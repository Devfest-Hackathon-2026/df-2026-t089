"""Train the two on-device models and write them into the app's assets.

  shelf_classifier.tflite : 224x224 RGB (0-255 float) -> softmax over labels.txt
  shelf_boundary.tflite   : 384x384 RGB (0-255 float) -> 24 per-column boundary probabilities

Usage:  python tools/train.py --data ../df-2026-t089/training_images
"""
import argparse, csv, datetime, json, random
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app/src/main/assets"
CACHE_W, CACHE_H = 384, 512
CLS_SIZE = 224
BND_SIZE = 384
BND_COLS = BND_SIZE // 16
SEED = 7


def load_cache(data: Path, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return dict(zip(z["names"], z["imgs"]))
    names, imgs = [], []
    for p in sorted(data.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")  # iPhone JPGs carry EXIF rotation
        imgs.append(np.asarray(im.resize((CACHE_W, CACHE_H), Image.BILINEAR)))
        names.append(p.name)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(names), imgs=np.stack(imgs))
    return dict(zip(names, imgs))


def read_csv(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def backbone(size):
    return keras.applications.MobileNetV3Small(
        input_shape=(size, size, 3), include_top=False, weights="imagenet",
        include_preprocessing=True, minimalistic=False)


def export_tflite(model, path: Path):
    tmp = ROOT / "tools/.cache/saved" / path.stem
    model.export(str(tmp))
    conv = tf.lite.TFLiteConverter.from_saved_model(str(tmp))
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    path.write_bytes(conv.convert())
    print("wrote", path, path.stat().st_size // 1024, "KB")


# ---------------- classifier ----------------
def rand_crop(img, rng, min_w=0.4, min_h=0.6):
    h, w, _ = img.shape
    cw, ch = int(w * rng.uniform(min_w, 1)), int(h * rng.uniform(min_h, 1))
    x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    return img[y:y + ch, x:x + cw]


def resize(img, w, h):
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), dtype=np.float32)


def jitter(x, rng):
    x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
    return np.clip(x, 0, 255)


def cls_batches(items, n_cls, rng, batch=32):
    counts = np.bincount([y for _, y in items], minlength=n_cls)
    w = np.array([1 / np.sqrt(counts[y]) for _, y in items])  # soften class imbalance
    w /= w.sum()
    while True:
        xs, ys = [], []
        for j in rng.choice(len(items), batch, p=w):
            img, y = items[j]
            c = rand_crop(img, rng)
            if rng.random() < 0.5:
                c = c[:, ::-1]
            xs.append(jitter(resize(c, CLS_SIZE, CLS_SIZE), rng))
            ys.append(y)
        yield np.stack(xs), np.array(ys)


def train_classifier(train, val, classes, epochs):
    rng = np.random.default_rng(SEED)
    base = backbone(CLS_SIZE)
    base.trainable = False
    inp = keras.Input((CLS_SIZE, CLS_SIZE, 3))
    x = keras.layers.GlobalAveragePooling2D()(base(inp, training=False))  # keep BatchNorm stats frozen when fine-tuning
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(len(classes), activation="softmax")(x)
    model = keras.Model(inp, out)
    steps = max(len(train) // 32, 8)
    vx = np.stack([resize(img, CLS_SIZE, CLS_SIZE) for img, _ in val]) if val else None
    vy = np.array([y for _, y in val]) if val else None
    vd = (vx, vy) if val else None
    for lr, unfreeze in ((1e-3, 0), (3e-5, 40)):
        if unfreeze:
            base.trainable = True
            for l in base.layers[:-unfreeze]:
                l.trainable = False
        model.compile(keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(cls_batches(train, len(classes), rng), steps_per_epoch=steps, epochs=epochs,
                  validation_data=vd, verbose=2)
    return model


# ---------------- boundary ----------------
def cuts_target(cuts):
    centers = (np.arange(BND_COLS) + 0.5) / BND_COLS
    t = np.zeros(BND_COLS, np.float32)
    for c in cuts:
        t = np.maximum(t, np.exp(-0.5 * ((centers - c) * BND_COLS / 0.6) ** 2))
    return t


def aug_real(img, cuts, rng):
    w = img.shape[1]
    cw = int(w * rng.uniform(0.7, 1.0))
    x0 = rng.integers(0, w - cw + 1)
    img = img[:, x0:x0 + cw]
    cuts = [(c * w - x0) / cw for c in cuts]
    cuts = [c for c in cuts if 0.02 < c < 0.98]
    if rng.random() < 0.5:
        img, cuts = img[:, ::-1], [1 - c for c in cuts]
    return img, cuts


def synth(by_class, rng):
    """Stitch 2-3 shelf photos of different categories side by side; the seams are boundaries."""
    k = rng.integers(2, 4)
    cls = rng.choice(list(by_class), size=k, replace=False)
    widths = rng.dirichlet(np.ones(k) * 2) * 0.7 + 0.3 / k
    parts, cuts, acc = [], [], 0.0
    for c, fw in zip(cls, widths):
        img = by_class[c][rng.integers(len(by_class[c]))]
        pw = max(int(CACHE_W * fw), 16)
        x0 = rng.integers(0, CACHE_W - pw + 1)
        parts.append(jitter(img[:, x0:x0 + pw].astype(np.float32), rng))
        acc += pw
        cuts.append(acc)
    img = np.concatenate(parts, axis=1).astype(np.uint8)
    return img, [c / acc for c in cuts[:-1]]


def bnd_batches(real, by_class, rng, batch=16):
    while True:
        xs, ys = [], []
        for _ in range(batch):
            if real and rng.random() < 0.5:
                img, cuts = real[rng.integers(len(real))]
                img, cuts = aug_real(img, cuts, rng)
                img = jitter(img.astype(np.float32), rng)
            else:
                img, cuts = synth(by_class, rng)
            xs.append(resize(np.asarray(img, np.uint8), BND_SIZE, BND_SIZE))
            ys.append(cuts_target(cuts))
        yield np.stack(xs), np.stack(ys)


def build_boundary():
    base = backbone(BND_SIZE)
    # last stride-16 feature map gives 24 columns at 384 px
    s16 = [l for l in base.layers if len(l.output.shape) == 4 and l.output.shape[1] == BND_SIZE // 16][-1]
    feat = keras.Model(base.input, s16.output)
    for l in feat.layers[:-30]:
        l.trainable = False
    inp = keras.Input((BND_SIZE, BND_SIZE, 3))
    x = keras.layers.Conv2D(64, 1, activation="relu")(feat(inp, training=False))
    x = keras.layers.AveragePooling2D((BND_SIZE // 16, 1))(x)  # collapse height -> per-column
    x = keras.layers.Reshape((BND_COLS, 64))(x)
    x = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(1, 3, padding="same", activation="sigmoid")(x)
    out = keras.layers.Reshape((BND_COLS,))(x)
    return keras.Model(inp, out)


def decode_peaks(p, thr):
    """Same decoding as BoundaryDetector.kt: local maxima above thr, sub-column refined."""
    out = []
    for i in range(len(p)):
        l = p[i - 1] if i > 0 else 0.0
        r = p[i + 1] if i < len(p) - 1 else 0.0
        if p[i] >= thr and p[i] >= l and p[i] > r:
            d = l - 2 * p[i] + r
            off = 0.5 * (l - r) / d if d < 0 else 0.0
            out.append((i + 0.5 + off) / len(p))
    return out


def pr(model, val, thr, tol=0.06):
    tp = fp = fn = 0
    for img, cuts in val:
        p = model.predict(resize(img, BND_SIZE, BND_SIZE)[None], verbose=0)[0]
        pred = decode_peaks(p, thr)
        used = set()
        for c in cuts:
            m = [i for i, q in enumerate(pred) if abs(q - c) < tol and i not in used]
            if m:
                used.add(m[0]); tp += 1
            else:
                fn += 1
        fp += len(pred) - len(used)
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1)


def train_boundary(real, by_class, steps, epochs):
    rng = np.random.default_rng(SEED)
    model = build_boundary()
    pos_w = 4.0

    def loss(y, p):
        p = tf.clip_by_value(p, 1e-6, 1 - 1e-6)
        return -tf.reduce_mean(pos_w * y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p))

    model.compile(keras.optimizers.Adam(5e-4), loss)
    model.fit(bnd_batches(real, by_class, rng), steps_per_epoch=steps, epochs=epochs, verbose=2)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "df-2026-t089/training_images")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--only", choices=["cls", "bnd"])
    ap.add_argument("--eval-only", action="store_true", help="train holdout models + write results, keep the app models")
    a = ap.parse_args()
    results_path = ROOT / "tools/results/eval.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    random.seed(SEED); np.random.seed(SEED); keras.utils.set_random_seed(SEED)
    imgs = load_cache(a.data, ROOT / "tools/.cache/images.npz")
    labels = [(r["file"], r["shelf"]) for r in read_csv(a.data / "labels.csv") if r["file"] in imgs]
    classes = sorted({s for _, s in labels})
    ASSETS.mkdir(parents=True, exist_ok=True)

    if a.only != "bnd":
        items = [(imgs[f], classes.index(s)) for f, s in labels]
        files = [f for f, _ in labels]
        rng = np.random.default_rng(SEED)
        val_idx = set()
        for c in range(len(classes)):  # stratified ~15% holdout, at least 1 per class with >=4 samples
            ids = [i for i, (_, y) in enumerate(items) if y == c]
            if len(ids) >= 4:
                val_idx |= set(rng.choice(ids, max(1, len(ids) * 15 // 100), replace=False).tolist())
        tr = [it for i, it in enumerate(items) if i not in val_idx]
        va = [it for i, it in enumerate(items) if i in val_idx]
        print(f"classifier: {len(classes)} classes, train {len(tr)} val {len(va)}")
        m = train_classifier(tr, va, classes, a.epochs)
        vi = sorted(val_idx)
        probs = m.predict(np.stack([resize(items[i][0], CLS_SIZE, CLS_SIZE) for i in vi]), verbose=0)
        rows = [{"file": files[i], "label": classes[items[i][1]],
                 "top": [[classes[k], round(float(p[k]), 3)] for k in np.argsort(-p)[:3]]} for i, p in zip(vi, probs)]
        acc = sum(r["label"] == r["top"][0][0] for r in rows) / len(rows)
        print(f"  holdout accuracy {acc:.3f}")
        results["classifier"] = {"accuracy": acc, "train": len(tr), "val": rows, "classes": classes,
                                 "epochs": a.epochs, "at": datetime.datetime.now().isoformat(timespec="seconds")}
        if not a.eval_only:
            final = train_classifier(items, [], classes, a.epochs)
            export_tflite(final, ASSETS / "shelf_classifier.tflite")
            (ASSETS / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    if a.only != "cls":
        real = [(imgs[r["file"]], [float(c) for c in r["cuts"].split()], r["file"])
                for r in read_csv(a.data / "boundaries.csv") if r["file"] in imgs]
        by_class = {}
        for f, s in labels:
            by_class.setdefault(s, []).append(imgs[f])
        rng = random.Random(SEED)
        rng.shuffle(real)
        names = [f for _, _, f in real]
        real = [(i, c) for i, c, _ in real]
        va, tr = real[:7], real[7:]
        print(f"boundary: real train {len(tr)} val {len(va)}")
        m = train_boundary(tr, by_class, 60, a.epochs)
        sweep = []
        for thr in (0.05, 0.1, 0.2, 0.3, 0.5):
            p, r = pr(m, va, thr)
            sweep.append({"thr": thr, "precision": p, "recall": r})
            print(f"  val thr={thr}: precision={p:.2f} recall={r:.2f}")
        rows = [{"file": n, "cuts": c, "probs": [round(float(x), 3) for x in m.predict(resize(img, BND_SIZE, BND_SIZE)[None], verbose=0)[0]]}
                for n, (img, c) in zip(names[:7], va)]
        results["boundary"] = {"sweep": sweep, "appThreshold": 0.1, "tolerance": 0.06, "train": len(tr), "val": rows,
                               "epochs": a.epochs, "at": datetime.datetime.now().isoformat(timespec="seconds")}
        if not a.eval_only:
            final = train_boundary(real, by_class, 60, a.epochs)
            export_tflite(final, ASSETS / "shelf_boundary.tflite")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", results_path)


if __name__ == "__main__":
    main()
