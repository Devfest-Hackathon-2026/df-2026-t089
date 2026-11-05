"""Train only the shelf boundary model from boundary_labels.csv. Product labels are not used.

Labeling rule (tools/boundary_labels.csv, cuts are x positions 0-1 read at mid image height):
  boundary     : shelf uprights, a change of fixture type, the edge between a shelf and another fixture or a wall
  not a boundary: a change of product, a refrigerator door frame

Usage:
  python tools/train_boundary.py --images boundary_training_images

The result goes to tools/out/. It does not replace app/src/main/assets/shelf_boundary.tflite.
"""
import argparse
import csv
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

import train as base  # 既存の学習スクリプトのモデル構造と前処理を使う

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tools/out"
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
HOLDOUT_PERIOD = 15
HOLDOUT_BLOCK = 3  # 撮影順に 15 枚ごとに連続 3 枚を検証に回し、似た写真が学習と検証に分かれるのを減らす
SEG_MARGIN = 0.04


def read_labels(path):
    with open(path, encoding="utf-8-sig") as f:
        return [(r["file"], [float(c) for c in r["cuts"].split()]) for r in csv.DictReader(f)]


def split(items):
    items = sorted(items, key=lambda it: it[0])
    val = [it for i, it in enumerate(items) if i % HOLDOUT_PERIOD < HOLDOUT_BLOCK]
    train = [it for i, it in enumerate(items) if i % HOLDOUT_PERIOD >= HOLDOUT_BLOCK]
    return train, val


def segments_of(real):
    """境界で区切った区間。境界をまたがないので、つなぎ合わせると継ぎ目だけが境界になる。"""
    out = []
    for img, cuts in real:
        edges = [0.0] + sorted(cuts) + [1.0]
        for a, b in zip(edges, edges[1:]):
            x0, x1 = int((a + SEG_MARGIN) * img.shape[1]), int((b - SEG_MARGIN) * img.shape[1])
            if x1 - x0 >= 48:
                out.append(img[:, x0:x1])
    return out


def stitched(segments, rng):
    """異なる区間を横に並べた合成画像。継ぎ目の位置が正解になる。"""
    k = rng.integers(2, 4)
    picks = [segments[i] for i in rng.choice(len(segments), size=k, replace=False)]
    widths = rng.dirichlet(np.ones(k) * 2) * 0.7 + 0.3 / k
    parts, cuts, acc = [], [], 0
    for seg, fraction in zip(picks, widths):
        w = max(int(base.CACHE_W * fraction), 16)
        part = base.resize(seg, w, base.CACHE_H)
        parts.append(base.jitter(part.astype(np.float32), rng))
        acc += w
        cuts.append(acc)
    return np.concatenate(parts, axis=1).astype(np.uint8), [c / acc for c in cuts[:-1]]


def batches(real, rng, batch=16, synth_ratio=0.3):
    segments = segments_of(real)
    while True:
        xs, ys = [], []
        for _ in range(batch):
            if rng.random() < synth_ratio:
                img, cuts = stitched(segments, rng)
            else:
                img, cuts = real[rng.integers(len(real))]
                img, cuts = base.aug_real(img, cuts, rng)
                img = base.jitter(img.astype(np.float32), rng)
            xs.append(base.resize(np.asarray(img, np.uint8), base.BND_SIZE, base.BND_SIZE))
            ys.append(base.cuts_target(cuts))
        yield np.stack(xs), np.stack(ys)


def train_model(real, steps, epochs, log_name):
    rng = np.random.default_rng(base.SEED)
    keras.utils.set_random_seed(base.SEED)
    model = base.build_boundary()
    pos_w = 4.0

    def loss(y, p):
        p = tf.clip_by_value(p, 1e-6, 1 - 1e-6)
        return -tf.reduce_mean(pos_w * y * tf.math.log(p) + (1 - y) * tf.math.log(1 - p))

    model.compile(keras.optimizers.Adam(5e-4), loss)
    model.fit(
        batches(real, rng), steps_per_epoch=steps, epochs=epochs, verbose=2,
        callbacks=[
            keras.callbacks.TensorBoard(str(OUT / "logs" / log_name)),
            keras.callbacks.CSVLogger(str(OUT / f"{log_name}.csv")),
        ],
    )
    return model


def score(predict, val, thr, tol):
    tp = fp = fn = 0
    errors = []
    for img, cuts in val:
        pred = base.decode_peaks(predict(img), thr)
        used = set()
        for c in cuts:
            candidates = [i for i, q in enumerate(pred) if abs(q - c) < tol and i not in used]
            if candidates:
                best = min(candidates, key=lambda i: abs(pred[i] - c))
                used.add(best)
                errors.append(abs(pred[best] - c))
                tp += 1
            else:
                fn += 1
        fp += len(pred) - len(used)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, float(np.mean(errors)) if errors else float("nan")


def report(name, predict, val):
    print(f"--- {name}", flush=True)
    best = (0.0, None)
    for tol in (0.06, 0.03):
        for thr in THRESHOLDS:
            p, r, f1, err = score(predict, val, thr, tol)
            print(f"  tol={tol} thr={thr}: precision={p:.2f} recall={r:.2f} f1={f1:.2f} mean_error={err:.3f}", flush=True)
            if tol == 0.06 and f1 > best[0]:
                best = (f1, thr)
    return best


def keras_predict(model):
    return lambda img: model.predict(base.resize(img, base.BND_SIZE, base.BND_SIZE)[None], verbose=0)[0]


def tflite_predict(path):
    interp = tf.lite.Interpreter(model_path=str(path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def predict(img):
        x = base.resize(img, base.BND_SIZE, base.BND_SIZE)[None].astype(inp["dtype"])
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        return interp.get_tensor(out["index"])[0]

    return predict


def save_previews(predict, val, names, thr):
    """検証画像に、見本の境界（緑）と予測した境界（赤）を重ねて保存する。"""
    folder = OUT / "val_preview"
    folder.mkdir(parents=True, exist_ok=True)
    for name, (img, cuts) in zip(names, val):
        canvas = Image.fromarray(img).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        for c in cuts:
            draw.line([(c * w, 0), (c * w, h * 0.5)], fill=(0, 255, 0), width=4)
        for c in base.decode_peaks(predict(img), thr):
            draw.line([(c * w, h * 0.5), (c * w, h)], fill=(255, 0, 0), width=4)
        canvas.save(folder / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, default=ROOT / "boundary_training_images")
    ap.add_argument("--labels", type=Path, default=ROOT / "tools/boundary_labels.csv")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    imgs = base.load_cache(a.images, ROOT / "tools/.cache/boundary_images.npz")
    labels = read_labels(a.labels)
    missing = [f for f, _ in labels if f not in imgs]
    if missing:
        raise SystemExit(f"labels refer to missing images: {missing[:5]}")

    train_items, val_items = split(labels)
    train_set = [(imgs[f], cuts) for f, cuts in train_items]
    val_set = [(imgs[f], cuts) for f, cuts in val_items]
    print(f"boundary only: train {len(train_set)} images, val {len(val_set)} images, "
          f"val boundaries {sum(len(c) for _, c in val_set)}", flush=True)

    existing = ROOT / "app/src/main/assets/shelf_boundary.tflite"
    report("existing shelf_boundary.tflite", tflite_predict(existing), val_set)

    start = time.time()
    model = train_model(train_set, a.steps, a.epochs, "holdout")
    print(f"holdout training took {(time.time() - start) / 60:.1f} min", flush=True)
    f1, thr = report("new model (trained without val images)", keras_predict(model), val_set)
    print(f"best on val: f1={f1:.2f} at thr={thr}", flush=True)
    save_previews(keras_predict(model), val_set, [f for f, _ in val_items], thr or 0.5)

    # 配布用は全画像で学習し直す。検証画像を含むので、精度は上の数字で判断する。
    final = train_model(train_set + val_set, a.steps, a.epochs, "final")
    base.export_tflite(final, OUT / "shelf_boundary.tflite")
    print(f"suggested threshold from holdout: {thr}", flush=True)


if __name__ == "__main__":
    main()
