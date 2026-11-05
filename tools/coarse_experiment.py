"""Compare ways to tell the coarse category of each vertical column of a store photo.

Labels: tools/coarse_labels.json (slices across the photo width). Each photo is cut into N_COLS columns and each
column gets the category of the slice that covers its centre.

Validation never mixes neighbouring shots: photos taken within GROUP_GAP_S seconds of each other form one group
(the same spot), and whole groups are held out together.

  python tools/coarse_experiment.py                                   # MobileNetV3 and two CLIP ViT-B-32
  python tools/coarse_experiment.py --clip MobileCLIP2-S0:dfndr2b --no-mobilenet
"""
import argparse, json, re, time, warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
PHOTOS = ROOT / "boundary_training_images"
OUT = ROOT / "tools/out/coarse"
N_COLS = int(__import__("os").environ.get("N_COLS", 16))  # columns per photo; fewer columns = fewer crops to encode on the phone
GROUP_GAP_S = 6.0
N_FOLDS = 6
SEED = 7

labels = json.loads((ROOT / "tools/coarse_labels.json").read_text(encoding="utf-8"))["labels"]
mapping = json.loads((ROOT / "tools/coarse_categories.json").read_text(encoding="utf-8"))
mapping.pop("_comment")
CLASSES = list(mapping) + ["商品なし"]
files = sorted(labels)


def column_labels(slices):
    out = []
    for c in range(N_COLS):
        x = (c + 0.5) / N_COLS * 1000
        cat = next((s["category"] for s in slices if s["x_start"] <= x < s["x_end"]), slices[-1]["category"])
        out.append(CLASSES.index(cat))
    return out


Y = np.array([column_labels(labels[f]["slices"]) for f in files])  # (images, N_COLS)


def groups_by_time():
    t = [int(m[0:2]) * 3600 + int(m[2:4]) * 60 + int(m[4:6]) + int(m[6:9]) / 1000
         for m in (re.search(r"_(\d{9})", f).group(1) for f in files)]
    g, out = 0, []
    for i, ti in enumerate(t):
        if i and ti - t[i - 1] > GROUP_GAP_S:
            g += 1
        out.append(g)
    return np.array(out)


groups = groups_by_time()
rng = np.random.default_rng(SEED)
unique = rng.permutation(np.unique(groups))
fold_of_group = {g: i % N_FOLDS for i, g in enumerate(unique)}
folds = np.array([fold_of_group[g] for g in groups])


def load(name):
    return ImageOps.exif_transpose(Image.open(PHOTOS / name)).convert("RGB")


# ---------------- features
def mobilenet_columns():
    import keras
    size = 384
    base = keras.applications.MobileNetV3Small(input_shape=(size, size, 3), include_top=False, weights="imagenet",
                                               include_preprocessing=True)
    s16 = [l for l in base.layers if len(l.output.shape) == 4 and l.output.shape[1] == size // 16][-1]
    feat = keras.Model(base.input, s16.output)
    out = []
    for f in files:
        x = np.asarray(load(f).resize((size, size), Image.BILINEAR), np.float32)[None]
        m = feat.predict(x, verbose=0)[0]  # (24, 24, C)
        cols = np.stack([part.mean(axis=1) for part in np.array_split(m, N_COLS, axis=1)], 1)  # (24 rows, N_COLS, C)
        thirds = np.stack([cols[i * 8:(i + 1) * 8].mean(0) for i in range(3)], 1)  # (N_COLS, 3, C)
        out.append(thirds.reshape(N_COLS, -1))
    return np.stack(out)


def clip_columns(model_name, pretrained):
    import open_clip, torch
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()
    tok = open_clip.get_tokenizer(model_name)
    out = []
    torch.set_num_threads(4)  # closer to a phone than all 16 PC cores
    with torch.no_grad():
        sample = preprocess(load(files[0]).crop((0, 0, 300, 900)))[None]
        model.encode_image(sample)
        t0 = time.time()
        for _ in range(5):
            model.encode_image(sample)
        per_crop_ms = (time.time() - t0) / 5 * 1000
        params = sum(p.numel() for p in model.visual.parameters()) / 1e6
        print(f"  {model_name}: image encoder {params:.1f}M params, {per_crop_ms:.0f} ms per crop on 4 PC threads", flush=True)
        for f in files:
            im = load(f)
            w, h = im.size
            crops = []
            for c in range(N_COLS):
                cx = (c + 0.5) / N_COLS * w
                half = 1.5 / N_COLS * w  # a window three columns wide gives the column some context
                crops.append(preprocess(im.crop((int(max(0, cx - half)), 0, int(min(w, cx + half)), h))))
            e = model.encode_image(torch.stack(crops))
            out.append(torch.nn.functional.normalize(e, dim=-1).numpy())
        prompts = {
            "飲み物・お酒": "bottled drinks, tea, juice, beer cans and liquor in a refrigerated case",
            "冷凍・アイス": "frozen food and ice cream in a freezer", "弁当・パン": "bento boxes, rice balls, sandwiches and bread",
            "お菓子・おつまみ": "snacks, chips, chocolate, candy and jerky", "食品": "cup noodles, canned food and packaged groceries",
            "日用品・雑貨": "toiletries, cosmetics, masks and household goods", "商品なし": "an empty store aisle, floor or wall",
        }
        texts = [f"a photo of {prompts[c]} on the shelves of a Japanese convenience store" for c in CLASSES]
        t = torch.nn.functional.normalize(model.encode_text(tok(texts)), dim=-1).numpy()
    return np.stack(out), t


# ---------------- evaluation
def smooth(probs, switch_cost=1.5):
    """Most likely class sequence along the columns, where changing class costs switch_cost (in log-prob)."""
    lp = np.log(np.clip(probs, 1e-6, 1))
    n, k = lp.shape
    score, back = lp[0].copy(), np.zeros((n, k), int)
    for i in range(1, n):
        trans = score[:, None] - switch_cost * (1 - np.eye(k))
        back[i] = trans.argmax(0)
        score = trans.max(0) + lp[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return np.array(path[::-1])


def report(name, probs):
    """probs: (images, N_COLS, classes) predicted on held-out folds."""
    raw = probs.argmax(-1)
    smoothed = np.stack([smooth(p) for p in probs])
    res = {}
    for kind, pred in (("raw", raw), ("smoothed", smoothed)):
        ok = pred == Y
        products = Y != CLASSES.index("商品なし")
        res[kind] = {"acc": ok.mean(), "acc_products": ok[products].mean(),
                     "macro_f1": f1_score(Y.ravel(), pred.ravel(), average="macro")}
    print(f"{name:42s} raw acc {res['raw']['acc']:.3f} (products {res['raw']['acc_products']:.3f}, macroF1 {res['raw']['macro_f1']:.3f})"
          f" | smoothed acc {res['smoothed']['acc']:.3f} (products {res['smoothed']['acc_products']:.3f}, macroF1 {res['smoothed']['macro_f1']:.3f})",
          flush=True)
    return res, smoothed


def cross_validate(features, make_model):
    probs = np.zeros(Y.shape + (len(CLASSES),))
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        Xtr, ytr = features[tr].reshape(-1, features.shape[-1]), Y[tr].ravel()
        model = make_model()
        model.fit(Xtr, ytr)
        p = model.predict_proba(features[te].reshape(-1, features.shape[-1]))
        full = np.zeros((p.shape[0], len(CLASSES)))
        full[:, model.classes_] = p
        probs[te] = full.reshape(te.sum(), N_COLS, -1)
    return probs


def with_neighbours(f):
    left = np.concatenate([f[:, :1], f[:, :-1]], 1)
    right = np.concatenate([f[:, 1:], f[:, -1:]], 1)
    return np.concatenate([f, left, right], -1)


def export_head(name, pre, emb, c):
    model = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(emb.reshape(-1, emb.shape[-1]), Y.ravel())
    assert list(model.classes_) == list(range(len(CLASSES))), "every class needs at least one labelled column"
    head = {"model": f"{name}/{pre}", "columns": N_COLS, "window_columns": 3, "classes": CLASSES,
            "weights": model.coef_.tolist(), "bias": model.intercept_.tolist(), "switch_cost": 1.5,
            "note": "embedding = L2-normalised image embedding of a crop three columns wide, full height; "
                    "probs = softmax(W e + b); then smooth along the columns"}
    path = OUT / f"head_{name}_{N_COLS}.json"
    path.write_text(json.dumps(head, ensure_ascii=False), encoding="utf-8")
    print("head written to", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", nargs="*", default=["ViT-B-32:laion2b_s34b_b79k", "ViT-B-32:openai"])
    ap.add_argument("--no-mobilenet", action="store_true")
    ap.add_argument("--export-head", type=float, metavar="C",
                    help="also fit logistic regression with this C on all photos and write out/coarse/head_<model>_<N_COLS>.json for the app")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} photos, {N_COLS} columns each, {groups.max() + 1} spots, {N_FOLDS} folds")
    counts = np.bincount(Y.ravel(), minlength=len(CLASSES))
    print("columns per class", dict(zip(CLASSES, counts.tolist())))
    majority = np.full(Y.shape + (len(CLASSES),), 1e-3)
    majority[..., counts.argmax()] = 1
    results = {"majority class": report("majority class (baseline)", majority)[0]}

    logreg = lambda c: (lambda: LogisticRegression(C=c, max_iter=3000, class_weight="balanced"))
    if not a.no_mobilenet:
        t = time.time()
        mob = mobilenet_columns()
        print(f"mobilenet features {mob.shape} in {time.time() - t:.0f}s")
        mob = mob / (np.linalg.norm(mob, axis=-1, keepdims=True) + 1e-6)
        for c in (0.1, 1.0):
            results[f"mobilenetv3 small + logreg C={c}"] = report(f"MobileNetV3 Small + logreg C={c}", cross_validate(mob, logreg(c)))[0]

    best = None
    for spec in a.clip:
        name, pre = spec.split(":")
        t = time.time()
        cache = OUT / f"emb_{name}_{pre}_{N_COLS}.npz"
        if cache.exists():
            z = np.load(cache)
            emb, text = z["emb"], z["text"]
        else:
            emb, text = clip_columns(name, pre)
            np.savez(cache, emb=emb, text=text)
        print(f"CLIP {name}/{pre} features {emb.shape} in {time.time() - t:.0f}s")
        if len(text) == len(CLASSES):  # the cached text prompts belong to the classes at the time of caching
            zero = np.exp(100 * emb @ text.T)
            zero /= zero.sum(-1, keepdims=True)
            results[f"clip {name}/{pre} zero-shot"] = report(f"{name} {pre} zero-shot (no labels)", zero)[0]
        for c in (0.3, 3.0):
            r, sm = report(f"{name} + logreg C={c}", cross_validate(emb, logreg(c)))
            results[f"clip {name}/{pre} logreg C={c}"] = r
            r2, sm2 = report(f"{name} + neighbours + logreg C={c}", cross_validate(with_neighbours(emb), logreg(c)))
            results[f"clip {name}/{pre} neighbours logreg C={c}"] = r2
            if best is None or r2["smoothed"]["acc"] > best[0]:
                best = (r2["smoothed"]["acc"], f"{name}/{pre} C={c}", sm2)
        if a.export_head:
            export_head(name, pre, emb, a.export_head)
    (OUT / f"experiment_{'_'.join(x.replace(':', '-') for x in a.clip)}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    np.save(OUT / "best_smoothed_pred.npy", best[2])
    print("best:", best[1], round(best[0], 3))


if __name__ == "__main__":
    main()
