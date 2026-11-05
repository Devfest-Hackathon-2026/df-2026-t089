"""Show where the coarse column model goes wrong: a confusion table and photos with label vs. prediction.

Uses held-out predictions only (each photo is predicted by a model that never saw its spot).
Run tools/coarse_experiment.py first so the embeddings are cached.

  N_COLS=8 python tools/coarse_review.py --clip MobileCLIP-S1:datacompdr --c 3
"""
import argparse, os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import LogisticRegression

import coarse_experiment as ce

COLORS = [(29, 120, 210), (240, 150, 20), (0, 160, 170), (120, 85, 60), (230, 57, 70), (40, 160, 80), (130, 130, 130), (150, 70, 200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="MobileCLIP-S1:datacompdr")
    ap.add_argument("--c", type=float, default=3.0)
    a = ap.parse_args()
    name, pre = a.clip.split(":")
    z = np.load(ce.OUT / f"emb_{name}_{pre}_{ce.N_COLS}.npz")
    probs = ce.cross_validate(z["emb"], lambda: LogisticRegression(C=a.c, max_iter=3000, class_weight="balanced"))
    pred = np.stack([ce.smooth(p) for p in probs])
    y = ce.Y
    k = len(ce.CLASSES)
    conf = np.zeros((k, k), int)
    for t, p in zip(y.ravel(), pred.ravel()):
        conf[t, p] += 1
    width = max(len(c) for c in ce.CLASSES) + 2
    print(f"accuracy {(pred == y).mean():.3f} over {y.size} columns; rows = label, columns = prediction")
    print(" " * width + "".join(f"{i:>5d}" for i in range(k)) + "   recall")
    for i, c in enumerate(ce.CLASSES):
        print(f"{i} {c:<{width - 2}}" + "".join(f"{v:>5d}" for v in conf[i]) + f"   {conf[i, i] / max(conf[i].sum(), 1):.2f}")
    per_photo = (pred == y).mean(1)
    print("photos with most columns right:", int((per_photo >= 0.75).sum()), "of", len(per_photo),
          "| photos below half right:", [ce.files[i][13:22] for i in np.flatnonzero(per_photo < 0.5)])

    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 16)
    legend = Image.new("RGB", (4 * 420, 40), "white")
    d = ImageDraw.Draw(legend)
    for i, c in enumerate(ce.CLASSES):
        d.rectangle([10 + i * 205, 10, 30 + i * 205, 30], fill=COLORS[i])
        d.text((36 + i * 205, 10), c, fill="black", font=font)
    for s in range(0, len(ce.files), 8):
        sheet = Image.new("RGB", (4 * 420, 2 * 620 + 40), "white")
        sheet.paste(legend, (0, 0))
        for j, f in enumerate(ce.files[s:s + 8]):
            i = s + j
            im = ce.load(f)
            im.thumbnail((410, 540))
            w, h = im.size
            canvas = Image.new("RGB", (w, h + 50), "white")
            canvas.paste(im, (0, 0))
            d = ImageDraw.Draw(canvas)
            for c in range(ce.N_COLS):
                x0, x1 = c * w / ce.N_COLS, (c + 1) * w / ce.N_COLS
                d.rectangle([x0, h + 2, x1, h + 22], fill=COLORS[y[i, c]])
                d.rectangle([x0, h + 26, x1, h + 46], fill=COLORS[pred[i, c]])
                if y[i, c] != pred[i, c]:
                    d.line([x0 + 3, h + 36, x1 - 3, h + 36], fill="white", width=3)
            d.text((2, h + 4), "正解", fill="white", font=font, stroke_width=2, stroke_fill="black")
            d.text((2, h + 28), "予測", fill="white", font=font, stroke_width=2, stroke_fill="black")
            x, ypos = (j % 4) * 420, 40 + (j // 4) * 620
            sheet.paste(canvas, (x, ypos))
            ImageDraw.Draw(sheet).text((x + 2, ypos + h + 52), f"{i} {f[13:22]}  {per_photo[i]:.0%}", fill="black", font=font)
        path = ce.OUT / f"review_{name}_{ce.N_COLS}_{s // 8}.jpg"
        sheet.save(path, quality=80)
    print("sheets in", ce.OUT)


if __name__ == "__main__":
    main()
