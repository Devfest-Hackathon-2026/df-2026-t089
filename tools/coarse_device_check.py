"""Make a test photo for CoarseClassifierDeviceTest and the column labels the phone should give for it.

The photo is turned upright and shrunk to the size the app classifies (long side 1280), so the phone and the PC
see the same pixels. The expected labels come from the same ONNX models and heads that go into the app.

  python tools/coarse_device_check.py PXL_20261105_132630390.jpg
  adb push tools/out/coarse/coarse_test.jpg /sdcard/Android/data/jp.devfest.shelfsight/files/coarse_test.jpg
  adb push tools/out/coarse/coarse_test.expected.json /sdcard/Android/data/jp.devfest.shelfsight/files/
"""
import json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app/src/main/assets/coarse"
OUT = ROOT / "tools/out/coarse"
LONG_SIDE = 1280
SIZE = 256


def crops(w, h, n, window):
    """Same as CoarseHead.cropFor."""
    out = []
    for c in range(n):
        cx, half = (c + 0.5) / n * w, window / 2 / n * w
        left, right = int(max(0, cx - half)), int(min(w, cx + half))
        width = right - left
        if width <= h:
            top = (h - width) // 2
            out.append((left, top, right, top + width))
        else:
            inset = (width - h) // 2
            out.append((left + inset, 0, left + inset + h, h))
    return out


def smooth(probs, cost):
    lp = np.log(np.clip(probs, 1e-6, 1))
    n, k = lp.shape
    score, back = lp[0].copy(), np.zeros((n, k), int)
    for i in range(1, n):
        trans = score[:, None] - cost * (1 - np.eye(k))
        back[i] = trans.argmax(0)
        score = trans.max(0) + lp[i]
    path = [int(score.argmax())]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    return path[::-1]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "PXL_20261105_132630390.jpg"
    im = ImageOps.exif_transpose(Image.open(ROOT / "boundary_training_images" / name)).convert("RGB")
    im.thumbnail((LONG_SIDE, LONG_SIDE), Image.BILINEAR)
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / "coarse_test.jpg", quality=95)
    im = Image.open(OUT / "coarse_test.jpg").convert("RGB")  # what the phone will decode
    expected = {"photo": name, "width": im.width, "height": im.height}
    for onnx in sorted(ASSETS.glob("*.onnx")):
        head = json.loads((ASSETS / f"{onnx.stem}.head.json").read_text(encoding="utf-8"))
        x = np.stack([np.asarray(im.crop(r).resize((SIZE, SIZE), Image.BILINEAR), np.float32).transpose(2, 0, 1) / 255
                      for r in crops(im.width, im.height, head["columns"], head["window_columns"])])
        e = ort.InferenceSession(str(onnx)).run(None, {"image": x})[0]
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        logits = e @ np.array(head["weights"]).T + np.array(head["bias"])
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        cols = [head["classes"][i] for i in smooth(p, head["switch_cost"])]
        expected[onnx.stem] = cols
        print(onnx.stem, cols)
    (OUT / "coarse_test.expected.json").write_text(json.dumps(expected, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", OUT / "coarse_test.jpg", OUT / "coarse_test.expected.json")


if __name__ == "__main__":
    main()
