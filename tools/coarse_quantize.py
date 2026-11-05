"""Make 8-bit versions of the walking classifier's image encoders and check they still agree with the originals.

Needs onnx, which does not fit in the main environment (see requirements-coarse.txt), so run it from a separate one:
  python -m venv <dir> && <dir>/Scripts/pip install onnx onnxruntime numpy pillow
  <dir>/Scripts/python tools/coarse_quantize.py

For every labelled photo it compares the column categories from the original and the 8-bit model (with the head
that goes into the app), and the time per photo on 4 PC threads. Writes app/src/main/assets/coarse/<name>-int8.onnx
and a copy of the head.

Result (2026-11-06, 63 photos): neither variant is worth using, so none of them are in the app.
- MatMul/Gemm only: agrees with fp32 on 99.6% of columns but is no faster (the encoders are mostly convolutions)
  and only 15% smaller.
- With Conv as well: 4x smaller but breaks the embeddings (13-29% of columns agree) and is 15x slower on the PC.
Static (calibrated) quantization might do better and has not been tried.
"""
import json, shutil, time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from PIL import Image, ImageOps

from coarse_device_check import crops, smooth

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app/src/main/assets/coarse"
PHOTOS = ROOT / "boundary_training_images"
SIZE = 256


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    return ort.InferenceSession(str(path), o)


def columns(sess, head, im):
    x = np.stack([np.asarray(im.crop(r).resize((SIZE, SIZE), Image.BILINEAR), np.float32).transpose(2, 0, 1) / 255
                  for r in crops(im.width, im.height, head["columns"], head["window_columns"])])
    t = time.perf_counter()
    e = sess.run(None, {"image": x})[0]
    ms = (time.perf_counter() - t) * 1000
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    logits = e @ np.array(head["weights"]).T + np.array(head["bias"])
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    return smooth(p, head["switch_cost"]), e, ms


def main():
    labels = json.loads((ROOT / "tools/coarse_labels.json").read_text(encoding="utf-8"))["labels"]
    photos = []
    for name in sorted(labels):
        im = ImageOps.exif_transpose(Image.open(PHOTOS / name)).convert("RGB")
        im.thumbnail((1280, 1280), Image.BILINEAR)
        photos.append((name, im))
    for model in ("MobileCLIP2-S0", "MobileCLIP-S1"):
        src = ASSETS / f"{model}.onnx"
        if not src.exists():
            continue
        head = json.loads((ASSETS / f"{model}.head.json").read_text(encoding="utf-8"))
        for ops, tag in ((["MatMul", "Gemm"], "int8"), (["MatMul", "Gemm", "Conv"], "int8conv")):
            dst = ASSETS / f"{model}-{tag}.onnx"
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=ops)
            a, b = session(src), session(dst)
            same, total, cos, ms_a, ms_b = 0, 0, [], [], []
            for name, im in photos:
                pa, ea, ta = columns(a, head, im)
                pb, eb, tb = columns(b, head, im)
                same += sum(x == y for x, y in zip(pa, pb))
                total += len(pa)
                cos.extend((ea * eb).sum(1).tolist())
                ms_a.append(ta)
                ms_b.append(tb)
            print(f"{model} {tag}: {dst.stat().st_size / 1e6:.0f} MB (fp32 {src.stat().st_size / 1e6:.0f} MB), "
                  f"columns agreeing with fp32 {same / total:.3f}, embedding cosine median {np.median(cos):.4f} min {min(cos):.3f}, "
                  f"ms per photo fp32 {np.median(ms_a):.0f} vs {np.median(ms_b):.0f}", flush=True)
            shutil.copy(ASSETS / f"{model}.head.json", ASSETS / f"{model}-{tag}.head.json")


if __name__ == "__main__":
    main()
