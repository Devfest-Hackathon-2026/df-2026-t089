"""Write the walking classifier's image encoder (app/src/main/assets/coarse/<model>.onnx) for the head already in the repo.

The heads (<model>.head.json) were fitted on open_clip's encode_image of each column crop. This exports the same
encoder so the app (CoarseClassifier) can run it unchanged:
  input  "image": (n, 3, 256, 256) RGB in 0..1 (the app does not subtract a mean or divide by a std)
  output "embedding": (n, d) encode_image output (the app L2-normalises it before the head)
open_clip's normalisation is put inside the graph, so it is the same as preprocess() whatever the model uses.

Needs open_clip_torch, torch, onnx and onnxscript (not the TensorFlow environment; see requirements-coarse.txt):
  python -m venv tools/.venv-clip
  tools/.venv-clip/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
  tools/.venv-clip/Scripts/pip install open_clip_torch timm onnx onnxscript onnxruntime pillow numpy
  tools/.venv-clip/Scripts/python tools/coarse_export_encoder.py [MobileCLIP2-S0]

The weights are Apple's research licence (apple-amlr), so the .onnx is not committed (.gitignore).
"""
import json, sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app/src/main/assets/coarse"
SIZE = 256


class Encoder(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, image):
        return self.model.encode_image((image - self.mean) / self.std)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "MobileCLIP2-S0"
    head = json.loads((ASSETS / f"{name}.head.json").read_text(encoding="utf-8"))
    pretrained = head["model"].split("/", 1)[1]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model.eval()
    norm = next(t for t in preprocess.transforms if t.__class__.__name__ == "Normalize")
    enc = Encoder(model, list(norm.mean), list(norm.std)).eval()
    out = ASSETS / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(enc, (torch.rand(2, 3, SIZE, SIZE),), str(out), input_names=["image"], output_names=["embedding"],
                          dynamic_axes={"image": {0: "n"}, "embedding": {0: "n"}}, opset_version=18, dynamo=False)

    # 確認: open_clip の preprocess → encode_image と、アプリと同じ前処理 → ONNX が同じ埋め込みになる
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (SIZE, SIZE, 3), dtype=np.uint8))
    with torch.no_grad():
        ref = model.encode_image(preprocess(img)[None]).numpy()
    x = (np.asarray(img, np.float32) / 255).transpose(2, 0, 1)[None]
    got = ort.InferenceSession(str(out)).run(None, {"image": x})[0]
    cos = float((ref * got).sum() / np.linalg.norm(ref) / np.linalg.norm(got))
    assert got.shape[1] == len(head["weights"][0]), f"embedding {got.shape[1]} does not match the head {len(head['weights'][0])}"
    print(f"{out} ({out.stat().st_size / 1e6:.0f} MB), mean {list(norm.mean)} std {list(norm.std)}, cosine to open_clip {cos:.5f}")
    assert cos > 0.999


if __name__ == "__main__":
    main()
