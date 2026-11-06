"""2b（グループ化）の DINOv2 埋め込みを Modal の GPU で計算する部分。s2_group.py --modal から使う。
長辺1920pxの画像と枠を送り、切り抜き・正方形化・埋め込みは GPU 側でやる（送る量を画像1枚分に抑える）。
"""
import io
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
MODEL = "facebook/dinov2-small"
GPU = ["L4", "A10G", "T4", "L40S"]  # 先頭から空いている GPU を使う


def _download():
    from transformers import AutoModel
    AutoModel.from_pretrained(MODEL)


image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "transformers==5.22.1", "pillow", "numpy", "onnxruntime")
         .run_function(_download)
         .add_local_file(HERE / "common.py", "/root/common.py")
         .add_local_file(HERE / "s2_crops.py", "/root/s2_crops.py")
         .add_local_file(HERE / "s2_group.py", "/root/s2_group.py"))
app = modal.App("convenience-products-dino", image=image)


@app.cls(gpu=GPU, max_containers=4, timeout=1800, scaledown_window=60)
class Dino:
    @modal.enter()
    def load(self):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(MODEL).to("cuda").eval().half()

    @modal.method()
    def run(self, stem: str, jpg: bytes, items: list):
        """items = [(切り抜きID, 枠)] → (stem, IDの列, float16 の埋め込み bytes)"""
        import numpy as np
        import torch
        from PIL import Image
        from s2_crops import crop
        from s2_group import square
        im = Image.open(io.BytesIO(jpg)).convert("RGB")
        x = np.stack([square(crop(im, box)) for _, box in items]).transpose(0, 3, 1, 2)
        with torch.no_grad():
            e = self.model(pixel_values=torch.from_numpy(x).cuda().half()).pooler_output.float().cpu().numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True)
        return stem, [i for i, _ in items], e.astype(np.float16).tobytes()
