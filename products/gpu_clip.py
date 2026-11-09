"""② 切り抜きの MobileCLIP 埋め込みを Modal で計算する部分。s2_crops.py --modal から使う。
アプリと同じ ONNX を onnxruntime-gpu で動かす（CUDA が使えなければコンテナの CPU で続ける）。
長辺1920pxの画像と枠を送り、切り抜き・縮小・埋め込みは向こうでやる（送る量を画像1枚分に抑える）。
"""
import io
from pathlib import Path

import modal

from common import CLIP_ONNX

HERE = Path(__file__).resolve().parent
GPU = ["L4", "A10G", "T4", "L40S"]  # 先頭から空いている GPU を使う
REMOTE_ONNX = "/products/work/models/mobileclip_s0_vision.onnx"  # 向こうの common.CLIP_ONNX と同じ場所

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("onnxruntime-gpu[cuda,cudnn]", "pillow", "numpy")
         .add_local_file(CLIP_ONNX, REMOTE_ONNX)
         .add_local_file(HERE / "common.py", "/root/common.py")
         .add_local_file(HERE / "s2_crops.py", "/root/s2_crops.py"))
app = modal.App("convenience-products-clip", image=image)


@app.cls(gpu=GPU, cpu=4, max_containers=4, timeout=1800, scaledown_window=60)
class Clip:
    @modal.enter()
    def load(self):
        import onnxruntime as ort
        try:
            ort.preload_dlls()  # pip で入れた CUDA/cuDNN を読み込む
        except Exception:
            pass
        from s2_crops import Embedder
        self.emb = Embedder()
        self.provider = self.emb.s.get_providers()[0]

    @modal.method()
    def run(self, stem: str, jpg: bytes, items: list):
        """items = [(枠番号, 枠)] → (stem, 枠番号の列, float16 埋め込み bytes, 使った実行環境)"""
        import numpy as np
        from PIL import Image
        from s2_crops import crop
        im = Image.open(io.BytesIO(jpg)).convert("RGB")
        cs = [crop(im, box) for _, box in items]
        E = np.concatenate([self.emb(cs[k:k + 64]) for k in range(0, len(cs), 64)])
        return stem, [i for i, _ in items], E.astype(np.float16).tobytes(), self.provider
