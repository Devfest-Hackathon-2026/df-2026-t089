"""③ YOLO の学習を Modal の GPU で行う部分。s4_train.py --modal から使う。
データセット（work/yolo）を Volume に上げ、向こうで学習し、エポックごとの結果を逐次返す（ダッシュボードに学習曲線を出すため）。
最後に best.pt と ONNX を返す。
"""
from pathlib import Path

import modal

GPU = ["L4", "A10G", "L40S", "T4"]  # 先頭から空いている GPU を使う
vol = modal.Volume.from_name("convenience-products-yolo", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("ultralytics", "onnx", "onnxslim", "onnxruntime"))
app = modal.App("convenience-products-train", image=image)


@app.function(gpu=GPU, volumes={"/data": vol}, timeout=4 * 3600, cpu=8, memory=16384)
def train(model: str, imgsz: int, epochs: int, batch: int, run: str):
    """学習しながら {"row": results.csv の1行} を返し、最後に {"best": bytes, "onnx": bytes, "csv": str} を返す"""
    import queue, threading
    from ultralytics import YOLO
    root = Path("/data/yolo")
    (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    q: queue.Queue = queue.Queue()
    m = YOLO(model)
    m.add_callback("on_fit_epoch_end", lambda tr: q.put({"epoch": tr.epoch + 1, "metrics": {
        k: float(v) for k, v in {**tr.label_loss_items(tr.tloss, prefix="train"), **tr.metrics}.items()}}))
    err = []

    def work():
        try:
            m.train(data=str(root / "data.yaml"), imgsz=imgsz, epochs=epochs, batch=batch, device=0, workers=8,
                    project="/tmp/runs", name=run, exist_ok=True, plots=False, max_det=600, cache=True)
        except Exception as e:  # 学習の失敗も呼び出し側に伝える
            err.append(repr(e))
        q.put(None)
    threading.Thread(target=work).start()
    while (x := q.get()) is not None:
        yield x
    if err:
        raise RuntimeError(err[0])
    out = Path("/tmp/runs") / run
    best = YOLO(str(out / "weights" / "best.pt"))
    onnx = best.export(format="onnx", imgsz=imgsz, simplify=True, opset=17)
    yield {"best": (out / "weights" / "best.pt").read_bytes(), "onnx": Path(onnx).read_bytes(),
           "csv": (out / "results.csv").read_text()}
