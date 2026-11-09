"""Modal GPU training and inference for the Gemini-labeled sku110k-2 detector."""
import json
import queue
import threading
import zipfile
from pathlib import Path

import modal


volume = modal.Volume.from_name("convenience-products-gemini-det", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("ultralytics==8.4.202"))
app = modal.App("convenience-products-gemini-det", image=image)


@app.function(gpu=["L40S", "A10G", "L4"], volumes={"/data": volume},
              timeout=4 * 3600, cpu=8, memory=32768)
def train(epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO

    root = Path("/tmp/gemini_det")
    root.mkdir()
    with zipfile.ZipFile("/data/gemini_det_train.zip") as archive:
        archive.extractall(root)
    events = queue.Queue()
    errors = []
    model = YOLO("/data/sku110k-2.pt")

    def epoch_end(trainer):
        events.put({"type": "epoch", "epoch": trainer.epoch + 1,
                    "metrics": {k: float(v) for k, v in trainer.metrics.items()
                                if isinstance(v, (int, float))}})

    model.add_callback("on_fit_epoch_end", epoch_end)

    def work():
        try:
            model.train(data=str(root / "data.yaml"), epochs=epochs, imgsz=imgsz,
                        batch=batch, device=0, workers=8, project="/tmp/runs",
                        name="gemini_det", exist_ok=True, plots=False, verbose=False,
                        patience=15, seed=20261110)
            run = Path("/tmp/runs/gemini_det")
            weights = (run / "weights/best.pt").read_bytes()
            Path("/data/sku110k-2-gemini.pt").write_bytes(weights)
            volume.commit()
            events.put({"type": "done", "weights": weights,
                        "csv": (run / "results.csv").read_text()})
        except Exception as exc:
            errors.append(repr(exc))
        events.put(None)

    threading.Thread(target=work).start()
    yield {"type": "started"}
    while (event := events.get()) is not None:
        yield event
    if errors:
        raise RuntimeError(errors[0])


@app.function(gpu=["L40S", "A10G", "L4"], volumes={"/data": volume},
              timeout=4 * 3600, cpu=8, memory=32768)
def predict(imgsz: int = 960):
    from ultralytics import YOLO

    root = Path("/tmp/gemini_det_view")
    root.mkdir()
    with zipfile.ZipFile("/data/gemini_det_view.zip") as archive:
        archive.extractall(root)
    names = json.loads((root / "names.json").read_text())
    model = YOLO("/data/sku110k-2-gemini.pt")
    images = {}
    total = 0
    for index, name in enumerate(names, 1):
        result = model.predict(str(root / "images" / name), imgsz=imgsz,
                               conf=0.05, max_det=1000, device=0, verbose=False)[0]
        boxes = []
        for xyxy, conf, cls in zip(result.boxes.xyxyn.tolist(), result.boxes.conf.tolist(),
                                   result.boxes.cls.tolist()):
            boxes.append([*[round(float(v), 4) for v in xyxy], round(float(conf), 4),
                          model.names[int(cls)]])
        images[name] = boxes
        total += len(boxes)
        if index % 50 == 0 or index == len(names):
            yield {"type": "progress", "images": index, "boxes": total}
    yield {"type": "done", "result": json.dumps({"images": images}, separators=(",", ":"))}
