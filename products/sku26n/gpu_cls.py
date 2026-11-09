"""Modal GPU functions for YOLO26n classification training."""
import json
import os
import queue
import threading
import time
import zipfile
from pathlib import Path

import modal


volume = modal.Volume.from_name("convenience-products-cls", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("ultralytics==8.4.202"))
app = modal.App("convenience-products-cls", image=image)


def link_data(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for row in manifest:
        source = root / row["path"]
        for name in ("stage", "full"):
            split = row["split"] if name == "stage" else "train"
            target = root / name / split / row["cat"] / f"{row['source']}_{row['id']}{source.suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source)
    (root / "full" / "val").symlink_to(root / "stage" / "val", target_is_directory=True)
    return manifest


@app.function(gpu=["L4", "A10G", "L40S", "T4"], volumes={"/data": volume},
              timeout=4 * 3600, cpu=8, memory=16384)
def train(epochs: int, imgsz: int, batch: int):
    from ultralytics import YOLO

    root = Path("/tmp/cls_data")
    root.mkdir()
    with zipfile.ZipFile("/data/cls_data.zip") as z:
        z.extractall(root)
    manifest = link_data(root)
    q = queue.Queue()
    errors = []
    model = YOLO("yolo26n-cls.pt")
    def epoch_end(trainer):
        q.put({"type": "epoch", "epoch": trainer.epoch + 1,
               "metrics": {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (int, float))}})
    model.add_callback("on_fit_epoch_end", epoch_end)

    def work():
        try:
            model.train(data=str(root / "stage"), epochs=epochs, imgsz=imgsz, batch=batch,
                        device=0, workers=8, project="/tmp/runs", name="validation",
                        exist_ok=True, plots=False, verbose=False)
            # The final model sees every labeled image. Validation metrics come only
            # from the first phase, where validation images are held out.
            final = YOLO("/tmp/runs/validation/weights/best.pt")
            final.train(data=str(root / "full"), epochs=3, imgsz=imgsz, batch=batch,
                        device=0, workers=8, project="/tmp/runs", name="all_data",
                        exist_ok=True, plots=False, verbose=False, val=False,
                        lr0=0.001)
            weights = Path("/tmp/runs/all_data/weights/last.pt").read_bytes()
            Path("/data/yolo26n-cls-best.pt").write_bytes(weights)
            volume.commit()
            q.put({"type": "done", "weights": weights,
                   "csv": Path("/tmp/runs/validation/results.csv").read_text()})
        except Exception as exc:
            errors.append(repr(exc))
        q.put(None)

    threading.Thread(target=work).start()
    yield {"type": "started", "images": len(manifest), "epochs": epochs, "final_epochs": 3,
           "started_at": time.time()}
    while (event := q.get()) is not None:
        yield event
    if errors:
        raise RuntimeError(errors[0])


@app.function(gpu=["L4", "A10G", "L40S", "T4"], volumes={"/data": volume},
              timeout=4 * 3600, cpu=8, memory=16384)
def classify_views():
    from PIL import Image
    from ultralytics import YOLO

    root = Path("/tmp/view")
    root.mkdir()
    with zipfile.ZipFile("/data/cls_view.zip") as z:
        z.extractall(root)
    det = json.loads((root / "det.json").read_text())
    model = YOLO("/data/yolo26n-cls-best.pt")
    result = {}
    total = 0
    for index, item in enumerate(det["images"], 1):
        path = root / "images" / item["name"]
        boxes = item["boxes"]
        classified = []
        with Image.open(path) as image:
            width, height = image.size
            for start in range(0, len(boxes), 128):
                chunk = boxes[start:start + 128]
                crops = []
                for box in chunk:
                    x0 = max(0, min(width - 1, int(box[0] * width)))
                    y0 = max(0, min(height - 1, int(box[1] * height)))
                    x1 = max(x0 + 1, min(width, int(box[2] * width)))
                    y1 = max(y0 + 1, min(height, int(box[3] * height)))
                    crops.append(image.crop((x0, y0, x1, y1)).convert("RGB"))
                preds = model.predict(source=crops, imgsz=224, batch=len(crops),
                                      device=0, verbose=False)
                for box, pred in zip(chunk, preds):
                    class_id = int(pred.probs.top1)
                    classified.append(box[:5] + [model.names[class_id],
                                                   round(float(pred.probs.top1conf), 4)])
        result[item["name"]] = classified
        total += len(classified)
        if index % 25 == 0 or index == len(det["images"]):
            yield {"type": "progress", "images": index, "boxes": total}
    yield {"type": "done", "result": json.dumps({"images": result}, separators=(",", ":"))}
