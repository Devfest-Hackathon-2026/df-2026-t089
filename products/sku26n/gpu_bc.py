"""Modal GPU jobs for direct B/C detector training with region-level ignore loss."""
import json
import os
import queue
import threading
import zipfile
from pathlib import Path

import modal

volume = modal.Volume.from_name("convenience-products-gemini-det", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("ultralytics==8.4.202")
         .add_local_python_source("ignore_bc"))
app = modal.App("convenience-products-direct-bc", image=image)


@app.function(gpu=["L40S", "A10G", "L4"], volumes={"/data": volume},
              timeout=4 * 3600, cpu=8, memory=32768)
def train(variant: str, epochs: int, batch_size: int, lr0: float | None = None, tag: str = ""):
    from ultralytics import YOLO
    import ignore_bc
    from ignore_bc import IgnoreTrainer

    if variant not in {"b", "c"}:
        raise ValueError(variant)
    root = Path("/tmp/bc")
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile("/data/bc_train.zip") as archive:
        archive.extractall(root)
    for split in ("train", "val"):
        for choice in ("b", "c"):
            target = root / choice / split / "images"
            target.mkdir(parents=True, exist_ok=True)
            for source in (root / split / "images").glob("*.jpg"):
                os.link(source, target / source.name)
    ignore = {}
    for split in ("train", "val"):
        for path in (root / "ignore" / split).glob("*.json"):
            ignore[path.stem] = json.loads(path.read_text())

    ignore_bc.IGNORE_BY_STEM = ignore

    events = queue.Queue()
    errors = []
    model = YOLO("/data/bc_original.pt")
    def epoch_end(trainer):
        events.put({"type": "epoch", "epoch": trainer.epoch + 1,
                    "metrics": {k: float(v) for k, v in trainer.metrics.items()
                                if isinstance(v, (int, float))}})
    model.add_callback("on_fit_epoch_end", epoch_end)

    def work():
        try:
            model.train(data=str(root / f"{variant}.yaml"), trainer=IgnoreTrainer,
                        epochs=epochs, imgsz=960, batch=batch_size, device=0, workers=8,
                        project="/tmp/bc_runs", name=variant, exist_ok=True,
                        plots=False, verbose=False, patience=12, seed=20261110,
                        mosaic=0, mixup=0, copy_paste=0, degrees=0, translate=0,
                        scale=0, shear=0, perspective=0, fliplr=0, flipud=0,
                        cls_pw=0.5, max_det=1000,
                        # optimizer=auto ignores lr0 and picks 0.01/(4+nc), 7.9e-5 for 122 classes
                        **({"optimizer": "AdamW", "lr0": lr0} if lr0 else {}))
            run = Path("/tmp/bc_runs") / variant
            weights = (run / "weights/best.pt").read_bytes()
            Path(f"/data/bc_{variant}{tag}.pt").write_bytes(weights)
            volume.commit()
            events.put({"type": "done", "weights": weights,
                        "csv": (run / "results.csv").read_text()})
        except Exception as exc:
            errors.append(repr(exc))
        events.put(None)

    threading.Thread(target=work).start()
    yield {"type": "started", "variant": variant}
    while (event := events.get()) is not None:
        yield event
    if errors:
        raise RuntimeError(errors[0])


@app.function(gpu=["L40S", "A10G", "L4"], volumes={"/data": volume},
              timeout=4 * 3600, cpu=8, memory=32768)
def predict(variant: str, tag: str = ""):
    from ultralytics import YOLO
    if variant not in {"b", "c"}:
        raise ValueError(variant)
    root = Path("/tmp/bc_view")
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile("/data/gemini_det_view.zip") as archive:
        archive.extractall(root)
    names = json.loads((root / "names.json").read_text())
    model = YOLO(f"/data/bc_{variant}{tag}.pt")
    images = {}
    total = 0
    for index, name in enumerate(names, 1):
        result = model.predict(str(root / "images" / name), imgsz=960,
                               conf=.05, max_det=1000, device=0, verbose=False)[0]
        boxes = []
        for xyxy, conf, cls in zip(result.boxes.xyxyn.tolist(), result.boxes.conf.tolist(),
                                   result.boxes.cls.tolist()):
            boxes.append([*[round(float(v), 4) for v in xyxy], round(float(conf), 4),
                          model.names[int(cls)]])
        images[name] = boxes
        total += len(boxes)
        if index % 50 == 0 or index == len(names):
            yield {"type": "progress", "images": index, "boxes": total}
    yield {"type": "done", "result": json.dumps({"images": images}, separators=(",", ":"), ensure_ascii=False)}
