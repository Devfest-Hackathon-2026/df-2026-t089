"""Export the COCO YOLO26n detector that the app uses to find people (class 0).

  python tools/export_person.py   # -> app/src/main/assets/person/yolo26n_416.onnx

The Ultralytics weights are AGPL-3.0.
"""
import os
import shutil
import tempfile
from pathlib import Path

from ultralytics import YOLO

OUT = Path(__file__).resolve().parent.parent / "app/src/main/assets/person/yolo26n_416.onnx"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as work:
        os.chdir(work)   # ultralytics downloads the weights into the current directory
        model = YOLO("yolo26n.pt")
        assert model.names[0] == "person"
        shutil.copyfile(model.export(format="onnx", imgsz=416, simplify=True, opset=17), OUT)
        os.chdir(OUT.parent)
    print(OUT)


if __name__ == "__main__":
    main()
