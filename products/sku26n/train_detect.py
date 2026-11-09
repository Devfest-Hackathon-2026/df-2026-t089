"""Fine-tune sku110k-2 on a snapshot of the ongoing Gemini annotations.

Run from the repository root:
  .venv/Scripts/python products/sku26n/train_detect.py [--prepare-only]
The train/validation split is by source photo. Artifacts live in products/work/sku26n/.
"""
import argparse
import json
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "products/work/sku26n"
VIEW = ROOT / "products/work/view"
TAXONOMY = ROOT / "products/konbinidb/taxonomy.json"
MIN_DET_CONF = 0.25  # label_all.py only sends these detections to Gemini


def snapshot(accepted_only: bool = False):
    data = (OUT / "label_all.jsonl").read_bytes()
    # A concurrent annotator may be writing its last line.
    if not data.endswith(b"\n"):
        data = data.rsplit(b"\n", 1)[0] + b"\n"
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    det = {item["name"]: item for item in json.loads((OUT / "det.json").read_text(encoding="utf-8"))["images"]}
    cats = sorted(k for group in json.loads(TAXONOMY.read_text(encoding="utf-8")).values() for k in group)
    names = cats + ["UNS"]
    ids = {cat: i for i, cat in enumerate(names)}
    by_image = defaultdict(dict)
    for row in rows:
        name, number = row["image"], row["box"]
        if name not in det or not isinstance(number, int) or number < 0 or number >= len(det[name]["boxes"]):
            raise ValueError(f"Unknown image or box: {name} #{number}")
        if number in by_image[name]:
            raise ValueError(f"Duplicate Gemini annotation: {name} #{number}")
        by_image[name][number] = row
    eligible = {}
    for name, boxes in by_image.items():
        expected = {i for i, b in enumerate(det[name]["boxes"]) if b[4] >= MIN_DET_CONF}
        if set(boxes) != expected:
            continue  # only use fully annotated photos
        labels = []
        for number, row in boxes.items():
            status = row["status"]
            if status == "ok":
                cat = row["cat"]
                if cat not in ids:
                    raise ValueError(f"Invalid category: {name} #{number}: {cat}")
            elif status in ("unsure", "multiple"):
                if accepted_only:
                    continue
                cat = "UNS"  # keep uncertain products as objects, not background
            elif status == "not_product":
                continue
            else:
                raise ValueError(f"Invalid Gemini status: {status}")
            x0, y0, x1, y1 = row["xyxy"]
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError(f"Invalid box: {name} #{number}")
            labels.append(f"{ids[cat]} {(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}")
        if labels:
            if not (VIEW / name).is_file():
                raise ValueError(f"Missing source photo: {name}")
            eligible[name] = labels
    if len(eligible) < 300:
        raise ValueError(f"Only {len(eligible)} usable photos; expected at least 300")
    return eligible, names, len(rows)


def prepare(accepted_only: bool = False):
    eligible, names, source_rows = snapshot(accepted_only)
    photos = sorted(eligible)
    random.Random(20261110).shuffle(photos)
    val = set(photos[:max(1, round(len(photos) * 0.15))])
    archive = OUT / "gemini_det_train.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for name in photos:
            split = "val" if name in val else "train"
            z.write(VIEW / name, f"{split}/images/{name}")
            z.writestr(f"{split}/labels/{Path(name).stem}.txt", "\n".join(eligible[name]) + "\n")
        yaml = {"path": "/tmp/gemini_det", "train": "train/images", "val": "val/images",
                "nc": len(names), "names": names}
        z.writestr("data.yaml", json.dumps(yaml))  # JSON is valid YAML
    manifest = {"source_rows": source_rows, "photos": len(photos), "train_photos": len(photos) - len(val),
                "val_photos": len(val), "boxes": sum(map(len, eligible.values())), "classes": len(names),
                "accepted_only": accepted_only,
                "validation_images": sorted(val)}
    (OUT / "gemini_det_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Snapshot: {manifest['photos']} photos, {manifest['boxes']} boxes, "
          f"{manifest['classes']} classes ({manifest['val_photos']} validation photos); "
          f"{archive.stat().st_size / 1e6:.1f} MB", flush=True)
    return archive


def prepare_views():
    archive = OUT / "gemini_det_view.zip"
    names = [item["name"] for item in json.loads((OUT / "det.json").read_text(encoding="utf-8"))["images"]]
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for name in names:
            z.write(VIEW / name, f"images/{name}")
        z.writestr("names.json", json.dumps(names))
    print(f"Inference set: {len(names)} photos; {archive.stat().st_size / 1e6:.1f} MB", flush=True)
    return archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", type=Path, default=OUT / "weights/sku110k-2.pt",
                    help="Initial checkpoint; use sku110k-2-gemini.pt to continue training")
    ap.add_argument("--accepted-only", action="store_true",
                    help="Train only on Gemini-accepted categories, ignoring unsure boxes")
    args = ap.parse_args()
    archive = None if args.skip_train else prepare(args.accepted_only)
    if args.prepare_only:
        return
    import modal  # noqa: F401
    from gpu_detect import app, predict, train, volume

    with volume.batch_upload(force=True) as upload:
        if archive:
            upload.put_file(str(archive), "/gemini_det_train.zip")
            upload.put_file(str(args.weights), "/sku110k-2.pt")
        upload.put_file(str(prepare_views()), "/gemini_det_view.zip")
    started = time.time()
    with app.run():
        if not args.skip_train:
            for event in train.remote_gen(args.epochs, args.imgsz, args.batch):
                if event["type"] == "epoch":
                    print(f"EPOCH {event['epoch']}/{args.epochs} elapsed={time.time()-started:.0f}s "
                          f"metrics={event['metrics']}", flush=True)
                elif event["type"] == "done":
                    (OUT / "sku110k-2-gemini.pt").write_bytes(event["weights"])
                    (OUT / "gemini_det_results.csv").write_text(event["csv"], encoding="utf-8")
                    print("Training complete; saved weights and validation results", flush=True)
                else:
                    print(event, flush=True)
        for event in predict.remote_gen(args.imgsz):
            if event["type"] == "progress":
                print(f"PREDICTED {event['images']} photos, {event['boxes']} boxes", flush=True)
            else:
                path = OUT / "pred_trained.json"
                path.write_text(event["result"], encoding="utf-8")
                print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
