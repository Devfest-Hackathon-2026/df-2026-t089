"""Train YOLO26n-cls on labeled store crops and KonbiniDB product images.

Run from the repository root: .venv/Scripts/python products/sku26n/train_cls.py
Artifacts are written under products/work/sku26n/ (ignored by Git).
"""
import argparse
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "products" / "work"
OUT = WORK / "sku26n"


def records():
    taxonomy = json.loads((ROOT / "products/konbinidb/taxonomy.json").read_text(encoding="utf-8"))
    valid = {code for group in taxonomy.values() for code in group}
    selected = set(json.loads((OUT / "selected_crops.json").read_text(encoding="utf-8")))
    judged = {row["id"]: row for row in json.loads((OUT / "verify_all.json").read_text(encoding="utf-8"))["rows"]}
    db_labels = {}
    for line in (WORK / "konbinidb/labels.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        db_labels[row["id"]] = row["cat"]
    result = []
    for cid in sorted(selected):
        row = judged[cid]
        if row.get("pass") is not True:
            raise ValueError(f"Unapproved store crop: {cid}")
        result.append(("crop", cid, row["cat"], WORK / "crops" / f"{cid}.jpg"))
    for path in sorted((WORK / "konbinidb/images").glob("*.webp")):
        if path.stem in db_labels and db_labels[path.stem] != "000":
            result.append(("db", path.stem, db_labels[path.stem], path))
    if len(result) != 11847 or len(selected) != 1850 or len(result) - len(selected) != 9997:
        raise ValueError(f"Unexpected source counts: total={len(result)}, crops={len(selected)}")
    bad = [(source, cid, cat) for source, cid, cat, path in result if cat not in valid or not path.is_file()]
    if bad:
        raise ValueError(f"Invalid category or missing image: {bad[:10]}")
    return result


def prepare():
    rows = records()
    by_cat = defaultdict(list)
    for row in rows:
        by_cat[row[2]].append(row)
    rng = random.Random(20261109)
    validation = set()
    for group in by_cat.values():
        rng.shuffle(group)
        validation.update((r[0], r[1]) for r in group[: min(len(group) - 1, max(1, round(len(group) * .1)))])
    manifest = [{"source": source, "id": cid, "cat": cat,
                 "path": f"images/{source}/{cid}{path.suffix.lower()}",
                 "split": "val" if (source, cid) in validation else "train"}
                for source, cid, cat, path in rows]
    archive = OUT / "cls_data.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for i, ((source, cid, cat, path), entry) in enumerate(zip(rows, manifest), 1):
            z.write(path, entry["path"])
            if i % 2000 == 0:
                print(f"Packed {i}/{len(rows)} images", flush=True)
        z.writestr("manifest.json", json.dumps(manifest, separators=(",", ":")))
    (OUT / "cls_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    (OUT / "selected_db.json").write_text(json.dumps(sorted(r[1] for r in rows if r[0] == "db")), encoding="utf-8")
    print(f"{archive}: {archive.stat().st_size / 1e6:.1f} MB, {len(rows)} images, "
          f"{len(by_cat)} classes, {len(validation)} validation", flush=True)
    return archive


def prepare_views():
    archive = OUT / "cls_view.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for path in sorted((WORK / "view").glob("*.jpg")):
            z.write(path, f"images/{path.name}")
        z.write(OUT / "det.json", "det.json")
    print(f"{archive}: {archive.stat().st_size / 1e6:.1f} MB", flush=True)
    return archive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--prepare-views", action="store_true")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    if args.prepare_views:
        prepare_views()
        return
    if args.classify:
        import modal
        from gpu_cls import app, classify_views, volume
        archive = OUT / "cls_view.zip"
        if not archive.exists():
            prepare_views()
        with volume.batch_upload(force=True) as upload:
            upload.put_file(str(archive), "/cls_view.zip")
        with app.run():
            for event in classify_views.remote_gen():
                if event["type"] == "progress":
                    print(f"CLASSIFIED {event['images']}/971 images, {event['boxes']} boxes", flush=True)
                else:
                    result = json.loads(event["result"])
                    path = OUT / "pred_m1.json"
                    path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
                    print(f"Saved {path}", flush=True)
        return
    archive = OUT / "cls_data.zip" if args.skip_prepare else prepare()
    if args.prepare_only:
        return
    import modal
    from gpu_cls import app, train, volume
    if not args.skip_upload:
        print("Uploading training data to Modal...", flush=True)
        with volume.batch_upload(force=True) as upload:
            upload.put_file(str(archive), "/cls_data.zip")
    started = time.time()
    with app.run():
        for event in train.remote_gen(args.epochs, args.imgsz, args.batch):
            if event["type"] == "started":
                print("TRAINING_STARTED " + json.dumps(event), flush=True)
            elif event["type"] == "epoch":
                elapsed = time.time() - started
                remaining = elapsed / event["epoch"] * (args.epochs - event["epoch"])
                print(f"EPOCH {event['epoch']}/{args.epochs} elapsed={elapsed:.0f}s eta_seconds={remaining:.0f} "
                      f"metrics={event['metrics']}", flush=True)
            elif event["type"] == "done":
                (OUT / "yolo26n-cls-best.pt").write_bytes(event["weights"])
                (OUT / "cls_results.csv").write_text(event["csv"], encoding="utf-8")
                print(f"Saved model: {OUT / 'yolo26n-cls-best.pt'}", flush=True)


if __name__ == "__main__":
    main()
