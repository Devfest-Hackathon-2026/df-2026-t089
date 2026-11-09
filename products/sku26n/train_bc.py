"""Prepare and run direct B/C detector training from the original sku110k-2 weights.

Both variants share the same annotation snapshot and photo split. Unknown product
regions are passed to the GPU loss as ignore boxes, never as negative examples.
"""
import argparse
import hashlib
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "products/work/sku26n"
VIEW = ROOT / "products/work/view"
TAXONOMY = ROOT / "products/konbinidb/taxonomy.json"
WEIGHTS = OUT / "weights/sku110k-2.pt"

# Thirty groups, each a disjoint union of the source taxonomy's 122 classes.
COARSE = {
    "G01 飲料・茶・水": "001 002 012 013 104",
    "G02 飲料・コーヒー": "006 007 072",
    "G03 飲料・炭酸・スポーツ": "008 009 010 011 121",
    "G04 飲料・果汁・乳製品": "003 004 005 014 015 016 017 018 019",
    "G05 飲料・栄養・その他": "020 122 123",
    "G06 酒・ビール・チューハイ": "101 102 127 128",
    "G07 酒・日本酒・洋酒": "103 129 130 131",
    "G08 菓子・スナック": "021 022 023 033 039",
    "G09 菓子・チョコ・飴": "024 025 026 027 028 029",
    "G10 菓子・焼き菓子": "030 031 034 035",
    "G11 菓子・ナッツ・栄養": "037 038 040",
    "G12 パン・おにぎり・弁当": "041 042 043",
    "G13 パン・サンドイッチ": "044 045 046",
    "G14 パン・菓子パン": "048 049 050 051",
    "G15 麺・調理済み": "055 056 057 058 059 060",
    "G16 即席・カップ麺": "061 062 064",
    "G17 即席・スープ・カレー": "065 066 067 068",
    "G18 即席・米・朝食": "070 071 074 075",
    "G19 即席・飲料・調味": "073 114 115 124 125",
    "G20 デザート・洋菓子": "076 077 078 079 080",
    "G21 デザート・和菓子・アイス": "081 082 083 084",
    "G22 生鮮・青果・サラダ": "085 086 087 088 089 090",
    "G23 惣菜・ホットスナック": "105 106 107 108 109 110",
    "G24 惣菜・食材": "111 112 113 116",
    "G25 日用品・衛生・紙": "091 092",
    "G26 日用品・洗面・ヘア": "094 095 119",
    "G27 日用品・化粧・美容": "096 097",
    "G28 日用品・電池・文具": "098 099",
    "G29 日用品・衣料・雑貨": "100 117 118 126",
    "G30 日用品・ペット": "120",
}


def mapping():
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    fine = sorted(code for group in taxonomy.values() for code in group)
    coarse = {}
    for group, members in COARSE.items():
        for code in members.split():
            if code in coarse:
                raise ValueError(f"Duplicate coarse category: {code}")
            coarse[code] = group
    if set(coarse) != set(fine) or len(COARSE) != 30:
        raise ValueError(f"Coarse mapping mismatch: missing={set(fine)-set(coarse)}, extra={set(coarse)-set(fine)}")
    return fine, coarse


def prepare():
    fine, coarse = mapping()
    source = (OUT / "label_all.jsonl").read_bytes()
    if not source.endswith(b"\n"):
        source = source.rsplit(b"\n", 1)[0] + b"\n"
    rows = [json.loads(line) for line in source.splitlines() if line]
    det = {im["name"]: im for im in json.loads((OUT / "det.json").read_text(encoding="utf-8"))["images"]}
    by_image = defaultdict(dict)
    for row in rows:
        name, box = row["image"], row["box"]
        if name not in det or not 0 <= box < len(det[name]["boxes"]) or box in by_image[name]:
            raise ValueError(f"Invalid or duplicate annotation: {name} #{box}")
        by_image[name][box] = row
    eligible = {}
    for name, annotated in by_image.items():
        expected = {i for i, b in enumerate(det[name]["boxes"]) if b[4] >= .25}
        if set(annotated) != expected:
            continue
        accepted, ignored = [], []
        for i, box in enumerate(det[name]["boxes"]):
            row = annotated.get(i)
            if row is None:
                ignored.append(box[:4])  # below the Gemini annotation threshold
            elif row["status"] == "ok":
                if row["cat"] not in fine:
                    raise ValueError(f"Unknown category {row['cat']}: {name} #{i}")
                accepted.append((row["cat"], row["xyxy"]))
            elif row["status"] in {"unsure", "multiple"}:
                ignored.append(row["xyxy"])
            elif row["status"] != "not_product":
                raise ValueError(f"Unknown status {row['status']}: {name} #{i}")
        if accepted:
            eligible[name] = (accepted, ignored)
    if len(eligible) < 300:
        raise ValueError(f"Only {len(eligible)} complete photos")
    names = sorted(eligible)
    random.Random(20261110).shuffle(names)
    val = set(names[:round(len(names) * .15)])
    fine_ids = {code: i for i, code in enumerate(fine)}
    groups = list(COARSE)
    coarse_ids = {name: i for i, name in enumerate(groups)}
    archive = OUT / "bc_train.zip"
    counts = Counter()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for name in names:
            split = "val" if name in val else "train"
            with Image.open(VIEW / name) as original:
                w, h = original.size
                scale = min(960 / w, 960 / h)
                nw, nh = round(w * scale), round(h * scale)
                left, top = (960 - nw) // 2, (960 - nh) // 2
                canvas = Image.new("RGB", (960, 960), (114, 114, 114))
                canvas.paste(original.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
                output = BytesIO()
                canvas.save(output, "JPEG", quality=90)
                z.writestr(f"{split}/images/{name}", output.getvalue())
            def place(box):
                x0, y0, x1, y1 = box
                return [(left + x0 * nw) / 960, (top + y0 * nh) / 960,
                        (left + x1 * nw) / 960, (top + y1 * nh) / 960]
            accepted, ignored = eligible[name]
            labels = {"b": [], "c": []}
            for code, box in accepted:
                x0, y0, x1, y1 = place(box)
                coords = f"{(x0+x1)/2:.6f} {(y0+y1)/2:.6f} {x1-x0:.6f} {y1-y0:.6f}"
                labels["b"].append(f"{fine_ids[code]} {coords}")
                labels["c"].append(f"{coarse_ids[coarse[code]]} {coords}")
                counts["accepted"] += 1
            for variant in ("b", "c"):
                z.writestr(f"{variant}/{split}/labels/{Path(name).stem}.txt", "\n".join(labels[variant]) + "\n")
            z.writestr(f"ignore/{split}/{Path(name).stem}.json", json.dumps([place(b) for b in ignored]))
            counts["ignored"] += len(ignored)
        for variant, class_names in (("b", fine), ("c", groups)):
            z.writestr(f"{variant}.yaml", json.dumps({"path": f"/tmp/bc/{variant}",
                "train": "train/images", "val": "val/images", "nc": len(class_names), "names": class_names}))
        z.writestr("coarse.json", json.dumps({"groups": COARSE, "mapping": coarse}, ensure_ascii=False))
    manifest = {"source_sha256": hashlib.sha256(source).hexdigest(), "source_rows": len(rows),
                "photos": len(names), "train_photos": len(names)-len(val), "val_photos": len(val),
                "accepted_boxes": counts["accepted"], "ignored_boxes": counts["ignored"],
                "fine_classes": len(fine), "coarse_classes": len(groups),
                "validation_images": sorted(val), "checkpoint_sha256": hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()}
    (OUT / "bc_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {manifest['photos']} photos, {counts['accepted']} accepted boxes, {counts['ignored']} ignored boxes; {archive.stat().st_size/1e6:.1f} MB", flush=True)
    return archive


def run(args):
    archive = None if args.reuse_upload else prepare()
    if args.prepare_only:
        return
    from gpu_bc import app, predict, train, volume
    from train_detect import prepare_views
    if not args.reuse_upload:
        with volume.batch_upload(force=True) as upload:
            upload.put_file(str(archive), "/bc_train.zip")
            upload.put_file(str(WEIGHTS), "/bc_original.pt")
            upload.put_file(str(prepare_views()), "/gemini_det_view.zip")
    start = time.time()
    with app.run():
        for variant in args.variants:
            name = variant + args.tag
            for event in train.remote_gen(variant, args.epochs, args.batch, args.lr0, args.tag):
                if event["type"] == "epoch":
                    print(f"{variant.upper()} epoch {event['epoch']}/{args.epochs} {time.time()-start:.0f}s {event['metrics']}", flush=True)
                elif event["type"] == "done":
                    (OUT / f"sku110k-2-{name}.pt").write_bytes(event["weights"])
                    (OUT / f"bc_{name}_results.csv").write_text(event["csv"], encoding="utf-8")
                    print(f"{variant.upper()} training complete", flush=True)
                else:
                    print(event, flush=True)
            for event in predict.remote_gen(variant, args.tag):
                if event["type"] == "progress":
                    print(f"{variant.upper()} predicted {event['images']} photos, {event['boxes']} boxes", flush=True)
                elif event["type"] == "done":
                    path = OUT / f"pred_{name}.json"
                    path.write_text(event["result"], encoding="utf-8")
                    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reuse-upload", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--variants", default="bc")
    parser.add_argument("--lr0", type=float, help="Explicit AdamW learning rate instead of optimizer=auto")
    parser.add_argument("--tag", default="", help="Suffix for output names, so experiments keep earlier results")
    run(parser.parse_args())
