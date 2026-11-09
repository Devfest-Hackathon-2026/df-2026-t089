"""YOLO26n-cls と MobileCLIP-S0 を、加工済みのコンビニDB と合格した店内の切り抜きで学習し、ダッシュボードの結果を作る。

  .venv/Scripts/python products/sku26n/train_two_models.py [--epochs 25] [--prepare-only]
店内の切り抜きの 20% は学習に使わず、実際の棚の画像での正解率を測る。出力は git に入らない products/work/sku26n/。
"""
import argparse, json, random, shutil, time, zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "products" / "work"
OUT = WORK / "sku26n"
AUG = WORK / "konbinidb" / "train_aug"
VIEW = WORK / "view"
HOLD_OUT = 0.2
SEED = 20261110


def gemini_store_rows() -> tuple[list[dict], dict]:
    """Gemini が確定した全部の枠。検証は B/C と同じ写真にして、検出器と同じ条件で比べる"""
    held = set(json.loads((OUT / "bc_manifest.json").read_text(encoding="utf-8"))["validation_images"])
    rows, files = [], {}
    for line in (OUT / "label_all.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] != "ok":
            continue
        cid = f"{Path(r['image']).stem}_{r['box']}"
        path = f"images/crop/{cid}.jpg"
        files[path] = (VIEW / r["image"], r["xyxy"])
        rows.append({"source": "crop", "id": cid, "cat": r["cat"], "path": path, "split": "val" if r["image"] in held else "train", "metric": r["image"] in held})
    return rows, files


def reviewed_store_rows() -> tuple[list[dict], dict]:
    selected = set(json.loads((OUT / "selected_crops.json").read_text(encoding="utf-8")))
    verified = {r["id"]: r for r in json.loads((OUT / "verify_all.json").read_text(encoding="utf-8"))["rows"] if r.get("pass")}
    rows, files = [], {}
    by_cat = defaultdict(list)
    for cid in sorted(selected & verified.keys()):
        by_cat[verified[cid]["cat"]].append(cid)
    rng = random.Random(SEED)
    held = set()
    for cat, ids in by_cat.items():
        rng.shuffle(ids)
        held.update(ids[:round(len(ids) * HOLD_OUT) if len(ids) >= 2 else 0])
    for cat, ids in sorted(by_cat.items()):
        for cid in sorted(ids):
            path = f"images/crop/{cid}.jpg"
            files[path] = WORK / "crops" / f"{cid}.jpg"
            rows.append({"source": "crop", "id": cid, "cat": cat, "path": path, "split": "val" if cid in held else "train", "metric": cid in held})
    return rows, files


def build_manifest(gemini: bool = False) -> tuple[list[dict], dict]:
    aug = json.loads((AUG / "manifest.json").read_text(encoding="utf-8"))
    rows, files = gemini_store_rows() if gemini else reviewed_store_rows()
    for r in aug["rows"]:
        for n in range(aug["variants"]):
            path = f"images/db/{r['id']}_{n}.jpg"
            files[path] = AUG / f"{r['id']}_{n}.jpg"
            rows.append({"source": "db", "id": f"{r['id']}_{n}", "cat": r["cat"], "path": path, "split": "train", "metric": False})
    # ultralytics は train と val のカテゴリが一致している必要があるので、val が無いカテゴリに学習用の 1 枚を重ねる（正解率には数えない）
    with_val = {r["cat"] for r in rows if r["split"] == "val"}
    first_train = {}
    for r in rows:
        if r["split"] == "train":
            first_train.setdefault(r["cat"], r)
    rows += [{**first_train[c], "split": "val", "metric": False} for c in sorted(first_train.keys() - with_val)]
    missing = [p for p, f in files.items() if not (f[0] if isinstance(f, tuple) else f).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files are missing, for example {missing[:3]}")
    return rows, files


def prepare(gemini: bool = False, tag: str = "") -> tuple[Path, Path]:
    rows, files = build_manifest(gemini)
    archive = OUT / f"two_models_train{tag}.zip"
    source, im = None, None
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
        for path, file in files.items():
            if not isinstance(file, tuple):
                z.write(file, path)
                continue
            if file[0] != source:  # 同じ写真の枠は続いて並ぶので、開き直さない
                source = file[0]
                with Image.open(source) as opened:
                    im = opened.convert("RGB")
            w, h = im.size
            x0, y0 = max(0, min(w - 1, int(file[1][0] * w))), max(0, min(h - 1, int(file[1][1] * h)))  # classify_boxes と同じ切り方
            buffer = BytesIO()
            im.crop((x0, y0, max(x0 + 1, min(w, int(file[1][2] * w))), max(y0 + 1, min(h, int(file[1][3] * h))))).save(buffer, "JPEG", quality=92)
            z.writestr(path, buffer.getvalue())
        z.writestr("manifest.json", json.dumps(rows, separators=(",", ":")))
    assert len(z.namelist()) == len(set(z.namelist())) == len(files) + 1, "同じ枠が重複している"
    train = sum(r["split"] == "train" for r in rows)
    print(f"{archive}: {archive.stat().st_size / 1e6:.0f} MB, train {train}, held out {sum(r['metric'] for r in rows)}, classes {len({r['cat'] for r in rows})}", flush=True)
    view = OUT / "cls_view.zip"
    if not view.exists():
        with zipfile.ZipFile(view, "w", compression=zipfile.ZIP_STORED) as z:
            for path in sorted((WORK / "view").glob("*.jpg")):
                z.write(path, f"images/{path.name}")
            z.write(OUT / "det.json", "det.json")
    return archive, view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--clip-epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--skip-prepare", action="store_true", help="作成済みの two_models_train.zip をそのまま使う")
    ap.add_argument("--gemini", action="store_true", help="店内の切り抜きに Gemini が確定した全部の枠を使う")
    ap.add_argument("--tag", default="", help="出力名の接尾辞。以前の結果を残す")
    a = ap.parse_args()
    archive, view = (OUT / f"two_models_train{a.tag}.zip", OUT / "cls_view.zip") if a.skip_prepare else prepare(a.gemini, a.tag)
    if a.prepare_only:
        return
    from gpu_two_models import app, train_both, volume
    print("Uploading...", flush=True)
    with volume.batch_upload(force=True) as up:
        up.put_file(str(archive), "/train.zip")
        up.put_file(str(view), "/view.zip")
    metrics, started = {}, time.time()
    with app.run():
        for e in train_both.remote_gen(a.epochs, a.imgsz, a.batch, a.clip_epochs):
            kind = e["type"]
            if kind == "yolo_done":
                (OUT / f"yolo26n-cls-aug{a.tag}.pt").write_bytes(e["weights"])
                (OUT / f"yolo26n_cls_aug{a.tag}_results.csv").write_text(e["csv"], encoding="utf-8")
                metrics["yolo26n-cls"] = e["metrics"]
                print(f"YOLO store crops: {e['metrics']}", flush=True)
            elif kind == "clip_done":
                (OUT / f"mobileclip_s0_head{a.tag}.pt").write_bytes(e["head"])
                metrics["mobileclip-s0"] = e["metrics"]
                print(f"CLIP store crops: {e['metrics']}", flush=True)
            elif kind == "pred":
                path = OUT / f"pred_{e['model']}{a.tag}.json"
                if path.exists() and e["model"] == "m1" and not (OUT / "pred_m1_before_aug.json").exists():
                    shutil.copy(path, OUT / "pred_m1_before_aug.json")
                path.write_text(e["json"], encoding="utf-8")
                print(f"Saved {path.name}", flush=True)
            else:
                print(f"{time.time() - started:.0f}s {json.dumps(e, ensure_ascii=False)[:220]}", flush=True)
            (OUT / f"two_models_metrics{a.tag}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DONE", json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
