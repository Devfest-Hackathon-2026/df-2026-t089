"""学習済みの YOLO26n-cls と MobileCLIP-S0 で Top-5 を出す（ダッシュボードのホバー表示と、精度の原因調査用）。

  .venv/Scripts/python products/sku26n/predict_top5.py
出力は git に入らない products/work/sku26n/:
  top5_m1.json / top5_clip.json   {画像名: [枠ごとの Top-5]}
  named_top5.json, named_emb.npy  商品名付きの切り抜きの Top-5 と MobileCLIP の埋め込み
  db_emb.npy, db_emb_rows.json    コンビニDB の加工画像の埋め込み
"""
import json, sys, zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dashboard import WORK, CROPS, named_crops  # noqa: E402  products/dashboard.py と同名なので先に読む
from gpu_two_models import app, predict_top5, volume  # noqa: E402

ARCHIVE = WORK / "named_crops.zip"


def main():
    ids = [r[0] for r in named_crops()]
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED) as z:
        for cid in ids:
            z.write(CROPS / f"{cid}.jpg", f"images/{cid}.jpg")
        z.writestr("ids.json", json.dumps(ids))
    print(f"{ARCHIVE.name}: {ARCHIVE.stat().st_size / 1e6:.0f} MB, {len(ids)} crops", flush=True)
    with volume.batch_upload(force=True) as up:
        up.put_file(str(ARCHIVE), "/named_crops.zip")
        up.put_file(str(WORK / "yolo26n-cls-aug.pt"), "/yolo26n-cls-aug.pt")
        up.put_file(str(WORK / "mobileclip_s0_head.pt"), "/mobileclip_s0_head.pt")
    with app.run():
        for e in predict_top5.remote_gen():
            if e["type"] == "top5":
                (WORK / f"top5_{e['model']}.json").write_text(e["json"], encoding="utf-8")
            elif e["type"] == "named":
                (WORK / "named_top5.json").write_text(e["json"], encoding="utf-8")
                np.save(WORK / "named_emb.npy", np.frombuffer(e["emb"], np.float16).reshape(len(ids), -1))
            elif e["type"] == "db_emb":
                (WORK / "db_emb_rows.json").write_text(json.dumps(e["rows"]), encoding="utf-8")
                np.save(WORK / "db_emb.npy", np.frombuffer(e["emb"], np.float16).reshape(len(e["rows"]), -1))
            print("saved", e["type"], e.get("model", ""), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
