"""③ 「商品らしいもの」1クラスの YOLO を、① で採用した枠（誤検出を除く）を正解として学習し、ONNX に書き出す。

  .venv/Scripts/python products/s4_train.py [--epochs 40] [--imgsz 960] [--model yolo11n.pt]
出力: work/yolo/（データセット）, work/train/<run>/（学習経過 results.csv）, work/detector.onnx
"""
import argparse, os, re, shutil, zlib

from common import WORK, config, log, read_json, run_stage, status, write_json
from s2_crops import accepted


def group(name: str) -> str:
    """動画から切り出した連番フレーム（IMG_8490_f00150 等）は同じ動画をまとめて train/val に振り分ける（リーク防止）"""
    return re.sub(r"_f\d+$", "", name)


def build_dataset():
    thr = config()["det_threshold"]
    rejects = read_json(WORK / "rejects.json", {})
    root = WORK / "yolo"
    shutil.rmtree(root, ignore_errors=True)
    split = {"train": [], "val": []}
    n_box = 0
    for p in sorted((WORK / "det").glob("*.json")):
        d, stem = read_json(p), p.stem
        sp = "val" if zlib.crc32(group(stem).encode()) % 10 == 0 else "train"
        (root / "images" / sp).mkdir(parents=True, exist_ok=True)
        (root / "labels" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy(WORK / "view" / f"{stem}.jpg", root / "images" / sp / f"{stem}.jpg")  # 長辺1280で十分（学習は960）
        lines = []
        for i in accepted(d, stem, thr, rejects):  # 小さい枠も正解に残す（外すと「小さいものは商品でない」と覚える）
            x0, y0, x1, y1 = d["boxes"][i]
            lines.append(f"0 {(x0 + x1) / 2:.5f} {(y0 + y1) / 2:.5f} {x1 - x0:.5f} {y1 - y0:.5f}")
        (root / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines))
        split[sp].append(stem)
        n_box += len(lines)
    (root / "data.yaml").write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: product\n")
    write_json(root / "split.json", split)
    log("train", f"データセット: train {len(split['train'])} 枚 / val {len(split['val'])} 枚, 枠 {n_box} 個", threshold=thr)
    return root / "data.yaml"


def train_modal(a):
    """work/yolo を Volume に上げて GPU で学習。エポックごとの結果を results.csv に書き足す（ダッシュボードの学習曲線用）"""
    import modal
    from gpu_train import app, train, vol
    run = f"{a.model.split('.')[0]}_{a.imgsz}_gpu"
    out = WORK / "train" / run
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.csv").unlink(missing_ok=True)
    status("train", phase="データを Modal に送信", note=f"{a.model} imgsz={a.imgsz} epochs={a.epochs} (Modal GPU)")
    with vol.batch_upload(force=True) as up:
        up.put_directory(str(WORK / "yolo"), "/yolo")
    keys = None
    with modal.enable_output(), app.run():
        for x in train.remote_gen(a.model, a.imgsz, a.epochs, a.batch, run):
            if "epoch" in x:
                m = {"epoch": x["epoch"], **x["metrics"]}
                if keys is None:
                    keys = list(m)
                    (out / "results.csv").write_text(",".join(keys) + "\n", encoding="utf-8")
                with open(out / "results.csv", "a", encoding="utf-8") as f:
                    f.write(",".join(str(m.get(k, "")) for k in keys) + "\n")
                status("train", done=x["epoch"], total=a.epochs, phase="学習 (Modal GPU)",
                       mAP50=round(m.get("metrics/mAP50(B)", 0), 3))
            else:
                (out / "best.pt").write_bytes(x["best"])
                (out / "results_full.csv").write_text(x["csv"], encoding="utf-8")
                (out / "model.onnx").write_bytes(x["onnx"])
    # 学習ごとに train/<run>/model.onnx に残し、推論で使うものを detector.json で指す（前のモデルと比べられるように）
    write_json(WORK / "detector.json", {"imgsz": a.imgsz, "run": run, "onnx": f"train/{run}/model.onnx"})
    log("train", f"ONNX を書き出し: work/train/{run}/model.onnx（推論で使用）", run=run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--modal", action="store_true", help="Modal の GPU で学習する（CPU だと数時間）")
    a = ap.parse_args()
    data = build_dataset()
    if a.modal:
        return train_modal(a)
    os.chdir(WORK)  # ultralytics は事前学習の重みをカレントに落とす
    from ultralytics import YOLO
    run = f"{a.model.split('.')[0]}_{a.imgsz}"
    status("train", total=a.epochs, done=0, note=f"{a.model} imgsz={a.imgsz} epochs={a.epochs} (CPU)")
    model = YOLO(a.model)
    model.add_callback("on_fit_epoch_end", lambda tr: status("train", done=tr.epoch + 1,
                                                                mAP50=round(float(tr.metrics.get("metrics/mAP50(B)", 0)), 3)))
    model.train(data=str(data), imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device="cpu", workers=4,
                project=str(WORK / "train"), name=run, exist_ok=True, plots=False, max_det=300)
    best = YOLO(str(WORK / "train" / run / "weights" / "best.pt"))
    onnx = best.export(format="onnx", imgsz=a.imgsz, simplify=True, opset=17)
    shutil.copy(onnx, WORK / "detector.onnx")
    write_json(WORK / "detector.json", {"imgsz": a.imgsz, "run": run})
    log("train", f"ONNX を書き出し: work/detector.onnx", run=run)


if __name__ == "__main__":
    run_stage("train", main)
