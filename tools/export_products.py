"""商品認識ダッシュボード（products/）で作ったモデルと照合先を、アプリの assets に書き出す。

  python tools/export_products.py --work <products/work のパス>

書き出すもの（app/src/main/assets/products/）
  detector.onnx   YOLO（商品らしいものの枠）。detector.json が指すモデル
  embed.onnx      切り抜きの埋め込み（MobileNetV3-small 128px、出力は正規化済み 128 次元）
  gallery.bin     照合先の埋め込み。float16、商品ごとに並べてある（n × 128）
  gallery.json    商品カテゴリの一覧（カテゴリ名・ジャンル・細分類・照合先の開始位置）と前処理の値
infer.py と同じく、社名か商品名があり Vision の結果が悪くない命名済みで、商品カテゴリが付いた切り抜きだけを照合先にする。
"""
import argparse
import json
import shutil
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "app/src/main/assets/products"


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", "").replace("　", "").lower()


def read(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    a = ap.parse_args()
    w = a.work
    OUT.mkdir(parents=True, exist_ok=True)

    det = read(w / "detector.json")
    shutil.copyfile(w / det.get("onnx", "detector.onnx"), OUT / "detector.onnx")
    shutil.copyfile(w / "embed_mnv3.onnx", OUT / "embed.onnx")
    cfg = read(w / "config.json")

    names = read(w / "names.json")
    cat = {**read(w / "category.json"), **read(w / "category_override.json")}  # 商品名 → 商品カテゴリ（products/s3c_category.py）
    excluded = set(json.loads((w / "label_exclude.json").read_text(encoding="utf-8"))) if (w / "label_exclude.json").exists() else set()
    good = {c: cat.get(norm(v.get("maker", "") + v.get("product", ""))) for c, v in names.items()
            if (v.get("maker") or v.get("product")) and v.get("vision") != "bad" and c not in excluded}
    good = {c: k for c, k in good.items() if k}
    z = np.load(w / "emb_mnv3.npz")
    ids, E = z["ids"].tolist(), z["E"]
    keep = [k for k, c in enumerate(ids) if c in good]
    lab = [good[ids[k]] for k in keep]
    order = np.argsort(lab, kind="stable")
    G = E[keep][order]
    G = (G / np.linalg.norm(G, axis=1, keepdims=True)).astype(np.float16)
    lab = [lab[i] for i in order]
    keys, starts = np.unique(lab, return_index=True)

    # カテゴリの粗いジャンル・細分類は、そのカテゴリの商品の多数決（products/common.py の category_levels と同じ）
    def vote(level):
        votes = {}
        for k, c in cat.items():
            if k in level:
                votes.setdefault(c, {}).setdefault(level[k], 0)
                votes[c][level[k]] += 1
        return {c: max(v, key=v.get) for c, v in votes.items()}
    genre, fine = vote(read(w / "genres.json")), vote(read(w / "fine.json"))
    products = [{"key": k, "name": k, "maker": "", "product": k, "genre": genre.get(k), "fine": fine.get(k), "start": int(s)}
                for k, s in zip(keys.tolist(), starts.tolist())]

    G.tofile(OUT / "gallery.bin")
    meta = {"source": det.get("run", ""), "detector_size": det["imgsz"], "embed_size": 128, "pad_color": [124, 116, 104],
            "dim": int(G.shape[1]), "count": int(G.shape[0]), "det_threshold": 0.25, "nms_iou": 0.5,
            "min_crop": [cfg.get("min_crop_w", 20), cfg.get("min_crop_h", 20)], "match_sim": 0.5,
            "context_lambda": [0.05, 0.05], "products": products}
    (OUT / "gallery.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"gallery {G.shape[0]} crops / {len(products)} categories -> {OUT}")


if __name__ == "__main__":
    main()
