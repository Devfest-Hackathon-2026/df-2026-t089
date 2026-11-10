"""学習済みの MobileCLIP の分類層（122 カテゴリ）を、アプリの assets に書き出す。

  .venv/Scripts/python tools/export_category_head.py

書き出すもの（app/src/main/assets/products/、git には入れない）
  category_head.bin   float32 のリトルエンディアン。1 層目の重み・バイアス、2 層目の重み・バイアスの順
  category_head.json  入力と中間層の次元、カテゴリの一覧（コード・名前・アプリの棚の種類）、信頼度調整の重み
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
HEAD = ROOT / "products/work/sku26n/mobileclip_s0_head_gem.pt"
# 周りの判定との一致で確からしさを付け直す重み（products/sku26n/neighbor_confidence.py が求める）
CONFIDENCE = ROOT / "products/work/sku26n/neighbor_confidence.json"
TAXONOMY = ROOT / "products/konbinidb/taxonomy.json"
OUT = ROOT / "app/src/main/assets/products"

# gallery.json に同じ名前が無いカテゴリの、アプリの棚の種類（labels.txt）
GENRE_CODES = {
    "ソフトドリンク": "001 003 004 007 010 014 104",
    "紙パック飲料・乳製品": "019 113",
    "栄養・エナジードリンク": "123 040",
    "酒": "101 102 127 128 129 130 131",
    "スナック菓子": "023 033",
    "チョコ・ビスケット": "024 025 030 031",
    "グミ・キャンディ": "027 029 034 035",
    "おつまみ・ナッツ": "038 039",
    "弁当・おにぎり": "042 044 045 055 056 057 058 059 060",
    "パン": "048 049 050",
    "カップ麺": "064 075",
    "加工食品": "067 070 072 073 114 115 124 125",
    "和洋菓子": "079 081",
    "アイス": "083 084",
    "冷凍食品": "085 086 087 116",
    "サンドイッチ・サラダ・惣菜": "088 089 090 106 107 108 109 110 111 112",
    "日用品・衛生用品": "092 094 095 100 118 119 120",
    "化粧品": "096",
    "スマホ用品・雑貨": "098 126",
    "文房具": "099",
    "靴下・衣料": "117",
}


def main():
    ckpt = torch.load(HEAD, map_location="cpu")
    state = {k: v.numpy().astype(np.float32) for k, v in ckpt["state"].items()}
    classes, dim, hidden = ckpt["classes"], ckpt["dim"], state["0.weight"].shape[0]
    names = {code: value[0] for group in json.loads(TAXONOMY.read_text(encoding="utf-8")).values() for code, value in group.items()}
    gallery = {p["name"]: p["genre"] for p in json.loads((OUT / "gallery.json").read_text(encoding="utf-8"))["products"]}
    extra = {code: genre for genre, codes in GENRE_CODES.items() for code in codes.split()}
    genres = {code: gallery.get(names[code]) or extra[code] for code in classes}  # 足りないカテゴリがあれば KeyError で止める
    assert set(genres.values()) <= set(gallery.values()), set(genres.values()) - set(gallery.values())

    body = np.concatenate([state[k].ravel() for k in ("0.weight", "0.bias", "3.weight", "3.bias")]).astype("<f4")
    (OUT / "category_head.bin").write_bytes(body.tobytes())
    (OUT / "category_head.json").write_text(json.dumps({"dim": dim, "hidden": hidden, "categories": [
        {"code": c, "name": names[c], "genre": genres[c]} for c in classes],
        "confidence": json.loads(CONFIDENCE.read_text(encoding="utf-8"))}, ensure_ascii=False, indent=0), encoding="utf-8")

    # 書き出した数値で、学習時の分類層と同じ確率になることを確かめる
    x = np.random.default_rng(0).normal(size=(8, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    raw = np.frombuffer((OUT / "category_head.bin").read_bytes(), "<f4")
    first = raw[:hidden * dim].reshape(hidden, dim)
    bias = raw[hidden * dim:hidden * dim + hidden]
    rest = raw[hidden * dim + hidden:]
    second, tail = rest[:len(classes) * hidden].reshape(len(classes), hidden), rest[len(classes) * hidden:]
    logits = np.maximum(x @ first.T + bias, 0) @ second.T + tail
    model = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(hidden, len(classes)))
    model.load_state_dict(ckpt["state"])
    expected = model.eval()(torch.from_numpy(x)).detach().numpy()
    assert np.allclose(logits, expected, atol=1e-3), "書き出した重みが学習時と違う"
    print(f"{len(classes)} categories, {len(raw)} floats, {len(raw) * 4 / 1e6:.2f} MB", flush=True)


if __name__ == "__main__":
    main()
