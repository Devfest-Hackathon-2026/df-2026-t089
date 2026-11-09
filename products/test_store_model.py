"""店モデルの推定を、作った点で確かめる。  python products/test_store_model.py"""
import json, random

from common import ROOT
from store_model import THEMES, StoreModel, text
from store_sim import calib, simulate

rng = random.Random(0)
calib_tax = json.loads((ROOT / "products/konbinidb/taxonomy.json").read_text(encoding="utf-8"))
m = StoreModel()


def face(u, v0, v1, cam_u, cat, genre, n=400):
    for _ in range(n):
        v = rng.uniform(v0, v1)
        m.add_product(cam_u, v, u + rng.gauss(0, 0.03), v, cat, genre)


face(2.44, 0, 3.6, 3.6, "カップ", "カップ麺")     # 背中合わせ = 島（3.6 m = 3x1.2 か 4x0.9）
face(1.56, 0, 3.6, 0.4, "菓子", "スナック菓子")
face(9.0, 0, 3.6, 7.8, "お茶", "ソフトドリンク")  # 壁際で冷やす商品 = ケース
face(-3.0, 0, 3.6, -1.8, "電池", "日用品・衛生用品")   # 壁際 = 壁面ゴンドラ
for v in range(12):
    m.add_camera(0.4, v * 0.3)
    m.add_camera(3.6, v * 0.3)
kinds = sorted(f["kind"] for f in m.estimate()["fixtures"])
assert kinds == ["case", "island", "wall"], (kinds, text(m.estimate()))
island = next(f for f in m.estimate()["fixtures"] if f["kind"] == "island")
assert abs(island["c"] - 2.0) < 0.1 and abs((island["b"] - island["a"]) - 3.6) < 0.01, island
assert island["faces"]["+"][0][0][0] == "カップ" and island["faces"]["-"][0][0][0] == "菓子", island

# テーマの表と分類表がずれていない（taxonomy.json を変えたら store_themes.json も直す）
tax = {v[0] for g in calib_tax.values() for v in g.values()}
listed = {c for t in THEMES for c in t["cats"]}
assert listed == tax, {"表にない": sorted(tax - listed), "分類にない": sorted(listed - tax)}

# テーマ: 生理用品の面に読み違いの充電器やピラフが混ざっても、テーマで外れる
m = StoreModel()
face(9.0, 0, 2.4, 7.8, "生理用品", "日用品・衛生用品", 300)
face(9.0, 0, 2.4, 7.8, "ティッシュ", "日用品・衛生用品", 120)
face(9.0, 0, 2.4, 7.8, "電池・充電器・ケーブル", "スマホ用品・雑貨", 60)
face(9.0, 0, 2.4, 7.8, "チャーハン・ピラフ", "弁当・おにぎり", 50)
f = m.estimate()["fixtures"][0]
side = next(iter(f["faces"]))   # 棚が 1 列だけだと店の向きは 90 度ごとに定まらないので、面の符号は決め打ちしない
assert f["themes"][side][0] == "紙・衛生・ケア用品", f
named = {c for mod in f["faces"][side] for c, _ in mod}
assert "生理用品" in named and not named & {"電池・充電器・ケーブル", "チャーハン・ピラフ"}, named
assert "[紙・衛生・ケア用品]" in text(m.estimate())
m = StoreModel(themes=[])   # 制約なしなら残る
face(9.0, 0, 2.4, 7.8, "生理用品", "日用品・衛生用品", 300)
face(9.0, 0, 2.4, 7.8, "電池・充電器・ケーブル", "スマホ用品・雑貨", 200)
assert any(c == "電池・充電器・ケーブル" for faces in m.estimate()["fixtures"][0]["faces"].values() for mod in faces for c, _ in mod)

# 全体: 認識が完璧・ずれなしなら、店をほぼ再現できるはず
r = simulate({"recog": "perfect", "drift": 0, "depth": 0.3, "seed": 3}, calib())["metrics"]
assert r["f1"] > 0.75 and r["cat_acc"] > 0.7, r
print("ok", {k: r[k] for k in ("f1", "kind_acc", "cat_acc", "pos_err")})
