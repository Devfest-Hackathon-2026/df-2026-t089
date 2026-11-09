"""店モデルの推定を、作った点で確かめる。  python products/test_store_model.py"""
import json, random

from common import ROOT
from store_model import THEMES, StoreModel, glass_restored, text
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
for u, vs in ((0.4, range(12)), (3.6, range(11, -1, -1))):   # 島を回り込んで歩く（歩いた線は島を横切れない）
    for v in vs:
        m.add_camera(u, v * 0.3)
    m.add_camera(u, 4.6)
kinds = sorted(f["kind"] for f in m.estimate()["fixtures"])
assert kinds == ["case", "island", "wall"], (kinds, text(m.estimate()))
island = next(f for f in m.estimate()["fixtures"] if f["kind"] == "island")
assert abs(island["c"] - 2.0) < 0.1 and abs((island["b"] - island["a"]) - 3.6) < 0.05, island
assert island["faces"]["+"][0][0][0] == "カップ" and island["faces"]["-"][0][0][0] == "菓子", island

# 直交する 2 つの棚の角は重ならない形に削られ、どちらも残る
m = StoreModel()
for i in range(300):
    z = i * 3.2 / 300
    m.add_product(3.8, z, 5.0 + rng.gauss(0, 0.03), z, "A", "")
    x = 1.9 + i * 3.4 / 300
    m.add_product(x, 1.9, x, 3.1 + rng.gauss(0, 0.03), "B", "")
for v in range(9):
    m.add_camera(3.8, v * 0.3)
racks = [f for f in m.estimate()["fixtures"]]
assert len(racks) == 2, text(m.estimate())
assert not m._overlaps(racks[0], racks[1]), text(m.estimate())

# ガラス: 視線に沿ってばらけた点から、カメラ側の端にガラスのケースを置き、点をその面へ戻す
m = StoreModel()
for i in range(500):
    cam_z = rng.uniform(0.2, 2.8)
    ahead = rng.uniform(-0.9, 0.9)
    depth = rng.uniform(1.6, 4.2)
    m.add_product(0.0, cam_z, depth, cam_z + ahead * depth / 3, "お茶", "ソフトドリンク")
    z = rng.uniform(0, 3)
    m.add_product(0.0, cam_z, -1.5 + rng.gauss(0, 0.03), z, "電池", "日用品・衛生用品")
for v in range(10):
    m.add_camera(0.0, v * 0.3)
view = m.estimate()
glass = [f for f in view["fixtures"] if f["kind"] == "glass"]
assert glass, text(view)
assert any((r := glass_restored(view, 0.0, z / 2, 3.5, z / 2)) and r[0] < 3.5 for z in range(7)), text(view)
assert glass_restored(view, 0.0, 1.5, -1.5, 1.5) is None   # 反対側の棚の点は動かさない

# 角: 直交する 2 面の端が少し離れていても交点にそろい、遠い面はそのまま
r = {"ax": 0, "c": 5.0, "a": 0.4, "b": 2.8}
q = {"ax": 1, "c": 0.0, "a": 2.0, "b": 4.7}
far = {"ax": 1, "c": 9.0, "a": 2.0, "b": 4.7}
StoreModel._snap_corners([r, q, far])
assert r["a"] == 0.0 and q["b"] == 5.0 and r["corner"] and q["corner"], (r, q)
assert far["a"] == 2.0 and far["b"] == 4.7 and "corner" not in far, far

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
