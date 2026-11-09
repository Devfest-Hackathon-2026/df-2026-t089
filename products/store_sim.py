"""コンビニを歩く様子を作り、アプリと同じ入力（1 秒ごとの商品＋位置）で店モデルを推定して、正解の店と比べる。

  python products/store_sim.py --batch     # 店を変えて条件ごとに測る -> work/store_sim_batch.json（ダッシュボードの ⑤ に出る）
  python products/store_sim.py --fixture   # アプリのテスト用データを作り直す（store_model.py を直したとき）

再現しているもの（数字の根拠）
  ・1 秒に 1 枚、胸の高さ 1.3 m、縦持ち、地図に置くのは大きい順に 20 個まで、奥行き 0.2〜3.5 m（AssistActivity / ProductRecognizer / ShelfMap の定数）
  ・カテゴリの読み違いと「名前を付けない」は実測（work/store_calib.json: 学習に使っていない切り抜きの実際の予測）。商品ごとの偏りも残る
  ・検出は写る大きさの関数。実機の記録 records/20261106_190954（225 枚、名前付き 2.5 個/枚、距離の中央値 1.06 m、0.23 m/s）に合うよう recall 0.75, px50 40 にした
仮定（実測がないもの。ダッシュボードで変えられる）
  ・深度の誤差と欠け、ARCore の位置のずれ。人による隠れは入れていない（画角は実機の値）
"""
import argparse, json, math, random, sys
from math import atan2, cos, hypot, pi, radians, sin

from common import ROOT, WORK, read_json, write_json
from store_model import CASE_DEPTH, END_DEPTH, GONDOLA_DEPTH, ISLAND_DEPTH, StoreModel, text

DEFAULTS = {"seed": 1, "mode": "scan", "recall": 0.75, "px50": 40.0, "geo": "all", "drift": 1.0, "depth": 1.0, "recog": "real", "speed": 0.25, "snap": 6}
F_PX = 943.0         # 実機の records.jsonl の intrinsics（fx 1415 / 1920x1080）を長辺 1280 px に換算
HFOV, VFOV = radians(41.8), radians(68.3)   # 同じく実機の値（縦持ち）
FACE_W, ROW_H = 0.09, 0.3
ROWS = (0.35, 0.65, 0.95, 1.25, 1.55)
MAX_NAMED, MAX_RANGE = 20, 3.5
ZONES = {"island": ["カップ麺", "加工食品", "スナック菓子", "チョコ・ビスケット", "グミ・キャンディ", "和洋菓子", "駄菓子・食玩", "おつまみ・ナッツ", "パン"],
         "drinks": ["ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品", "酒"],
         "chilled": ["弁当・おにぎり", "サンドイッチ・サラダ・惣菜", "チルドデザート", "冷凍食品", "アイス"],
         "goods": ["日用品・衛生用品", "化粧品", "文房具", "マスク", "スマホ用品・雑貨", "靴下・衣料", "その他"]}


def calib():
    c = read_json(WORK / "store_calib.json")
    if not c:
        raise ValueError("work/store_calib.json がありません（python products/store_calib.py）")
    return c


def pools(cal):
    """ジャンル → そのジャンルのカテゴリ（テストの切り抜きが 3 枚以上あるもの）"""
    out = {}
    for cat, groups in sorted(cal["cats"].items()):
        if sum(map(len, groups.values())) >= 3 and cat in cal["genre"]:
            out.setdefault(cal["genre"][cat], []).append(cat)
    return out


def make_store(seed, cal):
    """店の正解を作る。u, v は店の座標で、(0,0) が部屋の角"""
    rng = random.Random(seed)
    W, D = round(rng.uniform(9.0, 13.0), 1), round(rng.uniform(7.5, 11.0), 1)
    pool = pools(cal)
    every = [c for cs in pool.values() for c in cs]
    fixtures = []

    def cats_for(zone, n):
        names = [g for g in ZONES[zone] if g in pool] or list(pool)
        return [rng.choice(pool[rng.choice(names)]) for _ in range(n)]

    def add(kind, axis, c, a, n, w, depth, signs, zone):
        fixtures.append({"kind": kind, "axis": axis, "c": c, "a": a, "b": a + n * w, "d": depth, "n": n, "w": w,
                         "faces": {("+" if s == 1 else "-"): cats_for(zone, n) for s in signs}})

    def row(kind, axis, c, start, stop, depth, sign, zone, w):
        n = int((stop - start) / w)
        if n:
            add(kind, axis, c, start + (stop - start - n * w) / 2, n, w, depth, [sign], zone)

    row("case", 1, D - CASE_DEPTH / 2, 0.8, W - 0.8, CASE_DEPTH, -1, "drinks", 0.9)          # 奥の壁: 飲み物
    row("case", 0, W - CASE_DEPTH / 2, 1.8, D - 1.6, CASE_DEPTH, -1, "chilled", 0.9)         # 右の壁: 弁当など
    row("wall", 0, GONDOLA_DEPTH / 2, 0.6, D - 1.2, GONDOLA_DEPTH, 1, "goods", 1.2)          # 左の壁: 日用品
    reg_len = 2.4
    reg = {"kind": "register", "axis": 1, "c": 1.5, "a": W - 0.8 - reg_len, "b": W - 0.8, "d": 0.7, "n": 1, "w": reg_len, "faces": {}}
    fixtures.append(reg)

    # 島: 手前の通路と奥の通路を残し、列のあいだに 1.1〜1.4 m の通路をとる
    u0, u1 = GONDOLA_DEPTH + 1.3, W - CASE_DEPTH - 1.3
    v0, v1 = 3.0, D - CASE_DEPTH - 1.5
    u, rows = u0 + 0.5, []
    while u + ISLAND_DEPTH <= u1:
        rows.append(u + ISLAND_DEPTH / 2)
        u += ISLAND_DEPTH + rng.uniform(1.15, 1.4)
    for cu in rows:
        w = rng.choice((0.9, 1.2))
        n = max(2, int((v1 - v0 - 2 * END_DEPTH) / w))
        a = (v0 + v1) / 2 - n * w / 2
        add("island", 0, cu, a, n, w, ISLAND_DEPTH, [1, -1], "island")
        for sign, c in ((-1, a - END_DEPTH / 2), (1, a + n * w + END_DEPTH / 2)):
            if rng.random() < 0.8:
                add("end", 1, c, cu - 0.45, 1, 0.9, END_DEPTH, [sign], "island")

    products = []
    for fi, f in enumerate(fixtures):
        for sign, cats in f["faces"].items():
            s = 1 if sign == "+" else -1
            for m, main in enumerate(cats):
                for y in ROWS:   # 段ごとに商品がまとまっていて、4 分の 1 は同じジャンルの別カテゴリ
                    t = f["a"] + m * f["w"] + FACE_W / 2
                    while t < f["a"] + (m + 1) * f["w"]:
                        cat = main if rng.random() < 0.75 else rng.choice(pool.get(cal["genre"][main]) or every)
                        group = rng.choice(sorted(cal["cats"][cat]))
                        for _ in range(rng.randint(2, 4)):
                            if t < f["a"] + (m + 1) * f["w"]:
                                cross = f["c"] + s * f["d"] / 2
                                p = (cross, t) if f["axis"] == 0 else (t, cross)
                                products.append({"f": fi, "sign": s, "m": m, "p": p, "y": y, "cat": cat, "group": group})
                                t += FACE_W
    return {"W": W, "D": D, "fixtures": fixtures, "products": products, "rows": rows, "v0": v0, "v1": v1}


def auto_route(store):
    """入口から、島のあいだを蛇行して最後に店を一周する"""
    W, D, rows, v0, v1 = store["W"], store["D"], store["rows"], store["v0"], store["v1"]
    lanes = [GONDOLA_DEPTH + 0.9] + [(a + b) / 2 for a, b in zip(rows, rows[1:])] + [W - CASE_DEPTH - 0.9]
    if len(rows) == 1:
        lanes = [lanes[0], lanes[-1]]
    pts, near = [(W * 0.45, 0.8)], True
    low, high = v0 - 1.0, min(v1 + 1.0, D - CASE_DEPTH - 0.7)
    for lane in lanes:
        pts += [(lane, low if near else high), (lane, high if near else low)]
        near = not near
    # 店を一周して、最後にレジの前で会計して出る
    return pts + [(lanes[-1], high), (lanes[0], high), (lanes[0], low), (W - 2.0, low), (W - 2.0, 2.9), (W - 3.4, 2.9), (W * 0.45, 0.8)]


def rect_of(f):
    """什器の床の長方形 (u0, v0, u1, v1)"""
    lo, hi = f["c"] - f["d"] / 2, f["c"] + f["d"] / 2
    return (lo, f["a"], hi, f["b"]) if f["axis"] == 0 else (f["a"], lo, f["b"], hi)


def blocked(ox, oz, px, pz, rects, skip):
    """(ox,oz) から (px,pz) への線が、skip 以外の什器を通るか"""
    dx, dz = px - ox, pz - oz
    for i, (u0, v0, u1, v1) in enumerate(rects):
        if i == skip:
            continue
        lo, hi = 0.0, 1.0
        for o, d, a, b in ((ox, dx, u0, u1), (oz, dz, v0, v1)):
            if abs(d) < 1e-9:
                if not a <= o <= b:
                    lo, hi = 1.0, 0.0
                    break
            else:
                t0, t1 = sorted(((a - o) / d, (b - o) / d))
                lo, hi = max(lo, t0), min(hi, t1)
        if lo < hi:
            return True
    return False


def walk(route, speed, mode):
    """1 秒ごとの (u, v, 向き)"""
    out, i, u, v, t, along = [], 0, route[0][0], route[0][1], 0, 0.0
    while i + 1 < len(route):
        ex, ez = route[i + 1]
        d = hypot(ex - u, ez - v)
        if d <= speed:
            u, v, i = ex, ez, i + 1
            if i + 1 >= len(route) or d < 1e-6:
                continue
        else:
            along = atan2(ez - v, ex - u)
            u, v = u + (ex - u) / d * speed, v + (ez - v) / d * speed
        sway = radians(75) * sin(2 * pi * t / 6) if mode == "scan" else radians(8) * sin(2 * pi * t / 7)
        out.append((u, v, along + sway))
        t += 1
    return out


def simulate(cfg, cal=None, route=None):
    """1 回歩いて、一定間隔の推定と評価を返す"""
    cal = cal or calib()
    cfg = {**DEFAULTS, **cfg}
    rng = random.Random(cfg["seed"] * 7919 + 11)
    store = make_store(cfg["seed"], cal)
    rects = [rect_of(f) for f in store["fixtures"]]
    route = route or auto_route(store)
    steps = walk(route, cfg["speed"], cfg["mode"])
    if not steps:
        raise ValueError("route too short")
    # ARCore の世界は歩き始めの場所が原点で、店の向きとは無関係
    w0 = rng.uniform(0, 2 * pi)
    cw, sw = cos(w0), sin(w0)
    ou, ov = steps[0][0], steps[0][1]
    to_world = lambda u, v: ((u - ou) * cw - (v - ov) * sw, (u - ou) * sw + (v - ov) * cw)
    model = StoreModel()
    du = dv = de = 0.0   # ARCore の位置と向きのずれ
    genre = cal["genre"]
    by_group = lambda cat, group: cal["cats"][cat][group]
    frames, snaps = [], []
    seen = {}
    named = right = 0
    dists = []
    for t, (u, v, heading) in enumerate(steps):
        du += rng.gauss(0, 0.006 * cfg["drift"])
        dv += rng.gauss(0, 0.006 * cfg["drift"])
        de += rng.gauss(0, radians(0.03) * cfg["drift"])
        cu, cv = u + du, v + dv
        cam_x, cam_z = to_world(cu, cv)
        model.add_camera(cam_x, cam_z)
        visible = []
        for p in store["products"]:
            dx, dz = p["p"][0] - u, p["p"][1] - v
            flat = hypot(dx, dz)
            if not 0.2 <= flat <= MAX_RANGE:
                continue
            f = store["fixtures"][p["f"]]
            nu, nv = (p["sign"], 0) if f["axis"] == 0 else (0, p["sign"])
            cos_in = -(dx * nu + dz * nv) / flat
            ang = (atan2(dz, dx) - heading + pi) % (2 * pi) - pi
            dy = p["y"] - 1.3
            if cos_in <= 0.1 or abs(ang) > HFOV / 2 or abs(atan2(dy, flat)) > VFOV / 2:
                continue
            depth = hypot(flat, dy)
            px = FACE_W * cos_in * F_PX / depth
            if rng.random() > cfg["recall"] / (1 + math.exp(-(px - cfg["px50"]) / 5)):
                continue
            if not blocked(u, v, p["p"][0], p["p"][1], rects, p["f"]):
                visible.append((px * ROW_H * F_PX / depth, p, dx, dz, depth, ang))
        visible.sort(key=lambda x: -x[0])
        placed = 0
        for k, (_, p, dx, dz, depth, ang) in enumerate(visible):
            seen[(p["f"], p["sign"], p["m"])] = seen.get((p["f"], p["sign"], p["m"]), 0) + 1
            cat = ""   # 名前を付けるのは大きい順に 20 個まで。それ以外と「名前を付けない」は位置だけ使う
            if k < MAX_NAMED:
                cat = rng.choice(by_group(p["cat"], p["group"])) if cfg["recog"] == "real" else p["cat"]
                if cat:
                    named += 1
                    dists.append(depth)
                    right += cat == p["cat"]
            if not cat and cfg["geo"] == "named":
                continue
            if rng.random() < 0.15 * cfg["depth"]:   # 深度の穴。今の実装はその商品を置かない
                continue
            d = depth + rng.gauss(0, (0.02 + 0.02 * depth * depth) * cfg["depth"])
            if rng.random() < 0.03:
                d *= rng.uniform(0.6, 1.4)
            flat = d * hypot(dx, dz) / depth
            bearing = atan2(dz, dx) + rng.gauss(0, radians(0.5)) + de
            x, z = to_world(cu + flat * cos(bearing), cv + flat * sin(bearing))
            model.add_product(cam_x, cam_z, x, z, cat, genre.get(cat, ""))
            placed += 1
        for f in store["fixtures"]:   # レジの天板: ARCore の水平面（腰の高さ）
            if f["kind"] == "register":
                ang = (atan2(f["c"] - v, (f["a"] + f["b"]) / 2 - u) - heading + pi) % (2 * pi) - pi
                if abs(ang) < HFOV / 2 and hypot(f["c"] - v, (f["a"] + f["b"]) / 2 - u) < MAX_RANGE and rng.random() < 0.4:
                    for _ in range(4):
                        x, z = to_world(rng.uniform(f["a"], f["b"]) + du, f["c"] + rng.uniform(-f["d"] / 2, f["d"] / 2) + dv)
                        model.add_counter(x, z)
        frames.append({"t": t, "u": round(u, 2), "v": round(v, 2), "h": round(heading, 3), "n": placed})
        if t % cfg["snap"] == 0 or t == len(steps) - 1:
            est = model.estimate()
            snaps.append({"t": t, "model": light(est, w0, ou, ov), "metrics": evaluate(est, store, seen, w0, ou, ov)})
    final = snaps[-1]
    return {"cfg": cfg, "store": {k: store[k] for k in ("W", "D", "fixtures")}, "route": route, "frames": frames, "snaps": snaps,
            "text": text(model.estimate()), "cells": cells_in_store(model, w0, ou, ov), "named_acc": round(right / max(named, 1), 3), "named_per_frame": round(named / len(steps), 2), "named_dist": round(sorted(dists)[len(dists) // 2], 2) if dists else None, "metrics": final["metrics"]}


def in_store(x, z, w0, ou, ov):
    c, s = cos(w0), sin(w0)
    return x * c + z * s + ou, -x * s + z * c + ov


def corners(f, yaw):
    """推定した什器の四隅（世界座標）"""
    r = rect_of(f)
    c, s = cos(yaw), sin(yaw)
    return [(u * c - v * s, u * s + v * c) for u, v in ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))]


def light(est, w0, ou, ov):
    """ダッシュボードで描くために、推定を店の座標の四隅に直す"""
    out = []
    for f in est["fixtures"]:
        out.append({"kind": f["kind"], "h": f["h"], "n": f["n"], "w": round(f["w"], 2), "c": [[round(v, 3) for v in in_store(x, z, w0, ou, ov)] for x, z in corners(f, est["yaw"])],
                    "mods": modules(f, est["yaw"], w0, ou, ov), "main": main_label(f)})
    return out


def main_label(f):
    return {s: [m[0][0] if m else "" for m in mods] for s, mods in f["faces"].items()}


def modules(f, yaw, w0, ou, ov):
    """モジュールごとの (中心, 面ごとの法線と主な商品)。評価と表示で使う"""
    out, c, s = [], cos(yaw), sin(yaw)
    for i in range(f["n"]):
        t = f["a"] + (i + 0.5) * f["w"]
        u, v = (f["c"], t) if f["axis"] == 0 else (t, f["c"])
        x, z = in_store(u * c - v * s, u * s + v * c, w0, ou, ov)
        faces = []
        for sign, ms in f["faces"].items():
            nu, nv = (1, 0) if f["axis"] == 0 else (0, 1)
            k = 1 if sign == "+" else -1
            wx, wz = (nu * c - nv * s) * k, (nu * s + nv * c) * k
            cw, sw = cos(w0), sin(w0)
            faces.append({"n": [round(wx * cw + wz * sw, 3), round(-wx * sw + wz * cw, 3)], "cats": ms[i] if i < len(ms) else []})
        out.append({"u": round(x, 3), "v": round(z, 3), "w": f["w"], "faces": faces})
    return out


def cells_in_store(model, w0, ou, ov):
    return [[round(v, 2) for v in in_store(c[1] / c[0], c[2] / c[0], w0, ou, ov)] + [int(c[0])] for c in model.cells.values()]


def evaluate(est, store, seen, w0, ou, ov):
    """正解の什器モジュールと推定のモジュールを、床の重なりで対応づけて比べる"""
    gt = []
    for fi, f in enumerate(store["fixtures"]):
        for m in range(f["n"]):
            t0, t1 = f["a"] + m * f["w"], f["a"] + (m + 1) * f["w"]
            lo, hi = f["c"] - f["d"] / 2, f["c"] + f["d"] / 2
            r = (lo, t0, hi, t1) if f["axis"] == 0 else (t0, lo, t1, hi)
            gt.append({"f": fi, "m": m, "r": r, "kind": f["kind"], "faces": f["faces"], "axis": f["axis"], "seen": sum(seen.get((fi, s, m), 0) for s in (1, -1)) >= 5})
    pred = []
    for f in est["fixtures"]:
        for i, m in enumerate(modules(f, est["yaw"], w0, ou, ov)):
            half_w, half_d = f["w"] / 2, f["d"] / 2
            # 店の向きの推定は 90 度ごとに不定なので、推定の什器の向きを店の軸に揃えて長方形にする（誤差は位置の差に出る）
            cc, ss = cos(est["yaw"] - w0), sin(est["yaw"] - w0)
            tangent_along_u = (abs(ss) > abs(cc)) if f["axis"] == 0 else (abs(cc) > abs(ss))
            hu, hv = (half_w, half_d) if tangent_along_u else (half_d, half_w)
            pred.append({"r": (m["u"] - hu, m["v"] - hv, m["u"] + hu, m["v"] + hv), "kind": f["kind"], "faces": m["faces"]})
    pairs = []
    for gi, g in enumerate(gt):
        for pi_, p in enumerate(pred):
            iw = min(g["r"][2], p["r"][2]) - max(g["r"][0], p["r"][0])
            ih = min(g["r"][3], p["r"][3]) - max(g["r"][1], p["r"][1])
            if iw > 0 and ih > 0:
                inter = iw * ih
                union = (g["r"][2] - g["r"][0]) * (g["r"][3] - g["r"][1]) + (p["r"][2] - p["r"][0]) * (p["r"][3] - p["r"][1]) - inter
                if inter / union >= 0.25:
                    pairs.append((inter / union, gi, pi_))
    pairs.sort(reverse=True)
    used_g, used_p, matched = set(), set(), []
    for iou, gi, pi_ in pairs:
        if gi not in used_g and pi_ not in used_p:
            used_g.add(gi)
            used_p.add(pi_)
            matched.append((gi, pi_, iou))
    kind_ok = err = faces = cat_ok = genre_ok = 0
    cal = store_genre()
    rows = []
    for gi, pi_, iou in matched:
        g, p = gt[gi], pred[pi_]
        kind_ok += g["kind"] == p["kind"]
        err += hypot((g["r"][0] + g["r"][2] - p["r"][0] - p["r"][2]) / 2, (g["r"][1] + g["r"][3] - p["r"][1] - p["r"][3]) / 2)
        for sign, cats in g["faces"].items():
            n = (1, 0) if g["axis"] == 0 else (0, 1)
            n = (n[0] * (1 if sign == "+" else -1), n[1] * (1 if sign == "+" else -1))
            face = next((x for x in p["faces"] if x["n"][0] * n[0] + x["n"][1] * n[1] > 0.7 and x["cats"]), None)
            if face:
                faces += 1
                truth, top = cats[g["m"]], face["cats"][0][0]
                cat_ok += top == truth
                genre_ok += cal.get(top) == cal.get(truth)
                rows.append([g["kind"], truth, top, top == truth])
    seen_n = sum(g["seen"] for g in gt)
    seen_hit = sum(gt[gi]["seen"] for gi, _, _ in matched)
    by_kind = {}
    for g in gt:
        by_kind.setdefault(g["kind"], [0, 0])[1] += 1
    for gi, _, _ in matched:
        by_kind[gt[gi]["kind"]][0] += 1
    n = max(len(matched), 1)
    precision, recall = len(matched) / max(len(pred), 1), len(matched) / max(len(gt), 1)
    return {"gt": len(gt), "pred": len(pred), "matched": len(matched), "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 3), "seen_gt": seen_n, "seen_recall": round(seen_hit / max(seen_n, 1), 3),
            "kind_acc": round(kind_ok / n, 3), "pos_err": round(err / n, 3), "faces": faces, "cat_acc": round(cat_ok / max(faces, 1), 3),
            "genre_acc": round(genre_ok / max(faces, 1), 3), "by_kind": by_kind, "rows": rows}


_genre = {}


def store_genre():
    if not _genre:
        _genre.update(calib()["genre"])
    return _genre


CONDITIONS = [("実際の認識・棚を見回して歩く", {}),
              ("実際の認識・前を向いたまま", {"mode": "forward"}),
              ("認識が完璧（カテゴリの誤りなし）", {"recog": "perfect"}),
              ("ARCore のずれ 3 倍", {"drift": 3.0}),
              ("深度の誤差 2 倍", {"depth": 2.0}),
              ("検出が弱い（遠くの商品が見えない）", {"px50": 60.0}),
              ("名前が付いた商品だけで形も推定", {"geo": "named"})]


def batch(n):
    cal, out = calib(), []
    for name, cfg in CONDITIONS:
        ms = []
        for seed in range(1, n + 1):
            r = simulate({**cfg, "seed": seed, "snap": 10 ** 6}, cal)["metrics"]
            ms.append(r)
        keys = ("precision", "recall", "f1", "seen_recall", "kind_acc", "pos_err", "cat_acc", "genre_acc")
        out.append({"name": name, "n": n, **{k: round(sum(m[k] for m in ms) / n, 3) for k in keys}, "f1_min": min(m["f1"] for m in ms),
                    "by_kind": {k: round(sum(m["by_kind"].get(k, [0, 0])[0] for m in ms) / max(sum(m["by_kind"].get(k, [0, 0])[1] for m in ms), 1), 3) for k in ("island", "wall", "case", "end", "register")}})
        print(out[-1], flush=True)
    write_json(WORK / "store_sim_batch.json", {"rows": out, "seeds": n, "crops": cal["crops"], "named": cal["named"], "acc_named": cal["acc_named"]})


def fixture():
    """同じ観測列を流したときの text() が Python とアプリで同じかを確かめるデータ"""
    rng = random.Random(5)
    model = StoreModel()
    lines = []

    def obs(cx, cz, x, z, cat, genre):
        model.add_product(cx, cz, x, z, cat, genre)
        lines.append(f"P\t{cx}\t{cz}\t{x}\t{z}\t{cat}\t{genre}")

    yaw = radians(17)
    w = lambda u, v: (u * cos(yaw) - v * sin(yaw), u * sin(yaw) + v * cos(yaw))
    for i in range(1800):
        side = rng.choice([(1.0, 2.0, 0.0, 1, "A", "G1"), (1.88, 2.8, 0.0, -1, "B", "G2"), (6.0, 5.0, 2.0, -1, "D", "酒")])
        c, lane, a, s, cat, genre = side
        t = rng.uniform(a, a + 3.6)
        x, z = w(c + rng.gauss(0, 0.05), t)
        cx, cz = w(c + s * 1.2 if c != 6.0 else 5.0, t + rng.uniform(-1, 1))
        obs(round(cx, 4), round(cz, 4), round(x, 4), round(z, 4), cat, genre)
    for k in range(40):
        x, z = w(0.5 + k * 0.15, -1.0 + (k % 4) * 0.2)
        model.add_counter(round(x, 4), round(z, 4))
        lines.append(f"C\t{round(x, 4)}\t{round(z, 4)}")
    for k in range(30):
        x, z = w(3.2, k * 0.2)
        model.add_camera(round(x, 4), round(z, 4))
        lines.append(f"T\t{round(x, 4)}\t{round(z, 4)}")
    out = ROOT / "app/src/test/resources/store_model_trace.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n===\n" + text(model.estimate()) + "\n", encoding="utf-8", newline="\n")
    print(text(model.estimate()))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    if a.fixture:
        fixture()
    elif a.batch:
        batch(a.n)
    else:
        r = simulate({"snap": 30})
        print(json.dumps({k: v for k, v in r["metrics"].items() if k != "rows"}, ensure_ascii=False), r["named_acc"])
        print(r["text"])
