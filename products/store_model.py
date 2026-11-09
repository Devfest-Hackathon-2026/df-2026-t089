"""コンビニ記述言語（店のメンタルモデル）と、「商品＋位置」の連続観測からそのモデルを推定する処理。

■ 記述言語（StoreModel.estimate の結果と、text() の 1 行 1 什器の表記）
  店 = 部屋（床と壁の長方形）＋ 什器の並び。座標は店の向き yaw で回した u, v（棚は壁に平行か垂直なので軸に揃う）。
    store yaw=<度> room=<u0>,<v0>..<u1>,<v1>
    <種類> axis=<u|v> c=<正面方向の中心> span=<a>..<b> d=<奥行き> h=<高さ> mods=<個数>x<幅>
      m<番号> <+|->[<テーマ>]<カテゴリ>:<割合>,...      ← 什器の単位（モジュール）ごと・面（+ は軸の正の向き）ごとのテーマと主な商品
  テーマ: ゴンドラの片面には決まった組み合わせの商品が並ぶ（カップ麺とスープとレトルトは一緒、充電器と生理用品とピラフは一緒にならない）。
    定型は app/src/main/assets/store_themes.json（アプリと共有）。[...] はテーマを決められない面では出ない。
  種類:
    island   島ゴンドラ。奥行き 片面 0.44 m（背中合わせで 0.88）、高さ 1.8 m、幅 0.9 か 1.2 m を並べる
    wall     壁面ゴンドラ。壁を背にする
    case     冷蔵・冷凍ケース。壁を背にする（冷やす商品が半分以上の壁面の什器）
    end      エンド什器。島の両端に、島と直交する向きで付く
    register レジ。商品は写らないので、腰の高さの水平面（ARCore）から見つける
  axis は什器の正面が向く軸。

■ 推定（入力の点の集まりから毎回作り直すので、歩くほど更新される）
  1. 商品の位置を 10 cm の格子に貯める（回数・カテゴリ・カメラのいた向き）
  2. 点が軸に揃う回転を探して店の向きにする（棚は壁に平行か垂直）
  3. 軸に平行な線（棚の面）を長い順に取り出す。面の向きは、カメラがどちら側にいたかで決める
  4. 背中合わせの面 = 島、後ろを歩いていない外側の面 = 壁、島の端で直交する短い面 = エンド
  5. 幅を 0.9 / 1.2 m の倍数に揃え、モジュールごとに多数決で主な商品を決める
  6. 面ごとに観測に一番合うテーマを選び（隣のモジュールも少し参考にする）、テーマに合わないカテゴリは読み違いとして外す

アプリの jp.devfest.shelfsight.assist.StoreModel は同じ処理の移植。text() が一致することを StoreModelTest で確かめる
（直したら python products/store_sim.py --fixture でテストデータを作り直す）。
"""
import json
from bisect import bisect
from math import cos, floor, hypot, log, radians, sin
from pathlib import Path

THEMES = json.loads((Path(__file__).resolve().parents[1] / "app/src/main/assets/store_themes.json").read_text(encoding="utf-8"))
THEME_IN = 0.9            # 面の商品がテーマの中のカテゴリである確率
THEME_NEIGHBOR = 0.3      # 隣のモジュールの観測をどれだけ参考にするか
MIN_THEME_OBS = 3         # 名前付きの観測がこれ未満ならテーマを決めない
MIN_THEME_SHARE = 0.5     # テーマに合う観測がこの割合に届かない面は、定型にない面として何も外さない
CELL = 0.1
BAND = 0.18        # 面とみなす厚みの半分（深度の誤差を吸収する）
GAP = 0.5          # これより長く点が途切れたら別の面
MIN_LEN = 0.4
SAME_FACE = 0.45   # 同じ向きの面がこれより近いことはないので、1 つの面とみなす
MIN_OBS = 8        # 面と認める観測数の下限
ISLAND_DEPTH = 0.88
END_DEPTH = 0.44
GONDOLA_DEPTH = 0.45
CASE_DEPTH = 0.8
HEIGHT = {"island": 1.8, "wall": 1.8, "case": 2.0, "glass": 2.0, "end": 1.5, "register": 1.0}
FRONT = 0.85       # 面の前面とみなす、奥から数えた点の割合
EXPLAIN = 0.25     # 棚の箱からこの距離までの点は、その棚の商品として説明できる
TOUCH = 0.01       # 棚どうしが重なったとみなす食い込み
SHELF_COST = 3     # 新しく説明する点がこれ以下の棚は、既にある棚の重複なので足さない
MODULE_COST = 0
DEPTH_SCALES = (1.0, 0.66, 0.4)   # 面の裏に棚を置くときに試す奥行きの比
CORNER_BONUS = 12
CORNER_R = 0.6     # 直交する面の端がこの距離以内なら、同じ角とみなす
SIGHT_BEYOND = 0.5 # 線からこれ以上先の点へ届いた視線だけを、通り抜けた証拠にする（棚の帯は 0.4 m ほど）
SIGHT_MIN = 2      # 深度の外れ値 1 本では通路と決めない
TRAIL_MARGIN = 0.2 # 歩いた線が棚の端をこれだけかすめても、棚を否定しない
GLASS_GAP = 1.2    # ガラスの散らばりは疎なので、棚の面より遠くまで 1 つの塊とみなす
GLASS_SPREAD = 0.6 # 視線方向の厚みがこれ以上なら面ではなく散らばり（棚の帯は 0.4 m ほど）
GLASS_MIN = 8
GLASS_BONUS = 12
CHILLED = {"ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品", "酒", "冷凍食品", "アイス",
           "弁当・おにぎり", "サンドイッチ・サラダ・惣菜", "チルドデザート",
           "飲料", "お酒", "パン・米飯・軽食", "冷蔵・冷凍・デザート・惣菜"}   # 上は今のアプリの 26 分類、この行は 120 カテゴリの大分類


def _quantile(values, q):
    """(値, 重み) の重み付き分位点"""
    values = sorted(values)
    limit, s = sum(w for _, w in values) * q, 0.0
    for v, w in values:
        s += w
        if s >= limit:
            return v
    return values[-1][0]


def _front(pts, ax, sign):
    """面の前面の位置。点は棚の段と深度の誤差で奥行き方向に帯になるので、中央ではなくカメラ側の端を使う"""
    return _quantile([(p[ax], p[2]) for p in pts], FRONT if sign > 0 else 1 - FRONT)


def _overlap(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)


class StoreModel:
    def __init__(self, themes=None):
        self.themes = THEMES if themes is None else themes
        self.cells = {}      # (i, k) -> [回数, x の和, z の和, カメラ方向 x の和, カメラ方向 z の和, {カテゴリ: 回数}]
        self.counters = {}   # 腰の高さの水平面の格子 -> 回数
        self.trail = []      # 歩いた跡（30 cm ごと）
        self.genre = {}      # カテゴリ -> ジャンル
        self.obs = 0
        self.yaw = 0.0       # ラジアン
        self.yaw_at = 0      # 全範囲で向きを探したときの観測数

    def add_product(self, cam_x, cam_z, x, z, category, genre):
        """商品 1 つの観測（どちらも世界座標）。category が空なら位置だけ"""
        d = hypot(cam_x - x, cam_z - z)
        if d < 1e-6:
            return
        c = self.cells.setdefault((floor(x / CELL), floor(z / CELL)), [0.0, 0.0, 0.0, 0.0, 0.0, {}, []])
        c[0] += 1
        c[1] += x
        c[2] += z
        c[3] += (cam_x - x) / d
        c[4] += (cam_z - z) / d
        if all(hypot(v[0] - cam_x, v[1] - cam_z) >= 0.35 for v in c[6]):
            if len(c[6]) == 4:
                c[6].pop(0)
            c[6].append((cam_x, cam_z))
        if category:   # 名前なしの検出枠も、棚の形の手がかりにする
            c[5][category] = c[5].get(category, 0) + 1
            self.genre[category] = genre
        self.obs += 1

    def add_camera(self, x, z):
        if not self.trail or hypot(self.trail[-1][0] - x, self.trail[-1][1] - z) >= 0.3:
            self.trail.append((x, z))

    def add_counter(self, x, z):
        """腰の高さの水平面の上の点（レジの手がかり）"""
        k = (floor(x / CELL), floor(z / CELL))
        self.counters[k] = self.counters.get(k, 0) + 1

    def _fit_yaw(self):
        """点が軸に揃う回転（0〜90 度）を探す。向きは変わらないので、観測が倍になったときだけ全範囲を探し直す"""
        pts = [(c[1] / c[0], c[2] / c[0], min(c[0], 5.0)) for c in self.cells.values()]

        def score(deg):
            cs, sn = cos(radians(deg)), sin(radians(deg))
            us, vs = {}, {}
            for x, z, w in pts:
                u, v = floor((x * cs + z * sn) / 0.2), floor((-x * sn + z * cs) / 0.2)
                us[u] = us.get(u, 0.0) + w
                vs[v] = vs.get(v, 0.0) + w
            return sum(w * w for w in us.values()) + sum(w * w for w in vs.values())

        def best(cands):
            top, top_score = cands[0], -1.0
            for d in cands:
                s = score(d)
                if s > top_score:
                    top, top_score = d, s
            return top

        prev = self.yaw * 180 / 3.141592653589793
        if self.yaw_at == 0 or self.obs >= self.yaw_at * 2:
            self.yaw_at = self.obs
            prev = best([float(d) for d in range(0, 90, 3)])
        d = best([prev + 0.5 * k for k in range(-6, 7)])
        self.yaw = radians(d % 90.0)

    def _frame(self):
        cs, sn = cos(self.yaw), sin(self.yaw)
        pts = []
        for c in self.cells.values():
            x, z, nx, nz = c[1] / c[0], c[2] / c[0], c[3], c[4]
            cameras = [(cam_x * cs + cam_z * sn, -cam_x * sn + cam_z * cs) for cam_x, cam_z in c[6]]
            pts.append([x * cs + z * sn, -x * sn + z * cs, c[0], nx * cs + nz * sn, -nx * sn + nz * cs, c[5], cameras])
        trail = [(x * cs + z * sn, -x * sn + z * cs) for x, z in self.trail]
        return pts, trail

    @staticmethod
    def _cuts(trail, ax, line):
        """歩いた線がこの線を横切る位置（面に沿った座標）。そこは通路で、棚はつながらない"""
        out = []
        for p, q in zip(trail, trail[1:]):
            if (p[ax] - line) * (q[ax] - line) < 0:
                out.append(p[1 - ax] + (line - p[ax]) / (q[ax] - p[ax]) * (q[1 - ax] - p[1 - ax]))
        return sorted(out)

    @staticmethod
    def _sight_cuts(pts, ax, line):
        """視線がこの線を通り抜けて、その先の点に届いた位置。棚は透けないので、複数の視線が通った場所に棚はない"""
        hits = {}
        for p in pts:
            if abs(p[ax] - line) < SIGHT_BEYOND:   # 棚の帯の中の点は、その棚の奥行きの誤差
                continue
            for cam in p[6]:
                if (cam[ax] - line) * (p[ax] - line) < 0:
                    k = floor((cam[1 - ax] + (line - cam[ax]) / (p[ax] - cam[ax]) * (p[1 - ax] - cam[1 - ax])) / 0.2)
                    hits[k] = hits.get(k, 0) + 1
        return [(k + 0.5) * 0.2 for k, n in hits.items() if n >= SIGHT_MIN]

    def _open_at(self, pts, trail, ax, line):
        """この線の上で、棚がつながらない位置（歩いた場所と視線が通った場所）"""
        return sorted(self._cuts(trail, ax, line) + self._sight_cuts(pts, ax, line))

    def _runs(self, pts, trail):
        """軸に平行な棚の面を、点の多い線から順に取り出す。線は歩いた線で区切り、面にならなかった点は直交する棚のために残す"""
        runs, left, tried, cache = [], pts, set(), {}

        def cuts(ax, b):
            if (ax, b) not in cache:
                cache[ax, b] = self._open_at(pts, trail, ax, (b + 0.5) * 0.05)
            return cache[ax, b]

        while len(left) >= 5:
            score, axis, center, line = 0, 0, 0.0, 0
            for ax in (0, 1):
                bins = {}
                for p in left:
                    bins.setdefault(floor(p[ax] / 0.05), set()).add(floor(p[1 - ax] / 0.2))
                for b in sorted(bins):
                    if (ax, b) in tried:
                        continue
                    pieces = {}
                    for i in range(-3, 4):
                        for y in bins.get(b + i, ()):
                            k = bisect(cuts(ax, b), (y + 0.5) * 0.2)
                            pieces[k] = pieces.get(k, 0) + 1
                    s = max(pieces.values(), default=0)
                    if s > score:
                        score, axis, center, line = s, ax, (b + 0.5) * 0.05, b
            if score < 4:
                break
            tried.add((axis, line))
            walked = cuts(axis, line)
            band = sorted((p for p in left if abs(p[axis] - center) <= BAND), key=lambda p: p[1 - axis])
            taken, seg = set(), []
            for p in band + [None]:
                if seg and (p is None or p[1 - axis] - seg[-1][1 - axis] > GAP or bisect(walked, p[1 - axis]) != bisect(walked, seg[-1][1 - axis])):
                    a, b, w = seg[0][1 - axis], seg[-1][1 - axis], sum(q[2] for q in seg)
                    if b - a >= MIN_LEN and w >= MIN_OBS:
                        sign = 1 if sum(q[3 + axis] for q in seg) >= 0 else -1
                        runs.append({"ax": axis, "c": _front(seg, axis, sign), "a": a, "b": b, "s": sign, "w": w, "pts": seg})
                        taken |= {id(q) for q in seg}
                    seg = []
                if p is not None:
                    seg.append(p)
            left = [p for p in left if id(p) not in taken]
        return runs

    def _merge(self, runs, trail, pts):
        """同じ線の上で少しだけ途切れた面をつなぐ。間を歩いたか視線が通っていたら、通路なのでつながない"""
        out = []
        for r in sorted(runs, key=lambda r: (r["ax"], r["s"], r["a"])):
            hit = next((m for m in out if m["ax"] == r["ax"] and m["s"] == r["s"] and abs(m["c"] - r["c"]) < SAME_FACE and r["a"] - m["b"] < 1.0 and
                        not any(m["b"] <= t <= r["a"] for t in self._open_at(pts, trail, m["ax"], (m["c"] + r["c"]) / 2))), None)
            if hit is None:
                out.append(r)
            else:
                hit["b"], hit["w"], hit["pts"] = max(hit["b"], r["b"]), hit["w"] + r["w"], hit["pts"] + r["pts"]
                hit["c"] = _front(hit["pts"], hit["ax"], hit["s"])
        return out

    @staticmethod
    def _snap_corners(runs):
        """直交する 2 つの面の端が近ければ、そこは棚の角。両方の面の端を交点にそろえて、棚を角に当てる"""
        ends = []
        for r in runs:
            if r["ax"] != 0:
                continue
            for q in runs:
                if q["ax"] != 1:
                    continue
                for re_ in ("a", "b"):
                    for qe in ("a", "b"):
                        d = hypot(r[re_] - q["c"], q[qe] - r["c"])
                        if d < CORNER_R:
                            ends.append((d, r, re_, q, qe))
        used = set()
        for _, r, re_, q, qe in sorted(ends, key=lambda e: e[0]):
            if (id(r), re_) in used or (id(q), qe) in used:
                continue
            ra, qa = (q["c"] if re_ == "a" else r["a"]), (r["c"] if qe == "a" else q["a"])
            rb, qb = (q["c"] if re_ == "b" else r["b"]), (r["c"] if qe == "b" else q["b"])
            if rb - ra < MIN_LEN or qb - qa < MIN_LEN:
                continue
            used |= {(id(r), re_), (id(q), qe)}
            r["a"], r["b"], q["a"], q["b"] = ra, rb, qa, qb
            r["corner"] = q["corner"] = True

    @staticmethod
    def _has_wall_behind(r, runs, trail):
        """面の後ろを歩いたか、後ろに別の棚の面があれば、壁際の什器ではない"""
        back = lambda cross: (r["c"] - cross) * r["s"]
        t = 1 - r["ax"]
        if any(r["a"] <= p[t] <= r["b"] and back(p[r["ax"]]) > 0.9 for p in trail):
            return False
        return not any(o is not r and o["ax"] == r["ax"] and back(o["c"]) > 1.2 and _overlap(r["a"], r["b"], o["a"], o["b"]) > 0.3 for o in runs)

    @staticmethod
    def _island_pairs(runs):
        """Maximum-score one-to-one matching of observed opposite rack faces."""
        fronts = [r for r in runs if r["s"] == 1 and r["b"] - r["a"] >= 0.8]
        backs = [r for r in runs if r["s"] == -1 and r["b"] - r["a"] >= 0.8]
        options = []
        for a in fronts:
            row = []
            for j, b in enumerate(backs):
                if a["ax"] != b["ax"]:
                    continue
                depth = a["c"] - b["c"]
                if not 0.5 <= depth <= 1.3:
                    continue
                overlap = _overlap(a["a"], a["b"], b["a"], b["b"])
                shorter = min(a["b"] - a["a"], b["b"] - b["a"])
                if overlap <= 0.3 * shorter:
                    continue
                score = overlap / shorter + (1 - abs(depth - ISLAND_DEPTH) / ISLAND_DEPTH) + min(a["w"], b["w"]) / (a["w"] + b["w"])
                row.append((j, score))
            options.append(row)
        if len(backs) > 18:
            ranked = sorted(((i, j, score) for i, row in enumerate(options) for j, score in row), key=lambda x: -x[2])
            used_a, used_b, chosen = set(), set(), []
            for i, j, _ in ranked:
                if i not in used_a and j not in used_b:
                    used_a.add(i)
                    used_b.add(j)
                    chosen.append((fronts[i], backs[j]))
            return chosen
        from functools import lru_cache

        @lru_cache(None)
        def solve(i, used):
            if i == len(fronts):
                return 0.0, ()
            best = solve(i + 1, used)
            for j, score in options[i]:
                if used & (1 << j):
                    continue
                rest_score, rest = solve(i + 1, used | (1 << j))
                if score + rest_score > best[0]:
                    best = score + rest_score, ((i, j),) + rest
            return best

        return [(fronts[i], backs[j]) for i, j in solve(0, 0)[1]]

    def _classify(self, runs, trail):
        """面を什器に組み立てる。戻り値は (種類, 軸, 中心, a, b, 奥行き, [面...])"""
        out, islands, used = [], [], set()
        for A, B in self._island_pairs(runs):
            used.add(id(A))
            used.add(id(B))
            islands.append(("island", A["ax"], (A["c"] + B["c"]) / 2, min(A["a"], B["a"]), max(A["b"], B["b"]), A["c"] - B["c"], [A, B]))

        def take_ends():
            for r in runs:
                if id(r) in used or r["b"] - r["a"] > 1.5:
                    continue
                mid = (r["a"] + r["b"]) / 2
                for _, iax, ic, ia, ib, depth, _ in islands:
                    at_b = r["s"] == 1 and ib - 0.2 <= r["c"] <= ib + 0.8
                    at_a = r["s"] == -1 and ia - 0.8 <= r["c"] <= ia + 0.2
                    covers_edge = r["a"] <= ic - depth / 2 + 0.25 and r["b"] >= ic + depth / 2 - 0.25
                    if r["ax"] == 1 - iax and abs(mid - ic) < 0.3 and covers_edge and (at_a or at_b):
                        used.add(id(r))
                        out.append(("end", r["ax"], r["c"] - r["s"] * END_DEPTH / 2, ic - 0.45, ic + 0.45, END_DEPTH, [r]))
                        break

        take_ends()
        for r in runs:   # 片面しか見ていない島。後ろを歩いたか、後ろに棚が見えている
            if id(r) not in used and r["b"] - r["a"] >= 0.8 and not self._has_wall_behind(r, runs, trail):
                used.add(id(r))
                islands.append(("island", r["ax"], r["c"] - r["s"] * ISLAND_DEPTH / 2, r["a"], r["b"], ISLAND_DEPTH, [r]))
        take_ends()
        for r in runs:
            if id(r) in used:
                continue
            if self._has_wall_behind(r, runs, trail):
                named = [(c, n) for p in r["pts"] for c, n in p[5].items()]
                chilled = sum(n for c, n in named if self.genre.get(c) in CHILLED)
                kind, depth = ("case", CASE_DEPTH) if named and chilled >= sum(n for _, n in named) / 2 else ("wall", GONDOLA_DEPTH)
                out.append((kind, r["ax"], r["c"] - r["s"] * depth / 2, r["a"], r["b"], depth, [r]))
            else:
                islands.append(("island", r["ax"], r["c"] - r["s"] * ISLAND_DEPTH / 2, r["a"], r["b"], ISLAND_DEPTH, [r]))
        return islands + out

    @staticmethod
    def _layout(a, b, kind, faces, axis):
        """Fit mixed 0.9/1.2 m modules; strong endpoint observations can adjust the prior."""
        mid = (a + b) / 2
        if kind == "end":
            return mid - 0.45, [0.9]
        measured = b - a
        best3 = best4 = 0
        best_cost = float("inf")
        for n3 in range(13):
            for n4 in range(13):
                n = n3 + n4
                if not 1 <= n <= 12:
                    continue
                cost = abs(0.9 * n3 + 1.2 * n4 - measured) + n * 0.005
                if cost < best_cost:
                    best_cost, best3, best4 = cost, n3, n4
        count = best3 + best4
        prior = best3 * 0.9 + best4 * 1.2
        observations = sum(r["w"] for r in faces)
        # 角に当たった端は測った位置を信じる
        trust = 1.0 if any(r.get("corner") for r in faces) else max(0.0, min(1.0, observations / 30.0))
        target = prior + (measured - prior) * trust
        correction = (target - prior) / count
        small, large = 0.9 + correction, 1.2 + correction
        start = mid - target / 2
        pts = [p for r in faces for p in r["pts"]]
        best, best_evidence = [], float("-inf")

        def evaluate(widths):
            nonlocal best, best_evidence
            cats = [{} for _ in widths]
            at = start
            for i, width in enumerate(widths):
                end = at + width
                for p in pts:
                    coord = p[1 - axis]
                    if coord >= at and (coord < end or i == len(widths) - 1):
                        for cat, weight in p[5].items():
                            cats[i][cat] = cats[i].get(cat, 0) + weight
                at = end
            evidence = sum(max(m.values(), default=0) for m in cats)
            if evidence > best_evidence + 1e-6:
                best_evidence, best = evidence, widths[:]

        def enumerate_widths(n3, n4, widths):
            if n3 + n4 == 0:
                evaluate(widths)
                return
            if n3:
                widths.append(small)
                enumerate_widths(n3 - 1, n4, widths)
                widths.pop()
            if n4:
                widths.append(large)
                enumerate_widths(n3, n4 - 1, widths)
                widths.pop()

        enumerate_widths(best3, best4, [])
        return start, best

    def _in_theme(self, theme, cat):
        return cat in theme["cats"] or self.genre.get(cat) in theme["genres"]

    def _assign_themes(self, mods):
        """モジュールごとに、観測に一番合うテーマを選ぶ（決められなければ None）"""
        if not self.themes:
            return [None] * len(mods)
        n = len({c for t in self.themes for c in t["cats"]})
        scores = [[sum(v * (log(THEME_IN / len(t["cats"]) + (1 - THEME_IN) / n) if self._in_theme(t, c) else log((1 - THEME_IN) / n)) for c, v in m.items())
                   for t in self.themes] for m in mods]
        out = []
        for i, m in enumerate(mods):
            total = sum(m.values())
            if total < MIN_THEME_OBS:
                out.append(None)
                continue
            near = [scores[j] for j in (i - 1, i + 1) if 0 <= j < len(mods)]
            s = [scores[i][k] + THEME_NEIGHBOR * sum(x[k] for x in near) for k in range(len(self.themes))]
            theme = self.themes[s.index(max(s))]
            share = sum(v for c, v in m.items() if self._in_theme(theme, c)) / total
            out.append(theme if share >= MIN_THEME_SHARE else None)
        return out

    def _register(self, cs, sn):
        if len(self.counters) < 6:
            return None
        us = sorted(((i + 0.5) * CELL * cs + (k + 0.5) * CELL * sn) for i, k in self.counters)
        vs = sorted((-(i + 0.5) * CELL * sn + (k + 0.5) * CELL * cs) for i, k in self.counters)
        cut = len(us) // 20
        u0, u1, v0, v1 = us[cut], us[-1 - cut], vs[cut], vs[-1 - cut]
        axis = 0 if u1 - u0 < v1 - v0 else 1
        c, a, b = ((u0 + u1) / 2, v0, v1) if axis == 0 else ((v0 + v1) / 2, u0, u1)
        return {"kind": "register", "axis": axis, "c": c, "a": a, "b": b, "d": max(0.5, (u1 - u0) if axis == 0 else (v1 - v0)), "h": HEIGHT["register"], "n": 1, "w": b - a, "faces": {}, "themes": {}, "obs": sum(self.counters.values())}

    @staticmethod
    def _rect(f):
        c, a, b, depth = f["c"], f["a"], f["b"], f["d"]
        return (c - depth / 2, a, c + depth / 2, b) if f["axis"] == 0 else (a, c - depth / 2, b, c + depth / 2)

    @staticmethod
    def _crosses(p, q, rect, inset):
        lo = (rect[0] + inset, rect[1] + inset)
        hi = (rect[2] - inset, rect[3] - inset)
        if lo[0] >= hi[0] or lo[1] >= hi[1]:
            return False
        enter, leave = 0.0, 1.0
        for axis in (0, 1):
            delta = q[axis] - p[axis]
            if abs(delta) < 1e-9:
                if p[axis] <= lo[axis] or p[axis] >= hi[axis]:
                    return False
            else:
                a, b = (lo[axis] - p[axis]) / delta, (hi[axis] - p[axis]) / delta
                enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
                if enter >= leave:
                    return False
        return True

    @staticmethod
    def _corner_support(draft, drafts):
        def endpoints(face):
            c, a, b = face["c"], face["a"], face["b"]
            return ((c, a), (c, b)) if face["ax"] == 0 else ((a, c), (b, c))
        count = 0
        for face in draft[6]:
            for other in drafts:
                if other is draft:
                    continue
                for perpendicular in other[6]:
                    if perpendicular["ax"] != face["ax"] and any(
                            hypot(a[0] - b[0], a[1] - b[1]) < 0.35
                            for a in endpoints(face) for b in endpoints(perpendicular)):
                        count += 1
        return count

    def _glass_candidates(self, rest):
        """どの棚でも説明できない点の塊のうち、視線方向に大きくばらけたものはガラス越しの深度の失敗とみなす。
        ガラスは塊のカメラ側の端にあるとして、各点をカメラからの視線に沿ってガラス面へ戻す"""
        groups, left = [], list(rest)
        while left:
            group, frontier = [], [left.pop(0)]
            while frontier:
                p = frontier.pop()
                group.append(p)
                near = [q for q in left if hypot(p[0] - q[0], p[1] - q[1]) <= GLASS_GAP]
                left = [q for q in left if hypot(p[0] - q[0], p[1] - q[1]) > GLASS_GAP]
                frontier += near
            groups.append(group)
        out = []
        for group in groups:
            views = [(p, cam) for p in group for cam in p[6]]
            if len(group) < GLASS_MIN or not views:
                continue
            ax = 0 if sum(abs(p[0] - c[0]) for p, c in views) >= sum(abs(p[1] - c[1]) for p, c in views) else 1
            sign = 1 if sum(c[ax] - p[ax] for p, c in views) >= 0 else -1
            values = [(p[ax], p[2]) for p in group]
            if _quantile(values, 0.9) - _quantile(values, 0.1) < GLASS_SPREAD:
                continue
            face = _quantile(values, 0.9 if sign > 0 else 0.1)
            moved = []
            for p in group:
                hits = [restored_along(c[ax], c[1 - ax], p[ax], p[1 - ax], face, sign) for c in p[6]]
                hits = [h for h in hits if h is not None]
                if hits:
                    along = sum(hits) / len(hits)
                    moved.append([face, along, *p[2:]] if ax == 0 else [along, face, *p[2:]])
            if len(moved) < GLASS_MIN:
                continue
            a, b = min(p[1 - ax] for p in moved), max(p[1 - ax] for p in moved)
            if b - a < MIN_LEN:
                continue
            run = {"ax": ax, "c": face, "a": a, "b": b, "s": sign, "w": sum(p[2] for p in moved), "pts": moved}
            draft = ("glass", ax, face - sign * CASE_DEPTH / 2, a, b, CASE_DEPTH, [run])
            out.append({"draft": draft, "fixture": self._build(draft), "explains": {id(p) for p in group}, "bonus": GLASS_BONUS})
        return out

    def _valid(self, fixture, draft, pts, trail):
        rect = self._rect(fixture)
        if any(self._crosses(p, p, rect, TRAIL_MARGIN) for p in trail) or any(
                self._crosses(a, b, rect, TRAIL_MARGIN) for a, b in zip(trail, trail[1:])):
            return False
        wrong_side = views = 0
        for face in draft[6]:
            for p in face["pts"]:
                for cam in p[6]:
                    views += 1
                    if (cam[face["ax"]] - face["c"]) * face["s"] < -0.1:
                        wrong_side += 1
        if views >= 4 and wrong_side * 4 > views:
            return False
        own = {id(p) for face in draft[6] for p in face["pts"]}
        # 正面側のカメラから見えた点は、その棚の奥行きの誤差なので矛盾しない
        core = sum(id(p) not in own and rect[0] + 0.15 < p[0] < rect[2] - 0.15 and
                   rect[1] + 0.15 < p[1] < rect[3] - 0.15 and
                   not any((cam[face["ax"]] - face["c"]) * face["s"] > 0 for face in draft[6] for cam in p[6]) for p in pts)
        if fixture["kind"] == "glass":   # ガラスの奥に点があるのは当然なので、下は調べない
            return True
        return core <= max(6, len(pts) // 50)

    def _see_through(self, fixture, pts):
        """カメラから棚の向こうの点へ、棚を突き抜けて届いた点。不透明な棚では起きない"""
        rect, across, out = self._rect(fixture), fixture["axis"], set()
        low, high = fixture["c"] - fixture["d"] / 2 - 0.2, fixture["c"] + fixture["d"] / 2 + 0.2
        for p in pts:
            for cam in p[6]:
                if ((cam[across] < low and p[across] > high) or (cam[across] > high and p[across] < low)) and self._crosses(cam, p[:2], rect, 0.1):
                    out.add(id(p))
                    break
        return out

    @staticmethod
    def _glass_views(own):
        return max(3, own // 8)

    @staticmethod
    def _glass_front(fixture, draft):
        sign = draft[6][0]["s"]
        face = fixture["c"] + sign * fixture["d"] / 2
        return {**fixture, "kind": "glass", "c": face - sign * CASE_DEPTH / 2, "d": CASE_DEPTH, "h": HEIGHT["glass"]}

    def _overlaps(self, f, g):
        r, q = self._rect(f), self._rect(g)
        return min(r[2], q[2]) - max(r[0], q[0]) > TOUCH and min(r[3], q[3]) - max(r[1], q[1]) > TOUCH

    def _trim(self, candidate, other):
        """棚を長手方向に削って other に当たらないようにする。エンドは幅が決まっているので削らない"""
        kind, axis, c, a, b, depth, faces = candidate["draft"]
        if kind == "end":
            return False
        r, along = self._rect(other), 1 - axis
        lo, hi = r[along], r[along + 2]
        a2, b2 = (a, min(b, lo)) if (a + b) / 2 < (lo + hi) / 2 else (max(a, hi), b)
        if b2 - a2 < MIN_LEN or (a2, b2) == (a, b):
            return False
        candidate["draft"] = (kind, axis, c, a2, b2, depth, faces)
        candidate["fixture"] = self._build(candidate["draft"])
        return True

    def _end_touches_one_island(self, f, accepted):
        """エンドは島 1 つの端に付く。2 つの列をつなぐと通路がふさがる"""
        if f["kind"] != "end":
            return True
        rect, contacts = self._rect(f), 0
        for other in accepted:
            if other["fixture"]["kind"] == "island":
                q = self._rect(other["fixture"])
                contacts += rect[0] <= q[2] + 0.15 and rect[2] >= q[0] - 0.15 and rect[1] <= q[3] + 0.15 and rect[3] >= q[1] - 0.15
        return contacts == 1

    def _place(self, candidate, accepted):
        """棚は重なれないので、当たる棚を削る。削っても解けないなら置かない"""
        for _ in range(4):
            other = next((o for o in accepted if self._overlaps(candidate["fixture"], o["fixture"])), None)
            if other is None:
                return self._end_touches_one_island(candidate["fixture"], accepted)
            # 既に置いた棚を削るのは、幅の決まったエンドを島の先端に付けるときだけ
            if not (self._trim(candidate, other["fixture"]) or (candidate["fixture"]["kind"] == "end" and self._trim(other, candidate["fixture"]))):
                return False
        return False

    def _choose(self, candidates, accepted):
        """新しく説明できる点が棚のコストを上回る間だけ、いちばん得な棚を足す。点を説明できる最も少ない棚の組を選ぶため"""
        explained = set().union(*(c["explains"] for c in accepted))
        left = list(candidates)

        def gain(c):
            return len(c["explains"] - explained) + c["bonus"] - SHELF_COST - MODULE_COST * c["fixture"]["n"]

        while left:
            best = max(left, key=lambda c: (gain(c), c["fixture"]["kind"] != "end", -c["fixture"]["c"]))
            left.remove(best)
            if gain(best) <= 0:
                break
            if self._place(best, accepted):
                accepted.append(best)
                explained |= best["explains"]
        return accepted

    def _build(self, draft):
        kind, axis, c, a, b, depth, faces = draft
        a, widths = self._layout(a, b, kind, faces, axis)
        n = len(widths)
        w = sum(widths) / n
        edges = [a]
        for width in widths:
            edges.append(edges[-1] + width)
        per, theme_of = {}, {}
        for sign in (1, -1):   # 什器の軸の正の向きを見ている面が +
            sides = [r for r in faces if r["s"] == sign]
            if sides:
                mods = [{} for _ in range(n)]
                for r in sides:
                    for p in r["pts"]:
                        coord = p[1 - r["ax"]]
                        index = next((i for i in range(n - 1) if coord < edges[i + 1]), n - 1)
                        m = mods[index]
                        for cat, v in p[5].items():
                            m[cat] = m.get(cat, 0) + v
                side = "+" if sign == 1 else "-"
                theme_of[side] = self._assign_themes(mods)
                kept = [{c: v for c, v in m.items() if t is None or self._in_theme(t, c)} for m, t in zip(mods, theme_of[side])]
                per[side] = [[[cat, v / (sum(m.values()) or 1)] for cat, v in sorted(m.items(), key=lambda x: (-x[1], x[0]))[:3]] for m in kept]
        return {"kind": kind, "axis": axis, "c": c, "a": a, "b": edges[-1], "d": depth, "h": HEIGHT[kind], "n": n, "w": w, "widths": widths, "faces": per, "themes": {k: [t["name"] if t else None for t in v] for k, v in theme_of.items()}, "obs": sum(r["w"] for r in faces)}

    def estimate(self):
        """今の観測から店のモデルを作る"""
        if len(self.cells) < 30:
            return {"yaw": 0.0, "room": None, "fixtures": []}
        self._fit_yaw()
        pts, trail = self._frame()
        runs = self._merge(self._runs(pts, trail), trail, pts)
        self._snap_corners(runs)
        drafts = self._classify(runs, trail)
        opaque, doubtful, pending = [], [], list(drafts)
        while pending:
            draft = pending.pop(0)
            kind, axis, c, a, b, depth, faces = draft
            own = {id(p) for face in faces for p in face["pts"]}
            # 面があれば、その裏に棚がある。歩いた線や視線に当たるなら棚を薄くして、面の位置は動かさない
            fits = []
            for scale in (DEPTH_SCALES if len(faces) == 1 and kind != "end" else (1.0,)):
                d = depth * scale
                trial = (kind, axis, faces[0]["c"] - faces[0]["s"] * d / 2 if len(faces) == 1 and kind != "end" else c, a, b, d, faces)
                fixture = self._build(trial)
                if self._valid(fixture, trial, pts, trail):
                    fits.append((trial, fixture, self._see_through(fixture, pts)))
            clear = [f for f in fits if len(f[2]) < self._glass_views(len(own))]
            if not clear and len(faces) == 2:   # 島として成り立たなくても、面ごとの棚は残す
                pending += [("wall", r["ax"], r["c"] - r["s"] * GONDOLA_DEPTH / 2, r["a"], r["b"], GONDOLA_DEPTH, [r]) for r in faces]
                continue
            chosen = (clear or fits or [None])[0]
            if chosen is None:
                continue
            draft, fixture, through = chosen
            rect = self._rect(fixture)
            near = {id(p) for p in pts if rect[0] - EXPLAIN <= p[0] <= rect[2] + EXPLAIN and rect[1] - EXPLAIN <= p[1] <= rect[3] + EXPLAIN}
            item = {"draft": draft, "fixture": fixture, "explains": near | own, "bonus": CORNER_BONUS * self._corner_support(draft, drafts), "through": through, "own": len(own)}
            if clear:
                opaque.append(item)
            elif fixture["kind"] in ("wall", "case"):   # 視線に貫かれる棚はない。残る説明は壁際のガラスだけ
                doubtful.append(item)
        accepted = self._choose(opaque, [])
        explained = set().union(*(c["explains"] for c in accepted))
        for c in doubtful:
            c["fixture"] = self._glass_front(c["fixture"], c["draft"])
            c["explains"] = c["explains"] | (c["through"] - explained)
        accepted = self._choose(doubtful, accepted)
        explained = set().union(*(c["explains"] for c in accepted))
        glass = [c for c in self._glass_candidates([p for p in pts if id(p) not in explained])
                 if self._valid(c["fixture"], c["draft"], pts, trail)]
        accepted = self._choose(glass, accepted)
        fixtures = []
        for c in accepted:   # 削った棚はモジュール幅を合わせ直すので、最後にもう一度確かめる
            if not any(self._overlaps(c["fixture"], f) for f in fixtures):
                fixtures.append(c["fixture"])
        reg = self._register(cos(self.yaw), sin(self.yaw))
        if reg and not any(self._overlaps(reg, f) for f in fixtures):
            fixtures.append(reg)
        fixtures.sort(key=lambda f: (f["kind"], f["axis"], f["c"], f["a"]))
        return {"yaw": self.yaw, "room": self._room(fixtures, trail), "fixtures": fixtures}

    @staticmethod
    def _room(fixtures, trail):
        """壁際の什器の背中と、歩いた範囲から部屋を決める"""
        low, high = [[], []], [[], []]
        for f in fixtures:
            for ax, lo, hi in ((f["axis"], f["c"] - f["d"] / 2, f["c"] + f["d"] / 2), (1 - f["axis"], f["a"], f["b"])):
                low[ax].append(lo)
                high[ax].append(hi)
        for p in trail:
            for ax in (0, 1):
                low[ax].append(p[ax] - 0.5)
                high[ax].append(p[ax] + 0.5)
        return [min(low[0]), min(low[1]), max(high[0]), max(high[1])]


def restored_along(cam_ax, cam_along, x_ax, x_along, face, sign):
    """カメラから点への視線がガラス面 face と交わる位置（面に沿った座標）。視線がガラスに届かない点は None"""
    if (cam_ax - face) * sign < 0.3 or abs(x_ax - cam_ax) < 1e-6:
        return None
    t = (face - cam_ax) / (x_ax - cam_ax)
    return cam_along + t * (x_along - cam_along) if 0.05 <= t <= 1.3 else None


def glass_restored(view, cam_x, cam_z, x, z):
    """ガラス越しに測った点 (x, z) を、ガラス面の上へ戻した世界座標。ガラスに当たらない点は None（アプリの GlassProjection と同じ）"""
    cs, sn = cos(view["yaw"]), sin(view["yaw"])
    cam = (cam_x * cs + cam_z * sn, -cam_x * sn + cam_z * cs)
    raw = (x * cs + z * sn, -x * sn + z * cs)
    for f in view["fixtures"]:
        if f["kind"] != "glass":
            continue
        ax, sign = f["axis"], 1 if "+" in f["faces"] else -1
        face = f["c"] + sign * f["d"] / 2
        along = restored_along(cam[ax], cam[1 - ax], raw[ax], raw[1 - ax], face, sign)
        if along is not None and f["a"] <= along <= f["b"]:
            u, v = (face, along) if ax == 0 else (along, face)
            return u * cs - v * sn, u * sn + v * cs
    return None


def _fmt(v, digits=2):
    """四捨五入の桁そろえ。書式の丸めは言語で違う（0.125 が Python は 0.12、Java は 0.13）のでアプリと揃える"""
    k = 10 ** digits
    return f"{floor(v * k + 0.5) / k:.{digits}f}"


def text(model):
    """記述言語の文字列にする（cm の桁まで）"""
    if not model["fixtures"]:
        return "store (empty)"
    r = model["room"]
    lines = [f"store yaw={_fmt(model['yaw'] * 180 / 3.141592653589793, 1)} room={_fmt(r[0])},{_fmt(r[1])}..{_fmt(r[2])},{_fmt(r[3])}"]
    for f in model["fixtures"]:
        widths = f.get("widths", [f["w"]] * f["n"])
        modules = f"{f['n']}x{_fmt(f['w'])}" if all(abs(w - f["w"]) < 0.005 for w in widths) else "+".join(_fmt(w) for w in widths)
        lines.append(f"{f['kind']} axis={'uv'[f['axis']]} c={_fmt(f['c'])} span={_fmt(f['a'])}..{_fmt(f['b'])} d={_fmt(f['d'])} h={_fmt(f['h'], 1)} mods={modules}")
        for i in range(f["n"]):
            parts = [sign + (f"[{f['themes'][sign][i]}]" if f["themes"][sign][i] else "") + ",".join(f"{c}:{_fmt(s)}" for c, s in faces[i])
                     for sign, faces in sorted(f["faces"].items()) if i < len(faces) and faces[i]]
            if parts:
                lines.append(f"  m{i} " + " ".join(parts))
    return chr(10).join(lines)
