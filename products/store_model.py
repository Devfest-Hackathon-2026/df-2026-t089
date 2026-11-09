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
WIDTHS = (0.9, 1.2)
HEIGHT = {"island": 1.8, "wall": 1.8, "case": 2.0, "end": 1.5, "register": 1.0}
CHILLED = {"ソフトドリンク", "栄養・エナジードリンク", "紙パック飲料・乳製品", "酒", "冷凍食品", "アイス",
           "弁当・おにぎり", "サンドイッチ・サラダ・惣菜", "チルドデザート",
           "飲料", "お酒", "パン・米飯・軽食", "冷蔵・冷凍・デザート・惣菜"}   # 上は今のアプリの 26 分類、この行は 120 カテゴリの大分類


def _median(values):
    """(値, 重み) の重み付き中央値"""
    values = sorted(values)
    half, s = sum(w for _, w in values) / 2, 0.0
    for v, w in values:
        s += w
        if s >= half:
            return v
    return values[-1][0]


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
        c = self.cells.setdefault((floor(x / CELL), floor(z / CELL)), [0.0, 0.0, 0.0, 0.0, 0.0, {}])
        c[0] += 1
        c[1] += x
        c[2] += z
        c[3] += (cam_x - x) / d
        c[4] += (cam_z - z) / d
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
            pts.append([x * cs + z * sn, -x * sn + z * cs, c[0], nx * cs + nz * sn, -nx * sn + nz * cs, c[5]])
        trail = [(x * cs + z * sn, -x * sn + z * cs) for x, z in self.trail]
        return pts, trail

    @staticmethod
    def _runs(pts):
        """軸に平行な棚の面を、点の多い線から順に取り出す"""
        runs, left = [], pts
        while len(left) >= 5:
            score, axis, center = 0, 0, 0.0
            for ax in (0, 1):
                bins = {}
                for p in left:
                    bins.setdefault(floor(p[ax] / 0.05), set()).add(floor(p[1 - ax] / 0.2))
                for b in sorted(bins):
                    s = sum(len(bins.get(b + i, ())) for i in range(-3, 4))
                    if s > score:
                        score, axis, center = s, ax, (b + 0.5) * 0.05
            if score < 4:
                break
            band = sorted((p for p in left if abs(p[axis] - center) <= BAND), key=lambda p: p[1 - axis])
            left = [p for p in left if abs(p[axis] - center) > BAND]
            seg = []
            for p in band + [None]:
                if seg and (p is None or p[1 - axis] - seg[-1][1 - axis] > GAP):
                    a, b, w = seg[0][1 - axis], seg[-1][1 - axis], sum(q[2] for q in seg)
                    if b - a >= MIN_LEN and w >= MIN_OBS:
                        sign = 1 if sum(q[3 + axis] for q in seg) >= 0 else -1
                        runs.append({"ax": axis, "c": _median([(q[axis], q[2]) for q in seg]), "a": a, "b": b, "s": sign, "w": w, "pts": seg})
                    seg = []
                if p is not None:
                    seg.append(p)
        return runs

    @staticmethod
    def _merge(runs):
        """同じ線の上で少しだけ途切れた面をつなぐ"""
        out = []
        for r in sorted(runs, key=lambda r: (r["ax"], r["s"], r["a"])):
            hit = next((m for m in out if m["ax"] == r["ax"] and m["s"] == r["s"] and abs(m["c"] - r["c"]) < SAME_FACE and r["a"] - m["b"] < 1.0), None)
            if hit is None:
                out.append(r)
            else:
                hit["c"] = (hit["c"] * hit["w"] + r["c"] * r["w"]) / (hit["w"] + r["w"])
                hit["b"], hit["w"], hit["pts"] = max(hit["b"], r["b"]), hit["w"] + r["w"], hit["pts"] + r["pts"]
        return out

    @staticmethod
    def _has_wall_behind(r, runs, trail):
        """面の後ろを歩いたか、後ろに別の棚の面があれば、壁際の什器ではない"""
        back = lambda cross: (r["c"] - cross) * r["s"]
        t = 1 - r["ax"]
        if any(r["a"] <= p[t] <= r["b"] and back(p[r["ax"]]) > 0.9 for p in trail):
            return False
        return not any(o is not r and o["ax"] == r["ax"] and back(o["c"]) > 1.2 and _overlap(r["a"], r["b"], o["a"], o["b"]) > 0.3 for o in runs)

    def _classify(self, runs, trail):
        """面を什器に組み立てる。戻り値は (種類, 軸, 中心, a, b, 奥行き, [面...])"""
        out, islands, used = [], [], set()
        for A in sorted((r for r in runs if r["s"] == 1 and r["b"] - r["a"] >= 0.8), key=lambda r: r["a"] - r["b"]):
            if id(A) in used:
                continue
            Bs = [B for B in runs if id(B) not in used and B["ax"] == A["ax"] and B["s"] == -1 and 0.5 <= A["c"] - B["c"] <= 1.3
                  and _overlap(A["a"], A["b"], B["a"], B["b"]) > 0.3 * min(A["b"] - A["a"], B["b"] - B["a"])]
            if Bs:
                used.add(id(A))
                used.update(id(B) for B in Bs)
                c = (A["c"] + sum(B["c"] for B in Bs) / len(Bs)) / 2
                islands.append(("island", A["ax"], c, min([A["a"]] + [B["a"] for B in Bs]), max([A["b"]] + [B["b"] for B in Bs]), ISLAND_DEPTH, [A] + Bs))

        def take_ends():
            for r in runs:
                if id(r) in used or r["b"] - r["a"] > 1.5:
                    continue
                mid = (r["a"] + r["b"]) / 2
                for _, iax, ic, ia, ib, _, _ in islands:
                    at_b = r["s"] == 1 and ib - 0.2 <= r["c"] <= ib + 0.8
                    at_a = r["s"] == -1 and ia - 0.8 <= r["c"] <= ia + 0.2
                    if r["ax"] == 1 - iax and abs(mid - ic) < 0.6 and (at_a or at_b):
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
    def _snap(a, b, kind):
        """幅を 0.9 / 1.2 m の倍数に揃える"""
        mid, length = (a + b) / 2, b - a
        if kind == "end":
            return mid - 0.45, 1, 0.9
        n, w = max(((max(1, round(length / w)), w) for w in WIDTHS), key=lambda x: -abs(x[0] * x[1] - length))
        return mid - n * w / 2, n, w

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

    def estimate(self):
        """今の観測から店のモデルを作る"""
        if len(self.cells) < 30:
            return {"yaw": 0.0, "room": None, "fixtures": []}
        self._fit_yaw()
        pts, trail = self._frame()
        runs = self._merge(self._runs(pts))
        fixtures = []
        for kind, axis, c, a, b, depth, faces in self._classify(runs, trail):
            a, n, w = self._snap(a, b, kind)
            per, theme_of = {}, {}
            for sign in (1, -1):   # 什器の軸の正の向きを見ている面が +
                sides = [r for r in faces if r["s"] == sign]
                if sides:
                    mods = [{} for _ in range(n)]
                    for r in sides:
                        for p in r["pts"]:
                            m = mods[min(n - 1, max(0, floor((p[1 - r["ax"]] - a) / w)))]
                            for cat, v in p[5].items():
                                m[cat] = m.get(cat, 0) + v
                    side = "+" if sign == 1 else "-"
                    theme_of[side] = self._assign_themes(mods)
                    kept = [{c: v for c, v in m.items() if t is None or self._in_theme(t, c)} for m, t in zip(mods, theme_of[side])]
                    per[side] = [[[cat, v / (sum(m.values()) or 1)] for cat, v in sorted(m.items(), key=lambda x: (-x[1], x[0]))[:3]] for m in kept]
            fixtures.append({"kind": kind, "axis": axis, "c": c, "a": a, "b": a + n * w, "d": depth, "h": HEIGHT[kind], "n": n, "w": w, "faces": per, "themes": {k: [t["name"] if t else None for t in v] for k, v in theme_of.items()}, "obs": sum(r["w"] for r in faces)})
        reg = self._register(cos(self.yaw), sin(self.yaw))
        if reg:
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
        lines.append(f"{f['kind']} axis={'uv'[f['axis']]} c={_fmt(f['c'])} span={_fmt(f['a'])}..{_fmt(f['b'])} d={_fmt(f['d'])} h={_fmt(f['h'], 1)} mods={f['n']}x{_fmt(f['w'])}")
        for i in range(f["n"]):
            parts = [sign + (f"[{f['themes'][sign][i]}]" if f["themes"][sign][i] else "") + ",".join(f"{c}:{_fmt(s)}" for c, s in faces[i])
                     for sign, faces in sorted(f["faces"].items()) if i < len(faces) and faces[i]]
            if parts:
                lines.append(f"  m{i} " + " ".join(parts))
    return chr(10).join(lines)
