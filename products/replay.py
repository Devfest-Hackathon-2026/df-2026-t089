"""端末の記録モードで保存したデータを PC に取り込み、アプリと同じ処理で認識・店モデル・案内の言葉を計算し直す。

  python products/replay.py --pull                               # 端末から取り込む -> work/recordings/<名前>/（公開しない）
  python products/replay.py --run <名前> [--detector app|sku110k-2] [--hz 1]  # -> work/recordings/<名前>/replay_<検出器>.json
  python products/replay.py --fixture                            # アプリのテストデータを作り直す（案内の処理を直したとき）

結果はダッシュボード（8771）の「記録の再生」タブで見る。ShelfMap / Unprojector / StoreGuide / SceneJudge / ShelfDescriber の移植で、
タップの答えと AI への文脈が同じになることは StoreGuideTest（Kotlin 側）で確かめる。
shortcut: 分類はまだ今のアプリの埋め込み。M1 ができたら recognizer() で差し替える
"""
import argparse, json, re, shutil, subprocess, sys, time
from math import atan2, cos, degrees, floor, hypot, radians, sin, sqrt

from common import ROOT, WORK, read_json, write_json
from store_model import StoreModel, glass_restored, text as store_text

RECORDINGS = WORK / "recordings"
REMOTE = "/sdcard/Android/data/jp.devfest.shelfsight/files/recordings"
NAME = re.compile(r"^\d{8}_\d{6}$")
DETECTORS = ("app", "sku110k-2")
SRC = ROOT / "app/src/main/java/jp/devfest/shelfsight/assist"
DEVICE_HEIGHT_M = 1.3
MAX_NAMED = 20        # ProductRecognizer.MAX_CROPS
ESTIMATE_S = 3.0      # AssistActivity.STORE_ESTIMATE_MS
SNAP_EVERY = 3        # 再生用に店モデルを残す間隔（ステップ）
RANGE_M, FACING_M, FACING_DEG, MAX_SHELVES = 3.5, 1.8, 40.0, 4
FRONT_DEG, FRONT_M, ITEMS_ONLY_DEG, ITEMS_ONLY_M = 35.0, 2.2, 20.0, 1.5
FAR_TEXT = "近くに分かっている棚がありません。棚の方へ少し歩いてください。"
EDGE_M, OWNER_MARGIN_M = 0.1, 0.35  # StoreGuide: 見えるかを調べるとき什器の両端を縮める量、商品の置かれた什器を決める余裕
KINDS_PER_BAND, UPPER_M, LOWER_M = 3, 1.25, 0.75
MIN_PRODUCTS, OUTLIER_SHARE = 3, 0.25
MAX_MAP_LINES, MAX_ITEM_LINES, MAX_LOOKED_LINES = 30, 60, 10
KIND_NAMES = {"glass": "ガラスのケース", "island": "島の棚", "wall": "壁ぎわの棚", "case": "冷蔵・冷凍ケース", "end": "島の端の棚", "register": "レジ"}


def spoken_table():
    """声で聞きやすい言い方の表。アプリの ShelfDescriber.kt から読む（二重に持たない）"""
    src = (SRC / "ShelfDescriber.kt").read_text(encoding="utf-8")
    block = src[src.index("private val SPOKEN = mapOf("):]
    return dict(re.findall(r'"([^"]+)" to "([^"]+)"', block[:block.index("\n    )")]))


SPOKEN = spoken_table()


def spoken(category):
    return SPOKEN.get(category) or category.split("・")[0]


def group_table():
    g = json.loads((ROOT / "app/src/main/assets/coarse/categories.json").read_text(encoding="utf-8"))
    return {v: k for k, vs in g.items() if not k.startswith("_") for v in vs}


GROUPS = group_table()


# ---------------------------------------------------------------- ShelfMap / Unprojector

class Item:
    def __init__(self, name, genre, fine, pos, w, h, seen, last):
        self.name, self.genre, self.fine, self.pos, self.w, self.h, self.seen, self.last = name, genre, fine, pos, w, h, seen, last


class ShelfMap:
    MERGE_M, MAX_ITEMS = 0.25, 3000

    def __init__(self):
        self.items = []

    def add(self, name, genre, fine, pos, w, h, now):
        near = [i for i in self.items if i.name == name and dist3(i.pos, pos) < self.MERGE_M]
        if not near:
            self.items.append(Item(name, genre, fine, pos, w, h, 1, now))
            if len(self.items) > self.MAX_ITEMS:
                self.items.remove(min(self.items, key=lambda i: i.last))
            return
        s = min(near, key=lambda i: dist3(i.pos, pos))
        k = 1 / (s.seen + 1)
        s.pos = tuple(a + (b - a) * k for a, b in zip(s.pos, pos))
        s.w += (w - s.w) * k
        s.h += (h - s.h) * k
        s.seen += 1
        s.last = now


def dist3(a, b):
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def place(camera, along, depth_point, walls):
    """ShelfMap.place: 深度があればその点（近くの壁に寄せる）、なければ向きを壁と交わらせる"""
    near = lambda w, p: dist3(p, w["c"]) < w["r"] + 0.5
    sd = lambda w, p: dot(w["n"], [a - b for a, b in zip(p, w["c"])])
    if depth_point is not None:
        cand = [w for w in walls if abs(sd(w, depth_point)) < 0.3 and near(w, depth_point)]
        if not cand:
            return depth_point
        w = min(cand, key=lambda w: abs(sd(w, depth_point)))
        d = sd(w, depth_point)
        return tuple(p - n * d for p, n in zip(depth_point, w["n"]))
    v = [a - b for a, b in zip(along, camera)]
    length = max(sqrt(dot(v, v)), 1e-6)
    v = [x / length for x in v]
    hits = []
    for w in walls:
        denom = dot(w["n"], v)
        if abs(denom) < 0.2:
            continue
        t = dot(w["n"], [a - b for a, b in zip(w["c"], camera)]) / denom
        if not 0.2 <= t <= 3.5:
            continue
        p = tuple(c + d * t for c, d in zip(camera, v))
        if near(w, p):
            hits.append((t, p))
    return min(hits, key=lambda x: x[0])[1] if hits else None


def upright_to_sensor(u, v, rot):
    return {90: (v, 1 - u), 180: (1 - u, 1 - v), 270: (1 - v, u)}.get(rot, (u, v))


def world_at(u, v, depth, g):
    sx, sy = upright_to_sensor(u, v, g["rot"])
    x = (sx * g["w"] - g["cx"]) / g["fx"] * depth
    y = -(sy * g["h"] - g["cy"]) / g["fy"] * depth
    z = -depth
    m = g["m"]
    return (m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14])


def depth_at(u, v, g, grid):
    w, h, data = grid
    sx, sy = upright_to_sensor(u, v, g["rot"])
    dx = int((g["u0"] + sx * g["w"] * g["du"]) * w)
    dy = int((g["v0"] + sy * g["h"] * g["dv"]) * h)
    if not (0 <= dx < w and 0 <= dy < h):
        return None
    m = data[dy * w + dx] / 1000
    return m if 0.2 <= m <= 8.0 else None


def median_depth(box, g, grid):
    """Unprojector.medianDepthMeters: 枠の中央 40% を 9x9 点で見た中央値"""
    if grid is None:
        return None
    cu, cv = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hw, hh = (box[2] - box[0]) * 0.2, (box[3] - box[1]) * 0.2
    samples = sorted(d for r in range(9) for c in range(9) if (d := depth_at(cu - hw + 2 * hw * c / 8, cv - hh + 2 * hh * r / 8, g, grid)) is not None)
    return samples[len(samples) // 2] if samples else None


def size_in_meters(w, h, depth, g):
    fu, fv, iw, ih = (g["fy"], g["fx"], g["h"], g["w"]) if g["rot"] % 180 else (g["fx"], g["fy"], g["w"], g["h"])
    return w * iw * depth / fu, h * ih * depth / fv


# ---------------------------------------------------------------- SceneJudge

def area(b):
    return max(b[2] - b[0], 0) * max(b[3] - b[1], 0)


def main_product(boxes):
    s = sorted(boxes, key=area, reverse=True)
    if not s:
        return None
    first, second = s[0], area(s[1]) if len(s) > 1 else 0
    centered = first[0] < 0.75 and first[2] > 0.25 and first[1] < 0.75 and first[3] > 0.25
    return first if area(first) >= 0.12 and area(first) >= second * 2.5 and centered else None


def focus(boxes):
    """2 回たたいたときに調べるもの（SceneJudge.focus）"""
    if not boxes:
        return "NOTHING", None
    m = main_product(boxes)
    if m:
        return "PRODUCT", m
    largest = max(boxes, key=area)
    union = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    if area(largest) >= 0.25:
        return "PRODUCT", union if len(boxes) <= 4 else largest
    if len(boxes) <= 4 and area(union) >= 0.6 and area(largest) >= 0.06:
        return "PRODUCT", union
    spread = len(boxes) >= 6 and union[2] - union[0] >= 0.5 and union[3] - union[1] >= 0.35
    return ("SHELF" if spread and not any(area(b) >= 0.12 for b in boxes) else "STEP_BACK"), None


# ---------------------------------------------------------------- StoreGuide

def direction(a):
    side = "右" if a > 0 else "左"
    b = abs(a)
    return "前" if b <= 30 else f"{side}前" if b <= 60 else side if b <= 120 else f"{side}後ろ" if b <= 150 else "後ろ"


def azimuth(l, p):
    dx, dz = p[0] - l["p"][0], p[2] - l["p"][2]
    return degrees(atan2(dx * -l["fz"] + dz * l["fx"], dx * l["fx"] + dz * l["fz"]))


def _rect(f, across, along):
    """StoreGuide.rect: 店の座標での什器の四角形 (u0, v0, u1, v1)"""
    c0, c1 = f["c"] - f["d"] / 2 - across, f["c"] + f["d"] / 2 + across
    l0, l1 = min(f["a"], f["b"]) - along, max(f["a"], f["b"]) + along
    return (c0, l0, c1, l1) if f["axis"] == 0 else (l0, c0, l1, c1)


def _crosses(x0, y0, x1, y1, r):
    """StoreGuide.crosses: 線分が四角形を通るか（端の点で触れるだけなら通らない）"""
    if r[0] > r[2] or r[1] > r[3]:
        return False
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return t0 <= t1 and t1 > 1e-6 and t0 < 1 - 1e-6


def _blocked(view, u0, v0, u1, v1, skip):
    return any(f is not skip and _crosses(u0, v0, u1, v1, _rect(f, 0.0, -EDGE_M)) for f in view["fixtures"])


def visible_items(view, items, l):
    """StoreGuide.visibleItems: 置かれた什器の裏の面と、ほかの什器の陰にある商品を外す"""
    cs, sn = cos(view["yaw"]), sin(view["yaw"])
    lu, lv = l["p"][0] * cs + l["p"][2] * sn, -l["p"][0] * sn + l["p"][2] * cs
    out = []
    for i in items:
        iu, iv = i.pos[0] * cs + i.pos[2] * sn, -i.pos[0] * sn + i.pos[2] * cs
        owners = [f for f in view["fixtures"] if (r := _rect(f, OWNER_MARGIN_M, OWNER_MARGIN_M)) and r[0] <= iu <= r[2] and r[1] <= iv <= r[3]]
        owner = min(owners, key=lambda f: abs((iu if f["axis"] == 0 else iv) - f["c"])) if owners else None
        if owner is not None:
            if len(owner["faces"]) == 1:
                side = 1.0 if "+" in owner["faces"] else -1.0
            else:
                x = (iu if owner["axis"] == 0 else iv) - owner["c"]
                side = (x > 0) - (x < 0)
            if side != 0 and ((lu if owner["axis"] == 0 else lv) - owner["c"]) * side <= 0:
                continue
        if not _blocked(view, lu, lv, iu, iv, owner):
            out.append(i)
    return out


def near_shelves(view, l):
    cs, sn = cos(view["yaw"]), sin(view["yaw"])
    u, v = l["p"][0] * cs + l["p"][2] * sn, -l["p"][0] * sn + l["p"][2] * cs
    out = []
    for f in view["fixtures"]:
        for sign, modules in f["faces"].items():
            s = 1 if sign == "+" else -1
            face = f["c"] + s * f["d"] / 2
            cross, along = (u, v) if f["axis"] == 0 else (v, u)
            if (cross - face) * s <= 0:
                continue
            center = lambda i: f["a"] + (i + 0.5) * f["w"]
            valid = [i for i in range(len(modules)) if modules[i]]
            if not valid:
                continue
            i = min(valid, key=lambda k: abs(center(k) - along))
            pu, pv = (face, center(i)) if f["axis"] == 0 else (center(i), face)
            at = (pu * cs - pv * sn, l["p"][1], pu * sn + pv * cs)
            d = hypot(at[0] - l["p"][0], at[2] - l["p"][2])
            if d > RANGE_M:
                continue
            if _blocked(view, u, v, pu, pv, f):  # ほかの什器の陰で見えない
                continue
            nu, nv = (s, 0) if f["axis"] == 0 else (0, s)
            nx, nz = nu * cs - nv * sn, nu * sn + nv * cs
            facing = d <= FACING_M and -(nx * l["fx"] + nz * l["fz"]) >= cos(radians(FACING_DEG))
            out.append({"main": modules[i][0][0], "at": at, "d": d, "az": azimuth(l, at), "facing": facing})
    return sorted(out, key=lambda s: s["d"])


def split_kept(front):
    """ShelfDescriber.split のうち、残す商品の名前。場違いな商品を外す"""
    group = lambda i: GROUPS.get(i.genre) if i.genre else None
    kind = lambda i: i.fine or i.genre
    with_group = [i for i in front if group(i)]
    if not with_group:
        return set()
    counts = {}
    for i in with_group:
        counts[group(i)] = counts.get(group(i), 0) + 1
    main = max(counts, key=counts.get)
    keep = {g for g, c in counts.items() if g == main or c >= len(with_group) * OUTLIER_SHARE}
    in_group = [i for i in front if group(i) in keep]
    kinds = {}
    for i in in_group:
        kinds[kind(i)] = kinds.get(kind(i), 0) + 1
    top = max(kinds, key=kinds.get)
    good = {k for k, c in kinds.items() if k == top or c >= len(in_group) * OUTLIER_SHARE}
    return {i.name for i in in_group if kind(i) in good}


def aimed(i, l, max_deg, max_m):
    return hypot(i.pos[0] - l["p"][0], i.pos[2] - l["p"][2]) <= max_m and abs(azimuth(l, i.pos)) <= max_deg


def facing_sentence(main, items, floor_y):
    bands = []
    for band, lst in (("上の方", [i for i in items if i.pos[1] - floor_y >= UPPER_M]),
                      ("真ん中", [i for i in items if LOWER_M <= i.pos[1] - floor_y < UPPER_M]),
                      ("下の方", [i for i in items if i.pos[1] - floor_y < LOWER_M])):
        groups = {}
        for i in lst:
            groups.setdefault(spoken(i.name), []).append(i)
        kinds = [k for k, _ in sorted(groups.items(), key=lambda e: -sum(i.seen for i in e[1]))[:KINDS_PER_BAND]]
        if kinds:
            bands.append((band, kinds))
    top = main
    if top is None:
        by = {}
        for i in items:
            by.setdefault(i.name, []).append(i)
        if not by:
            return None
        top = max(by, key=lambda n: sum(i.seen for i in by[n]))
    shelf = f"{spoken(top)}の棚"
    if not bands:
        return f"{shelf}です。"
    if len(bands) >= 2 and all(set(k) == set(bands[0][1]) for _, k in bands):  # 段がどれも同じなら「全部〇〇です」
        return f"全部{'や'.join(bands[0][1])}です。"
    return "、".join(b + "には" + "や".join(k) for b, k in bands) + "があります。"  # 棚の名前は言わない


def tap_answer(view, items, l, floor_y):
    """StoreGuide.tapAnswer"""
    shelves = near_shelves(view, l) if view else []
    items = visible_items(view, items, l) if view else items
    facing = next((s for s in shelves if s["facing"]), None)
    front = [i for i in items if aimed(i, l, FRONT_DEG, FRONT_M)] if facing else [i for i in items if aimed(i, l, ITEMS_ONLY_DEG, ITEMS_ONLY_M)] if not shelves else []
    if facing or len(front) >= MIN_PRODUCTS:
        kept = split_kept(front)
        t = facing_sentence(facing["main"] if facing else None, [i for i in front if i.name in kept], floor_y)
        if t is None and facing:
            t = facing_sentence(facing["main"], [], 0)
        if t is not None:
            return {"kind": "sentence", "text": t}
    if not shelves:
        if view and view["fixtures"]:
            return {"kind": "sentence", "text": FAR_TEXT}
        return {"kind": "unknown", "text": "まだ棚が分かりません。棚の方を向いて、少し歩いてください。"}
    speak, seen = [], set()
    for s in sorted(shelves, key=lambda s: 0 if direction(s["az"]) == "前" else 1):  # 前の棚を先に、あとは近い順
        d = direction(s["az"])
        if d not in seen:
            seen.add(d)
            speak.append(s)
    speak = speak[:MAX_SHELVES]
    clauses = [f"{direction(s['az'])}には{spoken(s['main'])}の棚" + ("があります。" if k == len(speak) - 1 else "、") for k, s in enumerate(speak)]
    return {"kind": "around", "text": "".join(clauses), "clauses": clauses, "at": [list(s["at"]) for s in speak]}


def where(at, l):
    return f"{direction(azimuth(l, at))} {hypot(at[0] - l['p'][0], at[2] - l['p'][2]):.1f}m"


def context(view, l, items, looked, last):
    """StoreGuide.context（質問に答える AI に渡す文脈）"""
    out = []
    if l is None:
        out.append("- 位置はまだ分かりません。")
    else:
        out.append("- 方向と距離は、すべて利用者の今の位置と向きから見たものです。")
        cs, sn = (cos(view["yaw"]), sin(view["yaw"])) if view else (1, 0)
        if view and view.get("room"):
            r = view["room"]
            out.append(f"- 店の広さ: 約{r[2] - r[0]:.0f}m × {r[3] - r[1]:.0f}m（歩いた範囲から推定）")
        rows = []
        for f in (view["fixtures"] if view else []):
            cu, cv = (f["c"], (f["a"] + f["b"]) / 2) if f["axis"] == 0 else ((f["a"] + f["b"]) / 2, f["c"])
            at = (cu * cs - cv * sn, l["p"][1], cu * sn + cv * cs)
            shares = {}
            for ms in f["faces"].values():
                for m in ms:
                    for c, s in m:
                        shares[c] = shares.get(c, 0) + s
            goods = "、".join(c for c, _ in sorted(shares.items(), key=lambda e: -e[1])[:4])
            themes = "、".join(dict.fromkeys(t for ts in f.get("themes", {}).values() for t in ts if t))
            label = KIND_NAMES.get(f["kind"], f["kind"]) + (f"（テーマ: {themes}）" if themes else "")
            rows.append((hypot(at[0] - l["p"][0], at[2] - l["p"][2]), at, label + (f": {goods}" if goods else "")))
        rows.sort(key=lambda r: r[0])
        if rows:
            out.append("- 店の地図（近い順）:")
            out += [f"  - {where(at, l)}: {t}" for _, at, t in rows[:MAX_MAP_LINES]]
        if items:
            out.append("- 見つけた商品の種類（いちばん近いものの場所、近い順）:")
            nearest = {}
            for i in items:
                d = hypot(i.pos[0] - l["p"][0], i.pos[2] - l["p"][2])
                if i.name not in nearest or d < nearest[i.name][0]:
                    nearest[i.name] = (d, i)
            out += [f"  - {n}: {where(i.pos, l)}" for n, (_, i) in sorted(nearest.items(), key=lambda e: e[1][0])[:MAX_ITEM_LINES]]
        if looked:
            out.append("- 利用者が商品名を調べた商品（新しい順、場所は調べたときにいた所）:")
            out += [f"  - {n}: {where(at, l)}、{int((now - t) / 60000)} 分前" for n, at, t, now in reversed(looked[-MAX_LOOKED_LINES:])]
    if last:
        out.append(f"- 最後に利用者へ話したこと: {last}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- 取り込みと再生

def adb(*args):
    return subprocess.run(["adb", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")


def jsonl(path):
    out = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:   # 途中で止まった記録の最後の行
                pass
    return out


def set_status(**kw):
    write_json(RECORDINGS / "status.json", {"t": time.time(), **kw})


def make_preview(d):
    """ARCore MP4 のカメラトラックを、元の約 30 fps のまま縦向きで見られる動画にする。"""
    if (d / "preview30.mp4").exists() or not (d / "session.mp4").exists():
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg が見つかりません。動画プレビューを作れません")
    tmp = d / "preview30.tmp.mp4"
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(d / "session.mp4"),
           "-map", "0:v:0", "-vf", "transpose=clock,scale=720:-2", "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-y", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"動画プレビューを作れません: {r.stderr[-300:]}")
    tmp.replace(d / "preview30.mp4")


def pull():
    """端末にあって PC にない記録を取り込む。全部取れたものだけ .pulled を付ける"""
    r = adb("shell", "ls", REMOTE)
    names = [n for n in r.stdout.split() if NAME.match(n)]
    if r.returncode and not names:
        set_status(state="error", job="pull", msg="端末から一覧を取れません（接続を確認）")
        return
    todo = [n for n in names if not (RECORDINGS / n / ".pulled").exists()]
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    for k, n in enumerate(todo):
        set_status(state="running", job="pull", msg=f"{n} を取り込み中", done=k, total=len(todo))
        p = adb("pull", f"{REMOTE}/{n}", str(RECORDINGS))
        if p.returncode:
            set_status(state="error", job="pull", msg=f"{n}: {p.stderr[-200:]}")
            return
        (RECORDINGS / n / ".pulled").write_text("ok")
    for k, n in enumerate(names):
        d = RECORDINGS / n
        if (d / ".pulled").exists() and not (d / "preview30.mp4").exists():
            set_status(state="running", job="preview", msg=f"{n} の 30 fps 動画を作成中", done=k, total=len(names))
            make_preview(d)
    set_status(state="done", job="pull", msg=f"{len(todo)} 件を取り込みました（端末には {len(names)} 件）")


def recognizer(detector):
    """画像（RGB の配列）→ [{box, score, name, genre, fine, sim}]。アプリと同じく、名前を付けるのは大きい順に MAX_NAMED 個"""
    from infer import ProductRecognizer
    rec = ProductRecognizer()
    if detector == "sku110k-2":
        from ultralytics import YOLO
        model = YOLO(str(WORK / "sku26n" / "weights" / "sku110k-2.pt"))

        def detect(a):
            r = model.predict(a[..., ::-1].copy(), imgsz=640, conf=0.25, max_det=1000, device="cpu", verbose=False)[0]
            return r.boxes.xyxyn.cpu().numpy().reshape(-1, 4), r.boxes.conf.cpu().numpy()
        rec.detect = detect

    def run(a):
        dets, _ = rec(a)
        dets.sort(key=lambda d: -area(d["box"]))
        for k, d in enumerate(dets):
            if k >= MAX_NAMED:
                d["name"] = d["genre"] = d["fine"] = None
        return dets
    return run


def load_grid(path, size):
    import numpy as np
    if size is None or not path.exists():
        return None
    return size[0], size[1], np.fromfile(path, dtype="<u2").tolist()


def plane_state(state, updates, upto, pos):
    while pos < len(updates) and updates[pos]["t"] <= upto:
        u = updates[pos]
        state[u["id"]] = u
        pos += 1
    return pos


def world_of(m, x, z):
    return (m[0] * x + m[8] * z + m[12], m[1] * x + m[9] * z + m[13], m[2] * x + m[10] * z + m[14])


def light_model(view):
    out = []
    cs, sn = cos(view["yaw"]), sin(view["yaw"])
    for f in view["fixtures"]:
        lo, hi = f["c"] - f["d"] / 2, f["c"] + f["d"] / 2
        r = (lo, f["a"], hi, f["b"]) if f["axis"] == 0 else (f["a"], lo, f["b"], hi)
        corners = [[round(u * cs - v * sn, 3), round(u * sn + v * cs, 3)] for u, v in ((r[0], r[1]), (r[2], r[1]), (r[2], r[3]), (r[0], r[3]))]
        label = [f"[{t}]" for ts in f.get("themes", {}).values() for t in ts if t][:2] + [m[0][0] for ms in f["faces"].values() for m in ms if m][:3]
        out.append({"kind": f["kind"], "c": corners, "label": " ".join(dict.fromkeys(label))})
    return out


def replay(name, detector="app", hz=1.0, recognize=None):
    """記録を 1 回再生する。recognize はテスト用の差し替え"""
    from PIL import Image
    import numpy as np
    d = RECORDINGS / name
    keyframes = sorted(jsonl(d / "keyframes.jsonl"), key=lambda k: k["t"])
    if not keyframes:
        raise ValueError("keyframes.jsonl が空です")
    updates = sorted(jsonl(d / "planes.jsonl"), key=lambda p: p["t"])
    events = sorted(jsonl(d / "events.jsonl"), key=lambda e: e["t"])
    recognize = recognize or recognizer(detector)
    t0, step_ns = keyframes[0]["t"], 1e9 / hz
    chosen, nxt = [], t0
    for k in keyframes:
        if k["t"] >= nxt:
            chosen.append(k)
            nxt = k["t"] + step_ns * 0.95
    model, shelf, planes, ppos = StoreModel(), ShelfMap(), {}, 0
    observed = []   # (カメラ x, z, 点 x, z)。ガラス越しの点を戻して図に出す
    view, last_est, steps, last_spoken = None, None, [], ""
    stats = {"frames": 0, "tracked": 0, "boxes": 0, "named": 0, "with_depth": 0, "placed": 0, "ms": 0.0}
    for si, k in enumerate(chosen):
        g = k["geom"]
        set_status(state="running", job="replay", msg=f"{name} ({detector})", done=si, total=len(chosen))
        ppos = plane_state(planes, updates, k["t"], ppos)
        stats["frames"] += 1
        st = {"i": si, "n": k["n"], "t": round((k["t"] - t0) / 1e9, 2), "image": k["image"], "state": k["state"], "has_depth": bool(k.get("depth"))}
        cam = (g["m"][12], g["m"][13], g["m"][14])
        st["cam"] = [round(v, 3) for v in cam]
        # 画像認識は位置追跡に依存しない。追跡が止まった記録でも検出器の実験はできる。
        t1 = time.perf_counter()
        dets = recognize(np.asarray(Image.open(d / k["image"]).convert("RGB")))
        stats["ms"] += (time.perf_counter() - t1) * 1000
        if k["state"] != "TRACKING":
            st["dets"] = [[*[round(float(v), 4) for v in det["box"]], round(float(det["score"]), 3), det["name"], round(float(det["sim"]), 3), None, False] for det in dets]
            stats["boxes"] += len(dets)
            stats["named"] += sum(bool(det["name"]) for det in dets)
            steps.append(st)
            continue
        stats["tracked"] += 1
        flat = hypot(g["m"][8], g["m"][10])
        l = {"p": cam, "fx": -g["m"][8] / flat, "fz": -g["m"][10] / flat} if flat > 0.3 else None
        st["fwd"] = [round(l["fx"], 3), round(l["fz"], 3)] if l else None
        grid = load_grid(d / "depth" / f"{k['n']:05d}.d16", k.get("depth"))
        walls, counters = [], []
        for p in planes.values():
            if p["state"] != "TRACKING" or p["subsumed"]:
                continue
            m = p["m"]
            if p["type"] == "VERTICAL":
                walls.append({"c": (m[12], m[13], m[14]), "n": (m[4], m[5], m[6]), "r": max(p["ex"], p["ez"]) / 2})
            elif p["type"] == "HORIZONTAL_UPWARD_FACING" and -0.6 <= m[13] - cam[1] <= -0.1 and p["ex"] * p["ez"] >= 0.1:
                x, z = p["ex"] / 2, p["ez"] / 2
                counters += [world_of(m, a, b) for a, b in ((0, 0), (x, z), (-x, z), (x, -z), (-x, -z))]
        now_ms = (k["t"] - t0) / 1e6
        model.add_camera(cam[0], cam[2])
        out = []
        for det in dets:
            b, depth, pos = det["box"], median_depth(det["box"], g, grid), None
            stats["boxes"] += 1
            stats["with_depth"] += depth is not None
            cu, cv = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            if det["name"]:
                stats["named"] += 1
                pos = place(cam, world_at(cu, cv, 1.0, g), world_at(cu, cv, depth, g) if depth else None, walls)
                if pos:
                    w, h = size_in_meters(b[2] - b[0], b[3] - b[1], dist3(pos, cam), g)
                    shelf.add(det["name"], det["genre"], det["fine"], pos, w, h, now_ms)
                    model.add_product(cam[0], cam[2], pos[0], pos[2], det["name"], det["genre"] or "")
                    observed.append((cam[0], cam[2], pos[0], pos[2]))
                    stats["placed"] += 1
            elif depth is not None and depth <= RANGE_M:
                p = world_at(cu, cv, depth, g)
                model.add_product(cam[0], cam[2], p[0], p[2], "", "")
                observed.append((cam[0], cam[2], p[0], p[2]))
            out.append([*[round(float(v), 4) for v in b], round(float(det["score"]), 3), det["name"], round(float(det["sim"]), 3), None if depth is None else round(depth, 2), pos is not None])
        for c in counters:
            model.add_counter(c[0], c[2])
        if last_est is None or now_ms - last_est >= ESTIMATE_S * 1000:
            last_est = now_ms
            view = model.estimate()
        st["dets"] = out
        st["items"] = len(shelf.items)
        st["focus"] = focus([d_[:4] for d_ in out])[0]
        if l:
            ans = tap_answer(view, shelf.items, l, cam[1] - DEVICE_HEIGHT_M)
            st["tap"] = ans
            last_spoken = ans["text"]
            st["context"] = context(view, l, shelf.items, [], last_spoken)
        if view and si % SNAP_EVERY == 0:
            st["model"] = light_model(view)
        steps.append(st)
    if view is None:
        view = model.estimate()
    sec = (chosen[-1]["t"] - t0) / 1e9
    tracked = stats["tracked"]
    summary = {"seconds": round(sec, 1), "steps": len(steps), "tracked_ratio": round(tracked / max(stats["frames"], 1), 3),
               "boxes_per_step": round(stats["boxes"] / max(stats["frames"], 1), 1), "named_per_step": round(stats["named"] / max(stats["frames"], 1), 1),
               "depth_ratio": round(stats["with_depth"] / max(stats["boxes"], 1), 3), "placed_per_step": round(stats["placed"] / max(tracked, 1), 1),
               "ms_per_step": round(stats["ms"] / max(stats["frames"], 1)), "items": len(shelf.items), "fixtures": len(view["fixtures"]),
               "planes": len(planes), "keyframes": len(keyframes), "poses": len(jsonl(d / "poses.jsonl"))}
    timed = [{**e, "at": round((e["t"] - t0) / 1e9, 2)} for e in events]
    meta = read_json(d / "meta.json", {})
    audio_offset = max(0.0, (keyframes[0].get("wall_ms", 0) - meta.get("audio_start_ms", 0)) / 1000) if meta.get("audio_start_ms") and keyframes[0].get("wall_ms") else 0.0
    video_offset = max(0.0, (keyframes[0].get("wall_ms", 0) - meta.get("start_ms", 0)) / 1000) if meta.get("start_ms") and keyframes[0].get("wall_ms") else 0.0
    result = {"name": name, "detector": detector, "hz": hz, "meta": meta, "has_audio": (d / "audio.pcm").exists(), "audio_offset": audio_offset,
              "has_preview": (d / "preview30.mp4").exists(), "video_offset": video_offset,
              "summary": summary, "steps": steps, "events": timed,
              "text": store_text(view), "final_model": light_model(view),
              "glass_points": [[round(x, 3), round(z, 3), round(r[0], 3), round(r[1], 3)] for cx, cz, x, z in observed if (r := glass_restored(view, cx, cz, x, z))]}
    write_json(d / f"replay_{detector}.json", result)
    set_status(state="done", job="replay", msg=f"{name} ({detector}) を再生しました", done=len(chosen), total=len(chosen))
    return result


def recordings():
    """取り込み済みの記録の一覧"""
    out = []
    if RECORDINGS.exists():
        for p in sorted(RECORDINGS.iterdir(), reverse=True):
            if p.is_dir() and NAME.match(p.name):
                mp4 = p / "session.mp4"
                out.append({"name": p.name, "keyframes": len(list((p / "frames").glob("*.jpg"))), "mp4_mb": round(mp4.stat().st_size / 1e6) if mp4.exists() else 0,
                            "audio": (p / "audio.pcm").exists(), "app_audio": (p / "app_audio.pcm").exists(),
                            "preview": (p / "preview30.mp4").exists(),
                            "pulled": (p / ".pulled").exists(), "replays": [d for d in DETECTORS if (p / f"replay_{d}.json").exists()]})
    return out


def prompt_for(question, ctx):
    """Narrator.questionPrompt（アプリの文面をそのまま使う）"""
    src = (SRC / "Narrator.kt").read_text(encoding="utf-8")
    rules = re.search(r'SPEAKING_RULES = """(.*?)"""', src, re.S).group(1)
    body = re.search(r'fun questionPrompt\(.*?= """(.*?)"""\.trim\(\)', src, re.S).group(1)
    return body.replace("$context", ctx).replace("$question", question).replace("$SPEAKING_RULES", rules).strip()


def ask(name, detector, step, question):
    """ある瞬間の文脈と画像で、長押しの質問を Gemini に聞く（ダッシュボードのボタンを押したときだけ）"""
    import base64
    import requests
    from s3_compare import GEMINI, gemini_key
    d = RECORDINGS / name
    steps = read_json(d / f"replay_{detector}.json")["steps"]
    if not 0 <= step < len(steps):
        raise ValueError("bad step")
    s = steps[step]
    ctx = s.get("context") or "- 位置はまだ分かりません。\n"
    prompt = prompt_for(question, ctx)
    body = {"contents": [{"parts": [{"text": prompt}, {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode((d / s["image"]).read_bytes()).decode()}}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}}
    r = requests.post(GEMINI.format("gemini-3.5-flash-lite"), headers={"x-goog-api-key": gemini_key(), "Content-Type": "application/json"}, json=body, timeout=60)
    if r.status_code != 200:
        raise ValueError(f"Gemini HTTP {r.status_code}")
    text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    try:
        text = json.loads(text)["text"]
    except (json.JSONDecodeError, KeyError):
        pass
    return {"answer": text, "context": ctx}


def fixture():
    """タップの答えと AI の文脈が Python とアプリで同じかを確かめるデータ（StoreGuideTest）"""
    import random
    rng = random.Random(3)
    m, items, lines = StoreModel(), [], []

    def obs(cx, cz, x, z, cat, genre):
        m.add_product(cx, cz, x, z, cat, genre)
        lines.append(f"P\t{cx}\t{cz}\t{x}\t{z}\t{cat}\t{genre}")

    for _ in range(500):
        t = rng.uniform(0, 3.6)
        obs(0.0, round(t, 4), round(-1.2 + rng.gauss(0, 0.03), 4), round(t, 4), "カップラーメン", "カップ麺")
        t = rng.uniform(0, 3.6)
        obs(0.0, round(t, 4), round(1.2 + rng.gauss(0, 0.03), 4), round(t, 4), "お茶", "ソフトドリンク")
    for k in range(13):
        m.add_camera(0.0, k * 0.3)
        lines.append(f"T\t0.0\t{k * 0.3}")
    shelf = [("カップラーメン", "カップ麺", "カップ麺", -1.2, 1.5, 2.0), ("カップ焼きそば", "カップ麺", "カップ麺", -1.2, 1.0, 1.9), ("お茶", "ソフトドリンク", "緑茶", 1.2, 1.5, 2.0),
             ("麦茶", "ソフトドリンク", "麦茶", 1.2, 1.0, 1.8), ("コーラ", "ソフトドリンク", "コーラ", 1.2, 0.4, 2.1), ("チョコ", "チョコ・ビスケット", "チョコ", 1.2, 1.1, 2.2)]
    for n, g, f, x, y, z in shelf:
        items.append(Item(n, g, f, (x, y, z), 0.1, 0.2, 3, 0))
        lines.append(f"I\t{n}\t{g}\t{f}\t{x}\t{y}\t{z}\t3")
    view = m.estimate()
    lines.append("===")
    for p, fx, fz in (((0.0, 1.3, 2.0), 0.0, -1.0), ((0.5, 1.3, 2.0), 1.0, 0.0), ((-0.4, 1.3, 2.0), -1.0, 0.0), ((0.0, 1.3, 9.0), 0.0, -1.0)):
        l = {"p": p, "fx": fx, "fz": fz}
        ans = tap_answer(view, items, l, 0.0)
        ctx = context(view, l, items, [], "").replace("\n", "|")
        lines.append(f"L\t{p[0]}\t{p[1]}\t{p[2]}\t{fx}\t{fz}\t{ans['kind']}\t{ans['text']}\t{ctx}")
    (ROOT / "app/src/test/resources/store_guide_trace.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(x for x in lines if x.startswith("L")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true")
    ap.add_argument("--previews", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--detector", choices=DETECTORS, default="app")
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--fixture", action="store_true")
    a = ap.parse_args()
    try:
        if a.fixture:
            fixture()
        elif a.pull:
            pull()
        elif a.previews:
            for d in sorted(RECORDINGS.iterdir()):
                if d.is_dir() and NAME.match(d.name):
                    make_preview(d)
        elif a.run:
            if not NAME.match(a.run):
                raise ValueError("bad name")
            print(json.dumps(replay(a.run, a.detector, a.hz)["summary"], ensure_ascii=False))
    except Exception as e:
        set_status(state="error", job="replay" if a.run else "pull", msg=f"{type(e).__name__}: {e}")
        raise
    sys.exit(0)
