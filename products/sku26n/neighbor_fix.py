"""MobileCLIP の top1 より top2/top3 の方が周りの枠と合うなら置き換える（top2 優先）。

  .venv/Scripts/python products/sku26n/neighbor_fix.py [--fetch]
周りは同じ写真の近くの枠（検出の確からしさ 0.25 以上）。周りの判定には補正前の結果を使うので、処理の順番に依存しない。
設定は検証写真の半分で選び、残りの半分で精度を測る。出力は pred_clip_gem_fit.json。
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "products" / "work" / "sku26n"
TOP5 = OUT / "top5_clip_gem.json"
RADIUS = 3.0  # 自分の枠の長辺の何倍までを「周り」とするか


def geometry(d):
    """枠の中心（ピクセル）、長辺、周りとして使える枠かどうか"""
    b = np.array(d["boxes"], float).reshape(-1, 5) * [d["w"], d["h"], d["w"], d["h"], 1]
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]), b[:, 4] >= .25


def choose(top5, xy, long_side, ref, k, mode, margin, min_p):
    """各枠で使う順位（0=top1, 1=top2, 2=top3）を返す"""
    out = [0] * len(top5)
    if not len(top5):
        return out
    scores = [{c: p for c, p in t} for t in top5]
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    for i, top in enumerate(top5):
        near = [j for j in np.argsort(dist[i]) if j != i and ref[j] and dist[i, j] <= RADIUS * long_side[i]][:k]
        fit = (lambda c: sum(top5[j][0][0] == c for j in near)) if mode == "vote" else (lambda c: sum(scores[j].get(c, 0) for j in near))
        base = fit(top[0][0])
        for rank in (1, 2):  # 先に top2 を見るので top2 が優先される
            if top[rank][1] >= min_p and fit(top[rank][0]) > base + margin:
                out[i] = rank
                break
    return out


def self_check():
    pair = lambda a, b, c: [[a, .6], [b, .3], [c, .1], ["x", 0], ["y", 0]]
    tops = [pair("A", "B", "C"), pair("B", "A", "C"), pair("B", "A", "C"), pair("C", "A", "B")]
    xy, side, ref = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float), np.ones(4) * 2, [True] * 4
    assert choose(tops, xy, side, ref, 3, "vote", 0, 0)[0] == 1, "周りに多い top2 を選ぶ"
    tops = [pair("A", "B", "C"), pair("B", "A", "C"), pair("C", "A", "B"), pair("C", "A", "B")]
    assert choose(tops, xy, side, ref, 3, "vote", 0, 0)[0] == 1, "top2 も top3 も top1 より合うなら top2"
    tops = [pair("A", "B", "C"), pair("A", "B", "C"), pair("A", "B", "C"), pair("A", "B", "C")]
    assert choose(tops, xy, side, ref, 3, "vote", 0, 0)[0] == 0, "top1 が周りと合うなら変えない"


def fetch():
    from gpu_two_models import app, clip_top5
    with app.run():
        for text in clip_top5.remote_gen((OUT / "mobileclip_s0_head_gem.pt").read_bytes()):
            TOP5.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="Modal で全部の枠の Top-5 を出し直す")
    a = ap.parse_args()
    self_check()
    if a.fetch or not TOP5.exists():
        fetch()
    top5 = json.loads(TOP5.read_text(encoding="utf-8"))
    det = {x["name"]: x for x in json.loads((OUT / "det.json").read_text(encoding="utf-8"))["images"]}
    assert all(len(top5[n]) == len(d["boxes"]) for n, d in det.items()), "Top-5 と det.json の枠の数が違う"
    labels = {}
    for line in (OUT / "label_all.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            labels[r["image"], r["box"]] = r["cat"]
    val = sorted(json.loads((OUT / "bc_manifest.json").read_text(encoding="utf-8"))["validation_images"])
    tune, test = val[::2], val[1::2]

    geo = {n: geometry(d) for n, d in det.items()}
    run = lambda name, p: choose(top5[name], *geo[name], *p)

    def report(params, photos):
        hit = base = fixed = broke = changed = n = 0
        for name in photos:
            ranks = run(name, params)
            for i, rank in enumerate(ranks):
                cat = labels.get((name, i))
                if cat is None:
                    continue
                n += 1
                before, after = top5[name][i][0][0] == cat, top5[name][i][rank][0] == cat
                base += before
                hit += after
                changed += rank != 0
                fixed += after and not before
                broke += before and not after
        return {"n": n, "before": round(base / n, 4), "after": round(hit / n, 4), "changed": changed, "wrong_to_right": fixed, "right_to_wrong": broke}

    literal = (8, "vote", 0, 0)
    print("そのままのルール 近傍8・多数決 検証全体", report(literal, val), flush=True)
    grid = list(itertools.product((4, 8, 12, 20), ("vote", "prob"), (0, 1, 2), (0, .05, .1, .2)))
    results = [(report(p, tune)["after"], p) for p in grid]
    best = max(results, key=lambda x: x[0])[1]
    print("設定選択の半分で最良", best, report(best, tune), flush=True)
    print("残りの半分（設定に未使用）", report(best, test), flush=True)
    print("残りの半分のそのままのルール", report(literal, test), flush=True)

    output = {}
    for name, d in det.items():
        ranks = run(name, best)
        output[name] = [b[:5] + top5[name][i][r] for i, (b, r) in enumerate(zip(d["boxes"], ranks))]
    (OUT / "pred_clip_gem_fit.json").write_text(json.dumps({"images": output}, separators=(",", ":")), encoding="utf-8")
    print("Saved pred_clip_gem_fit.json", best, sum(map(len, output.values())), flush=True)


if __name__ == "__main__":
    main()
