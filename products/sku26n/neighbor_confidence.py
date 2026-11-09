"""MobileCLIP の top1 のカテゴリはそのままで、信頼度だけ周りの枠との一致度で調整する。

  .venv/Scripts/python products/sku26n/neighbor_confidence.py
重みは学習に使っていない検証写真で求める（学習写真では分類の確率が高く出すぎるため）。
評価は写真を半分に分けて交差させる。出力は pred_clip_gem_conf.json と neighbor_confidence.json。
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from neighbor_fix import OUT, RADIUS, TOP5, geometry

K = 4


def features(top5, xy, long_side, ref, k=K):
    """各枠の [top1 の logit, 周りがあるか, 周りの top1 が同じ割合, 周りでのそのカテゴリの平均確率]"""
    out = np.zeros((len(top5), 4))
    if not len(top5):
        return out
    scores = [{c: p for c, p in t} for t in top5]
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    for i, top in enumerate(top5):
        cat, p = top[0][0], float(np.clip(top[0][1], 1e-4, 1 - 1e-4))
        near = [j for j in np.argsort(dist[i]) if j != i and ref[j] and dist[i, j] <= RADIUS * long_side[i]][:k]
        out[i] = np.log(p / (1 - p)), bool(near), np.mean([top5[j][0][0] == cat for j in near]) if near else 0, np.mean([scores[j].get(cat, 0) for j in near]) if near else 0
    return out


def self_check():
    pair = lambda a, b, p: [[a, p], [b, 1 - p], ["x", 0], ["y", 0], ["z", 0]]
    tops = [pair("A", "B", .5), pair("A", "B", .8), pair("B", "A", .6)]
    x = features(tops, np.array([[0, 0], [1, 0], [2, 0]], float), np.ones(3) * 2, [True] * 3)
    assert np.allclose(x[0], [0, 1, .5, .6]), x[0]  # 周りは A(.8) と B（A が .4）
    x = features(tops[:1], np.zeros((1, 2)), np.ones(1), [True])
    assert np.allclose(x[0], [0, 0, 0, 0]), "周りが無くても動く"


def main():
    self_check()
    top5 = json.loads(TOP5.read_text(encoding="utf-8"))
    det = {x["name"]: x for x in json.loads((OUT / "det.json").read_text(encoding="utf-8"))["images"]}
    labels = {}
    for line in (OUT / "label_all.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["status"] == "ok":
            labels[r["image"], r["box"]] = r["cat"]
    val = sorted(json.loads((OUT / "bc_manifest.json").read_text(encoding="utf-8"))["validation_images"])
    feats = {n: features(top5[n], *geometry(d)) for n, d in det.items()}

    rows = [(n, i) for n in val for i in range(len(top5[n])) if (n, i) in labels]
    X = np.array([feats[n][i] for n, i in rows])
    y = np.array([top5[n][i][0][0] == labels[n, i] for n, i in rows])
    half = np.array([val.index(n) % 2 == 0 for n, _ in rows])
    fit = lambda a, b: LogisticRegression(max_iter=2000).fit(a, b)
    crossed = np.zeros(len(y))
    for train, test in ((half, ~half), (~half, half)):
        crossed[test] = fit(X[train], y[train]).predict_proba(X[test])[:, 1]

    raw = 1 / (1 + np.exp(-X[:, 0]))
    print(f"boxes {len(y)} AUROC raw {roc_auc_score(y, raw):.4f} adjusted {roc_auc_score(y, crossed):.4f}", flush=True)
    for cov in (.3, .5, .7, .85):
        count = int(cov * len(y))
        print(f"keep {cov:.0%}: raw {y[np.argsort(-raw)[:count]].mean():.3f} adjusted {y[np.argsort(-crossed)[:count]].mean():.3f}", flush=True)
    for threshold in (.5, .7, .8, .9):
        kept = crossed >= threshold
        print(f"adjusted >= {threshold}: keep {kept.mean():.3f} precision {y[kept].mean():.3f}", flush=True)

    model = fit(X, y)
    (OUT / "neighbor_confidence.json").write_text(json.dumps({"features": ["logit_top1", "has_neighbours", "neighbour_top1_share", "neighbour_mean_prob"], "coef": model.coef_[0].round(4).tolist(),
                                                              "intercept": round(float(model.intercept_[0]), 4), "k": K, "radius": RADIUS}, indent=1), encoding="utf-8")
    print("coef", model.coef_[0].round(3).tolist(), "intercept", round(float(model.intercept_[0]), 3), flush=True)
    output = {}
    for name, d in det.items():
        adjusted = model.predict_proba(feats[name])[:, 1] if len(feats[name]) else []
        output[name] = [b[:5] + [top5[name][i][0][0], round(float(adjusted[i]), 4)] for i, b in enumerate(d["boxes"])]
    (OUT / "pred_clip_gem_conf.json").write_text(json.dumps({"images": output}, separators=(",", ":")), encoding="utf-8")
    print("Saved pred_clip_gem_conf.json", sum(map(len, output.values())), flush=True)


if __name__ == "__main__":
    main()
