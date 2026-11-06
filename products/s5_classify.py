"""③b 商品分類の評価（最近傍）。命名済みの切り抜きを学習用（=照合先）とテスト用に分け、
テスト用の各切り抜きに「一番似ている学習用の切り抜きの名前」を付けて正解率を測る。

  .venv/Scripts/python products/s5_classify.py
・分け方はグループ単位（5グループに1つをテスト）。グループは見た目の近さで作ったので、写真単位で分けると
  同じグループのメンバーが両側に入り、正解率が甘く出る
・正解ラベル = 社名＋商品名（Gemini か手入力。表記ゆれを減らすため NFKC・空白除去で比べる）
・埋め込みは MobileCLIP-S0（アプリと同じ）と DINOv2-small（グループ化に使ったもの）を比べる
出力: work/classify.json
"""
import time, unicodedata, zlib

import numpy as np

from common import WORK, category_levels, crop_labels, log, read_json, run_stage, status, write_json

RERANK_K = 10  # 再順位付けの候補数
THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s).replace(" ", "").replace("　", "").lower()


def is_test(cid: str, groups_of: dict) -> bool:
    """テスト用か（グループ単位で 5 つに 1 つ）。埋め込みの学習 (s4b_embed.py) もこれで学習用だけを使う"""
    return zlib.crc32(str(groups_of.get(cid, cid)).encode()) % 5 == 0


def dino(ids):
    """DINOv2 の埋め込み（画像ごとのキャッシュから）。無いものは None"""
    out, cache = {}, {}
    for cid in ids:
        stem = cid.rsplit("_", 1)[0]
        if stem not in cache:
            p = WORK / "dino" / f"{stem}.npz"
            cache[stem] = np.load(p) if p.exists() else None
        z = cache[stem]
        out[cid] = z[cid] if z is not None and cid in z.files else None
    return out


def evaluate(name, S, ms, labels, is_test, ids, shown):
    """S = テスト×学習 の類似度。高い順に見て、名前ごとの上位5件を出す"""
    tr, te = np.where(~is_test)[0], np.where(is_test)[0]
    order = np.argsort(-S, axis=1)[:, :50]
    train_labels = {labels[i] for i in tr}
    rows = []
    for qi, q in enumerate(te):
        top, seen = [], set()
        for j in order[qi]:
            lab = labels[tr[j]]
            if lab not in seen:
                seen.add(lab)
                top.append((lab, float(S[qi, j]), ids[tr[j]]))
            if len(top) == 5:
                break
        rows.append({"q": ids[q], "true": labels[q], "known": labels[q] in train_labels, "top": top})
    known = [r for r in rows if r["known"]]
    unknown = [r for r in rows if not r["known"]]
    top1 = sum(r["top"][0][0] == r["true"] for r in known) / max(len(known), 1)
    top5 = sum(any(t[0] == r["true"] for t in r["top"]) for r in known) / max(len(known), 1)
    sweep = []
    for t in THRESHOLDS:
        ans = [r for r in known if r["top"][0][1] >= t]
        sweep.append({"t": t, "answered": round(len(ans) / max(len(known), 1), 3),
                      "acc": round(sum(r["top"][0][0] == r["true"] for r in ans) / max(len(ans), 1), 3),
                      "unknown_rejected": round(sum(r["top"][0][1] < t for r in unknown) / max(len(unknown), 1), 3)})
    # 実例: 当たり・外れを半分ずつ（毎回同じものが出るよう ID で並べる）
    rng = np.random.RandomState(0)
    ok = [r for r in known if r["top"][0][0] == r["true"]]
    ng = [r for r in known if r["top"][0][0] != r["true"]]
    ex = [rng.choice(len(x), min(len(x), 24), replace=False) for x in (ok, ng)]
    examples = [{**ok[i], "ok": True} for i in ex[0]] + [{**ng[i], "ok": False} for i in ex[1]]
    log("classify", f"{name}: Top-1 {top1:.3f} / Top-5 {top5:.3f}（既知 {len(known)} 枚, 未知 {len(unknown)} 枚, 照合先 {len(tr)} 枚）")
    return {"name": name, "top1": round(top1, 3), "top5": round(top5, 3), "known": len(known), "unknown": len(unknown),
            "gallery": len(tr), "ms_per_query": round(ms, 3), "sweep": sweep, "examples": examples, "shown": shown}


# 周辺カテゴリ補正の強さ（粗いジャンル λ, 細分類 λ）の組み合わせ
# 周辺補正の強さ（粗いジャンル, 細分類, カテゴリそのもの）の組み合わせ
COMBOS = [(g, f, c) for g in (0.0, 0.05, 0.1, 0.2, 0.4) for f in (0.0, 0.05, 0.1, 0.2, 0.4) for c in (0.0, 0.05, 0.1, 0.2)]


def mnv3():
    """学習した MobileNetV3 の埋め込み {切り抜きID: ベクトル}（s4b_embed.py の出力。無ければ None）"""
    p = WORK / "emb_mnv3.npz"
    if not p.exists():
        return None
    z = np.load(p)
    d = dict(zip(z["ids"].tolist(), z["E"]))
    d["_dim"] = z["E"].shape[1]
    return d


def clip_all():
    ids = read_json(WORK / "emb_ids.json", [])
    return ids, np.load(WORK / "emb_all.npy")


def mnv3_all(mn):
    ids = [k for k in mn if k != "_dim"]
    return ids, np.stack([mn[k] for k in ids])


def with_context(name, combos, base, all_ids, A, ids, labels, tr, te, stem, genre_of, fine_of, cat_of):
    """テスト用の切り抜きがある写真ごとに、写真内の合格（緑）の全切り抜きの候補を出し（1回だけ）、
    周辺カテゴリ（粗いジャンル・細分類）の補正の強さの組み合わせごとに選び直して正解率を測る"""
    from context import adjust
    crops = {c["id"]: c for c in read_json(WORK / "crops.json", [])}
    judge = read_json(WORK / "judge.json", {})
    pos = {c: k for k, c in enumerate(all_ids)}
    G = A[[pos[ids[j]] for j in tr]]
    G = G / np.linalg.norm(G, axis=1, keepdims=True)
    glab = [labels[j] for j in tr]
    gstem = stem[tr]
    by_img = {}
    for c in all_ids:
        if judge.get(c, {}).get("ok") and c in crops:
            by_img.setdefault(crops[c]["stem"], []).append(c)
    truth = {ids[q]: labels[q] for q in te}
    train_labels = set(glab)
    stat = {c: [0, 0, 0] for c in combos}  # 正解, 直った, 悪化
    known = 0
    for st in {stem[q] for q in te}:
        members = by_img.get(st, [])
        Q = A[[pos[c] for c in members]]
        S = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ G.T
        S[:, gstem == st] = -2.0
        cands = []
        for row in S:
            top, seen = [], set()
            for j in np.argsort(-row)[:60]:
                if glab[j] not in seen:
                    seen.add(glab[j])
                    top.append((glab[j], float(row[j])))
                if len(top) == 10:
                    break
            cands.append(top)
        centers = np.array([[(crops[c]["box"][0] + crops[c]["box"][2]) / 2, (crops[c]["box"][1] + crops[c]["box"][3]) / 2] for c in members])
        eval_idx = [k for k, c in enumerate(members) if c in truth and truth[c] in train_labels]
        known += len(eval_idx)
        for combo in combos:
            new = adjust(centers, cands, [(genre_of, combo[0]), (fine_of, combo[1]), (cat_of, combo[2])])
            for k in eval_idx:
                t = truth[members[k]]
                before, after = cands[k][0][0] == t, new[k][0] == t
                stat[combo][0] += after
                stat[combo][1] += after and not before
                stat[combo][2] += before and not after
    out = []
    for (lg, lf, lc), (correct, fixed, broken) in stat.items():
        top1 = correct / max(known, 1)
        nm = f"{name} ＋周辺補正 ジャンルλ={lg} 細分類λ={lf} カテゴリλ={lc}"
        log("classify", f"{nm}: Top-1 {top1:.3f}（直った {fixed} / 悪化 {broken}）")
        out.append({"name": nm, "top1": round(top1, 3), "top5": None, "known": known, "unknown": base["unknown"],
                    "gallery": base["gallery"], "ms_per_query": base["ms_per_query"], "fixed": fixed, "broken": broken,
                    "lambda_genre": lg, "lambda_fine": lf, "lambda_cat": lc, "sweep": [], "examples": [], "shown": {}})
    return out


def main():
    groups = read_json(WORK / "groups.json", {}).get("of", {})
    named = crop_labels()  # 正解 = 商品カテゴリ（②d）
    disp = {}
    # カテゴリごとの枚数（MobileNet 等の分類器を学習できるだけあるかの目安）
    per = {}
    for lab in named.values():
        per[lab] = per.get(lab, 0) + 1
    counts = {f">={k}": sum(c >= k for c in per.values()) for k in (1, 2, 5, 10, 20, 50)}
    top_products = sorted(per.items(), key=lambda x: -x[1])[:30]
    results = []
    clip_ids = read_json(WORK / "emb_ids.json", [])
    clip = np.load(WORK / "emb_all.npy")
    pos = {cid: k for k, cid in enumerate(clip_ids)}
    D = dino(list(named))
    ids = [cid for cid in named if cid in pos and D.get(cid) is not None]
    labels = [named[c] for c in ids]
    # グループ単位でテスト用を選ぶ（グループに入っていない切り抜きは自分の ID で）
    test = np.array([is_test(c, groups) for c in ids])
    tr, te = np.where(~test)[0], np.where(test)[0]
    unit = lambda E: E / np.linalg.norm(E, axis=1, keepdims=True)
    Ec = unit(np.stack([clip[pos[c]] for c in ids]).astype(np.float32))
    Ed = unit(np.stack([D[c] for c in ids]).astype(np.float32))

    stem = np.array([c.rsplit("_", 1)[0] for c in ids])
    same = stem[te][:, None] == stem[tr][None, :]  # 新しい写真で使う想定なので、同じ写真に写った照合先は使わない

    def sims(E):
        t0 = time.perf_counter()
        S = E[te] @ E[tr].T
        ms = (time.perf_counter() - t0) * 1000 / max(len(te), 1)
        S[same] = -2.0
        return S, ms
    Sc, msc = sims(Ec)
    Sd, msd = sims(Ed)
    # MobileCLIP で候補 RERANK_K 件 → その中を DINOv2 の類似度で並べ直す（候補外は選ばれない）
    cand = np.argpartition(-Sc, RERANK_K - 1, axis=1)[:, :RERANK_K]
    Sr = np.full_like(Sd, -2.0)
    np.put_along_axis(Sr, cand, np.take_along_axis(Sd, cand, 1), 1)
    shown = {lab: disp.get(lab, lab) for lab in set(labels)}
    ways = [("MobileCLIP-S0（アプリと同じ）", Sc, msc), ("DINOv2-small", Sd, msd),
            (f"MobileCLIP 上位{RERANK_K}件 → DINOv2 で再順位", Sr, msc + msd * RERANK_K / max(len(tr), 1)),
            ("MobileCLIP と DINOv2 の平均", (Sc + Sd) / 2, msc + msd)]
    mn = mnv3()
    if mn is not None:  # 学習した MobileNetV3（③b）。無い切り抜きは類似度 0
        Em = unit(np.stack([mn.get(c, np.zeros(mn["_dim"], np.float32)) for c in ids]) + 1e-9)
        Sm, msm = sims(Em)
        ways.append(("MobileNetV3-small（学習, 128px）", Sm, msm))
    status("classify", phase="最近傍で評価", done=0, total=len(ways))
    for k, (nm, S, ms) in enumerate(ways):
        results.append(evaluate(nm, S, ms, labels, test, ids, shown))
        status("classify", done=k + 1, total=len(ways))
    # 周辺カテゴリ補正（同じ写真の周りの商品の予想カテゴリで、候補を選び直す）。推論で使う埋め込みに対して比べる
    genre_of, fine_of = category_levels()
    if genre_of or fine_of:
        nm, (all_ids, A) = ("MobileNetV3-small（学習, 128px）", mnv3_all(mn)) if mn else ("MobileCLIP-S0（アプリと同じ）", clip_all())
        base = next(r for r in results if r["name"] == nm)
        results += with_context(nm, COMBOS, base, all_ids, A, ids, labels, tr, te, stem, genre_of, fine_of, {c: c for c in set(labels)})
    write_json(WORK / "classify.json", {
        "method": "最近傍（コサイン類似度）", "split": "グループ単位で 1/5 をテスト・同じ写真の照合先は使わない",
        "genres": len(set(genre_of.values())), "genre_labeled": len(genre_of), "fines": len(set(fine_of.values())),
        "labeled": len(ids), "products": len(per), "per_product": counts,
        "top_products": [[disp.get(k, k), v] for k, v in top_products], "results": results})


if __name__ == "__main__":
    run_stage("classify", main)
