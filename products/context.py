"""周りの商品のカテゴリで判定を補正する（同じカテゴリの商品は棚で固まっている）。③b の評価と ④ の推論で共通。
粗いジャンル（パン / サンドイッチ・惣菜 …）と細分類（緑茶 / コーヒー …）の2段で、周りと違う候補にペナルティをかける。

1. 各商品の候補（上位 K 件の名前と類似度）から、カテゴリの分布を作る（類似度の softmax をカテゴリごとに足す）
2. 周りの NB 個の分布を、近いほど重く平均する。棚は横に並ぶので縦の距離は ROW 倍に数える（同じ段の隣を重視）
3. 候補のスコア = 類似度 + Σ λ·log(周りでのそのカテゴリの確率)。カテゴリ不明の候補は補正しない
全部 NumPy の行列計算（1画像 100〜300 個で数 ms）。
"""
import numpy as np

NB = 8        # 周りとみなす近い商品の数
ROW = 3.0     # 縦方向の距離の重み（別の段の商品は遠い扱い）
TEMP = 20.0   # 類似度 → 確率の鋭さ
EPS = 0.02    # 周りに一度も出ないカテゴリでも確率 0 にしない（完全には禁止しない）


def neighbors(centers: np.ndarray) -> np.ndarray:
    """(n,n) の重み。各行は近い NB 個に 1/(距離+0.02) を割り振り、合計 1"""
    n = len(centers)
    W = np.zeros((n, n))
    if n < 2:
        return W
    d = centers[:, None] - centers[None]
    D = np.sqrt(d[..., 0] ** 2 + (ROW * d[..., 1]) ** 2)
    np.fill_diagonal(D, np.inf)
    k = min(NB, n - 1)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    W[rows, idx] = 1 / (D[rows, idx] + 0.02)
    return W / W.sum(1, keepdims=True)


def adjust(centers: np.ndarray, cands: list, levels: list):
    """centers: (n,2) 画像内の位置（0..1）。cands[i] = [(名前, 類似度), ...]（高い順）。
    levels = [(名前→カテゴリの dict, λ), ...]。戻り値: 各商品の (選んだ名前, 補正後スコア, 元の類似度)"""
    n = len(cands)
    K = max((len(c) for c in cands), default=0)
    if n == 0 or K == 0 or not any(lam and m for m, lam in levels):
        return [(c[0][0], c[0][1], c[0][1]) if c else (None, 0.0, 0.0) for c in cands]
    labs = [[x[0] for x in c] + [None] * (K - len(c)) for c in cands]
    sims = np.full((n, K), -np.inf)
    for i, c in enumerate(cands):
        sims[i, :len(c)] = [x[1] for x in c]
    ok = np.isfinite(sims)
    w = np.where(ok, np.exp(TEMP * (np.where(ok, sims, 0) - np.max(np.where(ok, sims, -1e9), 1, keepdims=True))), 0)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    Wn = neighbors(centers)
    score = sims.copy()
    for cat_of, lam in levels:
        if not lam or not cat_of:
            continue
        cats = {}
        L = np.array([[cats.setdefault(g, len(cats)) if (g := cat_of.get(l)) is not None else -1 for l in row] for row in labs])
        if not cats:
            continue
        has = L >= 0
        P = np.zeros((n, len(cats)))
        np.add.at(P, (np.nonzero(has)[0], L[has]), w[has])
        Q = Wn @ P
        Q = (Q + EPS) / (Q.sum(1, keepdims=True) + EPS * len(cats))
        score += lam * np.where(has, np.log(Q[np.arange(n)[:, None], np.maximum(L, 0)]), 0)
    best = np.argmax(score, 1)
    return [(labs[i][b], float(score[i, b]), float(sims[i, b])) if ok[i, b] else (None, 0.0, 0.0) for i, b in enumerate(best)]


if __name__ == "__main__":  # 自己チェック
    centers = np.array([[0.1, 0.1], [0.2, 0.1], [0.3, 0.1], [0.4, 0.1], [0.25, 0.11], [0.25, 0.6]])
    # 1) パンに囲まれた中で「ウェットティッシュ」が少しだけ似ていても、パンを選ぶ（粗いジャンル）
    gen = {"bread_a": "パン", "bread_b": "パン", "wipe": "日用品・衛生用品", "tea": "ソフトドリンク", "coffee": "ソフトドリンク"}
    cands = [[("bread_a", 0.9)], [("bread_b", 0.9)], [("bread_a", 0.88)], [("bread_b", 0.9)],
             [("wipe", 0.80), ("bread_a", 0.78)], [("tea", 0.9)]]
    assert adjust(centers, cands, [(gen, 0.0)])[4][0] == "wipe"
    assert adjust(centers, cands, [(gen, 0.05)])[4][0] == "bread_a"
    # 2) 茶の段の中でコーヒーが少しだけ似ていても、細分類で茶を選ぶ（粗いジャンルは同じソフトドリンクなので効かない）
    fine = {"tea": "緑茶", "tea2": "緑茶", "coffee": "コーヒー"}
    cands = [[("tea", 0.9)], [("tea2", 0.9)], [("tea", 0.9)], [("tea2", 0.9)], [("coffee", 0.80), ("tea", 0.78)], [("coffee", 0.9)]]
    assert adjust(centers, cands, [(gen, 0.05)])[4][0] == "coffee"
    assert adjust(centers, cands, [(gen, 0.05), (fine, 0.05)])[4][0] == "tea"
    # 3) 別の段（縦に離れた）コーヒーは、隣の段の判定にほとんど影響しない
    assert adjust(centers, cands, [(fine, 0.05)])[5][0] == "coffee"
    # 4) グミの段に紛れた「飲むヨーグルト」: 少しだけ似ていても、カテゴリそのものの段でグミを選ぶ
    cat = {k: k for k in ("グミ", "飲むヨーグルト")}
    cands = [[("グミ", 0.9)]] * 4 + [[("飲むヨーグルト", 0.80), ("グミ", 0.77)], [("飲むヨーグルト", 0.9)]]
    assert adjust(centers, cands, [])[4][0] == "飲むヨーグルト"
    assert adjust(centers, cands, [(cat, 0.1)])[4][0] == "グミ"
    print("ok: ジャンル・細分類・カテゴリのペナルティと、同じ段の隣の重視")
