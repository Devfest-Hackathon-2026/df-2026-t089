"""② ローカルモデル（DINOv2-small）で、合格（緑）の切り抜きのうち同じ商品どうしをグループにまとめる。
DINOv2 は「同じ物か」の照合に強い自己教師ありモデル。MobileCLIP（種類の判定向き）より細かい違いを見分ける。

  .venv/Scripts/python products/s2_group.py
まとめ方: 互いに上位 K 位以内（相互近傍）で類似度 group_sim 以上のペアをつなぎ、つながったものを
完全連結法で分け直す（グループ内の全ペアが group_sim 以上）。さらに色の類似度 group_color 以上も求める（色違い・味違い対策）。しきい値を変えても埋め込みの再計算は不要（--only-group）。
出力: work/dino/<画像>.npz（埋め込みキャッシュ）, work/groups.json {"of": {切り抜きID: グループ番号}, "groups": [...]}
"""
import argparse

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from common import TRAIN_DIR, WORK, config, load_image, log, read_json, run_stage, status, write_json
from s2_crops import crop

MODEL = "facebook/dinov2-small"
SIZE = 224
K = 10
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def square(c: Image.Image) -> np.ndarray:
    """縦長の商品が切れないよう、灰色で正方形に埋めてから縮める"""
    s = max(c.size)
    sq = Image.new("RGB", (s, s), (124, 116, 104))
    sq.paste(c, ((s - c.width) // 2, (s - c.height) // 2))
    return (np.asarray(sq.resize((SIZE, SIZE), Image.BICUBIC), np.float32) / 255 - MEAN) / STD


@torch.no_grad()
def embed_all(ids_by_stem, crops):
    model = AutoModel.from_pretrained(MODEL).eval()
    (WORK / "dino").mkdir(exist_ok=True)
    out = {}
    for n, (stem, ids) in enumerate(ids_by_stem.items()):
        cache = WORK / "dino" / f"{stem}.npz"
        have = dict(np.load(cache)) if cache.exists() else {}
        miss = [i for i in ids if i not in have]
        if miss:
            im = load_image(TRAIN_DIR / crops[miss[0]]["image"])
            for k in range(0, len(miss), 64):
                part = miss[k:k + 64]
                x = torch.from_numpy(np.stack([square(crop(im, crops[i]["box"])) for i in part]).transpose(0, 3, 1, 2))
                e = model(pixel_values=x).pooler_output.numpy()
                for i, v in zip(part, e / np.linalg.norm(e, axis=1, keepdims=True)):
                    have[i] = v.astype(np.float32)
            np.savez(cache, **have)
        out.update({i: have[i] for i in ids})
        status("group", done=n + 1, total=len(ids_by_stem), last=stem)
    return out


def embed_modal(ids_by_stem, crops):
    """足りない埋め込みだけを Modal の GPU で計算してキャッシュに書く（画像ごとに送り、届いた順に保存）"""
    import io

    import modal
    from gpu_dino import Dino, app
    (WORK / "dino").mkdir(exist_ok=True)
    todo = {}
    for stem, ids in ids_by_stem.items():
        cache = WORK / "dino" / f"{stem}.npz"
        have = set(np.load(cache).files) if cache.exists() else set()
        miss = [i for i in ids if i not in have]
        if miss:
            todo[stem] = miss

    def prep(stem):
        im = load_image(TRAIN_DIR / crops[todo[stem][0]]["image"])
        im.thumbnail((1920, 1920))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=92)
        return stem, b.getvalue(), [(i, crops[i]["box"]) for i in todo[stem]]

    log("group", f"Modal GPU で {sum(map(len, todo.values()))} 個を埋め込み（{len(todo)} 画像）")
    from concurrent.futures import ThreadPoolExecutor
    stems, saved = list(todo), set()
    with modal.enable_output(), app.run(), ThreadPoolExecutor(8) as ex:
        dino = Dino()
        for k in range(0, len(stems), 40):  # 回線が切れても 40 画像ずつ保存されるように
            chunk = list(ex.map(prep, stems[k:k + 40]))
            for attempt in range(3):
                try:
                    for stem, got, raw in dino.run.starmap(chunk, order_outputs=False):
                        cache = WORK / "dino" / f"{stem}.npz"
                        have = dict(np.load(cache)) if cache.exists() else {}
                        E = np.frombuffer(raw, np.float16).reshape(len(got), -1).astype(np.float32)
                        have.update(zip(got, E))
                        np.savez(cache, **have)
                        saved.add(stem)
                        status("group", done=len(saved), total=len(stems), last=stem, note="Modal GPU")
                    break
                except Exception as e:  # 通信エラー: 未保存の画像だけもう一度
                    chunk = [c for c in chunk if c[0] not in saved]
                    log("group", f"通信エラーで再送 ({attempt + 1}/3): 残り {len(chunk)} 画像", error=repr(e)[:200])


def cached(ids_by_stem):
    out = {}
    for stem, ids in ids_by_stem.items():
        cache = WORK / "dino" / f"{stem}.npz"
        if cache.exists():
            have = np.load(cache)
            out.update({i: have[i] for i in ids if i in have.files})
    return out


def color_hist(cid: str) -> np.ndarray:
    """色の特徴: サムネイル中央80%（背景・棚を除く）の HSV ヒストグラム。
    色のある画素は色相18段階（鮮やかさで重み）、色の無い画素（白・灰・黒）は明るさ3段階。合計1に正規化"""
    im = Image.open(WORK / "crops" / f"{cid}.jpg").convert("RGB")
    w, h = im.size
    hsv = np.asarray(im.crop((int(w * .1), int(h * .1), int(w * .9), int(h * .9))).convert("HSV"), np.float32) / 255
    H, S, V = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()
    chrom = (S > 0.25) & (V > 0.2)
    hh = np.bincount((H[chrom] * 18).astype(int) % 18, weights=S[chrom], minlength=18)
    vv = np.bincount(np.minimum((V[~chrom] * 3).astype(int), 2), minlength=3) * 0.5
    v = np.concatenate([hh, vv]).astype(np.float32)
    return v / max(v.sum(), 1e-6)


def colors(ids):
    """色の特徴をキャッシュ（work/color.npz）から読み、足りない分だけ計算する"""
    from concurrent.futures import ThreadPoolExecutor
    p = WORK / "color.npz"
    have = dict(np.load(p)) if p.exists() else {}
    miss = [i for i in ids if i not in have]
    if miss:
        log("group", f"色の特徴を {len(miss)} 個計算")
        with ThreadPoolExecutor(8) as ex:
            have.update(zip(miss, ex.map(color_hist, miss)))
        np.savez(p, **have)
    return np.stack([have[i] for i in ids])


def csim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """色ヒストグラムの共通部分（1=同じ色の分布）"""
    return np.minimum(a, b).sum(-1)


def group(E: np.ndarray, sim: float, C: np.ndarray | None = None, color: float = 0.0):
    """相互 K 近傍かつ類似度 sim 以上（かつ色の類似度 color 以上）をつないだ連結成分（Union-Find）を、
    完全連結法で分け直したグループ番号。グループ内の全ペアが両方の条件を満たす"""
    use_color = C is not None and color > 0
    n = len(E)
    k = min(K, n - 1)  # 件数が K 以下でも動くように
    if k < 1:
        return list(range(n))
    nn = np.zeros((n, k), int)
    ns = np.zeros((n, k), np.float32)
    for a in range(0, n, 2048):
        S = E[a:a + 2048] @ E.T
        np.fill_diagonal(S[:, a:a + 2048], -1)
        idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
        nn[a:a + 2048], ns[a:a + 2048] = idx, np.take_along_axis(S, idx, 1)
    nbr = [set(r) for r in nn]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j, s in zip(nn[i], ns[i]):
            if s >= sim and i in nbr[j] and (not use_color or csim(C[i], C[j]) >= color):
                parent[find(i)] = find(j)
    # つないだだけだと A≈B, B≈C で全然違う A と C が同じグループになる（競合製品が混ざる原因）。
    # 各グループの中を完全連結法で分け直し、「全員どうしが sim 以上」のまとまりだけを残す
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform
    comp = {}
    for i in range(n):
        comp.setdefault(find(i), []).append(i)
    label, nxt = [0] * n, 0
    for m in comp.values():
        if len(m) == 1:
            sub = [1]
        else:
            # 距離 = 2条件のうち厳しい方をしきい値で割ったもの。完全連結で 1 以下 = 全ペアが両方を満たす
            d = pdist(E[m], "cosine") / (1 - sim)
            if use_color:
                Cm = C[m]
                cd = np.concatenate([1 - csim(Cm[i], Cm[i + 1:]) for i in range(len(m) - 1)])
                d = np.maximum(d, cd / (1 - color))
            sub = fcluster(linkage(d, "complete"), t=1.0, criterion="distance")
        for i, s in zip(m, sub):
            label[i] = nxt + int(s)
        nxt += max(sub) + 1
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modal", action="store_true", help="埋め込みを Modal の GPU で計算する")
    ap.add_argument("--only-group", action="store_true", help="埋め込みは計算せず、キャッシュ済みの分でつなぎ直すだけ（しきい値変更用）")
    a = ap.parse_args()
    cfg = config()
    crops = {c["id"]: c for c in read_json(WORK / "crops.json", [])}
    judge = read_json(WORK / "judge.json", {})
    ids = [cid for cid, j in judge.items() if j["ok"] and cid in crops]
    by_stem = {}
    for cid in ids:
        by_stem.setdefault(crops[cid]["stem"], []).append(cid)
    if a.only_group:
        emb = cached(by_stem)
        log("group", f"しきい値 {cfg['group_sim']}・色 {cfg['group_color']} でつなぎ直し（埋め込み済み {len(emb)}/{len(ids)} 個）")
    elif a.modal:
        embed_modal(by_stem, crops)
        emb = cached(by_stem)
    else:
        log("group", f"合格 {len(ids)} 個を {MODEL} で埋め込み（CPU）", images=len(by_stem))
        emb = embed_all(by_stem, crops)
    ids = list(emb)
    E = np.stack([emb[i] for i in ids])
    C = colors(ids) if cfg["group_color"] > 0 else None
    root = group(E, cfg["group_sim"], C, cfg["group_color"])
    members = {}
    for cid, r in zip(ids, root):
        members.setdefault(r, []).append(cid)
    gs = sorted((m for m in members.values() if len(m) >= 2), key=len, reverse=True)
    pos = {cid: k for k, cid in enumerate(ids)}
    groups, of = [], {}
    for g, m in enumerate(gs):
        M = E[[pos[i] for i in m]]
        rep = m[int(np.argmax(M @ M.mean(0)))]  # 代表 = グループの平均に一番近い切り抜き
        groups.append({"id": g, "size": len(m), "images": len({crops[i]["stem"] for i in m}), "rep": rep, "members": m})
        of.update({i: g for i in m})
    write_json(WORK / "groups.json", {"of": of, "groups": groups, "sim": cfg["group_sim"], "color": cfg["group_color"], "model": MODEL})
    log("group", f"{len(groups)} グループ（2枚以上）に {len(of)} 個 / 1枚だけ {len(ids) - len(of)} 個",
        sim=cfg["group_sim"], color=cfg["group_color"], largest=[g["size"] for g in groups[:10]])


if __name__ == "__main__":
    run_stage("group", main)
