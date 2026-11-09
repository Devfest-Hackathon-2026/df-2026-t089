"""店モデルのシミュレーション（store_sim.py）で使う認識誤差を、実際のモデルの結果から作る。

  .venv/Scripts/python products/store_calib.py   -> work/store_calib.json

③b と同じ分け方のテスト用の切り抜き（学習に使っていない）に、照合先の最近傍でカテゴリを付けた結果を、
グループ（≒同じ商品）ごとに残す。シミュレーションの 1 商品 = 実際の 1 グループなので、
「この商品はいつも読み違える」という偏りがそのまま再現される。
shortcut: 周辺カテゴリ補正（+0.04 程度）は入れていない。写真の中の並びが要るため。入れるなら s5_classify.with_context を流用する
"""
import numpy as np

from common import WORK, category_levels, crop_labels, read_json, write_json
from infer import MATCH_SIM
from s5_classify import is_test


def main():
    groups = read_json(WORK / "groups.json", {}).get("of", {})
    labels = crop_labels()
    z = np.load(WORK / "emb_mnv3.npz")
    all_ids = z["ids"].tolist()  # npz は読むたびに展開するので 1 回だけ
    keep = [k for k, c in enumerate(all_ids) if c in labels]
    ids = [all_ids[k] for k in keep]
    E = z["E"][keep].astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    test = np.array([is_test(c, groups) for c in ids])
    tr, te = np.where(~test)[0], np.where(test)[0]
    stem = np.unique([c.rsplit("_", 1)[0] for c in ids], return_inverse=True)[1]
    S = E[te] @ E[tr].T
    S[stem[te][:, None] == stem[tr][None, :]] = -2.0  # 新しい写真で使う想定なので、同じ写真の照合先は使わない
    best, sim = S.argmax(1), S.max(1)
    out = {}
    for q, b, s in zip(te, best, sim):
        pred = labels[ids[tr[b]]] if s >= MATCH_SIM else ""  # "" = 名前を付けない（アプリの match_sim と同じ）
        out.setdefault(labels[ids[q]], {}).setdefault(str(groups.get(ids[q], ids[q])), []).append(pred)
    preds = [p for g in out.values() for ps in g.values() for p in ps]
    right = sum(p == c for c, g in out.items() for ps in g.values() for p in ps)
    named = sum(1 for p in preds if p)
    genre = category_levels()[0]
    write_json(WORK / "store_calib.json", {"match_sim": MATCH_SIM, "crops": len(preds), "named": round(named / len(preds), 3),
                                           "acc_named": round(right / max(named, 1), 3), "genre": genre, "cats": out})
    print(f"テスト {len(preds)} 枚 · 名前を付ける {named / len(preds):.1%} · そのうち正解 {right / max(named, 1):.1%} · カテゴリ {len(out)}")


if __name__ == "__main__":
    main()
