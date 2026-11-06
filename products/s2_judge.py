"""② 中盤: 軽量ローカルモデル（MobileCLIP-S0 のテキスト側）で「商品を特定できない切り抜き」をゼロショット判定する。
画像側の埋め込みは s2_crops.py で計算済みのものを使うので、ほぼ一瞬で終わる。

  .venv/Scripts/python products/s2_judge.py
出力: work/judge.json {切り抜きID: {"ok": 合格か, "top": 一番近い説明, "p_bad": 特定不能側の確率}}
"""
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from common import WORK, log, read_json, run_stage, status, write_json

REPO = "Xenova/mobileclip_s0"  # アプリの画像エンコーダと同じリポジトリのテキスト側
GOOD = ["a photo of a product package with a label", "a bottle of a drink", "a can of a drink", "a bag of snacks",
        "a box of a product", "a cosmetic product", "a packaged food", "a product hanging on a hook", "a pack of sanitary products",
        "a carton of milk", "a tube of toothpaste", "a pack of socks"]
BAD = ["a price tag", "an empty store shelf", "a shelf edge", "a store sign", "an advertising poster", "a floor",
       "a ceiling", "a person", "a refrigerator door", "rows of many different products", "a wall"]  # ぼけはルール側で判定（小さい切り抜きを落としすぎるため入れない）
MAX_BAD = 0.7  # 特定不能側の確率がこれを超えたら落とす（迷うものは残す）
SCALE = 100.0  # CLIP の logit scale


def text_embeds(texts):
    tok = AutoTokenizer.from_pretrained(REPO)
    s = ort.InferenceSession(hf_hub_download(REPO, "onnx/text_model.onnx"), providers=["CPUExecutionProvider"])
    ids = tok(texts, padding="max_length", max_length=77, truncation=True, return_tensors="np")["input_ids"].astype(np.int64)
    e = s.run(None, {"input_ids": ids})[0]
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def main():
    ids = read_json(WORK / "emb_ids.json", [])
    E = np.load(WORK / "emb_all.npy")
    T = text_embeds(GOOD + BAD)
    logits = SCALE * E @ T.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    p_bad = p[:, len(GOOD):].sum(1)
    top = p.argmax(1)
    labels = GOOD + BAD
    out = {cid: {"ok": bool(pb <= MAX_BAD), "top": labels[t], "p_bad": round(float(pb), 3)}
           for cid, pb, t in zip(ids, p_bad, top)}
    write_json(WORK / "judge.json", out)
    status("judge", done=len(ids), total=len(ids))
    tops = {}
    for v in out.values():
        if not v["ok"]:
            tops[v["top"]] = tops.get(v["top"], 0) + 1
    log("judge", f"{len(ids)} 個中 {sum(not v['ok'] for v in out.values())} 個を特定不能と判定", max_bad=MAX_BAD,
        dropped_by=dict(sorted(tops.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    run_stage("judge", main)
