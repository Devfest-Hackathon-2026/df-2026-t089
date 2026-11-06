"""③b 商品の埋め込みモデル（MobileNetV3-small, 128px）を学習し、命名済みの切り抜きを埋め込んで照合先を作る。
MobileCLIP（256px）より桁違いに軽く、1画像に100個以上の商品があっても数百 ms に収めるためのもの。

  .venv/Scripts/python products/s4b_embed.py [--epochs 30]
・学習に使うのは、③b の評価で「学習用」になるグループ側だけ（テスト側の正解率を公平に比べるため）
出力: work/embed_mnv3.onnx, work/emb_mnv3.npz（ルール通過の全切り抜きの埋め込み。命名済みのものが推論の照合先）, work/train_embed.csv
"""
import argparse, io, pickle

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from common import WORK, log, read_json, run_stage, status, write_json

SIZE = 128
PAD_COLOR = (124, 116, 104)


def square(arr: np.ndarray, size=SIZE) -> np.ndarray:
    """RGB の切り抜き → 灰色で正方形に埋めて size×size（縦長の商品の形を残す）。学習・照合先・推論で共通"""
    h, w = arr.shape[:2]
    s = max(h, w)
    out = np.empty((s, s, 3), np.uint8)
    out[:] = PAD_COLOR
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)


class MNV3:
    """学習済みの埋め込みモデル（入力 0..1 RGB の NCHW、出力は正規化済み 128 次元）"""
    def __init__(self):
        o = ort.SessionOptions()
        self.s = ort.InferenceSession(str(WORK / "embed_mnv3.onnx"), o, providers=["CPUExecutionProvider"])

    def __call__(self, squares: list[np.ndarray]) -> np.ndarray:
        x = (np.stack(squares).astype(np.float32) / 255).transpose(0, 3, 1, 2)
        return self.s.run(None, {"pixels": x})[0]


def thumb(cid):
    return np.asarray(Image.open(WORK / "crops" / f"{cid}.jpg").convert("RGB"))


def labeled():
    """命名済み（結果が悪いものを除く）の切り抜き → 正規化した名前。③b の評価と同じ規則"""
    from s5_classify import norm
    names = read_json(WORK / "names.json", {})
    return {cid: norm(v.get("maker", "") + v.get("product", "")) for cid, v in names.items()
            if (v.get("maker") or v.get("product")) and v.get("vision") != "bad" and (WORK / "crops" / f"{cid}.jpg").exists()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()
    import modal
    from gpu_embed_train import app, train, vol
    from s5_classify import is_test
    lab = labeled()
    of = read_json(WORK / "groups.json", {}).get("of", {})
    tr = [c for c in lab if not is_test(c, of)]
    cnt = {}
    for c in tr:
        cnt[lab[c]] = cnt.get(lab[c], 0) + 1
    classes = sorted(k for k, v in cnt.items() if v >= 2)  # 1枚しかない商品は「近づける相手」がいないので外す
    cid_of = {k: i for i, k in enumerate(classes)}
    tr = [c for c in tr if lab[c] in cid_of]
    log("embed", f"学習データ {len(tr)} 枚 / {len(classes)} 商品（学習用グループ側・2枚以上の商品）")
    status("embed", phase="学習データを作成・送信")
    jpgs = []
    for c in tr:
        b = io.BytesIO()
        Image.fromarray(square(thumb(c), 144)).save(b, "JPEG", quality=92)
        jpgs.append(b.getvalue())
    p = WORK / "embed_train.pkl"
    pickle.dump({"jpgs": jpgs, "labels": [cid_of[lab[c]] for c in tr], "n_classes": len(classes)}, open(p, "wb"))
    with vol.batch_upload(force=True) as up:
        up.put_file(str(p), "/train.pkl")
    csv = WORK / "train_embed.csv"
    csv.write_text("epoch,loss,train_acc\n", encoding="utf-8")
    with modal.enable_output(), app.run():
        for x in train.remote_gen(a.epochs):
            if "epoch" in x:
                with open(csv, "a", encoding="utf-8") as f:
                    f.write(f'{x["epoch"]},{x["loss"]},{x["train_acc"]}\n')
                status("embed", done=x["epoch"], total=a.epochs, phase="学習 (Modal GPU)", loss=x["loss"], train_acc=x["train_acc"])
            else:
                (WORK / "embed_mnv3.onnx").write_bytes(x["onnx"])
    # ルールを通過した全切り抜きを埋め込む（命名済みは照合先、それ以外も周辺ジャンル補正の評価に使う）
    m = MNV3()
    ids = [c for c in read_json(WORK / "emb_ids.json", []) if (WORK / "crops" / f"{c}.jpg").exists()]
    E = np.concatenate([m([square(thumb(c)) for c in ids[k:k + 256]]) for k in range(0, len(ids), 256)])
    np.savez(WORK / "emb_mnv3.npz", ids=np.array(ids), E=E.astype(np.float32))
    write_json(WORK / "embed.json", {"model": "MobileNetV3-small", "size": SIZE, "dim": int(E.shape[1]),
                                     "classes": len(classes), "train_crops": len(tr), "epochs": a.epochs})
    log("embed", f"埋め込みモデルを書き出し、{len(ids)} 枚を埋め込み")


if __name__ == "__main__":
    run_stage("embed", main)
