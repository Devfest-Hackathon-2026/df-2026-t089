"""④ 新しい店内写真から商品を全て検出し、商品名を付ける（最終成果物。Android でも同じ ONNX 2つを使う想定）。

  1. YOLO（detector.json が指すモデル）で「商品らしいもの」の枠を全部出す
  2. 各枠を切り抜いて埋め込み（③b で学習した MobileNetV3-small 128px。無ければ MobileCLIP）
  3. 命名済みの切り抜き（照合先）と照合して名前の候補を出し、周りの商品のジャンルで選び直す（context.py）

  .venv/Scripts/python products/infer.py [画像...]   # 省略時は test_images と学習に使っていない val 画像で評価
"""
import argparse, csv, os, time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from common import TEST_DIR, TRAIN_DIR, WORK, config, load_image, log, read_json, run_stage, status, write_json
from context import adjust
from s5_classify import norm

WORK_SIZE = 1280   # 処理用に縮める長辺（検出 640px・切り抜き 128px にはこれで十分）
MATCH_SIM = 0.5     # 補正前の類似度がこれ未満なら「照合先に無い商品」として名前を付けない
LAMBDA = (0.05, 0.05)  # 周辺カテゴリ補正の強さ（粗いジャンル, 細分類）。③b の評価で一番良かった組み合わせ


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = os.cpu_count() or 4
    return ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])


class ProductRecognizer:
    def __init__(self, conf=0.25, iou=0.5, lam=LAMBDA):
        dj = read_json(WORK / "detector.json")
        self.det = session(WORK / dj.get("onnx", "detector.onnx"))
        self.det_in = self.det.get_inputs()[0].name
        self.imgsz, self.det_name = dj["imgsz"], dj.get("run", "")
        self.conf, self.iou, self.lam = conf, iou, lam
        cfg = config()
        self.min_w, self.min_h = cfg["min_crop_w"], cfg["min_crop_h"]
        names = read_json(WORK / "names.json", {})
        good = {c: v for c, v in names.items() if (v.get("maker") or v.get("product")) and v.get("vision") != "bad"}
        if (WORK / "embed_mnv3.onnx").exists() and (WORK / "emb_mnv3.npz").exists():
            from s4b_embed import square
            self.emb_name, self.size = "MobileNetV3-small 128px（学習）", 128
            self.emb = session(WORK / "embed_mnv3.onnx")
            self.prep = lambda a: square(a, 128)
            self.run_emb = lambda x: self.emb.run(None, {"pixels": x})[0]
            z = np.load(WORK / "emb_mnv3.npz")
            ids, E = z["ids"].tolist(), z["E"]
        else:  # 学習前: アプリと同じ MobileCLIP（256px, 引き伸ばし）
            self.emb_name, self.size = "MobileCLIP-S0 256px", 256
            from s2_crops import Embedder
            e = Embedder()
            self.prep = lambda a: cv2.resize(a, (256, 256), interpolation=cv2.INTER_LINEAR)
            self.run_emb = lambda x: (lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))(e.s.run(None, {e.inp: x})[0])
            ids, E = read_json(WORK / "emb_ids.json", []), np.load(WORK / "emb_all.npy")
        keep = [k for k, c in enumerate(ids) if c in good]
        glab = [norm(good[ids[k]].get("maker", "") + good[ids[k]].get("product", "")) for k in keep]
        # 照合先を商品ごとに並べ、「商品ごとの最大類似度」を reduceat で一度に出せるようにする
        order = np.argsort(glab, kind="stable")
        G = E[keep][order]
        self.gallery = (G / np.linalg.norm(G, axis=1, keepdims=True)).astype(np.float32)
        self.glab = [glab[i] for i in order]
        products, self.starts = np.unique(self.glab, return_index=True)
        self.products = products.tolist()  # NumPy の文字列のままだと辞書の検索が遅い
        self.show = {norm(v.get("maker", "") + v.get("product", "")): f'{v.get("maker", "")} {v.get("product", "")}'.strip() for v in good.values()}
        self.genre_of = read_json(WORK / "genres.json", {})
        self.fine_of = read_json(WORK / "fine.json", {})

    def detect(self, a: np.ndarray):
        """レターボックスで正方形に → YOLO → 信頼度とNMSで絞る。枠は 0..1 の [x0,y0,x1,y1]"""
        h, w = a.shape[:2]
        s = self.imgsz / max(w, h)
        nw, nh = round(w * s), round(h * s)
        x = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        x[:nh, :nw] = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
        out = self.det.run(None, {self.det_in: (x.transpose(2, 0, 1)[None] / 255).astype(np.float32)})[0][0]
        cx, cy, bw, bh, sc = out[:5]
        m = sc >= self.conf
        b = np.stack([cx[m] - bw[m] / 2, cy[m] - bh[m] / 2, bw[m], bh[m]], 1)
        k = np.array(cv2.dnn.NMSBoxes(b.tolist(), sc[m].tolist(), self.conf, self.iou)).reshape(-1)
        b, sc = b[k], sc[m][k]
        return np.stack([b[:, 0] / nw, b[:, 1] / nh, (b[:, 0] + b[:, 2]) / nw, (b[:, 1] + b[:, 3]) / nh], 1).clip(0, 1), sc

    def __call__(self, im):
        a = np.asarray(im)  # RGB（カメラのフレームを想定。デコード時間は含めない）
        t0 = time.perf_counter()
        # 最初に長辺 WORK_SIZE へ1回だけ縮め、検出も切り抜きもそこから作る（元の 1200 万画素から切ると重い）
        k = WORK_SIZE / max(a.shape[:2])
        if k < 1:
            a = cv2.resize(a, (round(a.shape[1] * k), round(a.shape[0] * k)), interpolation=cv2.INTER_AREA)
        h, w = a.shape[:2]
        boxes, scores = self.detect(a)
        t1 = time.perf_counter()
        px = (boxes * [w, h, w, h]).astype(int)
        mw, mh = self.min_w * min(k, 1), self.min_h * min(k, 1)  # 最小サイズは元画像の画素で決めてあるので縮小に合わせる
        big = [i for i, (x0, y0, x1, y1) in enumerate(px) if x1 - x0 >= mw and y1 - y0 >= mh]
        sq = [self.prep(a[px[i, 1]:px[i, 3], px[i, 0]:px[i, 2]]) for i in big]
        t2 = time.perf_counter()
        E = self.run_emb((np.stack(sq).astype(np.float32) / 255).transpose(0, 3, 1, 2)) if sq else np.zeros((0, 1))
        t3 = time.perf_counter()
        cands = [[] for _ in boxes]
        if sq and len(self.gallery):
            P = np.maximum.reduceat(self.gallery @ E.T, self.starts, axis=0).T  # 商品ごとの最大類似度（縦方向の reduceat は速い）
            kk = min(10, P.shape[1])
            top = np.argpartition(-P, kk - 1, axis=1)[:, :kk]
            for r, i in enumerate(big):
                t = top[r][np.argsort(-P[r, top[r]])]
                cands[i] = [(self.products[j], float(P[r, j])) for j in t]
        centers = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1) if len(boxes) else np.zeros((0, 2))
        picked = adjust(centers, cands, [(self.genre_of, self.lam[0]), (self.fine_of, self.lam[1])])
        t4 = time.perf_counter()
        dets = []
        for i, b in enumerate(boxes):
            lab, score, sim = picked[i]
            ok = lab is not None and sim >= MATCH_SIM
            dets.append({"box": [round(float(v), 4) for v in b], "score": round(float(scores[i]), 3),
                         "name": self.show.get(lab, lab) if ok else None, "genre": self.genre_of.get(lab) if ok else None,
                         "fine": self.fine_of.get(lab) if ok else None,
                         "sim": round(sim, 3), "changed": bool(ok and cands[i] and cands[i][0][0] != lab)})
        ms = {"detect": round((t1 - t0) * 1000), "crop": round((t2 - t1) * 1000), "embed": round((t3 - t2) * 1000),
              "match": round((t4 - t3) * 1000), "total": round((t4 - t0) * 1000)}
        return dets, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    a = ap.parse_args()
    rec = ProductRecognizer()
    log("infer", f"検出 {rec.det_name}（{rec.imgsz}px）· 埋め込み {rec.emb_name} · 照合先 {len(rec.glab)} 枚 / {len(set(rec.glab))} 商品 · 周辺カテゴリ補正 λ(ジャンル, 細分類)={rec.lam}")
    if a.images:
        targets = [("x", Path(p)) for p in a.images]
    else:
        stems = set(read_json(WORK / "yolo" / "split.json", {"val": []})["val"])
        targets = [("test", p) for p in sorted(TEST_DIR.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        targets += [("val", p) for p in sorted(TRAIN_DIR.iterdir()) if p.stem in stems]
    for sub in ("infer", "view_infer"):
        (WORK / sub).mkdir(exist_ok=True)
    for f in (WORK / "infer").glob("*.json"):
        f.unlink()
    for _ in range(2):  # 最初の数回は初期化で遅いので計測から外す
        rec(load_image(targets[0][1]))
    times = []
    for n, (kind, p) in enumerate(targets):
        im = load_image(p)
        dets, ms = rec(im)
        times.append(ms)
        stem = f"{kind}_{p.stem}"
        v = im.copy()
        v.thumbnail((1280, 1280))
        v.save(WORK / "view_infer" / f"{stem}.jpg", quality=85)
        write_json(WORK / "infer" / f"{stem}.json", {"stem": stem, "image": f"{kind}/{p.name}", "view": f"view_infer/{stem}.jpg",
                                                     "dets": dets, "ms": ms})
        status("infer", done=n + 1, total=len(targets))
    T = {k: [t[k] for t in times] for k in times[0]}
    nd = [len(read_json(f)["dets"]) for f in (WORK / "infer").glob("*.json")]
    ev = {"images": len(times), "ms_total_mean": round(float(np.mean(T["total"]))), "ms_total_p90": round(float(np.percentile(T["total"], 90))),
          **{f"ms_{k}_mean": round(float(np.mean(v))) for k, v in T.items() if k != "total"},
          "dets_mean": round(float(np.mean(nd)), 1), "detector": f"{rec.det_name} ({rec.imgsz}px)", "embedder": rec.emb_name,
          "gallery_crops": len(rec.glab), "gallery_products": len(set(rec.glab)), "context_lambda": rec.lam,
          "cpu": f"このPCの CPU（{os.cpu_count()} スレッド）, onnxruntime"}
    res = WORK / "train" / read_json(WORK / "detector.json")["run"] / "results.csv"
    if res.exists():
        last = list(csv.DictReader(open(res, encoding="utf-8")))[-1]
        ev.update({k.strip().replace("metrics/", "val_"): round(float(v), 3) for k, v in last.items() if "metrics/" in k and v})
    write_json(WORK / "eval.json", ev)
    log("infer", "評価", **ev)


if __name__ == "__main__":
    run_stage("infer", main)
