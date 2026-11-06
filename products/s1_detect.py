"""① 店内写真から商品の枠をゼロショット検出（OWLv2）。画像ごとに work/det/<名前>.json を書くので途中で止めても再開できる。

  .venv/Scripts/python products/s1_detect.py [--limit N] [--only 名前...]
"""
import argparse, re, time, zlib
from pathlib import Path

import torch
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from common import TRAIN_DIR, WORK, list_images, load_image, log, run_stage, status, write_json

MODEL = "google/owlv2-base-patch16-ensemble"
QUERIES = ["a product package", "a bottle", "a can", "a box", "a bag of snacks", "a plastic pouch", "a carton"]
KEEP = 0.05          # これ未満は保存しない（採用しきい値はダッシュボードで後から決める）
MAX_AREA = 0.15      # 画像の15%を超える枠は棚・ポスターなので捨てる
VIEW_SIZE = 1280     # ダッシュボード表示用の縮小画像の長辺


def tiles(w, h):
    """全体 + 2×2（重なり20%）。OWLv2 は 960px に縮小するので、小さい商品は分割しないと取りこぼす"""
    out = [(0, 0, w, h)]
    tw, th = int(w * 0.6), int(h * 0.6)
    for x in (0, w - tw):
        for y in (0, h - th):
            out.append((x, y, x + tw, y + th))
    return out


@torch.no_grad()
def detect(model, proc, im):
    w, h = im.size
    ts = tiles(w, h)
    crops = [im.crop(t) for t in ts]
    dev = model.device
    inp = proc(text=[QUERIES] * len(crops), images=crops, return_tensors="pt").to(dev)
    with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):  # GPU 版 (s1_gpu.py) では半精度で速く
        out = model(**inp)
    sizes = torch.tensor([[max(c.size)] * 2 for c in crops], device=dev)  # 正方形にパディングされる
    res = proc.image_processor.post_process_object_detection(out, threshold=KEEP, target_sizes=sizes)
    boxes, scores = [], []
    for (x0, y0, x1, y1), r in zip(ts, res):
        tile_full = (x1 - x0, y1 - y0) == (w, h)
        for b, s in zip(r["boxes"].tolist(), r["scores"].tolist()):
            bx0, by0, bx1, by1 = max(b[0], 0), max(b[1], 0), min(b[2], x1 - x0), min(b[3], y1 - y0)
            # タイルの内側の境界で切れた枠は、別のタイルに完全な形で写っているので捨てる
            if not tile_full and ((bx0 < 3 and x0 > 0) or (by0 < 3 and y0 > 0)
                                  or (bx1 > x1 - x0 - 3 and x1 < w) or (by1 > y1 - y0 - 3 and y1 < h)):
                continue
            boxes.append([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0])
            scores.append(s)
    if not boxes:
        return [], []
    B, S = torch.tensor(boxes), torch.tensor(scores)
    area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    ok = (area < MAX_AREA * w * h) & (B[:, 2] - B[:, 0] > 0.01 * w) & (B[:, 3] - B[:, 1] > 0.01 * h)
    B, S = B[ok], S[ok]
    k = nms(B, S, 0.5)
    B, S = B[k], S[k]
    norm = torch.tensor([w, h, w, h], dtype=torch.float32)
    return [[round(v, 4) for v in b] for b in (B / norm).tolist()], [round(s, 3) for s in S.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--src", default=str(TRAIN_DIR))
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    imgs = list_images(Path(a.src))
    if a.only:
        imgs = [p for p in imgs if p.name in a.only]
    # 1枚40秒以上かかるので、途中で止めても偏らないよう順番をばらし、似た画像が続く動画フレームは最後に回す
    imgs.sort(key=lambda p: (re.search(r"_f\d+$", p.stem) is not None, zlib.crc32(p.name.encode())))
    if a.limit:
        imgs = imgs[: a.limit]
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).eval()
    todo = [p for p in imgs if a.redo or not (WORK / "det" / f"{p.stem}.json").exists()]
    log("detect", f"{len(imgs)} 枚中 {len(todo)} 枚を処理", model=MODEL, queries=QUERIES)
    for i, p in enumerate(todo):
        t = time.time()
        im = load_image(p)
        boxes, scores = detect(model, proc, im)
        v = im.copy()
        v.thumbnail((VIEW_SIZE, VIEW_SIZE))
        (WORK / "view").mkdir(exist_ok=True)
        v.save(WORK / "view" / f"{p.stem}.jpg", quality=85)
        sec = round(time.time() - t, 1)
        write_json(WORK / "det" / f"{p.stem}.json",
                   {"image": p.name, "src": str(p.parent.name), "w": im.width, "h": im.height,
                    "boxes": boxes, "scores": scores, "sec": sec})
        status("detect", done=len(imgs) - len(todo) + i + 1, total=len(imgs), last=p.name, sec_per_image=sec)
        log("detect", f"{p.name}: {sum(s >= 0.2 for s in scores)} 個 (>=0.2) / 候補 {len(scores)} 個, {sec}s")


if __name__ == "__main__":
    run_stage("detect", main)
