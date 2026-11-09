"""コンビニDB の商品画像を、店内の切り抜きに似せた学習データにする。

1. sku110k-2 で商品の枠を出し、いちばん大きい枠で切り抜く（白い背景が広いため）
2. 明るさを下げ、実際にありうる向きの Perspective Warp を 3 方向かけ、店内の切り抜きと同じ解像度まで落とす

  .venv/Scripts/python products/konbinidb/make_train.py [--limit 24]
出力: products/work/konbinidb/train_aug/<id>_<0..2>.jpg と manifest.json（git に入らない）
強さは実測に合わせている: 店内の切り抜きの明るさは 76〜153（中央 117）、コンビニDB の商品は平均 155
"""
import argparse, collections, hashlib, json, math, random
from pathlib import Path

import cv2
import numpy as np

import scrape as s

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "products" / "work" / "sku26n" / "weights" / "sku110k-2.pt"
DET = ROOT / "products" / "work" / "sku26n" / "det.json"
OUT = s.ROOT / "train_aug"
MANIFEST = OUT / "manifest.json"
PER_CATEGORY = 100
VARIANTS = 3
STORE_CONF = 0.25  # 店内の切り抜きの解像度を測るときの信頼度の下限
MIN_CONF = 0.10
DETECT_SIZE = 640
PAD_SCALE = 5
PAD_COLOR = 90
MIN_AREA = 0.05  # 画像に対する枠の面積。これより小さい枠は商品本体ではない
WHITE = 235
MIN_INK = 0.35  # 枠の中で、白くない部分の割合の下限
BRIGHTNESS = (0.5, 0.95)
YAW = (12, 35)  # 左右から見る角度（度）
PITCH = (8, 25)  # 上下から見る角度（度）
JPEG_QUALITY = 75
# 棚を左・右・上・下・斜めから見る向き（yaw の符号, pitch の符号）
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]


def targets() -> list[dict]:
    """1 カテゴリ 100 枚までの商品のうち、画像があり「No image」の代替画像でないもの"""
    items = [json.loads(x) for x in s.ITEMS.open(encoding="utf-8")]
    picked = [x for x in s.capped(items, PER_CATEGORY) if (s.IMAGES / f"{x['id']}.webp").exists()]
    digest = {x["id"]: hashlib.md5((s.IMAGES / f"{x['id']}.webp").read_bytes()).hexdigest() for x in picked}
    counts = collections.Counter(digest.values())
    return [x for x in picked if counts[digest[x["id"]]] < 5]  # 同じ画像が 5 枚以上あるものは代替画像


def store_long_sides() -> list[float]:
    """店内の切り抜きの長辺（px）の実測"""
    images = json.loads(DET.read_text(encoding="utf-8"))["images"]
    return [max((b[2] - b[0]) * im["w"], (b[3] - b[1]) * im["h"]) for im in images for b in im["boxes"] if b[4] >= STORE_CONF]


def detect(model, image: np.ndarray) -> tuple[list[float], float] | None:
    """商品の枠（元画像の 0..1）。棚向けのモデルなので、灰色の余白を足して商品を小さく写してから検出する"""
    h, w = image.shape[:2]
    scale = DETECT_SIZE / (max(h, w) * PAD_SCALE)
    small_w, small_h = max(1, round(w * scale)), max(1, round(h * scale))
    canvas = np.full((DETECT_SIZE, DETECT_SIZE, 3), PAD_COLOR, np.uint8)
    top, left = (DETECT_SIZE - small_h) // 2, (DETECT_SIZE - small_w) // 2
    canvas[top:top + small_h, left:left + small_w] = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    result = model.predict(canvas, imgsz=DETECT_SIZE, conf=MIN_CONF, max_det=30, device="cpu", verbose=False)[0]
    found = []
    for (x0, y0, x1, y1), conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        b = [max(0.0, (x0 - left) / small_w), max(0.0, (y0 - top) / small_h), min(1.0, (x1 - left) / small_w), min(1.0, (y1 - top) / small_h)]
        area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        if area < MIN_AREA:
            continue
        region = image[int(b[1] * h):max(int(b[3] * h), int(b[1] * h) + 1), int(b[0] * w):max(int(b[2] * w), int(b[0] * w) + 1)]
        if (region.min(axis=2) < WHITE).mean() >= MIN_INK:  # 白い余白だけの枠は商品ではない
            found.append((b, conf, area))
    if not found:
        return None
    largest = max(x[2] for x in found)
    b, conf, _ = max((x for x in found if x[2] >= largest * 0.4), key=lambda x: x[1])  # 大きい枠の中で確かなもの（盛り付け皿より袋を選ぶ）
    return b, conf


def warp(image: np.ndarray, yaw: float, pitch: float, fill: tuple[int, int, int]) -> np.ndarray:
    """平面の商品を yaw・pitch だけ回して見たときの画像（ピンホールカメラのホモグラフィ）"""
    h, w = image.shape[:2]
    f = 1.2 * max(w, h)
    cy, sy, cp, sp = math.cos(math.radians(yaw)), math.sin(math.radians(yaw)), math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    R = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    corners = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], float)
    moved = (R @ corners.T).T + [0, 0, f * 1.2]
    projected = moved[:, :2] / moved[:, 2:] * f
    projected -= projected.min(axis=0)
    source = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    size = np.ceil(projected.max(axis=0)).astype(int)
    M = cv2.getPerspectiveTransform(source, projected.astype(np.float32))
    return cv2.warpPerspective(image, M, (int(size[0]), int(size[1])), flags=cv2.INTER_AREA, borderValue=fill)


def variants(crop: np.ndarray, rng: random.Random, long_sides: list[float]) -> list[np.ndarray]:
    out = []
    for dx, dy in rng.sample(DIRECTIONS, VARIANTS):
        yaw = dx * rng.uniform(*YAW)
        pitch = dy * rng.uniform(*PITCH)
        fill = tuple(rng.randint(30, 110) for _ in range(3))  # 変形でできる隅は棚の暗い色で埋める
        image = warp(crop, yaw, pitch, fill)
        image = np.clip(image.astype(np.float32) * rng.uniform(*BRIGHTNESS), 0, 255).astype(np.uint8)
        scale = min(1.0, rng.choice(long_sides) / max(image.shape[:2]))
        size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
        out.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="無作為に N 件だけ（お試し用）")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    items = targets()
    if a.limit:
        items = random.Random(0).sample(items, a.limit)
    labels = {}
    for line in s.LABELS.open(encoding="utf-8"):
        x = json.loads(line)
        labels[x["id"]] = x["cat"]
    long_sides = store_long_sides()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for i, x in enumerate(items, 1):
        path = s.IMAGES / f"{x['id']}.webp"
        # 同じ商品なら毎回同じ変換になるよう商品ごとに乱数を分ける
        rng = random.Random(x["id"])
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        found = detect(model, image) if image is not None else None
        if found is None:
            skipped += 1
        else:
            box, conf = found
            H, W = image.shape[:2]
            crop = image[int(box[1] * H):max(int(box[3] * H), int(box[1] * H) + 1), int(box[0] * W):max(int(box[2] * W), int(box[0] * W) + 1)]
            for n, v in enumerate(variants(crop, rng, long_sides)):
                ok, encoded = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    raise RuntimeError(f"JPEG encoding failed: {x['id']}")
                encoded.tofile(str(OUT / f"{x['id']}_{n}.jpg"))
            rows.append({"id": x["id"], "name": x["name"], "chain": x["chain"], "cat": labels[x["id"]], "box": [round(v, 4) for v in box], "conf": round(conf, 3)})
        if i % 200 == 0 or i == len(items):
            print(f"{i}/{len(items)} 除外 {skipped}", flush=True)
    MANIFEST.write_text(json.dumps({"variants": VARIANTS, "skipped": skipped, "rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} 商品, {len(rows) * VARIANTS} 枚, 除外 {skipped}", flush=True)


if __name__ == "__main__":
    main()
