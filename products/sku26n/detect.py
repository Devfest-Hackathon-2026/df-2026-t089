"""店内写真から商品の枠を 26n-sku（YOLO26n を SKU-110K で学習したモデル）で切り出す。

  .venv/Scripts/python products/sku26n/detect.py [--imgsz 640] [--limit 10]
重みが無ければ Ultralytics Platform から取る（.env か環境変数の ULTRALYTICS_API_KEY が必要）。
出力: products/work/sku26n/det.json（git に入らない）。信頼度の足切りはダッシュボード側でするので、ここでは低めに全部残す。
"""
import argparse, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIEW = ROOT / "products" / "work" / "view"
OUT_DIR = ROOT / "products" / "work" / "sku26n"
URI = "ul://fatih/yolo26n-sku110k/sku110k-2"
WEIGHTS = OUT_DIR / "weights" / "sku110k-2.pt"


def api_key_from_dotenv():
    """ULTRALYTICS_API_KEY だけを .env から読む（値は表示しない）"""
    env = ROOT / ".env"
    if os.getenv("ULTRALYTICS_API_KEY") or not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "ULTRALYTICS_API_KEY" and v.strip():
            os.environ["ULTRALYTICS_API_KEY"] = v.strip().strip("\"'")


def fetch_weights():
    from ultralytics.utils.checks import resolve_platform_uri
    from ultralytics.utils.downloads import safe_download
    api_key_from_dotenv()
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    safe_download(resolve_platform_uri(URI), file=WEIGHTS, progress=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.05, help="保存する枠の下限。表示の足切りはダッシュボードのスライダーで")
    ap.add_argument("--limit", type=int, default=0, help="全体から等間隔に N 枚だけ（お試し用）")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "det.json")
    a = ap.parse_args()
    if a.weights == WEIGHTS and not WEIGHTS.exists():
        fetch_weights()

    from ultralytics import YOLO
    model = YOLO(str(a.weights))
    files = sorted(VIEW.glob("*.jpg"))
    if a.limit:
        files = [files[i * len(files) // a.limit] for i in range(min(a.limit, len(files)))]  # 同じ売り場に偏らないよう等間隔に
    images, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        r = model.predict(str(f), imgsz=a.imgsz, conf=a.conf, max_det=1000, device="cpu", verbose=False)[0]
        h, w = r.orig_shape
        boxes = [[round(v, 4) for v in b] + [round(c, 3)] for b, c in zip(r.boxes.xyxyn.tolist(), r.boxes.conf.tolist())]
        images.append({"name": f.name, "w": w, "h": h, "boxes": boxes})
        if i % 50 == 0 or i == len(files):
            print(f"{i}/{len(files)} ({time.time() - t0:.0f}s)", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"weights": a.weights.name, "imgsz": a.imgsz, "conf_min": a.conf, "images": images}), encoding="utf-8")
    print(f"{a.out}: {len(images)} 枚, 枠 {sum(len(x['boxes']) for x in images)} 個 (conf>={a.conf})")


if __name__ == "__main__":
    main()
