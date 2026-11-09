"""26n-sku の切り出しと、カテゴリ認識 3 手法の結果を見比べるダッシュボード。

  python products/sku26n/dashboard.py [--port 8771]   -> http://127.0.0.1:8771
各手法の結果は products/work/sku26n/pred_<手法>.json: {"images": {画像名: [[x0, y0, x1, y1, 信頼度, カテゴリ番号], ...]}}（座標は 0..1）
"""
import argparse, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
VIEW = HERE.parent / "work" / "view"
WORK = HERE.parent / "work" / "sku26n"
TAXONOMY = HERE.parent / "konbinidb" / "taxonomy.json"
METHODS = ["m1"]


def state(det: Path, pred_dir: Path) -> bytes:
    d = json.loads(det.read_text(encoding="utf-8")) if det.exists() else \
        {"weights": None, "images": [{"name": f.name, "boxes": []} for f in sorted(VIEW.glob("*.jpg"))]}
    d["methods"] = {m: json.loads((pred_dir / f"pred_{m}.json").read_text(encoding="utf-8"))["images"]
                    for m in METHODS if (pred_dir / f"pred_{m}.json").exists()}
    d["cats"] = {k: v[0] for g in json.loads(TAXONOMY.read_text(encoding="utf-8")).values() for k, v in g.items()}
    return json.dumps(d, ensure_ascii=False).encode()


def handler(det: Path, pred_dir: Path):
    names = {f.name for f in VIEW.glob("*.jpg")}

    class H(BaseHTTPRequestHandler):
        def send(self, body: bytes, ctype: str, cache: bool = False):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if cache:
                self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self.send((HERE / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/state":
                self.send(state(det, pred_dir), "application/json")
            elif path.startswith("/view/") and path[6:] in names:  # 一覧にある名前だけ返す
                self.send((VIEW / path[6:]).read_bytes(), "image/jpeg", cache=True)
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass

    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8771)
    ap.add_argument("--det", type=Path, default=WORK / "det.json")
    ap.add_argument("--pred-dir", type=Path, default=WORK)
    a = ap.parse_args()
    print(f"http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(a.det, a.pred_dir)).serve_forever()
