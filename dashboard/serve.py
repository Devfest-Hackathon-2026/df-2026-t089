"""Local dashboard for the shelf training data.

  python dashboard/serve.py [--data ../df-2026-t089/training_images] [--port 8765]
  -> http://127.0.0.1:8765
"""
import argparse, csv, datetime, io, json, time
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
IMG_EXT = {".jpg", ".jpeg", ".png"}

try:
    from PIL import Image, ImageOps
except ImportError:  # thumbnails fall back to the originals
    Image = None


def read_csv(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def build_data(data: Path):
    files = sorted(p.name for p in data.iterdir() if p.suffix.lower() in IMG_EXT)
    labels = {r["file"]: r["shelf"] for r in read_csv(data / "labels.csv")}
    cuts = {r["file"]: [float(c) for c in (r.get("cuts") or "").split()] for r in read_csv(data / "boundaries.csv")}
    model_labels = ROOT / "app/src/main/assets/labels.txt"
    return {
        "dataDir": str(data),
        "images": [{"file": f, "label": labels.get(f), "cuts": cuts.get(f)} for f in files],
        "missingFiles": sorted((set(labels) | set(cuts)) - set(files)),
        "modelClasses": [l for l in model_labels.read_text(encoding="utf-8").splitlines() if l] if model_labels.exists() else [],
    }


def build_results():
    ev = ROOT / "tools/results/eval.json"
    tests, at = [], None
    junit = ROOT / "app/build/test-results/testDebugUnitTest"
    for x in sorted(junit.glob("*.xml")) if junit.is_dir() else []:
        at = max(at or 0, x.stat().st_mtime)
        for tc in ET.parse(x).getroot().iter("testcase"):
            bad = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            tests.append({"class": tc.get("classname", "").split(".")[-1], "name": tc.get("name"), "time": float(tc.get("time") or 0),
                          "status": "skipped" if tc.find("skipped") is not None else "failed" if bad is not None else "passed",
                          "message": bad.get("message") if bad is not None else None})
    return {
        "eval": json.loads(ev.read_text(encoding="utf-8")) if ev.exists() else None,
        "unit": {"tests": tests, "at": datetime.datetime.fromtimestamp(at).isoformat(timespec="seconds") if at else None},
    }


class Handler(BaseHTTPRequestHandler):
    data: Path
    files: set
    cache = HERE / ".cache"

    def send(self, body: bytes, ctype: str, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(self.path.split("?")[0])
        if path in ("/", "/index.html"):
            return self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/data":
            d = build_data(self.data)
            Handler.files = {i["file"] for i in d["images"]}
            return self.send(json.dumps(d, ensure_ascii=False).encode(), "application/json; charset=utf-8")
        if path == "/api/progress":
            pf = ROOT / "tools/.cache/progress.json"
            body = pf.read_text(encoding="utf-8") if pf.exists() else "null"
            return self.send(f'{{"now": {time.time()}, "progress": {body}}}'.encode(), "application/json; charset=utf-8")
        if path == "/api/results":
            return self.send(json.dumps(build_results(), ensure_ascii=False).encode(), "application/json; charset=utf-8")
        for prefix in ("/img/", "/thumb/"):
            if path.startswith(prefix):
                name = path[len(prefix):]
                if name not in self.files:  # only names listed from the data dir: no path traversal
                    return self.send(b"not found", "text/plain", 404)
                if prefix == "/thumb/" and Image:
                    return self.send(self.thumb(name), "image/jpeg")
                return self.send((self.data / name).read_bytes(), "image/jpeg")
        self.send(b"not found", "text/plain", 404)

    def thumb(self, name):
        out = self.cache / (name + ".thumb.jpg")
        if not out.exists():
            self.cache.mkdir(exist_ok=True)
            im = ImageOps.exif_transpose(Image.open(self.data / name)).convert("RGB")
            im.thumbnail((360, 480))
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=80)
            out.write_bytes(buf.getvalue())
        return out.read_bytes()

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT.parent / "df-2026-t089/training_images")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if not a.data.is_dir():
        raise SystemExit(f"data dir not found: {a.data}  (use --data)")
    Handler.data = a.data.resolve()
    Handler.files = {i["file"] for i in build_data(Handler.data)["images"]}
    print(f"serving {Handler.data}\n  -> http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
