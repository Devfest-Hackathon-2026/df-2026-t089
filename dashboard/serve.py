"""Local dashboard for the shelf training data.

  python dashboard/serve.py [--data ../df-2026-t089/training_images] [--port 8765]
  -> http://127.0.0.1:8765
"""
import argparse, csv, datetime, io, json, shutil, threading, time
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


REVIEW = "labels_review.csv"  # who-checked-what log next to labels.csv; train.py ignores it
REVIEW_FIELDS = ["file", "status", "original", "label", "at"]
LOCK = threading.Lock()
_backed_up = False


def write_csv(path: Path, fields, rows):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:  # same format as the original: UTF-8, CRLF
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def save_label(data: Path, file: str, label):
    """label=None -> exclude the image (row removed from labels.csv). Returns the review row."""
    global _backed_up
    with LOCK:
        lp = data / "labels.csv"
        if not _backed_up and lp.exists():  # one backup per server run, before the first edit
            shutil.copy2(lp, data / f"labels.csv.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}")
            _backed_up = True
        rows = read_csv(lp)
        cur = {r["file"]: r["shelf"] for r in rows}
        review = {r["file"]: r for r in read_csv(data / REVIEW)}
        original = review[file]["original"] if file in review else cur.get(file, "")
        if label is None:
            rows = [r for r in rows if r["file"] != file]
        elif file in cur:
            for r in rows:
                if r["file"] == file:
                    r["shelf"] = label
        else:
            rows.append({"file": file, "shelf": label})
        write_csv(lp, ["file", "shelf"], rows)
        status = "excluded" if label is None else "ok" if label == original else "fixed"
        review[file] = {"file": file, "status": status, "original": original, "label": label or "",
                        "at": datetime.datetime.now().isoformat(timespec="seconds")}
        write_csv(data / REVIEW, REVIEW_FIELDS, review.values())
        return review[file]


def build_data(data: Path):
    files = sorted(p.name for p in data.iterdir() if p.suffix.lower() in IMG_EXT)
    labels = {r["file"]: r["shelf"] for r in read_csv(data / "labels.csv")}
    review = {r["file"]: r for r in read_csv(data / REVIEW)}
    cuts = {r["file"]: [float(c) for c in (r.get("cuts") or "").split()] for r in read_csv(data / "boundaries.csv")}
    model_labels = ROOT / "app/src/main/assets/labels.txt"
    return {
        "dataDir": str(data),
        "images": [{"file": f, "label": labels.get(f), "cuts": cuts.get(f), "review": review.get(f)} for f in files],
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

    def do_POST(self):
        # JSON-only: a cross-site form cannot send this content type without a CORS preflight, which we never answer
        if self.path != "/api/label" or self.headers.get("Content-Type") != "application/json":
            return self.send(b"bad request", "text/plain", 400)
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 10000)))
            file, label = body["file"], body.get("label")
            if file not in self.files:
                raise ValueError("unknown file")
            if label is not None:
                label = str(label).strip()
                if not label or len(label) > 40 or any(c in label for c in "\r\n,\""):
                    raise ValueError("invalid label")
        except (ValueError, KeyError, TypeError) as e:
            return self.send(json.dumps({"error": str(e)}).encode(), "application/json", 400)
        r = save_label(self.data, file, label)
        self.send(json.dumps(r, ensure_ascii=False).encode(), "application/json; charset=utf-8")

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
