"""26n-sku の切り出しと、カテゴリ認識 3 手法の結果を見比べるダッシュボード。

  python products/sku26n/dashboard.py [--port 8771]   -> http://127.0.0.1:8771
各手法の結果は products/work/sku26n/pred_<手法>.json: {"images": {画像名: [[x0, y0, x1, y1, 信頼度, カテゴリ番号], ...]}}（座標は 0..1）
"""
import argparse, json, math, os, re, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "konbinidb")]
from common import crop_labels  # noqa: E402
from scrape import IMAGES as DB_IMAGES, ITEMS as DB_ITEMS, LABELS as DB_LABELS, excluded  # noqa: E402

VIEW = HERE.parent / "work" / "view"
WORK = HERE.parent / "work" / "sku26n"
CROPS = HERE.parent / "work" / "crops"
NAMES = HERE.parent / "work" / "names.json"
SELECTED = WORK / "selected_crops.json"
SELECTED_DB = WORK / "selected_db.json"
TAXONOMY = HERE.parent / "konbinidb" / "taxonomy.json"
METHODS = ["gemini", "m1"]


def state(det: Path, pred_dir: Path) -> bytes:
    d = json.loads(det.read_text(encoding="utf-8")) if det.exists() else \
        {"weights": None, "images": [{"name": f.name, "boxes": []} for f in sorted(VIEW.glob("*.jpg"))]}
    d["methods"] = {m: json.loads((pred_dir / f"pred_{m}.json").read_text(encoding="utf-8"))["images"]
                    for m in METHODS if (pred_dir / f"pred_{m}.json").exists()}
    d["cats"] = {k: v[0] for g in json.loads(TAXONOMY.read_text(encoding="utf-8")).values() for k, v in g.items()}
    d["cats"]["NP"] = "商品でない"
    return json.dumps(d, ensure_ascii=False).encode()


_lock = threading.Lock()
_crops: list | None = None
_crop_ids: set = set()


def named_crops() -> list:
    """以前のダッシュボードの、商品名付きの切り抜き: [id, 商品名, カテゴリ, 確からしさ]。カテゴリ・商品名の順"""
    global _crops, _crop_ids
    with _lock:
        if _crops is None:
            names = json.loads(NAMES.read_text(encoding="utf-8"))
            rows = [[cid, names[cid]["best"], cat, names[cid].get("confidence")] for cid, cat in crop_labels().items()]
            _crops = sorted(rows, key=lambda r: (r[2], r[1]))
            _crop_ids = {r[0] for r in _crops}
    return _crops


def read_selected() -> set:
    return set(json.loads(SELECTED.read_text(encoding="utf-8"))) if SELECTED.exists() else set()


def read_selected_db() -> set:
    return set(json.loads(SELECTED_DB.read_text(encoding="utf-8"))) if SELECTED_DB.exists() else set()


def select(ids: list, on: bool, source: str = "crops"):
    if source == "crops":
        named_crops()
        allowed, path, read = _crop_ids, SELECTED, read_selected
    elif source == "db":
        allowed = {r[0] for r in konbinidb_view()["items"] if r[3] and r[5]}
        path, read = SELECTED_DB, read_selected_db
    else:
        raise ValueError("bad source")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids) or not isinstance(on, bool) or not set(ids) <= allowed:
        raise ValueError("bad selection")
    with _lock:
        s = read()
        s = s | set(ids) if on else s - set(ids)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(s)), encoding="utf-8")
        tmp.replace(path)


def konbinidb_view() -> dict:
    """コンビニDB の取得済みの全商品: [id, 商品名, チェーン, カテゴリ番号, サイトの分類, 画像があるか]。画像のあるものを先に"""
    labels = {}
    for line in DB_LABELS.open(encoding="utf-8"):
        x = json.loads(line)
        labels[x["id"]] = x["cat"]
    have = {f[:-5] for f in os.listdir(DB_IMAGES) if f.endswith(".webp")}
    rows = []
    for line in DB_ITEMS.open(encoding="utf-8"):
        x = json.loads(line)
        cat = None if excluded(x) or labels.get(x["id"]) == "000" else labels.get(x["id"])
        rows.append([x["id"], x["name"], x["chain"], cat, x["category"].rsplit("/", 1)[-1], x["id"] in have])
    rows.sort(key=lambda r: not r[5])
    selectable = {r[0] for r in rows if r[3] and r[5]}
    return {"items": rows, "images": len(have), "labeled": sum(bool(r[3]) for r in rows),
            "candidates": len(selectable), "selected": sorted(read_selected_db() & selectable)}


def store_sim_view(q):
    """店モデルのタブ: ブラウザからの条件を確かめて 1 回歩かせる"""
    import store_sim
    cfg = {}
    for k, lo, hi, kind in (("seed", 1, 9999, int), ("speed", 0.1, 1.5, float), ("drift", 0, 10, float), ("depth", 0, 5, float), ("px50", 10, 120, float), ("acc", 0, 1, float)):
        if k in q:
            v = kind(q[k][0])
            if not (math.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"bad {k}")
            cfg[k] = v
    for k, ok in (("mode", ("scan", "forward")), ("recog", ("real", "perfect"))):
        if k in q:
            if q[k][0] not in ok:
                raise ValueError(f"bad {k}")
            cfg[k] = q[k][0]
    for k in ("use_themes", "themed"):
        if k in q:
            if q[k][0] not in ("0", "1"):
                raise ValueError(f"bad {k}")
            cfg[k] = q[k][0] == "1"
    route = None
    if "route" in q:
        route = [tuple(float(x) for x in p.split(",")) for p in q["route"][0].split(";")]
        if not 2 <= len(route) <= 200 or not all(len(p) == 2 and all(math.isfinite(x) for x in p) for p in route):
            raise ValueError("bad route")
    return store_sim.simulate(cfg, route=route)


_replay_job = None


def replay_start(args):
    """記録の取り込み・再生は時間がかかるので別プロセスで動かす（1 度に 1 つ）"""
    global _replay_job
    if _replay_job and _replay_job.poll() is None:
        raise ValueError("実行中です")
    import replay
    replay.RECORDINGS.mkdir(parents=True, exist_ok=True)
    replay.set_status(state="running", job="start", msg="開始しています")
    _replay_job = subprocess.Popen([sys.executable, str(HERE.parent / "replay.py"), *args], cwd=HERE.parents[1], env={**os.environ, "PYTHONUTF8": "1"},
                                   stdout=open(replay.RECORDINGS / "job.out", "w", encoding="utf-8"), stderr=subprocess.STDOUT)


def replay_post(path, body):
    import replay
    if path == "/api/replay/pull":
        replay_start(["--pull"])
        return {}
    name, detector = body.get("name"), body.get("detector")
    if not (isinstance(name, str) and replay.NAME.match(name) and (replay.RECORDINGS / name).is_dir()) or detector not in replay.DETECTORS:
        raise ValueError("bad recording")
    if path == "/api/replay/run":
        hz = body.get("hz")
        if hz not in (1, 2, 5):
            raise ValueError("bad hz")
        replay_start(["--run", name, "--detector", detector, "--hz", str(hz)])
        return {}
    step, question = body.get("step"), body.get("question")
    if not isinstance(step, int) or not isinstance(question, str) or not 0 < len(question) <= 200:
        raise ValueError("bad question")
    return replay.ask(name, detector, step, question)


def replay_get(path):
    """(内容, 種類) を返す。記録の名前とファイル名は決まった形だけ通す（work/ の外や別のファイルを見せない）"""
    import replay
    if path == "/api/replay/list":
        return json.dumps(replay.recordings(), ensure_ascii=False).encode(), "application/json"
    if path == "/api/replay/status":
        s = json.loads((replay.RECORDINGS / "status.json").read_text(encoding="utf-8")) if (replay.RECORDINGS / "status.json").exists() else {}
        s["busy"] = bool(_replay_job and _replay_job.poll() is None)
        return json.dumps(s, ensure_ascii=False).encode(), "application/json"
    m = re.fullmatch(r"/api/replay/data/(\d{8}_\d{6})/([\w-]+)", path)
    if m and m[2] in replay.DETECTORS and (replay.RECORDINGS / m[1] / f"replay_{m[2]}.json").exists():
        return (replay.RECORDINGS / m[1] / f"replay_{m[2]}.json").read_bytes(), "application/json"
    m = re.fullmatch(r"/api/replay/file/(\d{8}_\d{6})/(frames/\d{5}\.jpg)", path)
    if m and (replay.RECORDINGS / m[1] / m[2]).exists():
        return (replay.RECORDINGS / m[1] / m[2]).read_bytes(), "image/jpeg"
    m = re.fullmatch(r"/api/replay/audio/(\d{8}_\d{6})\.wav", path)
    if m:
        import io
        import wave
        pcm = replay.RECORDINGS / m[1] / "audio.pcm"
        if pcm.exists():
            out = io.BytesIO()
            with wave.open(out, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm.read_bytes())
            return out.getvalue(), "audio/wav"
    m = re.fullmatch(r"/api/replay/depth/(\d{8}_\d{6})/(\d+)\.png", path)
    if m:
        import io
        import numpy as np
        from PIL import Image
        f = replay.RECORDINGS / m[1] / "depth" / f"{int(m[2]):05d}.d16"
        kf = next((k for k in replay.jsonl(replay.RECORDINGS / m[1] / "keyframes.jsonl") if k["n"] == int(m[2])), None)
        if f.exists() and kf and kf.get("depth"):
            w, h = kf["depth"]
            d = np.fromfile(f, dtype="<u2").reshape(h, w)
            im = Image.fromarray((255 - np.clip(d / 4000 * 255, 0, 255)).astype("uint8") * (d > 0)).rotate(-kf["geom"]["rot"], expand=True)
            b = io.BytesIO()
            im.save(b, "PNG")
            return b.getvalue(), "image/png"
    return None


def handler(det: Path, pred_dir: Path):
    names = {f.name for f in VIEW.glob("*.jpg")}

    class H(BaseHTTPRequestHandler):
        def send(self, body: bytes, ctype: str, cache: bool = False, status: int = 200):
            self.send_response(status)
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
            elif path == "/api/crops":
                self.send(json.dumps({"crops": named_crops(), "selected": sorted(read_selected())}, ensure_ascii=False).encode(), "application/json")
            elif path == "/api/konbinidb":
                self.send(json.dumps(konbinidb_view(), ensure_ascii=False).encode(), "application/json")
            elif path == "/replay":
                self.send((HERE.parent / "replay.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/replay_tab.js":
                self.send((HERE.parent / "replay_tab.js").read_bytes(), "text/javascript; charset=utf-8")
            elif path.startswith("/api/replay/"):
                r = replay_get(path)
                self.send(*r) if r else self.send_error(404)
            elif path == "/store":
                self.send((HERE.parent / "store.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/store_tab.js":
                self.send((HERE.parent / "store_tab.js").read_bytes(), "text/javascript; charset=utf-8")
            elif path == "/api/store_sim_batch":
                f = WORK.parent / "store_sim_batch.json"
                self.send(f.read_bytes() if f.exists() else b"{}", "application/json")
            elif path == "/api/store_sim":
                try:
                    self.send(json.dumps(store_sim_view(parse_qs(urlparse(self.path).query)), ensure_ascii=False).encode(), "application/json")
                except ValueError as e:
                    self.send_error(400, str(e))
            elif path.startswith("/view/") and path[6:] in names:  # 一覧にある名前だけ返す
                self.send((VIEW / path[6:]).read_bytes(), "image/jpeg", cache=True)
            elif path.startswith("/crop/") and path.endswith(".jpg") and (named_crops() and path[6:-4] in _crop_ids):
                self.send((CROPS / path[6:]).read_bytes(), "image/jpeg", cache=True)
            elif re.fullmatch(r"/db/\d+\.webp", path) and (DB_IMAGES / path[4:]).exists():
                self.send((DB_IMAGES / path[4:]).read_bytes(), "image/webp", cache=True)
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlparse(self.path).path
            if path.startswith("/api/replay/"):
                try:
                    out = replay_post(path, json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 10000)) or b"{}"))
                except (ValueError, TypeError, AttributeError, OSError) as e:
                    return self.send(str(e).encode(), "text/plain; charset=utf-8", status=400)
                return self.send(json.dumps(out, ensure_ascii=False).encode(), "application/json")
            if path != "/api/select":
                return self.send_error(404)
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                select(body.get("ids"), body.get("on"), body.get("source", "crops"))
            except (ValueError, TypeError, AttributeError) as e:
                return self.send_error(400, str(e))
            self.send(b"{}", "application/json")

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
