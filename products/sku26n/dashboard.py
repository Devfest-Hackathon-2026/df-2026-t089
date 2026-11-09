"""26n-sku の切り出しと、カテゴリ認識 3 手法の結果を見比べるダッシュボード。

  python products/sku26n/dashboard.py [--port 8771]   -> http://127.0.0.1:8771
各手法の結果は products/work/sku26n/pred_<手法>.json: {"images": {画像名: [[x0, y0, x1, y1, 信頼度, カテゴリ番号], ...]}}（座標は 0..1）
"""
import argparse, csv, json, os, re, subprocess, sys, threading, unicodedata
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
VERIFY = WORK / "verify_all.json"
TAXONOMY = HERE.parent / "konbinidb" / "taxonomy.json"
METHODS = ["trained", "b", "b_lr1e3", "c", "gemini"]
AUG = DB_IMAGES.parent / "train_aug"  # コンビニDB を店内の切り抜きに似せた学習データ（konbinidb/make_train.py）


_annotation_lock = threading.Lock()
_annotation_cache = {"key": None, "rows": [], "by_box": {}}


def annotation_rows(pred_dir: Path) -> list:
    """Return complete JSONL records while another process appends annotations."""
    log = pred_dir / "label_all.jsonl"
    if not log.exists():
        return []
    stat = log.stat()
    key = (str(log), stat.st_mtime_ns, stat.st_size)
    with _annotation_lock:
        if _annotation_cache["key"] != key:
            data = log.read_bytes()
            if not data.endswith(b"\n"):
                data = data.rsplit(b"\n", 1)[0] + b"\n"
            rows = [json.loads(line) for line in data.splitlines() if line]
            _annotation_cache.update(key=key, rows=rows,
                                     by_box={(r["image"], r["box"]): r for r in rows})
        return _annotation_cache["rows"]


def gemini_predictions(pred_dir: Path) -> dict:
    """Read the completed lines of the ongoing Gemini annotation run."""
    log = pred_dir / "label_all.jsonl"
    if not log.exists():
        saved = pred_dir / "pred_gemini.json"
        return json.loads(saved.read_text(encoding="utf-8"))["images"] if saved.exists() else {}
    codes = {"unsure": "UNS", "multiple": "MLT", "not_product": "NP"}
    images = {}
    for row in annotation_rows(pred_dir):
        cat = row["cat"] if row["status"] == "ok" else codes[row["status"]]
        box = row["xyxy"] + [row["det_conf"], cat, False, row.get("maker", ""), row.get("product", "")]
        images.setdefault(row["image"], []).append(box)
    return images


def confirmed_view(pred_dir: Path, params: dict) -> dict:
    query = params.get("q", [""])[0].strip().casefold()[:100]
    category = params.get("cat", [""])[0]
    try:
        page = max(0, int(params.get("page", ["0"])[0]))
    except ValueError:
        page = 0
    rows = [r for r in reversed(annotation_rows(pred_dir)) if r["status"] == "ok"
            and (not category or r["cat"] == category)
            and (not query or query in " ".join(str(r.get(k, "")) for k in ("maker", "product", "image", "cat")).casefold())]
    per_page = 100
    pages = max(1, (len(rows) + per_page - 1) // per_page)
    page = min(page, pages - 1)
    shown = rows[page * per_page:(page + 1) * per_page]
    return {"total": len(rows), "page": page, "pages": pages,
            "rows": [[r["image"], r["box"], r["cat"], r.get("maker", ""),
                      r.get("product", ""), r["det_conf"]] for r in shown]}


def confirmed_crop(pred_dir: Path, name: str, number: int) -> bytes | None:
    import io
    from PIL import Image

    annotation_rows(pred_dir)
    row = _annotation_cache["by_box"].get((name, number))
    if row is None or row["status"] != "ok":
        return None
    with Image.open(VIEW / name) as image:
        x0, y0, x1, y1 = row["xyxy"]
        pad_x, pad_y = (x1 - x0) * .05, (y1 - y0) * .05
        box = (int(max(0, x0 - pad_x) * image.width), int(max(0, y0 - pad_y) * image.height),
               int(min(1, x1 + pad_x) * image.width), int(min(1, y1 + pad_y) * image.height))
        crop = image.crop(box).convert("RGB")
        crop.thumbnail((240, 240))
        out = io.BytesIO()
        crop.save(out, "JPEG", quality=85)
        return out.getvalue()


def state(det: Path, pred_dir: Path) -> bytes:
    d = json.loads(det.read_text(encoding="utf-8")) if det.exists() else \
        {"weights": None, "images": [{"name": f.name, "boxes": []} for f in sorted(VIEW.glob("*.jpg"))]}
    d["methods"] = {m: json.loads((pred_dir / f"pred_{m}.json").read_text(encoding="utf-8"))["images"]
                    for m in METHODS if m != "gemini" and (pred_dir / f"pred_{m}.json").exists()}
    d["methods"]["gemini"] = gemini_predictions(pred_dir)
    manifest = pred_dir / "bc_manifest.json"
    if manifest.exists():
        details = json.loads(manifest.read_text(encoding="utf-8"))
        d["training"] = {}
        for variant in ("b", "c"):
            results = pred_dir / f"bc_{variant}_results.csv"
            if results.exists():
                with results.open(newline="", encoding="utf-8") as source:
                    epochs = list(csv.DictReader(source))
                best = max(epochs, key=lambda row: float(row["metrics/mAP50-95(B)"]))
                d["training"][variant] = {"photos": details["photos"],
                    "accepted": details["accepted_boxes"], "ignored": details["ignored_boxes"],
                    "epoch": int(best["epoch"]), "map50": float(best["metrics/mAP50(B)"]),
                    "map5095": float(best["metrics/mAP50-95(B)"]),
                    "precision": float(best["metrics/precision(B)"]),
                    "recall": float(best["metrics/recall(B)"])}
    d["cats"] = {k: v[0] for g in json.loads(TAXONOMY.read_text(encoding="utf-8")).values() for k, v in g.items()}
    d["cats"].update({"NP": "商品でない", "UNS": "断定できない", "MLT": "主役を絞れない"})
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


def verification_view() -> dict:
    """商品名ごとの代表切り抜きに対する Gemini 判定。合格時のカテゴリは新しい分類体系の番号。"""
    crops = named_crops()
    reps = {}
    for r in crops:
        reps.setdefault(re.sub(r"\s", "", unicodedata.normalize("NFKC", r[1])).lower(), r[0])
    rep_ids = set(reps.values())
    rows = json.loads(VERIFY.read_text(encoding="utf-8"))["rows"] if VERIFY.exists() else []
    judged = {r["id"]: [r["status"], r.get("pass") is True, r.get("cat") if r.get("pass") else None]
              for r in rows if r["id"] in rep_ids}
    return {"rows": judged, "representatives": len(reps), "judged": len(judged),
            "passed": sum(r[1] for r in judged.values()), "complete": set(judged) == rep_ids}


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


def aug_manifest() -> dict:
    f = AUG / "manifest.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"rows": [], "skipped": 0, "variants": 0}


def aug_crop(item_id: str) -> bytes | None:
    """sku110k-2 の枠で切り抜いたコンビニDB の画像（学習データの確認用。保存せず、その場で作る）"""
    import io
    from PIL import Image
    row = next((r for r in aug_manifest()["rows"] if r["id"] == item_id), None)
    if row is None:
        return None
    im = Image.open(DB_IMAGES / f"{item_id}.webp").convert("RGB")
    b = row["box"]
    out = io.BytesIO()
    im.crop((int(b[0] * im.width), int(b[1] * im.height), int(b[2] * im.width), int(b[3] * im.height))).save(out, "JPEG", quality=85)
    return out.getvalue()


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
    m = re.fullmatch(r"/api/replay/events/(\d{8}_\d{6})", path)
    if m and (replay.RECORDINGS / m[1]).is_dir():
        d = replay.RECORDINGS / m[1]
        keyframes = replay.jsonl(d / "keyframes.jsonl")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        first = keyframes[0] if keyframes else {}
        offset = max(0, (first.get("wall_ms", 0) - meta.get("start_ms", 0)) / 1000) if first else 0
        events = [{**e, "at": round((e["t"] - first["t"]) / 1e9, 2) if first else round((e["wall_ms"] - meta["start_ms"]) / 1000, 2)}
                  for e in replay.jsonl(d / "events.jsonl")]
        audio_start = meta.get("audio_start_ms")
        app_audio_start = meta.get("app_audio_start_ms")
        wall = first.get("wall_ms")
        return json.dumps({"events": events, "video_offset": offset,
                           "audio_offset": max(0, (wall - audio_start) / 1000) if wall and audio_start else 0,
                           "app_audio_offset": max(0, (wall - app_audio_start) / 1000) if wall and app_audio_start else 0}, ensure_ascii=False).encode(), "application/json"
    m = re.fullmatch(r"/api/replay/data/(\d{8}_\d{6})/([\w-]+)", path)
    if m and m[2] in replay.DETECTORS and (replay.RECORDINGS / m[1] / f"replay_{m[2]}.json").exists():
        return (replay.RECORDINGS / m[1] / f"replay_{m[2]}.json").read_bytes(), "application/json"
    m = re.fullmatch(r"/api/replay/file/(\d{8}_\d{6})/(frames/\d{5}\.jpg)", path)
    if m and (replay.RECORDINGS / m[1] / m[2]).exists():
        return (replay.RECORDINGS / m[1] / m[2]).read_bytes(), "image/jpeg"
    m = re.fullmatch(r"/api/replay/preview/(\d{8}_\d{6})\.mp4", path)
    if m and (replay.RECORDINGS / m[1] / "preview30.mp4").exists():
        return (replay.RECORDINGS / m[1] / "preview30.mp4").read_bytes(), "video/mp4"
    m = re.fullmatch(r"/api/replay/(audio|app_audio)/(\d{8}_\d{6})\.wav", path)
    if m:
        import io
        import wave
        app_audio = m[1] == "app_audio"
        pcm = replay.RECORDINGS / m[2] / ("app_audio.pcm" if app_audio else "audio.pcm")
        if pcm.exists():
            out = io.BytesIO()
            with wave.open(out, "wb") as wav:
                wav.setnchannels(2 if app_audio else 1)
                wav.setsampwidth(2)
                wav.setframerate(44100 if app_audio else 16000)
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
            elif path == "/api/confirmed":
                params = parse_qs(urlparse(self.path).query)
                self.send(json.dumps(confirmed_view(pred_dir, params), ensure_ascii=False).encode(), "application/json")
            elif (match := re.fullmatch(r"/api/confirmed/crop/([^/]+)/(\d+)\.jpg", path)) and match[1] in names:
                body = confirmed_crop(pred_dir, match[1], int(match[2]))
                self.send(body, "image/jpeg") if body else self.send_error(404)
            elif path == "/api/crops":
                self.send(json.dumps({"crops": named_crops(), "selected": sorted(read_selected()),
                                      "verification": verification_view()}, ensure_ascii=False).encode(), "application/json")
            elif path == "/api/konbinidb":
                self.send(json.dumps(konbinidb_view(), ensure_ascii=False).encode(), "application/json")
            elif path == "/replay":
                self.send((HERE.parent / "replay.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/replay_tab.js":
                self.send((HERE.parent / "replay_tab.js").read_bytes(), "text/javascript; charset=utf-8")
            elif path.startswith("/api/replay/"):
                r = replay_get(path)
                self.send(*r) if r else self.send_error(404)
            elif path.startswith("/view/") and path[6:] in names:  # 一覧にある名前だけ返す
                self.send((VIEW / path[6:]).read_bytes(), "image/jpeg", cache=True)
            elif path.startswith("/crop/") and path.endswith(".jpg") and (named_crops() and path[6:-4] in _crop_ids):
                self.send((CROPS / path[6:]).read_bytes(), "image/jpeg", cache=True)
            elif re.fullmatch(r"/db/\d+\.webp", path) and (DB_IMAGES / path[4:]).exists():
                self.send((DB_IMAGES / path[4:]).read_bytes(), "image/webp", cache=True)
            elif re.fullmatch(r"/api/top5/(m1|clip)/[^/]+\.jpg", path) and path.rsplit("/", 1)[1] in names and (WORK / "top5" / path[10:]).exists():
                self.send((WORK / "top5" / path[10:]).read_bytes(), "application/json", cache=True)
            elif path == "/api/train_aug":
                self.send(json.dumps(aug_manifest(), ensure_ascii=False).encode(), "application/json")
            elif re.fullmatch(r"/aug/\d+_[0-9]\.jpg", path) and (AUG / path[5:]).exists():
                self.send((AUG / path[5:]).read_bytes(), "image/jpeg", cache=True)
            elif re.fullmatch(r"/aug_crop/\d+\.jpg", path):
                body = aug_crop(path[10:-4])
                self.send(body, "image/jpeg", cache=True) if body else self.send_error(404)
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
