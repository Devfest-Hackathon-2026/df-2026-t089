"""商品認識パイプラインの進捗・中間結果・評価を見るローカルダッシュボード（標準ライブラリのみ）。

  python products/dashboard.py [--port 8770]   -> http://127.0.0.1:8770
"""
import argparse, csv, json, math, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from common import WORK, config, read_json, write_json

HERE = Path(__file__).resolve().parent
PY = sys.executable  # venv の Python で起動すれば、ボタンから走らせるスクリプトも同じ環境で動く
MODAL = str(Path(PY).parent / "modal")
# ボタン名 → コマンド。Modal を使うものは GPU で動く（①の検出、2b の埋め込み）
SCRIPTS = {"detect_gpu": [MODAL, "run", str(HERE / "s1_gpu.py")],
           "crops": [PY, str(HERE / "s2_crops.py")], "crops_gpu": [PY, str(HERE / "s2_crops.py"), "--modal"], "judge": [PY, str(HERE / "s2_judge.py")],
           "group": [PY, str(HERE / "s2_group.py")], "group_gpu": [PY, str(HERE / "s2_group.py"), "--modal"],
           "regroup": [PY, str(HERE / "s2_group.py"), "--only-group"], "name": [PY, str(HERE / "s3_name.py")],
           "compare": [PY, str(HERE / "s3_compare.py")], "classify": [PY, str(HERE / "s5_classify.py")],
           "train_gpu": [PY, str(HERE / "s4_train.py"), "--modal", "--model", "yolo11s.pt", "--epochs", "80", "--batch", "16"]}
CMP = WORK / "naming_compare"


def compare_view():
    """②c 命名方法の比較: 対象の切り抜き・方法ごとの結果・人が付けた正誤"""
    methods = {}
    for f in sorted(CMP.glob("*.json")):
        if f.stem not in ("sample", "eval"):
            methods[f.stem] = read_json(f)
    return {"sample": read_json(CMP / "sample.json", []), "methods": methods, "eval": read_json(CMP / "eval.json", {})}
RUNNING: dict = {}  # ボタンで起動中のスクリプト
_cache: dict = {"key": None, "rows": None}
_lock = threading.Lock()
TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json", ".jpg": "image/jpeg",
         ".png": "image/png", ".csv": "text/csv; charset=utf-8", ".jsonl": "text/plain; charset=utf-8"}


def product_rows():
    """全切り抜きの一覧（判定と名前を合成）。大きいので、元ファイルが変わったときだけ作り直す
    state: rule=ルールで除外(黒) / model=ローカルモデルで除外(青) / pass=合格(緑) / bad=Vision の結果が悪い(オレンジ) / None=未判定"""
    files = [WORK / f for f in ("crops.json", "judge.json", "names.json")]
    key = tuple(f.stat().st_mtime if f.exists() else 0 for f in files)
    with _lock:
        if _cache["key"] != key:
            crops, judge, names = (read_json(f, default) for f, default in zip(files, ([], {}, {})))
            rows = []
            for c in crops:
                j, n = judge.get(c["id"]), names.get(c["id"], {})
                st = "rule" if c["rule"] else None if j is None else "pass" if j["ok"] else "model"
                if st == "pass" and n.get("vision") == "bad" and n.get("source") != "manual":
                    st = "bad"
                why = c["rule"] or (j and f'{j["top"]} ({j["p_bad"]})') or ""
                rows.append([c["id"], c["image"], round(c["score"], 3), c["w"], c["h"], st, why,
                             n.get("maker", ""), n.get("product", "")])
            _cache.update(key=key, rows=rows)
        return _cache["rows"]


_gcache: dict = {"key": None, "data": None}


def groups_view():
    """グループ一覧。2枚以上のグループの後に、どのグループにも入らなかった合格（緑）の切り抜きを1枚グループとして並べる。
    名前は Vision/手入力の社名・商品名を数えて添える"""
    files = [WORK / "groups.json", WORK / "names.json", WORK / "crops.json", WORK / "judge.json"]
    key = tuple(f.stat().st_mtime if f.exists() else 0 for f in files)
    with _lock:
        if _gcache["key"] != key:
            g, names, crops, judge = (read_json(f, d) for f, d in zip(files, ({}, {}, [], {})))
            image = {c["id"]: c["image"] for c in crops}
            of = g.get("of", {})
            multi = [(gr["members"], gr["rep"]) for gr in g.get("groups", [])]
            single = [([c["id"]], c["id"]) for c in crops if judge.get(c["id"], {}).get("ok") and c["id"] not in of]
            out = []
            for k, (members, rep) in enumerate(multi + single):
                cnt = {}
                for m in members:
                    n = names.get(m, {})
                    if n.get("maker") or n.get("product"):
                        nm = " ".join(x for x in (n.get("maker"), n.get("product")) if x)
                        cnt[nm] = cnt.get(nm, 0) + 1
                out.append({"id": k, "size": len(members), "images": len({image.get(m) for m in members}), "rep": rep,
                            "members": [[m, image.get(m, "")] for m in members],
                            "names": sorted(cnt.items(), key=lambda x: -x[1])})
            _gcache.update(key=key, data={"groups": out, "sim": g.get("sim"), "color": g.get("color"), "model": g.get("model"),
                                          "multi": len(multi), "single": len(single),
                                          "grouped": sum(len(m) for m, _ in multi)})
        return _gcache["data"]


def summary(rows):
    s = {"all": len(rows), "none": 0, "rule": 0, "model": 0, "pass": 0, "bad": 0, "named": 0}
    for r in rows:
        s[r[5] or "none"] += 1
        s["named"] += bool(r[7] or r[8])
    s["products"] = len({(r[7], r[8]) for r in rows if r[7] or r[8]})
    return s


def run_script(name, args):
    if name in RUNNING and RUNNING[name].poll() is None:
        raise ValueError(f"{name} は実行中です")
    if name.startswith(("group", "regroup")) and any(k.startswith(("group", "regroup")) and p.poll() is None for k, p in RUNNING.items()):
        raise ValueError("グループ化を実行中です")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    RUNNING[name] = subprocess.Popen([*SCRIPTS[name], *args], cwd=HERE.parent, env=env,
                                     stdout=open(WORK / f"{name}.out", "w", encoding="utf-8"), stderr=subprocess.STDOUT)


def state():
    det = []
    for p in sorted((WORK / "det").glob("*.json")):
        d = read_json(p)
        if d:
            det.append({"stem": p.stem, "image": d["image"], "scores": d["scores"], "sec": d.get("sec")})
    lines = []
    if (WORK / "log.jsonl").exists():
        lines = (WORK / "log.jsonl").read_text(encoding="utf-8").splitlines()[-400:]
    train = []
    for p in sorted((WORK / "train").glob("*/results.csv")):
        with open(p, encoding="utf-8") as f:
            train.append({"run": p.parent.name, "rows": [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]})
    infer = [read_json(p) for p in sorted((WORK / "infer").glob("*.json"))]
    return {"status": read_json(WORK / "status.json", {}), "config": config(), "now": time.time(),
            "log": [json.loads(l) for l in lines if l.strip()], "det": det,
            "rejects": read_json(WORK / "rejects.json", {}),
            "summary": summary(product_rows()), "train": train,
            "running": [k for k, p in RUNNING.items() if p.poll() is None],
            "vision_used": read_json(WORK / "vision_usage.json", {}).get(time.strftime("%Y-%m"), 0),
            "infer": [x for x in infer if x], "eval": read_json(WORK / "eval.json"),
            "experiments": read_json(WORK / "experiments.json", [])}


def post(path, body):
    """ブラウザからの編集の受け口なので、型と範囲をここで確かめる"""
    if path == "/api/config":
        lim = {"det_threshold": (0.01, 0.99), "group_sim": (0.3, 0.99), "group_color": (0, 0.99), "min_crop_w": (0, 2000), "min_crop_h": (0, 2000)}
        cur = read_json(WORK / "config.json", {})
        for k, v in body.items():
            if k not in lim or not isinstance(v, (int, float)) or not math.isfinite(v) or not lim[k][0] <= v <= lim[k][1]:
                raise ValueError(f"bad config {k}={v!r}")
            cur[k] = v
        write_json(WORK / "config.json", cur)
    elif path == "/api/reject":
        stem, idx = body.get("stem"), body.get("idx")
        if not isinstance(stem, str) or not (WORK / "det" / f"{stem}.json").exists() or not isinstance(idx, int):
            raise ValueError("bad reject")
        r = read_json(WORK / "rejects.json", {})
        s = set(r.get(stem, []))
        s ^= {idx}
        r[stem] = sorted(s)
        write_json(WORK / "rejects.json", r)
    elif path == "/api/name":
        cid, maker, product = body.get("id"), body.get("maker"), body.get("product")
        if not isinstance(cid, str) or not all(isinstance(x, str) and len(x) <= 100 for x in (maker, product)):
            raise ValueError("bad name")
        with _lock:
            n = read_json(WORK / "names.json", {})
            if maker.strip() or product.strip():
                n[cid] = {**n.get(cid, {}), "maker": maker.strip(), "product": product.strip(), "source": "manual"}
            else:
                n.pop(cid, None)
            write_json(WORK / "names.json", n)
    elif path == "/api/compare_eval":
        cid, meth, v = body.get("id"), body.get("method"), body.get("v")
        if not isinstance(cid, str) or meth not in {f.stem for f in CMP.glob("*.json")} - {"sample", "eval"} or v not in (1, 0, None):
            raise ValueError("bad eval")
        with _lock:
            e = read_json(CMP / "eval.json", {})
            if v is None:
                e.get(cid, {}).pop(meth, None)
            else:
                e.setdefault(cid, {})[meth] = v
            write_json(CMP / "eval.json", e)
    elif path == "/api/run":
        name, k = body.get("script"), body.get("n", 50)
        if name not in SCRIPTS or not isinstance(k, int) or not 1 <= k <= 100000:
            raise ValueError("bad run")
        run_script(name, ["--n", str(k)] if name == "name" else [])
    else:
        raise KeyError(path)


class H(BaseHTTPRequestHandler):
    def send(self, code, data: bytes, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        if ctype == TYPES[".json"] or ctype.startswith("text/html"):  # 画像はキャッシュさせて4秒ごとの更新でちらつかせない
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self.send(200, (HERE / "dashboard.html").read_bytes(), TYPES[".html"])
        if path == "/api/compare":
            return self.send(200, json.dumps(compare_view(), ensure_ascii=False).encode(), TYPES[".json"])
        if path == "/api/groups":
            return self.send(200, json.dumps(groups_view(), ensure_ascii=False).encode(), TYPES[".json"])
        if path == "/api/products":
            return self.send(200, json.dumps(product_rows(), ensure_ascii=False).encode(), TYPES[".json"])
        if path.startswith("/api/crop/"):
            cid = path[10:]
            return self.send(200, json.dumps({"name": read_json(WORK / "names.json", {}).get(cid),
                                              "judge": read_json(WORK / "judge.json", {}).get(cid)}, ensure_ascii=False).encode(), TYPES[".json"])
        if path == "/api/state":
            return self.send(200, json.dumps(state(), ensure_ascii=False).encode(), TYPES[".json"])
        if path.startswith("/work/"):
            f = (WORK / path[6:]).resolve()
            if f.is_file() and f.is_relative_to(WORK.resolve()):  # work/ の外は見せない
                return self.send(200, f.read_bytes(), TYPES.get(f.suffix.lower(), "application/octet-stream"))
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 10000:
                raise ValueError("too large")
            post(urlparse(self.path).path, json.loads(self.rfile.read(n) or b"{}"))
            self.send(200, b"{}", TYPES[".json"])
        except (ValueError, KeyError, AttributeError, json.JSONDecodeError) as e:
            self.send(400, str(e).encode(), "text/plain")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    port = ap.parse_args().port
    print(f"http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
