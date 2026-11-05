"""Run the cloud shelf prompt on test photos and write an input/output report.

The app and this script share app/src/main/assets/cloud, so a prompt tuned here behaves the same on the phone.

  GEMINI_API_KEY comes from the environment or local.properties (neither is committed).

  python tools/cloud_test.py                        # the frontal shelf photos below
  python tools/cloud_test.py --images a.jpg b.jpg   # other shelf photos
  python tools/cloud_test.py --dry-run              # build the requests without sending anything

The report goes to tools/out/cloud_test/report.html. tools/out is not committed because photos can show customers.
"""
import argparse, base64, html, io, json, os, sys, time, urllib.error, urllib.request
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app/src/main/assets"
OUT = ROOT / "tools/out/cloud_test"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
API_REVISION = "2026-05-20"
DEFAULT_MODELS = "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite"
RETRY_DELAYS = (0.3, 0.3, 1, 3, 8, 20)  # longer than the app: this run wants an answer for every photo
MAX_LONG_SIDE = 1920
PHOTOS = ROOT / "boundary_training_images"

# Chosen by eye: taken from the front, with no customers visible.
FRONTAL_SHELVES = ["PXL_20261105_132704139.jpg", "PXL_20261105_132737718.jpg", "PXL_20261105_132951988.jpg",
                   "PXL_20261105_132955455.jpg", "PXL_20261105_133014564.jpg", "PXL_20261105_133018764.jpg",
                   "PXL_20261105_133041690.jpg"]


def local_property(name):
    path = ROOT / "local.properties"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                return line[len(name) + 1:].strip()
    return None


def build_task(name):
    categories = [c for c in (ASSETS / "labels.txt").read_text(encoding="utf-8").splitlines() if c]
    prompt = (ASSETS / f"cloud/{name}.prompt.txt").read_text(encoding="utf-8")
    schema = (ASSETS / f"cloud/{name}.schema.json").read_text(encoding="utf-8")
    prompt = prompt.replace("{{CATEGORIES}}", "\n".join(f"- {c}" for c in categories))
    schema = schema.replace('"{{CATEGORIES_ENUM}}"', json.dumps(categories, ensure_ascii=False))
    return prompt, json.loads(schema)


def load_jpeg(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((MAX_LONG_SIDE, MAX_LONG_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return buf.getvalue(), im


def request_body(model, prompt, schema, jpeg):
    return {
        "model": model,
        "store": False,  # do not keep photos on the server after answering
        "input": [
            {"type": "text", "text": prompt},
            {"type": "image", "data": base64.b64encode(jpeg).decode(), "mime_type": "image/jpeg", "resolution": "high"},
        ],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": schema},
    }


def call_with_fallback(api_key, models, prompt, schema, jpeg):
    """Same policy as GeminiClient.kt: a busy model falls over to the next one."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        model = models[attempt % len(models)]
        try:
            return (*call(api_key, request_body(model, prompt, schema, jpeg)), model, attempt)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 503) or attempt == len(RETRY_DELAYS):
                raise
            time.sleep(RETRY_DELAYS[attempt])


def call(api_key, body):
    req = urllib.request.Request(ENDPOINT, json.dumps(body).encode(), method="POST", headers={
        "x-goog-api-key": api_key, "Content-Type": "application/json", "Api-Revision": API_REVISION})
    start = time.time()
    with urllib.request.urlopen(req, timeout=120) as res:
        reply = json.loads(res.read())
    if reply.get("status") != "completed":
        raise RuntimeError(f"status={reply.get('status')} errors={reply.get('errors')}")
    text = "".join(part.get("text", "") for step in reply.get("steps", []) if step.get("type") == "model_output"
                   for part in step.get("content", []) if part.get("type") == "text")
    return json.loads(text), time.time() - start, reply.get("usage", {})


def render_shelf(row):
    bars = "".join(
        f'<div class="col" style="left:{c["x_start"] / 10}%;width:{(c["x_end"] - c["x_start"]) / 10}%">'
        f'<b>{html.escape(c["category"])}</b></div>' for c in row["output"].get("columns", []))
    lines = "".join(
        f'<li><b>{html.escape(c["category"])}</b> {c["x_start"]}–{c["x_end"]} ({c["confidence"]}) {html.escape(c["evidence"])}</li>'
        for c in row["output"].get("columns", []))
    return bars, f'<p>棚の正面: {row["output"].get("is_shelf_front")}</p><ol>{lines}</ol>'


def write_report(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        if "error" in row:
            bars, details = "", f'<p class="err">{html.escape(row["error"])}</p>'
        else:
            bars, details = render_shelf(row)
            details += f'<p class="meta">{html.escape(row["model"])} / {row["seconds"]:.1f} 秒 / 再試行 {row["retries"]} 回 / トークン {row["usage"].get("total_tokens", "?")}</p>'
        cards.append(f'<section><div class="img"><img src="{html.escape(row["preview"])}">{bars}</div>'
                     f'<div><h2>{html.escape(row["file"])}</h2>{details}</div></section>')
    page = """<!DOCTYPE html><html lang="ja"><meta charset="utf-8"><title>クラウド判別のテスト</title><style>
body{font:15px system-ui,sans-serif;margin:24px;background:#fafafa;color:#222}
section{display:flex;gap:20px;margin:0 0 24px;padding:16px;background:#fff;border:1px solid #ddd;border-radius:8px}
.img{position:relative;flex:0 0 360px}img{width:100%;display:block}
.col{position:absolute;top:0;bottom:0;border:2px solid #e53935;box-sizing:border-box}
.col b{background:#e53935;color:#fff;font-size:12px;padding:1px 4px}
h2{font-size:14px;margin:0 0 8px}.err{color:#c62828}.meta{color:#777;font-size:13px}pre{white-space:pre-wrap;background:#f3f3f3;padding:8px}
</style><h1>クラウド判別のテスト</h1>""" + "".join(cards) + "</html>"
    (OUT / "report.html").write_text(page, encoding="utf-8")
    (OUT / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUT / "report.html"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", type=Path)
    ap.add_argument("--models", help="comma separated, tried in order; default as in the app")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    models = (a.models or local_property("GEMINI_MODELS") or DEFAULT_MODELS).split(",")
    api_key = os.environ.get("GEMINI_API_KEY") or local_property("GEMINI_API_KEY")
    if not api_key and not a.dry_run:
        sys.exit("GEMINI_API_KEY is not set. Add GEMINI_API_KEY=... to local.properties.")

    shelves = a.images if a.images is not None else [PHOTOS / f for f in FRONTAL_SHELVES]
    jobs = [("shelf_columns", p) for p in shelves]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, path in jobs:
        prompt, schema = build_task(task)
        jpeg, im = load_jpeg(path)
        preview = OUT / f"{path.stem}.preview.jpg"
        small = im.copy()
        small.thumbnail((720, 720))
        small.save(preview, quality=80)
        row = {"task": task, "file": path.name, "preview": preview.name}
        if a.dry_run:
            body = request_body(models[0], prompt, schema, jpeg)
            print(f"[dry-run] {task} {path.name}: image {len(jpeg) // 1024} KB, request {len(json.dumps(body)) // 1024} KB, "
                  f"prompt {len(prompt)} chars, schema keys {list(schema['properties'])}")
            continue
        try:
            row["output"], row["seconds"], row["usage"], row["model"], row["retries"] = call_with_fallback(
                api_key, models, prompt, schema, jpeg)
            print(f"{task} {path.name}: {row['model']} {row['seconds']:.1f}s retries={row['retries']} "
                  f"{json.dumps(row['output'], ensure_ascii=False)}")
        except urllib.error.HTTPError as e:  # the body is Google's message; it does not contain the key
            row["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:500]}"
            print(f"{task} {path.name}: {row['error']}")
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"{task} {path.name}: {row['error']}")
        rows.append(row)
    if not a.dry_run:
        print("report:", write_report(rows))


if __name__ == "__main__":
    main()
