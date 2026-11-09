"""sku110k-2 の枠を、元の写真から切り抜いて Gemini で 122 カテゴリに分類する。

  .venv/Scripts/python products/sku26n/label_all.py --boxes 200       無作為な画像を丸ごと、枠が N 個になるまで（お試し）
  .venv/Scripts/python products/sku26n/label_all.py --boxes 0         全部
主役を 1 種類に絞れないもの、種類を断定できないものは除外する。止まっても続きから再開できる。
出力: products/work/sku26n/label_all*.json（git に入らない）
"""
import argparse, json, random, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from label_gemini import CATALOG, GEMINI, MODEL, TAX, WORK, gemini_key, inline, load  # noqa: E402

PER_REQUEST = 20
CROP_SIDE = 512
SHELF_SIDE = 768
MARGIN = 0.05
STATUS = ("ok", "not_product", "unsure", "multiple")
PROMPT = f"""コンビニの店内の写真から切り出した商品を分類してください。
最初の画像は棚全体です。続く画像が切り抜きで、それぞれの前に番号があります。

status は次のどれか 1 つ:
- "ok": 主役の商品が 1 種類に絞れて、カテゴリを断定できる。同じ商品が並んでいたり積み重なっていたりするのは問題ない。
- "multiple": 別の種類の商品（同じブランドの味違いも別の種類とする）が 2 つ以上あり、どれも全体が写っている。主役以外の種類が小さく写っているだけ、または一部しか写っていないだけなら "multiple" にせず、メインの 1 つで判定する。
- "unsure": カテゴリを断定できない（ぼやけている、暗い、一部しか写っていない）。少しでも迷ったら "unsure" にする。棚全体の様子だけから推測しない。
- "not_product": 値札、POP、棚、人など商品でないもの。

"ok" のときだけ cat に下のカテゴリ番号を入れる。

カテゴリ一覧:
{CATALOG}

JSON だけを返す: {{"results": {{"番号": {{"status": "ok", "cat": "カテゴリ番号"}}, ...}}}}"""


def crop(im, b):
    w, h = (b[2] - b[0]) * MARGIN, (b[3] - b[1]) * MARGIN
    return im.crop((int(max(0, b[0] - w) * im.width), int(max(0, b[1] - h) * im.height), int(min(1, b[2] + w) * im.width), int(min(1, b[3] + h) * im.height)))


def ask(im, boxes):
    parts = [{"text": PROMPT}, inline(im, SHELF_SIDE)]
    for i, b in enumerate(boxes):
        parts += [{"text": f"切り抜き {i}:"}, inline(crop(im, b), CROP_SIDE)]
    body = {"contents": [{"parts": parts}], "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "mediaResolution": "MEDIA_RESOLUTION_MEDIUM", "thinkingConfig": {"thinkingLevel": "low"}}}
    r = None
    for _ in range(4):
        r = requests.post(GEMINI.format(MODEL), headers={"x-goog-api-key": gemini_key()}, json=body, timeout=300)
        if r.status_code == 200:
            break
        print(f"HTTP {r.status_code}: {r.text[:200]}", flush=True)
    if r is None or r.status_code != 200:
        raise RuntimeError("Gemini request failed")
    d = r.json()
    txt = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    return json.JSONDecoder().raw_decode(txt[txt.find("{"):])[0].get("results", {}), d.get("usageMetadata", {})


def label_image(item, conf):
    name, im = item["name"], load(item["name"])
    kept = [(i, b) for i, b in enumerate(item["boxes"]) if b[4] >= conf]
    rows, usage = [], {"in": 0, "out": 0}
    for k in range(0, len(kept), PER_REQUEST):
        chunk = kept[k:k + PER_REQUEST]
        got, u = ask(im, [b for _, b in chunk])
        usage["in"] += u.get("promptTokenCount", 0)
        usage["out"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        for n, (i, b) in enumerate(chunk):
            a = got.get(str(n)) or {}
            status = a.get("status") if a.get("status") in STATUS else "unsure"
            cat = str(a.get("cat", "")).zfill(3)
            if status == "ok" and cat not in TAX:
                status = "unsure"
            rows.append({"image": name, "box": i, "xyxy": b[:4], "det_conf": b[4], "status": status, "cat": cat if status == "ok" else None})
    return rows, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", type=int, default=200, help="画像を丸ごと、枠がこの数を超えるまで。0 は全部")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=WORK / "label_all.json")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    images = [x for x in json.loads((WORK / "det.json").read_text(encoding="utf-8"))["images"] if any(b[4] >= a.conf for b in x["boxes"])]
    random.Random(a.seed).shuffle(images)
    if a.boxes:
        picked, count = [], 0
        for x in images:
            if count >= a.boxes:
                break
            picked.append(x)
            count += sum(b[4] >= a.conf for b in x["boxes"])
        images = picked
    cache = a.out.with_suffix(".jsonl")
    rows = [json.loads(x) for x in cache.open(encoding="utf-8")] if cache.exists() else []
    done = {r["image"] for r in rows}
    images = [x for x in images if x["name"] not in done]
    usage = {"in": 0, "out": 0}
    with ThreadPoolExecutor(a.workers) as ex, cache.open("a", encoding="utf-8") as f:
        futures = [ex.submit(label_image, x, a.conf) for x in images]
        for n, future in enumerate(as_completed(futures), 1):
            try:
                got, u = future.result()
            except Exception as e:
                print(f"失敗: {e!r}", flush=True)
                continue
            rows += got
            usage = {k: usage[k] + u[k] for k in usage}
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in got)
            f.flush()
            if n % 25 == 0 or n == len(images):
                print(f"{n}/{len(images)} 画像", flush=True)
    a.out.write_text(json.dumps({"rows": rows}, ensure_ascii=False), encoding="utf-8")
    print(f"枠 {len(rows)} 個, 入力 {usage['in']} / 出力 {usage['out']} トークン, 約 ${usage['in'] / 1e6 * 0.5 + usage['out'] / 1e6 * 3:.3f}", flush=True)


if __name__ == "__main__":
    main()
