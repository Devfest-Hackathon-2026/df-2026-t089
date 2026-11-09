"""商品名付きの切り抜き（同じ商品名の代表）を Gemini で判定する。元のラベルは見せず、社名・商品名・カテゴリを画像だけから答えさせる。
主役の商品を 1 種類に絞れないもの、商品を断定できないものは必ず除外する。読み取った名前が元のラベルと同じ商品のものだけ合格にする。

  .venv/Scripts/python products/sku26n/verify_gemini.py --limit 200
出力: products/work/sku26n/verify.json
"""
import argparse, json, random, re, sys, unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent), str(HERE.parent / "konbinidb")]
from dashboard import WORK, named_crops  # noqa: E402  products/dashboard.py と同名なので、他より先に読む
from classify import CATALOG, MODEL, TAX  # noqa: E402
from label_gemini import inline, load  # noqa: E402
from s3_compare import GEMINI, gemini_key  # noqa: E402

DET = HERE.parent / "work" / "det"
OUT = WORK / "verify.json"
PER_REQUEST = 10
CROP_SIDE = 768
PROMPT = f"""コンビニの棚の写真から切り出した商品画像を判定してください。それぞれの画像の前に番号があります。

status は次のどれか 1 つ:
- "ok": 主役の商品が 1 種類に絞れて、その商品が何かを断定できる。
- "multiple": 主役の商品を 1 種類に絞れない（別の種類の商品が同じくらい目立つ）。同じ商品が並んでいたり積み重なっていたりするのは問題なく、その商品が主役なら "ok" にする。縁に少し見切れているだけの隣の商品は無視してよい。
- "unsure": 商品が何かを断定できない（文字が読めない、一部しか写っていない、ぼやけている、商品でない）。

断定の基準:
- パッケージに読み取れる商品名やブランドがあるときだけ "ok" にする。色や形、棚の並びからの推測は絶対にしない。少しでも迷ったら "unsure" にする。
- "ok" のときだけ、maker に社名（例: サントリー）、product に商品名（例: 龍角散、綾鷹ほうじ茶）、cat に下のカテゴリ番号を入れる。社名がパッケージで読めなくても、商品名から確実に分かるなら入れてよい。分からなければ "unsure" にする。

カテゴリ一覧:
{CATALOG}

JSON だけを返す: {{"results": {{"番号": {{"status": "ok", "maker": "社名", "product": "商品名", "cat": "カテゴリ番号"}}, ...}}}}"""
MATCH_PROMPT = """コンビニ商品の名前の組が、同じ商品かどうかを判定してください。A は元のラベル、B は画像から読み取った名前です。
表記の違い、社名の有無、B に容量が書かれていないことは同じ商品として扱う。味・種類・シリーズが違うものは別の商品とする。
JSON だけを返す: {"same": {"番号": true, ...}}

"""


def post(body: dict) -> tuple[dict, dict]:
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
    return json.JSONDecoder().raw_decode(txt[txt.find("{"):])[0], d.get("usageMetadata", {})


def known(text) -> bool:
    """社名・商品名が本当に答えられているか（読めないときに "unsure" などと書かれることがある）"""
    return bool(text) and str(text).strip().lower() not in ("unsure", "multiple", "unknown", "不明", "なし", "n/a", "-")


def source_crop(cid: str) -> Image.Image:
    """元の写真から、切り抜きの枠で切り直す"""
    stem, i = cid.rsplit("_", 1)
    box = json.loads((DET / f"{stem}.json").read_text(encoding="utf-8"))["boxes"][int(i)]
    im = load(f"{stem}.jpg")
    return im.crop((int(box[0] * im.width), int(box[1] * im.height), int(box[2] * im.width), int(box[3] * im.height)))


def judge(rows: list) -> tuple[list[dict], dict]:
    parts: list[dict] = [{"text": PROMPT}]
    for i, r in enumerate(rows):
        parts += [{"text": f"画像 {i}:"}, inline(source_crop(r[0]), CROP_SIDE)]
    got, usage = post({"contents": [{"parts": parts}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "mediaResolution": "MEDIA_RESOLUTION_HIGH",
                                            "thinkingConfig": {"thinkingLevel": "low"}}})
    out = []
    for i, r in enumerate(rows):
        a = got.get("results", {}).get(str(i)) or {}
        status = a.get("status") if a.get("status") in ("ok", "multiple", "unsure") else "unsure"
        cat = str(a.get("cat", "")).zfill(3)
        if status == "ok" and not (known(a.get("maker")) and known(a.get("product")) and cat in TAX):
            status = "unsure"  # 3 つ揃わないものは断定できていない
        ok = status == "ok"
        out.append({"id": r[0], "label": r[1], "old_cat": r[2], "status": status, "maker": a.get("maker") if ok else None,
                    "product": a.get("product") if ok else None, "cat": cat if ok else None})
    return out, usage


def add_match(rows: list[dict]) -> dict:
    """読み取った名前が元のラベルと同じ商品か（ok のものだけ）"""
    targets = [r for r in rows if r["status"] == "ok"]
    usage = {"promptTokenCount": 0, "candidatesTokenCount": 0, "thoughtsTokenCount": 0}
    for k in range(0, len(targets), 100):
        chunk = targets[k:k + 100]
        pairs = {i: {"A": r["label"], "B": f"{r['maker']} {r['product']}"} for i, r in enumerate(chunk)}
        got, u = post({"contents": [{"parts": [{"text": MATCH_PROMPT + json.dumps(pairs, ensure_ascii=False)}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}})
        for key in usage:
            usage[key] += u.get(key, 0)
        for i, r in enumerate(chunk):
            r["same"] = got.get("same", {}).get(str(i)) is True
    return usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    a = ap.parse_args()
    first: dict[str, list] = {}
    for r in named_crops():
        first.setdefault(re.sub(r"\s", "", unicodedata.normalize("NFKC", r[1])).lower(), r)
    reps = list(first.values())
    rows = random.Random(0).sample(reps, min(a.limit, len(reps)))
    batches = [rows[i:i + PER_REQUEST] for i in range(0, len(rows), PER_REQUEST)]
    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(judge, batches))
    out = [x for batch, _ in results for x in batch]
    usages = [u for _, u in results] + [add_match(out)]
    for r in out:
        r["pass"] = r["status"] == "ok" and r.get("same") is True
    OUT.write_text(json.dumps({"rows": out}, ensure_ascii=False), encoding="utf-8")
    t_in = sum(u.get("promptTokenCount", 0) for u in usages)
    t_out = sum(u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0) for u in usages)
    print(f"reps {len(reps)}, judged {len(out)}, pass {sum(r['pass'] for r in out)}, in {t_in}, out {t_out}, ${t_in / 1e6 * 0.5 + t_out / 1e6 * 3:.3f}")


if __name__ == "__main__":
    main()
