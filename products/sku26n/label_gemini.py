"""sku110k-2 の切り抜きを、元の写真から切り直して Gemini で 120 カテゴリに分類する。

  .venv/Scripts/python products/sku26n/label_gemini.py --limit 3
出力: products/work/sku26n/pred_gemini.json（ダッシュボードの列）
"""
import argparse, base64, io, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "konbinidb")]
from classify import CATALOG, MODEL, TAX  # noqa: E402
from s3_compare import GEMINI, gemini_key  # noqa: E402

WORK = HERE.parent / "work" / "sku26n"
ORIGINALS = HERE.parents[1] / "training_images"
VIEW = HERE.parent / "work" / "view"
NOT_PRODUCT = "NP"
PER_REQUEST = 20
MARGIN = 0.15
CROP_SIDE = 512
SHELF_SIDE = 768
PROMPT = f"""コンビニの店内の写真から切り出した商品を分類してください。
最初の画像は棚全体です。続く画像が切り抜きで、それぞれの前に番号があります。

- 切り抜きの中央にある商品 1 つを、カテゴリ一覧のどれか 1 つの番号で答える。縁に写る隣の商品は無視する。
- 文字が読めなくても、形・容器・棚全体の様子から最も近いカテゴリを選ぶ。
- 値札・POP・棚・人など商品でないものは "{NOT_PRODUCT}" にする。
- conf は確からしさ（0〜1）。推測しかできないときは低くする。

カテゴリ一覧:
{CATALOG}

JSON だけを返す: {{"assign": {{"切り抜きの番号": {{"cat": "カテゴリ番号", "conf": 0.9}}, ...}}}}"""


def inline(im: Image.Image, side: int) -> dict:
    im = im.copy()
    im.thumbnail((side, side))
    b = io.BytesIO()
    im.convert("RGB").save(b, "JPEG", quality=88)
    return {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(b.getvalue()).decode()}}


def load(name: str) -> Image.Image:
    """元の写真（なければ表示用の 1280px）"""
    stem = Path(name).stem
    original = next((p for p in (ORIGINALS / f"{stem}.jpg", ORIGINALS / f"{stem}.JPG") if p.exists()), None)
    return ImageOps.exif_transpose(Image.open(original or VIEW / name)).convert("RGB")


def crop(im: Image.Image, b: list[float]) -> Image.Image:
    w, h = (b[2] - b[0]) * MARGIN, (b[3] - b[1]) * MARGIN
    return im.crop((int(max(0, b[0] - w) * im.width), int(max(0, b[1] - h) * im.height),
                    int(min(1, b[2] + w) * im.width), int(min(1, b[3] + h) * im.height)))


def ask(shelf: Image.Image, im: Image.Image, boxes: list[list[float]]) -> tuple[dict, dict]:
    parts = [{"text": PROMPT}, inline(shelf, SHELF_SIDE)]
    for i, b in enumerate(boxes):
        parts += [{"text": f"切り抜き {i}:"}, inline(crop(im, b), CROP_SIDE)]
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "mediaResolution": "MEDIA_RESOLUTION_MEDIUM",
                                 "thinkingConfig": {"thinkingLevel": "low"}}}
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
    return json.JSONDecoder().raw_decode(txt[txt.find("{"):])[0].get("assign", {}), d.get("usageMetadata", {})


def label(im_info: dict, conf: float) -> tuple[str, list, dict]:
    name = im_info["name"]
    kept = [b for b in im_info["boxes"] if b[4] >= conf]
    im = load(name)
    out, usage = [], {"in": 0, "out": 0}
    for k in range(0, len(kept), PER_REQUEST):
        chunk = kept[k:k + PER_REQUEST]
        got, u = ask(im, im, chunk)
        usage["in"] += u.get("promptTokenCount", 0)
        usage["out"] += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
        for i, b in enumerate(chunk):
            a = got.get(str(i)) or {}
            cat = str(a.get("cat", ""))
            cat = NOT_PRODUCT if cat == NOT_PRODUCT else cat.zfill(3) if cat.zfill(3) in TAX else None
            if cat:
                out.append(b[:5] + [cat, a.get("conf")])
    return name, out, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=3, help="先頭から N 枚")
    ap.add_argument("--conf", type=float, default=0.25, help="分類する枠の信頼度の下限")
    a = ap.parse_args()
    images = json.loads((WORK / "det.json").read_text(encoding="utf-8"))["images"][:a.limit]
    with ThreadPoolExecutor(3) as ex:
        results = list(ex.map(lambda x: label(x, a.conf), images))
    (WORK / "pred_gemini.json").write_text(json.dumps({"images": {n: o for n, o, _ in results}}, ensure_ascii=False), encoding="utf-8")
    t_in = sum(u["in"] for *_, u in results)
    t_out = sum(u["out"] for *_, u in results)
    print(f"枠 {sum(len(o) for _, o, _ in results)} 個, 入力 {t_in} / 出力 {t_out} トークン, 約 ${t_in / 1e6 * 0.5 + t_out / 1e6 * 3:.3f}")


if __name__ == "__main__":
    main()
