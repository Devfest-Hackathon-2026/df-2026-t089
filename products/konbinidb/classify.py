"""コンビニDB の商品を、商品名とサイトのカテゴリから 122 カテゴリ（taxonomy.json）に Gemini で振り分ける。

  .venv/Scripts/python products/konbinidb/classify.py [--sample 300] [--out labels.jsonl]
出力: products/work/konbinidb/labels.jsonl {"id", "cat"}。追記式なので続きから再開できる（同じ id は後の行が優先）。
"""
import argparse, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from s3_compare import GEMINI, gemini_key  # noqa: E402
from scrape import excluded  # noqa: E402

ROOT = HERE.parent / "work" / "konbinidb"
MODEL = "gemini-3-flash-preview"
BATCH = 80
TAX = {k: v for g in json.loads((HERE / "taxonomy.json").read_text(encoding="utf-8")).values() for k, v in g.items()}
CATALOG = "\n".join(f"{k}: {n}（{ex}）" for k, (n, ex) in TAX.items())
RULES = f"""コンビニの商品を、下のカテゴリ一覧のどれか1つに振り分けてください。番号で答えること。
- 商品の主な中身で分類する。「サイトの分類」は参考程度にし、名前と食い違うときは名前を優先する。
- 味の名前では分類しない（コーラ味のグミは 026、抹茶味のアイスはアイスのカテゴリ）。
- 温度帯より商品の種類を優先する。ただし、麺・米飯・惣菜の冷凍品は冷凍のカテゴリ（085・086・087）に入れる。
- ごはんとおかず、または主食 2 品のセットは 043。ただしお好み焼きと焼そばのセットは 108。
- カップ入りの即席麺は 061・062・064・075、袋の即席麺と乾麺は 065、調理済みのチルド麺は 055〜059。
- 090 にしてよいのは、調理済みでそのままおかずになる冷蔵の惣菜だけ。「その他のスナック菓子」023 や 090 を、別の種類の商品の受け皿にしない。
- 店頭で作るドリンクも中身で分ける（アイスコーヒーは 006、カフェラテは 007、抹茶ラテ・フラッペは 104）。
- 必ず一覧のどれか 1 つの番号にする。「該当なし」は使わない。

カテゴリ一覧:
{CATALOG}
"""


def ask(batch: list[dict], rules: str = RULES) -> dict[str, str]:
    items = {i: f"{x['name']}（サイトの分類: {x['category']}）" for i, x in enumerate(batch)}
    prompt = rules + f"\n商品（番号: 名前）: {json.dumps(items, ensure_ascii=False)}\n" + 'JSON だけを返す: {"assign": {"商品の番号": "カテゴリ番号", ...}}'
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}}
    for attempt in range(4):
        r = requests.post(GEMINI.format(MODEL), headers={"x-goog-api-key": gemini_key()}, json=body, timeout=300)
        if r.status_code == 200:
            break
        print(f"HTTP {r.status_code}（再試行 {attempt + 1}）", flush=True)
    else:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    txt = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    got = json.JSONDecoder().raw_decode(txt[txt.find("{"):])[0].get("assign", {})
    out = {}
    for i, c in got.items():
        c = str(c).zfill(3)
        if str(i).isdigit() and int(i) < len(batch) and c in TAX:
            out[batch[int(i)]["id"]] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="無作為に N 件だけ（お試し用）")
    ap.add_argument("--out", type=Path, default=ROOT / "labels.jsonl")
    a = ap.parse_args()
    items = [x for x in map(json.loads, (ROOT / "items.jsonl").open(encoding="utf-8")) if not excluded(x)]
    if a.sample:
        items = random.Random(0).sample(items, min(a.sample, len(items)))
    labels = {}
    if a.out.exists():
        for line in a.out.open(encoding="utf-8"):
            x = json.loads(line)
            labels[x["id"]] = x["cat"]
    todo = sorted((x for x in items if x["id"] not in labels), key=lambda x: x["category"])  # 似た商品を同じ回に
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(items)} 件 / 対象 {len(todo)} 件 / {len(batches)} 回", flush=True)
    with ThreadPoolExecutor(8) as ex, a.out.open("a", encoding="utf-8") as f:
        for n, got in enumerate(ex.map(ask, batches), 1):
            f.writelines(json.dumps({"id": k, "cat": v}) + "\n" for k, v in got.items())
            f.flush()
            if n % 20 == 0 or n == len(batches):
                print(f"{n}/{len(batches)}", flush=True)


if __name__ == "__main__":
    main()
