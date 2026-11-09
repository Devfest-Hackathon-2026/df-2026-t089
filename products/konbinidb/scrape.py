"""コンビニDB（konbinidb.com）の商品名・画像を集める。途中で止めても続きから再開できる。

  .venv/Scripts/python products/konbinidb/scrape.py list    [--max-pages N]  一覧ページから商品名・カテゴリ・画像 URL
  .venv/Scripts/python products/konbinidb/scrape.py images  [--limit N] [--chain セブンイレブン] [--per-category 100]  商品画像

相手のサイトに負担をかけないよう 1 リクエストずつ、間隔を 0.5 秒以上あける。
画像の権利は各社にあり、サイトは無断転載を禁じているので、出力は git に入らない products/work/konbinidb/ だけに置く。
"""
import argparse, html, json, random, re, sys, time, urllib.error, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "work" / "konbinidb"
ITEMS = ROOT / "items.jsonl"
LABELS = ROOT / "labels.jsonl"
PAGES = ROOT / "done_pages.txt"
IMAGES = ROOT / "images"
SITEMAPS = ROOT / "sitemaps"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; shelf-research/0.1)"}
INTERVAL = 0.5
CHAINS = ("セブンイレブン", "ナチュラルローソン", "ローソンストア100", "ローソン", "ファミリーマート", "ミニストップ")
_last = 0.0


def get(url: str) -> bytes | None:
    """404 は None。429/5xx は待って再試行し、403 か再試行切れは止める（ブロックされたら続けない）"""
    global _last
    for attempt in range(5):
        time.sleep(max(0.0, _last + INTERVAL - time.time()))
        _last = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 403:
                sys.exit(f"403 を受けたので停止します: {url}")
            if e.code != 429 and e.code < 500:
                raise
            print(f"HTTP {e.code}、{60 * (attempt + 1)} 秒待ちます", flush=True)
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"通信エラー {e}、{60 * (attempt + 1)} 秒待ちます", flush=True)
        time.sleep(60 * (attempt + 1))
    sys.exit(f"再試行しても取得できないので停止します: {url}")


def leaf_categories() -> list[str]:
    xml = (SITEMAPS / "product-category-all-sitemap.xml").read_text(encoding="utf-8")
    cats = [u for u in re.findall(r"<loc>([^<]+)</loc>", xml) if "/en/" not in u and "/ko/" not in u]
    return sorted(u for u in cats if not any(v != u and v.startswith(u) for v in cats))


def parse(page: str, cat: str) -> list[dict]:
    out = {}
    for card in page.split('<div class="p-entry-product">')[1:]:
        u = re.search(r'href="(https://konbinidb\.com/product/[^"]*/(\d+)/)"', card)
        name = re.search(r'p-entry-item__ttl"[^>]*>\s*(.*?)\s*</h3>', card, re.S)
        img = re.search(r'<img src="([^"]+)"', card)
        if not (u and name):
            continue
        chain = next((c for c in CHAINS if f">{c}</span>" in card), None)
        out[u[2]] = {"id": u[2], "url": u[1], "name": html.unescape(name[1]), "image": img[1] if img else None, "chain": chain, "category": cat}
    return list(out.values())


def stage_list(max_pages: int):
    ROOT.mkdir(parents=True, exist_ok=True)
    done = set(PAGES.read_text(encoding="utf-8").split()) if PAGES.exists() else set()
    seen = {json.loads(x)["id"] for x in ITEMS.open(encoding="utf-8")} if ITEMS.exists() else set()
    leaves = leaf_categories()
    print(f"末端カテゴリ {len(leaves)} / 取得済み {len(seen)} 件", flush=True)
    for n, base in enumerate(leaves, 1):
        cat = base.split("/product/")[1].strip("/")
        p = 1
        while True:
            url = base if p == 1 else f"{base}page/{p}/"
            if url in done:
                p += 1
                continue
            if max_pages and p > max_pages:
                break
            body = get(url)
            if body is None:
                break
            page = body.decode("utf-8", "ignore")
            items = [x for x in parse(page, cat) if x["id"] not in seen]
            with ITEMS.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in items)
            seen.update(x["id"] for x in items)
            with PAGES.open("a", encoding="utf-8") as f:
                f.write(url + "\n")
            has_next = 'rel="next"' in page
            print(f"[{n}/{len(leaves)}] {cat} p{p}: +{len(items)} 計 {len(seen)}", flush=True)
            if not has_next:
                break
            p += 1
    print(f"完了: {len(seen)} 件", flush=True)


def excluded(x: dict) -> bool:
    """店の棚に並ばない商品（キャラクターグッズ・くじ・オンライン限定）は分類も学習もしない"""
    return x["category"].rsplit("/", 1)[-1] in ("goods", "lottery") or "オンライン限定" in x["name"]


def capped(items: list[dict], n: int) -> list[dict]:
    """分類済みの商品を、カテゴリごとに n 件まで選ぶ。取得済みの画像を先に数え、残りは無作為（毎回同じ結果）"""
    labels = {}
    for line in LABELS.open(encoding="utf-8"):
        x = json.loads(line)
        labels[x["id"]] = x["cat"]
    by_cat: dict[str, list[dict]] = {}
    for x in items:
        cat = labels.get(x["id"])
        if cat and cat != "000" and not excluded(x):
            by_cat.setdefault(cat, []).append(x)
    rng, out = random.Random(0), []
    for _, xs in sorted(by_cat.items()):
        rng.shuffle(xs)
        xs.sort(key=lambda x: not (IMAGES / f"{x['id']}.webp").exists())
        out += xs[:n]
    return out


def stage_images(limit: int, chain: str | None, per_category: int):
    IMAGES.mkdir(parents=True, exist_ok=True)
    items = [x for x in map(json.loads, ITEMS.open(encoding="utf-8")) if not chain or x["chain"] == chain]
    if per_category:
        items = capped(items, per_category)
    todo = [x for x in items if x["image"] and not (IMAGES / f"{x['id']}.webp").exists()]
    if limit:
        todo = todo[:limit]
    print(f"画像 全 {len(items)} 件 / 未取得 {len(todo)} 件", flush=True)
    for i, x in enumerate(todo, 1):
        body = get(x["image"])
        if body:
            tmp = IMAGES / f"{x['id']}.tmp"
            tmp.write_bytes(body)
            tmp.replace(IMAGES / f"{x['id']}.webp")
        if i % 100 == 0 or i == len(todo):
            print(f"{i}/{len(todo)}", flush=True)
    print("完了", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["list", "images"])
    ap.add_argument("--max-pages", type=int, default=0, help="カテゴリごとのページ数の上限（動作確認用）")
    ap.add_argument("--limit", type=int, default=0, help="画像の枚数の上限（動作確認用）")
    ap.add_argument("--chain", help="画像を取るチェーン（例: セブンイレブン）。省略すると全チェーン")
    ap.add_argument("--per-category", type=int, default=0, help="カテゴリごとの画像の上限（classify.py の結果を使う）。0 は上限なし")
    a = ap.parse_args()
    stage_list(a.max_pages) if a.stage == "list" else stage_images(a.limit, a.chain, a.per_category)
