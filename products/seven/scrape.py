"""セブン‐イレブン公式サイト（www.sej.co.jp）の商品ページと商品画像を集める。途中で止めても続きから再開できる。

  .venv/Scripts/python products/seven/scrape.py all [--max-pages N]   一覧 → 商品ページ → 画像

相手のサイトに負担をかけないよう 1 リクエストずつ、間隔をあける。ブロックされたら（403 やボット対策の確認ページ）止まる。
画像の権利はセブン‐イレブン等にあるので、出力は git に入らない products/work/seven/ だけに置き、公開しない。
"""
import argparse, html, json, re, sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "konbinidb"))
import scrape as http  # noqa: E402  konbinidb と同じ、間隔あけと再試行つきの get

ROOT = Path(__file__).resolve().parents[1] / "work" / "seven"
BASE = "https://www.sej.co.jp"
VISITED = ROOT / "visited.txt"
FRONTIER = ROOT / "frontier.json"
CODES = ROOT / "codes.jsonl"
DETAILS = ROOT / "items.jsonl"
IMAGES = ROOT / "images"
PAGE_INTERVAL = 1.0
IMAGE_INTERVAL = 0.5
START = "/products/a/"
SEEDS = ["/products/", "/sitemap.html"]
REPEATED_VIEW = re.compile(r"/l(50|100)/")  # 15 件ずつのページと同じ内容なのでたどらない


def fetch_page(path: str) -> str | None:
    http.INTERVAL = PAGE_INTERVAL
    body = http.get(BASE + path)
    if body is None:
        return None
    if len(body) < 5000 and b"Incapsula" in body:  # 確認ページは突破しない
        sys.exit(f"ボット対策の確認ページが出たので停止します: {path}")
    return body.decode("utf-8", "ignore")


def links(page: str) -> tuple[list[str], set[str]]:
    """一覧ページ → (たどる一覧ページ, 商品コード)"""
    found, codes = [], set()
    for href in re.findall(r'href="([^"#?]+)', page):
        href = re.sub(r"/{2,}", "/", href.replace(BASE, ""))
        if not href.startswith(START):
            continue
        m = re.match(r"/products/a/item/(\w+)/", href)
        if m:
            codes.add(m[1])
        elif not href.endswith((".css", ".js")) and not REPEATED_VIEW.search(href):
            found.append(href)
    return found, codes


def stage_list(max_pages: int):
    ROOT.mkdir(parents=True, exist_ok=True)
    visited = set(VISITED.read_text(encoding="utf-8").split()) if VISITED.exists() else set()
    seen = {json.loads(x)["code"] for x in CODES.open(encoding="utf-8")} if CODES.exists() else set()
    frontier = deque(p for p in (json.loads(FRONTIER.read_text(encoding="utf-8")) if FRONTIER.exists() else SEEDS + [START]) if not REPEATED_VIEW.search(p))
    queued = set(frontier) | visited
    n = 0
    while frontier and not (max_pages and n >= max_pages):
        path = frontier.popleft()
        page = fetch_page(path)
        n += 1
        with VISITED.open("a", encoding="utf-8") as f:
            f.write(path + "\n")
        visited.add(path)
        if page is not None:
            found, codes = links(page)
            with CODES.open("a", encoding="utf-8") as f:
                f.writelines(json.dumps({"code": c, "listing": path}) + "\n" for c in sorted(codes - seen))
            seen |= codes
            for p in found:
                if p not in queued:
                    queued.add(p)
                    frontier.append(p)
        if n % 25 == 0 or not frontier:
            FRONTIER.write_text(json.dumps(list(frontier)), encoding="utf-8")
            print(f"一覧 {len(visited)} ページ / 残り {len(frontier)} / 商品 {len(seen)}", flush=True)
    FRONTIER.write_text(json.dumps(list(frontier)), encoding="utf-8")
    return not frontier


def stage_items():
    done = {json.loads(x)["code"] for x in DETAILS.open(encoding="utf-8")} if DETAILS.exists() else set()
    todo = [x for x in map(json.loads, CODES.open(encoding="utf-8")) if x["code"] not in done]
    print(f"商品ページ 未取得 {len(todo)}", flush=True)
    for i, x in enumerate(todo, 1):
        page = fetch_page(f"/products/a/item/{x['code']}/")
        name = re.search(r"<h1>(.*?)</h1>", page or "", re.S)
        images = list(dict.fromkeys(re.findall(r'https://img-afd[^"\']+/item-image/[^"\']+', page or "")))
        with DETAILS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({**x, "name": html.unescape(name[1]).strip() if name else None, "images": images}, ensure_ascii=False) + "\n")
        if i % 100 == 0 or i == len(todo):
            print(f"商品ページ {i}/{len(todo)}", flush=True)


def stage_images():
    IMAGES.mkdir(parents=True, exist_ok=True)
    http.INTERVAL = IMAGE_INTERVAL
    todo = [(x["code"], n, u) for x in map(json.loads, DETAILS.open(encoding="utf-8")) for n, u in enumerate(x["images"])]
    todo = [t for t in todo if not (IMAGES / f"{t[0]}_{t[1]}.jpg").exists()]
    print(f"画像 未取得 {len(todo)}", flush=True)
    for i, (code, n, url) in enumerate(todo, 1):
        body = http.get(url)
        if body:
            tmp = IMAGES / f"{code}_{n}.tmp"
            tmp.write_bytes(body)
            tmp.replace(IMAGES / f"{code}_{n}.jpg")
        if i % 100 == 0 or i == len(todo):
            print(f"画像 {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["all", "list"])
    ap.add_argument("--max-pages", type=int, default=0, help="一覧のページ数の上限（動作確認用）")
    a = ap.parse_args()
    finished = stage_list(a.max_pages)
    if a.stage == "all" and finished:
        stage_items()
        stage_images()
        print("完了", flush=True)
