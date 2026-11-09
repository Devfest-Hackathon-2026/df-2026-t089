"""②d 商品名を商品カテゴリ（牛乳 / 飲むヨーグルト / プリン … 全体で約300）にまとめる。③b の学習・評価と ④ の推論・アプリはこのカテゴリを当てる。
画像は使わず、細分類（fine.json）ごとに商品名のテキストを Gemini に渡し、その細分類を分けるカテゴリを決めて各商品を振り分ける。

  .venv/Scripts/python products/s3c_category.py            # カテゴリ一覧が無い細分類は一覧から作る。ある細分類は一覧に振り分けるだけ
  .venv/Scripts/python products/s3c_category.py --redo     # カテゴリ一覧も作り直す
出力: products/categories.json {細分類: [カテゴリ, ...]}（一覧。コミットする。細かすぎた味・食感の分け方は手でまとめ直してある）
      work/category.json {正規化した商品名: カテゴリ}。手で直したものは work/category_override.json（ダッシュボードの ②d タブ）が優先
"""
import argparse, json
from concurrent.futures import ThreadPoolExecutor

import requests

from common import TAXONOMY, WORK, log, read_json, run_stage, status, write_json
from s3_compare import GEMINI, gemini_key
from s5_classify import norm

MODEL = "gemini-3-flash-preview"
TARGET = 300  # 全体のカテゴリ数の目安。細分類ごとに商品数に比例して割り振る
OTHER = "その他"  # 細分類が無い商品


def ask(fine, items, k, cats=None):
    if cats:
        task = f"次の商品を、カテゴリ一覧のどれか1つに振り分けてください。一覧に無い言葉は使わないこと。\nカテゴリ一覧: {json.dumps(cats, ensure_ascii=False)}\n"
    else:
        task = (f"次の商品（コンビニの「{fine}」）を、買い物客が探すときに使う商品カテゴリ {k} 個前後に分けてください。"
                "粒度の例: 牛乳、飲むヨーグルト、プリン、緑茶、ポテトチップス、ツナマヨおにぎり ではなく おにぎり（具は分けない）、"
                "メーカー名・味・容量では分けない。カテゴリ名は短い一般名詞（ほかの細分類と紛れない名前）。\n")
    prompt = (task + f"商品（番号: 名前）: {json.dumps({i: n for i, n in enumerate(items)}, ensure_ascii=False)}\n"
              'JSON だけを返す: {"categories": ["カテゴリ", ...], "assign": {"番号": "カテゴリ", ...}}')
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}}
    for attempt in range(3):
        r = requests.post(GEMINI.format(MODEL), headers={"x-goog-api-key": gemini_key()}, json=body, timeout=300)
        if r.status_code == 200:
            break
        log("category", f"{fine}: HTTP {r.status_code}（再試行 {attempt + 1}）")
    else:
        raise RuntimeError(f"{fine}: HTTP {r.status_code}: {r.text[:300]}")
    txt = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    d = json.JSONDecoder().raw_decode(txt[txt.find("{"):])[0]
    cats = cats or [c for c in d.get("categories", []) if isinstance(c, str) and c.strip()]
    got = {items[int(i)]: c for i, c in d.get("assign", {}).items() if str(i).isdigit() and int(i) < len(items) and c in cats}
    return cats, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", action="store_true")
    redo = ap.parse_args().redo
    names = read_json(WORK / "names.json", {})
    fine_of = read_json(WORK / "fine.json", {})
    disp = {}
    for v in names.values():
        if (v.get("maker") or v.get("product")) and v.get("vision") != "bad":
            disp[norm(v.get("maker", "") + v.get("product", ""))] = f'{v.get("maker", "")} {v.get("product", "")}'.strip()
    by_fine = {}
    for k in disp:
        by_fine.setdefault(fine_of.get(k, OTHER), []).append(k)
    tax = {} if redo else read_json(TAXONOMY, {})
    have = {} if redo else read_json(WORK / "category.json", {})
    jobs = []
    for f, keys in by_fine.items():
        todo = [k for k in keys if k not in have]
        if todo:
            jobs.append((f, todo, max(2, round(TARGET * len(keys) / len(disp)))))
    log("category", f"{len(disp)} 商品 / {len(by_fine)} 細分類。{sum(len(j[1]) for j in jobs)} 商品を振り分ける")

    def run(job):
        f, todo, k = job
        cats, got = tax.get(f), {}
        for b in range(0, len(todo), 300):  # 1回の出力が長くなりすぎないよう分ける。2回目以降は1回目の一覧に振り分ける
            cats, g = ask(f, [disp[x] for x in todo[b:b + 300]], k, cats)
            got.update(g)
        return f, cats, got

    done = 0
    with ThreadPoolExecutor(8) as ex:
        for f, cats, got in ex.map(run, jobs):
            tax[f] = cats
            back = {disp[x]: x for x in by_fine[f]}
            have.update({back[n]: c for n, c in got.items() if n in back})
            done += 1
            write_json(TAXONOMY, tax)
            write_json(WORK / "category.json", have)
            status("category", done=done, total=len(jobs))
    # 別の細分類に同じ名前のカテゴリがあれば同じカテゴリとして扱う（紅茶と乳飲料のミルクティーなど）
    used = {c for c in have.values()}
    log("category", f"カテゴリ付け {len(have)}/{len(disp)} 商品 → {len(used)} カテゴリ", missing=len(disp) - len(have))


if __name__ == "__main__":
    run_stage("category", main)
