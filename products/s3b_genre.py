"""②c 命名済みの商品名にジャンル（棚の種類）を付ける。ジャンルは既存の棚認識アプリと同じ training_images/labels.csv の種類。
画像は使わず、商品名のテキストだけを 200件ずつ Gemini に渡す（安い）。済んだものは飛ばす。

  .venv/Scripts/python products/s3b_genre.py
  .venv/Scripts/python products/s3b_genre.py --level fine   # 細分類（茶 / コーヒー / 菓子パン / サンドイッチ …）
出力: work/genres.json {正規化した商品名: ジャンル}, work/fine.json {正規化した商品名: 細分類}
"""
import json
from concurrent.futures import ThreadPoolExecutor

import requests

from common import ROOT, WORK, log, read_json, run_stage, status, write_json
from s3_compare import GEMINI, gemini_key
from s5_classify import norm

MODEL = "gemini-3-flash-preview"
# 細分類: 同じジャンル（例: ソフトドリンク）の中でも棚で分かれて並ぶ単位（茶とコーヒー、菓子パンとサンドイッチ…）
FINE = ["緑茶", "麦茶・ほうじ茶・その他のお茶", "紅茶", "コーヒー", "水", "炭酸飲料", "果汁・野菜ジュース", "スポーツ・機能性飲料",
        "乳飲料・ヨーグルト飲料", "豆乳", "エナジードリンク", "栄養ドリンク", "ビール類", "チューハイ・サワー・ハイボール", "日本酒・焼酎",
        "ワイン・洋酒", "菓子パン", "食パン・調理パン", "サンドイッチ", "おにぎり", "弁当", "惣菜・サラダ", "カップ麺", "袋麺",
        "レトルト・缶詰・インスタント", "調味料", "冷凍食品", "アイス", "チルドデザート・ヨーグルト", "和菓子", "洋菓子・ケーキ",
        "チョコレート", "ビスケット・クッキー", "ポテトチップス・スナック", "せんべい・米菓", "グミ", "キャンディ・ガム・タブレット",
        "ナッツ・おつまみ・珍味", "駄菓子・食玩", "シリアル・栄養補助食品", "メイク化粧品", "スキンケア", "ヘアケア・ボディケア",
        "オーラルケア", "ティッシュ・生理用品・紙製品", "マスク", "医薬品・救急用品", "洗剤・日用雑貨", "文房具", "電池・スマホ用品・雑貨",
        "衣料・靴下", "ギフトカード・その他"]


def genre_list():
    cats = []
    for line in (ROOT / "training_images" / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
        g = line.split(",", 1)[1].strip() if "," in line else ""
        if g and g not in cats and g != "shelf":
            cats.append(g)
    return cats


def ask(items, cats):
    prompt = ("次のコンビニ商品を、ジャンル一覧のどれか1つに分類してください。一覧に無い言葉は使わないこと。\n"
              f"ジャンル一覧: {json.dumps(cats, ensure_ascii=False)}\n"
              f"商品（番号: 名前）: {json.dumps({i: n for i, n in enumerate(items)}, ensure_ascii=False)}\n"
              'JSON だけを返す: {"番号": "ジャンル", ...}')
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingLevel": "low"}}}
    r = requests.post(GEMINI.format(MODEL), headers={"x-goog-api-key": gemini_key()}, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.json().get('error', {}).get('message', '')[:300]}")
    txt = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
    d = json.JSONDecoder().raw_decode(txt[txt.find("{"):])[0]
    return {items[int(k)]: v for k, v in d.items() if str(k).isdigit() and int(k) < len(items) and v in cats}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="genre", choices=["genre", "fine"], help="genre=棚の26種類 / fine=細分類")
    level = ap.parse_args().level
    cats = genre_list() if level == "genre" else FINE
    out = WORK / ("genres.json" if level == "genre" else "fine.json")
    names = read_json(WORK / "names.json", {})
    disp = {}
    for v in names.values():
        if (v.get("maker") or v.get("product")) and v.get("vision") != "bad":
            disp[norm(v.get("maker", "") + v.get("product", ""))] = f'{v.get("maker", "")} {v.get("product", "")}'.strip()
    have = read_json(out, {})
    todo = [k for k in disp if k not in have]
    log("genre", f"{len(disp)} 商品中 {len(todo)} 商品に {level} を付ける", cats=len(cats))
    batches = [todo[k:k + 200] for k in range(0, len(todo), 200)]
    done = 0
    with ThreadPoolExecutor(8) as ex:
        for res in ex.map(lambda b: (b, ask([disp[k] for k in b], cats)), batches):
            b, got = res
            back = {disp[k]: k for k in b}
            have.update({back[n]: g for n, g in got.items() if n in back})
            done += 1
            write_json(out, have)
            status("genre", done=done, total=len(batches))
    cnt = {}
    for g in have.values():
        cnt[g] = cnt.get(g, 0) + 1
    log("genre", f"{level} 付け {len(have)}/{len(disp)} 商品", counts=dict(sorted(cnt.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    run_stage("genre", main)
