"""② 後半: 命名。既定は gemini-3-flash-preview で、2枚以上のグループの代表を命名して全メンバーに写す（--engine gemini）。
--engine vision: 未命名の合格（緑）の切り抜きを、Cloud Vision API の Web Detection（Google の逆画像検索）で1枚ずつ命名する。
検索結果が悪ければ命名しない（ダッシュボードでオレンジ）。名前は社名と商品名に分ける。

  .venv/Scripts/python products/s3_name.py [--n 50]
・1枚ずつ調べる。同じ社名・商品名になったものはダッシュボードで1つの商品にまとめて表示する
・キーは .env の GCP_API_KEY。生の結果は work/vision/<ID>.json に保存し、同じ切り抜きで二度課金しない
出力: work/names.json {切り抜きID: {maker, product, source: vision|manual, vision: good|bad, ...}}
"""
import argparse, base64, io, re, time

import requests

from common import ROOT, TRAIN_DIR, WORK, load_image, log, read_json, run_stage, status, write_json
from s2_crops import crop

URL = "https://vision.googleapis.com/v1/images:annotate"
FREE_PER_MONTH = 1000  # Web Detection の無料枠（月あたりの件数）
# 商品名として意味のない一般語（best guess がこれだけなら「結果が悪い」）
GENERIC = {"product", "products", "packaging and labeling", "packaging", "label", "brand", "bottle", "plastic bottle",
           "glass bottle", "convenience store", "store", "shelf", "retail", "supermarket", "cosmetics", "snack", "food",
           "drink", "box", "carton", "font", "logo", "advertising", "plastic", "paper", "personal care", "skin care",
           "beauty", "health", "liquid", "water", "juice", "tea", "beer", "sock", "socks", "textile", "material property",
           "rectangle", "japan", "japanese", "graphics", "design", "pattern", "illustration", "art", "image", "photograph",
           "medicine", "pharmaceutical drug", "household supply", "hygiene", "lipstick", "toothbrush", "cosmetic",
           "makeup", "comb", "hair care", "electric blue", "magenta", "pink", "red", "white", "black", "transparency",
           "convenience food", "flavor", "ingredient", "fast food", "candy", "confectionery", "pharmacy", "drugstore"}
# 社名の辞書（表記ゆれ → 正式名）。web 検索結果の文字列に出てきたものを社名にする
MAKERS = {"花王": ["花王", "kao"], "ライオン": ["ライオン", "lion corporation"], "資生堂": ["資生堂", "shiseido"],
          "P&G": ["p&g", "procter", "ピーアンドジー"], "ユニリーバ": ["ユニリーバ", "unilever"], "ユニ・チャーム": ["ユニ・チャーム", "ユニチャーム", "unicharm"],
          "小林製薬": ["小林製薬", "kobayashi"], "ロート製薬": ["ロート製薬", "rohto"], "大正製薬": ["大正製薬", "taisho"],
          "サントリー": ["サントリー", "suntory"], "アサヒ": ["アサヒ", "asahi"], "キリン": ["キリン", "kirin"],
          "サッポロ": ["サッポロ", "sapporo"], "伊藤園": ["伊藤園", "ito en"], "コカ・コーラ": ["コカ・コーラ", "coca-cola", "coca cola"],
          "ダイドー": ["ダイドー", "dydo"], "明治": ["明治", "meiji"], "森永": ["森永", "morinaga"], "ロッテ": ["ロッテ", "lotte"],
          "グリコ": ["グリコ", "glico"], "カルビー": ["カルビー", "calbee"], "湖池屋": ["湖池屋", "koikeya"], "日清": ["日清", "nissin"],
          "東洋水産": ["東洋水産", "マルちゃん", "maruchan"], "ブルボン": ["ブルボン", "bourbon"], "不二家": ["不二家", "fujiya"],
          "ハウス食品": ["ハウス食品", "house foods"], "味の素": ["味の素", "ajinomoto"], "キッコーマン": ["キッコーマン", "kikkoman"],
          "ヤクルト": ["ヤクルト", "yakult"], "雪印メグミルク": ["雪印", "megmilk"], "ファミリーマート": ["ファミリーマート", "familymart"],
          "セブン-イレブン": ["セブン-イレブン", "セブンイレブン", "7-eleven", "seven-eleven"], "ローソン": ["ローソン", "lawson"],
          "ジレット": ["gillette", "ジレット"], "ジョンソン・エンド・ジョンソン": ["johnson & johnson", "ジョンソン・エンド・ジョンソン", "band-aid"],
          "アース製薬": ["アース製薬", "earth corporation"], "エステー": ["エステー", "st corporation"], "コーセー": ["コーセー", "kose"],
          "マンダム": ["マンダム", "mandom"], "ピジョン": ["ピジョン", "pigeon"], "レッドブル": ["red bull", "レッドブル"],
          "モンスター": ["monster energy", "モンスターエナジー"], "ネスレ": ["ネスレ", "nestle", "nestlé"]}


def api_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "GCP_API_KEY":
            return v.strip().strip('"\'')
    raise SystemExit(".env に GCP_API_KEY がありません")


def crop_jpeg(c: dict) -> bytes:
    im = crop(load_image(TRAIN_DIR / c["image"]), c["box"], pad=0.08)
    im.thumbnail((640, 640))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=90)
    return b.getvalue()


def judge(r: dict):
    """Vision の結果から (good?, maker, product, 根拠)。根拠が弱い・一般語だけなら good=False"""
    w = r.get("webDetection", {})
    best = next((x.get("label", "") for x in w.get("bestGuessLabels", []) if x.get("label")), "")
    ents = [(e["description"], e.get("score", 0)) for e in w.get("webEntities", []) if e.get("description")]
    pages = [p.get("pageTitle", "") for p in w.get("pagesWithMatchingImages", [])]
    matches = len(w.get("fullMatchingImages", [])) + len(w.get("partialMatchingImages", []))
    specific = [d for d, s in ents if d.lower() not in GENERIC and s >= 0.5]
    name = best if best and best.lower() not in GENERIC else (specific[0] if specific else "")
    good = bool(name) and (matches > 0 or len(pages) > 0 or bool(specific))
    text = " ".join([best] + [d for d, _ in ents] + pages).lower()
    maker = next((m for m, al in MAKERS.items() if any(a in best.lower() for a in al)), None) \
        or next((m for m, al in MAKERS.items() if any(a in text for a in al)), "")
    product = name
    for a in MAKERS.get(maker, []):  # 商品名から社名の部分を取り除く
        product = re.sub(re.escape(a), "", product, flags=re.I).strip(" -・")
    return good, maker, product or name, {"best": best, "entities": ents[:6], "pages": pages[:3], "matches": matches,
                                          "error": r.get("error", {}).get("message")}


GEMINI_MODEL = "gemini-3-flash-preview"  # ②c の比較で選んだモデル
MIN_CONF = 0.5  # Gemini の確信度がこれ未満なら「結果が悪い」として命名しない


def main_gemini(n, workers=32):
    """2枚以上のグループの代表を Gemini で命名し、同じグループの全メンバーに写す。大きいグループから。
    無料枠の1日上限に当たったら止める（続きは再実行で）"""
    from concurrent.futures import ThreadPoolExecutor
    from s3_compare import m_gemini
    crops = {c["id"]: c for c in read_json(WORK / "crops.json", [])}
    groups = [g for g in read_json(WORK / "groups.json", {}).get("groups", []) if g["size"] >= 2]
    names = read_json(WORK / "names.json", {})
    todo = [g for g in groups if not any(m in names for m in g["members"]) and g["rep"] in crops][:n]
    log("name", f"{GEMINI_MODEL} で {len(groups)} グループ中 未命名 {len(todo)} グループを命名（今回の上限 {n}）")
    ask = m_gemini(GEMINI_MODEL)
    good = bad = 0
    cost = 0.0
    stop = []

    def one(g):
        if stop:
            return g, None
        try:
            return g, ask(crop_jpeg(crops[g["rep"]]))
        except Exception as e:
            msg = str(e)
            if "429" in msg and ("per day" in msg.lower() or "free tier" in msg.lower() or "free_tier" in msg):
                stop.append(msg)  # 1日の上限・無料枠: 以降は送らない（1分あたりの上限は m_gemini 内で待って再試行）
            return g, {"error": msg[:200]}
    done = 0
    with ThreadPoolExecutor(workers) as ex:  # 1枚 15秒前後かかるので並列で。1分あたりの上限に当たれば待って再試行
        for g, r in ex.map(one, todo):
            if r is None:
                continue
            done += 1
            if "error" in r:
                log("name", f"グループ #{g['id']}: エラー", error=r["error"])
                continue
            cost += r.get("cost", 0)
            conf = r.get("confidence")
            ok = bool(r["product"]) and (not isinstance(conf, (int, float)) or conf >= MIN_CONF)
            entry = {"maker": r["maker"] if ok else "", "product": r["product"] if ok else "", "source": "gemini",
                     "model": GEMINI_MODEL, "vision": "good" if ok else "bad",  # vision=bad → ② でオレンジ
                     "confidence": conf, "group": g["id"], "rep": g["rep"], "best": f'{r["maker"]} {r["product"]}'.strip()}
            for m in g["members"]:
                if names.get(m, {}).get("source") != "manual":  # 手で直した名前は上書きしない
                    names[m] = entry
            good, bad = good + ok, bad + (not ok)
            if done % 20 == 0:
                write_json(WORK / "names.json", names)
            status("name", done=done, total=len(todo), phase=f"{GEMINI_MODEL} で命名", cost_usd=round(cost, 4))
    write_json(WORK / "names.json", names)
    log("name", f"命名 {good} グループ / 結果が悪い {bad} グループ / 費用 約 ${cost:.3f}" + (" / 1日の上限で停止" if stop else ""),
        error=stop[0][:200] if stop else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="今回命名する数（Gemini はグループ数、Vision は切り抜き数）")
    ap.add_argument("--engine", default="gemini", choices=["gemini", "vision"])
    ap.add_argument("--workers", type=int, default=32, help="Gemini の同時リクエスト数")
    a = ap.parse_args()
    if a.engine == "gemini":
        return main_gemini(a.n, a.workers)
    key = api_key()
    crops = {c["id"]: c for c in read_json(WORK / "crops.json", [])}
    judge_ = read_json(WORK / "judge.json", {})
    names = read_json(WORK / "names.json", {})
    usage = read_json(WORK / "vision_usage.json", {})
    month = time.strftime("%Y-%m")
    (WORK / "vision").mkdir(exist_ok=True)
    # 対象: 合格（緑）で未命名・未試行。スコアの高い（はっきり写った）ものから
    todo = sorted((cid for cid, j in judge_.items() if j["ok"] and cid not in names and cid in crops),
                  key=lambda cid: -crops[cid]["score"])
    left = FREE_PER_MONTH - usage.get(month, 0)
    log("name", f"未命名の合格 {len(todo)} 個のうち最大 {a.n} 個を命名（今月の無料枠の残り {left} 件）")
    called = good_n = bad_n = 0
    k = 0
    while k < len(todo) and called < a.n:
        batch = todo[k:k + min(8, a.n - called)]
        k += len(batch)
        if usage.get(month, 0) + len(batch) > FREE_PER_MONTH:
            log("name", "今月の無料枠に達したので停止", used=usage.get(month, 0))
            break
        body = {"requests": [{"image": {"content": base64.b64encode(crop_jpeg(crops[c])).decode()},
                              "features": [{"type": "WEB_DETECTION", "maxResults": 10}],
                              "imageContext": {"webDetectionParams": {"includeGeoResults": False}}} for c in batch]}
        r = requests.post(URL, headers={"x-goog-api-key": key}, json=body, timeout=60)  # キーは URL に載せない
        if r.status_code != 200:
            raise RuntimeError(f"Vision API {r.status_code}: {r.text[:300]}")
        usage[month] = usage.get(month, 0) + len(batch)
        write_json(WORK / "vision_usage.json", usage)
        called += len(batch)
        for cid, res in zip(batch, r.json()["responses"]):
            write_json(WORK / "vision" / f"{cid}.json", res)
            ok, maker, product, why = judge(res)
            if ok:
                names[cid] = {"maker": maker, "product": product, "source": "vision", "vision": "good", **why}
                good_n += 1
            else:
                names[cid] = {"maker": "", "product": "", "source": "vision", "vision": "bad", **why}
                bad_n += 1
        write_json(WORK / "names.json", names)
        status("name", done=called, total=min(a.n, len(todo)), used_this_month=usage[month])
    write_json(WORK / "names.json", names)
    log("name", f"API {called} 件（良 {good_n} / 悪 {bad_n}）",
        used_this_month=usage.get(month, 0))


if __name__ == "__main__":
    run_stage("name", main)
