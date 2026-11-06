"""②c 命名方法の比較実験: 同じ切り抜き N 枚を、いくつかの方法で「社名・商品名」に命名して並べる。
正誤はダッシュボードで人が付ける（work/naming_compare/eval.json）。

  .venv/Scripts/python products/s3_compare.py [--n 20] [--methods vision_web,gemini-3-flash-preview,...]
出力: work/naming_compare/sample.json（対象の切り抜き。2回目以降も同じものを使う）
      work/naming_compare/<方法>.json（結果。2回目以降は未実行とエラーの切り抜きだけやり直す。--redo で全部取り直し）
"""
import argparse, base64, json, random, time
from concurrent.futures import ThreadPoolExecutor

import requests

from common import ROOT, WORK, log, read_json, run_stage, status, write_json
from s3_name import MAKERS, URL as VISION_URL, api_key, crop_jpeg, judge as judge_web

OUT = WORK / "naming_compare"
GEMINI = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
# 料金（USD / 100万トークン, 入力・出力）。https://ai.google.dev/pricing を 2026-11 に確認
PRICE = {"gemini-3.8-flash": (0.75, 3.75), "gemini-3.6-flash": (0.75, 3.75), "gemini-3.5-flash-lite": (0.30, 2.50),
         "gemini-3.1-flash-lite": (0.25, 1.50), "gemini-3-flash-preview": (0.50, 3.00)}
VISION_PRICE = 3.5 / 1000  # Cloud Vision 1機能1枚あたり（無料枠 1000件/月 を超えた場合）
PROMPT = ("この画像はコンビニの棚の商品1つを切り抜いたものです。パッケージに見える文字・ロゴ・デザインから、"
          "メーカー（社名）と商品名を日本で売られている正式な表記で答えてください。味・フレーバー・容量の違いも商品名に含めてください。"
          "確信が持てない場合は推測せず空文字にしてください。"
          'JSON だけを返す: {"maker": "社名", "product": "商品名", "confidence": 0〜1}')
METHODS = ["vision_web", "vision_text_logo", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3-flash-preview",
           "gemini-3.8-flash", "gemini-3-flash-preview+search", "gemma-4-26b-a4b-it", "gemma-4-31b-it"]


def gemini_key():
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "GEMINI_API_KEY":
            return v.strip().strip('"\'')
    raise SystemExit(".env に GEMINI_API_KEY がありません")


def pick_sample(n):
    """2枚以上のグループの代表のうち、80px 以上で見分けやすいものから n 枚（グループが無ければ合格から）"""
    crops = {c["id"]: c for c in read_json(WORK / "crops.json", [])}
    judge = read_json(WORK / "judge.json", {})
    big = lambda i: i in crops and crops[i]["w"] >= 80 and crops[i]["h"] >= 80 and judge.get(i, {}).get("ok")
    reps = [g["rep"] for g in read_json(WORK / "groups.json", {}).get("groups", []) if big(g["rep"])]
    pool = reps or [i for i in crops if big(i)]
    random.Random(0).shuffle(pool)
    return [{"id": i, "image": crops[i]["image"], "box": crops[i]["box"]} for i in pool[:n]]


def vision(jpg: bytes, features):
    r = requests.post(VISION_URL, headers={"x-goog-api-key": api_key()}, timeout=60, json={"requests": [{
        "image": {"content": base64.b64encode(jpg).decode()}, "features": [{"type": f, "maxResults": 10} for f in features]}]})
    if r.status_code != 200:  # raise_for_status は URL（= API キー入り）をエラー文に含めるので使わない
        raise RuntimeError(f"HTTP {r.status_code}: {r.json().get('error', {}).get('message', '')[:150]}")
    return r.json()["responses"][0]


def m_vision_web(jpg):
    raw = vision(jpg, ["WEB_DETECTION"])
    ok, maker, product, why = judge_web(raw)
    return {"maker": maker if ok else "", "product": product if ok else "", "note": f"best guess: {why['best']}" + ("" if ok else "（結果が悪いので不採用）"),
            "cost": VISION_PRICE}


def m_vision_text_logo(jpg):
    """文字認識（OCR）とロゴ検出から組み立てる: 社名=ロゴか社名辞書に当たった語、商品名=大きい文字の上位2行"""
    raw = vision(jpg, ["TEXT_DETECTION", "LOGO_DETECTION"])
    logo = next((x["description"] for x in raw.get("logoAnnotations", [])), "")
    words = raw.get("textAnnotations", [])[1:]
    text = raw.get("textAnnotations", [{}])[0].get("description", "") if raw.get("textAnnotations") else ""

    def height(w):
        ys = [v.get("y", 0) for v in w["boundingPoly"]["vertices"]]
        return max(ys) - min(ys)
    big = sorted(words, key=height, reverse=True)[:6]  # 大きい文字ほど商品名らしい
    product = " ".join(dict.fromkeys(w["description"] for w in big))
    maker = logo or next((m for m, al in MAKERS.items() if any(a in text.lower() for a in al)), "")
    return {"maker": maker, "product": product, "note": "OCR: " + text.replace("\n", " / ")[:120], "cost": 2 * VISION_PRICE}


def m_gemini(model, search=False):
    def run(jpg):
        body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpg).decode()}},
                                        {"text": PROMPT}]}],
                "generationConfig": {"temperature": 0.1, "thinkingConfig": {"thinkingLevel": "low"}}}
        if model.startswith("gemma"):  # Gemma は思考設定・JSON モード非対応。本文から JSON を拾う
            body["generationConfig"] = {"temperature": 0.1}
        elif search:
            body["tools"] = [{"google_search": {}}]  # 検索で裏付け（JSON モード指定とは併用できないので本文から JSON を拾う）
        else:
            body["generationConfig"]["responseMimeType"] = "application/json"
        r = None
        for attempt in range(4):
            r = requests.post(GEMINI.format(model), headers={"x-goog-api-key": gemini_key()}, json=body, timeout=120)  # キーは URL に載せない（エラー文に URL が出るため）
            if r.status_code not in (429, 500, 503):
                break
            time.sleep(4 * (attempt + 1))  # 混雑・一時的な上限は少し待って再試行
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.json().get('error', {}).get('message', '')[:600]}")
        j = r.json()
        txt = "".join(p.get("text", "") for p in j["candidates"][0]["content"]["parts"])
        k = txt.find("{")  # JSON の後ろに文字が付くことがあるので、先頭の JSON だけを読む
        d = json.JSONDecoder().raw_decode(txt[k:])[0] if k >= 0 else {}
        d = d if isinstance(d, dict) else {}
        u = j.get("usageMetadata", {})
        pin, pout = PRICE.get(model, (0, 0))
        cost = (u.get("promptTokenCount", 0) * pin + (u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)) * pout) / 1e6
        if search:
            cost += 35 / 1000  # 検索グラウンディング 1リクエストあたり（無料枠超過時の目安）
        return {"maker": d.get("maker", "") or "", "product": d.get("product", "") or "",
                "note": f"confidence {d.get('confidence', '-')}", "confidence": d.get("confidence"), "cost": cost}
    return run


def method(name):
    if name == "vision_web":
        return m_vision_web
    if name == "vision_text_logo":
        return m_vision_text_logo
    if name.endswith("+search"):
        return m_gemini(name[:-7], search=True)
    return m_gemini(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--redo", action="store_true", help="済んだ方法も取り直す")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sample = read_json(OUT / "sample.json")
    if not sample or len(sample) != a.n:
        sample = pick_sample(a.n)
        write_json(OUT / "sample.json", sample)
    with ThreadPoolExecutor(8) as ex:
        jpgs = dict(zip([s["id"] for s in sample], ex.map(crop_jpeg, sample)))
    methods = [m for m in a.methods.split(",") if m]
    log("compare", f"{len(sample)} 枚を {len(methods)} 方法で命名", methods=methods)
    for k, name in enumerate(methods):
        old = {} if a.redo else read_json(OUT / f"{name}.json", {}).get("results", {})
        todo = [s for s in sample if s["id"] not in old or "error" in old[s["id"]]]  # 未実行とエラーだけやり直す
        if not todo:
            continue
        fn = method(name)

        def one(s):
            t = time.time()
            try:
                return s["id"], {**fn(jpgs[s["id"]]), "sec": round(time.time() - t, 2)}
            except Exception as e:
                return s["id"], {"maker": "", "product": "", "error": str(e)[:200], "sec": round(time.time() - t, 2), "cost": 0}
        with ThreadPoolExecutor(5) as ex:
            res = {**old, **dict(ex.map(one, todo))}
        res = {s["id"]: res[s["id"]] for s in sample}
        errs = sum("error" in r for r in res.values())
        write_json(OUT / f"{name}.json", {"method": name, "results": res, "errors": errs,
                                          "sec_mean": round(sum(r["sec"] for r in res.values()) / len(res), 2),
                                          "cost_total": round(sum(r.get("cost", 0) for r in res.values()), 5)})
        status("compare", done=k + 1, total=len(methods), phase=name)
        log("compare", f"{name}: 命名 {sum(bool(r['maker'] or r['product']) for r in res.values())}/{len(res)} 枚, エラー {errs}",
            sec_mean=round(sum(r['sec'] for r in res.values()) / len(res), 2))


if __name__ == "__main__":
    run_stage("compare", main)
