"""商品認識パイプラインの共通部分。各段階は work/ に結果・状態・ログを書き、ダッシュボードはそれを読むだけ。"""
import json, os, sys, threading, time
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "training_images"
TEST_DIR = ROOT / "test_images"
WORK = ROOT / "products" / "work"
# MobileCLIP-S0 の画像側。Apple の研究用ライセンスなので git に入れず、アプリにも同梱しない（無ければ s2_crops.Embedder が取ってくる）
CLIP_ONNX = WORK / "models" / "mobileclip_s0_vision.onnx"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
WORK.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")  # 同時に書く他プロセスと一時ファイルが衝突しないように
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    for k in range(20):  # ダッシュボードが書きかけを読まないよう置き換える。Windows は読み中だと失敗するので少し待って再試行
        try:
            return tmp.replace(path)
        except PermissionError:
            time.sleep(0.05 * (k + 1))
    tmp.replace(path)


def read_json(path: Path, default=None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def log(stage: str, msg: str, **kw):
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "stage": stage, "msg": msg, **kw}
    print(f"[{stage}] {msg}", kw if kw else "", flush=True)
    with open(WORK / "log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def status(stage: str, **kw):
    """work/status.json[stage] を更新（state: running/done/error, done/total など）"""
    p = WORK / "status.json"
    try:
        s = read_json(p, {})
    except OSError as e:  # 他プロセスが置き換え中だと読めない。進捗表示は失敗しても本体の処理を止めない
        print(f"[status] 読み込み失敗（無視）: {e!r}", flush=True)
        return
    prev = s.get(stage, {})
    cur = {**prev, **kw, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    # 終了予定の計算用: 今のフェーズ（total か phase が変わるまで）の開始時刻と、その時点の done を覚える
    if "done" in kw and (kw.get("total", prev.get("total")) != prev.get("total") or kw.get("phase", prev.get("phase")) != prev.get("phase")
                         or "t0" not in prev or kw.get("state") == "running"):
        cur["t0"], cur["done0"] = time.time(), kw["done"]
    if kw.get("state") == "running" and "done" not in kw:  # 段階の開始: 前回の進捗を消す
        for k in ("done", "total", "phase", "t0", "done0"):
            cur.pop(k, None)
    s[stage] = cur
    try:
        write_json(p, s)
    except OSError as e:  # 進捗表示は失敗しても本体の処理を止めない（他プロセスが読み中だと Windows で置き換えに失敗する）
        print(f"[status] 書き込み失敗（無視）: {e!r}", flush=True)


def config() -> dict:
    """ダッシュボードから変えられる設定（しきい値など）"""
    # min_crop_w/h: 元画像でこのピクセル数未満の枠は、商品を見分けられないので切り抜き・名前付けに使わない
    # group_sim: ② で同じ商品とみなす DINOv2 の類似度 / group_color: 色ヒストグラムの類似度（0 で色を見ない）
    return {"det_threshold": 0.18, "min_crop_w": 20, "min_crop_h": 20, "group_sim": 0.8, "group_color": 0.8,
            **read_json(WORK / "config.json", {})}


def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def load_image(p: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def run_stage(stage: str, fn):
    """段階の開始・終了・例外を status とログに残す"""
    status(stage, state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), error=None)
    log(stage, "開始", argv=sys.argv[1:])
    try:
        fn()
    except BaseException as e:
        status(stage, state="error", error=repr(e))
        log(stage, f"エラー: {e!r}")
        raise
    status(stage, state="done")
    log(stage, "完了")


TAXONOMY = ROOT / "products" / "categories.json"  # 細分類 → 商品カテゴリの一覧（s3c_category.py が作る。写真を含まないのでコミットする）


def category_of() -> dict:
    """正規化した商品名 → 商品カテゴリ（Gemini の振り分けに、ダッシュボードで手で直したものを上書き）"""
    return {**read_json(WORK / "category.json", {}), **read_json(WORK / "category_override.json", {})}


def crop_labels() -> dict:
    """学習・評価・照合先に使う切り抜き → 商品カテゴリ。命名済み（Vision の結果が悪いものを除く）でカテゴリが付いたものだけ"""
    import unicodedata
    cat = category_of()
    excluded = set(read_json(WORK / "label_exclude.json", []))  # ダッシュボードの教師データ確認で外した切り抜き
    out = {}
    for cid, v in read_json(WORK / "names.json", {}).items():
        if (v.get("maker") or v.get("product")) and v.get("vision") != "bad" and cid not in excluded:
            c = cat.get(unicodedata.normalize("NFKC", v.get("maker", "") + v.get("product", "")).replace(" ", "").replace("　", "").lower())
            if c:
                out[cid] = c
    return out


def category_levels() -> tuple[dict, dict]:
    """周辺補正用: カテゴリ → 粗いジャンル, カテゴリ → 細分類（どちらもそのカテゴリの商品の多数決。tools/export_products.py も同じ）"""
    cat = category_of()

    def vote(level: dict) -> dict:
        votes = {}
        for k, c in cat.items():
            if k in level:
                votes.setdefault(c, {}).setdefault(level[k], 0)
                votes[c][level[k]] += 1
        return {c: max(v, key=v.get) for c, v in votes.items()}
    return vote(read_json(WORK / "genres.json", {})), vote(read_json(WORK / "fine.json", {}))
