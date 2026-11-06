"""② 前半: ① の枠（しきい値以上）を全部切り抜き、ルールで「明らかに商品を特定できない」ものを保守的に落とし、
残りを MobileCLIP で埋め込む。

  .venv/Scripts/python products/s2_crops.py [--modal]   # --modal: 埋め込みを Modal の GPU で計算（CPU だと1枚20秒かかる）
出力: work/crops/<id>.jpg（表示用）, work/emb/<画像>.npz（埋め込みキャッシュ）,
      work/crops.json（全切り抜きと判定）, work/emb_all.npy + work/emb_ids.json（ルール合格分の埋め込み）
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageStat

from common import CLIP_ONNX, TRAIN_DIR, WORK, config, load_image, log, read_json, run_stage, status, write_json

CLIP_SIZE = 256
PAD = 0.04          # 枠の外側を少し含める（パッケージの縁まで）
MAX_ASPECT = 8      # これより細長い枠は棚の縁・値札レール
MIN_STD = 6         # 濃淡の標準偏差がこれ未満 = ほぼ無地
MIN_SHARP = 4       # ラプラシアンの分散がこれ未満 = 強いぼけ（保守的にかなり低め）


class Embedder:
    def __init__(self):
        if not CLIP_ONNX.exists():
            import shutil
            from huggingface_hub import hf_hub_download
            CLIP_ONNX.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download("Xenova/mobileclip_s0", "onnx/vision_model.onnx"), CLIP_ONNX)
        cuda = "CUDAExecutionProvider" in ort.get_available_providers()  # Modal の GPU 上 (gpu_clip.py) なら CUDA
        self.s = ort.InferenceSession(str(CLIP_ONNX), providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda
                                      else ["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, crops: list[Image.Image]) -> np.ndarray:
        """アプリ (ClipClassifier.kt) と同じ前処理: 256×256 に引き伸ばし、0..1 の RGB、NCHW"""
        x = np.stack([np.asarray(c.resize((CLIP_SIZE, CLIP_SIZE), Image.BILINEAR), np.float32) / 255 for c in crops])
        e = self.s.run(None, {self.inp: x.transpose(0, 3, 1, 2)})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def crop(im: Image.Image, box, pad=PAD) -> Image.Image:
    w, h = im.size
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    return im.crop((int(max(0, x0 - px) * w), int(max(0, y0 - py) * h), int(min(1, x1 + px) * w), int(min(1, y1 + py) * h)))


def accepted(d, stem, thr, rejects):
    """しきい値以上で、ダッシュボードで誤検出にされていない枠（YOLO の正解に使う）"""
    rej = set(rejects.get(stem, []))
    return [i for i, s in enumerate(d["scores"]) if s >= thr and i not in rej]


def rule(c: Image.Image, wpx, hpx, cfg):
    """明らかに特定不能な理由（無ければ None）。迷うものは残す"""
    if wpx < cfg["min_crop_w"] or hpx < cfg["min_crop_h"]:
        return f"小さすぎ {wpx}×{hpx}px"
    if max(wpx / hpx, hpx / wpx) > MAX_ASPECT:
        return "極端に細長い"
    g = c.convert("L")
    g.thumbnail((160, 160))
    if ImageStat.Stat(g).stddev[0] < MIN_STD:
        return "ほぼ無地"
    if ImageStat.Stat(g.filter(ImageFilter.FIND_EDGES)).var[0] < MIN_SHARP:
        return "強いぼけ"
    return None


def cut(p, cfg, rejects):
    """1画像ぶん: 切り抜き・サムネイル保存・ルール判定。埋め込みが足りない枠も返す（軽いので手元で並列に回す）"""
    d, stem = read_json(p), p.stem
    rej = set(rejects.get(stem, []))
    cache = WORK / "emb" / f"{stem}.npz"
    have = set(np.load(cache).files) if cache.exists() else set()
    im, rows, need = None, [], []
    for i, s in enumerate(d["scores"]):
        if s < cfg["det_threshold"]:
            continue
        x0, y0, x1, y1 = d["boxes"][i]
        wpx, hpx = round((x1 - x0) * d["w"]), round((y1 - y0) * d["h"])
        cid = f"{stem}_{i}"
        thumb = WORK / "crops" / f"{cid}.jpg"
        if thumb.exists():
            c = Image.open(thumb).convert("RGB")
        else:
            im = im or load_image(TRAIN_DIR / d["image"])
            c = crop(im, d["boxes"][i])
            c.thumbnail((160, 160))
            c.save(thumb, quality=85)
        reason = "手動で誤検出" if i in rej else rule(c, wpx, hpx, cfg)
        if reason is None and str(i) not in have:
            need.append((str(i), d["boxes"][i]))
        rows.append({"id": cid, "stem": stem, "image": d["image"], "box": d["boxes"][i], "score": s,
                     "w": wpx, "h": hpx, "rule": reason})
    return stem, d["image"], rows, need


def save_emb(stem, got: dict):
    cache = WORK / "emb" / f"{stem}.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    have.update(got)
    np.savez(cache, **have)


def embed_local(need, images):
    emb = Embedder()
    for n, (stem, items) in enumerate(need.items()):
        im = load_image(TRAIN_DIR / images[stem])
        cs = [crop(im, box) for _, box in items]
        E = np.concatenate([emb(cs[k:k + 32]) for k in range(0, len(cs), 32)])
        save_emb(stem, {i: v for (i, _), v in zip(items, E)})
        status("crops", done=n + 1, total=len(need), phase="埋め込み (CPU)")


def embed_modal(need, images):
    """足りない埋め込みを Modal で計算。40画像ずつ送って届いた順に保存し、通信エラーは未保存分だけ送り直す"""
    import io
    from concurrent.futures import ThreadPoolExecutor

    import modal
    from gpu_clip import Clip, app

    def prep(stem):
        im = load_image(TRAIN_DIR / images[stem])
        im.thumbnail((1920, 1920))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=92)
        return stem, b.getvalue(), need[stem]

    stems, saved, providers = list(need), set(), set()
    with modal.enable_output(), app.run(), ThreadPoolExecutor(8) as ex:
        clip = Clip()
        for k in range(0, len(stems), 40):
            chunk = list(ex.map(prep, stems[k:k + 40]))
            for attempt in range(3):
                try:
                    for stem, got, raw, prov in clip.run.starmap(chunk, order_outputs=False):
                        save_emb(stem, dict(zip(got, np.frombuffer(raw, np.float16).reshape(len(got), -1).astype(np.float32))))
                        saved.add(stem)
                        providers.add(prov)
                        status("crops", done=len(saved), total=len(stems), phase=f"埋め込み (Modal {prov})")
                    break
                except Exception as e:  # 通信エラー: 未保存の画像だけもう一度
                    chunk = [c for c in chunk if c[0] not in saved]
                    log("crops", f"通信エラーで再送 ({attempt + 1}/3): 残り {len(chunk)} 画像", error=repr(e)[:200])
    log("crops", f"Modal で {len(saved)}/{len(stems)} 画像を埋め込み", providers=sorted(providers))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modal", action="store_true", help="埋め込みを Modal（GPU）で計算する")
    a = ap.parse_args()
    cfg = config()
    rejects = read_json(WORK / "rejects.json", {})
    dets = sorted((WORK / "det").glob("*.json"))
    for sub in ("emb", "crops"):
        (WORK / sub).mkdir(exist_ok=True)
    # 1) 切り抜き＋ルール判定（手元・並列）。終わった時点で全件をテーブルに出す（埋め込み前は「未判定」）
    out, need, images = [], {}, {}
    with ThreadPoolExecutor(8) as ex:
        for n, (stem, image, rows, miss) in enumerate(ex.map(lambda p: cut(p, cfg, rejects), dets)):
            out += rows
            images[stem] = image
            if miss:
                need[stem] = miss
            status("crops", done=n + 1, total=len(dets), phase="切り抜き＋ルール判定")
    write_json(WORK / "crops.json", out)
    reasons = {}
    for o in out:
        if o["rule"]:
            reasons[o["rule"].split(" ")[0]] = reasons.get(o["rule"].split(" ")[0], 0) + 1
    log("crops", f"切り抜き {len(out)} 個（{len(dets)} 画像）/ ルールで除外 {sum(reasons.values())} 個 / 埋め込みが必要 {sum(map(len, need.values()))} 個",
        threshold=cfg["det_threshold"], reasons=reasons)
    # 2) 足りない埋め込みを計算
    if need:
        (embed_modal if a.modal else embed_local)(need, images)
    # 3) ルール合格分の埋め込みをまとめる（次の段階が使う）
    ids, vecs, missing = [], [], 0
    by = {}
    for o in out:
        if o["rule"] is None:
            by.setdefault(o["stem"], []).append(o["id"])
    for stem, cids in by.items():
        cache = WORK / "emb" / f"{stem}.npz"
        have = np.load(cache) if cache.exists() else None
        for cid in cids:
            i = cid.rsplit("_", 1)[1]
            if have is not None and i in have.files:
                ids.append(cid)
                vecs.append(have[i])
            else:
                missing += 1
    np.save(WORK / "emb_all.npy", np.asarray(vecs, np.float32).reshape(len(vecs), -1) if vecs else np.zeros((0, 512), np.float32))
    write_json(WORK / "emb_ids.json", ids)
    log("crops", f"埋め込み済み {len(ids)} 個" + (f"（{missing} 個は未計算。再実行で続きから）" if missing else ""))


if __name__ == "__main__":
    run_stage("crops", main)
