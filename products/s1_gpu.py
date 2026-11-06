"""① を Modal の GPU で実行する版。検出の中身は s1_detect.detect と同じで、結果も同じ work/det/*.json に書く。

  .venv/Scripts/modal run products/s1_gpu.py [--limit N]
処理済みの画像は飛ばす。送るのは長辺1920pxに縮めた画像。
"""
import io, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
MODEL = "google/owlv2-base-patch16-ensemble"
SEND_SIZE = 1920  # OWLv2 はタイルごとに960pxへ縮めるので、これ以上送っても結果は変わらない
CHUNK = 40
GPU = os.environ.get("OWL_GPU", "L4,A10G,T4,L40S").split(",")  # 先頭から空いている GPU を使う（L4 だけだと空き待ちで止まった）


def _download():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    Owlv2Processor.from_pretrained(MODEL)
    Owlv2ForObjectDetection.from_pretrained(MODEL)


image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "torchvision==0.29.2", "transformers==5.22.1", "pillow")
         .run_function(_download)  # 重みはイメージに焼き込み、起動のたびに落とさない
         .add_local_file(HERE / "common.py", "/root/common.py")
         .add_local_file(HERE / "s1_detect.py", "/root/s1_detect.py"))
app = modal.App("convenience-products-owlv2", image=image)


@app.cls(gpu=GPU, max_containers=4, timeout=1800, scaledown_window=60)
class Owl:
    @modal.enter()
    def load(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(MODEL).to("cuda").eval()

    @modal.method()
    def run(self, name: str, jpg: bytes):
        from PIL import Image
        from s1_detect import detect
        t = time.time()
        boxes, scores = detect(self.model, self.proc, Image.open(io.BytesIO(jpg)).convert("RGB"))
        return name, boxes, scores, round(time.time() - t, 2)


@app.local_entrypoint()
def main(limit: int = 0):
    from common import TRAIN_DIR, WORK, list_images, load_image, log, status, write_json
    from s1_detect import VIEW_SIZE

    imgs = list_images(TRAIN_DIR)
    todo = [p for p in imgs if not (WORK / "det" / f"{p.stem}.json").exists()]
    skipped = len(imgs) - len(todo)
    if limit:
        todo = todo[:limit]
    (WORK / "view").mkdir(exist_ok=True)
    meta = {}

    def prep(p):
        """元画像の大きさ（小さすぎる枠の判定に使う）を控え、表示用と送信用の縮小画像を作る"""
        im = load_image(p)
        meta[p.name] = (p, im.width, im.height)
        v = im.copy()
        v.thumbnail((VIEW_SIZE, VIEW_SIZE))
        v.save(WORK / "view" / f"{p.stem}.jpg", quality=85)
        im.thumbnail((SEND_SIZE, SEND_SIZE))
        b = io.BytesIO()
        im.save(b, "JPEG", quality=92)
        return p.name, b.getvalue()

    t0 = time.time()
    status("detect", state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"), error=None, note=f"Modal GPU ({'/'.join(GPU)}×最大4)")
    log("detect", f"GPU で {len(todo)} 枚を処理（{skipped} 枚は処理済み）", gpu=GPU, model=MODEL)
    done = skipped
    owl = Owl()
    # 回線が切れやすいので、CHUNK 枚ずつ送って結果をすぐ保存し、失敗した塊だけ送り直す
    with ThreadPoolExecutor(8) as ex:
        for k in range(0, len(todo), CHUNK):
            chunk = list(ex.map(prep, todo[k:k + CHUNK]))
            for attempt in range(3):
                try:
                    for name, boxes, scores, sec in owl.run.starmap(chunk, order_outputs=False):
                        p, w, h = meta[name]
                        write_json(WORK / "det" / f"{p.stem}.json", {"image": name, "src": p.parent.name, "w": w, "h": h,
                                                                     "boxes": boxes, "scores": scores, "sec": sec, "gpu": "/".join(GPU)})
                        done += 1
                        status("detect", done=done, total=len(imgs), last=name, sec_per_image=sec)
                    break
                except Exception as e:  # 通信エラー: 未保存の分だけもう一度
                    chunk = [c for c in chunk if not (WORK / "det" / f"{Path(c[0]).stem}.json").exists()]
                    log("detect", f"通信エラーで再送 ({attempt + 1}/3): 残り {len(chunk)} 枚", error=repr(e)[:200])
            log("detect", f"{done}/{len(imgs)} 枚", elapsed_s=round(time.time() - t0))
    status("detect", state="done" if done == len(imgs) else "error",
           error=None if done == len(imgs) else f"{len(imgs) - done} 枚が未処理（再実行すれば続きから）")
    log("detect", f"GPU 処理完了 {len(todo)} 枚 / {round(time.time() - t0)} 秒")
