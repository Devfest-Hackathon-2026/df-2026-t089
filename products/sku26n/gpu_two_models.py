"""YOLO26n-cls と MobileCLIP-S0 を Modal の GPU で学習し、店内の切り抜きを分類する部分。train_two_models.py から使う。"""
import json
import queue
import threading
import zipfile
from pathlib import Path

import modal

ONNX = Path(__file__).resolve().parents[1] / "work" / "models" / "mobileclip_s0_vision.onnx"
REMOTE_ONNX = "/root/mobileclip_s0_vision.onnx"
CLIP_SIZE = 256
STORE_WEIGHT = 10.0  # 店内の棚から切り出した画像の損失の重み（加工したコンビニDB の画像は 1）
volume = modal.Volume.from_name("convenience-products-two-models", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libglib2.0-0")
         .pip_install("ultralytics==8.4.202", "onnxruntime-gpu[cuda,cudnn]", "pillow", "numpy")
         .add_local_file(ONNX, REMOTE_ONNX))
app = modal.App("convenience-products-two-models", image=image)


def link_dataset(root: Path, manifest: list[dict]) -> Path:
    """ultralytics の分類データの形（train|val/カテゴリ/画像）にリンクを作る"""
    ds = root / "ds"
    for row in manifest:
        target = ds / row["split"] / row["cat"] / f"{row['source']}__{Path(row['path']).name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(root / row["path"])
    return ds


def load_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


def score(rows: list[dict], top5: list[list[str]]) -> dict:
    """店内の切り抜き（学習に使っていないもの）での正解率"""
    n = max(len(rows), 1)
    return {"n": len(rows), "top1": round(sum(r["cat"] == t[0] for r, t in zip(rows, top5)) / n, 4),
            "top5": round(sum(r["cat"] in t for r, t in zip(rows, top5)) / n, 4)}


def classify_boxes(view: Path, predict) -> "Iterator[dict]":
    """全画像の枠を分類する。predict(切り抜きのリスト) → [(カテゴリ, 確からしさ), ...]"""
    det = json.loads((view / "det.json").read_text())
    result, total = {}, 0
    for index, item in enumerate(det["images"], 1):
        boxes, out = item["boxes"], []
        im = load_rgb(view / "images" / item["name"])
        w, h = im.size
        for start in range(0, len(boxes), 128):
            chunk = boxes[start:start + 128]
            crops = []
            for b in chunk:
                x0, y0 = max(0, min(w - 1, int(b[0] * w))), max(0, min(h - 1, int(b[1] * h)))
                crops.append(im.crop((x0, y0, max(x0 + 1, min(w, int(b[2] * w))), max(y0 + 1, min(h, int(b[3] * h))))))
            out += [b[:5] + [cat, round(prob, 4)] for b, (cat, prob) in zip(chunk, predict(crops))]
        result[item["name"]] = out
        total += len(out)
        if index % 100 == 0:
            yield {"type": "progress", "images": index, "boxes": total}
    yield {"type": "pred", "json": json.dumps({"images": result}, separators=(",", ":"))}


def train_yolo(root, ds, epochs, imgsz, batch, q, errors, device=0, project="/tmp/runs"):
    import torch.nn.functional as F
    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify.train import ClassificationTrainer

    class WeightedDataset(ClassificationDataset):
        def __getitem__(self, i):
            sample = super().__getitem__(i)
            sample["weight"] = STORE_WEIGHT if Path(self.samples[i][0]).name.startswith("crop__") else 1.0
            return sample

    def weighted_loss(preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        weight = batch["weight"].to(preds.device, dtype=preds.dtype)
        loss = (weight * F.cross_entropy(preds, batch["cls"], reduction="none")).sum() / weight.sum()
        return loss, {"loss": loss.detach()}

    class WeightedTrainer(ClassificationTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            if mode != "train":
                return super().build_dataset(img_path, mode, batch)
            return WeightedDataset(img_path, self.args, augment=True, prefix="train", names=self.data["names"])

        def _setup_train(self, *args, **kwargs):
            super()._setup_train(*args, **kwargs)
            # EMA を作ったあとで差し替える。保存される EMA のモデルには標準の損失が残り、普通に読み込める
            self.model.criterion = weighted_loss

    def epoch_end(trainer):
        q.put({"type": "yolo_epoch", "epoch": trainer.epoch + 1, "metrics": {k: float(v) for k, v in trainer.metrics.items() if isinstance(v, (int, float))}})
    try:
        model = YOLO("yolo26n-cls.pt")
        model.add_callback("on_fit_epoch_end", epoch_end)
        # パッケージの文字が裏返らないよう左右反転は使わない
        model.train(trainer=WeightedTrainer, data=str(ds), epochs=epochs, imgsz=imgsz, batch=batch, device=device, workers=8 if device == 0 else 0, project=project, name="yolo",
                    exist_ok=True, plots=False, verbose=False, fliplr=0.0)
    except Exception as exc:
        errors.append(repr(exc))
    q.put(None)


@app.function(gpu=["L4", "A10G", "L40S", "T4"], volumes={"/data": volume}, timeout=4 * 3600, cpu=8, memory=32768)
def train_both(epochs: int, imgsz: int, batch: int, clip_epochs: int):
    import numpy as np
    import onnxruntime as ort
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from PIL import Image
    from ultralytics import YOLO

    root, view = Path("/tmp/data"), Path("/tmp/view")
    root.mkdir(), view.mkdir()
    with zipfile.ZipFile("/data/train.zip") as z:
        z.extractall(root)
    with zipfile.ZipFile("/data/view.zip") as z:
        z.extractall(view)
    manifest = json.loads((root / "manifest.json").read_text())
    ds = link_dataset(root, manifest)
    held_out = [r for r in manifest if r.get("metric")]
    yield {"type": "started", "images": len(manifest), "held_out": len(held_out), "classes": len({r["cat"] for r in manifest})}

    # ---- YOLO26n-cls
    q, errors = queue.Queue(), []
    threading.Thread(target=train_yolo, args=(root, ds, epochs, imgsz, batch, q, errors)).start()
    while (event := q.get()) is not None:
        yield event
    if errors:
        raise RuntimeError(errors[0])
    yolo = YOLO("/tmp/runs/yolo/weights/best.pt")

    def yolo_top(crops):
        return [p.probs for p in yolo.predict(source=crops, imgsz=imgsz, batch=len(crops), device=0, verbose=False)]

    top5 = []
    for k in range(0, len(held_out), 128):
        top5 += [[yolo.names[int(i)] for i in p.top5] for p in yolo_top([load_rgb(root / r["path"]) for r in held_out[k:k + 128]])]
    metrics = score(held_out, top5)
    yield {"type": "yolo_done", "metrics": metrics, "weights": Path("/tmp/runs/yolo/weights/best.pt").read_bytes(), "csv": Path("/tmp/runs/yolo/results.csv").read_text()}
    for event in classify_boxes(view, lambda crops: [(yolo.names[int(p.top1)], float(p.top1conf)) for p in yolo_top(crops)]):
        yield {**event, "model": "m1"}
    del yolo
    torch.cuda.empty_cache()

    # ---- MobileCLIP-S0（画像エンコーダーは固定し、分類層だけ学習する）
    try:
        ort.preload_dlls()
    except Exception:
        pass
    session = ort.InferenceSession(REMOTE_ONNX, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    yield {"type": "clip_started", "provider": session.get_providers()[0]}

    def embed(images):
        x = np.stack([np.asarray(i.resize((CLIP_SIZE, CLIP_SIZE), Image.BILINEAR), np.float32) / 255 for i in images]).transpose(0, 3, 1, 2)
        e = session.run(None, {input_name: x})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def embed_rows(rows):
        return np.concatenate([embed([load_rgb(root / r["path"]) for r in rows[k:k + 128]]) for k in range(0, len(rows), 128)])

    classes = sorted({r["cat"] for r in manifest})
    index = {c: i for i, c in enumerate(classes)}
    train_rows = [r for r in manifest if r["split"] == "train"]
    X, y = torch.from_numpy(embed_rows(train_rows)).cuda(), torch.tensor([index[r["cat"]] for r in train_rows]).cuda()
    X_val, y_val = torch.from_numpy(embed_rows(held_out)).cuda(), torch.tensor([index[r["cat"]] for r in held_out]).cuda()
    sample_weight = torch.tensor([STORE_WEIGHT if r["source"] == "crop" else 1.0 for r in train_rows], dtype=X.dtype, device=X.device)
    counts = torch.bincount(y, minlength=len(classes)).float().clamp(min=1)
    head = nn.Sequential(nn.Linear(X.shape[1], 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, len(classes))).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=clip_epochs * (len(X) // 512 + 1))
    weight = (counts.sum() / counts).sqrt()  # 商品が少ないカテゴリを無視しないように
    best, best_state = -1.0, None
    for epoch in range(1, clip_epochs + 1):
        head.train()
        perm = torch.randperm(len(X), device=X.device)
        for k in range(0, len(X), 512):
            idx = perm[k:k + 512]
            losses = F.cross_entropy(head(X[idx]), y[idx], weight=weight, reduction="none")
            loss = (sample_weight[idx] * losses).sum() / sample_weight[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        head.eval()
        with torch.no_grad():
            top1 = (head(X_val).argmax(1) == y_val).float().mean().item() if len(X_val) else 0.0
        if top1 > best:
            best, best_state = top1, {k: v.cpu().clone() for k, v in head.state_dict().items()}
        yield {"type": "clip_epoch", "epoch": epoch, "metrics": {"store_top1": round(top1, 4), "loss": round(loss.item(), 4)}}
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        ranked = head(X_val).topk(min(5, len(classes)), dim=1).indices.tolist()
    metrics = score(held_out, [[classes[i] for i in t] for t in ranked])
    import io
    buffer = io.BytesIO()
    torch.save({"state": best_state, "classes": classes, "dim": int(X.shape[1])}, buffer)
    yield {"type": "clip_done", "metrics": metrics, "head": buffer.getvalue()}

    def clip_predict(crops):
        with torch.no_grad():
            probs = head(torch.from_numpy(embed(crops)).cuda()).softmax(1)
        top = probs.max(1)
        return [(classes[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]

    for event in classify_boxes(view, clip_predict):
        yield {**event, "model": "clip"}
    yield {"type": "finished"}
