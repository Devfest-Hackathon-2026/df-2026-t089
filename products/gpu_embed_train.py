"""③b 商品の埋め込みモデル（MobileNetV3-small, 128px → 128次元）を Modal の GPU で学習する部分。s4b_embed.py から使う。
同じ商品が近く・違う商品が遠くなるよう ArcFace 型の損失で学習し、ONNX（正規化込み、入力 0..1 RGB）で返す。
"""
import io

import modal

GPU = ["L4", "A10G", "L40S", "T4"]
SIZE = 128
vol = modal.Volume.from_name("convenience-products-embed", create_if_missing=True)


def _download():
    import torchvision
    torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")


image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.15.0", "torchvision==0.29.2", "pillow", "numpy", "onnx", "onnxscript")
         .run_function(_download))
app = modal.App("convenience-products-embed", image=image)


@app.function(gpu=GPU, volumes={"/data": vol}, timeout=3600, cpu=8, memory=16384)
def train(epochs: int = 30, dim: int = 128):
    """/data/train.pkl = {"jpgs": [bytes], "labels": [int], "n_classes": int} → エポックごとの記録を返し、最後に ONNX"""
    import pickle
    import numpy as np
    import torch, torch.nn as nn, torch.nn.functional as F
    import torchvision
    from PIL import Image
    d = pickle.load(open("/data/train.pkl", "rb"))
    imgs = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in d["jpgs"]]  # 正方形 144px（手元で作成済み）
    y = torch.tensor(d["labels"])
    dev = "cuda"

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            m = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
            self.features, self.pool = m.features, nn.AdaptiveAvgPool2d(1)
            self.head = nn.Sequential(nn.Linear(576, 512), nn.Hardswish(), nn.Dropout(0.3), nn.Linear(512, dim))
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):  # x: 0..1 RGB（アプリ・推論側はこれだけ渡せばよい）
            x = (x - self.mean) / self.std
            return F.normalize(self.head(self.pool(self.features(x)).flatten(1)), dim=1)

    net = Net().to(dev)
    W = nn.Parameter(torch.randn(d["n_classes"], dim, device=dev) * 0.01)  # 商品ごとの中心
    # 過学習対策で重み減衰を強く（特徴抽出部だけ学習率を下げると学習不足で正解率が下がった）
    opt = torch.optim.AdamW([*net.parameters(), W], lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * (len(imgs) // 256 + 1))
    s, margin = 30.0, 0.2
    X = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()  # uint8 N×3×144×144

    def augment(xb):
        """位置・大きさのずれ、明るさ・コントラスト、一部を隠す。色相は変えない（色違い＝別商品を見分けたいので）"""
        n = xb.shape[0]
        out = torch.empty(n, 3, SIZE, SIZE, device=dev)
        for i in range(n):
            sc = np.random.uniform(0.8, 1.0)
            w = int(144 * sc)
            x0, y0 = np.random.randint(0, 145 - w, 2)
            out[i] = F.interpolate(xb[i:i + 1, :, y0:y0 + w, x0:x0 + w].float(), size=SIZE, mode="bilinear", align_corners=False)[0]
        out = out / 255
        for i in range(n):  # 一部を隠す（パッケージの1か所だけで覚えないように）
            if np.random.rand() < 0.5:
                h, w = np.random.randint(16, 56, 2)
                y0, x0 = np.random.randint(0, SIZE - h), np.random.randint(0, SIZE - w)
                out[i, :, y0:y0 + h, x0:x0 + w] = torch.rand(3, 1, 1, device=dev)
        b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
        c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
        m = out.mean((1, 2, 3), keepdim=True)
        return ((out - m) * c + m * b).clamp(0, 1)

    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        tot = cor = 0
        loss_sum = 0.0
        for k in range(0, len(X), 256):
            idx = perm[k:k + 256]
            xb, yb = augment(X[idx].to(dev)), y[idx].to(dev)
            e = net(xb)
            cos = e @ F.normalize(W, dim=1).T
            # ArcFace: 正解クラスの角度に margin を足してから scale 倍
            th = torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
            logits = s * torch.where(F.one_hot(yb, d["n_classes"]).bool(), torch.cos(th + margin), cos)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(idx)
            cor += (cos.argmax(1) == yb).sum().item()
            tot += len(idx)
        yield {"epoch": ep + 1, "loss": round(loss_sum / tot, 4), "train_acc": round(cor / tot, 4)}

    net.eval().cpu()
    torch.onnx.export(net, (torch.rand(2, 3, SIZE, SIZE),), "/tmp/embed.onnx", input_names=["pixels"], output_names=["embedding"],
                      dynamic_axes={"pixels": {0: "n"}, "embedding": {0: "n"}}, opset_version=17, dynamo=False)
    yield {"onnx": open("/tmp/embed.onnx", "rb").read()}
