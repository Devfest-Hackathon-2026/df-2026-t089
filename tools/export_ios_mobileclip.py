"""iOS 用に、MobileCLIP-S0 の入力を 1 枚に固定したモデルを書き出す。

  .venv/Scripts/python tools/export_ios_mobileclip.py

app/src/main/assets/products/mobileclip_s0.onnx は、Android 版が切り抜きを最大 8 枚まとめて入れるので、
枚数（batch_size）が決まっていない。iOS 版は Core ML（Neural Engine・GPU）に計算を任せるが、形の決まっていない
モデルは Core ML に渡せず（onlyAllowStaticInputShapes）、全部 CPU で動いて ONNX Runtime の作業用メモリが増え続ける。
そこで枚数を 1 に固定し、形から計算していた部分（Shape → Gather → Reshape など）を定数に畳んで ios/Models/ に書き出す。

  ios/Models/mobileclip_s0.onnx  入力 [1, 3, 256, 256]、出力 [1, 512]。元のモデルと出力が同じことを確かめてから書く

必要なもの: pip install onnx onnxruntime
"""
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "app/src/main/assets/products/mobileclip_s0.onnx"
OUT = ROOT / "ios/Models/mobileclip_s0.onnx"


def main() -> None:
    m = onnx.load(str(SRC))
    for v in list(m.graph.input) + list(m.graph.output):
        d = v.type.tensor_type.shape.dim[0]
        d.ClearField("dim_param")
        d.dim_value = 1
    del m.graph.value_info[:]
    m = onnx.shape_inference.infer_shapes(m)
    with tempfile.TemporaryDirectory() as tmp:
        fixed = Path(tmp) / "fixed.onnx"
        folded = Path(tmp) / "folded.onnx"
        onnx.save(m, str(fixed))
        # 定数の畳み込みだけ（BASIC）。これより上の最適化は ONNX Runtime 専用の演算子を入れるので使わない
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        so.optimized_model_filepath = str(folded)
        ort.InferenceSession(str(fixed), so, providers=["CPUExecutionProvider"])
        out = onnx.load(str(folded))
    # 書き出しのときに付く、使っていない演算子の集まりの宣言を外す
    used = {n.domain for n in out.graph.node}
    keep = [o for o in out.opset_import if o.domain in used or o.domain == ""]
    del out.opset_import[:]
    out.opset_import.extend(keep)
    onnx.checker.check_model(out)
    dynamic = [n.op_type for n in out.graph.node if n.op_type in ("Shape", "Gather", "ConstantOfShape")]
    assert not dynamic, f"形から計算する演算子が残っている: {dynamic}"

    a = ort.InferenceSession(str(SRC), providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(out.SerializeToString(), providers=["CPUExecutionProvider"])
    name = a.get_inputs()[0].name
    for seed in range(3):
        x = np.random.default_rng(seed).random((1, 3, 256, 256), dtype=np.float32)
        diff = float(np.abs(a.run(None, {name: x})[0] - b.run(None, {name: x})[0]).max())
        assert diff < 1e-5, f"元のモデルと出力が違う（最大 {diff}）"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(out, str(OUT))
    print(f"{OUT.relative_to(ROOT)}: {OUT.stat().st_size / 1e6:.1f}MB, {len(out.graph.node)} nodes")


if __name__ == "__main__":
    main()
