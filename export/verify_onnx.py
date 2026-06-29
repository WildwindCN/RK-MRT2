"""ONNX 模型导出 + 验证一体化

创建模型 → 固定权重 → 导出 ONNX → 对比输出验证精度
"""
import os
import sys
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import DepthFormerConfig, SpectroStreamConfig
from models.depthformer import DepthBodyAR
from models.spectrostream import SpectroStreamDecoder


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def verify_model(name, pt_model, example_input, input_names, output_names,
                 dynamic_axes, output_dir, rtol=1e-4):
    """导出并验证单个模型"""
    set_seed(42)
    print(f"\n[{name}]")

    # PyTorch 输出
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(example_input)
    if isinstance(pt_out, tuple):
        pt_outs = [o.detach().numpy() for o in pt_out]
    else:
        pt_outs = [pt_out.detach().numpy()]

    # ONNX 导出
    path = os.path.join(output_dir, f"{name}.onnx")
    torch.onnx.export(
        pt_model, example_input, path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )

    # 结构验证
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    print(f"  Nodes: {len(onnx_model.graph.node)}")

    # ONNX Runtime 推理
    if isinstance(example_input, tuple):
        ort_inputs = {name: t.numpy() for name, t in zip(input_names, example_input)}
    else:
        ort_inputs = {input_names[0]: example_input.numpy()}

    session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    ort_outs = session.run(None, ort_inputs)

    # 对比
    for i, (pt, ort_o) in enumerate(zip(pt_outs, ort_outs)):
        diff = np.abs(pt - ort_o)
        max_d = diff.max()
        mean_d = diff.mean()
        pt_f = pt.reshape(-1)
        ort_f = ort_o.reshape(-1)
        cos = np.dot(pt_f, ort_f) / (np.linalg.norm(pt_f) * np.linalg.norm(ort_f))
        status = "PASS" if cos > 0.9999 else "FAIL"
        print(f"  Out[{i}] {pt.shape}: max_diff={max_d:.2e}, mean_diff={mean_d:.2e}, cos={cos:.8f} [{status}]")
        if cos < 0.9999:
            raise RuntimeError(f"Verification failed for {name} output {i}: cos={cos}")

    return path


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "exported")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("RK-MRT2 ONNX Export & Verification")
    print("=" * 60)

    # 1. Depth Body
    cfg = DepthFormerConfig()
    db = DepthBodyAR(cfg)
    B, Q, D = 2, cfg.num_codebooks, cfg.temporal_spec.model_dims
    x_db = torch.randn(B, Q, D)
    verify_model("depth_body", db, x_db,
                 ["x"], ["logits"],
                 {"x": {0: "batch"}, "logits": {0: "batch"}},
                 output_dir)

    # 2. Codec Decoder
    scfg = SpectroStreamConfig()
    decoder = SpectroStreamDecoder(scfg)
    T = 25
    x_dec = torch.randn(1, T, scfg.embedding_dim)
    verify_model("codec_decoder", decoder, x_dec,
                 ["embeddings"], ["stft_features"],
                 {"embeddings": {0: "batch", 1: "time"},
                  "stft_features": {0: "batch", 3: "stft_time"}},
                 output_dir)

    print("\n" + "=" * 60)
    print("All models exported and verified!")
    print("=" * 60)


if __name__ == "__main__":
    main()
