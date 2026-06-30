"""Temporal Body ONNX 数值验证

对比 PyTorch vs ONNX Runtime 输出:
1. 单帧推理 (随机权重)
2. 多帧 stateful 循环 (KV cache 传播)
3. 真实权重验证 (如果 checkpoint 可用)

用法:
    python export/verify_temporal_onnx.py                          # 随机权重
    python export/verify_temporal_onnx.py --weights <path.pt>     # 真实权重
"""
import os
import sys
import argparse
import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import DepthFormerConfig
from models.depthformer import TemporalBodyStateful


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def cosine_similarity(a, b):
    a_f = a.reshape(-1).astype(np.float64)
    b_f = b.reshape(-1).astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f)))


def trim_kv(pt_kv, max_len):
    """Trim PyTorch KV cache to max_len (matching ONNX export wrapper logic)"""
    sk, sv = pt_kv
    if sk.shape[2] > max_len:
        sk = sk[:, :, -max_len:, :]
        sv = sv[:, :, -max_len:, :]
    return (sk, sv)


def verify_single_frame(model, onnx_path, cfg, max_kv_len, T_cond):
    """单帧推理验证: 对比 PyTorch 和 ONNX Runtime 输出"""
    print("\n" + "=" * 60)
    print("Test 1: Single-Frame Inference")
    print("=" * 60)

    num_layers = cfg.temporal_spec.num_layers
    num_heads = cfg.temporal_spec.num_heads
    dim_per_head = cfg.temporal_spec.dim_per_head
    model_dim = cfg.temporal_spec.model_dims
    encoder_dim = cfg.encoder_spec.model_dims

    set_seed(123)
    x = torch.randn(1, 1, model_dim)
    cond = torch.randn(1, T_cond, encoder_dim)

    # 初始化 KV caches
    kv_caches = []
    for _ in range(num_layers):
        kv_caches.append({
            "self_kv": (
                torch.zeros(1, num_heads, max_kv_len, dim_per_head),
                torch.zeros(1, num_heads, max_kv_len, dim_per_head),
            ),
            "cross_kv": None,
        })

    # PyTorch 推理
    model.eval()
    with torch.no_grad():
        pt_out, pt_new_caches = model(x, cond, kv_caches)

    # ONNX Runtime 推理
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {
        "x": x.numpy(),
        "cond": cond.numpy(),
    }
    for i in range(num_layers):
        ort_inputs[f"self_k_{i}"] = kv_caches[i]["self_kv"][0].numpy()
        ort_inputs[f"self_v_{i}"] = kv_caches[i]["self_kv"][1].numpy()

    ort_outputs = session.run(None, ort_inputs)
    ort_out = ort_outputs[0]

    # 对比输出
    cos = cosine_similarity(pt_out.numpy(), ort_out)
    print(f"  Output cos={cos:.8f}  {'[PASS]' if cos > 0.9999 else '[FAIL]'}")

    # 对比 KV caches (trim PyTorch to match ONNX max_len)
    all_pass = cos > 0.9999
    for i in range(num_layers):
        pt_sk, pt_sv = trim_kv(pt_new_caches[i]["self_kv"], max_kv_len)
        ort_k = ort_outputs[1 + i * 2]
        ort_v = ort_outputs[1 + i * 2 + 1]
        cos_k = cosine_similarity(pt_sk.numpy(), ort_k)
        cos_v = cosine_similarity(pt_sv.numpy(), ort_v)
        layer_pass = cos_k > 0.9999 and cos_v > 0.9999
        if not layer_pass:
            all_pass = False
        if i < 3 or not layer_pass:
            print(f"  Layer[{i}] K cos={cos_k:.8f} V cos={cos_v:.8f}  [{'PASS' if layer_pass else 'FAIL'}]")

    return all_pass


def verify_stateful_loop(model, onnx_path, cfg, max_kv_len, T_cond, num_frames=20):
    """多帧 stateful 循环验证: 逐帧对比 PyTorch 和 ONNX Runtime"""
    print("\n" + "=" * 60)
    print(f"Test 2: Stateful Loop ({num_frames} frames)")
    print("=" * 60)

    num_layers = cfg.temporal_spec.num_layers
    num_heads = cfg.temporal_spec.num_heads
    dim_per_head = cfg.temporal_spec.dim_per_head
    model_dim = cfg.temporal_spec.model_dims
    encoder_dim = cfg.encoder_spec.model_dims

    # 固定输入序列 (确保可复现)
    set_seed(456)
    x_frames = [torch.randn(1, 1, model_dim) for _ in range(num_frames)]
    cond = torch.randn(1, T_cond, encoder_dim)

    # 初始化 KV caches (随机初始, PyTorch 和 ONNX 相同)
    set_seed(789)
    pt_kv = []
    ort_kv_inputs = {"cond": cond.numpy()}
    for i in range(num_layers):
        k = torch.randn(1, num_heads, max_kv_len, dim_per_head) * 0.01
        v = torch.randn(1, num_heads, max_kv_len, dim_per_head) * 0.01
        pt_kv.append({
            "self_kv": (k.clone(), v.clone()),
            "cross_kv": None,
        })
        ort_kv_inputs[f"self_k_{i}"] = k.numpy()
        ort_kv_inputs[f"self_v_{i}"] = v.numpy()

    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    model.eval()

    max_cos_drift = 0.0
    min_cos = 1.0
    all_pass = True

    for frame_idx in range(num_frames):
        x = x_frames[frame_idx]

        # PyTorch 推理
        with torch.no_grad():
            pt_out, pt_new_kv = model(x, cond, pt_kv)

        # ONNX Runtime 推理
        ort_kv_inputs["x"] = x.numpy()
        ort_outputs = session.run(None, ort_kv_inputs)

        # 对比输出
        cos = cosine_similarity(pt_out.numpy(), ort_outputs[0])
        min_cos = min(min_cos, cos)
        drift = abs(1.0 - cos)
        max_cos_drift = max(max_cos_drift, drift)

        if cos < 0.9999 and all_pass:
            all_pass = False
            print(f"  Frame {frame_idx}: cos={cos:.8f}  [FAIL] — drift detected!")
        elif frame_idx < 5 or frame_idx == num_frames - 1:
            print(f"  Frame {frame_idx}: cos={cos:.8f}  [{'PASS' if cos > 0.9999 else 'FAIL'}]")

        # 更新 PyTorch KV caches (trim to match ONNX export wrapper)
        pt_kv = [{"self_kv": trim_kv(c["self_kv"], max_kv_len), "cross_kv": c["cross_kv"]}
                 for c in pt_new_kv]

        # 更新 ONNX KV inputs
        for i in range(num_layers):
            ort_kv_inputs[f"self_k_{i}"] = ort_outputs[1 + i * 2]
            ort_kv_inputs[f"self_v_{i}"] = ort_outputs[1 + i * 2 + 1]

    print(f"\n  Min cosine: {min_cos:.8f}, Max drift: {max_cos_drift:.2e}")
    print(f"  Result: {'[PASS]' if all_pass else '[FAIL]'}")

    return all_pass


def verify_cross_attention(model, onnx_path, cfg, max_kv_len, T_cond):
    """交叉注意力验证: 确认 cond → cross-attn 计算在 ONNX 中正确"""
    print("\n" + "=" * 60)
    print("Test 3: Cross-Attention Correctness")
    print("=" * 60)

    num_layers = cfg.temporal_spec.num_layers
    num_heads = cfg.temporal_spec.num_heads
    dim_per_head = cfg.temporal_spec.dim_per_head
    model_dim = cfg.temporal_spec.model_dims
    encoder_dim = cfg.encoder_spec.model_dims

    # 使用非零 cond 和不同的输入
    set_seed(111)
    x = torch.randn(1, 1, model_dim)
    cond_a = torch.randn(1, T_cond, encoder_dim)
    cond_b = torch.randn(1, T_cond, encoder_dim)  # 不同的 cond

    kv_caches = []
    for _ in range(num_layers):
        kv_caches.append({
            "self_kv": (
                torch.zeros(1, num_heads, max_kv_len, dim_per_head),
                torch.zeros(1, num_heads, max_kv_len, dim_per_head),
            ),
            "cross_kv": None,
        })

    # PyTorch — cond_a
    with torch.no_grad():
        pt_out_a, _ = model(x, cond_a, kv_caches)
        pt_out_b, _ = model(x, cond_b, kv_caches)

    # 确认不同 cond 产生不同输出
    pt_diff = (pt_out_a - pt_out_b).abs().max().item()
    print(f"  PyTorch max_diff(cond_a, cond_b): {pt_diff:.6f}")
    assert pt_diff > 0.001, "Cross-attention not working: cond doesn't affect output"
    print(f"  [PASS] Different conditions produce different outputs")

    # ONNX Runtime — 验证 ONNX 输出与 PyTorch 一致
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_inputs = {"x": x.numpy(), "cond": cond_a.numpy()}
    for i in range(num_layers):
        ort_inputs[f"self_k_{i}"] = kv_caches[i]["self_kv"][0].numpy()
        ort_inputs[f"self_v_{i}"] = kv_caches[i]["self_kv"][1].numpy()

    ort_outputs = session.run(None, ort_inputs)
    cos = cosine_similarity(pt_out_a.numpy(), ort_outputs[0])
    print(f"  ONNX vs PyTorch (cond_a) cos={cos:.8f}  [{'PASS' if cos > 0.9999 else 'FAIL'}]")

    return cos > 0.9999


def main():
    parser = argparse.ArgumentParser(description="Verify Temporal Body ONNX")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to converted PyTorch weights (.pt)")
    parser.add_argument("--onnx_path", type=str, default=None,
                        help="Path to existing ONNX file (skip export if provided)")
    parser.add_argument("--output_dir", default="./exported")
    parser.add_argument("--max_kv_len", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Number of frames for stateful loop test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = DepthFormerConfig()
    T_cond = 50

    print("=" * 60)
    print("Temporal Body ONNX Verification")
    print("=" * 60)
    print(f"  Layers: {cfg.temporal_spec.num_layers}")
    print(f"  Dim: {cfg.temporal_spec.model_dims}")
    print(f"  Heads: {cfg.temporal_spec.num_heads}")
    print(f"  Max KV len: {args.max_kv_len}")
    print(f"  Weights: {args.weights or 'random (init)'}")

    # 创建模型
    model = TemporalBodyStateful(cfg)

    # 加载权重 (如果提供)
    if args.weights:
        print(f"\nLoading weights: {args.weights}")
        state = torch.load(args.weights, map_location="cpu")
        # 支持嵌套结构: state['depthformer'] 或直接 flat dict
        if 'depthformer' in state:
            depth_state = state['depthformer']
        else:
            depth_state = state
        # 只加载 temporal body 的权重 (key 前缀: "temporal_body.")
        temporal_state = {}
        for k, v in depth_state.items():
            if k.startswith("temporal_body."):
                temporal_state[k[len("temporal_body."):]] = v
        if temporal_state:
            missing, unexpected = model.load_state_dict(temporal_state, strict=False)
            print(f"  Loaded temporal body: {len(temporal_state)} params")
            if missing:
                print(f"  Missing keys: {len(missing)} (e.g. {missing[:3]})")
            if unexpected:
                print(f"  Unexpected keys: {unexpected}")
        else:
            print(f"  WARNING: No temporal_body.* keys found in checkpoint")

    # Step 1: 导出现有 ONNX (使用 export_onnx 的逻辑)
    onnx_path = args.onnx_path or os.path.join(args.output_dir, "temporal_body.onnx")

    if args.onnx_path and os.path.exists(args.onnx_path):
        print(f"\nUsing existing ONNX: {args.onnx_path}")
    else:
        print(f"\nExporting ONNX to: {onnx_path}")
        from export.export_onnx import export_temporal_body
        export_temporal_body(args.output_dir, args.max_kv_len, model=model)

    # Step 2: 结构验证
    print("\n" + "=" * 60)
    print("ONNX Structure Check")
    print("=" * 60)
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Nodes: {len(onnx_model.graph.node)}")
    print(f"  Inputs: {len(onnx_model.graph.input)}")
    for inp in onnx_model.graph.input[:5]:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in inp.type.tensor_type.shape.dim]
        print(f"    {inp.name}: {shape}")
    if len(onnx_model.graph.input) > 5:
        print(f"    ... ({len(onnx_model.graph.input) - 5} more)")
    print(f"  Outputs: {len(onnx_model.graph.output)}")
    for out in onnx_model.graph.output[:3]:
        shape = [d.dim_value if d.dim_value else d.dim_param for d in out.type.tensor_type.shape.dim]
        print(f"    {out.name}: {shape}")
    print(f"  Opset: {onnx_model.opset_import[0].version}")
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    # Step 3: 数值验证
    results = {}
    results["single_frame"] = verify_single_frame(model, onnx_path, cfg, args.max_kv_len, T_cond)
    results["stateful_loop"] = verify_stateful_loop(model, onnx_path, cfg, args.max_kv_len, T_cond, args.num_frames)
    results["cross_attn"] = verify_cross_attention(model, onnx_path, cfg, args.max_kv_len, T_cond)

    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    for name, passed in results.items():
        print(f"  {name}: {'[PASS]' if passed else '[FAIL]'}")
    all_pass = all(results.values())
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
