"""ONNX 导出脚本

将模型拆分为 3 个独立 graph 导出:
1. Temporal Body (单帧, stateful KV cache)
2. Depth Body (静态 [B, Q, D] → [B, Q, vocab])
3. Codec Decoder (静态 [B, T, D] → [B, C, F, T'])

所有图均避免 einsum/CumSum/STFT 等 RKNN 不支持算子。

用法:
    python export/export_onnx.py [--output_dir ./exported]
"""

import os
import sys
import argparse
import torch
import torch.onnx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import DepthFormerConfig, SpectroStreamConfig
from models.depthformer import TemporalBodyStateful, DepthBodyAR
from models.spectrostream import SpectroStreamDecoder


def export_temporal_body(output_dir: str, max_kv_len: int = 512):
    """导出 Temporal Body 为单帧 stateful ONNX graph

    输入:
      x:            [1, 1, 1024]  当前帧
      conditioning: [1, T_cond, 256]  MIDI encoder 输出 (预计算)
      self_k_cache: [1, 8, max_kv_len, 128]  每层 self-attn K cache
      self_v_cache: [1, 8, max_kv_len, 128]  每层 self-attn V cache
      cross_k:      [1, 8, T_cond, 128]  每层 cross-attn K (预计算)
      cross_v:      [1, 8, T_cond, 128]  每层 cross-attn V (预计算)

    输出:
      output:       [1, 1, 1024]
      self_k_new:   [1, 8, max_kv_len, 128]  更新后的 K cache
      self_v_new:   [1, 8, max_kv_len, 128]  更新后的 V cache

    注意: cross_k/cross_v 不变 (conditioning 固定), 不作为输出
    """
    print("\n[1/3] Exporting Temporal Body (single-frame stateful)...")

    cfg = DepthFormerConfig()
    model = TemporalBodyStateful(cfg)
    model.eval()

    T_cond = 50  # 条件序列长度 (示例)

    # 输入 tensors
    x = torch.randn(1, 1, cfg.temporal_spec.model_dims)
    cond = torch.randn(1, T_cond, cfg.encoder_spec.model_dims)

    # 为每层创建 KV cache 占位符 (self-attn only, cross-attn 从 conditioning 预计算)
    num_layers = cfg.temporal_spec.num_layers
    num_heads = cfg.temporal_spec.num_heads
    dim_per_head = cfg.temporal_spec.dim_per_head

    self_k_caches = []
    self_v_caches = []
    for _ in range(num_layers):
        self_k_caches.append(torch.randn(1, num_heads, max_kv_len, dim_per_head))
        self_v_caches.append(torch.randn(1, num_heads, max_kv_len, dim_per_head))

    # 构建带 KV cache 的推理
    class TemporalBodyExportWrapper(torch.nn.Module):
        def __init__(self, body, num_layers, max_len):
            super().__init__()
            self.body = body
            self.num_layers = num_layers
            self.max_len = max_len

        def forward(self, x, cond, *kv_inputs):
            # kv_inputs: [self_k_0, self_v_0, self_k_1, self_v_1, ...]
            kv_caches = []
            for i in range(self.num_layers):
                kv_caches.append({
                    "self_kv": (kv_inputs[i * 2], kv_inputs[i * 2 + 1]),
                    "cross_kv": None,  # 从 conditioning 动态计算
                })

            output, new_caches = self.body(x, cond, kv_caches)

            # 更新 self-attn KV cache (Concat + Slice to keep max_len)
            outputs = [output]
            for i, cache in enumerate(new_caches):
                sk, sv = cache["self_kv"]
                # Trim to max_len if needed (sliding window)
                if sk.shape[2] > self.max_len:
                    sk = sk[:, :, -self.max_len:, :]
                    sv = sv[:, :, -self.max_len:, :]
                outputs.extend([sk, sv])

            return tuple(outputs)

    wrapper = TemporalBodyExportWrapper(model, num_layers, max_kv_len)

    # 动态轴: kv cache 的时间维度可变
    dynamic_axes = {"x": {0: "batch"}, "cond": {0: "batch"}}
    input_names = ["x", "cond"]
    for i in range(num_layers):
        input_names.append(f"self_k_{i}")
        input_names.append(f"self_v_{i}")
        dynamic_axes[f"self_k_{i}"] = {0: "batch", 2: "kv_len"}
        dynamic_axes[f"self_v_{i}"] = {0: "batch", 2: "kv_len"}

    output_names = ["output"]
    for i in range(num_layers):
        output_names.append(f"self_k_out_{i}")
        output_names.append(f"self_v_out_{i}")
        dynamic_axes[f"self_k_out_{i}"] = {0: "batch", 2: "kv_len"}
        dynamic_axes[f"self_v_out_{i}"] = {0: "batch", 2: "kv_len"}

    # 构建完整输入
    all_inputs = [x, cond] + self_k_caches + self_v_caches

    path = os.path.join(output_dir, "temporal_body.onnx")
    torch.onnx.export(
        wrapper, tuple(all_inputs), path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"  → {path}")
    print(f"     Params: {params/1e6:.1f}M")
    print(f"     Inputs:  {len(input_names)} ({', '.join(input_names[:4])} ...)")
    print(f"     Outputs: {len(output_names)} ({', '.join(output_names[:4])} ...)")
    print(f"     Max KV length: {max_kv_len}")


def export_depth_body(output_dir: str):
    """导出 Depth Body 为静态 ONNX graph

    输入:  x: [B, Q, D]  (B=batch, Q=num_codebooks=12, D=temporal_dim=1024)
    输出:  logits: [B, Q, vocab_size]  (vocab_size=1030)

    RKNN 兼容: 全 MatMul + Reshape, 无 einsum
    """
    print("\n[2/3] Exporting Depth Body...")

    cfg = DepthFormerConfig()
    model = DepthBodyAR(cfg)
    model.eval()

    B, Q, D = 1, cfg.num_codebooks, cfg.temporal_spec.model_dims
    x = torch.randn(B, Q, D)

    # Use torch.export + ONNX exporter for better fidelity
    path = os.path.join(output_dir, "depth_body.onnx")
    torch.onnx.export(
        model, x, path,
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=True,
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"  → {path}")
    print(f"     Params: {params/1e6:.1f}M")
    print(f"     Input:  [B, {Q}, {D}]")
    print(f"     Output: [B, {Q}, {cfg.vocab_size}]")


def export_codec_decoder(output_dir: str, T: int = 25):
    """导出 SpectroStream Codec Decoder 为静态 ONNX graph

    输入:  embeddings: [1, T, 256]  (T=25 frames at 25Hz = 1 second)
    输出:  stft: [1, 4, 480, T*4]  (T*4=100 frames at 100Hz)

    iSTFT 在 CPU 端完成 (RKNN 不支持 STFT)
    """
    print("\n[3/3] Exporting Codec Decoder...")

    cfg = SpectroStreamConfig()
    model = SpectroStreamDecoder(cfg)
    model.eval()

    x = torch.randn(1, T, cfg.embedding_dim)

    path = os.path.join(output_dir, "codec_decoder.onnx")
    torch.onnx.export(
        model, x, path,
        input_names=["embeddings"],
        output_names=["stft_features"],
        dynamic_axes={
            "embeddings": {0: "batch", 1: "time"},
            "stft_features": {0: "batch", 3: "stft_time"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    params = sum(p.numel() for p in model.parameters())
    out_T = T * cfg.total_time_stride
    print(f"  → {path}")
    print(f"     Params: {params/1e6:.1f}M")
    print(f"     Input:  [1, {T}, {cfg.embedding_dim}]")
    print(f"     Output: [1, {cfg.num_channels}, {cfg.num_bins}, {out_T}]")


def main():
    parser = argparse.ArgumentParser(description="Export MRT2 models to ONNX")
    parser.add_argument("--output_dir", default="./exported",
                        help="Output directory for ONNX files")
    parser.add_argument("--max_kv_len", type=int, default=512,
                        help="Maximum KV cache length for temporal body")
    parser.add_argument("--codec_T", type=int, default=25,
                        help="Number of input frames for codec decoder (25 = 1 second)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 60)
    print("RK-MRT2 ONNX Export")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    export_temporal_body(args.output_dir, args.max_kv_len)
    export_depth_body(args.output_dir)
    export_codec_decoder(args.output_dir, args.codec_T)

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)

    print("\nNext steps:")
    print("  1. Verify ONNX: python export/verify_onnx.py")
    print("  2. Convert to RKNN: python export/convert_rknn.py")
    print("  3. C++ Runtime integration")


if __name__ == "__main__":
    main()
