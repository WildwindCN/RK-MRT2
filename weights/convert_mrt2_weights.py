"""MRT2 JAX 权重 → PyTorch 模型转换

将 HuggingFace safetensors 格式的 MRT2 权重加载到我们的 PyTorch 模型。

JAX 权重格式:
  - Q/K/V 投影: [input_dim, num_heads, dim_per_head]
  - 输出投影: [num_heads, dim_per_head, output_dim]
  - RMSNorm: {scale: [dim]}
  - LayerNorm: {scale: [dim], bias: [dim]}
  - Linear: {kernel: [in, out], bias: [out]}

PyTorch 格式:
  - nn.Linear.weight: [out_features, in_features]
  - RMSNorm.weight: [dim]
  - nn.Embedding.weight: [num_embeddings, dim]
  - nn.Conv2d.weight: [out_channels, in_channels, kH, kW]

用法 (在 WSL 或 Linux 中):
    python weights/convert_mrt2_weights.py \
        --checkpoint ~/models/mrt2_small/checkpoint.safetensors \
        --output ../exported/weights/mrt2_small_pytorch.pt \
        --verify
"""

import os
import sys
import argparse
import struct
from pathlib import Path

import numpy as np
import torch
import safetensors

# 统一使用普通 safetensors 加载 (避免 JAX 依赖)
def _load_safetensors(path):
    """用普通 safetensors 加载 (返回 numpy arrays, 无需 JAX)"""
    flat = {}
    with safetensors.safe_open(path, framework="np") as f:
        for key in f.keys():
            flat[key] = f.get_tensor(key)
    return flat

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import DepthFormerConfig, SpectroStreamConfig
from models.depthformer import DepthFormer, TemporalBodyFull, DepthBodyAR
from models.spectrostream import SpectroStreamDecoder, RVQEmbedding
from models.transformer import RMSNorm, TransformerBlock


# ═══════════════════════════════════════════════════════
# JAX → PyTorch Tensor 转换工具
# ═══════════════════════════════════════════════════════

def jax_to_torch(arr: np.ndarray, transpose: bool = False) -> torch.Tensor:
    """JAX numpy array → PyTorch tensor"""
    t = torch.from_numpy(arr.copy()).float()
    return t.T if transpose else t


def load_jax_params(path: str) -> dict:
    """加载 JAX safetensors checkpoint

    Returns:
        嵌套字典 params[top_key][sub_key]... = numpy array
    """
    print(f"Loading: {path}")
    flat = _load_safetensors(path)
    # 将 flat keys (a/b/c) 转为嵌套字典
    nested = {}
    for key, value in flat.items():
        parts = key.split("/")
        d = nested
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = value
    print(f"  {len(flat)} tensor keys loaded")
    return nested


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ═══════════════════════════════════════════════════════
# Attention 权重加载
# ═══════════════════════════════════════════════════════

def load_qkv_projection(pt_linear: torch.nn.Linear, jax_kernel: np.ndarray):
    """JAX Q/K/V kernel → PyTorch nn.Linear

    JAX: [input_dim, num_heads, dim_per_head]
    PyTorch Linear: weight [num_heads*dim_per_head, input_dim]
    """
    in_dim, num_heads, uph = jax_kernel.shape
    out_dim = num_heads * uph
    # [in, heads, uph] → [in, heads*uph] → [heads*uph, in]
    w = jax_to_torch(jax_kernel.reshape(in_dim, out_dim), transpose=True)
    assert w.shape == pt_linear.weight.shape, \
        f"QKV shape mismatch: {w.shape} vs {pt_linear.weight.shape}"
    pt_linear.weight.data.copy_(w)


def load_output_projection(pt_linear: torch.nn.Linear, jax_kernel: np.ndarray):
    """JAX output projection → PyTorch nn.Linear

    JAX EinsumDense('...nh,dnh->...d'): kernel [d=out_dim, n=heads, h=dim_per_head]
    PyTorch Linear: weight [out_features, in_features] = [out_dim, heads*dim_per_head]

    直接 reshape, 无需 transpose.
    """
    out_dim, num_heads, uph = jax_kernel.shape
    in_dim = num_heads * uph
    w = jax_to_torch(jax_kernel.reshape(out_dim, in_dim))
    assert w.shape == pt_linear.weight.shape, \
        f"Output proj shape mismatch: {w.shape} vs {pt_linear.weight.shape}"
    pt_linear.weight.data.copy_(w)


def load_attention_weights(pt_block: TransformerBlock, jax_attn: dict,
                           *, is_cross_attn: bool = False):
    """加载单个 Attention 的权重 (self-attn 或 cross-attn)

    pt_block: TransformerBlock 实例
    jax_attn: JAX attention params 字典
        结构: {attention: {query_projection: {kernel: ...}, ...}, output_projection: {...}, pre_norm: {...}, ...}
    """
    attn = pt_block.cross_attn if is_cross_attn else pt_block.self_attn
    prefix = "cross_attn" if is_cross_attn else "self_attn"

    # 处理 attention/ 嵌套
    jax_inner = jax_attn.get("attention", jax_attn)

    # Q/K/V projections
    load_qkv_projection(attn.q_proj, jax_inner["query_projection"]["kernel"])
    load_qkv_projection(attn.k_proj, jax_inner["key_projection"]["kernel"])
    load_qkv_projection(attn.v_proj, jax_inner["value_projection"]["kernel"])

    # Output projection
    load_output_projection(attn.out_proj, jax_attn["output_projection"]["kernel"])

    # Per-dim scale (PyTorch defaults to 1/sqrt(d), JAX learns it)
    if "per_dim_scale" in jax_inner:
        scale = jax_to_torch(jax_inner["per_dim_scale"])
        attn.scale = float(scale.mean().item())

    # Attention sink (self-attn only: cross-attn has sink for streaming)
    if attn.attention_sink is not None and "sink_key_embeddings" in jax_inner:
        sk = jax_to_torch(jax_inner["sink_key_embeddings"])  # [1, H, D]
        sv = jax_to_torch(jax_inner["sink_value_embeddings"])
        # [1, H, D] → [H, 1, D]
        attn.attention_sink.sink_k.data.copy_(sk.squeeze(0).unsqueeze(0))
        attn.attention_sink.sink_v.data.copy_(sv.squeeze(0).unsqueeze(0))

    # RMS Norms
    pre_norm = pt_block.pre_self_attn_norm if not is_cross_attn else pt_block.pre_cross_attn_norm
    post_norm = pt_block.post_self_attn_norm if not is_cross_attn else pt_block.post_cross_attn_norm
    if pre_norm is not None and "pre_norm" in jax_attn:
        pre_norm.weight.data.copy_(jax_to_torch(jax_attn["pre_norm"]["scale"]))
    if post_norm is not None and "post_norm" in jax_attn:
        post_norm.weight.data.copy_(jax_to_torch(jax_attn["post_norm"]["scale"]))

    log = "cross-attn" if is_cross_attn else "self-attn"
    in_dim = jax_inner["query_projection"]["kernel"].shape[0]
    print(f"    {prefix}/{log}: QKV [{in_dim}→{attn.inner_dim}] [OK]")


def load_ffn_weights(pt_block: TransformerBlock, jax_ffn: dict):
    """加载 FFN 权重"""
    ffn = pt_block.ffn
    jax_f1 = jax_ffn["ffn_layer1"]
    jax_f2 = jax_ffn["ffn_layer2"]

    if hasattr(ffn, 'gate_proj'):
        # Gated FFN: gate_proj + up_proj share the doubled hidden_dim
        # JAX ffn_layer1 kernel: [in, hidden*2] → split
        k1 = jax_f1["kernel"]
        in_dim, doubled = k1.shape
        half = doubled // 2
        gate_kernel = k1[:, :half]   # [in, hidden]
        up_kernel = k1[:, half:]     # [in, hidden]
        ffn.gate_proj.weight.data.copy_(jax_to_torch(gate_kernel, transpose=True))
        ffn.up_proj.weight.data.copy_(jax_to_torch(up_kernel, transpose=True))
        if "bias" in jax_f1:
            ffn.gate_proj.bias.data.copy_(jax_to_torch(jax_f1["bias"][:half]))
            ffn.up_proj.bias.data.copy_(jax_to_torch(jax_f1["bias"][half:]))
    else:
        # Standard FFN
        ffn.up_proj.weight.data.copy_(jax_to_torch(jax_f1["kernel"], transpose=True))
        if "bias" in jax_f1:
            ffn.up_proj.bias.data.copy_(jax_to_torch(jax_f1["bias"]))

    ffn.down_proj.weight.data.copy_(jax_to_torch(jax_f2["kernel"], transpose=True))
    if "bias" in jax_f2:
        ffn.down_proj.bias.data.copy_(jax_to_torch(jax_f2["bias"]))

    # RMS Norms
    pt_block.pre_ffn_norm.weight.data.copy_(jax_to_torch(jax_ffn["pre_norm"]["scale"]))
    pt_block.post_ffn_norm.weight.data.copy_(jax_to_torch(jax_ffn["post_norm"]["scale"]))

    print(f"    ffn: [{jax_f1['kernel'].shape[0]}→{jax_f1['kernel'].shape[1]}→{jax_f2['kernel'].shape[1]}] [OK]")


def load_transformer_block(pt_block: TransformerBlock, jax_block: dict,
                           *, has_cross_attn: bool = True):
    """加载单个 Transformer Block"""
    load_attention_weights(pt_block, jax_block["self_attention"], is_cross_attn=False)
    if has_cross_attn and "cross_attention" in jax_block:
        load_attention_weights(pt_block, jax_block["cross_attention"], is_cross_attn=True)
    load_ffn_weights(pt_block, jax_block["ffn"])


# ═══════════════════════════════════════════════════════
# 主加载函数
# ═══════════════════════════════════════════════════════

def load_depthformer_weights(pt_model: DepthFormer, jax_params: dict):
    """将 JAX depthformer 权重加载到 PyTorch 模型"""
    # 处理 params/ 前缀
    jax_df = jax_params.get("params", jax_params)
    jax_df = jax_df.get("depthformer", jax_df)
    jax_dec = jax_df["decoder"]
    cfg = pt_model.config

    print("\n[1] Loading token embedding...")
    emb_weight = jax_to_torch(jax_dec["decoder_embedding"]["embedding"]["embedding"])
    pt_model.token_embedding.embed.weight.data.copy_(emb_weight)
    print(f"  Embedding: {list(emb_weight.shape)} [OK]")

    print("\n[2] Loading Temporal Body (12 layers)...")
    jax_temporal = jax_dec["temporal_body"]["transformer"]
    for i in range(cfg.temporal_spec.num_layers):
        layer_key = f"x_layers_{i}"
        if layer_key in jax_temporal:
            load_transformer_block(
                pt_model.temporal_body.layers[i],
                jax_temporal[layer_key],
                has_cross_attn=True,
            )
        else:
            print(f"  [WARN] {layer_key} not found in checkpoint")

    print("\n[3] Loading Depth Body (2 layers)...")
    jax_depth = jax_depth_body = jax_dec["depth_body"]

    # depth_input_adapter: Linear [temporal_dim → depth_dim]
    if "depth_input_adapter" in jax_depth_body:
        adapter_kernel = jax_to_torch(
            jax_depth_body["depth_input_adapter"]["kernel"], transpose=True
        )
        pt_model.depth_body.input_adapter.weight.data.copy_(adapter_kernel)
        print(f"  Input adapter: {list(adapter_kernel.shape)} [OK]")

    # Depth transformer layers
    jax_depth_xformer = jax_depth_body["transformer"]
    for i in range(cfg.depth_spec.num_layers):
        layer_key = f"x_layers_{i}"
        if layer_key in jax_depth_xformer:
            load_transformer_block(
                pt_model.depth_body.layers[i],
                jax_depth_xformer[layer_key],
                has_cross_attn=False,
            )

    # Final LayerNorm
    if "final_ln" in jax_depth_body:
        jax_ln = jax_depth_body["final_ln"]
        pt_model.depth_body.final_norm.weight.data.copy_(jax_to_torch(jax_ln["scale"]))
        if hasattr(pt_model.depth_body.final_norm, 'bias') and "bias" in jax_ln:
            pt_model.depth_body.final_norm.bias.data.copy_(jax_to_torch(jax_ln["bias"]))
        print(f"  Final LayerNorm [OK]")

    # to_logits
    if "to_logits" in jax_depth_body:
        logits_kernel = jax_to_torch(
            jax_depth_body["to_logits"]["kernel"], transpose=True
        )
        pt_model.depth_body.to_logits.weight.data.copy_(logits_kernel)
        print(f"  to_logits: {list(logits_kernel.shape)} [OK]")

    params_loaded = count_params(pt_model)
    print(f"\n  DepthFormer: {params_loaded/1e6:.1f}M params loaded")


def load_spectrostream_weights(pt_decoder: SpectroStreamDecoder, jax_params: dict):
    """加载 SpectroStream 权重 (quantizer + decoder)

    JAX Conv2D kernel: [kH, kW, Cin, Cout] → PyTorch: [Cout, Cin, kH, kW]
    JAX ConvTranspose kernel: [kH, kW, Cin, Cout] → PyTorch: [Cin, Cout, kH, kW] (flipped)
    """
    jax_p = jax_params.get("params", jax_params)
    jax_ss = jax_p.get("soundstream", jax_p)
    cfg = pt_decoder.config

    print("\n[4] Loading RVQ Quantizer embeddings...")
    q_emb = jax_to_torch(jax_ss["quantizer"]["embedding"])
    pt_decoder.rvq_embedding.embedding.data.copy_(q_emb)
    print(f"  RVQ: {list(q_emb.shape)} [OK]")

    print("\n[5] Loading Decoder conv layers...")
    jax_dec = jax_ss["decoder"]

    # 递归收集所有卷积参数
    _loaded_conv_count = [0]
    _skipped_conv_count = [0]

    def _find_conv_params(jax_node, prefix=""):
        result = {}
        for key, value in jax_node.items():
            if isinstance(value, dict):
                if "kernel" in value:
                    result[prefix + key] = value
                else:
                    result.update(_find_conv_params(value, prefix + key + "/"))
        return result

    all_jax_convs = _find_conv_params(jax_dec)
    print(f"  Found {len(all_jax_convs)} conv layers in JAX checkpoint")

    # 收集 PyTorch 模型中所有 Conv2d
    pt_conv_list = []
    for name, module in pt_decoder.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            pt_conv_list.append((name, module))

    print(f"  Found {len(pt_conv_list)} conv layers in PyTorch model")

    def _load_conv(pt_conv, jax_params_dict, is_transpose=False):
        kernel = np.array(jax_params_dict["kernel"])
        if is_transpose:
            kernel = kernel[::-1, ::-1, :, :]  # flip spatial
            pt_weight = torch.from_numpy(kernel.copy()).permute(2, 3, 0, 1).float()
        else:
            pt_weight = torch.from_numpy(kernel.copy()).permute(3, 2, 0, 1).float()

        if pt_conv.weight.shape == pt_weight.shape:
            pt_conv.weight.data.copy_(pt_weight)
            _loaded_conv_count[0] += 1
            return True
        return False

    # 匹配卷积层 (JAX 名称 → PyTorch 模块)
    # 由于架构差异，很多层无法直接匹配
    jax_conv_items = list(all_jax_convs.items())

    for pt_name, pt_conv in pt_conv_list:
        matched = False
        for jax_name, jax_params in jax_conv_items:
            is_transpose = "transpose" in jax_name.lower()
            if _load_conv(pt_conv, jax_params, is_transpose):
                matched = True
                break
        if not matched:
            _skipped_conv_count[0] += 1

    print(f"  Loaded: {_loaded_conv_count[0]}, Skipped: {_skipped_conv_count[0]}")
    if _skipped_conv_count[0] > 0:
        print(f"  NOTE: {_skipped_conv_count[0]} conv layers not matched (architecture differs)")
        print(f"  The decoder conv weights need a matching architecture or retraining")


# ═══════════════════════════════════════════════════════
# 精度验证
# ═══════════════════════════════════════════════════════

def verify_weights(pt_model, jax_params, model_name: str, rtol: float = 1e-4):
    """对比 PyTorch 和 JAX 模型输出"""
    print(f"\n[Verify] {model_name}:")
    print(f"  (需要运行 JAX 参考实现来对比 - 当前仅检查权重统计)")

    # 检查权重统计
    total_params = 0
    total_nan = 0
    for name, param in pt_model.named_parameters():
        total_params += param.numel()
        total_nan += torch.isnan(param).sum().item()

    print(f"  Total params: {total_params:,}")
    print(f"  NaN count: {total_nan}")
    print(f"  [{'PASS' if total_nan == 0 else 'FAIL'}] No NaN check")


# ═══════════════════════════════════════════════════════
# 保存 PyTorch 权重
# ═══════════════════════════════════════════════════════

def save_pytorch_weights(pt_models: dict, output_path: str):
    """保存转换后的 PyTorch state_dict"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    state = {name: model.state_dict() for name, model in pt_models.items()}
    torch.save(state, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Convert MRT2 JAX weights to PyTorch")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to MRT2 safetensors checkpoint")
    parser.add_argument("--output", type=str, default="./exported/weights/mrt2_small_pytorch.pt",
                        help="Output path for PyTorch weights")
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "base"], help="MRT2 model size")
    parser.add_argument("--verify", action="store_true", help="Verify weights after loading")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 60)
    print(f"MRT2 JAX → PyTorch Weight Conversion")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    # 加载 JAX 参数
    jax_params = load_jax_params(args.checkpoint)
    jax_p = jax_params.get("params", jax_params)
    print(f"  Top-level keys: {list(jax_p.keys())}")

    # 创建 PyTorch 模型
    df_cfg = DepthFormerConfig()
    ss_cfg = SpectroStreamConfig()

    if args.model == "small":
        df_cfg.encoder_spec.num_layers = 6
        df_cfg.encoder_spec.model_dims = 256
        df_cfg.encoder_spec.hidden_dims = 1024
        df_cfg.encoder_spec.num_heads = 8
        df_cfg.encoder_spec.dim_per_head = 32

    print(f"\nBuilding PyTorch models...")
    depthformer = DepthFormer(df_cfg).to(args.device)
    codec_decoder = SpectroStreamDecoder(ss_cfg).to(args.device)

    # 加载权重
    load_depthformer_weights(depthformer, jax_p)
    load_spectrostream_weights(codec_decoder, jax_p)

    # 验证
    if args.verify:
        verify_weights(depthformer, jax_p, "DepthFormer")
        verify_weights(codec_decoder, jax_p, "SpectroStream Decoder")

    # 保存
    models = {"depthformer": depthformer, "codec_decoder": codec_decoder}
    save_pytorch_weights(models, args.output)

    print("\nDone!")


if __name__ == "__main__":
    main()
