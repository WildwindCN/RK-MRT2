"""训练框架 → 推理框架 对齐规范

本文档定义训练框架产出的模型需要满足的接口/格式约束，
确保训练好的权重可以直接加载到 RK-MRT2 推理框架并导出 ONNX → RKNN。

用法 (验证训练产出):
    python spec/training_inference_spec.py --checkpoint /path/to/training_output.pt
"""

import os, sys, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import DepthFormerConfig, SpectroStreamConfig
from models.depthformer import DepthFormer
from models.spectrostream import SpectroStreamDecoder

# ═══════════════════════════════════════════════════════
# 一、模型配置约束
# ═══════════════════════════════════════════════════════

# 以下参数必须与推理框架一致 —— 否则权重 shape 不匹配
REQUIRED_CONFIG = {
    # DepthFormer
    "temporal_num_layers": 12,
    "temporal_model_dims": 1024,
    "temporal_hidden_dims": 4096,
    "temporal_num_heads": 8,
    "temporal_dim_per_head": 128,
    "temporal_ffn_gated": False,

    "depth_num_layers": 2,
    "depth_model_dims": 768,
    "depth_hidden_dims": 3072,
    "depth_num_heads": 6,
    "depth_dim_per_head": 128,

    "num_codebooks": 12,
    "codebook_size": 1024,
    "num_reserved_tokens": 6,

    "max_past_horizon": 41,
    "num_attention_sink_embeddings": 1,
    "use_rope": False,  # NoPE

    # SpectroStream Decoder
    "ss_embedding_dim": 256,
    "ss_num_quantizers": 64,
    "ss_codebook_size": 1024,
    "ss_num_bins": 480,
    "ss_num_channels": 4,
    "ss_total_time_stride": 4,
    "ss_total_freq_stride": 96,

    # Audio
    "sample_rate": 48000,
    "frame_rate": 25.0,
}

# ═══════════════════════════════════════════════════════
# 二、权重保存格式
# ═══════════════════════════════════════════════════════

EXPECTED_STATE_DICT_KEYS = {
    "depthformer",    # DepthFormer state_dict
    "codec_decoder",  # SpectroStreamDecoder state_dict
}

# 保存示例:
# torch.save({
#     "depthformer": model.depthformer.state_dict(),
#     "codec_decoder": model.codec_decoder.state_dict(),
#     "config": { ... },  # 可选, 训练配置快照
# }, "training_output.pt")

# ═══════════════════════════════════════════════════════
# 三、Token 格式规范
# ═══════════════════════════════════════════════════════

def validate_token_format():
    """Token 格式说明"""
    print("""
Token 格式规范:
  - 训练数据: per-RVQ 索引, 范围 [0, codebook_size-1] = [0, 1023]
  - 模型内部: 自动转换为全词表索引
      full_token = num_reserved + rvq_idx * codebook_size + local_token
  - SOS token: 0 (全词表中的 reserved 0)
  - 全词表: vocab_size = num_codebooks * codebook_size + num_reserved = 12294
  - 每 RVQ 层有效范围: [num_reserved + rvq*codebook_size, num_reserved + (rvq+1)*codebook_size)

训练时:
  - 输入 tokens: [B, T, Q] int32, values in [0, codebook_size-1]
  - 模型内部调用 _pad_sos_and_offset() 自动转换
  - 损失计算使用全词表 logits [B, T, Q, 12294] vs targets [B, T, Q]

推理时:
  - 采样 logits 后, 使用 valid_range 限制到当前 RVQ 层的词表范围
  - 采样的 token 需减去偏移才能用于 codec decoder
  - codec decoder 接收 per-RVQ tokens [0, codebook_size-1]
""")

# ═══════════════════════════════════════════════════════
# 四、ONNX 兼容性检查
# ═══════════════════════════════════════════════════════

ONNX_FORBIDDEN_OPS = {
    "einsum",
    "torch.cumsum",
    "torch.stft", "torch.istft",
    "F.grid_sample",
    "torch.nonzero",
}

def check_onnx_compatibility(model: torch.nn.Module) -> list[str]:
    """检查模型中是否有 ONNX 不兼容的操作"""
    issues = []

    import inspect
    for name, module in model.named_modules():
        try:
            src = inspect.getsource(module.forward)
            for op in ONNX_FORBIDDEN_OPS:
                # 排除注释中的引用
                for line in src.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith('"""'):
                        continue
                    if op in stripped:
                        issues.append(f"{name}: uses {op}")
                        break
        except (TypeError, OSError):
            pass

    return issues

# ═══════════════════════════════════════════════════════
# 五、Decoder 架构要求 (NPU 部署)
# ═══════════════════════════════════════════════════════

DECODER_REQUIREMENTS = """
SpectroStream Decoder NPU 部署要求:

1. 架构必须是纯卷积 (Conv/ConvTranspose) + 激活函数
   - RKNN 不支持 STFT/iSTFT (CPU 端执行)
   - RKNN 不支持 einsum (用 MatMul 替代)

2. 输入: RVQ token embeddings [B, T, 256]
   输出: STFT features [B, 4, 480, T*4]

3. 建议使用标准 ConvTranspose 架构 (而非 interpolate+conv)
   - JAX SpectroStream 使用 7-stage ConvTranspose2d
   - 每 stage: ConvTranspose2d (stride > 1) → ELU → Conv2d(3×3) → residual
   - RKNN ConvTranspose 有硬件加速

4. 参数量目标: ~30M (匹配 ~16kbps 码率)

5. 训练建议:
   - 冻结 RVQ quantizer (64×1024×256)
   - 只训练 decoder 卷积层
   - Loss: L1 + multi-scale STFT loss
   - 训练数据: 高质量音乐 (48kHz 立体声)
"""

# ═══════════════════════════════════════════════════════
# 六、验证工具
# ═══════════════════════════════════════════════════════

def validate_checkpoint(ckpt_path: str, device: str = "cpu"):
    """验证训练产出是否符合推理框架要求"""
    print("=" * 60)
    print("Training → Inference Compatibility Check")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 60)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 1. 检查必需 key
    print("\n[1] State dict keys:")
    missing_keys = EXPECTED_STATE_DICT_KEYS - set(state.keys())
    extra_keys = set(state.keys()) - EXPECTED_STATE_DICT_KEYS

    if missing_keys:
        print(f"  [FAIL] Missing: {missing_keys}")
        return False
    else:
        print(f"  [PASS] All required keys present")

    if extra_keys:
        print(f"  [INFO] Extra keys (ignored): {extra_keys}")

    # 2. 加载并验证 shape
    print("\n[2] Building reference models...")
    df_cfg = DepthFormerConfig()
    ss_cfg = SpectroStreamConfig()

    ref_df = DepthFormer(df_cfg).to(device)
    ref_dec = SpectroStreamDecoder(ss_cfg).to(device)

    print("\n[3] Loading weights...")
    df_missing, df_unexpected = ref_df.load_state_dict(
        state["depthformer"], strict=False
    )
    dec_missing, dec_unexpected = ref_dec.load_state_dict(
        state["codec_decoder"], strict=False
    )

    print(f"  DepthFormer: {len(df_missing)} missing, {len(df_unexpected)} unexpected")
    if df_missing:
        print(f"    Missing (first 5): {df_missing[:5]}")
    if df_unexpected:
        print(f"    Unexpected (first 5): {df_unexpected[:5]}")

    print(f"  Decoder: {len(dec_missing)} missing, {len(dec_unexpected)} unexpected")

    # 3. 前向传播测试
    print("\n[4] Forward pass test...")
    ref_df.eval()
    ref_dec.eval()

    B, T_cond, T_tok = 1, 50, 10
    cond = torch.randn(B, T_cond, df_cfg.encoder_spec.model_dims)
    tokens = torch.randint(0, df_cfg.codebook_size, (B, T_tok, df_cfg.num_codebooks))

    with torch.no_grad():
        logits, loss = ref_df(cond, tokens, return_loss=True)
        ppl = math.exp(min(loss.item(), 20))

    assert not torch.isnan(logits).any(), "NaN in DepthFormer output!"
    print(f"  DepthFormer: loss={loss.item():.4f}, ppl={ppl:.1f}, shape={list(logits.shape)}")
    print(f"  [PASS] No NaN")

    # Decoder forward
    emb = torch.randn(1, 25, ss_cfg.embedding_dim)
    with torch.no_grad():
        stft = ref_dec(emb)
    assert not torch.isnan(stft).any(), "NaN in Decoder output!"
    print(f"  Decoder: shape={list(stft.shape)}")
    print(f"  [PASS] No NaN")

    # 4. ONNX 兼容性
    print("\n[5] ONNX compatibility:")
    df_issues = check_onnx_compatibility(ref_df)
    dec_issues = check_onnx_compatibility(ref_dec)
    total_issues = df_issues + dec_issues

    if total_issues:
        print(f"  [WARN] {len(total_issues)} potential issues:")
        for issue in total_issues:
            print(f"    - {issue}")
    else:
        print(f"  [PASS] No forbidden ops detected")

    # 5. 参数统计
    print("\n[6] Parameter summary:")
    df_params = sum(p.numel() for p in ref_df.parameters())
    dec_params = sum(p.numel() for p in ref_dec.parameters())
    print(f"  DepthFormer: {df_params/1e6:.1f}M")
    print(f"  Decoder:     {dec_params/1e6:.1f}M")
    print(f"  Total:       {(df_params+dec_params)/1e6:.1f}M")

    print("\n" + "=" * 60)
    success = (len(df_missing) < 10 and "NaN" not in str(logits))
    print(f"Overall: {'[PASS]' if success else '[FAIL]'}")
    print("=" * 60)

    return success


def print_spec():
    """打印完整对齐规范"""
    print(__doc__)
    print("\n" + "=" * 60)
    print("REQUIRED CONFIG")
    print("=" * 60)
    for k, v in REQUIRED_CONFIG.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("TOKEN FORMAT")
    print("=" * 60)
    validate_token_format()

    print("=" * 60)
    print("DECODER REQUIREMENTS")
    print("=" * 60)
    print(DECODER_REQUIREMENTS)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training-Inference Alignment Spec")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to training output checkpoint to validate")
    parser.add_argument("--spec", action="store_true", default=True,
                        help="Print the full spec")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print_spec()

    if args.checkpoint:
        print("\n\n")
        ok = validate_checkpoint(args.checkpoint, args.device)
        sys.exit(0 if ok else 1)
