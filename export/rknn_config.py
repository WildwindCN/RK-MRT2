"""RKNN 转换配置

每个 graph 的 RKNN 编译参数, 针对 RK3576 NPU 优化。

RKNN-Toolkit2 安装要求:
- x86 Linux (Ubuntu 20.04/22.04)
- Python 3.8-3.10
- pip install rknn-toolkit2

用法 (在 Linux 构建机上):
    python deploy/convert_all.py --graph all --precision fp16
"""

# ═══════════════════════════════════════════════════════
# 通用 RKNN 配置
# ═══════════════════════════════════════════════════════

COMMON_CONFIG = {
    "target_platform": "rk3576",
    "optimization_level": 3,
    "float_dtype": "float16",
}

# ═══════════════════════════════════════════════════════
# 各 Graph 的专用配置
# ═══════════════════════════════════════════════════════

TEMPORAL_BODY_CONFIG = {
    **COMMON_CONFIG,
    "model_name": "temporal_body",
    "description": "12-layer Temporal Body (single-frame stateful, 27 inputs)",
    "onnx_file": "temporal_body_sim.onnx",
    "rknn_file": "temporal_body.rknn",
    # 27 inputs: x + cond + attn_mask + 12 self_k + 12 self_v
    # 25 outputs: output + 12 self_k_out + 12 self_v_out
    # KV cache: 42 positions (41 window + 1 sink), mask: 44 positions
    # ~365MB FP16 weights
}

DEPTH_BODY_CONFIG = {
    **COMMON_CONFIG,
    "model_name": "depth_body",
    "description": "2-layer Depth Body (RVQ AR)",
    "onnx_file": "depth_body.onnx",
    "rknn_file": "depth_body.rknn",
    # 输入: [B, 12, 1024], 输出: [B, 12, 12294]
    # 注意: FP16 可能溢出 (token 嵌入放大), 如不行用 FP32
}

CODEC_DECODER_CONFIG = {
    **COMMON_CONFIG,
    "model_name": "codec_decoder",
    "description": "SpectroStream ConvTranspose Decoder",
    "onnx_file": "codec_decoder.onnx",
    "rknn_file": "codec_decoder.rknn",
    # 纯卷积 → INT8 量化可选
}

# ═══════════════════════════════════════════════════════
# 精度验证阈值
# ═══════════════════════════════════════════════════════

VERIFICATION_THRESHOLDS = {
    "cosine_similarity": 0.99,   # 余弦相似度 ≥ 0.99
    "max_relative_error": 0.01,  # 最大相对误差 < 1%
    "nan_inf_ratio": 0.0,        # NaN/Inf 比例必须为 0
}
