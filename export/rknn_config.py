"""RKNN 转换配置

每个 graph 的 RKNN 编译参数, 针对 RK3588 NPU 优化。

RKNN-Toolkit2 安装要求:
- x86 Linux (Ubuntu 20.04/22.04)
- Python 3.8-3.10
- pip install rknn-toolkit2

用法 (在 Linux 构建机上):
    python export/convert_rknn.py --onnx_dir ./exported --output_dir ./rknn_models
"""

import os

# ═══════════════════════════════════════════════════════
# 通用 RKNN 配置
# ═══════════════════════════════════════════════════════

COMMON_CONFIG = {
    "target_platform": "rk3588",
    "optimization_level": 3,
    "output_tensor_type": "float16",
    "do_quantization": False,
    "inputs_zp": {},
    "outputs_zp": {},
}

# ═══════════════════════════════════════════════════════
# 各 Graph 的专用配置
# ═══════════════════════════════════════════════════════

TEMPORAL_BODY_CONFIG = {
    **COMMON_CONFIG,
    "model_name": "temporal_body",
    "description": "12-layer Temporal Body (single-frame stateful)",
    "onnx_file": "temporal_body.onnx",
    "rknn_file": "temporal_body.rknn",
    # FP16 量化 (避免 INT8 精度损失)
    "quantized_dtype": "float16",
    # 大模型分段加载 (SRAM 32KB tiling)
    "output_tensor_type": "float16",
    # 输入归一化 (可选, 取决于训练数据分布)
    "mean_values": None,
    "std_values": None,
    # KV cache 维度: [B, H, L, D] per layer
    # 26 inputs (x + cond + 12 self_k + 12 self_v)
    # 25 outputs (out + 12 self_k + 12 self_v)
}

DEPTH_BODY_CONFIG = {
    **COMMON_CONFIG,
    "model_name": "depth_body",
    "description": "2-layer Depth Body (RVQ AR)",
    "onnx_file": "depth_body.onnx",
    "rknn_file": "depth_body.rknn",
    "quantized_dtype": "float16",
    # 输入: [B, 12, 1024]
    # 输出: [B, 12, 1030]
    # 注意: FP16 可能溢出 (iPhone 移植经验: 15.7% NaN)
    # 对策: 先用 FP16 验证, 如不行改用 FP32 CPU
    "output_tensor_type": "float16",
}

CODEC_DECODER_CONFIG = {
    **COMMON_CONFIG,
    "model_name": "codec_decoder",
    "description": "SpectroStream ConvTranspose Decoder",
    "onnx_file": "codec_decoder.onnx",
    "rknn_file": "codec_decoder.rknn",
    # 纯卷积 → INT8 量化安全
    "quantized_dtype": "int8",
    "quantized_method": "layer",
    "quantized_algorithm": "mmse",  # 最小化均方误差
    "output_tensor_type": "float16",
    # Conv+ELU 融合加速
    "optimization_level": 3,
}

# ═══════════════════════════════════════════════════════
# INT8 量化配置 (校准数据集)
# ═══════════════════════════════════════════════════════

INT8_QUANT_CONFIG = {
    "quantized_dtype": "int8",
    "quantized_method": "layer",
    "quantized_algorithm": "mmse",
    "do_quantization": True,
    "dataset": None,  # 需提供代表性校准数据集
    # 校准图像/数据数量
    "quant_img_num": 100,
    # 批量大小
    "batch_size": 1,
}

# ═══════════════════════════════════════════════════════
# 精度验证阈值
# ═══════════════════════════════════════════════════════

VERIFICATION_THRESHOLDS = {
    "cosine_similarity": 0.99,   # 余弦相似度 ≥ 0.99
    "max_relative_error": 0.01,  # 最大相对误差 < 1%
    "nan_inf_ratio": 0.0,        # NaN/Inf 比例必须为 0
}
