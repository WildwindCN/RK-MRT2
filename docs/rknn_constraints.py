"""RKNN/RKLLM 算子兼容性约束

基于 RKNN-Toolkit2 v1.6.0+ OP Support 和 RKLLM v1.2.1 文档
"""

# ═══════════════════════════════════════════════════════
# RKNN 支持的算子 (用于 SpectroStream Decoder + Depth Body)
# ═══════════════════════════════════════════════════════

RKNN_SUPPORTED = {
    # 激活函数
    "activation": ["Relu", "LeakyRelu", "Elu", "HardSigmoid", "HardSwish",
                   "Sigmoid", "Tanh", "Gelu", "Swish", "PRelu"],
    # 卷积
    "conv": ["Conv", "ConvTranspose"],
    # 池化
    "pool": ["AveragePool", "GlobalAveragePool", "MaxPool", "GlobalMaxPool"],
    # 归一化
    "norm": ["BatchNormalization", "InstanceNormalization",
             "LayerNormalization"],  # LayerNorm from v1.5.0+
    # 矩阵运算
    "matmul": ["Gemm", "MatMul"],
    # 变换
    "transform": ["Concat", "Flatten", "Reshape", "Transpose",
                  "Squeeze", "Unsqueeze", "Slice", "Split", "Tile"],
    # 缩减
    "reduce": ["ReduceMax", "ReduceMin", "ReduceMean",
               "ReduceSum"],  # from v1.6.0+
    # 数据移动
    "data": ["Cast", "Clip", "Gather", "Pad",
             "Resize"],  # Resize: nearest/bilinear only
    # 比较
    "compare": ["Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual", "Where"],
    # 数学
    "math": ["Abs", "Exp", "Pow", "Sqrt", "Cos", "Sin"],
    # Special
    "special": ["Softmax", "Dropout", "Identity", "ArgMax", "ArgMin"],
    # 量化
    "quant": ["QuantizeLinear", "DequantizeLinear"],
}

# ═══════════════════════════════════════════════════════
# RKNN 不支持 / 需避免的算子
# ═══════════════════════════════════════════════════════

RKNN_UNSUPPORTED = {
    "einsum": "需用 MatMul + Reshape + Transpose 替代",
    "CumSum": "需用 CPU 端实现",
    "STFT": "只在 CPU 端执行 (Signalsmith DSP)",
    "iSTFT": "只在 CPU 端执行 (Signalsmith DSP)",
    "GridSample": "不支持",
    "GatherND": "不支持",
    "GatherElements": "不支持",
    "OneHot": "不支持",
    "NonZero": "不支持",
    "Log": "不支持 (但 Softmax 内部 log_softmax 可用)",
    "Neg": "不支持 (用 Mul by -1 替代)",
    "Erf": "不支持 (GELU 的 tanh 近似可用)",
    "Mod": "不支持",
    "RandomNormal": "不支持 (采样在 CPU)",
    "Multinomial": "不支持 (采样在 CPU)",
}

# ═══════════════════════════════════════════════════════
# RK3588 特有约束
# ═══════════════════════════════════════════════════════

RK3588_CONSTRAINTS = {
    "sram_per_op": "32KB",             # 单次操作 SRAM 上限
    "max_resolution": "8192×8192",     # 最大 tensor 空间分辨率
    "quantization": ["INT8", "FP16", "Mixed (INT8+FP16)"],
    "w4a16": False,                     # RK3576 only
    "int4_matmul": True,                # int4×int4→int16 on RK3588
    "onnx_opset": "12-19",            # v1.6.0+
    "dynamic_shape": "部分支持",       # 需预编译 max_len, 运行时不可超
    "lstm_batch": 1,                   # LSTM/GRU batch_size 必须为 1
    "resize_modes": ["nearest", "bilinear"],  # 仅这两种
    "fusion_ops": [                     # 自动融合加速
        "Conv-SiLU", "Conv-Swish",
        "Conv-HardSwish", "Conv-HardSigmoid",
        "Conv-GELU", "Conv-Sigmoid",
    ],
}

# ═══════════════════════════════════════════════════════
# RKLLM 约束 (用于 Temporal Body)
# ═══════════════════════════════════════════════════════

RKLLM_INFO = {
    "version": "v1.2.1 (June 2025)",
    "quantization": "w8a8_g128",       # 权重 INT8, 激活 INT8, group_size=128
    "supported_architectures": [
        "LLAMA", "TinyLLAMA", "Qwen2/2.5/3", "Gemma2/3",
        "Phi2/3", "MiniCPM3/4", "InternLM2", "ChatGLM3-6B",
        "DeepSeek-R1-Distill", "RWKV7",
    ],
    "cross_attention": "v1.2.1+",      # 支持交叉注意力
    "multi_batch": True,               # v1.2.1+
    "function_calling": True,          # v1.2.1+
    "conversion": "x86 only",          # 只能在 x86 PC 上做转换
    "max_model_size": "~7B",           # 8GB RAM
    "speed_example": "TinyLlama 1.1B: 10-15 tok/s on NPU",
}

# ═══════════════════════════════════════════════════════
# 我们的算子兼容性检查
# ═══════════════════════════════════════════════════════

# SpectroStream Decoder:
#   nn.Conv2d          → Conv ✓
#   nn.ConvTranspose2d → ConvTranspose ✓
#   F.interpolate      → Resize (nearest) ✓
#   nn.ELU             → Elu ✓
#   F.pad              → Pad ✓
#   .transpose         → Transpose ✓
#   .reshape/.view     → Reshape ✓
#   .squeeze           → Squeeze ✓
#   F.embedding        → Gather ✓

# DepthFormer:
#   nn.Linear          → Gemm/MatMul ✓
#   RMSNorm            → ReduceMean + Mul + Div → 需分解
#   F.softmax          → Softmax ✓
#   torch.matmul       → MatMul ✓
#   .transpose         → Transpose ✓
#   .view/.reshape     → Reshape ✓
#   torch.tanh         → Tanh ✓
#   F.gelu             → Gelu ✓
#   F.cross_entropy    → CPU 端计算
#   Attention Sink     → Concat ✓
#   KV Cache           → 外部 buffer (Slice + Concat) ✓
#   Sampling           → CPU 端 (argmax/topk)

# 潜在问题:
#   1. RMSNorm 需分解为 ReduceMean + Pow + Div + Mul
#      或等待 RKNN 原生支持 (已支持 LayerNorm, RMSNorm 类似)
#   2. 滑动窗口 attention 的 mask 构造需在 CPU 端预计算
#   3. Attention Sink 的 Concat 操作需显式实现
#   4. KV Cache 用 Slice/Concat 更新 (SRAM 32KB 限制)
#   5. GELU 可用 tanh 近似: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
