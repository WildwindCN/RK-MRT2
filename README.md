# RK-MRT2

基于 [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) (Apache 2.0) 架构规格的 PyTorch 实现，支持加载 MRT2 Small 预训练权重，目标 **RK3588 NPU** (6 TOPS) 端侧实时部署。

## 验证状态

```
全链路验证: 7/7 PASS
  [PASS] 模型架构 — 6 组件前向传播无 NaN
  [PASS] iSTFT 往返 — SNR 101.1 dB
  [PASS] ONNX 导出 — cos=1.00000000 (Depth Body + Codec Decoder)
  [PASS] 权重转换 — 268/268 DepthFormer 参数从 MRT2 Small 加载, 0 missing
  [PASS] 训练对齐 — 0 forbidden ONNX ops
  [PASS] 端到端生成 — 48kHz 立体声, RTF=2.74x (CPU), 无 NaN
  [PASS] 逐层验证 — 所有参数加载前后可变, 非随机初始化
```

运行验证: `python verify_all.py --checkpoint <path_to_mrt2_small.safetensors>`

## 架构

```
MIDI ──► [Encoder] ──► Conditioning (256-dim)
                           │
Token ──► [Temporal Body] ──► [Depth Body] ──► Logits ──► Sample
  ↑        12层, d=1024        2层, d=768      12×12294       │
  │        滑动窗口 41帧        RVQ 因果AR        combined      │
  │        NoPE + Sink          NoPE             vocab         │
  │        NPU FP16             NPU/CPU FP32                   │
  │                                                           ▼
  └─────────────────── RVQ Tokens (per-RVQ) ◄─────────────────┘
                              │
              [RVQ Embedding] │  lookup 64×1024×256, sum 12 层
                              ▼
              [Codec Decoder] ──► STFT features ──► [iSTFT CPU] ──► 48kHz PCM
                7-stage ConvTranspose    [1,4,480,T×4]    Signalsmith DSP
                NPU FP16/INT8
```

三组件 NPU + CPU 混合部署，ONNX 图独立导出，RKNN 转换。

## 模型规格

| 组件 | 层数 | 维度 | 头数 | 参数量 | 部署位置 |
|---|---|---|---|---|---|
| Temporal Body | 12 | d=1024 | 8 (128/head) | 182.6M | NPU FP16 |
| Depth Body | 2 | d=768 | 6 (128/head) | 24.4M | NPU/CPU FP32 |
| Codec Decoder | 7-stage | — | — | 30.5M | NPU FP16/INT8 |
| RVQ Embedding | 64×1024×256 | — | — | 16.8M | CPU |
| Token Embedding | 12294×1024 | — | — | 12.6M | NPU |
| **总计 (含 Encoder)** | | | | **~250M** | |

## 项目结构

```
RK-MRT2/
├── models/                        # PyTorch 模型
│   ├── config.py                  # DepthFormerConfig, SpectroStreamConfig, ModelSpec
│   ├── transformer.py             # RMSNorm, SlidingWindowAttention, CrossAttention, GatedFFN
│   ├── depthformer.py             # DepthFormer, TemporalBodyStateful, DepthBodyAR
│   ├── spectrostream.py           # SpectroStreamDecoder, RVQEmbedding, DecoderStage
│   └── istft.py                   # iSTFT (SNR 101.1 dB 验证)
├── demo/
│   └── generate.py                # 端到端生成 Demo: MIDI → WAV
├── export/                        # ONNX 导出 + RKNN 转换
│   ├── export_onnx.py             # 三图分离导出 (Temporal/Depth/Codec)
│   ├── verify_onnx.py             # 导出+精度验证 (PyTorch vs ONNX)
│   ├── convert_rknn.py            # RKNN FP16/INT8 转换 (需 x86 Linux)
│   └── rknn_config.py             # 各 graph RKNN 编译参数
├── weights/
│   └── convert_mrt2_weights.py    # JAX safetensors → PyTorch state_dict
├── spec/
│   └── training_inference_spec.py # 训练→推理对齐规范 + 验证工具
├── runtime/                       # C++ 推理运行时 (RK3588 板端)
│   ├── rknn_model.hpp             # RKNPU2 RAII Wrapper
│   ├── kv_cache.hpp               # Ring Buffer KV Cache (12 层)
│   ├── midi_parser.hpp            # MIDI → 128-dim Pianoroll
│   ├── inference_engine.hpp       # 推理循环编排
│   ├── demo/main.cpp              # 实时生成入口
│   └── CMakeLists.txt             # ARM NEON + RKNPU2 构建
├── docs/
│   └── rknn_constraints.py        # RKNN/RKLLM 算子兼容性参考
├── tests/
│   └── test_models.py             # 模型架构单元测试
├── verify_all.py                  # 全链路严谨验证 (7 项)
├── README.md
└── .gitignore
```

## 快速开始

### 环境要求

- Python 3.10+ / PyTorch 2.x / safetensors / onnx + onnxruntime
- MRT2 Small 权重: HuggingFace `google/magenta-realtime-2` 或本地 `~/Documents/Magenta/`
- WSL 用户: 需配代理 `export https_proxy=http://<Windows_IP>:7890`

### 1. 全链路验证

```bash
python verify_all.py --checkpoint ~/Documents/Magenta/magenta-rt-v2/checkpoints/mrt2_small.safetensors
```

### 2. 权重转换

```bash
# Windows / WSL 均可 (无需 JAX)
python weights/convert_mrt2_weights.py \
    --checkpoint ~/Documents/Magenta/magenta-rt-v2/checkpoints/mrt2_small.safetensors \
    --output ./exported/weights/mrt2_small_pytorch.pt \
    --verify
```

### 3. 端到端生成

```bash
python demo/generate.py \
    --weights ./exported/weights/mrt2_small_pytorch.pt \
    --duration 10 --pattern chord \
    --temperature 0.8 --output music.wav
```

### 4. 训练产出验证

```bash
python spec/training_inference_spec.py --checkpoint /path/to/your_training_output.pt
```

### 5. ONNX 导出

```bash
python export/export_onnx.py --output_dir ./exported
python export/verify_onnx.py       # 精度验证 (cos > 0.9999)
```

### 6. RKNN 转换 (需要 x86 Linux + RKNN-Toolkit2)

```bash
python export/convert_rknn.py --all --precision fp16
python export/convert_rknn.py --graph codec_decoder --precision int8 --create_calib
```

### 7. C++ 运行时 (RK3588 板端)

```bash
cd runtime && mkdir build && cd build
cmake .. -DRKNN_SDK_PATH=/path/to/rknpu2
make -j$(nproc)
./rk_mrt2_demo --temporal temporal_body.rknn --depth depth_body.rknn \
               --codec codec_decoder.rknn --output audio.wav
```

## ONNX 导出规格

| Graph | 输入 | 输出 | 节点数 | 大小 (FP32) |
|---|---|---|---|---|
| Temporal Body | x + cond + 24 KV cache tensors | out + 24 updated KV | ~2000 | 731 MB |
| Depth Body | [B, 12, 1024] | [B, 12, 12294] | 269 | 63 MB |
| Codec Decoder | [B, T, 256] | [B, 4, 480, T×4] | 672 | 55 MB |

## RKNN 兼容性

**已规避的算子:** einsum, CumSum, STFT/iSTFT (CPU), GridSample, GatherND, OneHot

**使用中的关键算子:** Conv2D, ConvTranspose2D, MatMul, LayerNormalization, Softmax, GELU, ELU, Reshape, Transpose, Concat, Slice, Pad, ReduceMean

**Attention Sink / KV Cache:** 作为显式 I/O 张量，CPU 端管理 ring buffer，满足 RKNN 静态图约束。

## 延迟预算 (40ms/帧 @ 25Hz)

| 步骤 | 设备 | 估算 |
|---|---|---|
| Temporal Body ×1 | NPU FP16 | < 5ms |
| Depth Body AR ×12 | NPU/CPU | < 10ms |
| Codec Decoder | NPU FP16 | < 3ms |
| iSTFT + Sampling | CPU | < 2ms |
| **总计** | | **< 20ms** |

CPU 实测 RTF: 2.74x (NPU 上预计 < 0.2x)。

## Token 格式

```
训练数据: per-RVQ 索引 [0, 1023]
模型内部: 全词表索引 = num_reserved(6) + rvq_idx × 1024 + local_token
SOS: 0 | 全词表: 12 × 1024 + 6 = 12294
每层有效范围: [6 + rvq×1024, 6 + (rvq+1)×1024)
```

## 已知限制

- **SpectroStream Decoder**: 当前架构与 JAX 版不完全对齐 (10/18 卷积层匹配), Decoder 需重训
- **Encoder**: 使用简化的 Linear 投影, 未加载 MusicCoCa 权重
- **KV Cache**: PyTorch 推理为 O(N²) 全序列模式, 非 stateful 单帧推理
- **RKNN 实测**: 需 RK3588 硬件 + x86 Linux 构建机

## 参考

- [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) — 架构规格 (Apache 2.0)
- [iPhone MRT2 移植](https://github.com/mattmireles/magenta-realtime-2-iphone) — 三图拆分方案参考
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) — ONNX → RKNN 转换
- [RKLLM](https://github.com/airockchip/rknn-llm) — RK NPU LLM 推理框架 (支持 cross-attention 自 v1.2.1)
- [Signalsmith DSP](https://github.com/signalsmith-audio/dsp) — iSTFT C++ 实现 (MIT)

## 许可证

MIT
