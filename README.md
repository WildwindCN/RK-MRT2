# RK-MRT2: MRT2 架构 PyTorch 实现 → RK3588 NPU 部署

基于 [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) (Apache 2.0) 架构规格的 PyTorch 参考实现，目标 RK3588 NPU (6 TOPS) 端侧部署。

## 架构

```
MIDI ──► [Encoder] ──► Conditioning (256-dim)
                           │
Token ──► [Temporal Body] ──► [Depth Body] ──► Logits ──► Sample
  ↑        12层, d=1024        2层, d=768       12×1030       │
  │        滑动窗口 41帧        RVQ 因果AR                     │
  │        NPU FP16             NPU FP16 / CPU FP32           │
  │                                                           ▼
  └──────────────────── RVQ Tokens ◄──────────────────────────┘
                              │
              [RVQ Embedding] │  lookup + sum
                              ▼
              [Codec Decoder] ──► STFT features ──► [iSTFT CPU] ──► 48kHz PCM
                7-stage ConvTranspose               Signalsmith DSP
                NPU FP16/INT8
```

三组件 NPU + CPU 混合部署，ONNX 图独立导出，RKNN 转换。

## 项目结构

```
RK-MRT2/
├── models/                # PyTorch 参考实现 (~246M 参数)
│   ├── config.py          # DepthFormerConfig, SpectroStreamConfig
│   ├── transformer.py     # 注意力/FFN/归一化 (ONNX 兼容, 无 einsum)
│   ├── depthformer.py     # TemporalBody + DepthBody + 完整 DepthFormer
│   └── spectrostream.py   # SpectroStream Decoder + RVQ Embedding
├── export/                # ONNX 导出 + RKNN 转换
│   ├── export_onnx.py     # 三图分离导出
│   ├── verify_onnx.py     # 导出验证 (PyTorch vs ONNX)
│   ├── convert_rknn.py    # RKNN FP16/INT8 转换
│   └── rknn_config.py     # RKNN 编译参数
├── runtime/               # C++ 推理运行时 (RK3588)
│   ├── rknn_model.hpp     # RKNPU2 RAII Wrapper
│   ├── kv_cache.hpp       # Ring Buffer KV Cache Manager
│   ├── midi_parser.hpp    # MIDI → Pianoroll
│   ├── inference_engine.hpp  # 推理循环编排
│   ├── demo/main.cpp      # 实时生成 Demo
│   └── CMakeLists.txt     # 构建配置
├── tests/                 # 测试
│   └── test_models.py     # 模型架构验证
└── docs/                  # 文档
    └── rknn_constraints.py  # RKNN 算子兼容性
```

## 快速开始

### 1. PyTorch 模型验证

```bash
cd RK-MRT2
python tests/test_models.py
```

### 2. ONNX 导出

```bash
python export/verify_onnx.py    # 导出 + 精度验证 (cos > 0.9999)
```

### 3. RKNN 转换 (需要 x86 Linux + RKNN-Toolkit2)

```bash
# FP16 转换
python export/convert_rknn.py --all --precision fp16

# INT8 量化 (需要校准数据)
python export/convert_rknn.py --graph codec_decoder --precision int8 --create_calib
```

### 4. C++ 运行时 (RK3588 板端)

```bash
cd runtime
mkdir build && cd build
cmake .. -DRKNN_SDK_PATH=/path/to/rknpu2
make -j$(nproc)
./rk_mrt2_demo --temporal temporal_body.rknn --depth depth_body.rknn \
               --codec codec_decoder.rknn --midi input.mid --output audio.wav
```

## 模型规格

| 组件 | 层数 | 维度 | 头数 | 参数量 | 部署位置 |
|---|---|---|---|---|---|
| Temporal Body | 12 | d=1024 | 8 | 182.6M | NPU FP16 |
| Depth Body | 2 | d=768 | 6 | 15.7M | NPU/CPU |
| Codec Decoder | 7-stage | — | — | 30.5M | NPU FP16/INT8 |
| RVQ Embedding | 64×1024×256 | — | — | 16.8M | CPU |
| **总计** | | | | **~246M** | |

## RKNN 兼容性

- 无 einsum (全部 MatMul + Reshape + Transpose)
- 无 CumSum、无 STFT (iSTFT 走 CPU)
- 滑动窗口注意力 + Attention Sink
- KV Cache 作为显式 I/O (静态图兼容)
- 采样/Gather/码本查找在 CPU 端

## 延迟预算 (40ms/帧 @ 25Hz)

| 步骤 | 估算 |
|---|---|
| Temporal Body (NPU) | < 5ms |
| Depth Body AR ×12 (NPU/CPU) | < 10ms |
| Codec Decoder (NPU) | < 3ms |
| iSTFT + 采样 (CPU) | < 2ms |
| **总计** | **< 20ms** |

## 参考

- [Magenta RealTime 2](https://github.com/magenta/magenta-realtime) — 架构规格 (Apache 2.0)
- [iPhone MRT2 移植](https://github.com/mattmireles/magenta-realtime-2-iphone) — 三图拆分方案
- [RKNN-Toolkit2](https://github.com/airockchip/rknn-toolkit2) — ONNX → RKNN 转换
- [RKLLM](https://github.com/airockchip/rknn-llm) — RK NPU LLM 推理框架
- [Signalsmith DSP](https://github.com/signalsmith-audio/dsp) — iSTFT C++ 实现 (MIT)
- [SHARD (EuroMLSys '26)](https://eurosys.org/) — NPU SRAM tiling 方法

## 许可证

MIT
