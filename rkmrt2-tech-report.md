# RK-MRT2 项目技术架构与进度报告

**日期**: 2026-06-30 | **仓库**: [github.com/WildwindCN/RK-MRT2](https://github.com/WildwindCN/RK-MRT2)

---

## 一、项目目标

将 MRT2 (Magenta RealTime 2) 架构移植到 RK3576 NPU 端侧，搭建 **Camera → CLIP → Adapter → Tokens → Decoder → 48kHz 立体声音频** 的实时视觉驱动音乐生成管线。

```
推理管线 (RK3576):
  Camera → CLIP ViT-B/32 (512d) → Adapter → Tokens [T, 12] → Decoder (28MB .rknn) → 48kHz 立体声

训练管线 (8×A800):
  Audio → STFT → Encoder (9.4M, 训练) → RVQ (17.5M, 冻结 MRT2 Small) → Decoder (36.6M, 微调) → Audio
```

---

## 二、模型架构

### 2.1 SpectroStream Encoder (音频 → 瓶颈)

| 属性 | 值 |
|------|-----|
| 输入 | [B, 4, 480, T_stft] |
| 输出 | [B, T/4, 256] |
| 参数 | 9.4M |
| 架构 | 7 级 EncoderResidualUnit 降采样 (ratios: 4× time, 96× freq) |
| 通道 | 4→32→64→64→128→128→128→256→512→256→256 |
| 状态 | **从头训练** (JAX 编码器权重未公开) |

### 2.2 RVQ (连续嵌入 → 离散 Token)

| 属性 | 值 |
|------|-----|
| 码本 | 64 quantizers × 1024 codebook × 256-dim |
| 截断 | 12 层 (rvq_truncation_level=12) |
| 参数 | 17.5M |
| 比特率 | 3.0 kbps |
| 状态 | **冻结** (MRT2 Small checkpoint, token 空间与 MusicCoCa 对齐) |

### 2.3 SpectroStream Decoder (Token → 音频)

| 属性 | 值 |
|------|-----|
| 输入 | [B, T, 256] |
| 输出 | [B, 4, 480, T×4] |
| 参数 | 36.6M |
| 架构 | 7 级 DecoderResidualUnit (ConvTranspose2d + Conv2d × 2/stage) |
| Channel Splits | 2 组 (decoder_0 后分组, decoder_1..6 并行) |
| 状态 | **微调** (MRT2 Small 初始化, 部分层需重训) |

### 2.4 精度验证结果

| 测试 | 结果 |
|------|------|
| iSTFT 往返 SNR | **106.1 dB** (近无损) |
| 随机权重管线 SNR | **~0 dB** (确认无恒等捷径) |
| NaN/Inf 检查 | **0 处** (全部 6 阶段通过) |
| 编码器形状 | [B,4,480,T] → [B,T/4,256] ✅ |
| 解码器形状 | [B,T,256] → [B,4,480,T×4] ✅ |
| RVQ 权重加载 | [64,1024,256] ✅ |
| 端到端结构 | 全部跨组件形状兼容 ✅ |

---

## 三、RK3576 部署状态

### 3.1 RKNN 转换 (全部完成)

| 模型 | 大小 | 输入 | 状态 |
|------|------|------|------|
| `depth_body.rknn` | 32 MB | [1,12,1024] | ✅ |
| `codec_decoder.rknn` | 28 MB | [1,T,256] | ✅ |
| `temporal_body.rknn` | 360 MB | 27 inputs | ✅ |
| **总计** | **420 MB** | | **3/3** |

### 3.2 Temporal Body 优化

- KV 窗口: 512 → 42 位置 (41 滑动窗 + 1 attention sink)
- 外部 attention mask 输入 [1,1,1,44]
- 节点数 4595 → 1895 (ONNX simplify)
- C++ Runtime: ring buffer (capacity=512) + window extract/merge

### 3.3 Decoder 部署 (待训练)

- 当前: Decoder 28MB .rknn 已转换 (旧架构)
- 新架构 36.6M 参数, 训练后重新导出 ONNX → RKNN
- 预计 RKNN 大小: ~29 MB (FP16)

---

## 四、JAX 对齐状态

### 4.1 已对齐

| 组件 | 对齐度 | 说明 |
|------|--------|------|
| STFT/iSTFT | 100% | Hann 窗, frame=960, step=480, fft=960 |
| Encoder 架构 | 100% | 7 级 Conv2d 降采样, 因果时间填充, 对称频率填充 |
| RVQ 量化 | 100% | 距离公式 + straight-through estimator |
| Decoder 架构 | 100% | 2 convs/stage + channel_splits=2 |
| Decoder 填充 | 100% | ConvTranspose 手动 causal/symmetric 裁剪 |
| Config 参数 | 100% | 全部值对齐 JAX `stft_spectrostream_40ms_generic_48khz_stereo_config` |

### 4.2 已知差异

| 差异 | 影响 | 原因 |
|------|------|------|
| Encoder channel_splits: 1×1 Conv vs ParallelChannels | 功能等价, 参数量略增 131K | JAX 编码器权重未公开, 从头训练 |
| Decoder 权重仅 6/47 可加载 | 需重训 | 新架构 (2 convs/stage + channel_splits) vs 旧 (1 conv/stage 无 split) |

---

## 五、训练准备 (待数据就绪)

### 5.1 数据需求

| 属性 | 值 |
|------|------|
| 来源 | ACE Studio 合成音频 |
| 时长 | 200 小时 |
| 格式 | 48kHz 立体声 |
| 片段 | 10s (480,000 samples) |
| 增强 | 随机增益 ±3dB |

### 5.2 训练配置

| 参数 | 值 |
|------|------|
| GPU | 8×A800 (Slurm) |
| Batch | 8/GPU × 8 = 64 |
| 优化器 | AdamW (lr=3e-4, beta=0.8/0.99) |
| 调度 | Cosine warmup 5k + Cosine decay |
| 精度 | bf16 |
| Phase 1 | epochs 1-50: 冻结 Decoder, 只训练 Encoder |
| Phase 2 | epochs 51-200: 解冻 Decoder, 联合微调 (lr=1.5e-4) |
| 预计耗时 | 50-70 小时 |

### 5.3 损失函数

```
L_total = L1(waveform, reconstructed)
        + Σ_i L1(|STFT_i(waveform)|, |STFT_i(reconstructed)|)  (3 scales: 512/1024/2048)
        + 0.01 × MSE(encoder_embeddings, quantized_embeddings.detach())
```

---

## 六、技术决策记录

1. **固定 KV 窗口 (42)**: sliding window=41 + sink=1, C++ ring buffer (512) 管理长历史
2. **外部 Attention Mask**: [1,1,1,44], C++ runtime 预计算, 解决早期帧 padding + 窗口组合
3. **Cross-Attention 内部计算**: cond 固定, K/V 在 NPU 内每帧重算 (避免额外 I/O)
4. **冻 RVQ 训 codec**: 保持 token 空间与 MRT2/MusicCoCa 对齐, Adapter 对比学习目标不变
5. **1×1 Conv 替代 ParallelChannels**: 编码器权重从头训练, 功能等价, 简化实现
6. **bf16 训练 + float32 STFT**: STFT/iSTFT 在 autocast 下不兼容, 强制浮点精度

---

## 七、下一步

| 任务 | 优先级 | 依赖 |
|------|--------|------|
| ACE Studio 数据就绪 | **阻塞** | — |
| 编码器训练 (Phase 1) | 高 | 数据就绪 |
| 解码器微调 (Phase 2) | 高 | Phase 1 完成 |
| Decoder ONNX 导出 + RKNN 转换 | 高 | 训练完成 |
| Temporal Body RKNN 板端验证 | 中 | RK3576 硬件 |
| Adapter 对比学习训练 | 中 | Decoder 训练完成 |
| 端到端管线集成测试 | 中 | Adapter + Decoder 就绪 |
