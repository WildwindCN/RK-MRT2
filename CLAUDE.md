# CLAUDE.md

此文件为 Claude Code 在 RK-MRT2 仓库中工作提供指引。

## 项目目标

将 MRT2 (Magenta RealTime 2) 架构移植到 RK3588 NPU 端侧部署。完整管线：MIDI → DepthFormer AR Transformer → SpectroStream Codec Decoder → 48kHz 立体声音频。

## 环境

- **Windows (主开发)**: Python 3.13, torch 2.x, safetensors, onnx+onnxruntime
- **WSL (RKNN 转换)**: Ubuntu 22.04, Python 3.10, torch 2.4.0+cpu, onnx<1.17, rknn-toolkit2==2.3.2
- **代理**: WSL 内 `export https_proxy=http://172.27.224.1:7890`

## 关键命令

```bash
# 全链路验证
python verify_all.py --checkpoint ~/Documents/Magenta/magenta-rt-v2/checkpoints/mrt2_small.safetensors

# 模型测试
python tests/test_models.py

# 权重转换
python weights/convert_mrt2_weights.py --checkpoint <path> --output <out.pt> --verify

# 端到端生成 (需权重)
python demo/generate.py --weights <weights.pt> --duration 10 --output music.wav

# ONNX 导出
python export/verify_onnx.py

# RKNN 转换 (WSL 中)
wsl -d Ubuntu-22.04
export https_proxy=http://172.27.224.1:7890
cd /mnt/d/workspace/TwiddleX/RK-MRT2
python3 deploy/convert_all.py --graph depth_body --precision fp16
```

## 当前进度 (2026-06-29)

| Phase | 状态 |
|---|---|
| 1a. SpectroStream Decoder PyTorch | 完成 |
| 1b. DepthFormer Transformer PyTorch | 完成 |
| 1c. 权重转换 (JAX→PyTorch) | 完成 — 268/268 参数 0 missing |
| 2. ONNX 导出 | 完成 — 3 图导出, Temporal Body cos=1.00000000 |
| 3. RKNN 转换 (RK3576) | **完成** — 全部 3 图 FP16 已产出 (420MB total) |
| 4. C++ Runtime | KV cache + mask 预计算完成, 待板端编译 |

**RKNN 模型产出 (RK3576)**:
| Graph | 大小 | 输入 | 说明 |
|---|---|---|---|
| Depth Body | 32 MB | [1,12,1024] | 2-layer RVQ AR |
| Codec Decoder | 28 MB | [1,T,256] | 7-stage ConvTranspose |
| Temporal Body | 360 MB | 27 inputs | 12-layer stateful, 42-pos window + attn mask |

**关键验证结果 (7/7 PASS)**:
- iSTFT SNR 101.1 dB
- ONNX cos=1.00000000 (Temporal Body 3/3 tests: single-frame + stateful loop + cross-attn)
- DepthFormer 268/268 参数从 checkpoint 加载
- 端到端 RTF=2.74x (CPU)

**Temporal Body ONNX 优化**:
- KV cache: 512→42 位置 (41 window + 1 sink)
- 输入: 26→27 (新增 attn_mask [1,1,1,44])
- 动态轴: 全部移除, 节点 4595→1895
- ONNX simplify: 消除 Cast/Shape/Gather/Where 等动态 op

**已知待办**:
- Decoder 需重训 (当前架构与 JAX 不完全对齐, 10/18 卷积层匹配)
- RK3576 板端实测 (需 librknnrt.so + C++ 编译)
- 端到端 RKNN vs ONNX 数值精度对比

## 文件结构

```
RK-MRT2/
├── models/           # PyTorch 模型 (~250M 参数)
│   ├── config.py, transformer.py, depthformer.py, spectrostream.py, istft.py
├── demo/generate.py  # 端到端生成
├── export/           # ONNX 导出 + RKNN 转换
├── weights/          # JAX→PyTorch 权重转换
├── deploy/           # RK3588 板端部署流水线
├── spec/             # 训练-推理对齐规范
├── runtime/          # C++ 推理运行时
├── tests/            # 测试
├── verify_all.py     # 全链路验证 (7 项)
└── rknn_models/      # 产出: *.rknn
```

## RKNN 环境

- 用 `pip install rknn-toolkit2` (PyPI 有 manylinux wheel)
- 需要 torch 2.4.0 CPU + onnx<1.17
- RKNN-Toolkit2 不支持 onnx.mapping (v1.17+ 移除), 需降级 onnx
- 动态 batch 需用 `input_size_list` 固定形状
- `float_dtype` 控制精度 (非 `output_tensor_type`)
