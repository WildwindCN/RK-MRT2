"""SpectroStream Encoder —— PyTorch 实现 (JAX 严格对齐)

将 STFT 特征编码为瓶颈嵌入 (25Hz frame rate)。

架构严格对齐 JAX spectrostream_encoder_config:
- 时间填充: causal (仅左侧, 无未来帧)
- 频率填充: symmetric (JAX pad_freq 公式)
- 输出投影: Flatten + 1x1 Conv residual (dense 1280→256 per timestep)

参考: magenta_rt/jax/spectrostream.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderResidualUnit(nn.Module):
    """下采样残差单元 (严格对齐 JAX conv2d_residual_unit(transposed=False))

    主路径:
      ELU → Conv2d(3×3, s=1) → ELU → Conv2d(k=max(3,2s), s=s)
    快捷方式:
      AvgPool2d(s) [+ causal time pad] + Conv2d(1×1) [if needed]
    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]):
        super().__init__()
        self.stride = stride  # (freq_stride, time_stride) in PyTorch [B,C,H=freq,W=time]
        sf, st = stride  # freq stride, time stride
        kernel_f = max(3, 2 * sf)
        kernel_t = max(3, 2 * st)

        # Conv1: 3×3, stride=1 (pre-activated)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                               stride=1, padding=0, bias=True)
        # Conv2: resample kernel, stride=(sf, st) (pre-activated)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=(kernel_f, kernel_t),
                               stride=stride, padding=0, bias=True)

        # Shortcut
        need_shortcut = (stride != (1, 1)) or (in_channels != out_channels)
        if need_shortcut:
            sc = []
            if stride != (1, 1):
                sc.append(nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0))
            if in_channels != out_channels:
                sc.append(nn.Conv2d(in_channels, out_channels, 1, bias=True))
            self.shortcut = nn.Sequential(*sc)
        else:
            self.shortcut = nn.Identity()

    def _causal_time_pad(self, x, pad_left):
        """时间维度因果填充: 仅左侧"""
        return F.pad(x, (pad_left, 0, 0, 0)) if pad_left > 0 else x

    def _symmetric_freq_pad(self, x, kernel_size, stride):
        """频率维度对称填充: JAX pad_freq 公式"""
        pad = max((kernel_size - 1) * 1 + 1 - stride, 0)
        return F.pad(x, (0, 0, pad // 2, pad - pad // 2)) if pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H=freq, W=time]
        residual = x

        # Conv1: ELU → causal time pad(2 left) → symmetric freq pad(1,1) → conv
        x = F.elu(x)
        x = self._causal_time_pad(x, 2)         # 3×3 kernel, causal: kt-1=2 left
        x = self._symmetric_freq_pad(x, 3, 1)    # 3×3 kernel, stride=1: pad=(3-1+1-1)=2 → (1,1)
        x = self.conv1(x)

        # Conv2: ELU → causal time pad → symmetric freq pad → strided conv
        x = F.elu(x)
        kf, kt = self.conv2.kernel_size
        sf, st = self.stride
        x = self._causal_time_pad(x, kt - 1)     # causal: kt-1 left
        x = self._symmetric_freq_pad(x, kf, sf)  # JAX pad_freq formula
        x = self.conv2(x)

        # Shortcut with causal time padding before AvgPool
        if self.stride != (1, 1) and hasattr(self, 'shortcut') and len(self.shortcut) > 0:
            sc = residual
            if isinstance(self.shortcut[0], nn.AvgPool2d):
                # JAX: AvgPool2D with time_padding='semicausal'
                pool_kernel = self.shortcut[0].kernel_size
                sc = self._causal_time_pad(sc, pool_kernel[1] - 1)
            sc = self.shortcut(sc)
        else:
            sc = self.shortcut(residual)

        # Match shapes
        mt = min(x.shape[2], sc.shape[2])
        mf = min(x.shape[3], sc.shape[3])
        return x[:, :, :mt, :mf] + sc[:, :, :mt, :mf]


class SpectroStreamEncoder(nn.Module):
    """SpectroStream 编码器 (严格 JAX 架构对齐)

    输入: [B, 4, 480, T_stft]
    输出: [B, T_stft/4, 256]

    通道追踪 (对齐 JAX):
      base_conv:   4  → 32   (k=7×7, s=1)
      enc_0:      32  → 64   (s=1×2, mult=2)
      enc_1:      64  → 64   (s=1×2, mult=1)
      enc_2:      64  → 128  (s=1×3, mult=2)
      enc_3:     128  → 128  (s=1×2, mult=1)
      enc_4:     128  → 128  (s=1×2, mult=1)
      enc_5:     128  → 256  (s=2×2, mult=2)
      expand:    256  → 512  (1×1 Conv, 替代 JAX ParallelChannels)
      enc_6:     512  → 256  (s=2×1, mult=1)
      bottleneck:256  → 256  (s=1×1)
      output:    1280 → 256  (Flatten freq 5×256=1280 → 1×1 Conv residual)
    """

    def __init__(self, config):
        super().__init__()
        base_depth = config.encoder_base_conv_depth   # 32
        base_size = config.encoder_base_conv_size      # 7
        ratios = list(config.ratios)    # ((1,2),(1,2),(1,3),(1,2),(1,2),(2,2),(2,1))
        mults = list(config.mults)      # (2,1,2,1,1,2,1)
        channel_splits = config.channel_splits          # 2
        channel_recombo_block = config.channel_recombo_block  # -2
        num_channels = config.num_channels              # 4
        num_features = config.num_features              # 256

        num_blocks = len(ratios)  # 7
        # JAX: num_blocks = len(ratios) + 1 = 8, block %= 8, -2→6
        if channel_recombo_block < 0:
            channel_recombo_block = (num_blocks + 1) + channel_recombo_block  # 6

        split_at = channel_recombo_block  # 6 → encoder_6 is post-split

        # Base conv: 4→32, k=7×7, causal+symmetric padding
        self.base_conv = nn.Conv2d(
            num_channels, base_depth,
            kernel_size=(base_size, base_size), stride=1, padding=0, bias=True)

        # Pre-split: encoder_0..5
        in_ch = base_depth
        out_ch = base_depth
        self.pre_stages = nn.ModuleList()
        for i in range(split_at):
            out_ch = int(round(out_ch * mults[i]))
            rt, rf = ratios[i]  # JAX: (time_ratio, freq_ratio)
            self.pre_stages.append(EncoderResidualUnit(in_ch, out_ch, stride=(rf, rt)))
            in_ch = out_ch

        # Channel expansion: 1×1 Conv(256→512) 替代 JAX ParallelChannels(group=2)
        # 由于从头训练, 功能等价; 权重加载不可用(JAX encoder 权重未公开)
        if channel_splits and channel_splits > 1:
            self.channel_expand = nn.Conv2d(out_ch, out_ch * channel_splits, 1, bias=True)
            in_ch = out_ch * channel_splits
        else:
            self.channel_expand = nn.Identity()

        # Post-split: encoder_6 (JAX: output_channels 在 split 前计算)
        out_ch = int(round(out_ch * mults[split_at]))
        rt6, rf6 = ratios[split_at]
        self.post_stages = nn.ModuleList()
        self.post_stages.append(EncoderResidualUnit(in_ch, out_ch, stride=(rf6, rt6)))
        in_ch = out_ch

        # Bottleneck: stride=(1,1), no channel change
        self.bottleneck = EncoderResidualUnit(out_ch, out_ch, stride=(1, 1))

        # Output projection: JAX Flatten freq→channels → 1×1 Conv residual
        # [B, C=256, H=freq=f, W=time=t] → flatten freq: [B, C*H, W] = [B, 1280, t]
        # Main: Conv1d(1280→256)  per timestep
        # Shortcut: ELU→Conv1d(1280→1280)→ELU→Conv1d(1280→256)
        freq_bins = config.num_bins // config.total_freq_stride  # 480/96 = 5
        flat_channels = freq_bins * out_ch  # 5*256 = 1280

        # Use Conv2d as 1D conv: [B, C, W, 1]
        self.output_main = nn.Conv2d(flat_channels, num_features, 1, bias=True)
        self.output_shortcut = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(flat_channels, flat_channels, 1, bias=True),
            nn.ELU(),
            nn.Conv2d(flat_channels, num_features, 1, bias=True),
        )

    def _causal_time_pad(self, x, pad_left):
        return F.pad(x, (pad_left, 0, 0, 0)) if pad_left > 0 else x

    def _symmetric_freq_pad(self, x, kernel_size, stride):
        pad = max((kernel_size - 1) + 1 - stride, 0)
        return F.pad(x, (0, 0, pad // 2, pad - pad // 2)) if pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 4, 480, T_stft]

        # Base conv: causal time pad (6 left) + symmetric freq pad (3,3)
        ks_h, ks_w = self.base_conv.kernel_size  # (7, 7) → H=freq, W=time
        x = self._causal_time_pad(x, ks_w - 1)        # time: 6 left
        x = self._symmetric_freq_pad(x, ks_h, 1)      # freq: (7-1+1-1)=6 → (3,3)
        x = self.base_conv(x)

        # Pre-split stages
        for stage in self.pre_stages:
            x = stage(x)

        # Channel expansion
        x = self.channel_expand(x)

        # Post-split stages
        for stage in self.post_stages:
            x = stage(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Output: JAX Flatten freq → 1×1 Conv residual
        # x: [B, C=256, H=freq=5, W=time]
        B_s, C_s, H_f, W_t = x.shape

        # Flatten freq into channels: [B, C*H, W] → unsqueeze → [B, C*H, W, 1]
        flat = x.reshape(B_s, C_s * H_f, W_t).unsqueeze(-1)

        # Main: 1×1 conv (dense 1280→256 per timestep)
        main = self.output_main(flat).squeeze(-1)  # [B, 256, time]

        # Shortcut: two 1×1 convs with ELU
        sc = self.output_shortcut(flat).squeeze(-1)  # [B, 256, time]

        # [B, C, T] → [B, T, C]
        return (main + sc).transpose(1, 2)
