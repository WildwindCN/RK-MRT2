"""SpectroStream Decoder —— PyTorch 实现 (JAX 严格对齐)

将 RVQ 离散 Token 解码为 STFT 频域表示。
iSTFT 在 CPU 端完成。

架构严格对齐 JAX spectrostream_decoder_config + conv2d_residual_unit(transposed=True):
- 每级 2 个卷积: ConvTranspose(k=max(3,2s), s=s) + Conv2d(3×3, s=1)
- ConvTranspose: 无预填充, padding=0, 输出后手动 causal/symmetric 裁剪
- Channel splits (ParallelChannels): decoder_0 后分 2 组并行
- Shortcut: Upsample(nearest) + Conv2d(1×1) [if needed]

参考: magenta_rt/jax/spectrostream.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpectroStreamConfig


class RVQEmbedding(nn.Module):
    """RVQ Token → Embedding (冻结, MRT2 Small 权重)"""

    def __init__(self, config: SpectroStreamConfig):
        super().__init__()
        self.num_quantizers = config.num_quantizers
        self.codebook_size = config.codebook_size
        self.embedding_dim = config.embedding_dim
        self.truncation_level = config.rvq_truncation_level
        self.embedding = nn.Parameter(
            torch.randn(config.num_quantizers, config.codebook_size, config.embedding_dim) * 0.02
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        B, T, Q = codes.shape
        quantized = torch.zeros(B, T, self.embedding_dim, device=codes.device, dtype=self.embedding.dtype)
        for i in range(Q):
            idx = codes[:, :, i].long().clamp(0, self.codebook_size - 1)
            quantized = quantized + F.embedding(idx, self.embedding[i])
        return quantized


class DecoderResidualUnit(nn.Module):
    """解码器残差单元 (严格对齐 JAX conv2d_residual_unit(transposed=True))

    每级 2 个预激活卷积:
      ELU → ConvTranspose(k=max(3,2s), s=s) [if strided] 或 Conv2d(3×3, s=1)
      ELU → Conv2d(3×3, s=1)  [始终添加]

    Shortcut:
      Upsample(nearest, scale=s) + Conv2d(1×1) [if channels differ]
    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int]):
        super().__init__()
        st, sf = stride  # time_ratio, freq_ratio
        kernel_t = max(3, 2 * st)
        kernel_f = max(3, 2 * sf)

        # Conv1: strided (transposed) or 3×3
        if stride != (1, 1):
            self.conv1 = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=(kernel_t, kernel_f),
                stride=stride, padding=0, bias=True,
            )
            self.k1_t, self.k1_f = kernel_t, kernel_f
            self.s1_t, self.s1_f = st, sf
            self.is_transposed = True
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=1, padding=0, bias=True)
            self.is_transposed = False

        # Conv2: always 3×3, stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=0, bias=True)

        # Shortcut: Upsample + Conv1x1
        need_sc = (stride != (1, 1)) or (in_channels != out_channels)
        if need_sc:
            sc = []
            if in_channels != out_channels:
                sc.append(nn.Conv2d(in_channels, out_channels, 1, bias=True))
            if stride != (1, 1):
                sc.append(nn.Upsample(scale_factor=stride, mode='nearest'))
            self.shortcut = nn.Sequential(*sc)
        else:
            self.shortcut = nn.Identity()

        self.activation = nn.ELU()

    def _causal_pad_time(self, x, pad_left):
        """时间维度因果填充 (仅左侧)"""
        if pad_left <= 0:
            return x
        return F.pad(x, (0, 0, pad_left, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T=time, F=freq]
        residual = x

        # Conv1: ELU → pad → conv (transposed or regular)
        x = self.activation(x)
        if self.is_transposed:
            # ConvTranspose2d with padding=0. Output = (T_in-1)*st + kt
            # Manually crop for causal time + symmetric freq.
            x = self.conv1(x)
            # Causal trim on time (left side only): remove kt-st from left
            trim_t = self.k1_t - self.s1_t
            x = x[:, :, trim_t:, :]
            # Symmetric trim on freq: remove (kf-sf)//2 from each side
            trim_f = self.k1_f - self.s1_f
            x = x[:, :, :, trim_f // 2 : -(trim_f - trim_f // 2) or None]
        else:
            # Regular Conv2d: causal time pad + symmetric freq pad
            x = self._causal_pad_time(x, 2)
            x = F.pad(x, (1, 1, 0, 0))
            x = self.conv1(x)

        # Conv2: ELU → causal time pad(2,0) + symmetric freq pad(1,1) → 3×3 conv
        x = self.activation(x)
        x = self._causal_pad_time(x, 2)
        x = F.pad(x, (1, 1, 0, 0))
        x = self.conv2(x)

        # Shortcut
        residual = self.shortcut(residual)

        # Match shapes
        mt = min(x.shape[2], residual.shape[2])
        mf = min(x.shape[3], residual.shape[3])
        return x[:, :, :mt, :mf] + residual[:, :, :mt, :mf]


class SpectroStreamDecoder(nn.Module):
    """SpectroStream 解码器 (JAX 严格架构对齐)

    输入: [B, T, 256] token embeddings (25Hz)
    输出: [B, 4, 480, T*4] STFT features

    通道追踪 (对齐 JAX, channel_splits=2):
      input_layer:     256 → 2560 → reshape → [B, 512, T, 5]
      input_residual:  512 → 512  (bottleneck, stride=1)
      decoder_0:       512 → 512  (stride=(2,1))
      --- ParallelChannels(group=2) splits channels: 512→2×256 ---
      decoder_1:       256 → 128  (stride=(2,2), mult=2)
      decoder_2:       128 → 128  (stride=(1,2), mult=1)
      decoder_3:       128 → 128  (stride=(1,2), mult=1)
      decoder_4:       128 → 64   (stride=(1,3), mult=2)
      decoder_5:        64 → 64   (stride=(1,2), mult=1)
      decoder_6:        64 → 32   (stride=(1,2), mult=2)
      output_layer:     32 → 2    (per group, kernel=7×7)
      concat:           2+2 = 4   (final output channels)
    """

    def __init__(self, config: SpectroStreamConfig):
        super().__init__()
        base_depth = config.decoder_base_conv_depth   # 64
        base_size = config.decoder_base_conv_size      # 7
        ratios = list(config.ratios)
        mults = list(config.mults)
        channel_splits = config.channel_splits          # 2
        channel_recombo_block = config.channel_recombo_block  # -2
        num_bins = config.num_bins                      # 480
        num_features = config.embedding_dim             # 256
        num_output_channels = config.num_channels       # 4

        num_blocks = len(ratios)  # 7
        # JAX: num_blocks+1=8, recombo_block %= 8, -2→6
        # Meaning: decoder_0 runs at full channels, then split before decoder_1
        if channel_recombo_block < 0:
            channel_recombo_block = (num_blocks + 1) + channel_recombo_block  # 6

        # Full-channel output = base_depth * prod(mults)
        output_channels = base_depth * int(math.prod(mults))  # 64*8 = 512
        input_bins = num_bins // config.total_freq_stride      # 480/96 = 5
        proj_filters = input_bins * output_channels            # 5*512 = 2560

        # Input layer: Residual(1×1 conv, shortcut: 1×1→ELU→1×1)
        # Matches JAX spectrostream_decoder_config "input_layer"
        self.input_conv = nn.Conv2d(num_features, proj_filters, kernel_size=1, bias=True)
        self.input_shortcut = nn.Sequential(
            nn.Conv2d(num_features, proj_filters, 1, bias=True),
            nn.ELU(),
            nn.Conv2d(proj_filters, proj_filters, 1, bias=True),
        )
        self.input_bins = input_bins
        self.output_channels = output_channels

        # Input residual unit (bottleneck after reshape, before decoder stages)
        # JAX: conv2d_residual_unit(transposed=True, strides=(1,1))
        self.input_residual = DecoderResidualUnit(output_channels, output_channels, stride=(1, 1))

        # Decoder stages (reverse order: decoder_0 uses ratios[-1], decoder_1 uses ratios[-2], etc.)
        reversed_ratios = ratios[::-1]  # ((2,1), (2,2), (1,2), (1,2), (1,3), (1,2), (1,2))
        reversed_mults = mults[::-1]    # (1, 2, 1, 1, 2, 1, 2)

        # decoder_0: full channels (before split)
        # decoder_1..6: split into groups (after split)
        split_after = num_blocks - channel_recombo_block  # 7-6=1 → split after decoder_0

        curr_channels = output_channels  # 512
        self.pre_split_stages = nn.ModuleList()
        for i in range(split_after):
            next_channels = int(round(curr_channels / reversed_mults[i]))
            rt, rf = reversed_ratios[i]
            self.pre_split_stages.append(DecoderResidualUnit(curr_channels, next_channels, stride=(rt, rf)))
            curr_channels = next_channels

        # Split: divide channels into groups
        self.num_groups = channel_splits if channel_splits else 1
        group_channels = curr_channels // self.num_groups  # 512/2 = 256

        # Post-split stages (decoder_1..6): per-group channels
        self.post_split_stages = nn.ModuleList()
        for i in range(split_after, num_blocks):
            next_channels = int(round(group_channels / reversed_mults[i]))
            rt, rf = reversed_ratios[i]
            self.post_split_stages.append(DecoderResidualUnit(group_channels, next_channels, stride=(rt, rf)))
            group_channels = next_channels

        # Output layer (per group): ELU → Conv2d(base_size×base_size, filters=num_output_channels/num_groups)
        out_per_group = num_output_channels // self.num_groups  # 4/2 = 2
        self.output_activation = nn.ELU()
        self.output_conv = nn.Conv2d(
            group_channels, out_per_group,
            kernel_size=(base_size, base_size), stride=1, padding=0, bias=True,
        )
        self.base_size = base_size

        self.rvq_embedding = RVQEmbedding(config)

    def _to_conv_format(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, D, T, 1]"""
        return x.transpose(1, 2).unsqueeze(-1)

    def _causal_time_pad(self, x, pad_left):
        if pad_left <= 0:
            return x
        return F.pad(x, (0, 0, pad_left, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        T_in = x.shape[1]

        # [B, T, D] → [B, D, T, 1]
        x = self._to_conv_format(x)

        # Input projection (Residual: main + shortcut)
        main = self.input_conv(x)
        shortcut = self.input_shortcut(x)
        x = main + shortcut  # [B, 2560, T, 1]

        # Reshape: [B, proj_filters, T, 1] → [B, output_channels, T, input_bins]
        # JAX: ExpandDims → Reshape([input_bins, output_channels])
        x = x.squeeze(-1)                                        # [B, 2560, T]
        x = x.permute(0, 2, 1)                                   # [B, T, 2560]
        x = x.reshape(B, T_in, self.input_bins, self.output_channels)  # [B, T, 5, 512]
        x = x.permute(0, 3, 1, 2)                                # [B, 512, T, 5]

        # Input residual (bottleneck)
        x = self.input_residual(x)

        # Pre-split decoder stages
        for stage in self.pre_split_stages:
            x = stage(x)

        # Channel split: divide into groups, each processes independently, concat
        if self.num_groups > 1:
            # x: [B, C, T, F] → split C into groups → process each → concat
            chunks = x.chunk(self.num_groups, dim=1)
            outputs = []
            for chunk in chunks:
                for stage in self.post_split_stages:
                    chunk = stage(chunk)
                # Output layer (per group)
                chunk = self.output_activation(chunk)
                # Causal time pad + symmetric freq pad for 7×7 output conv
                ks = self.base_size
                chunk = self._causal_time_pad(chunk, ks - 1)
                chunk = F.pad(chunk, ((ks - 1) // 2, ks - 1 - (ks - 1) // 2, 0, 0))
                chunk = self.output_conv(chunk)
                outputs.append(chunk)
            x = torch.cat(outputs, dim=1)  # concat along channel dim
        else:
            for stage in self.post_split_stages:
                x = stage(x)
            x = self.output_activation(x)
            ks = self.base_size
            x = self._causal_time_pad(x, ks - 1)
            x = F.pad(x, ((ks - 1) // 2, ks - 1 - (ks - 1) // 2, 0, 0))
            x = self.output_conv(x)

        # [B, C, T, F] → [B, C, F, T]
        return x.transpose(2, 3)

    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        embeddings = self.rvq_embedding(tokens)
        return self.forward(embeddings)
