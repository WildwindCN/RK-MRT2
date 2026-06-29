"""SpectroStream Decoder —— PyTorch 参考实现

将 RVQ 离散 Token 解码为 STFT 频域表示。
iSTFT 在 CPU 端完成（RKNN 不支持 STFT 算子）。

架构: RVQ Embedding → 7-stage Upsample + Conv Decoder → STFT features
每个 stage: ELU → Upsample(time×freq) → Conv2d(3×3) → Residual

参考: magenta_rt/jax/spectrostream.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SpectroStreamConfig


class RVQEmbedding(nn.Module):
    """残差向量量化 (RVQ) 的 Token → Embedding 转换

    输入: [B, T, Q] int32 (Q 层 RVQ token indices)
    输出: [B, T, D] float (Q 层嵌入求和)

    等效于 JAX ResidualVectorQuantizer.codes_to_embeddings(use_gather=True)
    """

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


class DecoderStage(nn.Module):
    """单个解码器 stage: 上采样 + 卷积 + 残差

    - 上采样: interpolate (time×freq)
    - 卷积: 3×3 Conv2d (causal in time)
    - 残差 shortcut: 上采样 + 1×1 Conv (if channel change)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        causal: bool = True,
    ):
        super().__init__()
        self.stride = stride
        self.causal = causal

        self.activation = nn.ELU()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1,
            padding=0, bias=True,
        )

        # Shortcut
        need_shortcut = (stride != (1, 1)) or (in_channels != out_channels)
        if need_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=True)
        else:
            self.shortcut = nn.Identity()

    def _causal_pad_time(self, x: torch.Tensor, pad: int) -> torch.Tensor:
        """时间维度左侧 causal padding"""
        if pad == 0:
            return x
        return F.pad(x, (0, 0, pad, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        residual = x
        T_in, F_in = x.shape[2], x.shape[3]

        # 上采样
        new_t, new_f = T_in, F_in
        if self.stride != (1, 1):
            new_t = T_in * self.stride[0]
            new_f = F_in * self.stride[1]
            x = F.interpolate(x, size=(new_t, new_f), mode='nearest')

        # 卷积 (Pre-activation: ELU before conv)
        x = self.activation(x)
        # Causal padding in time (2 left, 0 right) + symmetric in freq (1 each side)
        x = self._causal_pad_time(x, 2)
        x = F.pad(x, (1, 1, 0, 0))
        x = self.conv(x)
        # After 3×3 conv with padding: time = new_t + 2 - 2 = new_t, freq = new_f + 2 - 2 = new_f
        # So x already has shape [B, C, new_t, new_f]

        # Shortcut
        if self.stride != (1, 1):
            residual = F.interpolate(residual, size=(new_t, new_f), mode='nearest')
        residual = self.shortcut(residual)

        # Match shapes
        min_t = min(x.shape[2], residual.shape[2])
        min_f = min(x.shape[3], residual.shape[3])
        x = x[:, :, :min_t, :min_f]
        residual = residual[:, :, :min_t, :min_f]

        return x + residual


class SpectroStreamDecoder(nn.Module):
    """SpectroStream 解码器

    输入: [B, T, D] token embeddings (25Hz)
    输出: [B, num_channels, num_bins, T*time_stride] STFT features
    """

    def __init__(self, config: SpectroStreamConfig):
        super().__init__()
        self.config = config
        self.total_time_stride = config.total_time_stride
        self.total_freq_stride = config.total_freq_stride

        base_depth = config.decoder_base_conv_depth  # 64
        mults = config.mults  # (2, 1, 2, 1, 1, 2, 1)
        input_bins = config.num_bins // config.total_freq_stride  # 480/96 = 5

        # 初始通道数: base_depth * prod(mults) = 64 * 8 = 512
        init_channels = base_depth * int(math.prod(mults))
        # 输入投影: embedding_dim → input_bins * init_channels (= 5*512 = 2560)
        proj_filters = input_bins * init_channels

        self.input_conv = nn.Conv2d(
            config.embedding_dim, proj_filters,
            kernel_size=1, bias=True,
        )
        self.input_shortcut = nn.Sequential(
            nn.Conv2d(config.embedding_dim, proj_filters, 1, bias=True),
            nn.ELU(),
            nn.Conv2d(proj_filters, proj_filters, 1, bias=True),
        )

        self.input_bins = input_bins
        self.init_channels = init_channels

        # 解码器阶段 (反转 ratios 和 mults → 上采样)
        reversed_ratios = config.ratios[::-1]
        reversed_mults = config.mults[::-1]

        self.stages = nn.ModuleList()
        curr_channels = init_channels
        for (rt, rf), mult in zip(reversed_ratios, reversed_mults):
            next_channels = int(round(curr_channels / mult))
            self.stages.append(
                DecoderStage(curr_channels, next_channels, stride=(rt, rf), causal=config.causal)
            )
            curr_channels = next_channels

        # 最终输出: → num_channels
        self.output_activation = nn.ELU()
        ks = config.decoder_base_conv_size
        self.output_conv = nn.Conv2d(
            curr_channels, config.num_channels,
            kernel_size=(ks, ks),
            stride=1, padding=0, bias=True,
        )

        self.rvq_embedding = RVQEmbedding(config)

    def _to_conv_format(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, D, T, 1]"""
        return x.transpose(1, 2).unsqueeze(-1)

    def _causal_pad(self, x: torch.Tensor, pad_time: int, pad_freq: int) -> torch.Tensor:
        """Causal padding (time left only)"""
        if pad_time > 0:
            x = F.pad(x, (0, 0, pad_time, 0))
        if pad_freq > 0:
            x = F.pad(x, (pad_freq // 2, pad_freq - pad_freq // 2))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] token embeddings (25Hz)

        Returns:
            [B, num_channels, num_bins, T*time_stride] STFT features
        """
        B = x.shape[0]
        T_in = x.shape[1]

        # [B, T, D] → [B, D, T, 1]
        x = self._to_conv_format(x)

        # 输入投影: [B, D, T, 1] → [B, proj_filters, T, 1]
        main = self.input_conv(x)
        shortcut = self.input_shortcut(x)
        x = main + shortcut  # [B, input_bins * init_channels, T, 1]

        # Reshape: [B, proj_filters, T, 1] → [B, init_channels, T, input_bins]
        # 等效于 JAX: Reshape([input_bins, output_channels])
        x = x.squeeze(-1)  # [B, proj_filters, T]
        x = x.permute(0, 2, 1)  # [B, T, proj_filters]
        x = x.reshape(B, T_in, self.input_bins, self.init_channels)  # [B, T, input_bins, output_channels]
        x = x.permute(0, 3, 1, 2)  # [B, output_channels, T, input_bins]

        # 各解码器 stage
        for stage in self.stages:
            x = stage(x)

        # 最终输出层
        x = self.output_activation(x)
        x = self._causal_pad(x, self.config.decoder_base_conv_size - 1,
                             self.config.decoder_base_conv_size - 1)
        x = self.output_conv(x)

        # [B, C, T', F] → [B, C, F, T']
        x = x.transpose(2, 3)

        return x

    def decode_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """从 RVQ tokens 直接解码"""
        embeddings = self.rvq_embedding(tokens)
        return self.forward(embeddings)
