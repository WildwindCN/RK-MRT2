"""SpectroStream Codec 损失函数

Multi-scale STFT loss + L1 波形损失 + RVQ commitment loss.

参考: EnCodec / SoundStream 的判别器-free 训练配方
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT 幅度损失

    在多个 FFT 尺度上比较原始和重建音频的频谱幅度。

    Args:
        fft_sizes: FFT 大小列表 (默认 [512, 1024, 2048])
        hop_sizes: 对应的 hop 大小 (默认 [128, 256, 512])
        win_lengths: 对应的窗长度 (默认与 fft_sizes 相同)
    """

    def __init__(self, fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_lengths=None):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths or fft_sizes

        # 预创建 Hann 窗口
        self.windows = nn.ParameterList([
            nn.Parameter(torch.hann_window(w), requires_grad=False)
            for w in self.win_lengths
        ])

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 原始音频 [B, C, T]
            x_hat: 重建音频 [B, C, T]
        Returns:
            scalar loss
        """
        loss = torch.tensor(0.0, device=x.device)
        for fft, hop, win_len, window in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths, self.windows
        ):
            # 对每个声道独立计算
            for ch in range(x.shape[1]):
                x_stft = torch.stft(x[:, ch], n_fft=fft, hop_length=hop,
                                    win_length=win_len, window=window,
                                    return_complex=True).abs()
                x_hat_stft = torch.stft(x_hat[:, ch], n_fft=fft, hop_length=hop,
                                        win_length=win_len, window=window,
                                        return_complex=True).abs()
                loss = loss + F.l1_loss(x_stft, x_hat_stft)
        return loss / (len(self.fft_sizes) * x.shape[1])


class CodecLoss(nn.Module):
    """Codec 训练总损失

    L_total = L_waveform + L_stft + lambda_commit * L_commitment

    Args:
        lambda_wave: 波形 L1 损失权重
        lambda_stft: Multi-scale STFT 损失权重
        lambda_commit: RVQ commitment 损失权重
    """

    def __init__(self, lambda_wave: float = 1.0, lambda_stft: float = 1.0,
                 lambda_commit: float = 0.01):
        super().__init__()
        self.lambda_wave = lambda_wave
        self.lambda_stft = lambda_stft
        self.lambda_commit = lambda_commit
        self.stft_loss = MultiScaleSTFTLoss()

    def forward(self, waveform: torch.Tensor, reconstructed: torch.Tensor,
                encoder_embeddings: torch.Tensor = None,
                quantized_embeddings: torch.Tensor = None) -> dict:
        """
        Args:
            waveform: 原始音频 [B, C, T]
            reconstructed: 重建音频 [B, C, T]
            encoder_embeddings: 编码器输出 [B, T_enc, D]
            quantized_embeddings: RVQ 量化后嵌入 [B, T_enc, D]
        Returns:
            dict with keys: loss, wave_loss, stft_loss, commit_loss
        """
        # 对齐长度
        min_len = min(waveform.shape[-1], reconstructed.shape[-1])
        waveform = waveform[..., :min_len]
        reconstructed = reconstructed[..., :min_len]

        # L1 波形损失
        wave_loss = F.l1_loss(waveform, reconstructed)

        # Multi-scale STFT 损失
        stft_l = self.stft_loss(waveform, reconstructed)

        # 总损失
        total = self.lambda_wave * wave_loss + self.lambda_stft * stft_l

        # RVQ commitment loss: encoder 输出靠近量化值
        commit_loss = torch.tensor(0.0, device=waveform.device)
        if encoder_embeddings is not None and quantized_embeddings is not None:
            commit_loss = F.mse_loss(encoder_embeddings, quantized_embeddings.detach())
            total = total + self.lambda_commit * commit_loss

        return {
            'loss': total,
            'wave_loss': wave_loss,
            'stft_loss': stft_l,
            'commit_loss': commit_loss,
        }
