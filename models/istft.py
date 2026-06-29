"""iSTFT 音频重建模块

将 SpectroStream Decoder 输出的 STFT 频域特征转为 PCM 音频。

生产环境使用 Signalsmith DSP (MIT, header-only C++),
当前 PyTorch 实现用于本地验证和 Demo。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ISTFTLayer(nn.Module):
    """逆短时傅里叶变换 (iSTFT)

    输入: [B, num_channels, num_bins, T_stft] 频域表示
    输出: [B, num_samples] 时域波形

    SpectroStream 的 STFT 格式:
    - frame_length=960, frame_step=480, fft_length=960
    - num_bins=480 (fft_length/2, keep_dc=True → 去掉 Nyquist bin)
    - num_channels=4 (2 个音频通道 × 2 实部/虚部)
    - 立体声输出: 2 通道交替排列
    """

    def __init__(
        self,
        frame_length: int = 960,
        frame_step: int = 480,
        fft_length: int = 960,
        num_bins: int = 480,
        num_channels: int = 4,
        keep_dc: bool = True,
    ):
        super().__init__()
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length
        self.num_bins = num_bins
        self.num_channels = num_channels
        self.keep_dc = keep_dc
        self.num_audio_channels = num_channels // 2

        # Hann window
        self.register_buffer(
            "window",
            torch.hann_window(frame_length, periodic=True),
        )

    def _complex_to_channels(self, stft_features: torch.Tensor) -> torch.Tensor:
        """[B, 4, F, T] → complex [B, 2, F+1, T]"""
        B, C, n_bins, n_frames = stft_features.shape
        assert C == self.num_channels, f"Expected {self.num_channels} channels, got {C}"
        assert n_bins == self.num_bins, f"Expected {self.num_bins} bins, got {n_bins}"

        stft_reshaped = stft_features.view(B, self.num_audio_channels, 2, n_bins, n_frames)
        real = stft_reshaped[:, :, 0]  # [B, 2, F, T]
        imag = stft_reshaped[:, :, 1]

        if self.keep_dc:
            real = torch.nn.functional.pad(real, (0, 0, 0, 1))
            imag = torch.nn.functional.pad(imag, (0, 0, 0, 1))
        else:
            real = torch.nn.functional.pad(real, (0, 0, 1, 0))
            imag = torch.nn.functional.pad(imag, (0, 0, 1, 0))

        complex_spec = torch.complex(real, imag)  # [B, 2, F+1, T]
        return complex_spec

    def forward(self, stft_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stft_features: [B, num_channels, num_bins, T_stft]

        Returns:
            pcm: [B, num_audio_channels, num_samples] 立体声 PCM
        """
        B = stft_features.shape[0]

        # 转为复数频谱
        complex_spec = self._complex_to_channels(stft_features)  # [B, 2, F+1, T]

        # 对每个音频通道做 iSTFT
        pcm_list = []
        for ch in range(self.num_audio_channels):
            spec_ch = complex_spec[:, ch]  # [B, F+1, T] — correct for torch.istft

            waveform = torch.istft(
                spec_ch,
                n_fft=self.fft_length,
                hop_length=self.frame_step,
                win_length=self.frame_length,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                length=None,
                return_complex=False,
            )
            pcm_list.append(waveform)

        pcm = torch.stack(pcm_list, dim=1)  # [B, 2, num_samples]
        return pcm


def test_istft():
    """验证 iSTFT 往返: STFT → iSTFT ≈ identity"""
    import numpy as np

    layer = ISTFTLayer()
    layer.eval()

    # 生成测试信号
    sample_rate = 48000
    duration = 1.0
    t = torch.arange(0, int(sample_rate * duration)) / sample_rate
    # 440Hz 正弦波 + 880Hz 泛音
    signal = torch.sin(2 * torch.pi * 440 * t) + 0.5 * torch.sin(2 * torch.pi * 880 * t)
    signal = signal.unsqueeze(0).unsqueeze(1)  # [1, 1, T]
    signal_stereo = signal.repeat(1, 2, 1)  # [1, 2, T]

    # STFT
    window = torch.hann_window(960, periodic=True)
    spec = torch.stft(
        signal_stereo.view(2, -1),
        n_fft=960, hop_length=480, win_length=960,
        window=window, center=True,
        return_complex=True,
    )  # [2, F+1, T]

    # 打包为 SpectroStream 格式: [1, 4, F, T]
    spec_real = spec.real  # [2, F+1, T]
    spec_imag = spec.imag

    # 去掉 Nyquist bin (keep_dc=True → last bin)
    spec_real = spec_real[:, :-1]  # [2, F, T]
    spec_imag = spec_imag[:, :-1]

    stft_packed = torch.stack([
        spec_real[0], spec_imag[0],
        spec_real[1], spec_imag[1],
    ], dim=0).unsqueeze(0)  # [1, 4, F, T]

    # iSTFT
    with torch.no_grad():
        reconstructed = layer(stft_packed)

    # 对齐延迟
    min_len = min(signal_stereo.shape[2], reconstructed.shape[2])
    orig = signal_stereo[:, :, :min_len]
    recon = reconstructed[:, :, :min_len]

    # SNR
    noise = orig - recon
    snr = 10 * torch.log10((orig ** 2).mean() / (noise ** 2).mean())
    print(f"  iSTFT round-trip SNR: {snr.item():.1f} dB")

    assert snr > 60, f"SNR too low: {snr.item():.1f} dB"
    print("  [PASS] iSTFT verification")


if __name__ == "__main__":
    test_istft()
