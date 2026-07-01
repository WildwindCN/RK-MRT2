"""SpectroStream Codec 数据加载器

从音频目录加载 48kHz 立体声文件, 在线计算 STFT, 提供训练 DataLoader。

用法:
    from training.codec_dataset import create_dataloader
    loader = create_dataloader("/data/ace_studio/", batch_size=8, num_workers=4)
"""
import os, glob, random
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class CodecDataset(Dataset):
    """音频片段数据集

    扫描目录下所有 .wav/.flac 文件, 切分为固定长度片段。
    在线增益增强。不预计算 STFT (保证训练时梯度流通)。

    Args:
        data_dir: 音频文件目录
        segment_samples: 每片段采样数 (默认 480000 = 10s @ 48kHz)
        sample_rate: 目标采样率 (默认 48000)
        stereo: 是否立体声
    """

    def __init__(self, data_dir: str, segment_samples: int = 480000,
                 sample_rate: int = 48000, stereo: bool = True):
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.stereo = stereo
        self.target_channels = 2 if stereo else 1

        # 扫描音频文件
        self.files = []
        for ext in ('*.wav', '*.flac', '*.mp3', '*.ogg', '*.m4a'):
            self.files.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
        if not self.files:
            raise ValueError(f"No audio files found in {data_dir}")

        # 预扫描: 计算每个文件可切分的片段数
        self.segments = []
        for fpath in self.files:
            try:
                info = torchaudio.info(fpath)
                total_samples = info.num_frames
                n_segs = max(0, total_samples // segment_samples)
                for i in range(n_segs):
                    self.segments.append((fpath, i * segment_samples))
            except Exception:
                continue

        print(f"CodecDataset: {len(self.files)} files, {len(self.segments)} segments "
              f"({len(self.segments) * segment_samples / sample_rate / 3600:.1f}h total)")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        fpath, offset = self.segments[idx]
        chunk, sr = torchaudio.load(fpath, frame_offset=offset,
                                     num_frames=self.segment_samples)
        # 重采样 (如果需要)
        if sr != self.sample_rate:
            chunk = torchaudio.functional.resample(chunk, sr, self.sample_rate)
        # 声道处理
        if chunk.shape[0] == 1 and self.target_channels == 2:
            chunk = chunk.repeat(2, 1)
        elif chunk.shape[0] > self.target_channels:
            chunk = chunk[:self.target_channels, :]
        # 长度对齐
        if chunk.shape[1] < self.segment_samples:
            chunk = F.pad(chunk, (0, self.segment_samples - chunk.shape[1]))
        # 随机增益增强 (±3dB, 在峰值归一化之前)
        if random.random() < 0.5:
            gain_db = random.uniform(-3, 3)
            gain_linear = 10 ** (gain_db / 20.0)
            chunk = chunk * gain_linear
        return chunk  # [channels, samples]


def create_dataloader(data_dir: str, batch_size: int = 8, num_workers: int = 4,
                      segment_seconds: float = 10.0, sample_rate: int = 48000,
                      ddp: bool = False, rank: int = 0, world_size: int = 1,
                      prefetch_factor: int = 2):
    """创建训练 DataLoader

    Args:
        data_dir: 音频目录
        batch_size: 每 GPU batch size
        num_workers: 数据加载线程数
        segment_seconds: 片段长度 (秒)
        sample_rate: 采样率
        ddp: 是否使用 DistributedSampler
        rank: 当前进程 rank
        world_size: 总进程数
        prefetch_factor: 每个 worker 预取 batch 数
    """
    segment_samples = int(segment_seconds * sample_rate)
    dataset = CodecDataset(data_dir, segment_samples=segment_samples,
                           sample_rate=sample_rate)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                  shuffle=True) if ddp else None

    loader = DataLoader(dataset, batch_size=batch_size,
                        sampler=sampler,
                        shuffle=(sampler is None),
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0),
                        prefetch_factor=prefetch_factor if num_workers > 0 else None,
                        drop_last=True)
    return loader, dataset
