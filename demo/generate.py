"""端到端推理 Demo: MIDI → Audio 生成

完整管线:
  1. MIDI Encoder: pianoroll [T, 128] → conditioning [T, 256]
  2. DepthFormer: AR 生成 RVQ tokens (25Hz)
  3. SpectroStream Decoder: tokens → STFT features
  4. iSTFT: STFT features → 48kHz 立体声 PCM
  5. 保存为 WAV

用法:
    python demo/generate.py --midi input.mid --output output.wav --duration 10 --temperature 0.8
"""

import os
import sys
import argparse
import time
import math
import struct
import wave

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import DepthFormerConfig, SpectroStreamConfig
from models.depthformer import DepthFormer
from models.spectrostream import SpectroStreamDecoder, RVQEmbedding
from models.istft import ISTFTLayer


# ═══════════════════════════════════════════════════════
# MIDI Encoder (简化版)
# ═══════════════════════════════════════════════════════

class MIDIEncoder(torch.nn.Module):
    """Pianoroll → Conditioning Embedding

    简化版: 128-dim pianoroll → Linear → encoder_dim
    完整版应包含 MusicCoCa style token + Instrument embedding
    """

    def __init__(self, pianoroll_dim: int = 128, encoder_dim: int = 256):
        super().__init__()
        self.proj = torch.nn.Linear(pianoroll_dim, encoder_dim, bias=False)
        self.norm = torch.nn.LayerNorm(encoder_dim)

    def forward(self, pianoroll: torch.Tensor) -> torch.Tensor:
        """
        pianoroll: [B, T, 128] float (0.0-1.0)
        returns:   [B, T, encoder_dim]
        """
        return self.norm(self.proj(pianoroll))


# ═══════════════════════════════════════════════════════
# 采样工具
# ═══════════════════════════════════════════════════════

def sample_topk(logits: torch.Tensor, temperature: float = 0.8, top_k: int = 50) -> torch.Tensor:
    """Top-k 采样

    logits: [..., vocab_size]
    returns: [...] int indices
    """
    if temperature > 0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
        min_top = top_values[..., -1:]
        logits = torch.where(logits >= min_top, logits, -float("inf"))

    # Softmax + sample
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs.view(-1, logits.size(-1)), 1).view(logits.shape[:-1])


# ═══════════════════════════════════════════════════════
# 主生成器
# ═══════════════════════════════════════════════════════

class Generator:
    """端到端音乐生成器"""

    def __init__(self, df_cfg: DepthFormerConfig, ss_cfg: SpectroStreamConfig,
                 device: torch.device = torch.device("cpu")):
        self.cfg = df_cfg
        self.ss_cfg = ss_cfg
        self.device = device

        # 组件
        self.encoder = MIDIEncoder(encoder_dim=df_cfg.encoder_spec.model_dims).to(device)
        self.depthformer = DepthFormer(df_cfg).to(device)
        self.codec_decoder = SpectroStreamDecoder(ss_cfg).to(device)
        self.istft = ISTFTLayer(
            frame_length=ss_cfg.stft_frame_length,
            frame_step=ss_cfg.stft_frame_step,
            fft_length=ss_cfg.stft_fft_length,
            num_bins=ss_cfg.num_bins,
            num_channels=ss_cfg.num_channels,
        ).to(device)

        # 模式
        self.eval()
        self._reset_state()

    def eval(self):
        self.encoder.eval()
        self.depthformer.eval()
        self.codec_decoder.eval()
        self.istft.eval()

    def _reset_state(self):
        """重置生成状态"""
        self.token_history = []       # 每帧的 per-RVQ token [Q] (0..codebook_size-1)
        self.full_token_history = []  # 每帧的 full vocab token [Q] (含 RVQ offset)
        self.embedding_history = []   # 每帧的 embedding [D_temp]
        self.audio_buffer = []        # 累积 PCM

    @torch.no_grad()
    def generate(self, pianoroll: torch.Tensor, num_frames: int,
                 temperature: float = 0.8, top_k: int = 50,
                 show_progress: bool = True) -> np.ndarray:
        """
        Args:
            pianoroll: [T_cond, 128] MIDI pianoroll
            num_frames: 生成的帧数 (25Hz)
            temperature: 采样温度
            top_k: Top-k 采样参数

        Returns:
            pcm: [2, num_samples] 48kHz 立体声 PCM
        """
        B = 1
        Q = self.cfg.num_codebooks
        D_temp = self.cfg.temporal_spec.model_dims
        D_enc = self.cfg.encoder_spec.model_dims

        # 1. Encode MIDI conditioning
        pr = pianoroll.unsqueeze(0).to(self.device)  # [1, T_cond, 128]
        conditioning = self.encoder(pr)               # [1, T_cond, 256]

        self._reset_state()

        # 2. AR 生成循环
        for frame in range(num_frames):
            # 上一帧的 tokens (第一帧用 BOS)
            if frame == 0:
                prev_tokens = torch.full(
                    (B, 1, Q), self.cfg.sos_id,
                    dtype=torch.long, device=self.device
                )
            else:
                # full_token_history 存储完整 vocab token (含 RVQ offset)
                prev_tokens = torch.tensor(
                    [self.full_token_history[-1]],
                    dtype=torch.long, device=self.device
                ).unsqueeze(0)  # [1, 1, Q]

            # Token → Embedding → Mean
            embedded = self.depthformer.token_embedding(prev_tokens)  # [1, 1, Q, D_temp]
            temporal_input = embedded.mean(dim=-2)  # [1, 1, D_temp]

            # Temporal Body (完整序列模式)
            # 构建完整输入序列 [prev_embedding_0, ..., prev_embedding_{frame-1}, current]
            if frame == 0:
                full_temporal_input = temporal_input
            else:
                prev_embeddings = torch.stack(self.embedding_history, dim=0)  # [frame, D_temp]
                prev_embeddings = prev_embeddings.unsqueeze(0)  # [1, frame, D_temp]
                full_temporal_input = torch.cat([prev_embeddings, temporal_input], dim=1)  # [1, frame+1, D_temp]

            temporal_out = self.depthformer.temporal_body(
                full_temporal_input, conditioning
            )  # [1, frame+1, D_temp]
            current_temporal = temporal_out[:, -1:]  # [1, 1, D_temp]

            # Depth Body AR (逐 RVQ 层)
            rvq_tokens = []      # per-RVQ indices (0..codebook_size-1)
            full_tokens = []     # full vocab indices (with RVQ offset)
            depth_input = current_temporal.squeeze(1)  # [1, D_temp]

            for rvq_idx in range(Q):
                # 构建 depth body 输入: [1, rvq_idx+1, D_temp]
                if rvq_idx == 0:
                    depth_seq = depth_input.unsqueeze(1)  # [1, 1, D_temp]
                else:
                    # 之前 RVQ token 的 embeddings (用 full token)
                    prev_full = torch.tensor(
                        full_tokens, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, rvq_idx]
                    prev_emb = self.depthformer.token_embedding(prev_full)  # [1, rvq_idx, D_temp]
                    depth_seq = torch.cat([
                        depth_input.unsqueeze(1),
                        prev_emb,
                    ], dim=1)  # [1, rvq_idx+1, D_temp]

                logits = self.depthformer.depth_body(depth_seq)  # [1, rvq_idx+1, vocab]
                next_logits = logits[:, -1]  # [1, vocab]

                # 采样 (只从有效 codebook 范围采样)
                valid_mask = torch.zeros(self.cfg.vocab_size, device=self.device)
                min_val = self.cfg.num_reserved_tokens + rvq_idx * self.cfg.codebook_size
                max_val = min_val + self.cfg.codebook_size
                valid_mask[min_val:max_val] = 1.0
                masked_logits = next_logits + torch.log(valid_mask + 1e-10)

                sampled = sample_topk(masked_logits, temperature, top_k)
                full_token = sampled.item()
                rvq_token = full_token - self.cfg.num_reserved_tokens - rvq_idx * self.cfg.codebook_size

                full_tokens.append(full_token)
                rvq_tokens.append(rvq_token)

                # 下一轮 depth input: embed this token (full vocab index)
                if rvq_idx < Q - 1:
                    tok_tensor = torch.tensor(
                        [[full_token]], dtype=torch.long, device=self.device
                    )
                    depth_input = self.depthformer.token_embedding(tok_tensor).squeeze(1)  # [1, D_temp]

            # 保存本帧结果
            self.token_history.append(rvq_tokens)       # per-RVQ (for codec decoder)
            self.full_token_history.append(full_tokens)  # full vocab (for next frame embedding)
            self.embedding_history.append(current_temporal.squeeze(1).squeeze(0))

            if show_progress and (frame + 1) % 25 == 0:
                sec = (frame + 1) / 25
                print(f"\r  Generated {frame + 1}/{num_frames} frames ({sec:.1f}s)", end="", flush=True)

        if show_progress:
            print()

        # 3. Codec Decoder: tokens → STFT
        all_tokens = torch.tensor(self.token_history, dtype=torch.long, device=self.device)
        all_tokens = all_tokens.unsqueeze(0)  # [1, T, Q]
        stft_features = self.codec_decoder.decode_from_tokens(all_tokens)  # [1, 4, 480, T*4]

        # 4. iSTFT: STFT → PCM
        pcm = self.istft(stft_features)  # [1, 2, num_samples]
        pcm = pcm.squeeze(0).cpu().numpy()  # [2, num_samples]

        return pcm


# ═══════════════════════════════════════════════════════
# WAV 保存
# ═══════════════════════════════════════════════════════

def save_wav(path: str, pcm: np.ndarray, sample_rate: int = 48000):
    """保存立体声 PCM 为 WAV 文件"""
    pcm_int16 = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.T.tobytes())
    print(f"  Saved: {path} ({pcm.shape[1]/sample_rate:.1f}s, {sample_rate}Hz stereo)")


# ═══════════════════════════════════════════════════════
# 简易 MIDI → Pianoroll (无需外部依赖)
# ═══════════════════════════════════════════════════════

def create_test_pianoroll(num_frames: int = 250, pattern: str = "scale") -> np.ndarray:
    """创建测试 pianoroll (无需真实 MIDI 文件)

    pattern: "scale" (上行音阶), "chord" (和弦进行), "silence" (静音)
    """
    pr = np.zeros((num_frames, 128), dtype=np.float32)
    frame_rate = 25  # Hz

    if pattern == "scale":
        # C 大调上行音阶, 每 4 帧换一个音
        scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C4-C5
        for i, pitch in enumerate(scale):
            start = i * num_frames // len(scale)
            end = (i + 1) * num_frames // len(scale)
            pr[start:end, pitch] = 0.8

    elif pattern == "chord":
        # C-F-G-C 和弦进行
        chords = [
            ([60, 64, 67], 0),       # C major
            ([65, 69, 72], 1),       # F major
            ([67, 71, 74], 2),       # G major
            ([60, 64, 67], 3),       # C major
        ]
        for notes, idx in chords:
            start = idx * num_frames // 4
            end = (idx + 1) * num_frames // 4
            for pitch in notes:
                pr[start:end, pitch] = 0.8
                # 每 4 帧触发一次 onset
                for f in range(start, end, 4):
                    if f < num_frames:
                        pr[f, pitch] = 1.0  # onset

    elif pattern == "silence":
        pass  # all zeros

    return pr


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RK-MRT2 End-to-End Generation Demo")
    parser.add_argument("--midi", type=str, default=None, help="MIDI file path (not yet supported)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output WAV path")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--pattern", type=str, default="chord",
                        choices=["scale", "chord", "silence"],
                        help="Test pianoroll pattern")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to converted PyTorch weights (.pt)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    num_frames = int(args.duration * 25)

    print("=" * 60)
    print("RK-MRT2: End-to-End Music Generation")
    print(f"  Duration: {args.duration}s ({num_frames} frames @ 25Hz)")
    print(f"  Temperature: {args.temperature}, Top-k: {args.top_k}")
    print(f"  Pattern: {args.pattern}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # 初始化模型
    print("\n[1] Building models...")
    df_cfg = DepthFormerConfig()
    ss_cfg = SpectroStreamConfig()

    gen = Generator(df_cfg, ss_cfg, device=device)

    # 加载权重
    if args.weights:
        print(f"  Loading weights: {args.weights}")
        state = torch.load(args.weights, map_location=device, weights_only=False)
        gen.depthformer.load_state_dict(state["depthformer"], strict=False)
        gen.codec_decoder.load_state_dict(state["codec_decoder"], strict=False)
        print("  Weights loaded [OK]")

    total_params = sum(p.numel() for p in gen.encoder.parameters())
    total_params += sum(p.numel() for p in gen.depthformer.parameters())
    total_params += sum(p.numel() for p in gen.codec_decoder.parameters())
    print(f"  Total parameters: {total_params/1e6:.1f}M")

    # 创建 pianoroll
    print(f"\n[2] Creating test pianoroll ({args.pattern})...")
    pr = create_test_pianoroll(num_frames, args.pattern)
    pr = torch.from_numpy(pr).float()
    print(f"  Shape: {list(pr.shape)}")

    # 生成
    print(f"\n[3] Generating {num_frames} frames...")
    t0 = time.time()
    pcm = gen.generate(pr, num_frames, args.temperature, args.top_k)
    elapsed = time.time() - t0
    audio_dur = pcm.shape[1] / ss_cfg.audio_sample_rate
    rtf = elapsed / audio_dur

    print(f"\n  Generation complete:")
    print(f"    Wall time: {elapsed:.1f}s")
    print(f"    Audio duration: {audio_dur:.1f}s")
    print(f"    RTF: {rtf:.2f}x (real-time factor)")

    # 保存
    print(f"\n[4] Saving audio...")
    save_wav(args.output, pcm, int(ss_cfg.audio_sample_rate))

    # 统计
    print(f"\n[5] Token statistics:")
    all_tokens = np.array(gen.token_history)
    for rvq in range(min(4, df_cfg.num_codebooks)):
        tokens = all_tokens[:, rvq]
        unique = len(np.unique(tokens))
        print(f"    RVQ{rvq}: {unique} unique tokens / {len(tokens)} frames")

    print("\nDone!")


if __name__ == "__main__":
    main()
