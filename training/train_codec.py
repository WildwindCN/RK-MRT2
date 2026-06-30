"""SpectroStream Codec 端到端训练脚本

完整训练流程:
  音频 → STFT → Encoder → RVQ(冻结) → Decoder → iSTFT → 重建音频
  Loss: L1 + Multi-scale STFT + RVQ Commitment

训练策略:
  Phase 1 (epochs 1-50): 冻结 Decoder + RVQ, 只训练 Encoder
  Phase 2 (epochs 51-200): 解冻 Decoder, 联合微调

用法 (DDP):
  torchrun --nproc_per_node=8 training/train_codec.py --data_dir /data/ace --epochs 200
"""

import os, sys, argparse, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import SpectroStreamConfig
from models.spectrostream import SpectroStreamDecoder, RVQEmbedding
from models.spectrostream_encoder import SpectroStreamEncoder
from models.istft import ISTFTLayer
from training.codec_dataset import create_dataloader
from training.codec_loss import CodecLoss


def setup_ddp():
    """初始化 DDP 环境"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def load_checkpoint_state(path, cfg, encoder, decoder):
    """从 MRT2 Small checkpoint 加载 RVQ + Decoder 初始化权重

    Decoder 架构已对齐 JAX, 可直接加载。Encoder 不加载 (从头训练)。
    RVQ 加载后冻结。
    """
    state = torch.load(path, map_location='cpu', weights_only=True)
    codec_state = state['codec_decoder']

    # Load RVQ
    rvq_weight = codec_state['rvq_embedding.embedding']
    if rvq_weight.shape == (64, 1024, 256):
        rvq = RVQEmbedding(cfg)
        rvq.embedding.data.copy_(rvq_weight)
        for p in rvq.parameters():
            p.requires_grad = False
        print(f"  RVQ loaded: {list(rvq_weight.shape)} (frozen)")
    else:
        raise RuntimeError(f"Unexpected RVQ shape: {rvq_weight.shape}")

    # Load Decoder (partial — only matching keys)
    decoder_state = {}
    for k, v in codec_state.items():
        if k.startswith('rvq_embedding.'):
            continue
        if hasattr(decoder, k.split('.')[0]) or any(
            k.startswith(p) for p in ['input_conv', 'input_shortcut',
                                       'input_residual', 'stages', 'output_']
        ):
            decoder_state[k] = v
    missing, unexpected = decoder.load_state_dict(decoder_state, strict=False)
    print(f"  Decoder: {len(decoder_state)} params loaded, "
          f"{len(missing)} missing (untrained), {len(unexpected)} unexpected")
    if missing:
        print(f"    First 5 missing: {missing[:5]}")

    return rvq


def compute_stft(waveform, cfg):
    """在线计算 STFT 特征 [B, C_audio, N] → [B, 4, 480, T_stft]

    严格对齐 JAX slice_and_bitcast:
      complex STFT [B, T, 481, C_audio] → view_as_real → [B, T, 481, 2*C_audio]
      → drop Nyquist → [B, T, 480, 4]
    输出格式: [B, 4, 480, T_stft] (BCFT)
    """
    window = torch.hann_window(cfg.stft_fft_length, periodic=True,
                               device=waveform.device)
    B, C_audio, N = waveform.shape

    stfts = []
    for ch in range(C_audio):
        stft = torch.stft(
            waveform[:, ch],
            n_fft=cfg.stft_fft_length,
            hop_length=cfg.stft_frame_step,
            win_length=cfg.stft_frame_length,
            window=window,
            center=True,
            return_complex=True,
        )  # [B, 481, T_stft] complex
        real_imag = torch.view_as_real(stft)  # [B, 481, T_stft, 2]
        stfts.append(real_imag)

    # JAX: view(float_dtype) → [B, T_stft, 481, 2*C_audio]
    # PyTorch: stack channels, permute to BCFT
    # [B, 481, T, 2] x 2 → stack → [B, 2, 481, T, 2] → permute → [B, 2, 2, 481, T] → reshape → [B, 4, 481, T]
    stft_in = torch.stack(stfts, dim=1)  # [B, C_audio, 481, T, 2]
    stft_in = stft_in.permute(0, 1, 4, 2, 3)  # [B, C_audio, 2, 481, T]
    stft_in = stft_in.reshape(B, C_audio * 2, cfg.stft_fft_length // 2 + 1, -1)  # [B, 4, 481, T]
    # Drop Nyquist bin (keep_dc=True): keep first 480 of 481 freq bins
    stft_in = stft_in[:, :, :cfg.num_bins, :]  # [B, 4, 480, T_stft]
    return stft_in


def compute_istft(stft_features, cfg):
    """逆 STFT: [B, 4, 480, T] → [B, 2, N_samples]

    使用 ISTFTLayer 进行确定性重建。
    """
    istft = ISTFTLayer(
        frame_length=cfg.stft_frame_length,
        frame_step=cfg.stft_frame_step,
        fft_length=cfg.stft_fft_length,
        num_bins=cfg.num_bins,
        num_channels=cfg.num_channels,
    ).to(stft_features.device)
    return istft(stft_features)


def rvq_encode(embeddings, rvq, cfg):
    """RVQ 量化: 连续嵌入 → 离散 tokens → 量化嵌入

    严格对齐 JAX ResidualVectorQuantizer._quantize():
      distances = -2 * input @ codebook.T + ||codebook||²
      indices = argmin(distances)
      quantized = input + stop_gradient(quantized_raw - input)  (straight-through)

    Encoder 通过 straight-through estimator 接收重建损失的梯度,
    加上 codec_loss 中的 commitment loss。

    Args:
        embeddings: [B, T_enc, 256] (requires_grad)
        rvq: RVQEmbedding (frozen, MRT2 Small weights)
    Returns:
        quantized: [B, T_enc, 256] 量化后嵌入 (straight-through gradient)
        tokens: [B, T_enc, 12] 离散 tokens (no grad)
    """
    B, T, D = embeddings.shape
    residual = embeddings
    quantized_sum_raw = torch.zeros_like(embeddings)
    tokens_list = []

    for q in range(cfg.rvq_truncation_level):  # 12 levels
        codebook = rvq.embedding[q]  # [1024, 256]
        # JAX: distances = -2 * residual @ codebook.T + ||codebook||²
        codebook_norm = (codebook * codebook).sum(dim=-1)  # [1024]
        scores = torch.matmul(residual.reshape(-1, D), codebook.T)  # [B*T, 1024]
        scores = scores - 0.5 * codebook_norm.unsqueeze(0)
        indices = scores.argmax(dim=-1)  # [B*T]
        tokens = indices.reshape(B, T)

        # Lookup (no grad for raw quantized values)
        with torch.no_grad():
            quantized_raw = F.embedding(tokens, codebook)  # [B, T, 256]
        quantized_sum_raw = quantized_sum_raw + quantized_raw
        residual = residual - quantized_raw
        tokens_list.append(tokens)

    tokens = torch.stack(tokens_list, dim=-1)  # [B, T, 12]

    # Straight-through estimator (JAX stop_gradient equivalent):
    # forward = quantized_sum_raw (actual codebook sum)
    # backward gradient flows directly to embeddings
    quantized = embeddings + (quantized_sum_raw - embeddings).detach()
    return quantized, tokens


def train():
    parser = argparse.ArgumentParser(description='SpectroStream Codec Training')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str,
                        default='exported/weights/mrt2_small_pytorch.pt')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--segment_seconds', type=float, default=10.0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--precision', type=str, default='bf16',
                        choices=['fp32', 'fp16', 'bf16'])
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--phase1_epochs', type=int, default=50,
                        help='Epochs with decoder frozen (encoder-only training)')
    args = parser.parse_args()

    # DDP setup
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f'cuda:{local_rank}')
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
        print("=" * 60)
        print("SpectroStream Codec Training")
        print(f"  Data: {args.data_dir}")
        print(f"  GPUs: {world_size}, Batch/GPU: {args.batch_size}")
        print(f"  Total batch: {world_size * args.batch_size}")
        print(f"  Epochs: {args.epochs}, LR: {args.lr}")
        print(f"  Precision: {args.precision}")
        print("=" * 60)

    # Config & Models
    cfg = SpectroStreamConfig()

    encoder = SpectroStreamEncoder(cfg).to(device)
    decoder = SpectroStreamDecoder(cfg).to(device)

    # Load RVQ (frozen) + Decoder init from checkpoint
    rvq = load_checkpoint_state(args.checkpoint, cfg, encoder, decoder)
    rvq = rvq.to(device)

    # Loss
    criterion = CodecLoss().to(device)

    # Phase 1: freeze decoder, train encoder only
    for p in decoder.parameters():
        p.requires_grad = False

    trainable_params = list(encoder.parameters())
    if is_main:
        enc_p = sum(p.numel() for p in encoder.parameters())
        dec_p = sum(p.numel() for p in decoder.parameters())
        print(f"  Encoder: {enc_p/1e6:.1f}M, Decoder: {dec_p/1e6:.1f}M "
              f"(frozen in Phase 1)")
        print(f"  Trainable: {sum(p.numel() for p in trainable_params)/1e6:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr,
                                   betas=(0.8, 0.99), weight_decay=1e-5)

    # Data
    loader, dataset = create_dataloader(
        args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers,
        segment_seconds=args.segment_seconds,
        ddp=True, rank=rank, world_size=world_size,
    )

    total_steps = len(loader) * args.epochs
    warmup_steps = min(5000, total_steps // 10)

    # AMP scaler
    use_amp = args.precision in ('fp16', 'bf16')
    amp_dtype = torch.bfloat16 if args.precision == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.precision == 'fp16'))

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        loader.sampler.set_epoch(epoch)
        encoder.train()
        if epoch > args.phase1_epochs:
            decoder.train()

        epoch_loss = 0.0
        epoch_wave = 0.0
        epoch_stft = 0.0
        epoch_commit = 0.0

        # Phase 2: unfreeze decoder
        if epoch == args.phase1_epochs + 1:
            for p in decoder.parameters():
                p.requires_grad = True
            trainable_params = list(encoder.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr * 0.5,
                                           betas=(0.8, 0.99), weight_decay=1e-5)
            if is_main:
                dec_trainable = sum(p.numel() for p in decoder.parameters())
                print(f"\n  Phase 2: Decoder unfrozen ({dec_trainable/1e6:.1f}M "
                      f"params), LR reduced to {args.lr * 0.5}")

        for batch_idx, waveform in enumerate(loader):
            global_step += 1
            waveform = waveform.to(device)

            # Cosine warmup + exponential decay schedule
            if global_step < warmup_steps:
                lr_scale = global_step / warmup_steps
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * lr_scale

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                # STFT
                stft_in = compute_stft(waveform, cfg)

                # Encoder
                embeddings = encoder(stft_in)  # [B, T_enc, 256]

                # RVQ (frozen, no grad)
                quantized, tokens = rvq_encode(embeddings, rvq, cfg)

                # Decoder
                stft_out = decoder(quantized)

                # iSTFT
                reconstructed = compute_istft(stft_out, cfg)

                # Loss
                loss_dict = criterion(waveform, reconstructed, embeddings, quantized)

            optimizer.zero_grad()
            if args.precision == 'fp16':
                scaler.scale(loss_dict['loss']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

            epoch_loss += loss_dict['loss'].item()
            epoch_wave += loss_dict['wave_loss'].item()
            epoch_stft += loss_dict['stft_loss'].item()
            epoch_commit += loss_dict['commit_loss'].item()

            if is_main and global_step % args.log_interval == 0:
                writer.add_scalar('loss/total', loss_dict['loss'].item(), global_step)
                writer.add_scalar('loss/waveform', loss_dict['wave_loss'].item(), global_step)
                writer.add_scalar('loss/stft', loss_dict['stft_loss'].item(), global_step)
                writer.add_scalar('loss/commit', loss_dict['commit_loss'].item(), global_step)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step)
                print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                      f"loss={loss_dict['loss'].item():.4f} "
                      f"wave={loss_dict['wave_loss'].item():.4f} "
                      f"stft={loss_dict['stft_loss'].item():.4f}")

        n_batches = len(loader)
        avg_loss = epoch_loss / n_batches
        avg_wave = epoch_wave / n_batches
        avg_stft = epoch_stft / n_batches
        avg_commit = epoch_commit / n_batches

        if is_main:
            writer.add_scalar('epoch/loss', avg_loss, epoch)
            writer.add_scalar('epoch/waveform', avg_wave, epoch)
            writer.add_scalar('epoch/stft', avg_stft, epoch)
            writer.add_scalar('epoch/commit', avg_commit, epoch)

            phase = "Phase1(encoder-only)" if epoch <= args.phase1_epochs else "Phase2(joint)"
            print(f"\n  Epoch {epoch} [{phase}] avg_loss={avg_loss:.4f}")

            if epoch % args.save_interval == 0:
                ckpt_path = os.path.join(args.output_dir, f'codec_epoch{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict(),
                    'rvq': rvq.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'cfg': cfg,
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")

    if is_main:
        # Final save
        final_path = os.path.join(args.output_dir, 'codec_final.pt')
        torch.save({
            'epoch': args.epochs,
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'rvq': rvq.state_dict(),
            'cfg': cfg,
        }, final_path)
        print(f"\n  Final checkpoint: {final_path}")
        writer.close()

    cleanup_ddp()


if __name__ == '__main__':
    train()
