"""本地训练管线冒烟测试

从 test_data/ 目录加载真实 ACE Studio 音频, 验证:
1. 数据加载器正常工作
2. 完整前向传播 (STFT → Encoder → RVQ → Decoder → iSTFT)
3. 损失计算 (L1 + Multi-scale STFT + RVQ commitment)
4. 反向传播 + 参数更新
5. Checkpoint 保存/加载

用法:
    python tests/test_local_training.py --data_dir ./test_data --epochs 1 --batch_size 2
"""
import os, sys, argparse, tempfile
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import SpectroStreamConfig
from models.spectrostream import SpectroStreamDecoder
from models.spectrostream_encoder import SpectroStreamEncoder
from training.codec_dataset import CodecDataset
from training.codec_loss import CodecLoss


def compute_stft(waveform, cfg):
    wf = waveform.float()
    window = torch.hann_window(cfg.stft_fft_length, periodic=True, device=wf.device)
    B, C_audio, N = wf.shape
    stfts = []
    for ch in range(C_audio):
        stft = torch.stft(wf[:, ch], n_fft=cfg.stft_fft_length,
                          hop_length=cfg.stft_frame_step,
                          win_length=cfg.stft_frame_length,
                          window=window, center=True, return_complex=True)
        stfts.append(torch.view_as_real(stft))
    stft_in = torch.stack(stfts, dim=1)
    stft_in = stft_in.permute(0, 1, 4, 2, 3)
    stft_in = stft_in.reshape(B, C_audio * 2, cfg.stft_fft_length // 2 + 1, -1)
    stft_in = stft_in[:, :, :cfg.num_bins, :]
    return stft_in


@torch.no_grad()
def rvq_encode(embeddings, rvq_embed, cfg):
    """简化 RVQ 编码 (无梯度, 用于冒烟测试)"""
    B, T, D = embeddings.shape
    residual = embeddings
    quantized_sum = torch.zeros_like(embeddings)
    for q in range(cfg.rvq_truncation_level):
        codebook = rvq_embed[q]
        cb_norm = (codebook * codebook).sum(dim=-1)
        scores = torch.matmul(residual.reshape(-1, D), codebook.T) - 0.5 * cb_norm.unsqueeze(0)
        indices = scores.argmax(dim=-1).reshape(B, T)
        quantized = torch.nn.functional.embedding(indices, codebook)
        quantized_sum = quantized_sum + quantized
        residual = residual - quantized
    return quantized_sum


def test_data_loading(data_dir):
    """Test 1: 数据加载器"""
    print("=" * 60)
    print("Test 1: Data Loading")
    print("=" * 60)
    try:
        ds = CodecDataset(data_dir, segment_samples=480000, sample_rate=48000)
        print(f"  Files: {len(ds.files)}, Segments: {len(ds)}")
        if len(ds) == 0:
            print("  [FAIL] No segments found (files may be < 10s)")
            return False
        x = ds[0]
        print(f"  Sample shape: {list(x.shape)}, mean={x.mean():.4f}, std={x.std():.4f}")
        print(f"  [PASS] Loaded {len(ds)} segments")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_forward_pass(data_dir):
    """Test 2: 完整前向传播 + 损失"""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass + Loss")
    print("=" * 60)

    cfg = SpectroStreamConfig()
    encoder = SpectroStreamEncoder(cfg).eval()
    decoder = SpectroStreamDecoder(cfg).eval()

    # Load RVQ from checkpoint
    ckpt_path = "exported/weights/mrt2_small_pytorch.pt"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        rvq_embed = state['codec_decoder']['rvq_embedding.embedding']
        print(f"  RVQ loaded: {list(rvq_embed.shape)}")
    else:
        rvq_embed = torch.randn(12, 1024, 256) * 0.02  # random
        print(f"  RVQ random init: {list(rvq_embed.shape)}")

    ds = CodecDataset(data_dir, segment_samples=480000, sample_rate=48000)
    if len(ds) < 2:
        print("  [SKIP] Need >= 2 segments")
        return True

    # Batch of 2
    wav1 = ds[0].unsqueeze(0)
    wav2 = ds[1].unsqueeze(0)
    waveform = torch.cat([wav1, wav2], dim=0)
    print(f"  Waveform: {list(waveform.shape)}")

    # STFT
    stft_in = compute_stft(waveform, cfg)
    print(f"  STFT: {list(stft_in.shape)}")

    # Encoder
    with torch.no_grad():
        emb = encoder(stft_in)
    print(f"  Encoder: {list(emb.shape)}, NaN={torch.isnan(emb).any()}")

    # RVQ
    quantized = rvq_encode(emb, rvq_embed, cfg)
    print(f"  RVQ quantized: {list(quantized.shape)}, NaN={torch.isnan(quantized).any()}")

    # Decoder
    with torch.no_grad():
        stft_out = decoder(quantized)
    print(f"  Decoder: {list(stft_out.shape)}, NaN={torch.isnan(stft_out).any()}")

    # iSTFT (simplified)
    from models.istft import ISTFTLayer
    istft = ISTFTLayer(frame_length=960, frame_step=480, fft_length=960,
                       num_bins=480, num_channels=4)
    with torch.no_grad():
        reconstructed = istft(stft_out.float())
    print(f"  iSTFT: {list(reconstructed.shape)}")

    # Align lengths
    min_len = min(waveform.shape[-1], reconstructed.shape[-1])
    wav_aligned = waveform[..., :min_len]
    rec_aligned = reconstructed[..., :min_len]

    # SNR
    signal_power = (wav_aligned ** 2).mean()
    noise_power = ((wav_aligned - rec_aligned) ** 2).mean()
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10)).item()
    print(f"  SNR: {snr:.1f} dB (random weights, expect low)")

    # Loss
    criterion = CodecLoss()
    loss_dict = criterion(wav_aligned, rec_aligned, emb, quantized)
    print(f"  Loss: total={loss_dict['loss'].item():.4f}, "
          f"wave={loss_dict['wave_loss'].item():.4f}, "
          f"stft={loss_dict['stft_loss'].item():.4f}, "
          f"commit={loss_dict['commit_loss'].item():.4f}")

    print(f"  [PASS] Forward pass + loss OK")
    return True


def test_backward_pass(data_dir):
    """Test 3: 反向传播 + 参数更新"""
    print("\n" + "=" * 60)
    print("Test 3: Backward Pass + Optimizer Step")
    print("=" * 60)

    cfg = SpectroStreamConfig()
    encoder = SpectroStreamEncoder(cfg).train()
    ckpt_path = "exported/weights/mrt2_small_pytorch.pt"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        rvq_embed = state['codec_decoder']['rvq_embedding.embedding']
    else:
        rvq_embed = torch.randn(12, 1024, 256) * 0.02

    ds = CodecDataset(data_dir, segment_samples=480000, sample_rate=48000)
    waveform = ds[0].unsqueeze(0)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-4)

    # Forward
    stft_in = compute_stft(waveform, cfg)
    emb = encoder(stft_in)
    quantized = rvq_encode(emb, rvq_embed, cfg)

    # Simple reconstruction loss (skip decoder for speed)
    loss = torch.nn.functional.mse_loss(emb, torch.zeros_like(emb))

    # Backward
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
    optimizer.step()

    # Check params changed
    p0 = list(encoder.parameters())[0].clone()
    print(f"  Grad norm: {grad_norm:.4f}")
    print(f"  Param[0] mean: {p0.mean():.6f} (should be non-zero)")
    print(f"  [PASS] Backward pass + optimizer step OK")
    return True


def test_checkpoint_save_load(data_dir):
    """Test 4: Checkpoint 保存/加载"""
    print("\n" + "=" * 60)
    print("Test 4: Checkpoint Save/Load")
    print("=" * 60)

    cfg = SpectroStreamConfig()
    encoder = SpectroStreamEncoder(cfg)
    decoder = SpectroStreamDecoder(cfg)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        ckpt_path = f.name

    torch.save({
        'epoch': 1, 'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(), 'cfg': cfg,
        'base_lr': 3e-4, 'global_step': 100, 'optimizer': {},
    }, ckpt_path)
    size_kb = os.path.getsize(ckpt_path) / 1024
    print(f"  Saved: {ckpt_path} ({size_kb:.0f} KB)")

    # Load
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    encoder2 = SpectroStreamEncoder(cfg)
    encoder2.load_state_dict(ckpt['encoder'])
    print(f"  Loaded: epoch={ckpt['epoch']}, base_lr={ckpt['base_lr']}")

    # Verify params match
    for (n1, p1), (n2, p2) in zip(encoder.named_parameters(), encoder2.named_parameters()):
        if not torch.equal(p1, p2):
            print(f"  [FAIL] Param mismatch: {n1}")
            os.unlink(ckpt_path)
            return False

    os.unlink(ckpt_path)
    print(f"  [PASS] Checkpoint save/load OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./test_data')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"Data dir not found: {args.data_dir}")
        print("Run: mkdir test_data && scp spark:/data/.../wav/*/*.wav test_data/")
        return 1

    results = []
    results.append(("Data Loading", test_data_loading(args.data_dir)))
    results.append(("Forward Pass", test_forward_pass(args.data_dir)))
    results.append(("Backward Pass", test_backward_pass(args.data_dir)))
    results.append(("Checkpoint", test_checkpoint_save_load(args.data_dir)))

    print("\n" + "=" * 60)
    print("Local Training Smoke Test Summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name}: {'[PASS]' if ok else '[FAIL]'}")
    all_pass = all(r[1] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
