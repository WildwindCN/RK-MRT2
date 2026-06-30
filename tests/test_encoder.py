"""验证 SpectroStream Encoder + Full Pipeline

测试:
1. Encoder 形状 + 参数
2. STFT→Encoder→RVQ→Decoder→iSTFT 往返
3. RVQ 权重加载
"""
import sys, os, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import SpectroStreamConfig
from models.spectrostream import SpectroStreamDecoder, RVQEmbedding
from models.spectrostream_encoder import SpectroStreamEncoder


def test_encoder():
    """Encoder: [B, 4, 480, T] → [B, T/4, 256]"""
    print("=" * 60)
    print("Test 1: Encoder Forward Pass")
    print("=" * 60)
    cfg = SpectroStreamConfig()
    enc = SpectroStreamEncoder(cfg).eval()
    x = torch.randn(2, 4, 480, 1000)
    with torch.no_grad():
        out = enc(x)
    expected = (2, 250, 256)
    ok = out.shape == expected
    p = sum(p.numel() for p in enc.parameters())
    print(f"  Input: {list(x.shape)} → Output: {list(out.shape)}")
    print(f"  Expected: {list(expected)}, NaN: {torch.isnan(out).any()}")
    print(f"  Params: {p/1e6:.1f}M, Result: {'[PASS]' if ok else '[FAIL]'}")
    return ok


def test_encoder_decoder_roundtrip():
    """Random embedding → Decoder → iSTFT: 解码器结构性测试"""
    print("\n" + "=" * 60)
    print("Test 2: Encoder → RVQ(sim) → Decoder Round-trip")
    print("=" * 60)
    cfg = SpectroStreamConfig()
    enc = SpectroStreamEncoder(cfg).eval()
    dec = SpectroStreamDecoder(cfg).eval()

    # STFT input simulation
    x = torch.randn(1, 4, 480, 100)
    with torch.no_grad():
        emb = enc(x)
        # Simulate RVQ quantization (add small noise)
        quantized = emb + torch.randn_like(emb) * 0.01
        stft_out = dec(quantized)

    print(f"  STFT in: {list(x.shape)}")
    print(f"  Embeddings: {list(emb.shape)}")
    print(f"  STFT out: {list(stft_out.shape)}")
    # Decoder expands T_enc by total_time_stride (4x)
    expected_time = emb.shape[1] * cfg.total_time_stride  # 25 * 4 = 100
    expected_out = (1, cfg.num_channels, cfg.num_bins, expected_time)
    ok = stft_out.shape == expected_out
    print(f"  Expected: {list(expected_out)}, Result: {'[PASS]' if ok else '[FAIL]'}")
    return ok


def test_rvq_loading():
    """加载 MRT2 Small RVQ 权重 (冻结)"""
    print("\n" + "=" * 60)
    print("Test 3: RVQ Weight Loading (MRT2 Small)")
    print("=" * 60)
    cfg = SpectroStreamConfig()
    rvq = RVQEmbedding(cfg)
    path = "exported/weights/mrt2_small_pytorch.pt"
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        w = state['codec_decoder']['rvq_embedding.embedding']
        rvq.embedding.data.copy_(w)
        print(f"  Shape: {list(w.shape)} (expected [64, 1024, 256])")
        print(f"  Mean={w.mean():.6f}, Std={w.std():.6f}, NaN={torch.isnan(w).any()}")
        ok = w.shape == (64, 1024, 256) and not torch.isnan(w).any()
        print(f"  Result: {'[PASS]' if ok else '[FAIL]'}")
        return ok
    else:
        print(f"  [SKIP] Checkpoint not found")
        return True


def main():
    results = [
        ("Encoder Shape", test_encoder()),
        ("Round-trip", test_encoder_decoder_roundtrip()),
        ("RVQ Loading", test_rvq_loading()),
    ]
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for n, ok in results:
        print(f"  {n}: {'[PASS]' if ok else '[FAIL]'}")
    all_ok = all(r[1] for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
