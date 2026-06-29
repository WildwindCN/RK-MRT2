"""模型架构验证测试"""
import torch
import sys
sys.path.insert(0, ".")

from models.config import DepthFormerConfig, SpectroStreamConfig
from models.depthformer import DepthFormer, TemporalBodyStateful, DepthBodyAR
from models.spectrostream import SpectroStreamDecoder, RVQEmbedding


def test_config():
    """验证配置"""
    cfg = DepthFormerConfig()
    assert cfg.temporal_spec.num_layers == 12
    assert cfg.temporal_spec.model_dims == 1024
    assert cfg.depth_spec.num_layers == 2
    assert cfg.depth_spec.model_dims == 768
    assert cfg.vocab_size == 12294  # num_codebooks * codebook_size + num_reserved
    assert cfg.per_rvq_vocab_size == 1030  # codebook_size + num_reserved
    assert cfg.num_codebooks == 12
    print("  [OK] Config")


def test_temporal_body():
    """验证 Temporal Body"""
    cfg = DepthFormerConfig()
    model = TemporalBodyStateful(cfg)
    model.eval()

    B, T, D = 2, 1, cfg.temporal_spec.model_dims
    D_enc = cfg.encoder_spec.model_dims  # 256
    x = torch.randn(B, T, D)
    cond = torch.randn(B, 50, D_enc)

    with torch.no_grad():
        out, kv_caches = model(x, cond)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    assert len(kv_caches) == 12
    for cache in kv_caches:
        assert "self_kv" in cache
        assert "cross_kv" in cache

    params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] TemporalBody: {params/1e6:.1f}M params, out shape={out.shape}")


def test_depth_body():
    """验证 Depth Body"""
    cfg = DepthFormerConfig()
    model = DepthBodyAR(cfg)
    model.eval()

    B, Q, D = 4, cfg.num_codebooks, cfg.temporal_spec.model_dims
    x = torch.randn(B, Q, D)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (B, Q, cfg.vocab_size), f"Expected {(B, Q, cfg.vocab_size)}, got {logits.shape}"

    params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] DepthBody: {params/1e6:.1f}M params, logits shape={logits.shape}")


def test_full_depthformer():
    """验证完整 DepthFormer"""
    cfg = DepthFormerConfig()
    model = DepthFormer(cfg)
    model.eval()

    B, T_cond, T_tok = 2, 50, 20
    D_enc = cfg.encoder_spec.model_dims
    cond = torch.randn(B, T_cond, D_enc)
    tokens = torch.randint(0, cfg.codebook_size, (B, T_tok, cfg.num_codebooks))

    with torch.no_grad():
        logits, loss = model(cond, tokens, return_loss=True)

    expected = (B, T_tok, cfg.num_codebooks, cfg.vocab_size)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    assert loss.item() > 0

    params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] DepthFormer: {params/1e6:.1f}M params, logits={list(logits.shape)}, loss={loss.item():.4f}")


def test_spectrostream_decoder():
    """验证 SpectroStream Decoder"""
    cfg = SpectroStreamConfig()
    model = SpectroStreamDecoder(cfg)
    model.eval()

    B, T, D = 1, 25, cfg.embedding_dim
    x = torch.randn(B, T, D)

    with torch.no_grad():
        out = model(x)

    expected_T = T * cfg.total_time_stride
    assert out.shape[0] == B
    assert out.shape[1] == cfg.num_channels  # 4
    assert out.shape[2] == cfg.num_bins       # 480
    assert out.shape[3] == expected_T         # T * 4

    params = sum(p.numel() for p in model.parameters())
    print(f"  [OK] SpectroStream Decoder: {params/1e6:.1f}M params, out shape={list(out.shape)}")


def test_rvq_embedding():
    """验证 RVQ Embedding"""
    cfg = SpectroStreamConfig()
    emb = RVQEmbedding(cfg)
    emb.eval()

    B, T, Q = 2, 10, cfg.rvq_truncation_level
    codes = torch.randint(0, cfg.codebook_size, (B, T, Q))

    with torch.no_grad():
        out = emb(codes)

    assert out.shape == (B, T, cfg.embedding_dim)
    print(f"  [OK] RVQ Embedding: {sum(p.numel() for p in emb.parameters())/1e6:.1f}M params, out shape={list(out.shape)}")


if __name__ == "__main__":
    print("=" * 60)
    print("RK-MRT2 模型架构验证")
    print("=" * 60)

    test_config()
    test_temporal_body()
    test_depth_body()
    test_full_depthformer()
    test_spectrostream_decoder()
    test_rvq_embedding()

    print("\n" + "=" * 60)
    print("全部通过!")
    print("=" * 60)
