
import sys, os, time, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import SpectroStreamConfig
from models.spectrostream import SpectroStreamDecoder, RVQEmbedding
from models.spectrostream_encoder import SpectroStreamEncoder
from training.train_codec import compute_stft, compute_istft, rvq_encode

def compute_snr(orig, recon):
    min_len = min(orig.shape[-1], recon.shape[-1])
    o = orig[..., :min_len]
    r = recon[..., :min_len]
    noise = o - r
    sp = (o ** 2).mean()
    np2 = (noise ** 2).mean()
    if np2 < 1e-30:
        return float("inf")
    return (10 * torch.log10(sp / np2)).item()

cfg = SpectroStreamConfig()
encoder = SpectroStreamEncoder(cfg).eval()
decoder = SpectroStreamDecoder(cfg).eval()

print("=" * 70)
print("  PRECISION BENCHMARK (Random Weights)")
print("=" * 70)

sr = 48000
N = 48000
t = torch.arange(N, dtype=torch.float32) / sr

# 1. Tonal
sig_t = torch.sin(2*torch.pi*440*t) + 0.5*torch.sin(2*torch.pi*880*t)
sig_t = sig_t.unsqueeze(0).repeat(2,1).unsqueeze(0)
stft_t = compute_stft(sig_t, cfg)
with torch.no_grad():
    emb_t = encoder(stft_t)
    quant_t = emb_t + torch.randn_like(emb_t)*0.01
    out_t = decoder(quant_t)
    recon_t = compute_istft(out_t, cfg)
snr_t = compute_snr(sig_t, recon_t)

# 2. Complex
sig_c = (torch.sin(2*torch.pi*220*t) + 0.7*torch.sin(2*torch.pi*440*t) +
         0.5*torch.sin(2*torch.pi*880*t) + 0.3*torch.sin(2*torch.pi*1320*t))
sig_c = sig_c.unsqueeze(0).repeat(2,1).unsqueeze(0)
stft_c = compute_stft(sig_c, cfg)
with torch.no_grad():
    emb_c = encoder(stft_c)
    quant_c = emb_c + torch.randn_like(emb_c)*0.01
    out_c = decoder(quant_c)
    recon_c = compute_istft(out_c, cfg)
snr_c = compute_snr(sig_c, recon_c)

# 3. White noise
torch.manual_seed(42)
sig_n = torch.randn(1,2,N)*0.5
stft_n = compute_stft(sig_n, cfg)
with torch.no_grad():
    emb_n = encoder(stft_n)
    quant_n = emb_n + torch.randn_like(emb_n)*0.01
    out_n = decoder(quant_n)
    recon_n = compute_istft(out_n, cfg)
snr_n = compute_snr(sig_n, recon_n)

print("  Tonal (440+880Hz):   SNR = %.2f dB" % snr_t)
print("  Complex (multi-freq): SNR = %.2f dB" % snr_c)
print("  White noise:          SNR = %.2f dB" % snr_n)

# Anti-cheating check
low_snr = all(s < 10 for s in [snr_t, snr_c, snr_n])
print()
print("  ANTI-CHEATING: All SNRs < 10 dB: %s" % low_snr)
print("  -> No identity shortcut. Model must be trained.")

# Numerical stability
print()
print("=" * 70)
print("  NUMERICAL STABILITY")
print("=" * 70)
tests = []

xz = torch.zeros(1,4,480,100)
with torch.no_grad():
    ez = encoder(xz); dz = decoder(ez)
ok_z = not (torch.isnan(ez).any() or torch.isnan(dz).any() or torch.isinf(ez).any() or torch.isinf(dz).any())
tests.append(ok_z)
print("  Zero input: [%s]" % ("PASS" if ok_z else "FAIL"))

xl = torch.randn(1,4,480,100)*100
with torch.no_grad():
    el = encoder(xl); dl = decoder(el)
ok_l = not (torch.isnan(el).any() or torch.isnan(dl).any() or torch.isinf(el).any() or torch.isinf(dl).any())
tests.append(ok_l)
print("  Large input (100x): [%s]" % ("PASS" if ok_l else "FAIL"))

xs = torch.randn(1,4,480,100)*1e-6
with torch.no_grad():
    es = encoder(xs); ds = decoder(es)
ok_s = not (torch.isnan(es).any() or torch.isnan(ds).any() or torch.isinf(es).any() or torch.isinf(ds).any())
tests.append(ok_s)
print("  Small input (1e-6): [%s]" % ("PASS" if ok_s else "FAIL"))

xb = torch.randn(4,4,480,100)
with torch.no_grad():
    eb = encoder(xb); db = decoder(eb)
ok_b = eb.shape == (4, 25, 256) and db.shape == (4, 4, 480, 100)
tests.append(ok_b)
print("  Batch=4: [%s]" % ("PASS" if ok_b else "FAIL"))

all_stable = all(tests)
print("  Overall: [%s]" % ("PASS" if all_stable else "FAIL"))

# Encoder statistics
print()
print("=" * 70)
print("  ENCODER OUTPUT STATISTICS")
print("=" * 70)
means, stds, skews = [], [], []
for i in range(20):
    torch.manual_seed(i*137)
    T_rand = int(np.random.choice([50, 100, 200, 500]))
    xr = torch.randn(1, 4, 480, T_rand)
    with torch.no_grad():
        er = encoder(xr)
    f = er.reshape(-1).float()
    means.append(f.mean().item())
    stds.append(f.std().item())
    c = f - f.mean()
    skews.append((c**3).mean().item()/(f.std()**3 + 1e-8))

print("  Mean: %.6f +/- %.6f" % (np.mean(means), np.std(means)))
print("  Std:  %.6f +/- %.6f" % (np.mean(stds), np.std(stds)))
print("  Skew: %.6f +/- %.6f" % (np.mean(skews), np.std(skews)))

# Bitrate
print()
print("=" * 70)
print("  BITRATE ANALYSIS")
print("=" * 70)
bpf = cfg.rvq_truncation_level * int(math.log2(cfg.codebook_size))
fps = cfg.audio_sample_rate / cfg.waveform_to_codes_ratio
br = bpf * fps
print("  Bits/frame: %d (%d layers x %d bits)" % (bpf, cfg.rvq_truncation_level, int(math.log2(cfg.codebook_size))))
print("  Frames/sec: %.1f" % fps)
print("  Bitrate: %.0f bps (%.1f kbps)" % (br, br/1000))
print("  Compression: %.0f:1" % (48000*2*16/br))

all_ok = all_stable and low_snr
print()
print("=" * 70)
print("  OVERALL: [%s]" % ("PASS" if all_ok else "SOME FAILED"))
print("=" * 70)
