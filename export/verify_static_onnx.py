"""Verify static-shape ONNX with attention mask."""
import sys, os, torch, numpy as np, onnxruntime as ort
sys.path.insert(0, '.')
from models.config import DepthFormerConfig
from models.depthformer import TemporalBodyStateful

torch.manual_seed(42)
np.random.seed(42)
cfg = DepthFormerConfig()
model = TemporalBodyStateful(cfg)
model.eval()

MAX_KV = 42
num_layers = cfg.temporal_spec.num_layers
num_heads = cfg.temporal_spec.num_heads
dim_per_head = cfg.temporal_spec.dim_per_head

def cos(a, b):
    a_f = a.numpy().reshape(-1).astype(np.float64) if hasattr(a, 'numpy') else a.reshape(-1).astype(np.float64)
    b_f = b.numpy().reshape(-1).astype(np.float64) if hasattr(b, 'numpy') else b.reshape(-1).astype(np.float64)
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f)))

# Export ONNX from same model (uses the earlier test_static_export.py output)
onnx_path = './exported/temporal_body.onnx'
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# Test 1: Full cache (42 positions filled)
print("=== Test 1: Full KV cache (42 pos) ===")
x = torch.randn(1, 1, 1024)
cond = torch.randn(1, 50, 256)
mask = torch.zeros(1, 1, 1, 44)  # All positions valid (full window)

kv_caches = []
for _ in range(num_layers):
    kv_caches.append({
        'self_kv': (
            torch.randn(1, num_heads, MAX_KV, dim_per_head) * 0.1,
            torch.randn(1, num_heads, MAX_KV, dim_per_head) * 0.1,
        ),
        'cross_kv': None,
    })

with torch.no_grad():
    pt_out, pt_new = model(x, cond, kv_caches, attention_mask=mask)

ort_inputs = {'x': x.numpy(), 'cond': cond.numpy(), 'attn_mask': mask.numpy()}
for i in range(num_layers):
    ort_inputs[f'self_k_{i}'] = kv_caches[i]['self_kv'][0].numpy()
    ort_inputs[f'self_v_{i}'] = kv_caches[i]['self_kv'][1].numpy()

ort_outputs = session.run(None, ort_inputs)
print(f'  Output cos: {cos(pt_out, ort_outputs[0]):.8f}')
for i in range(3):
    pt_sk = pt_new[i]['self_kv'][0][:, :, -MAX_KV:, :]
    pt_sv = pt_new[i]['self_kv'][1][:, :, -MAX_KV:, :]
    print(f'  Layer[{i}] K cos: {cos(pt_sk, ort_outputs[1+i*2]):.8f}, V cos: {cos(pt_sv, ort_outputs[1+i*2+1]):.8f}')

# Test 2: Empty cache (first frame, zero KV)
print("\n=== Test 2: Empty KV cache (first frame, zeros padded) ===")
mask_empty = torch.zeros(1, 1, 1, 44)
# Mask out padded positions (1 to 42, only current at 43 and sink at 0 are valid)
mask_empty[0, 0, 0, 1:43] = float('-inf')

kv_empty = []
for _ in range(num_layers):
    kv_empty.append({
        'self_kv': (
            torch.zeros(1, num_heads, MAX_KV, dim_per_head),
            torch.zeros(1, num_heads, MAX_KV, dim_per_head),
        ),
        'cross_kv': None,
    })

with torch.no_grad():
    pt_out_e, pt_new_e = model(x, cond, kv_empty, attention_mask=mask_empty)

ort_inputs_e = {'x': x.numpy(), 'cond': cond.numpy(), 'attn_mask': mask_empty.numpy()}
for i in range(num_layers):
    ort_inputs_e[f'self_k_{i}'] = kv_empty[i]['self_kv'][0].numpy()
    ort_inputs_e[f'self_v_{i}'] = kv_empty[i]['self_kv'][1].numpy()

ort_outputs_e = session.run(None, ort_inputs_e)
print(f'  Output cos: {cos(pt_out_e, ort_outputs_e[0]):.8f}')

# Test 3: Stateful loop with mask progression
print("\n=== Test 3: Stateful loop (5 frames, mask progression) ===")
x_frames = [torch.randn(1, 1, 1024) for _ in range(5)]
cond3 = torch.randn(1, 50, 256)

# Init
pt_kv = []
ort_inputs_3 = {'cond': cond3.numpy()}
for i in range(num_layers):
    pt_kv.append({
        'self_kv': (
            torch.zeros(1, num_heads, MAX_KV, dim_per_head),
            torch.zeros(1, num_heads, MAX_KV, dim_per_head),
        ),
        'cross_kv': None,
    })
    ort_inputs_3[f'self_k_{i}'] = np.zeros((1, num_heads, MAX_KV, dim_per_head), dtype=np.float32)
    ort_inputs_3[f'self_v_{i}'] = np.zeros((1, num_heads, MAX_KV, dim_per_head), dtype=np.float32)

all_pass = True
for fi in range(5):
    # Build mask for this frame
    m = torch.zeros(1, 1, 1, 44)
    valid_positions = fi + 1  # Number of valid positions in KV cache
    padded = MAX_KV - valid_positions
    if padded > 0:
        m[0, 0, 0, 1:1 + padded] = float('-inf')
    # Also mask old positions beyond window
    total_valid = 1 + valid_positions  # sink + valid cache positions
    if total_valid > 42:  # 41 window + 1 sink
        old = total_valid - 42
        m[0, 0, 0, 1:1 + old] = float('-inf')

    # PT
    with torch.no_grad():
        pt_out, pt_new_kv = model(x_frames[fi], cond3, pt_kv, attention_mask=m)

    # ORT
    ort_inputs_3['x'] = x_frames[fi].numpy()
    ort_inputs_3['attn_mask'] = m.numpy()
    ort_outputs_3 = session.run(None, ort_inputs_3)

    c = cos(pt_out, ort_outputs_3[0])
    if fi < 3 or c < 0.9999:
        print(f'  Frame {fi}: cos={c:.8f} {"[PASS]" if c > 0.9999 else "[FAIL]"}')
    if c < 0.9999:
        all_pass = False

    # Update
    pt_kv = []
    for i in range(num_layers):
        sk = pt_new_kv[i]['self_kv'][0]
        sv = pt_new_kv[i]['self_kv'][1]
        if sk.shape[2] > MAX_KV:
            sk = sk[:, :, -MAX_KV:, :]
            sv = sv[:, :, -MAX_KV:, :]
        pt_kv.append({'self_kv': (sk, sv), 'cross_kv': None})
        ort_inputs_3[f'self_k_{i}'] = ort_outputs_3[1 + i * 2]
        ort_inputs_3[f'self_v_{i}'] = ort_outputs_3[1 + i * 2 + 1]

print(f'\n  Result: {"[ALL PASS]" if all_pass else "[FAIL]"}')
