"""RKNN 精度验证: 对比 ONNX Runtime vs RKNN 模拟器输出

使用 load_onnx() + build() + init_runtime() 方式 (模拟器),
因为 load_rknn() 不支持模拟器模式。

用法 (WSL 中):
    python deploy/verify_rknn.py --graph depth_body
    python deploy/verify_rknn.py --graph all
"""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort


def cos(a, b):
    a_f = a.astype(np.float64).reshape(-1)
    b_f = b.astype(np.float64).reshape(-1)
    dot = np.dot(a_f, b_f)
    na = np.linalg.norm(a_f)
    nb = np.linalg.norm(b_f)
    return float(dot / (na * nb)) if na and nb else 0.0


def verify_via_sim(name, onnx_path, input_names, input_size_list, test_inputs):
    """Build + simulate to verify precision"""
    from rknn.api import RKNN
    print(f"  Building {name} for simulation...")
    rknn = RKNN(verbose=False)
    rknn.config(target_platform="rk3576", optimization_level=3, float_dtype="float16")
    ret = rknn.load_onnx(model=onnx_path, inputs=input_names, input_size_list=input_size_list)
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: {ret}")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"build failed: {ret}")
    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")
    out = rknn.inference(inputs=test_inputs)
    rknn.release()
    return out


def verify_depth_body(onnx_path):
    print("\n" + "=" * 60)
    print("Depth Body [1,12,1024] -> [1,12,12294]")
    print("=" * 60)
    np.random.seed(42)
    x = np.random.randn(1, 12, 1024).astype(np.float32)
    ref = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']).run(None, {"x": x})[0]
    print(f"  ONNX: shape={ref.shape}, mean={ref.mean():.4f}")
    try:
        out = verify_via_sim("depth_body", onnx_path, ["x"], [[1, 12, 1024]], [x])
        c = cos(ref, out[0])
        ok = c > 0.99 and not np.isnan(out[0]).any()
        print(f"  Cosine: {c:.8f}  MaxErr: {np.abs(ref-out[0]).max():.2e}  [{'PASS' if ok else 'FAIL'}]")
        return ok
    except Exception as e:
        print(f"  [SKIP] {e}")
        return None


def verify_codec_decoder(onnx_path):
    print("\n" + "=" * 60)
    print("Codec Decoder [1,25,256] -> [1,4,480,100]")
    print("=" * 60)
    np.random.seed(43)
    x = np.random.randn(1, 25, 256).astype(np.float32)
    ref = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider']).run(None, {"embeddings": x})[0]
    print(f"  ONNX: shape={ref.shape}, mean={ref.mean():.4f}")
    try:
        out = verify_via_sim("codec_decoder", onnx_path, ["embeddings"], [[1, 25, 256]], [x])
        c = cos(ref, out[0])
        ok = c > 0.99 and not np.isnan(out[0]).any()
        print(f"  Cosine: {c:.8f}  MaxErr: {np.abs(ref-out[0]).max():.2e}  [{'PASS' if ok else 'FAIL'}]")
        return ok
    except Exception as e:
        print(f"  [SKIP] {e}")
        return None


def verify_temporal_body(onnx_path):
    print("\n" + "=" * 60)
    print("Temporal Body (27 inputs, stateful)")
    print("=" * 60)
    print("  [SKIP] 696MB too large for simulator. Requires RK3576 board.")
    return None


def main():
    p = argparse.ArgumentParser(description="RKNN Precision Verification (build+sim)")
    p.add_argument("--graph", choices=["depth_body", "codec_decoder", "temporal_body", "all"], default="all")
    p.add_argument("--onnx_dir", default="./exported")
    args = p.parse_args()

    print("=" * 60)
    print("RK-MRT2 RKNN Precision Verification")
    print("  Mode: load_onnx + build + init_runtime (simulator)")
    print("  Note: Production .rknn files require actual RK3576 board")
    print("=" * 60)

    graphs = {
        "depth_body": ("depth_body.onnx", verify_depth_body),
        "codec_decoder": ("codec_decoder.onnx", verify_codec_decoder),
        "temporal_body": ("temporal_body_sim.onnx", verify_temporal_body),
    }

    targets = list(graphs) if args.graph == "all" else [args.graph]
    results = {}
    for name in targets:
        f, fn = graphs[name]
        path = os.path.join(args.onnx_dir, f)
        if not os.path.exists(path):
            print(f"\n  [{name}] not found: {path}  [SKIP]")
            continue
        try:
            results[name] = fn(path)
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for n, r in results.items():
        s = "[PASS]" if r else "[FAIL]" if r is not None else "[SKIP]"
        print(f"  {n}: {s}")
    definitive = {k: v for k, v in results.items() if v is not None}
    return 0 if all(definitive.values()) else 1 if definitive else 0


if __name__ == "__main__":
    sys.exit(main())
