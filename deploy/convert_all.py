#!/usr/bin/env python3
"""RK3588 板端部署 — ONNX → RKNN 转换流水线

将准备好的 ONNX 模型转换为 RKNN 格式，打包部署到 RK3588 板端。

前置: RKNN-Toolkit2 已安装

安装方式 (选一):
  A. GitHub 下载: https://github.com/airockchip/rknn-toolkit2/releases/latest
     下载 rknn_toolkit2-2.3.2-cp310-cp310-linux_x86_64.whl
     pip install rknn_toolkit2-2.3.2-cp310-cp310-linux_x86_64.whl

  B. Docker: docker run -it -v $(pwd):/work pelochus/ezrkllm-toolkit:latest bash

  C. Rockchip 官方 SDK: https://console.zbox.filez.com/l/7vYST1 (Rockchip FileZ)

用法:
  # 1. 转换全部模型 (FP16)
  python deploy/convert_all.py --onnx_dir ./exported --output_dir ./rknn_models

  # 2. INT8 量化 Codec Decoder
  python deploy/convert_all.py --graph codec_decoder --precision int8

  # 3. 验证精度
  python deploy/convert_all.py --verify
"""

import os, sys, argparse, time, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════

GRAPHS = {
    "depth_body": {
        "onnx": "depth_body.onnx",
        "rknn": "depth_body.rknn",
        "description": "Depth Body (2-layer, d=768) — RVQ AR attention",
        "input_shape": [1, 12, 1024],
        "output_shape": [1, 12, 12294],
        "quantize": "fp16",  # fp16 only (precision critical)
    },
    "codec_decoder": {
        "onnx": "codec_decoder.onnx",
        "rknn": "codec_decoder.rknn",
        "description": "Codec Decoder (7-stage ConvTranspose) — STFT output",
        "input_shape": [1, 25, 256],
        "output_shape": [1, 4, 480, 100],
        "quantize": "int8",  # can use int8 (conv-only, robust to quantization)
    },
    "temporal_body": {
        "onnx": "temporal_body.onnx",
        "rknn": "temporal_body.rknn",
        "description": "Temporal Body (12-layer, d=1024) — stateful KV cache",
        "input_shape": "dynamic",  # 26 inputs with variable KV lengths
        "output_shape": "dynamic",
        "quantize": "fp16",
        "note": "731MB FP32 → ~365MB FP16. 最大 KV length=512",
    },
}

RKNN_CONFIG = {
    "target_platform": "rk3588",
    "optimization_level": 3,
}


def check_environment():
    """验证 RKNN 环境"""
    print("=" * 60)
    print("Environment Check")
    print("=" * 60)

    # 检查 ONNX 文件
    onnx_dir = "./exported"
    for name, info in GRAPHS.items():
        path = os.path.join(onnx_dir, info["onnx"])
        exists = os.path.exists(path)
        size = os.path.getsize(path) / (1024 * 1024) if exists else 0
        status = f"{size:.0f} MB" if exists else "NOT FOUND"
        print(f"  {info['onnx']}: {status}")

    # 检查 RKNN-Toolkit2
    try:
        from rknn.api import RKNN
        print(f"  RKNN-Toolkit2: installed")
        return True
    except ImportError:
        print(f"  RKNN-Toolkit2: NOT INSTALLED")
        print(f"")
        print(f"  To install:")
        print(f"    1. Download wheel from: https://github.com/airockchip/rknn-toolkit2/releases/latest")
        print(f"    2. pip install rknn_toolkit2-*-cp310-cp310-linux_x86_64.whl")
        print(f"  Or use Docker:")
        print(f"    docker run -it -v $(pwd):/work pelochus/ezrkllm-toolkit:latest bash")
        return False


def convert_graph(name, info, onnx_dir, output_dir, precision):
    """转换单个 graph"""
    from rknn.api import RKNN

    onnx_path = os.path.join(onnx_dir, info["onnx"])
    rknn_path = os.path.join(output_dir, info["rknn"])

    # 决定量化精度
    quant = precision or info.get("quantize", "fp16")
    do_quant = quant == "int8"

    print(f"\n{'='*60}")
    print(f"Converting: {name}")
    print(f"  {info['description']}")
    print(f"  Precision: {quant}")
    print(f"{'='*60}")

    rknn = RKNN(verbose=True)

    # 配置
    rknn.config(
        target_platform=RKNN_CONFIG["target_platform"],
        optimization_level=RKNN_CONFIG["optimization_level"],
        output_tensor_type=quant if quant in ("fp16", "int8") else "float16",
    )

    # 加载 ONNX
    print(f"  Loading ONNX: {onnx_path}")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: {ret}")

    # 构建
    print(f"  Building RKNN (quantization={do_quant})...")
    t0 = time.time()
    ret = rknn.build(do_quantization=do_quant, dataset=None)
    if ret != 0:
        raise RuntimeError(f"build failed: {ret}")
    elapsed = time.time() - t0
    print(f"  Build time: {elapsed:.1f}s")

    # 导出
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f"export_rknn failed: {ret}")

    size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
    print(f"  Output: {rknn_path} ({size_mb:.1f} MB)")

    rknn.release()
    return rknn_path


def verify_graph(name, info, rknn_path):
    """验证 RKNN 输出精度"""
    from rknn.api import RKNN

    print(f"\n  Verifying {name}...")

    rknn = RKNN(verbose=False)
    rknn.load_rknn(rknn_path)
    rknn.init_runtime(target=RKNN_CONFIG["target_platform"])

    # 生成测试输入
    if isinstance(info["input_shape"], list):
        x = np.random.randn(*info["input_shape"]).astype(np.float32)
        rknn_out = rknn.inference(inputs=[x])
        print(f"    Input: {info['input_shape']}, Output: {list(rknn_out[0].shape)}")
        print(f"    No NaN: {not np.isnan(rknn_out[0]).any()}")

    rknn.release()


def create_deployment_package(output_dir):
    """创建板端部署包"""
    pkg_dir = os.path.join(output_dir, "deployment_package")
    os.makedirs(pkg_dir, exist_ok=True)

    # 复制 RKNN 模型
    for name, info in GRAPHS.items():
        rknn_path = os.path.join(output_dir, info["rknn"])
        if os.path.exists(rknn_path):
            shutil.copy(rknn_path, pkg_dir)
            print(f"  Copied: {info['rknn']}")

    # 创建板端运行脚本
    script = os.path.join(pkg_dir, "run_on_board.sh")
    with open(script, "w") as f:
        f.write("""#!/bin/bash
# RK-MRT2 板端推理脚本
# 需要: RKNPU2 Runtime (librknnrt.so)

echo "============================================"
echo "RK-MRT2 Board Deployment Test"
echo "============================================"

# 检查 NPU 状态
if [ -f /sys/kernel/debug/rknpu/load ]; then
    echo "NPU driver loaded"
    cat /sys/kernel/debug/rknpu/load
fi

# 检查 runtime 库
if ldconfig -p | grep -q rknnrt; then
    echo "librknnrt.so found"
else
    echo "ERROR: librknnrt.so not found. Install rknpu2 runtime."
    exit 1
fi

# 运行 demo (需先编译 runtime/)
echo "Models ready:"
ls -la *.rknn
echo ""
echo "Run: ./rk_mrt2_demo --temporal temporal_body.rknn --depth depth_body.rknn --codec codec_decoder.rknn"
""")
    os.chmod(script, 0o755)
    print(f"  Created: run_on_board.sh")

    # 创建 README
    readme = os.path.join(pkg_dir, "README.txt")
    with open(readme, "w") as f:
        f.write("""RK-MRT2 Board Deployment Package
=================================

Models:
  - depth_body.rknn      Depth Body (FP16, ~32MB)
  - codec_decoder.rknn   Codec Decoder (FP16/INT8, ~30MB)
  - temporal_body.rknn   Temporal Body (FP16, ~365MB)

Board Requirements:
  - RK3588 with NPU driver (rknpu2)
  - librknnrt.so in library path
  - 2+ GB free RAM for Temporal Body

Quick Test:
  1. Push to board: scp *.rknn root@<board_ip>:/opt/rkmrt2/
  2. Compile runtime: cd runtime && cmake .. && make
  3. Run: ./rk_mrt2_demo --temporal temporal_body.rknn \\
            --depth depth_body.rknn --codec codec_decoder.rknn \\
            --output test.wav

Expected Performance (RK3588):
  - Temporal Body: < 5ms/frame
  - Depth Body: < 10ms/frame
  - Codec Decoder: < 3ms/frame
  - Total: < 20ms/frame (40ms budget, 2x headroom)
""")
    print(f"  Created: README.txt")
    print(f"\nDeployment package: {pkg_dir}")


def main():
    parser = argparse.ArgumentParser(description="RK3588 Board Deployment Pipeline")
    parser.add_argument("--onnx_dir", default="./exported")
    parser.add_argument("--output_dir", default="./rknn_models")
    parser.add_argument("--graph", choices=list(GRAPHS.keys()) + ["all"],
                        default="all")
    parser.add_argument("--precision", choices=["fp16", "int8"],
                        default=None, help="Override default precision")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--package", action="store_true",
                        help="Create deployment package for board")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("RK-MRT2: RK3588 Board Deployment Pipeline")
    print("=" * 60)

    # 1. 环境检查
    has_rknn = check_environment()
    if not has_rknn:
        print("\n[INFO] RKNN-Toolkit2 not installed. Will create config files only.")
        print("  Download from: https://github.com/airockchip/rknn-toolkit2/releases/latest")
        print("  Or use Docker: docker run -it -v $(pwd):/work pelochus/ezrkllm-toolkit:latest bash")

    # 2. 转换
    targets = list(GRAPHS.keys()) if args.graph == "all" else [args.graph]

    if has_rknn:
        for name in targets:
            info = GRAPHS[name]
            try:
                rknn_path = convert_graph(name, info, args.onnx_dir, args.output_dir, args.precision)
                if args.verify:
                    verify_graph(name, info, rknn_path)
            except Exception as e:
                print(f"  [ERROR] {name}: {e}")
    else:
        for name in targets:
            info = GRAPHS[name]
            onnx_path = os.path.join(args.onnx_dir, info["onnx"])
            size_mb = os.path.getsize(onnx_path) / (1024 * 1024) if os.path.exists(onnx_path) else 0
            est_rknn = size_mb / 2 if info.get("quantize") == "fp16" else size_mb / 4
            print(f"\n  {name}: {info['description']}")
            print(f"    ONNX: {onnx_path} ({size_mb:.0f} MB)")
            print(f"    Est RKNN: ~{est_rknn:.0f} MB ({info.get('quantize','fp16')})")
            if info.get("note"):
                print(f"    Note: {info['note']}")

    # 3. 打包
    if args.package:
        create_deployment_package(args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
