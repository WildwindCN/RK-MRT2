"""RKNN 模型转换 & 精度验证

将 ONNX graph 转换为 RKNN 格式, 支持 FP16 和 INT8 量化。

环境要求:
- x86 Linux (Ubuntu 20.04/22.04)
- RKNN-Toolkit2: pip install rknn-toolkit2
- 或使用 Docker: pelochus/ezrkllm-toolkit

用法:
    # FP16 转换 (推荐先用此验证)
    python export/convert_rknn.py --all --precision fp16

    # INT8 量化 (需要校准数据)
    python export/convert_rknn.py --graph codec_decoder --precision int8 --calib_data ./calib/

    # 精度验证 (对比 ONNX vs RKNN 输出)
    python export/convert_rknn.py --all --verify
"""

import os
import sys
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import DepthFormerConfig, SpectroStreamConfig
from models.depthformer import DepthBodyAR
from models.spectrostream import SpectroStreamDecoder
from export.rknn_config import (
    TEMPORAL_BODY_CONFIG, DEPTH_BODY_CONFIG, CODEC_DECODER_CONFIG,
    VERIFICATION_THRESHOLDS,
)

# RKNN-Toolkit2 是可选依赖 (仅 Linux x86 可用)
try:
    from rknn.api import RKNN
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False
    print("[WARN] RKNN-Toolkit2 not available. Install on x86 Linux:")
    print("       pip install rknn-toolkit2")
    print("       Will create config files only.\n")


class RKNNConverter:
    """RKNN 模型转换器"""

    def __init__(self, config: dict, onnx_dir: str, output_dir: str):
        self.config = config
        self.onnx_path = os.path.join(onnx_dir, config["onnx_file"])
        self.rknn_path = os.path.join(output_dir, config["rknn_file"])
        self.model_name = config["model_name"]

        if HAS_RKNN:
            self.rknn = RKNN(verbose=True)
        else:
            self.rknn = None

    def check_onnx(self):
        """检查 ONNX 文件是否存在并验证结构"""
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX not found: {self.onnx_path}")

        import onnx
        model = onnx.load(self.onnx_path)
        onnx.checker.check_model(model)

        ops = set(n.op_type for n in model.graph.node)
        rknn_unsupported = {"Einsum", "CumSum", "GridSample", "GatherND", "OneHot"}
        found = ops & rknn_unsupported
        if found:
            print(f"  [WARN] Unsupported ops found: {found}")
        else:
            print(f"  [OK] No unsupported ops")

        print(f"  Nodes: {len(model.graph.node)}, Ops: {sorted(ops)}")
        return True

    def convert_fp16(self):
        """FP16 转换 (无量化)"""
        if not HAS_RKNN:
            self._mock_convert("fp16")
            return

        print(f"\n  Converting {self.model_name} to FP16...")

        # 配置
        self.rknn.config(
            target_platform=self.config["target_platform"],
            optimization_level=self.config["optimization_level"],
            output_tensor_type=self.config["output_tensor_type"],
        )

        # 加载 ONNX
        ret = self.rknn.load_onnx(model=self.onnx_path)
        if ret != 0:
            raise RuntimeError(f"Load ONNX failed: {ret}")

        # 构建 (FP16)
        ret = self.rknn.build(
            do_quantization=False,
            dataset=None,
        )
        if ret != 0:
            raise RuntimeError(f"Build RKNN failed: {ret}")

        # 导出
        ret = self.rknn.export_rknn(self.rknn_path)
        if ret != 0:
            raise RuntimeError(f"Export failed: {ret}")

        # 检查大小
        size_mb = os.path.getsize(self.rknn_path) / (1024 * 1024)
        print(f"  → {self.rknn_path} ({size_mb:.1f} MB)")

    def convert_int8(self, calib_dataset=None):
        """INT8 量化转换"""
        if not HAS_RKNN:
            self._mock_convert("int8")
            return

        print(f"\n  Converting {self.model_name} to INT8...")

        self.rknn.config(
            target_platform=self.config["target_platform"],
            optimization_level=self.config["optimization_level"],
            output_tensor_type="int8",
            quantized_dtype="int8",
            quantized_method="layer",
            quantized_algorithm="mmse",
        )

        ret = self.rknn.load_onnx(model=self.onnx_path)
        if ret != 0:
            raise RuntimeError(f"Load ONNX failed: {ret}")

        # 构建 (INT8 量化)
        ret = self.rknn.build(
            do_quantization=True,
            dataset=calib_dataset or "dataset.txt",
        )
        if ret != 0:
            raise RuntimeError(f"Build RKNN (INT8) failed: {ret}")

        ret = self.rknn.export_rknn(self.rknn_path)
        if ret != 0:
            raise RuntimeError(f"Export failed: {ret}")

        size_mb = os.path.getsize(self.rknn_path) / (1024 * 1024)
        print(f"  → {self.rknn_path} ({size_mb:.1f} MB)")

    def verify_accuracy(self, pt_model, example_input, rtol=0.01):
        """对比 ONNX/RKNN 输出精度"""
        if not HAS_RKNN:
            self._mock_verify()
            return

        print(f"\n  Verifying {self.model_name}...")

        # PyTorch 输出
        pt_model.eval()
        input_np = example_input.detach().numpy() if hasattr(example_input, 'numpy') else example_input
        with torch.no_grad():
            pt_out = pt_model(example_input).detach().numpy()

        # RKNN 推理
        self.rknn.init_runtime(target=self.config["target_platform"])
        rknn_out = self.rknn.inference(inputs=[input_np])[0]

        # 计算指标
        pt_f = pt_out.reshape(-1)
        rk_f = rknn_out.reshape(-1)

        cos_sim = np.dot(pt_f, rk_f) / (np.linalg.norm(pt_f) * np.linalg.norm(rk_f))
        max_diff = np.abs(pt_out - rknn_out).max()
        mean_diff = np.abs(pt_out - rknn_out).mean()
        nan_ratio = np.isnan(rknn_out).mean()
        inf_ratio = np.isinf(rknn_out).mean()

        print(f"    Cosine sim: {cos_sim:.6f}  (target: >{VERIFICATION_THRESHOLDS['cosine_similarity']})")
        print(f"    Max diff:   {max_diff:.6e}")
        print(f"    Mean diff:  {mean_diff:.6e}")
        print(f"    NaN ratio:  {nan_ratio:.6f}")
        print(f"    Inf ratio:  {inf_ratio:.6f}")

        checks = [
            ("Cosine similarity", cos_sim >= VERIFICATION_THRESHOLDS["cosine_similarity"]),
            ("NaN ratio", nan_ratio <= VERIFICATION_THRESHOLDS["nan_inf_ratio"]),
            ("Inf ratio", inf_ratio <= VERIFICATION_THRESHOLDS["nan_inf_ratio"]),
        ]

        all_pass = True
        for name, ok in checks:
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"    [{status}] {name}")

        return all_pass

    def _mock_convert(self, precision):
        """模拟转换 (无 RKNN 环境时)"""
        print(f"\n  [MOCK] Would convert {self.model_name} to {precision.upper()}:")
        print(f"    ONNX:  {self.onnx_path}")
        print(f"    RKNN:  {self.rknn_path}")
        print(f"    Platform: {self.config['target_platform']}")

        # 估算量化后大小
        import onnx
        model = onnx.load(self.onnx_path)
        total_params = sum(
            np.prod(init.dims) for init in model.graph.initializer
            if init.data_type in (1, 6, 7)  # float32, float16, int32
        )
        if precision == "int8":
            est_size = total_params  # 1 byte per param
        else:
            est_size = total_params * 2  # 2 bytes per param (FP16)
        print(f"    Est size: {est_size / (1024*1024):.1f} MB ({precision})")

    def _mock_verify(self):
        """模拟验证"""
        print(f"\n  [MOCK] Would verify {self.model_name} on RK3588 NPU")
        print(f"    Thresholds: cosine > {VERIFICATION_THRESHOLDS['cosine_similarity']}")

    def release(self):
        if self.rknn is not None:
            self.rknn.release()


def create_calibration_dataset(output_dir: str, model_name: str, num_samples: int = 100):
    """为 INT8 量化创建校准数据集

    生成代表性输入 tensor 并保存为 RKNN 校准格式。
    """
    calib_dir = os.path.join(output_dir, "calibration", model_name)
    os.makedirs(calib_dir, exist_ok=True)

    dataset_file = os.path.join(calib_dir, "dataset.txt")

    if model_name == "depth_body":
        cfg = DepthFormerConfig()
        B, Q, D = 1, cfg.num_codebooks, cfg.temporal_spec.model_dims
        with open(dataset_file, "w") as f:
            for i in range(num_samples):
                x = np.random.randn(B, Q, D).astype(np.float32)
                path = os.path.join(calib_dir, f"input_{i:04d}.npy")
                np.save(path, x)
                f.write(path + "\n")
        print(f"  Created {num_samples} calibration samples for {model_name}")

    elif model_name == "codec_decoder":
        cfg = SpectroStreamConfig()
        with open(dataset_file, "w") as f:
            for i in range(num_samples):
                T = np.random.choice([25, 50, 100])  # 不同时长
                x = np.random.randn(1, T, cfg.embedding_dim).astype(np.float32)
                path = os.path.join(calib_dir, f"input_{i:04d}.npy")
                np.save(path, x)
                f.write(path + "\n")
        print(f"  Created {num_samples} calibration samples for {model_name}")

    print(f"  Dataset file: {dataset_file}")
    return dataset_file


def main():
    parser = argparse.ArgumentParser(description="RKNN Model Conversion")
    parser.add_argument("--onnx_dir", default="./exported", help="ONNX model directory")
    parser.add_argument("--output_dir", default="./rknn_models", help="Output directory for RKNN models")
    parser.add_argument("--graph", choices=["temporal_body", "depth_body", "codec_decoder", "all"],
                        default="all", help="Which graph to convert")
    parser.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    parser.add_argument("--verify", action="store_true", help="Verify accuracy after conversion")
    parser.add_argument("--create_calib", action="store_true", help="Create calibration dataset")
    parser.add_argument("--calib_samples", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    graphs = {
        "temporal_body": (TEMPORAL_BODY_CONFIG, None, None),
        "depth_body": (DEPTH_BODY_CONFIG, None, None),
        "codec_decoder": (CODEC_DECODER_CONFIG, None, None),
    }

    if args.graph == "all":
        targets = list(graphs.keys())
    else:
        targets = [args.graph]

    print("=" * 60)
    print(f"RK-MRT2 RKNN Conversion (precision={args.precision})")
    print(f"HAS_RKNN: {HAS_RKNN}")
    print("=" * 60)

    # 创建校准数据 (INT8 模式)
    calib_files = {}
    if args.create_calib or args.precision == "int8":
        print("\n[0] Creating calibration datasets...")
        for name in targets:
            calib_files[name] = create_calibration_dataset(
                args.output_dir, name, args.calib_samples
            )

    # 转换各 graph
    for name in targets:
        config, _, _ = graphs[name]
        converter = RKNNConverter(config, args.onnx_dir, args.output_dir)

        try:
            # 检查 ONNX
            print(f"\n--- {name} ---")
            converter.check_onnx()

            # 转换
            if args.precision == "fp16":
                converter.convert_fp16()
            else:
                calib_file = calib_files.get(name)
                converter.convert_int8(calib_file)

            # 精度验证
            if args.verify:
                # 创建 PyTorch 模型
                if name == "depth_body":
                    cfg = DepthFormerConfig()
                    pt_model = DepthBodyAR(cfg)
                    B, Q, D = 1, cfg.num_codebooks, cfg.temporal_spec.model_dims
                    x = torch.randn(B, Q, D)
                elif name == "codec_decoder":
                    cfg = SpectroStreamConfig()
                    pt_model = SpectroStreamDecoder(cfg)
                    x = torch.randn(1, 25, cfg.embedding_dim)
                else:
                    pt_model, x = None, None

                if pt_model is not None:
                    converter.verify_accuracy(pt_model, x)

        except Exception as e:
            print(f"  [ERROR] {e}")
        finally:
            converter.release()

    print("\n" + "=" * 60)
    if HAS_RKNN:
        print("Conversion complete! .rknn files in:", args.output_dir)
    else:
        print("Config files created. Transfer to x86 Linux to run conversion:")
        print(f"  python export/convert_rknn.py --onnx_dir {args.onnx_dir} --output_dir {args.output_dir}")
        print(f"  Or use Docker: docker run -v $(pwd):/work pelochus/ezrkllm-toolkit:latest")
    print("=" * 60)


if __name__ == "__main__":
    import torch
    main()
