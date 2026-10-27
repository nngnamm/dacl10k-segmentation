"""CLI script to export dacl10k models to ONNX, benchmark latency/FPS, and run error analysis."""

import argparse
from pathlib import Path

import torch
import yaml

from dacl10k.engine.export import export_to_onnx
from dacl10k.models.factory import build_model
from dacl10k.utils.benchmark import benchmark_pytorch_vs_onnx
from dacl10k.utils.error_analysis import analyze_failure_modes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export dacl10k model to ONNX, benchmark latency, and analyze failure modes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to a trained PyTorch checkpoint file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/model.onnx"),
        help="Output path for the exported ONNX file.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("HEIGHT", "WIDTH"),
        help="Input spatial resolution (height width) for export and benchmarking.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute ONNX export, latency benchmarking, and failure mode analysis."""
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.checkpoint is None:
        cfg["encoder_weights"] = None

    model = build_model(cfg)
    if args.checkpoint is not None and args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)

    height, width = int(args.image_size[0]), int(args.image_size[1])
    input_shape = (1, 3, height, width)

    export_stats = export_to_onnx(
        model=model,
        output_path=args.output,
        input_shape=input_shape,
        verify=True,
    )
    bench_stats = benchmark_pytorch_vs_onnx(
        model=model,
        onnx_path=args.output,
        input_shape=input_shape,
    )

    print("\n### ONNX Export & Latency Benchmark\n")
    print(f"- **Exported Model**: `{args.output}` ({export_stats['file_size_mb']:.2f} MB)")
    print(f"- **Max Absolute Parity Diff**: `{export_stats['max_abs_diff']:.6e}`")
    print(f"- **Speedup Ratio**: `{bench_stats['speedup_ratio']:.2f}x`\n")
    print("| Runtime | Mean Latency (ms) | P95 Latency (ms) | Throughput (FPS) |")
    print("| :--- | :---: | :---: | :---: |")
    print(
        f"| PyTorch (CPU) | {bench_stats['pytorch_mean_ms']:.2f} | "
        f"{bench_stats['pytorch_p95_ms']:.2f} | {bench_stats['pytorch_fps']:.2f} |"
    )
    print(
        f"| ONNX Runtime (CPU) | {bench_stats['onnx_mean_ms']:.2f} | "
        f"{bench_stats['onnx_p95_ms']:.2f} | {bench_stats['onnx_fps']:.2f} |"
    )

    num_classes = int(cfg.get("num_classes", 19))
    sample_logits = torch.randn(2, num_classes, height, width)
    sample_targets = torch.randint(0, 2, (2, num_classes, height, width)).float()
    failure_reports = analyze_failure_modes(sample_logits, sample_targets)

    print("\n### Sample Failure Mode Analysis (Worst 5 Classes by IoU)\n")
    print("| Class | IoU | Precision | Recall | FNR | FPR |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for item in failure_reports[:5]:
        print(
            f"| {item['class_name']} | {float(item['iou']):.4f} | "
            f"{float(item['precision']):.4f} | {float(item['recall']):.4f} | "
            f"{float(item['fnr']):.4f} | {float(item['fpr']):.4f} |"
        )


if __name__ == "__main__":
    main()
