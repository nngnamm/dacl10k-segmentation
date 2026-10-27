"""Unit tests for ONNX export, latency benchmarking, and failure mode analysis."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from dacl10k.engine import export_to_onnx
from dacl10k.models import build_model
from dacl10k.utils import analyze_failure_modes, benchmark_pytorch_vs_onnx


def test_onnx_export_and_dynamic_shapes(tmp_path: Path) -> None:
    """Verify ONNX export parity and dynamic batch/spatial resolution inference."""
    torch.manual_seed(42)
    model = build_model(
        {
            "arch": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "num_classes": 19,
        }
    )
    onnx_path = tmp_path / "test.onnx"

    export_info = export_to_onnx(
        model=model,
        output_path=onnx_path,
        input_shape=(1, 3, 64, 64),
        opset_version=17,
        verify=True,
    )

    assert onnx_path.exists()
    assert export_info["max_abs_diff"] < 1e-3
    assert export_info["file_size_mb"] > 0.0

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dynamic_input = np.random.randn(2, 3, 96, 96).astype(np.float32)
    ort_logits = session.run(["logits"], {"input": dynamic_input})[0]
    assert ort_logits.shape == (2, 19, 96, 96)


def test_benchmark_and_error_analysis(tmp_path: Path) -> None:
    """Verify benchmark_pytorch_vs_onnx metrics and analyze_failure_modes class reports."""
    torch.manual_seed(42)
    model = build_model(
        {
            "arch": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "num_classes": 19,
        }
    )
    onnx_path = tmp_path / "bench.onnx"
    export_to_onnx(model, onnx_path, input_shape=(1, 3, 64, 64), verify=False)

    bench = benchmark_pytorch_vs_onnx(
        model=model,
        onnx_path=onnx_path,
        input_shape=(1, 3, 64, 64),
        warmup=1,
        runs=2,
    )

    expected_bench_keys = {
        "pytorch_mean_ms",
        "pytorch_p95_ms",
        "pytorch_fps",
        "onnx_mean_ms",
        "onnx_p95_ms",
        "onnx_fps",
        "speedup_ratio",
    }
    assert expected_bench_keys.issubset(bench.keys())
    for key in expected_bench_keys:
        assert bench[key] > 0.0

    logits = torch.randn(2, 19, 32, 32)
    targets = torch.randint(0, 2, (2, 19, 32, 32)).float()
    reports = analyze_failure_modes(logits, targets, thresholds=0.5)

    assert len(reports) == 19
    expected_report_keys = {
        "class_name",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "iou",
        "fnr",
        "fpr",
    }
    ious = [float(r["iou"]) for r in reports]
    assert ious == sorted(ious), "Expected failure mode reports sorted by ascending IoU"
    for r in reports:
        assert expected_report_keys.issubset(r.keys())
