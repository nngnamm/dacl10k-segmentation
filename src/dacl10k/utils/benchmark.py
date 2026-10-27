"""Latency and throughput benchmarking utilities for PyTorch vs. ONNX Runtime."""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


def benchmark_pytorch_vs_onnx(
    model: nn.Module,
    onnx_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    warmup: int = 3,
    runs: int = 10,
) -> dict[str, float]:
    """Benchmark inference latency and throughput between PyTorch CPU and ONNX Runtime CPU.

    Args:
        model: PyTorch segmentation model.
        onnx_path: Path to the exported ONNX model file.
        input_shape: Input tensor shape `(B, C, H, W)`.
        warmup: Number of untimed warmup iterations.
        runs: Number of timed benchmark iterations.

    Returns:
        Dictionary containing latency (ms), FPS, and speedup metrics.
    """
    model = model.cpu()
    model.eval()

    dummy_torch = torch.randn(*input_shape, dtype=torch.float32, device="cpu")
    dummy_np = dummy_torch.numpy()
    batch_size = input_shape[0]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_torch)

        pytorch_times_ms: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(dummy_torch)
            t1 = time.perf_counter()
            pytorch_times_ms.append((t1 - t0) * 1000.0)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for _ in range(warmup):
        _ = session.run(["logits"], {"input": dummy_np})

    onnx_times_ms: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run(["logits"], {"input": dummy_np})
        t1 = time.perf_counter()
        onnx_times_ms.append((t1 - t0) * 1000.0)

    pt_arr = np.array(pytorch_times_ms, dtype=np.float64)
    ort_arr = np.array(onnx_times_ms, dtype=np.float64)

    pytorch_mean_ms = float(np.mean(pt_arr))
    pytorch_p95_ms = float(np.percentile(pt_arr, 95))
    pytorch_fps = float((batch_size * 1000.0) / max(pytorch_mean_ms, 1e-9))

    onnx_mean_ms = float(np.mean(ort_arr))
    onnx_p95_ms = float(np.percentile(ort_arr, 95))
    onnx_fps = float((batch_size * 1000.0) / max(onnx_mean_ms, 1e-9))

    speedup_ratio = float(pytorch_mean_ms / max(onnx_mean_ms, 1e-9))

    return {
        "pytorch_mean_ms": pytorch_mean_ms,
        "pytorch_p95_ms": pytorch_p95_ms,
        "pytorch_fps": pytorch_fps,
        "onnx_mean_ms": onnx_mean_ms,
        "onnx_p95_ms": onnx_p95_ms,
        "onnx_fps": onnx_fps,
        "speedup_ratio": speedup_ratio,
    }
