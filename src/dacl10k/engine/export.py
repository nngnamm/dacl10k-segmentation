"""ONNX model export and parity verification utilities."""

import inspect
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    opset_version: int = 17,
    verify: bool = True,
) -> dict[str, float]:
    """Export a PyTorch segmentation model to ONNX with dynamic axes and parity check.

    Args:
        model: Instantiated PyTorch segmentation model returning raw logits.
        output_path: Destination file path for the exported `.onnx` model.
        input_shape: Dummy input tensor shape `(B, C, H, W)`.
        opset_version: ONNX opset version to target.
        verify: Whether to verify numerical parity with ONNX Runtime.

    Returns:
        Dictionary with `"max_abs_diff"` and `"file_size_mb"`.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu()
    model.eval()

    dummy_input = torch.randn(*input_shape, dtype=torch.float32, device="cpu")
    dynamic_axes = {
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "logits": {0: "batch_size", 2: "height", 3: "width"},
    }

    export_kwargs: dict[str, object] = {
        "input_names": ["input"],
        "output_names": ["logits"],
        "opset_version": opset_version,
        "dynamic_axes": dynamic_axes,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(out_file),
            **export_kwargs,
        )

    onnx.checker.check_model(str(out_file))

    max_abs_diff = 0.0
    if verify:
        with torch.no_grad():
            torch_out = model(dummy_input).detach().cpu().numpy()

        session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        ort_out = session.run(["logits"], {"input": dummy_input.numpy()})[0]

        max_abs_diff = float(np.max(np.abs(torch_out - ort_out)))
        assert max_abs_diff < 1e-3, (
            f"ONNX parity verification failed: max_abs_diff={max_abs_diff:.6f} >= 1e-3"
        )

    file_size_mb = os.path.getsize(out_file) / (1024 * 1024)
    return {
        "max_abs_diff": max_abs_diff,
        "file_size_mb": float(file_size_mb),
    }
