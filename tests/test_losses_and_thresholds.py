"""Unit tests for multi-label loss functions and per-class threshold tuning."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from dacl10k.losses import AsymmetricLoss, MultiLabelDiceLoss, build_loss
from dacl10k.utils import optimize_per_class_thresholds


def test_loss_forward_passes() -> None:
    """Verify BCEWithLogitsLoss, MultiLabelDiceLoss, AsymmetricLoss, and factory losses."""
    torch.manual_seed(42)
    logits = torch.randn(2, 19, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 19, 64, 64)).float()

    loss_modules = [
        nn.BCEWithLogitsLoss(),
        MultiLabelDiceLoss(),
        AsymmetricLoss(),
        build_loss({"loss": "bce"}),
        build_loss({"loss": "bce_dice"}),
        build_loss({"loss": "asymmetric"}),
        build_loss({"loss": "asl"}),
    ]

    for criterion in loss_modules:
        loss = criterion(logits, targets)
        assert loss.ndim == 0, f"Expected scalar loss for {criterion}, got shape {loss.shape}"
        assert not torch.isnan(loss), f"NaN loss encountered for {criterion}"
        assert not torch.isinf(loss), f"Inf loss encountered for {criterion}"
        assert loss.item() >= 0.0

    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss({"loss": "invalid_loss"})


def test_threshold_optimizer() -> None:
    """Verify optimize_per_class_thresholds returns 19 valid optimal thresholds."""
    rng = np.random.default_rng(42)
    all_logits = rng.standard_normal((4, 19, 16, 16)).astype(np.float32)
    all_targets = rng.integers(0, 2, size=(4, 19, 16, 16)).astype(np.float32)

    candidate_thresholds = np.linspace(0.1, 0.9, 9)
    optimal = optimize_per_class_thresholds(
        all_logits=all_logits,
        all_targets=all_targets,
        thresholds=candidate_thresholds,
    )

    assert isinstance(optimal, np.ndarray)
    assert optimal.shape == (19,)
    assert np.issubdtype(optimal.dtype, np.floating)
    assert np.all((optimal >= 0.1) & (optimal <= 0.9))
