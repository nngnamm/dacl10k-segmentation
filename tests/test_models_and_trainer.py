"""Unit tests for model factory, multi-label IoU metric, and Accelerate Trainer."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dacl10k.engine import Trainer
from dacl10k.models import build_model
from dacl10k.utils import MultiLabelIoUMetric


def test_model_factory_shapes() -> None:
    """Verify build_model outputs [B, 19, H, W] for unetplusplus and segformer."""
    dummy_input = torch.randn(2, 3, 128, 128)

    unetpp_cfg = {
        "arch": "unetplusplus",
        "encoder_name": "resnet18",
        "num_classes": 19,
        "encoder_weights": None,
    }
    unetpp_model = build_model(unetpp_cfg).eval()
    with torch.no_grad():
        out_unetpp = unetpp_model(dummy_input)
    assert out_unetpp.shape == (2, 19, 128, 128)

    segformer_cfg = {
        "arch": "segformer",
        "encoder_name": "mit_b0",
        "num_classes": 19,
        "encoder_weights": None,
    }
    segformer_model = build_model(segformer_cfg).eval()
    with torch.no_grad():
        out_segformer = segformer_model(dummy_input)
    assert out_segformer.shape == (2, 19, 128, 128)

    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model({"arch": "invalid_arch_name", "encoder_weights": None})


def test_multilabel_iou_metric() -> None:
    """Verify MultiLabelIoUMetric returns 1.0 for exact match and < 0.01 for inverted."""
    metric = MultiLabelIoUMetric(num_classes=19, threshold=0.5)

    # Create target tensor where every class has positive pixels
    targets = torch.zeros(2, 19, 16, 16, dtype=torch.float32)
    targets[:, :, :8, :8] = 1.0

    # High positive logits where targets == 1, high negative logits where targets == 0
    perfect_logits = torch.where(targets == 1.0, torch.tensor(10.0), torch.tensor(-10.0))
    metric.update(perfect_logits, targets)
    res_perfect = metric.compute()
    assert pytest.approx(res_perfect["miou"], abs=1e-5) == 1.0
    assert len(res_perfect["per_class_iou"]) == 19

    # Completely inverted predictions
    metric.reset()
    inverted_logits = -perfect_logits
    metric.update(inverted_logits, targets)
    res_inverted = metric.compute()
    assert res_inverted["miou"] < 0.01


def test_trainer_checkpoint_save_and_resume(tmp_path: Path) -> None:
    """Verify Trainer runs 1 epoch, saves last.pt and best.pt, and restores state cleanly."""
    torch.manual_seed(42)
    cfg = {
        "arch": "unet",
        "encoder_name": "resnet18",
        "encoder_weights": None,
        "num_classes": 19,
        "epochs": 1,
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "use_wandb": False,
        "checkpoint_dir": str(tmp_path),
        "mixed_precision": "no",
    }

    images = torch.randn(4, 3, 64, 64)
    masks = torch.randint(0, 2, (4, 19, 64, 64)).to(torch.float32)
    dataset = TensorDataset(images, masks)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = build_model(cfg)
    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    metrics = trainer.fit()

    last_ckpt = tmp_path / "last.pt"
    best_ckpt = tmp_path / "best.pt"
    assert last_ckpt.exists()
    assert best_ckpt.exists()
    assert "miou" in metrics

    # Create a fresh model and trainer and verify load_checkpoint restores weights and epoch
    fresh_model = build_model(cfg)
    fresh_trainer = Trainer(
        cfg=cfg,
        model=fresh_model,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    loaded_epoch, loaded_best_miou = fresh_trainer.load_checkpoint(last_ckpt)
    assert loaded_epoch == 0
    assert pytest.approx(loaded_best_miou, abs=1e-6) == metrics["best_miou"]

    orig_state = trainer.accelerator.unwrap_model(trainer.model).state_dict()
    loaded_state = fresh_trainer.accelerator.unwrap_model(fresh_trainer.model).state_dict()
    for key in orig_state:
        assert torch.allclose(orig_state[key].cpu(), loaded_state[key].cpu())


def test_multilabel_iou_metric_ignores_empty_classes() -> None:
    """Verify classes with zero ground-truth and predicted positives
    are excluded from mean IoU.
    """
    metric = MultiLabelIoUMetric(num_classes=3, threshold=0.5)

    targets = torch.zeros(1, 3, 4, 4, dtype=torch.float32)
    targets[:, 0, :2, :2] = 1.0
    targets[:, 1, :2, :2] = 1.0

    logits = torch.full_like(targets, -10.0)

    # Correct predictions for classes 0 and 1.
    logits[:, 0, :2, :2] = 10.0
    logits[:, 1, :2, :2] = 10.0

    # Class 2 has no positive pixels in either target or prediction.
    metric.update(logits, targets)
    result = metric.compute()

    assert result["miou"] == pytest.approx(1.0, abs=1e-5)
    assert len(result["per_class_iou"]) == 3
