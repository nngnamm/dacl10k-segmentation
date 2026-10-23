
"""Unit tests for model factory, EfficientNet encoders, IoU metric, and Trainer."""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dacl10k.engine import Trainer
from dacl10k.models import build_model
from dacl10k.utils import MultiLabelIoUMetric


def test_model_factory_shapes() -> None:
    """Verify U-Net++ and SegFormer output the expected segmentation masks."""
    dummy_input = torch.randn(2, 3, 128, 128)
    expected_shape = (2, 19, 128, 128)

    configs = [
        {
            "arch": "unetplusplus",
            "encoder_name": "resnet18",
            "num_classes": 19,
            "encoder_weights": None,
        },
        {
            "arch": "segformer",
            "encoder_name": "mit_b0",
            "num_classes": 19,
            "encoder_weights": None,
        },
    ]

    for cfg in configs:
        model = build_model(cfg).eval()

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == expected_shape, (
            f"Architecture {cfg['arch']} returned {tuple(output.shape)}, "
            f"expected {expected_shape}"
        )

    # Unsupported architectures must raise a clear error.
    with pytest.raises(ValueError, match="Unsupported architecture"):
        build_model(
            {
                "arch": "invalid_arch_name",
                "encoder_weights": None,
            }
        )


@pytest.mark.parametrize(
    "encoder_name",
    [
        "efficientnet-b0",
        "efficientnet-b3",
        "efficientnet-b4",
        "efficientnet-b5",
    ],
)
def test_efficientnet_encoder_shapes(encoder_name: str) -> None:
    """Verify each EfficientNet encoder produces a 19-channel segmentation mask."""
    cfg = {
        "arch": "unet",
        "encoder_name": encoder_name,
        "num_classes": 19,
        "encoder_weights": None,
    }

    model = build_model(cfg).eval()
    dummy_input = torch.randn(1, 3, 128, 128)

    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (1, 19, 128, 128)
    assert output.shape == expected_shape, (
        f"Encoder {encoder_name} returned {tuple(output.shape)}, "
        f"expected {expected_shape}"
    )


def test_multilabel_iou_metric() -> None:
    """Verify IoU is approximately 1 for perfect predictions and low for inverted predictions."""
    metric = MultiLabelIoUMetric(num_classes=19, threshold=0.5)

    targets = torch.zeros(2, 19, 16, 16, dtype=torch.float32)
    targets[:, :, :8, :8] = 1.0

    perfect_logits = torch.where(
        targets == 1.0,
        torch.tensor(10.0),
        torch.tensor(-10.0),
    )

    metric.update(perfect_logits, targets)
    res_perfect = metric.compute()

    assert pytest.approx(res_perfect["miou"], abs=1e-5) == 1.0
    assert len(res_perfect["per_class_iou"]) == 19

    metric.reset()

    inverted_logits = -perfect_logits
    metric.update(inverted_logits, targets)
    res_inverted = metric.compute()

    assert res_inverted["miou"] < 0.01


def test_trainer_checkpoint_save_and_resume(tmp_path: Path) -> None:
    """Verify one training epoch, checkpoint saving, and checkpoint restoration."""
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
    masks = torch.randint(
        0, 2, (4, 19, 64, 64)
    ).to(torch.float32)

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

    assert last_ckpt.exists(), "Training did not save last.pt"
    assert best_ckpt.exists(), "Training did not save best.pt"
    assert "miou" in metrics
    assert "best_miou" in metrics

    # Load the checkpoint into a new model and trainer.
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
    loaded_state = (
        fresh_trainer.accelerator.unwrap_model(fresh_trainer.model).state_dict()
    )

    assert orig_state.keys() == loaded_state.keys()

    for key in orig_state:
        assert torch.allclose(
            orig_state[key].cpu(),
            loaded_state[key].cpu(),
        ), f"Checkpoint parameter mismatch: {key}"
