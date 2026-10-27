"""Loss factory for multi-label semantic segmentation."""

from typing import Any

import torch
import torch.nn as nn

from dacl10k.losses.asymmetric_loss import AsymmetricLoss
from dacl10k.losses.dice_loss import MultiLabelDiceLoss


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy with Logits and Multi-Label Soft Dice Loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        """Initialize the combined BCE + Dice loss wrapper.

        Args:
            bce_weight: Weight applied to BCEWithLogitsLoss.
            dice_weight: Weight applied to MultiLabelDiceLoss.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiLabelDiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of BCEWithLogitsLoss and MultiLabelDiceLoss.

        Args:
            logits: Raw model logits of shape [B, C, H, W].
            targets: Binary ground truth tensor of shape [B, C, H, W].

        Returns:
            Scalar combined loss.
        """
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


def build_loss(cfg: dict[str, Any]) -> nn.Module:
    """Build a multi-label segmentation loss module from configuration.

    Args:
        cfg: Dictionary containing loss configuration (e.g., `{"loss": "bce_dice"}`).

    Returns:
        Instantiated PyTorch nn.Module loss function.

    Raises:
        ValueError: If the requested loss name is unknown.
    """
    loss_name = str(cfg.get("loss", "bce_dice")).lower()

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name == "bce_dice":
        return BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)
    if loss_name in ("asymmetric", "asl"):
        return AsymmetricLoss()

    raise ValueError(
        f"Unknown loss '{loss_name}'. Supported losses: 'bce', 'bce_dice', 'asymmetric', 'asl'."
    )
