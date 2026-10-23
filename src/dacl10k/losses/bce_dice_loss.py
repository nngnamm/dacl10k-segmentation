"""BCE + Dice loss for multi-label segmentation (sigmoid, not softmax)."""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Weighted sum of BCEWithLogits and multi-label Dice."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits and targets both have shape (B, C, H, W)."""
        return (
            self.bce_weight * self.bce(logits, targets)
            + self.dice_weight * self.dice(logits, targets)
        )