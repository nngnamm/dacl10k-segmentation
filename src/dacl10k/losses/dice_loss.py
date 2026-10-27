"""Multi-label soft Dice loss for semantic segmentation."""

import torch
import torch.nn as nn


class MultiLabelDiceLoss(nn.Module):
    """Soft Dice loss computed per channel for multi-label segmentation masks."""

    def __init__(self, smooth: float = 1.0, eps: float = 1e-7) -> None:
        """Initialize the MultiLabelDiceLoss module.

        Args:
            smooth: Laplace smoothing factor added to numerator and denominator.
            eps: Numerical stability constant added to denominator.
        """
        super().__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute channel-averaged soft Dice loss.

        Args:
            logits: Raw model logits of shape [B, C, H, W].
            targets: Binary ground truth tensor of shape [B, C, H, W].

        Returns:
            Scalar soft Dice loss (1.0 - mean channel Dice score).
        """
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(0, 2, 3))
        cardinality = probs.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth + self.eps)
        return 1.0 - dice_score.mean()
