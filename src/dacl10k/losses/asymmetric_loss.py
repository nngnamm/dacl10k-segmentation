"""Asymmetric Loss for multi-label semantic segmentation."""

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (ASL) for multi-label classification and segmentation."""

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        """Initialize the AsymmetricLoss module.

        Args:
            gamma_pos: Focusing parameter for positive samples.
            gamma_neg: Focusing parameter for negative samples.
            clip: Probability margin for shifting negative sample probabilities.
            eps: Numerical stability epsilon for logarithmic operations.
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Asymmetric Loss between raw logits and binary targets.

        Args:
            logits: Raw model logits of shape [B, C, H, W].
            targets: Binary ground truth tensor of shape [B, C, H, W].

        Returns:
            Scalar mean-reduced asymmetric loss.
        """
        probs = torch.sigmoid(logits)
        p_neg = torch.clamp(probs - self.clip, min=0.0)

        pos_loss = (
            -targets
            * (1.0 - probs).pow(self.gamma_pos)
            * torch.log(torch.clamp(probs, min=self.eps))
        )
        neg_loss = (
            -(1.0 - targets)
            * p_neg.pow(self.gamma_neg)
            * torch.log(torch.clamp(1.0 - p_neg, min=self.eps))
        )

        loss = pos_loss + neg_loss
        return loss.mean()
