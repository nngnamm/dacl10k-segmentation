"""Per-class probability threshold optimization utilities."""

import numpy as np


def optimize_per_class_thresholds(
    all_logits: np.ndarray,
    all_targets: np.ndarray,
    thresholds: np.ndarray = np.linspace(0.1, 0.9, 9),  # noqa: B008
) -> np.ndarray:
    """Find optimal per-class probability thresholds maximizing F1 score.

    Args:
        all_logits: Array of raw logits of shape [N, C, ...] (where C = 19).
        all_targets: Array of binary ground truth targets with matching shape [N, C, ...].
        thresholds: 1D numpy array of candidate probability thresholds to evaluate.

    Returns:
        1D numpy float array of shape [19] containing the optimal threshold per class.
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(all_logits, -88.0, 88.0)))
    num_classes = probs.shape[1]
    optimal_thresholds = np.full(num_classes, 0.5, dtype=np.float64)
    eps = 1e-7

    for c in range(num_classes):
        class_probs = probs[:, c].reshape(-1)
        class_targets = (all_targets[:, c].reshape(-1) >= 0.5).astype(np.float64)
        target_sum = class_targets.sum()

        best_f1 = -1.0
        best_thresh = float(thresholds[0])

        for thresh in thresholds:
            preds = (class_probs >= thresh).astype(np.float64)
            tp = np.sum(preds * class_targets)
            pred_sum = preds.sum()
            f1 = float((2.0 * tp + eps) / (pred_sum + target_sum + eps))

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)

        optimal_thresholds[c] = best_thresh

    return optimal_thresholds
