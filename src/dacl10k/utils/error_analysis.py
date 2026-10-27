"""Per-class failure mode and error analysis for multi-label segmentation."""

import numpy as np
import torch

from dacl10k.data.constants import DACL10K_CLASSES


def analyze_failure_modes(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: np.ndarray | float = 0.5,
) -> list[dict[str, float | str]]:
    """Compute per-class confusion metrics and sort by ascending IoU to surface failure modes.

    Args:
        logits: Raw model logits tensor of shape `[B, 19, H, W]`.
        targets: Binary ground truth tensor of shape `[B, 19, H, W]`.
        thresholds: Scalar float threshold or 1D numpy array of 19 per-class thresholds.

    Returns:
        List of 19 per-class metric dictionaries sorted by `"iou"` ascending.
    """
    probs = torch.sigmoid(logits.detach().cpu())
    targets_bin = (targets.detach().cpu() >= 0.5).to(torch.float32)

    num_classes = len(DACL10K_CLASSES)
    if isinstance(thresholds, np.ndarray):
        thresh_list = [float(t) for t in thresholds.reshape(-1)]
    else:
        thresh_list = [float(thresholds)] * num_classes

    eps = 1e-7
    reports: list[dict[str, float | str]] = []

    for c_idx, class_name in enumerate(DACL10K_CLASSES):
        thresh = thresh_list[c_idx]
        c_preds = (probs[:, c_idx] >= thresh).to(torch.float32)
        c_targs = targets_bin[:, c_idx]

        tp = float((c_preds * c_targs).sum().item())
        fp = float((c_preds * (1.0 - c_targs)).sum().item())
        fn = float(((1.0 - c_preds) * c_targs).sum().item())

        precision = float(tp / (tp + fp + eps))
        recall = float(tp / (tp + fn + eps))
        iou = float(tp / (tp + fp + fn + eps))
        fnr = float(fn / (tp + fn + eps))
        fpr = float(fp / (tp + fp + eps))

        reports.append(
            {
                "class_name": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "fnr": fnr,
                "fpr": fpr,
            }
        )

    reports.sort(key=lambda item: float(item["iou"]))
    return reports
