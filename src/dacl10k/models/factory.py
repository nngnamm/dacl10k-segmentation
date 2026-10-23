"""Config-driven model factory using segmentation_models_pytorch."""

from typing import Any

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a multi-label segmentation model from configuration.

    Args:
        cfg: Dictionary containing model architecture parameters:
            - arch: Architecture name ("segformer", "unetplusplus", "unet", etc.).
            - encoder_name: Backbone encoder name (e.g., "resnet34", "mit_b2").
            - num_classes: Number of output segmentation classes (default: 19).
            - encoder_weights: Pretrained weights name or None (default: "imagenet").

    Returns:
        Instantiated PyTorch nn.Module returning raw logits.

    Raises:
        ValueError: If the requested architecture is not supported.
    """
    arch = str(cfg.get("arch", "unetplusplus")).lower()
    encoder_name = cfg.get("encoder_name", "resnet34")
    num_classes = int(cfg.get("num_classes", 19))
    encoder_weights = cfg.get("encoder_weights", "imagenet")

    arch_map: dict[str, type[nn.Module]] = {
        "segformer": smp.Segformer,
        "unetplusplus": smp.UnetPlusPlus,
        "unet": smp.Unet,
        "deeplabv3": smp.DeepLabV3,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "manet": smp.MAnet,
    }

    if arch not in arch_map:
        supported = ", ".join(sorted(arch_map.keys()))
        raise ValueError(f"Unsupported architecture '{arch}'. Supported architectures: {supported}")

    model_cls = arch_map[arch]
    return model_cls(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
