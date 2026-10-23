"""U-Net with an ImageNet-pretrained EfficientNet encoder (B0/B3/B4/B5)."""

import segmentation_models_pytorch as smp
import torch.nn as nn

# Short name -> timm encoder name ("tu-" = timm universal encoder in smp).
ENCODER_REGISTRY = {
    "efficientnet-b0": "tu-efficientnet_b0",
    "efficientnet-b3": "tu-efficientnet_b3",
    "efficientnet-b4": "tu-efficientnet_b4",
    "efficientnet-b5": "tu-efficientnet_b5",
}


def build_efficientnet_unet(
    name: str, num_classes: int = 19, pretrained: bool = True
) -> nn.Module:
    """Return a U-Net that outputs raw logits of shape (B, num_classes, H, W).

    No sigmoid is applied: the loss and the metric do it, which is more stable.
    """
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(ENCODER_REGISTRY)}")
    return smp.Unet(
        encoder_name=ENCODER_REGISTRY[name],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=num_classes,
        activation=None,
    )