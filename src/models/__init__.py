import segmentation_models_pytorch as smp
from .swin_model import SwinLargeSegmenter

def build_model(cfg: dict):
    arch = cfg["model"]["arch"]
    num_classes = cfg["data"]["num_classes"]
    pretrained = cfg["model"].get("pretrained", True)

    if arch == "swin_large":
        # Swin dùng backbone_name (hoặc lấy encoder_name nếu có)
        backbone_name = cfg["model"].get("backbone_name", cfg["model"].get("encoder_name", "tu-swin_large_patch4_window12_384"))
        return SwinLargeSegmenter(
            backbone_name=backbone_name,
            num_classes=num_classes,
            img_size=cfg["data"]["img_size"],
            pretrained=pretrained,
            grad_checkpointing=cfg["model"].get("grad_checkpointing", True)
        )
    elif arch == "segformer":
        # SegFormer dùng encoder_name
        encoder_name = cfg["model"].get("encoder_name", "mit_b5")
        return smp.Segformer(
            encoder_name=encoder_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Kiến trúc '{arch}' không được hỗ trợ trong dự án này!")