from .unet_models import UNetArchitecture
from .eva02_model import EVA02Segmenter

def build_model(cfg: dict):
    arch = cfg["model"]["arch"]
    num_classes = cfg["data"]["num_classes"]

    if arch == "unet":
        return UNetArchitecture(
            in_channels=cfg["model"].get("in_channels", 3),
            num_classes=num_classes,
            use_attention=False
        )
    elif arch == "att_unet":
        return UNetArchitecture(
            in_channels=cfg["model"].get("in_channels", 3),
            num_classes=num_classes,
            use_attention=True
        )
    elif arch == "eva02_large":
        return EVA02Segmenter(
            backbone_name=cfg["model"]["backbone_name"],
            num_classes=num_classes,
            img_size=cfg["data"]["img_size"],
            pretrained=cfg["model"].get("pretrained", True),
            grad_checkpointing=cfg["model"].get("grad_checkpointing", True)
        )
    else:
        raise ValueError(f"Kiến trúc '{arch}' không được hỗ trợ!")