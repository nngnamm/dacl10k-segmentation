import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class EVA02Segmenter(nn.Module):
    def __init__(
        self,
        backbone_name: str = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        num_classes: int = 19,
        img_size: int = 448,
        pretrained: bool = True,
        grad_checkpointing: bool = True
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            img_size=img_size,
            out_indices=(-4, -3, -2, -1)
        )
        if grad_checkpointing:
            self.encoder.set_grad_checkpointing(True)

        in_channels_list = self.encoder.feature_info.channels()
        fpn_dim = 256

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, fpn_dim),
                nn.GELU()
            ) for ch in in_channels_list
        ])

        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(fpn_dim * len(in_channels_list), fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, fpn_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        features = self.encoder(x)
        proj_feats = [conv(f) for f, conv in zip(features, self.lateral_convs)]
        fused = torch.cat(proj_feats, dim=1)
        logits = self.fpn_bottleneck(fused)
        return F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)