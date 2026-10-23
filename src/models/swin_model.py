import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinLargeSegmenter(nn.Module):
    def __init__(
        self,
        backbone_name: str = "tu-swin_large_patch4_window12_384",
        num_classes: int = 19,
        img_size: int = 384,
        pretrained: bool = True,
        grad_checkpointing: bool = True
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            img_size=self.img_size,
            out_indices=(0, 1, 2, 3)
        )
        if grad_checkpointing and hasattr(self.encoder, "set_grad_checkpointing"):
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

    def _to_nchw(self, feat: torch.Tensor, expected_ch: int) -> torch.Tensor:
        if feat.ndim == 4 and feat.shape[1] != expected_ch and feat.shape[-1] == expected_ch:
            return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[-2:]
        if orig_size != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)

        raw_features = self.encoder(x)
        features = [
            self._to_nchw(feat, ch)
            for feat, ch in zip(raw_features, self.encoder.feature_info.channels(), strict=True)
        ]

        proj_feats = []
        target_size = features[0].shape[-2:]
        for f_feat, conv in zip(features, self.lateral_convs, strict=True):
            if f_feat.shape[-2:] != target_size:
                f_feat = F.interpolate(f_feat, size=target_size, mode="bilinear", align_corners=False)
            proj_feats.append(conv(f_feat))

        fused = torch.cat(proj_feats, dim=1)
        logits = self.fpn_bottleneck(fused)
        return F.interpolate(logits, size=orig_size, mode="bilinear", align_corners=False)