import torch.nn as nn
import segmentation_models_pytorch as smp

class MultiLabelComboLoss(nn.Module):
    def __init__(self, focal_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.focal_w = focal_weight
        self.dice_w = dice_weight
        self.focal = smp.losses.FocalLoss(mode="multilabel")
        self.dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)

    def forward(self, logits, targets):
        return self.focal_w * self.focal(logits, targets.long()) + \
               self.dice_w * self.dice(logits, targets.float())