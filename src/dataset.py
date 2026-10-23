import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

DACL10K_CLASSES = [
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket",
    "Hollowareas", "Cavity", "Spalling", "Graffiti", "Weathering",
    "Restformwork", "ExposedRebars", "Bearing", "EJoint", "Drainage",
    "PEquipment", "JTape", "WConccor"
]
CLASS_TO_IDX = {c.lower(): i for i, c in enumerate(DACL10K_CLASSES)}

class DACL10kDataset(Dataset):
    def __init__(self, hf_split, transforms=None):
        self.dataset = hf_split
        self.transforms = transforms
        self.num_classes = len(DACL10K_CLASSES)

    def __len__(self):
        return len(self.dataset)

    def _rasterize(self, item, h, w):
        mask = np.zeros((h, w, self.num_classes), dtype=np.float32)
        ann = item.get("polylines", item.get("ground_truth", {}))
        poly_list = ann.get("polylines", []) if isinstance(ann, dict) else (ann if isinstance(ann, list) else [])

        for poly in poly_list:
            label = poly.get("label", "").lower().replace(" ", "")
            if label not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[label]
            for pts in poly.get("points", []):
                abs_pts = np.array([[int(x * w), int(y * h)] for x, y in pts], dtype=np.int32)
                if len(abs_pts) >= 3:
                    ch = np.ascontiguousarray(mask[:, :, cls_idx])
                    cv2.fillPoly(ch, [abs_pts], 1.0)
                    mask[:, :, cls_idx] = ch
        return mask

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = np.array(item["image"].convert("RGB"))
        h, w, _ = image.shape
        mask = self._rasterize(item, h, w)

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask

def get_transforms(img_size: int = 512, is_train: bool = True):
    if is_train:
        return A.Compose([
            A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.RandomCrop(height=img_size, width=img_size, p=1.0),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])