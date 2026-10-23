import json
from pathlib import Path
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
    def __init__(self, root_dir, split="train", transforms=None):
        self.root_dir = Path(root_dir)
        self.split = "train" if split == "train" else "val"
        self.transforms = transforms
        self.num_classes = len(DACL10K_CLASSES)

        # Trỏ tới thư mục images và annotations chuẩn
        self.img_dir = self.root_dir / "images" / self.split
        self.ann_dir = self.root_dir / "annotations" / self.split
        
        if not self.img_dir.exists() and self.split == "val":
            self.img_dir = self.root_dir / "images" / "validation"
            self.ann_dir = self.root_dir / "annotations" / "validation"

        self.image_paths = sorted(list(self.img_dir.glob("*.jpg")) + list(self.img_dir.glob("*.png")))

    def __len__(self):
        return len(self.image_paths)

    def _rasterize(self, ann_path, h, w):
        mask = np.zeros((h, w, self.num_classes), dtype=np.float32)
        if not ann_path.exists():
            return mask
            
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_data = json.load(f)

        for obj in ann_data.get("objects", []):
            label = obj.get("classTitle", "").lower().replace(" ", "")
            if label not in CLASS_TO_IDX:
                continue
            cls_idx = CLASS_TO_IDX[label]
            
            points = obj.get("points", {})
            exterior = points.get("exterior", [])
            if len(exterior) >= 3:
                abs_pts = np.array(exterior, dtype=np.int32).reshape((-1, 1, 2))
                ch = np.ascontiguousarray(mask[:, :, cls_idx])
                cv2.fillPoly(ch, [abs_pts], 1.0)
                mask[:, :, cls_idx] = ch
        return mask

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_bgr = cv2.imread(str(img_path))
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        ann_path = self.ann_dir / f"{img_path.stem}.json"
        mask = self._rasterize(ann_path, h, w)

        if self.transforms:
            aug = self.transforms(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"].permute(2, 0, 1).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask

def get_transforms(img_size: int = 384, is_train: bool = True):
    if is_train:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])