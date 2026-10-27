"""Albumentations data augmentation pipelines for dacl10k segmentation."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(split: str, image_size: tuple[int, int] = (512, 512)) -> A.Compose:
    """Create the Albumentations augmentation pipeline for a given dataset split.

    Args:
        split: Dataset split name ("train", "val", or "test").
        image_size: Target (height, width) tuple.

    Returns:
        Configured A.Compose pipeline.
    """
    height, width = image_size

    if split == "train":
        return A.Compose(
            [
                A.PadIfNeeded(min_height=height, min_width=width, border_mode=0),
                A.RandomCrop(height=height, width=width, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    if split in ("val", "test"):
        return A.Compose(
            [
                A.Resize(height=height, width=width),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )

    raise ValueError(f"Unknown split '{split}'. Expected one of: 'train', 'val', 'test'.")
