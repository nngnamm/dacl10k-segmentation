"""PyTorch Dataset implementation for multi-label dacl10k semantic segmentation."""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dacl10k.data.rasterizer import rasterize_annotations


class DACL10kDataset(Dataset):
    """Multi-label semantic segmentation dataset for dacl10k."""

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        transforms: A.Compose | None = None,
        image_paths: list[str | Path] | None = None,
        annotation_paths: list[str | Path] | None = None,
    ) -> None:
        """Initialize the DACL10kDataset.

        Args:
            root_dir: Root directory containing `images/{split}` and `annotations/{split}`.
            split: Dataset split ("train", "val", or "test").
            transforms: Optional Albumentations Compose pipeline.
            image_paths: Optional explicit list of image paths.
            annotation_paths: Optional explicit list of annotation JSON paths.
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms

        if image_paths is not None and annotation_paths is not None:
            if len(image_paths) != len(annotation_paths):
                raise ValueError("image_paths and annotation_paths must have the same length.")
            self.samples: list[tuple[Path, Path]] = [
                (Path(img_p), Path(ann_p))
                for img_p, ann_p in zip(image_paths, annotation_paths, strict=True)
            ]
        else:
            img_dir = self.root_dir / "images" / self.split
            ann_dir = self.root_dir / "annotations" / self.split
            self.samples = []

            if img_dir.exists():
                valid_exts = {".jpg", ".jpeg", ".png"}
                img_files = sorted(
                    p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts
                )
                for img_path in img_files:
                    ann_path = ann_dir / f"{img_path.stem}.json"
                    if not ann_path.exists():
                        alt_ann_path = ann_dir / f"{img_path.name}.json"
                        if alt_ann_path.exists():
                            ann_path = alt_ann_path
                    if ann_path.exists():
                        self.samples.append((img_path, ann_path))

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Fetch and preprocess a single image and its multi-label mask.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys "image" ([3, H, W] float32), "mask" ([19, H, W] float32),
            and "id" (str image stem).
        """
        img_path, ann_path = self.samples[idx]

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]
        mask = rasterize_annotations(ann_path, height=height, width=width)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image_out = transformed["image"]
            mask_out = transformed["mask"]
        else:
            image_out = torch.from_numpy(np.transpose(image, (2, 0, 1))).to(torch.float32) / 255.0
            mask_out = torch.from_numpy(np.transpose(mask, (2, 0, 1))).to(torch.float32)

        if isinstance(image_out, np.ndarray):
            image_tensor = torch.from_numpy(np.transpose(image_out, (2, 0, 1))).to(torch.float32)
        else:
            image_tensor = image_out.to(torch.float32)

        if isinstance(mask_out, np.ndarray):
            if mask_out.ndim == 3 and mask_out.shape[-1] == 19 and mask_out.shape[0] != 19:
                mask_tensor = torch.from_numpy(np.transpose(mask_out, (2, 0, 1))).to(torch.float32)
            else:
                mask_tensor = torch.from_numpy(mask_out).to(torch.float32)
        else:
            if mask_out.ndim == 3 and mask_out.shape[-1] == 19 and mask_out.shape[0] != 19:
                mask_tensor = mask_out.permute(2, 0, 1).contiguous().to(torch.float32)
            else:
                mask_tensor = mask_out.to(torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "id": img_path.stem,
        }
