
"""PyTorch Dataset for preprocessed DACL10K NumPy files."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from dacl10k.data.transforms import get_transforms


CLASS_NAMES = (
    "Crack",
    "ACrack",
    "Wetspot",
    "Efflorescence",
    "Rust",
    "Rockpocket",
    "Hollowareas",
    "Cavity",
    "Spalling",
    "Graffiti",
    "Weathering",
    "Restformwork",
    "ExposedRebars",
    "Bearing",
    "EJoint",
    "Drainage",
    "PEquipment",
    "JTape",
    "WConccor",
)


class DACL10kDataset(Dataset):
    """Read 512x512 RGB images and 19-channel binary masks from .npy files."""

    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transforms=None,
    ) -> None:
        self.root_dir = Path(root_dir)

        split = split.lower()
        if split == "train":
            self.split = "train"
        elif split in {"val", "valid", "validation"}:
            self.split = "validation"
        else:
            raise ValueError(
                f"Unsupported split {split!r}; use 'train' or 'val'."
            )

        self.image_dir = self.root_dir / "images" / self.split
        self.annotation_dir = self.root_dir / "annotations" / self.split

        for directory in (self.image_dir, self.annotation_dir):
            if not directory.is_dir():
                raise FileNotFoundError(
                    f"Dataset directory does not exist: {directory}"
                )

        image_index = {
            path.stem: path
            for path in self.image_dir.glob("*.npy")
        }
        annotation_index = {
            path.stem: path
            for path in self.annotation_dir.glob("*.npy")
        }

        if not image_index:
            raise FileNotFoundError(
                f"No .npy images found in {self.image_dir}"
            )
        if not annotation_index:
            raise FileNotFoundError(
                f"No .npy masks found in {self.annotation_dir}"
            )

        missing_masks = sorted(set(image_index) - set(annotation_index))
        missing_images = sorted(set(annotation_index) - set(image_index))

        if missing_masks or missing_images:
            raise ValueError(
                "Image/mask filename mismatch. "
                f"Images without masks: {missing_masks[:5]}; "
                f"masks without images: {missing_images[:5]}."
            )

        self.samples = [
            (image_index[stem], annotation_index[stem])
            for stem in sorted(image_index)
        ]

        self.transforms = transforms or get_transforms(
            "train" if self.split == "train" else "val",
            image_size=(512, 512),
        )

        print(
            f"DACL10K {self.split}: "
            f"{len(self.samples)} matched image-mask pairs."
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, mask_path = self.samples[index]

        # Both files contain numeric NumPy arrays, not pickled Python objects.
        image = np.load(image_path, allow_pickle=False)
        mask = np.load(mask_path, allow_pickle=False)

        if image.dtype != np.uint8 or image.ndim != 3:
            raise ValueError(
                f"Invalid image at {image_path}: "
                f"shape={image.shape}, dtype={image.dtype}"
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, got {image.shape}"
            )

        expected_mask_shape = (
            image.shape[0],
            image.shape[1],
            self.NUM_CLASSES,
        )
        if mask.shape != expected_mask_shape:
            raise ValueError(
                f"Invalid mask at {mask_path}: shape={mask.shape}; "
                f"expected {expected_mask_shape}"
            )

        if mask.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 mask at {mask_path}, got {mask.dtype}"
            )

        if not np.isin(mask, (0, 1)).all():
            raise ValueError(
                f"Mask contains values other than 0 and 1: {mask_path}"
            )

        # Transforms expect HWC NumPy arrays and return CHW float tensors.
        image_tensor, mask_tensor = self.transforms(image, mask)

        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise RuntimeError(
                f"Invalid transformed image shape: {tuple(image_tensor.shape)}"
            )

        if mask_tensor.ndim != 3:
            raise RuntimeError(
                f"Invalid transformed mask shape: {tuple(mask_tensor.shape)}"
            )

        if mask_tensor.shape[0] != self.NUM_CLASSES:
            raise RuntimeError(
                f"Expected {self.NUM_CLASSES} mask channels, "
                f"got {mask_tensor.shape[0]}"
            )

        if image_tensor.shape[1:] != mask_tensor.shape[1:]:
            raise RuntimeError(
                "Transformed image and mask have different spatial sizes."
            )

        return image_tensor, mask_tensor
