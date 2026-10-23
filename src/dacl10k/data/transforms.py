
"""Preprocessing and augmentation for DACL10K segmentation."""

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance


class DACL10kTransforms:
    """Resize images and masks and convert them to PyTorch tensors."""

    def __init__(
        self,
        split: str,
        image_size: tuple[int, int],
    ) -> None:
        self.split = split.lower()
        self.height, self.width = image_size

        if self.height <= 0 or self.width <= 0:
            raise ValueError("image_size must contain positive dimensions.")

        self.mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        ).reshape(1, 1, 3)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform an RGB image and its HWC multi-label mask."""

        image = np.asarray(image, dtype=np.uint8)
        mask = np.asarray(mask, dtype=np.uint8)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image [H, W, 3], got {image.shape}"
            )

        if mask.ndim == 2:
            mask = mask[..., None]

        if mask.ndim != 3:
            raise ValueError(
                f"Expected mask [H, W, C], got {mask.shape}"
            )

        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                "Image and mask must have the same height and width."
            )

        # DACL10K .npy images are RGB uint8 arrays.
        image_pil = Image.fromarray(image).convert("RGB")

        # Apply augmentation only to training images.
        # Geometric transformations must also be applied to the masks.
        if self.split == "train":
            if random.random() < 0.5:
                image_pil = image_pil.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
                mask = np.flip(mask, axis=1).copy()

            # Photometric transformations apply only to the image.
            if random.random() < 0.3:
                image_pil = ImageEnhance.Brightness(
                    image_pil
                ).enhance(random.uniform(0.85, 1.15))

                image_pil = ImageEnhance.Contrast(
                    image_pil
                ).enhance(random.uniform(0.85, 1.15))

        # PIL image.size is (width, height), whereas NumPy uses (height, width).
        # Skip resizing when the preprocessed image already has target dimensions.
        if image_pil.size != (self.width, self.height):
            image_pil = image_pil.resize(
                (self.width, self.height),
                resample=Image.Resampling.BILINEAR,
            )

        # DACL10K masks are HWC binary arrays.
        # Skip the expensive per-channel resize when dimensions already match.
        if mask.shape[:2] != (self.height, self.width):
            resized_channels = []

            for channel_idx in range(mask.shape[2]):
                channel = Image.fromarray(mask[:, :, channel_idx])

                channel = channel.resize(
                    (self.width, self.height),
                    resample=Image.Resampling.NEAREST,
                )

                resized_channels.append(
                    np.asarray(channel, dtype=np.uint8)
                )

            mask = np.stack(resized_channels, axis=-1)

        # Normalize RGB image using ImageNet statistics.
        image_array = np.asarray(
            image_pil, dtype=np.float32
        ) / 255.0

        image_array = (image_array - self.mean) / self.std

        # Convert image HWC -> CHW.
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image_array.transpose(2, 0, 1))
        ).float()

        # Convert mask HWC -> CHW.
        # Preserve the binary 0/1 labels without normalization.
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask.transpose(2, 0, 1))
        ).float()

        return image_tensor, mask_tensor


def get_transforms(
    split: str,
    image_size: tuple[int, int] = (512, 512),
) -> DACL10kTransforms:
    """Build preprocessing for the requested dataset split."""

    normalized_split = split.lower()

    if normalized_split in {"val", "valid", "validation"}:
        normalized_split = "val"
    elif normalized_split != "train":
        raise ValueError(
            f"Unsupported transform split: {split!r}. "
            "Use 'train' or 'val'."
        )

    return DACL10kTransforms(
        split=normalized_split,
        image_size=image_size,
    )
