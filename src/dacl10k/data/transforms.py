
"""Preprocessing and augmentation for DACL10K semantic segmentation."""

import random

import numpy as np
import torch
from PIL import Image, ImageEnhance


class DACL10kTransforms:
    """Transform RGB images and multi-label segmentation masks."""

    def __init__(
        self,
        split: str,
        image_size: tuple[int, int] = (512, 512),
    ) -> None:
        self.split = split.lower()
        self.height, self.width = image_size

        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                "image_size must contain positive dimensions."
            )

        # ImageNet statistics for the pretrained EfficientNet encoder.
        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        ).reshape(1, 1, 3)

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        ).reshape(1, 1, 3)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return an RGB tensor [3,H,W] and mask tensor [C,H,W].

        The raw Hugging Face DACL10K images may have arbitrary spatial
        dimensions. Their polygon masks are rasterized directly at the
        requested target size by DACL10kDataset.
        """

        image = np.asarray(image, dtype=np.uint8)
        mask = np.asarray(mask, dtype=np.uint8)

        # Validate the RGB image.
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Expected RGB image [H,W,3], got {image.shape}."
            )

        # Convert a single-channel mask to HWC format.
        if mask.ndim == 2:
            mask = mask[..., None]

        if mask.ndim != 3 or mask.shape[2] == 0:
            raise ValueError(
                f"Expected mask [H,W,C] with C > 0, got {mask.shape}."
            )

        target_shape = (self.height, self.width)

        # Two valid cases:
        # 1. Image and mask have matching source dimensions.
        # 2. The mask has already been rasterized at the target dimensions.
        # The second case is used by the raw Hugging Face polygon dataset.
        if (
            image.shape[:2] != mask.shape[:2]
            and mask.shape[:2] != target_shape
        ):
            raise ValueError(
                "Image and mask dimensions do not match, and the mask "
                f"is not at target size {target_shape}. "
                f"Image={image.shape[:2]}, mask={mask.shape[:2]}."
            )

        image_pil = Image.fromarray(image).convert("RGB")

        # Augmentation is restricted to training data.
        if self.split == "train":
            # Apply the same geometric flip to image and mask.
            if random.random() < 0.5:
                image_pil = image_pil.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
                mask = np.flip(mask, axis=1).copy()

            # Brightness and contrast affect the image only.
            if random.random() < 0.3:
                image_pil = ImageEnhance.Brightness(
                    image_pil
                ).enhance(random.uniform(0.85, 1.15))

                image_pil = ImageEnhance.Contrast(
                    image_pil
                ).enhance(random.uniform(0.85, 1.15))

        # PIL uses (width, height); NumPy uses (height, width).
        # Skip unnecessary image resizing when dimensions already match.
        if image_pil.size != (self.width, self.height):
            image_pil = image_pil.resize(
                (self.width, self.height),
                resample=Image.Resampling.BILINEAR,
            )

        # Resize masks only when they are not already at target dimensions.
        # Nearest-neighbor interpolation preserves discrete binary labels.
        if mask.shape[:2] != target_shape:
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

        # Normalize image values using ImageNet statistics.
        image_array = np.asarray(
            image_pil,
            dtype=np.float32,
        ) / 255.0

        image_array = (image_array - self.mean) / self.std

        # Convert image from HWC to CHW.
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                image_array.transpose(2, 0, 1)
            )
        ).float()

        # Convert mask from HWC to CHW without normalizing its labels.
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(
                mask.transpose(2, 0, 1)
            )
        ).float()

        return image_tensor, mask_tensor


def get_transforms(
    split: str,
    image_size: tuple[int, int] = (512, 512),
) -> DACL10kTransforms:
    """Create transforms for the train or validation split."""

    normalized_split = split.lower()

    if normalized_split in {"val", "valid", "validation"}:
        normalized_split = "val"
    elif normalized_split != "train":
        raise ValueError(
            f"Unsupported split {split!r}. Use 'train' or 'val'."
        )

    return DACL10kTransforms(
        split=normalized_split,
        image_size=image_size,
    )
