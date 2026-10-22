
"""PyTorch dataset for DACL10K multi-label semantic segmentation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class DACL10kDataset(Dataset):
    """Load DACL10K images and convert polygon JSON files into masks."""

    CLASS_NAMES = CLASS_NAMES
    NUM_CLASSES = len(CLASS_NAMES)

    def __init__(
        self,
        root_dir: str | Path,
        split: str,
        transforms: Any = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = self._normalize_split(split)
        self.transforms = transforms

        self.image_dir = self._find_split_dir(
            self.root_dir / "images", self.split
        )
        self.annotation_dir = self._find_split_dir(
            self.root_dir / "annotations", self.split
        )

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}"
            )

        if not self.annotation_dir.is_dir():
            raise FileNotFoundError(
                f"Annotation directory not found: {self.annotation_dir}"
            )

        image_index = {
            path.name.lower(): path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        image_stems = {
            path.stem.lower(): path
            for path in self.image_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        }

        annotation_files = sorted(self.annotation_dir.glob("*.json"))

        if not annotation_files:
            raise FileNotFoundError(
                f"No JSON annotations found in {self.annotation_dir}"
            )

        self.samples: list[tuple[Path, Path]] = []
        missing_images: list[str] = []

        for annotation_path in annotation_files:
            # imageName in the annotation is the preferred image identifier.
            with annotation_path.open("r", encoding="utf-8") as file:
                annotation = json.load(file)

            image_name = Path(
                annotation.get(
                    "imageName",
                    annotation_path.stem + ".jpg",
                )
            ).name

            image_path = image_index.get(image_name.lower())

            if image_path is None:
                image_path = image_stems.get(
                    Path(image_name).stem.lower()
                )

            if image_path is None:
                image_path = image_stems.get(
                    annotation_path.stem.lower()
                )

            if image_path is None:
                missing_images.append(image_name)
                continue

            self.samples.append((image_path, annotation_path))

        if missing_images:
            examples = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"{len(missing_images)} annotation files have no matching "
                f"image in {self.image_dir}. Examples: {examples}"
            )

        if not self.samples:
            raise RuntimeError(
                f"No matched image/annotation pairs found for split {split!r}."
            )

        if self.transforms is None:
            self.transforms = get_transforms(
                self.split,
                image_size=(512, 512),
            )

        print(
            f"DACL10K {self.split}: loaded "
            f"{len(self.samples)} image/annotation pairs."
        )

    @staticmethod
    def _normalize_split(split: str) -> str:
        split = split.lower()

        if split == "train":
            return "train"

        if split in {"val", "valid", "validation"}:
            return "validation"

        raise ValueError(
            f"Unsupported split {split!r}. Use 'train' or 'val'."
        )

    @staticmethod
    def _find_split_dir(parent: Path, split: str) -> Path:
        """Support both 'val' and 'validation' folder names."""
        candidates = (
            [parent / "train"]
            if split == "train"
            else [parent / "validation", parent / "val", parent / "valid"]
        )

        for path in candidates:
            if path.is_dir():
                return path

        # Return the standard expected path to make error messages clear.
        return candidates[0]

    @classmethod
    def _make_mask(
        cls,
        annotation: dict[str, Any],
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Rasterize polygons into an HWC binary mask with 19 channels."""

        annotation_width = int(
            annotation.get("imageWidth", image_width)
        )
        annotation_height = int(
            annotation.get("imageHeight", image_height)
        )

        if annotation_width <= 0 or annotation_height <= 0:
            raise ValueError(
                "Annotation imageWidth and imageHeight must be positive."
            )

        if annotation_width != image_width or annotation_height != image_height:
            scale_x = image_width / annotation_width
            scale_y = image_height / annotation_height
        else:
            scale_x = scale_y = 1.0

        class_to_index = {
            name: idx for idx, name in enumerate(cls.CLASS_NAMES)
        }

        # One independent binary mask per class; classes may overlap.
        channel_images = [
            Image.new("L", (image_width, image_height), 0)
            for _ in cls.CLASS_NAMES
        ]

        for shape in annotation.get("shapes", []):
            label = shape.get("label")

            if label not in class_to_index:
                continue

            shape_type = shape.get("shape_type")
            if shape_type not in (None, "polygon", "polygone"):
                continue

            points = shape.get("points", [])
            if len(points) < 3:
                continue

            scaled_points = [
                (
                    round(float(point[0]) * scale_x),
                    round(float(point[1]) * scale_y),
                )
                for point in points
                if len(point) >= 2
            ]

            if len(scaled_points) < 3:
                continue

            channel_idx = class_to_index[label]
            draw = ImageDraw.Draw(channel_images[channel_idx])
            draw.polygon(scaled_points, fill=1, outline=1)

        channels = [
            np.asarray(channel, dtype=np.uint8)
            for channel in channel_images
        ]

        return np.stack(channels, axis=-1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, annotation_path = self.samples[index]

        with Image.open(image_path) as image_file:
            image = np.asarray(
                image_file.convert("RGB"),
                dtype=np.uint8,
            ).copy()

        with annotation_path.open("r", encoding="utf-8") as file:
            annotation = json.load(file)

        height, width = image.shape[:2]

        mask = self._make_mask(
            annotation,
            image_width=width,
            image_height=height,
        )

        image_tensor, mask_tensor = self.transforms(image, mask)

        expected_image_shape = (3, 512, 512)
        expected_mask_shape = (self.NUM_CLASSES, 512, 512)

        if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
            raise RuntimeError(
                f"Invalid image tensor shape: {tuple(image_tensor.shape)}"
            )

        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != self.NUM_CLASSES:
            raise RuntimeError(
                f"Invalid mask tensor shape: {tuple(mask_tensor.shape)}"
            )

        if image_tensor.shape[1:] != mask_tensor.shape[1:]:
            raise RuntimeError(
                "Transformed image and mask have inconsistent spatial sizes."
            )

        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
            raise RuntimeError("Segmentation mask must be binary.")

        return image_tensor, mask_tensor
