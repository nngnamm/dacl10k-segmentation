
"""DACL10K dataset from Hugging Face FiftyOne export."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from dacl10k.data.transforms import get_transforms


CLASS_NAMES = (
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust",
    "Rockpocket", "Hollowareas", "Cavity", "Spalling", "Graffiti",
    "Weathering", "Restformwork", "ExposedRebars", "Bearing",
    "EJoint", "Drainage", "PEquipment", "JTape", "WConccor",
)


def _is_point(value) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


class DACL10kDataset(Dataset):
    """Read Hugging Face images and generate multi-label polygon masks."""

    NUM_CLASSES = len(CLASS_NAMES)
    CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __init__(self, root_dir, split, transforms=None):
        self.root_dir = Path(root_dir)
        requested_split = split.lower()

        if requested_split in {"val", "valid", "validation"}:
            self.split = "validation"
        elif requested_split == "train":
            self.split = "train"
        else:
            raise ValueError("split must be 'train' or 'validation'")

        samples_file = self.root_dir / "samples.json"
        image_root = self.root_dir / "data"

        if not samples_file.is_file():
            raise FileNotFoundError(f"Metadata not found: {samples_file}")
        if not image_root.is_dir():
            raise FileNotFoundError(f"Images not found: {image_root}")

        with samples_file.open("r", encoding="utf-8") as f:
            records = json.load(f)["samples"]

        # Use metadata tags, not filename guesses, to determine the split.
        split_records = [
            record for record in records
            if self.split in {
                str(tag).lower() for tag in (record.get("tags") or [])
            }
        ]

        self.samples = []
        missing_images = []
        missing_annotations = []

        for record in split_records:
            filepath = record.get("filepath")
            segmentation = record.get("semseg_ground_truth_filled")

            if not filepath:
                raise ValueError("A DACL10K record has no filepath.")

            filename = str(filepath).replace("\\", "/").rsplit("/", 1)[-1]
            image_path = image_root / str(filepath).replace("\\", "/")

            # Some metadata may contain paths from a different machine.
            # Reconstruct the path using the known image directory if needed.
            if not image_path.is_file():
                matches = list(image_root.rglob(filename))
                image_path = matches[0] if matches else image_path

            if not image_path.is_file():
                missing_images.append(filename)
                continue

            if not isinstance(segmentation, dict):
                missing_annotations.append(filename)
                continue

            self.samples.append((image_path, segmentation))

        if missing_images:
            raise FileNotFoundError(
                f"{len(missing_images)} {self.split} images are missing; "
                f"examples: {missing_images[:5]}"
            )
        if missing_annotations:
            raise ValueError(
                f"{len(missing_annotations)} {self.split} samples lack "
                f"segmentation annotations; examples: {missing_annotations[:5]}"
            )
        if not self.samples:
            raise RuntimeError(f"No samples found for {self.split}")

        self.transforms = transforms or get_transforms(
            "train" if self.split == "train" else "val",
            image_size=(512, 512),
        )
        self.height = self.transforms.height
        self.width = self.transforms.width

        print(
            f"DACL10K HF {self.split}: "
            f"{len(self.samples)} image-mask pairs."
        )

    def _make_mask(self, segmentation):
        """Rasterize normalized polygon points into a binary HWC mask."""
        mask = np.zeros(
            (self.height, self.width, self.NUM_CLASSES),
            dtype=np.uint8,
        )

        layers = {}

        for polyline in segmentation.get("polylines", []):
            label = str(polyline.get("label", "")).strip()
            class_index = self.CLASS_TO_INDEX.get(label)
            if class_index is None:
                continue

            points = polyline.get("points") or []
            if not points:
                continue

            # FiftyOne may represent points as one contour or multiple contours.
            contours = [points] if _is_point(points[0]) else points

            for contour in contours:
                if not isinstance(contour, list) or len(contour) < 3:
                    continue
                if not all(_is_point(point) for point in contour):
                    continue

                normalized = all(
                    -1e-6 <= float(x) <= 1.000001
                    and -1e-6 <= float(y) <= 1.000001
                    for x, y in contour
                )

                if not normalized:
                    raise ValueError(
                        "Expected normalized polygon coordinates in [0,1]; "
                        "found pixel coordinates."
                    )

                pixel_points = [
                    (
                        max(0, min(self.width - 1, round(float(x) * (self.width - 1)))),
                        max(0, min(self.height - 1, round(float(y) * (self.height - 1)))),
                    )
                    for x, y in contour
                ]

                if class_index not in layers:
                    layers[class_index] = Image.new(
                        "L", (self.width, self.height), 0
                    )

                ImageDraw.Draw(layers[class_index]).polygon(
                    pixel_points, fill=1
                )

        for class_index, layer in layers.items():
            mask[:, :, class_index] = np.asarray(
                layer, dtype=np.uint8
            )

        return mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, segmentation = self.samples[index]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            image = np.asarray(image, dtype=np.uint8).copy()

        mask = self._make_mask(segmentation)
        image_tensor, mask_tensor = self.transforms(image, mask)

        if image_tensor.shape != (3, self.height, self.width):
            raise RuntimeError(
                f"Unexpected image shape: {tuple(image_tensor.shape)}"
            )
        if mask_tensor.shape != (19, self.height, self.width):
            raise RuntimeError(
                f"Unexpected mask shape: {tuple(mask_tensor.shape)}"
            )
        if not torch.isfinite(image_tensor).all():
            raise RuntimeError(f"Non-finite image tensor: {image_path}")
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)):
            raise RuntimeError(f"Non-binary mask: {image_path}")

        return image_tensor, mask_tensor
