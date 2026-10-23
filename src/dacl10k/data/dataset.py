"""DACL10K dataset loaded from a Hugging Face / FiftyOne export.

Expected folder layout (DATA_ROOT):
    DATA_ROOT/samples.json   # metadata, polygons, split tags
    DATA_ROOT/data/...       # image files

Each item is (image [3,H,W] float, mask [19,H,W] float in {0,1}).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from dacl10k.data.transforms import get_transforms

CLASS_NAMES = (
    "Crack", "ACrack", "Wetspot", "Efflorescence", "Rust",
    "Rockpocket", "Hollowareas", "Cavity", "Spalling", "Graffiti",
    "Weathering", "Restformwork", "ExposedRebars", "Bearing",
    "EJoint", "Drainage", "PEquipment", "JTape", "WConccor",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLIT_ALIASES = {
    "train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
}


def _is_point(value) -> bool:
    """Return True if value looks like an (x, y) coordinate pair."""
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _find_segmentation(record: dict, label_field: str | None):
    """Return the FiftyOne `Polylines` dict of one sample record.

    If `label_field` is None, auto-detect the first field that contains a
    "polylines" key (the field name differs between exports).
    """
    if label_field is not None:
        value = record.get(label_field)
        return value if isinstance(value, dict) else None
    for value in record.values():
        if isinstance(value, dict) and "polylines" in value:
            return value
    return None


class DACL10kDataset(Dataset):
    """Read images and rasterize polygon annotations into multi-label masks."""

    NUM_CLASSES = len(CLASS_NAMES)
    CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __init__(
        self,
        root_dir,
        split,
        transforms=None,
        image_size: tuple[int, int] = (512, 512),
        label_field: str | None = None,
    ):
        """
        Args:
            root_dir: folder containing samples.json and data/.
            split: "train" or "validation" ("val" is accepted).
            transforms: optional callable (image, mask) -> tensors; needs
                `.height` and `.width` attributes.
            image_size: (height, width) used when `transforms` is None.
            label_field: name of the polyline field in samples.json;
                None means auto-detect.
        """
        self.root_dir = Path(root_dir)
        if split.lower() not in SPLIT_ALIASES:
            raise ValueError("split must be 'train' or 'validation'")
        self.split = SPLIT_ALIASES[split.lower()]

        samples_file = self.root_dir / "samples.json"
        image_root = self.root_dir / "data"
        if not samples_file.is_file():
            raise FileNotFoundError(f"Metadata not found: {samples_file}")
        if not image_root.is_dir():
            raise FileNotFoundError(f"Images not found: {image_root}")

        with samples_file.open("r", encoding="utf-8") as f:
            records = json.load(f)["samples"]

        # Determine the split from metadata tags, not from filenames.
        split_records = [
            r for r in records
            if self.split in {str(t).lower() for t in (r.get("tags") or [])}
        ]

        self.samples = []
        missing_images, missing_annotations = [], []
        image_index = None  # built lazily, only if a direct path lookup fails

        for record in split_records:
            filepath = str(record.get("filepath", "")).replace("\\", "/")
            filename = filepath.rsplit("/", 1)[-1]

            # filepath is usually relative to DATA_ROOT, e.g. data/data_0/x.jpg.
            image_path = self.root_dir / filepath.lstrip("/")
            if not image_path.is_file():
                # Fallback: match by basename (handles absolute paths in metadata).
                if image_index is None:
                    image_index = {
                        p.name.lower(): p
                        for p in image_root.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
                    }
                image_path = image_index.get(filename.lower())

            if image_path is None or not image_path.is_file():
                missing_images.append(filename)
                continue

            segmentation = _find_segmentation(record, label_field)
            if segmentation is None:
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
            image_size=image_size,
        )
        self.height = self.transforms.height
        self.width = self.transforms.width

        print(f"DACL10K {self.split}: {len(self.samples)} image-mask pairs.")

    def _make_mask(self, segmentation) -> np.ndarray:
        """Rasterize normalized polygons into a binary (H, W, C) uint8 mask."""
        mask = np.zeros((self.height, self.width, self.NUM_CLASSES), dtype=np.uint8)
        layers = {}  # class index -> PIL image, so overlapping polygons are OR-ed

        for polyline in segmentation.get("polylines", []):
            label = str(polyline.get("label", "")).strip()
            class_index = self.CLASS_TO_INDEX.get(label)
            if class_index is None:
                continue

            points = polyline.get("points") or []
            if not points:
                continue

            # FiftyOne stores either one contour or a list of contours.
            contours = [points] if _is_point(points[0]) else points

            for contour in contours:
                if not isinstance(contour, list) or len(contour) < 3:
                    continue
                if not all(_is_point(p) for p in contour):
                    continue

                inside_unit_square = all(
                    -1e-6 <= float(x) <= 1.000001 and -1e-6 <= float(y) <= 1.000001
                    for x, y in contour
                )
                if not inside_unit_square:
                    raise ValueError(
                        "Expected normalized polygon coordinates in [0,1]; "
                        "found pixel coordinates."
                    )

                # Normalized [0,1] -> pixel coordinates at the target size.
                pixel_points = [
                    (
                        max(0, min(self.width - 1, round(float(x) * (self.width - 1)))),
                        max(0, min(self.height - 1, round(float(y) * (self.height - 1)))),
                    )
                    for x, y in contour
                ]

                if class_index not in layers:
                    layers[class_index] = Image.new("L", (self.width, self.height), 0)
                ImageDraw.Draw(layers[class_index]).polygon(pixel_points, fill=1)

        for class_index, layer in layers.items():
            mask[:, :, class_index] = np.asarray(layer, dtype=np.uint8)
        return mask

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        image_path, segmentation = self.samples[index]

        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8).copy()

        mask = self._make_mask(segmentation)
        image_tensor, mask_tensor = self.transforms(image, mask)

        # Cheap shape checks only; per-sample value scans slow down the loader.
        if image_tensor.shape != (3, self.height, self.width):
            raise RuntimeError(f"Bad image shape {tuple(image_tensor.shape)}: {image_path}")
        if mask_tensor.shape != (self.NUM_CLASSES, self.height, self.width):
            raise RuntimeError(f"Bad mask shape {tuple(mask_tensor.shape)}: {image_path}")

        return image_tensor, mask_tensor