"""OpenCV polygon rasterizer for Supervisely JSON annotations."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dacl10k.data.constants import CLASS_TO_IDX, NUM_CLASSES


def rasterize_annotations(
    json_path_or_dict: str | Path | dict[str, Any], height: int, width: int
) -> np.ndarray:
    """Rasterize Supervisely polygon annotations into a multi-label binary mask.

    Args:
        json_path_or_dict: Path to the Supervisely JSON file or a parsed dictionary.
        height: Target mask height.
        width: Target mask width.

    Returns:
        Multi-hot binary mask of shape (height, width, NUM_CLASSES) and dtype uint8.
    """
    if isinstance(json_path_or_dict, (str, Path)):
        with open(json_path_or_dict, encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(json_path_or_dict, dict):
        data = json_path_or_dict
    else:
        raise TypeError(f"Expected str, Path, or dict, got {type(json_path_or_dict)}")

    mask = np.zeros((height, width, NUM_CLASSES), dtype=np.uint8)

    for obj in data.get("objects", []):
        class_title = obj.get("classTitle")
        if class_title not in CLASS_TO_IDX:
            continue

        class_idx = CLASS_TO_IDX[class_title]
        points = obj.get("points", {})
        exterior = points.get("exterior", [])
        if not exterior or len(exterior) < 3:
            continue

        channel = np.ascontiguousarray(mask[:, :, class_idx])
        pts_ext = np.array(exterior, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(channel, [pts_ext], color=1)

        interior = points.get("interior", [])
        for hole in interior:
            if len(hole) >= 3:
                pts_int = np.array(hole, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(channel, [pts_int], color=0)

        mask[:, :, class_idx] = channel

    return mask
