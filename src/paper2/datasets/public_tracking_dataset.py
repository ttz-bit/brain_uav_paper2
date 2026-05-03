from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PublicTrackingSample:
    image: np.ndarray
    target_center: np.ndarray
    bbox_xywh: np.ndarray
    valid: bool
    sequence_id: str
    frame_id: str
    meta: dict[str, Any]
    water_mask: np.ndarray | None = None


def resolve_image_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str.replace("\\", "/"))
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def load_bgr_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img
