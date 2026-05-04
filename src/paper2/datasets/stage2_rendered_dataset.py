from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paper2.datasets.public_tracking_dataset import (
    PublicTrackingSample,
    load_bgr_image,
    resolve_image_path,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class Stage2RenderedDataset:
    def __init__(
        self,
        root: str | Path,
        split: str,
        project_root: str | Path | None = None,
        max_samples: int | None = None,
        only_stage: str | None = None,
        load_water_mask: bool = False,
    ):
        self.root = Path(root)
        self.split = str(split)
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.only_stage = str(only_stage) if only_stage else None
        self.load_water_mask = bool(load_water_mask)

        label_path = self.root / "labels" / f"{self.split}.jsonl"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing labels: {label_path}")

        rows = _read_jsonl(label_path)
        if self.only_stage:
            rows = [r for r in rows if str(r.get("stage", "")) == self.only_stage]
        if max_samples is not None:
            rows = rows[: max(0, int(max_samples))]
        if not rows:
            raise ValueError(f"Empty Stage2 dataset: root={self.root}, split={self.split}, only_stage={self.only_stage}")
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def _resolve_meta_path(self, path_raw: Any) -> Path | None:
        if path_raw is None:
            return None
        path_str = str(path_raw).replace("\\", "/")
        if not path_str:
            return None
        path = Path(path_str)
        if path.exists():
            return path
        if not path.is_absolute():
            candidate = (self.project_root / path).resolve()
            if candidate.exists():
                return candidate
        return path

    def _load_water_mask_crop(self, row: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray | None:
        meta = dict(row.get("meta", {}))
        if not self.load_water_mask:
            return None
        direct_crop_path = self._resolve_meta_path(meta.get("water_mask_crop_path"))
        if direct_crop_path is not None and direct_crop_path.exists():
            crop = cv2.imread(str(direct_crop_path), cv2.IMREAD_GRAYSCALE)
            if crop is None:
                return None
            h, w = int(image_shape[0]), int(image_shape[1])
            if crop.shape[:2] != (h, w):
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
            return crop
        crop_origin = (
            meta.get("crop_origin_bg_px")
            or meta.get("crop_origin_xy")
            or meta.get("crop_bg_xy")
            or meta.get("crop_top_left")
        )
        mask_path = self._resolve_meta_path(meta.get("water_mask_path"))
        if crop_origin is None or mask_path is None or not mask_path.exists():
            return None
        if not isinstance(crop_origin, (list, tuple)) or len(crop_origin) < 2:
            return None
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        h, w = int(image_shape[0]), int(image_shape[1])
        x1 = int(round(float(crop_origin[0])))
        y1 = int(round(float(crop_origin[1])))
        if x1 < 0 or y1 < 0 or x1 + w > mask.shape[1] or y1 + h > mask.shape[0]:
            return None
        crop = mask[y1 : y1 + h, x1 : x1 + w].copy()
        if crop.shape[:2] != (h, w):
            return None
        return crop

    def __getitem__(self, idx: int) -> PublicTrackingSample:
        row = self._rows[idx]
        image_path = resolve_image_path(str(row["image_path"]), self.project_root)
        image = load_bgr_image(image_path)

        center = np.asarray(row["target_center_px"], dtype=np.float32).reshape(2)
        bbox = np.asarray(row["bbox_xywh"], dtype=np.float32).reshape(4)
        valid = bool(row.get("obs_valid", row.get("meta", {}).get("obs_valid", True)))

        return PublicTrackingSample(
            image=image,
            target_center=center,
            bbox_xywh=bbox,
            valid=valid,
            sequence_id=str(row["sequence_id"]),
            frame_id=str(row["frame_id"]),
            meta=dict(row.get("meta", {})),
            water_mask=self._load_water_mask_crop(row, image.shape[:2]),
        )


def build_stage2_rendered_dataset(
    root: str | Path,
    split: str,
    project_root: str | Path | None = None,
    max_samples: int | None = None,
    only_stage: str | None = None,
    load_water_mask: bool = False,
) -> Stage2RenderedDataset:
    return Stage2RenderedDataset(
        root=root,
        split=split,
        project_root=project_root,
        max_samples=max_samples,
        only_stage=only_stage,
        load_water_mask=load_water_mask,
    )
