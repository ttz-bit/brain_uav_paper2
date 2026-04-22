from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from paper2.datasets.public_tracking_dataset import (
    PublicTrackingSample,
    load_bgr_image,
    resolve_image_path,
)
from paper2.datasets.unified_schema import validate_record


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SeaDronesSeeDataset:
    def __init__(
        self,
        root: str | Path,
        split: str,
        project_root: str | Path | None = None,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()

        manifest_path = self.root / "manifests" / f"records_{split}.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        rows = _read_jsonl(manifest_path)
        if max_samples is not None:
            rows = rows[: max(0, int(max_samples))]
        if not rows:
            raise ValueError(f"Empty dataset for split={split}: {manifest_path}")

        for row in rows:
            validate_record(row)
            if str(row.get("dataset_name", "")) != "SeaDronesSee":
                raise KeyError("dataset_name must be SeaDronesSee for SeaDronesSeeDataset")
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> PublicTrackingSample:
        row = self._rows[idx]
        crop_path = resolve_image_path(str(row["crop_path"]), self.project_root)
        image = load_bgr_image(crop_path)

        center = np.asarray(row["center_px"], dtype=np.float32).reshape(2)
        bbox = np.asarray(row["bbox_xywh"], dtype=np.float32).reshape(4)
        visible = bool(int(row.get("visible", 1)))
        occluded = bool(int(row.get("occluded", 0)))
        truncated = bool(int(row.get("truncated", 0)))
        valid = visible and (not occluded) and (not truncated)

        return PublicTrackingSample(
            image=image,
            target_center=center,
            bbox_xywh=bbox,
            valid=valid,
            sequence_id=str(row["sequence_id"]),
            frame_id=str(row["frame_id"]),
            meta=dict(row.get("meta", {})),
        )


def build_seadronessee_dataset(
    root: str | Path,
    split: str,
    project_root: str | Path | None = None,
    max_samples: int | None = None,
) -> SeaDronesSeeDataset:
    return SeaDronesSeeDataset(
        root=root,
        split=split,
        project_root=project_root,
        max_samples=max_samples,
    )

