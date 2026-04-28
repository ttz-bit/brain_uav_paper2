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
    ):
        self.root = Path(root)
        self.split = str(split)
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.only_stage = str(only_stage) if only_stage else None

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
        )


def build_stage2_rendered_dataset(
    root: str | Path,
    split: str,
    project_root: str | Path | None = None,
    max_samples: int | None = None,
    only_stage: str | None = None,
) -> Stage2RenderedDataset:
    return Stage2RenderedDataset(
        root=root,
        split=split,
        project_root=project_root,
        max_samples=max_samples,
        only_stage=only_stage,
    )

