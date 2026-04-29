from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class WaterConstraint(Protocol):
    def is_water_world(self, xy_world: np.ndarray) -> bool:
        ...


@dataclass(frozen=True)
class AllWaterConstraint:
    def is_water_world(self, xy_world: np.ndarray) -> bool:
        xy = np.asarray(xy_world, dtype=float).reshape(-1)
        return bool(xy.size >= 2 and np.isfinite(xy[:2]).all())


def build_water_constraint(cfg: dict | None = None) -> WaterConstraint:
    mode = "all_water" if cfg is None else str(cfg.get("mode", "all_water"))
    if mode == "all_water":
        return AllWaterConstraint()
    raise ValueError(f"Unsupported water constraint mode: {mode}")
