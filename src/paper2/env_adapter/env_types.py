from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


TerminationReason = Literal[
    "running",
    "captured",
    "timeout",
    "out_of_bounds",
    "target_out_of_bounds",
    "safety_violation",
]


@dataclass(frozen=True)
class EnvObservation:
    aircraft_pos_world: np.ndarray
    target_pos_world: np.ndarray
    truth_crop_center_world: np.ndarray
    crop_valid_flag: bool


@dataclass(frozen=True)
class EnvStepInfo:
    reason: TerminationReason
    distance_to_target: float
    mode: str
    crop_valid_flag: bool
    target_out_of_bounds: bool


@dataclass(frozen=True)
class EnvStepResult:
    observation: EnvObservation
    reward: float
    done: bool
    info: EnvStepInfo
