from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class AircraftState:
    t: float
    pos_world: np.ndarray
    vel_world: np.ndarray
    heading: float


@dataclass
class NoFlyZoneState:
    center_world: np.ndarray
    radius_world: float


@dataclass
class TargetTruthState:
    t: float
    pos_world: np.ndarray
    vel_world: np.ndarray
    heading: float
    motion_mode: str


@dataclass
class VisionObservation:
    t: float
    detected: bool
    center_px: Optional[Tuple[float, float]]
    bbox_xywh: Optional[Tuple[float, float, float, float]]
    score: float
    crop_path: Optional[str]
    crop_center_world: Optional[Tuple[float, float]]
    gsd: Optional[float]
    meta: Optional[Dict[str, Any]] = None


@dataclass
class TargetEstimateState:
    t: float
    pos_world_est: np.ndarray
    vel_world_est: np.ndarray
    cov: np.ndarray
    obs_conf: float
    obs_age: float