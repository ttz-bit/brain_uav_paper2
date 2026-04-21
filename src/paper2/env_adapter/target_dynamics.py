from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from paper2.common.types import TargetTruthState


@dataclass
class TargetMotionInternalState:
    mode: str
    omega: float
    steps_to_switch: int
    segment_id: int
    evasive_active: bool


def _sample_speed(cfg: dict[str, Any], rng: np.random.Generator) -> float:
    smin = float(cfg["speed_range"]["min"])
    smax = float(cfg["speed_range"]["max"])
    return float(rng.uniform(smin, smax))


def _sample_turn_rate(cfg: dict[str, Any], rng: np.random.Generator) -> float:
    omin = float(cfg["turn_rate_range"]["min"])
    omax = float(cfg["turn_rate_range"]["max"])
    return float(rng.uniform(omin, omax))


def _sample_switch_interval(cfg: dict[str, Any], rng: np.random.Generator) -> int:
    imin = int(cfg["switch_interval_range"]["min"])
    imax = int(cfg["switch_interval_range"]["max"])
    return int(rng.integers(imin, imax + 1))


def sample_motion_mode(cfg: dict[str, Any], rng: np.random.Generator) -> str:
    probs = cfg["mode_probs"]
    names = ["cv", "turn", "piecewise", "evasive"]
    pvals = np.array([float(probs[k]) for k in names], dtype=float)
    pvals = pvals / pvals.sum()
    return str(rng.choice(names, p=pvals))


def init_motion_internal(
    mode: str, cfg: dict[str, Any], rng: np.random.Generator
) -> TargetMotionInternalState:
    return TargetMotionInternalState(
        mode=mode,
        omega=_sample_turn_rate(cfg, rng),
        steps_to_switch=_sample_switch_interval(cfg, rng),
        segment_id=0,
        evasive_active=False,
    )


def make_initial_target_truth(
    t: float, pos_world: np.ndarray, mode: str, cfg: dict[str, Any], rng: np.random.Generator
) -> TargetTruthState:
    speed = _sample_speed(cfg, rng)
    heading = float(rng.uniform(-math.pi, math.pi))
    vel = np.array([speed * math.cos(heading), speed * math.sin(heading)], dtype=float)
    return TargetTruthState(
        t=float(t),
        pos_world=pos_world.astype(float),
        vel_world=vel,
        heading=heading,
        motion_mode=mode,
    )


def _from_heading_speed(heading: float, speed: float) -> np.ndarray:
    return np.array([speed * math.cos(heading), speed * math.sin(heading)], dtype=float)


def propagate_target_truth(
    truth: TargetTruthState,
    internal: TargetMotionInternalState,
    dt: float,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> TargetTruthState:
    speed = float(np.linalg.norm(truth.vel_world))
    heading = float(truth.heading)
    omega = float(internal.omega)

    if internal.mode == "cv":
        vel = truth.vel_world.copy()
    elif internal.mode == "turn":
        heading = heading + omega * dt
        vel = _from_heading_speed(heading, speed)
    elif internal.mode == "piecewise":
        internal.steps_to_switch -= 1
        if internal.steps_to_switch <= 0:
            delta = float(rng.uniform(-0.75, 0.75))
            heading = heading + delta
            speed = _sample_speed(cfg, rng)
            vel = _from_heading_speed(heading, speed)
            internal.segment_id += 1
            internal.steps_to_switch = _sample_switch_interval(cfg, rng)
        else:
            vel = truth.vel_world.copy()
    elif internal.mode == "evasive":
        emin = float(cfg["evasive_intensity_range"]["min"])
        emax = float(cfg["evasive_intensity_range"]["max"])
        if rng.uniform() < 0.25:
            internal.evasive_active = not internal.evasive_active
        intensity = float(rng.uniform(emin, emax))
        sign = -1.0 if rng.uniform() < 0.5 else 1.0
        burst = sign * intensity if internal.evasive_active else sign * intensity * 0.4
        heading = heading + burst
        speed = float(np.clip(_sample_speed(cfg, rng), cfg["speed_range"]["min"], cfg["speed_range"]["max"]))
        vel = _from_heading_speed(heading, speed)
    else:
        raise ValueError(f"Unsupported target mode: {internal.mode}")

    pos = truth.pos_world + vel * dt
    return TargetTruthState(
        t=float(truth.t + dt),
        pos_world=pos.astype(float),
        vel_world=vel.astype(float),
        heading=float(math.atan2(vel[1], vel[0])),
        motion_mode=internal.mode,
    )
