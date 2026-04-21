from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import numpy as np

from paper2.common.types import AircraftState, NoFlyZoneState, TargetTruthState
from paper2.env_adapter.target_dynamics import (
    TargetMotionInternalState,
    init_motion_internal,
    make_initial_target_truth,
    sample_motion_mode,
)


@dataclass
class EpisodeInit:
    aircraft: AircraftState
    target_truth: TargetTruthState
    target_internal: TargetMotionInternalState
    no_fly_zones: list[NoFlyZoneState]


def _sample_point(area: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    return np.array(
        [
            rng.uniform(float(area["x_min"]), float(area["x_max"])),
            rng.uniform(float(area["y_min"]), float(area["y_max"])),
        ],
        dtype=float,
    )


def _point_inside_nfz(point: np.ndarray, nfz: NoFlyZoneState) -> bool:
    return float(np.linalg.norm(point - nfz.center_world)) <= float(nfz.radius_world)


def _sample_no_fly_zones(cfg: dict[str, Any], rng: np.random.Generator) -> list[NoFlyZoneState]:
    area = cfg["area"]
    nfz_cfg = cfg["no_fly_zone"]
    count = int(rng.integers(int(nfz_cfg["count_min"]), int(nfz_cfg["count_max"]) + 1))
    zones: list[NoFlyZoneState] = []
    for _ in range(count):
        center = _sample_point(area, rng)
        radius = float(rng.uniform(float(nfz_cfg["radius_min"]), float(nfz_cfg["radius_max"])))
        zones.append(NoFlyZoneState(center_world=center, radius_world=radius))
    return zones


def _time_to_area_exit(pos: np.ndarray, vel: np.ndarray, area: dict[str, float]) -> float:
    times: list[float] = []
    px = float(pos[0])
    py = float(pos[1])
    vx = float(vel[0])
    vy = float(vel[1])
    if vx > 1e-8:
        times.append((float(area["x_max"]) - px) / vx)
    elif vx < -1e-8:
        times.append((float(area["x_min"]) - px) / vx)
    if vy > 1e-8:
        times.append((float(area["y_max"]) - py) / vy)
    elif vy < -1e-8:
        times.append((float(area["y_min"]) - py) / vy)
    pos_times = [t for t in times if t > 0.0]
    if not pos_times:
        return float("inf")
    return float(min(pos_times))


def sample_episode_init(cfg: dict[str, Any], rng: np.random.Generator) -> EpisodeInit:
    area = cfg["area"]
    min_dist = float(cfg["min_init_dist"])
    max_dist = float(cfg["max_init_dist"])
    resample_limit = int(cfg["resample_limit"])
    truth_crop_horizon_sec = float(cfg["truth_crop_horizon_sec"])
    aircraft_speed = float(cfg["aircraft"]["speed"])
    dyn_cfg = cfg["target_dynamics"]

    for _ in range(resample_limit):
        zones = _sample_no_fly_zones(cfg, rng)
        aircraft_pos = _sample_point(area, rng)
        target_pos = _sample_point(area, rng)
        dist = float(np.linalg.norm(target_pos - aircraft_pos))

        if dist < min_dist or dist > max_dist:
            continue

        if any(_point_inside_nfz(aircraft_pos, z) for z in zones):
            continue
        if any(_point_inside_nfz(target_pos, z) for z in zones):
            continue

        heading = float(rng.uniform(-math.pi, math.pi))
        aircraft_vel = np.array(
            [aircraft_speed * math.cos(heading), aircraft_speed * math.sin(heading)],
            dtype=float,
        )
        aircraft = AircraftState(
            t=0.0,
            pos_world=aircraft_pos,
            vel_world=aircraft_vel,
            heading=heading,
        )

        mode = sample_motion_mode(dyn_cfg, rng)
        truth = make_initial_target_truth(
            t=0.0,
            pos_world=target_pos,
            mode=mode,
            cfg=dyn_cfg,
            rng=rng,
        )
        # Keep phase1a episodes away from trivial early target-out-of-bounds failures.
        crop_center = truth.pos_world + truth.vel_world * truth_crop_horizon_sec
        x_min = float(area["x_min"])
        x_max = float(area["x_max"])
        y_min = float(area["y_min"])
        y_max = float(area["y_max"])
        crop_in_area = x_min <= float(crop_center[0]) <= x_max and y_min <= float(crop_center[1]) <= y_max
        if not crop_in_area:
            continue
        if _time_to_area_exit(truth.pos_world, truth.vel_world, area) < 20.0:
            continue

        internal = init_motion_internal(mode, dyn_cfg, rng)
        return EpisodeInit(
            aircraft=aircraft,
            target_truth=truth,
            target_internal=internal,
            no_fly_zones=zones,
        )

    raise RuntimeError("Failed to sample a valid episode init under constraints.")
