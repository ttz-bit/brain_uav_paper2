from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from paper2.common.types import TargetTruthState
from paper2.env_adapter.water_constraint import WaterConstraint, build_water_constraint


MOTION_MODES = ("cv", "turn", "piecewise", "evasive")


@dataclass
class Phase3TargetInternalState:
    mode: str
    omega: float
    steps_to_switch: int
    segment_id: int = 0


def sample_phase3_motion_mode(cfg: dict[str, Any], rng: np.random.Generator) -> str:
    probs = cfg["mode_probs"]
    weights = np.array([float(probs[name]) for name in MOTION_MODES], dtype=float)
    weights = weights / max(float(weights.sum()), 1e-12)
    return str(rng.choice(list(MOTION_MODES), p=weights))


def sample_phase3_initial_target(
    cfg: dict[str, Any],
    rng: np.random.Generator,
    *,
    mode: str | None = None,
    water_constraint: WaterConstraint | None = None,
) -> tuple[TargetTruthState, Phase3TargetInternalState]:
    mode_name = sample_phase3_motion_mode(cfg, rng) if mode is None else str(mode)
    if mode_name not in MOTION_MODES:
        raise ValueError(f"Unsupported phase3 target motion mode: {mode_name}")

    area = cfg["area"]
    margin = float(cfg.get("init_margin", 0.0))
    x_min = float(area["x_min"]) + margin
    x_max = float(area["x_max"]) - margin
    y_min = float(area["y_min"]) + margin
    y_max = float(area["y_max"]) - margin
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("phase3_target_motion init_margin leaves no valid sampling area.")

    water = water_constraint or build_water_constraint(cfg.get("water_constraint"))
    attempts = int(cfg.get("water_constraint", {}).get("max_resample_attempts", 16))
    pos = None
    for _ in range(max(1, attempts)):
        candidate = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
        if water.is_water_world(candidate):
            pos = candidate
            break
    if pos is None:
        raise RuntimeError("Failed to sample phase3 target initial position on water.")
    speed = _sample_speed(cfg, rng)
    heading = float(rng.uniform(-math.pi, math.pi))
    truth = TargetTruthState(
        t=0.0,
        pos_world=pos,
        vel_world=_from_heading_speed(heading, speed),
        heading=heading,
        motion_mode=mode_name,
    )
    internal = Phase3TargetInternalState(
        mode=mode_name,
        omega=_sample_turn_rate(cfg, rng),
        steps_to_switch=_sample_switch_interval(cfg, rng),
    )
    return truth, internal


def propagate_phase3_target_truth(
    truth: TargetTruthState,
    internal: Phase3TargetInternalState,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    *,
    aircraft_pos_world: np.ndarray | None = None,
    water_constraint: WaterConstraint | None = None,
) -> TargetTruthState:
    dt = float(cfg["dt"])
    speed = _clip_speed(float(np.linalg.norm(truth.vel_world)), cfg)
    heading = float(truth.heading)

    if internal.mode == "cv":
        pass
    elif internal.mode == "turn":
        heading += float(internal.omega) * dt
    elif internal.mode == "piecewise":
        internal.steps_to_switch -= 1
        if internal.steps_to_switch <= 0:
            delta_cfg = cfg["piecewise_heading_delta_range"]
            heading += float(rng.uniform(float(delta_cfg["min"]), float(delta_cfg["max"])))
            speed = _sample_speed(cfg, rng)
            internal.segment_id += 1
            internal.steps_to_switch = _sample_switch_interval(cfg, rng)
    elif internal.mode == "evasive":
        heading = _evasive_heading(heading, truth.pos_world, aircraft_pos_world, cfg, rng)
        speed = _sample_speed(cfg, rng)
    else:
        raise ValueError(f"Unsupported phase3 target motion mode: {internal.mode}")

    water = water_constraint or build_water_constraint(cfg.get("water_constraint"))
    attempts = int(cfg.get("water_constraint", {}).get("max_resample_attempts", 16))
    next_pos = None
    for attempt in range(max(1, attempts)):
        candidate_heading = heading if attempt == 0 else float(rng.uniform(-math.pi, math.pi))
        vel = _from_heading_speed(candidate_heading, speed)
        candidate_pos = np.asarray(truth.pos_world, dtype=float) + vel * dt
        candidate_pos, candidate_heading = _reflect_if_needed(candidate_pos, candidate_heading, cfg)
        if water.is_water_world(candidate_pos):
            next_pos = candidate_pos
            heading = candidate_heading
            break
    if next_pos is None:
        next_pos = np.asarray(truth.pos_world, dtype=float).copy()
        heading = float(truth.heading)
    vel = _from_heading_speed(heading, speed)
    return TargetTruthState(
        t=float(truth.t + dt),
        pos_world=next_pos.astype(float),
        vel_world=vel.astype(float),
        heading=float(math.atan2(vel[1], vel[0])),
        motion_mode=internal.mode,
    )


def generate_phase3_target_trajectory(
    cfg: dict[str, Any],
    *,
    seed: int,
    frames: int | None = None,
    mode: str | None = None,
    aircraft_positions_world: list[np.ndarray] | None = None,
    water_constraint: WaterConstraint | None = None,
) -> list[TargetTruthState]:
    rng = np.random.default_rng(int(seed))
    frame_count = int(frames if frames is not None else cfg["frames_per_sequence"])
    if frame_count <= 0:
        raise ValueError("frames must be positive.")

    water = water_constraint or build_water_constraint(cfg.get("water_constraint"))
    truth, internal = sample_phase3_initial_target(cfg, rng, mode=mode, water_constraint=water)
    rows = [truth]
    for idx in range(1, frame_count):
        aircraft_pos = None
        if aircraft_positions_world is not None and idx - 1 < len(aircraft_positions_world):
            aircraft_pos = np.asarray(aircraft_positions_world[idx - 1], dtype=float)
        truth = propagate_phase3_target_truth(
            truth,
            internal,
            cfg,
            rng,
            aircraft_pos_world=aircraft_pos,
            water_constraint=water,
        )
        rows.append(truth)
    return rows


def summarize_phase3_target_trajectories(
    trajectories: list[list[TargetTruthState]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    speeds: list[float] = []
    step_distances: list[float] = []
    out_of_bounds = 0
    total_frames = 0
    mode_counts = {name: 0 for name in MOTION_MODES}
    max_step = 0.0

    for traj in trajectories:
        if not traj:
            continue
        mode_counts[str(traj[0].motion_mode)] = mode_counts.get(str(traj[0].motion_mode), 0) + 1
        prev = None
        for truth in traj:
            total_frames += 1
            speeds.append(float(np.linalg.norm(truth.vel_world)))
            if not _inside_area(truth.pos_world, cfg):
                out_of_bounds += 1
            if prev is not None:
                step = float(np.linalg.norm(truth.pos_world - prev.pos_world))
                step_distances.append(step)
                max_step = max(max_step, step)
            prev = truth

    speed_arr = np.asarray(speeds, dtype=float)
    step_arr = np.asarray(step_distances, dtype=float)
    speed_max = float(cfg["speed_range"]["max"])
    max_allowed_step = speed_max * float(cfg["dt"]) * float(cfg["continuity"]["max_step_distance_factor"])
    return {
        "num_trajectories": int(len(trajectories)),
        "total_frames": int(total_frames),
        "mode_counts": mode_counts,
        "speed_min": float(speed_arr.min()) if speed_arr.size else 0.0,
        "speed_mean": float(speed_arr.mean()) if speed_arr.size else 0.0,
        "speed_max": float(speed_arr.max()) if speed_arr.size else 0.0,
        "step_distance_mean": float(step_arr.mean()) if step_arr.size else 0.0,
        "step_distance_max": float(max_step),
        "max_allowed_step_distance": float(max_allowed_step),
        "out_of_bounds_frames": int(out_of_bounds),
        "out_of_bounds_ratio": float(out_of_bounds / max(1, total_frames)),
        "continuity_ok": bool(max_step <= max_allowed_step + 1e-9),
        "bounds_ok": bool(out_of_bounds == 0),
    }


def _sample_speed(cfg: dict[str, Any], rng: np.random.Generator) -> float:
    return float(rng.uniform(float(cfg["speed_range"]["min"]), float(cfg["speed_range"]["max"])))


def _clip_speed(speed: float, cfg: dict[str, Any]) -> float:
    return float(np.clip(speed, float(cfg["speed_range"]["min"]), float(cfg["speed_range"]["max"])))


def _sample_turn_rate(cfg: dict[str, Any], rng: np.random.Generator) -> float:
    return float(rng.uniform(float(cfg["turn_rate_range"]["min"]), float(cfg["turn_rate_range"]["max"])))


def _sample_switch_interval(cfg: dict[str, Any], rng: np.random.Generator) -> int:
    return int(rng.integers(int(cfg["switch_interval_range"]["min"]), int(cfg["switch_interval_range"]["max"]) + 1))


def _from_heading_speed(heading: float, speed: float) -> np.ndarray:
    return np.array([speed * math.cos(heading), speed * math.sin(heading)], dtype=float)


def _evasive_heading(
    heading: float,
    target_pos: np.ndarray,
    aircraft_pos: np.ndarray | None,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    evasive_cfg = cfg["evasive"]
    if aircraft_pos is not None and aircraft_pos.size >= 2:
        rel = np.asarray(target_pos, dtype=float)[:2] - np.asarray(aircraft_pos, dtype=float)[:2]
        dist = float(np.linalg.norm(rel))
        if dist <= float(evasive_cfg["trigger_distance"]) and dist > 1e-8:
            away_heading = float(math.atan2(rel[1], rel[0]))
            gain = float(evasive_cfg["heading_gain"])
            return _wrap_angle((1.0 - gain) * heading + gain * away_heading)

    jitter_cfg = evasive_cfg["random_jitter_range"]
    return _wrap_angle(heading + float(rng.uniform(float(jitter_cfg["min"]), float(jitter_cfg["max"]))))


def _reflect_if_needed(pos: np.ndarray, heading: float, cfg: dict[str, Any]) -> tuple[np.ndarray, float]:
    area = cfg["area"]
    margin = float(cfg.get("boundary_margin", 0.0))
    x_min = float(area["x_min"]) + margin
    x_max = float(area["x_max"]) - margin
    y_min = float(area["y_min"]) + margin
    y_max = float(area["y_max"]) - margin
    next_pos = np.asarray(pos, dtype=float).copy()
    next_heading = float(heading)

    if next_pos[0] < x_min:
        next_pos[0] = x_min + (x_min - next_pos[0])
        next_heading = math.pi - next_heading
    elif next_pos[0] > x_max:
        next_pos[0] = x_max - (next_pos[0] - x_max)
        next_heading = math.pi - next_heading

    if next_pos[1] < y_min:
        next_pos[1] = y_min + (y_min - next_pos[1])
        next_heading = -next_heading
    elif next_pos[1] > y_max:
        next_pos[1] = y_max - (next_pos[1] - y_max)
        next_heading = -next_heading

    next_pos[0] = float(np.clip(next_pos[0], x_min, x_max))
    next_pos[1] = float(np.clip(next_pos[1], y_min, y_max))
    return next_pos, _wrap_angle(next_heading)


def _inside_area(pos: np.ndarray, cfg: dict[str, Any]) -> bool:
    area = cfg["area"]
    x = float(pos[0])
    y = float(pos[1])
    return (
        float(area["x_min"]) <= x <= float(area["x_max"])
        and float(area["y_min"]) <= y <= float(area["y_max"])
    )


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)
