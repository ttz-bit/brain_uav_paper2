from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np

from paper2.common.types import AircraftState, TargetTruthState
from paper2.env_adapter.phase3_target_motion import (
    MOTION_MODES,
    generate_phase3_target_trajectory,
)
from paper2.env_adapter.water_constraint import WaterConstraint, build_water_constraint


STAGES = ("far", "mid", "terminal")


@dataclass(frozen=True)
class Phase3TaskFrame:
    sequence_id: str
    frame_id: int
    t: float
    stage: str
    range_xy_km: float
    range_3d_km: float
    gsd_km_per_px: float
    target_state_world: dict[str, Any]
    aircraft_state: dict[str, Any]
    crop_center_world: list[float]
    center_px: list[float]
    target_on_water: bool
    motion_mode: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_phase3_stage(range_xy_km: float, stage_cfg: dict[str, Any]) -> str:
    r = float(range_xy_km)
    for stage in STAGES:
        cfg = stage_cfg[stage]
        if float(cfg["range_min_km"]) <= r <= float(cfg["range_max_km"]):
            return stage
    raise ValueError(f"range_xy_km={r:.3f} does not fit any phase3 task stage.")


def sample_phase3_task_sequence(
    *,
    sequence_idx: int,
    target_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    seed: int,
    frames: int | None = None,
    mode: str | None = None,
    water_constraint: WaterConstraint | None = None,
) -> list[Phase3TaskFrame]:
    rng = np.random.default_rng(int(seed))
    water = water_constraint or build_water_constraint(target_cfg.get("water_constraint"))
    frame_count = int(frames if frames is not None else target_cfg["frames_per_sequence"])
    mode_name = mode if mode is not None else MOTION_MODES[sequence_idx % len(MOTION_MODES)]
    target_rows = generate_phase3_target_trajectory(
        target_cfg,
        seed=int(seed),
        frames=frame_count,
        mode=mode_name,
        water_constraint=water,
    )

    sequence_id = f"phase3_seq_{sequence_idx:06d}"
    frames_out: list[Phase3TaskFrame] = []
    stage_plan = _approach_stage_plan(frame_count, sequence_idx)
    stage_offsets = _smooth_center_offsets_px(stage_plan, stage_cfg, rng)
    bearing = float(rng.uniform(-math.pi, math.pi))
    bearing_rate = float(rng.uniform(-0.035, 0.035))
    stage_seen = {stage: 0 for stage in STAGES}
    stage_total = {stage: int(stage_plan.count(stage)) for stage in STAGES}
    for frame_id, target_truth in enumerate(target_rows):
        stage = stage_plan[frame_id]
        stage_params = stage_cfg[stage]
        local_idx = stage_seen[stage]
        stage_seen[stage] += 1
        range_xy = _approach_range_km(stage, local_idx, stage_total[stage], stage_cfg, rng)
        bearing = _wrap_angle(bearing + bearing_rate + float(rng.normal(0.0, 0.01)))
        aircraft_xy = np.asarray(target_truth.pos_world, dtype=float)[:2] - range_xy * np.array(
            [math.cos(bearing), math.sin(bearing)],
            dtype=float,
        )
        aircraft_z = float(stage_cfg.get("aircraft_altitude_km", 12.0))
        aircraft_pos = np.array([aircraft_xy[0], aircraft_xy[1], aircraft_z], dtype=float)
        target_xyz = np.array([target_truth.pos_world[0], target_truth.pos_world[1], 0.0], dtype=float)
        range_3d = float(np.linalg.norm(aircraft_pos - target_xyz))

        gsd = float(stage_params["gsd_km_per_px"])
        image_size = int(stage_cfg.get("image_size", 256))
        offset_px = stage_offsets[frame_id]
        crop_center = np.asarray(target_truth.pos_world, dtype=float)[:2] - offset_px * gsd
        center_px = _world_to_image_px(
            target_xy=np.asarray(target_truth.pos_world, dtype=float)[:2],
            crop_center_xy=crop_center,
            gsd_km_per_px=gsd,
            image_size=image_size,
        )

        aircraft = AircraftState(
            t=float(target_truth.t),
            pos_world=aircraft_pos,
            vel_world=np.zeros(3, dtype=float),
            heading=float(math.atan2(target_truth.pos_world[1] - aircraft_xy[1], target_truth.pos_world[0] - aircraft_xy[0])),
            speed=None,
            meta={"source": "phase3_task_sampler", "unit": "km"},
        )
        frames_out.append(
            Phase3TaskFrame(
                sequence_id=sequence_id,
                frame_id=int(frame_id),
                t=float(target_truth.t),
                stage=stage,
                range_xy_km=range_xy,
                range_3d_km=range_3d,
                gsd_km_per_px=gsd,
                target_state_world=_target_truth_to_dict(target_truth),
                aircraft_state=_aircraft_to_dict(aircraft),
                crop_center_world=[float(crop_center[0]), float(crop_center[1])],
                center_px=[float(center_px[0]), float(center_px[1])],
                target_on_water=bool(water.is_water_world(target_truth.pos_world)),
                motion_mode=str(target_truth.motion_mode),
            )
        )
    return frames_out


def _approach_stage_plan(frame_count: int, sequence_idx: int) -> list[str]:
    """Return contiguous far->mid->terminal stages while keeping corpus counts balanced."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    base = int(frame_count) // len(STAGES)
    remainder = int(frame_count) % len(STAGES)
    counts = {stage: base for stage in STAGES}
    for extra_idx in range(remainder):
        counts[STAGES[(int(sequence_idx) + extra_idx) % len(STAGES)]] += 1
    plan: list[str] = []
    for stage in STAGES:
        plan.extend([stage] * counts[stage])
    return plan


def _approach_range_km(
    stage: str,
    local_idx: int,
    stage_count: int,
    stage_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> float:
    params = stage_cfg[stage]
    r_min = float(params["range_min_km"])
    r_max = float(params["range_max_km"])
    if stage_count <= 1:
        return float(0.5 * (r_min + r_max))
    # Jitter inside each monotonic slot. The 0.2..0.8 margin prevents adjacent
    # slots from crossing while avoiding a perfectly uniform synthetic pattern.
    frac = (float(local_idx) + float(rng.uniform(0.2, 0.8))) / float(stage_count)
    return float(r_max - frac * (r_max - r_min))


def _smooth_center_offsets_px(
    stage_plan: list[str],
    stage_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    jitter_cfg = stage_cfg.get("crop_jitter_px", {})
    offsets: list[np.ndarray] = []
    offset = np.zeros(2, dtype=float)
    for stage in stage_plan:
        limit = float(jitter_cfg.get(stage, 0.0))
        step_sigma = max(0.25, limit * 0.12)
        offset = offset + rng.normal(0.0, step_sigma, size=2)
        norm = float(np.linalg.norm(offset))
        if limit > 0.0 and norm > limit:
            offset = offset / max(norm, 1e-9) * limit
        offsets.append(offset.astype(float).copy())
    return offsets


def summarize_phase3_task_frames(frames: list[Phase3TaskFrame], stage_cfg: dict[str, Any]) -> dict[str, Any]:
    stage_counts = {stage: 0 for stage in STAGES}
    invalid_stage_ranges = 0
    out_of_image = 0
    not_water = 0
    ranges = []
    offcenter = []
    image_size = float(stage_cfg.get("image_size", 256))
    center = np.array([image_size * 0.5, image_size * 0.5], dtype=float)
    for row in frames:
        stage_counts[row.stage] = stage_counts.get(row.stage, 0) + 1
        ranges.append(float(row.range_xy_km))
        try:
            got_stage = classify_phase3_stage(row.range_xy_km, stage_cfg)
        except ValueError:
            invalid_stage_ranges += 1
        else:
            if got_stage != row.stage:
                invalid_stage_ranges += 1
        px = np.asarray(row.center_px, dtype=float)
        if not (0.0 <= float(px[0]) < image_size and 0.0 <= float(px[1]) < image_size):
            out_of_image += 1
        offcenter.append(float(np.linalg.norm(px - center)))
        if not row.target_on_water:
            not_water += 1

    range_arr = np.asarray(ranges, dtype=float)
    off_arr = np.asarray(offcenter, dtype=float)
    return {
        "num_frames": int(len(frames)),
        "stage_counts": stage_counts,
        "range_xy_min_km": float(range_arr.min()) if range_arr.size else 0.0,
        "range_xy_mean_km": float(range_arr.mean()) if range_arr.size else 0.0,
        "range_xy_max_km": float(range_arr.max()) if range_arr.size else 0.0,
        "offcenter_px_mean": float(off_arr.mean()) if off_arr.size else 0.0,
        "offcenter_px_max": float(off_arr.max()) if off_arr.size else 0.0,
        "invalid_stage_ranges": int(invalid_stage_ranges),
        "center_px_out_of_image": int(out_of_image),
        "target_not_water": int(not_water),
        "accepted": bool(
            len(frames) > 0
            and all(stage_counts.get(stage, 0) > 0 for stage in STAGES)
            and invalid_stage_ranges == 0
            and out_of_image == 0
            and not_water == 0
        ),
    }


def _world_to_image_px(
    *,
    target_xy: np.ndarray,
    crop_center_xy: np.ndarray,
    gsd_km_per_px: float,
    image_size: int,
) -> np.ndarray:
    half = float(image_size) * 0.5
    gsd = max(float(gsd_km_per_px), 1e-12)
    u = half + (float(target_xy[0]) - float(crop_center_xy[0])) / gsd
    v = half - (float(target_xy[1]) - float(crop_center_xy[1])) / gsd
    return np.array([u, v], dtype=float)


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _target_truth_to_dict(truth: TargetTruthState) -> dict[str, Any]:
    return {
        "t": float(truth.t),
        "pos_world": [float(v) for v in np.asarray(truth.pos_world, dtype=float).tolist()],
        "vel_world": [float(v) for v in np.asarray(truth.vel_world, dtype=float).tolist()],
        "heading": float(truth.heading),
        "motion_mode": str(truth.motion_mode),
        "unit": "km",
    }


def _aircraft_to_dict(aircraft: AircraftState) -> dict[str, Any]:
    return {
        "t": float(aircraft.t),
        "pos_world": [float(v) for v in np.asarray(aircraft.pos_world, dtype=float).tolist()],
        "vel_world": [float(v) for v in np.asarray(aircraft.vel_world, dtype=float).tolist()],
        "heading": float(aircraft.heading),
        "unit": "km",
    }
