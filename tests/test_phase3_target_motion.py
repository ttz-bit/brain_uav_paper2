from __future__ import annotations

from pathlib import Path

import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.phase3_target_motion import (
    MOTION_MODES,
    generate_phase3_target_trajectory,
    summarize_phase3_target_trajectories,
)


def _cfg() -> dict:
    return load_yaml(Path(__file__).resolve().parents[1] / "configs" / "env.yaml")["phase3_target_motion"]


def test_phase3_target_motion_is_reproducible():
    cfg = _cfg()
    a = generate_phase3_target_trajectory(cfg, seed=123, frames=12, mode="turn")
    b = generate_phase3_target_trajectory(cfg, seed=123, frames=12, mode="turn")
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert np.allclose(left.pos_world, right.pos_world)
        assert np.allclose(left.vel_world, right.vel_world)
        assert left.motion_mode == right.motion_mode


def test_phase3_target_motion_all_modes_stay_valid():
    cfg = _cfg()
    trajectories = [
        generate_phase3_target_trajectory(cfg, seed=100 + idx, frames=40, mode=mode)
        for idx, mode in enumerate(MOTION_MODES)
    ]
    report = summarize_phase3_target_trajectories(trajectories, cfg)
    assert report["bounds_ok"] is True
    assert report["continuity_ok"] is True
    assert float(report["speed_min"]) >= float(cfg["speed_range"]["min"]) - 1e-9
    assert float(report["speed_max"]) <= float(cfg["speed_range"]["max"]) + 1e-9
    for mode in MOTION_MODES:
        assert report["mode_counts"][mode] == 1


def test_phase3_evasive_motion_responds_to_near_aircraft():
    cfg = _cfg()
    aircraft_positions = [np.array([2000.0, 2000.0], dtype=float) for _ in range(8)]
    traj = generate_phase3_target_trajectory(
        cfg,
        seed=77,
        frames=8,
        mode="evasive",
        aircraft_positions_world=aircraft_positions,
    )
    assert len(traj) == 8
    assert all(np.isfinite(row.pos_world).all() for row in traj)
