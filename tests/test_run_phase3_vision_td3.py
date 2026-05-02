from __future__ import annotations

import numpy as np

from scripts.run_phase3_vision_td3 import _gate_and_smooth_estimate, _stage_for_range, _vision_estimate_from_row
from paper2.common.types import TargetEstimateState


def test_stage_for_range_uses_phase3_thresholds():
    cfg = {
        "far": {"range_min_km": 1500.0, "range_max_km": 2000.0},
        "mid": {"range_min_km": 600.0, "range_max_km": 1500.0},
        "terminal": {"range_min_km": 50.0, "range_max_km": 600.0},
    }

    assert _stage_for_range(1800.0, cfg) == "far"
    assert _stage_for_range(900.0, cfg) == "mid"
    assert _stage_for_range(120.0, cfg) == "terminal"
    assert _stage_for_range(10.0, cfg) == "terminal"


def test_vision_estimate_replays_dataset_world_error_on_current_target():
    row = {
        "image_path": "dummy.png",
        "sequence_id": "seq",
        "gsd_km_per_px": 1.0,
        "meta": {
            "crop_center_world": [100.0, 100.0],
            "gsd": 1.0,
            "target_state_world": {"x": 100.0, "y": 100.0},
        },
    }

    estimate, error = _vision_estimate_from_row(
        row=row,
        pred_center_px=(138.0, 128.0),
        pred_conf=0.9,
        current_target_xy=np.array([500.0, 700.0], dtype=float),
        current_target_z=123.0,
        t=0.0,
    )

    assert np.allclose(estimate.pos_world_est, [510.0, 700.0, 123.0])
    assert np.isclose(error, 10.0)
    assert estimate.obs_conf == 0.9


class _Args:
    disable_estimate_gating = False
    gate_far_km = 300.0
    gate_mid_km = 120.0
    gate_terminal_km = 25.0
    gain_far = 0.25
    gain_mid = 0.50
    gain_terminal = 0.80


def _estimate(pos, *, t=0.0):
    return TargetEstimateState(
        t=float(t),
        pos_world_est=np.asarray(pos, dtype=float),
        vel_world_est=np.zeros(3, dtype=float),
        cov=np.eye(6, dtype=float),
        obs_conf=1.0,
        obs_age=0.0,
        meta={"source": "test"},
    )


def test_estimate_gate_rejects_large_terminal_jump():
    prev = _estimate([100.0, 100.0, 12.0], t=0.0)
    cand = _estimate([180.0, 100.0, 12.0], t=1.0)

    estimate, accepted, innovation, gain = _gate_and_smooth_estimate(cand, prev, stage="terminal", args=_Args())

    assert accepted is False
    assert np.isclose(innovation, 80.0)
    assert gain == 0.0
    assert np.allclose(estimate.pos_world_est, [100.0, 100.0, 12.0])
    assert estimate.meta["gate_accepted"] is False


def test_estimate_gate_smooths_plausible_terminal_update():
    prev = _estimate([100.0, 100.0, 12.0], t=0.0)
    cand = _estimate([110.0, 100.0, 12.0], t=1.0)

    estimate, accepted, innovation, gain = _gate_and_smooth_estimate(cand, prev, stage="terminal", args=_Args())

    assert accepted is True
    assert np.isclose(innovation, 10.0)
    assert np.isclose(gain, 0.80)
    assert np.allclose(estimate.pos_world_est, [108.0, 100.0, 12.0])
    assert estimate.meta["gate_accepted"] is True
