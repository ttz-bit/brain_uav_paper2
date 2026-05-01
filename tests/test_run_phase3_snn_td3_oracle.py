from __future__ import annotations

import numpy as np

from paper2.common.types import TargetTruthState
from scripts.run_phase3_snn_td3_oracle import _episode_summary
from scripts.run_phase3_snn_td3_oracle import _init_phase3_dynamic_target


def test_episode_summary_accepts_paper1_segment_capture():
    rows = [
        {
            "range_to_target_km": 5.1,
            "target_est_error_km": 0.0,
            "zone_violation": False,
            "safety_margin_violation": False,
            "done_reason": "captured",
        }
    ]

    summary = _episode_summary(rows, capture_radius_km=5.0)

    assert summary["captured"] is True


def test_phase3_dynamic_target_can_start_from_paper1_goal():
    class DummyBridge:
        def get_target_truth(self):
            return TargetTruthState(
                t=0.0,
                pos_world=np.array([10.0, 20.0, 30.0], dtype=float),
                vel_world=np.zeros(3, dtype=float),
                heading=0.0,
                motion_mode="static_goal",
            )

    cfg = {
        "mode_probs": {"cv": 1.0, "turn": 0.0, "piecewise": 0.0, "evasive": 0.0},
        "area": {"x_min": 0.0, "x_max": 100.0, "y_min": 0.0, "y_max": 100.0},
        "init_margin": 1.0,
        "water_constraint": {"mode": "all_water"},
        "speed_range": {"min": 0.01, "max": 0.01},
        "turn_rate_range": {"min": 0.0, "max": 0.0},
        "switch_interval_range": {"min": 2, "max": 2},
    }

    truth, _ = _init_phase3_dynamic_target(DummyBridge(), cfg, np.random.default_rng(1), "paper1_goal")

    assert np.allclose(truth.pos_world, [10.0, 20.0])
    assert truth.vel_world.shape == (2,)
