from __future__ import annotations

import numpy as np

from scripts.run_phase3_vision_td3 import _stage_for_range, _vision_estimate_from_row


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
