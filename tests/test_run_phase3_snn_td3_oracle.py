from __future__ import annotations

from scripts.run_phase3_snn_td3_oracle import _episode_summary


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
