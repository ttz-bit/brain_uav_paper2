from __future__ import annotations

from pathlib import Path

import numpy as np

from paper2.common.config import load_yaml
from paper2.render.phase3_task_sampler import (
    STAGES,
    classify_phase3_stage,
    sample_phase3_task_sequence,
    summarize_phase3_task_frames,
)


def _cfg() -> dict:
    return load_yaml(Path(__file__).resolve().parents[1] / "configs" / "env.yaml")


def test_classify_phase3_stage_uses_range_thresholds():
    stage_cfg = _cfg()["phase3_task_stages"]
    assert classify_phase3_stage(1700.0, stage_cfg) == "far"
    assert classify_phase3_stage(900.0, stage_cfg) == "mid"
    assert classify_phase3_stage(100.0, stage_cfg) == "terminal"


def test_phase3_task_sampler_outputs_valid_frames():
    cfg = _cfg()
    rows = sample_phase3_task_sequence(
        sequence_idx=0,
        target_cfg=cfg["phase3_target_motion"],
        stage_cfg=cfg["phase3_task_stages"],
        seed=123,
        frames=9,
    )
    assert len(rows) == 9
    assert {row.stage for row in rows} == set(STAGES)
    for row in rows:
        assert row.target_on_water is True
        assert row.range_xy_km >= cfg["phase3_task_stages"][row.stage]["range_min_km"]
        assert row.range_xy_km <= cfg["phase3_task_stages"][row.stage]["range_max_km"]
        assert 0.0 <= row.center_px[0] < cfg["phase3_task_stages"]["image_size"]
        assert 0.0 <= row.center_px[1] < cfg["phase3_task_stages"]["image_size"]
        target = np.asarray(row.target_state_world["pos_world"], dtype=float)
        aircraft = np.asarray(row.aircraft_state["pos_world"], dtype=float)
        assert np.isclose(np.linalg.norm(aircraft[:2] - target[:2]), row.range_xy_km)


def test_phase3_task_sampler_sequence_is_far_to_terminal_approach():
    cfg = _cfg()
    rows = sample_phase3_task_sequence(
        sequence_idx=0,
        target_cfg=cfg["phase3_target_motion"],
        stage_cfg=cfg["phase3_task_stages"],
        seed=123,
        frames=40,
    )
    stages = [row.stage for row in rows]
    assert stages == sorted(stages, key=lambda stage: STAGES.index(stage))
    assert stages[0] == "far"
    assert stages[-1] == "terminal"

    ranges = np.asarray([row.range_xy_km for row in rows], dtype=float)
    assert np.all(np.diff(ranges) <= 0.0)

    centers = np.asarray([row.center_px for row in rows], dtype=float)
    jumps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    assert float(jumps.max()) <= 32.0


def test_phase3_task_sampler_summary_accepts_balanced_smoke():
    cfg = _cfg()
    rows = []
    for seq_idx in range(3):
        rows.extend(
            sample_phase3_task_sequence(
                sequence_idx=seq_idx,
                target_cfg=cfg["phase3_target_motion"],
                stage_cfg=cfg["phase3_task_stages"],
                seed=200 + seq_idx,
                frames=6,
            )
        )
    summary = summarize_phase3_task_frames(rows, cfg["phase3_task_stages"])
    assert summary["accepted"] is True
    for stage in STAGES:
        assert summary["stage_counts"][stage] > 0
