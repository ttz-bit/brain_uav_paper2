from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset
from scripts.render_phase3_task_dataset import (
    _DistractorTrack,
    _advance_distractor_track,
    _collect_targets,
    _point_on_water,
    _target_template_allowed,
)


def test_render_phase3_task_dataset_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "phase3_task_smoke"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    render_cmd = [
        sys.executable,
        str(root / "scripts" / "render_phase3_task_dataset.py"),
        "--out-root",
        str(out_root),
        "--sequences",
        "4",
        "--frames",
        "4",
    ]
    check_cmd = [
        sys.executable,
        str(root / "scripts" / "check_phase3_task_dataset.py"),
        "--dataset-root",
        str(out_root),
    ]
    subprocess.run(render_cmd, cwd=root, env=env, check=True)
    subprocess.run(check_cmd, cwd=root, env=env, check=True)
    report = json.loads((out_root / "reports" / "phase3_task_dataset_qc.json").read_text(encoding="utf-8"))
    assert report["pass"] is True
    assert report["total_frames"] == 16


def test_target_template_filter_rejects_non_topdown_views():
    assert _target_template_allowed(
        Path("target_boat_top_001.png"),
        allow_keywords=("top",),
        reject_keywords=("side", "oblique"),
    )
    assert not _target_template_allowed(
        Path("target_boat_side_001.png"),
        allow_keywords=("top",),
        reject_keywords=("side", "oblique"),
    )
    assert not _target_template_allowed(
        Path("target_boat_oblique_001.png"),
        allow_keywords=("top",),
        reject_keywords=("side", "oblique"),
    )


def test_collect_targets_ignores_backup_split_directories(tmp_path):
    assets_root = tmp_path / "assets"
    canonical = assets_root / "target_templates" / "alpha_png"
    backup = assets_root / "target_templates" / "_backup_old"
    for split in ("train", "val", "test"):
        (canonical / split).mkdir(parents=True)
        (canonical / split / f"target_boat_top_{split}_0001.png").write_bytes(b"png")
        (backup / split).mkdir(parents=True)
        (backup / split / f"target_boat_top_{split}_backup.png").write_bytes(b"png")

    out = _collect_targets(
        assets_root,
        allow_keywords=("top",),
        reject_keywords=("side",),
    )

    assert {split: len(paths) for split, paths in out.items()} == {"train": 1, "val": 1, "test": 1}
    for paths in out.values():
        assert all("_backup_old" not in str(path) for path in paths)


def test_render_phase3_task_dataset_qc_reports_sequence_consistency(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out_root = tmp_path / "phase3_task_smoke"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "render_phase3_task_dataset.py"),
            "--out-root",
            str(out_root),
            "--sequences",
            "4",
            "--frames",
            "4",
        ],
        cwd=root,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "check_phase3_task_dataset.py"),
            "--dataset-root",
            str(out_root),
        ],
        cwd=root,
        env=env,
        check=True,
    )
    report = json.loads((out_root / "reports" / "phase3_task_dataset_qc.json").read_text(encoding="utf-8"))
    assert report["sequence_background_violations"] == 0
    assert report["sequence_target_violations"] == 0
    assert report["target_water_ratio_violations"] == 0


def test_stage2_rendered_dataset_accepts_crop_origin_aliases(tmp_path):
    root = tmp_path / "dataset"
    image_dir = root / "images" / "train"
    label_dir = root / "labels"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    cv2.imwrite(str(image_dir / "sample.png"), image)

    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[1:5, 1:5] = 255
    mask_path = mask_dir / "water_mask.png"
    cv2.imwrite(str(mask_path), mask)

    row = {
        "image_path": "images/train/sample.png",
        "split": "train",
        "sequence_id": "train_0000",
        "frame_id": "0000",
        "stage": "far",
        "target_center_px": [2.0, 2.0],
        "bbox_xywh": [1.0, 1.0, 2.0, 2.0],
        "visibility": 1.0,
        "background_asset_id": "bg_1",
        "target_asset_id": "target_1",
        "distractor_asset_ids": [],
        "motion_mode": "cv",
        "land_overlap_ratio": 0.0,
        "shore_buffer_overlap_ratio": 0.0,
        "scale_px": 2.0,
        "angle_deg": 0.0,
        "obs_valid": True,
        "meta": {
            "asset_mode": "real",
            "water_mask_path": str(mask_path),
            "crop_top_left": [1, 1],
            "target_state_world": {"x": 0.0, "y": 0.0},
        },
    }
    (label_dir / "train.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    ds = build_stage2_rendered_dataset(
        root=root,
        split="train",
        project_root=root,
        load_water_mask=True,
    )
    sample = ds[0]
    assert sample.water_mask is not None
    assert sample.water_mask.shape == (4, 4)
    assert int(sample.water_mask[0, 0]) == 255
    assert int(sample.water_mask[-1, -1]) == 255


def test_distractor_track_advances_without_collision_or_land(tmp_path):
    mask = np.ones((24, 24), dtype=np.uint8) * 255
    mask[:2, :] = 0
    rng = np.random.default_rng(7)
    track = _DistractorTrack(
        asset_path="track_a.png",
        center_px=np.array([10.0, 12.0], dtype=float),
        heading=0.0,
        speed_px=5.0,
        motion_mode="cv",
        turn_rate=0.0,
        steps_to_switch=8,
        scale_px=4.0,
        radius_px=2.5,
        count_requested=1,
    )
    peer = _DistractorTrack(
        asset_path="track_b.png",
        center_px=np.array([22.0, 12.0], dtype=float),
        heading=0.0,
        speed_px=1.0,
        motion_mode="cv",
        turn_rate=0.0,
        steps_to_switch=8,
        scale_px=4.0,
        radius_px=2.5,
        count_requested=1,
    )

    _advance_distractor_track(
        track,
        rng,
        mask_u8=mask,
        target_center=(16.0, 18.0),
        target_clearance_px=6.0,
        other_tracks=[track, peer],
    )

    assert _point_on_water(mask, float(track.center_px[0]), float(track.center_px[1]))
    assert float(np.hypot(track.center_px[0] - peer.center_px[0], track.center_px[1] - peer.center_px[1])) >= (
        track.radius_px + peer.radius_px + 6.0
    )
    assert float(np.hypot(track.center_px[0] - 16.0, track.center_px[1] - 18.0)) >= 6.0
