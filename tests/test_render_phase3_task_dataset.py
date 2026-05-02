from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.render_phase3_task_dataset import _collect_targets, _target_template_allowed


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
        "3",
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
    assert report["total_frames"] == 12


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
            "3",
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
