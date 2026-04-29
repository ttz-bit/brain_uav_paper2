from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


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
