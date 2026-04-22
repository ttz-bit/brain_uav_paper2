import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


def _make_record(crop_rel: str, frame_id: str) -> dict:
    return {
        "image_path": crop_rel,
        "dataset_name": "SeaDronesSee",
        "task_name": "single_target_localization",
        "sequence_id": "1",
        "frame_id": frame_id,
        "orig_image_path": "D:/datasets/SeaDronesSee/dummy.jpg",
        "orig_image_size": [1920, 1080],
        "crop_path": crop_rel,
        "crop_size": [128, 128],
        "center_px": [320.0, 240.0],
        "bbox_xywh": [10, 20, 30, 40],
        "visible": 1,
        "occluded": 0,
        "truncated": 0,
        "target_id": "main_target",
        "category_name": "vessel",
        "category_id": 1,
        "crop_center_world": None,
        "gsd": None,
        "world_unit": None,
        "split": "train",
        "source_track": "seadronessee_sot/1",
        "meta": {"mot_source_relpath": f"images/train/{frame_id}.png"},
    }


def test_smoke_train_public_vision_runs_end_to_end(tmp_path: Path):
    project_root = tmp_path
    proc_root = project_root / "data" / "processed" / "seadronessee"
    (proc_root / "manifests").mkdir(parents=True, exist_ok=True)
    crops_dir = proc_root / "crops" / "train"
    crops_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(16):
        frame = f"{i+1:06d}"
        rel = f"data/processed/seadronessee/crops/train/seq_0001_frame_{frame}.png"
        rows.append(_make_record(rel, frame))
        img = np.full((128, 128, 3), 20 + i * 4, dtype=np.uint8)
        cv2.imwrite(str(crops_dir / f"seq_0001_frame_{frame}.png"), img)

    manifest = proc_root / "manifests" / "records_train.jsonl"
    manifest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    out_dir = project_root / "outputs" / "train" / "smoke_public_vision"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_train_public_vision.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--root",
        str(proc_root),
        "--split",
        "train",
        "--project-root",
        str(project_root),
        "--max-samples",
        "16",
        "--batch-size",
        "8",
        "--steps",
        "5",
        "--out-dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr + "\n" + proc.stdout

    report_path = out_dir / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["num_samples"] == 16
    assert report["steps"] == 5
    assert (out_dir / "linear_weights.npy").exists()
    assert (out_dir / "visuals").exists()

