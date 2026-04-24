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
        "center_px_crop": [64.0, 64.0],
        "bbox_xywh_crop": [49.0, 44.0, 30.0, 40.0],
        "crop_origin_xy": [256, 176],
        "crop_box_xyxy": [256, 176, 384, 304],
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


def test_eval_public_vision_runs_with_saved_weights(tmp_path: Path):
    project_root = tmp_path
    proc_root = project_root / "data" / "processed" / "seadronessee"
    (proc_root / "manifests").mkdir(parents=True, exist_ok=True)
    crops_train = proc_root / "crops" / "train"
    crops_train.mkdir(parents=True, exist_ok=True)
    crops_val = proc_root / "crops" / "val"
    crops_val.mkdir(parents=True, exist_ok=True)

    train_rows = []
    val_rows = []
    for i in range(16):
        frame = f"{i+1:06d}"
        rel_train = f"data/processed/seadronessee/crops/train/seq_0001_frame_{frame}.png"
        rel_val = f"data/processed/seadronessee/crops/val/seq_0001_frame_{frame}.png"
        rec_train = _make_record(rel_train, frame)
        rec_train["split"] = "train"
        rec_val = _make_record(rel_val, frame)
        rec_val["split"] = "val"
        train_rows.append(rec_train)
        val_rows.append(rec_val)
        img = np.full((128, 128, 3), 20 + i * 4, dtype=np.uint8)
        cv2.imwrite(str(crops_train / f"seq_0001_frame_{frame}.png"), img)
        cv2.imwrite(str(crops_val / f"seq_0001_frame_{frame}.png"), img)

    (proc_root / "manifests" / "records_train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train_rows) + "\n",
        encoding="utf-8",
    )
    (proc_root / "manifests" / "records_val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows) + "\n",
        encoding="utf-8",
    )

    train_out = project_root / "outputs" / "train" / "public_vision" / "unit_train"
    train_script = Path(__file__).resolve().parents[1] / "scripts" / "train_public_vision.py"
    train_cmd = [
        sys.executable,
        str(train_script),
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
        "--learning-rate",
        "0.001",
        "--out-dir",
        str(train_out),
    ]
    train_proc = subprocess.run(train_cmd, capture_output=True, text=True)
    assert train_proc.returncode == 0, train_proc.stderr + "\n" + train_proc.stdout
    assert (train_out / "linear_weights.npy").exists()
    assert (train_out / "feature_norm_stats.npz").exists()

    eval_out = project_root / "outputs" / "eval" / "public_vision" / "unit_eval"
    eval_script = Path(__file__).resolve().parents[1] / "scripts" / "eval_public_vision.py"
    eval_cmd = [
        sys.executable,
        str(eval_script),
        "--root",
        str(proc_root),
        "--split",
        "val",
        "--project-root",
        str(project_root),
        "--max-samples",
        "16",
        "--weights-path",
        str(train_out / "linear_weights.npy"),
        "--norm-stats-path",
        str(train_out / "feature_norm_stats.npz"),
        "--out-dir",
        str(eval_out),
    ]
    eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True)
    assert eval_proc.returncode == 0, eval_proc.stderr + "\n" + eval_proc.stdout

    report_path = eval_out / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task"] == "eval_public_vision"
    assert report["purpose"] == "evaluation"
    assert report["split"] == "val"
    assert "mse" in report and report["mse"] >= 0.0
    assert "mae" in report and report["mae"] >= 0.0
