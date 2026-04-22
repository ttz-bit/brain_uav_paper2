import json
from pathlib import Path

import cv2
import numpy as np

from paper2.datasets.seadronessee_dataset import build_seadronessee_dataset


def _make_record(crop_rel: str, sequence_id: str, frame_id: str) -> dict:
    return {
        "image_path": crop_rel,
        "dataset_name": "SeaDronesSee",
        "task_name": "single_target_localization",
        "sequence_id": sequence_id,
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
        "source_track": f"seadronessee_sot/{sequence_id}",
        "meta": {"mot_source_relpath": f"images/train/{frame_id}.png"},
    }


def test_seadronessee_dataset_reader_loads_manifest_and_images(tmp_path: Path):
    project_root = tmp_path
    proc_root = project_root / "data" / "processed" / "seadronessee"
    (proc_root / "manifests").mkdir(parents=True, exist_ok=True)
    crops_dir = proc_root / "crops" / "train"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img0 = np.full((128, 128, 3), 120, dtype=np.uint8)
    img1 = np.full((128, 128, 3), 180, dtype=np.uint8)
    p0 = crops_dir / "seq_0001_frame_000001.png"
    p1 = crops_dir / "seq_0001_frame_000002.png"
    cv2.imwrite(str(p0), img0)
    cv2.imwrite(str(p1), img1)

    rec0 = _make_record("data/processed/seadronessee/crops/train/seq_0001_frame_000001.png", "1", "000001")
    rec1 = _make_record("data/processed/seadronessee/crops/train/seq_0001_frame_000002.png", "1", "000002")
    manifest = proc_root / "manifests" / "records_train.jsonl"
    manifest.write_text(
        json.dumps(rec0, ensure_ascii=False) + "\n" + json.dumps(rec1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    ds = build_seadronessee_dataset(
        root=proc_root,
        split="train",
        project_root=project_root,
    )
    assert len(ds) == 2
    s0 = ds[0]
    assert s0.image.shape == (128, 128, 3)
    assert s0.sequence_id == "1"
    assert s0.frame_id == "000001"
    assert s0.valid is True
    assert float(s0.bbox_xywh[2]) == 30.0
    assert float(s0.target_center[0]) == 320.0

