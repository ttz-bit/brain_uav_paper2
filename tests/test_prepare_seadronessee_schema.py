from pathlib import Path

from scripts.prepare_seadronessee import _build_unified_record
from paper2.datasets.unified_schema import validate_record


def test_build_unified_record_matches_schema_required_keys():
    rec = _build_unified_record(
        split="train",
        sequence_id="1",
        frame_id="000001",
        img_path=Path(r"D:\datasets\SeaDronesSee\dummy.jpg"),
        raw_rel="images/train/000001.jpg",
        crop_rel="data/processed/seadronessee/crops/train/seq_0001_frame_000001.png",
        crop_size=128,
        bbox_xywh=(10, 20, 30, 40),
        center_px=(25.0, 40.0),
        center_px_crop=(64.0, 64.0),
        bbox_xywh_crop=(49.0, 44.0, 30.0, 40.0),
        crop_origin_xy=(-39, -24),
        crop_box_xyxy=(-39, -24, 89, 104),
        img_w=1920,
        img_h=1080,
    )
    validate_record(rec)
