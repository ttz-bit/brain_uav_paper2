from pathlib import Path

import pytest

from paper2.datasets.unified_schema import validate_record
from scripts.prepare_seadronessee import _build_unified_record


def _base_record() -> dict:
    return _build_unified_record(
        split="train",
        sequence_id="1",
        frame_id="000001",
        img_path=Path(r"D:\datasets\SeaDronesSee\dummy.jpg"),
        raw_rel="images/train/000001.jpg",
        crop_rel="data/processed/seadronessee/crops/train/seq_0001_frame_000001.png",
        crop_size=128,
        bbox_xywh=(10, 20, 30, 40),
        center_px=(25.0, 40.0),
        img_w=1920,
        img_h=1080,
    )


def test_unified_schema_validation_accepts_valid_record():
    rec = _base_record()
    validate_record(rec)


def test_unified_schema_validation_rejects_invalid_split():
    rec = _base_record()
    rec["split"] = "dev"
    with pytest.raises(KeyError):
        validate_record(rec)


def test_unified_schema_validation_rejects_out_of_range_center():
    rec = _base_record()
    rec["center_px"] = [99999.0, 40.0]
    with pytest.raises(KeyError):
        validate_record(rec)

