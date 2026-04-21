REQUIRED_KEYS = [
    "image_path",
    "dataset_name",
    "task_name",
    "sequence_id",
    "frame_id",
    "orig_image_path",
    "orig_image_size",
    "crop_path",
    "crop_size",
    "center_px",
    "bbox_xywh",
    "visible",
    "occluded",
    "truncated",
    "target_id",
    "category_name",
    "category_id",
    "crop_center_world",
    "gsd",
    "world_unit",
    "split",
    "source_track",
    "meta",
]


def validate_record(record: dict) -> None:
    for key in REQUIRED_KEYS:
        if key not in record:
            raise KeyError(f"Missing required key: {key}")