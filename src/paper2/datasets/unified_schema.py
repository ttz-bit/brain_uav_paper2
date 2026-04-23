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
    "center_px_crop",
    "bbox_xywh_crop",
    "crop_origin_xy",
    "crop_box_xyxy",
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


def _fail(msg: str) -> None:
    raise KeyError(msg)


def _is_finite_number(x: object) -> bool:
    return isinstance(x, (int, float)) and x == x and x not in (float("inf"), float("-inf"))


def _require_non_empty_str(record: dict, key: str) -> str:
    val = record.get(key)
    if not isinstance(val, str) or not val.strip():
        _fail(f"Invalid non-empty string field: {key}")
    return val


def _require_pair(record: dict, key: str) -> tuple[float, float]:
    val = record.get(key)
    if not isinstance(val, list) or len(val) != 2:
        _fail(f"Invalid 2-element field: {key}")
    x, y = val
    if not _is_finite_number(x) or not _is_finite_number(y):
        _fail(f"Invalid numeric values in field: {key}")
    return float(x), float(y)


def validate_record(record: dict) -> None:
    for key in REQUIRED_KEYS:
        if key not in record:
            raise KeyError(f"Missing required key: {key}")

    _require_non_empty_str(record, "image_path")
    _require_non_empty_str(record, "dataset_name")
    _require_non_empty_str(record, "task_name")
    _require_non_empty_str(record, "sequence_id")
    _require_non_empty_str(record, "frame_id")
    _require_non_empty_str(record, "orig_image_path")
    _require_non_empty_str(record, "crop_path")
    _require_non_empty_str(record, "target_id")
    _require_non_empty_str(record, "category_name")
    _require_non_empty_str(record, "source_track")

    split = _require_non_empty_str(record, "split")
    if split not in {"train", "val", "test"}:
        _fail(f"Invalid split: {split}")

    ow, oh = _require_pair(record, "orig_image_size")
    cw, ch = _require_pair(record, "crop_size")
    cx, cy = _require_pair(record, "center_px")
    cx_crop, cy_crop = _require_pair(record, "center_px_crop")
    crop_ox, crop_oy = _require_pair(record, "crop_origin_xy")
    if ow <= 0 or oh <= 0:
        _fail("orig_image_size must be positive")
    if cw <= 0 or ch <= 0:
        _fail("crop_size must be positive")
    if cx < 0 or cx >= ow or cy < 0 or cy >= oh:
        _fail("center_px is out of orig_image_size range")
    if cx_crop < 0 or cx_crop >= cw or cy_crop < 0 or cy_crop >= ch:
        _fail("center_px_crop is out of crop_size range")
    if not _is_finite_number(crop_ox) or not _is_finite_number(crop_oy):
        _fail("crop_origin_xy must contain finite numbers")

    bbox = record.get("bbox_xywh")
    if not isinstance(bbox, list) or len(bbox) != 4:
        _fail("Invalid 4-element field: bbox_xywh")
    bx, by, bw, bh = bbox
    if not all(_is_finite_number(v) for v in (bx, by, bw, bh)):
        _fail("bbox_xywh must be finite numbers")
    if float(bw) <= 0 or float(bh) <= 0:
        _fail("bbox_xywh width/height must be positive")

    bbox_crop = record.get("bbox_xywh_crop")
    if not isinstance(bbox_crop, list) or len(bbox_crop) != 4:
        _fail("Invalid 4-element field: bbox_xywh_crop")
    cbx, cby, cbw, cbh = bbox_crop
    if not all(_is_finite_number(v) for v in (cbx, cby, cbw, cbh)):
        _fail("bbox_xywh_crop must be finite numbers")
    if float(cbw) <= 0 or float(cbh) <= 0:
        _fail("bbox_xywh_crop width/height must be positive")

    crop_box = record.get("crop_box_xyxy")
    if not isinstance(crop_box, list) or len(crop_box) != 4:
        _fail("Invalid 4-element field: crop_box_xyxy")
    x1, y1, x2, y2 = crop_box
    if not all(_is_finite_number(v) for v in (x1, y1, x2, y2)):
        _fail("crop_box_xyxy must be finite numbers")
    if float(x2) <= float(x1) or float(y2) <= float(y1):
        _fail("crop_box_xyxy must satisfy x2>x1 and y2>y1")
    if abs(float(crop_ox) - float(x1)) > 1e-3 or abs(float(crop_oy) - float(y1)) > 1e-3:
        _fail("crop_origin_xy must match crop_box_xyxy top-left")
    if abs((float(x2) - float(x1)) - float(cw)) > 1e-3 or abs((float(y2) - float(y1)) - float(ch)) > 1e-3:
        _fail("crop_box_xyxy size must match crop_size")

    for flag_key in ("visible", "occluded", "truncated"):
        flag_val = record.get(flag_key)
        if flag_val not in (0, 1):
            _fail(f"{flag_key} must be 0 or 1")

    category_id = record.get("category_id")
    if not isinstance(category_id, int) or category_id < 0:
        _fail("category_id must be non-negative int")

    crop_center_world = record.get("crop_center_world")
    if crop_center_world is not None:
        if not isinstance(crop_center_world, list) or len(crop_center_world) != 2:
            _fail("crop_center_world must be None or [x, y]")
        if not all(_is_finite_number(v) for v in crop_center_world):
            _fail("crop_center_world must contain finite numbers")

    gsd = record.get("gsd")
    if gsd is not None:
        if not _is_finite_number(gsd) or float(gsd) <= 0:
            _fail("gsd must be None or a positive finite number")

    world_unit = record.get("world_unit")
    if world_unit is not None and (not isinstance(world_unit, str) or not world_unit.strip()):
        _fail("world_unit must be None or non-empty string")

    meta = record.get("meta")
    if not isinstance(meta, dict):
        _fail("meta must be a dict")
