from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper2.common.config import load_yaml


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _crop_origin_from_meta(meta: dict) -> list[float] | None:
    for key in ("crop_origin_bg_px", "crop_origin_xy", "crop_bg_xy", "crop_top_left"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="data/rendered/paper2_task_v1.0.0_smoke")
    parser.add_argument("--config", type=str, default="configs/env.yaml")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    stage_cfg = load_yaml(args.config)["phase3_task_stages"]
    labels_dir = root / "labels"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    total = 0
    stage_counts = {"far": 0, "mid": 0, "terminal": 0}
    split_counts = {}
    center_out = 0
    bbox_bad = 0
    not_water = 0
    invalid_stage = 0
    center_bbox_delta_violations = 0
    target_water_ratio_violations = 0
    distractor_count_shortfalls = 0
    distractor_count_over_max = 0
    distractor_water_ratio_violations = 0
    sequence_background_violations = 0
    sequence_target_violations = 0
    sequence_stage_order_violations = 0
    sequence_range_monotonic_violations = 0
    sequence_motion_mode_violations = 0
    sequence_distractor_count_violations = 0
    target_motion_step_violations = 0
    crop_origin_missing = 0
    water_mask_missing = 0
    water_mask_crop_violations = 0
    water_mask_target_center_violations = 0
    ranges = []
    offcenter = []
    center_bbox_deltas = []
    target_water_ratios = []
    distractor_counts = []
    distractor_water_ratios = []
    target_motion_steps = []
    sequence_backgrounds: dict[str, str] = {}
    sequence_targets: dict[str, str] = {}
    rows_by_sequence: dict[str, list[dict]] = {}
    motion_mode_sequence_counts: dict[str, int] = {}
    water_mask_cache: dict[str, np.ndarray | None] = {}

    for split in ("train", "val", "test"):
        label_path = labels_dir / f"{split}.jsonl"
        if not label_path.exists():
            errors.append(f"missing_labels:{label_path}")
            continue
        rows = _read_jsonl(label_path)
        split_counts[split] = len(rows)
        for row in rows:
            total += 1
            stage = str(row["stage"])
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            img_path = Path(row["image_path"])
            abs_img = img_path if img_path.is_absolute() else Path.cwd() / img_path
            img = cv2.imread(str(abs_img), cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"bad_image:{abs_img}")
                continue
            h, w = img.shape[:2]
            cx, cy = [float(v) for v in row["target_center_px"]]
            if not (0.0 <= cx < float(w) and 0.0 <= cy < float(h)):
                center_out += 1
            bx, by, bw, bh = [float(v) for v in row["bbox_xywh"]]
            if bw <= 0.0 or bh <= 0.0 or bx >= w or by >= h or bx + bw <= 0.0 or by + bh <= 0.0:
                bbox_bad += 1
            bbox_cx = bx + 0.5 * bw
            bbox_cy = by + 0.5 * bh
            center_bbox_delta = float(np.hypot(cx - bbox_cx, cy - bbox_cy))
            center_bbox_deltas.append(center_bbox_delta)
            meta = dict(row.get("meta", {}))
            asset_mode = str(meta.get("asset_mode", "")).lower()
            water_mask_path_raw = meta.get("water_mask_path")
            crop_origin = _crop_origin_from_meta(meta)
            if asset_mode == "real":
                if not water_mask_path_raw:
                    water_mask_missing += 1
                if crop_origin is None:
                    crop_origin_missing += 1
            if water_mask_path_raw:
                if not isinstance(crop_origin, (list, tuple)) or len(crop_origin) < 2:
                    crop_origin_missing += 1
                else:
                    mask_path = Path(str(water_mask_path_raw))
                    if not mask_path.is_absolute():
                        mask_path = Path.cwd() / mask_path
                    mask_key = str(mask_path)
                    if mask_key not in water_mask_cache:
                        water_mask_cache[mask_key] = cv2.imread(mask_key, cv2.IMREAD_GRAYSCALE)
                    mask = water_mask_cache[mask_key]
                    if mask is None:
                        water_mask_crop_violations += 1
                    else:
                        x1 = int(round(float(crop_origin[0])))
                        y1 = int(round(float(crop_origin[1])))
                        if x1 < 0 or y1 < 0 or x1 + w > mask.shape[1] or y1 + h > mask.shape[0]:
                            water_mask_crop_violations += 1
                        else:
                            mx = min(max(int(round(x1 + cx)), 0), mask.shape[1] - 1)
                            my = min(max(int(round(y1 + cy)), 0), mask.shape[0] - 1)
                            if mask[my, mx] <= 0:
                                water_mask_target_center_violations += 1
            seq_id = str(row["sequence_id"])
            rows_by_sequence.setdefault(seq_id, []).append(row)
            bg_path = str(meta.get("background_path", row.get("background_asset_id", "")))
            target_path = str(meta.get("target_asset_path", row.get("target_asset_id", "")))
            prev_bg = sequence_backgrounds.setdefault(seq_id, bg_path)
            if bg_path != prev_bg:
                sequence_background_violations += 1
            prev_target = sequence_targets.setdefault(seq_id, target_path)
            if target_path != prev_target:
                sequence_target_violations += 1
            scale_px = float(meta.get("scale_px", row.get("scale_px", 0.0)))
            max_center_bbox_delta = max(8.0, 1.5 * scale_px)
            if center_bbox_delta > max_center_bbox_delta:
                center_bbox_delta_violations += 1
            target_water_ratio = float(meta.get("target_water_ratio", 1.0))
            target_water_ratios.append(target_water_ratio)
            if target_water_ratio < 0.98:
                target_water_ratio_violations += 1
            distractor_ids = list(row.get("distractor_asset_ids", []))
            distractor_count = int(meta.get("distractor_count", len(distractor_ids)))
            distractor_count_requested = int(meta.get("distractor_count_requested", 0))
            distractor_counts.append(distractor_count)
            if distractor_count < distractor_count_requested:
                distractor_count_shortfalls += 1
            if distractor_count > 3 or distractor_count_requested > 3:
                distractor_count_over_max += 1
            for ratio_raw in list(meta.get("distractor_water_ratios", [])):
                ratio = float(ratio_raw)
                distractor_water_ratios.append(ratio)
                if ratio < 0.98:
                    distractor_water_ratio_violations += 1
            if not bool(meta.get("target_on_water", False)):
                not_water += 1
            r = float(meta.get("range_xy_km", -1.0))
            ranges.append(r)
            expected_stage = stage_cfg.get(stage)
            if not expected_stage:
                invalid_stage += 1
            elif not (float(expected_stage["range_min_km"]) <= r <= float(expected_stage["range_max_km"])):
                invalid_stage += 1
            offcenter.append(float(np.hypot(cx - 0.5 * w, cy - 0.5 * h)))

    stage_order = {"far": 0, "mid": 1, "terminal": 2}
    for seq_id, seq_rows in rows_by_sequence.items():
        seq_rows = sorted(seq_rows, key=lambda row: int(row["frame_id"]))
        seq_stages = [str(row["stage"]) for row in seq_rows]
        seq_stage_indices = [stage_order.get(stage, -1) for stage in seq_stages]
        if any(idx < 0 for idx in seq_stage_indices) or any(
            seq_stage_indices[i] > seq_stage_indices[i + 1] for i in range(len(seq_stage_indices) - 1)
        ):
            sequence_stage_order_violations += 1

        seq_ranges = [float(dict(row.get("meta", {})).get("range_xy_km", -1.0)) for row in seq_rows]
        if any(seq_ranges[i] + 1e-6 < seq_ranges[i + 1] for i in range(len(seq_ranges) - 1)):
            sequence_range_monotonic_violations += 1

        seq_modes = {str(row.get("motion_mode", "")) for row in seq_rows}
        if len(seq_modes) != 1:
            sequence_motion_mode_violations += 1
        else:
            mode = next(iter(seq_modes))
            motion_mode_sequence_counts[mode] = motion_mode_sequence_counts.get(mode, 0) + 1

        seq_distractor_counts = [int(dict(row.get("meta", {})).get("distractor_count", 0)) for row in seq_rows]
        if len(set(seq_distractor_counts)) > 1:
            sequence_distractor_count_violations += 1

        prev_xy = None
        for row in seq_rows:
            target_state = dict(dict(row.get("meta", {})).get("target_state_world", {}))
            xy = np.array([float(target_state.get("x", 0.0)), float(target_state.get("y", 0.0))], dtype=float)
            if prev_xy is not None:
                step = float(np.linalg.norm(xy - prev_xy))
                target_motion_steps.append(step)
                if step > 0.1:
                    target_motion_step_violations += 1
            prev_xy = xy

    range_arr = np.asarray(ranges, dtype=float)
    off_arr = np.asarray(offcenter, dtype=float)
    center_bbox_delta_arr = np.asarray(center_bbox_deltas, dtype=float)
    target_water_ratio_arr = np.asarray(target_water_ratios, dtype=float)
    distractor_count_arr = np.asarray(distractor_counts, dtype=float)
    distractor_water_ratio_arr = np.asarray(distractor_water_ratios, dtype=float)
    target_motion_step_arr = np.asarray(target_motion_steps, dtype=float)
    expected_motion_modes = {"cv", "turn", "piecewise", "evasive"}
    motion_mode_coverage_ok = expected_motion_modes.issubset(set(motion_mode_sequence_counts))
    report = {
        "dataset_root": str(root),
        "total_frames": int(total),
        "split_counts": split_counts,
        "stage_counts": stage_counts,
        "sequence_count": int(len(rows_by_sequence)),
        "motion_mode_sequence_counts": motion_mode_sequence_counts,
        "motion_mode_coverage_ok": bool(motion_mode_coverage_ok),
        "range_xy_min_km": float(range_arr.min()) if range_arr.size else 0.0,
        "range_xy_mean_km": float(range_arr.mean()) if range_arr.size else 0.0,
        "range_xy_max_km": float(range_arr.max()) if range_arr.size else 0.0,
        "target_motion_step_km_mean": float(target_motion_step_arr.mean()) if target_motion_step_arr.size else 0.0,
        "target_motion_step_km_max": float(target_motion_step_arr.max()) if target_motion_step_arr.size else 0.0,
        "offcenter_px_mean": float(off_arr.mean()) if off_arr.size else 0.0,
        "offcenter_px_max": float(off_arr.max()) if off_arr.size else 0.0,
        "center_bbox_delta_px_mean": float(center_bbox_delta_arr.mean()) if center_bbox_delta_arr.size else 0.0,
        "center_bbox_delta_px_max": float(center_bbox_delta_arr.max()) if center_bbox_delta_arr.size else 0.0,
        "target_water_ratio_min": float(target_water_ratio_arr.min()) if target_water_ratio_arr.size else 0.0,
        "target_water_ratio_mean": float(target_water_ratio_arr.mean()) if target_water_ratio_arr.size else 0.0,
        "distractor_count_mean": float(distractor_count_arr.mean()) if distractor_count_arr.size else 0.0,
        "distractor_count_min": int(distractor_count_arr.min()) if distractor_count_arr.size else 0,
        "distractor_count_max": int(distractor_count_arr.max()) if distractor_count_arr.size else 0,
        "distractor_water_ratio_min": float(distractor_water_ratio_arr.min()) if distractor_water_ratio_arr.size else 0.0,
        "distractor_water_ratio_mean": float(distractor_water_ratio_arr.mean()) if distractor_water_ratio_arr.size else 0.0,
        "center_px_out_of_image": int(center_out),
        "bbox_bad": int(bbox_bad),
        "center_bbox_delta_violations": int(center_bbox_delta_violations),
        "target_water_ratio_violations": int(target_water_ratio_violations),
        "distractor_count_shortfalls": int(distractor_count_shortfalls),
        "distractor_count_over_max": int(distractor_count_over_max),
        "distractor_water_ratio_violations": int(distractor_water_ratio_violations),
        "sequence_background_violations": int(sequence_background_violations),
        "sequence_target_violations": int(sequence_target_violations),
        "sequence_stage_order_violations": int(sequence_stage_order_violations),
        "sequence_range_monotonic_violations": int(sequence_range_monotonic_violations),
        "sequence_motion_mode_violations": int(sequence_motion_mode_violations),
        "sequence_distractor_count_violations": int(sequence_distractor_count_violations),
        "target_motion_step_violations": int(target_motion_step_violations),
        "crop_origin_missing": int(crop_origin_missing),
        "water_mask_missing": int(water_mask_missing),
        "water_mask_crop_violations": int(water_mask_crop_violations),
        "water_mask_target_center_violations": int(water_mask_target_center_violations),
        "target_not_water": int(not_water),
        "invalid_stage_ranges": int(invalid_stage),
        "errors": errors[:100],
        "pass": bool(
            total > 0
            and not errors
            and all(split_counts.get(split, 0) > 0 for split in ("train", "val", "test"))
            and all(stage_counts.get(stage, 0) > 0 for stage in ("far", "mid", "terminal"))
            and center_out == 0
            and bbox_bad == 0
            and center_bbox_delta_violations == 0
            and target_water_ratio_violations == 0
            and distractor_count_shortfalls == 0
            and distractor_count_over_max == 0
            and distractor_water_ratio_violations == 0
            and sequence_background_violations == 0
            and sequence_target_violations == 0
            and sequence_stage_order_violations == 0
            and sequence_range_monotonic_violations == 0
            and sequence_motion_mode_violations == 0
            and sequence_distractor_count_violations == 0
            and target_motion_step_violations == 0
            and crop_origin_missing == 0
            and water_mask_missing == 0
            and water_mask_crop_violations == 0
            and water_mask_target_center_violations == 0
            and motion_mode_coverage_ok
            and not_water == 0
            and invalid_stage == 0
        ),
    }
    out_path = reports_dir / "phase3_task_dataset_qc.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
