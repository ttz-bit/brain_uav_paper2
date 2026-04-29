from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="data/rendered/paper2_task_v1.0.0_smoke")
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
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
    ranges = []
    offcenter = []

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
            meta = dict(row.get("meta", {}))
            if not bool(meta.get("target_on_water", False)):
                not_water += 1
            r = float(meta.get("range_xy_km", -1.0))
            ranges.append(r)
            if stage == "far" and not (1500.0 <= r <= 2000.0):
                invalid_stage += 1
            elif stage == "mid" and not (600.0 <= r <= 1500.0):
                invalid_stage += 1
            elif stage == "terminal" and not (50.0 <= r <= 600.0):
                invalid_stage += 1
            offcenter.append(float(np.hypot(cx - 0.5 * w, cy - 0.5 * h)))

    range_arr = np.asarray(ranges, dtype=float)
    off_arr = np.asarray(offcenter, dtype=float)
    report = {
        "dataset_root": str(root),
        "total_frames": int(total),
        "split_counts": split_counts,
        "stage_counts": stage_counts,
        "range_xy_min_km": float(range_arr.min()) if range_arr.size else 0.0,
        "range_xy_mean_km": float(range_arr.mean()) if range_arr.size else 0.0,
        "range_xy_max_km": float(range_arr.max()) if range_arr.size else 0.0,
        "offcenter_px_mean": float(off_arr.mean()) if off_arr.size else 0.0,
        "offcenter_px_max": float(off_arr.max()) if off_arr.size else 0.0,
        "center_px_out_of_image": int(center_out),
        "bbox_bad": int(bbox_bad),
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
