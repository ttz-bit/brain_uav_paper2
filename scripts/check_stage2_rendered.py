from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_smoke_v0")
    p.add_argument("--max-low-visibility-ratio", type=float, default=0.10)
    p.add_argument("--min-positive-visibility", type=float, default=0.35)
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    labels_dir = root / "labels"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    split_asset_sets: dict[str, dict[str, set[str]]] = {}
    total = 0
    center_in_bounds = 0
    bbox_valid = 0
    low_vis = 0

    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}.jsonl"
        if not path.exists():
            errors.append(f"missing_labels:{path}")
            continue
        rows = _read_jsonl(path)
        split_asset_sets[split] = {"background": set(), "target": set(), "distractor": set()}
        for r in rows:
            total += 1
            image_path = Path(r["image_path"])
            abs_image = image_path if image_path.is_absolute() else (Path.cwd() / image_path)
            if not abs_image.exists():
                errors.append(f"missing_image:{abs_image}")
                continue
            img = cv2.imread(str(abs_image), cv2.IMREAD_COLOR)
            if img is None:
                errors.append(f"bad_image:{abs_image}")
                continue
            h, w = img.shape[:2]
            cx, cy = float(r["target_center_px"][0]), float(r["target_center_px"][1])
            if 0.0 <= cx < float(w) and 0.0 <= cy < float(h):
                center_in_bounds += 1
            bx, by, bw, bh = [float(x) for x in r["bbox_xywh"]]
            if bw > 0 and bh > 0 and bx + bw > 0 and by + bh > 0 and bx < w and by < h:
                bbox_valid += 1

            vis = float(r.get("visibility", 0.0))
            if vis < float(args.min_positive_visibility):
                low_vis += 1

            split_asset_sets[split]["background"].add(str(r["background_asset_id"]))
            split_asset_sets[split]["target"].add(str(r["target_asset_id"]))
            for d in r.get("distractor_asset_ids", []):
                split_asset_sets[split]["distractor"].add(str(d))

    leakage = {"background": 0, "target": 0, "distractor": 0}
    for key in leakage.keys():
        train = split_asset_sets.get("train", {}).get(key, set())
        val = split_asset_sets.get("val", {}).get(key, set())
        test = split_asset_sets.get("test", {}).get(key, set())
        leakage[key] = len(train.intersection(val)) + len(train.intersection(test)) + len(val.intersection(test))

    center_rate = float(center_in_bounds / max(1, total))
    bbox_rate = float(bbox_valid / max(1, total))
    low_vis_ratio = float(low_vis / max(1, total))

    report = {
        "dataset_root": str(root),
        "total_frames": total,
        "center_in_bounds_rate": center_rate,
        "bbox_valid_rate": bbox_rate,
        "low_visibility_ratio": low_vis_ratio,
        "split_asset_leakage": leakage,
        "pass": (
            len(errors) == 0
            and center_rate >= 1.0
            and bbox_rate >= 1.0
            and low_vis_ratio <= float(args.max_low_visibility_ratio)
            and all(v == 0 for v in leakage.values())
        ),
        "errors": errors[:100],
    }
    out_path = reports_dir / "qc_summary.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
