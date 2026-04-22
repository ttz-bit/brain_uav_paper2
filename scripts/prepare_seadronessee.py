from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from paper2.datasets.unified_schema import validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=str, default=r"D:\datasets\SeaDronesSee", help="SeaDronesSee raw root")
    parser.add_argument(
        "--out-root",
        type=str,
        default=r"D:\Projects\brain_uav_paper2\data\processed\seadronessee",
        help="Processed output root",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--crop-size", type=int, default=128, help="Square crop size")
    parser.add_argument("--max-sequences", type=int, default=None, help="Process first N sequences only")
    parser.add_argument("--qc-per-split", type=int, default=30, help="Save at most N QC samples per split")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite crops/manifests")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_lines(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sorted_seq_items(seq_map: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(seq_map.items(), key=lambda kv: int(Path(kv[0]).stem))


def parse_annotation_line(line: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Bad annotation line: {line}")
    return tuple(int(float(v)) for v in parts)  # type: ignore[return-value]


def resolve_raw_image_path(raw_root: Path, raw_rel: str) -> Path:
    rel = Path(raw_rel.replace("\\", "/"))
    parts = list(rel.parts)
    if "images" not in parts:
        raise ValueError(f"'images' not found in raw relpath: {raw_rel}")
    idx = parts.index("images")
    if idx + 2 >= len(parts):
        raise ValueError(f"Unexpected raw relpath format: {raw_rel}")

    split = parts[idx + 1]
    stem = Path(parts[idx + 2]).stem
    jpg_path = raw_root / "extracted" / "mot" / "Compressed" / split / f"{stem}.jpg"
    if jpg_path.exists():
        return jpg_path
    png_path = raw_root / "extracted" / "mot" / "Compressed" / split / f"{stem}.png"
    if png_path.exists():
        return png_path
    raise FileNotFoundError(f"Cannot resolve local image for: {raw_rel}")


def make_center_crop(img: np.ndarray, center_x: float, center_y: float, crop_size: int) -> np.ndarray:
    half = crop_size // 2
    h, w = img.shape[:2]
    x1 = int(round(center_x)) - half
    y1 = int(round(center_y)) - half
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)
    if pad_left or pad_top or pad_right or pad_bottom:
        img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        x1 += pad_left
        y1 += pad_top
        x2 += pad_left
        y2 += pad_top

    crop = img[y1:y2, x1:x2]
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        crop = cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    return crop


def draw_bbox_overlay(img: np.ndarray, bbox_xywh: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox_xywh
    vis = img.copy()
    cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cx = x + w / 2.0
    cy = y + h / 2.0
    cv2.circle(vis, (int(round(cx)), int(round(cy))), 5, (0, 0, 255), -1)
    return vis


def _build_unified_record(
    split: str,
    sequence_id: str,
    frame_id: str,
    img_path: Path,
    raw_rel: str,
    crop_rel: str,
    crop_size: int,
    bbox_xywh: tuple[int, int, int, int],
    center_px: tuple[float, float],
    img_w: int,
    img_h: int,
) -> dict[str, Any]:
    x, y, w, h = bbox_xywh
    cx, cy = center_px
    record = {
        "image_path": crop_rel,
        "dataset_name": "SeaDronesSee",
        "task_name": "single_target_localization",
        "sequence_id": str(sequence_id),
        "frame_id": str(frame_id),
        "orig_image_path": str(img_path).replace("\\", "/"),
        "orig_image_size": [int(img_w), int(img_h)],
        "crop_path": crop_rel,
        "crop_size": [int(crop_size), int(crop_size)],
        "center_px": [float(cx), float(cy)],
        "bbox_xywh": [int(x), int(y), int(w), int(h)],
        "visible": 1,
        "occluded": 0,
        "truncated": 0,
        "target_id": "main_target",
        "category_name": "vessel",
        "category_id": 1,
        "crop_center_world": None,
        "gsd": None,
        "world_unit": None,
        "split": split,
        "source_track": f"seadronessee_sot/{split}/{sequence_id}",
        "meta": {
            "mot_source_relpath": raw_rel,
        },
    }
    return record


def process_split(
    split: str,
    raw_root: Path,
    out_root: Path,
    crop_size: int,
    max_sequences: int | None,
    qc_per_split: int,
    overwrite: bool,
) -> tuple[dict[str, Any], set[str]]:
    json_path = raw_root / "sot" / f"SeaDronesSee_{split}.json"
    ann_dir = raw_root / "sot" / ("test_annotations_first_frame" if split == "test" else f"{split}_annotations")

    manifests_dir = out_root / "manifests"
    crops_dir = out_root / "crops" / split
    qc_dir = out_root / "qc" / split
    stats_dir = out_root / "stats"
    ensure_dir(manifests_dir)
    ensure_dir(crops_dir)
    ensure_dir(qc_dir)
    ensure_dir(stats_dir)

    data = read_json(json_path)
    seq_ids = sorted(data.keys(), key=lambda x: int(x))
    if max_sequences is not None:
        seq_ids = seq_ids[:max_sequences]

    records: list[dict[str, Any]] = []
    qc_saved = 0
    mismatches: list[dict[str, Any]] = []
    source_tracks: set[str] = set()
    summary: dict[str, Any] = {
        "split": split,
        "num_sequences": 0,
        "num_records": 0,
        "missing_images": 0,
        "bad_annotations": 0,
        "invalid_bbox": 0,
        "schema_errors": 0,
        "boundary_crop_count": 0,
        "length_mismatch_sequences": 0,
        "length_mismatch_total_extra_frames": 0,
        "length_mismatch_total_extra_annotations": 0,
    }

    for seq_id in tqdm(seq_ids, desc=f"Processing {split}"):
        seq_map = data[seq_id]
        if not isinstance(seq_map, dict):
            raise TypeError(f"Expected dict for sequence {seq_id}, got {type(seq_map).__name__}")

        seq_items = sorted_seq_items(seq_map)
        ann_path = ann_dir / f"{seq_id}.txt"
        if not ann_path.exists():
            if split == "test":
                continue
            raise FileNotFoundError(f"Missing annotation file: {ann_path}")

        ann_lines = read_lines(ann_path)
        if len(seq_items) != len(ann_lines):
            mismatches.append(
                {
                    "split": split,
                    "sequence_id": str(seq_id),
                    "num_frames": len(seq_items),
                    "num_annotations": len(ann_lines),
                    "frames_minus_annotations": len(seq_items) - len(ann_lines),
                }
            )
            summary["length_mismatch_sequences"] += 1
            if len(seq_items) > len(ann_lines):
                summary["length_mismatch_total_extra_frames"] += len(seq_items) - len(ann_lines)
            else:
                summary["length_mismatch_total_extra_annotations"] += len(ann_lines) - len(seq_items)
            print(
                f"[WARN] Length mismatch split={split}, seq={seq_id}: "
                f"{len(seq_items)} frames vs {len(ann_lines)} annotations."
            )

        pair_count = min(len(seq_items), len(ann_lines))
        if pair_count == 0:
            continue
        summary["num_sequences"] += 1

        for i in range(pair_count):
            local_frame_name, raw_rel = seq_items[i]
            try:
                bbox_xywh = parse_annotation_line(ann_lines[i])
            except Exception:
                summary["bad_annotations"] += 1
                continue
            x, y, w, h = bbox_xywh
            if w <= 0 or h <= 0:
                summary["invalid_bbox"] += 1
                continue

            try:
                img_path = resolve_raw_image_path(raw_root, raw_rel)
            except FileNotFoundError:
                summary["missing_images"] += 1
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                summary["missing_images"] += 1
                continue
            img_h, img_w = img.shape[:2]
            center_x = x + w / 2.0
            center_y = y + h / 2.0

            half = crop_size // 2
            if (
                int(round(center_x)) - half < 0
                or int(round(center_y)) - half < 0
                or int(round(center_x)) + half > img_w
                or int(round(center_y)) + half > img_h
            ):
                summary["boundary_crop_count"] += 1

            crop = make_center_crop(img, center_x, center_y, crop_size)
            frame_id = Path(local_frame_name).stem
            crop_name = f"seq_{int(seq_id):04d}_frame_{frame_id}.png"
            crop_path = crops_dir / crop_name
            crop_rel = f"data/processed/seadronessee/crops/{split}/{crop_name}"
            if overwrite or (not crop_path.exists()):
                cv2.imwrite(str(crop_path), crop)

            record = _build_unified_record(
                split=split,
                sequence_id=str(seq_id),
                frame_id=frame_id,
                img_path=img_path,
                raw_rel=raw_rel,
                crop_rel=crop_rel,
                crop_size=crop_size,
                bbox_xywh=bbox_xywh,
                center_px=(center_x, center_y),
                img_w=img_w,
                img_h=img_h,
            )
            try:
                validate_record(record)
            except KeyError:
                summary["schema_errors"] += 1
                continue

            records.append(record)
            summary["num_records"] += 1
            source_tracks.add(str(record["source_track"]))

            if qc_saved < qc_per_split:
                vis = draw_bbox_overlay(img, bbox_xywh)
                qc_overlay_path = qc_dir / f"seq_{int(seq_id):04d}_frame_{frame_id}_overlay.jpg"
                qc_crop_path = qc_dir / f"seq_{int(seq_id):04d}_frame_{frame_id}_crop.jpg"
                cv2.imwrite(str(qc_overlay_path), vis)
                cv2.imwrite(str(qc_crop_path), crop)
                qc_saved += 1

    manifest_path = manifests_dir / f"records_{split}.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    mismatch_path = stats_dir / f"length_mismatches_{split}.json"
    with mismatch_path.open("w", encoding="utf-8") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=2)

    summary["length_mismatch_report"] = str(mismatch_path).replace("\\", "/")
    summary["manifest_path"] = str(manifest_path).replace("\\", "/")

    summary_path = stats_dir / f"summary_{split}.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[DONE] split={split}")
    print(f"  manifest: {manifest_path}")
    print(f"  summary : {summary_path}")
    print(f"  mismatch: {mismatch_path}")
    print(f"  records : {summary['num_records']}")
    print(f"  qc saved: {qc_saved}")
    return summary, source_tracks


def build_split_leakage_report(split_to_sequences: dict[str, set[str]]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    splits = sorted(split_to_sequences.keys())
    for i, sa in enumerate(splits):
        for sb in splits[i + 1 :]:
            overlap = sorted(split_to_sequences[sa].intersection(split_to_sequences[sb]))
            pairs.append(
                {
                    "split_a": sa,
                    "split_b": sb,
                    "overlap_count": len(overlap),
                    "overlap_sequence_ids_sample": overlap[:20],
                }
            )
    return {
        "all_clear": all(p["overlap_count"] == 0 for p in pairs),
        "pairs": pairs,
    }


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    stats_dir = out_root / "stats"
    ensure_dir(stats_dir)

    split_to_sequences: dict[str, set[str]] = {}
    split_summaries: dict[str, dict[str, Any]] = {}
    for split in args.splits:
        summary, seqs = process_split(
            split=split,
            raw_root=raw_root,
            out_root=out_root,
            crop_size=args.crop_size,
            max_sequences=args.max_sequences,
            qc_per_split=args.qc_per_split,
            overwrite=args.overwrite,
        )
        split_to_sequences[split] = seqs
        split_summaries[split] = summary

    leakage = build_split_leakage_report(split_to_sequences)
    leakage_path = stats_dir / "split_leakage_report.json"
    with leakage_path.open("w", encoding="utf-8") as f:
        json.dump(leakage, f, ensure_ascii=False, indent=2)
    print(f"[DONE] split leakage report: {leakage_path}")

    aggregate = {
        "splits": list(args.splits),
        "split_summaries": split_summaries,
        "split_leakage_report": str(leakage_path).replace("\\", "/"),
        "all_clear": leakage["all_clear"],
    }
    aggregate_path = stats_dir / "summary_all.json"
    with aggregate_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"[DONE] aggregate summary: {aggregate_path}")


if __name__ == "__main__":
    main()
