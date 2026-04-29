from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper2.common.config import load_yaml
from paper2.render.phase3_task_sampler import Phase3TaskFrame, sample_phase3_task_sequence


STAGE_SCALE_PX = {
    "far": 10.0,
    "mid": 18.0,
    "terminal": 32.0,
}


def _split_for_sequence(seq_idx: int, total: int) -> str:
    if total >= 3:
        if seq_idx == total - 1:
            return "test"
        if seq_idx == total - 2:
            return "val"
    train_n = max(1, int(round(total * 0.70)))
    val_n = max(1, int(round(total * 0.15)))
    if seq_idx < train_n:
        return "train"
    if seq_idx < train_n + val_n:
        return "val"
    return "test"


def _make_ocean_background(image_size: int, rng: np.random.Generator) -> np.ndarray:
    base = np.zeros((image_size, image_size, 3), dtype=np.float32)
    y = np.linspace(0.0, 1.0, image_size, dtype=np.float32)[:, None]
    base[:, :, 0] = 115.0 + 25.0 * y
    base[:, :, 1] = 95.0 + 35.0 * y
    base[:, :, 2] = 40.0 + 15.0 * y
    noise = rng.normal(0.0, 6.0, size=base.shape).astype(np.float32)
    wave = 8.0 * np.sin(np.linspace(0.0, 10.0, image_size, dtype=np.float32))[None, :, None]
    img = np.clip(base + noise + wave, 0.0, 255.0).astype(np.uint8)
    return cv2.GaussianBlur(img, (3, 3), 0.0)


def _draw_target(canvas: np.ndarray, frame: Phase3TaskFrame, rng: np.random.Generator) -> tuple[list[float], float]:
    cx, cy = float(frame.center_px[0]), float(frame.center_px[1])
    scale = float(STAGE_SCALE_PX.get(frame.stage, 18.0))
    heading = float(frame.target_state_world["heading"])
    length = scale
    width = max(4.0, scale * 0.45)

    direction = np.array([np.cos(heading), -np.sin(heading)], dtype=float)
    side = np.array([-direction[1], direction[0]], dtype=float)
    nose = np.array([cx, cy], dtype=float) + direction * (0.55 * length)
    tail = np.array([cx, cy], dtype=float) - direction * (0.45 * length)
    p1 = nose
    p2 = tail + side * (0.5 * width)
    p3 = tail - side * (0.5 * width)
    pts = np.round(np.stack([p1, p2, p3], axis=0)).astype(np.int32)

    color = (
        int(rng.integers(185, 235)),
        int(rng.integers(185, 235)),
        int(rng.integers(210, 250)),
    )
    cv2.fillConvexPoly(canvas, pts, color)
    cv2.polylines(canvas, [pts], isClosed=True, color=(35, 45, 55), thickness=1, lineType=cv2.LINE_AA)

    x, y, w, h = cv2.boundingRect(pts)
    img_h, img_w = canvas.shape[:2]
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(img_w, int(x + w))
    y1 = min(img_h, int(y + h))
    bbox = [float(x0), float(y0), float(max(1, x1 - x0)), float(max(1, y1 - y0))]
    visibility = 1.0 if x0 < x1 and y0 < y1 else 0.0
    return bbox, visibility


def _label_row(frame: Phase3TaskFrame, image_path: Path, project_root: Path, split: str, seq_idx: int, bbox: list[float], visibility: float) -> dict:
    try:
        rel_image = image_path.relative_to(project_root).as_posix()
    except ValueError:
        rel_image = str(image_path)
    target = frame.target_state_world
    row = {
        "image_path": rel_image,
        "split": split,
        "sequence_id": frame.sequence_id,
        "frame_id": f"{int(frame.frame_id):04d}",
        "stage": frame.stage,
        "observation_source": f"phase3_{frame.stage}",
        "gsd_km_per_px": float(frame.gsd_km_per_px),
        "gsd_m_per_px": float(frame.gsd_km_per_px * 1000.0),
        "target_center_px": [float(frame.center_px[0]), float(frame.center_px[1])],
        "bbox_xywh": bbox,
        "visibility": float(visibility),
        "background_asset_id": f"phase3_ocean_{split}_{seq_idx:06d}",
        "target_asset_id": f"phase3_target_{split}",
        "distractor_asset_ids": [],
        "motion_mode": frame.motion_mode,
        "land_overlap_ratio": 0.0,
        "shore_buffer_overlap_ratio": 0.0,
        "scale_px": float(STAGE_SCALE_PX.get(frame.stage, 18.0)),
        "angle_deg": float(np.degrees(target["heading"])),
        "obs_valid": bool(visibility > 0.0),
        "meta": {
            "dataset_name": "paper2_task_v1.0.0_smoke",
            "unit": "km",
            "range_xy_km": float(frame.range_xy_km),
            "range_3d_km": float(frame.range_3d_km),
            "aircraft_state": frame.aircraft_state,
            "crop_center_world": frame.crop_center_world,
            "crop_center_world_x": float(frame.crop_center_world[0]),
            "crop_center_world_y": float(frame.crop_center_world[1]),
            "target_state_world": {
                "x": float(target["pos_world"][0]),
                "y": float(target["pos_world"][1]),
                "vx": float(target["vel_world"][0]),
                "vy": float(target["vel_world"][1]),
                "heading": float(target["heading"]),
                "motion_mode": str(target["motion_mode"]),
                "unit": "km",
            },
            "target_world_x": float(target["pos_world"][0]),
            "target_world_y": float(target["pos_world"][1]),
            "target_world_vx": float(target["vel_world"][0]),
            "target_world_vy": float(target["vel_world"][1]),
            "center_x": float(frame.center_px[0]),
            "center_y": float(frame.center_px[1]),
            "bbox_xywh": bbox,
            "visibility": float(visibility),
            "gsd": float(frame.gsd_km_per_px),
            "gsd_km_per_px": float(frame.gsd_km_per_px),
            "perception_stage": frame.stage,
            "target_on_water": bool(frame.target_on_water),
            "land_overlap_ratio": 0.0,
            "shore_buffer_overlap_ratio": 0.0,
            "scale_px": float(STAGE_SCALE_PX.get(frame.stage, 18.0)),
            "obs_valid": bool(visibility > 0.0),
        },
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/env.yaml")
    parser.add_argument("--out-root", type=str, default="data/rendered/paper2_task_v1.0.0_smoke")
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    target_cfg = cfg["phase3_target_motion"]
    stage_cfg = cfg["phase3_task_stages"]
    image_size = int(stage_cfg.get("image_size", 256))
    base_seed = int(args.seed if args.seed is not None else target_cfg["seed"])

    out_root = Path(args.out_root)
    images_dir = out_root / "images"
    labels_dir = out_root / "labels"
    meta_dir = out_root / "meta"
    reports_dir = out_root / "reports"
    for path in (images_dir, labels_dir, meta_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    label_files = {split: (labels_dir / f"{split}.jsonl").open("w", encoding="utf-8") for split in ("train", "val", "test")}
    manifest_path = out_root / "manifest.jsonl"
    total_rows = 0
    stage_counts = {"far": 0, "mid": 0, "terminal": 0}
    split_counts = {"train": 0, "val": 0, "test": 0}
    try:
        with manifest_path.open("w", encoding="utf-8") as manifest_f:
            for seq_idx in range(int(args.sequences)):
                split = _split_for_sequence(seq_idx, int(args.sequences))
                rows = sample_phase3_task_sequence(
                    sequence_idx=seq_idx,
                    target_cfg=target_cfg,
                    stage_cfg=stage_cfg,
                    seed=base_seed + seq_idx,
                    frames=int(args.frames),
                )
                for frame in rows:
                    rng = np.random.default_rng(base_seed + seq_idx * 100000 + int(frame.frame_id))
                    canvas = _make_ocean_background(image_size, rng)
                    bbox, visibility = _draw_target(canvas, frame, rng)
                    image_path = images_dir / split / frame.sequence_id / f"{int(frame.frame_id):04d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(image_path), canvas)

                    label = _label_row(frame, image_path.resolve(), project_root, split, seq_idx, bbox, visibility)
                    line = json.dumps(label, ensure_ascii=False)
                    label_files[split].write(line + "\n")
                    manifest_f.write(line + "\n")
                    total_rows += 1
                    stage_counts[frame.stage] = stage_counts.get(frame.stage, 0) + 1
                    split_counts[split] = split_counts.get(split, 0) + 1
    finally:
        for f in label_files.values():
            f.close()

    generation_config = {
        "task": "render_phase3_task_dataset",
        "config": str(args.config),
        "out_root": str(out_root),
        "seed": base_seed,
        "sequences": int(args.sequences),
        "frames_per_sequence": int(args.frames),
        "image_size": image_size,
        "unit": "km",
        "stage_counts": stage_counts,
        "split_counts": split_counts,
        "total_frames": total_rows,
    }
    (meta_dir / "generation_config.json").write_text(json.dumps(generation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "dataset_qc.json").write_text(json.dumps(generation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(generation_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
