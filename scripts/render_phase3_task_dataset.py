from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper2.common.config import load_yaml
from paper2.render.phase3_task_sampler import Phase3TaskFrame, sample_phase3_task_sequence
from paper2.render.compositor import alpha_blend_center, read_bgra, rotate_bgra


STAGE_SCALE_PX = {
    "far": 10.0,
    "mid": 18.0,
    "terminal": 32.0,
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


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


def _iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _infer_split(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    for split in ("train", "val", "test"):
        if split in parts:
            return split
    return None


def _collect_backgrounds(assets_root: Path, water_mask_root: Path, skip_review: bool) -> dict[str, list[dict]]:
    backgrounds_root = assets_root / "backgrounds"
    review_paths = _load_review_backgrounds(water_mask_root) if skip_review else set()
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for image_path in _iter_images(backgrounds_root):
        split = _infer_split(image_path)
        if split is None:
            continue
        if str(image_path.resolve()) in review_paths:
            continue
        try:
            rel = image_path.resolve().relative_to(backgrounds_root.resolve())
        except ValueError:
            continue
        mask_path = water_mask_root / "deep_water_masks" / rel.parent / f"{image_path.stem}_deep_water_mask.png"
        if not mask_path.exists():
            mask_path = water_mask_root / "masks" / rel.parent / f"{image_path.stem}_water_mask.png"
        if not mask_path.exists():
            continue
        out[split].append({"image_path": image_path, "mask_path": mask_path})
    return out


def _load_review_backgrounds(water_mask_root: Path) -> set[str]:
    report_path = water_mask_root / "reports" / "water_mask_report.json"
    if not report_path.exists():
        return set()
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    out = set()
    for row in report.get("rows", []):
        if bool(row.get("review", False)):
            out.add(str(Path(row["image_path"]).resolve()))
    return out


def _collect_targets(assets_root: Path) -> dict[str, list[Path]]:
    roots = [
        assets_root / "target_templates" / "alpha_png",
        assets_root / "target_templates",
    ]
    out: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    seen: set[Path] = set()
    for root in roots:
        for path in _iter_images(root):
            if path.suffix.lower() != ".png":
                continue
            split = _infer_split(path)
            if split is None:
                continue
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out[split].append(path)
    return out


def _extract_patch_inside(background_bgr: np.ndarray, x1: int, y1: int, image_size: int) -> np.ndarray:
    return background_bgr[y1 : y1 + image_size, x1 : x1 + image_size].copy()


def _resize_bgra_to_long_side(img_bgra: np.ndarray, long_side_px: float, image_size: int) -> np.ndarray:
    h, w = img_bgra.shape[:2]
    target_long = max(2, min(int(round(float(long_side_px))), int(image_size * 0.85)))
    ratio = target_long / max(1.0, float(max(h, w)))
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    return cv2.resize(img_bgra, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _alpha_water_ratio(mask_u8: np.ndarray, overlay_bgra: np.ndarray, center_x: float, center_y: float) -> float:
    h, w = mask_u8.shape[:2]
    oh, ow = overlay_bgra.shape[:2]
    x1 = int(round(center_x - ow / 2))
    y1 = int(round(center_y - oh / 2))
    x2 = x1 + ow
    y2 = y1 + oh
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(w, x2)
    iy2 = min(h, y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    ox1 = ix1 - x1
    oy1 = iy1 - y1
    ox2 = ox1 + (ix2 - ix1)
    oy2 = oy1 + (iy2 - iy1)
    alpha = overlay_bgra[oy1:oy2, ox1:ox2, 3] > 10
    denom = int(alpha.sum())
    if denom <= 0:
        return 0.0
    water = mask_u8[iy1:iy2, ix1:ix2] > 0
    return float((alpha & water).sum() / float(denom))


def _render_real_asset_frame(
    frame: Phase3TaskFrame,
    *,
    split: str,
    backgrounds_by_split: dict[str, list[dict]],
    targets_by_split: dict[str, list[Path]],
    image_size: int,
    rng: np.random.Generator,
    min_target_water_ratio: float,
    min_target_visibility: float,
    placement_attempts: int,
    points_per_background: int,
) -> tuple[np.ndarray, list[float], float, dict]:
    bg_pool = backgrounds_by_split.get(split, [])
    target_pool = targets_by_split.get(split, [])
    if not bg_pool:
        raise RuntimeError(f"No usable real backgrounds for split={split}.")
    if not target_pool:
        raise RuntimeError(f"No target templates for split={split}.")

    cx = float(frame.center_px[0])
    cy = float(frame.center_px[1])
    best_candidate: tuple[float, np.ndarray, list[float], float, dict] | None = None
    for _ in range(max(1, int(placement_attempts))):
        bg_rec = bg_pool[int(rng.integers(0, len(bg_pool)))]
        bg_path = Path(bg_rec["image_path"])
        mask_path = Path(bg_rec["mask_path"])
        bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bg is None or mask is None:
            continue
        h, w = bg.shape[:2]
        if h < image_size or w < image_size:
            continue
        if mask.shape[:2] != bg.shape[:2]:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        target_path = target_pool[int(rng.integers(0, len(target_pool)))]
        target = read_bgra(target_path)
        target = _resize_bgra_to_long_side(target, STAGE_SCALE_PX.get(frame.stage, 18.0), image_size)
        angle_deg = -float(np.degrees(frame.target_state_world["heading"]))
        target = rotate_bgra(target, angle_deg)

        x_min = int(np.ceil(cx))
        y_min = int(np.ceil(cy))
        x_max = int(np.floor(float(w - image_size) + cx))
        y_max = int(np.floor(float(h - image_size) + cy))
        if x_min > x_max or y_min > y_max:
            continue
        valid = mask > 0
        bounded = np.zeros_like(valid, dtype=bool)
        bounded[y_min : y_max + 1, x_min : x_max + 1] = True
        valid &= bounded

        target_radius = int(max(2, round(0.45 * max(target.shape[:2]))))
        if target_radius > 2:
            dist = cv2.distanceTransform((mask > 0).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=3)
            preferred = valid & (dist >= float(target_radius))
            if int(preferred.sum()) > 0:
                valid = preferred

        yy, xx = np.where(valid)
        if len(xx) <= 0:
            continue
        sample_n = min(max(1, int(points_per_background)), len(xx))
        sample_idx = rng.choice(len(xx), size=sample_n, replace=False)
        for idx_raw in sample_idx:
            idx = int(idx_raw)
            target_bg_x = float(xx[idx])
            target_bg_y = float(yy[idx])
            x1 = int(round(target_bg_x - cx))
            y1 = int(round(target_bg_y - cy))
            if x1 < 0 or y1 < 0 or x1 + image_size > w or y1 + image_size > h:
                continue
            patch_mask = mask[y1 : y1 + image_size, x1 : x1 + image_size]
            if patch_mask[int(round(cy)), int(round(cx))] <= 0:
                continue
            ratio = _alpha_water_ratio(patch_mask, target, cx, cy)
            canvas = _extract_patch_inside(bg, x1, y1, image_size)
            bbox_tuple, visibility = alpha_blend_center(canvas, target, cx, cy)
            if visibility < min_target_visibility:
                continue
            bbox = [float(v) for v in bbox_tuple]
            asset_meta = {
                "asset_mode": "real",
                "background_path": str(bg_path),
                "water_mask_path": str(mask_path),
                "target_asset_path": str(target_path),
                "target_water_ratio": float(ratio),
            }
            if ratio >= min_target_water_ratio:
                return canvas, bbox, float(visibility), asset_meta
            score = float(ratio) + 0.05 * float(visibility)
            if ratio >= 0.85 and (best_candidate is None or score > best_candidate[0]):
                best_candidate = (score, canvas, bbox, float(visibility), asset_meta)

    if best_candidate is not None:
        _, canvas, bbox, visibility, asset_meta = best_candidate
        asset_meta = dict(asset_meta)
        asset_meta["placement_fallback"] = "relaxed_water_ratio"
        return canvas, bbox, visibility, asset_meta
    raise RuntimeError(f"Could not place target on water for split={split}, stage={frame.stage}.")


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


def _label_row(
    frame: Phase3TaskFrame,
    image_path: Path,
    project_root: Path,
    dataset_name: str,
    split: str,
    seq_idx: int,
    bbox: list[float],
    visibility: float,
    asset_meta: dict | None = None,
) -> dict:
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
        "background_asset_id": Path(str((asset_meta or {}).get("background_path", f"phase3_ocean_{split}_{seq_idx:06d}"))).stem,
        "target_asset_id": Path(str((asset_meta or {}).get("target_asset_path", f"phase3_target_{split}"))).stem,
        "distractor_asset_ids": [],
        "motion_mode": frame.motion_mode,
        "land_overlap_ratio": 0.0,
        "shore_buffer_overlap_ratio": 0.0,
        "scale_px": float(STAGE_SCALE_PX.get(frame.stage, 18.0)),
        "angle_deg": float(np.degrees(target["heading"])),
        "obs_valid": bool(visibility > 0.0),
        "meta": {
            "dataset_name": str(dataset_name),
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
            "asset_mode": str((asset_meta or {}).get("asset_mode", "procedural")),
            "background_path": (asset_meta or {}).get("background_path"),
            "water_mask_path": (asset_meta or {}).get("water_mask_path"),
            "target_asset_path": (asset_meta or {}).get("target_asset_path"),
            "target_water_ratio": float((asset_meta or {}).get("target_water_ratio", 1.0)),
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
    parser.add_argument("--asset-mode", choices=["procedural", "real"], default="procedural")
    parser.add_argument("--assets-root", type=str, default="data/assets/source_stage2_v2")
    parser.add_argument("--water-mask-root", type=str, default=None)
    parser.add_argument("--include-review-backgrounds", action="store_true")
    parser.add_argument("--min-target-water-ratio", type=float, default=0.98)
    parser.add_argument("--min-target-visibility", type=float, default=0.35)
    parser.add_argument("--placement-attempts", type=int, default=160)
    parser.add_argument("--points-per-background", type=int, default=64)
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    target_cfg = cfg["phase3_target_motion"]
    stage_cfg = cfg["phase3_task_stages"]
    image_size = int(stage_cfg.get("image_size", 256))
    base_seed = int(args.seed if args.seed is not None else target_cfg["seed"])
    assets_root = Path(args.assets_root).resolve()
    water_mask_root = (
        Path(args.water_mask_root).resolve()
        if args.water_mask_root is not None
        else (assets_root / "water_masks_auto").resolve()
    )
    backgrounds_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    targets_by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    if args.asset_mode == "real":
        backgrounds_by_split = _collect_backgrounds(
            assets_root,
            water_mask_root,
            skip_review=not bool(args.include_review_backgrounds),
        )
        targets_by_split = _collect_targets(assets_root)
        missing_bg = [s for s in ("train", "val", "test") if not backgrounds_by_split.get(s)]
        missing_target = [s for s in ("train", "val", "test") if not targets_by_split.get(s)]
        if missing_bg or missing_target:
            raise RuntimeError(
                f"Missing real assets. missing_background_splits={missing_bg}, missing_target_splits={missing_target}, "
                f"assets_root={assets_root}, water_mask_root={water_mask_root}"
            )

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
                    asset_meta: dict | None = None
                    if args.asset_mode == "real":
                        canvas, bbox, visibility, asset_meta = _render_real_asset_frame(
                            frame,
                            split=split,
                            backgrounds_by_split=backgrounds_by_split,
                            targets_by_split=targets_by_split,
                            image_size=image_size,
                            rng=rng,
                            min_target_water_ratio=float(args.min_target_water_ratio),
                            min_target_visibility=float(args.min_target_visibility),
                            placement_attempts=int(args.placement_attempts),
                            points_per_background=int(args.points_per_background),
                        )
                    else:
                        canvas = _make_ocean_background(image_size, rng)
                        bbox, visibility = _draw_target(canvas, frame, rng)
                        asset_meta = {"asset_mode": "procedural"}
                    image_path = images_dir / split / frame.sequence_id / f"{int(frame.frame_id):04d}.png"
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(image_path), canvas)

                    label = _label_row(
                        frame,
                        image_path.resolve(),
                        project_root,
                        out_root.name,
                        split,
                        seq_idx,
                        bbox,
                        visibility,
                        asset_meta,
                    )
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
        "asset_mode": str(args.asset_mode),
        "assets_root": str(assets_root) if args.asset_mode == "real" else None,
        "water_mask_root": str(water_mask_root) if args.asset_mode == "real" else None,
        "background_counts": {k: len(v) for k, v in backgrounds_by_split.items()} if args.asset_mode == "real" else None,
        "target_counts": {k: len(v) for k, v in targets_by_split.items()} if args.asset_mode == "real" else None,
        "stage_counts": stage_counts,
        "split_counts": split_counts,
        "total_frames": total_rows,
    }
    (meta_dir / "generation_config.json").write_text(json.dumps(generation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "dataset_qc.json").write_text(json.dumps(generation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(generation_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
