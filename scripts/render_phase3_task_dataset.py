from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from paper2.common.config import load_yaml
from paper2.render.phase3_task_sampler import Phase3TaskFrame, sample_phase3_task_sequence
from paper2.render.compositor import alpha_blend_center, read_bgra, rotate_bgra, trim_bgra_to_alpha_bbox


STAGE_SCALE_PX = {
    "far": 10.0,
    "mid": 18.0,
    "terminal": 32.0,
}
DISTRACTOR_SCALE_RANGE = (0.45, 0.85)
DISTRACTOR_MOTION_MODES = ("cv", "turn", "piecewise")
DISTRACTOR_COUNT_MAX_LIMIT = 3
DISTRACTOR_CENTER_BUFFER_PX = 6.0
DISTRACTOR_SPEED_RANGE = {
    "far": (0.22, 0.85),
    "mid": (0.18, 0.72),
    "terminal": (0.12, 0.52),
}
DEFAULT_DISTRACTOR_ALLOW_KEYWORDS = ("ship_like", "small", "boat", "canoe", "kayak")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_TARGET_ALLOW_KEYWORDS = ("top", "overhead", "aerial", "bird", "vertical")
DEFAULT_TARGET_REJECT_KEYWORDS = ("side", "oblique", "profile", "tilt", "perspective", "front", "rear")
DEFAULT_DISTRACTOR_REJECT_KEYWORDS = DEFAULT_TARGET_REJECT_KEYWORDS


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


def _iter_split_pngs(root: Path, split: str) -> list[Path]:
    split_root = root / split
    if not split_root.exists():
        return []
    return sorted(p for p in split_root.iterdir() if p.is_file() and p.suffix.lower() == ".png")


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


def _split_keywords(text: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in str(text).split(",") if x.strip())


def _target_name_tokens(path: Path) -> set[str]:
    return {x for x in re.split(r"[^a-z0-9]+", path.stem.lower()) if x}


def _template_allowed(
    path: Path,
    *,
    allow_keywords: tuple[str, ...],
    reject_keywords: tuple[str, ...],
) -> bool:
    text = str(path).lower().replace("\\", "/")
    tokens = _target_name_tokens(path)
    if any(k and (k in tokens or k in text) for k in reject_keywords):
        return False
    if not allow_keywords:
        return True
    return any(k and (k in tokens or k in text) for k in allow_keywords)


def _target_template_allowed(
    path: Path,
    *,
    allow_keywords: tuple[str, ...],
    reject_keywords: tuple[str, ...],
) -> bool:
    return _template_allowed(path, allow_keywords=allow_keywords, reject_keywords=reject_keywords)


def _collect_targets(
    assets_root: Path,
    *,
    allow_keywords: tuple[str, ...] = DEFAULT_TARGET_ALLOW_KEYWORDS,
    reject_keywords: tuple[str, ...] = DEFAULT_TARGET_REJECT_KEYWORDS,
) -> dict[str, list[Path]]:
    root = assets_root / "target_templates" / "alpha_png"
    out: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    seen: set[Path] = set()
    for split in ("train", "val", "test"):
        for path in _iter_split_pngs(root, split):
            if not _target_template_allowed(path, allow_keywords=allow_keywords, reject_keywords=reject_keywords):
                continue
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out[split].append(path)
    return out


def _collect_distractors(
    assets_root: Path,
    *,
    allow_keywords: tuple[str, ...] = DEFAULT_TARGET_ALLOW_KEYWORDS,
    reject_keywords: tuple[str, ...] = DEFAULT_TARGET_REJECT_KEYWORDS,
) -> dict[str, list[Path]]:
    root = assets_root / "distractor_templates" / "alpha_png"
    out: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    seen: set[Path] = set()
    for split in ("train", "val", "test"):
        for path in _iter_split_pngs(root, split):
            if not _template_allowed(path, allow_keywords=allow_keywords, reject_keywords=reject_keywords):
                continue
            rp = path.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out[split].append(path)
    return out


def _infer_background_category_from_path(path: Path) -> str:
    low = str(path).lower().replace("\\", "/")
    if "/open_sea" in low:
        return "open_sea"
    if "/coastal" in low:
        return "coastal"
    if "/island_complex" in low:
        return "island_complex"
    if "/port" in low:
        return "port"
    return "unknown"


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


def _point_on_water(mask_u8: np.ndarray, x: float, y: float) -> bool:
    h, w = mask_u8.shape[:2]
    ix = int(round(float(x)))
    iy = int(round(float(y)))
    if ix < 0 or iy < 0 or ix >= w or iy >= h:
        return False
    return bool(mask_u8[iy, ix] > 0)


def _wrap_angle(value: float) -> float:
    return float((float(value) + float(np.pi)) % (2.0 * float(np.pi)) - float(np.pi))


def _prepare_distractor_overlay(track: "_DistractorTrack", image_size: int) -> np.ndarray:
    distractor = trim_bgra_to_alpha_bbox(read_bgra(track.asset_path))
    distractor = _resize_bgra_to_long_side(distractor, track.scale_px, image_size)
    distractor = rotate_bgra(distractor, float(np.degrees(track.heading)))
    return trim_bgra_to_alpha_bbox(distractor)


def _distractor_water_ratio_at(
    mask_u8: np.ndarray,
    track: "_DistractorTrack",
    center_x: float,
    center_y: float,
    image_size: int,
) -> float:
    overlay = _prepare_distractor_overlay(track, image_size)
    return _alpha_water_ratio(mask_u8, overlay, center_x, center_y)


@dataclass
class _DistractorTrack:
    asset_path: str
    center_px: np.ndarray
    heading: float
    speed_px: float
    motion_mode: str
    turn_rate: float
    steps_to_switch: int
    scale_px: float
    radius_px: float
    count_requested: int


def _sample_distractor_track(
    *,
    distractor_pool: list[Path],
    mask_u8: np.ndarray,
    target_center: tuple[float, float],
    target_clearance_px: float,
    rng: np.random.Generator,
    stage: str,
    image_size: int,
    count_requested: int,
    scale_range: tuple[float, float],
    min_target_distance_px: float,
    min_water_ratio: float,
    existing_tracks: list[_DistractorTrack] | None = None,
) -> _DistractorTrack | None:
    if not distractor_pool:
        return None
    water = mask_u8 > 0
    yy_all, xx_all = np.where(water)
    if len(xx_all) <= 0:
        return None
    target_cx, target_cy = float(target_center[0]), float(target_center[1])
    scale_min = max(0.05, float(min(scale_range[0], scale_range[1])))
    scale_max = max(scale_min, float(max(scale_range[0], scale_range[1])))
    existing_tracks = list(existing_tracks or [])
    for _ in range(96):
        path = distractor_pool[int(rng.integers(0, len(distractor_pool)))]
        distractor = trim_bgra_to_alpha_bbox(read_bgra(path))
        scale_px = float(STAGE_SCALE_PX.get(stage, 18.0)) * float(rng.uniform(scale_min, scale_max))
        distractor = _resize_bgra_to_long_side(distractor, scale_px, image_size)
        distractor = rotate_bgra(distractor, float(rng.uniform(0.0, 360.0)))
        distractor = trim_bgra_to_alpha_bbox(distractor)
        dh, dw = distractor.shape[:2]
        radius = max(2.0, 0.5 * float(max(dh, dw)))

        idx = int(rng.integers(0, len(xx_all)))
        dx = float(xx_all[idx])
        dy = float(yy_all[idx])
        if dx - radius < 0 or dy - radius < 0 or dx + radius >= image_size or dy + radius >= image_size:
            continue
        if float(np.hypot(dx - target_cx, dy - target_cy)) < float(max(min_target_distance_px, target_clearance_px)):
            continue
        if any(
            float(np.hypot(dx - other.center_px[0], dy - other.center_px[1]))
            < float(radius + other.radius_px + DISTRACTOR_CENTER_BUFFER_PX)
            for other in existing_tracks
        ):
            continue
        water_ratio = _alpha_water_ratio(mask_u8, distractor, dx, dy)
        if water_ratio < float(min_water_ratio):
            continue
        motion_mode = str(rng.choice(DISTRACTOR_MOTION_MODES, p=[0.55, 0.3, 0.15]))
        heading = float(rng.uniform(-np.pi, np.pi))
        speed_low, speed_high = DISTRACTOR_SPEED_RANGE.get(stage, (0.18, 0.72))
        if int(count_requested) >= 3:
            speed_high = min(speed_high, 0.62)
        speed_px = float(rng.uniform(float(speed_low), float(speed_high)))
        turn_rate = float(rng.uniform(-0.03, 0.03))
        steps_to_switch = int(rng.integers(10, 28))
        return _DistractorTrack(
            asset_path=str(path),
            center_px=np.array([dx, dy], dtype=float),
            heading=heading,
            speed_px=speed_px,
            motion_mode=motion_mode,
            turn_rate=turn_rate,
            steps_to_switch=steps_to_switch,
            scale_px=float(scale_px),
            radius_px=float(radius),
            count_requested=int(count_requested),
        )
    return None


def _advance_distractor_track(
    track: _DistractorTrack,
    rng: np.random.Generator,
    mask_u8: np.ndarray | None = None,
    target_center: tuple[float, float] | None = None,
    target_clearance_px: float = 0.0,
    min_water_ratio: float = 0.0,
    other_tracks: list[_DistractorTrack] | None = None,
) -> None:
    def _is_safe(candidate_xy: np.ndarray) -> tuple[bool, _DistractorTrack | None]:
        if target_center is not None:
            target_dist = float(np.hypot(candidate_xy[0] - float(target_center[0]), candidate_xy[1] - float(target_center[1])))
            if target_dist < float(target_clearance_px):
                return False, None
        if mask_u8 is not None:
            h, w = mask_u8.shape[:2]
            ix = int(round(float(candidate_xy[0])))
            iy = int(round(float(candidate_xy[1])))
            if ix < 0 or iy < 0 or ix >= w or iy >= h or mask_u8[iy, ix] <= 0:
                return False, None
            if min_water_ratio > 0.0:
                water_ratio = _distractor_water_ratio_at(mask_u8, track, float(candidate_xy[0]), float(candidate_xy[1]), w)
                if water_ratio < float(min_water_ratio):
                    return False, None
        if other_tracks:
            nearest: _DistractorTrack | None = None
            nearest_dist = float("inf")
            for other in other_tracks:
                if other is track:
                    continue
                sep = float(np.hypot(candidate_xy[0] - other.center_px[0], candidate_xy[1] - other.center_px[1]))
                required = float(track.radius_px + other.radius_px + DISTRACTOR_CENTER_BUFFER_PX)
                if sep < required:
                    if sep < nearest_dist:
                        nearest = other
                        nearest_dist = sep
                    return False, nearest
        return True, None

    def _proposal(step_scale: float) -> np.ndarray:
        delta = np.array([np.cos(track.heading), -np.sin(track.heading)], dtype=float) * float(track.speed_px) * float(step_scale)
        return track.center_px + delta

    if track.motion_mode == "turn":
        track.heading = _wrap_angle(track.heading + track.turn_rate)
    elif track.motion_mode == "piecewise":
        track.steps_to_switch -= 1
        if track.steps_to_switch <= 0:
            track.heading = _wrap_angle(track.heading + float(rng.uniform(-0.8, 0.8)))
            track.speed_px = float(np.clip(track.speed_px * float(rng.uniform(0.85, 1.15)), 0.18, 1.8))
            track.steps_to_switch = int(rng.integers(8, 24))
    prev = track.center_px.copy()
    step_scales = (1.0, 0.78, 0.58, 0.42)
    for attempt, step_scale in enumerate(step_scales):
        candidate = _proposal(step_scale)
        safe, nearest_other = _is_safe(candidate)
        if safe:
            track.center_px = candidate
            track.heading = _wrap_angle(track.heading)
            return

        if nearest_other is not None:
            target_vec = candidate - nearest_other.center_px
            if float(np.linalg.norm(target_vec)) > 1e-9:
                away = float(np.arctan2(target_vec[1], target_vec[0]))
                track.heading = _wrap_angle(away + float(rng.uniform(-0.45, 0.45)))
        elif target_center is not None:
            target_vec = candidate - np.array([float(target_center[0]), float(target_center[1])], dtype=float)
            if float(np.linalg.norm(target_vec)) > 1e-9:
                away = float(np.arctan2(target_vec[1], target_vec[0]))
                track.heading = _wrap_angle(away + float(rng.uniform(-0.4, 0.4)))
        elif mask_u8 is not None:
            track.heading = _wrap_angle(track.heading + float(np.pi) + float(rng.uniform(-0.35, 0.35)))
        else:
            track.heading = _wrap_angle(track.heading + float(rng.uniform(-0.55, 0.55)))
        track.speed_px = float(np.clip(track.speed_px * float(rng.uniform(0.85, 0.98)), 0.12, 1.8))
        if attempt == len(step_scales) - 1:
            break

    track.center_px = prev
    track.heading = _wrap_angle(track.heading + float(rng.uniform(-0.25, 0.25)))


def _render_distractor_track(
    canvas: np.ndarray,
    mask_u8: np.ndarray,
    track: _DistractorTrack,
    *,
    target_center: tuple[float, float],
    min_target_distance_px: float,
    min_water_ratio: float,
) -> tuple[list[float], float, float] | None:
    image_size = int(canvas.shape[0])
    cx, cy = float(track.center_px[0]), float(track.center_px[1])
    if cx < 0.0 or cy < 0.0 or cx >= image_size or cy >= image_size:
        return None
    if float(np.hypot(cx - float(target_center[0]), cy - float(target_center[1]))) < float(min_target_distance_px):
        return None
    distractor = _prepare_distractor_overlay(track, image_size)
    water_ratio = _alpha_water_ratio(mask_u8, distractor, cx, cy)
    if water_ratio < float(min_water_ratio):
        return None
    scratch = canvas.copy()
    bbox_tuple, visibility = alpha_blend_center(scratch, distractor, cx, cy)
    if visibility <= 0.0:
        return None
    canvas[:, :, :] = scratch
    return [float(v) for v in bbox_tuple], float(visibility), float(water_ratio)


def _render_real_asset_frame(
    frame: Phase3TaskFrame,
    *,
    split: str,
    backgrounds_by_split: dict[str, list[dict]],
    targets_by_split: dict[str, list[Path]],
    distractors_by_split: dict[str, list[Path]],
    image_size: int,
    rng: np.random.Generator,
    min_target_water_ratio: float,
    min_target_visibility: float,
    placement_attempts: int,
    points_per_background: int,
    distractor_count_min: int = 0,
    distractor_count_max: int = 0,
    min_distractor_target_distance_px: float = 48.0,
    distractor_scale_min: float = DISTRACTOR_SCALE_RANGE[0],
    distractor_scale_max: float = DISTRACTOR_SCALE_RANGE[1],
    fixed_background: dict | None = None,
    fixed_target: Path | None = None,
    sequence_state: dict | None = None,
    allow_relaxed_water_ratio: bool = False,
) -> tuple[np.ndarray, list[float], float, dict]:
    bg_pool = backgrounds_by_split.get(split, [])
    target_pool = targets_by_split.get(split, [])
    if not bg_pool:
        raise RuntimeError(f"No usable real backgrounds for split={split}.")
    if not target_pool:
        raise RuntimeError(f"No target templates for split={split}.")
    distractor_pool = distractors_by_split.get(split, [])
    if int(distractor_count_max) > 0 and not distractor_pool:
        raise RuntimeError(f"No distractor templates for split={split}.")

    cx = float(frame.center_px[0])
    cy = float(frame.center_px[1])
    best_candidate: tuple[float, np.ndarray, list[float], float, dict] | None = None
    for _ in range(max(1, int(placement_attempts))):
        bg_rec = fixed_background if fixed_background is not None else bg_pool[int(rng.integers(0, len(bg_pool)))]
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

        target_path = Path(fixed_target) if fixed_target is not None else target_pool[int(rng.integers(0, len(target_pool)))]
        target = read_bgra(target_path)
        target = trim_bgra_to_alpha_bbox(target)
        target = _resize_bgra_to_long_side(target, STAGE_SCALE_PX.get(frame.stage, 18.0), image_size)
        angle_deg = -float(np.degrees(frame.target_state_world["heading"]))
        target = rotate_bgra(target, angle_deg)
        target = trim_bgra_to_alpha_bbox(target)
        target_clearance_px = max(float(min_distractor_target_distance_px), 0.55 * float(max(target.shape[:2])) + 6.0)

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
            distractor_meta: list[dict] = []
            if int(distractor_count_max) > 0 and distractor_pool:
                tracks = None if sequence_state is None else sequence_state.get("distractor_tracks")
                if tracks is None:
                    desired = int(rng.integers(max(0, int(distractor_count_min)), max(0, int(distractor_count_max)) + 1))
                    tracks = []
                    for _ in range(desired):
                        track = _sample_distractor_track(
                            distractor_pool=distractor_pool,
                            mask_u8=patch_mask,
                            target_center=(cx, cy),
                            target_clearance_px=float(target_clearance_px),
                            rng=rng,
                            stage=frame.stage,
                            image_size=image_size,
                            count_requested=desired,
                            scale_range=(float(distractor_scale_min), float(distractor_scale_max)),
                            min_target_distance_px=float(min_distractor_target_distance_px),
                            min_water_ratio=float(min_target_water_ratio),
                            existing_tracks=tracks,
                        )
                        if track is not None:
                            tracks.append(track)
                    if sequence_state is not None:
                        sequence_state["distractor_tracks"] = tracks
                        sequence_state["distractor_count_requested"] = int(len(tracks))
                        sequence_state["distractor_count_desired"] = int(desired)
                for track in tracks:
                    other_tracks = [other for other in tracks if other is not track]
                    _advance_distractor_track(
                        track,
                        rng,
                        patch_mask,
                        target_center=(cx, cy),
                        target_clearance_px=float(target_clearance_px),
                        min_water_ratio=float(min_target_water_ratio),
                        other_tracks=other_tracks,
                    )
                    rendered = _render_distractor_track(
                        canvas,
                        patch_mask,
                        track,
                        target_center=(cx, cy),
                        min_target_distance_px=float(min_distractor_target_distance_px),
                        min_water_ratio=float(min_target_water_ratio),
                        )
                    if rendered is not None:
                        bbox_tuple, visibility_d, water_ratio_d = rendered
                        distractor_meta.append(
                            {
                                "asset_path": track.asset_path,
                                "bbox_xywh": bbox_tuple,
                                "center_px": [float(track.center_px[0]), float(track.center_px[1])],
                                "radius_px": float(track.radius_px),
                                "water_ratio": float(water_ratio_d),
                                "visibility": float(visibility_d),
                                "count_requested": int(sequence_state.get("distractor_count_requested", len(tracks)) if sequence_state is not None else len(tracks)),
                                "count_desired": int(sequence_state.get("distractor_count_desired", len(tracks)) if sequence_state is not None else len(tracks)),
                            }
                        )
                    else:
                        raise RuntimeError(
                            f"Failed to render distractor track asset={track.asset_path} split={split} "
                            f"sequence={frame.sequence_id} frame={frame.frame_id}"
                        )
            bbox_tuple, visibility = alpha_blend_center(canvas, target, cx, cy)
            if visibility < min_target_visibility:
                continue
            bbox = [float(v) for v in bbox_tuple]
            active_distractor_tracks = []
            if sequence_state is not None and sequence_state.get("distractor_tracks") is not None:
                active_distractor_tracks = list(sequence_state.get("distractor_tracks") or [])
            requested_distractors = int(sequence_state.get("distractor_count_requested", len(active_distractor_tracks))) if sequence_state is not None else int(len(distractor_meta))
            asset_meta = {
                "asset_mode": "real",
                "background_path": str(bg_path),
                "water_mask_path": str(mask_path),
                "background_size_px": [int(w), int(h)],
                "crop_origin_bg_px": [int(x1), int(y1)],
                "target_bg_center_px": [float(target_bg_x), float(target_bg_y)],
                "background_category": _infer_background_category_from_path(bg_path),
                "target_asset_path": str(target_path),
                "target_water_ratio": float(ratio),
                "distractor_asset_paths": [d["asset_path"] for d in distractor_meta],
                "distractor_bboxes_xywh": [d["bbox_xywh"] for d in distractor_meta],
                "distractor_water_ratios": [d["water_ratio"] for d in distractor_meta],
                "distractor_visibilities": [d["visibility"] for d in distractor_meta],
                "distractor_count_requested": int(requested_distractors),
                "distractor_count": int(len(active_distractor_tracks)),
                "distractor_count_active": int(len(active_distractor_tracks)),
                "distractor_count_rendered": int(len(distractor_meta)),
            }
            if ratio >= min_target_water_ratio:
                return canvas, bbox, float(visibility), asset_meta
            score = float(ratio) + 0.05 * float(visibility)
            if allow_relaxed_water_ratio and ratio >= 0.85 and (best_candidate is None or score > best_candidate[0]):
                best_candidate = (score, canvas, bbox, float(visibility), asset_meta)

    if best_candidate is not None:
        _, canvas, bbox, visibility, asset_meta = best_candidate
        asset_meta = dict(asset_meta)
        asset_meta["placement_fallback"] = "relaxed_water_ratio"
        return canvas, bbox, visibility, asset_meta
    raise RuntimeError(f"Could not place target on water for split={split}, stage={frame.stage}.")


def _place_distractors(
    canvas: np.ndarray,
    mask_u8: np.ndarray,
    distractor_pool: list[Path],
    *,
    target_center: tuple[float, float],
    target_overlay: np.ndarray,
    stage: str,
    rng: np.random.Generator,
    count_min: int,
    count_max: int,
    min_target_distance_px: float,
    scale_range: tuple[float, float],
    min_water_ratio: float,
    min_visibility: float,
) -> tuple[list[dict], bool]:
    count_min = max(0, int(count_min))
    count_max = max(count_min, int(count_max))
    if count_max <= 0:
        return [], True
    if not distractor_pool:
        return [], False

    desired = int(rng.integers(count_min, count_max + 1))
    image_h, image_w = canvas.shape[:2]
    target_long = float(max(target_overlay.shape[:2]))
    target_cx, target_cy = float(target_center[0]), float(target_center[1])
    scale_min = max(0.05, float(min(scale_range[0], scale_range[1])))
    scale_max = max(scale_min, float(max(scale_range[0], scale_range[1])))
    placed: list[dict] = []

    water = mask_u8 > 0
    yy_all, xx_all = np.where(water)
    if len(xx_all) <= 0:
        return [], False

    for _ in range(desired):
        accepted = False
        for _attempt in range(96):
            path = distractor_pool[int(rng.integers(0, len(distractor_pool)))]
            distractor = read_bgra(path)
            distractor = trim_bgra_to_alpha_bbox(distractor)
            scale = float(STAGE_SCALE_PX.get(stage, 18.0)) * float(
                rng.uniform(scale_min, scale_max)
            )
            distractor = _resize_bgra_to_long_side(distractor, scale, image_w)
            distractor = rotate_bgra(distractor, float(rng.uniform(0.0, 360.0)))
            distractor = trim_bgra_to_alpha_bbox(distractor)
            dh, dw = distractor.shape[:2]
            radius = max(2.0, 0.5 * float(max(dh, dw)))

            idx = int(rng.integers(0, len(xx_all)))
            dx = float(xx_all[idx])
            dy = float(yy_all[idx])
            if dx - radius < 0 or dy - radius < 0 or dx + radius >= image_w or dy + radius >= image_h:
                continue
            target_clearance = max(float(min_target_distance_px), 0.55 * target_long + radius + 6.0)
            if float(np.hypot(dx - target_cx, dy - target_cy)) < target_clearance:
                continue
            if any(float(np.hypot(dx - d["center_px"][0], dy - d["center_px"][1])) < radius + d["radius_px"] + 4.0 for d in placed):
                continue
            water_ratio = _alpha_water_ratio(mask_u8, distractor, dx, dy)
            if water_ratio < min_water_ratio:
                continue

            scratch = canvas.copy()
            bbox_tuple, visibility = alpha_blend_center(scratch, distractor, dx, dy)
            if visibility < min_visibility:
                continue
            canvas[:, :, :] = scratch
            placed.append(
                {
                    "asset_path": str(path),
                    "bbox_xywh": [float(v) for v in bbox_tuple],
                    "center_px": [float(dx), float(dy)],
                    "radius_px": float(radius),
                    "water_ratio": float(water_ratio),
                    "visibility": float(visibility),
                    "count_requested": int(desired),
                }
            )
            accepted = True
            break
        if not accepted:
            return placed, False
    return placed, True


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
        "background_category": str((asset_meta or {}).get("background_category", "unknown")),
        "target_asset_id": Path(str((asset_meta or {}).get("target_asset_path", f"phase3_target_{split}"))).stem,
        "distractor_asset_ids": [
            Path(str(p)).stem for p in (asset_meta or {}).get("distractor_asset_paths", [])
        ],
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
            "background_category": str((asset_meta or {}).get("background_category", "unknown")),
            "background_path": (asset_meta or {}).get("background_path"),
            "water_mask_path": (asset_meta or {}).get("water_mask_path"),
            "background_size_px": (asset_meta or {}).get("background_size_px"),
            "crop_origin_bg_px": (asset_meta or {}).get("crop_origin_bg_px"),
            "crop_origin_xy": (asset_meta or {}).get("crop_origin_bg_px"),
            "crop_bg_xy": (asset_meta or {}).get("crop_origin_bg_px"),
            "crop_top_left": (asset_meta or {}).get("crop_origin_bg_px"),
            "target_bg_center_px": (asset_meta or {}).get("target_bg_center_px"),
            "target_asset_path": (asset_meta or {}).get("target_asset_path"),
            "target_water_ratio": float((asset_meta or {}).get("target_water_ratio", 1.0)),
            "distractor_asset_paths": (asset_meta or {}).get("distractor_asset_paths", []),
            "distractor_bboxes_xywh": (asset_meta or {}).get("distractor_bboxes_xywh", []),
            "distractor_water_ratios": (asset_meta or {}).get("distractor_water_ratios", []),
            "distractor_visibilities": (asset_meta or {}).get("distractor_visibilities", []),
            "distractor_count_requested": int((asset_meta or {}).get("distractor_count_requested", 0)),
            "distractor_count": int((asset_meta or {}).get("distractor_count", 0)),
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
    parser.add_argument("--assets-root", type=str, default="data/assets/source_stage2")
    parser.add_argument("--target-assets-root", type=str, default="data/assets/source_stage2_v2")
    parser.add_argument("--distractor-assets-root", type=str, default=None)
    parser.add_argument("--water-mask-root", type=str, default=None)
    parser.add_argument("--include-review-backgrounds", action="store_true")
    parser.add_argument("--min-target-water-ratio", type=float, default=0.98)
    parser.add_argument("--min-target-visibility", type=float, default=0.35)
    parser.add_argument("--allow-relaxed-water-ratio", action="store_true")
    parser.add_argument("--placement-attempts", type=int, default=160)
    parser.add_argument("--points-per-background", type=int, default=64)
    parser.add_argument("--sequence-placement-attempts", type=int, default=24)
    parser.add_argument("--distractor-count-min", type=int, default=0)
    parser.add_argument("--distractor-count-max", type=int, default=DISTRACTOR_COUNT_MAX_LIMIT)
    parser.add_argument("--min-distractor-target-distance-px", type=float, default=48.0)
    parser.add_argument("--distractor-scale-min", type=float, default=DISTRACTOR_SCALE_RANGE[0])
    parser.add_argument("--distractor-scale-max", type=float, default=DISTRACTOR_SCALE_RANGE[1])
    parser.add_argument(
        "--target-allow-keywords",
        type=str,
        default=",".join(DEFAULT_TARGET_ALLOW_KEYWORDS),
        help="Comma-separated keywords required for real target templates. Empty string allows all non-rejected targets.",
    )
    parser.add_argument(
        "--target-reject-keywords",
        type=str,
        default=",".join(DEFAULT_TARGET_REJECT_KEYWORDS),
        help="Comma-separated keywords rejected for real target templates.",
    )
    parser.add_argument(
        "--distractor-allow-keywords",
        type=str,
        default=",".join(DEFAULT_DISTRACTOR_ALLOW_KEYWORDS),
        help="Comma-separated keywords required for real distractor templates. Empty string allows all non-rejected distractors.",
    )
    parser.add_argument(
        "--distractor-reject-keywords",
        type=str,
        default=",".join(DEFAULT_DISTRACTOR_REJECT_KEYWORDS),
        help="Comma-separated keywords rejected for real distractor templates.",
    )
    args = parser.parse_args()
    if int(args.distractor_count_min) < 0 or int(args.distractor_count_max) < 0:
        raise ValueError("Distractor counts must be non-negative.")
    if int(args.distractor_count_min) > int(args.distractor_count_max):
        raise ValueError("--distractor-count-min cannot exceed --distractor-count-max.")
    if int(args.distractor_count_max) > DISTRACTOR_COUNT_MAX_LIMIT:
        raise ValueError(f"--distractor-count-max cannot exceed {DISTRACTOR_COUNT_MAX_LIMIT}.")
    if float(args.distractor_scale_min) <= 0.0 or float(args.distractor_scale_max) <= 0.0:
        raise ValueError("Distractor scale factors must be positive.")

    project_root = Path.cwd().resolve()
    cfg = load_yaml(Path(args.config))
    target_cfg = cfg["phase3_target_motion"]
    stage_cfg = cfg["phase3_task_stages"]
    image_size = int(stage_cfg.get("image_size", 256))
    base_seed = int(args.seed if args.seed is not None else target_cfg["seed"])
    assets_root = Path(args.assets_root).resolve()
    target_assets_root = (
        Path(args.target_assets_root).resolve()
        if args.target_assets_root is not None
        else assets_root
    )
    distractor_assets_root = (
        Path(args.distractor_assets_root).resolve()
        if args.distractor_assets_root is not None
        else assets_root
    )
    water_mask_root = (
        Path(args.water_mask_root).resolve()
        if args.water_mask_root is not None
        else (assets_root / "water_masks_auto").resolve()
    )
    backgrounds_by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    targets_by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    distractors_by_split: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    if args.asset_mode == "real":
        backgrounds_by_split = _collect_backgrounds(
            assets_root,
            water_mask_root,
            skip_review=not bool(args.include_review_backgrounds),
        )
        targets_by_split = _collect_targets(
            target_assets_root,
            allow_keywords=_split_keywords(args.target_allow_keywords),
            reject_keywords=_split_keywords(args.target_reject_keywords),
        )
        if not any(targets_by_split.values()) and target_assets_root != assets_root:
            targets_by_split = _collect_targets(
                assets_root,
                allow_keywords=_split_keywords(args.target_allow_keywords),
                reject_keywords=_split_keywords(args.target_reject_keywords),
            )
        if int(args.distractor_count_max) > 0:
            distractors_by_split = _collect_distractors(
                distractor_assets_root,
                allow_keywords=_split_keywords(args.distractor_allow_keywords),
                reject_keywords=_split_keywords(args.distractor_reject_keywords),
            )
            if not any(distractors_by_split.values()) and distractor_assets_root == assets_root:
                fallback_root = (assets_root.parent / "source_stage2").resolve()
                if fallback_root.exists() and fallback_root != distractor_assets_root:
                    distractor_assets_root = fallback_root
                    distractors_by_split = _collect_distractors(
                        distractor_assets_root,
                        allow_keywords=_split_keywords(args.distractor_allow_keywords),
                        reject_keywords=_split_keywords(args.distractor_reject_keywords),
                    )
        missing_bg = [s for s in ("train", "val", "test") if not backgrounds_by_split.get(s)]
        missing_target = [s for s in ("train", "val", "test") if not targets_by_split.get(s)]
        missing_distractor = [
            s for s in ("train", "val", "test") if int(args.distractor_count_max) > 0 and not distractors_by_split.get(s)
        ]
        if missing_bg or missing_target or missing_distractor:
            raise RuntimeError(
                f"Missing real assets. missing_background_splits={missing_bg}, missing_target_splits={missing_target}, "
                f"missing_distractor_splits={missing_distractor}, assets_root={assets_root}, "
                f"target_assets_root={target_assets_root}, distractor_assets_root={distractor_assets_root}, "
                f"water_mask_root={water_mask_root}"
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
                rendered_sequence: list[tuple[Phase3TaskFrame, np.ndarray, list[float], float, dict]] = []
                last_error: Exception | None = None
                sequence_attempts = max(1, int(args.sequence_placement_attempts))
                for seq_attempt in range(sequence_attempts):
                    rendered_sequence = []
                    seq_background: dict | None = None
                    seq_target: Path | None = None
                    sequence_state: dict = {}
                    if args.asset_mode == "real":
                        seq_rng = np.random.default_rng(base_seed + seq_idx * 1000003 + seq_attempt * 10007)
                        bg_pool = backgrounds_by_split.get(split, [])
                        target_pool = targets_by_split.get(split, [])
                        if not bg_pool or not target_pool:
                            raise RuntimeError(f"Missing real assets for split={split}.")
                        seq_background = bg_pool[int(seq_rng.integers(0, len(bg_pool)))]
                        seq_target = target_pool[int(seq_rng.integers(0, len(target_pool)))]
                    try:
                        for frame in rows:
                            rng = np.random.default_rng(
                                base_seed + seq_idx * 100000 + int(frame.frame_id) + seq_attempt * 10000019
                            )
                            asset_meta: dict | None = None
                            if args.asset_mode == "real":
                                canvas, bbox, visibility, asset_meta = _render_real_asset_frame(
                                    frame,
                                    split=split,
                                    backgrounds_by_split=backgrounds_by_split,
                                    targets_by_split=targets_by_split,
                                    distractors_by_split=distractors_by_split,
                                    image_size=image_size,
                                    rng=rng,
                                    min_target_water_ratio=float(args.min_target_water_ratio),
                                    min_target_visibility=float(args.min_target_visibility),
                                    placement_attempts=int(args.placement_attempts),
                                    points_per_background=int(args.points_per_background),
                                    distractor_count_min=int(args.distractor_count_min),
                                    distractor_count_max=int(args.distractor_count_max),
                                    min_distractor_target_distance_px=float(args.min_distractor_target_distance_px),
                                    distractor_scale_min=float(args.distractor_scale_min),
                                    distractor_scale_max=float(args.distractor_scale_max),
                                    fixed_background=seq_background,
                                    fixed_target=seq_target,
                                    sequence_state=sequence_state,
                                    allow_relaxed_water_ratio=bool(args.allow_relaxed_water_ratio),
                                )
                            else:
                                canvas = _make_ocean_background(image_size, rng)
                                bbox, visibility = _draw_target(canvas, frame, rng)
                                asset_meta = {"asset_mode": "procedural"}
                            rendered_sequence.append((frame, canvas, bbox, float(visibility), dict(asset_meta or {})))
                    except RuntimeError as exc:
                        last_error = exc
                        continue
                    break
                if not rendered_sequence or len(rendered_sequence) != len(rows):
                    raise RuntimeError(
                        f"Could not render full sequence seq_idx={seq_idx}, split={split}, "
                        f"attempts={sequence_attempts}. last_error={last_error}"
                    )

                for frame, canvas, bbox, visibility, asset_meta in rendered_sequence:
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
        "target_assets_root": str(target_assets_root) if args.asset_mode == "real" else None,
        "distractor_assets_root": str(distractor_assets_root) if args.asset_mode == "real" else None,
        "water_mask_root": str(water_mask_root) if args.asset_mode == "real" else None,
        "background_counts": {k: len(v) for k, v in backgrounds_by_split.items()} if args.asset_mode == "real" else None,
        "target_counts": {k: len(v) for k, v in targets_by_split.items()} if args.asset_mode == "real" else None,
        "distractor_counts": {k: len(v) for k, v in distractors_by_split.items()} if args.asset_mode == "real" else None,
        "target_template_filter": {
            "allow_keywords": list(_split_keywords(args.target_allow_keywords)),
            "reject_keywords": list(_split_keywords(args.target_reject_keywords)),
        }
        if args.asset_mode == "real"
        else None,
        "distractor_template_filter": {
            "allow_keywords": list(_split_keywords(args.distractor_allow_keywords)),
            "reject_keywords": list(_split_keywords(args.distractor_reject_keywords)),
        }
        if args.asset_mode == "real"
        else None,
        "sequence_policy": {
            "fixed_background_per_sequence": bool(args.asset_mode == "real"),
            "fixed_target_template_per_sequence": bool(args.asset_mode == "real"),
            "allow_relaxed_water_ratio": bool(args.allow_relaxed_water_ratio),
            "min_target_water_ratio": float(args.min_target_water_ratio),
            "distractor_count_min": int(args.distractor_count_min),
            "distractor_count_max": int(args.distractor_count_max),
            "min_distractor_target_distance_px": float(args.min_distractor_target_distance_px),
            "distractor_scale_min": float(args.distractor_scale_min),
            "distractor_scale_max": float(args.distractor_scale_max),
        },
        "stage_counts": stage_counts,
        "split_counts": split_counts,
        "total_frames": total_rows,
    }
    (meta_dir / "generation_config.json").write_text(json.dumps(generation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (reports_dir / "dataset_qc.json").write_text(json.dumps(generation_config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(generation_config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
