from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from paper2.render.asset_registry import AssetRecord, AssetRegistry
from paper2.render.compositor import alpha_blend_center, read_bgra, resize_bgra_with_scale, rotate_bgra
from paper2.render.coordinate_mapper import background_px_to_world, world_to_background_px, world_to_image
from paper2.render.motion_sampler import generate_motion_sequence, sample_mode
from paper2.render.perturbations import apply_perturbations
from paper2.render.schema import RenderedFrameRecord


def _extract_patch(background_bgr: np.ndarray, cx: float, cy: float, size: int) -> np.ndarray:
    h, w = background_bgr.shape[:2]
    half = size // 2
    x1 = int(round(cx)) - half
    y1 = int(round(cy)) - half
    x2 = x1 + size
    y2 = y1 + size

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)
    if pad_l or pad_t or pad_r or pad_b:
        padded = cv2.copyMakeBorder(background_bgr, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        x1 += pad_l
        y1 += pad_t
        x2 += pad_l
        y2 += pad_t
        return padded[y1:y2, x1:x2].copy()
    return background_bgr[y1:y2, x1:x2].copy()


def _water_mask(patch_bgr: np.ndarray) -> np.ndarray:
    # Conservative heuristic for sea/harbor water:
    # require plausible HSV range + blue/cyan dominance, and suppress bright gray land/concrete.
    b = patch_bgr[:, :, 0].astype(np.int16)
    g = patch_bgr[:, :, 1].astype(np.int16)
    r = patch_bgr[:, :, 2].astype(np.int16)
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(np.int16)
    s = hsv[:, :, 1].astype(np.int16)
    v = hsv[:, :, 2].astype(np.int16)

    hsv_water = ((h >= 70) & (h <= 135) & (s >= 30) & (v >= 20)) | ((h >= 55) & (h <= 150) & (s >= 45) & (v >= 15))
    blue_dom = b >= (np.maximum(r, g) + 6)
    cyan_dom = (b >= (g - 4)) & (g >= (r + 3))
    not_bright_gray = ~((s <= 28) & (v >= 90))
    m = (hsv_water & (blue_dom | cyan_dom) & not_bright_gray).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = _select_primary_water_component(m, patch_bgr)
    return m


def _safe_water_mask(mask_u8: np.ndarray, margin_px: int = 3) -> np.ndarray:
    if margin_px <= 0:
        return mask_u8
    k = max(3, int(margin_px) * 2 + 1)
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask_u8, kernel, iterations=1)


def _select_primary_water_component(mask_u8: np.ndarray, patch_bgr: np.ndarray) -> np.ndarray:
    # Keep the most plausible "open water" connected component to avoid piers/roads false positives.
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return mask_u8

    h, w = mask_u8.shape[:2]
    b = patch_bgr[:, :, 0].astype(np.int16)
    g = patch_bgr[:, :, 1].astype(np.int16)
    r = patch_bgr[:, :, 2].astype(np.int16)
    blue_dom = b - np.maximum(g, r)

    best_id = -1
    best_score = -1.0
    for cid in range(1, num):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < max(80, (h * w) // 300):
            continue
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        cw = int(stats[cid, cv2.CC_STAT_WIDTH])
        ch = int(stats[cid, cv2.CC_STAT_HEIGHT])
        touches_border = x <= 0 or y <= 0 or (x + cw) >= (w - 1) or (y + ch) >= (h - 1)
        comp = labels == cid
        if int(comp.sum()) <= 0:
            continue
        mean_blue = float(np.mean(blue_dom[comp]))
        # Prefer large border-connected components with stronger blue dominance.
        score = float(area) + (5000.0 if touches_border else 0.0) + mean_blue * 40.0
        if score > best_score:
            best_score = score
            best_id = cid

    if best_id <= 0:
        return mask_u8
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    out[labels == best_id] = 255
    return out


def _water_with_clearance(mask_u8: np.ndarray, min_clearance_px: int) -> np.ndarray:
    if min_clearance_px <= 0:
        return mask_u8
    binary = (mask_u8 > 0).astype(np.uint8)
    if int(binary.sum()) <= 0:
        return mask_u8
    dist = cv2.distanceTransform(binary, distanceType=cv2.DIST_L2, maskSize=3)
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    out[dist >= float(min_clearance_px)] = 255
    return out


def _harmonize_overlay_to_background(
    bg_bgr: np.ndarray,
    overlay_bgra: np.ndarray,
    center_x: float,
    center_y: float,
    strength: float = 0.35,
) -> np.ndarray:
    """Match overlay tone to local background so pasted targets look less synthetic."""
    out = overlay_bgra.copy()
    h, w = bg_bgr.shape[:2]
    oh, ow = out.shape[:2]
    x1 = int(round(center_x - ow / 2))
    y1 = int(round(center_y - oh / 2))
    x2 = x1 + ow
    y2 = y1 + oh

    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(w, x2)
    iy2 = min(h, y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return out

    ox1 = ix1 - x1
    oy1 = iy1 - y1
    ox2 = ox1 + (ix2 - ix1)
    oy2 = oy1 + (iy2 - iy1)

    fg_bgr = out[oy1:oy2, ox1:ox2, :3].astype(np.float32)
    fg_alpha = out[oy1:oy2, ox1:ox2, 3]
    mask = fg_alpha > 10
    if int(mask.sum()) < 12:
        return out

    bg_local = bg_bgr[iy1:iy2, ix1:ix2, :3].astype(np.float32)
    fg_pix = fg_bgr[mask]
    bg_pix = bg_local[mask]
    fg_mean = fg_pix.mean(axis=0)
    bg_mean = bg_pix.mean(axis=0)
    fg_std = fg_pix.std(axis=0) + 1e-3
    bg_std = bg_pix.std(axis=0) + 1e-3

    matched = (fg_bgr - fg_mean) / fg_std * bg_std + bg_mean
    fg_bgr = fg_bgr * (1.0 - float(strength)) + matched * float(strength)
    fg_bgr = np.clip(fg_bgr, 0.0, 255.0).astype(np.uint8)
    out[oy1:oy2, ox1:ox2, :3] = fg_bgr

    # Slightly soften alpha edge to reduce hard cut-out boundary.
    alpha_f = out[:, :, 3].astype(np.float32)
    alpha_blur = cv2.GaussianBlur(alpha_f, (0, 0), sigmaX=0.5, sigmaY=0.5)
    out[:, :, 3] = np.clip(alpha_blur, 0.0, 255.0).astype(np.uint8)
    return out


def _snap_to_water(mask_u8: np.ndarray, x: float, y: float, max_radius: int = 64) -> tuple[float, float]:
    h, w = mask_u8.shape[:2]
    xi = int(np.clip(round(x), 0, w - 1))
    yi = int(np.clip(round(y), 0, h - 1))
    if mask_u8[yi, xi] > 0:
        return float(xi), float(yi)
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return float(xi), float(yi)
    d2 = (xs - xi) * (xs - xi) + (ys - yi) * (ys - yi)
    idx = int(np.argmin(d2))
    if d2[idx] > max_radius * max_radius:
        return float(xi), float(yi)
    return float(xs[idx]), float(ys[idx])


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

    alpha = overlay_bgra[oy1:oy2, ox1:ox2, 3]
    fg = alpha > 10
    denom = int(fg.sum())
    if denom <= 0:
        return 0.0
    water = mask_u8[iy1:iy2, ix1:ix2] > 0
    num = int((fg & water).sum())
    return float(num / float(denom))


def _alpha_overlap_ratio(mask_u8: np.ndarray, overlay_bgra: np.ndarray, center_x: float, center_y: float) -> float:
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

    alpha = overlay_bgra[oy1:oy2, ox1:ox2, 3]
    fg = alpha > 10
    denom = int(fg.sum())
    if denom <= 0:
        return 0.0
    m = mask_u8[iy1:iy2, ix1:ix2] > 0
    num = int((fg & m).sum())
    return float(num / float(denom))


def _overlay_visibility(overlay_bgra: np.ndarray, center_x: float, center_y: float, w: int, h: int) -> float:
    oh, ow = overlay_bgra.shape[:2]
    x1 = int(round(center_x - ow / 2))
    y1 = int(round(center_y - oh / 2))
    x2 = x1 + ow
    y2 = y1 + oh

    alpha = overlay_bgra[:, :, 3] > 10
    total = int(alpha.sum())
    if total <= 0:
        return 0.0

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
    visible = int((overlay_bgra[oy1:oy2, ox1:ox2, 3] > 10).sum())
    return float(visible / float(total))


def _sample_water_center(
    mask_u8: np.ndarray,
    overlay_bgra: np.ndarray,
    x0: float,
    y0: float,
    rng: np.random.Generator,
    min_ratio: float = 0.85,
    tries: int = 24,
    local_radius: int = 24,
    global_fallback: bool = False,
) -> tuple[float, float]:
    h, w = mask_u8.shape[:2]
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return float(np.clip(x0, 0, w - 1)), float(np.clip(y0, 0, h - 1))

    best_x = float(np.clip(x0, 0, w - 1))
    best_y = float(np.clip(y0, 0, h - 1))
    best_score = -1.0

    # Candidate 1: nearest water center.
    nx, ny = _snap_to_water(mask_u8, x0, y0, max_radius=max(h, w))
    candidates = [(nx, ny)]
    # Candidate 2+: prefer local water pixels around current center.
    lx1 = max(0, int(round(x0)) - int(local_radius))
    ly1 = max(0, int(round(y0)) - int(local_radius))
    lx2 = min(w, int(round(x0)) + int(local_radius) + 1)
    ly2 = min(h, int(round(y0)) + int(local_radius) + 1)
    local = mask_u8[ly1:ly2, lx1:lx2] > 0
    lys, lxs = np.where(local)
    if len(lxs) > 0:
        n_local = min(tries, len(lxs))
        idx = rng.choice(len(lxs), size=n_local, replace=False)
        for i in idx:
            candidates.append((float(lxs[int(i)] + lx1), float(lys[int(i)] + ly1)))
    elif global_fallback:
        n = min(tries, len(xs))
        idx = rng.choice(len(xs), size=n, replace=False)
        for i in idx:
            candidates.append((float(xs[int(i)]), float(ys[int(i)])))

    for cx, cy in candidates:
        ratio = _alpha_water_ratio(mask_u8, overlay_bgra, cx, cy)
        dist = float(np.hypot(cx - x0, cy - y0))
        score = ratio - 0.0015 * dist
        if score > best_score:
            best_score = score
            best_x, best_y = cx, cy
        if ratio >= min_ratio:
            return cx, cy
    return best_x, best_y


def _random_water_point(mask_u8: np.ndarray, rng: np.random.Generator) -> tuple[float, float] | None:
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return None
    i = int(rng.integers(0, len(xs)))
    return float(xs[i]), float(ys[i])


def _bbox_from_center(overlay_bgra: np.ndarray, center_x: float, center_y: float, w: int, h: int) -> tuple[int, int, int, int]:
    oh, ow = overlay_bgra.shape[:2]
    x1 = int(round(center_x - ow / 2))
    y1 = int(round(center_y - oh / 2))
    x2 = x1 + ow
    y2 = y1 + oh
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(w, x2)
    iy2 = min(h, y2)
    return (ix1, iy1, max(0, ix2 - ix1), max(0, iy2 - iy1))


def _iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    ua = float(max(1, aw * ah) + max(1, bw * bh) - inter)
    return inter / ua


@dataclass
class RenderSplitResult:
    split: str
    sequences: int
    frames: int
    used_background_ids: set[str]
    used_target_ids: set[str]
    used_distractor_ids: set[str]
    label_path: Path


class Stage2Renderer:
    def __init__(
        self,
        cfg: dict,
        registry: AssetRegistry,
        project_root: Path,
        output_root: Path,
        rng: np.random.Generator,
    ):
        self.cfg = cfg
        self.registry = registry
        self.project_root = project_root
        self.output_root = output_root
        self.rng = rng

    def _stage_for_frame(self, frame_idx: int, frames_per_sequence: int) -> str:
        ratio = float(frame_idx) / max(1.0, float(frames_per_sequence - 1))
        if ratio < 0.4:
            return "far"
        if ratio < 0.75:
            return "mid"
        return "terminal"

    def _image_rel_path(self, split: str, seq_id: int, frame_id: int) -> str:
        image_abs = self.output_root / "images" / split / f"seq_{seq_id:04d}_frame_{frame_id:04d}.png"
        try:
            rel = image_abs.resolve().relative_to(self.project_root.resolve())
            return str(rel).replace("\\", "/")
        except Exception:
            # Fallback to absolute path when output_root is outside project_root.
            return str(image_abs.resolve()).replace("\\", "/")

    def _read_background(self, asset: AssetRecord) -> np.ndarray:
        img = cv2.imread(str(asset.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read background: {asset.path}")
        return img

    @staticmethod
    def _mask_ratio(mask_u8: np.ndarray) -> float:
        if mask_u8.size <= 0:
            return 0.0
        return float((mask_u8 > 0).mean())

    def _sample_background(self, split: str) -> AssetRecord:
        # Conservative stage2 smoke rule: avoid "port" backgrounds until dedicated harbor-water masking is ready.
        pool = [a for a in self.registry.get("background", split) if str(a.category).lower() != "port"]
        if not pool:
            pool = self.registry.get("background", split)
        if not pool:
            raise ValueError(f"No background assets for split={split}")
        idx = int(self.rng.integers(0, len(pool)))
        return pool[idx]

    def render_split(self, split: str, num_sequences: int) -> RenderSplitResult:
        ds_name = str(self.output_root.name)
        frames_per_sequence = int(self.cfg["dataset"]["frames_per_sequence"])
        image_size = int(self.cfg["dataset"]["image_size"])
        world_size_m = float(self.cfg["dataset"]["world_size_m"])
        motion_probs = dict(self.cfg["motion_modes"])
        perturb_cfg = dict(self.cfg["perturbations"])
        distractor_cfg = dict(self.cfg["distractors"])
        stages_cfg = dict(self.cfg["stages"])
        target_cfg = dict(self.cfg.get("target", {}))
        continuity_cfg = dict(self.cfg.get("continuity", {}))
        placement_cfg = dict(self.cfg.get("placement", {}))
        max_pos_step_px = float(continuity_cfg.get("max_position_shift_ratio", 0.15)) * float(image_size)
        max_crop_shift_ratio = float(continuity_cfg.get("max_crop_shift_ratio", 0.15))
        max_scale_change_ratio = float(continuity_cfg.get("max_scale_change_ratio", 0.05))
        max_angle_change_deg = float(continuity_cfg.get("max_angle_change_deg", 12.0))
        min_visibility = float(placement_cfg.get("min_visibility", 0.35))
        max_truncation_ratio = float(placement_cfg.get("max_truncation_ratio", 0.20))
        require_water_mask = bool(placement_cfg.get("require_water_mask", True))
        max_land_overlap = float(placement_cfg.get("max_land_overlap", 0.0))
        max_shore_overlap = float(placement_cfg.get("max_shore_buffer_overlap", 0.05))
        min_shore_clearance_px = int(placement_cfg.get("min_shore_clearance_px", 5))
        allowed_target_categories = {str(x) for x in target_cfg.get("allowed_categories", ["boat_top", "boat_oblique"])}
        heading_noise_deg = float(target_cfg.get("heading_noise_deg", 10.0))

        images_dir = self.output_root / "images" / split
        labels_dir = self.output_root / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_path = labels_dir / f"{split}.jsonl"

        used_background_ids: set[str] = set()
        used_target_ids: set[str] = set()
        used_distractor_ids: set[str] = set()

        with label_path.open("w", encoding="utf-8") as f:
            for seq_idx in range(num_sequences):
                # Ensure chosen background contains enough water area for maritime rendering.
                bg = None
                bg_img = None
                bg_water_global = None
                for _ in range(24):
                    cand = self._sample_background(split)
                    cand_img = self._read_background(cand)
                    cand_water = _safe_water_mask(_water_mask(cand_img), margin_px=6)
                    # Require enough deep-water area (away from shore) to avoid unavoidable beaching.
                    cand_core_water = _water_with_clearance(
                        cand_water, int(max(4, placement_cfg.get("min_shore_clearance_px", 5)))
                    )
                    if self._mask_ratio(cand_water) >= 0.03 and self._mask_ratio(cand_core_water) >= 0.02:
                        bg = cand
                        bg_img = cand_img
                        bg_water_global = cand_water
                        break
                if bg is None or bg_img is None or bg_water_global is None:
                    raise RuntimeError(f"No usable water background found for split={split}")
                bg_target_water_global = _water_with_clearance(
                    bg_water_global, int(max(4, placement_cfg.get("min_shore_clearance_px", 5)))
                )
                d_num = int(self.rng.integers(int(distractor_cfg["min_count"]), int(distractor_cfg["max_count"]) + 1))
                distractors = self.registry.sample_many("distractor", split, d_num, self.rng)

                used_background_ids.add(bg.asset_id)
                for d in distractors:
                    used_distractor_ids.add(d.asset_id)

                # Enforce allowed target viewpoints for aerial realism.
                split_targets = self.registry.get("target", split)
                allowed_targets = [a for a in split_targets if str(a.category) in allowed_target_categories]
                if not allowed_targets:
                    raise RuntimeError(
                        f"No allowed target templates in split={split}; expected categories={sorted(allowed_target_categories)}"
                    )
                target = allowed_targets[int(self.rng.integers(0, len(allowed_targets)))]
                used_target_ids.add(target.asset_id)
                target_bgra = read_bgra(target.path)
                distractor_bgras = [read_bgra(d.path) for d in distractors]

                mode = sample_mode(motion_probs, self.rng)
                # Start with far-stage dt; the per-frame gsd/dt is still saved in labels.
                motion = generate_motion_sequence(mode, frames_per_sequence, dt=1.0, world_size_m=world_size_m, rng=self.rng)
                prev_target_xy: tuple[float, float] | None = None
                prev_crop_center: tuple[float, float] | None = None
                prev_scale: float | None = None
                prev_angle_deg: float | None = None
                prev_valid_patch: np.ndarray | None = None
                prev_valid_bbox: tuple[int, int, int, int] | None = None
                prev_valid_vis: float | None = None
                prev_valid_tx_ty: tuple[float, float] | None = None
                prev_valid_land: float | None = None
                prev_valid_shore: float | None = None
                prev_valid_trunc: float | None = None
                prev_valid_angle: float | None = None
                prev_valid_scale: float | None = None
                # Keep target scale stable per sequence by default (more realistic temporal continuity).
                fixed_scale_range = target_cfg.get("fixed_scale_range", [0.14, 0.20])
                seq_target_scale = float(self.rng.uniform(float(fixed_scale_range[0]), float(fixed_scale_range[1])))
                scale_mode = str(target_cfg.get("scale_mode", "fixed_sequence"))

                for frame_idx, state in enumerate(motion):
                    stage_name = self._stage_for_frame(frame_idx, frames_per_sequence)
                    stage_cfg = dict(stages_cfg[stage_name])
                    gsd = float(stage_cfg["gsd_m_per_px"])
                    center_jitter_px = float(stage_cfg["center_jitter_px"])

                    crop_center_x = state.x + float(self.rng.normal(0.0, center_jitter_px * gsd))
                    crop_center_y = state.y + float(self.rng.normal(0.0, center_jitter_px * gsd))
                    crop_center_x = float(np.clip(crop_center_x, 0.0, world_size_m))
                    crop_center_y = float(np.clip(crop_center_y, 0.0, world_size_m))
                    if prev_crop_center is not None:
                        pcx, pcy = prev_crop_center
                        dx = crop_center_x - pcx
                        dy = crop_center_y - pcy
                        d = float(np.hypot(dx, dy))
                        max_crop_step_m = max_crop_shift_ratio * float(image_size) * gsd
                        if d > max_crop_step_m and d > 1e-6:
                            s = max_crop_step_m / d
                            crop_center_x = pcx + dx * s
                            crop_center_y = pcy + dy * s

                    bg_cx, bg_cy = world_to_background_px(crop_center_x, crop_center_y, world_size_m, bg_img.shape[1], bg_img.shape[0])
                    patch = _extract_patch(bg_img, bg_cx, bg_cy, image_size)
                    water = _extract_patch(bg_water_global, bg_cx, bg_cy, image_size)
                    if water.ndim == 3:
                        water = water[:, :, 0]
                    water = water.astype(np.uint8)

                    # If local patch has too little water, move crop center to nearest global water.
                    if self._mask_ratio(water) < 0.20:
                        tgt_bg_x, tgt_bg_y = world_to_background_px(state.x, state.y, world_size_m, bg_img.shape[1], bg_img.shape[0])
                        nwx, nwy = _snap_to_water(bg_water_global, tgt_bg_x, tgt_bg_y, max_radius=max(bg_img.shape[0], bg_img.shape[1]))
                        crop_center_x, crop_center_y = background_px_to_world(nwx, nwy, world_size_m, bg_img.shape[1], bg_img.shape[0])
                        if prev_crop_center is not None:
                            pcx, pcy = prev_crop_center
                            dx = crop_center_x - pcx
                            dy = crop_center_y - pcy
                            d = float(np.hypot(dx, dy))
                            max_crop_step_m = max_crop_shift_ratio * float(image_size) * gsd
                            if d > max_crop_step_m and d > 1e-6:
                                s = max_crop_step_m / d
                                crop_center_x = pcx + dx * s
                                crop_center_y = pcy + dy * s
                                nwx, nwy = world_to_background_px(
                                    crop_center_x, crop_center_y, world_size_m, bg_img.shape[1], bg_img.shape[0]
                                )
                        bg_cx, bg_cy = nwx, nwy
                        patch = _extract_patch(bg_img, bg_cx, bg_cy, image_size)
                        water = _extract_patch(bg_water_global, bg_cx, bg_cy, image_size)
                        if water.ndim == 3:
                            water = water[:, :, 0]
                        water = water.astype(np.uint8)

                    tx, ty = world_to_image(state.x, state.y, crop_center_x, crop_center_y, gsd, image_size)
                    if scale_mode == "stage_progressive":
                        scale_min, scale_max = stage_cfg["target_scale_range"]
                        target_scale = float(self.rng.uniform(float(scale_min), float(scale_max)))
                    else:
                        target_scale = seq_target_scale
                    if prev_scale is not None:
                        s_lo = prev_scale * (1.0 - max_scale_change_ratio)
                        s_hi = prev_scale * (1.0 + max_scale_change_ratio)
                        target_scale = float(np.clip(target_scale, s_lo, s_hi))
                    target_scale = float(max(0.01, target_scale))
                    target_patch = resize_bgra_with_scale(target_bgra, target_scale, image_size=image_size)
                    heading_deg = float(np.degrees(np.arctan2(state.vy, state.vx)))
                    angle_deg = heading_deg + float(self.rng.uniform(-heading_noise_deg, heading_noise_deg))
                    if prev_angle_deg is not None:
                        d_ang = angle_deg - prev_angle_deg
                        while d_ang > 180.0:
                            d_ang -= 360.0
                        while d_ang < -180.0:
                            d_ang += 360.0
                        d_ang = float(np.clip(d_ang, -max_angle_change_deg, max_angle_change_deg))
                        angle_deg = prev_angle_deg + d_ang
                    target_patch = rotate_bgra(target_patch, angle_deg)

                    # Keep target away from shoreline to avoid unrealistic "beaching" samples.
                    th, tw = target_patch.shape[:2]
                    shoreline_margin = int(max(min_shore_clearance_px, np.clip(round(0.55 * max(th, tw)), 4, 48)))
                    target_water = _extract_patch(bg_target_water_global, bg_cx, bg_cy, image_size)
                    if target_water.ndim == 3:
                        target_water = target_water[:, :, 0]
                    target_water = target_water.astype(np.uint8)
                    # If local deep-water mask is too small, recenter crop to nearest global deep-water point.
                    if self._mask_ratio(target_water) < 0.01:
                        tgt_bg_x, tgt_bg_y = world_to_background_px(state.x, state.y, world_size_m, bg_img.shape[1], bg_img.shape[0])
                        nwx, nwy = _snap_to_water(
                            bg_target_water_global, tgt_bg_x, tgt_bg_y, max_radius=max(bg_img.shape[0], bg_img.shape[1])
                        )
                        crop_center_x, crop_center_y = background_px_to_world(nwx, nwy, world_size_m, bg_img.shape[1], bg_img.shape[0])
                        if prev_crop_center is not None:
                            pcx, pcy = prev_crop_center
                            dx = crop_center_x - pcx
                            dy = crop_center_y - pcy
                            d = float(np.hypot(dx, dy))
                            max_crop_step_m = max_crop_shift_ratio * float(image_size) * gsd
                            if d > max_crop_step_m and d > 1e-6:
                                s = max_crop_step_m / d
                                crop_center_x = pcx + dx * s
                                crop_center_y = pcy + dy * s
                                nwx, nwy = world_to_background_px(
                                    crop_center_x, crop_center_y, world_size_m, bg_img.shape[1], bg_img.shape[0]
                                )
                        bg_cx, bg_cy = nwx, nwy
                        patch = _extract_patch(bg_img, bg_cx, bg_cy, image_size)
                        water = _extract_patch(bg_water_global, bg_cx, bg_cy, image_size)
                        if water.ndim == 3:
                            water = water[:, :, 0]
                        water = water.astype(np.uint8)
                        target_water = _extract_patch(bg_target_water_global, bg_cx, bg_cy, image_size)
                        if target_water.ndim == 3:
                            target_water = target_water[:, :, 0]
                        target_water = target_water.astype(np.uint8)
                        tx, ty = world_to_image(state.x, state.y, crop_center_x, crop_center_y, gsd, image_size)
                    if self._mask_ratio(target_water) < 0.01:
                        target_water = _water_with_clearance(water, shoreline_margin)
                    if self._mask_ratio(target_water) < 0.01:
                        target_water = water
                    tx, ty = _sample_water_center(
                        target_water,
                        target_patch,
                        tx,
                        ty,
                        self.rng,
                        min_ratio=0.985,
                        tries=64,
                        local_radius=24,
                        global_fallback=False,
                    )
                    # Keep temporal continuity: strict per-frame jump cap.
                    if prev_target_xy is not None:
                        px, py = prev_target_xy
                        dx, dy = tx - px, ty - py
                        dist = float(np.hypot(dx, dy))
                        max_step = min(float(target_cfg.get("max_step_px", 16.0)), max_pos_step_px)
                        if dist > max_step and dist > 1e-6:
                            s = max_step / dist
                            tx = px + dx * s
                            ty = py + dy * s
                            tx, ty = _sample_water_center(
                                target_water,
                                target_patch,
                                tx,
                                ty,
                                self.rng,
                                min_ratio=0.98,
                                tries=32,
                                local_radius=20,
                                global_fallback=False,
                            )
                    # Hard constraint: target must remain on water.
                    if _alpha_water_ratio(target_water, target_patch, tx, ty) < 0.98:
                        if prev_target_xy is not None:
                            tx, ty = _sample_water_center(
                                target_water,
                                target_patch,
                                prev_target_xy[0],
                                prev_target_xy[1],
                                self.rng,
                                min_ratio=0.99,
                                tries=64,
                                local_radius=20,
                                global_fallback=False,
                            )
                        else:
                            tx, ty = _sample_water_center(
                                target_water,
                                target_patch,
                                tx,
                                ty,
                                self.rng,
                                min_ratio=0.99,
                                tries=64,
                                local_radius=20,
                                global_fallback=False,
                            )
                    # Final hard continuity guard: never allow large frame-to-frame pixel jump.
                    if prev_target_xy is not None:
                        px, py = prev_target_xy
                        final_step = float(np.hypot(tx - px, ty - py))
                        max_step = min(float(target_cfg.get("max_step_px", 8.0)), max_pos_step_px)
                        if final_step > max_step:
                            tx, ty = px, py

                    # Final hard placement guard: target center must be on water.
                    iy = int(np.clip(round(ty), 0, image_size - 1))
                    ix = int(np.clip(round(tx), 0, image_size - 1))
                    if int(target_water[iy, ix]) <= 0:
                        max_step = float(target_cfg.get("max_step_px", 8.0))
                        local_snap_radius = int(max(6, round(max_step * 2.0)))
                        tx2, ty2 = _snap_to_water(target_water, tx, ty, max_radius=local_snap_radius)
                        iy2 = int(np.clip(round(ty2), 0, image_size - 1))
                        ix2 = int(np.clip(round(tx2), 0, image_size - 1))
                        if int(target_water[iy2, ix2]) > 0:
                            tx, ty = tx2, ty2
                        elif prev_target_xy is not None:
                            px, py = prev_target_xy
                            py_i = int(np.clip(round(py), 0, image_size - 1))
                            px_i = int(np.clip(round(px), 0, image_size - 1))
                            if int(target_water[py_i, px_i]) > 0:
                                tx, ty = px, py
                            # else: keep tx,ty as-is to avoid sudden teleports.

                    water_binary = (water > 0).astype(np.uint8)
                    land_mask = ((water_binary == 0).astype(np.uint8)) * 255
                    shore_mask = (((water_binary > 0) & (cv2.distanceTransform(water_binary, cv2.DIST_L2, 3) < float(shoreline_margin))).astype(np.uint8)) * 255

                    # Placement retry under hard semantic constraints.
                    for _ in range(12):
                        land_overlap_ratio = _alpha_overlap_ratio(land_mask, target_patch, tx, ty)
                        shore_overlap_ratio = _alpha_overlap_ratio(shore_mask, target_patch, tx, ty)
                        if land_overlap_ratio <= max_land_overlap and shore_overlap_ratio <= max_shore_overlap:
                            break
                        anchor_x, anchor_y = (prev_target_xy[0], prev_target_xy[1]) if prev_target_xy is not None else (tx, ty)
                        max_step = min(float(target_cfg.get("max_step_px", 8.0)), max_pos_step_px)
                        tx, ty = _sample_water_center(
                            target_water,
                            target_patch,
                            anchor_x,
                            anchor_y,
                            self.rng,
                            min_ratio=0.99,
                            tries=48,
                            local_radius=int(max(6, round(max_step))),
                            global_fallback=False,
                        )

                    land_overlap_ratio = _alpha_overlap_ratio(land_mask, target_patch, tx, ty)
                    shore_overlap_ratio = _alpha_overlap_ratio(shore_mask, target_patch, tx, ty)
                    pre_vis = _overlay_visibility(target_patch, tx, ty, image_size, image_size)
                    truncation_ratio = max(0.0, 1.0 - pre_vis)
                    max_step = min(float(target_cfg.get("max_step_px", 8.0)), max_pos_step_px)
                    is_semantic_ok = (
                        float(land_overlap_ratio) <= float(max_land_overlap)
                        and float(shore_overlap_ratio) <= float(max_shore_overlap)
                        and float(pre_vis) >= float(min_visibility)
                        and float(truncation_ratio) <= float(max_truncation_ratio)
                        and (not require_water_mask or _alpha_water_ratio(water, target_patch, tx, ty) >= 0.96)
                    )
                    if not is_semantic_ok:
                        for _ in range(96):
                            p = _random_water_point(target_water, self.rng)
                            if p is None:
                                break
                            cx, cy = _sample_water_center(
                                target_water,
                                target_patch,
                                p[0],
                                p[1],
                                self.rng,
                                min_ratio=0.99,
                                tries=24,
                                local_radius=int(max(6, round(max_step))),
                                global_fallback=False,
                            )
                            if prev_target_xy is not None:
                                d = float(np.hypot(cx - prev_target_xy[0], cy - prev_target_xy[1]))
                                if d > max_step + 1e-6:
                                    continue
                            l = _alpha_overlap_ratio(land_mask, target_patch, cx, cy)
                            s = _alpha_overlap_ratio(shore_mask, target_patch, cx, cy)
                            v = _overlay_visibility(target_patch, cx, cy, image_size, image_size)
                            t = max(0.0, 1.0 - v)
                            if (
                                float(l) <= float(max_land_overlap)
                                and float(s) <= float(max_shore_overlap)
                                and float(v) >= float(min_visibility)
                                and float(t) <= float(max_truncation_ratio)
                                and (not require_water_mask or _alpha_water_ratio(water, target_patch, cx, cy) >= 0.96)
                            ):
                                tx, ty = cx, cy
                                land_overlap_ratio = l
                                shore_overlap_ratio = s
                                pre_vis = v
                                truncation_ratio = t
                                break
                    is_semantic_ok = (
                        float(land_overlap_ratio) <= float(max_land_overlap)
                        and float(shore_overlap_ratio) <= float(max_shore_overlap)
                        and float(pre_vis) >= float(min_visibility)
                        and float(truncation_ratio) <= float(max_truncation_ratio)
                        and (not require_water_mask or _alpha_water_ratio(water, target_patch, tx, ty) >= 0.96)
                    )
                    if (not is_semantic_ok) and (prev_target_xy is not None):
                        px, py = prev_target_xy
                        l = _alpha_overlap_ratio(land_mask, target_patch, px, py)
                        s = _alpha_overlap_ratio(shore_mask, target_patch, px, py)
                        v = _overlay_visibility(target_patch, px, py, image_size, image_size)
                        t = max(0.0, 1.0 - v)
                        if (
                            float(l) <= float(max_land_overlap)
                            and float(s) <= float(max_shore_overlap)
                            and float(v) >= float(min_visibility)
                            and float(t) <= float(max_truncation_ratio)
                            and (not require_water_mask or _alpha_water_ratio(water, target_patch, px, py) >= 0.96)
                        ):
                            tx, ty = px, py
                            land_overlap_ratio = l
                            shore_overlap_ratio = s
                            pre_vis = v
                            truncation_ratio = t

                    # Absolute last guard: forbid large frame-to-frame pixel jumps.
                    if prev_target_xy is not None:
                        px, py = prev_target_xy
                        d = float(np.hypot(tx - px, ty - py))
                        max_step = min(float(target_cfg.get("max_step_px", 8.0)), max_pos_step_px)
                        if d > max_step + 1e-6:
                            tx, ty = px, py
                            land_overlap_ratio = _alpha_overlap_ratio(land_mask, target_patch, tx, ty)
                            shore_overlap_ratio = _alpha_overlap_ratio(shore_mask, target_patch, tx, ty)
                            pre_vis = _overlay_visibility(target_patch, tx, ty, image_size, image_size)
                            truncation_ratio = max(0.0, 1.0 - pre_vis)

                    target_patch = _harmonize_overlay_to_background(patch, target_patch, tx, ty, strength=0.35)
                    bbox, vis = alpha_blend_center(patch, target_patch, tx, ty)
                    obs_valid = (
                        (not require_water_mask or _alpha_water_ratio(water, target_patch, tx, ty) >= 0.96)
                        and float(land_overlap_ratio) <= float(max_land_overlap)
                        and float(shore_overlap_ratio) <= float(max_shore_overlap)
                        and float(vis) >= float(min_visibility)
                        and float(truncation_ratio) <= float(max_truncation_ratio)
                    )

                    # Absolute write-time fallback:
                    # if this frame violates hard constraints or jumps too far, reuse last valid frame.
                    max_step = min(float(target_cfg.get("max_step_px", 8.0)), max_pos_step_px)
                    step_fail = False
                    if prev_valid_tx_ty is not None:
                        step_fail = float(np.hypot(tx - prev_valid_tx_ty[0], ty - prev_valid_tx_ty[1])) > (max_step + 1e-6)
                    hard_fail = (
                        float(land_overlap_ratio) > float(max_land_overlap)
                        or float(shore_overlap_ratio) > float(max_shore_overlap)
                        or float(vis) < float(min_visibility)
                        or float(truncation_ratio) > float(max_truncation_ratio)
                        or (
                            require_water_mask
                            and _alpha_water_ratio(water, target_patch, tx, ty) < 0.96
                        )
                    )
                    if (hard_fail or step_fail) and prev_valid_patch is not None and prev_valid_bbox is not None and prev_valid_vis is not None and prev_valid_tx_ty is not None:
                        patch = prev_valid_patch.copy()
                        tx, ty = prev_valid_tx_ty
                        bbox = prev_valid_bbox
                        vis = float(prev_valid_vis)
                        land_overlap_ratio = float(prev_valid_land if prev_valid_land is not None else 0.0)
                        shore_overlap_ratio = float(prev_valid_shore if prev_valid_shore is not None else 0.0)
                        truncation_ratio = float(prev_valid_trunc if prev_valid_trunc is not None else 0.0)
                        angle_deg = float(prev_valid_angle if prev_valid_angle is not None else angle_deg)
                        target_scale = float(prev_valid_scale if prev_valid_scale is not None else target_scale)
                        obs_valid = True

                    if bool(obs_valid):
                        prev_valid_patch = patch.copy()
                        prev_valid_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        prev_valid_vis = float(vis)
                        prev_valid_tx_ty = (float(tx), float(ty))
                        prev_valid_land = float(land_overlap_ratio)
                        prev_valid_shore = float(shore_overlap_ratio)
                        prev_valid_trunc = float(truncation_ratio)
                        prev_valid_angle = float(angle_deg)
                        prev_valid_scale = float(target_scale)

                    prev_target_xy = (tx, ty)
                    prev_crop_center = (crop_center_x, crop_center_y)
                    prev_scale = target_scale
                    prev_angle_deg = angle_deg

                    d_ids: list[str] = []
                    placed_bboxes: list[tuple[int, int, int, int]] = [bbox]
                    d_min, d_max = distractor_cfg["scale_range"]
                    for d_asset, d_img in zip(distractors, distractor_bgras):
                        d_scale = float(self.rng.uniform(float(d_min), float(d_max)))
                        d_patch = resize_bgra_with_scale(d_img, d_scale, image_size=image_size)
                        placed = False
                        for _ in range(24):
                            p = _random_water_point(water, self.rng)
                            if p is None:
                                break
                            d_water = _water_with_clearance(water, min_clearance_px=2)
                            if self._mask_ratio(d_water) < 0.01:
                                d_water = water
                            d_center_x, d_center_y = _sample_water_center(
                                d_water,
                                d_patch,
                                p[0],
                                p[1],
                                self.rng,
                                min_ratio=0.96,
                                tries=24,
                                local_radius=28,
                                global_fallback=True,
                            )
                            if _alpha_water_ratio(water, d_patch, d_center_x, d_center_y) < 0.96:
                                continue
                            d_bbox = _bbox_from_center(d_patch, d_center_x, d_center_y, image_size, image_size)
                            # No collision/overlap with target and already placed distractors.
                            if any(_iou_xywh(d_bbox, b2) > 0.02 for b2 in placed_bboxes):
                                continue
                            d_patch = _harmonize_overlay_to_background(patch, d_patch, d_center_x, d_center_y, strength=0.30)
                            alpha_blend_center(patch, d_patch, d_center_x, d_center_y)
                            d_ids.append(d_asset.asset_id)
                            placed_bboxes.append(d_bbox)
                            placed = True
                            break
                        if not placed:
                            continue

                    patch = apply_perturbations(patch, perturb_cfg, self.rng)

                    rel_img_path = self._image_rel_path(split, seq_idx, frame_idx)
                    abs_img_path = self.project_root / Path(rel_img_path)
                    abs_img_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(abs_img_path), patch)

                    row = RenderedFrameRecord(
                        image_path=str(rel_img_path).replace("\\", "/"),
                        split=split,
                        sequence_id=f"{split}_{seq_idx:04d}",
                        frame_id=f"{frame_idx:04d}",
                        stage=stage_name,
                        observation_source=str(stage_cfg["observation_source"]),
                        gsd_m_per_px=gsd,
                        target_center_px=[float(tx), float(ty)],
                        bbox_xywh=[float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                        visibility=float(vis),
                        background_asset_id=bg.asset_id,
                        target_asset_id=target.asset_id,
                        distractor_asset_ids=d_ids,
                        motion_mode=mode,
                        land_overlap_ratio=float(land_overlap_ratio),
                        shore_buffer_overlap_ratio=float(shore_overlap_ratio),
                        scale_px=float(max(target_patch.shape[0], target_patch.shape[1])),
                        angle_deg=float(angle_deg),
                        obs_valid=bool(obs_valid),
                        meta={
                            "dataset_name": ds_name,
                            "crop_center_world": [crop_center_x, crop_center_y],
                            "crop_center_world_x": float(crop_center_x),
                            "crop_center_world_y": float(crop_center_y),
                            "target_state_world": {
                                "x": float(state.x),
                                "y": float(state.y),
                                "vx": float(state.vx),
                                "vy": float(state.vy),
                                "heading_deg": float(heading_deg),
                            },
                            "target_world_x": float(state.x),
                            "target_world_y": float(state.y),
                            "target_world_vx": float(state.vx),
                            "target_world_vy": float(state.vy),
                            "target_heading_deg": float(heading_deg),
                            "center_x": float(tx),
                            "center_y": float(ty),
                            "bbox_xywh": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                            "visibility": float(vis),
                            "gsd": float(gsd),
                            "perception_stage": str(stage_name),
                            "land_overlap_ratio": float(land_overlap_ratio),
                            "shore_buffer_overlap_ratio": float(shore_overlap_ratio),
                            "scale_px": float(max(target_patch.shape[0], target_patch.shape[1])),
                            "scale_factor": float(target_scale),
                            "angle_deg": float(angle_deg),
                            "obs_valid": bool(obs_valid),
                        },
                    )
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

        return RenderSplitResult(
            split=split,
            sequences=num_sequences,
            frames=num_sequences * frames_per_sequence,
            used_background_ids=used_background_ids,
            used_target_ids=used_target_ids,
            used_distractor_ids=used_distractor_ids,
            label_path=label_path,
        )
