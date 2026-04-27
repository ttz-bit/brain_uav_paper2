from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from paper2.render.asset_registry import AssetRecord, AssetRegistry
from paper2.render.compositor import alpha_blend_center, read_bgra, resize_bgra_with_scale, rotate_bgra
from paper2.render.coordinate_mapper import world_to_background_px, world_to_image
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
    # Lightweight heuristic: water tends to be blue/green with medium saturation.
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    m1 = (h >= 70) & (h <= 135) & (s >= 25) & (v >= 20)
    m2 = (h >= 50) & (h <= 160) & (s >= 35) & (v >= 15)
    m = (m1 | m2).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    return m


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


def _sample_water_center(
    mask_u8: np.ndarray,
    overlay_bgra: np.ndarray,
    x0: float,
    y0: float,
    rng: np.random.Generator,
    min_ratio: float = 0.85,
    tries: int = 24,
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
    # Candidate 2+: random water pixels.
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
        return f"data/rendered/{self.cfg['dataset']['name']}/images/{split}/seq_{seq_id:04d}_frame_{frame_id:04d}.png"

    def _read_background(self, asset: AssetRecord) -> np.ndarray:
        img = cv2.imread(str(asset.path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read background: {asset.path}")
        return img

    def render_split(self, split: str, num_sequences: int) -> RenderSplitResult:
        ds_name = str(self.cfg["dataset"]["name"])
        frames_per_sequence = int(self.cfg["dataset"]["frames_per_sequence"])
        image_size = int(self.cfg["dataset"]["image_size"])
        world_size_m = float(self.cfg["dataset"]["world_size_m"])
        motion_probs = dict(self.cfg["motion_modes"])
        perturb_cfg = dict(self.cfg["perturbations"])
        distractor_cfg = dict(self.cfg["distractors"])
        stages_cfg = dict(self.cfg["stages"])

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
                bg = self.registry.sample_one("background", split, self.rng)
                d_num = int(self.rng.integers(int(distractor_cfg["min_count"]), int(distractor_cfg["max_count"]) + 1))
                distractors = self.registry.sample_many("distractor", split, d_num, self.rng)

                used_background_ids.add(bg.asset_id)
                for d in distractors:
                    used_distractor_ids.add(d.asset_id)

                bg_img = self._read_background(bg)
                # Prefer top-view templates for aerial realism; fallback to any target if none available.
                split_targets = self.registry.get("target", split)
                top_targets = [a for a in split_targets if a.category == "boat_top"]
                if top_targets:
                    target = top_targets[int(self.rng.integers(0, len(top_targets)))]
                else:
                    target = self.registry.sample_one("target", split, self.rng)
                used_target_ids.add(target.asset_id)
                target_bgra = read_bgra(target.path)
                distractor_bgras = [read_bgra(d.path) for d in distractors]

                mode = sample_mode(motion_probs, self.rng)
                # Start with far-stage dt; the per-frame gsd/dt is still saved in labels.
                motion = generate_motion_sequence(mode, frames_per_sequence, dt=1.0, world_size_m=world_size_m, rng=self.rng)

                for frame_idx, state in enumerate(motion):
                    stage_name = self._stage_for_frame(frame_idx, frames_per_sequence)
                    stage_cfg = dict(stages_cfg[stage_name])
                    gsd = float(stage_cfg["gsd_m_per_px"])
                    center_jitter_px = float(stage_cfg["center_jitter_px"])

                    crop_center_x = state.x + float(self.rng.normal(0.0, center_jitter_px * gsd))
                    crop_center_y = state.y + float(self.rng.normal(0.0, center_jitter_px * gsd))
                    crop_center_x = float(np.clip(crop_center_x, 0.0, world_size_m))
                    crop_center_y = float(np.clip(crop_center_y, 0.0, world_size_m))

                    bg_cx, bg_cy = world_to_background_px(crop_center_x, crop_center_y, world_size_m, bg_img.shape[1], bg_img.shape[0])
                    patch = _extract_patch(bg_img, bg_cx, bg_cy, image_size)
                    water = _water_mask(patch)

                    tx, ty = world_to_image(state.x, state.y, crop_center_x, crop_center_y, gsd, image_size)
                    scale_min, scale_max = stage_cfg["target_scale_range"]
                    target_scale = float(self.rng.uniform(float(scale_min), float(scale_max)))
                    target_patch = resize_bgra_with_scale(target_bgra, target_scale, image_size=image_size)
                    # Only top-view templates are rotated by heading to avoid "capsized" side-view artifacts.
                    if target.category == "boat_top":
                        heading_deg = float(np.degrees(np.arctan2(state.vy, state.vx)))
                        target_patch = rotate_bgra(target_patch, heading_deg)
                    tx, ty = _sample_water_center(water, target_patch, tx, ty, self.rng, min_ratio=0.96, tries=48)
                    # Hard constraint: target must stay on water; fallback to random water tries.
                    t_ratio = _alpha_water_ratio(water, target_patch, tx, ty)
                    if t_ratio < 0.96:
                        for _ in range(24):
                            p = _random_water_point(water, self.rng)
                            if p is None:
                                break
                            cx, cy = p
                            if _alpha_water_ratio(water, target_patch, cx, cy) >= 0.96:
                                tx, ty = cx, cy
                                t_ratio = 1.0
                                break
                    bbox, vis = alpha_blend_center(patch, target_patch, tx, ty)

                    d_ids: list[str] = []
                    d_min, d_max = distractor_cfg["scale_range"]
                    for d_asset, d_img in zip(distractors, distractor_bgras):
                        d_scale = float(self.rng.uniform(float(d_min), float(d_max)))
                        d_patch = resize_bgra_with_scale(d_img, d_scale, image_size=image_size)
                        p = _random_water_point(water, self.rng)
                        if p is None:
                            continue
                        d_center_x, d_center_y = p
                        d_center_x, d_center_y = _sample_water_center(
                            water,
                            d_patch,
                            d_center_x,
                            d_center_y,
                            self.rng,
                            min_ratio=0.93,
                            tries=24,
                        )
                        if _alpha_water_ratio(water, d_patch, d_center_x, d_center_y) < 0.93:
                            continue
                        alpha_blend_center(patch, d_patch, d_center_x, d_center_y)
                        d_ids.append(d_asset.asset_id)

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
                        meta={
                            "dataset_name": ds_name,
                            "crop_center_world": [crop_center_x, crop_center_y],
                            "target_state_world": {
                                "x": float(state.x),
                                "y": float(state.y),
                                "vx": float(state.vx),
                                "vy": float(state.vy),
                            },
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
