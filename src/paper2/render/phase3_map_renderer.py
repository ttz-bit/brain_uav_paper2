from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paper2.render.compositor import alpha_blend_center, read_bgra, rotate_bgra, trim_bgra_to_alpha_bbox
from paper2.render.physical_scale import target_dimensions_px_from_km


@dataclass
class Phase3MapScene:
    scene_id: str
    split: str
    background_path: str
    water_mask_path: str
    target_asset_path: str
    background_bgr: np.ndarray
    water_mask: np.ndarray
    target_bgra: np.ndarray
    anchor_world_km: np.ndarray
    anchor_bg_px: np.ndarray
    km_per_bg_px: float
    map_width_km: float
    map_height_km: float
    distractor_tracks: list["Phase3MapDistractorTrack"]
    prev_crop_center_world: np.ndarray | None = None


@dataclass
class Phase3MapRenderResult:
    image_bgr: np.ndarray
    water_mask_crop: np.ndarray
    bbox_xywh: list[float]
    visibility: float
    meta: dict[str, Any]


@dataclass
class Phase3MapDistractorTrack:
    asset_path: str
    bg_px: np.ndarray
    heading: float
    speed_bg_px: float
    scale_px_by_stage: dict[str, float]
    radius_bg_px: float
    count_requested: int


class Phase3MapRenderer:
    """Render Phase3 observations from one fixed real map per sequence.

    The renderer maps a local Phase3 km coordinate window onto a selected
    background image and crops that same map over time. Stage-specific GSD is
    implemented by resizing a crop whose source size is image_size * gsd.
    """

    def __init__(
        self,
        *,
        stage_cfg: dict[str, Any],
        backgrounds_by_split: dict[str, list[dict]],
        targets_by_split: dict[str, list[Path]],
        distractors_by_split: dict[str, list[Path]] | None = None,
        rng: np.random.Generator | None = None,
        local_map_size_km: float = 12.0,
        min_target_water_ratio: float = 0.98,
        min_target_visibility: float = 0.35,
        distractor_count_min: int = 0,
        distractor_count_max: int = 0,
        distractor_scale_min: float = 0.45,
        distractor_scale_max: float = 0.85,
        min_distractor_target_distance_px: float = 48.0,
        scene_attempts: int = 96,
    ) -> None:
        self.stage_cfg = stage_cfg
        self.backgrounds_by_split = backgrounds_by_split
        self.targets_by_split = targets_by_split
        self.distractors_by_split = distractors_by_split or {"train": [], "val": [], "test": []}
        self.rng = rng if rng is not None else np.random.default_rng()
        self.local_map_size_km = max(1.0, float(local_map_size_km))
        self.min_target_water_ratio = float(min_target_water_ratio)
        self.min_target_visibility = float(min_target_visibility)
        self.distractor_count_min = max(0, int(distractor_count_min))
        self.distractor_count_max = max(self.distractor_count_min, int(distractor_count_max))
        self.distractor_scale_min = max(0.05, float(min(distractor_scale_min, distractor_scale_max)))
        self.distractor_scale_max = max(self.distractor_scale_min, float(max(distractor_scale_min, distractor_scale_max)))
        self.min_distractor_target_distance_px = float(min_distractor_target_distance_px)
        self.scene_attempts = max(1, int(scene_attempts))

    def create_scene(self, *, split: str, sequence_id: str, frames: list[Any]) -> Phase3MapScene:
        if not frames:
            raise RuntimeError("Cannot create Phase3 map scene without frames.")
        bg_pool = self.backgrounds_by_split.get(split, [])
        target_pool = self.targets_by_split.get(split, [])
        if not bg_pool:
            raise RuntimeError(f"No Phase3 map backgrounds for split={split}.")
        if not target_pool:
            raise RuntimeError(f"No Phase3 map target templates for split={split}.")

        for _ in range(self.scene_attempts):
            bg_rec = bg_pool[int(self.rng.integers(0, len(bg_pool)))]
            bg_path = Path(bg_rec["image_path"])
            mask_path = Path(bg_rec["mask_path"])
            bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if bg is None or mask is None:
                continue
            h, w = bg.shape[:2]
            if h < 4 or w < 4:
                continue
            if mask.shape[:2] != bg.shape[:2]:
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            target_path = target_pool[int(self.rng.integers(0, len(target_pool)))]
            target_bgra = trim_bgra_to_alpha_bbox(read_bgra(target_path))

            km_per_bg_px = self.local_map_size_km / float(min(w, h))
            map_width_km = float(w) * km_per_bg_px
            map_height_km = float(h) * km_per_bg_px
            anchor = self._sample_anchor(mask, frames, km_per_bg_px)
            if anchor is None:
                continue
            anchor_world = np.asarray(frames[0].target_state_world["pos_world"], dtype=float).reshape(-1)[:2]
            scene = Phase3MapScene(
                scene_id=str(sequence_id),
                split=str(split),
                background_path=str(bg_path),
                water_mask_path=str(mask_path),
                target_asset_path=str(target_path),
                background_bgr=bg,
                water_mask=mask.astype(np.uint8),
                target_bgra=target_bgra,
                anchor_world_km=anchor_world.astype(float),
                anchor_bg_px=np.asarray(anchor, dtype=float),
                km_per_bg_px=float(km_per_bg_px),
                map_width_km=float(map_width_km),
                map_height_km=float(map_height_km),
                distractor_tracks=[],
            )
            scene.distractor_tracks = self._init_distractors(scene)
            return scene
        raise RuntimeError(f"Could not create continuous Phase3 map scene for split={split}, sequence={sequence_id}.")

    def create_live_scene(
        self,
        *,
        split: str,
        sequence_id: str,
        target_world_xy: np.ndarray,
    ) -> Phase3MapScene:
        target_xy = np.asarray(target_world_xy, dtype=float).reshape(2)
        gsd = float(self.stage_cfg["far"]["gsd_km_per_px"])
        frame = SimpleNamespace(
            frame_id=0,
            stage="far",
            gsd_km_per_px=gsd,
            crop_center_world=[float(target_xy[0]), float(target_xy[1])],
            target_state_world={
                "pos_world": [float(target_xy[0]), float(target_xy[1])],
                "heading": 0.0,
            },
        )
        return self.create_scene(split=split, sequence_id=sequence_id, frames=[frame])

    def render_frame(self, scene: Phase3MapScene, frame: Any) -> Phase3MapRenderResult:
        image_size = int(self.stage_cfg.get("image_size", 256))
        stage = str(frame.stage)
        gsd = float(frame.gsd_km_per_px)
        crop_center_world = np.asarray(frame.crop_center_world, dtype=float).reshape(2)
        target_world = np.asarray(frame.target_state_world["pos_world"], dtype=float).reshape(-1)[:2]
        crop_bg = self.world_to_bg(scene, crop_center_world)
        target_bg = self.world_to_bg(scene, target_world)
        crop_size_bg = max(2.0, float(image_size) * gsd / max(scene.km_per_bg_px, 1e-12))
        canvas = _extract_resized_patch(scene.background_bgr, crop_bg[0], crop_bg[1], crop_size_bg, image_size, cv2.INTER_LINEAR)
        water_crop = _extract_resized_patch(scene.water_mask, crop_bg[0], crop_bg[1], crop_size_bg, image_size, cv2.INTER_NEAREST)
        if water_crop.ndim == 3:
            water_crop = water_crop[:, :, 0]
        water_crop = water_crop.astype(np.uint8)

        target_center = self.bg_to_image(target_bg, crop_bg, crop_size_bg, image_size)
        target_length_px, target_width_px = target_dimensions_px_from_km(
            gsd_km_per_px=gsd,
            stage_cfg=self.stage_cfg,
            image_size=image_size,
        )
        target = _resize_bgra_to_long_side(scene.target_bgra, target_length_px, image_size)
        target = rotate_bgra(target, -float(np.degrees(frame.target_state_world["heading"])))
        target = trim_bgra_to_alpha_bbox(target)

        distractor_asset_paths: list[str] = []
        distractor_bboxes: list[list[float]] = []
        distractor_water_ratios: list[float] = []
        distractor_visibilities: list[float] = []
        for track in scene.distractor_tracks:
            self._advance_distractor(track, scene)
            rendered = self._render_distractor(canvas, water_crop, scene, track, crop_bg, crop_size_bg, target_center, stage)
            if rendered is not None:
                bbox_d, vis_d, wr_d = rendered
                distractor_asset_paths.append(track.asset_path)
                distractor_bboxes.append(bbox_d)
                distractor_visibilities.append(float(vis_d))
                distractor_water_ratios.append(float(wr_d))

        target_water_ratio = _alpha_water_ratio(water_crop, target, target_center[0], target_center[1])
        bbox_tuple, visibility = alpha_blend_center(canvas, target, target_center[0], target_center[1])
        if visibility < self.min_target_visibility:
            raise RuntimeError(f"Phase3 map target visibility too low: sequence={scene.scene_id}, frame={frame.frame_id}")
        if target_water_ratio < self.min_target_water_ratio:
            raise RuntimeError(
                f"Phase3 map target water ratio too low: ratio={target_water_ratio:.3f}, "
                f"sequence={scene.scene_id}, frame={frame.frame_id}"
            )

        crop_half = 0.5 * crop_size_bg
        meta = {
            "asset_mode": "real_map",
            "render_mode": "phase3_map",
            "map_scene_id": scene.scene_id,
            "background_path": scene.background_path,
            "water_mask_path": scene.water_mask_path,
            "target_asset_path": scene.target_asset_path,
            "background_size_px": [int(scene.background_bgr.shape[1]), int(scene.background_bgr.shape[0])],
            "target_bg_center_px": [float(target_bg[0]), float(target_bg[1])],
            "crop_center_bg_px": [float(crop_bg[0]), float(crop_bg[1])],
            "crop_origin_bg_px": [
                float(crop_bg[0] - crop_half),
                float(crop_bg[1] - crop_half),
            ],
            "crop_box_bg_xyxy": [
                float(crop_bg[0] - crop_half),
                float(crop_bg[1] - crop_half),
                float(crop_bg[0] + crop_half),
                float(crop_bg[1] + crop_half),
            ],
            "origin_world_km": [float(scene.anchor_world_km[0]), float(scene.anchor_world_km[1])],
            "anchor_bg_px": [float(scene.anchor_bg_px[0]), float(scene.anchor_bg_px[1])],
            "km_per_bg_px": float(scene.km_per_bg_px),
            "map_width_km": float(scene.map_width_km),
            "map_height_km": float(scene.map_height_km),
            "target_water_ratio": float(target_water_ratio),
            "target_length_px": float(target_length_px),
            "target_width_px": float(target_width_px),
            "distractor_asset_paths": distractor_asset_paths,
            "distractor_bboxes_xywh": distractor_bboxes,
            "distractor_water_ratios": distractor_water_ratios,
            "distractor_visibilities": distractor_visibilities,
            "distractor_count_requested": int(len(scene.distractor_tracks)),
            "distractor_count": int(len(scene.distractor_tracks)),
        }
        return Phase3MapRenderResult(
            image_bgr=canvas,
            water_mask_crop=water_crop,
            bbox_xywh=[float(v) for v in bbox_tuple],
            visibility=float(visibility),
            meta=meta,
        )

    def render_truth(
        self,
        scene: Phase3MapScene,
        *,
        truth: Any,
        stage: str,
        frame_id: int,
    ) -> Phase3MapRenderResult:
        target_xy = np.asarray(truth.pos_world, dtype=float).reshape(-1)[:2]
        gsd = float(self.stage_cfg[str(stage)]["gsd_km_per_px"])
        jitter_cfg = dict(self.stage_cfg.get("crop_jitter_px", {}))
        jitter_px = float(jitter_cfg.get(str(stage), 0.0))
        candidate = target_xy + self.rng.normal(0.0, jitter_px * gsd, size=2)
        if scene.prev_crop_center_world is None:
            crop_center = candidate
        else:
            crop_center = 0.82 * scene.prev_crop_center_world + 0.18 * candidate
            max_step = max(0.25, 0.20 * float(self.stage_cfg.get("image_size", 256)) * gsd)
            delta = crop_center - scene.prev_crop_center_world
            dist = float(np.linalg.norm(delta))
            if dist > max_step and dist > 1e-9:
                crop_center = scene.prev_crop_center_world + delta / dist * max_step
        scene.prev_crop_center_world = np.asarray(crop_center, dtype=float)
        frame = SimpleNamespace(
            frame_id=int(frame_id),
            stage=str(stage),
            gsd_km_per_px=gsd,
            crop_center_world=[float(crop_center[0]), float(crop_center[1])],
            target_state_world={
                "pos_world": [float(target_xy[0]), float(target_xy[1])],
                "vel_world": [float(v) for v in np.asarray(truth.vel_world, dtype=float).reshape(-1)[:2]],
                "heading": float(truth.heading),
                "motion_mode": str(truth.motion_mode),
            },
        )
        return self.render_frame(scene, frame)

    def world_to_bg(self, scene: Phase3MapScene, xy_world_km: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_world_km, dtype=float).reshape(2)
        delta = xy - scene.anchor_world_km
        return np.array(
            [
                float(scene.anchor_bg_px[0] + delta[0] / scene.km_per_bg_px),
                float(scene.anchor_bg_px[1] - delta[1] / scene.km_per_bg_px),
            ],
            dtype=float,
        )

    @staticmethod
    def bg_to_image(bg_xy: np.ndarray, crop_bg_xy: np.ndarray, crop_size_bg: float, image_size: int) -> np.ndarray:
        scale = float(image_size) / max(float(crop_size_bg), 1e-12)
        return np.array(
            [
                float(0.5 * image_size + (float(bg_xy[0]) - float(crop_bg_xy[0])) * scale),
                float(0.5 * image_size + (float(bg_xy[1]) - float(crop_bg_xy[1])) * scale),
            ],
            dtype=float,
        )

    def _sample_anchor(self, mask: np.ndarray, frames: list[Any], km_per_bg_px: float) -> tuple[float, float] | None:
        image_size = int(self.stage_cfg.get("image_size", 256))
        h, w = mask.shape[:2]
        water = mask > 0
        if int(water.sum()) <= 0:
            return None
        dist = cv2.distanceTransform(water.astype(np.uint8), cv2.DIST_L2, 3)
        initial = np.asarray(frames[0].target_state_world["pos_world"], dtype=float).reshape(-1)[:2]
        target_world = [np.asarray(f.target_state_world["pos_world"], dtype=float).reshape(-1)[:2] for f in frames]
        crop_world = [np.asarray(f.crop_center_world, dtype=float).reshape(2) for f in frames]
        offsets_target = [(xy - initial) / float(km_per_bg_px) for xy in target_world]
        offsets_crop = [(xy - initial) / float(km_per_bg_px) for xy in crop_world]
        max_crop_radius = max(
            0.5 * image_size * float(f.gsd_km_per_px) / max(float(km_per_bg_px), 1e-12) for f in frames
        )
        yy, xx = np.where(dist >= max(4.0, min(96.0, max_crop_radius * 0.08)))
        if len(xx) <= 0:
            yy, xx = np.where(water)
        if len(xx) <= 0:
            return None
        for _ in range(512):
            idx = int(self.rng.integers(0, len(xx)))
            ax = float(xx[idx])
            ay = float(yy[idx])
            ok = True
            for t_off, c_off, f in zip(offsets_target, offsets_crop, frames):
                target_px = np.array([ax + float(t_off[0]), ay - float(t_off[1])], dtype=float)
                crop_px = np.array([ax + float(c_off[0]), ay - float(c_off[1])], dtype=float)
                crop_radius = 0.5 * image_size * float(f.gsd_km_per_px) / max(float(km_per_bg_px), 1e-12)
                if not (crop_radius <= crop_px[0] < w - crop_radius and crop_radius <= crop_px[1] < h - crop_radius):
                    ok = False
                    break
                ix = int(round(float(target_px[0])))
                iy = int(round(float(target_px[1])))
                if ix < 0 or iy < 0 or ix >= w or iy >= h or mask[iy, ix] <= 0:
                    ok = False
                    break
            if ok:
                return ax, ay
        return None

    def _init_distractors(self, scene: Phase3MapScene) -> list[Phase3MapDistractorTrack]:
        count_max = self.distractor_count_max
        pool = self.distractors_by_split.get(scene.split, [])
        if count_max <= 0 or not pool:
            return []
        desired = int(self.rng.integers(self.distractor_count_min, count_max + 1))
        if desired <= 0:
            return []
        water = scene.water_mask > 0
        yy, xx = np.where(water)
        tracks: list[Phase3MapDistractorTrack] = []
        if len(xx) <= 0:
            return tracks
        for _ in range(desired):
            for _attempt in range(96):
                idx = int(self.rng.integers(0, len(xx)))
                bg_xy = np.array([float(xx[idx]), float(yy[idx])], dtype=float)
                if float(np.hypot(bg_xy[0] - scene.anchor_bg_px[0], bg_xy[1] - scene.anchor_bg_px[1])) < 12.0:
                    continue
                path = pool[int(self.rng.integers(0, len(pool)))]
                scale_by_stage: dict[str, float] = {}
                for stage in ("far", "mid", "terminal"):
                    gsd = float(self.stage_cfg[stage]["gsd_km_per_px"])
                    target_len, _ = target_dimensions_px_from_km(
                        gsd_km_per_px=gsd,
                        stage_cfg=self.stage_cfg,
                        image_size=int(self.stage_cfg.get("image_size", 256)),
                    )
                    scale_by_stage[stage] = float(target_len * self.rng.uniform(self.distractor_scale_min, self.distractor_scale_max))
                tracks.append(
                    Phase3MapDistractorTrack(
                        asset_path=str(path),
                        bg_px=bg_xy,
                        heading=float(self.rng.uniform(-np.pi, np.pi)),
                        speed_bg_px=float(self.rng.uniform(0.15, 0.65)),
                        scale_px_by_stage=scale_by_stage,
                        radius_bg_px=max(2.0, max(scale_by_stage.values()) * 0.5),
                        count_requested=desired,
                    )
                )
                break
        return tracks

    def _advance_distractor(self, track: Phase3MapDistractorTrack, scene: Phase3MapScene) -> None:
        step = np.array([np.cos(track.heading), -np.sin(track.heading)], dtype=float) * float(track.speed_bg_px)
        candidate = track.bg_px + step
        h, w = scene.water_mask.shape[:2]
        ix = int(round(float(candidate[0])))
        iy = int(round(float(candidate[1])))
        if ix < 0 or iy < 0 or ix >= w or iy >= h or scene.water_mask[iy, ix] <= 0:
            track.heading = float((track.heading + np.pi + self.rng.uniform(-0.45, 0.45) + np.pi) % (2 * np.pi) - np.pi)
            return
        track.bg_px = candidate
        track.heading = float((track.heading + self.rng.uniform(-0.025, 0.025) + np.pi) % (2 * np.pi) - np.pi)

    def _render_distractor(
        self,
        canvas: np.ndarray,
        water_crop: np.ndarray,
        scene: Phase3MapScene,
        track: Phase3MapDistractorTrack,
        crop_bg: np.ndarray,
        crop_size_bg: float,
        target_center: np.ndarray,
        stage: str,
    ) -> tuple[list[float], float, float] | None:
        image_size = int(canvas.shape[0])
        center = self.bg_to_image(track.bg_px, crop_bg, crop_size_bg, image_size)
        if center[0] < 0.0 or center[1] < 0.0 or center[0] >= image_size or center[1] >= image_size:
            return None
        if float(np.hypot(center[0] - target_center[0], center[1] - target_center[1])) < self.min_distractor_target_distance_px:
            return None
        try:
            distractor = trim_bgra_to_alpha_bbox(read_bgra(track.asset_path))
        except Exception:
            return None
        scale_px = float(track.scale_px_by_stage.get(stage, max(track.scale_px_by_stage.values())))
        distractor = _resize_bgra_to_long_side(distractor, scale_px, image_size)
        distractor = rotate_bgra(distractor, float(np.degrees(track.heading)))
        distractor = trim_bgra_to_alpha_bbox(distractor)
        water_ratio = _alpha_water_ratio(water_crop, distractor, center[0], center[1])
        if water_ratio < 0.85:
            return None
        bbox_tuple, visibility = alpha_blend_center(canvas, distractor, center[0], center[1])
        if visibility <= 0.0:
            return None
        return [float(v) for v in bbox_tuple], float(visibility), float(water_ratio)


def _extract_resized_patch(
    image: np.ndarray,
    cx: float,
    cy: float,
    source_size: float,
    out_size: int,
    interpolation: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(2, int(round(float(source_size))))
    half = 0.5 * float(side)
    x1 = int(round(float(cx) - half))
    y1 = int(round(float(cy) - half))
    x2 = x1 + side
    y2 = y1 + side
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)
    src = image
    if pad_l or pad_t or pad_r or pad_b:
        src = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        x1 += pad_l
        y1 += pad_t
        x2 += pad_l
        y2 += pad_t
    patch = src[y1:y2, x1:x2].copy()
    if patch.shape[0] != out_size or patch.shape[1] != out_size:
        patch = cv2.resize(patch, (int(out_size), int(out_size)), interpolation=interpolation)
    return patch


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
