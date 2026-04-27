from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from paper2.render.asset_registry import AssetRecord, AssetRegistry
from paper2.render.compositor import alpha_blend_center, read_bgra, resize_bgra_with_scale
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
                target = self.registry.sample_one("target", split, self.rng)
                d_num = int(self.rng.integers(int(distractor_cfg["min_count"]), int(distractor_cfg["max_count"]) + 1))
                distractors = self.registry.sample_many("distractor", split, d_num, self.rng)

                used_background_ids.add(bg.asset_id)
                used_target_ids.add(target.asset_id)
                for d in distractors:
                    used_distractor_ids.add(d.asset_id)

                bg_img = self._read_background(bg)
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

                    tx, ty = world_to_image(state.x, state.y, crop_center_x, crop_center_y, gsd, image_size)
                    scale_min, scale_max = stage_cfg["target_scale_range"]
                    target_scale = float(self.rng.uniform(float(scale_min), float(scale_max)))
                    target_patch = resize_bgra_with_scale(target_bgra, target_scale, image_size=image_size)
                    bbox, vis = alpha_blend_center(patch, target_patch, tx, ty)

                    d_ids: list[str] = []
                    d_min, d_max = distractor_cfg["scale_range"]
                    for d_asset, d_img in zip(distractors, distractor_bgras):
                        d_center_x = float(self.rng.uniform(0.1, 0.9) * image_size)
                        d_center_y = float(self.rng.uniform(0.1, 0.9) * image_size)
                        d_scale = float(self.rng.uniform(float(d_min), float(d_max)))
                        d_patch = resize_bgra_with_scale(d_img, d_scale, image_size=image_size)
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
