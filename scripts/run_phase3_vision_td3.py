from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paper2.common.config import load_yaml
from paper2.common.types import NoFlyZoneState, TargetEstimateState, TargetTruthState, VisionObservation
from paper2.env_adapter.paper1_bridge import Paper1EnvBridge
from paper2.env_adapter.phase3_target_motion import propagate_phase3_target_truth, sample_phase3_initial_target
from paper2.env_adapter.world_frame import paper2_xyz_to_paper1_xyz
from paper2.paper1_method.curriculum import parse_curriculum_mix
from paper2.render.asset_registry import AssetRegistry, load_asset_inventory
from paper2.render.coordinate_mapper import WorldState
from paper2.models.cnn_heatmap import HeatmapCNN
from paper2.models.snn_heatmap import HeatmapSNN, peak_argmax_2d, soft_argmax_2d
from paper2.render.phase3_map_renderer import Phase3MapRenderer
from paper2.render.renderer_stage2 import Stage2Renderer
from paper2.planning.paper1_td3_policy import Paper1TD3Policy
from paper2.tracking.kalman import ConstantVelocityKalmanFilter
from paper2.tracking.vision_to_estimate import image_point_to_world_xy, vision_observation_to_target_estimate
from scripts.render_phase3_task_dataset import _collect_backgrounds, _collect_distractors, _collect_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/env.yaml")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument(
        "--vision-source",
        choices=["replay_dataset", "phase3_map_live", "stage2_live_render", "live_render"],
        default="replay_dataset",
    )
    p.add_argument("--render-config", type=str, default="configs/render_stage2_c_v1.yaml")
    p.add_argument("--phase3-live-generation-config", type=str, default="")
    p.add_argument("--phase3-live-local-map-size-km", type=float, default=0.0)
    p.add_argument("--phase3-live-render-retries", type=int, default=4)
    p.add_argument("--phase3-live-render-reanchor-on-failure", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--render-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--vision-weights", type=str, required=True)
    p.add_argument("--td3-checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["snn", "ann"], default="snn")
    p.add_argument("--vision-model", choices=["auto", "cnn_coord", "cnn_heatmap", "snn_heatmap"], default="auto")
    p.add_argument("--decode-method", choices=["auto", "argmax", "softargmax"], default="auto")
    p.add_argument("--estimate-filter", choices=["none", "gated_smooth", "kalman"], default="gated_smooth")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--paper1-curriculum-level", choices=["easy", "easy_two_zone", "medium", "hard"], default="hard")
    p.add_argument("--paper1-curriculum-mix", type=str, default=None)
    p.add_argument("--phase3-target-init", choices=["paper1_goal", "random_water"], default="paper1_goal")
    p.add_argument("--target-z-policy", choices=["keep_current_goal_z"], default="keep_current_goal_z")
    p.add_argument("--capture-radius-km", type=float, default=None)
    p.add_argument(
        "--capture-mode",
        choices=["estimate_goal", "true_target"],
        default="estimate_goal",
        help="estimate_goal preserves Paper1 termination; true_target only ends capture when the real moving target is reached.",
    )
    p.add_argument("--max-vision-samples", type=int, default=None)
    p.add_argument("--disable-estimate-gating", action="store_true")
    p.add_argument("--gate-far-km", type=float, default=5.0)
    p.add_argument("--gate-mid-km", type=float, default=2.0)
    p.add_argument("--gate-terminal-km", type=float, default=0.5)
    p.add_argument("--gain-far", type=float, default=0.05)
    p.add_argument("--gain-mid", type=float, default=0.25)
    p.add_argument("--gain-terminal", type=float, default=0.80)
    p.add_argument("--kf-process-accel-std", type=float, default=0.08)
    p.add_argument("--kf-process-accel-std-far", type=float, default=0.04)
    p.add_argument("--kf-process-accel-std-mid", type=float, default=0.08)
    p.add_argument("--kf-process-accel-std-terminal", type=float, default=0.15)
    p.add_argument("--kf-gate-far-km", type=float, default=5.0)
    p.add_argument("--kf-gate-mid-km", type=float, default=2.0)
    p.add_argument("--kf-gate-terminal-km", type=float, default=0.5)
    p.add_argument("--kf-max-reject-streak", type=int, default=100)
    p.add_argument("--kf-max-obs-age-before-reinit", type=float, default=120.0)
    p.add_argument("--kf-nis-gate-threshold", type=float, default=11.83)
    p.add_argument("--bootstrap-estimate-from-goal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--initial-goal-position-std-km", type=float, default=5.0)
    p.add_argument("--initial-goal-velocity-std-km-s", type=float, default=2.0)
    p.add_argument("--meas-sigma-far-px", type=float, default=80.0)
    p.add_argument("--meas-sigma-mid-px", type=float, default=32.0)
    p.add_argument("--meas-sigma-terminal-px", type=float, default=24.0)
    p.add_argument(
        "--kf-terminal-mode",
        choices=["kalman", "raw"],
        default="raw",
        help="When --estimate-filter=kalman, optionally bypass KF in terminal range to avoid end-game lag.",
    )
    p.add_argument(
        "--terminal-controller",
        choices=["td3", "pure_pursuit", "blend"],
        default="td3",
        help="Optional evaluation-time terminal guidance based on the current target estimate.",
    )
    p.add_argument("--terminal-controller-range-km", type=float, default=80.0)
    p.add_argument("--terminal-controller-blend", type=float, default=0.5)
    p.add_argument("--vision-input-size", type=int, default=0)
    p.add_argument("--visual-audit-count", type=int, default=16)
    p.add_argument("--out-dir", type=str, default="outputs/phase3_vision_td3/cnn_td3_snn_hard")
    return p.parse_args()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for CNN vision TD3 evaluation.") from exc
    return torch, nn


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return str(device)


def _to_tensor_image(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def _read_jsonl(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows is not None and len(rows) >= int(max_rows):
                break
    return rows


def _resolve_image_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def _target_truth_to_world_state(truth: TargetTruthState) -> WorldState:
    pos = np.asarray(truth.pos_world, dtype=float).reshape(-1)
    vel = np.asarray(truth.vel_world, dtype=float).reshape(-1)
    return WorldState(
        x=float(pos[0]),
        y=float(pos[1]),
        vx=float(vel[0]) if vel.size > 0 else 0.0,
        vy=float(vel[1]) if vel.size > 1 else 0.0,
        heading=float(truth.heading),
    )


def _phase3_map_frame_to_row(
    *,
    result: Any,
    truth: TargetTruthState,
    stage: str,
    stage_cfg: dict[str, Any],
    image_path: str,
    sequence_id: str,
    frame_id: str,
) -> dict[str, Any]:
    target_xy = np.asarray(truth.pos_world, dtype=float).reshape(-1)[:2]
    target_vel = np.asarray(truth.vel_world, dtype=float).reshape(-1)
    meta = dict(result.meta)
    target_center_px = np.asarray(meta["target_bg_center_px"], dtype=float).reshape(2)
    crop_bg = np.asarray(meta["crop_center_bg_px"], dtype=float).reshape(2)
    crop_box = np.asarray(meta["crop_box_bg_xyxy"], dtype=float).reshape(4)
    crop_size_bg = max(float(crop_box[2] - crop_box[0]), 1e-9)
    image_size = int(stage_cfg.get("image_size", 256))
    target_center_px = np.array(
        [
            0.5 * image_size + (target_center_px[0] - crop_bg[0]) * float(image_size) / crop_size_bg,
            0.5 * image_size + (target_center_px[1] - crop_bg[1]) * float(image_size) / crop_size_bg,
        ],
        dtype=float,
    )
    gsd_km_per_px = float(stage_cfg[str(stage)]["gsd_km_per_px"])
    half = float(image_size) * 0.5
    crop_center_xy = np.array(
        [
            float(target_xy[0]) - (float(target_center_px[0]) - half) * gsd_km_per_px,
            float(target_xy[1]) - (half - float(target_center_px[1])) * gsd_km_per_px,
        ],
        dtype=float,
    )
    meta.update(
        {
            "source": "phase3_map_live",
            "observation_geometry_unit": "km",
            "crop_center_world": [float(crop_center_xy[0]), float(crop_center_xy[1])],
            "crop_center_world_x": float(crop_center_xy[0]),
            "crop_center_world_y": float(crop_center_xy[1]),
            "target_state_world": {
                "x": float(target_xy[0]),
                "y": float(target_xy[1]),
                "vx": float(target_vel[0]) if target_vel.size > 0 else 0.0,
                "vy": float(target_vel[1]) if target_vel.size > 1 else 0.0,
                "heading": float(truth.heading),
                "motion_mode": str(truth.motion_mode),
                "unit": "km",
            },
            "target_world_x": float(target_xy[0]),
            "target_world_y": float(target_xy[1]),
            "target_center_px": [float(target_center_px[0]), float(target_center_px[1])],
            "center_x": float(target_center_px[0]),
            "center_y": float(target_center_px[1]),
            "gsd": float(gsd_km_per_px),
            "gsd_km_per_px": float(gsd_km_per_px),
        }
    )
    return {
        "image_path": str(image_path),
        "sequence_id": str(sequence_id),
        "frame_id": str(frame_id),
        "stage": str(stage),
        "vision_source": "phase3_map_live",
        "render_mode": "phase3_map",
        "background_asset_id": Path(str(meta.get("background_path", "phase3_map"))).stem,
        "target_asset_id": Path(str(meta.get("target_asset_path", "phase3_target"))).stem,
        "distractor_asset_ids": [Path(str(p)).stem for p in meta.get("distractor_asset_paths", [])],
        "gsd_km_per_px": float(gsd_km_per_px),
        "target_center_px": [float(target_center_px[0]), float(target_center_px[1])],
        "bbox_xywh": list(result.bbox_xywh),
        "visibility": float(result.visibility),
        "obs_valid": bool(result.visibility > 0.0),
        "meta": meta,
    }


def _live_frame_to_phase3_row(
    *,
    live_frame: Any,
    truth: TargetTruthState,
    stage: str,
    stage_cfg: dict[str, Any],
    image_path: str,
    sequence_id: str,
    frame_id: str,
) -> dict[str, Any]:
    """Expose live-rendered pixels with Phase3 km-based observation geometry."""
    target_xy = np.asarray(truth.pos_world, dtype=float).reshape(-1)[:2]
    target_vel = np.asarray(truth.vel_world, dtype=float).reshape(-1)
    target_center_px = np.asarray(live_frame.target_center_px, dtype=float).reshape(2)
    image_size = int(stage_cfg.get("image_size", 256))
    half = float(image_size) * 0.5
    gsd_km_per_px = float(stage_cfg[str(stage)]["gsd_km_per_px"])
    crop_center_xy = np.array(
        [
            float(target_xy[0]) - (float(target_center_px[0]) - half) * gsd_km_per_px,
            float(target_xy[1]) - (half - float(target_center_px[1])) * gsd_km_per_px,
        ],
        dtype=float,
    )

    meta = dict(live_frame.meta)
    meta.update(
        {
            "source": "live_render_phase3_geometry",
            "render_geometry_unit": "renderer_local",
            "observation_geometry_unit": "km",
            "crop_center_world": [float(crop_center_xy[0]), float(crop_center_xy[1])],
            "crop_center_world_x": float(crop_center_xy[0]),
            "crop_center_world_y": float(crop_center_xy[1]),
            "target_state_world": {
                "x": float(target_xy[0]),
                "y": float(target_xy[1]),
                "vx": float(target_vel[0]) if target_vel.size > 0 else 0.0,
                "vy": float(target_vel[1]) if target_vel.size > 1 else 0.0,
                "heading_deg": float(np.degrees(float(truth.heading))),
            },
            "target_world_x": float(target_xy[0]),
            "target_world_y": float(target_xy[1]),
            "target_center_px": [float(target_center_px[0]), float(target_center_px[1])],
            "center_x": float(target_center_px[0]),
            "center_y": float(target_center_px[1]),
            "gsd": float(gsd_km_per_px),
            "gsd_km_per_px": float(gsd_km_per_px),
            "render_gsd_m_per_px": float(live_frame.gsd_m_per_px),
        }
    )

    return {
        "image_path": str(image_path),
        "sequence_id": str(sequence_id),
        "frame_id": str(frame_id),
        "stage": str(live_frame.stage),
        "vision_source": "live_render",
        "render_mode": "live_render",
        "background_asset_id": str(live_frame.background_asset_id),
        "target_asset_id": str(live_frame.target_asset_id),
        "distractor_asset_ids": list(live_frame.distractor_asset_ids),
        "render_background_asset_id": str(live_frame.background_asset_id),
        "render_target_asset_id": str(live_frame.target_asset_id),
        "render_distractor_asset_ids": list(live_frame.distractor_asset_ids),
        "render_obs_valid": bool(live_frame.obs_valid),
        "render_visibility": float(live_frame.visibility),
        "render_land_overlap_ratio": float(live_frame.land_overlap_ratio),
        "render_shore_buffer_overlap_ratio": float(live_frame.shore_buffer_overlap_ratio),
        "gsd_km_per_px": float(gsd_km_per_px),
        "render_gsd_m_per_px": float(live_frame.gsd_m_per_px),
        "target_center_px": [float(target_center_px[0]), float(target_center_px[1])],
        "meta": meta,
    }


def _sample_gt_center_px(row: dict[str, Any]) -> tuple[float, float] | None:
    if "target_center_px" in row and isinstance(row["target_center_px"], (list, tuple)) and len(row["target_center_px"]) >= 2:
        return float(row["target_center_px"][0]), float(row["target_center_px"][1])
    meta = dict(row.get("meta", {}))
    if "center_x" in meta and "center_y" in meta:
        return float(meta["center_x"]), float(meta["center_y"])
    if "target_center_px" in meta and isinstance(meta["target_center_px"], (list, tuple)) and len(meta["target_center_px"]) >= 2:
        return float(meta["target_center_px"][0]), float(meta["target_center_px"][1])
    return None


def _write_vision_audit_image(
    *,
    image_bgr: np.ndarray,
    row: dict[str, Any],
    pred_center_px: tuple[float, float],
    stage: str,
    episode: int,
    step: int,
    out_dir: Path,
) -> tuple[str, float]:
    gt = _sample_gt_center_px(row)
    if gt is None:
        return "", float("nan")
    px = float(pred_center_px[0])
    py = float(pred_center_px[1])
    gx = float(gt[0])
    gy = float(gt[1])
    err_px = float(np.hypot(px - gx, py - gy))

    vis = image_bgr.copy()
    cv2.circle(vis, (int(round(gx)), int(round(gy))), 5, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, (int(round(px)), int(round(py))), 5, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.line(vis, (int(round(gx)), int(round(gy))), (int(round(px)), int(round(py))), (255, 255, 0), 1, cv2.LINE_AA)
    label = f"ep={episode:03d} step={step:04d} {stage} gt=green pred=red err={err_px:.1f}px"
    cv2.putText(vis, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ep{episode:03d}_step{step:04d}_{stage}_err{err_px:06.2f}px.jpg"
    cv2.imwrite(str(path), vis)
    return str(path), err_px


def _make_cnn_model(nn: Any):
    class _SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            return torch.sigmoid(self.head(self.backbone(x)))

    torch, _ = _import_torch()
    return _SmallCNN()


def _load_vision_model(weights: Path, device: str, requested: str = "auto"):
    torch, nn = _import_torch()
    ckpt = torch.load(weights, map_location=device)
    model_type = str(ckpt.get("model_type", "cnn_coord"))
    if requested != "auto":
        model_type = str(requested)
    if model_type == "cnn_coord":
        model = _make_cnn_model(nn).to(device)
    elif model_type == "cnn_heatmap":
        model = HeatmapCNN(
            width=int(ckpt.get("width", 32)),
            arch=str(ckpt.get("cnn_arch", ckpt.get("arch", "enhanced"))),
        ).to(device)
    elif model_type == "snn_heatmap":
        model = HeatmapSNN(
            beta=float(ckpt.get("beta", 0.95)),
            num_steps=int(ckpt.get("num_steps", 12)),
            train_encoding=str(ckpt.get("train_encoding", "direct")),
            eval_encoding=str(ckpt.get("eval_encoding", "direct")),
            arch=str(ckpt.get("snn_arch", ckpt.get("arch", "enhanced"))),
        ).to(device)
    else:
        raise RuntimeError(f"Unsupported vision model_type={model_type}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt, model_type


def _predict_vision(
    model: Any,
    model_type: str,
    image_bgr: np.ndarray,
    device: str,
    *,
    input_size: int = 0,
    decode_method: str = "auto",
    softargmax_temperature: float = 20.0,
) -> tuple[float, float, float]:
    torch, _ = _import_torch()
    orig_h, orig_w = image_bgr.shape[:2]
    proc_h, proc_w = orig_h, orig_w
    x_img = image_bgr
    if int(input_size) > 0 and (orig_h != int(input_size) or orig_w != int(input_size)):
        proc_w = proc_h = int(input_size)
        x_img = cv2.resize(image_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    with torch.inference_mode():
        x = torch.from_numpy(_to_tensor_image(x_img)).unsqueeze(0).to(device)
        if model_type == "cnn_coord":
            pred = model(x)[0].detach().cpu().numpy()
            px = float(np.clip(pred[0], 0.0, 1.0) * proc_w)
            py = float(np.clip(pred[1], 0.0, 1.0) * proc_h)
            conf = float(np.clip(pred[2], 0.0, 1.0))
            sx = orig_w / max(1.0, float(proc_w))
            sy = orig_h / max(1.0, float(proc_h))
            return (float(px * sx), float(py * sy), conf)
        outputs = model(x) if model_type == "cnn_heatmap" else model(x, stochastic=False)
        if str(decode_method) == "softargmax":
            pred_xy = soft_argmax_2d(outputs["heatmap_logits"], temperature=float(softargmax_temperature))[0].detach().cpu().numpy()
        else:
            pred_xy = peak_argmax_2d(outputs["heatmap_logits"])[0].detach().cpu().numpy()
        px = float(np.clip(pred_xy[0], 0.0, 1.0) * max(1, proc_w - 1))
        py = float(np.clip(pred_xy[1], 0.0, 1.0) * max(1, proc_h - 1))
        conf = float(torch.sigmoid(outputs["conf_logits"])[0].detach().cpu().item())
    sx = orig_w / max(1.0, float(proc_w))
    sy = orig_h / max(1.0, float(proc_h))
    return (float(px * sx), float(py * sy), float(np.clip(conf, 0.0, 1.0)))


def _stage_for_range(range_km: float, stage_cfg: dict[str, Any]) -> str:
    r = float(range_km)
    for name in ("terminal", "mid", "far"):
        cfg = stage_cfg[name]
        if float(cfg["range_min_km"]) <= r <= float(cfg["range_max_km"]):
            return name
    if r < float(stage_cfg["terminal"]["range_min_km"]):
        return "terminal"
    return "far"


def _create_phase3_live_scene(
    renderer: Phase3MapRenderer,
    *,
    split: str,
    sequence_id: str,
    target_xy: np.ndarray,
) -> Any:
    return renderer.create_live_scene(
        split=str(split),
        sequence_id=str(sequence_id),
        target_world_xy=np.asarray(target_xy, dtype=float).reshape(-1)[:2],
    )


def _group_rows_by_stage(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = {"far": [], "mid": [], "terminal": []}
    for row in rows:
        stage = str(row.get("stage", row.get("meta", {}).get("perception_stage", "")))
        if stage in grouped:
            grouped[stage].append(row)
    return grouped


def _sample_stage_row(grouped: dict[str, list[dict[str, Any]]], stage: str, rng: np.random.Generator) -> dict[str, Any]:
    pool = grouped.get(stage) or []
    if not pool:
        raise RuntimeError(f"No vision samples available for stage={stage}.")
    return pool[int(rng.integers(0, len(pool)))]


def _init_phase3_dynamic_target(
    bridge: Paper1EnvBridge,
    target_cfg: dict[str, Any],
    rng: np.random.Generator,
    init_mode: str,
) -> tuple[TargetTruthState, Any]:
    sampled_truth, internal = sample_phase3_initial_target(target_cfg, rng)
    if init_mode == "random_water":
        return sampled_truth, internal
    if init_mode != "paper1_goal":
        raise ValueError(f"Unsupported phase3 target init mode: {init_mode}")

    paper1_goal = bridge.get_target_truth()
    pos_xy = np.asarray(paper1_goal.pos_world, dtype=float).reshape(-1)[:2]
    return (
        TargetTruthState(
            t=0.0,
            pos_world=pos_xy.astype(float),
            vel_world=np.asarray(sampled_truth.vel_world, dtype=float).reshape(-1)[:2],
            heading=float(sampled_truth.heading),
            motion_mode=str(sampled_truth.motion_mode),
        ),
        internal,
    )


def _vision_estimate_from_row(
    *,
    row: dict[str, Any],
    stage: str = "mid",
    pred_center_px: tuple[float, float],
    pred_conf: float,
    current_target_xy: np.ndarray,
    current_target_z: float,
    t: float,
    args: argparse.Namespace | None = None,
) -> tuple[TargetEstimateState, float]:
    meta = dict(row.get("meta", {}))
    label_crop_center = np.asarray(meta["crop_center_world"], dtype=float).reshape(2)
    label_target = meta["target_state_world"]
    label_target_xy = np.array([float(label_target["x"]), float(label_target["y"])], dtype=float)
    gsd = float(meta.get("gsd", row.get("gsd_km_per_px")))

    crop_center_for_current_target = np.asarray(current_target_xy, dtype=float).reshape(2) + (
        label_crop_center - label_target_xy
    )
    pred_world_xy = image_point_to_world_xy(
        pred_center_px,
        crop_center_for_current_target,
        gsd,
        image_size=(256, 256),
    )
    error_km = float(np.linalg.norm(pred_world_xy - np.asarray(current_target_xy, dtype=float).reshape(2)))
    obs = VisionObservation(
        t=float(t),
        detected=True,
        center_px=(float(pred_center_px[0]), float(pred_center_px[1])),
        bbox_xywh=None,
        score=float(pred_conf),
        crop_path=str(row.get("image_path")),
        crop_center_world=(float(crop_center_for_current_target[0]), float(crop_center_for_current_target[1])),
        gsd=float(gsd),
        meta={
            "source": "cnn_dataset_error_replay",
            "row_sequence_id": row.get("sequence_id"),
            "perception_stage": str(stage),
            "measurement_sigma_px": _vision_measurement_sigma_px(stage, pred_conf, args=args),
        },
    )
    est = vision_observation_to_target_estimate(
        obs,
        image_size=(256, 256),
        pixel_sigma=None,
        z_value=float(current_target_z),
    )
    return est, error_km


def _stage_gate_threshold(stage: str, args: argparse.Namespace) -> float:
    if stage == "terminal":
        return float(args.gate_terminal_km)
    if stage == "mid":
        return float(args.gate_mid_km)
    return float(args.gate_far_km)


def _stage_update_gain(stage: str, args: argparse.Namespace) -> float:
    if stage == "terminal":
        return float(np.clip(args.gain_terminal, 0.0, 1.0))
    if stage == "mid":
        return float(np.clip(args.gain_mid, 0.0, 1.0))
    return float(np.clip(args.gain_far, 0.0, 1.0))


def _vision_measurement_sigma_px(
    stage: str,
    pred_conf: float,
    *,
    args: argparse.Namespace | None = None,
) -> float:
    stage = str(stage)
    if stage == "terminal":
        base = float(getattr(args, "meas_sigma_terminal_px", 24.0))
    elif stage == "mid":
        base = float(getattr(args, "meas_sigma_mid_px", 32.0))
    else:
        base = float(getattr(args, "meas_sigma_far_px", 80.0))
    conf = float(np.clip(pred_conf, 0.35, 1.0))
    return float(base / conf)


def _initial_goal_prior_estimate(
    *,
    truth: TargetTruthState,
    target_z_km: float,
    position_std_km: float,
    velocity_std_km_s: float,
) -> TargetEstimateState:
    pos_xy = np.asarray(truth.pos_world, dtype=float).reshape(-1)[:2]
    pos = np.array([float(pos_xy[0]), float(pos_xy[1]), float(target_z_km)], dtype=float)
    vel = np.zeros(3, dtype=float)
    pos_var = max(float(position_std_km), 1.0e-6) ** 2
    vel_var = max(float(velocity_std_km_s), 1.0e-6) ** 2
    cov = np.diag([pos_var, pos_var, pos_var, vel_var, vel_var, vel_var]).astype(float)
    return TargetEstimateState(
        t=float(truth.t),
        pos_world_est=pos,
        vel_world_est=vel,
        cov=cov,
        obs_conf=1.0,
        obs_age=0.0,
        meta={
            "source": "initial_goal_prior",
            "valid": True,
            "position_std_km": float(position_std_km),
            "velocity_std_km_s": float(velocity_std_km_s),
        },
    )


def _gate_and_smooth_estimate(
    candidate: TargetEstimateState,
    previous: TargetEstimateState | None,
    *,
    stage: str,
    args: argparse.Namespace,
) -> tuple[TargetEstimateState, bool, float, float]:
    if previous is None or bool(args.disable_estimate_gating):
        meta = dict(candidate.meta or {})
        meta.update(
            {
                "gate_enabled": not bool(args.disable_estimate_gating),
                "gate_accepted": True,
                "gate_innovation_km": 0.0,
                "gate_threshold_km": _stage_gate_threshold(stage, args),
                "update_gain": 1.0,
            }
        )
        candidate.meta = meta
        return candidate, True, 0.0, 1.0

    cand_pos = np.asarray(candidate.pos_world_est, dtype=float).reshape(-1)
    prev_pos = np.asarray(previous.pos_world_est, dtype=float).reshape(-1)
    prev_vel = np.asarray(previous.vel_world_est, dtype=float).reshape(-1)
    dim = min(cand_pos.size, prev_pos.size)
    dt = max(float(candidate.t) - float(previous.t), 1.0)
    predicted = prev_pos.copy()
    if prev_vel.size >= dim:
        predicted[:dim] = prev_pos[:dim] + prev_vel[:dim] * dt
    innovation = float(np.linalg.norm(cand_pos[:2] - predicted[:2]))
    threshold = _stage_gate_threshold(stage, args)
    gain = _stage_update_gain(stage, args)

    if innovation > threshold:
        meta = dict(previous.meta or {})
        meta.update(
            {
                "source": "vision_observation_gated",
                "valid": True,
                "gate_enabled": True,
                "gate_accepted": False,
                "gate_innovation_km": float(innovation),
                "gate_threshold_km": float(threshold),
                "update_gain": 0.0,
                "rejected_measurement_pos_world": cand_pos.tolist(),
            }
        )
        cov = np.asarray(previous.cov, dtype=float).copy() * 1.5
        gated = TargetEstimateState(
            t=float(candidate.t),
            pos_world_est=predicted,
            vel_world_est=prev_vel.copy(),
            cov=cov,
            obs_conf=min(float(previous.obs_conf), float(candidate.obs_conf)) * 0.5,
            obs_age=float(previous.obs_age) + dt,
            meta=meta,
        )
        return gated, False, innovation, 0.0

    fused_pos = predicted.copy()
    fused_pos[:dim] = (1.0 - gain) * predicted[:dim] + gain * cand_pos[:dim]
    fused_vel = prev_vel.copy()
    if fused_vel.size >= dim:
        fused_vel[:dim] = (fused_pos[:dim] - prev_pos[:dim]) / dt
    else:
        fused_vel = np.zeros_like(fused_pos)
        fused_vel[:dim] = (fused_pos[:dim] - prev_pos[:dim]) / dt
    meta = dict(candidate.meta or {})
    meta.update(
        {
            "source": "vision_observation_gated",
            "valid": True,
            "gate_enabled": True,
            "gate_accepted": True,
            "gate_innovation_km": float(innovation),
            "gate_threshold_km": float(threshold),
            "update_gain": float(gain),
            "raw_measurement_pos_world": cand_pos.tolist(),
        }
    )
    fused_cov = (1.0 - gain) * np.asarray(previous.cov, dtype=float) + gain * np.asarray(candidate.cov, dtype=float)
    fused = TargetEstimateState(
        t=float(candidate.t),
        pos_world_est=fused_pos,
        vel_world_est=fused_vel,
        cov=fused_cov,
        obs_conf=float(candidate.obs_conf),
        obs_age=0.0,
        meta=meta,
    )
    return fused, True, innovation, gain


def _apply_estimate_filter(
    candidate: TargetEstimateState,
    previous: TargetEstimateState | None,
    *,
    stage: str,
    args: argparse.Namespace,
    kf: ConstantVelocityKalmanFilter | None,
) -> tuple[TargetEstimateState, bool, float, float]:
    if str(args.estimate_filter) == "none":
        return candidate, True, 0.0, 1.0
    if str(args.estimate_filter) == "gated_smooth":
        return _gate_and_smooth_estimate(candidate, previous, stage=stage, args=args)
    if kf is None:
        raise RuntimeError("Kalman filter requested but not initialized.")
    if stage == "terminal" and str(args.kf_terminal_mode) == "raw":
        meta = dict(candidate.meta or {})
        meta.update(
            {
                "source": "vision_observation_terminal_raw",
                "kalman_terminal_bypass": True,
                "kalman_accepted": True,
            }
        )
        candidate.meta = meta
        kf.reset_from_estimate(candidate)
        return candidate, True, 0.0, 1.0
    kf.process_accel_std = _kf_stage_process_accel_std(stage, args)
    gate = _kf_stage_threshold(stage, args)
    estimate, info = kf.update(candidate, gate_threshold=float(gate))
    return estimate, bool(info.accepted), float(info.innovation_norm), 1.0 if bool(info.accepted) else 0.0


def _kf_stage_threshold(stage: str, args: argparse.Namespace) -> float:
    if stage == "terminal":
        return float(args.kf_gate_terminal_km)
    if stage == "mid":
        return float(args.kf_gate_mid_km)
    return float(args.kf_gate_far_km)


def _kf_stage_process_accel_std(stage: str, args: argparse.Namespace) -> float:
    if stage == "terminal":
        return float(getattr(args, "kf_process_accel_std_terminal", args.kf_process_accel_std))
    if stage == "mid":
        return float(getattr(args, "kf_process_accel_std_mid", args.kf_process_accel_std))
    return float(getattr(args, "kf_process_accel_std_far", args.kf_process_accel_std))


def _active_gate_threshold(stage: str, args: argparse.Namespace) -> float:
    if str(args.estimate_filter) == "kalman":
        return _kf_stage_threshold(stage, args)
    return _stage_gate_threshold(stage, args)


def _set_goal_from_estimate(bridge: Paper1EnvBridge, estimate: TargetEstimateState, target_z_km: float) -> None:
    pos = np.asarray(estimate.pos_world_est, dtype=float).reshape(-1)
    pos3 = np.array([pos[0], pos[1], float(target_z_km)], dtype=float)
    bridge.env.goal = paper2_xyz_to_paper1_xyz(pos3, world_size_km=float(bridge.world_size_km)).astype(np.float32)
    if hasattr(bridge.env, "best_goal_distance_so_far"):
        bridge.env.best_goal_distance_so_far = bridge.env._goal_distance(bridge.env.state[:3])
    if hasattr(bridge.env, "last_segment_goal_distance"):
        bridge.env.last_segment_goal_distance = bridge.env.best_goal_distance_so_far
    if hasattr(bridge.env, "last_goal_reached_by_segment"):
        bridge.env.last_goal_reached_by_segment = False


def _terminal_pursuit_action(bridge: Paper1EnvBridge) -> np.ndarray:
    env = bridge.env
    state = np.asarray(env.state, dtype=float).reshape(-1)
    goal = np.asarray(env.goal, dtype=float).reshape(-1)
    rel = goal[:3] - state[:3]
    xy_norm = max(float(np.linalg.norm(rel[:2])), 1.0e-6)
    desired_gamma = float(math.atan2(float(rel[2]), xy_norm))
    desired_psi = float(math.atan2(float(rel[1]), float(rel[0])))
    delta_gamma = desired_gamma - float(state[3])
    delta_psi = float((desired_psi - float(state[4]) + math.pi) % (2.0 * math.pi) - math.pi)
    limits = bridge.get_aircraft_state().control_limits or {}
    delta_gamma_max = float(limits.get("delta_gamma_max", getattr(env.scenario, "delta_gamma_max", 0.14)))
    delta_psi_max = float(limits.get("delta_psi_max", getattr(env.scenario, "delta_psi_max", 0.2)))
    return np.array(
        [
            float(np.clip(delta_gamma, -delta_gamma_max, delta_gamma_max)),
            float(np.clip(delta_psi, -delta_psi_max, delta_psi_max)),
        ],
        dtype=np.float32,
    )


def _select_planner_action(
    bridge: Paper1EnvBridge,
    policy: Paper1TD3Policy,
    *,
    stage: str,
    range_pre: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, str]:
    td3_action = policy.act(bridge.env._get_obs())
    if stage != "terminal" or float(range_pre) > float(args.terminal_controller_range_km):
        return td3_action, "td3"
    mode = str(args.terminal_controller)
    if mode == "td3":
        return td3_action, "td3"
    pursuit_action = _terminal_pursuit_action(bridge)
    if mode == "pure_pursuit":
        return pursuit_action, "pure_pursuit"
    blend = float(np.clip(args.terminal_controller_blend, 0.0, 1.0))
    action = (1.0 - blend) * td3_action.astype(float) + blend * pursuit_action.astype(float)
    return action.astype(np.float32), "blend"


def _zone_violation(pos_world: np.ndarray, zones: list[NoFlyZoneState], *, include_safety_margin: bool) -> bool:
    pos = np.asarray(pos_world, dtype=float).reshape(-1)
    if pos.size < 3:
        return False
    for zone in zones:
        center = np.asarray(zone.center_world, dtype=float).reshape(-1)
        radius = float(zone.radius_world)
        if include_safety_margin:
            radius += float(zone.safety_margin)
        dxy = float(np.linalg.norm(pos[:2] - center[:2]))
        if zone.geometry == "hemisphere":
            if dxy <= radius:
                z_limit = math.sqrt(max(radius * radius - dxy * dxy, 0.0))
                if pos[2] <= center[2] + z_limit:
                    return True
        elif dxy <= radius:
            return True
    return False


def _episode_summary(rows: list[dict[str, Any]], capture_radius_km: float) -> dict[str, Any]:
    if not rows:
        return {
            "steps_executed": 0,
            "captured": False,
            "min_range_km": 0.0,
            "final_range_km": 0.0,
            "mean_est_error_km": 0.0,
            "mean_vision_error_km": 0.0,
            "zone_violation_count": 0,
            "safety_margin_violation_count": 0,
            "done_reason": "empty",
        }
    ranges = np.asarray([float(r["range_to_target_km"]) for r in rows], dtype=float)
    est_errors = np.asarray([float(r["target_est_error_km"]) for r in rows], dtype=float)
    vision_errors = np.asarray([float(r["vision_error_km"]) for r in rows], dtype=float)
    done_reason = str(rows[-1]["done_reason"])
    return {
        "steps_executed": int(len(rows)),
        "captured": bool(done_reason == "captured" or float(ranges.min()) <= float(capture_radius_km)),
        "min_range_km": float(ranges.min()),
        "final_range_km": float(ranges[-1]),
        "mean_est_error_km": float(est_errors.mean()) if est_errors.size else 0.0,
        "mean_vision_error_km": float(vision_errors.mean()) if vision_errors.size else 0.0,
        "zone_violation_count": int(sum(1 for r in rows if bool(r["zone_violation"]))),
        "safety_margin_violation_count": int(sum(1 for r in rows if bool(r["safety_margin_violation"]))),
        "done_reason": done_reason,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    target_cfg = cfg["phase3_target_motion"]
    stage_cfg = cfg["phase3_task_stages"]
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(args.dataset_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch, _ = _import_torch()
    device = _resolve_device(torch, args.device)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    vision_model, vision_ckpt, vision_model_type = _load_vision_model(
        Path(args.vision_weights).resolve(),
        device,
        requested=str(args.vision_model),
    )
    vision_input_size = int(args.vision_input_size) if int(args.vision_input_size) > 0 else int(vision_ckpt.get("input_size", 256))
    decode_method = str(args.decode_method)
    if decode_method == "auto":
        decode_method = str(vision_ckpt.get("decode_method", "softargmax"))
        if decode_method == "heatmap_argmax":
            decode_method = "argmax"

    vision_source = str(args.vision_source)
    replay_mode = vision_source == "replay_dataset"
    stage2_live_mode = vision_source in {"live_render", "stage2_live_render"}
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {"far": [], "mid": [], "terminal": []}
    live_renderer: Stage2Renderer | None = None
    phase3_live_renderer: Phase3MapRenderer | None = None
    if replay_mode:
        label_path = dataset_root / "labels" / f"{args.eval_split}.jsonl"
        rows = _read_jsonl(label_path, max_rows=args.max_vision_samples)
        grouped = _group_rows_by_stage(rows)
    elif stage2_live_mode:
        render_cfg = load_yaml(args.render_config)
        inventory_csv = Path(render_cfg["assets"]["inventory_csv"]).expanduser().resolve()
        if not inventory_csv.is_absolute():
            inventory_csv = (Path(__file__).resolve().parents[1] / inventory_csv).resolve()
        assets = load_asset_inventory(inventory_csv)
        registry = AssetRegistry(assets)
        live_assets_dir = Path(args.out_dir).resolve() / "_live_render_assets"
        live_assets_dir.mkdir(parents=True, exist_ok=True)
        live_renderer = Stage2Renderer(
            cfg=render_cfg,
            registry=registry,
            project_root=Path(__file__).resolve().parents[1],
            output_root=live_assets_dir,
            rng=np.random.default_rng(int(args.seed)),
        )
    else:
        generation_config_path = (
            Path(args.phase3_live_generation_config).resolve()
            if str(args.phase3_live_generation_config).strip()
            else (dataset_root / "meta" / "generation_config.json").resolve()
        )
        if not generation_config_path.exists():
            raise FileNotFoundError(
                f"Phase3 map live render needs a generation config: {generation_config_path}. "
                "Pass --phase3-live-generation-config or use a Phase3 rendered dataset root."
            )
        gen_cfg = json.loads(generation_config_path.read_text(encoding="utf-8"))
        assets_root = Path(str(gen_cfg["assets_root"])).resolve()
        target_assets_root = Path(str(gen_cfg.get("target_assets_root") or gen_cfg["assets_root"])).resolve()
        distractor_assets_root = Path(str(gen_cfg.get("distractor_assets_root") or gen_cfg["assets_root"])).resolve()
        water_mask_root = Path(str(gen_cfg.get("water_mask_root") or (assets_root / "water_masks_auto"))).resolve()
        background_filter = dict(gen_cfg.get("background_filter") or {})
        target_filter = dict(gen_cfg.get("target_template_filter") or {})
        distractor_filter = dict(gen_cfg.get("distractor_template_filter") or {})
        backgrounds_by_split = _collect_backgrounds(
            assets_root,
            water_mask_root,
            skip_review=True,
            allow_keywords=tuple(background_filter.get("allow_keywords") or ()),
            reject_keywords=tuple(background_filter.get("reject_keywords") or ()),
        )
        targets_by_split = _collect_targets(
            target_assets_root,
            allow_keywords=tuple(target_filter.get("allow_keywords") or ()),
            reject_keywords=tuple(target_filter.get("reject_keywords") or ()),
        )
        distractors_by_split = _collect_distractors(
            distractor_assets_root,
            allow_keywords=tuple(distractor_filter.get("allow_keywords") or ()),
            reject_keywords=tuple(distractor_filter.get("reject_keywords") or ()),
        )
        policy_cfg = dict(gen_cfg.get("sequence_policy") or {})
        local_map_size_km = float(args.phase3_live_local_map_size_km)
        if local_map_size_km <= 0.0:
            far_crop_km = float(stage_cfg["far"]["gsd_km_per_px"]) * float(stage_cfg.get("image_size", 256))
            motion_budget_km = float(target_cfg["speed_range"]["max"]) * float(args.steps) * 2.5
            local_map_size_km = max(float(gen_cfg.get("local_map_size_km") or 24.0), far_crop_km + motion_budget_km)
        phase3_live_renderer = Phase3MapRenderer(
            stage_cfg=stage_cfg,
            backgrounds_by_split=backgrounds_by_split,
            targets_by_split=targets_by_split,
            distractors_by_split=distractors_by_split,
            rng=np.random.default_rng(int(args.seed)),
            local_map_size_km=local_map_size_km,
            min_target_water_ratio=float(policy_cfg.get("min_target_water_ratio", 0.98)),
            min_target_visibility=0.35,
            distractor_count_min=int(policy_cfg.get("distractor_count_min", 0)),
            distractor_count_max=int(policy_cfg.get("distractor_count_max", 0)),
            distractor_scale_min=float(policy_cfg.get("distractor_scale_min", 0.45)),
            distractor_scale_max=float(policy_cfg.get("distractor_scale_max", 0.85)),
            min_distractor_target_distance_px=float(policy_cfg.get("min_distractor_target_distance_px", 48.0)),
            scene_attempts=128,
        )
    paper1_curriculum_mix = parse_curriculum_mix(
        args.paper1_curriculum_mix,
        fallback_level=str(args.paper1_curriculum_level),
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    visual_audit_dir = out_dir / "visuals" / "vision_audit"
    visual_audit_saved = 0
    live_render_failure_count = 0
    live_render_reanchor_count = 0
    policy: Paper1TD3Policy | None = None
    policy_diag: dict[str, Any] = {}
    env_source = "unknown"

    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + ep
        ep_rng = np.random.default_rng(ep_seed)
        vision_rng = np.random.default_rng(ep_seed + 100000)
        bridge = Paper1EnvBridge(seed=ep_seed, curriculum_mix=paper1_curriculum_mix)
        env_source = str(bridge.env_source)
        bridge.reset(seed=ep_seed)
        if policy is None:
            policy = Paper1TD3Policy.from_env(
                bridge.env,
                checkpoint_path=args.td3_checkpoint,
                model_type=str(args.model),
                device=str(args.device),
                allow_random_init=False,
            )
            policy_diag = policy.diagnostics(bridge.env._get_obs())

        live_scene = live_renderer.create_live_scene(split=str(args.render_split)) if live_renderer is not None else None
        truth2d, internal = _init_phase3_dynamic_target(
            bridge,
            target_cfg,
            ep_rng,
            str(args.phase3_target_init),
        )
        phase3_live_scene = (
            _create_phase3_live_scene(
                phase3_live_renderer,
                split=str(args.render_split),
                sequence_id=f"phase3_live_{ep:04d}",
                target_xy=np.asarray(truth2d.pos_world, dtype=float).reshape(-1)[:2],
            )
            if phase3_live_renderer is not None
            else None
        )
        zones = bridge.get_no_fly_zones()
        ep_rows: list[dict[str, Any]] = []
        capture_radius_km = (
            float(args.capture_radius_km)
            if args.capture_radius_km is not None
            else float(getattr(bridge.env.scenario, "goal_radius", 5.0))
        )
        initial_target_z = float(np.asarray(bridge.env.goal, dtype=float).reshape(-1)[2])
        prev_estimate: TargetEstimateState | None = None
        if bool(args.bootstrap_estimate_from_goal) and str(args.phase3_target_init) == "paper1_goal":
            prev_estimate = _initial_goal_prior_estimate(
                truth=truth2d,
                target_z_km=initial_target_z,
                position_std_km=float(args.initial_goal_position_std_km),
                velocity_std_km_s=float(args.initial_goal_velocity_std_km_s),
            )
        kf: ConstantVelocityKalmanFilter | None = None
        if str(args.estimate_filter) == "kalman":
            kf = ConstantVelocityKalmanFilter(
                dim=3,
                process_accel_std=float(args.kf_process_accel_std),
                max_reject_streak=int(args.kf_max_reject_streak),
                nis_gate_threshold=float(args.kf_nis_gate_threshold),
                max_obs_age_before_reinit=float(args.kf_max_obs_age_before_reinit),
            )
            if prev_estimate is not None:
                kf.reset_from_estimate(prev_estimate)

        for step_idx in range(int(args.steps)):
            live_render_retry_count = 0
            aircraft = bridge.get_aircraft_state()
            target_z = float(np.asarray(bridge.env.goal, dtype=float).reshape(-1)[2])
            truth_pos = np.array([truth2d.pos_world[0], truth2d.pos_world[1], target_z], dtype=float)
            aircraft_pos_pre = np.asarray(aircraft.pos_world, dtype=float)
            range_pre = float(np.linalg.norm(aircraft_pos_pre[:3] - truth_pos[:3]))
            stage = _stage_for_range(range_pre, stage_cfg)
            if replay_mode:
                sample_row = _sample_stage_row(grouped, stage, vision_rng)
                image_path = _resolve_image_path(str(sample_row["image_path"]), project_root)
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(f"Could not read vision sample image: {image_path}")
            elif stage2_live_mode:
                if live_renderer is None or live_scene is None:
                    raise RuntimeError("Live renderer was not initialized.")
                live_state = _target_truth_to_world_state(truth2d)
                live_frame = live_renderer.render_live_frame(live_scene, live_state, stage)
                sample_row = _live_frame_to_phase3_row(
                    live_frame=live_frame,
                    truth=truth2d,
                    stage=stage,
                    stage_cfg=stage_cfg,
                    image_path=f"live_render/ep{ep:03d}_step{step_idx:04d}.png",
                    sequence_id=f"live_{ep:04d}",
                    frame_id=f"{step_idx:04d}",
                )
                image = live_frame.image_bgr
            else:
                if phase3_live_renderer is None or phase3_live_scene is None:
                    raise RuntimeError("Phase3 map live renderer was not initialized.")
                max_retries = max(0, int(args.phase3_live_render_retries))
                last_render_error: RuntimeError | None = None
                live_result = None
                for attempt in range(max_retries + 1):
                    phase3_live_renderer.rng = np.random.default_rng(
                        ep_seed * 1000003 + step_idx + attempt * 9176
                    )
                    try:
                        live_result = phase3_live_renderer.render_truth(
                            phase3_live_scene,
                            truth=truth2d,
                            stage=stage,
                            frame_id=step_idx,
                        )
                        live_render_retry_count = int(attempt)
                        break
                    except RuntimeError as exc:
                        last_render_error = exc
                        live_render_failure_count += 1
                        if (not bool(args.phase3_live_render_reanchor_on_failure)) or attempt >= max_retries:
                            raise RuntimeError(
                                f"Phase3 map live render failed after {attempt + 1} attempt(s), "
                                f"episode={ep}, step={step_idx}, stage={stage}. last_error={exc}"
                            ) from exc
                        live_render_reanchor_count += 1
                        phase3_live_scene = _create_phase3_live_scene(
                            phase3_live_renderer,
                            split=str(args.render_split),
                            sequence_id=f"phase3_live_{ep:04d}_reanchor_{step_idx:04d}_{attempt + 1}",
                            target_xy=np.asarray(truth2d.pos_world, dtype=float).reshape(-1)[:2],
                        )
                if live_result is None:
                    raise RuntimeError(
                        f"Phase3 map live render failed unexpectedly, episode={ep}, "
                        f"step={step_idx}, stage={stage}. last_error={last_render_error}"
                    )
                sample_row = _phase3_map_frame_to_row(
                    result=live_result,
                    truth=truth2d,
                    stage=stage,
                    stage_cfg=stage_cfg,
                    image_path=f"phase3_map_live/ep{ep:03d}_step{step_idx:04d}.png",
                    sequence_id=f"phase3_map_live_{ep:04d}",
                    frame_id=f"{step_idx:04d}",
                )
                sample_row.setdefault("meta", {})["live_render_retry_count"] = int(live_render_retry_count)
                sample_row["meta"]["live_render_reanchor_count_total"] = int(live_render_reanchor_count)
                image = live_result.image_bgr

            pred_x, pred_y, pred_conf = _predict_vision(
                vision_model,
                vision_model_type,
                image,
                device,
                input_size=int(vision_input_size),
                decode_method=decode_method,
                softargmax_temperature=float(vision_ckpt.get("softargmax_temperature", 20.0)),
            )
            raw_estimate, vision_error_km = _vision_estimate_from_row(
                row=sample_row,
                stage=stage,
                pred_center_px=(pred_x, pred_y),
                pred_conf=pred_conf,
                current_target_xy=np.asarray(truth2d.pos_world, dtype=float).reshape(2),
                current_target_z=target_z,
                t=float(aircraft.t),
                args=args,
            )
            estimate, gate_accepted, gate_innovation_km, gate_gain = _apply_estimate_filter(
                raw_estimate,
                prev_estimate,
                stage=stage,
                args=args,
                kf=kf,
            )
            prev_estimate = estimate
            _set_goal_from_estimate(bridge, estimate, target_z)
            action, action_source = _select_planner_action(
                bridge,
                policy,
                stage=stage,
                range_pre=range_pre,
                args=args,
            )
            step_result = bridge.step(action)

            aircraft_pos = np.asarray(step_result.observation.aircraft_pos_world, dtype=float)
            range_to_target = float(np.linalg.norm(aircraft_pos[:3] - truth_pos[:3]))
            est_error = float(np.linalg.norm(np.asarray(estimate.pos_world_est, dtype=float)[:3] - truth_pos[:3]))
            done = bool(step_result.done)
            done_reason = str(step_result.info.reason)
            true_capture = range_to_target <= float(capture_radius_km)
            if str(args.capture_mode) == "true_target":
                if true_capture:
                    done = True
                    done_reason = "captured"
                elif done_reason == "captured":
                    done = False
                    done_reason = "estimate_goal_reached"
            violation = _zone_violation(aircraft_pos, zones, include_safety_margin=False)
            safety_violation = _zone_violation(aircraft_pos, zones, include_safety_margin=True)
            vision_audit_path = ""
            vision_pixel_error_px = float("nan")
            if int(args.visual_audit_count) > 0 and visual_audit_saved < int(args.visual_audit_count):
                vision_audit_path, vision_pixel_error_px = _write_vision_audit_image(
                    image_bgr=image,
                    row=sample_row,
                    pred_center_px=(pred_x, pred_y),
                    stage=stage,
                    episode=ep,
                    step=step_idx,
                    out_dir=visual_audit_dir,
                )
                if vision_audit_path:
                    visual_audit_saved += 1
            gt_center = _sample_gt_center_px(sample_row)
            row = {
                "episode": int(ep),
                "step": int(step_idx),
                "t": float(aircraft.t),
                "vision_source": vision_source,
                "observer": str(vision_model_type),
                "estimate_filter": str(args.estimate_filter),
                "planner": f"paper1_{args.model}_td3",
                "render_mode": str(sample_row.get("render_mode", "replay_dataset")),
                "vision_stage": stage,
                "vision_sequence_id": str(sample_row.get("sequence_id")),
                "vision_frame_id": str(sample_row.get("frame_id")),
                "vision_gt_x": float(gt_center[0]) if gt_center is not None else float("nan"),
                "vision_gt_y": float(gt_center[1]) if gt_center is not None else float("nan"),
                "vision_pred_x": float(pred_x),
                "vision_pred_y": float(pred_y),
                "vision_pixel_error_px": float(vision_pixel_error_px),
                "vision_conf": float(pred_conf),
                "vision_gate_accepted": bool(gate_accepted),
                "vision_gate_innovation_km": float(gate_innovation_km),
                "vision_gate_gain": float(gate_gain),
                "vision_gate_threshold_km": float(_active_gate_threshold(stage, args)),
                "aircraft_x": float(aircraft_pos[0]),
                "aircraft_y": float(aircraft_pos[1]),
                "aircraft_z": float(aircraft_pos[2]),
                "target_x": float(truth_pos[0]),
                "target_y": float(truth_pos[1]),
                "target_z": float(truth_pos[2]),
                "estimate_x": float(estimate.pos_world_est[0]),
                "estimate_y": float(estimate.pos_world_est[1]),
                "estimate_z": float(estimate.pos_world_est[2]),
                "action_source": str(action_source),
                "action_delta_gamma": float(action[0]),
                "action_delta_psi": float(action[1]),
                "range_to_target_km": range_to_target,
                "target_est_error_km": est_error,
                "vision_error_km": float(vision_error_km),
                "zone_violation": bool(violation),
                "safety_margin_violation": bool(safety_violation),
                "done": bool(done),
                "done_reason": str(done_reason),
                "reward": float(step_result.reward),
                "vision_audit_image_path": str(vision_audit_path),
                "live_render_retry_count": int(live_render_retry_count),
                "live_render_reanchor_count_total": int(live_render_reanchor_count),
            }
            ep_rows.append(row)
            all_rows.append(row)
            if done:
                break
            truth2d = propagate_phase3_target_truth(
                truth2d,
                internal,
                target_cfg,
                ep_rng,
                aircraft_pos_world=aircraft_pos[:2],
            )

        summary = _episode_summary(ep_rows, capture_radius_km)
        summary.update({"episode": int(ep), "seed": int(ep_seed), "observer": str(vision_model_type)})
        summaries.append(summary)

    traj_path = out_dir / "trajectory.jsonl"
    with traj_path.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = out_dir / "summary.csv"
    if summaries:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    min_ranges = np.asarray([float(s["min_range_km"]) for s in summaries], dtype=float)
    final_ranges = np.asarray([float(s["final_range_km"]) for s in summaries], dtype=float)
    est_errors = np.asarray([float(s["mean_est_error_km"]) for s in summaries], dtype=float)
    vision_errors = np.asarray([float(s["mean_vision_error_km"]) for s in summaries], dtype=float)
    vision_px_errors = np.asarray([float(r["vision_pixel_error_px"]) for r in all_rows if np.isfinite(float(r.get("vision_pixel_error_px", float("nan"))))], dtype=float)
    gate_rejections = int(sum(1 for r in all_rows if not bool(r.get("vision_gate_accepted", True))))
    violation_counts = np.asarray([int(s["zone_violation_count"]) for s in summaries], dtype=int)
    safety_violation_counts = np.asarray([int(s["safety_margin_violation_count"]) for s in summaries], dtype=int)
    capture_count = int(sum(1 for s in summaries if bool(s["captured"])))

    report = {
        "task": "run_phase3_vision_td3",
        "purpose": "vision_observer_closed_loop_eval",
        "config": str(args.config),
        "env_source": env_source,
        "vision_source": vision_source,
        "planner": f"paper1_{args.model}_td3",
        "observer": str(vision_model_type),
        "observer_arch": str(
            vision_ckpt.get(
                "cnn_arch" if vision_model_type == "cnn_heatmap" else "snn_arch",
                vision_ckpt.get("arch", "unknown"),
            )
        ),
        "estimate_filter": str(args.estimate_filter),
        "dataset_root": str(dataset_root),
        "eval_split": str(args.eval_split),
        "render_config": str(args.render_config),
        "render_split": str(args.render_split),
        "vision_weights_path": str(Path(args.vision_weights).expanduser()),
        "td3_checkpoint_path": str(Path(args.td3_checkpoint).expanduser()),
        "vision_train_split": str(vision_ckpt.get("train_split", "unknown")),
        "vision_val_split": str(vision_ckpt.get("val_split", "unknown")),
        "phase3_target_init": str(args.phase3_target_init),
        "capture_mode": str(args.capture_mode),
        "terminal_controller": str(args.terminal_controller),
        "terminal_controller_range_km": float(args.terminal_controller_range_km),
        "terminal_controller_blend": float(args.terminal_controller_blend),
        "paper1_curriculum_level": str(args.paper1_curriculum_level),
        "paper1_curriculum_mix": paper1_curriculum_mix,
        "episodes": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "seed": int(args.seed),
        "unit": "km",
        "capture_radius_km": float(args.capture_radius_km if args.capture_radius_km is not None else 5.0),
        "vision_sample_counts": {k: int(len(v)) for k, v in grouped.items()} if replay_mode else {},
        "live_render_enabled": bool(not replay_mode),
        "phase3_live_render_retries": int(args.phase3_live_render_retries),
        "phase3_live_render_reanchor_on_failure": bool(args.phase3_live_render_reanchor_on_failure),
        "estimate_gating": {
            "enabled": not bool(args.disable_estimate_gating),
            "gate_far_km": float(args.gate_far_km),
            "gate_mid_km": float(args.gate_mid_km),
            "gate_terminal_km": float(args.gate_terminal_km),
            "gain_far": float(args.gain_far),
            "gain_mid": float(args.gain_mid),
            "gain_terminal": float(args.gain_terminal),
            "kf_terminal_mode": str(args.kf_terminal_mode),
            "kf_process_accel_std_far": float(args.kf_process_accel_std_far),
            "kf_process_accel_std_mid": float(args.kf_process_accel_std_mid),
            "kf_process_accel_std_terminal": float(args.kf_process_accel_std_terminal),
            "kf_gate_far_km": float(args.kf_gate_far_km),
            "kf_gate_mid_km": float(args.kf_gate_mid_km),
            "kf_gate_terminal_km": float(args.kf_gate_terminal_km),
            "kf_max_reject_streak": int(args.kf_max_reject_streak),
            "kf_max_obs_age_before_reinit": float(args.kf_max_obs_age_before_reinit),
            "kf_nis_gate_threshold": float(args.kf_nis_gate_threshold),
            "meas_sigma_far_px": float(args.meas_sigma_far_px),
            "meas_sigma_mid_px": float(args.meas_sigma_mid_px),
            "meas_sigma_terminal_px": float(args.meas_sigma_terminal_px),
            "bootstrap_estimate_from_goal": bool(args.bootstrap_estimate_from_goal),
            "initial_goal_position_std_km": float(args.initial_goal_position_std_km),
            "initial_goal_velocity_std_km_s": float(args.initial_goal_velocity_std_km_s),
        },
        "policy_diagnostics": policy_diag,
        "metrics": {
            "episodes_completed": int(len(summaries)),
            "total_steps": int(len(all_rows)),
            "capture_count": capture_count,
            "capture_rate": float(capture_count / max(1, len(summaries))),
            "min_range_mean_km": float(min_ranges.mean()) if min_ranges.size else 0.0,
            "final_range_mean_km": float(final_ranges.mean()) if final_ranges.size else 0.0,
            "target_est_error_mean_km": float(est_errors.mean()) if est_errors.size else 0.0,
            "vision_error_mean_km": float(vision_errors.mean()) if vision_errors.size else 0.0,
            "vision_pixel_error_mean_px": float(vision_px_errors.mean()) if vision_px_errors.size else 0.0,
            "vision_pixel_error_p90_px": float(np.percentile(vision_px_errors, 90)) if vision_px_errors.size else 0.0,
            "vision_gate_rejection_count": int(gate_rejections),
            "vision_gate_rejection_rate": float(gate_rejections / max(1, len(all_rows))),
            "live_render_failure_count": int(live_render_failure_count),
            "live_render_reanchor_count": int(live_render_reanchor_count),
            "zone_violation_total": int(violation_counts.sum()) if violation_counts.size else 0,
            "zone_violation_episode_count": int((violation_counts > 0).sum()) if violation_counts.size else 0,
            "safety_margin_violation_total": int(safety_violation_counts.sum()) if safety_violation_counts.size else 0,
            "safety_margin_violation_episode_count": int((safety_violation_counts > 0).sum())
            if safety_violation_counts.size
            else 0,
        },
        "acceptance": {
            "all_episodes_ran": bool(len(summaries) == int(args.episodes)),
            "vision_weights_loaded": True,
            "checkpoint_loaded": bool(policy is not None and not policy.random_init),
            "no_zone_violations": bool((violation_counts.sum() if violation_counts.size else 0) == 0),
            "finite_ranges": bool(np.isfinite(min_ranges).all() and np.isfinite(final_ranges).all()),
            "vision_source_valid": bool(replay_mode or live_renderer is not None or phase3_live_renderer is not None),
        },
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "trajectory_path": str(traj_path),
            "summary_csv_path": str(csv_path),
            "live_render_assets_dir": str((out_dir / "_live_render_assets")) if not replay_mode else "",
            "vision_audit_dir": str(visual_audit_dir),
        },
    }
    report["accepted"] = bool(
        report["acceptance"]["all_episodes_ran"]
        and report["acceptance"]["vision_weights_loaded"]
        and report["acceptance"]["checkpoint_loaded"]
        and report["acceptance"]["finite_ranges"]
        and report["acceptance"]["no_zone_violations"]
        and report["acceptance"]["vision_source_valid"]
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
