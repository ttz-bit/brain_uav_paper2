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
from paper2.planning.paper1_td3_policy import Paper1TD3Policy
from paper2.tracking.vision_to_estimate import image_point_to_world_xy, vision_observation_to_target_estimate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/env.yaml")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--cnn-weights", type=str, required=True)
    p.add_argument("--td3-checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["snn", "ann"], default="snn")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--paper1-curriculum-level", choices=["easy", "easy_two_zone", "medium", "hard"], default="hard")
    p.add_argument("--paper1-curriculum-mix", type=str, default=None)
    p.add_argument("--phase3-target-init", choices=["paper1_goal", "random_water"], default="paper1_goal")
    p.add_argument("--target-z-policy", choices=["keep_current_goal_z"], default="keep_current_goal_z")
    p.add_argument("--capture-radius-km", type=float, default=None)
    p.add_argument("--max-vision-samples", type=int, default=None)
    p.add_argument("--disable-estimate-gating", action="store_true")
    p.add_argument("--gate-far-km", type=float, default=300.0)
    p.add_argument("--gate-mid-km", type=float, default=120.0)
    p.add_argument("--gate-terminal-km", type=float, default=25.0)
    p.add_argument("--gain-far", type=float, default=0.25)
    p.add_argument("--gain-mid", type=float, default=0.50)
    p.add_argument("--gain-terminal", type=float, default=0.80)
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


def _load_cnn(weights: Path, device: str):
    torch, nn = _import_torch()
    model = _make_cnn_model(nn).to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _predict_cnn(model: Any, image_bgr: np.ndarray, device: str) -> tuple[float, float, float]:
    torch, _ = _import_torch()
    with torch.no_grad():
        x = torch.from_numpy(_to_tensor_image(image_bgr)).unsqueeze(0).to(device)
        pred = model(x)[0].detach().cpu().numpy()
    h, w = image_bgr.shape[:2]
    return (
        float(np.clip(pred[0], 0.0, 1.0) * w),
        float(np.clip(pred[1], 0.0, 1.0) * h),
        float(np.clip(pred[2], 0.0, 1.0)),
    )


def _stage_for_range(range_km: float, stage_cfg: dict[str, Any]) -> str:
    r = float(range_km)
    for name in ("terminal", "mid", "far"):
        cfg = stage_cfg[name]
        if float(cfg["range_min_km"]) <= r <= float(cfg["range_max_km"]):
            return name
    if r < float(stage_cfg["terminal"]["range_min_km"]):
        return "terminal"
    return "far"


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
    pred_center_px: tuple[float, float],
    pred_conf: float,
    current_target_xy: np.ndarray,
    current_target_z: float,
    t: float,
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
        meta={"source": "cnn_dataset_error_replay", "row_sequence_id": row.get("sequence_id")},
    )
    est = vision_observation_to_target_estimate(
        obs,
        image_size=(256, 256),
        pixel_sigma=8.0,
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
    cnn_model, cnn_ckpt = _load_cnn(Path(args.cnn_weights).resolve(), device)

    label_path = dataset_root / "labels" / f"{args.eval_split}.jsonl"
    rows = _read_jsonl(label_path, max_rows=args.max_vision_samples)
    grouped = _group_rows_by_stage(rows)
    paper1_curriculum_mix = parse_curriculum_mix(
        args.paper1_curriculum_mix,
        fallback_level=str(args.paper1_curriculum_level),
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
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

        truth2d, internal = _init_phase3_dynamic_target(
            bridge,
            target_cfg,
            ep_rng,
            str(args.phase3_target_init),
        )
        zones = bridge.get_no_fly_zones()
        ep_rows: list[dict[str, Any]] = []
        capture_radius_km = (
            float(args.capture_radius_km)
            if args.capture_radius_km is not None
            else float(getattr(bridge.env.scenario, "goal_radius", 5.0))
        )
        prev_estimate: TargetEstimateState | None = None

        for step_idx in range(int(args.steps)):
            aircraft = bridge.get_aircraft_state()
            target_z = float(np.asarray(bridge.env.goal, dtype=float).reshape(-1)[2])
            truth_pos = np.array([truth2d.pos_world[0], truth2d.pos_world[1], target_z], dtype=float)
            aircraft_pos_pre = np.asarray(aircraft.pos_world, dtype=float)
            range_pre = float(np.linalg.norm(aircraft_pos_pre[:3] - truth_pos[:3]))
            stage = _stage_for_range(range_pre, stage_cfg)
            sample_row = _sample_stage_row(grouped, stage, vision_rng)
            image_path = _resolve_image_path(str(sample_row["image_path"]), project_root)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read vision sample image: {image_path}")
            pred_x, pred_y, pred_conf = _predict_cnn(cnn_model, image, device)
            raw_estimate, vision_error_km = _vision_estimate_from_row(
                row=sample_row,
                pred_center_px=(pred_x, pred_y),
                pred_conf=pred_conf,
                current_target_xy=np.asarray(truth2d.pos_world, dtype=float).reshape(2),
                current_target_z=target_z,
                t=float(aircraft.t),
            )
            estimate, gate_accepted, gate_innovation_km, gate_gain = _gate_and_smooth_estimate(
                raw_estimate,
                prev_estimate,
                stage=stage,
                args=args,
            )
            prev_estimate = estimate
            _set_goal_from_estimate(bridge, estimate, target_z)
            action = policy.act(bridge.env._get_obs())
            step_result = bridge.step(action)

            aircraft_pos = np.asarray(step_result.observation.aircraft_pos_world, dtype=float)
            range_to_target = float(np.linalg.norm(aircraft_pos[:3] - truth_pos[:3]))
            est_error = float(np.linalg.norm(np.asarray(estimate.pos_world_est, dtype=float)[:3] - truth_pos[:3]))
            violation = _zone_violation(aircraft_pos, zones, include_safety_margin=False)
            safety_violation = _zone_violation(aircraft_pos, zones, include_safety_margin=True)
            row = {
                "episode": int(ep),
                "step": int(step_idx),
                "t": float(aircraft.t),
                "observer": "cnn",
                "planner": f"paper1_{args.model}_td3",
                "vision_stage": stage,
                "vision_sequence_id": str(sample_row.get("sequence_id")),
                "vision_frame_id": str(sample_row.get("frame_id")),
                "vision_pred_x": float(pred_x),
                "vision_pred_y": float(pred_y),
                "vision_conf": float(pred_conf),
                "vision_gate_accepted": bool(gate_accepted),
                "vision_gate_innovation_km": float(gate_innovation_km),
                "vision_gate_gain": float(gate_gain),
                "vision_gate_threshold_km": float(_stage_gate_threshold(stage, args)),
                "aircraft_x": float(aircraft_pos[0]),
                "aircraft_y": float(aircraft_pos[1]),
                "aircraft_z": float(aircraft_pos[2]),
                "target_x": float(truth_pos[0]),
                "target_y": float(truth_pos[1]),
                "target_z": float(truth_pos[2]),
                "estimate_x": float(estimate.pos_world_est[0]),
                "estimate_y": float(estimate.pos_world_est[1]),
                "estimate_z": float(estimate.pos_world_est[2]),
                "action_delta_gamma": float(action[0]),
                "action_delta_psi": float(action[1]),
                "range_to_target_km": range_to_target,
                "target_est_error_km": est_error,
                "vision_error_km": float(vision_error_km),
                "zone_violation": bool(violation),
                "safety_margin_violation": bool(safety_violation),
                "done": bool(step_result.done),
                "done_reason": str(step_result.info.reason),
                "reward": float(step_result.reward),
            }
            ep_rows.append(row)
            all_rows.append(row)
            if step_result.done:
                break
            truth2d = propagate_phase3_target_truth(
                truth2d,
                internal,
                target_cfg,
                ep_rng,
                aircraft_pos_world=aircraft_pos[:2],
            )

        summary = _episode_summary(ep_rows, capture_radius_km)
        summary.update({"episode": int(ep), "seed": int(ep_seed), "observer": "cnn"})
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
    gate_rejections = int(sum(1 for r in all_rows if not bool(r.get("vision_gate_accepted", True))))
    violation_counts = np.asarray([int(s["zone_violation_count"]) for s in summaries], dtype=int)
    safety_violation_counts = np.asarray([int(s["safety_margin_violation_count"]) for s in summaries], dtype=int)
    capture_count = int(sum(1 for s in summaries if bool(s["captured"])))

    report = {
        "task": "run_phase3_vision_td3",
        "purpose": "cnn_observer_closed_loop_eval",
        "config": str(args.config),
        "env_source": env_source,
        "planner": f"paper1_{args.model}_td3",
        "observer": "cnn",
        "dataset_root": str(dataset_root),
        "eval_split": str(args.eval_split),
        "cnn_weights_path": str(Path(args.cnn_weights).expanduser()),
        "td3_checkpoint_path": str(Path(args.td3_checkpoint).expanduser()),
        "cnn_train_split": str(cnn_ckpt.get("train_split", "unknown")),
        "cnn_val_split": str(cnn_ckpt.get("val_split", "unknown")),
        "phase3_target_init": str(args.phase3_target_init),
        "paper1_curriculum_level": str(args.paper1_curriculum_level),
        "paper1_curriculum_mix": paper1_curriculum_mix,
        "episodes": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "seed": int(args.seed),
        "unit": "km",
        "capture_radius_km": float(args.capture_radius_km if args.capture_radius_km is not None else 5.0),
        "vision_sample_counts": {k: int(len(v)) for k, v in grouped.items()},
        "estimate_gating": {
            "enabled": not bool(args.disable_estimate_gating),
            "gate_far_km": float(args.gate_far_km),
            "gate_mid_km": float(args.gate_mid_km),
            "gate_terminal_km": float(args.gate_terminal_km),
            "gain_far": float(args.gain_far),
            "gain_mid": float(args.gain_mid),
            "gain_terminal": float(args.gain_terminal),
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
            "vision_gate_rejection_count": int(gate_rejections),
            "vision_gate_rejection_rate": float(gate_rejections / max(1, len(all_rows))),
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
        },
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "trajectory_path": str(traj_path),
            "summary_csv_path": str(csv_path),
        },
    }
    report["accepted"] = bool(
        report["acceptance"]["all_episodes_ran"]
        and report["acceptance"]["vision_weights_loaded"]
        and report["acceptance"]["checkpoint_loaded"]
        and report["acceptance"]["finite_ranges"]
        and report["acceptance"]["no_zone_violations"]
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
