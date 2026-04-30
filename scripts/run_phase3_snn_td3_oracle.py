from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from paper2.common.config import load_yaml
from paper2.common.types import NoFlyZoneState, TargetEstimateState, TargetTruthState
from paper2.env_adapter.paper1_bridge import Paper1EnvBridge
from paper2.env_adapter.phase3_target_motion import (
    propagate_phase3_target_truth,
    sample_phase3_initial_target,
)
from paper2.env_adapter.world_frame import paper2_xyz_to_paper1_xyz
from paper2.planning.paper1_td3_policy import Paper1TD3Policy
from paper2.tracking.vision_to_estimate import oracle_target_estimate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/env.yaml")
    p.add_argument("--observer", choices=["gt", "noisy"], default="gt")
    p.add_argument("--model", choices=["snn", "ann"], default="snn")
    p.add_argument("--td3-checkpoint", type=str, default=None)
    p.add_argument("--allow-random-init", action="store_true")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--noise-std-km", type=float, default=2.0)
    p.add_argument("--target-z-km", type=float, default=0.0)
    p.add_argument("--capture-radius-km", type=float, default=50.0)
    p.add_argument("--out-dir", type=str, default="outputs/phase3_snn_td3_oracle/gt_eval")
    return p.parse_args()


def _truth_with_z(truth: TargetTruthState, z_km: float) -> TargetTruthState:
    pos = np.asarray(truth.pos_world, dtype=float).reshape(-1)
    vel = np.asarray(truth.vel_world, dtype=float).reshape(-1)
    if pos.size == 2:
        pos = np.array([pos[0], pos[1], float(z_km)], dtype=float)
    if vel.size == 2:
        vel = np.array([vel[0], vel[1], 0.0], dtype=float)
    return TargetTruthState(
        t=float(truth.t),
        pos_world=pos,
        vel_world=vel,
        heading=float(truth.heading),
        motion_mode=str(truth.motion_mode),
    )


def _make_estimate(
    truth: TargetTruthState,
    *,
    observer: str,
    noise_std_km: float,
    rng: np.random.Generator,
) -> TargetEstimateState:
    if observer == "gt":
        return oracle_target_estimate(truth, noise_std_m=0.0, rng=rng, obs_conf=1.0)
    if observer == "noisy":
        return oracle_target_estimate(truth, noise_std_m=float(noise_std_km), rng=rng, obs_conf=0.9)
    raise ValueError(f"Unsupported observer: {observer}")


def _set_goal_from_estimate(bridge: Paper1EnvBridge, estimate: TargetEstimateState, target_z_km: float) -> None:
    pos = np.asarray(estimate.pos_world_est, dtype=float).reshape(-1)
    if pos.size == 2:
        pos3 = np.array([pos[0], pos[1], float(target_z_km)], dtype=float)
    else:
        pos3 = np.array([pos[0], pos[1], pos[2]], dtype=float)
    bridge.env.goal = paper2_xyz_to_paper1_xyz(pos3, world_size_km=float(bridge.world_size_km)).astype(np.float32)


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
            "zone_violation_count": 0,
            "safety_margin_violation_count": 0,
        }
    ranges = np.asarray([float(r["range_to_target_km"]) for r in rows], dtype=float)
    est_errors = np.asarray([float(r["target_est_error_km"]) for r in rows], dtype=float)
    violations = int(sum(1 for r in rows if bool(r["zone_violation"])))
    safety_violations = int(sum(1 for r in rows if bool(r["safety_margin_violation"])))
    return {
        "steps_executed": int(len(rows)),
        "captured": bool(float(ranges.min()) <= float(capture_radius_km)),
        "min_range_km": float(ranges.min()),
        "final_range_km": float(ranges[-1]),
        "mean_est_error_km": float(est_errors.mean()) if est_errors.size else 0.0,
        "zone_violation_count": violations,
        "safety_margin_violation_count": safety_violations,
        "done_reason": str(rows[-1]["done_reason"]),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    target_cfg = cfg["phase3_target_motion"]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    policy: Paper1TD3Policy | None = None
    policy_diag: dict[str, Any] = {}

    for ep in range(int(args.episodes)):
        ep_seed = int(args.seed) + ep
        ep_rng = np.random.default_rng(ep_seed)
        bridge = Paper1EnvBridge(seed=ep_seed)
        bridge.reset(seed=ep_seed)
        if policy is None:
            policy = Paper1TD3Policy.from_env(
                bridge.env,
                checkpoint_path=args.td3_checkpoint,
                model_type=str(args.model),
                device=str(args.device),
                allow_random_init=bool(args.allow_random_init),
            )
            policy_diag = policy.diagnostics(bridge.env._get_obs())

        zones = bridge.get_no_fly_zones()
        truth2d, internal = sample_phase3_initial_target(target_cfg, ep_rng)
        ep_rows: list[dict[str, Any]] = []
        done_reason = "running"

        for step_idx in range(int(args.steps)):
            aircraft = bridge.get_aircraft_state()
            truth = _truth_with_z(truth2d, float(args.target_z_km))
            estimate = _make_estimate(
                truth,
                observer=str(args.observer),
                noise_std_km=float(args.noise_std_km),
                rng=rng,
            )
            _set_goal_from_estimate(bridge, estimate, float(args.target_z_km))
            obs_vec = bridge.env._get_obs()
            action = policy.act(obs_vec)

            target_pos = np.asarray(truth.pos_world, dtype=float)
            aircraft_pos = np.asarray(aircraft.pos_world, dtype=float)
            range_to_target = float(np.linalg.norm(aircraft_pos[:3] - target_pos[:3]))
            est_error = float(np.linalg.norm(np.asarray(estimate.pos_world_est, dtype=float)[:3] - target_pos[:3]))
            violation = _zone_violation(aircraft_pos, zones, include_safety_margin=False)
            safety_violation = _zone_violation(aircraft_pos, zones, include_safety_margin=True)

            step_result = bridge.step(action)
            done_reason = str(step_result.info.reason)
            row = {
                "episode": int(ep),
                "step": int(step_idx),
                "t": float(aircraft.t),
                "observer": str(args.observer),
                "planner": f"paper1_{args.model}_td3",
                "aircraft_x": float(aircraft_pos[0]),
                "aircraft_y": float(aircraft_pos[1]),
                "aircraft_z": float(aircraft_pos[2]),
                "target_x": float(target_pos[0]),
                "target_y": float(target_pos[1]),
                "target_z": float(target_pos[2]),
                "estimate_x": float(estimate.pos_world_est[0]),
                "estimate_y": float(estimate.pos_world_est[1]),
                "estimate_z": float(estimate.pos_world_est[2]),
                "action_delta_gamma": float(action[0]),
                "action_delta_psi": float(action[1]),
                "range_to_target_km": range_to_target,
                "target_est_error_km": est_error,
                "zone_violation": bool(violation),
                "safety_margin_violation": bool(safety_violation),
                "done": bool(step_result.done),
                "done_reason": done_reason,
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

        summary = _episode_summary(ep_rows, float(args.capture_radius_km))
        summary.update({"episode": int(ep), "seed": int(ep_seed), "observer": str(args.observer)})
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
    violation_counts = np.asarray([int(s["zone_violation_count"]) for s in summaries], dtype=int)
    safety_violation_counts = np.asarray([int(s["safety_margin_violation_count"]) for s in summaries], dtype=int)
    capture_count = int(sum(1 for s in summaries if bool(s["captured"])))

    report = {
        "task": "run_phase3_snn_td3_oracle",
        "purpose": "main_planner_closed_loop_eval",
        "config": str(args.config),
        "paper1_root": None,
        "env_source": "paper2_local_paper1_physics",
        "planner": f"paper1_{args.model}_td3",
        "checkpoint_path": None if args.td3_checkpoint is None else str(Path(args.td3_checkpoint).expanduser()),
        "random_init": bool(policy.random_init if policy is not None else False),
        "observer": str(args.observer),
        "episodes": int(args.episodes),
        "steps_per_episode": int(args.steps),
        "seed": int(args.seed),
        "unit": "km",
        "capture_radius_km": float(args.capture_radius_km),
        "noise_std_km": float(args.noise_std_km if args.observer == "noisy" else 0.0),
        "policy_diagnostics": policy_diag,
        "metrics": {
            "episodes_completed": int(len(summaries)),
            "total_steps": int(len(all_rows)),
            "capture_count": capture_count,
            "capture_rate": float(capture_count / max(1, len(summaries))),
            "min_range_mean_km": float(min_ranges.mean()) if min_ranges.size else 0.0,
            "final_range_mean_km": float(final_ranges.mean()) if final_ranges.size else 0.0,
            "target_est_error_mean_km": float(est_errors.mean()) if est_errors.size else 0.0,
            "zone_violation_total": int(violation_counts.sum()) if violation_counts.size else 0,
            "zone_violation_episode_count": int((violation_counts > 0).sum()) if violation_counts.size else 0,
            "safety_margin_violation_total": int(safety_violation_counts.sum()) if safety_violation_counts.size else 0,
            "safety_margin_violation_episode_count": int((safety_violation_counts > 0).sum())
            if safety_violation_counts.size
            else 0,
        },
        "acceptance": {
            "all_episodes_ran": bool(len(summaries) == int(args.episodes)),
            "checkpoint_loaded": bool(policy is not None and not policy.random_init),
            "no_zone_violations": bool((violation_counts.sum() if violation_counts.size else 0) == 0),
            "gt_est_error_zero": bool(str(args.observer) != "gt" or (est_errors.size > 0 and float(est_errors.max()) < 1e-9)),
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
        and report["acceptance"]["checkpoint_loaded"]
        and report["acceptance"]["finite_ranges"]
        and report["acceptance"]["no_zone_violations"]
        and (str(args.observer) != "gt" or report["acceptance"]["gt_est_error_zero"])
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
