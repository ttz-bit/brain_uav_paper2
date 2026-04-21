from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.dynamic_env_phase1a import DynamicTargetEnvPhase1A
from paper2.env_adapter.env_types import EnvStepResult


def _simple_truth_chase_action(aircraft_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    vec = target_pos - aircraft_pos
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-8:
        return np.zeros(2, dtype=float)
    return vec / nrm


def _safe_truth_chase_action(
    aircraft_pos: np.ndarray,
    target_pos: np.ndarray,
    area: dict,
    zones: list,
    margin: float = 30.0,
) -> np.ndarray:
    desired = _simple_truth_chase_action(aircraft_pos, target_pos)
    repel = np.zeros(2, dtype=float)

    # Boundary repulsion
    dx_min = float(aircraft_pos[0] - area["x_min"])
    dx_max = float(area["x_max"] - aircraft_pos[0])
    dy_min = float(aircraft_pos[1] - area["y_min"])
    dy_max = float(area["y_max"] - aircraft_pos[1])
    if dx_min < margin:
        repel += np.array([1.0 / max(dx_min, 1.0), 0.0], dtype=float)
    if dx_max < margin:
        repel += np.array([-1.0 / max(dx_max, 1.0), 0.0], dtype=float)
    if dy_min < margin:
        repel += np.array([0.0, 1.0 / max(dy_min, 1.0)], dtype=float)
    if dy_max < margin:
        repel += np.array([0.0, -1.0 / max(dy_max, 1.0)], dtype=float)

    # NFZ repulsion
    for z in zones:
        diff = aircraft_pos - z.center_world
        dist = float(np.linalg.norm(diff))
        soft = float(z.radius_world + margin)
        if dist < soft and dist > 1e-8:
            repel += diff / dist * (soft - dist) / max(soft, 1.0)

    action = desired + 1.2 * repel
    nrm = float(np.linalg.norm(action))
    if nrm < 1e-8:
        return np.zeros(2, dtype=float)
    return action / nrm


def _get_env_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "env.yaml"


def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run_single_controller(
    episodes: int,
    max_steps: int,
    controller_name: str,
    action_fn: Callable[[DynamicTargetEnvPhase1A], np.ndarray],
) -> tuple[dict, list[dict]]:
    env_cfg = load_yaml(_get_env_config_path())
    env = DynamicTargetEnvPhase1A(env_cfg)

    out_dir = _get_project_root() / "outputs" / "eval" / "phase1a_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"episodes_{controller_name}.jsonl"

    rows = []
    with result_path.open("w", encoding="utf-8") as f:
        for ep in range(episodes):
            obs = env.reset(seed=1000 + ep)
            a0 = obs.aircraft_pos_world
            t0 = obs.target_pos_world
            initial_dist = float(np.linalg.norm(a0 - t0))
            mode = env.get_target_truth().motion_mode

            done = False
            step_idx = 0
            reason = "running"
            max_crop_jump = 0.0
            prev_crop = obs.truth_crop_center_world
            captured = False
            crop_invalid_steps = 0
            target_oob_steps = 0

            while not done and step_idx < max_steps:
                action = action_fn(env)
                step_result: EnvStepResult = env.step(action)
                obs = step_result.observation
                done = step_result.done
                step_idx += 1

                crop = obs.truth_crop_center_world
                crop_jump = float(np.linalg.norm(crop - prev_crop))
                max_crop_jump = max(max_crop_jump, crop_jump)
                prev_crop = crop

                if not bool(obs.crop_valid_flag):
                    crop_invalid_steps += 1
                if bool(step_result.info.target_out_of_bounds):
                    target_oob_steps += 1

                reason = str(step_result.info.reason)
                if reason == "captured":
                    captured = True

            final_dist = float(
                np.linalg.norm(env.get_aircraft_state().pos_world - env.get_target_truth().pos_world)
            )
            row = {
                "episode": ep,
                "mode": mode,
                "initial_dist": initial_dist,
                "final_dist": final_dist,
                "distance_improved": final_dist < initial_dist,
                "captured": captured,
                "steps": step_idx,
                "reason": reason,
                "max_crop_center_jump": max_crop_jump,
                "crop_invalid_steps": crop_invalid_steps,
                "target_oob_steps": target_oob_steps,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mode_stats: dict[str, dict[str, int]] = {}
    for r in rows:
        mode_stats.setdefault(r["mode"], {"count": 0, "captured": 0})
        mode_stats[r["mode"]]["count"] += 1
        mode_stats[r["mode"]]["captured"] += int(bool(r["captured"]))

    summary = {
        "controller": controller_name,
        "episodes": episodes,
        "improved_ratio": float(sum(int(r["distance_improved"]) for r in rows) / max(1, len(rows))),
        "capture_ratio": float(sum(int(r["captured"]) for r in rows) / max(1, len(rows))),
        "avg_crop_invalid_steps": float(sum(r["crop_invalid_steps"] for r in rows) / max(1, len(rows))),
        "target_out_of_bounds_episodes": int(sum(1 for r in rows if r["target_oob_steps"] > 0)),
        "termination_reasons": {
            k: sum(1 for r in rows if r["reason"] == k)
            for k in sorted(set(r["reason"] for r in rows))
        },
        "mode_stats": mode_stats,
        "result_path": str(result_path),
    }
    with (out_dir / f"summary_{controller_name}.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary, rows


def run_sanity(episodes: int, max_steps: int) -> dict:
    env_cfg = load_yaml(_get_env_config_path())
    area = env_cfg["phase1a"]["area"]
    direct_summary, _ = _run_single_controller(
        episodes=episodes,
        max_steps=max_steps,
        controller_name="direct_chase",
        action_fn=lambda env: _simple_truth_chase_action(
            aircraft_pos=env.get_aircraft_state().pos_world,
            target_pos=env.get_target_truth().pos_world,
        ),
    )
    safe_summary, _ = _run_single_controller(
        episodes=episodes,
        max_steps=max_steps,
        controller_name="safe_chase",
        action_fn=lambda env: _safe_truth_chase_action(
            aircraft_pos=env.get_aircraft_state().pos_world,
            target_pos=env.get_target_truth().pos_world,
            area=area,
            zones=env.get_no_fly_zones(),
        ),
    )
    comparison = {
        "episodes": episodes,
        "direct_chase": direct_summary,
        "safe_chase": safe_summary,
        "delta_out_of_bounds": safe_summary["termination_reasons"].get("out_of_bounds", 0)
        - direct_summary["termination_reasons"].get("out_of_bounds", 0),
    }
    out_dir = _get_project_root() / "outputs" / "eval" / "phase1a_sanity"
    with (out_dir / "summary_compare.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    summary = run_sanity(episodes=args.episodes, max_steps=args.max_steps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
