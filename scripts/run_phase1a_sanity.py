from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

import numpy as np

from paper2.common.config import load_yaml
from paper2.common.types import NoFlyZoneState
from paper2.env_adapter.dynamic_env_phase1a import DynamicTargetEnvPhase1A
from paper2.env_adapter.env_types import EnvStepResult

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


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
    zones: list[NoFlyZoneState],
    margin: float = 30.0,
) -> np.ndarray:
    desired = _simple_truth_chase_action(aircraft_pos, target_pos)
    repel = np.zeros(2, dtype=float)

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


def _run_check_phase1a() -> tuple[bool, str]:
    root = _get_project_root()
    cmd = [sys.executable, str(root / "scripts" / "check_phase1a.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, output


def _dump_trace(trace_path: Path, rows: list[dict]) -> None:
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def _plot_episode(
    plot_path: Path,
    episode_rows: list[dict],
    zones: list[NoFlyZoneState],
    area: dict,
    title: str,
) -> None:
    if plt is None:
        return
    aircraft_xy = np.array([r["aircraft_pos_world"] for r in episode_rows], dtype=float)
    target_xy = np.array([r["target_pos_world"] for r in episode_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(aircraft_xy[:, 0], aircraft_xy[:, 1], label="aircraft", linewidth=1.8)
    ax.plot(target_xy[:, 0], target_xy[:, 1], label="target", linewidth=1.8)
    ax.scatter(aircraft_xy[0, 0], aircraft_xy[0, 1], marker="o", s=30, label="aircraft_start")
    ax.scatter(target_xy[0, 0], target_xy[0, 1], marker="o", s=30, label="target_start")
    ax.scatter(aircraft_xy[-1, 0], aircraft_xy[-1, 1], marker="x", s=40, label="aircraft_end")
    ax.scatter(target_xy[-1, 0], target_xy[-1, 1], marker="x", s=40, label="target_end")

    for z in zones:
        circle = plt.Circle((z.center_world[0], z.center_world[1]), z.radius_world, fill=False, linestyle="--")
        ax.add_patch(circle)

    ax.set_xlim(float(area["x_min"]), float(area["x_max"]))
    ax.set_ylim(float(area["y_min"]), float(area["y_max"]))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _run_single_controller(
    episodes: int,
    max_steps: int,
    controller_name: str,
    action_fn: Callable[[DynamicTargetEnvPhase1A], np.ndarray],
) -> tuple[dict, list[dict]]:
    env_cfg = load_yaml(_get_env_config_path())
    env = DynamicTargetEnvPhase1A(env_cfg)
    area = env_cfg["phase1a"]["area"]

    out_dir = _get_project_root() / "outputs" / "eval" / "phase1a_sanity"
    traces_dir = out_dir / "traces"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / f"episodes_{controller_name}.jsonl"

    rows = []
    with result_path.open("w", encoding="utf-8") as f:
        for ep in range(episodes):
            obs = env.reset(seed=1000 + ep)
            zones = env.get_no_fly_zones()
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

            trace_rows = [
                {
                    "step": 0,
                    "aircraft_pos_world": obs.aircraft_pos_world.tolist(),
                    "target_pos_world": obs.target_pos_world.tolist(),
                    "truth_crop_center_world": obs.truth_crop_center_world.tolist(),
                    "action": [0.0, 0.0],
                    "reward": 0.0,
                    "done": False,
                    "reason": "running",
                    "crop_valid_flag": bool(obs.crop_valid_flag),
                }
            ]

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

                trace_rows.append(
                    {
                        "step": step_idx,
                        "aircraft_pos_world": obs.aircraft_pos_world.tolist(),
                        "target_pos_world": obs.target_pos_world.tolist(),
                        "truth_crop_center_world": obs.truth_crop_center_world.tolist(),
                        "action": action.tolist(),
                        "reward": float(step_result.reward),
                        "done": bool(step_result.done),
                        "reason": reason,
                        "crop_valid_flag": bool(obs.crop_valid_flag),
                    }
                )

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

            trace_path = traces_dir / f"{controller_name}_ep{ep:03d}.json"
            _dump_trace(trace_path, trace_rows)
            _plot_episode(
                plots_dir / f"{controller_name}_ep{ep:03d}.png",
                trace_rows,
                zones=zones,
                area=area,
                title=f"{controller_name} ep{ep:03d} reason={reason}",
            )

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


def _evaluate_acceptance(
    check_passed: bool,
    compare: dict,
    improved_ratio_min: float,
    avg_crop_invalid_steps_max: float,
) -> tuple[bool, dict]:
    direct = compare["direct_chase"]
    safe = compare["safe_chase"]
    d_term = direct["termination_reasons"]
    s_term = safe["termination_reasons"]

    cond = {
        "check_phase1a_passed": bool(check_passed),
        "safe_improved_ratio": float(safe["improved_ratio"]) >= float(improved_ratio_min),
        "safe_avg_crop_invalid_steps": float(safe["avg_crop_invalid_steps"]) <= float(avg_crop_invalid_steps_max),
        "safe_out_of_bounds_not_worse_than_direct": int(s_term.get("out_of_bounds", 0))
        <= int(d_term.get("out_of_bounds", 0)),
        "safe_safety_violation_eq_zero": int(s_term.get("safety_violation", 0)) == 0,
        "safe_reason_distribution_non_degenerate": len(s_term.keys()) >= 2,
    }
    return all(cond.values()), cond


def run_sanity(
    episodes: int,
    max_steps: int,
    improved_ratio_min: float,
    avg_crop_invalid_steps_max: float,
) -> tuple[dict, bool]:
    env_cfg = load_yaml(_get_env_config_path())
    area = env_cfg["phase1a"]["area"]

    check_ok, check_output = _run_check_phase1a()
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

    compare = {
        "episodes": episodes,
        "direct_chase": direct_summary,
        "safe_chase": safe_summary,
        "delta_out_of_bounds": safe_summary["termination_reasons"].get("out_of_bounds", 0)
        - direct_summary["termination_reasons"].get("out_of_bounds", 0),
    }
    pass_all, conditions = _evaluate_acceptance(
        check_passed=check_ok,
        compare=compare,
        improved_ratio_min=improved_ratio_min,
        avg_crop_invalid_steps_max=avg_crop_invalid_steps_max,
    )

    out_dir = _get_project_root() / "outputs" / "eval" / "phase1a_sanity"
    with (out_dir / "summary_compare.json").open("w", encoding="utf-8") as f:
        json.dump(compare, f, ensure_ascii=False, indent=2)

    acceptance = {
        "phase": "phase1a",
        "status": "PASS" if pass_all else "FAIL",
        "check_phase1a_passed": check_ok,
        "check_phase1a_output": check_output,
        "sanity_passed": pass_all,
        "episodes": episodes,
        "thresholds": {
            "safe_improved_ratio_min": improved_ratio_min,
            "safe_avg_crop_invalid_steps_max": avg_crop_invalid_steps_max,
            "safe_out_of_bounds_not_worse_than_direct": True,
            "safe_safety_violation_eq": 0,
            "safe_reason_distribution_non_degenerate": True,
        },
        "conditions": conditions,
        "direct_chase": direct_summary,
        "safe_chase": safe_summary,
        "artifacts": {
            "traces_dir": str(out_dir / "traces"),
            "plots_dir": str(out_dir / "plots"),
            "compare_json": str(out_dir / "summary_compare.json"),
        },
    }
    with (out_dir / "acceptance.json").open("w", encoding="utf-8") as f:
        json.dump(acceptance, f, ensure_ascii=False, indent=2)
    return acceptance, pass_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--safe-improved-ratio-min", type=float, default=0.85)
    parser.add_argument("--safe-avg-crop-invalid-steps-max", type=float, default=1.0)
    args = parser.parse_args()

    acceptance, passed = run_sanity(
        episodes=args.episodes,
        max_steps=args.max_steps,
        improved_ratio_min=args.safe_improved_ratio_min,
        avg_crop_invalid_steps_max=args.safe_avg_crop_invalid_steps_max,
    )
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
    if passed:
        print("PHASE1A ACCEPTANCE: PASS")
        raise SystemExit(0)
    print("PHASE1A ACCEPTANCE: FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
