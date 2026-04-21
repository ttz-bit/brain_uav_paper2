from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.dynamic_env_phase1a import DynamicTargetEnvPhase1A


def _simple_truth_chase_action(aircraft_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    vec = target_pos - aircraft_pos
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-8:
        return np.zeros(2, dtype=float)
    return vec / nrm


def run_sanity(episodes: int, max_steps: int) -> dict:
    env_cfg = load_yaml(Path("configs/env.yaml"))
    env = DynamicTargetEnvPhase1A(env_cfg)

    out_dir = Path("outputs/eval/phase1a_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "episodes.jsonl"

    rows = []
    with result_path.open("w", encoding="utf-8") as f:
        for ep in range(episodes):
            obs = env.reset(seed=1000 + ep)
            a0 = obs["aircraft_pos_world"]
            t0 = obs["target_pos_world"]
            initial_dist = float(np.linalg.norm(a0 - t0))
            mode = env.get_target_truth().motion_mode

            done = False
            step_idx = 0
            reason = "running"
            max_crop_jump = 0.0
            prev_crop = obs["truth_crop_center_world"]
            captured = False

            while not done and step_idx < max_steps:
                action = _simple_truth_chase_action(
                    aircraft_pos=env.get_aircraft_state().pos_world,
                    target_pos=env.get_target_truth().pos_world,
                )
                obs, _, done, info = env.step(action)
                step_idx += 1

                crop = obs["truth_crop_center_world"]
                crop_jump = float(np.linalg.norm(crop - prev_crop))
                max_crop_jump = max(max_crop_jump, crop_jump)
                prev_crop = crop

                reason = str(info["reason"])
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
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mode_stats: dict[str, dict[str, int]] = {}
    for r in rows:
        mode_stats.setdefault(r["mode"], {"count": 0, "captured": 0})
        mode_stats[r["mode"]]["count"] += 1
        mode_stats[r["mode"]]["captured"] += int(bool(r["captured"]))

    summary = {
        "episodes": episodes,
        "improved_ratio": float(sum(int(r["distance_improved"]) for r in rows) / max(1, len(rows))),
        "capture_ratio": float(sum(int(r["captured"]) for r in rows) / max(1, len(rows))),
        "termination_reasons": {
            k: sum(1 for r in rows if r["reason"] == k)
            for k in sorted(set(r["reason"] for r in rows))
        },
        "mode_stats": mode_stats,
        "result_path": str(result_path),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    summary = run_sanity(episodes=args.episodes, max_steps=args.max_steps)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
