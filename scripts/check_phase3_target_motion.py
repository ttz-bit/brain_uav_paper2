from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.phase3_target_motion import (
    MOTION_MODES,
    generate_phase3_target_trajectory,
    summarize_phase3_target_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/env.yaml")
    parser.add_argument("--sequences", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    env_cfg = load_yaml(Path(args.config))
    cfg = env_cfg["phase3_target_motion"]
    sequences = int(args.sequences if args.sequences is not None else cfg["default_sequences"])
    frames = int(args.frames if args.frames is not None else cfg["frames_per_sequence"])
    base_seed = int(args.seed if args.seed is not None else cfg["seed"])

    trajectories = []
    for idx in range(sequences):
        mode = MOTION_MODES[idx] if idx < len(MOTION_MODES) else None
        trajectories.append(
            generate_phase3_target_trajectory(
                cfg,
                seed=base_seed + idx,
                frames=frames,
                mode=mode,
            )
        )

    report = summarize_phase3_target_trajectories(trajectories, cfg)
    report.update(
        {
            "task": "check_phase3_target_motion",
            "config": str(args.config),
            "seed": base_seed,
            "frames_per_sequence": frames,
            "acceptance": {
                "all_modes_present": all(report["mode_counts"].get(mode, 0) > 0 for mode in MOTION_MODES),
                "speed_in_range": (
                    float(report["speed_min"]) >= float(cfg["speed_range"]["min"]) - 1e-9
                    and float(report["speed_max"]) <= float(cfg["speed_range"]["max"]) + 1e-9
                ),
                "continuity_ok": bool(report["continuity_ok"]),
                "bounds_ok": bool(report["bounds_ok"]),
            },
        }
    )
    report["accepted"] = bool(all(report["acceptance"].values()))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase3_target_motion_check.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)

    # Keep numpy imported in this script path so missing scientific deps fail early.
    assert np.isfinite(float(report["speed_mean"]))


if __name__ == "__main__":
    main()
