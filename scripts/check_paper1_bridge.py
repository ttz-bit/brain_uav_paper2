from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paper2.env_adapter.paper1_bridge import Paper1EnvBridge
from paper2.env_adapter.world_frame import paper1_xy_to_paper2_xy, paper2_xy_to_paper1_xy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper1-root",
        type=str,
        default=None,
        help="Optional original Paper1 repo root for compatibility checks. Omit to use Paper2 local physics.",
    )
    parser.add_argument("--world-size-km", type=float, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    paper1_root = Path(args.paper1_root) if args.paper1_root else None
    bridge = Paper1EnvBridge(
        paper1_root=paper1_root,
        world_size_km=args.world_size_km,
        seed=int(args.seed),
    )
    obs0 = bridge.reset(seed=int(args.seed))
    zones = bridge.get_no_fly_zones()
    reasons: list[str] = []
    done = False
    for _ in range(int(args.steps)):
        result = bridge.step(np.zeros(2, dtype=np.float32))
        reasons.append(str(result.info.reason))
        done = bool(result.done)
        if done:
            break

    sample_xy = np.array([100.0, -50.0], dtype=float)
    round_trip = paper2_xy_to_paper1_xy(
        paper1_xy_to_paper2_xy(sample_xy, world_size_km=float(bridge.world_size_km)),
        world_size_km=float(bridge.world_size_km),
    )
    report = {
        "task": "check_paper1_bridge",
        "paper1_root": None if paper1_root is None else str(paper1_root),
        "env_source": bridge.env_source,
        "seed": int(args.seed),
        "unit": "km",
        "world_size_km": float(bridge.world_size_km),
        "paper1_speed_km_s": float(bridge.env.scenario.speed),
        "paper1_dt_s": float(bridge.env.scenario.dt),
        "paper1_max_steps": int(bridge.env.scenario.max_steps),
        "paper1_max_range_km": float(bridge.env.scenario.speed * bridge.env.scenario.dt * bridge.env.scenario.max_steps),
        "steps_requested": int(args.steps),
        "steps_executed": len(reasons),
        "done": done,
        "aircraft_pos_shape": list(obs0.aircraft_pos_world.shape),
        "target_pos_shape": list(obs0.target_pos_world.shape),
        "truth_crop_center_shape": list(obs0.truth_crop_center_world.shape),
        "zone_count": len(zones),
        "zone_geometries": sorted({z.geometry for z in zones}),
        "reasons": reasons,
        "coordinate_round_trip_ok": bool(np.allclose(sample_xy, round_trip)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["aircraft_pos_shape"] != [3]:
        raise SystemExit("paper1 bridge aircraft state must be 3D")
    if report["target_pos_shape"] != [3]:
        raise SystemExit("paper1 bridge target state must be 3D")
    if not report["coordinate_round_trip_ok"]:
        raise SystemExit("paper1/paper2 world-frame round trip failed")


if __name__ == "__main__":
    main()
