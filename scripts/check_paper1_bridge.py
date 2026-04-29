from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from paper2.env_adapter.paper1_bridge import Paper1EnvBridge
from paper2.env_adapter.world_frame import paper1_xy_to_paper2_xy, paper2_xy_to_paper1_xy


def _default_paper1_root() -> Path | None:
    env_root = os.environ.get("PAPER1_REPO_ROOT")
    if env_root:
        return Path(env_root)
    local = Path(__file__).resolve().parents[1] / ".external" / "brain_uav"
    if local.exists():
        return local
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper1-root", type=str, default=None)
    parser.add_argument("--world-size-m", type=float, default=4000.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    paper1_root = Path(args.paper1_root) if args.paper1_root else _default_paper1_root()
    bridge = Paper1EnvBridge(
        paper1_root=paper1_root,
        world_size_m=float(args.world_size_m),
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
        paper1_xy_to_paper2_xy(sample_xy, world_size_m=float(args.world_size_m)),
        world_size_m=float(args.world_size_m),
    )
    report = {
        "task": "check_paper1_bridge",
        "paper1_root": None if paper1_root is None else str(paper1_root),
        "seed": int(args.seed),
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
