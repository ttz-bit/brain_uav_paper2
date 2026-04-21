from pathlib import Path
import numpy as np

from paper2.common.config import load_yaml
from paper2.common.types import (
    AircraftState,
    NoFlyZoneState,
    TargetTruthState,
    VisionObservation,
    TargetEstimateState,
)


def main():
    env_cfg = load_yaml(Path("configs/env.yaml"))
    render_cfg = load_yaml(Path("configs/render.yaml"))
    vision_cfg = load_yaml(Path("configs/vision.yaml"))

    print("Loaded project:", env_cfg["project"]["name"])
    print("World unit:", env_cfg["world"]["unit_name"])
    print("Patch size:", render_cfg["render"]["patch_size"])
    print("Task:", vision_cfg["task"]["name"])

    aircraft = AircraftState(
        t=0.0,
        pos_world=np.array([0.0, 0.0], dtype=float),
        vel_world=np.array([0.0, 0.0], dtype=float),
        heading=0.0,
    )

    nfz = NoFlyZoneState(
        center_world=np.array([1.0, 1.0], dtype=float),
        radius_world=1.0,
    )

    target = TargetTruthState(
        t=0.0,
        pos_world=np.array([2.0, 3.0], dtype=float),
        vel_world=np.array([0.5, 0.0], dtype=float),
        heading=0.1,
        motion_mode="cv",
    )

    vision_obs = VisionObservation(
        t=0.0,
        detected=True,
        center_px=(64.0, 63.0),
        bbox_xywh=(58.0, 55.0, 12.0, 16.0),
        score=0.95,
        crop_path="data/processed/demo/sample.png",
        crop_center_world=None,
        gsd=None,
        meta={"source": "debug"},
    )

    estimate = TargetEstimateState(
        t=0.0,
        pos_world_est=np.array([2.1, 2.9], dtype=float),
        vel_world_est=np.array([0.4, 0.1], dtype=float),
        cov=np.eye(4, dtype=float),
        obs_conf=0.95,
        obs_age=0.0,
    )

    print("AircraftState OK:", aircraft)
    print("NoFlyZoneState OK:", nfz)
    print("TargetTruthState OK:", target)
    print("VisionObservation OK:", vision_obs)
    print("TargetEstimateState OK:", estimate)


if __name__ == "__main__":
    main()