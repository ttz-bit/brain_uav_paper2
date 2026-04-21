from __future__ import annotations

from typing import Any

import numpy as np

from paper2.common.types import AircraftState, NoFlyZoneState, TargetTruthState


def check_termination(
    aircraft: AircraftState,
    target: TargetTruthState,
    zones: list[NoFlyZoneState],
    step_idx: int,
    cfg: dict[str, Any],
) -> tuple[bool, str]:
    capture_radius = float(cfg["capture_radius"])
    max_steps = int(cfg["max_steps"])
    area = cfg["area"]

    dist = float(np.linalg.norm(aircraft.pos_world - target.pos_world))
    if dist <= capture_radius:
        return True, "captured"

    if step_idx >= max_steps:
        return True, "timeout"

    x = float(aircraft.pos_world[0])
    y = float(aircraft.pos_world[1])
    if x < float(area["x_min"]) or x > float(area["x_max"]) or y < float(area["y_min"]) or y > float(area["y_max"]):
        return True, "out_of_bounds"

    for zone in zones:
        if float(np.linalg.norm(aircraft.pos_world - zone.center_world)) <= float(zone.radius_world):
            return True, "safety_violation"

    return False, "running"
