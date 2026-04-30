from __future__ import annotations

import numpy as np

from paper2.common.types import AircraftState, NoFlyZoneState, TargetEstimateState
from paper2.control.phase3_safe_controller import Phase3SafeController


def _aircraft() -> AircraftState:
    return AircraftState(
        t=0.0,
        pos_world=np.array([0.0, 0.0, 100.0], dtype=float),
        vel_world=np.array([2.5, 0.0, 0.0], dtype=float),
        heading=0.0,
        speed=2.5,
        gamma=0.0,
        psi=0.0,
        control_limits={"delta_gamma_max": 0.14, "delta_psi_max": 0.2, "gamma_max": 0.6},
    )


def _estimate() -> TargetEstimateState:
    return TargetEstimateState(
        t=0.0,
        pos_world_est=np.array([1000.0, 0.0, 0.0], dtype=float),
        vel_world_est=np.zeros(3, dtype=float),
        cov=np.eye(4, dtype=float),
        obs_conf=1.0,
        obs_age=0.0,
    )


def test_safe_controller_turns_when_zone_blocks_line_of_sight():
    controller = Phase3SafeController()
    zone = NoFlyZoneState(
        center_world=np.array([350.0, 0.0, 0.0], dtype=float),
        radius_world=180.0,
        geometry="hemisphere",
        safety_margin=80.0,
    )

    action = controller.act(_aircraft(), _estimate(), [zone])

    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert abs(float(action[1])) > 0.01


def test_safe_controller_points_to_target_without_zones():
    action = Phase3SafeController().act(_aircraft(), _estimate(), [])

    assert action.shape == (2,)
    assert abs(float(action[1])) < 1e-6


def test_safe_controller_climbs_for_target_inside_safety_buffer():
    controller = Phase3SafeController()
    zone = NoFlyZoneState(
        center_world=np.array([900.0, 0.0, 0.0], dtype=float),
        radius_world=180.0,
        geometry="hemisphere",
        safety_margin=80.0,
    )
    aircraft = _aircraft()
    estimate = _estimate()
    estimate.pos_world_est = np.array([930.0, 0.0, 0.0], dtype=float)

    action = controller.act(aircraft, estimate, [zone])

    assert action.shape == (2,)
    assert float(action[0]) > 0.0
