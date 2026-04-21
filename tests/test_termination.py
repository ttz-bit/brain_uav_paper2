from pathlib import Path

import numpy as np

from paper2.common.config import load_yaml
from paper2.common.types import AircraftState, NoFlyZoneState, TargetTruthState
from paper2.env_adapter.termination import check_termination


def _base_cfg() -> dict:
    return load_yaml(Path(__file__).resolve().parents[1] / "configs" / "env.yaml")["phase1a"]


def _aircraft(x: float, y: float) -> AircraftState:
    return AircraftState(t=0.0, pos_world=np.array([x, y], dtype=float), vel_world=np.zeros(2), heading=0.0)


def _target(x: float, y: float) -> TargetTruthState:
    return TargetTruthState(
        t=0.0,
        pos_world=np.array([x, y], dtype=float),
        vel_world=np.zeros(2),
        heading=0.0,
        motion_mode="cv",
    )


def test_termination_reason_captured():
    cfg = _base_cfg()
    done, reason = check_termination(_aircraft(10, 10), _target(10, 10), [], 0, cfg)
    assert done is True
    assert reason == "captured"


def test_termination_reason_timeout():
    cfg = _base_cfg()
    done, reason = check_termination(_aircraft(10, 10), _target(500, 500), [], int(cfg["max_steps"]), cfg)
    assert done is True
    assert reason == "timeout"


def test_termination_reason_out_of_bounds():
    cfg = _base_cfg()
    done, reason = check_termination(_aircraft(-1, 10), _target(500, 500), [], 1, cfg)
    assert done is True
    assert reason == "out_of_bounds"


def test_termination_reason_target_out_of_bounds():
    cfg = _base_cfg()
    done, reason = check_termination(_aircraft(10, 10), _target(1200, 1200), [], 1, cfg)
    assert done is True
    assert reason == "target_out_of_bounds"


def test_termination_reason_safety_violation():
    cfg = _base_cfg()
    nfz = [NoFlyZoneState(center_world=np.array([10, 10], dtype=float), radius_world=5.0)]
    done, reason = check_termination(_aircraft(10, 10), _target(500, 500), nfz, 1, cfg)
    assert done is True
    assert reason == "safety_violation"
