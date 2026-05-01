from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from paper2.env_adapter.paper1_bridge import Paper1EnvBridge
from paper2.env_adapter.world_frame import paper1_xy_to_paper2_xy, paper2_xy_to_paper1_xy


def _paper1_root() -> Path | None:
    env_root = os.environ.get("PAPER1_REPO_ROOT")
    if env_root:
        return Path(env_root)
    local = Path(__file__).resolve().parents[1] / ".external" / "brain_uav"
    if local.exists():
        return local
    return None


def test_world_frame_round_trip():
    xy = np.array([123.0, -456.0], dtype=float)
    got = paper2_xy_to_paper1_xy(paper1_xy_to_paper2_xy(xy, world_size_km=2625.0), world_size_km=2625.0)
    assert np.allclose(got, xy)


def test_paper1_bridge_reset_step_contract():
    bridge = Paper1EnvBridge(seed=7)
    obs = bridge.reset(seed=7)
    assert bridge.env_source == "paper2_local_paper1_method"
    assert np.isclose(bridge.world_size_km, 2625.0)
    assert obs.aircraft_pos_world.shape == (3,)
    assert obs.target_pos_world.shape == (3,)
    assert obs.truth_crop_center_world.shape == (2,)

    aircraft = bridge.get_aircraft_state()
    assert aircraft.pos_world.shape == (3,)
    assert aircraft.vel_world.shape == (3,)
    assert aircraft.speed is not None
    assert np.isclose(aircraft.speed, 2.5)
    assert aircraft.gamma is not None
    assert aircraft.psi is not None
    assert aircraft.meta is not None
    assert aircraft.meta["unit"] == "km"

    zones = bridge.get_no_fly_zones()
    assert zones
    assert all(z.geometry == "hemisphere" for z in zones)
    assert all(z.center_world.shape == (3,) for z in zones)

    result = bridge.step(np.zeros(2, dtype=np.float32))
    assert np.isfinite(result.reward)
    assert result.info.reason in {
        "running",
        "captured",
        "timeout",
        "out_of_bounds",
        "target_out_of_bounds",
        "safety_violation",
    }


def test_paper1_bridge_external_compatibility_if_available():
    root = _paper1_root()
    if root is None:
        pytest.skip("paper1 repo not available; set PAPER1_REPO_ROOT")

    bridge = Paper1EnvBridge(paper1_root=root, seed=7)
    obs = bridge.reset(seed=7)
    assert bridge.env_source == "external_paper1"
    assert np.isclose(bridge.world_size_km, 2625.0)
    assert obs.aircraft_pos_world.shape == (3,)
    assert np.isclose(bridge.get_aircraft_state().speed, 2.5)
