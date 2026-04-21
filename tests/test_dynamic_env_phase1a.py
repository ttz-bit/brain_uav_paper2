from pathlib import Path

import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.dynamic_env_phase1a import DynamicTargetEnvPhase1A


def _cfg() -> dict:
    return load_yaml(Path(__file__).resolve().parents[1] / "configs" / "env.yaml")


def test_phase1a_env_reset_reproducible_under_fixed_seed():
    cfg = _cfg()
    env = DynamicTargetEnvPhase1A(cfg)
    obs_a = env.reset(seed=123)
    obs_b = env.reset(seed=123)
    assert np.allclose(obs_a.aircraft_pos_world, obs_b.aircraft_pos_world)
    assert np.allclose(obs_a.target_pos_world, obs_b.target_pos_world)
    assert np.allclose(obs_a.truth_crop_center_world, obs_b.truth_crop_center_world)


def test_phase1a_truth_crop_center_exact_formula():
    cfg = _cfg()
    env = DynamicTargetEnvPhase1A(cfg)
    env.reset(seed=123)
    truth = env.get_target_truth()
    expected = truth.pos_world + truth.vel_world * float(cfg["phase1a"]["truth_crop_horizon_sec"])
    got = env.get_truth_crop_center_world()
    assert np.allclose(got, expected)


def test_phase1a_env_step_returns_typed_payload():
    cfg = _cfg()
    env = DynamicTargetEnvPhase1A(cfg)
    obs = env.reset(seed=123)
    assert np.isfinite(obs.truth_crop_center_world).all()
    done = False
    for _ in range(20):
        result = env.step(np.array([1.0, 0.0], dtype=float))
        assert np.isfinite(result.reward)
        assert np.isfinite(result.observation.aircraft_pos_world).all()
        assert np.isfinite(result.observation.target_pos_world).all()
        assert np.isfinite(result.observation.truth_crop_center_world).all()
        assert result.info.reason in {
            "running",
            "captured",
            "timeout",
            "out_of_bounds",
            "target_out_of_bounds",
            "safety_violation",
        }
        done = result.done
        if done:
            break


def test_phase1a_env_terminates_by_timeout_with_zero_action():
    cfg = _cfg()
    cfg["phase1a"]["capture_radius"] = 0.0
    cfg["phase1a"]["aircraft"]["speed"] = 0.0
    cfg["phase1a"]["target_dynamics"]["speed_range"]["min"] = 0.0
    cfg["phase1a"]["target_dynamics"]["speed_range"]["max"] = 0.0
    cfg["phase1a"]["max_steps"] = 5
    env = DynamicTargetEnvPhase1A(cfg)
    env.reset(seed=99)

    done = False
    reason = "running"
    while not done:
        result = env.step(np.array([0.0, 0.0], dtype=float))
        done = result.done
        reason = str(result.info.reason)
    assert reason == "timeout"
