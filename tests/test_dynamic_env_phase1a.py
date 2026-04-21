import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.dynamic_env_phase1a import DynamicTargetEnvPhase1A


def test_phase1a_env_step_and_truth_crop_center_world():
    cfg = load_yaml("configs/env.yaml")
    env = DynamicTargetEnvPhase1A(cfg)
    obs = env.reset(seed=123)
    assert "truth_crop_center_world" in obs
    assert np.isfinite(obs["truth_crop_center_world"]).all()

    done = False
    for _ in range(20):
        obs, reward, done, info = env.step(np.array([1.0, 0.0], dtype=float))
        assert np.isfinite(reward)
        assert np.isfinite(obs["aircraft_pos_world"]).all()
        assert np.isfinite(obs["target_pos_world"]).all()
        assert np.isfinite(obs["truth_crop_center_world"]).all()
        assert "reason" in info
        if done:
            break


def test_phase1a_env_terminates_by_timeout_with_zero_action():
    cfg = load_yaml("configs/env.yaml")
    cfg["phase1a"]["max_steps"] = 5
    env = DynamicTargetEnvPhase1A(cfg)
    env.reset(seed=99)

    done = False
    reason = "running"
    while not done:
        _, _, done, info = env.step(np.array([0.0, 0.0], dtype=float))
        reason = str(info["reason"])
    assert reason in {"timeout", "captured", "out_of_bounds", "safety_violation"}
