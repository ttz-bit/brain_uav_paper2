from __future__ import annotations

import numpy as np

from paper2.paper1_method.config import ExperimentConfig
from paper2.paper1_method.envs import StaticNoFlyTrajectoryEnv


def test_paper1_method_env_runs_inside_paper2_namespace():
    cfg = ExperimentConfig()
    env = StaticNoFlyTrajectoryEnv(cfg.scenario, cfg.rewards, seed=7)

    obs, info = env.reset(seed=7)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, terminated, truncated, step_info = env.step(action)

    assert obs.shape == (24,)
    assert next_obs.shape == (24,)
    assert np.isfinite(obs).all()
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["curriculum_level"] == "hard"
    assert "goal_distance" in step_info


def test_paper1_method_keeps_paper1_training_constants():
    cfg = ExperimentConfig()

    assert cfg.scenario.speed == 2.5
    assert cfg.scenario.ground_warning_height == 4.0
    assert cfg.rewards.goal_reward == 5000.0
    assert cfg.rewards.collision_penalty == 12000.0
    assert cfg.training.actor_freeze_steps == 25000
    assert cfg.training.success_sample_bias == 4.0
    assert cfg.training.terminal_geo_regularization_enabled is True
