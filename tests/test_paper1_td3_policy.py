from __future__ import annotations

import numpy as np
import pytest

from paper2.env_adapter.paper1_bridge import Paper1EnvBridge
from paper2.planning.paper1_td3_policy import Paper1TD3Policy


def test_local_env_observation_matches_paper1_td3_shape():
    bridge = Paper1EnvBridge(seed=7)
    bridge.reset(seed=7)

    obs = bridge.env._get_obs()

    assert obs.shape == (24,)
    assert np.isfinite(obs).all()


def test_td3_policy_random_init_smoke_if_torch_available():
    pytest.importorskip("torch")
    bridge = Paper1EnvBridge(seed=7)
    bridge.reset(seed=7)
    policy = Paper1TD3Policy.from_env(
        bridge.env,
        checkpoint_path=None,
        model_type="snn",
        device="cpu",
        allow_random_init=True,
    )

    action = policy.act(bridge.env._get_obs())

    assert action.shape == (2,)
    assert np.isfinite(action).all()
    assert abs(float(action[0])) <= bridge.env.scenario.delta_gamma_max + 1e-6
    assert abs(float(action[1])) <= bridge.env.scenario.delta_psi_max + 1e-6
