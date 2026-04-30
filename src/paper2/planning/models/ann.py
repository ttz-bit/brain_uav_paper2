from __future__ import annotations

import torch
from torch import nn

from paper2.env_adapter.paper1_local_env import ScenarioConfig
from paper2.planning.models.scaling import FixedObsScaler


class ANNPolicyActor(nn.Module):
    """Paper1-compatible ANN actor baseline for continuous flight control."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        action_limit: torch.Tensor,
        scenario: ScenarioConfig,
    ) -> None:
        super().__init__()
        self.obs_scaler = FixedObsScaler(scenario, state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.register_buffer("action_limit", action_limit)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        final_linear = self.net[4]
        nn.init.uniform_(final_linear.weight, -1e-3, 1e-3)
        nn.init.uniform_(final_linear.bias, -1e-3, 1e-3)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.obs_scaler(obs)) * self.action_limit


class ANNCritic(nn.Module):
    """Paper1-compatible TD3 critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, scenario: ScenarioConfig) -> None:
        super().__init__()
        self.obs_scaler = FixedObsScaler(scenario, state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, 1),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.obs_scaler(obs), action], dim=-1))
