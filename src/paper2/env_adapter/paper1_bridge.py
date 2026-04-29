from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import sys

import numpy as np

from paper2.common.types import AircraftState, NoFlyZoneState, TargetTruthState
from paper2.env_adapter.env_types import EnvObservation, EnvStepInfo, EnvStepResult
from paper2.env_adapter.world_frame import paper1_xy_to_paper2_xy, paper1_xyz_to_paper2_xyz


def _add_paper1_src_to_path(paper1_root: str | Path | None) -> None:
    if paper1_root is None:
        return
    src = Path(paper1_root).expanduser().resolve() / "src"
    if not src.exists():
        raise FileNotFoundError(f"paper1 src directory not found: {src}")
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


class Paper1EnvBridge:
    """Adapt paper1 StaticNoFlyTrajectoryEnv to the paper2 environment protocol."""

    def __init__(
        self,
        env: Any | None = None,
        *,
        paper1_root: str | Path | None = None,
        world_size_km: float | None = None,
        seed: int | None = None,
    ) -> None:
        _add_paper1_src_to_path(paper1_root)
        if env is None:
            from brain_uav.config import RewardConfig, ScenarioConfig
            from brain_uav.envs import StaticNoFlyTrajectoryEnv

            env = StaticNoFlyTrajectoryEnv(ScenarioConfig(), RewardConfig(), seed=seed)
        self.env = env
        self.world_size_km = (
            float(world_size_km)
            if world_size_km is not None
            else float(self.env.scenario.world_xy) * 2.0
        )
        self._last_obs: np.ndarray | None = None
        self._last_info: dict[str, Any] = {}

    def reset(self, seed: int | None = None) -> EnvObservation:
        obs, info = self.env.reset(seed=seed)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = dict(info or {})
        return self._build_observation()

    def step(self, action: Any) -> EnvStepResult:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size != 2:
            raise ValueError("Paper1 action must be [delta_gamma, delta_psi].")

        obs, reward, terminated, truncated, info = self.env.step(action_arr)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._last_info = dict(info or {})
        done = bool(terminated or truncated)
        reason = self._map_outcome(str(self._last_info.get("outcome", "running")), done)
        return EnvStepResult(
            observation=self._build_observation(),
            reward=float(reward),
            done=done,
            info=EnvStepInfo(
                reason=reason,
                distance_to_target=float(self._last_info.get("goal_distance", math.nan)),
                mode=str(self._last_info.get("curriculum_level", "paper1_static_goal")),
                crop_valid_flag=True,
                target_out_of_bounds=False,
            ),
        )

    def get_aircraft_state(self) -> AircraftState:
        state = np.asarray(self.env.state, dtype=float).reshape(-1)
        if state.size < 5:
            raise RuntimeError("paper1 env.state must be [x, y, z, gamma, psi].")
        pos = paper1_xyz_to_paper2_xyz(state[:3], world_size_km=self.world_size_km)
        gamma = float(state[3])
        psi = float(state[4])
        speed = float(self.env.scenario.speed)
        vel = np.array(
            [
                speed * math.cos(gamma) * math.cos(psi),
                speed * math.cos(gamma) * math.sin(psi),
                speed * math.sin(gamma),
            ],
            dtype=float,
        )
        return AircraftState(
            t=float(getattr(self.env, "steps", 0) * self.env.scenario.dt),
            pos_world=pos,
            vel_world=vel,
            heading=psi,
            speed=speed,
            gamma=gamma,
            psi=psi,
            control_limits={
                "delta_gamma_max": float(self.env.scenario.delta_gamma_max),
                "delta_psi_max": float(self.env.scenario.delta_psi_max),
                "gamma_max": float(self.env.scenario.gamma_max),
            },
            meta={
                "source": "paper1",
                "unit": "km",
                "speed_unit": "km/s",
                "state_order": ["x", "y", "z", "gamma", "psi"],
                "world_size_km": self.world_size_km,
            },
        )

    def get_target_truth(self) -> TargetTruthState:
        goal = np.asarray(self.env.goal, dtype=float).reshape(-1)
        if goal.size < 3:
            raise RuntimeError("paper1 env.goal must be [x, y, z].")
        pos = paper1_xyz_to_paper2_xyz(goal[:3], world_size_km=self.world_size_km)
        return TargetTruthState(
            t=float(getattr(self.env, "steps", 0) * self.env.scenario.dt),
            pos_world=pos,
            vel_world=np.zeros(3, dtype=float),
            heading=0.0,
            motion_mode="static_goal",
        )

    def get_no_fly_zones(self) -> list[NoFlyZoneState]:
        zones: list[NoFlyZoneState] = []
        for zone in self.env.zones:
            center_xy = paper1_xy_to_paper2_xy(zone.center_xy, world_size_km=self.world_size_km)
            zones.append(
                NoFlyZoneState(
                    center_world=np.array([center_xy[0], center_xy[1], 0.0], dtype=float),
                    radius_world=float(zone.radius),
                    geometry="hemisphere",
                    safety_margin=float(self.env.scenario.warning_distance),
                    meta={
                        "source": "paper1",
                        "unit": "km",
                        "paper1_center_xy": np.asarray(zone.center_xy).tolist(),
                        "world_size_km": self.world_size_km,
                    },
                )
            )
        return zones

    def get_truth_crop_center_world(self) -> np.ndarray:
        return self.get_target_truth().pos_world[:2].copy()

    def export_scenario(self) -> dict[str, Any]:
        return dict(self.env.export_scenario())

    def _build_observation(self) -> EnvObservation:
        aircraft = self.get_aircraft_state()
        target = self.get_target_truth()
        return EnvObservation(
            aircraft_pos_world=aircraft.pos_world.copy(),
            target_pos_world=target.pos_world.copy(),
            truth_crop_center_world=target.pos_world[:2].copy(),
            crop_valid_flag=True,
        )

    @staticmethod
    def _map_outcome(outcome: str, done: bool) -> str:
        if not done or outcome == "running":
            return "running"
        if outcome == "goal":
            return "captured"
        if outcome == "collision":
            return "safety_violation"
        if outcome in {"boundary", "ground"}:
            return "out_of_bounds"
        if outcome == "timeout":
            return "timeout"
        return "timeout"
