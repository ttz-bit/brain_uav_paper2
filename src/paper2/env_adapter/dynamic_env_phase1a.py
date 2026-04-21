from __future__ import annotations

from typing import Any
import math

import numpy as np

from paper2.common.types import AircraftState, NoFlyZoneState, TargetTruthState
from paper2.env_adapter.scene_sampler import EpisodeInit, sample_episode_init
from paper2.env_adapter.target_dynamics import (
    TargetMotionInternalState,
    propagate_target_truth,
)
from paper2.env_adapter.termination import check_termination


class DynamicTargetEnvPhase1A:
    def __init__(self, env_cfg: dict[str, Any]):
        self._cfg_root = env_cfg
        self._cfg = env_cfg["phase1a"]
        self._dt = float(self._cfg["dt"])
        self._rng = np.random.default_rng(int(self._cfg["seed"]))

        self._step_idx = 0
        self._aircraft: AircraftState | None = None
        self._target_truth: TargetTruthState | None = None
        self._target_internal: TargetMotionInternalState | None = None
        self._zones: list[NoFlyZoneState] = []

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

        init_state: EpisodeInit = sample_episode_init(self._cfg, self._rng)
        self._step_idx = 0
        self._aircraft = init_state.aircraft
        self._target_truth = init_state.target_truth
        self._target_internal = init_state.target_internal
        self._zones = init_state.no_fly_zones
        return self._build_obs()

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._assert_ready()
        action_vec = np.asarray(action, dtype=float).reshape(-1)
        if action_vec.size != 2:
            raise ValueError("Action must be a 2D vector.")

        speed = float(self._cfg["aircraft"]["speed"])
        norm = float(np.linalg.norm(action_vec))
        if norm < 1e-8:
            vel = np.zeros(2, dtype=float)
        else:
            vel = action_vec / norm * speed

        aircraft = self._aircraft
        assert aircraft is not None
        aircraft_new_pos = aircraft.pos_world + vel * self._dt
        if np.linalg.norm(vel) > 1e-8:
            heading = float(math.atan2(vel[1], vel[0]))
        else:
            heading = float(aircraft.heading)

        self._aircraft = AircraftState(
            t=float(aircraft.t + self._dt),
            pos_world=aircraft_new_pos.astype(float),
            vel_world=vel.astype(float),
            heading=heading,
        )

        truth = self._target_truth
        internal = self._target_internal
        assert truth is not None and internal is not None
        self._target_truth = propagate_target_truth(
            truth=truth,
            internal=internal,
            dt=self._dt,
            cfg=self._cfg["target_dynamics"],
            rng=self._rng,
        )
        self._step_idx += 1

        done, reason = check_termination(
            aircraft=self._aircraft,
            target=self._target_truth,
            zones=self._zones,
            step_idx=self._step_idx,
            cfg=self._cfg,
        )
        dist = float(np.linalg.norm(self._aircraft.pos_world - self._target_truth.pos_world))
        reward = -dist
        info = {
            "reason": reason,
            "distance_to_target": dist,
            "mode": self._target_truth.motion_mode,
        }
        return self._build_obs(), reward, done, info

    def get_aircraft_state(self) -> AircraftState:
        self._assert_ready()
        assert self._aircraft is not None
        return self._aircraft

    def get_target_truth(self) -> TargetTruthState:
        self._assert_ready()
        assert self._target_truth is not None
        return self._target_truth

    def get_no_fly_zones(self) -> list[NoFlyZoneState]:
        self._assert_ready()
        return list(self._zones)

    def get_truth_crop_center_world(self) -> np.ndarray:
        self._assert_ready()
        assert self._target_truth is not None
        horizon = float(self._cfg["truth_crop_horizon_sec"])
        center = self._target_truth.pos_world + self._target_truth.vel_world * horizon
        return center.astype(float)

    def _build_obs(self) -> dict[str, Any]:
        return {
            "aircraft_pos_world": self.get_aircraft_state().pos_world.copy(),
            "target_pos_world": self.get_target_truth().pos_world.copy(),
            "truth_crop_center_world": self.get_truth_crop_center_world().copy(),
        }

    def _assert_ready(self) -> None:
        if self._aircraft is None or self._target_truth is None or self._target_internal is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
