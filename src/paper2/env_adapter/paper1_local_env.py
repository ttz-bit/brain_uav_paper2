from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass
class ScenarioConfig:
    """Self-contained copy of the Paper1 physical口径 used by Paper2 phase 3."""

    dt: float = 1.0
    speed: float = 2.5
    gamma_max: float = 0.6
    delta_gamma_max: float = 0.14
    delta_psi_max: float = 0.2
    target_distance: float = 1750.0
    world_xy_margin_ratio: float = 0.75
    world_xy: float | None = None
    world_z_min: float = 0.1
    world_z_max: float | None = None
    goal_radius: float = 5.0
    max_steps: int = 1000
    no_fly_count_range: tuple[int, int] = (1, 3)
    no_fly_radius_range: tuple[float, float] = (200.0, 250.0)
    warning_distance: float = 80.0
    no_fly_clearance: float = 120.0
    start_z_ratio_range: tuple[float, float] = (0.18, 0.36)
    goal_z_ratio_range: tuple[float, float] = (0.18, 0.36)

    def __post_init__(self) -> None:
        if self.world_xy is None:
            self.world_xy = float(self.target_distance) * float(self.world_xy_margin_ratio)
        if self.world_z_max is None:
            self.world_z_max = float(self.world_xy) / 3.0


@dataclass
class RewardConfig:
    progress_weight: float = 1.0
    step_penalty: float = 0.01
    goal_bonus: float = 100.0
    collision_penalty: float = 100.0
    boundary_penalty: float = 100.0
    warning_penalty: float = 2.0


@dataclass
class Zone:
    center_xy: np.ndarray
    radius: float


class StaticNoFlyTrajectoryEnv:
    """Minimal Paper1-style 3D aircraft environment kept inside Paper2.

    State order follows Paper1: [x, y, z, gamma, psi], units are km and seconds.
    No-fly zones are hemispheres with centers on z=0.
    """

    metadata = {"render_modes": []}
    source_name = "paper2_local_paper1_physics"

    def __init__(
        self,
        scenario: ScenarioConfig | None = None,
        reward: RewardConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.scenario = scenario or ScenarioConfig()
        self.reward = reward or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(5, dtype=np.float32)
        self.goal = np.zeros(3, dtype=np.float32)
        self.zones: list[Zone] = []
        self.steps = 0
        self._prev_goal_distance = 0.0
        self._outcome = "running"

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        del options
        self.steps = 0
        self._outcome = "running"
        self.state, self.goal, self.zones = self._sample_scenario()
        self._prev_goal_distance = self._goal_distance()
        return self._get_obs(), self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.size != 2:
            raise ValueError("Action must be [delta_gamma, delta_psi].")

        prev_dist = self._goal_distance()
        self._apply_action(action)
        self.steps += 1
        terminated, truncated, outcome = self._termination()
        self._outcome = outcome
        reward = self._compute_reward(prev_dist, outcome)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._info()

    def export_scenario(self) -> dict[str, Any]:
        return {
            "source": self.source_name,
            "scenario": asdict(self.scenario),
            "reward": asdict(self.reward),
            "state": self.state.astype(float).tolist(),
            "goal": self.goal.astype(float).tolist(),
            "zones": [
                {"center_xy": z.center_xy.astype(float).tolist(), "radius": float(z.radius)}
                for z in self.zones
            ],
        }

    def _sample_scenario(self) -> tuple[np.ndarray, np.ndarray, list[Zone]]:
        s = self.scenario
        world_xy = float(s.world_xy)
        world_z_max = float(s.world_z_max)
        target_distance = float(s.target_distance)

        angle = float(self.rng.uniform(-math.pi, math.pi))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        midpoint = self.rng.uniform(-0.12 * world_xy, 0.12 * world_xy, size=2)
        lateral_offset = float(self.rng.uniform(-0.18, 0.18) * target_distance)
        start_xy = midpoint - 0.5 * target_distance * direction - 0.5 * lateral_offset * lateral
        goal_xy = midpoint + 0.5 * target_distance * direction + 0.5 * lateral_offset * lateral
        start_xy = np.clip(start_xy, -0.92 * world_xy, 0.92 * world_xy)
        goal_xy = np.clip(goal_xy, -0.92 * world_xy, 0.92 * world_xy)

        start_z = float(self.rng.uniform(*s.start_z_ratio_range) * world_z_max)
        goal_z = float(self.rng.uniform(*s.goal_z_ratio_range) * world_z_max)
        psi = float(math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0]))
        gamma = float(self.rng.uniform(-0.05, 0.05))
        state = np.array([start_xy[0], start_xy[1], start_z, gamma, psi], dtype=np.float32)
        goal = np.array([goal_xy[0], goal_xy[1], goal_z], dtype=np.float32)
        zones = self._sample_zones(start_xy, goal_xy)
        return state, goal, zones

    def _sample_zones(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> list[Zone]:
        s = self.scenario
        count = int(self.rng.integers(int(s.no_fly_count_range[0]), int(s.no_fly_count_range[1]) + 1))
        zones: list[Zone] = []
        path = np.asarray(goal_xy, dtype=float) - np.asarray(start_xy, dtype=float)
        for _ in range(count):
            zone = self._sample_one_zone(start_xy, path, zones)
            if zone is not None:
                zones.append(zone)
        return zones

    def _sample_one_zone(self, start_xy: np.ndarray, path: np.ndarray, existing: list[Zone]) -> Zone | None:
        s = self.scenario
        path_len = max(float(np.linalg.norm(path)), 1e-6)
        unit = path / path_len
        normal = np.array([-unit[1], unit[0]], dtype=float)
        world_xy = float(s.world_xy)
        for _ in range(80):
            radius = float(self.rng.uniform(*s.no_fly_radius_range))
            along = float(self.rng.uniform(0.25, 0.75) * path_len)
            lateral = float(self.rng.uniform(-0.22, 0.22) * path_len)
            center = start_xy + along * unit + lateral * normal
            if np.any(np.abs(center) > 0.92 * world_xy):
                continue
            clearance = radius + float(s.no_fly_clearance)
            if float(np.linalg.norm(center - start_xy)) < clearance:
                continue
            goal_xy = start_xy + path
            if float(np.linalg.norm(center - goal_xy)) < clearance:
                continue
            if any(float(np.linalg.norm(center - z.center_xy)) < 0.6 * (radius + z.radius) for z in existing):
                continue
            return Zone(center_xy=center.astype(np.float32), radius=radius)
        return None

    def _apply_action(self, action: np.ndarray) -> None:
        s = self.scenario
        delta_gamma = float(np.clip(action[0], -float(s.delta_gamma_max), float(s.delta_gamma_max)))
        delta_psi = float(np.clip(action[1], -float(s.delta_psi_max), float(s.delta_psi_max)))
        gamma = float(np.clip(float(self.state[3]) + delta_gamma, -float(s.gamma_max), float(s.gamma_max)))
        psi = _wrap_angle(float(self.state[4]) + delta_psi)
        step = float(s.speed) * float(s.dt)
        dx = step * math.cos(gamma) * math.cos(psi)
        dy = step * math.cos(gamma) * math.sin(psi)
        dz = step * math.sin(gamma)
        self.state[:3] = (self.state[:3].astype(float) + np.array([dx, dy, dz], dtype=float)).astype(np.float32)
        self.state[3] = np.float32(gamma)
        self.state[4] = np.float32(psi)

    def _termination(self) -> tuple[bool, bool, str]:
        s = self.scenario
        if self._goal_distance() <= float(s.goal_radius):
            return True, False, "goal"
        if self._in_no_fly_zone(self.state[:3]):
            return True, False, "collision"
        if abs(float(self.state[0])) > float(s.world_xy) or abs(float(self.state[1])) > float(s.world_xy):
            return True, False, "boundary"
        if float(self.state[2]) < float(s.world_z_min) or float(self.state[2]) > float(s.world_z_max):
            return True, False, "ground"
        if int(self.steps) >= int(s.max_steps):
            return False, True, "timeout"
        return False, False, "running"

    def _compute_reward(self, prev_dist: float, outcome: str) -> float:
        progress = float(prev_dist - self._goal_distance())
        reward = float(self.reward.progress_weight) * progress - float(self.reward.step_penalty)
        if self._near_no_fly_zone(self.state[:3]):
            reward -= float(self.reward.warning_penalty)
        if outcome == "goal":
            reward += float(self.reward.goal_bonus)
        elif outcome == "collision":
            reward -= float(self.reward.collision_penalty)
        elif outcome in {"boundary", "ground"}:
            reward -= float(self.reward.boundary_penalty)
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        rel = self.goal.astype(float) - self.state[:3].astype(float)
        return np.concatenate(
            [
                self.state.astype(float),
                self.goal.astype(float),
                rel,
                np.array([self._goal_distance()], dtype=float),
            ]
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "outcome": self._outcome,
            "goal_distance": self._goal_distance(),
            "curriculum_level": "paper2_local_paper1_physics",
            "speed_km_s": float(self.scenario.speed),
            "dt_s": float(self.scenario.dt),
            "world_xy_km": float(self.scenario.world_xy),
        }

    def _goal_distance(self) -> float:
        return float(np.linalg.norm(self.goal.astype(float) - self.state[:3].astype(float)))

    def _in_no_fly_zone(self, pos: np.ndarray) -> bool:
        p = np.asarray(pos, dtype=float).reshape(-1)
        for zone in self.zones:
            dxy = float(np.linalg.norm(p[:2] - zone.center_xy.astype(float)))
            if dxy <= float(zone.radius):
                z_limit = math.sqrt(max(float(zone.radius) ** 2 - dxy * dxy, 0.0))
                if p[2] <= z_limit:
                    return True
        return False

    def _near_no_fly_zone(self, pos: np.ndarray) -> bool:
        p = np.asarray(pos, dtype=float).reshape(-1)
        for zone in self.zones:
            if float(np.linalg.norm(p[:2] - zone.center_xy.astype(float))) <= float(zone.radius) + float(
                self.scenario.warning_distance
            ):
                return True
        return False


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)
