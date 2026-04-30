from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from paper2.common.types import AircraftState, NoFlyZoneState, TargetEstimateState


@dataclass
class Phase3SafeController:
    """Paper2 self-contained online controller inspired by Paper1 planners.

    The controller combines attraction to the target with APF-style no-fly-zone
    repulsion and a tangential bypass waypoint when the target line of sight is
    blocked by an inflated no-fly zone.
    """

    attractive_gain: float = 1.0
    repulsive_gain: float = 7000.0
    clearance_extra_km: float = 35.0
    tangent_gain: float = 0.85

    def act(
        self,
        aircraft: AircraftState,
        estimate: TargetEstimateState,
        zones: list[NoFlyZoneState],
    ) -> np.ndarray:
        pos = _as_3d(aircraft.pos_world)
        target = _as_3d(estimate.pos_world_est)
        target = self._project_target_to_safe_standoff(pos, target, zones)
        waypoint = self._target_or_bypass_waypoint(pos, target, zones)
        force = self.attractive_gain * (waypoint - pos)
        force += self._zone_repulsion(pos, zones)
        if float(np.linalg.norm(force[:2])) < 1e-9:
            force = target - pos
        return _direction_to_action(aircraft, force)

    def _project_target_to_safe_standoff(
        self,
        pos: np.ndarray,
        target: np.ndarray,
        zones: list[NoFlyZoneState],
    ) -> np.ndarray:
        safe_target = target.copy()
        for zone in zones:
            center = _zone_center_3d(zone)
            inflated = _inflated_radius(zone, self.clearance_extra_km)
            offset = safe_target[:2] - center[:2]
            dist = float(np.linalg.norm(offset))
            if dist >= inflated:
                continue
            if dist < 1e-9:
                offset = pos[:2] - center[:2]
                dist = float(np.linalg.norm(offset))
            if dist < 1e-9:
                offset = np.array([1.0, 0.0], dtype=float)
                dist = 1.0
            safe_xy = center[:2] + offset / dist * inflated
            safe_z = max(float(safe_target[2]), center[2] + 0.35 * inflated)
            safe_target = np.array([safe_xy[0], safe_xy[1], safe_z], dtype=float)
        return safe_target

    def _target_or_bypass_waypoint(
        self,
        pos: np.ndarray,
        target: np.ndarray,
        zones: list[NoFlyZoneState],
    ) -> np.ndarray:
        blocker = self._nearest_line_blocker(pos, target, zones)
        if blocker is None:
            return target
        zone, side = blocker
        center = _zone_center_3d(zone)
        rel = target[:2] - pos[:2]
        rel_norm = max(float(np.linalg.norm(rel)), 1e-9)
        along = rel / rel_norm
        lateral = np.array([-along[1], along[0]], dtype=float) * float(side)
        inflated = _inflated_radius(zone, self.clearance_extra_km)
        waypoint_xy = center[:2] + lateral * inflated * 1.35 + along * inflated * 0.25
        waypoint_z = max(float(pos[2]), float(target[2]), center[2] + 0.35 * inflated)
        tangent_waypoint = np.array([waypoint_xy[0], waypoint_xy[1], waypoint_z], dtype=float)
        return self.tangent_gain * tangent_waypoint + (1.0 - self.tangent_gain) * target

    def _nearest_line_blocker(
        self,
        pos: np.ndarray,
        target: np.ndarray,
        zones: list[NoFlyZoneState],
    ) -> tuple[NoFlyZoneState, float] | None:
        segment = target[:2] - pos[:2]
        seg_len2 = float(np.dot(segment, segment))
        if seg_len2 < 1e-9:
            return None
        best: tuple[NoFlyZoneState, float, float] | None = None
        heading = _heading_vector(float(pos[2]), segment)
        for zone in zones:
            center = _zone_center_3d(zone)
            inflated = _inflated_radius(zone, self.clearance_extra_km)
            t = float(np.clip(np.dot(center[:2] - pos[:2], segment) / seg_len2, 0.0, 1.0))
            if t <= 0.02 or t >= 0.98:
                continue
            closest_xy = pos[:2] + t * segment
            lateral_dist = float(np.linalg.norm(center[:2] - closest_xy))
            if lateral_dist > inflated:
                continue
            forward_dist = float(t * math.sqrt(seg_len2))
            side_value = _cross2d(heading, center[:2] - pos[:2])
            side = -1.0 if side_value >= 0.0 else 1.0
            if best is None or forward_dist < best[1]:
                best = (zone, forward_dist, side)
        if best is None:
            return None
        return best[0], best[2]

    def _zone_repulsion(self, pos: np.ndarray, zones: list[NoFlyZoneState]) -> np.ndarray:
        repulsion = np.zeros(3, dtype=float)
        for zone in zones:
            center = _zone_center_3d(zone)
            vector = pos - center
            distance = max(float(np.linalg.norm(vector)), 1e-6)
            influence = _inflated_radius(zone, self.clearance_extra_km)
            if distance < influence:
                strength = self.repulsive_gain * ((1.0 / distance) - (1.0 / influence)) / (distance**2)
                repulsion += strength * (vector / distance)
        return repulsion


def _direction_to_action(aircraft: AircraftState, direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if direction.size < 3:
        direction = np.array([direction[0], direction[1], 0.0], dtype=float)
    gamma = float(aircraft.gamma if aircraft.gamma is not None else 0.0)
    psi = float(aircraft.psi if aircraft.psi is not None else aircraft.heading)
    limits = dict(aircraft.control_limits or {})
    dg_max = float(limits.get("delta_gamma_max", 0.14))
    dp_max = float(limits.get("delta_psi_max", 0.2))
    gamma_max = float(limits.get("gamma_max", 0.6))
    xy_norm = max(float(np.linalg.norm(direction[:2])), 1e-9)
    desired_gamma = float(math.atan2(direction[2], xy_norm))
    desired_psi = float(math.atan2(direction[1], direction[0]))
    delta_gamma = float(np.clip(desired_gamma - gamma, -dg_max, dg_max))
    next_gamma = float(np.clip(gamma + delta_gamma, -gamma_max, gamma_max))
    delta_gamma = float(next_gamma - gamma)
    delta_psi = float(np.clip(_wrap_angle(desired_psi - psi), -dp_max, dp_max))
    return np.array([delta_gamma, delta_psi], dtype=np.float32)


def _as_3d(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 2:
        return np.array([arr[0], arr[1], 0.0], dtype=float)
    return np.array([arr[0], arr[1], arr[2]], dtype=float)


def _zone_center_3d(zone: NoFlyZoneState) -> np.ndarray:
    return _as_3d(zone.center_world)


def _inflated_radius(zone: NoFlyZoneState, extra_km: float) -> float:
    return float(zone.radius_world) + float(zone.safety_margin) + float(extra_km)


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _heading_vector(z: float, xy_direction: np.ndarray) -> np.ndarray:
    del z
    n = max(float(np.linalg.norm(xy_direction)), 1e-9)
    return np.asarray(xy_direction, dtype=float) / n
