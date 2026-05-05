from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paper2.common.types import TargetEstimateState


@dataclass
class KalmanUpdateInfo:
    accepted: bool
    initialized: bool
    innovation_norm: float
    gate_threshold: float | None
    innovation_nis: float | None = None
    reinitialized: bool = False


class ConstantVelocityKalmanFilter:
    """Small constant-velocity Kalman filter for target position estimates."""

    def __init__(
        self,
        *,
        dim: int = 3,
        process_accel_std: float = 1.0,
        initial_position_var: float = 1.0e4,
        initial_velocity_var: float = 1.0e4,
        max_reject_streak: int = 3,
        nis_gate_threshold: float = 11.83,
        max_obs_age_before_reinit: float = 20.0,
    ) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError("dim must be positive.")
        self.process_accel_std = float(process_accel_std)
        self.initial_position_var = float(initial_position_var)
        self.initial_velocity_var = float(initial_velocity_var)
        self.max_reject_streak = int(max_reject_streak)
        self.nis_gate_threshold = float(nis_gate_threshold)
        self.max_obs_age_before_reinit = float(max_obs_age_before_reinit)
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.t: float | None = None
        self.obs_conf = 0.0
        self.obs_age = 0.0
        self.meta: dict = {}
        self.reject_streak = 0

    @property
    def initialized(self) -> bool:
        return self.x is not None and self.P is not None and self.t is not None

    def reset(self) -> None:
        self.x = None
        self.P = None
        self.t = None
        self.obs_conf = 0.0
        self.obs_age = 0.0
        self.meta = {}
        self.reject_streak = 0

    def reset_from_estimate(self, estimate: TargetEstimateState) -> None:
        pos = _as_dim(estimate.pos_world_est, self.dim, fill=0.0)
        vel = _as_dim(estimate.vel_world_est, self.dim, fill=0.0)
        self.x = np.concatenate([pos, vel]).astype(float)
        cov = np.asarray(estimate.cov, dtype=float)
        if cov.shape == (self.dim * 2, self.dim * 2) and np.isfinite(cov).all():
            self.P = cov.copy()
        else:
            self.P = np.diag(
                [self.initial_position_var] * self.dim + [self.initial_velocity_var] * self.dim
            ).astype(float)
        self.t = float(estimate.t)
        self.obs_conf = float(estimate.obs_conf)
        self.obs_age = float(estimate.obs_age)
        self.meta = dict(estimate.meta or {})
        self.reject_streak = 0

    def predict(self, t: float) -> TargetEstimateState:
        if not self.initialized:
            raise RuntimeError("Kalman filter must be initialized before predict().")
        assert self.x is not None and self.P is not None and self.t is not None
        dt = max(float(t) - float(self.t), 0.0)
        if dt > 0.0:
            f = np.eye(self.dim * 2, dtype=float)
            f[: self.dim, self.dim :] = np.eye(self.dim, dtype=float) * dt
            q = self._process_noise(dt)
            self.x = f @ self.x
            self.P = f @ self.P @ f.T + q
            self.t = float(t)
            self.obs_age += dt
        return self.to_estimate(meta_extra={"source": "kalman_predict"})

    def update(
        self,
        measurement: TargetEstimateState,
        *,
        gate_threshold: float | None = None,
    ) -> tuple[TargetEstimateState, KalmanUpdateInfo]:
        if not np.isfinite(np.asarray(measurement.pos_world_est, dtype=float)).all():
            if not self.initialized:
                raise ValueError("Cannot initialize Kalman filter from non-finite measurement.")
            pred = self.predict(float(measurement.t))
            return pred, KalmanUpdateInfo(False, False, float("inf"), gate_threshold, None, False)

        if not self.initialized:
            self.reset_from_estimate(measurement)
            est = self.to_estimate(
                meta_extra={
                    "source": "kalman_update",
                    "kalman_initialized": True,
                    "kalman_accepted": True,
                    "kalman_innovation_norm": 0.0,
                    "kalman_innovation_nis": 0.0,
                    "kalman_gate_threshold": gate_threshold,
                }
            )
            return est, KalmanUpdateInfo(True, True, 0.0, gate_threshold, 0.0, False)

        self.predict(float(measurement.t))
        assert self.x is not None and self.P is not None
        z_full = _as_dim(measurement.pos_world_est, self.dim)
        z_dim = min(int(np.asarray(measurement.pos_world_est).size), self.dim)
        z = z_full[:z_dim]
        h = np.zeros((z_dim, self.dim * 2), dtype=float)
        h[:, :z_dim] = np.eye(z_dim, dtype=float)
        r = _measurement_position_cov(measurement.cov, z_dim)
        innovation = z - h @ self.x
        innovation_norm = float(np.linalg.norm(innovation[: min(2, z_dim)]))
        innovation_xy = innovation[: min(2, z_dim)]
        h_xy = h[: min(2, z_dim), :]
        r_xy = r[: min(2, z_dim), : min(2, z_dim)]
        s_xy = h_xy @ self.P @ h_xy.T + r_xy
        nis = float(innovation_xy.T @ np.linalg.pinv(s_xy) @ innovation_xy)
        hard_gate = float(gate_threshold) if gate_threshold is not None else None
        accepted = nis <= self.nis_gate_threshold
        if hard_gate is not None:
            accepted = accepted or innovation_norm <= hard_gate

        if not accepted:
            self.reject_streak += 1
            reject_streak = int(self.reject_streak)
            self.P = self.P * 1.5
            self.obs_conf = min(self.obs_conf, float(measurement.obs_conf)) * 0.5
            self.meta = dict(self.meta)
            if reject_streak >= self.max_reject_streak or self.obs_age >= self.max_obs_age_before_reinit:
                self.reset_from_estimate(measurement)
                est = self.to_estimate(
                    meta_extra={
                        "source": "kalman_update",
                        "kalman_accepted": True,
                        "kalman_reinitialized": True,
                        "kalman_reject_streak": reject_streak,
                        "kalman_innovation_norm": innovation_norm,
                        "kalman_innovation_nis": nis,
                        "kalman_gate_threshold": hard_gate,
                        "raw_measurement_pos_world": z_full.tolist(),
                    }
                )
                return est, KalmanUpdateInfo(True, True, innovation_norm, gate_threshold, nis, True)
            est = self.to_estimate(
                meta_extra={
                    "source": "kalman_update",
                    "kalman_accepted": False,
                    "kalman_reject_streak": int(self.reject_streak),
                    "kalman_innovation_norm": innovation_norm,
                    "kalman_innovation_nis": nis,
                    "kalman_gate_threshold": hard_gate,
                    "raw_measurement_pos_world": z_full.tolist(),
                }
            )
            return est, KalmanUpdateInfo(False, False, innovation_norm, gate_threshold, nis, False)

        s = h @ self.P @ h.T + r
        k = self.P @ h.T @ np.linalg.pinv(s)
        self.x = self.x + k @ innovation
        i = np.eye(self.dim * 2, dtype=float)
        # Joseph form is a little more expensive, but keeps covariance symmetric/positive in long runs.
        self.P = (i - k @ h) @ self.P @ (i - k @ h).T + k @ r @ k.T
        self.P = 0.5 * (self.P + self.P.T)
        self.obs_conf = float(measurement.obs_conf)
        self.obs_age = 0.0
        self.meta = dict(measurement.meta or {})
        self.reject_streak = 0
        est = self.to_estimate(
            meta_extra={
                "source": "kalman_update",
                "kalman_accepted": True,
                "kalman_innovation_norm": innovation_norm,
                "kalman_innovation_nis": nis,
                "kalman_gate_threshold": gate_threshold,
                "raw_measurement_pos_world": z_full.tolist(),
            }
        )
        return est, KalmanUpdateInfo(True, False, innovation_norm, gate_threshold, nis, False)

    def to_estimate(self, *, meta_extra: dict | None = None) -> TargetEstimateState:
        if not self.initialized:
            raise RuntimeError("Kalman filter is not initialized.")
        assert self.x is not None and self.P is not None and self.t is not None
        meta = dict(self.meta)
        if meta_extra:
            meta.update(meta_extra)
        return TargetEstimateState(
            t=float(self.t),
            pos_world_est=self.x[: self.dim].copy(),
            vel_world_est=self.x[self.dim :].copy(),
            cov=self.P.copy(),
            obs_conf=float(self.obs_conf),
            obs_age=float(self.obs_age),
            meta=meta,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        q = np.zeros((self.dim * 2, self.dim * 2), dtype=float)
        a2 = max(self.process_accel_std, 1.0e-12) ** 2
        q[: self.dim, : self.dim] = np.eye(self.dim) * (0.25 * dt**4 * a2)
        q[: self.dim, self.dim :] = np.eye(self.dim) * (0.5 * dt**3 * a2)
        q[self.dim :, : self.dim] = np.eye(self.dim) * (0.5 * dt**3 * a2)
        q[self.dim :, self.dim :] = np.eye(self.dim) * (dt**2 * a2)
        return q


def _as_dim(values: np.ndarray, dim: int, *, fill: float = np.nan) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(int(dim), float(fill), dtype=float)
    n = min(arr.size, int(dim))
    if n > 0:
        out[:n] = arr[:n]
    return out


def _measurement_position_cov(cov: np.ndarray, dim: int) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    if c.ndim == 2 and c.shape[0] >= dim and c.shape[1] >= dim and np.isfinite(c[:dim, :dim]).all():
        r = c[:dim, :dim].copy()
    else:
        r = np.eye(dim, dtype=float)
    min_var = 1.0e-9
    r = 0.5 * (r + r.T)
    diag = np.diag(r).copy()
    diag[diag < min_var] = min_var
    np.fill_diagonal(r, diag)
    return r
