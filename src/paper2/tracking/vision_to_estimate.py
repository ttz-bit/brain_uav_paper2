from __future__ import annotations

from typing import Sequence

import numpy as np

from paper2.common.types import TargetEstimateState, TargetTruthState, VisionObservation


def image_point_to_world_xy(
    center_px: Sequence[float],
    crop_center_world: Sequence[float],
    gsd: float,
    image_size: Sequence[int],
) -> np.ndarray:
    if len(center_px) < 2:
        raise ValueError("center_px must contain x/y pixel coordinates.")
    if len(crop_center_world) < 2:
        raise ValueError("crop_center_world must contain x/y world coordinates.")
    if len(image_size) < 2:
        raise ValueError("image_size must contain height/width.")

    h = float(image_size[0])
    w = float(image_size[1])
    u = float(center_px[0])
    v = float(center_px[1])
    cx = float(crop_center_world[0])
    cy = float(crop_center_world[1])
    g = float(gsd)
    return np.array([cx + (u - 0.5 * w) * g, cy + (0.5 * h - v) * g], dtype=float)


def vision_observation_to_target_estimate(
    obs: VisionObservation,
    *,
    image_size: Sequence[int] = (256, 256),
    default_cov_m2: float = 1.0e4,
    pixel_sigma: float | None = None,
    z_value: float | None = None,
) -> TargetEstimateState:
    if not obs.detected or obs.center_px is None or obs.crop_center_world is None or obs.gsd is None:
        dim = 3 if z_value is not None else 2
        pos = np.full(dim, np.nan, dtype=float)
        return TargetEstimateState(
            t=float(obs.t),
            pos_world_est=pos,
            vel_world_est=np.zeros(dim, dtype=float),
            cov=np.eye(dim * 2, dtype=float) * float(default_cov_m2),
            obs_conf=0.0,
            obs_age=0.0,
            meta={"source": "vision_observation", "valid": False, **dict(obs.meta or {})},
        )

    xy = image_point_to_world_xy(
        center_px=obs.center_px,
        crop_center_world=obs.crop_center_world,
        gsd=float(obs.gsd),
        image_size=image_size,
    )
    if z_value is None:
        pos = xy
    else:
        pos = np.array([xy[0], xy[1], float(z_value)], dtype=float)
    dim = int(pos.size)
    sigma_px = _measurement_sigma_px(obs, pixel_sigma=pixel_sigma)
    sigma_m = max(float(obs.gsd) * float(sigma_px), 1e-6)
    pos_var = sigma_m * sigma_m
    # A single visual detection observes position only. Keep velocity uncertainty broad so
    # KF initialization/reinitialization does not become overconfident in zero velocity.
    vel_var = max(float(default_cov_m2), pos_var)
    cov = np.diag([pos_var] * dim + [vel_var] * dim).astype(float)
    meta = {"source": "vision_observation", "valid": True, "pixel_sigma": float(sigma_px)}
    if obs.meta:
        meta.update(dict(obs.meta))
    return TargetEstimateState(
        t=float(obs.t),
        pos_world_est=pos,
        vel_world_est=np.zeros(dim, dtype=float),
        cov=cov,
        obs_conf=float(obs.score),
        obs_age=0.0,
        meta=meta,
    )


def _measurement_sigma_px(obs: VisionObservation, *, pixel_sigma: float | None) -> float:
    meta = dict(obs.meta or {})
    override = meta.get("measurement_sigma_px")
    if override is not None:
        return max(float(override), 1.0e-6)
    if pixel_sigma is not None:
        return max(float(pixel_sigma), 1.0e-6)
    stage = str(meta.get("perception_stage", "")).strip().lower()
    if stage == "far":
        return 48.0
    if stage == "mid":
        return 32.0
    if stage == "terminal":
        return 24.0
    return 32.0


def oracle_target_estimate(
    truth: TargetTruthState,
    *,
    noise_std_m: float = 0.0,
    rng: np.random.Generator | None = None,
    obs_conf: float = 1.0,
) -> TargetEstimateState:
    pos = np.asarray(truth.pos_world, dtype=float).copy()
    vel = np.asarray(truth.vel_world, dtype=float).copy()
    if float(noise_std_m) > 0.0:
        generator = rng if rng is not None else np.random.default_rng()
        pos = pos + generator.normal(0.0, float(noise_std_m), size=pos.shape)
    dim = int(pos.size)
    variance = max(float(noise_std_m) ** 2, 1.0e-6)
    return TargetEstimateState(
        t=float(truth.t),
        pos_world_est=pos,
        vel_world_est=vel,
        cov=np.eye(dim * 2, dtype=float) * variance,
        obs_conf=float(obs_conf),
        obs_age=0.0,
        meta={"source": "oracle", "noise_std_m": float(noise_std_m)},
    )
