from __future__ import annotations

import numpy as np

from paper2.common.types import TargetEstimateState, VisionObservation
from paper2.tracking.kalman import ConstantVelocityKalmanFilter
from paper2.tracking.vision_to_estimate import vision_observation_to_target_estimate


def _estimate(pos, *, t=0.0, conf=1.0, cov_scale=1.0):
    pos = np.asarray(pos, dtype=float)
    dim = int(pos.size)
    return TargetEstimateState(
        t=float(t),
        pos_world_est=pos,
        vel_world_est=np.zeros(dim, dtype=float),
        cov=np.eye(dim * 2, dtype=float) * float(cov_scale),
        obs_conf=float(conf),
        obs_age=0.0,
        meta={"source": "test"},
    )


def test_kalman_initializes_and_predicts():
    kf = ConstantVelocityKalmanFilter(dim=3, process_accel_std=0.1)
    init = _estimate([10.0, 20.0, 30.0], t=0.0)
    est, info = kf.update(init)
    assert info.initialized is True
    assert np.allclose(est.pos_world_est, [10.0, 20.0, 30.0])

    pred = kf.predict(2.0)
    assert np.allclose(pred.pos_world_est, [10.0, 20.0, 30.0])
    assert pred.obs_age >= 2.0


def test_kalman_gate_rejects_large_jump():
    kf = ConstantVelocityKalmanFilter(dim=3, process_accel_std=0.1)
    kf.update(_estimate([0.0, 0.0, 0.0], t=0.0))
    est, info = kf.update(_estimate([100.0, 0.0, 0.0], t=1.0), gate_threshold=10.0)
    assert info.accepted is False
    assert info.innovation_norm > 10.0
    assert np.allclose(est.pos_world_est, [0.0, 0.0, 0.0], atol=1e-6)


def test_kalman_hard_gate_rejects_large_jump_even_with_loose_covariance():
    kf = ConstantVelocityKalmanFilter(dim=3, process_accel_std=0.1)
    kf.update(_estimate([0.0, 0.0, 0.0], t=0.0, cov_scale=1.0e6))
    est, info = kf.update(_estimate([20.0, 0.0, 0.0], t=1.0, cov_scale=1.0e6), gate_threshold=5.0)
    assert info.accepted is False
    assert info.innovation_norm > 5.0
    assert np.allclose(est.pos_world_est[:2], [0.0, 0.0], atol=1e-6)


def test_vision_measurement_keeps_velocity_covariance_broad():
    obs = VisionObservation(
        t=0.0,
        detected=True,
        center_px=(128.0, 128.0),
        bbox_xywh=None,
        score=1.0,
        crop_path=None,
        crop_center_world=(10.0, 20.0),
        gsd=0.005,
        meta={"measurement_sigma_px": 4.0},
    )
    est = vision_observation_to_target_estimate(obs, z_value=12.0, default_cov_m2=25.0)

    assert np.allclose(np.diag(est.cov)[:3], [0.0004, 0.0004, 0.0004])
    assert np.allclose(np.diag(est.cov)[3:], [25.0, 25.0, 25.0])
