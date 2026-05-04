from __future__ import annotations

import numpy as np

from paper2.common.types import TargetEstimateState
from paper2.tracking.kalman import ConstantVelocityKalmanFilter


def _estimate(pos, *, t=0.0, conf=1.0):
    pos = np.asarray(pos, dtype=float)
    dim = int(pos.size)
    return TargetEstimateState(
        t=float(t),
        pos_world_est=pos,
        vel_world_est=np.zeros(dim, dtype=float),
        cov=np.eye(dim * 2, dtype=float),
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
