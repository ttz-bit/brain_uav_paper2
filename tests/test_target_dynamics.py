import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.target_dynamics import (
    init_motion_internal,
    make_initial_target_truth,
    propagate_target_truth,
)


def test_target_dynamics_all_modes_are_numerically_stable():
    cfg = load_yaml("configs/env.yaml")["phase1a"]["target_dynamics"]
    rng = np.random.default_rng(7)
    modes = ["cv", "turn", "piecewise", "evasive"]
    for mode in modes:
        truth = make_initial_target_truth(
            t=0.0,
            pos_world=np.array([100.0, 120.0], dtype=float),
            mode=mode,
            cfg=cfg,
            rng=rng,
        )
        internal = init_motion_internal(mode, cfg, rng)
        for _ in range(30):
            truth = propagate_target_truth(truth, internal, dt=1.0, cfg=cfg, rng=rng)
            assert np.isfinite(truth.pos_world).all()
            assert np.isfinite(truth.vel_world).all()
            assert np.isfinite(truth.heading)
