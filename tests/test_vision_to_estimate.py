from __future__ import annotations

import numpy as np

from paper2.common.types import TargetTruthState, VisionObservation
from paper2.tracking.vision_to_estimate import (
    image_point_to_world_xy,
    oracle_target_estimate,
    vision_observation_to_target_estimate,
)


def test_image_point_to_world_xy_matches_render_convention():
    got = image_point_to_world_xy(
        center_px=(138.0, 118.0),
        crop_center_world=(2000.0, 2000.0),
        gsd=10.0,
        image_size=(256, 256),
    )
    assert np.allclose(got, np.array([2100.0, 2100.0]))


def test_vision_observation_to_target_estimate_valid_2d():
    obs = VisionObservation(
        t=1.0,
        detected=True,
        center_px=(128.0, 128.0),
        bbox_xywh=None,
        score=0.8,
        crop_path=None,
        crop_center_world=(100.0, 200.0),
        gsd=5.0,
    )
    est = vision_observation_to_target_estimate(obs, image_size=(256, 256))
    assert np.allclose(est.pos_world_est, np.array([100.0, 200.0]))
    assert est.vel_world_est.shape == (2,)
    assert est.cov.shape == (4, 4)
    assert est.obs_conf == 0.8


def test_oracle_target_estimate_preserves_truth_without_noise():
    truth = TargetTruthState(
        t=2.0,
        pos_world=np.array([1.0, 2.0, 3.0]),
        vel_world=np.array([0.1, 0.2, 0.3]),
        heading=0.0,
        motion_mode="static_goal",
    )
    est = oracle_target_estimate(truth, noise_std_m=0.0)
    assert np.allclose(est.pos_world_est, truth.pos_world)
    assert np.allclose(est.vel_world_est, truth.vel_world)
    assert est.cov.shape == (6, 6)
