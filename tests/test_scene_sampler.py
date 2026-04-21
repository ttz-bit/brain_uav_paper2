import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.scene_sampler import sample_episode_init


def test_scene_sampler_respects_distance_and_nfz_constraints():
    cfg = load_yaml("configs/env.yaml")["phase1a"]
    rng = np.random.default_rng(11)
    ep = sample_episode_init(cfg, rng)

    dist = float(np.linalg.norm(ep.aircraft.pos_world - ep.target_truth.pos_world))
    assert dist >= float(cfg["min_init_dist"])
    assert dist <= float(cfg["max_init_dist"])

    for z in ep.no_fly_zones:
        d_air = float(np.linalg.norm(ep.aircraft.pos_world - z.center_world))
        d_tgt = float(np.linalg.norm(ep.target_truth.pos_world - z.center_world))
        assert d_air > float(z.radius_world)
        assert d_tgt > float(z.radius_world)
