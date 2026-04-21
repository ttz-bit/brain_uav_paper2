from pathlib import Path
import numpy as np

from paper2.common.config import load_yaml
from paper2.env_adapter.dynamic_env_phase1a import DynamicTargetEnvPhase1A
from paper2.env_adapter.env_types import EnvObservation, EnvStepResult


ALLOWED_REASONS = {
    "running",
    "captured",
    "timeout",
    "out_of_bounds",
    "target_out_of_bounds",
    "safety_violation",
}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    env_cfg = load_yaml(root / "configs" / "env.yaml")
    exp_cfg = load_yaml(root / "configs" / "experiment.yaml")

    print("Experiment:", exp_cfg["experiment"]["name"])
    print("Current stage:", exp_cfg["experiment"]["current_stage"])
    print("Env project stage:", env_cfg["project"]["stage"])

    env = DynamicTargetEnvPhase1A(env_cfg)
    obs = env.reset(seed=2026)
    assert isinstance(obs, EnvObservation)
    assert np.isfinite(obs.aircraft_pos_world).all()
    assert np.isfinite(obs.target_pos_world).all()
    assert np.isfinite(obs.truth_crop_center_world).all()
    print("Reset observation OK")

    for idx in range(5):
        action = np.array([1.0, 0.2], dtype=float)
        result = env.step(action)
        assert isinstance(result, EnvStepResult)
        assert result.info.reason in ALLOWED_REASONS
        assert np.isfinite(result.observation.truth_crop_center_world).all()
        print(
            f"Step {idx + 1}: done={result.done}, reason={result.info.reason}, "
            f"crop_valid={result.observation.crop_valid_flag}"
        )
        if result.done:
            break

    center = env.get_truth_crop_center_world()
    assert np.isfinite(center).all()
    print("get_truth_crop_center_world() OK")
    print("Phase1A check passed.")


if __name__ == "__main__":
    main()
