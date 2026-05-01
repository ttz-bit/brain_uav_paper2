from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Paper1TD3Policy:
    """Paper1 SNN/ANN-TD3 actor adapter used as the Paper2 main planner."""

    actor: Any
    device: str
    model_type: str
    checkpoint_path: Path | None = None
    random_init: bool = False

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        checkpoint_path: str | Path | None,
        model_type: str = "snn",
        device: str = "auto",
        hidden_dim: int = 128,
        snn_time_window: int = 4,
        snn_backend: str = "torch",
        allow_random_init: bool = False,
    ) -> "Paper1TD3Policy":
        torch = _import_torch()
        resolved_device = _resolve_device(torch, device)
        obs = env._get_obs()
        state_dim = int(np.asarray(obs).shape[0])
        action_dim = 2
        payload: dict[str, Any] | None = None
        checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        if checkpoint is not None:
            payload = torch.load(checkpoint, map_location=resolved_device)
            cfg = _checkpoint_config(payload)
            hidden_dim = int(cfg.get("hidden_dim", hidden_dim))
            snn_time_window = int(cfg.get("snn_time_window", snn_time_window))
            snn_backend = str(cfg.get("snn_backend", snn_backend))
        elif not allow_random_init:
            raise ValueError("checkpoint_path is required unless allow_random_init=True.")

        actor = _make_actor(
            torch,
            env.scenario,
            model_type=model_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            snn_time_window=snn_time_window,
            snn_backend=snn_backend,
        ).to(resolved_device)
        if payload is not None:
            state_dict = _checkpoint_state_dict(payload)
            actor.load_state_dict(state_dict)
        actor.eval()
        return cls(
            actor=actor,
            device=resolved_device,
            model_type=str(model_type),
            checkpoint_path=checkpoint,
            random_init=payload is None,
        )

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = _import_torch()
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
            action = self.actor(obs_tensor)
        return action.detach().cpu().numpy()[0].astype(np.float32)

    def diagnostics(self, obs: np.ndarray) -> dict[str, float]:
        if not hasattr(self.actor, "forward_with_diagnostics"):
            return {}
        torch = _import_torch()
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs_arr, dtype=torch.float32, device=self.device)
            _, diag = self.actor.forward_with_diagnostics(obs_tensor)
        return {str(k): float(v) if isinstance(v, (int, float)) else v for k, v in diag.items()}


def _make_actor(
    torch: Any,
    scenario: Any,
    *,
    model_type: str,
    state_dim: int,
    action_dim: int,
    hidden_dim: int,
    snn_time_window: int,
    snn_backend: str,
) -> Any:
    action_limit = torch.tensor(
        [float(scenario.delta_gamma_max), float(scenario.delta_psi_max)],
        dtype=torch.float32,
    )
    if model_type == "snn":
        from paper2.paper1_method.models.snn import SNNPolicyActor

        return SNNPolicyActor(
            state_dim,
            action_dim,
            hidden_dim,
            snn_time_window,
            action_limit,
            scenario,
            backend=snn_backend,
        )
    if model_type == "ann":
        from paper2.paper1_method.models.ann import ANNPolicyActor

        return ANNPolicyActor(state_dim, action_dim, hidden_dim, action_limit, scenario)
    raise ValueError(f"Unsupported model_type: {model_type}")


def _checkpoint_state_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        for key in ("state_dict", "actor_state_dict", "actor"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if all(hasattr(v, "shape") for v in payload.values()):
            return payload
    raise ValueError("Unsupported checkpoint format: expected state_dict/actor_state_dict or raw state dict.")


def _checkpoint_config(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cfg = payload.get("config") or payload.get("cfg") or {}
    if not isinstance(cfg, dict):
        return {}
    training = cfg.get("training", cfg)
    if not isinstance(training, dict):
        return {}
    return dict(training)


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    return device


def _import_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Paper1TD3Policy.") from exc
    return torch
