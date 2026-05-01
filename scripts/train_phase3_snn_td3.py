from __future__ import annotations

import argparse
import json
import math
import random
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from paper2.env_adapter.paper1_bridge import Paper1EnvBridge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["snn", "ann"], default="snn")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--total-timesteps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay-size", type=int, default=200000)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--warmup-strategy", choices=["random", "expert"], default="random")
    p.add_argument("--bc-steps", type=int, default=0)
    p.add_argument("--bc-updates", type=int, default=0)
    p.add_argument("--bc-batch-size", type=int, default=128)
    p.add_argument("--bc-learning-rate", type=float, default=1e-3)
    p.add_argument("--bc-eval-episodes", type=int, default=4)
    p.add_argument("--bc-heldout-steps", type=int, default=2048)
    p.add_argument("--actor-lr", type=float, default=1e-3)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--policy-noise", type=float, default=0.015)
    p.add_argument("--noise-clip", type=float, default=0.03)
    p.add_argument("--exploration-noise", type=float, default=0.02)
    p.add_argument("--policy-delay", type=int, default=2)
    p.add_argument("--actor-freeze-steps", type=int, default=25000)
    p.add_argument("--actor-grad-clip-norm", type=float, default=1.0)
    p.add_argument("--critic-grad-clip-norm", type=float, default=1.0)
    p.add_argument("--actor-rl-scale-alpha", type=float, default=2.5)
    p.add_argument("--bc-regularization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--terminal-geo-regularization", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--terminal-geo-radius", type=float, default=250.0)
    p.add_argument("--terminal-geo-lambda", type=float, default=3000.0)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--snn-time-window", type=int, default=4)
    p.add_argument("--snn-backend", choices=["torch", "cupy"], default="torch")
    p.add_argument("--eval-interval", type=int, default=10000)
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default="outputs/phase3_snn_td3/train_snn_td3")
    return p.parse_args()


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.obs: np.ndarray | None = None
        self.action: np.ndarray | None = None
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs: np.ndarray | None = None
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.size = 0
        self.pos = 0

    def __len__(self) -> int:
        return int(self.size)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        if self.obs is None:
            self.obs = np.zeros((self.capacity, *obs.shape), dtype=np.float32)
            self.action = np.zeros((self.capacity, *action.shape), dtype=np.float32)
            self.next_obs = np.zeros((self.capacity, *next_obs.shape), dtype=np.float32)
        idx = self.pos
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx, 0] = float(reward)
        self.next_obs[idx] = next_obs
        self.done[idx, 0] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, torch: Any, device: str) -> dict[str, Any]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        return {
            "obs": torch.tensor(self.obs[idx], dtype=torch.float32, device=device),
            "action": torch.tensor(self.action[idx], dtype=torch.float32, device=device),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32, device=device),
            "next_obs": torch.tensor(self.next_obs[idx], dtype=torch.float32, device=device),
            "done": torch.tensor(self.done[idx], dtype=torch.float32, device=device),
        }


def main() -> None:
    args = parse_args()
    torch = _import_torch()
    device = _resolve_device(torch, str(args.device))
    _seed_everything(torch, int(args.seed))

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bridge = Paper1EnvBridge(seed=int(args.seed))
    obs = bridge.env.reset(seed=int(args.seed))[0]
    env = bridge.env
    state_dim = int(obs.shape[0])
    action_dim = 2
    action_low = np.array([-float(env.scenario.delta_gamma_max), -float(env.scenario.delta_psi_max)], dtype=np.float32)
    action_high = np.array([float(env.scenario.delta_gamma_max), float(env.scenario.delta_psi_max)], dtype=np.float32)
    action_limit = torch.tensor(action_high, dtype=torch.float32)
    from paper2.planning.models import ANNCritic

    actor = _make_actor(args, env.scenario, state_dim, action_dim, action_limit).to(device)
    actor_target = _make_actor(args, env.scenario, state_dim, action_dim, action_limit).to(device)
    actor_target.load_state_dict(actor.state_dict())
    critic1 = ANNCritic(state_dim, action_dim, int(args.hidden_dim), env.scenario).to(device)
    critic2 = ANNCritic(state_dim, action_dim, int(args.hidden_dim), env.scenario).to(device)
    critic1_target = ANNCritic(state_dim, action_dim, int(args.hidden_dim), env.scenario).to(device)
    critic2_target = ANNCritic(state_dim, action_dim, int(args.hidden_dim), env.scenario).to(device)
    critic1_target.load_state_dict(critic1.state_dict())
    critic2_target.load_state_dict(critic2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=float(args.actor_lr))
    critic_opt = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=float(args.critic_lr))
    replay = ReplayBuffer(int(args.replay_size))
    bc_trace: list[dict[str, Any]] = []
    bc_reference_actor = None
    expert_eval: dict[str, Any] = {}
    bc_eval: dict[str, Any] = {}
    bc_action_fit: dict[str, Any] = {}

    if int(args.bc_steps) > 0:
        expert_eval = _evaluate_expert_policy(
            seed=int(args.seed) + 3571,
            episodes=int(args.bc_eval_episodes),
            action_low=action_low,
            action_high=action_high,
        )
        bc_trace = _behavior_clone_actor(
            torch=torch,
            actor=actor,
            actor_target=actor_target,
            env=env,
            replay=replay,
            args=args,
            device=device,
            action_low=action_low,
            action_high=action_high,
        )
        if bool(args.bc_regularization):
            bc_reference_actor = deepcopy(actor).to(device)
            bc_reference_actor.eval()
            for param in bc_reference_actor.parameters():
                param.requires_grad = False
        bc_eval = _evaluate_policy(
            torch,
            actor,
            device,
            seed=int(args.seed) + 3571,
            episodes=int(args.bc_eval_episodes),
            action_low=action_low,
            action_high=action_high,
        )
        bc_action_fit = _evaluate_bc_action_fit(
            torch=torch,
            actor=actor,
            device=device,
            seed=int(args.seed) + 4243,
            steps=int(args.bc_heldout_steps),
            action_low=action_low,
            action_high=action_high,
        )
        print(
            f"[BC-EVAL] expert_outcomes={expert_eval.get('outcomes', {})} "
            f"actor_outcomes={bc_eval.get('outcomes', {})} "
            f"heldout_action_rmse={bc_action_fit.get('action_rmse', 0.0):.6f}"
        )

    obs, _ = env.reset(seed=int(args.seed))
    ep_return = 0.0
    ep_len = 0
    episode_returns: deque[float] = deque(maxlen=20)
    episode_lengths: deque[int] = deque(maxlen=20)
    outcomes: dict[str, int] = {}
    best_eval_score = -float("inf")
    loss_trace: list[dict[str, Any]] = []
    eval_trace: list[dict[str, Any]] = []
    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_rl_actor_loss = 0.0
    last_scaled_rl_actor_loss = 0.0
    last_actor_rl_scale = 1.0
    last_bc_loss = 0.0
    last_bc_lambda = 0.0
    last_terminal_geo_loss = 0.0
    last_terminal_geo_lambda = 0.0

    for step in range(1, int(args.total_timesteps) + 1):
        if step <= int(args.warmup_steps):
            if str(args.warmup_strategy) == "expert":
                action = _expert_action(env, action_low, action_high)
            else:
                action = np.random.uniform(action_low, action_high).astype(np.float32)
        else:
            action = _select_action(torch, actor, obs, device, float(args.exploration_noise), action_low, action_high)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        replay.add(obs, action, float(reward), next_obs, done)
        obs = next_obs
        ep_return += float(reward)
        ep_len += 1

        if len(replay) >= int(args.batch_size):
            batch = replay.sample(int(args.batch_size), torch, device)
            with torch.no_grad():
                noise = (torch.randn_like(batch["action"]) * float(args.policy_noise)).clamp(
                    -float(args.noise_clip),
                    float(args.noise_clip),
                )
                next_action = (actor_target(batch["next_obs"]) + noise).clamp(
                    torch.tensor(action_low, dtype=torch.float32, device=device),
                    torch.tensor(action_high, dtype=torch.float32, device=device),
                )
                target_q = torch.min(
                    critic1_target(batch["next_obs"], next_action),
                    critic2_target(batch["next_obs"], next_action),
                )
                target_q = batch["reward"] + (1.0 - batch["done"]) * float(args.gamma) * target_q

            current_q1 = critic1(batch["obs"], batch["action"])
            current_q2 = critic2(batch["obs"], batch["action"])
            critic_loss = torch.nn.functional.mse_loss(current_q1, target_q) + torch.nn.functional.mse_loss(
                current_q2,
                target_q,
            )
            critic_opt.zero_grad()
            critic_loss.backward()
            if float(args.critic_grad_clip_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    list(critic1.parameters()) + list(critic2.parameters()),
                    max_norm=float(args.critic_grad_clip_norm),
                )
            critic_opt.step()
            last_critic_loss = float(critic_loss.detach().cpu())

            if step % int(args.policy_delay) == 0:
                _soft_update(critic1, critic1_target, float(args.tau))
                _soft_update(critic2, critic2_target, float(args.tau))
                if step > int(args.actor_freeze_steps):
                    (
                        actor_loss,
                        rl_actor_loss,
                        scaled_rl_actor_loss,
                        bc_loss,
                        bc_lambda,
                        actor_rl_scale,
                        terminal_geo_loss,
                        terminal_geo_lambda,
                    ) = _compute_actor_loss_terms(
                        torch=torch,
                        actor=actor,
                        critic1=critic1,
                        obs=batch["obs"],
                        action_low=torch.tensor(action_low, dtype=torch.float32, device=device),
                        action_high=torch.tensor(action_high, dtype=torch.float32, device=device),
                        args=args,
                        step=step,
                        bc_reference_actor=bc_reference_actor,
                    )
                    actor_opt.zero_grad()
                    actor_loss.backward()
                    if float(args.actor_grad_clip_norm) > 0.0:
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=float(args.actor_grad_clip_norm))
                    actor_opt.step()
                    _soft_update(actor, actor_target, float(args.tau))
                    last_actor_loss = float(actor_loss.detach().cpu())
                    last_rl_actor_loss = float(rl_actor_loss.detach().cpu())
                    last_scaled_rl_actor_loss = float(scaled_rl_actor_loss.detach().cpu())
                    last_bc_loss = float(bc_loss.detach().cpu())
                    last_bc_lambda = float(bc_lambda)
                    last_actor_rl_scale = float(actor_rl_scale)
                    last_terminal_geo_loss = float(terminal_geo_loss.detach().cpu())
                    last_terminal_geo_lambda = float(terminal_geo_lambda)

        if done:
            outcome = str(info.get("outcome", "unknown"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            episode_returns.append(float(ep_return))
            episode_lengths.append(int(ep_len))
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0

        if step % int(args.log_interval) == 0 or step == int(args.total_timesteps):
            row = {
                "step": int(step),
                "replay_size": int(len(replay)),
                "avg_return_20": float(np.mean(episode_returns)) if episode_returns else 0.0,
                "avg_length_20": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
                "actor_loss": float(last_actor_loss),
                "critic_loss": float(last_critic_loss),
                "actor_phase": "frozen" if step <= int(args.actor_freeze_steps) else "active",
                "rl_actor_loss": float(last_rl_actor_loss),
                "scaled_rl_actor_loss": float(last_scaled_rl_actor_loss),
                "actor_rl_scale": float(last_actor_rl_scale),
                "bc_loss": float(last_bc_loss),
                "bc_lambda": float(last_bc_lambda),
                "terminal_geo_loss": float(last_terminal_geo_loss),
                "terminal_geo_lambda": float(last_terminal_geo_lambda),
                "outcomes": dict(outcomes),
            }
            loss_trace.append(row)
            print(
                f"[TD3] step={step}/{args.total_timesteps} replay={len(replay)} "
                f"avg_return_20={row['avg_return_20']:.3f} actor_phase={row['actor_phase']} "
                f"actor_loss={last_actor_loss:.4f} "
                f"critic_loss={last_critic_loss:.4f} outcomes={outcomes}"
            )

        if step % int(args.eval_interval) == 0 or step == int(args.total_timesteps):
            eval_row = _evaluate_policy(
                torch,
                actor,
                device,
                seed=int(args.seed) + step,
                episodes=int(args.eval_episodes),
                action_low=action_low,
                action_high=action_high,
            )
            eval_row["step"] = int(step)
            eval_trace.append(eval_row)
            score = (
                float(eval_row["capture_rate"]) * 1000.0
                - float(eval_row["collision_count"]) * 100.0
                - float(eval_row["ground_count"]) * 100.0
                - float(eval_row["boundary_count"]) * 100.0
            )
            if score > best_eval_score:
                best_eval_score = score
                _save_checkpoint(out_dir / "model_best.pth", actor, args, env, state_dim, action_dim, eval_row)
            print(
                f"[EVAL] step={step} capture_rate={eval_row['capture_rate']:.3f} "
                f"collision_count={eval_row['collision_count']} ground_count={eval_row['ground_count']} "
                f"boundary_count={eval_row['boundary_count']} timeout_count={eval_row['timeout_count']}"
            )

    _save_checkpoint(out_dir / "model_last.pth", actor, args, env, state_dim, action_dim, eval_trace[-1] if eval_trace else {})
    report = {
        "task": "train_phase3_snn_td3",
        "purpose": "main_planner_training",
        "model": str(args.model),
        "device": device,
        "total_timesteps": int(args.total_timesteps),
        "bc_pretrain": {
            "bc_steps": int(args.bc_steps),
            "bc_updates": int(args.bc_updates),
            "bc_batch_size": int(args.bc_batch_size),
            "bc_learning_rate": float(args.bc_learning_rate),
            "bc_eval_episodes": int(args.bc_eval_episodes),
            "bc_heldout_steps": int(args.bc_heldout_steps),
            "final_bc_loss": float(bc_trace[-1]["bc_loss"]) if bc_trace else None,
            "bc_regularization": bool(args.bc_regularization),
            "expert_eval": expert_eval,
            "bc_actor_eval": bc_eval,
            "bc_action_fit": bc_action_fit,
        },
        "warmup_strategy": str(args.warmup_strategy),
        "actor_freeze_steps": int(args.actor_freeze_steps),
        "actor_grad_clip_norm": float(args.actor_grad_clip_norm),
        "critic_grad_clip_norm": float(args.critic_grad_clip_norm),
        "actor_rl_scale_alpha": float(args.actor_rl_scale_alpha),
        "terminal_geo_regularization": bool(args.terminal_geo_regularization),
        "terminal_geo_radius": float(args.terminal_geo_radius),
        "terminal_geo_lambda": float(args.terminal_geo_lambda),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "outcomes": outcomes,
        "final_train": loss_trace[-1] if loss_trace else {},
        "best_eval": max(eval_trace, key=lambda r: float(r["capture_rate"])) if eval_trace else {},
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "loss_trace_path": str(out_dir / "loss_trace.json"),
            "bc_trace_path": str(out_dir / "bc_trace.json"),
            "eval_trace_path": str(out_dir / "eval_trace.json"),
            "best_weights_path": str(out_dir / "model_best.pth"),
            "last_weights_path": str(out_dir / "model_last.pth"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "bc_trace.json").write_text(json.dumps(bc_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "loss_trace.json").write_text(json.dumps(loss_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "eval_trace.json").write_text(json.dumps(eval_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _behavior_clone_actor(
    *,
    torch: Any,
    actor: Any,
    actor_target: Any,
    env: Any,
    replay: ReplayBuffer,
    args: argparse.Namespace,
    device: str,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> list[dict[str, Any]]:
    obs_arr, action_arr, expert_outcomes = _collect_expert_rollout(
        env=env,
        replay=replay,
        seed=int(args.seed) + 7919,
        steps=int(args.bc_steps),
        action_low=action_low,
        action_high=action_high,
    )
    if obs_arr.size == 0:
        return []

    optimizer = torch.optim.Adam(actor.parameters(), lr=float(args.bc_learning_rate))
    updates = int(args.bc_updates) if int(args.bc_updates) > 0 else int(args.bc_steps)
    updates = max(1, updates)
    batch_size = max(1, min(int(args.bc_batch_size), int(obs_arr.shape[0])))
    trace: list[dict[str, Any]] = []
    log_every = max(1, updates // 10)

    for update in range(1, updates + 1):
        idx = np.random.randint(0, int(obs_arr.shape[0]), size=batch_size)
        obs_batch = torch.tensor(obs_arr[idx], dtype=torch.float32, device=device)
        action_batch = torch.tensor(action_arr[idx], dtype=torch.float32, device=device)
        pred = actor(obs_batch)
        loss = torch.nn.functional.mse_loss(pred, action_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if update == 1 or update % log_every == 0 or update == updates:
            row = {
                "update": int(update),
                "bc_loss": float(loss.detach().cpu()),
                "expert_samples": int(obs_arr.shape[0]),
                "expert_outcomes": dict(expert_outcomes),
            }
            trace.append(row)
            print(
                f"[BC] update={update}/{updates} loss={row['bc_loss']:.6f} "
                f"samples={obs_arr.shape[0]} expert_outcomes={expert_outcomes}"
            )

    actor_target.load_state_dict(actor.state_dict())
    return trace


def _collect_expert_rollout(
    *,
    env: Any,
    replay: ReplayBuffer,
    seed: int,
    steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    outcomes: dict[str, int] = {}
    obs, _ = env.reset(seed=int(seed))
    for _ in range(max(0, int(steps))):
        action = _expert_action(env, action_low, action_high)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        replay.add(obs, action, float(reward), next_obs, done)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        action_list.append(np.asarray(action, dtype=np.float32))
        if done:
            outcome = str(info.get("outcome", "unknown"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            obs, _ = env.reset()
        else:
            obs = next_obs
    if not obs_list:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), outcomes
    return np.stack(obs_list).astype(np.float32), np.stack(action_list).astype(np.float32), outcomes


def _expert_action(env: Any, action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    state = np.asarray(env.state, dtype=float)
    pos = state[:3]
    goal = np.asarray(env.goal, dtype=float)
    command = goal - pos

    scenario = env.scenario
    ground_guard = max(float(scenario.ground_warning_height) * 1.4, float(scenario.world_z_min) + 5.0)
    if pos[2] < ground_guard:
        command[2] += (ground_guard - pos[2]) * 6.0
    upper_guard = float(scenario.world_z_max) - 10.0
    if pos[2] > upper_guard:
        command[2] -= (pos[2] - upper_guard) * 6.0

    for zone in env.zones:
        center = np.asarray(zone.center_xy, dtype=float)
        away_xy = pos[:2] - center
        dxy = max(float(np.linalg.norm(away_xy)), 1e-6)
        influence = float(zone.radius) + float(scenario.warning_distance) + 40.0
        to_goal_xy = goal[:2] - pos[:2]
        path_len = max(float(np.linalg.norm(to_goal_xy)), 1e-6)
        path_unit = to_goal_xy / path_len
        to_center = center - pos[:2]
        along = float(np.dot(to_center, path_unit))
        closest = pos[:2] + np.clip(along, 0.0, path_len) * path_unit
        cross_vec = center - closest
        cross_dist = float(np.linalg.norm(cross_vec))
        corridor = float(zone.radius) + float(scenario.warning_distance)
        if 0.0 < along < min(path_len, corridor * 3.0) and cross_dist < corridor:
            normal = np.array([-path_unit[1], path_unit[0]], dtype=float)
            side = 1.0 if float(np.dot(center - pos[:2], normal)) <= 0.0 else -1.0
            command[:2] += normal * side * (corridor - cross_dist + 1.0) * 8.0
            target_clearance_z = float(zone.radius) + float(scenario.no_fly_clearance) * 0.7
            if pos[2] < target_clearance_z:
                command[2] += (target_clearance_z - pos[2]) * 8.0
        if dxy < influence:
            strength = ((influence - dxy) / influence) ** 2
            command[:2] += (away_xy / dxy) * strength * influence * 8.0
        if dxy < float(zone.radius):
            z_cap = math.sqrt(max(float(zone.radius) ** 2 - dxy * dxy, 0.0))
            target_clearance_z = z_cap + float(scenario.no_fly_clearance) * 0.8
            if pos[2] < target_clearance_z:
                command[2] += (target_clearance_z - pos[2]) * 8.0

    xy_norm = max(float(np.linalg.norm(command[:2])), 1e-6)
    desired_gamma = math.atan2(float(command[2]), xy_norm)
    desired_gamma = float(np.clip(desired_gamma, -float(scenario.gamma_max), float(scenario.gamma_max)))
    desired_psi = math.atan2(float(command[1]), float(command[0]))
    delta_gamma = desired_gamma - float(state[3])
    delta_psi = _wrap_angle(desired_psi - float(state[4]))
    action = np.array([delta_gamma, delta_psi], dtype=np.float32)
    return np.clip(action, action_low, action_high).astype(np.float32)


def _make_actor(args: argparse.Namespace, scenario: Any, state_dim: int, action_dim: int, action_limit: Any) -> Any:
    if args.model == "snn":
        from paper2.planning.models import SNNPolicyActor

        return SNNPolicyActor(
            state_dim,
            action_dim,
            int(args.hidden_dim),
            int(args.snn_time_window),
            action_limit,
            scenario,
            backend=str(args.snn_backend),
        )
    from paper2.planning.models import ANNPolicyActor

    return ANNPolicyActor(state_dim, action_dim, int(args.hidden_dim), action_limit, scenario)


def _select_action(
    torch: Any,
    actor: Any,
    obs: np.ndarray,
    device: str,
    exploration_noise: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    with torch.no_grad():
        obs_tensor = torch.tensor(obs[None, :], dtype=torch.float32, device=device)
        action = actor(obs_tensor).detach().cpu().numpy()[0]
    if exploration_noise > 0.0:
        action = action + np.random.normal(0.0, float(exploration_noise), size=action.shape)
    return np.clip(action, action_low, action_high).astype(np.float32)


def _evaluate_policy(
    torch: Any,
    actor: Any,
    device: str,
    *,
    seed: int,
    episodes: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, Any]:
    captures = 0
    collisions = 0
    ground = 0
    boundary = 0
    timeouts = 0
    steps_total = 0
    returns = []
    outcomes: dict[str, int] = {}
    for ep in range(int(episodes)):
        bridge = Paper1EnvBridge(seed=int(seed) + ep)
        obs, _ = bridge.env.reset(seed=int(seed) + ep)
        done = False
        ep_return = 0.0
        info: dict[str, Any] = {}
        while not done:
            action = _select_action(torch, actor, obs, device, 0.0, action_low, action_high)
            obs, reward, terminated, truncated, info = bridge.env.step(action)
            ep_return += float(reward)
            steps_total += 1
            done = bool(terminated or truncated)
        outcome = str(info.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        captures += int(outcome == "goal")
        collisions += int(outcome == "collision")
        ground += int(outcome == "ground")
        boundary += int(outcome == "boundary")
        timeouts += int(outcome == "timeout")
        returns.append(ep_return)
    return {
        "episodes": int(episodes),
        "outcomes": outcomes,
        "capture_count": int(captures),
        "capture_rate": float(captures / max(1, episodes)),
        "collision_count": int(collisions),
        "ground_count": int(ground),
        "boundary_count": int(boundary),
        "timeout_count": int(timeouts),
        "steps_total": int(steps_total),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
    }


def _evaluate_expert_policy(
    *,
    seed: int,
    episodes: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, Any]:
    captures = 0
    collisions = 0
    ground = 0
    boundary = 0
    timeouts = 0
    steps_total = 0
    returns = []
    outcomes: dict[str, int] = {}
    for ep in range(int(episodes)):
        bridge = Paper1EnvBridge(seed=int(seed) + ep)
        obs, _ = bridge.env.reset(seed=int(seed) + ep)
        done = False
        ep_return = 0.0
        info: dict[str, Any] = {}
        while not done:
            action = _expert_action(bridge.env, action_low, action_high)
            obs, reward, terminated, truncated, info = bridge.env.step(action)
            ep_return += float(reward)
            steps_total += 1
            done = bool(terminated or truncated)
        outcome = str(info.get("outcome", "unknown"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        captures += int(outcome == "goal")
        collisions += int(outcome == "collision")
        ground += int(outcome == "ground")
        boundary += int(outcome == "boundary")
        timeouts += int(outcome == "timeout")
        returns.append(ep_return)
    return {
        "episodes": int(episodes),
        "outcomes": outcomes,
        "capture_count": int(captures),
        "capture_rate": float(captures / max(1, episodes)),
        "collision_count": int(collisions),
        "ground_count": int(ground),
        "boundary_count": int(boundary),
        "timeout_count": int(timeouts),
        "steps_total": int(steps_total),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
    }


def _evaluate_bc_action_fit(
    *,
    torch: Any,
    actor: Any,
    device: str,
    seed: int,
    steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, Any]:
    obs_arr, action_arr, outcomes = _collect_expert_rollout_for_fit(
        seed=int(seed),
        steps=int(steps),
        action_low=action_low,
        action_high=action_high,
    )
    if obs_arr.size == 0:
        return {
            "samples": 0,
            "action_mse": None,
            "action_rmse": None,
            "action_mae": None,
            "expert_outcomes": outcomes,
        }
    with torch.no_grad():
        pred = actor(torch.tensor(obs_arr, dtype=torch.float32, device=device)).detach().cpu().numpy()
    err = pred.astype(np.float64) - action_arr.astype(np.float64)
    abs_err = np.abs(err)
    return {
        "samples": int(obs_arr.shape[0]),
        "action_mse": float(np.mean(err**2)),
        "action_rmse": float(np.sqrt(np.mean(err**2))),
        "action_mae": float(np.mean(abs_err)),
        "delta_gamma_mae": float(np.mean(abs_err[:, 0])),
        "delta_psi_mae": float(np.mean(abs_err[:, 1])),
        "delta_gamma_p90_abs": float(np.percentile(abs_err[:, 0], 90)),
        "delta_psi_p90_abs": float(np.percentile(abs_err[:, 1], 90)),
        "expert_outcomes": outcomes,
    }


def _collect_expert_rollout_for_fit(
    *,
    seed: int,
    steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    bridge = Paper1EnvBridge(seed=int(seed))
    env = bridge.env
    obs, _ = env.reset(seed=int(seed))
    obs_list: list[np.ndarray] = []
    action_list: list[np.ndarray] = []
    outcomes: dict[str, int] = {}
    for _ in range(max(0, int(steps))):
        action = _expert_action(env, action_low, action_high)
        next_obs, _, terminated, truncated, info = env.step(action)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        action_list.append(np.asarray(action, dtype=np.float32))
        if bool(terminated or truncated):
            outcome = str(info.get("outcome", "unknown"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            obs, _ = env.reset()
        else:
            obs = next_obs
    if not obs_list:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), outcomes
    return np.stack(obs_list).astype(np.float32), np.stack(action_list).astype(np.float32), outcomes


def _compute_actor_loss_terms(
    *,
    torch: Any,
    actor: Any,
    critic1: Any,
    obs: Any,
    action_low: Any,
    action_high: Any,
    args: argparse.Namespace,
    step: int,
    bc_reference_actor: Any | None,
) -> tuple[Any, Any, Any, Any, float, float, Any, float]:
    actor_actions = actor(obs)
    q_values = critic1(obs, actor_actions)
    rl_actor_loss = -q_values.mean()
    q_scale = q_values.detach().abs().mean().clamp(min=1.0)
    actor_rl_scale = float(float(args.actor_rl_scale_alpha) / float(q_scale.item()))
    scaled_rl_actor_loss = rl_actor_loss * actor_rl_scale

    bc_lambda = 0.0
    bc_loss = torch.zeros((), dtype=torch.float32, device=obs.device)
    if bc_reference_actor is not None:
        bc_lambda = _bc_lambda(step_hint=int(step))
        with torch.no_grad():
            bc_actions = bc_reference_actor(obs)
        bc_loss = torch.nn.functional.mse_loss(actor_actions, bc_actions)

    terminal_geo_lambda = 0.0
    terminal_geo_loss = torch.zeros((), dtype=torch.float32, device=obs.device)
    if bool(args.terminal_geo_regularization):
        terminal_geo_loss = _terminal_geo_loss(
            torch=torch,
            obs=obs,
            actor_actions=actor_actions,
            action_low=action_low,
            action_high=action_high,
            terminal_geo_radius=float(args.terminal_geo_radius),
        )
        if float(terminal_geo_loss.detach().abs().cpu()) > 0.0:
            terminal_geo_lambda = float(args.terminal_geo_lambda)

    actor_loss = scaled_rl_actor_loss + bc_lambda * bc_loss + terminal_geo_lambda * terminal_geo_loss
    return (
        actor_loss,
        rl_actor_loss,
        scaled_rl_actor_loss,
        bc_loss,
        bc_lambda,
        actor_rl_scale,
        terminal_geo_loss,
        terminal_geo_lambda,
    )


def _bc_lambda(*, step_hint: int) -> float:
    if step_hint < 75000:
        return 500.0
    if step_hint < 150000:
        return 150.0
    if step_hint < 250000:
        return 30.0
    return 5.0


def _terminal_geo_loss(
    *,
    torch: Any,
    obs: Any,
    actor_actions: Any,
    action_low: Any,
    action_high: Any,
    terminal_geo_radius: float,
) -> Any:
    rel_goal = obs[:, 5:8]
    goal_distance = torch.linalg.norm(rel_goal, dim=1)
    eligible_mask = goal_distance <= float(terminal_geo_radius)
    if not torch.any(eligible_mask):
        return torch.zeros((), dtype=torch.float32, device=obs.device)

    gamma = obs[:, 3]
    psi = obs[:, 4]
    dx = rel_goal[:, 0]
    dy = rel_goal[:, 1]
    dz = rel_goal[:, 2]
    horizontal = torch.sqrt(torch.clamp(dx.square() + dy.square(), min=1e-9))
    target_gamma = torch.atan2(dz, horizontal)
    target_psi = torch.atan2(dy, dx)
    delta_gamma = torch.clamp(target_gamma - gamma, min=float(action_low[0].item()), max=float(action_high[0].item()))
    delta_psi = torch.clamp(
        _wrap_angle_tensor(torch, target_psi - psi),
        min=float(action_low[1].item()),
        max=float(action_high[1].item()),
    )
    target_action = torch.stack([delta_gamma, delta_psi], dim=-1)
    return torch.nn.functional.mse_loss(actor_actions[eligible_mask], target_action[eligible_mask])


def _wrap_angle_tensor(torch: Any, value: Any) -> Any:
    two_pi = float(2.0 * np.pi)
    return torch.remainder(value + float(np.pi), two_pi) - float(np.pi)


def _save_checkpoint(
    path: Path,
    actor: Any,
    args: argparse.Namespace,
    env: Any,
    state_dim: int,
    action_dim: int,
    eval_row: dict[str, Any],
) -> None:
    torch = _import_torch()
    payload = {
        "state_dict": actor.state_dict(),
        "config": {
            "training": {
                "hidden_dim": int(args.hidden_dim),
                "snn_time_window": int(args.snn_time_window),
                "snn_backend": str(args.snn_backend),
            },
            "scenario": env.scenario.__dict__,
        },
        "metadata": {
            "model": str(args.model),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "unit": "km",
            "eval": eval_row,
        },
    }
    torch.save(payload, path)


def _soft_update(model: Any, target: Any, tau: float) -> None:
    for param, target_param in zip(model.parameters(), target.parameters()):
        target_param.data.copy_(float(tau) * param.data + (1.0 - float(tau)) * target_param.data)


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return device


def _import_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train_phase3_snn_td3.py.") from exc
    return torch


if __name__ == "__main__":
    main()
