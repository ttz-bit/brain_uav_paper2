from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the minimal Paper2 closed-loop comparison suite: oracle, SNN no-KF, and KF ablations."
    )
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--config", type=str, default="configs/env.yaml")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--vision-weights", type=str, required=True)
    p.add_argument("--td3-checkpoint", type=str, required=True)
    p.add_argument("--vision-source", choices=["replay_dataset", "live_render", "phase3_map_live"], default="replay_dataset")
    p.add_argument("--render-config", type=str, default="configs/render_stage2_c_v1.yaml")
    p.add_argument("--render-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--vision-model", choices=["auto", "cnn_coord", "cnn_heatmap", "snn_heatmap"], default="auto")
    p.add_argument("--decode-method", choices=["auto", "argmax", "softargmax"], default="auto")
    p.add_argument("--model", choices=["snn", "ann"], default="snn")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--paper1-curriculum-level", choices=["easy", "easy_two_zone", "medium", "hard"], default="hard")
    p.add_argument("--paper1-curriculum-mix", type=str, default=None)
    p.add_argument("--phase3-target-init", choices=["paper1_goal", "random_water"], default="paper1_goal")
    p.add_argument("--target-z-policy", choices=["keep_current_goal_z"], default="keep_current_goal_z")
    p.add_argument("--capture-radius-km", type=float, default=5.0)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260430)
    p.add_argument("--vision-input-size", type=int, default=0)
    p.add_argument("--max-vision-samples", type=int, default=None)
    p.add_argument("--terminal-controller-range-km", type=float, default=80.0)
    p.add_argument("--terminal-controller-blend", type=float, default=0.5)
    p.add_argument("--plot-episode", type=int, default=0)
    p.add_argument("--out-root", type=str, default="outputs/phase3_closed_loop/minimal_suite")
    p.add_argument("--skip-oracle", action="store_true")
    p.add_argument("--skip-no-kf", action="store_true")
    p.add_argument("--skip-kf-raw", action="store_true")
    p.add_argument("--skip-kf-pure", action="store_true")
    p.add_argument("--skip-plot", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _unique_out_root(base: Path) -> Path:
    if not base.exists():
        return base
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base / stamp


def _build_vision_cmd(
    *,
    project_root: Path,
    args: argparse.Namespace,
    out_dir: Path,
    estimate_filter: str,
    terminal_controller: str,
    capture_mode: str = "true_target",
    kf_terminal_mode: str = "raw",
) -> list[str]:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_phase3_vision_td3.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--eval-split",
        str(args.eval_split),
        "--vision-source",
        str(args.vision_source),
        "--render-config",
        str(Path(args.render_config).resolve()),
        "--render-split",
        str(args.render_split),
        "--vision-weights",
        str(Path(args.vision_weights).resolve()),
        "--td3-checkpoint",
        str(Path(args.td3_checkpoint).resolve()),
        "--model",
        str(args.model),
        "--vision-model",
        str(args.vision_model),
        "--decode-method",
        str(args.decode_method),
        "--estimate-filter",
        str(estimate_filter),
        "--device",
        str(args.device),
        "--episodes",
        str(int(args.episodes)),
        "--steps",
        str(int(args.steps)),
        "--seed",
        str(int(args.seed)),
        "--paper1-curriculum-level",
        str(args.paper1_curriculum_level),
        "--phase3-target-init",
        str(args.phase3_target_init),
        "--target-z-policy",
        str(args.target_z_policy),
        "--capture-mode",
        str(capture_mode),
        "--capture-radius-km",
        str(float(args.capture_radius_km)),
        "--terminal-controller",
        str(terminal_controller),
        "--terminal-controller-range-km",
        str(float(args.terminal_controller_range_km)),
        "--terminal-controller-blend",
        str(float(args.terminal_controller_blend)),
        "--out-dir",
        str(out_dir),
    ]
    if args.paper1_curriculum_mix is not None:
        cmd.extend(["--paper1-curriculum-mix", str(args.paper1_curriculum_mix)])
    if args.vision_input_size > 0:
        cmd.extend(["--vision-input-size", str(int(args.vision_input_size))])
    if args.max_vision_samples is not None:
        cmd.extend(["--max-vision-samples", str(int(args.max_vision_samples))])
    if estimate_filter == "kalman":
        cmd.extend(["--kf-terminal-mode", str(kf_terminal_mode)])
    return cmd


def _build_oracle_cmd(*, project_root: Path, args: argparse.Namespace, out_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_phase3_snn_td3_oracle.py"),
        "--config",
        str(Path(args.config).resolve()),
        "--observer",
        "gt",
        "--target-mode",
        "phase3_dynamic",
        "--phase3-target-init",
        str(args.phase3_target_init),
        "--paper1-curriculum-level",
        str(args.paper1_curriculum_level),
        "--model",
        str(args.model),
        "--td3-checkpoint",
        str(Path(args.td3_checkpoint).resolve()),
        "--device",
        str(args.device),
        "--episodes",
        str(int(args.episodes)),
        "--steps",
        str(int(args.steps)),
        "--seed",
        str(int(args.seed)),
        "--target-z-policy",
        str(args.target_z_policy),
        "--capture-radius-km",
        str(float(args.capture_radius_km)),
        "--out-dir",
        str(out_dir),
    ]
    if args.paper1_curriculum_mix is not None:
        cmd.extend(["--paper1-curriculum-mix", str(args.paper1_curriculum_mix)])
    return cmd


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    out_root = _unique_out_root(Path(args.out_root).expanduser().resolve())
    out_root.mkdir(parents=True, exist_ok=True)

    plan: list[dict[str, Any]] = []
    runs: list[tuple[str, Path]] = []

    if not args.skip_oracle:
        oracle_dir = out_root / "oracle_true_target_v1"
        cmd = _build_oracle_cmd(project_root=project_root, args=args, out_dir=oracle_dir)
        plan.append({"label": "oracle_true_target_v1", "out_dir": str(oracle_dir), "cmd": cmd})
        _run(cmd, cwd=project_root)
        runs.append(("Oracle", oracle_dir))

    if not args.skip_no_kf:
        no_kf_dir = out_root / "snn_no_kf_v2"
        cmd = _build_vision_cmd(
            project_root=project_root,
            args=args,
            out_dir=no_kf_dir,
            estimate_filter="none",
            terminal_controller="td3",
        )
        plan.append({"label": "snn_no_kf_v2", "out_dir": str(no_kf_dir), "cmd": cmd})
        _run(cmd, cwd=project_root)
        runs.append(("SNN no KF", no_kf_dir))

    if not args.skip_kf_raw:
        kf_raw_dir = out_root / "snn_kf_terminal_raw_v3"
        cmd = _build_vision_cmd(
            project_root=project_root,
            args=args,
            out_dir=kf_raw_dir,
            estimate_filter="kalman",
            terminal_controller="td3",
            kf_terminal_mode="raw",
        )
        plan.append({"label": "snn_kf_terminal_raw_v3", "out_dir": str(kf_raw_dir), "cmd": cmd})
        _run(cmd, cwd=project_root)
        runs.append(("SNN KF terminal raw", kf_raw_dir))

    if not args.skip_kf_pure:
        kf_pure_dir = out_root / "snn_kf_terminal_pure_v1"
        cmd = _build_vision_cmd(
            project_root=project_root,
            args=args,
            out_dir=kf_pure_dir,
            estimate_filter="kalman",
            terminal_controller="pure_pursuit",
            kf_terminal_mode="raw",
        )
        plan.append({"label": "snn_kf_terminal_pure_v1", "out_dir": str(kf_pure_dir), "cmd": cmd})
        _run(cmd, cwd=project_root)
        runs.append(("SNN KF terminal pure pursuit", kf_pure_dir))

    (out_root / "run_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.skip_plot and runs:
        plot_dir = out_root / "figures"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_cmd = [
            sys.executable,
            str(project_root / "scripts" / "plot_phase3_closed_loop_results.py"),
            "--config",
            str(Path(args.config).resolve()),
            "--episode",
            str(int(args.plot_episode)),
            "--output-dir",
            str(plot_dir),
        ]
        for label, run_dir in runs:
            plot_cmd.extend(["--run", label, str(run_dir)])
        _run(plot_cmd, cwd=project_root)

    print(json.dumps({"out_root": str(out_root), "runs": [str(run_dir) for _, run_dir in runs]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
