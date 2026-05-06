from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper2.common.config import load_yaml
from paper2.env_adapter.paper1_bridge import Paper1EnvBridge
from paper2.paper1_method.curriculum import parse_curriculum_mix


@dataclass
class RunBundle:
    label: str
    run_dir: Path
    report: dict[str, Any]
    summary_rows: list[dict[str, Any]]
    trajectory_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        required=True,
        help="Repeatable. Provide a label and a run directory containing report.json, summary.csv, and trajectory.jsonl.",
    )
    p.add_argument("--config", type=str, default="configs/env.yaml")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs/phase3_closed_loop/figures")
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_run_dir(path_text: str) -> Path:
    return Path(path_text).expanduser().resolve()


def _load_run(label: str, run_dir: Path) -> RunBundle:
    report_path = run_dir / "report.json"
    summary_path = run_dir / "summary.csv"
    traj_path = run_dir / "trajectory.jsonl"
    if not report_path.exists():
        raise FileNotFoundError(f"Missing report.json in {run_dir}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv in {run_dir}")
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectory.jsonl in {run_dir}")
    return RunBundle(
        label=label,
        run_dir=run_dir,
        report=_read_json(report_path),
        summary_rows=_read_csv(summary_path),
        trajectory_rows=_read_jsonl(traj_path),
    )


def _metric_value(report: dict[str, Any], key: str) -> float:
    metrics = report.get("metrics", {})
    value = metrics.get(key)
    if value is None:
        return float("nan")
    return float(value)


def _curriculum_mix_from_report(report: dict[str, Any]) -> dict[str, float]:
    mix = report.get("paper1_curriculum_mix")
    fallback_level = str(report.get("paper1_curriculum_level", "hard"))
    if isinstance(mix, dict):
        return {str(k): float(v) for k, v in mix.items()}
    if isinstance(mix, str) and mix.strip():
        return parse_curriculum_mix(mix, fallback_level=fallback_level)
    return parse_curriculum_mix(None, fallback_level=fallback_level)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _episode_rows(rows: list[dict[str, Any]], episode: int) -> list[dict[str, Any]]:
    out = [r for r in rows if int(r.get("episode", -1)) == int(episode)]
    out.sort(key=lambda r: int(r.get("step", 0)))
    return out


def _zones_for_episode(report: dict[str, Any], episode: int) -> list[Any]:
    seed = int(report.get("seed", 0)) + int(episode)
    bridge = Paper1EnvBridge(seed=seed, curriculum_mix=_curriculum_mix_from_report(report))
    bridge.reset(seed=seed)
    return bridge.get_no_fly_zones()


def _plot_metrics(runs: list[RunBundle], out_path: Path) -> None:
    labels = [r.label for r in runs]
    metric_specs = [
        ("capture_rate", "capture_rate"),
        ("min_range_mean_km", "min_range_mean_km"),
        ("target_est_error_mean_km", "target_est_error_mean_km"),
        ("vision_error_mean_km", "vision_error_mean_km"),
    ]
    values = []
    for metric_key, _ in metric_specs:
        values.append([_metric_value(r.report, metric_key) for r in runs])

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes = axes.flatten()
    for ax, (metric_key, title), ys in zip(axes, metric_specs, values):
        x = np.arange(len(labels), dtype=float)
        colors = list(plt.cm.tab10.colors)
        ax.bar(x, ys, color=[colors[i % len(colors)] for i in range(len(labels))])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if metric_key == "capture_rate":
            ax.set_ylim(0.0, 1.0)
        elif np.isfinite(np.asarray(ys, dtype=float)).any():
            ymax = float(np.nanmax(np.asarray(ys, dtype=float)))
            ax.set_ylim(0.0, ymax * 1.15 if ymax > 0 else 1.0)
    fig.suptitle("Phase 3 Closed-Loop Comparison")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_trajectories(runs: list[RunBundle], episode: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.2), constrained_layout=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        rows = _episode_rows(run.trajectory_rows, episode)
        if not rows:
            ax.set_title(f"{run.label} ep{episode:03d} (no rows)")
            ax.axis("off")
            continue

        aircraft_xy = np.array([[float(r["aircraft_x"]), float(r["aircraft_y"])] for r in rows], dtype=float)
        target_xy = np.array([[float(r["target_x"]), float(r["target_y"])] for r in rows], dtype=float)
        zone_flag = np.array([bool(r.get("zone_violation", False)) for r in rows], dtype=bool)
        safety_flag = np.array([bool(r.get("safety_margin_violation", False)) for r in rows], dtype=bool)

        ax.plot(aircraft_xy[:, 0], aircraft_xy[:, 1], label="aircraft", linewidth=1.8)
        ax.plot(target_xy[:, 0], target_xy[:, 1], label="target", linewidth=1.8)
        ax.scatter(aircraft_xy[0, 0], aircraft_xy[0, 1], s=28, marker="o", label="aircraft start")
        ax.scatter(target_xy[0, 0], target_xy[0, 1], s=28, marker="o", label="target start")
        ax.scatter(aircraft_xy[-1, 0], aircraft_xy[-1, 1], s=36, marker="x", label="aircraft end")
        ax.scatter(target_xy[-1, 0], target_xy[-1, 1], s=36, marker="x", label="target end")
        if zone_flag.any():
            ax.scatter(aircraft_xy[zone_flag, 0], aircraft_xy[zone_flag, 1], s=18, c="red", label="zone violation")
        if safety_flag.any():
            ax.scatter(aircraft_xy[safety_flag, 0], aircraft_xy[safety_flag, 1], s=18, c="orange", label="safety margin")

        try:
            zones = _zones_for_episode(run.report, episode)
            for zone in zones:
                circle = plt.Circle(
                    (float(zone.center_world[0]), float(zone.center_world[1])),
                    float(zone.radius_world),
                    fill=False,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.6,
                )
                ax.add_patch(circle)
                if float(zone.safety_margin) > 0.0:
                    margin_circle = plt.Circle(
                        (float(zone.center_world[0]), float(zone.center_world[1])),
                        float(zone.radius_world + zone.safety_margin),
                        fill=False,
                        linestyle=":",
                        linewidth=0.9,
                        alpha=0.35,
                    )
                    ax.add_patch(margin_circle)
        except Exception:
            pass

        ax.set_title(run.label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Episode {episode:03d} Trajectories")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_ranges(runs: list[RunBundle], episode: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for run in runs:
        rows = _episode_rows(run.trajectory_rows, episode)
        if not rows:
            continue
        steps = np.array([int(r["step"]) for r in rows], dtype=int)
        ranges = np.array([_safe_float(r["range_to_target_km"]) for r in rows], dtype=float)
        ax.plot(steps, ranges, linewidth=1.8, label=run.label)
    ax.set_xlabel("step")
    ax.set_ylabel("range_to_target_km")
    ax.set_title(f"Episode {episode:03d} Range Curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_estimation_errors(runs: list[RunBundle], episode: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    for run in runs:
        rows = _episode_rows(run.trajectory_rows, episode)
        if not rows:
            continue
        steps = np.array([int(r["step"]) for r in rows], dtype=int)
        est_err = np.array([_safe_float(r.get("target_est_error_km")) for r in rows], dtype=float)
        vis_err = np.array([_safe_float(r.get("vision_error_km")) for r in rows], dtype=float)
        ax.plot(steps, est_err, linewidth=1.5, label=f"{run.label} estimate error")
        if np.isfinite(vis_err).any():
            ax.plot(steps, vis_err, linewidth=1.0, linestyle="--", alpha=0.8, label=f"{run.label} vision error")
    ax.set_xlabel("step")
    ax.set_ylabel("error_km")
    ax.set_title(f"Episode {episode:03d} Estimation Errors")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_target_zoom(runs: list[RunBundle], episode: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.0), constrained_layout=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        rows = _episode_rows(run.trajectory_rows, episode)
        if not rows:
            ax.set_title(f"{run.label} ep{episode:03d} (no rows)")
            ax.axis("off")
            continue
        target_xy = np.array([[float(r["target_x"]), float(r["target_y"])] for r in rows], dtype=float)
        origin = target_xy[0].copy()
        delta = target_xy - origin
        ax.plot(delta[:, 0], delta[:, 1], linewidth=2.0, label="target")
        ax.scatter(delta[0, 0], delta[0, 1], s=28, marker="o", label="start")
        ax.scatter(delta[-1, 0], delta[-1, 1], s=36, marker="x", label="end")
        ax.set_title(run.label)
        ax.set_xlabel("target_dx_km")
        ax.set_ylabel("target_dy_km")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Episode {episode:03d} Target Motion Zoom")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_estimate_tracks(runs: list[RunBundle], episode: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.0), constrained_layout=True)
    if len(runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, runs):
        rows = _episode_rows(run.trajectory_rows, episode)
        if not rows:
            ax.set_title(f"{run.label} ep{episode:03d} (no rows)")
            ax.axis("off")
            continue
        target_xy = np.array([[float(r["target_x"]), float(r["target_y"])] for r in rows], dtype=float)
        estimate_xy = np.array([[float(r["estimate_x"]), float(r["estimate_y"])] for r in rows], dtype=float)
        ax.plot(target_xy[:, 0], target_xy[:, 1], linewidth=2.0, label="target")
        if np.isfinite(estimate_xy).all():
            ax.plot(estimate_xy[:, 0], estimate_xy[:, 1], linewidth=1.5, alpha=0.85, label="estimate")
        ax.scatter(target_xy[0, 0], target_xy[0, 1], s=28, marker="o", label="target start")
        ax.scatter(target_xy[-1, 0], target_xy[-1, 1], s=36, marker="x", label="target end")
        points = target_xy
        if np.isfinite(estimate_xy).all():
            points = np.vstack([target_xy, estimate_xy])
        center = np.nanmedian(points, axis=0)
        radius = float(np.nanpercentile(np.linalg.norm(points - center, axis=1), 90)) if len(points) else 10.0
        radius = max(radius, 30.0)
        ax.set_xlim(center[0] - radius * 1.25, center[0] + radius * 1.25)
        ax.set_ylim(center[1] - radius * 1.25, center[1] + radius * 1.25)
        ax.set_title(run.label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle(f"Episode {episode:03d} Target And Estimate Tracks")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _stage_to_value(stage: str) -> int:
    if stage == "terminal":
        return 0
    if stage == "mid":
        return 1
    if stage == "far":
        return 2
    return -1


def _stage_to_gsd(stage: str) -> float:
    if stage == "terminal":
        return 0.005
    if stage == "mid":
        return 0.010
    if stage == "far":
        return 0.020
    return float("nan")


def _plot_stage_usage(runs: list[RunBundle], episode: int, out_path: Path, csv_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True, sharex=True)
    summary_rows: list[dict[str, Any]] = []
    for run in runs:
        rows = _episode_rows(run.trajectory_rows, episode)
        if not rows:
            continue
        steps = np.array([int(r["step"]) for r in rows], dtype=int)
        stages = [str(r.get("vision_stage", "")) for r in rows]
        stage_values = np.array([_stage_to_value(s) for s in stages], dtype=float)
        gsds = np.array([_stage_to_gsd(s) for s in stages], dtype=float)
        axes[0].plot(steps, stage_values, linewidth=1.6, label=run.label)
        axes[1].plot(steps, gsds, linewidth=1.6, label=run.label)
        for stage in ("far", "mid", "terminal"):
            count = sum(1 for s in stages if s == stage)
            summary_rows.append(
                {
                    "label": run.label,
                    "episode": int(episode),
                    "stage": stage,
                    "steps": int(count),
                    "ratio": float(count / max(1, len(stages))),
                    "gsd_km_per_px": _stage_to_gsd(stage),
                    "image_rate_hz": 1.0,
                }
            )

    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["terminal", "mid", "far"])
    axes[0].set_ylabel("stage")
    axes[0].set_title(f"Episode {episode:03d} Perception Stage Schedule")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("gsd_km_per_px")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    if summary_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = [_load_run(label, _resolve_run_dir(run_dir)) for label, run_dir in args.run]

    comparison_rows: list[dict[str, Any]] = []
    for run in runs:
        row = {
            "label": run.label,
            "run_dir": str(run.run_dir),
            "accepted": bool(run.report.get("accepted", False)),
            "capture_rate": _metric_value(run.report, "capture_rate"),
            "min_range_mean_km": _metric_value(run.report, "min_range_mean_km"),
            "final_range_mean_km": _metric_value(run.report, "final_range_mean_km"),
            "target_est_error_mean_km": _metric_value(run.report, "target_est_error_mean_km"),
            "vision_error_mean_km": _metric_value(run.report, "vision_error_mean_km"),
            "zone_violation_total": _metric_value(run.report, "zone_violation_total"),
            "safety_margin_violation_total": _metric_value(run.report, "safety_margin_violation_total"),
            "vision_gate_rejection_rate": _metric_value(run.report, "vision_gate_rejection_rate"),
        }
        comparison_rows.append(row)

    (output_dir / "comparison_report.json").write_text(
        json.dumps({"episode": int(args.episode), "runs": comparison_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "comparison_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()) if comparison_rows else [])
        if comparison_rows:
            writer.writeheader()
            writer.writerows(comparison_rows)

    _plot_metrics(runs, output_dir / "metrics_compare.png")
    _plot_trajectories(runs, int(args.episode), output_dir / f"trajectories_ep{int(args.episode):03d}.png")
    _plot_ranges(runs, int(args.episode), output_dir / f"ranges_ep{int(args.episode):03d}.png")
    _plot_estimation_errors(runs, int(args.episode), output_dir / f"errors_ep{int(args.episode):03d}.png")
    _plot_target_zoom(runs, int(args.episode), output_dir / f"target_zoom_ep{int(args.episode):03d}.png")
    _plot_estimate_tracks(runs, int(args.episode), output_dir / f"estimate_tracks_ep{int(args.episode):03d}.png")
    _plot_stage_usage(
        runs,
        int(args.episode),
        output_dir / f"stage_usage_ep{int(args.episode):03d}.png",
        output_dir / f"stage_usage_ep{int(args.episode):03d}.csv",
    )

    print(json.dumps({"output_dir": str(output_dir), "runs": comparison_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
