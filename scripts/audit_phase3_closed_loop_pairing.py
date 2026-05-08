from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunAudit:
    label: str
    run_dir: Path
    report: dict[str, Any]
    summary_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Audit Phase3 closed-loop runs for paired-seed consistency, common protocol "
            "settings, and basic acceptance/safety conditions."
        )
    )
    p.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "RUN_DIR"),
        help="Repeatable. Run directory must contain report.json and summary.csv.",
    )
    p.add_argument(
        "--comparison-report",
        type=str,
        default="",
        help="Optional comparison_report.json/txt produced by plot_phase3_closed_loop_results.py.",
    )
    p.add_argument("--out", type=str, default="", help="Optional path to write the audit JSON report.")
    p.add_argument("--expected-seed", type=int, default=None)
    p.add_argument("--expected-episodes", type=int, default=None)
    p.add_argument("--expected-steps", type=int, default=None)
    p.add_argument("--expected-capture-mode", type=str, default="true_target")
    p.add_argument("--expected-phase3-target-init", type=str, default="paper1_goal")
    p.add_argument(
        "--strict-safety-margin",
        action="store_true",
        help="Fail if safety_margin_violation_total is non-zero. By default this is reported as a warning.",
    )
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_run(label: str, run_dir: Path) -> RunAudit:
    report_path = run_dir / "report.json"
    summary_path = run_dir / "summary.csv"
    if not report_path.exists():
        raise FileNotFoundError(f"{label}: missing report.json at {report_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"{label}: missing summary.csv at {summary_path}")
    return RunAudit(
        label=label,
        run_dir=run_dir,
        report=_read_json(report_path),
        summary_rows=_read_csv(summary_path),
    )


def _runs_from_comparison(path: Path) -> list[tuple[str, Path]]:
    data = _read_json(path)
    out: list[tuple[str, Path]] = []
    for item in list(data.get("runs", [])):
        label = str(item.get("label", "")).strip()
        run_dir = str(item.get("run_dir", "")).strip()
        if not label or not run_dir:
            raise ValueError(f"Invalid run entry in {path}: {item}")
        out.append((label, Path(run_dir).expanduser().resolve()))
    return out


def _normalized(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalized(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list):
        return [_normalized(v) for v in value]
    return value


def _same_value(values: list[Any]) -> bool:
    if not values:
        return True
    first = _normalized(values[0])
    return all(_normalized(v) == first for v in values[1:])


def _summary_seeds(run: RunAudit) -> list[int]:
    seeds: list[int] = []
    for row in run.summary_rows:
        if "seed" not in row:
            raise ValueError(f"{run.label}: summary.csv missing seed column")
        seeds.append(int(row["seed"]))
    return seeds


def _summary_episodes(run: RunAudit) -> list[int]:
    episodes: list[int] = []
    for row in run.summary_rows:
        if "episode" not in row:
            raise ValueError(f"{run.label}: summary.csv missing episode column")
        episodes.append(int(row["episode"]))
    return episodes


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _is_boundary_failure(row: dict[str, Any]) -> bool:
    reason = str(row.get("done_reason", "")).strip().lower()
    return (not _as_bool(row.get("captured", False))) and reason in {"out_of_bounds", "target_out_of_bounds"}


def _reason_counts(run: RunAudit) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in run.summary_rows:
        reason = str(row.get("done_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _non_captured_episodes(run: RunAudit) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for row in run.summary_rows:
        if _as_bool(row.get("captured", False)):
            continue
        episodes.append(
            {
                "episode": int(row.get("episode", -1)),
                "seed": int(row.get("seed", -1)),
                "done_reason": str(row.get("done_reason", "unknown")),
                "final_range_km": float(row.get("final_range_km", "nan")),
                "min_range_km": float(row.get("min_range_km", "nan")),
                "steps_executed": int(row.get("steps_executed", -1)),
            }
        )
    return episodes


def _valid_capture_summary(run: RunAudit) -> dict[str, Any]:
    total = len(run.summary_rows)
    captured = sum(1 for row in run.summary_rows if _as_bool(row.get("captured", False)))
    valid_rows = [row for row in run.summary_rows if not _is_boundary_failure(row)]
    valid_captured = sum(1 for row in valid_rows if _as_bool(row.get("captured", False)))
    boundary_failures = [
        {
            "episode": int(row.get("episode", -1)),
            "seed": int(row.get("seed", -1)),
            "done_reason": str(row.get("done_reason", "unknown")),
        }
        for row in run.summary_rows
        if _is_boundary_failure(row)
    ]
    return {
        "capture": f"{captured}/{total}",
        "capture_rate": float(captured / max(1, total)),
        "valid_capture": f"{valid_captured}/{len(valid_rows)}",
        "valid_capture_rate": float(valid_captured / max(1, len(valid_rows))),
        "boundary_failure_count": int(len(boundary_failures)),
        "boundary_failure_episodes": boundary_failures,
    }


def _metric(report: dict[str, Any], key: str) -> Any:
    return dict(report.get("metrics", {})).get(key)


def _add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any = None, severity: str = "error") -> None:
    checks.append({"name": name, "passed": bool(passed), "severity": severity, "detail": detail})


def _collect_runs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    runs: list[tuple[str, Path]] = []
    if args.comparison_report:
        runs.extend(_runs_from_comparison(Path(args.comparison_report).expanduser().resolve()))
    if args.run:
        runs.extend((str(label), Path(run_dir).expanduser().resolve()) for label, run_dir in args.run)
    if not runs:
        raise SystemExit("Provide at least one --run LABEL RUN_DIR or --comparison-report.")
    labels = [label for label, _ in runs]
    if len(labels) != len(set(labels)):
        raise SystemExit(f"Run labels must be unique: {labels}")
    return runs


def main() -> None:
    args = parse_args()
    run_specs = _collect_runs(args)
    runs = [_load_run(label, run_dir) for label, run_dir in run_specs]
    checks: list[dict[str, Any]] = []

    common_keys = [
        "config",
        "env_source",
        "planner",
        "td3_checkpoint_path",
        "seed",
        "episodes",
        "steps_per_episode",
        "phase3_target_init",
        "paper1_curriculum_level",
        "paper1_curriculum_mix",
        "capture_radius_km",
    ]
    for key in common_keys:
        present = [key in r.report for r in runs]
        values = [r.report.get(key) for r in runs if key in r.report]
        if all(present):
            _add_check(
                checks,
                f"common_report_field:{key}",
                _same_value(values),
                {r.label: r.report.get(key) for r in runs},
            )
        else:
            _add_check(
                checks,
                f"common_report_field:{key}",
                False,
                {"missing_in": [r.label for r, has_key in zip(runs, present) if not has_key]},
                severity="warning",
            )

    if args.expected_seed is not None:
        _add_check(
            checks,
            "expected_seed",
            all(int(r.report.get("seed", -1)) == int(args.expected_seed) for r in runs),
            {r.label: r.report.get("seed") for r in runs},
        )
    if args.expected_episodes is not None:
        _add_check(
            checks,
            "expected_episodes",
            all(int(r.report.get("episodes", -1)) == int(args.expected_episodes) for r in runs),
            {r.label: r.report.get("episodes") for r in runs},
        )
    if args.expected_steps is not None:
        _add_check(
            checks,
            "expected_steps_per_episode",
            all(int(r.report.get("steps_per_episode", -1)) == int(args.expected_steps) for r in runs),
            {r.label: r.report.get("steps_per_episode") for r in runs},
        )

    _add_check(
        checks,
        "expected_phase3_target_init",
        all(str(r.report.get("phase3_target_init", "")) == str(args.expected_phase3_target_init) for r in runs),
        {r.label: r.report.get("phase3_target_init") for r in runs},
    )

    vision_runs = [r for r in runs if r.report.get("task") == "run_phase3_vision_td3"]
    if vision_runs:
        missing_capture_mode = [r.label for r in vision_runs if "capture_mode" not in r.report]
        if missing_capture_mode:
            _add_check(
                checks,
                "vision_capture_mode_present",
                False,
                {"missing_in": missing_capture_mode},
                severity="warning",
            )
        vision_runs_with_capture_mode = [r for r in vision_runs if "capture_mode" in r.report]
        _add_check(
            checks,
            "vision_capture_mode",
            all(str(r.report.get("capture_mode", "")) == str(args.expected_capture_mode) for r in vision_runs_with_capture_mode),
            {r.label: r.report.get("capture_mode") for r in vision_runs_with_capture_mode},
        )
        _add_check(
            checks,
            "vision_source_is_live",
            all(str(r.report.get("vision_source", "")) == "phase3_map_live" for r in vision_runs),
            {r.label: r.report.get("vision_source") for r in vision_runs},
            severity="warning",
        )

    oracle_runs = [r for r in runs if r.report.get("task") == "run_phase3_snn_td3_oracle"]
    for r in oracle_runs:
        _add_check(
            checks,
            f"oracle_protocol:{r.label}",
            str(r.report.get("observer", "")) == "gt" and str(r.report.get("target_mode", "")) == "phase3_dynamic",
            {"observer": r.report.get("observer"), "target_mode": r.report.get("target_mode")},
        )

    seed_lists = {r.label: _summary_seeds(r) for r in runs}
    episode_lists = {r.label: _summary_episodes(r) for r in runs}
    first_seed_list = next(iter(seed_lists.values()))
    first_episode_list = next(iter(episode_lists.values()))
    _add_check(checks, "summary_seed_lists_identical", all(v == first_seed_list for v in seed_lists.values()), seed_lists)
    _add_check(
        checks,
        "summary_episode_indices_identical",
        all(v == first_episode_list for v in episode_lists.values()),
        episode_lists,
    )
    expected_seed_base = int(args.expected_seed if args.expected_seed is not None else runs[0].report.get("seed", 0))
    expected_episode_count = int(
        args.expected_episodes if args.expected_episodes is not None else runs[0].report.get("episodes", len(first_seed_list))
    )
    expected_seeds = list(range(expected_seed_base, expected_seed_base + expected_episode_count))
    expected_episodes = list(range(expected_episode_count))
    _add_check(checks, "summary_seeds_contiguous", first_seed_list == expected_seeds, {"actual": first_seed_list, "expected": expected_seeds})
    _add_check(
        checks,
        "summary_episode_indices_contiguous",
        first_episode_list == expected_episodes,
        {"actual": first_episode_list, "expected": expected_episodes},
    )

    non_captured_by_run = {r.label: _non_captured_episodes(r) for r in runs}
    non_captured_episode_sets = {
        label: [int(item["episode"]) for item in episodes] for label, episodes in non_captured_by_run.items()
    }
    first_non_captured = next(iter(non_captured_episode_sets.values()))
    _add_check(
        checks,
        "non_captured_episode_sets_identical",
        all(v == first_non_captured for v in non_captured_episode_sets.values()),
        non_captured_episode_sets,
        severity="warning",
    )
    boundary_failure_sets = {
        r.label: [int(item["episode"]) for item in _valid_capture_summary(r)["boundary_failure_episodes"]]
        for r in runs
    }
    first_boundary_failures = next(iter(boundary_failure_sets.values()))
    _add_check(
        checks,
        "boundary_failure_episode_sets_identical",
        all(v == first_boundary_failures for v in boundary_failure_sets.values()),
        boundary_failure_sets,
        severity="warning",
    )

    for r in runs:
        metrics = dict(r.report.get("metrics", {}))
        _add_check(checks, f"accepted:{r.label}", bool(r.report.get("accepted", False)), r.report.get("acceptance", {}))
        _add_check(
            checks,
            f"episodes_completed:{r.label}",
            int(metrics.get("episodes_completed", len(r.summary_rows))) == int(r.report.get("episodes", len(r.summary_rows))),
            {"metrics": metrics.get("episodes_completed"), "report": r.report.get("episodes"), "summary_rows": len(r.summary_rows)},
        )
        _add_check(
            checks,
            f"hard_zone_violations_zero:{r.label}",
            int(float(metrics.get("zone_violation_total", 0))) == 0,
            metrics.get("zone_violation_total"),
        )
        safety_total = int(float(metrics.get("safety_margin_violation_total", 0)))
        _add_check(
            checks,
            f"safety_margin_violations_zero:{r.label}",
            safety_total == 0,
            safety_total,
            severity="error" if args.strict_safety_margin else "warning",
        )

    metrics_table = []
    for r in runs:
        metrics_table.append(
            {
                "label": r.label,
                "run_dir": str(r.run_dir),
                "accepted": bool(r.report.get("accepted", False)),
                "capture_rate": _metric(r.report, "capture_rate"),
                "min_range_mean_km": _metric(r.report, "min_range_mean_km"),
                "final_range_mean_km": _metric(r.report, "final_range_mean_km"),
                "target_est_error_mean_km": _metric(r.report, "target_est_error_mean_km"),
                "vision_error_mean_km": _metric(r.report, "vision_error_mean_km"),
                "vision_pixel_error_mean_px": _metric(r.report, "vision_pixel_error_mean_px"),
                "vision_pixel_error_p90_px": _metric(r.report, "vision_pixel_error_p90_px"),
                "zone_violation_total": _metric(r.report, "zone_violation_total"),
                "safety_margin_violation_total": _metric(r.report, "safety_margin_violation_total"),
                "live_render_failure_count": _metric(r.report, "live_render_failure_count"),
                "live_render_reanchor_count": _metric(r.report, "live_render_reanchor_count"),
                "vision_gate_rejection_rate": _metric(r.report, "vision_gate_rejection_rate"),
                "valid_capture_summary": _valid_capture_summary(r),
                "done_reason_counts": _reason_counts(r),
                "non_captured_episodes": non_captured_by_run[r.label],
            }
        )

    failed_errors = [c for c in checks if not c["passed"] and c["severity"] == "error"]
    failed_warnings = [c for c in checks if not c["passed"] and c["severity"] == "warning"]
    audit = {
        "task": "audit_phase3_closed_loop_pairing",
        "accepted": len(failed_errors) == 0,
        "error_count": len(failed_errors),
        "warning_count": len(failed_warnings),
        "runs": metrics_table,
        "checks": checks,
    }
    text = json.dumps(audit, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    if failed_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
