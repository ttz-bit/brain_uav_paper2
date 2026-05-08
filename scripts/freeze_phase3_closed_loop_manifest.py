from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Freeze Phase3 closed-loop paper results into a traceable manifest JSON."
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
        help="Optional comparison report containing runs with label/run_dir.",
    )
    p.add_argument("--audit-report", type=str, default="", help="Optional pairing audit JSON.")
    p.add_argument("--visual-eval", action="append", nargs=2, metavar=("LABEL", "REPORT_JSON"), default=[])
    p.add_argument("--paper-tag", type=str, default="phase3_closed_loop_main")
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--out", type=str, required=True)
    p.add_argument(
        "--hash-large-files",
        action="store_true",
        help="Also hash model checkpoints. Disabled by default to keep manifest creation fast.",
    )
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _runs_from_comparison(path: Path) -> list[tuple[str, Path]]:
    data = _read_json(path)
    runs: list[tuple[str, Path]] = []
    for item in list(data.get("runs", [])):
        label = str(item.get("label", "")).strip()
        run_dir = str(item.get("run_dir", "")).strip()
        if not label or not run_dir:
            raise ValueError(f"Invalid run entry in {path}: {item}")
        runs.append((label, Path(run_dir).expanduser().resolve()))
    return runs


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


def _git_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except Exception:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_record(path_text: str | None, *, hash_file: bool = False) -> dict[str, Any]:
    if not path_text:
        return {"path": ""}
    path = Path(path_text).expanduser()
    record: dict[str, Any] = {"path": str(path)}
    if path.exists():
        stat = path.stat()
        record.update({"exists": True, "size_bytes": int(stat.st_size)})
        if hash_file:
            record["sha256"] = _sha256(path)
    else:
        record["exists"] = False
    return record


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("done_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _non_captured(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if _as_bool(row.get("captured", False)):
            continue
        out.append(
            {
                "episode": int(row.get("episode", -1)),
                "seed": int(row.get("seed", -1)),
                "done_reason": str(row.get("done_reason", "unknown")),
                "steps_executed": int(row.get("steps_executed", -1)),
                "min_range_km": float(row.get("min_range_km", "nan")),
                "final_range_km": float(row.get("final_range_km", "nan")),
            }
        )
    return out


def _metric(report: dict[str, Any], key: str) -> Any:
    return dict(report.get("metrics", {})).get(key)


def _run_manifest(label: str, run_dir: Path, *, hash_large_files: bool) -> dict[str, Any]:
    report_path = run_dir / "report.json"
    summary_path = run_dir / "summary.csv"
    trajectory_path = run_dir / "trajectory.jsonl"
    if not report_path.exists():
        raise FileNotFoundError(f"{label}: missing {report_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"{label}: missing {summary_path}")
    report = _read_json(report_path)
    summary_rows = _read_csv(summary_path)
    metrics = dict(report.get("metrics", {}))
    return {
        "label": label,
        "run_dir": str(run_dir),
        "task": report.get("task"),
        "purpose": report.get("purpose"),
        "accepted": bool(report.get("accepted", False)),
        "protocol": {
            "seed": report.get("seed"),
            "episodes": report.get("episodes"),
            "steps_per_episode": report.get("steps_per_episode"),
            "phase3_target_init": report.get("phase3_target_init"),
            "capture_mode": report.get("capture_mode"),
            "paper1_curriculum_level": report.get("paper1_curriculum_level"),
            "paper1_curriculum_mix": report.get("paper1_curriculum_mix"),
            "capture_radius_km": report.get("capture_radius_km"),
            "vision_source": report.get("vision_source"),
            "estimate_filter": report.get("estimate_filter"),
            "observer": report.get("observer"),
            "target_mode": report.get("target_mode"),
        },
        "artifacts": {
            "report": _file_record(str(report_path)),
            "summary_csv": _file_record(str(summary_path)),
            "trajectory_jsonl": _file_record(str(trajectory_path)),
            "vision_weights": _file_record(report.get("vision_weights_path"), hash_file=hash_large_files),
            "td3_checkpoint": _file_record(report.get("td3_checkpoint_path"), hash_file=hash_large_files),
            "dataset_root": {"path": str(report.get("dataset_root", ""))},
            "config": _file_record(report.get("config")),
            "render_config": _file_record(report.get("render_config")),
        },
        "metrics": {
            "episodes_completed": metrics.get("episodes_completed"),
            "total_steps": metrics.get("total_steps"),
            "capture_count": metrics.get("capture_count"),
            "capture_rate": metrics.get("capture_rate"),
            "min_range_mean_km": metrics.get("min_range_mean_km"),
            "final_range_mean_km": metrics.get("final_range_mean_km"),
            "target_est_error_mean_km": metrics.get("target_est_error_mean_km"),
            "vision_error_mean_km": metrics.get("vision_error_mean_km"),
            "vision_pixel_error_mean_px": metrics.get("vision_pixel_error_mean_px"),
            "vision_pixel_error_p90_px": metrics.get("vision_pixel_error_p90_px"),
            "zone_violation_total": metrics.get("zone_violation_total"),
            "safety_margin_violation_total": metrics.get("safety_margin_violation_total"),
            "live_render_failure_count": metrics.get("live_render_failure_count"),
            "live_render_reanchor_count": metrics.get("live_render_reanchor_count"),
            "vision_gate_rejection_rate": metrics.get("vision_gate_rejection_rate"),
        },
        "episode_outcomes": {
            "done_reason_counts": _reason_counts(summary_rows),
            "non_captured_episodes": _non_captured(summary_rows),
        },
        "acceptance": report.get("acceptance", {}),
    }


def _visual_eval_manifest(label: str, path: Path) -> dict[str, Any]:
    report = _read_json(path)
    metrics = dict(report.get("metrics", {}))
    return {
        "label": label,
        "report": _file_record(str(path)),
        "dataset": report.get("dataset", {}),
        "weights": _file_record(report.get("weights_path")),
        "separation_check": report.get("separation_check", {}),
        "eval_config": report.get("eval_config", {}),
        "metrics": {
            "pixel_error_mean": metrics.get("pixel_error_mean"),
            "pixel_error_p90": metrics.get("pixel_error_p90"),
            "world_error_mean_m": metrics.get("world_error_mean_m"),
            "world_error_p90_m": metrics.get("world_error_p90_m"),
            "center_baseline_improve_ratio": metrics.get("center_baseline_improve_ratio"),
            "offcenter_count": metrics.get("offcenter_count"),
            "offcenter_pixel_error_mean": metrics.get("offcenter_pixel_error_mean"),
            "offcenter_center_baseline_improve_ratio": metrics.get("offcenter_center_baseline_improve_ratio"),
            "pred_on_land_count": metrics.get("pred_on_land_count"),
            "pred_on_land_ratio": metrics.get("pred_on_land_ratio"),
            "stage_pixel_error_mean": metrics.get("stage_pixel_error_mean"),
            "background_pixel_error_mean": metrics.get("background_pixel_error_mean"),
            "prediction_stats": metrics.get("prediction_stats"),
            "diagnostics": metrics.get("diagnostics"),
        },
    }


def main() -> None:
    args = parse_args()
    run_specs = _collect_runs(args)
    audit_path = Path(args.audit_report).expanduser().resolve() if args.audit_report else None
    manifest = {
        "task": "freeze_phase3_closed_loop_manifest",
        "paper_tag": str(args.paper_tag),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": str(args.notes),
        "code": {
            "git_root": _git_value(["rev-parse", "--show-toplevel"]),
            "commit": _git_value(["rev-parse", "HEAD"]),
            "branch": _git_value(["branch", "--show-current"]),
            "dirty_status": _git_value(["status", "--short"]),
        },
        "pairing_audit": _read_json(audit_path) if audit_path and audit_path.exists() else None,
        "closed_loop_runs": [
            _run_manifest(label, run_dir, hash_large_files=bool(args.hash_large_files))
            for label, run_dir in run_specs
        ],
        "visual_evals": [
            _visual_eval_manifest(str(label), Path(path).expanduser().resolve()) for label, path in args.visual_eval
        ],
        "paper_reporting_guidance": {
            "overall_capture_rate_note": "Report capture_rate over all 64 paired episodes.",
            "valid_scenario_note": (
                "If non-captured episodes are identical across methods and caused by out_of_bounds, "
                "also report valid-scenario capture over the non-boundary subset."
            ),
            "safety_note": (
                "Hard no-fly-zone violations are separate from safety-margin incursions; do not claim "
                "strict safety-margin satisfaction when safety_margin_violation_total is non-zero."
            ),
        },
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(out_path), "closed_loop_run_count": len(run_specs)}, indent=2))


if __name__ == "__main__":
    main()
