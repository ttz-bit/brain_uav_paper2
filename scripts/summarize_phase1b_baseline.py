from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase1b_baseline_runs.json",
        help="Run-list config JSON path",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root used to resolve relative paths",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/reports",
        help="Summary output directory",
    )
    return parser.parse_args()


def _resolve(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_seed(run_name: str) -> int | None:
    m = re.search(r"seed(\d+)", run_name)
    if m is None:
        return None
    return int(m.group(1))


def _artifact_paths(report_path: Path, report: dict[str, Any], project_root: Path) -> dict[str, Path]:
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}
    default_dir = report_path.parent
    visual_dir = _resolve(str(artifacts.get("visual_dir", default_dir / "visuals")), project_root)
    weights_path = _resolve(str(artifacts.get("weights_path", default_dir / "linear_weights.npy")), project_root)
    report_json = _resolve(str(artifacts.get("report_path", report_path)), project_root)
    return {
        "visual_dir": visual_dir,
        "weights_path": weights_path,
        "report_path": report_json,
    }


def _check_artifacts(paths: dict[str, Path], purpose: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    if not paths["report_path"].exists():
        missing.append("report.json")
    if purpose != "evaluation" and (not paths["weights_path"].exists()):
        missing.append("linear_weights.npy")
    vis_ok = paths["visual_dir"].exists() and any(paths["visual_dir"].glob("*.jpg"))
    if not vis_ok:
        missing.append("visuals/*.jpg")
    return len(missing) == 0, missing


def _check_metrics(report: dict[str, Any], purpose: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if purpose == "evaluation":
        mse = report.get("mse")
        mae = report.get("mae")
        if not isinstance(mse, (int, float)) or not np.isfinite(float(mse)) or float(mse) < 0.0:
            errors.append("invalid_mse")
        if not isinstance(mae, (int, float)) or not np.isfinite(float(mae)) or float(mae) < 0.0:
            errors.append("invalid_mae")
    else:
        initial_loss = float(report.get("initial_loss", "nan"))
        final_loss = float(report.get("final_loss", "nan"))
        improve_ratio = float(report.get("improve_ratio", "nan"))
        if not (final_loss < initial_loss):
            errors.append("final_loss_not_lower_than_initial")
        if not (improve_ratio > 0.0):
            errors.append("improve_ratio_not_positive")
    return len(errors) == 0, errors


def _stability_by_split(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for split in sorted({str(r["split"]) for r in rows if str(r.get("split", "")).strip()}):
        split_rows = [r for r in rows if str(r["split"]) == split]
        all_positive = all(bool(r["metrics_ok"]) for r in split_rows)
        improve_values = [
            float(r["improve_ratio"])
            for r in split_rows
            if isinstance(r.get("improve_ratio"), (int, float))
        ]
        out[split] = {
            "num_runs": len(split_rows),
            "all_metrics_ok": all_positive,
            "improve_ratio_min": min(improve_values) if improve_values else None,
            "improve_ratio_max": max(improve_values) if improve_values else None,
            "run_names": [str(r["run_name"]) for r in split_rows],
        }
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = _resolve(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = _resolve(args.config, project_root)
    cfg = _read_json(cfg_path)
    run_items = cfg.get("runs")
    if not isinstance(run_items, list) or len(run_items) == 0:
        raise ValueError(f"Invalid or empty runs list in config: {cfg_path}")

    rows: list[dict[str, Any]] = []
    for item in run_items:
        if not isinstance(item, dict):
            raise TypeError("Each run item must be a dict")

        run_name = str(item.get("run_name", "")).strip()
        report_path_str = str(item.get("report_path", "")).strip()
        expected_split = str(item.get("expected_split", "")).strip()
        purpose = str(item.get("purpose", "fit_check")).strip() or "fit_check"
        if not run_name or not report_path_str:
            raise ValueError(f"Invalid run entry: {item}")

        report_path = _resolve(report_path_str, project_root)
        if not report_path.exists():
            rows.append(
                {
                    "run_name": run_name,
                    "split": expected_split,
                    "purpose": purpose,
                    "seed": _infer_seed(run_name),
                    "num_samples": None,
                    "initial_loss": None,
                    "final_loss": None,
                    "improve_ratio": None,
                    "metrics_ok": False,
                    "artifacts_ok": False,
                    "overall_ok": False,
                    "issues": "missing_report_json",
                    "report_path": str(report_path),
                }
            )
            continue

        report = _read_json(report_path)
        split = str(report.get("split", "")).strip() or expected_split
        metrics_ok, metric_errors = _check_metrics(report, purpose)
        artifact_paths = _artifact_paths(report_path, report, project_root)
        artifacts_ok, missing_artifacts = _check_artifacts(artifact_paths, purpose)

        issues = []
        if expected_split and split != expected_split:
            issues.append(f"split_mismatch(expected={expected_split},actual={split})")
        issues.extend(metric_errors)
        issues.extend([f"missing_{x}" for x in missing_artifacts])

        row = {
            "run_name": run_name,
            "split": split,
            "purpose": purpose,
            "seed": _infer_seed(run_name),
            "num_samples": int(report.get("num_samples", 0)),
            "initial_loss": (
                float(report.get("initial_loss", "nan")) if "initial_loss" in report else None
            ),
            "final_loss": (
                float(report.get("final_loss", "nan")) if "final_loss" in report else None
            ),
            "improve_ratio": (
                float(report.get("improve_ratio", "nan")) if "improve_ratio" in report else None
            ),
            "mse": float(report.get("mse", "nan")) if "mse" in report else None,
            "mae": float(report.get("mae", "nan")) if "mae" in report else None,
            "metrics_ok": metrics_ok,
            "artifacts_ok": artifacts_ok,
            "overall_ok": metrics_ok and artifacts_ok and (len(issues) == 0),
            "issues": ";".join(issues) if issues else "",
            "report_path": str(report_path),
        }
        rows.append(row)

    # Training/evaluation separation note for current phase.
    # If all registered runs are "fit_check", this summary is only about fitting stability.
    purposes = {str(r["purpose"]) for r in rows}
    eval_separation_ok = "evaluation" in purposes

    stability = _stability_by_split(rows)
    summary = {
        "phase": "phase1b_baseline_summary",
        "config_path": str(cfg_path),
        "num_runs": len(rows),
        "all_runs_overall_ok": all(bool(r["overall_ok"]) for r in rows),
        "all_metrics_ok": all(bool(r["metrics_ok"]) for r in rows),
        "all_artifacts_ok": all(bool(r["artifacts_ok"]) for r in rows),
        "eval_separation_ok": eval_separation_ok,
        "eval_separation_note": (
            "No dedicated evaluation run registered; current runs are fit checks."
            if not eval_separation_ok
            else "Dedicated evaluation run exists."
        ),
        "stability_by_split": stability,
        "runs": rows,
    }

    csv_path = out_dir / "phase1b_baseline_summary.csv"
    json_path = out_dir / "phase1b_baseline_summary.json"

    fieldnames = [
        "run_name",
        "split",
        "purpose",
        "seed",
        "num_samples",
        "initial_loss",
        "final_loss",
        "improve_ratio",
        "mse",
        "mae",
        "metrics_ok",
        "artifacts_ok",
        "overall_ok",
        "issues",
        "report_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] csv: {csv_path}")
    print(f"[DONE] json: {json_path}")


if __name__ == "__main__":
    main()
