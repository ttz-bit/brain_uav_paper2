from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/phase3a_snn_runs.json",
        help="SNN experiment list config",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root for relative-path resolution",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/reports",
        help="Output directory for summary artifacts",
    )
    return parser.parse_args()


def _resolve(path_str: str, project_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (project_root / p).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_valid_metric(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _check_train_metrics(report: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    initial = report.get("initial_loss")
    final = report.get("final_loss")
    improve = report.get("improve_ratio")
    if not _is_valid_metric(initial):
        errors.append("invalid_initial_loss")
    if not _is_valid_metric(final):
        errors.append("invalid_final_loss")
    if not _is_valid_metric(improve):
        errors.append("invalid_improve_ratio")
    if not errors:
        if not (float(final) < float(initial)):
            errors.append("final_loss_not_lower_than_initial")
        if not (float(improve) > 0.0):
            errors.append("improve_ratio_not_positive")
    return len(errors) == 0, errors


def _check_eval_metrics(report: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for key in ("mse", "mae", "mse_xy", "mae_xy"):
        val = report.get(key)
        if not _is_valid_metric(val):
            errors.append(f"invalid_{key}")
        elif float(val) < 0.0:
            errors.append(f"negative_{key}")
    return len(errors) == 0, errors


def _check_train_artifacts(train_report_path: Path, report: dict[str, Any], project_root: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    report_path = _resolve(str(artifacts.get("report_path", train_report_path)), project_root)
    weights_path = _resolve(str(artifacts.get("weights_path", train_report_path.parent / "model.pth")), project_root)
    visual_dir = _resolve(str(artifacts.get("visual_dir", train_report_path.parent / "visuals")), project_root)

    if not report_path.exists():
        missing.append("train_report.json")
    if not weights_path.exists():
        missing.append("train_weights")
    if not (visual_dir.exists() and any(visual_dir.glob("*.jpg"))):
        missing.append("train_visuals")
    return len(missing) == 0, missing


def _check_eval_artifacts(eval_report_path: Path, report: dict[str, Any], project_root: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    report_path = _resolve(str(artifacts.get("report_path", eval_report_path)), project_root)
    visual_dir = _resolve(str(artifacts.get("visual_dir", eval_report_path.parent / "visuals")), project_root)

    if not report_path.exists():
        missing.append("eval_report.json")
    if not (visual_dir.exists() and any(visual_dir.glob("*.jpg"))):
        missing.append("eval_visuals")
    return len(missing) == 0, missing


def _safe_float(x: Any) -> float | None:
    if _is_valid_metric(x):
        return float(x)
    return None


def _aggregate_by_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sample_values = sorted({int(r["samples"]) for r in rows if isinstance(r.get("samples"), int)})
    for sample_count in sample_values:
        sample_rows = [r for r in rows if int(r["samples"]) == sample_count]
        eval_mse = [float(r["eval_mse"]) for r in sample_rows if isinstance(r.get("eval_mse"), (int, float))]
        eval_mae = [float(r["eval_mae"]) for r in sample_rows if isinstance(r.get("eval_mae"), (int, float))]
        out[str(sample_count)] = {
            "num_experiments": len(sample_rows),
            "all_ok": all(bool(r["overall_ok"]) for r in sample_rows),
            "train_all_ok": all(bool(r["train_metrics_ok"]) for r in sample_rows),
            "eval_all_ok": all(bool(r["eval_metrics_ok"]) for r in sample_rows),
            "eval_mse_min": min(eval_mse) if eval_mse else None,
            "eval_mse_max": max(eval_mse) if eval_mse else None,
            "eval_mae_min": min(eval_mae) if eval_mae else None,
            "eval_mae_max": max(eval_mae) if eval_mae else None,
            "experiment_ids": [str(r["experiment_id"]) for r in sample_rows],
        }
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = _resolve(args.out_dir, project_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = _resolve(args.config, project_root)
    cfg = _read_json(cfg_path)
    experiments = cfg.get("experiments")
    if not isinstance(experiments, list) or len(experiments) == 0:
        raise ValueError(f"Invalid or empty experiments in config: {cfg_path}")

    rows: list[dict[str, Any]] = []
    for item in experiments:
        if not isinstance(item, dict):
            raise TypeError("Each experiment entry must be a dict")
        exp_id = str(item.get("experiment_id", "")).strip()
        seed = item.get("seed")
        samples = item.get("samples")
        train_path_str = str(item.get("train_report_path", "")).strip()
        eval_path_str = str(item.get("eval_report_path", "")).strip()
        if not exp_id or not train_path_str or not eval_path_str:
            raise ValueError(f"Invalid experiment item: {item}")

        train_report_path = _resolve(train_path_str, project_root)
        eval_report_path = _resolve(eval_path_str, project_root)
        issues: list[str] = []

        if not train_report_path.exists():
            rows.append(
                {
                    "experiment_id": exp_id,
                    "seed": seed,
                    "samples": samples,
                    "train_initial_loss": None,
                    "train_final_loss": None,
                    "train_improve_ratio": None,
                    "eval_mse": None,
                    "eval_mae": None,
                    "eval_mse_xy": None,
                    "eval_mae_xy": None,
                    "num_steps": None,
                    "beta": None,
                    "train_metrics_ok": False,
                    "eval_metrics_ok": False,
                    "train_artifacts_ok": False,
                    "eval_artifacts_ok": False,
                    "overall_ok": False,
                    "issues": "missing_train_report_json",
                    "train_report_path": str(train_report_path),
                    "eval_report_path": str(eval_report_path),
                }
            )
            continue
        if not eval_report_path.exists():
            rows.append(
                {
                    "experiment_id": exp_id,
                    "seed": seed,
                    "samples": samples,
                    "train_initial_loss": None,
                    "train_final_loss": None,
                    "train_improve_ratio": None,
                    "eval_mse": None,
                    "eval_mae": None,
                    "eval_mse_xy": None,
                    "eval_mae_xy": None,
                    "num_steps": None,
                    "beta": None,
                    "train_metrics_ok": False,
                    "eval_metrics_ok": False,
                    "train_artifacts_ok": False,
                    "eval_artifacts_ok": False,
                    "overall_ok": False,
                    "issues": "missing_eval_report_json",
                    "train_report_path": str(train_report_path),
                    "eval_report_path": str(eval_report_path),
                }
            )
            continue

        train_report = _read_json(train_report_path)
        eval_report = _read_json(eval_report_path)

        if str(train_report.get("purpose", "")).strip() != "fit_check":
            issues.append("train_purpose_not_fit_check")
        if str(eval_report.get("purpose", "")).strip() != "evaluation":
            issues.append("eval_purpose_not_evaluation")

        train_metrics_ok, train_metric_errors = _check_train_metrics(train_report)
        eval_metrics_ok, eval_metric_errors = _check_eval_metrics(eval_report)
        train_artifacts_ok, train_artifact_errors = _check_train_artifacts(train_report_path, train_report, project_root)
        eval_artifacts_ok, eval_artifact_errors = _check_eval_artifacts(eval_report_path, eval_report, project_root)

        issues.extend(train_metric_errors)
        issues.extend(eval_metric_errors)
        issues.extend(train_artifact_errors)
        issues.extend(eval_artifact_errors)

        num_steps = eval_report.get("num_steps", train_report.get("num_steps"))
        beta = eval_report.get("beta", train_report.get("beta"))
        row = {
            "experiment_id": exp_id,
            "seed": int(seed) if isinstance(seed, int) else seed,
            "samples": int(samples) if isinstance(samples, int) else samples,
            "train_initial_loss": _safe_float(train_report.get("initial_loss")),
            "train_final_loss": _safe_float(train_report.get("final_loss")),
            "train_improve_ratio": _safe_float(train_report.get("improve_ratio")),
            "eval_mse": _safe_float(eval_report.get("mse")),
            "eval_mae": _safe_float(eval_report.get("mae")),
            "eval_mse_xy": _safe_float(eval_report.get("mse_xy")),
            "eval_mae_xy": _safe_float(eval_report.get("mae_xy")),
            "num_steps": int(num_steps) if isinstance(num_steps, int) else num_steps,
            "beta": _safe_float(beta),
            "train_metrics_ok": train_metrics_ok,
            "eval_metrics_ok": eval_metrics_ok,
            "train_artifacts_ok": train_artifacts_ok,
            "eval_artifacts_ok": eval_artifacts_ok,
            "overall_ok": train_metrics_ok and eval_metrics_ok and train_artifacts_ok and eval_artifacts_ok and len(issues) == 0,
            "issues": ";".join(issues) if issues else "",
            "train_report_path": str(train_report_path),
            "eval_report_path": str(eval_report_path),
        }
        rows.append(row)

    summary = {
        "phase": "phase3a_snn_summary",
        "config_path": str(cfg_path),
        "num_experiments": len(rows),
        "all_overall_ok": all(bool(r["overall_ok"]) for r in rows),
        "all_train_metrics_ok": all(bool(r["train_metrics_ok"]) for r in rows),
        "all_eval_metrics_ok": all(bool(r["eval_metrics_ok"]) for r in rows),
        "all_artifacts_ok": all(bool(r["train_artifacts_ok"]) and bool(r["eval_artifacts_ok"]) for r in rows),
        "train_eval_separation_ok": all(
            ("train_purpose_not_fit_check" not in str(r["issues"]))
            and ("eval_purpose_not_evaluation" not in str(r["issues"]))
            for r in rows
        ),
        "by_samples": _aggregate_by_samples(rows),
        "experiments": rows,
    }

    csv_path = out_dir / "phase3a_snn_summary.csv"
    json_path = out_dir / "phase3a_snn_summary.json"

    fieldnames = [
        "experiment_id",
        "seed",
        "samples",
        "num_steps",
        "beta",
        "train_initial_loss",
        "train_final_loss",
        "train_improve_ratio",
        "eval_mse",
        "eval_mae",
        "eval_mse_xy",
        "eval_mae_xy",
        "train_metrics_ok",
        "eval_metrics_ok",
        "train_artifacts_ok",
        "eval_artifacts_ok",
        "overall_ok",
        "issues",
        "train_report_path",
        "eval_report_path",
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
