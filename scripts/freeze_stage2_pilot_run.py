from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--render-report", type=str, default=None)
    p.add_argument("--qc-report", type=str, default=None)
    p.add_argument("--temporal-report", type=str, default=None)
    p.add_argument("--fit-report", type=str, default=None)
    p.add_argument("--eval-report", type=str, default=None)
    p.add_argument("--config", type=str, default="configs/render_stage2_pilot_v1.yaml")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Default: outputs/stage2_pilot_baselines/freezes/<dataset_name>",
    )
    return p.parse_args()


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    dataset_name = dataset_root.name

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (Path("outputs") / "stage2_pilot_baselines" / "freezes" / dataset_name).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = dataset_root / "reports"
    render_report = Path(args.render_report).resolve() if args.render_report else (reports_dir / "smoke_render_report.json")
    qc_report = Path(args.qc_report).resolve() if args.qc_report else (reports_dir / "qc_summary.json")
    temporal_report = Path(args.temporal_report).resolve() if args.temporal_report else (reports_dir / "temporal_continuity_report.json")
    fit_report = Path(args.fit_report).resolve() if args.fit_report else Path("outputs/stage2_pilot_baselines/cnn_fit_v1/report.json").resolve()
    eval_report = Path(args.eval_report).resolve() if args.eval_report else Path("outputs/stage2_pilot_baselines/cnn_eval_v1/report.json").resolve()
    config_path = Path(args.config).resolve()

    rr = _load_json(render_report)
    qr = _load_json(qc_report)
    tr = _load_json(temporal_report)
    fr = _load_json(fit_report)
    er = _load_json(eval_report)

    summary = {
        "task": "freeze_stage2_pilot_run",
        "dataset_root": str(dataset_root),
        "dataset_name": dataset_name,
        "checks": {
            "render_report_exists": bool(rr),
            "qc_pass": bool(qr.get("pass", False)),
            "temporal_pass": bool(tr.get("pass", False)),
            "fit_train_improved": bool(fr.get("success_criteria", {}).get("train_final_lt_initial", False))
            and bool(fr.get("success_criteria", {}).get("train_improve_ratio_gt_0", False)),
            "fit_val_improved": bool(fr.get("success_criteria", {}).get("val_final_lt_initial", False))
            and bool(fr.get("success_criteria", {}).get("val_improve_ratio_gt_0", False)),
            "eval_split_test": str(er.get("dataset", {}).get("eval_split", "")) == "test",
        },
        "metrics_snapshot": {
            "qc": {
                "center_bias_ratio": qr.get("center_bias_ratio"),
                "land_overlap_violations": qr.get("land_overlap_violations"),
                "shore_overlap_violations": qr.get("shore_overlap_violations"),
                "obs_invalid_count": qr.get("obs_invalid_count"),
            },
            "temporal": tr.get("global", {}),
            "fit": fr.get("metrics", {}),
            "eval": er.get("metrics", {}),
        },
        "source_paths": {
            "render_report": str(render_report),
            "qc_report": str(qc_report),
            "temporal_report": str(temporal_report),
            "fit_report": str(fit_report),
            "eval_report": str(eval_report),
            "config": str(config_path),
        },
    }
    summary["freeze_pass"] = bool(all(summary["checks"].values()))

    copied = {
        "render_report": _copy_if_exists(render_report, out_dir / "reports" / "smoke_render_report.json"),
        "qc_report": _copy_if_exists(qc_report, out_dir / "reports" / "qc_summary.json"),
        "temporal_report": _copy_if_exists(temporal_report, out_dir / "reports" / "temporal_continuity_report.json"),
        "fit_report": _copy_if_exists(fit_report, out_dir / "reports" / "cnn_fit_report.json"),
        "eval_report": _copy_if_exists(eval_report, out_dir / "reports" / "cnn_eval_report.json"),
        "config": _copy_if_exists(config_path, out_dir / "config" / "render_stage2_pilot_v1.yaml"),
    }
    summary["copied_artifacts"] = copied

    out_report = out_dir / "freeze_report.json"
    out_report.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

