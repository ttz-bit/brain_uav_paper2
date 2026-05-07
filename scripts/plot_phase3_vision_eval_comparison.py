from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAGES = ("far", "mid", "terminal")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Phase3 vision eval comparisons from SNN/CNN report.json files."
    )
    p.add_argument("--snn-report", type=str, required=True, help="SNN eval report.json, or a .txt file containing JSON.")
    p.add_argument("--cnn-report", type=str, required=True, help="CNN eval report.json, or a .txt file containing JSON.")
    p.add_argument("--snn-label", type=str, default="SNN-enhanced")
    p.add_argument("--cnn-label", type=str, default="CNN-enhanced")
    p.add_argument("--output-dir", type=str, default="outputs/reports/phase3_vision_eval_compare")
    p.add_argument("--title", type=str, default="Phase3 Formal Vision Eval Comparison")
    p.add_argument(
        "--stage-gsd-km-per-px",
        type=str,
        default="far=0.020,mid=0.010,terminal=0.005",
        help="Comma-separated stage GSD map used to estimate stage world error from stage pixel error.",
    )
    return p.parse_args()


def _read_report(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Report is not valid JSON: {path}") from exc


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("metrics", {})
    return value if isinstance(value, dict) else {}


def _metric(report: dict[str, Any], key: str) -> float:
    value = _metrics(report).get(key)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _nested_metric(report: dict[str, Any], key: str) -> dict[str, float]:
    value = _metrics(report).get(key, {})
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in value.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _parse_stage_gsd(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --stage-gsd-km-per-px item: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    for stage in STAGES:
        if stage not in out:
            raise ValueError(f"Missing GSD for stage={stage}")
    return out


def _model_type(report: dict[str, Any]) -> str:
    cfg = report.get("eval_config", {})
    if isinstance(cfg, dict):
        model_type = str(cfg.get("checkpoint_model_type", "")).strip()
        if model_type:
            return model_type
    return "unknown"


def _common_context(snn: dict[str, Any], cnn: dict[str, Any]) -> dict[str, Any]:
    snn_dataset = snn.get("dataset", {}) if isinstance(snn.get("dataset", {}), dict) else {}
    cnn_dataset = cnn.get("dataset", {}) if isinstance(cnn.get("dataset", {}), dict) else {}
    snn_cfg = snn.get("eval_config", {}) if isinstance(snn.get("eval_config", {}), dict) else {}
    cnn_cfg = cnn.get("eval_config", {}) if isinstance(cnn.get("eval_config", {}), dict) else {}
    return {
        "same_dataset_root": snn_dataset.get("root") == cnn_dataset.get("root"),
        "same_eval_split": snn_dataset.get("eval_split") == cnn_dataset.get("eval_split"),
        "same_num_eval": snn_dataset.get("num_eval") == cnn_dataset.get("num_eval"),
        "same_decode_method": snn_cfg.get("decode_method") == cnn_cfg.get("decode_method"),
        "same_water_logit_constraint": snn_cfg.get("water_logit_constraint") == cnn_cfg.get("water_logit_constraint"),
        "snn_model_type": _model_type(snn),
        "cnn_model_type": _model_type(cnn),
        "dataset_root": snn_dataset.get("root"),
        "eval_split": snn_dataset.get("eval_split"),
        "num_eval": snn_dataset.get("num_eval"),
    }


def _annotate_bars(ax: plt.Axes, bars, *, fmt: str = "{:.2f}") -> None:
    for bar in bars:
        h = float(bar.get_height())
        if not np.isfinite(h):
            continue
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _bar_pair(
    ax: plt.Axes,
    labels: list[str],
    snn_values: list[float],
    cnn_values: list[float],
    snn_label: str,
    cnn_label: str,
    *,
    ylabel: str,
    title: str,
    percent: bool = False,
) -> None:
    x = np.arange(len(labels), dtype=float)
    width = 0.36
    snn_bars = ax.bar(x - width / 2.0, snn_values, width, label=snn_label, color="#3465a4")
    cnn_bars = ax.bar(x + width / 2.0, cnn_values, width, label=cnn_label, color="#cc6b32")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    if percent:
        ax.set_ylim(0.0, max(1.0, float(np.nanmax([*snn_values, *cnn_values])) * 1.20))
        _annotate_bars(ax, snn_bars, fmt="{:.1f}")
        _annotate_bars(ax, cnn_bars, fmt="{:.1f}")
    else:
        ymax = float(np.nanmax([v for v in [*snn_values, *cnn_values] if np.isfinite(v)] or [1.0]))
        ax.set_ylim(0.0, ymax * 1.18 if ymax > 0.0 else 1.0)
        _annotate_bars(ax, snn_bars)
        _annotate_bars(ax, cnn_bars)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(label: str, report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "pixel_error_mean",
        "pixel_error_p90",
        "world_error_mean_m",
        "world_error_p90_m",
        "center_baseline_improve_ratio",
        "offcenter_center_baseline_improve_ratio",
        "pred_on_land_ratio",
    ):
        rows.append({"method": label, "metric_group": "overall", "metric": key, "value": _metric(report, key)})
    pred_stats = _metrics(report).get("prediction_stats", {})
    if isinstance(pred_stats, dict) and "rounded_unique_pred_xy" in pred_stats:
        rows.append(
            {
                "method": label,
                "metric_group": "diagnostic",
                "metric": "rounded_unique_pred_xy",
                "value": float(pred_stats["rounded_unique_pred_xy"]),
            }
        )
    for stage, value in _nested_metric(report, "stage_pixel_error_mean").items():
        rows.append({"method": label, "metric_group": "stage_pixel_error_mean", "metric": stage, "value": value})
    for bg, value in _nested_metric(report, "background_pixel_error_mean").items():
        rows.append({"method": label, "metric_group": "background_pixel_error_mean", "metric": bg, "value": value})
    return rows


def _write_markdown(
    path: Path,
    *,
    title: str,
    snn_label: str,
    cnn_label: str,
    snn: dict[str, Any],
    cnn: dict[str, Any],
    context: dict[str, Any],
    figure_name: str,
    stage_gsd: dict[str, float],
) -> None:
    snn_stage = _nested_metric(snn, "stage_pixel_error_mean")
    cnn_stage = _nested_metric(cnn, "stage_pixel_error_mean")
    lines = [
        f"# {title}",
        "",
        f"![comparison]({figure_name})",
        "",
        "## Protocol Check",
        "",
        f"- same_dataset_root: {context['same_dataset_root']}",
        f"- same_eval_split: {context['same_eval_split']}",
        f"- same_num_eval: {context['same_num_eval']}",
        f"- same_decode_method: {context['same_decode_method']}",
        f"- same_water_logit_constraint: {context['same_water_logit_constraint']}",
        "",
        "## Key Metrics",
        "",
        "| method | pixel mean | pixel p90 | world mean m | world p90 m | land ratio |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {snn_label} | {_metric(snn, 'pixel_error_mean'):.4f} | {_metric(snn, 'pixel_error_p90'):.4f} | "
            f"{_metric(snn, 'world_error_mean_m'):.4f} | {_metric(snn, 'world_error_p90_m'):.4f} | "
            f"{_metric(snn, 'pred_on_land_ratio'):.6f} |"
        ),
        (
            f"| {cnn_label} | {_metric(cnn, 'pixel_error_mean'):.4f} | {_metric(cnn, 'pixel_error_p90'):.4f} | "
            f"{_metric(cnn, 'world_error_mean_m'):.4f} | {_metric(cnn, 'world_error_p90_m'):.4f} | "
            f"{_metric(cnn, 'pred_on_land_ratio'):.6f} |"
        ),
        "",
        "## Stage Pixel Error Mean",
        "",
        "| stage | SNN | CNN | CNN/SNN |",
        "|---|---:|---:|---:|",
    ]
    for stage in STAGES:
        s = snn_stage.get(stage, float("nan"))
        c = cnn_stage.get(stage, float("nan"))
        ratio = c / s if np.isfinite(s) and abs(s) > 1e-12 else float("nan")
        lines.append(f"| {stage} | {s:.4f} | {c:.4f} | {ratio:.2f} |")
    lines.extend(
        [
            "",
            "## Estimated Stage World Error Mean",
            "",
            "| stage | SNN m | CNN m |",
            "|---|---:|---:|",
        ]
    )
    for stage in STAGES:
        s = snn_stage.get(stage, float("nan")) * stage_gsd[stage] * 1000.0
        c = cnn_stage.get(stage, float("nan")) * stage_gsd[stage] * 1000.0
        lines.append(f"| {stage} | {s:.2f} | {c:.2f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    snn = _read_report(args.snn_report)
    cnn = _read_report(args.cnn_report)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    context = _common_context(snn, cnn)
    labels = [args.snn_label, args.cnn_label]

    stage_labels = list(STAGES)
    stage_gsd = _parse_stage_gsd(args.stage_gsd_km_per_px)
    snn_stage = _nested_metric(snn, "stage_pixel_error_mean")
    cnn_stage = _nested_metric(cnn, "stage_pixel_error_mean")

    backgrounds = sorted(set(_nested_metric(snn, "background_pixel_error_mean")) | set(_nested_metric(cnn, "background_pixel_error_mean")))
    snn_bg = _nested_metric(snn, "background_pixel_error_mean")
    cnn_bg = _nested_metric(cnn, "background_pixel_error_mean")

    pred_stats_snn = _metrics(snn).get("prediction_stats", {})
    pred_stats_cnn = _metrics(cnn).get("prediction_stats", {})
    if not isinstance(pred_stats_snn, dict):
        pred_stats_snn = {}
    if not isinstance(pred_stats_cnn, dict):
        pred_stats_cnn = {}

    snn_stage_px = [snn_stage.get(stage, float("nan")) for stage in stage_labels]
    cnn_stage_px = [cnn_stage.get(stage, float("nan")) for stage in stage_labels]
    snn_stage_world_m = [snn_stage.get(stage, float("nan")) * stage_gsd[stage] * 1000.0 for stage in stage_labels]
    cnn_stage_world_m = [cnn_stage.get(stage, float("nan")) * stage_gsd[stage] * 1000.0 for stage in stage_labels]

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.8), constrained_layout=True)
    axes = axes.flatten()

    _bar_pair(
        axes[0],
        stage_labels,
        snn_stage_px,
        cnn_stage_px,
        args.snn_label,
        args.cnn_label,
        ylabel="pixel error (px)",
        title="Stage Mean Pixel Error",
    )

    _bar_pair(
        axes[1],
        stage_labels,
        snn_stage_world_m,
        cnn_stage_world_m,
        args.snn_label,
        args.cnn_label,
        ylabel="estimated world error (m)",
        title="Stage Mean World Error",
    )

    _bar_pair(
        axes[2],
        ["mean", "p90"],
        [_metric(snn, "pixel_error_mean"), _metric(snn, "pixel_error_p90")],
        [_metric(cnn, "pixel_error_mean"), _metric(cnn, "pixel_error_p90")],
        args.snn_label,
        args.cnn_label,
        ylabel="pixel error (px)",
        title="Overall Pixel Error",
    )

    _bar_pair(
        axes[3],
        ["mean", "p90"],
        [_metric(snn, "world_error_mean_m"), _metric(snn, "world_error_p90_m")],
        [_metric(cnn, "world_error_mean_m"), _metric(cnn, "world_error_p90_m")],
        args.snn_label,
        args.cnn_label,
        ylabel="world error (m)",
        title="World-Space Error",
    )

    _bar_pair(
        axes[4],
        backgrounds,
        [snn_bg.get(bg, float("nan")) for bg in backgrounds],
        [cnn_bg.get(bg, float("nan")) for bg in backgrounds],
        args.snn_label,
        args.cnn_label,
        ylabel="pixel error (px)",
        title="Background Mean Pixel Error",
    )

    _bar_pair(
        axes[5],
        ["center improve", "offcenter improve", "pred on land"],
        [
            _metric(snn, "center_baseline_improve_ratio") * 100.0,
            _metric(snn, "offcenter_center_baseline_improve_ratio") * 100.0,
            _metric(snn, "pred_on_land_ratio") * 100.0,
        ],
        [
            _metric(cnn, "center_baseline_improve_ratio") * 100.0,
            _metric(cnn, "offcenter_center_baseline_improve_ratio") * 100.0,
            _metric(cnn, "pred_on_land_ratio") * 100.0,
        ],
        args.snn_label,
        args.cnn_label,
        ylabel="percent (%)",
        title="Quality Diagnostics",
        percent=True,
    )

    _bar_pair(
        axes[6],
        ["unique pred xy"],
        [float(pred_stats_snn.get("rounded_unique_pred_xy", float("nan")))],
        [float(pred_stats_cnn.get("rounded_unique_pred_xy", float("nan")))],
        args.snn_label,
        args.cnn_label,
        ylabel="count",
        title="Prediction Diversity",
    )

    axes[7].axis("off")
    protocol_lines = [
        "Protocol Check",
        f"same dataset: {context['same_dataset_root']}",
        f"same split: {context['same_eval_split']}",
        f"same num eval: {context['same_num_eval']}",
        f"same decode: {context['same_decode_method']}",
        f"same water constraint: {context['same_water_logit_constraint']}",
        f"eval split: {context.get('eval_split')}",
        f"num eval: {context.get('num_eval')}",
    ]
    axes[7].text(0.02, 0.98, "\n".join(protocol_lines), va="top", ha="left", fontsize=12)

    axes[0].legend(loc="upper left")
    fig.suptitle(args.title)
    out_png = out_dir / "phase3_vision_eval_comparison.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    rows = _summary_rows(args.snn_label, snn) + _summary_rows(args.cnn_label, cnn)
    _write_csv(out_dir / "phase3_vision_eval_comparison.csv", rows)
    (out_dir / "phase3_vision_eval_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(
        out_dir / "phase3_vision_eval_comparison.md",
        title=args.title,
        snn_label=args.snn_label,
        cnn_label=args.cnn_label,
        snn=snn,
        cnn=cnn,
        context=context,
        figure_name=out_png.name,
        stage_gsd=stage_gsd,
    )
    print(json.dumps({"figure": str(out_png), "csv": str(out_dir / "phase3_vision_eval_comparison.csv")}, indent=2))


if __name__ == "__main__":
    main()
