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
from matplotlib.ticker import MaxNLocator


STAGES = ("far", "mid", "terminal")
COLORS = {"SNN-enhanced": "#2f5597", "CNN-enhanced": "#c55a11"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create publication-style Phase3 SNN/CNN visual localization comparison figures."
    )
    p.add_argument("--snn-report", type=str, required=True)
    p.add_argument("--cnn-report", type=str, required=True)
    p.add_argument("--snn-label", type=str, default="SNN-enhanced")
    p.add_argument("--cnn-label", type=str, default="CNN-enhanced")
    p.add_argument("--output-dir", type=str, default="outputs/reports/phase3_vision_publication")
    p.add_argument("--max-cdf-error-m", type=float, default=250.0)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _read_json(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_records_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise KeyError("Report missing artifacts dict.")
    path_text = artifacts.get("sample_errors_all_path")
    if not path_text:
        raise KeyError("Report missing artifacts.sample_errors_all_path.")
    path = Path(str(path_text)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing sample error records: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in sample_errors_all_path: {path}")
    return [dict(x) for x in data]


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("metrics", {})
    return value if isinstance(value, dict) else {}


def _metric(report: dict[str, Any], key: str) -> float:
    try:
        return float(_metrics(report).get(key))
    except Exception:
        return float("nan")


def _stage_values(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for rec in records:
        stage = str(rec.get("stage", "")).lower()
        if stage not in out:
            continue
        value = rec.get("world_error_m")
        if value is None:
            continue
        try:
            f = float(value)
        except Exception:
            continue
        if np.isfinite(f):
            out[stage].append(f)
    return {stage: np.asarray(vals, dtype=float) for stage, vals in out.items()}


def _summary(label: str, report: dict[str, Any], stage_map: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = [
        {"method": label, "group": "overall", "metric": "world_error_mean_m", "value": _metric(report, "world_error_mean_m")},
        {"method": label, "group": "overall", "metric": "world_error_p90_m", "value": _metric(report, "world_error_p90_m")},
        {"method": label, "group": "overall", "metric": "pixel_error_mean", "value": _metric(report, "pixel_error_mean")},
        {"method": label, "group": "overall", "metric": "pixel_error_p90", "value": _metric(report, "pixel_error_p90")},
        {"method": label, "group": "overall", "metric": "pred_on_land_ratio", "value": _metric(report, "pred_on_land_ratio")},
    ]
    for stage in STAGES:
        vals = stage_map.get(stage, np.asarray([], dtype=float))
        if vals.size:
            rows.extend(
                [
                    {"method": label, "group": "stage_world_error_m", "metric": f"{stage}_mean", "value": float(vals.mean())},
                    {"method": label, "group": "stage_world_error_m", "metric": f"{stage}_median", "value": float(np.median(vals))},
                    {"method": label, "group": "stage_world_error_m", "metric": f"{stage}_p90", "value": float(np.percentile(vals, 90))},
                    {"method": label, "group": "stage_world_error_m", "metric": f"{stage}_p99", "value": float(np.percentile(vals, 99))},
                ]
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.tick_params(axis="both", labelsize=9)


def _bar_with_points(
    ax: plt.Axes,
    *,
    stage_labels: list[str],
    snn_stage: dict[str, np.ndarray],
    cnn_stage: dict[str, np.ndarray],
    snn_label: str,
    cnn_label: str,
) -> None:
    x = np.arange(len(stage_labels), dtype=float)
    width = 0.34
    snn_mean = [float(snn_stage[s].mean()) for s in stage_labels]
    cnn_mean = [float(cnn_stage[s].mean()) for s in stage_labels]
    snn_p90 = [float(np.percentile(snn_stage[s], 90)) for s in stage_labels]
    cnn_p90 = [float(np.percentile(cnn_stage[s], 90)) for s in stage_labels]
    ax.bar(x - width / 2, snn_mean, width, color=COLORS.get(snn_label, "#2f5597"), label=snn_label)
    ax.bar(x + width / 2, cnn_mean, width, color=COLORS.get(cnn_label, "#c55a11"), label=cnn_label)
    ax.scatter(x - width / 2, snn_p90, marker="D", s=28, color="#1b2a41", label="P90")
    ax.scatter(x + width / 2, cnn_p90, marker="D", s=28, color="#1b2a41")
    for xi, mean in zip(x - width / 2, snn_mean):
        ax.text(xi, mean, f"{mean:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, mean in zip(x + width / 2, cnn_mean):
        ax.text(xi, mean, f"{mean:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in stage_labels])
    ax.set_ylabel("World localization error (m)")
    ax.set_title("(a) Stage-wise mean error with P90 markers", loc="left", fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    _style_axes(ax)


def _cdf(ax: plt.Axes, label: str, values: np.ndarray, *, color: str, max_error: float) -> None:
    vals = values[np.isfinite(values)]
    vals = np.sort(vals)
    if vals.size == 0:
        return
    y = np.arange(1, vals.size + 1, dtype=float) / float(vals.size)
    vals = np.clip(vals, 0.0, float(max_error))
    ax.plot(vals, y, linewidth=2.0, color=color, label=label)


def _boxplot(ax: plt.Axes, *, snn_stage: dict[str, np.ndarray], cnn_stage: dict[str, np.ndarray]) -> None:
    data: list[np.ndarray] = []
    positions: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    for idx, stage in enumerate(STAGES):
        base = float(idx + 1)
        data.extend([snn_stage[stage], cnn_stage[stage]])
        positions.extend([base - 0.16, base + 0.16])
        colors.extend([COLORS["SNN-enhanced"], COLORS["CNN-enhanced"]])
        labels.append(stage.capitalize())
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.24,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    ax.set_xticks(np.arange(1, len(STAGES) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("World localization error (m)")
    ax.set_title("(c) Stage-wise error distribution", loc="left", fontsize=11)
    _style_axes(ax)


def _diagnostic_panel(ax: plt.Axes, snn: dict[str, Any], cnn: dict[str, Any], snn_label: str, cnn_label: str) -> None:
    labels = ["Mean", "P90", "Land pred."]
    snn_vals = [
        _metric(snn, "world_error_mean_m"),
        _metric(snn, "world_error_p90_m"),
        _metric(snn, "pred_on_land_ratio") * 100.0,
    ]
    cnn_vals = [
        _metric(cnn, "world_error_mean_m"),
        _metric(cnn, "world_error_p90_m"),
        _metric(cnn, "pred_on_land_ratio") * 100.0,
    ]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    ax.bar(x - width / 2, snn_vals, width, color=COLORS.get(snn_label, "#2f5597"), label=snn_label)
    ax.bar(x + width / 2, cnn_vals, width, color=COLORS.get(cnn_label, "#c55a11"), label=cnn_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("m for errors; % for land predictions")
    ax.set_title("(d) Overall metrics", loc="left", fontsize=11)
    _style_axes(ax)


def main() -> None:
    args = parse_args()
    snn = _read_json(args.snn_report)
    cnn = _read_json(args.cnn_report)
    snn_records = _read_records_from_report(snn)
    cnn_records = _read_records_from_report(cnn)
    snn_stage = _stage_values(snn_records)
    cnn_stage = _stage_values(cnn_records)
    missing = [stage for stage in STAGES if snn_stage[stage].size == 0 or cnn_stage[stage].size == 0]
    if missing:
        raise ValueError(f"Missing stage records for: {missing}")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)
    axes = axes.flatten()
    _bar_with_points(
        axes[0],
        stage_labels=list(STAGES),
        snn_stage=snn_stage,
        cnn_stage=cnn_stage,
        snn_label=args.snn_label,
        cnn_label=args.cnn_label,
    )
    _cdf(
        axes[1],
        args.snn_label,
        np.concatenate([snn_stage[s] for s in STAGES]),
        color=COLORS.get(args.snn_label, "#2f5597"),
        max_error=float(args.max_cdf_error_m),
    )
    _cdf(
        axes[1],
        args.cnn_label,
        np.concatenate([cnn_stage[s] for s in STAGES]),
        color=COLORS.get(args.cnn_label, "#c55a11"),
        max_error=float(args.max_cdf_error_m),
    )
    axes[1].set_xlabel("World localization error (m)")
    axes[1].set_ylabel("Cumulative fraction")
    axes[1].set_title("(b) Error cumulative distribution", loc="left", fontsize=11)
    axes[1].set_xlim(0.0, float(args.max_cdf_error_m))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, loc="lower right")
    _style_axes(axes[1])
    _boxplot(axes[2], snn_stage=snn_stage, cnn_stage=cnn_stage)
    _diagnostic_panel(axes[3], snn, cnn, args.snn_label, args.cnn_label)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, loc="upper left")
    fig.suptitle("SNN vs. CNN Visual Localization on the Held-out Maritime Test Set", y=1.02)

    out_png = out_dir / "phase3_vision_world_error_publication.png"
    out_pdf = out_dir / "phase3_vision_world_error_publication.pdf"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    rows = _summary(args.snn_label, snn, snn_stage) + _summary(args.cnn_label, cnn, cnn_stage)
    _write_csv(out_dir / "phase3_vision_world_error_summary.csv", rows)
    context = {
        "snn_report": str(Path(args.snn_report).expanduser().resolve()),
        "cnn_report": str(Path(args.cnn_report).expanduser().resolve()),
        "figure_png": str(out_png),
        "figure_pdf": str(out_pdf),
        "stage_sample_counts": {
            "snn": {stage: int(snn_stage[stage].size) for stage in STAGES},
            "cnn": {stage: int(cnn_stage[stage].size) for stage in STAGES},
        },
        "note": "All stage panels use per-sample world_error_m from sample_errors_all.json.",
    }
    (out_dir / "phase3_vision_world_error_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure_png": str(out_png), "figure_pdf": str(out_pdf)}, indent=2))


if __name__ == "__main__":
    main()
