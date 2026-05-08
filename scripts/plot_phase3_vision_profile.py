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


COLORS = {
    "SNN-enhanced": "#2f5597",
    "CNN-enhanced": "#c55a11",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create publication-style plots for Phase3 vision profiling results.")
    p.add_argument("--profile-report", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="outputs/reports/phase3_vision_profile_figures")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _read_json(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing profile report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _result_by_label(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("profile report must contain results list")
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        label = str(row.get("label", "")).strip()
        if label:
            out[label] = row
    return out


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key))
    except Exception:
        return float("nan")


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.85)
    ax.tick_params(labelsize=9)


def _annotate(ax: plt.Axes, bars, *, fmt: str) -> None:
    for bar in bars:
        value = float(bar.get_height())
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _bar(ax: plt.Axes, labels: list[str], values: list[float], *, ylabel: str, title: str, log: bool = False) -> None:
    x = np.arange(len(labels), dtype=float)
    colors = [COLORS.get(label, "#666666") for label in labels]
    bars = ax.bar(x, values, color=colors, width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11)
    if log:
        ax.set_yscale("log")
    _annotate(ax, bars, fmt="{:.2f}" if max(values) < 1000 else "{:.0f}")
    _style(ax)


def _spike_panel(ax: plt.Axes, snn_row: dict[str, Any]) -> None:
    diag = snn_row.get("spike_diagnostics", {})
    if not isinstance(diag, dict) or not diag:
        ax.axis("off")
        ax.text(0.02, 0.95, "No SNN spike diagnostics available.", va="top", ha="left")
        return
    items = sorted((str(k).replace("spike_rate_", ""), float(v)) for k, v in diag.items())
    labels = [k.upper() for k, _ in items]
    values = [v * 100.0 for _, v in items]
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(x, values, color=COLORS["SNN-enhanced"], width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean spike rate (%)")
    ax.set_title("(d) SNN activation sparsity", loc="left", fontsize=11)
    _annotate(ax, bars, fmt="{:.1f}")
    _style(ax)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    report = _read_json(args.profile_report)
    by_label = _result_by_label(report)
    labels = [label for label in ("SNN-enhanced", "CNN-enhanced") if label in by_label]
    if len(labels) < 2:
        labels = list(by_label)
    if len(labels) < 2:
        raise ValueError("Need at least two profiled models.")

    rows = [by_label[label] for label in labels]
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

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.2), constrained_layout=True)
    axes = axes.flatten()
    _bar(
        axes[0],
        labels,
        [_f(row, "latency_ms_per_image_mean") for row in rows],
        ylabel="Latency per image (ms)",
        title="(a) GPU inference latency",
        log=True,
    )
    _bar(
        axes[1],
        labels,
        [_f(row, "throughput_images_per_s") for row in rows],
        ylabel="Throughput (images/s)",
        title="(b) GPU throughput",
        log=True,
    )
    _bar(
        axes[2],
        labels,
        [_f(row, "params") / 1000.0 for row in rows],
        ylabel="Parameters (K)",
        title="(c) Model size",
        log=False,
    )
    snn_row = by_label.get("SNN-enhanced", rows[0])
    _spike_panel(axes[3], snn_row)
    fig.suptitle("Computational Profile of Vision Localization Models", y=1.02)
    out_png = out_dir / "phase3_vision_profile.png"
    out_pdf = out_dir / "phase3_vision_profile.pdf"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    table_rows: list[dict[str, Any]] = []
    for row in rows:
        table_rows.append(
            {
                "label": row.get("label"),
                "model_type": row.get("model_type"),
                "arch": row.get("arch"),
                "num_steps": row.get("num_steps"),
                "params": row.get("params"),
                "latency_ms_per_image_mean": row.get("latency_ms_per_image_mean"),
                "throughput_images_per_s": row.get("throughput_images_per_s"),
                "latency_ms_per_batch_mean": row.get("latency_ms_per_batch_mean"),
                "latency_ms_per_batch_p90": row.get("latency_ms_per_batch_p90"),
            }
        )
    _write_csv(out_dir / "phase3_vision_profile_summary.csv", table_rows)
    context = {
        "profile_report": str(Path(args.profile_report).expanduser().resolve()),
        "figure_png": str(out_png),
        "figure_pdf": str(out_pdf),
        "interpretation_note": (
            "This profile measures conventional GPU/PyTorch latency. It should not be used to claim "
            "neuromorphic energy efficiency unless hardware energy is directly measured."
        ),
    }
    (out_dir / "phase3_vision_profile_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure_png": str(out_png), "figure_pdf": str(out_pdf)}, indent=2))


if __name__ == "__main__":
    main()
