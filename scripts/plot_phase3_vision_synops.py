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
    "Dense MACs": "#7f7f7f",
    "Proxy SynOps": "#4472c4",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create publication-style figures from Phase3 vision MACs/SynOps proxy analysis."
    )
    p.add_argument("--synops-report", type=str, required=True)
    p.add_argument("--profile-report", type=str, default="", help="Optional GPU latency profile report.")
    p.add_argument("--output-dir", type=str, default="outputs/reports/phase3_vision_synops_figures")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _read_json(path_text: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _results(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = report.get("results", {})
    if not isinstance(value, dict):
        raise ValueError("Report must contain results dict.")
    return {str(k): dict(v) for k, v in value.items() if isinstance(v, dict)}


def _profile_by_label(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not report:
        return {}
    rows = report.get("results", [])
    if not isinstance(rows, list):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("label"):
            out[str(row["label"])] = row
    return out


def _f(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key))
    except Exception:
        return float("nan")


def _billions(value: float) -> float:
    return float(value) / 1.0e9


def _style(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#dddddd", linewidth=0.7, alpha=0.85)
    ax.tick_params(labelsize=9)


def _annotate(ax: plt.Axes, bars, *, fmt: str = "{:.2f}") -> None:
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


def _ops_panel(ax: plt.Axes, results: dict[str, dict[str, Any]]) -> None:
    labels = ["SNN-enhanced", "CNN-enhanced"]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    dense = [_billions(_f(results[label], "dense_macs_per_image")) for label in labels]
    proxy = [_billions(_f(results[label], "proxy_synops_per_image")) for label in labels]
    bars1 = ax.bar(x - width / 2, dense, width, color=COLORS["Dense MACs"], label="Dense MACs")
    bars2 = ax.bar(x + width / 2, proxy, width, color=COLORS["Proxy SynOps"], label="Proxy SynOps")
    ax.set_xticks(x)
    ax.set_xticklabels(["SNN", "CNN"])
    ax.set_ylabel("Operations per image (G)")
    ax.set_yscale("log")
    ax.set_title("(a) Dense MACs and spike-gated SynOps proxy", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    _annotate(ax, bars1, fmt="{:.2f}")
    _annotate(ax, bars2, fmt="{:.2f}")
    _style(ax)


def _ratio_panel(ax: plt.Axes, results: dict[str, dict[str, Any]]) -> None:
    labels = ["SNN proxy / SNN dense", "SNN proxy / CNN dense", "CNN proxy / CNN dense"]
    snn = results["SNN-enhanced"]
    cnn = results["CNN-enhanced"]
    values = [
        _f(snn, "proxy_synops_per_image") / max(_f(snn, "dense_macs_per_image"), 1.0e-12),
        _f(snn, "proxy_synops_per_image") / max(_f(cnn, "dense_macs_per_image"), 1.0e-12),
        _f(cnn, "proxy_synops_per_image") / max(_f(cnn, "dense_macs_per_image"), 1.0e-12),
    ]
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(x, [v * 100.0 for v in values], color=["#4472c4", "#70ad47", "#c55a11"], width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=16, ha="right")
    ax.set_ylabel("Ratio (%)")
    ax.set_title("(b) Operation proxy ratios", loc="left", fontsize=11)
    _annotate(ax, bars, fmt="{:.1f}")
    _style(ax)


def _snn_layer_panel(ax: plt.Axes, results: dict[str, dict[str, Any]]) -> None:
    rows = list(results["SNN-enhanced"].get("layer_rows", []))
    rows = [dict(r) for r in rows if isinstance(r, dict)]
    order = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "skip2", "heatmap_head.0", "heatmap_head.2"]
    row_by_layer = {str(r.get("layer")): r for r in rows}
    labels = [name for name in order if name in row_by_layer]
    activity = [_f(row_by_layer[name], "mean_input_activity") * 100.0 for name in labels]
    x = np.arange(len(labels), dtype=float)
    bars = ax.bar(x, activity, color="#2f5597", width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Input activity (%)")
    ax.set_title("(c) SNN presynaptic activity by layer", loc="left", fontsize=11)
    _annotate(ax, bars, fmt="{:.1f}")
    _style(ax)


def _layer_ops_panel(ax: plt.Axes, results: dict[str, dict[str, Any]]) -> None:
    rows = list(results["SNN-enhanced"].get("layer_rows", []))
    rows = [dict(r) for r in rows if isinstance(r, dict)]
    rows.sort(key=lambda r: _f(r, "proxy_synops_per_image"), reverse=True)
    top = rows[:7]
    labels = [str(r.get("layer")) for r in top]
    values = [_billions(_f(r, "proxy_synops_per_image")) for r in top]
    y = np.arange(len(labels), dtype=float)
    bars = ax.barh(y, values, color="#4472c4")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Proxy SynOps per image (G)")
    ax.set_title("(d) SNN proxy SynOps contributors", loc="left", fontsize=11)
    for bar in bars:
        value = float(bar.get_width())
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", ha="left", fontsize=8)
    _style(ax, grid_axis="x")


def _write_csv(path: Path, results: dict[str, dict[str, Any]], profile: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for label, row in results.items():
        p = profile.get(label, {})
        rows.append(
            {
                "label": label,
                "dense_macs_per_image": row.get("dense_macs_per_image"),
                "proxy_synops_per_image": row.get("proxy_synops_per_image"),
                "proxy_synops_to_dense_macs_ratio": row.get("proxy_synops_to_dense_macs_ratio"),
                "latency_ms_per_image_mean": p.get("latency_ms_per_image_mean"),
                "throughput_images_per_s": p.get("throughput_images_per_s"),
                "num_steps": row.get("num_steps"),
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    synops_report = _read_json(args.synops_report)
    profile_report = _read_json(args.profile_report) if args.profile_report else None
    results = _results(synops_report)
    missing = [label for label in ("SNN-enhanced", "CNN-enhanced") if label not in results]
    if missing:
        raise ValueError(f"Missing required results: {missing}")
    profile = _profile_by_label(profile_report)
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
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), constrained_layout=True)
    axes = axes.flatten()
    _ops_panel(axes[0], results)
    _ratio_panel(axes[1], results)
    _snn_layer_panel(axes[2], results)
    _layer_ops_panel(axes[3], results)
    fig.suptitle("MACs and Spike-Gated SynOps Proxy Analysis", y=1.02)
    out_png = out_dir / "phase3_vision_synops.png"
    out_pdf = out_dir / "phase3_vision_synops.pdf"
    fig.savefig(out_png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    _write_csv(out_dir / "phase3_vision_synops_summary.csv", results, profile)
    context = {
        "synops_report": str(Path(args.synops_report).expanduser().resolve()),
        "profile_report": str(Path(args.profile_report).expanduser().resolve()) if args.profile_report else "",
        "figure_png": str(out_png),
        "figure_pdf": str(out_pdf),
        "caveat": (
            "SynOps is an operation-count proxy based on observed spike activity. It is not measured hardware energy."
        ),
    }
    (out_dir / "phase3_vision_synops_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"figure_png": str(out_png), "figure_pdf": str(out_pdf)}, indent=2))


if __name__ == "__main__":
    main()
