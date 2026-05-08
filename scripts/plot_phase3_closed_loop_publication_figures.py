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


METHOD_ORDER = ["Oracle-GT", "SNN no-KF", "SNN KF/raw", "SNN full-KF", "CNN KF/raw"]
COLORS = {
    "Oracle-GT": "#595959",
    "SNN no-KF": "#6c8ebf",
    "SNN KF/raw": "#2f5597",
    "SNN full-KF": "#8faadc",
    "CNN KF/raw": "#c55a11",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create publication-style Phase3 closed-loop ablation figures from final paper metrics."
    )
    p.add_argument(
        "--metrics-csv",
        type=str,
        default="configs/phase3_closed_loop_publication_metrics.csv",
        help="CSV with method,capture,total,valid_capture,valid_total,est_err_m,vision_err_m,hard_viol,safety_margin.",
    )
    p.add_argument("--output-dir", type=str, default="outputs/reports/phase3_closed_loop_publication")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _read_rows(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"Empty metrics CSV: {path}")
    by_method = {str(r.get("method", "")): r for r in rows}
    ordered = [by_method[m] for m in METHOD_ORDER if m in by_method]
    ordered.extend(r for r in rows if str(r.get("method", "")) not in METHOD_ORDER)
    return ordered


def _f(row: dict[str, Any], key: str) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def _i(row: dict[str, Any], key: str) -> int:
    return int(round(_f(row, key)))


def _style(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#dddddd", linewidth=0.7, alpha=0.85)
    ax.tick_params(labelsize=9)


def _annotate_bars(ax: plt.Axes, bars, *, fmt: str = "{:.1f}", dy: float = 0.0) -> None:
    for bar in bars:
        value = float(bar.get_height())
        if not np.isfinite(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows: list[dict[str, Any]] = []
    for r in rows:
        cap = f"{_i(r, 'capture')}/{_i(r, 'total')}"
        valid_cap = f"{_i(r, 'valid_capture')}/{_i(r, 'valid_total')}"
        vision = _f(r, "vision_err_m")
        csv_rows.append(
            {
                "Method": r["method"],
                "Capture": cap,
                "Valid Capture": valid_cap,
                "Est. Err. (m)": f"{_f(r, 'est_err_m'):.2f}",
                "Vision Err. (m)": "-" if not np.isfinite(vision) else f"{vision:.2f}",
                "Hard Viol.": str(_i(r, "hard_viol")),
                "Safety Margin": str(_i(r, "safety_margin")),
                "Role": r.get("role", ""),
            }
        )
    with (out_dir / "closed_loop_ablation_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "| Method | Capture | Valid Capture | Est. Err. (m) | Vision Err. (m) | Hard Viol. | Safety Margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in csv_rows:
        lines.append(
            f"| {row['Method']} | {row['Capture']} | {row['Valid Capture']} | "
            f"{row['Est. Err. (m)']} | {row['Vision Err. (m)']} | "
            f"{row['Hard Viol.']} | {row['Safety Margin']} |"
        )
    (out_dir / "closed_loop_ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_main_ablation(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> None:
    labels = [str(r["method"]) for r in rows]
    x = np.arange(len(rows), dtype=float)
    est = np.asarray([_f(r, "est_err_m") for r in rows], dtype=float)
    vision = np.asarray([_f(r, "vision_err_m") for r in rows], dtype=float)
    capture = np.asarray([_f(r, "capture") / max(_f(r, "total"), 1.0) * 100.0 for r in rows], dtype=float)
    valid_capture = np.asarray(
        [_f(r, "valid_capture") / max(_f(r, "valid_total"), 1.0) * 100.0 for r in rows],
        dtype=float,
    )
    safety = np.asarray([_f(r, "safety_margin") for r in rows], dtype=float)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.4), constrained_layout=True)
    axes = axes.ravel()

    bars = axes[0].bar(x, est, color=[COLORS.get(label, "#666666") for label in labels], width=0.62)
    axes[0].set_title("(a) Target-estimation error")
    axes[0].set_ylabel("Mean error (m)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylim(0.0, max(float(np.nanmax(est)) * 1.28, 1.0))
    _annotate_bars(axes[0], bars, fmt="{:.2f}", dy=max(float(np.nanmax(est)) * 0.02, 0.4))
    _style(axes[0])

    finite_mask = np.isfinite(vision)
    width = 0.34
    bars1 = axes[1].bar(x - width / 2, est, width, color="#4c78a8", label="Estimator")
    bars2 = axes[1].bar(x[finite_mask] + width / 2, vision[finite_mask], width, color="#f58518", label="Raw vision")
    axes[1].set_title("(b) Estimator vs. visual measurement")
    axes[1].set_ylabel("Mean error (m)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    both = np.concatenate([est[np.isfinite(est)], vision[np.isfinite(vision)]])
    axes[1].set_ylim(0.0, max(float(np.nanmax(both)) * 1.25, 1.0))
    axes[1].legend(frameon=False, loc="upper left")
    _annotate_bars(axes[1], bars1, fmt="{:.1f}", dy=max(float(np.nanmax(both)) * 0.02, 0.4))
    _annotate_bars(axes[1], bars2, fmt="{:.1f}", dy=max(float(np.nanmax(both)) * 0.02, 0.4))
    _style(axes[1])

    cap_width = 0.34
    cap_bars = axes[2].bar(
        x - cap_width / 2,
        capture,
        cap_width,
        color="#2f5597",
        label="Overall capture",
    )
    valid_bars = axes[2].bar(
        x + cap_width / 2,
        valid_capture,
        cap_width,
        color="#70ad47",
        label="Valid-scenario capture",
    )
    axes[2].set_title("(c) Capture performance")
    axes[2].set_ylabel("Capture rate (%)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=18, ha="right")
    axes[2].set_ylim(92.0, 101.5)
    axes[2].legend(frameon=False, loc="lower left", ncol=1)
    _style(axes[2])
    for bars_obj, counts_key, totals_key in [
        (cap_bars, "capture", "total"),
        (valid_bars, "valid_capture", "valid_total"),
    ]:
        for bar, row in zip(bars_obj, rows):
            value = float(bar.get_height())
            axes[2].text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.12,
                f"{_i(row, counts_key)}/{_i(row, totals_key)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    safety_bars = axes[3].bar(x, safety, color="#7f7f7f", width=0.62)
    hard = np.asarray([_f(r, "hard_viol") for r in rows], dtype=float)
    axes[3].scatter(x, hard, color="#c00000", marker="x", s=48, linewidths=1.7)
    axes[3].set_title("(d) Safety outcomes")
    axes[3].set_ylabel("Safety-margin incursion count")
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels, rotation=18, ha="right")
    axes[3].set_ylim(0.0, max(float(np.nanmax(safety)) * 1.18, 1.0))
    _annotate_bars(axes[3], safety_bars, fmt="{:.0f}", dy=max(float(np.nanmax(safety)) * 0.015, 5.0))
    for xi, value in zip(x, hard):
        axes[3].text(xi, value + max(float(np.nanmax(safety)) * 0.02, 10.0), f"{int(value)}", ha="center", va="bottom", fontsize=8, color="#c00000")
    axes[3].text(
        0.02,
        0.94,
        "Hard no-fly-zone violations: 0 for all methods",
        transform=axes[3].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
    )
    _style(axes[3])

    fig.suptitle("Closed-Loop Ablation Under Paired Dynamic-Target Episodes", y=1.04)
    fig.text(
        0.5,
        -0.01,
        "Capture is true-target capture over all 64 episodes; valid-scenario capture excludes common boundary failures.",
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.savefig(out_dir / "closed_loop_ablation_main.png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_dir / "closed_loop_ablation_main.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_kf_sensitivity(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> None:
    by_method = {str(r["method"]): r for r in rows}
    required = ["SNN no-KF", "SNN KF/raw", "SNN full-KF"]
    if any(k not in by_method for k in required):
        return
    no_kf = _f(by_method["SNN no-KF"], "est_err_m")
    kf_raw = _f(by_method["SNN KF/raw"], "est_err_m")
    full_kf = _f(by_method["SNN full-KF"], "est_err_m")
    delta = no_kf - kf_raw
    pct = delta / max(no_kf, 1.0e-12) * 100.0
    sensitivity = full_kf - kf_raw

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    labels = ["SNN no-KF", "SNN KF/raw", "SNN full-KF"]
    values = [no_kf, kf_raw, full_kf]
    x = np.arange(len(labels), dtype=float)
    bars = axes[0].bar(x, values, color=[COLORS.get(label, "#666666") for label in labels], width=0.58)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=14, ha="right")
    axes[0].set_ylabel("Mean estimation error (m)")
    axes[0].set_title("(a) SNN estimator ablations")
    axes[0].set_ylim(0.0, max(values) * 1.22)
    _annotate_bars(axes[0], bars, fmt="{:.2f}", dy=max(values) * 0.02)
    _style(axes[0])

    effect_labels = ["KF gain\n(no-KF - KF/raw)", "Terminal sensitivity\n(full-KF - KF/raw)"]
    effect_values = [delta, sensitivity]
    colors = ["#70ad47", "#7f7f7f"]
    bars2 = axes[1].bar(np.arange(2), effect_values, color=colors, width=0.52)
    axes[1].axhline(0.0, color="#333333", linewidth=0.9)
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(effect_labels)
    axes[1].set_ylabel("Error difference (m)")
    axes[1].set_title("(b) Effect size")
    pad = max(abs(float(np.nanmax(effect_values))), abs(float(np.nanmin(effect_values))), 0.5)
    axes[1].set_ylim(-pad * 1.45, pad * 1.45)
    for bar, value in zip(bars2, effect_values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.2f} m",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )
    axes[1].text(
        0.02,
        0.96,
        f"KF/raw reduces SNN mean error by {delta:.2f} m ({pct:.2f}%).",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
    )
    _style(axes[1])
    fig.suptitle("State-Estimation Ablation and Terminal Filtering Sensitivity", y=1.04)
    fig.savefig(out_dir / "closed_loop_kf_sensitivity.png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_dir / "closed_loop_kf_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_table(rows: list[dict[str, Any]], out_dir: Path, dpi: int) -> None:
    table_path = out_dir / "closed_loop_ablation_table.csv"
    with table_path.open("r", encoding="utf-8", newline="") as f:
        table_rows = list(csv.DictReader(f))
    columns = ["Method", "Capture", "Valid Capture", "Est. Err. (m)", "Vision Err. (m)", "Hard Viol.", "Safety Margin"]
    cell_text = [[row[col] for col in columns] for row in table_rows]

    fig, ax = plt.subplots(figsize=(11.8, 2.8), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#bfbfbf")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
        elif table_rows[r - 1]["Method"] == "SNN KF/raw":
            cell.set_facecolor("#eaf1fb")
    ax.set_title("Closed-Loop Ablation Summary", fontsize=13, pad=10)
    fig.savefig(out_dir / "closed_loop_ablation_table.png", dpi=int(dpi), bbox_inches="tight")
    fig.savefig(out_dir / "closed_loop_ablation_table.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.metrics_csv)
    out_dir = Path(args.output_dir).expanduser().resolve()
    _write_outputs(rows, out_dir)
    _plot_main_ablation(rows, out_dir, int(args.dpi))
    _plot_kf_sensitivity(rows, out_dir, int(args.dpi))
    _plot_table(rows, out_dir, int(args.dpi))
    context = {
        "metrics_csv": str(Path(args.metrics_csv).expanduser().resolve()),
        "outputs": {
            "main": str(out_dir / "closed_loop_ablation_main.png"),
            "kf_sensitivity": str(out_dir / "closed_loop_kf_sensitivity.png"),
            "table": str(out_dir / "closed_loop_ablation_table.png"),
        },
        "interpretation_note": (
            "These aggregate figures use the final paired closed-loop summary metrics. "
            "Per-episode scatter and representative trajectory curves require original summary.csv and trajectory.jsonl files."
        ),
    }
    (out_dir / "closed_loop_publication_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(context["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
