from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import Circle, Rectangle


STAGE_TITLES = {
    "far": "Far stage",
    "mid": "Mid stage",
    "terminal": "Terminal stage",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a publication-style construction figure for the Phase3 task-oriented maritime dataset."
    )
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--split", type=str, default="test")
    p.add_argument(
        "--sequence-id",
        type=str,
        default="",
        help="Optional sequence id. When provided, far/mid/terminal panels are selected from this sequence.",
    )
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Omit the figure-level title for compact paper layouts.",
    )
    p.add_argument("--output-dir", type=str, default="outputs/reports/phase3_dataset_construction")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _load_manifest(dataset_root: Path) -> list[dict[str, Any]]:
    path = dataset_root / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    return rows


def _resolve_path(path_text: str, *, project_root: Path, dataset_root: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute() and path.exists():
        return path
    candidates = [
        project_root / path_text,
        dataset_root / path_text,
        dataset_root / path.name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


def _read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _target_on_checkerboard(path: Path, size: int = 256) -> Image.Image:
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    tile = 16
    pix = canvas.load()
    for y in range(size):
        for x in range(size):
            shade = 235 if ((x // tile) + (y // tile)) % 2 == 0 else 210
            pix[x, y] = (shade, shade, shade, 255)
    scale = min(size * 0.76 / max(1, rgba.width), size * 0.48 / max(1, rgba.height))
    new_size = (max(1, int(rgba.width * scale)), max(1, int(rgba.height * scale)))
    resized = rgba.resize(new_size, Image.Resampling.LANCZOS)
    xy = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.alpha_composite(resized, xy)
    return canvas.convert("RGB")


def _crop_background(row: dict[str, Any], *, project_root: Path, dataset_root: Path) -> Image.Image:
    meta = row.get("meta", {})
    bg_path_text = str(meta.get("background_path", ""))
    if bg_path_text:
        bg_path = _resolve_path(bg_path_text, project_root=project_root, dataset_root=dataset_root)
        if bg_path.exists():
            bg = _read_rgb(bg_path)
            crop_box = meta.get("crop_box_bg_xyxy")
            if isinstance(crop_box, list) and len(crop_box) == 4:
                x1, y1, x2, y2 = [float(v) for v in crop_box]
                crop = bg.crop((x1, y1, x2, y2))
                return crop.resize((256, 256), Image.Resampling.LANCZOS)
            return bg.resize((256, 256), Image.Resampling.LANCZOS)
    image_path = _resolve_path(str(row["image_path"]), project_root=project_root, dataset_root=dataset_root)
    return _read_rgb(image_path)


def _score_row(row: dict[str, Any], stage: str) -> float:
    cx, cy = [float(v) for v in row.get("target_center_px", [128.0, 128.0])]
    bbox = row.get("bbox_xywh", [0, 0, 0, 0])
    bw, bh = float(bbox[2]), float(bbox[3])
    center_score = -float(np.hypot(cx - 128.0, cy - 128.0))
    size_score = min(bw, 28.0) + min(bh, 18.0)
    category_bonus = 3.0 if str(row.get("background_category", "")) in {"coastal", "island_complex"} else 0.0
    water_bonus = 5.0 if bool(row.get("meta", {}).get("target_on_water", False)) else 0.0
    stage_bonus = {"far": 0.0, "mid": 2.0, "terminal": 4.0}.get(stage, 0.0)
    return center_score + size_score + category_bonus + water_bonus + stage_bonus


def _valid_rows(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    return [
        r
        for r in rows
        if str(r.get("split")) == split
        and bool(r.get("obs_valid", True))
        and int(r.get("meta", {}).get("distractor_count", r.get("distractor_count", 0))) == 0
        and float(r.get("land_overlap_ratio", 0.0)) <= 1.0e-9
    ]


def _select_stage(rows: list[dict[str, Any]], *, split: str, stage: str) -> dict[str, Any]:
    candidates = [r for r in _valid_rows(rows, split=split) if str(r.get("stage")) == stage]
    if not candidates:
        candidates = [r for r in rows if str(r.get("stage")) == stage]
    if not candidates:
        raise ValueError(f"No manifest row found for stage={stage}")
    return max(candidates, key=lambda r: _score_row(r, stage))


def _select_stage_triplet(
    rows: list[dict[str, Any]],
    *,
    split: str,
    sequence_id: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    usable = _valid_rows(rows, split=split)
    if sequence_id:
        seq_rows = [r for r in usable if str(r.get("sequence_id")) == sequence_id]
        if not seq_rows:
            raise ValueError(f"No valid rows for sequence_id={sequence_id!r} in split={split!r}")
    else:
        by_seq: dict[str, list[dict[str, Any]]] = {}
        for row in usable:
            by_seq.setdefault(str(row.get("sequence_id", "")), []).append(row)
        complete: list[tuple[float, str, list[dict[str, Any]]]] = []
        for seq, seq_rows_candidate in by_seq.items():
            stages = {str(r.get("stage")) for r in seq_rows_candidate}
            if {"far", "mid", "terminal"}.issubset(stages):
                stage_best = [
                    max([r for r in seq_rows_candidate if str(r.get("stage")) == st], key=lambda r: _score_row(r, st))
                    for st in ("far", "mid", "terminal")
                ]
                score = sum(_score_row(r, str(r.get("stage"))) for r in stage_best)
                complete.append((float(score), seq, seq_rows_candidate))
        if complete:
            complete.sort(reverse=True, key=lambda item: item[0])
            seq_rows = complete[0][2]
        else:
            return (
                _select_stage(rows, split=split, stage="far"),
                _select_stage(rows, split=split, stage="mid"),
                _select_stage(rows, split=split, stage="terminal"),
            )
    selected = []
    for stage in ("far", "mid", "terminal"):
        candidates = [r for r in seq_rows if str(r.get("stage")) == stage]
        if not candidates:
            raise ValueError(f"Sequence {sequence_id!r} does not contain stage={stage!r}")
        selected.append(max(candidates, key=lambda r: _score_row(r, stage)))
    return selected[0], selected[1], selected[2]


def _draw_label(ax: plt.Axes, row: dict[str, Any], *, color: str = "#c00000", show_text: bool = True) -> None:
    cx, cy = [float(v) for v in row["target_center_px"]]
    x, y, w, h = [float(v) for v in row["bbox_xywh"]]
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.1))
    ax.add_patch(Circle((cx, cy), radius=2.5, facecolor=color, edgecolor="white", linewidth=0.6, zorder=4))
    if show_text:
        ax.text(
            0.03,
            0.95,
            rf"$p_t=({cx:.1f},{cy:.1f})$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2},
        )


def _imshow(ax: plt.Axes, image: Image.Image, title: str) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("#888888")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(dataset_root)
    far, mid, terminal = _select_stage_triplet(rows, split=str(args.split), sequence_id=str(args.sequence_id))
    flow_row = mid

    rendered_path = _resolve_path(str(flow_row["image_path"]), project_root=project_root, dataset_root=dataset_root)
    target_path = _resolve_path(
        str(flow_row.get("meta", {}).get("target_asset_path", "")),
        project_root=project_root,
        dataset_root=dataset_root,
    )
    background = _crop_background(flow_row, project_root=project_root, dataset_root=dataset_root)
    target = _target_on_checkerboard(target_path) if target_path.exists() else _read_rgb(rendered_path)
    rendered = _read_rgb(rendered_path)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.55), constrained_layout=True)
    axes = axes.ravel()

    _imshow(axes[0], background, "(a) Maritime background crop")
    axes[0].text(
        0.03,
        0.95,
        str(flow_row.get("background_category", "background")).replace("_", " "),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.2},
    )

    _imshow(axes[1], target, "(b) Target crop and mask")
    axes[1].text(
        0.03,
        0.95,
        "alpha PNG target",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=7.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.2},
    )

    _imshow(axes[2], rendered, "(c) Copy-paste composition")
    _draw_label(axes[2], flow_row, show_text=False)
    axes[2].annotate(
        "",
        xy=tuple(float(v) for v in flow_row["target_center_px"]),
        xytext=(35, 35),
        arrowprops={"arrowstyle": "->", "color": "#c00000", "lw": 1.2, "linestyle": "--"},
    )

    _imshow(axes[3], rendered, "(d) Pixel annotation")
    _draw_label(axes[3], flow_row)

    for ax, row, letter in zip(axes[4:7], [far, mid, terminal], ["(e)", "(f)", "(g)"]):
        img_path = _resolve_path(str(row["image_path"]), project_root=project_root, dataset_root=dataset_root)
        img = _read_rgb(img_path)
        stage = str(row.get("stage", ""))
        _imshow(ax, img, f"{letter} {STAGE_TITLES.get(stage, stage.title())}")
        _draw_label(ax, row, show_text=False)
        scale = float(row.get("scale_px", row.get("target_length_px", 0.0)))
        gsd = float(row.get("gsd_m_per_px", 0.0))
        rng = float(row.get("meta", {}).get("range_xy_km", float("nan")))
        ax.text(
            0.03,
            0.95,
            f"GSD {gsd:.1f} m/px\nrange {rng:.0f} km\nlength {scale:.0f} px",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 1.2},
        )

    axes[7].axis("off")
    axes[7].set_title("(h) Pixel-to-world label", fontsize=10, pad=6)
    axes[7].text(
        0.5,
        0.72,
        r"$p_t=(u_t,v_t)$",
        ha="center",
        va="center",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eaf2fb", "edgecolor": "#2f5597"},
    )
    axes[7].annotate(
        "",
        xy=(0.5, 0.50),
        xytext=(0.5, 0.64),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
    )
    axes[7].text(
        0.5,
        0.42,
        "UAV pose +\ncamera geometry",
        ha="center",
        va="center",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "#f7f7f7", "edgecolor": "#999999"},
    )
    axes[7].annotate(
        "",
        xy=(0.5, 0.22),
        xytext=(0.5, 0.34),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#333333"},
    )
    axes[7].text(
        0.5,
        0.13,
        r"$z_t=(x_t,y_t)$",
        ha="center",
        va="center",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eaf7ea", "edgecolor": "#548235"},
    )

    if not bool(args.no_title):
        fig.suptitle("Task-oriented Maritime Dataset Construction and Stage-aware Samples", y=1.015)
    fig.savefig(out_dir / "phase3_dataset_construction.png", dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(out_dir / "phase3_dataset_construction.pdf", bbox_inches="tight")

    selection = {
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "flow_sample": {
            "image_path": str(rendered_path),
            "target_asset_path": str(target_path),
            "stage": str(flow_row.get("stage")),
            "sequence_id": str(flow_row.get("sequence_id")),
            "frame_id": str(flow_row.get("frame_id")),
        },
        "stage_samples": [
            {
                "stage": str(r.get("stage")),
                "image_path": str(_resolve_path(str(r["image_path"]), project_root=project_root, dataset_root=dataset_root)),
                "sequence_id": str(r.get("sequence_id")),
                "frame_id": str(r.get("frame_id")),
            }
            for r in [far, mid, terminal]
        ],
        "outputs": {
            "png": str(out_dir / "phase3_dataset_construction.png"),
            "pdf": str(out_dir / "phase3_dataset_construction.pdf"),
        },
    }
    (out_dir / "phase3_dataset_construction_selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
