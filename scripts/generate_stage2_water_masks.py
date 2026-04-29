from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper2.render.renderer_stage2 import _safe_water_mask, _water_mask, _water_with_clearance


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Batch-generate initial water masks for stage2/phase3 background assets. "
            "The masks are automatic first-pass labels; inspect preview images before "
            "using them as final placement constraints."
        )
    )
    p.add_argument(
        "--background-root",
        type=str,
        default="data/assets/source_stage2/backgrounds",
        help="Root directory containing background images.",
    )
    p.add_argument(
        "--out-root",
        type=str,
        default="data/assets/source_stage2/water_masks_auto",
        help="Output directory for generated masks, previews, and report.",
    )
    p.add_argument(
        "--safe-margin-px",
        type=int,
        default=6,
        help="Erode the raw water mask by this many pixels to avoid shoreline placement.",
    )
    p.add_argument(
        "--clearance-px",
        type=int,
        default=5,
        help="Create an additional deep-water mask requiring this pixel clearance from non-water.",
    )
    p.add_argument(
        "--min-water-ratio",
        type=float,
        default=0.03,
        help="Images below this safe-water ratio are flagged for manual review.",
    )
    p.add_argument(
        "--preview-alpha",
        type=float,
        default=0.38,
        help="Overlay strength for preview images.",
    )
    p.add_argument("--limit", type=int, default=0, help="If >0, process at most this many images.")
    return p.parse_args()


def iter_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def rel_output_path(src: Path, root: Path, out_root: Path, suffix: str) -> Path:
    rel = src.relative_to(root)
    return out_root / rel.parent / f"{rel.stem}{suffix}.png"


def make_preview(bgr: np.ndarray, safe_mask: np.ndarray, deep_mask: np.ndarray, alpha: float) -> np.ndarray:
    preview = bgr.copy()
    overlay = np.zeros_like(preview, dtype=np.uint8)
    overlay[safe_mask > 0] = (40, 210, 60)
    overlay[(safe_mask > 0) & (deep_mask == 0)] = (0, 210, 210)
    blended = cv2.addWeighted(preview, 1.0 - float(alpha), overlay, float(alpha), 0.0)
    return np.where((safe_mask[:, :, None] > 0), blended, preview)


def mask_ratio(mask: np.ndarray) -> float:
    if mask.size <= 0:
        return 0.0
    return float((mask > 0).mean())


def main() -> None:
    args = parse_args()
    background_root = Path(args.background_root).resolve()
    out_root = Path(args.out_root).resolve()
    mask_root = out_root / "masks"
    deep_mask_root = out_root / "deep_water_masks"
    preview_root = out_root / "previews"
    report_dir = out_root / "reports"
    for path in (mask_root, deep_mask_root, preview_root, report_dir):
        path.mkdir(parents=True, exist_ok=True)

    image_paths = iter_images(background_root)
    if int(args.limit) > 0:
        image_paths = image_paths[: int(args.limit)]

    rows: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for src in image_paths:
        try:
            bgr = read_bgr(src)
            raw = _water_mask(bgr)
            safe = _safe_water_mask(raw, margin_px=int(args.safe_margin_px))
            deep = _water_with_clearance(safe, min_clearance_px=int(args.clearance_px))

            mask_path = rel_output_path(src, background_root, mask_root, "_water_mask")
            deep_path = rel_output_path(src, background_root, deep_mask_root, "_deep_water_mask")
            preview_path = rel_output_path(src, background_root, preview_root, "_water_preview")
            for p in (mask_path, deep_path, preview_path):
                p.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(mask_path), safe)
            cv2.imwrite(str(deep_path), deep)
            cv2.imwrite(str(preview_path), make_preview(bgr, safe, deep, alpha=float(args.preview_alpha)))

            safe_ratio = mask_ratio(safe)
            deep_ratio = mask_ratio(deep)
            rows.append(
                {
                    "image_path": str(src),
                    "mask_path": str(mask_path),
                    "deep_water_mask_path": str(deep_path),
                    "preview_path": str(preview_path),
                    "width": int(bgr.shape[1]),
                    "height": int(bgr.shape[0]),
                    "safe_water_ratio": safe_ratio,
                    "deep_water_ratio": deep_ratio,
                    "review": bool(safe_ratio < float(args.min_water_ratio)),
                }
            )
        except Exception as exc:
            failed.append({"image_path": str(src), "error": str(exc)})

    ratios = np.asarray([float(r["safe_water_ratio"]) for r in rows], dtype=float)
    deep_ratios = np.asarray([float(r["deep_water_ratio"]) for r in rows], dtype=float)
    review_rows = [r for r in rows if bool(r["review"])]
    report = {
        "task": "generate_stage2_water_masks",
        "background_root": str(background_root),
        "out_root": str(out_root),
        "num_images": int(len(image_paths)),
        "processed": int(len(rows)),
        "failed": int(len(failed)),
        "review_count": int(len(review_rows)),
        "safe_margin_px": int(args.safe_margin_px),
        "clearance_px": int(args.clearance_px),
        "min_water_ratio": float(args.min_water_ratio),
        "safe_water_ratio_min": float(ratios.min()) if ratios.size else 0.0,
        "safe_water_ratio_mean": float(ratios.mean()) if ratios.size else 0.0,
        "safe_water_ratio_max": float(ratios.max()) if ratios.size else 0.0,
        "deep_water_ratio_min": float(deep_ratios.min()) if deep_ratios.size else 0.0,
        "deep_water_ratio_mean": float(deep_ratios.mean()) if deep_ratios.size else 0.0,
        "deep_water_ratio_max": float(deep_ratios.max()) if deep_ratios.size else 0.0,
        "rows": rows,
        "failed_examples": failed[:20],
        "review_examples": review_rows[:20],
        "accepted": bool(len(rows) > 0 and len(failed) == 0),
    }
    report_path = report_dir / "water_mask_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows"}}, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
