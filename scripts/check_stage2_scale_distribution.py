from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument(
        "--bins",
        type=str,
        default="0,4,8,12,16,24,32,48,64,96,128,256",
        help="Comma-separated pixel bins for scale_px.",
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _hist(values: np.ndarray, bins: list[float]) -> list[dict]:
    if values.size == 0:
        return []
    h, edges = np.histogram(values, bins=np.array(bins, dtype=np.float64))
    out: list[dict] = []
    total = max(1, int(values.size))
    for i, c in enumerate(h.tolist()):
        out.append(
            {
                "bin": f"[{edges[i]:.1f},{edges[i+1]:.1f})",
                "count": int(c),
                "ratio": float(c / total),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    labels_dir = root / "labels"
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    bins = [float(x.strip()) for x in str(args.bins).split(",") if x.strip()]
    bins = sorted(bins)
    if len(bins) < 2:
        raise ValueError("Need at least 2 bin edges")

    split_values: dict[str, list[float]] = defaultdict(list)
    split_visibility: dict[str, list[float]] = defaultdict(list)
    split_center_dist: dict[str, list[float]] = defaultdict(list)

    for split in ("train", "val", "test"):
        rows = _load_jsonl(labels_dir / f"{split}.jsonl")
        for r in rows:
            s = float(r.get("scale_px", r.get("meta", {}).get("scale_px", 0.0)))
            split_values[split].append(s)
            split_visibility[split].append(float(r.get("visibility", 0.0)))
            cx, cy = [float(v) for v in r.get("target_center_px", [128.0, 128.0])]
            split_center_dist[split].append(float(np.hypot(cx - 128.0, cy - 128.0)))

    report: dict[str, dict] = {"dataset_root": str(root), "splits": {}}
    for split in ("train", "val", "test"):
        vals = np.array(split_values[split], dtype=np.float64)
        vis = np.array(split_visibility[split], dtype=np.float64)
        cdist = np.array(split_center_dist[split], dtype=np.float64)
        report["splits"][split] = {
            "num_frames": int(vals.size),
            "scale_px": {
                "mean": float(vals.mean()) if vals.size else 0.0,
                "p10": float(np.percentile(vals, 10)) if vals.size else 0.0,
                "p50": float(np.percentile(vals, 50)) if vals.size else 0.0,
                "p90": float(np.percentile(vals, 90)) if vals.size else 0.0,
                "hist": _hist(vals, bins),
            },
            "visibility": {
                "mean": float(vis.mean()) if vis.size else 0.0,
                "p10": float(np.percentile(vis, 10)) if vis.size else 0.0,
                "p50": float(np.percentile(vis, 50)) if vis.size else 0.0,
                "p90": float(np.percentile(vis, 90)) if vis.size else 0.0,
            },
            "center_distance_px": {
                "mean": float(cdist.mean()) if cdist.size else 0.0,
                "p50": float(np.percentile(cdist, 50)) if cdist.size else 0.0,
                "p90": float(np.percentile(cdist, 90)) if cdist.size else 0.0,
            },
        }

    out = reports_dir / "scale_distribution_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
