from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="List exact Stage 2 temporal-continuity violations with sample metadata."
    )
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--max-center-step-px", type=float, default=20.0)
    p.add_argument("--max-world-step-m", type=float, default=80.0)
    p.add_argument("--max-scale-change-ratio", type=float, default=0.05)
    p.add_argument("--max-angle-change-deg", type=float, default=12.0)
    p.add_argument("--max-crop-step-px", type=float, default=38.4)
    p.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Default: <dataset-root>/reports/temporal_violation_examples.json",
    )
    return p.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _step_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return float(math.hypot(float(bx) - float(ax), float(by) - float(ay)))


def _angle_delta(a_deg: float, b_deg: float) -> float:
    d = float(b_deg) - float(a_deg)
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return float(d)


def _image_path(root: Path, row: dict[str, Any]) -> str:
    p = Path(str(row.get("image_path", "")))
    if p.is_absolute():
        return str(p)
    return str((root.parent.parent.parent / p).resolve()) if str(p).startswith("data/") else str((Path.cwd() / p).resolve())


def _world(row: dict[str, Any]) -> dict[str, float]:
    w = row.get("meta", {}).get("target_state_world", {})
    return {
        "x": float(w.get("x", row.get("meta", {}).get("target_world_x", 0.0))),
        "y": float(w.get("y", row.get("meta", {}).get("target_world_y", 0.0))),
        "vx": float(w.get("vx", row.get("meta", {}).get("target_world_vx", 0.0))),
        "vy": float(w.get("vy", row.get("meta", {}).get("target_world_vy", 0.0))),
    }


def _crop(row: dict[str, Any]) -> list[float]:
    c = row.get("meta", {}).get("crop_center_world", [0.0, 0.0])
    return [float(c[0]), float(c[1])]


def _scale(row: dict[str, Any]) -> float:
    meta = row.get("meta", {})
    return float(meta.get("scale_factor", row.get("scale_px", meta.get("scale_px", 0.0))))


def _angle(row: dict[str, Any]) -> float:
    meta = row.get("meta", {})
    return float(row.get("angle_deg", meta.get("angle_deg", 0.0)))


def _center(row: dict[str, Any]) -> list[float]:
    c = row.get("target_center_px", [0.0, 0.0])
    return [float(c[0]), float(c[1])]


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    out_json = (
        Path(args.out_json).resolve()
        if args.out_json
        else (root / "reports" / "temporal_violation_examples.json").resolve()
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    by_seq: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_labels: list[str] = []
    total_rows = 0
    for split in ("train", "val", "test"):
        label_path = root / "labels" / f"{split}.jsonl"
        if not label_path.exists():
            missing_labels.append(str(label_path))
            continue
        rows = _load_jsonl(label_path)
        total_rows += len(rows)
        for r in rows:
            by_seq[str(r.get("sequence_id", ""))].append(r)

    events: list[dict[str, Any]] = []
    seq_summary: dict[str, dict[str, Any]] = {}

    for seq_id, rows in sorted(by_seq.items()):
        rows = sorted(rows, key=lambda r: int(r.get("frame_id", 0)))
        seq_events: list[dict[str, Any]] = []
        for i in range(1, len(rows)):
            a = rows[i - 1]
            b = rows[i]
            ac = _center(a)
            bc = _center(b)
            aw = _world(a)
            bw = _world(b)
            acr = _crop(a)
            bcr = _crop(b)
            agsd = float(a.get("gsd_m_per_px", a.get("meta", {}).get("gsd", 1.0)))
            bgsd = float(b.get("gsd_m_per_px", b.get("meta", {}).get("gsd", 1.0)))
            gsd = max(1e-6, 0.5 * (agsd + bgsd))

            center_step = _step_xy(ac[0], ac[1], bc[0], bc[1])
            world_step = _step_xy(aw["x"], aw["y"], bw["x"], bw["y"])
            crop_step = _step_xy(acr[0], acr[1], bcr[0], bcr[1]) / gsd
            scale_ratio = _scale(b) / max(1e-6, _scale(a))
            angle_delta = _angle_delta(_angle(a), _angle(b))

            violation_types: list[str] = []
            if center_step > float(args.max_center_step_px):
                violation_types.append("center_step")
            if world_step > float(args.max_world_step_m):
                violation_types.append("world_step")
            if abs(scale_ratio - 1.0) > float(args.max_scale_change_ratio) + 1e-6:
                violation_types.append("scale_change")
            if abs(angle_delta) > float(args.max_angle_change_deg) + 1e-6:
                violation_types.append("angle_change")
            if crop_step > float(args.max_crop_step_px) + 1e-6:
                violation_types.append("crop_step")

            if not violation_types:
                continue

            event = {
                "sequence_id": seq_id,
                "split": str(b.get("split", a.get("split", ""))),
                "from_frame": str(a.get("frame_id")),
                "to_frame": str(b.get("frame_id")),
                "from_stage": str(a.get("stage", "")),
                "to_stage": str(b.get("stage", "")),
                "violation_types": violation_types,
                "center_step_px": center_step,
                "world_step_m": world_step,
                "world_excess_m": max(0.0, world_step - float(args.max_world_step_m)),
                "scale_ratio": scale_ratio,
                "angle_delta_deg": angle_delta,
                "crop_step_px": crop_step,
                "motion_mode": str(b.get("motion_mode", a.get("motion_mode", ""))),
                "background_asset_id": str(b.get("background_asset_id", "")),
                "target_asset_id": str(b.get("target_asset_id", "")),
                "from_image_path": _image_path(root, a),
                "to_image_path": _image_path(root, b),
                "from_world": aw,
                "to_world": bw,
                "from_center_px": ac,
                "to_center_px": bc,
                "from_crop_center_world": acr,
                "to_crop_center_world": bcr,
            }
            events.append(event)
            seq_events.append(event)

        if seq_events:
            seq_summary[seq_id] = {
                "split": seq_events[0]["split"],
                "num_events": len(seq_events),
                "violation_types": sorted({t for e in seq_events for t in e["violation_types"]}),
                "max_world_step_m": max(float(e["world_step_m"]) for e in seq_events),
                "events": seq_events,
            }

    event_type_counts: dict[str, int] = defaultdict(int)
    for event in events:
        for t in event["violation_types"]:
            event_type_counts[t] += 1

    bad_sequence_count = len(seq_summary)
    only_world_step = set(event_type_counts.keys()) <= {"world_step"}
    recommendation = (
        "Formal freeze should require pass=true. Because violations are limited, use the listed sequences "
        "for visual inspection; if you need a publishable frozen dataset, rerun the full render with the latest "
        "renderer or regenerate a clean dataset version. Temporary model training may proceed, but do not call "
        "this dataset QC-passed in the paper until temporal pass=true."
    )
    if bad_sequence_count <= 10 and only_world_step:
        recommendation = (
            "Only a few world-step events were found. This is small enough for temporary training, but the formal "
            "dataset should still be regenerated or repaired before freezing, because the Stage 2 temporal gate "
            "requires world_step_violations=0."
        )

    report = {
        "dataset_root": str(root),
        "num_rows": total_rows,
        "num_sequences": len(by_seq),
        "missing_labels": missing_labels,
        "thresholds": {
            "max_center_step_px": float(args.max_center_step_px),
            "max_world_step_m": float(args.max_world_step_m),
            "max_scale_change_ratio": float(args.max_scale_change_ratio),
            "max_angle_change_deg": float(args.max_angle_change_deg),
            "max_crop_step_px": float(args.max_crop_step_px),
        },
        "num_violation_events": len(events),
        "num_bad_sequences": bad_sequence_count,
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "bad_sequences": seq_summary,
        "recommendation": recommendation,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] wrote {out_json}")
    print(f"rows={total_rows} sequences={len(by_seq)} violation_events={len(events)} bad_sequences={bad_sequence_count}")
    print(f"event_type_counts={dict(sorted(event_type_counts.items()))}")
    for seq_id, s in seq_summary.items():
        print(
            f"- {seq_id} split={s['split']} events={s['num_events']} "
            f"types={','.join(s['violation_types'])} max_world_step_m={s['max_world_step_m']:.3f}"
        )
        for e in s["events"][:3]:
            print(
                f"  frame {e['from_frame']}->{e['to_frame']} "
                f"stage {e['from_stage']}->{e['to_stage']} "
                f"world={e['world_step_m']:.3f} excess={e['world_excess_m']:.3f} "
                f"image={e['to_image_path']}"
            )
    print(recommendation)


if __name__ == "__main__":
    main()
