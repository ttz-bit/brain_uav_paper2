from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_smoke_v0")
    p.add_argument("--max-center-step-px", type=float, default=20.0)
    p.add_argument("--max-world-step-m", type=float, default=80.0)
    p.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Default: <dataset-root>/reports/temporal_continuity_report.json",
    )
    return p.parse_args()


def _load_rows(label_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not label_path.exists():
        return rows
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _step(a: list[float], b: list[float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    return float(np.hypot(bx - ax, by - ay))


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    out_json = (
        Path(args.out_json).resolve()
        if args.out_json
        else (root / "reports" / "temporal_continuity_report.json").resolve()
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    by_seq: dict[str, list[dict]] = defaultdict(list)
    total_rows = 0
    for split in ("train", "val", "test"):
        label_path = root / "labels" / f"{split}.jsonl"
        rows = _load_rows(label_path)
        total_rows += len(rows)
        for r in rows:
            by_seq[str(r["sequence_id"])].append(r)

    seq_reports: dict[str, dict] = {}
    all_center_steps: list[float] = []
    all_world_steps: list[float] = []
    center_violations = 0
    world_violations = 0

    for seq_id, rows in by_seq.items():
        rows = sorted(rows, key=lambda x: int(x["frame_id"]))
        center_steps: list[float] = []
        world_steps: list[float] = []
        seq_bad: list[dict] = []
        for i in range(1, len(rows)):
            a = rows[i - 1]
            b = rows[i]
            c_step = _step(a["target_center_px"], b["target_center_px"])
            aw = a["meta"]["target_state_world"]
            bw = b["meta"]["target_state_world"]
            w_step = _step([aw["x"], aw["y"]], [bw["x"], bw["y"]])
            center_steps.append(c_step)
            world_steps.append(w_step)
            if c_step > float(args.max_center_step_px) or w_step > float(args.max_world_step_m):
                seq_bad.append(
                    {
                        "from_frame": str(a["frame_id"]),
                        "to_frame": str(b["frame_id"]),
                        "center_step_px": c_step,
                        "world_step_m": w_step,
                    }
                )
            if c_step > float(args.max_center_step_px):
                center_violations += 1
            if w_step > float(args.max_world_step_m):
                world_violations += 1

        all_center_steps.extend(center_steps)
        all_world_steps.extend(world_steps)

        seq_reports[seq_id] = {
            "num_frames": len(rows),
            "num_steps": max(0, len(rows) - 1),
            "center_step_px": {
                "mean": float(np.mean(center_steps)) if center_steps else 0.0,
                "p95": float(np.percentile(center_steps, 95)) if center_steps else 0.0,
                "max": float(np.max(center_steps)) if center_steps else 0.0,
            },
            "world_step_m": {
                "mean": float(np.mean(world_steps)) if world_steps else 0.0,
                "p95": float(np.percentile(world_steps, 95)) if world_steps else 0.0,
                "max": float(np.max(world_steps)) if world_steps else 0.0,
            },
            "jump_events": seq_bad[:20],
            "num_jump_events": len(seq_bad),
        }

    report = {
        "dataset_root": str(root),
        "num_rows": int(total_rows),
        "num_sequences": int(len(by_seq)),
        "thresholds": {
            "max_center_step_px": float(args.max_center_step_px),
            "max_world_step_m": float(args.max_world_step_m),
        },
        "global": {
            "center_step_px": {
                "mean": float(np.mean(all_center_steps)) if all_center_steps else 0.0,
                "p95": float(np.percentile(all_center_steps, 95)) if all_center_steps else 0.0,
                "max": float(np.max(all_center_steps)) if all_center_steps else 0.0,
            },
            "world_step_m": {
                "mean": float(np.mean(all_world_steps)) if all_world_steps else 0.0,
                "p95": float(np.percentile(all_world_steps, 95)) if all_world_steps else 0.0,
                "max": float(np.max(all_world_steps)) if all_world_steps else 0.0,
            },
            "center_step_violations": int(center_violations),
            "world_step_violations": int(world_violations),
        },
        "pass": bool(center_violations == 0 and world_violations == 0),
        "sequences": seq_reports,
    }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

