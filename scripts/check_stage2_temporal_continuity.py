from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_smoke_v0")
    p.add_argument("--mode", type=str, default="tracking", choices=["tracking", "localization"])
    p.add_argument("--max-center-step-px", type=float, default=20.0)
    p.add_argument("--max-world-step-m", type=float, default=55.0)
    p.add_argument("--max-port-world-step-m", type=float, default=40.0)
    p.add_argument("--max-scale-change-ratio", type=float, default=0.05)
    p.add_argument("--max-angle-change-deg", type=float, default=12.0)
    p.add_argument("--max-crop-step-px", type=float, default=38.4)
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


def _world_step_budget(row: dict, args: argparse.Namespace) -> float:
    bg_category = str(row.get("meta", {}).get("background_category", "")).strip().lower()
    if bg_category == "port":
        return float(args.max_port_world_step_m)
    return float(args.max_world_step_m)


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
    all_crop_steps: list[float] = []
    scale_change_violations = 0
    angle_change_violations = 0
    crop_step_violations = 0
    center_violations = 0
    world_violations = 0

    for seq_id, rows in by_seq.items():
        rows = sorted(rows, key=lambda x: int(x["frame_id"]))
        center_steps: list[float] = []
        world_steps: list[float] = []
        crop_steps: list[float] = []
        seq_bad: list[dict] = []
        seq_scale_change_violations = 0
        seq_angle_change_violations = 0
        seq_crop_step_violations = 0
        bg_ids = {str(r.get("background_asset_id", "")) for r in rows}
        tgt_ids = {str(r.get("target_asset_id", "")) for r in rows}
        for i in range(1, len(rows)):
            a = rows[i - 1]
            b = rows[i]
            c_step = _step(a["target_center_px"], b["target_center_px"])
            aw = a["meta"]["target_state_world"]
            bw = b["meta"]["target_state_world"]
            w_step = _step([aw["x"], aw["y"]], [bw["x"], bw["y"]])
            w_budget = _world_step_budget(b, args)
            ac = a["meta"].get("crop_center_world", [0.0, 0.0])
            bc = b["meta"].get("crop_center_world", [0.0, 0.0])
            agsd = float(a.get("gsd_m_per_px", a["meta"].get("gsd", 1.0)))
            bgsd = float(b.get("gsd_m_per_px", b["meta"].get("gsd", 1.0)))
            gsd = max(1e-6, 0.5 * (agsd + bgsd))
            crop_step_m = _step([float(ac[0]), float(ac[1])], [float(bc[0]), float(bc[1])])
            crop_step_px = crop_step_m / gsd
            a_scale = float(a.get("meta", {}).get("scale_factor", a.get("scale_px", a["meta"].get("scale_px", 0.0))))
            b_scale = float(b.get("meta", {}).get("scale_factor", b.get("scale_px", b["meta"].get("scale_px", 0.0))))
            scale_ratio = float(b_scale / max(1e-6, a_scale))
            a_ang = float(a.get("angle_deg", a["meta"].get("angle_deg", 0.0)))
            b_ang = float(b.get("angle_deg", b["meta"].get("angle_deg", 0.0)))
            d_ang = b_ang - a_ang
            while d_ang > 180.0:
                d_ang -= 360.0
            while d_ang < -180.0:
                d_ang += 360.0

            center_steps.append(c_step)
            world_steps.append(w_step)
            crop_steps.append(crop_step_px)
            if (
                c_step > float(args.max_center_step_px)
                or w_step > w_budget
                or abs(scale_ratio - 1.0) > float(args.max_scale_change_ratio) + 1e-6
                or abs(d_ang) > float(args.max_angle_change_deg) + 1e-6
                or crop_step_px > float(args.max_crop_step_px) + 1e-6
            ):
                seq_bad.append(
                    {
                        "from_frame": str(a["frame_id"]),
                        "to_frame": str(b["frame_id"]),
                        "center_step_px": c_step,
                        "world_step_m": w_step,
                        "world_step_budget_m": w_budget,
                        "scale_ratio": scale_ratio,
                        "angle_delta_deg": float(d_ang),
                        "crop_step_px": crop_step_px,
                    }
                )
            if c_step > float(args.max_center_step_px):
                center_violations += 1
            if w_step > w_budget:
                world_violations += 1
            if abs(scale_ratio - 1.0) > float(args.max_scale_change_ratio) + 1e-6:
                scale_change_violations += 1
                seq_scale_change_violations += 1
            if abs(d_ang) > float(args.max_angle_change_deg) + 1e-6:
                angle_change_violations += 1
                seq_angle_change_violations += 1
            if crop_step_px > float(args.max_crop_step_px) + 1e-6:
                crop_step_violations += 1
                seq_crop_step_violations += 1

        all_center_steps.extend(center_steps)
        all_world_steps.extend(world_steps)
        all_crop_steps.extend(crop_steps)

        seq_reports[seq_id] = {
            "num_frames": len(rows),
            "num_steps": max(0, len(rows) - 1),
            "background_fixed": bool(len(bg_ids) == 1),
            "target_fixed": bool(len(tgt_ids) == 1),
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
            "crop_step_px": {
                "mean": float(np.mean(crop_steps)) if crop_steps else 0.0,
                "p95": float(np.percentile(crop_steps, 95)) if crop_steps else 0.0,
                "max": float(np.max(crop_steps)) if crop_steps else 0.0,
            },
            "scale_change_violations": int(seq_scale_change_violations),
            "angle_change_violations": int(seq_angle_change_violations),
            "crop_step_violations": int(seq_crop_step_violations),
            "jump_events": seq_bad[:20],
            "num_jump_events": len(seq_bad),
        }

    fixed_bg_ok = all(v.get("background_fixed", False) for v in seq_reports.values()) if seq_reports else True
    fixed_target_ok = all(v.get("target_fixed", False) for v in seq_reports.values()) if seq_reports else True
    report = {
        "dataset_root": str(root),
        "mode": str(args.mode),
        "num_rows": int(total_rows),
        "num_sequences": int(len(by_seq)),
        "thresholds": {
            "max_center_step_px": float(args.max_center_step_px),
            "max_world_step_m": float(args.max_world_step_m),
            "max_port_world_step_m": float(args.max_port_world_step_m),
            "max_scale_change_ratio": float(args.max_scale_change_ratio),
            "max_angle_change_deg": float(args.max_angle_change_deg),
            "max_crop_step_px": float(args.max_crop_step_px),
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
            "crop_step_px": {
                "mean": float(np.mean(all_crop_steps)) if all_crop_steps else 0.0,
                "p95": float(np.percentile(all_crop_steps, 95)) if all_crop_steps else 0.0,
                "max": float(np.max(all_crop_steps)) if all_crop_steps else 0.0,
            },
            "center_step_violations": int(center_violations),
            "world_step_violations": int(world_violations),
            "scale_change_violations": int(scale_change_violations),
            "angle_change_violations": int(angle_change_violations),
            "crop_step_violations": int(crop_step_violations),
            "background_fixed_violations": int(sum(1 for v in seq_reports.values() if not v.get("background_fixed", False))),
            "target_fixed_violations": int(sum(1 for v in seq_reports.values() if not v.get("target_fixed", False))),
        },
        "pass": bool(
            (
                center_violations == 0
                and world_violations == 0
                and scale_change_violations == 0
                and angle_change_violations == 0
                and crop_step_violations == 0
                and fixed_bg_ok
                and fixed_target_ok
            )
            if str(args.mode) == "tracking"
            else True
        ),
        "sequences": seq_reports,
    }

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
