from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2.common.config import load_yaml
from paper2.render.phase3_task_sampler import sample_phase3_task_sequence, summarize_phase3_task_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/env.yaml")
    parser.add_argument("--sequences", type=int, default=12)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/reports")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))
    target_cfg = cfg["phase3_target_motion"]
    stage_cfg = cfg["phase3_task_stages"]
    base_seed = int(args.seed if args.seed is not None else target_cfg["seed"])
    rows = []
    for seq_idx in range(int(args.sequences)):
        rows.extend(
            sample_phase3_task_sequence(
                sequence_idx=seq_idx,
                target_cfg=target_cfg,
                stage_cfg=stage_cfg,
                seed=base_seed + seq_idx,
                frames=int(args.frames),
            )
        )

    summary = summarize_phase3_task_frames(rows, stage_cfg)
    report = {
        "task": "check_phase3_task_sampler",
        "config": str(args.config),
        "seed": base_seed,
        "sequences": int(args.sequences),
        "frames_per_sequence": int(args.frames),
        "summary": summary,
        "accepted": bool(summary["accepted"]),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "phase3_task_sampler_check.json"
    sample_path = out_dir / "phase3_task_sampler_samples.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_path.write_text(
        json.dumps([row.to_dict() for row in rows[: min(20, len(rows))]], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
