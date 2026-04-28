from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Stage 2C full render + QC + 36 sheets.")
    p.add_argument("--config", type=str, default="configs/render_stage2_c_v1.yaml")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--inventory-csv", type=str, default="data/assets/stage2/asset_inventory.csv")
    p.add_argument("--out-root", type=str, default=None)
    p.add_argument("--max-center-bias-ratio", type=float, default=0.95)
    p.add_argument("--max-center-step-px", type=float, default=20.0)
    p.add_argument("--max-world-step-m", type=float, default=80.0)
    p.add_argument("--max-scale-change-ratio", type=float, default=0.05)
    p.add_argument("--max-angle-change-deg", type=float, default=12.0)
    p.add_argument("--max-crop-step-px", type=float, default=38.4)
    p.add_argument("--sheet-frames", type=int, default=20)
    p.add_argument("--sheet-cols", type=int, default=5)
    p.add_argument("--draw-labels", action="store_true")
    p.add_argument("--skip-rebalance", action="store_true")
    return p.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    cfg_path = Path(args.config).resolve()
    inv_csv = Path(args.inventory_csv).resolve()

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out_root = Path(args.out_root).resolve() if args.out_root else (project_root / "data" / "rendered" / str(cfg["dataset"]["name"])).resolve()

    py = sys.executable

    if not args.skip_rebalance:
        _run(
            [
                py,
                "scripts/rebalance_stage2_background_categories.py",
                "--inventory-csv",
                str(inv_csv),
                "--categories",
                "open_sea,coastal,island_complex,port",
                "--splits",
                "train,val,test",
                "--min-per-split",
                "2",
            ],
            cwd=project_root,
        )
        _run(
            [
                py,
                "scripts/rebalance_stage2_inventory.py",
                "--inventory-csv",
                str(inv_csv),
                "--asset-type",
                "target",
                "--category",
                "boat_top",
                "--splits",
                "train,val,test",
                "--min-per-split",
                "1",
            ],
            cwd=project_root,
        )

    _run(
        [
            py,
            "scripts/render_stage2_smoke.py",
            "--config",
            str(cfg_path),
            "--out-root",
            str(out_root),
            "--inventory-csv",
            str(inv_csv),
            "--project-root",
            str(project_root),
        ],
        cwd=project_root,
    )

    _run(
        [
            py,
            "scripts/check_stage2_rendered.py",
            "--dataset-root",
            str(out_root),
            "--max-center-bias-ratio",
            str(float(args.max_center_bias_ratio)),
        ],
        cwd=project_root,
    )

    _run(
        [
            py,
            "scripts/check_stage2_temporal_continuity.py",
            "--dataset-root",
            str(out_root),
            "--max-center-step-px",
            str(float(args.max_center_step_px)),
            "--max-world-step-m",
            str(float(args.max_world_step_m)),
            "--max-scale-change-ratio",
            str(float(args.max_scale_change_ratio)),
            "--max-angle-change-deg",
            str(float(args.max_angle_change_deg)),
            "--max-crop-step-px",
            str(float(args.max_crop_step_px)),
        ],
        cwd=project_root,
    )

    for split in ("train", "val", "test"):
        cmd = [
            py,
            "scripts/make_stage2_stage_background_sheets.py",
            "--dataset-root",
            str(out_root),
            "--split",
            split,
            "--background-categories",
            "open_sea,coastal,island_complex,port",
            "--frames",
            str(int(args.sheet_frames)),
            "--cols",
            str(int(args.sheet_cols)),
            "--inventory-csv",
            str(inv_csv),
        ]
        if args.draw_labels:
            cmd.append("--draw-labels")
        _run(cmd, cwd=project_root)

    print(f"[DONE] Stage2C pipeline complete: {out_root}")


if __name__ == "__main__":
    main()
