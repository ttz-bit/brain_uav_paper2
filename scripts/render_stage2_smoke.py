from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from paper2.common.config import load_yaml
from paper2.render.asset_registry import AssetRegistry, load_asset_inventory
from paper2.render.renderer_stage2 import Stage2Renderer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/render_stage2.yaml")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--inventory-csv", type=str, default=None)
    p.add_argument("--out-root", type=str, default=None)
    return p.parse_args()


def _make_contact_sheet(image_paths: list[Path], out_path: Path, cols: int = 5) -> None:
    if not image_paths:
        return
    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in image_paths]
    imgs = [x for x in imgs if x is not None]
    if not imgs:
        return
    h, w = imgs[0].shape[:2]
    cols = max(1, int(cols))
    rows = int(np.ceil(len(imgs) / cols))
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r = idx // cols
        c = idx % cols
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet)


def _first_sequence_paths(images_root: Path, split: str, frames: int) -> list[Path]:
    paths = []
    for i in range(frames):
        p = images_root / split / f"seq_0000_frame_{i:04d}.png"
        if p.exists():
            paths.append(p)
    return paths


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    cfg = load_yaml(args.config)
    inventory_csv = Path(args.inventory_csv).resolve() if args.inventory_csv else (project_root / cfg["assets"]["inventory_csv"]).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else (project_root / "data" / "rendered" / cfg["dataset"]["name"]).resolve()

    seed = int(cfg["seed"]["global_seed"])
    rng = np.random.default_rng(seed)

    assets = load_asset_inventory(inventory_csv)
    registry = AssetRegistry(assets)
    renderer = Stage2Renderer(cfg=cfg, registry=registry, project_root=project_root, output_root=out_root, rng=rng)

    split_results = []
    for split, split_cfg in cfg["splits"].items():
        res = renderer.render_split(split=split, num_sequences=int(split_cfg["sequences"]))
        split_results.append(res)

    meta_dir = out_root / "meta"
    reports_dir = out_root / "reports"
    meta_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    generation_cfg_path = meta_dir / "generation_config.yaml"
    generation_cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    seed_plan = {
        "global_seed": seed,
        "splits": {
            str(s.split): {
                "sequences": int(s.sequences),
                "frames_per_sequence": int(cfg["dataset"]["frames_per_sequence"]),
            }
            for s in split_results
        },
    }
    (meta_dir / "seed_plan.json").write_text(json.dumps(seed_plan, ensure_ascii=False, indent=2), encoding="utf-8")

    lock_manifest = {
        "dataset_name": cfg["dataset"]["name"],
        "inventory_csv": str(inventory_csv),
        "used_assets": {
            s.split: {
                "background_ids": sorted(s.used_background_ids),
                "target_ids": sorted(s.used_target_ids),
                "distractor_ids": sorted(s.used_distractor_ids),
            }
            for s in split_results
        },
    }
    (meta_dir / "asset_lock_manifest.json").write_text(
        json.dumps(lock_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dataset_stats = {
        "dataset_name": cfg["dataset"]["name"],
        "image_size": int(cfg["dataset"]["image_size"]),
        "frames_per_sequence": int(cfg["dataset"]["frames_per_sequence"]),
        "splits": {
            s.split: {
                "sequences": int(s.sequences),
                "frames": int(s.frames),
                "label_path": str(s.label_path),
            }
            for s in split_results
        },
        "total_frames": int(sum(s.frames for s in split_results)),
    }
    (meta_dir / "dataset_stats.json").write_text(json.dumps(dataset_stats, ensure_ascii=False, indent=2), encoding="utf-8")

    images_root = out_root / "images"
    frames = int(cfg["dataset"]["frames_per_sequence"])
    for split in cfg["splits"].keys():
        sheet_paths = _first_sequence_paths(images_root, split, frames)
        _make_contact_sheet(sheet_paths, reports_dir / "contact_sheets" / f"{split}_seq0000.jpg", cols=5)

    smoke_report = {
        "task": "render_stage2_smoke",
        "status": "PASS",
        "dataset_root": str(out_root),
        "inventory_csv": str(inventory_csv),
        "splits": dataset_stats["splits"],
        "total_frames": dataset_stats["total_frames"],
        "artifacts": {
            "meta_dir": str(meta_dir),
            "reports_dir": str(reports_dir),
            "asset_lock_manifest": str(meta_dir / "asset_lock_manifest.json"),
            "dataset_stats": str(meta_dir / "dataset_stats.json"),
        },
    }
    (reports_dir / "smoke_render_report.json").write_text(json.dumps(smoke_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(smoke_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
