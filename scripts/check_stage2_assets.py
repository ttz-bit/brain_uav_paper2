from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from paper2.render.asset_registry import load_asset_inventory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--inventory-csv", type=str, default="data/assets/stage2/asset_inventory.csv")
    p.add_argument("--out-json", type=str, default="data/assets/stage2/reports/asset_qc_report.json")
    return p.parse_args()


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    inventory = load_asset_inventory(args.inventory_csv)
    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    missing_files: list[str] = []
    invalid_shapes: list[str] = []
    invalid_alpha: list[str] = []

    hash_by_type: dict[str, dict[str, set[str]]] = {}
    counts: dict[str, dict[str, int]] = {}

    for a in inventory:
        path = a.path
        counts.setdefault(a.asset_type, {}).setdefault(a.split, 0)
        counts[a.asset_type][a.split] += 1
        if not path.exists():
            missing_files.append(str(path))
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            missing_files.append(str(path))
            continue
        if img.shape[1] != a.width or img.shape[0] != a.height:
            invalid_shapes.append(str(path))
        if a.asset_type in {"target", "distractor"}:
            if img.ndim != 3 or img.shape[2] != 4:
                invalid_alpha.append(str(path))
            else:
                alpha = img[:, :, 3]
                if int((alpha > 10).sum()) <= 0:
                    invalid_alpha.append(str(path))

        digest = _file_hash(path)
        hash_by_type.setdefault(a.asset_type, {}).setdefault(digest, set()).add(a.split)

    overlap: dict[str, int] = {}
    overlap_examples: dict[str, list[dict[str, object]]] = {}
    for asset_type, digest_map in hash_by_type.items():
        overlap_count = 0
        ex: list[dict[str, object]] = []
        for digest, splits in digest_map.items():
            if len(splits) > 1:
                overlap_count += 1
                if len(ex) < 10:
                    ex.append({"hash": digest[:12], "splits": sorted(splits)})
        overlap[asset_type] = overlap_count
        overlap_examples[asset_type] = ex

    report = {
        "inventory_csv": str(Path(args.inventory_csv).resolve()),
        "num_assets": len(inventory),
        "counts_by_type_split": counts,
        "all_files_exist": len(missing_files) == 0,
        "shape_ok": len(invalid_shapes) == 0,
        "alpha_png_ok": len(invalid_alpha) == 0,
        "split_overlap_count": overlap,
        "split_overlap_examples": overlap_examples,
        "errors": {
            "missing_files": missing_files[:50],
            "invalid_shapes": invalid_shapes[:50],
            "invalid_alpha": invalid_alpha[:50],
        },
        "pass": (
            len(missing_files) == 0
            and len(invalid_shapes) == 0
            and len(invalid_alpha) == 0
            and all(v == 0 for v in overlap.values())
        ),
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
