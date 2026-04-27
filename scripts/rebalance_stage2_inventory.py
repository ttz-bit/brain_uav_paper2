from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rebalance split assignment for a specific asset type/category in stage2 inventory."
    )
    p.add_argument(
        "--inventory-csv",
        type=str,
        default="data/assets/stage2/asset_inventory.csv",
        help="Path to asset_inventory.csv",
    )
    p.add_argument("--asset-type", type=str, default="target")
    p.add_argument("--category", type=str, default="boat_top")
    p.add_argument(
        "--splits",
        type=str,
        default="train,val,test",
        help="Comma-separated split order used for round-robin assignment.",
    )
    p.add_argument(
        "--min-per-split",
        type=int,
        default=1,
        help="Minimum required count per split for this category. If unmet, exit with code 1.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inventory_csv = Path(args.inventory_csv).resolve()
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    if not splits:
        raise SystemExit("No valid splits provided.")

    with inventory_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise SystemExit(f"Invalid CSV header: {inventory_csv}")

    subset = [
        row
        for row in rows
        if str(row.get("asset_type", "")) == str(args.asset_type)
        and str(row.get("category", "")) == str(args.category)
    ]
    if not subset:
        raise SystemExit(
            f"No rows found for asset_type={args.asset_type}, category={args.category} in {inventory_csv}"
        )

    required = int(args.min_per_split) * len(splits)
    if len(subset) < required:
        raise SystemExit(
            f"Not enough rows to satisfy min-per-split: have={len(subset)}, need={required} "
            f"(asset_type={args.asset_type}, category={args.category}, splits={splits})"
        )

    subset_sorted = sorted(subset, key=lambda r: str(r.get("asset_id", "")))
    id_to_split: dict[str, str] = {}
    for i, row in enumerate(subset_sorted):
        assigned = splits[i % len(splits)]
        id_to_split[str(row.get("asset_id", ""))] = assigned

    for row in rows:
        aid = str(row.get("asset_id", ""))
        if aid in id_to_split:
            row["split"] = id_to_split[aid]

    with inventory_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = {s: 0 for s in splits}
    for row in rows:
        if str(row.get("asset_type", "")) == str(args.asset_type) and str(row.get("category", "")) == str(args.category):
            sp = str(row.get("split", ""))
            counts[sp] = counts.get(sp, 0) + 1

    print(f"[DONE] rebalanced: {inventory_csv}")
    print(
        f"[INFO] asset_type={args.asset_type}, category={args.category}, "
        f"counts_by_split={counts}"
    )


if __name__ == "__main__":
    main()

