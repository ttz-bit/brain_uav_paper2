from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebalance stage2 background categories across splits.")
    p.add_argument("--inventory-csv", type=str, default="data/assets/stage2/asset_inventory.csv")
    p.add_argument(
        "--categories",
        type=str,
        default="open_sea,coastal,island_complex,port",
        help="Comma-separated background categories to rebalance.",
    )
    p.add_argument("--splits", type=str, default="train,val,test")
    p.add_argument("--min-per-split", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inventory_csv = Path(args.inventory_csv).resolve()
    categories = [c.strip() for c in str(args.categories).split(",") if c.strip()]
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    min_per_split = int(args.min_per_split)

    if not categories:
        raise SystemExit("No valid categories provided.")
    if not splits:
        raise SystemExit("No valid splits provided.")
    if min_per_split <= 0:
        raise SystemExit("min-per-split must be > 0.")

    with inventory_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise SystemExit(f"Invalid CSV header: {inventory_csv}")

    for cat in categories:
        subset = [
            r
            for r in rows
            if str(r.get("asset_type", "")) == "background" and str(r.get("category", "")) == cat
        ]
        required = len(splits) * min_per_split
        if len(subset) < required:
            raise SystemExit(
                f"Category '{cat}' not enough assets: have={len(subset)}, need>={required} "
                f"for splits={splits} and min-per-split={min_per_split}."
            )
        subset = sorted(subset, key=lambda r: str(r.get("asset_id", "")))
        for i, r in enumerate(subset):
            r["split"] = splits[i % len(splits)]

    with inventory_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[DONE] rebalanced backgrounds in {inventory_csv}")
    for cat in categories:
        counts = {s: 0 for s in splits}
        for r in rows:
            if str(r.get("asset_type", "")) == "background" and str(r.get("category", "")) == cat:
                counts[str(r.get("split", ""))] = counts.get(str(r.get("split", "")), 0) + 1
        print(f"[INFO] category={cat}, counts_by_split={counts}")


if __name__ == "__main__":
    main()
