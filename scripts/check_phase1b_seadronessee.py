from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper2.datasets.unified_schema import validate_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\Projects\brain_uav_paper2\data\processed\seadronessee",
        help="Processed SeaDronesSee root",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifests_dir = root / "manifests"
    stats_dir = root / "stats"
    qc_dir = root / "qc"

    errors: list[str] = []
    split_source_tracks: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}

    if not manifests_dir.exists():
        errors.append(f"Missing manifests dir: {manifests_dir}")
    if not stats_dir.exists():
        errors.append(f"Missing stats dir: {stats_dir}")
    if not qc_dir.exists():
        errors.append(f"Missing qc dir: {qc_dir}")

    for split in args.splits:
        manifest_path = manifests_dir / f"records_{split}.jsonl"
        summary_path = stats_dir / f"summary_{split}.json"
        qc_split_dir = qc_dir / split

        if not manifest_path.exists():
            errors.append(f"Missing manifest: {manifest_path}")
            continue
        if not summary_path.exists():
            errors.append(f"Missing summary: {summary_path}")
        if not qc_split_dir.exists():
            errors.append(f"Missing qc split dir: {qc_split_dir}")

        rows = _read_jsonl(manifest_path)
        split_counts[split] = len(rows)
        if len(rows) == 0:
            errors.append(f"Empty manifest: {manifest_path}")
            continue

        tracks: set[str] = set()
        schema_errors = 0
        for row in rows:
            try:
                validate_record(row)
            except KeyError as e:
                schema_errors += 1
                errors.append(f"Schema error in {manifest_path}: {e}")
                continue
            tracks.add(str(row["source_track"]))
        if schema_errors > 0:
            errors.append(f"{manifest_path} schema_errors={schema_errors}")
        split_source_tracks[split] = tracks

    leakage_path = stats_dir / "split_leakage_report.json"
    pairs: list[dict] = []
    splits = sorted(split_source_tracks.keys())
    for i, sa in enumerate(splits):
        for sb in splits[i + 1 :]:
            overlap = sorted(split_source_tracks[sa].intersection(split_source_tracks[sb]))
            pairs.append(
                {
                    "split_a": sa,
                    "split_b": sb,
                    "overlap_count": len(overlap),
                    "overlap_source_tracks_sample": overlap[:20],
                }
            )
    leakage = {
        "all_clear": all(p["overlap_count"] == 0 for p in pairs),
        "pairs": pairs,
        "leakage_key": "source_track",
    }
    leakage_path.write_text(json.dumps(leakage, ensure_ascii=False, indent=2), encoding="utf-8")
    if not leakage["all_clear"]:
        for p in pairs:
            if p["overlap_count"] > 0:
                errors.append(f"Split overlap detected {p['split_a']}-{p['split_b']}: {p['overlap_count']}")

    report = {
        "phase": "phase1b_seadronessee",
        "status": "PASS" if not errors else "FAIL",
        "root": str(root),
        "splits": args.splits,
        "split_counts": split_counts,
        "errors": errors,
    }
    out_path = stats_dir / "phase1b_check_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        print("PHASE1B SEADRONESSEE: FAIL")
        raise SystemExit(1)
    print("PHASE1B SEADRONESSEE: PASS")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
