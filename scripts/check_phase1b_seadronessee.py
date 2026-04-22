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
    parser.add_argument(
        "--min-qc-per-split",
        type=int,
        default=30,
        help="Minimum number of QC overlay samples required for each split",
    )
    parser.add_argument(
        "--allow-length-mismatch",
        action="store_true",
        help="Allow length_mismatch_sequences > 0 from summary stats",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _leakage_id(row: dict) -> str:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        raise KeyError("Missing meta for leakage id")
    raw_rel = str(meta.get("mot_source_relpath", "")).strip().replace("\\", "/")
    if not raw_rel:
        raise KeyError("Missing meta.mot_source_relpath for leakage id")

    # Normalize split marker so leakage check is not trivially bypassed by split folder names.
    norm = raw_rel.replace("/train/", "/").replace("/val/", "/").replace("/test/", "/")
    return norm


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifests_dir = root / "manifests"
    stats_dir = root / "stats"
    qc_dir = root / "qc"

    errors: list[str] = []
    split_leakage_ids: dict[str, set[str]] = {}
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

        leakage_ids: set[str] = set()
        schema_errors = 0
        for row in rows:
            try:
                validate_record(row)
            except KeyError as e:
                schema_errors += 1
                errors.append(f"Schema error in {manifest_path}: {e}")
                continue
            try:
                leakage_ids.add(_leakage_id(row))
            except KeyError as e:
                schema_errors += 1
                errors.append(f"Leakage id error in {manifest_path}: {e}")
        if schema_errors > 0:
            errors.append(f"{manifest_path} schema_errors={schema_errors}")
        split_leakage_ids[split] = leakage_ids

        if summary_path.exists():
            summary = _read_json(summary_path)
            if int(summary.get("schema_errors", 0)) > 0:
                errors.append(f"{summary_path} reports schema_errors={summary.get('schema_errors')}")
            if not args.allow_length_mismatch and int(summary.get("length_mismatch_sequences", 0)) > 0:
                errors.append(
                    f"{summary_path} reports length_mismatch_sequences={summary.get('length_mismatch_sequences')}"
                )

        if qc_split_dir.exists():
            qc_overlay_count = len(list(qc_split_dir.glob("*_overlay.jpg")))
            if qc_overlay_count < int(args.min_qc_per_split):
                errors.append(
                    f"QC samples too few in {qc_split_dir}: "
                    f"{qc_overlay_count} < {int(args.min_qc_per_split)}"
                )

    leakage_path = stats_dir / "split_leakage_report.json"
    pairs: list[dict] = []
    splits = sorted(split_leakage_ids.keys())
    for i, sa in enumerate(splits):
        for sb in splits[i + 1 :]:
            overlap = sorted(split_leakage_ids[sa].intersection(split_leakage_ids[sb]))
            pairs.append(
                {
                    "split_a": sa,
                    "split_b": sb,
                    "overlap_count": len(overlap),
                    "overlap_leakage_ids_sample": overlap[:20],
                }
            )
    leakage = {
        "all_clear": all(p["overlap_count"] == 0 for p in pairs),
        "pairs": pairs,
        "leakage_key": "normalized_meta.mot_source_relpath",
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
