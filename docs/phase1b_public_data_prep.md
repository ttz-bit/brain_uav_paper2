# Phase 1B Public Data Prep Note

## 1. Stage Goal

Phase 1B prepares public data for unified local-window single-target localization,
with SeaDronesSee as the first completed external dataset line.

## 2. Frozen In Phase 1B (SeaDronesSee Line)

- unified record output in `jsonl` format under `data/processed/seadronessee/manifests`
- per-record schema check with `paper2.datasets.unified_schema.validate_record`
- split leakage check report `stats/split_leakage_report.json`
- QC artifacts under `data/processed/seadronessee/qc/<split>/`
- phase check entry: `scripts/check_phase1b_seadronessee.py`

## 3. Not Frozen Yet

- AOT pipeline implementation and validation
- shared train script loading both AOT and SeaDronesSee outputs
- final world-unit/gsd values (still pending upstream physical calibration)

## 4. Acceptance Entry

1. Run preprocessing:

`python scripts/prepare_seadronessee.py --raw-root <raw_root> --out-root <out_root> --splits train val --overwrite`

2. Run check:

`python scripts/check_phase1b_seadronessee.py --root <out_root> --splits train val`

Passing criteria:

- report status is `PASS`
- command exits with code `0`
