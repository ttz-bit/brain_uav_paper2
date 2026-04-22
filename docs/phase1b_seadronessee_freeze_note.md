# Phase 1B SeaDronesSee Freeze Note (Strict Recheck)

Date: 2026-04-22

## Scope

This note freezes the current SeaDronesSee processing line for Paper2 Phase 1B, after the strict recheck pass.

## Recheck Command

```powershell
.\.venv\Scripts\python.exe scripts\check_phase1b_seadronessee.py --root data\processed\seadronessee --splits train val
```

## Result

- status: PASS
- split_counts:
  - train: 26464
  - val: 30489
- errors: []

Source: `data/processed/seadronessee/stats/phase1b_check_report.json`

## Leakage Check

- all_clear: true
- overlap_count(train,val): 0
- leakage_key: normalized_meta.mot_source_relpath

Source: `data/processed/seadronessee/stats/split_leakage_report.json`

## Schema / Data Integrity

- train summary:
  - schema_errors: 0
  - missing_images: 0
  - invalid_bbox: 0
  - bad_annotations: 0
  - length_mismatch_policy: skip
  - length_mismatch_sequences: 0
  - length_mismatch_sequences_skipped: 1
- val summary:
  - schema_errors: 0
  - missing_images: 0
  - invalid_bbox: 0
  - bad_annotations: 0
  - length_mismatch_policy: skip
  - length_mismatch_sequences: 0
  - length_mismatch_sequences_skipped: 0

Sources:
- `data/processed/seadronessee/stats/summary_train.json`
- `data/processed/seadronessee/stats/summary_val.json`

## QC / File Consistency Spot Checks

- QC overlay samples:
  - train: 50
  - val: 50
- Manifest-to-crop path existence check:
  - train: 26464 rows, 0 missing crop files
  - val: 30489 rows, 0 missing crop files

## Notes

- One train sequence still has frame/annotation length mismatch (`sequence_id=32`, -3 frames vs annotations), recorded in:
  - `data/processed/seadronessee/stats/length_mismatches_train.json`
- Current policy is `skip`, so this mismatched sequence is excluded and strict Phase 1B check still passes.
