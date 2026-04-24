# Phase 1B Baseline Criteria

This document freezes how we summarize and judge Phase 1B baseline runs.

## Goal

Build a stable baseline reference before model upgrades in later phases.

## Required Inputs

- `report.json` from each registered run.
- Run list config: `configs/phase1b_baseline_runs.json`.

## Summary Script

Use:

```bash
python scripts/summarize_phase1b_baseline.py \
  --config configs/phase1b_baseline_runs.json \
  --out-dir outputs/reports
```

## Required Artifacts Per Run

- `report.json`
- `linear_weights.npy`
- at least one file in `visuals/*.jpg`

If any required artifact is missing, the run is marked as failed in summary.

## Metric Pass Rules

A run passes metric checks only when both conditions are true:

- `final_loss < initial_loss`
- `improve_ratio > 0`

## Stability Rule (Cross-Seed)

For each split (`train`, `val`), all registered runs should pass metric checks.

The summary reports:

- `stability_by_split.<split>.all_metrics_ok`
- min/max `improve_ratio` across runs in that split.

## Training vs Evaluation Separation

Current Phase 1B records are mostly `fit_check` runs.

- `fit_check`: training updates are applied on the selected split.
- `evaluation`: no parameter update, only forward-pass evaluation.

The summary reports `eval_separation_ok`.
If `false`, Phase 1B baseline is still valid for fitting stability, but not yet a formal generalization evaluation protocol.
