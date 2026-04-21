# Phase 1A Dynamic Env Freeze Note

## 1. Stage Goal

Phase 1A focuses on a runnable dynamic-target environment scaffold under abstract units,
without connecting the paper1 final environment bridge or visual model inference.

This stage is considered complete when:

- dynamic target motion is generated in four modes (`cv/turn/piecewise/evasive`)
- episode sampling constraints are enforced
- environment `reset/step` loop is runnable
- termination logic is stable and explicit
- sanity and check scripts can verify the above deterministically

## 2. Frozen In Phase 1A

- `DynamicTargetEnvPhase1A` as the minimal runnable env entry
- typed outputs for `reset/step`:
  - `EnvObservation`
  - `EnvStepResult`
- termination reason names:
  - `running`
  - `captured`
  - `timeout`
  - `out_of_bounds`
  - `target_out_of_bounds`
  - `safety_violation`
- single step-budget control via `phase1a.max_steps` (no dual step-limit source)
- Phase 1A protocol extension method:
  - `get_truth_crop_center_world()`

## 3. Not Frozen Yet

- final physical units and scene scaling
- final paper1 bridge bindings
- phase2 render crop policy wiring (`render.yaml` side)
- phase3+ closed-loop metric formulas

## 4. Acceptance Entry

- `scripts/check_phase1a.py`
- `scripts/run_phase1a_sanity.py`

Both should pass before moving to phase2 data/render integration work.

`run_phase1a_sanity.py` provides hard-gated acceptance with explicit thresholds,
writes `outputs/eval/phase1a_sanity/acceptance.json`, and exits with:

- `0` for PASS
- `1` for FAIL
