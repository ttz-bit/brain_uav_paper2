# Stage 2 Final Rendering Specification

## 1. Scope

This document defines the final Stage 2 rendering design for Paper 2.
It is the execution and acceptance baseline for:

- Stage 2B pilot dataset generation
- Stage 2C full dataset generation
- pre-training QC gates before ANN/SNN baseline training

---

## 2. Three-Stage Observation Design

All rendered samples must belong to one of three stages:

- `far`
- `mid`
- `terminal`

All three stages keep the same visual viewpoint family (`top` / `near-top`) and differ by scale, field coverage, offset tolerance, and update semantics.

### 2.1 Stage Table

| Item | far | mid | terminal |
|---|---:|---:|---:|
| Role | coarse cue / search | main localization | terminal precision refinement |
| observation_source | `external_cue` | `onboard_midrange` | `onboard_terminal` |
| image size | 256x256 | 256x256 | 256x256 |
| heatmap size | 64x64 | 64x64 | 64x64 |
| GSD (m/px) | 25 | 25 | 10 |
| target long side (px) | 10-24 | 24-64 | 64-140 |
| center offset range (px) | 20-70 | 10-40 | 0-20 |
| update period (s) | 1.0 | 1.0 | 0.5 |
| training weight | auxiliary | primary | primary |
| target ratio | 15% | 40% | 45% |

---

## 3. Data Composition Rules

### 3.1 Split coverage

Each split (`train/val/test`) must cover all 4 background categories:

- `open_sea`
- `coastal`
- `island_complex`
- `port`

If any required category is missing in a split, rendering must fail fast.

### 3.2 Anti-leakage

Across `train/val/test`:

- background assets must not leak
- target template assets must not leak
- distractor template assets must not leak

---

## 4. Trajectory-First Constraint (Critical)

Rendering must follow:

1. sample/update world trajectory
2. constrain world trajectory to valid water area
3. map world state to image coordinates
4. render image and labels

Never use per-frame random paste as primary logic.

### 4.1 Port-special constraints

For `port` backgrounds, stricter constraints apply:

- tighter shoreline clearance
- smaller allowed world-step
- tighter water snap radius

Recommended values:

- `port_min_shore_clearance_px`: 12-14
- `port_max_world_step_m`: 40-50
- `port_water_snap_radius_px`: 24-36

---

## 5. Hard Acceptance Gates

## 5.1 Semantic gate (`check_stage2_rendered.py`)

Required pass conditions:

- `land_overlap_violations == 0`
- `shore_overlap_violations == 0` with `max_shore_overlap <= 0.01`
- `truncation_violations == 0`
- `obs_invalid_count == 0`
- `split_asset_leakage.background == 0`
- `split_asset_leakage.target == 0`
- `split_asset_leakage.distractor == 0`

## 5.2 Temporal gate (`check_stage2_temporal_continuity.py`)

Required pass conditions:

- `center_step_violations == 0`
- `world_step_violations == 0` with non-port `max_world_step_m = 55` and port `max_port_world_step_m = 40`
- `scale_change_violations == 0`
- `angle_change_violations == 0`
- `crop_step_violations == 0`
- `background_fixed_violations == 0`
- `target_fixed_violations == 0`

## 5.3 Center-bias policy

Center bias is task-dependent for online tracking.
Use documented threshold policy:

- strict default: `max-center-bias-ratio = 0.90`
- tracking-oriented tolerance: `max-center-bias-ratio = 0.95`

Any non-default threshold must be explicitly recorded in experiment notes.

---

## 6. Stage 2B Entry Criteria

Stage 2B is accepted only when all items below hold:

1. semantic gate: pass
2. temporal gate: pass
3. split coverage: all required background categories present in each split
4. anti-leakage: pass
5. manual contact-sheet inspection: no obvious beaching, dock-overlap, or label drift

---

## 7. Stage 2C Entry Criteria

Stage 2C full dataset generation is allowed only when:

1. Stage 2B acceptance is complete
2. ANN baseline converges on 2B
3. SNN baseline converges on 2B
4. per-stage (`far/mid/terminal`) metrics are reported and stable

## 7.1 Current formal run policy

The current formal Stage 2C renderer is allowed to retry an impossible sequence start.
If the first valid frame cannot be placed after all hard-constrained retries, the renderer discards the current sequence candidate and resamples background / target / motion until the sequence becomes feasible or the retry budget is exhausted.

Operationally:

- render first;
- then run semantic QC;
- then run temporal QC;
- do not couple QC into the main render job.

The formal render pipeline should use the latest renderer implementation with:

- buffered per-sequence label writes, so failed retry attempts leave no stale JSONL rows;
- synchronized hard-fail fallback state, so reused pixels and `target_state_world` stay aligned;
- final world-step hard guard;
- cached per-frame semantic masks;
- PNG compression level 1 for faster writes;
- per-frame `water_mask_crop_path`;
- per-frame `distractor_bboxes_xywh`;
- detailed temporal violation locator for debugging only.

---

## 8. Execution Checklist

Before rendering:

- [ ] rebalance inventory so each split has all required background categories
- [ ] verify no missing assets in target/distractor categories

After rendering:

- [ ] run `check_stage2_rendered.py`
- [ ] run `check_stage2_temporal_continuity.py`
- [ ] confirm water-mask metadata before water-logit or land-penalty training
- [ ] confirm distractor bbox metadata before distractor-repel training
- [ ] run scale distribution inspection
- [ ] generate contact sheets (with overlays)
- [ ] archive reports and config snapshot under dataset `meta/reports`
