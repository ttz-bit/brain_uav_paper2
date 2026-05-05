# Data Change Log

## 2026-04-27 - Distractor Template Count Alignment

- Scope: `data/assets/distractor_templates_v1`
- Operation: removed 1 sample from `test` split
- Removed file: `distractor_ship_like_010_boat0077_rgba.png`
- Split counts before: `train/val/test = 14/3/4` (total `21`)
- Split counts after: `train/val/test = 14/3/3` (total `20`)
- Reason: align with experiment document constraint (`distractor templates = 15-20`) and fix dataset size to `20`.
- Sync updates: `manifests/distractor_template_splits.csv` updated to 20 rows.
- Safety: removed sample moved to `_removed/` for rollback.

## 2026-05-04 - Stage 2C Formal Render Workflow Sync

- Scope: `configs/render_stage2_c_v1.yaml`, `src/paper2/render/renderer_stage2.py`, `scripts/render_stage2_smoke.py`
- Operation: aligned formal Stage 2C rendering with current repo progress.
- Updated behavior:
  - sequence-level retry is enabled when the first frame of a candidate sequence is infeasible;
  - final world-step hard guard remains enforced;
  - per-frame semantic masks are cached to reduce repeated mask construction;
  - PNG write compression remains low for faster writes.
- Operational workflow:
  - render first, then run semantic QC and temporal QC separately;
  - use `tmux` for the long render job;
  - use `scripts/inspect_stage2_temporal_violations.py` only for debugging failed temporal QC.
- Acceptance reminder:
  - formal dataset freeze still requires `check_stage2_rendered.py` and `check_stage2_temporal_continuity.py` to pass.

## 2026-05-05 - Stage 2C v1.0.1 Render/QC Consistency Fix

- Scope: `src/paper2/render/renderer_stage2.py`, `scripts/check_stage2_rendered.py`, `scripts/check_stage2_temporal_continuity.py`, Stage 2 documentation.
- Commit: `f19360d fix(stage2): tighten render qc and retry consistency`.
- Operation: invalidated the previous partial `paper2_render_v1.0.1` run and documented the required clean rerender.
- Updated behavior:
  - sequence labels are buffered and written only after the sequence succeeds, preventing stale rows from failed retry attempts;
  - hard-fail fallback now synchronizes `frame_state` with reused pixels/crop/water-mask metadata;
  - semantic QC defaults match formal spec: `max_shore_overlap=0.01`, `max_truncation_ratio=0.15`;
  - temporal QC now uses non-port `max_world_step_m=55` and port `max_port_world_step_m=40`.
- Formal rerender target:
  - remove `data/rendered/paper2_render_v1.0.1`;
  - rerender with `--out-root data/rendered/paper2_render_v1.0.1`;
  - run semantic QC and temporal QC as separate post-render steps.
