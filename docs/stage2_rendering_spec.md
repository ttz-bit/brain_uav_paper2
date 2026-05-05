# Stage 2 Rendering Specification (Hard Constraints)

This specification is the acceptance baseline for Stage 2 rendering.
For the current formal dataset, render first and run QC separately after render completion.

## 1. Semantic Placement

- Target must be placed on `water_valid_mask` only.
- `land_overlap_ratio` must be `0.0`.
- `shore_buffer_overlap_ratio` must be `<= 0.01`.
- `visibility` must be `>= 0.35`.
- `truncation_ratio` (`1 - visibility`) must be `<= 0.15`.

## 2. Temporal Continuity

- Same sequence keeps fixed `background_asset_id`.
- Same sequence keeps fixed `target_asset_id`.
- Per-frame target center step: `<= 0.15 * image_size` pixels.
- Per-frame crop center step: `<= 0.15 * image_size` pixels (converted by current `gsd_m_per_px`).
- Per-frame world target step: `<= 55m` for non-port backgrounds and `<= 40m` for `port` backgrounds.
- Per-frame scale change ratio: `abs(scale_t / scale_t-1 - 1) <= 0.05`.
- Per-frame angle change: `<= 12 deg`.

## 3. Viewpoint and Realism

- Allowed positive target categories are controlled by config:
  - default: `boat_top`, `boat_oblique`.
- Side-view templates are excluded from positive targets.
- Target orientation follows motion heading with bounded noise.
- Overlay appearance is harmonized to local background to reduce cutout artifacts.
- Current renderer behavior:
  - cached per-frame semantic masks are used to avoid repeated mask construction;
  - PNG write compression is intentionally low for speed;
  - impossible first-frame sequence candidates are retried at the sequence level instead of writing a partial sequence;
  - labels are buffered per sequence and written only after the whole sequence succeeds;
  - hard-fail fallback keeps the reused image/crop/mask and `target_state_world` metadata synchronized.

## 4. Required Labels

Each frame record must include:

- `target_center_px`, `bbox_xywh`, `visibility`
- `land_overlap_ratio`, `shore_buffer_overlap_ratio`
- `scale_px`, `angle_deg`, `obs_valid`
- `meta.crop_center_world`, `meta.target_state_world`, `meta.gsd`, `meta.perception_stage`
- `meta.water_mask_crop_path`
- `meta.distractor_bboxes_xywh`

## 5. Automatic QC Gates

Rendering is accepted only when both checks pass, after the render job finishes:

1. `scripts/check_stage2_rendered.py`
2. `scripts/check_stage2_temporal_continuity.py`

Any hard-constraint violation causes `pass=false`.

Current formal QC defaults are aligned with `configs/render_stage2_c_v1.yaml`:

- semantic QC: `max_shore_overlap=0.01`, `max_truncation_ratio=0.15`
- temporal QC: `max_world_step_m=55`, `max_port_world_step_m=40`
