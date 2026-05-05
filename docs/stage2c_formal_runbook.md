# Stage 2C Formal Render Runbook

This runbook reflects the current formal Stage 2C workflow.

## Goal

Generate `data/rendered/paper2_render_v1.0.1`, then validate it with semantic and temporal QC as separate steps.

`configs/render_stage2_c_v1.yaml` still carries the historical dataset name, so the formal rerender must pass `--out-root data/rendered/paper2_render_v1.0.1` explicitly.

## Current renderer behavior

- strict no-bad-frame write policy;
- per-sequence labels are buffered in memory and written only after the full sequence succeeds;
- hard-fail fallback synchronizes image pixels, crop center, water mask, target state, and target world-state metadata;
- final world-step hard guard;
- cached per-frame semantic masks;
- low PNG compression for faster writes;
- sequence-level retry when the first frame of a candidate sequence is infeasible.
- frame labels include `meta.water_mask_crop_path` and `meta.distractor_bboxes_xywh` for auxiliary water-logit and distractor-repel supervision.

## Recommended execution order

1. Pull the latest code.
2. Remove any partial formal dataset.
3. Run render only inside `tmux`.
4. Run semantic QC after render completes.
5. Run temporal QC after semantic QC passes.
6. Generate contact sheets and archive reports.

## Render only

```bash
cd ~/projects/brain_uav_paper2
git pull
git log --oneline -1
rm -rf data/rendered/paper2_render_v1.0.1
mkdir -p logs/server_runs

tmux new -s paper2_render_full_v101
```

Inside the tmux session:

```bash
cd ~/projects/brain_uav_paper2
conda activate paper2

python -B scripts/render_stage2_smoke.py \
  --config configs/render_stage2_c_v1.yaml \
  --inventory-csv data/assets/stage2/asset_inventory.csv \
  --out-root data/rendered/paper2_render_v1.0.1 \
  2>&1 | tee logs/server_runs/render_stage2_v101.log
```

## Semantic QC

```bash
python -B scripts/check_stage2_rendered.py \
  --dataset-root data/rendered/paper2_render_v1.0.1 \
  --max-center-bias-ratio 0.95
```

## Temporal QC

```bash
python -B scripts/check_stage2_temporal_continuity.py \
  --dataset-root data/rendered/paper2_render_v1.0.1 \
  --max-center-step-px 20 \
  --max-world-step-m 55 \
  --max-port-world-step-m 40 \
  --max-scale-change-ratio 0.05 \
  --max-angle-change-deg 12 \
  --max-crop-step-px 38.4
```

## Debug helper

Use this only when temporal QC fails and you need exact violating sequence/frame pairs:

```bash
python -B scripts/inspect_stage2_temporal_violations.py \
  --dataset-root data/rendered/paper2_render_v1.0.1
```

## Acceptance target

- `check_stage2_rendered.py`: `pass=true`
- `check_stage2_temporal_continuity.py`: `pass=true`
- `world_step_violations=0`
- `max_world_step_m=55`, `max_port_world_step_m=40`
- water-mask metadata complete for any run using `--water-logit-constraint` or `--land-penalty-weight > 0`
- distractor bbox metadata available before using `--distractor-repel-weight > 0`
- no asset leakage across splits
