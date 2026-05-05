# brain_uav_paper2

Independent experiment codebase for Paper 2 of the brain-inspired UAV project.

## Overview

This repository focuses on Paper 2 visual perception and data pipeline work while keeping a frozen bridge interface to the corrected Paper 1 environment.

Current focus:

- keep Paper 1 interface/protocol stable;
- build public-vision external validation pipeline;
- prepare SeaDronesSee into unified manifest format for training;
- generate and freeze the Stage 2C formal rendered dataset for Paper 2;
- keep closed-loop bridge as placeholder until Paper 1 physical calibration is finalized.

## Current Milestone Status

- Phase 1A: interface/protocol freeze in place.
- Phase 1B: SeaDronesSee external data pipeline implemented and accepted.
  - preprocessing done for `train/val`;
  - unified schema validation enabled;
  - split leakage / QC / mismatch checks implemented;
  - training entry script for public vision available.
- Phase 3A (SNN public vision): training/evaluation summary accepted on server.
  - summary gate: `all_overall_ok=true`, `all_artifacts_ok=true`, `train_eval_separation_ok=true`;
  - frozen record: `docs/phase3a_snn_freeze_note.md`;
  - summary artifacts: `outputs/reports/phase3a_snn_summary.json` and `.csv`.
- Stage 2C formal rendering: active full-dataset generation stage.
  - current trusted rerender target: `data/rendered/paper2_render_v1.0.1`;
  - config: `configs/render_stage2_c_v1.yaml` with explicit `--out-root` override for v1.0.1;
  - renderer now uses strict no-bad-frame write policy, buffered per-sequence label writes, fallback world-state synchronization, final world-step guard, sequence retry, and cached per-frame semantic masks;
  - labels include per-frame `water_mask_crop_path` and `distractor_bboxes_xywh` for auxiliary supervision;
  - recommended workflow is render first, then run QC and temporal checks separately;
  - operational runbook: `docs/stage2c_formal_runbook.md`.

Not finalized yet:

- final Paper 1 physical world-unit and dynamics calibration;
- final closed-loop bridge implementation;
- final Stage 2C freeze report after semantic + temporal QC pass;
- final SNN/CNN vision training on the frozen formal rendered dataset.
- phase 3 minimal closed-loop suite and paper-2-specific runbook are in `docs/phase3_minimal_closed_loop_runbook.md`.

## Key Paths

- `scripts/prepare_seadronessee.py`: convert raw SeaDronesSee to processed crops + manifests.
- `scripts/check_phase1b_seadronessee.py`: phase 1B acceptance checks.
- `scripts/train_public_vision.py`: baseline public vision training entry.
- `scripts/train_public_vision_snn.py`: phase3a SNN training entry.
- `scripts/eval_public_vision_snn.py`: phase3a SNN evaluation entry.
- `scripts/summarize_phase3a_snn.py`: phase3a SNN summary + acceptance check.
- `scripts/render_stage2_smoke.py`: Stage 2 renderer entry; used for both smoke and full Stage 2C formal rendering.
- `scripts/check_stage2_rendered.py`: Stage 2 semantic/QC gate.
- `scripts/check_stage2_temporal_continuity.py`: Stage 2 temporal-continuity gate.
- `scripts/inspect_stage2_temporal_violations.py`: detailed temporal violation locator.
- `scripts/make_stage2_stage_background_sheets.py`: visual contact sheets by split/stage/background.
- `src/paper2/datasets/unified_schema.py`: manifest schema and hard validation.
- `src/paper2/datasets/seadronessee_dataset.py`: dataset reader using crop-space labels.
- `src/paper2/env_adapter/interfaces.py`: frozen environment protocol.
- `src/paper2/env_adapter/paper1_bridge.py`: placeholder bridge (kept intentionally unimplemented).

## Phase 1B Data Format Notes

Each manifest record includes both original-image and crop-space target fields.

Core fields:

- original-image space: `center_px`, `bbox_xywh`
- crop space: `center_px_crop`, `bbox_xywh_crop`
- crop geometry: `crop_origin_xy`, `crop_box_xyxy`, `crop_size`

The training reader consumes crop-space labels by default.

## Quick Start (Server)

```bash
cd ~/projects/brain_uav_paper2
conda activate paper2
export PYTHONPATH=src
```

Preprocess:

```bash
python scripts/prepare_seadronessee.py \
  --raw-root /data_8T/clt/SeaDronesSee \
  --out-root data/processed/seadronessee \
  --splits train val \
  --overwrite
```

Check:

```bash
python scripts/check_phase1b_seadronessee.py \
  --root data/processed/seadronessee \
  --splits train val \
  --allow-length-mismatch
```

Train baseline:

```bash
python scripts/train_public_vision.py \
  --root data/processed/seadronessee \
  --split train \
  --project-root ~/projects/brain_uav_paper2 \
  --max-samples 4096 \
  --batch-size 64 \
  --steps 400 \
  --learning-rate 0.001 \
  --grad-clip 0.5 \
  --out-dir outputs/train/public_vision/run_example
```

## Stage 2C Formal Render Quick Start

Use the split workflow for formal runs: render first, then QC. Keep the render task in `tmux`.

```bash
cd ~/projects/brain_uav_paper2
git pull
git log --oneline -1
conda activate paper2
rm -rf data/rendered/paper2_render_v1.0.1
```

Render only:

```bash
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

After rendering completes, run QC separately:

```bash
python -B scripts/check_stage2_rendered.py \
  --dataset-root data/rendered/paper2_render_v1.0.1 \
  --max-center-bias-ratio 0.95

python -B scripts/check_stage2_temporal_continuity.py \
  --dataset-root data/rendered/paper2_render_v1.0.1 \
  --max-center-step-px 20 \
  --max-world-step-m 55 \
  --max-port-world-step-m 40 \
  --max-scale-change-ratio 0.05 \
  --max-angle-change-deg 12 \
  --max-crop-step-px 38.4
```

Formal freeze requires both reports to show `pass=true`.
