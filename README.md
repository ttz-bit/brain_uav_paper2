# brain_uav_paper2

Independent experiment codebase for Paper 2 of the brain-inspired UAV project.

## Overview

This repository focuses on Paper 2 visual perception and data pipeline work while keeping a frozen bridge interface to the corrected Paper 1 environment.

Current focus:

- keep Paper 1 interface/protocol stable;
- build public-vision external validation pipeline;
- prepare SeaDronesSee into unified manifest format for training;
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

Not finalized yet:

- final Paper 1 physical world-unit and dynamics calibration;
- final closed-loop bridge implementation;
- final high-performance vision model (current train script is baseline).

## Key Paths

- `scripts/prepare_seadronessee.py`: convert raw SeaDronesSee to processed crops + manifests.
- `scripts/check_phase1b_seadronessee.py`: phase 1B acceptance checks.
- `scripts/train_public_vision.py`: baseline public vision training entry.
- `scripts/train_public_vision_snn.py`: phase3a SNN training entry.
- `scripts/eval_public_vision_snn.py`: phase3a SNN evaluation entry.
- `scripts/summarize_phase3a_snn.py`: phase3a SNN summary + acceptance check.
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
