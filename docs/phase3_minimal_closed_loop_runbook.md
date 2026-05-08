# Phase 3 Minimal Closed-Loop Runbook

This runbook is the current priority path for Paper 2.

## What this is

Paper 2 is not redoing Paper 1. It replaces the known target coordinate input with a visual estimate:

1. dynamic target truth is generated;
2. an image is rendered or replayed for the current perception stage;
3. SNN/CNN localize the target center;
4. image coordinates are mapped to world coordinates;
5. KF may optionally smooth the estimate;
6. Paper1 SNN-TD3 replans online from that estimate;
7. capture is judged by the true target, not the estimate.

## Current ablation matrix

The closed-loop ablation should evaluate the contribution and stability of the visual estimation chain, not claim that one terminal filtering setting is dramatically better than another.

Use this final matrix:

- `A0 Oracle-GT + TD3`
- `A1 SNN + no-KF + TD3`
- `A2 SNN + KF/raw + TD3` (main method)
- `A3 SNN + full-KF + TD3`
- `A4 CNN + KF/raw + TD3` (architecture baseline, not an internal SNN ablation)

Each group answers a different question:

- `Oracle-GT + TD3`: upper bound with privileged true target state.
- `SNN + no-KF + TD3`: whether frame-wise visual localization alone can close the loop.
- `SNN + KF/raw + TD3`: whether stage-aware state estimation stabilizes estimates while preserving Oracle-level capture.
- `SNN + full-KF + TD3`: whether the terminal raw choice is a performance risk, and whether the system is sensitive to terminal filtering.
- `CNN + KF/raw + TD3`: layer-matched non-spiking visual baseline.

## Server run

Use `tmux`.

Note: the model files under `outputs/` are local artifacts and are not pushed by Git. Make sure the server already has the same files at the same paths, or copy them there before you start.

```bash
cd ~/projects/brain_uav_paper2
git pull
git log --oneline -1
conda activate paper2

mkdir -p logs/server_runs
tmux new -s paper2_phase3_minimal
```

Inside the tmux session:

```bash
cd ~/projects/brain_uav_paper2
conda activate paper2
export PYTHONPATH=src

python -B scripts/run_phase3_minimal_closed_loop_suite.py \
  --dataset-root /data_8T/clt/projects/brain_uav_paper2/data/rendered/phase3_map_main_no_port_no_distractor \
  --vision-source phase3_map_live \
  --vision-weights outputs/phase3_task_real_main/snn_enhanced_clean_noport_fast_v1/model_best.pth \
  --cnn-vision-weights outputs/phase3_task_real_main/cnn_enhanced_clean_noport_v1/model_best.pth \
  --td3-checkpoint outputs/paper1_method/models/td3_snn_hard.pt \
  --episodes 64 \
  --steps 1000 \
  --seed 20260430 \
  --capture-radius-km 5 \
  --paper1-curriculum-level hard \
  --phase3-target-init paper1_goal \
  --target-z-policy keep_current_goal_z \
  2>&1 | tee logs/server_runs/phase3_minimal_closed_loop.log
```

## Visual verification

Run the visual module separately. This is the independent SNN/CNN check.

SNN:

```bash
python -B scripts/eval_stage2_pilot_snn_heatmap.py \
  --dataset-root /data_8T/clt/projects/brain_uav_paper2/data/rendered/phase3_map_main_no_port_no_distractor \
  --eval-split test \
  --weights outputs/phase3_task_real_main/snn_enhanced_clean_noport_fast_v1/model_best.pth \
  --out-dir outputs/phase3_task_real_main/snn_enhanced_clean_noport_fast_v1_eval_test
```

CNN:

```bash
python -B scripts/eval_stage2_pilot_cnn.py \
  --dataset-root /data_8T/clt/projects/brain_uav_paper2/data/rendered/phase3_map_main_no_port_no_distractor \
  --eval-split test \
  --weights outputs/phase3_task_real_main/cnn_enhanced_clean_noport_v1/model_best.pth \
  --out-dir outputs/phase3_task_real_main/cnn_enhanced_clean_noport_v1_eval_test
```

## Figures and table

`run_phase3_minimal_closed_loop_suite.py` calls `plot_phase3_closed_loop_results.py` after the runs finish. The plotting script writes:

- `ablation_table.md`: paper-style ablation table in meters.
- `ablation_error_capture.png`: Figure A, mean estimation error with capture-rate marker.
- `paired_snn_no_kf_vs_kf_raw.png`: Figure B, paired per-episode SNN no-KF vs SNN KF/raw scatter.
- `representative_error_curve_ep*.png`: Figure C, range and estimation error over time for one representative episode.

Recommended table columns:

`Method`, `Capture`, `Valid Capture`, `Est. Err. (m)`, `Vision Err. (m)`, `Hard Viol.`, `Safety Margin`.

## What to write in the paper

- Filtering slightly improves SNN estimation accuracy, if the measured gain is small.
- The full-KF variant should be interpreted as a terminal filtering sensitivity check, not as a failed competitor.
- If full-KF and KF/raw are nearly identical, write that terminal raw does not degrade closed-loop performance and is kept as a conservative design to avoid potential terminal lag under highly maneuvering targets.
- If CNN has lower closed-loop mean estimation error in the saturated protocol, do not claim SNN wins by capture rate. Interpret the SNN advantage through the held-out visual benchmark, especially far-stage and difficult-background long-tail localization robustness.
- All capture claims must use true-target capture, not estimate-goal capture.
