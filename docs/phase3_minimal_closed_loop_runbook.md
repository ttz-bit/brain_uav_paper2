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

## Current recommendation

Do not wait for the full real-time renderer to finish before running the paper.
Use the minimal closed-loop suite first:

- `Oracle true target`
- `SNN no KF`
- `SNN + KF terminal raw`
- `SNN + KF terminal pure pursuit`

The first two are the key comparison. The KF variants are ablations.

## Server run

Use `tmux`.

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
  --dataset-root /data_8T/clt/projects/brain_uav_paper2/data/rendered/paper2_task_v1.0.0_real_main \
  --vision-weights outputs/debug/snn_heatmap_overfit64_v2_softargmax/model_best.pth \
  --td3-checkpoint outputs/paper1_method/models/td3_snn_hard.pt \
  --episodes 16 \
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
  --dataset-root /data_8T/clt/projects/brain_uav_paper2/data/rendered/paper2_task_v1.0.0_real_main \
  --eval-split test \
  --weights outputs/debug/snn_heatmap_overfit64_v2_softargmax/model_best.pth \
  --out-dir outputs/phase3_task_real_main/snn_eval_test
```

CNN:

```bash
python -B scripts/eval_stage2_pilot_cnn.py \
  --dataset-root /data_8T/clt/projects/brain_uav_paper2/data/rendered/paper2_task_v1.0.0_real_main \
  --eval-split test \
  --weights outputs/phase3_task_real_main/cnn_baseline_stream_bs8/model_best.pth \
  --out-dir outputs/phase3_task_real_main/cnn_eval_test
```

## What to write in the paper

- visual stage error by `far/mid/terminal`;
- SNN vs CNN on the same rendered test set;
- closed-loop comparison with true-target capture;
- KF as a negative or weak ablation if it does not help;
- terminal controller as a small ablation, not the main story.
