# Phase 3A SNN Vision Freeze Note

Date: 2026-04-26

## Scope

This note freezes the current Phase 3A SNN public-vision training/evaluation status
for Paper2 using the server run summary.

## Summary Command

```bash
python scripts/summarize_phase3a_snn.py --config configs/phase3a_snn_runs.json --out-dir outputs/reports
```

## Overall Result

- phase: `phase3a_snn_summary`
- num_experiments: `4`
- all_overall_ok: `true`
- all_train_metrics_ok: `true`
- all_eval_metrics_ok: `true`
- all_artifacts_ok: `true`
- train_eval_separation_ok: `true`

Source:
- `outputs/reports/phase3a_snn_summary.json`
- `outputs/reports/phase3a_snn_summary.csv`

## By-Samples Aggregation

- `1024` samples:
  - num_experiments: `2`
  - all_ok: `true`
  - eval_mse_min/max: `0.0470198728 / 0.1215555668`
  - eval_mae_min/max: `0.1533902436 / 0.2252235413`
- `2048` samples:
  - num_experiments: `2`
  - all_ok: `true`
  - eval_mse_min/max: `0.0387878045 / 0.1001643762`
  - eval_mae_min/max: `0.1277103424 / 0.1901404113`

## Experiment Table (Server Run)

| experiment_id | seed | samples | num_steps | beta | initial_loss | final_loss | improve_ratio | eval_mse | eval_mae | eval_mse_xy | eval_mae_xy | overall_ok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| snn_1024_seed42 | 42 | 1024 | 8 | 0.95 | 100.125000 | 0.547945 | 0.994527 | 0.047020 | 0.153390 | 0.007175 | 0.058455 | true |
| snn_1024_seed7 | 7 | 1024 | 8 | 0.95 | 100.097649 | 1.043321 | 0.989577 | 0.121556 | 0.225224 | 0.005034 | 0.041632 | true |
| snn_2048_seed42 | 42 | 2048 | 8 | 0.95 | 100.125038 | 0.415020 | 0.995855 | 0.038788 | 0.127710 | 0.004169 | 0.034675 | true |
| snn_2048_seed7 | 7 | 2048 | 8 | 0.95 | 100.097687 | 0.753394 | 0.992473 | 0.100164 | 0.190140 | 0.001557 | 0.014153 | true |

## Acceptance Judgment

Phase 3A SNN summary is accepted under the current criteria:

1. Training fit checks pass (`final_loss < initial_loss`, `improve_ratio > 0`) for all runs.
2. Evaluation metrics are valid and non-negative for all runs.
3. Required artifacts are present for train/eval runs.
4. Train/evaluation purpose separation is valid.

## Notes

- This freeze note records the server result at:
  - project root: `/data_8T/clt/projects/brain_uav_paper2`
- If newer reruns are generated, update this note together with
  `outputs/reports/phase3a_snn_summary.{json,csv}`.
