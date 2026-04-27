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

