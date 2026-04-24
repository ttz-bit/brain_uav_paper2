from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper2.datasets.seadronessee_dataset import build_seadronessee_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\Projects\brain_uav_paper2\data\processed\seadronessee",
        help="Processed public dataset root (SeaDronesSee for now)",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument(
        "--weights-path",
        type=str,
        required=True,
        help="Path to saved linear_weights.npy from train_public_vision.py",
    )
    parser.add_argument(
        "--norm-stats-path",
        type=str,
        default=None,
        help=(
            "Optional path to feature_norm_stats.npz. "
            "If omitted, try sibling file near weights, then fallback to current split stats."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "eval" / "public_vision"),
    )
    return parser.parse_args()


def _extract_feature(img_bgr: np.ndarray) -> np.ndarray:
    small = cv2.resize(img_bgr, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    feat = small.reshape(-1)
    return np.concatenate([feat, np.array([1.0], dtype=np.float32)], axis=0)


def _target_from_sample(sample) -> np.ndarray:
    h, w = sample.image.shape[:2]
    cx = float(sample.target_center[0]) / max(1.0, float(w))
    cy = float(sample.target_center[1]) / max(1.0, float(h))
    conf = 1.0 if bool(sample.valid) else 0.0
    return np.array([cx, cy, conf], dtype=np.float32)


def _load_norm_stats(
    x: np.ndarray, weights_path: Path, norm_stats_path: Path | None
) -> tuple[np.ndarray, np.ndarray, str]:
    chosen = norm_stats_path
    if chosen is None:
        sibling = weights_path.with_name("feature_norm_stats.npz")
        if sibling.exists():
            chosen = sibling

    if chosen is not None and chosen.exists():
        data = np.load(chosen)
        mean = np.asarray(data["mean"], dtype=np.float32)
        std = np.asarray(data["std"], dtype=np.float32)
        std = np.maximum(std, 1e-6)
        return mean, std, f"loaded:{chosen}"

    # Fallback: compute from the current evaluation split.
    # This keeps script robust for old runs that did not save norm stats.
    x_core = x[:, :-1]
    mean = np.mean(x_core, axis=0, keepdims=True)
    std = np.std(x_core, axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32), "computed_from_eval_split"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    w = np.load(weights_path)
    if w.ndim != 2:
        raise ValueError(f"weights must be 2D matrix, got shape={w.shape}")

    norm_path = Path(args.norm_stats_path).resolve() if args.norm_stats_path else None

    ds = build_seadronessee_dataset(
        root=args.root,
        split=args.split,
        project_root=project_root,
        max_samples=args.max_samples,
    )

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for i in range(len(ds)):
        s = ds[i]
        features.append(_extract_feature(s.image))
        targets.append(_target_from_sample(s))
    x_raw = np.stack(features, axis=0)
    y = np.stack(targets, axis=0)

    x = x_raw.copy()
    mean, std, norm_source = _load_norm_stats(x, weights_path, norm_path)
    if x.shape[1] > 1:
        x[:, :-1] = (x[:, :-1] - mean.reshape(1, -1)) / std.reshape(1, -1)

    if w.shape[0] != x.shape[1]:
        raise ValueError(f"Weight dim mismatch: weights={w.shape}, features={x.shape}")

    pred = x @ w
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))

    vis_count = min(8, len(ds))
    vis_indices = rng.choice(len(ds), size=vis_count, replace=False) if vis_count > 0 else []
    for rank, i in enumerate(vis_indices):
        s = ds[int(i)]
        feat = _extract_feature(s.image)
        if feat.shape[0] != x.shape[1]:
            raise ValueError("Feature dimension mismatch in visualization pass")
        feat[:-1] = (feat[:-1] - mean.reshape(-1)) / std.reshape(-1)
        one_pred = feat @ w
        if not np.isfinite(one_pred).all():
            one_pred = np.nan_to_num(one_pred, nan=0.0, posinf=1.0, neginf=0.0)

        h_img, w_img = s.image.shape[:2]
        gt_center_conf = _target_from_sample(s)
        pred_x = int(np.clip(one_pred[0], 0.0, 1.0) * w_img)
        pred_y = int(np.clip(one_pred[1], 0.0, 1.0) * h_img)
        pred_conf = float(np.clip(one_pred[2], 0.0, 1.0))
        gt_x = int(np.clip(gt_center_conf[0], 0.0, 1.0) * w_img)
        gt_y = int(np.clip(gt_center_conf[1], 0.0, 1.0) * h_img)
        gt_conf = float(np.clip(gt_center_conf[2], 0.0, 1.0))

        vis = s.image.copy()
        cv2.circle(vis, (gt_x, gt_y), 4, (0, 255, 0), -1)
        cv2.circle(vis, (pred_x, pred_y), 4, (0, 0, 255), -1)
        cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)
        cv2.putText(
            vis,
            (
                f"gt_center=({gt_x},{gt_y}) pred_center=({pred_x},{pred_y}) "
                f"gt_conf={gt_conf:.2f} pred_conf={pred_conf:.2f}"
            ),
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(vis_dir / f"sample_{rank:02d}.jpg"), vis)

    report = {
        "task": "eval_public_vision",
        "dataset": "SeaDronesSee",
        "split": args.split,
        "purpose": "evaluation",
        "num_samples": int(len(ds)),
        "feature_dim": int(x.shape[1]),
        "output_dim": int(w.shape[1]),
        "mse": mse,
        "mae": mae,
        "weights_path": str(weights_path),
        "normalization_source": norm_source,
        "artifacts": {
            "visual_dir": str(vis_dir),
            "report_path": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
