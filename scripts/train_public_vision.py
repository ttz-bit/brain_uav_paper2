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
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "train" / "public_vision"),
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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    ds = build_seadronessee_dataset(
        root=args.root,
        split=args.split,
        project_root=args.project_root,
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

    # Keep final bias column unchanged, normalize other feature dims for stable optimization.
    x = x_raw.copy()
    if x.shape[1] > 1:
        x_core = x[:, :-1]
        mean = np.mean(x_core, axis=0, keepdims=True)
        std = np.std(x_core, axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)
        x[:, :-1] = (x_core - mean) / std

    n, d = x.shape
    out_dim = y.shape[1]
    w = rng.normal(0.0, 0.01, size=(d, out_dim)).astype(np.float32)

    def mse(pred: np.ndarray, gt: np.ndarray) -> float:
        return float(np.mean((pred - gt) ** 2))

    initial_loss = mse(x @ w, y)
    loss_trace: list[float] = []

    steps = max(1, int(args.steps))
    batch_size = max(1, int(args.batch_size))
    lr = float(args.learning_rate)

    for step in range(steps):
        idx = rng.integers(0, n, size=min(batch_size, n))
        xb = x[idx]
        yb = y[idx]
        pred = xb @ w
        err = pred - yb
        grad = (2.0 / float(len(xb))) * (xb.T @ err)
        grad = np.clip(grad, -float(args.grad_clip), float(args.grad_clip))
        w = w - lr * grad.astype(np.float32)
        loss_trace.append(mse(x @ w, y))
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[SMOKE] step={step + 1:03d}/{steps}, loss={loss_trace[-1]:.6f}")

    final_loss = mse(x @ w, y)
    improve_ratio = (initial_loss - final_loss) / max(initial_loss, 1e-12)

    vis_count = min(8, len(ds))
    vis_indices = rng.choice(len(ds), size=vis_count, replace=False) if vis_count > 0 else []
    for rank, i in enumerate(vis_indices):
        s = ds[int(i)]
        feat = _extract_feature(s.image)
        if x.shape[1] > 1:
            feat[:-1] = (feat[:-1] - mean.reshape(-1)) / std.reshape(-1)
        pred = feat @ w
        if not np.isfinite(pred).all():
            pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

        h_img, w_img = s.image.shape[:2]
        gt_center_conf = _target_from_sample(s)
        pred_x = int(np.clip(pred[0], 0.0, 1.0) * w_img)
        pred_y = int(np.clip(pred[1], 0.0, 1.0) * h_img)
        pred_conf = float(np.clip(pred[2], 0.0, 1.0))
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
        "task": "train_public_vision",
        "dataset": "SeaDronesSee",
        "split": args.split,
        "num_samples": int(n),
        "feature_dim": int(d),
        "output_dim": int(out_dim),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "learning_rate": float(lr),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "improve_ratio": float(improve_ratio),
        "artifacts": {
            "visual_dir": str(vis_dir),
            "weights_path": str(out_dir / "linear_weights.npy"),
            "report_path": str(out_dir / "report.json"),
        },
    }
    np.save(out_dir / "linear_weights.npy", w)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
