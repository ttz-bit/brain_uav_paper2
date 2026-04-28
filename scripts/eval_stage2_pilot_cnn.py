from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_pilot_v6")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--weights",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "stage2_pilot_baselines"
            / "cnn_fit"
            / "model_best.pth"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "stage2_pilot_baselines" / "cnn_eval"),
    )
    return p.parse_args()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as e:
        raise RuntimeError("PyTorch is required. Install torch in your environment.") from e
    return torch, nn, F


def _to_tensor_image(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def _target_from_sample(sample) -> np.ndarray:
    h, w = sample.image.shape[:2]
    cx = float(sample.target_center[0]) / max(1.0, float(w))
    cy = float(sample.target_center[1]) / max(1.0, float(h))
    conf = 1.0 if bool(sample.valid) else 0.0
    return np.array([cx, cy, conf], dtype=np.float32)


def _pixel_error(pred: np.ndarray, gt: np.ndarray, h: int, w: int) -> float:
    px = float(np.clip(pred[0], 0.0, 1.0) * w)
    py = float(np.clip(pred[1], 0.0, 1.0) * h)
    gx = float(np.clip(gt[0], 0.0, 1.0) * w)
    gy = float(np.clip(gt[1], 0.0, 1.0) * h)
    return float(np.hypot(px - gx, py - gy))


def _make_visual(img_bgr: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    pred_x = int(np.clip(pred[0], 0.0, 1.0) * w_img)
    pred_y = int(np.clip(pred[1], 0.0, 1.0) * h_img)
    gt_x = int(np.clip(gt[0], 0.0, 1.0) * w_img)
    gt_y = int(np.clip(gt[1], 0.0, 1.0) * h_img)
    vis = img_bgr.copy()
    cv2.circle(vis, (gt_x, gt_y), 4, (0, 255, 0), -1)
    cv2.circle(vis, (pred_x, pred_y), 4, (0, 0, 255), -1)
    cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)
    cv2.putText(vis, f"gt=({gt_x},{gt_y}) pred=({pred_x},{pred_y})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main() -> None:
    args = parse_args()
    torch, nn, F = _import_torch()

    device = "cpu"
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = "cuda"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_root = Path(args.dataset_root).resolve()
    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    vis_dir = out_dir / "visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    ds = build_stage2_rendered_dataset(
        root=dataset_root,
        split=args.eval_split,
        project_root=project_root,
        max_samples=args.max_eval_samples,
    )

    class _SmallCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            return torch.sigmoid(self.head(self.backbone(x)))

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")

    model = _SmallCNN().to(device)
    ckpt = torch.load(weights, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    losses: list[float] = []
    px_errors: list[float] = []
    conf_mae: list[float] = []
    per_stage: dict[str, list[float]] = {"far": [], "mid": [], "terminal": []}
    rows_for_report: list[dict] = []

    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x_np = _to_tensor_image(s.image)
            gt = _target_from_sample(s)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            pred = model(x)[0].detach().cpu().numpy()

            pred_xy = torch.tensor(pred[:2], dtype=torch.float32)
            gt_xy = torch.tensor(gt[:2], dtype=torch.float32)
            pred_c = torch.tensor(pred[2], dtype=torch.float32)
            gt_c = torch.tensor(gt[2], dtype=torch.float32)
            loss = float(F.smooth_l1_loss(pred_xy, gt_xy).item() + F.binary_cross_entropy(pred_c, gt_c).item())
            losses.append(loss)

            h, w = s.image.shape[:2]
            err = _pixel_error(pred, gt, h, w)
            px_errors.append(err)
            conf_mae.append(float(abs(pred[2] - gt[2])))

            st = str(s.meta.get("perception_stage", "unknown"))
            if st in per_stage:
                per_stage[st].append(err)

            if i < 16:
                vis = _make_visual(s.image, pred, gt)
                cv2.imwrite(str(vis_dir / f"eval_sample_{i:02d}.jpg"), vis)

            if i < 200:
                rows_for_report.append(
                    {
                        "sequence_id": str(s.sequence_id),
                        "frame_id": str(s.frame_id),
                        "stage": st,
                        "pixel_error": err,
                    }
                )

    report = {
        "task": "eval_stage2_pilot_cnn",
        "purpose": "generalization_eval",
        "dataset": {
            "name": "stage2_rendered",
            "root": str(dataset_root),
            "eval_split": str(args.eval_split),
            "num_eval": int(len(ds)),
        },
        "separation_check": {
            "eval_split": str(args.eval_split),
            "trained_train_split": str(ckpt.get("train_split", "unknown")),
            "trained_val_split": str(ckpt.get("val_split", "unknown")),
            "train_eval_disjoint_expected": bool(str(args.eval_split) == "test"),
        },
        "weights_path": str(weights),
        "device": str(device),
        "metrics": {
            "eval_loss_mean": float(np.mean(losses)) if losses else 0.0,
            "eval_loss_p90": float(np.percentile(losses, 90)) if losses else 0.0,
            "pixel_error_mean": float(np.mean(px_errors)) if px_errors else 0.0,
            "pixel_error_p90": float(np.percentile(px_errors, 90)) if px_errors else 0.0,
            "conf_mae_mean": float(np.mean(conf_mae)) if conf_mae else 0.0,
            "stage_pixel_error_mean": {
                k: (float(np.mean(v)) if v else 0.0) for k, v in per_stage.items()
            },
        },
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "sample_errors_path": str(out_dir / "sample_errors_head.json"),
            "visual_dir": str(vis_dir),
        },
    }

    (out_dir / "sample_errors_head.json").write_text(json.dumps(rows_for_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

