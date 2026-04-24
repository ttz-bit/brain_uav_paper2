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
        help="Processed SeaDronesSee root",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--weights-path", type=str, required=True, help="Path to model.pth from train_public_vision_cnn.py")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "eval" / "public_vision_cnn"),
    )
    return parser.parse_args()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as e:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "PyTorch is required for eval_public_vision_cnn.py. "
            "Please install torch in your server environment."
        ) from e
    return torch, nn


def _to_tensor_image(img_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def _target_from_sample(sample) -> np.ndarray:
    h, w = sample.image.shape[:2]
    cx = float(sample.target_center[0]) / max(1.0, float(w))
    cy = float(sample.target_center[1]) / max(1.0, float(h))
    conf = 1.0 if bool(sample.valid) else 0.0
    return np.array([cx, cy, conf], dtype=np.float32)


def _make_visual(img_bgr: np.ndarray, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    pred_x = int(np.clip(pred[0], 0.0, 1.0) * w_img)
    pred_y = int(np.clip(pred[1], 0.0, 1.0) * h_img)
    pred_conf = float(np.clip(pred[2], 0.0, 1.0))
    gt_x = int(np.clip(gt[0], 0.0, 1.0) * w_img)
    gt_y = int(np.clip(gt[1], 0.0, 1.0) * h_img)
    gt_conf = float(np.clip(gt[2], 0.0, 1.0))

    vis = img_bgr.copy()
    cv2.circle(vis, (gt_x, gt_y), 4, (0, 255, 0), -1)
    cv2.circle(vis, (pred_x, pred_y), 4, (0, 0, 255), -1)
    cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)
    cv2.putText(
        vis,
        (
            f"gt=({gt_x},{gt_y}) pred=({pred_x},{pred_y}) "
            f"gt_c={gt_conf:.2f} pred_c={pred_conf:.2f}"
        ),
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def main() -> None:
    args = parse_args()
    torch, nn = _import_torch()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cpu"
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = "cuda"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
            z = self.backbone(x)
            return torch.sigmoid(self.head(z))

    model = _SmallCNN().to(device)
    weights_path = Path(args.weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    ckpt = torch.load(weights_path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain key: state_dict")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            x = torch.from_numpy(_to_tensor_image(s.image)).unsqueeze(0).to(device)
            pred = model(x)[0].detach().cpu().numpy()
            gt = _target_from_sample(s)
            preds.append(pred.astype(np.float32))
            gts.append(gt.astype(np.float32))

    pred_arr = np.stack(preds, axis=0)
    gt_arr = np.stack(gts, axis=0)
    mse = float(np.mean((pred_arr - gt_arr) ** 2))
    mae = float(np.mean(np.abs(pred_arr - gt_arr)))
    mse_xy = float(np.mean((pred_arr[:, :2] - gt_arr[:, :2]) ** 2))
    mae_xy = float(np.mean(np.abs(pred_arr[:, :2] - gt_arr[:, :2])))

    vis_count = min(8, len(ds))
    vis_indices = rng.choice(len(ds), size=vis_count, replace=False) if vis_count > 0 else []
    for rank, idx in enumerate(vis_indices):
        s = ds[int(idx)]
        vis = _make_visual(s.image, pred_arr[int(idx)], gt_arr[int(idx)])
        cv2.imwrite(str(vis_dir / f"sample_{rank:02d}.jpg"), vis)

    report = {
        "task": "eval_public_vision_cnn",
        "dataset": "SeaDronesSee",
        "split": args.split,
        "purpose": "evaluation",
        "device": device,
        "num_samples": int(len(ds)),
        "mse": mse,
        "mae": mae,
        "mse_xy": mse_xy,
        "mae_xy": mae_xy,
        "weights_path": str(weights_path),
        "artifacts": {
            "visual_dir": str(vis_dir),
            "report_path": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
