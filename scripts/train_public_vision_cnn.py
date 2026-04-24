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
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "train" / "public_vision_cnn"),
    )
    return parser.parse_args()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:  # pragma: no cover - runtime environment dependent
        raise RuntimeError(
            "PyTorch is required for train_public_vision_cnn.py. "
            "Please install torch in your server environment."
        ) from e
    return torch, nn, F, DataLoader, Dataset


def _to_tensor_image(img_bgr: np.ndarray) -> np.ndarray:
    # Convert BGR uint8 [H,W,C] to RGB float32 [C,H,W] in [0,1].
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
    torch, nn, F, DataLoader, Dataset = _import_torch()
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

    images: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for i in range(len(ds)):
        s = ds[i]
        images.append(_to_tensor_image(s.image))
        targets.append(_target_from_sample(s))

    x_np = np.stack(images, axis=0).astype(np.float32)
    y_np = np.stack(targets, axis=0).astype(np.float32)

    class _NumpyVisionDataset(Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = x
            self.y = y

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, idx: int):
            return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])

    loader = DataLoader(
        _NumpyVisionDataset(x_np, y_np),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        drop_last=False,
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
            out = self.head(z)
            # Keep outputs in [0,1] range for normalized (x,y,conf).
            return torch.sigmoid(out)

    model = _SmallCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    def _batch_loss(pred, target):
        pred_xy = pred[:, :2]
        gt_xy = target[:, :2]
        pred_c = pred[:, 2]
        gt_c = target[:, 2]
        # xy: smooth L1 regression, conf: BCE.
        loss_xy = F.smooth_l1_loss(pred_xy, gt_xy)
        loss_c = F.binary_cross_entropy(pred_c, gt_c)
        return loss_xy + loss_c

    x_all = torch.from_numpy(x_np).to(device)
    y_all = torch.from_numpy(y_np).to(device)

    with torch.no_grad():
        model.eval()
        initial_pred = model(x_all)
        initial_loss = float(_batch_loss(initial_pred, y_all).item())

    loss_trace: list[float] = []
    epochs = max(1, int(args.epochs))
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = _batch_loss(pred, yb)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            pred_all = model(x_all)
            epoch_loss = float(_batch_loss(pred_all, y_all).item())
            loss_trace.append(epoch_loss)
        print(f"[CNN] epoch={epoch + 1:03d}/{epochs}, loss={epoch_loss:.6f}")

    final_loss = float(loss_trace[-1]) if loss_trace else initial_loss
    improve_ratio = (initial_loss - final_loss) / max(initial_loss, 1e-12)

    vis_count = min(8, len(ds))
    vis_indices = rng.choice(len(ds), size=vis_count, replace=False) if vis_count > 0 else []
    with torch.no_grad():
        model.eval()
        for rank, idx in enumerate(vis_indices):
            s = ds[int(idx)]
            img_tensor = torch.from_numpy(_to_tensor_image(s.image)).unsqueeze(0).to(device)
            pred = model(img_tensor)[0].detach().cpu().numpy()
            gt = _target_from_sample(s)
            vis = _make_visual(s.image, pred, gt)
            cv2.imwrite(str(vis_dir / f"sample_{rank:02d}.jpg"), vis)

    weights_path = out_dir / "model.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": int(args.seed),
            "split": args.split,
            "num_samples": int(len(ds)),
        },
        weights_path,
    )

    report = {
        "task": "train_public_vision_cnn",
        "dataset": "SeaDronesSee",
        "split": args.split,
        "purpose": "fit_check",
        "device": device,
        "num_samples": int(len(ds)),
        "batch_size": int(args.batch_size),
        "epochs": int(epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "initial_loss": float(initial_loss),
        "final_loss": float(final_loss),
        "improve_ratio": float(improve_ratio),
        "artifacts": {
            "visual_dir": str(vis_dir),
            "weights_path": str(weights_path),
            "report_path": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
