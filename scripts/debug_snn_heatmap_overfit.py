from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--split", choices=["train", "val", "test"], default="train")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-steps", type=int, default=12)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--heatmap-size", type=int, default=64)
    p.add_argument("--heatmap-sigma", type=float, default=1.5)
    p.add_argument("--heatmap-weight", type=float, default=1.0)
    p.add_argument("--coord-weight", type=float, default=5.0)
    p.add_argument("--conf-weight", type=float, default=0.2)
    p.add_argument("--softargmax-temperature", type=float, default=20.0)
    p.add_argument("--train-encoding", choices=["rate", "direct"], default="direct")
    p.add_argument("--eval-encoding", choices=["rate", "direct"], default="direct")
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/debug/snn_heatmap_overfit64")
    return p.parse_args()


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch and snntorch are required for SNN debug.") from exc
    return torch, DataLoader, Dataset


def _to_tensor_image(img_bgr: np.ndarray, input_size: int) -> np.ndarray:
    if int(input_size) > 0:
        sz = int(input_size)
        h, w = img_bgr.shape[:2]
        if h != sz or w != sz:
            img_bgr = cv2.resize(img_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def _target_from_sample(sample) -> np.ndarray:
    h, w = sample.image.shape[:2]
    cx = float(sample.target_center[0]) / max(1.0, float(w))
    cy = float(sample.target_center[1]) / max(1.0, float(h))
    conf = 1.0 if bool(sample.valid) else 0.0
    return np.array([cx, cy, conf], dtype=np.float32)


def _pixel_error_norm(pred_xy: np.ndarray, gt_xy: np.ndarray, h: int, w: int) -> float:
    px = float(np.clip(pred_xy[0], 0.0, 1.0) * w)
    py = float(np.clip(pred_xy[1], 0.0, 1.0) * h)
    gx = float(np.clip(gt_xy[0], 0.0, 1.0) * w)
    gy = float(np.clip(gt_xy[1], 0.0, 1.0) * h)
    return float(np.hypot(px - gx, py - gy))


def _make_visual(img_bgr: np.ndarray, pred_xy: np.ndarray, gt: np.ndarray, err: float) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    pred_x = int(np.clip(pred_xy[0], 0.0, 1.0) * w_img)
    pred_y = int(np.clip(pred_xy[1], 0.0, 1.0) * h_img)
    gt_x = int(np.clip(gt[0], 0.0, 1.0) * w_img)
    gt_y = int(np.clip(gt[1], 0.0, 1.0) * h_img)
    vis = img_bgr.copy()
    cv2.circle(vis, (gt_x, gt_y), 4, (0, 255, 0), -1)
    cv2.circle(vis, (pred_x, pred_y), 4, (0, 0, 255), -1)
    cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)
    cv2.putText(
        vis,
        f"gt=({gt_x},{gt_y}) pred=({pred_x},{pred_y}) err={err:.1f}px",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def _materialize(ds: Any, input_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in range(len(ds)):
        sample = ds[i]
        xs.append(_to_tensor_image(sample.image, input_size=input_size))
        ys.append(_target_from_sample(sample))
    return np.stack(xs, axis=0).astype(np.float32), np.stack(ys, axis=0).astype(np.float32)


def _tensor_diag_to_float(diag: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in diag.items():
        if hasattr(value, "detach"):
            out[str(key)] = float(value.detach().cpu().item())
        else:
            out[str(key)] = float(value)
    return out


def main() -> None:
    args = parse_args()
    torch, DataLoader, Dataset = _import_torch()
    from paper2.models.snn_heatmap import HeatmapSNN, heatmap_loss, peak_argmax_2d

    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

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
        split=str(args.split),
        project_root=project_root,
        max_samples=max(1, int(args.samples)),
    )
    x_np, y_np = _materialize(ds, input_size=int(args.input_size))

    class _NumpyDataset(Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = x
            self.y = y

        def __len__(self) -> int:
            return int(self.x.shape[0])

        def __getitem__(self, idx: int):
            return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])

    loader = DataLoader(
        _NumpyDataset(x_np, y_np),
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        drop_last=False,
    )
    model = HeatmapSNN(
        beta=float(args.beta),
        num_steps=int(args.num_steps),
        train_encoding=str(args.train_encoding),
        eval_encoding=str(args.eval_encoding),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    def _loss(outputs, targets):
        return heatmap_loss(
            outputs,
            targets,
            heatmap_size=int(args.heatmap_size),
            sigma=float(args.heatmap_sigma),
            coord_weight=float(args.coord_weight),
            heatmap_weight=float(args.heatmap_weight),
            conf_weight=float(args.conf_weight),
            softargmax_temperature=float(args.softargmax_temperature),
        )

    def _eval() -> dict[str, Any]:
        model.eval()
        losses: list[float] = []
        px_errors: list[float] = []
        pred_x_px: list[float] = []
        pred_y_px: list[float] = []
        diag_acc: dict[str, list[float]] = {}
        with torch.no_grad():
            for s in range(0, int(x_np.shape[0]), max(1, int(args.batch_size))):
                e = min(s + max(1, int(args.batch_size)), int(x_np.shape[0]))
                xb = torch.from_numpy(x_np[s:e]).to(device)
                yb = torch.from_numpy(y_np[s:e]).to(device)
                outputs = model(xb, stochastic=False, return_diagnostics=True)
                loss, _ = _loss(outputs, yb)
                losses.append(float(loss.detach().cpu().item()) * int(e - s))
                pred_xy = peak_argmax_2d(outputs["heatmap_logits"]).detach().cpu().numpy()
                for pxy, gxy in zip(pred_xy, y_np[s:e, :2]):
                    px_errors.append(_pixel_error_norm(pxy, gxy, int(args.input_size), int(args.input_size)))
                    pred_x_px.append(float(np.clip(pxy[0], 0.0, 1.0) * int(args.input_size)))
                    pred_y_px.append(float(np.clip(pxy[1], 0.0, 1.0) * int(args.input_size)))
                for key, value in _tensor_diag_to_float(outputs["diagnostics"]).items():
                    diag_acc.setdefault(key, []).append(value)
        rounded_unique = {(int(round(x)), int(round(y))) for x, y in zip(pred_x_px, pred_y_px)}
        diag = {k: float(np.mean(v)) for k, v in diag_acc.items()}
        return {
            "loss": float(sum(losses) / max(1, int(x_np.shape[0]))),
            "pixel_error_mean": float(np.mean(px_errors)) if px_errors else 0.0,
            "pixel_error_p90": float(np.percentile(px_errors, 90)) if px_errors else 0.0,
            "rounded_unique_pred_xy": int(len(rounded_unique)),
            "pred_x_std_px": float(np.std(pred_x_px)) if pred_x_px else 0.0,
            "pred_y_std_px": float(np.std(pred_y_px)) if pred_y_px else 0.0,
            "diagnostics": diag,
        }

    initial = _eval()
    best = dict(initial)
    best_epoch = 0
    best_path = out_dir / "model_best.pth"
    trace: list[dict[str, Any]] = [{"epoch": 0, "metrics": initial}]
    epochs = max(1, int(args.epochs))
    eval_interval = max(1, int(args.eval_interval))

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(xb, stochastic=True)
            loss, _ = _loss(outputs, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        if epoch % eval_interval == 0 or epoch == epochs:
            metrics = _eval()
            metrics["train_batch_loss_mean"] = float(np.mean(train_losses)) if train_losses else 0.0
            trace.append({"epoch": int(epoch), "metrics": metrics})
            if float(metrics["pixel_error_mean"]) < float(best["pixel_error_mean"]):
                best = dict(metrics)
                best_epoch = int(epoch)
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_type": "snn_heatmap",
                        "purpose": "small_sample_overfit_debug",
                        "dataset_root": str(dataset_root),
                        "split": str(args.split),
                        "samples": int(len(ds)),
                        "input_size": int(args.input_size),
                        "num_steps": int(args.num_steps),
                        "beta": float(args.beta),
                        "train_encoding": str(args.train_encoding),
                        "eval_encoding": str(args.eval_encoding),
                    },
                    best_path,
                )
            print(
                "[SNN-OVERFIT] "
                f"epoch={epoch:03d}/{epochs} "
                f"loss={metrics['loss']:.6f} "
                f"px={metrics['pixel_error_mean']:.2f} "
                f"unique={metrics['rounded_unique_pred_xy']} "
                f"logit_std={metrics['diagnostics'].get('heatmap_logit_std', 0.0):.4f} "
                f"spk3={metrics['diagnostics'].get('spike_rate_l3', 0.0):.4f}"
            )

    model.load_state_dict(torch.load(best_path, map_location=device)["state_dict"])
    final = _eval()

    with torch.no_grad():
        model.eval()
        for rank, idx in enumerate(rng.choice(len(ds), size=min(16, len(ds)), replace=False)):
            sample = ds[int(idx)]
            x = torch.from_numpy(_to_tensor_image(sample.image, int(args.input_size))).unsqueeze(0).to(device)
            outputs = model(x, stochastic=False)
            pred_xy = peak_argmax_2d(outputs["heatmap_logits"])[0].detach().cpu().numpy()
            gt = _target_from_sample(sample)
            err = _pixel_error_norm(pred_xy, gt[:2], sample.image.shape[0], sample.image.shape[1])
            cv2.imwrite(str(vis_dir / f"sample_{rank:02d}.jpg"), _make_visual(sample.image, pred_xy, gt, err))

    report = {
        "task": "debug_snn_heatmap_overfit",
        "purpose": "small_sample_overfit_debug",
        "dataset": {
            "root": str(dataset_root),
            "split": str(args.split),
            "num_samples": int(len(ds)),
        },
        "device": str(device),
        "hyperparams": {
            "epochs": int(epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "num_steps": int(args.num_steps),
            "beta": float(args.beta),
            "train_encoding": str(args.train_encoding),
            "eval_encoding": str(args.eval_encoding),
            "heatmap_size": int(args.heatmap_size),
            "heatmap_sigma": float(args.heatmap_sigma),
            "coord_weight": float(args.coord_weight),
            "heatmap_weight": float(args.heatmap_weight),
            "conf_weight": float(args.conf_weight),
        },
        "metrics": {
            "initial": initial,
            "best": best,
            "best_epoch": int(best_epoch),
            "final_loaded_best": final,
        },
        "success_criteria": {
            "pixel_error_mean_lt_5px": bool(float(best["pixel_error_mean"]) < 5.0),
            "rounded_unique_pred_xy_gt_10": bool(int(best["rounded_unique_pred_xy"]) > 10),
            "heatmap_logit_std_gt_0_01": bool(float(best["diagnostics"].get("heatmap_logit_std", 0.0)) > 0.01),
            "spike_rate_l3_gt_0": bool(float(best["diagnostics"].get("spike_rate_l3", 0.0)) > 0.0),
        },
        "diagnosis": {
            "suspect_fixed_prediction": bool(int(best["rounded_unique_pred_xy"]) <= 10),
            "suspect_constant_logits": bool(float(best["diagnostics"].get("heatmap_logit_std", 0.0)) <= 0.01),
            "suspect_dead_spikes": bool(float(best["diagnostics"].get("spike_rate_l3", 0.0)) <= 0.0),
        },
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "trace_path": str(out_dir / "trace.json"),
            "best_weights_path": str(best_path),
            "visual_dir": str(vis_dir),
        },
    }
    (out_dir / "trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
