from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset
from paper2.models.cnn_heatmap import HeatmapCNN
from paper2.models.snn_heatmap import heatmap_loss, peak_argmax_2d, soft_argmax_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN heatmap baseline on Stage2 rendered data.")
    p.add_argument("--dataset-root", type=str, default="data/rendered/paper2_render_v1.0.0")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--train-split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--heatmap-size", type=int, default=64)
    p.add_argument("--heatmap-sigma", type=float, default=1.5)
    p.add_argument("--heatmap-weight", type=float, default=1.0)
    p.add_argument("--coord-weight", type=float, default=5.0)
    p.add_argument("--conf-weight", type=float, default=0.2)
    p.add_argument("--softargmax-temperature", type=float, default=20.0)
    p.add_argument("--decode-method", type=str, default="softargmax", choices=["argmax", "softargmax"])
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--cnn-arch", type=str, default="enhanced", choices=["legacy", "enhanced"])
    p.add_argument("--init-weights", type=str, default="")
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument(
        "--selection-metric",
        type=str,
        default="val_pixel_error",
        choices=["val_loss", "val_pixel_error", "val_center_improve"],
    )
    p.add_argument("--train-eval-max-samples", type=int, default=2048)
    p.add_argument("--val-eval-max-samples", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--amp", type=str, default="none", choices=["none", "fp16", "bf16"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--strict-no-leak", action="store_true")
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--num-interop-threads", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "stage2_formal_baselines" / "cnn_heatmap_fit_v1"),
    )
    return p.parse_args()


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset, Subset
    except Exception as exc:
        raise RuntimeError("PyTorch is required for train_stage2_pilot_cnn_heatmap.py.") from exc
    return torch, DataLoader, Dataset, Subset


def _to_tensor_image(img_bgr: np.ndarray, input_size: int) -> np.ndarray:
    sz = int(input_size)
    if sz > 0 and img_bgr.shape[:2] != (sz, sz):
        img_bgr = cv2.resize(img_bgr, (sz, sz), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def _target_from_sample(sample) -> np.ndarray:
    h, w = sample.image.shape[:2]
    return np.array(
        [
            float(sample.target_center[0]) / max(1.0, float(w)),
            float(sample.target_center[1]) / max(1.0, float(h)),
            1.0 if bool(sample.valid) else 0.0,
        ],
        dtype=np.float32,
    )


def _pixel_error_norm(pred_xy: np.ndarray, gt_xy: np.ndarray, h: int, w: int) -> float:
    px = float(np.clip(pred_xy[0], 0.0, 1.0) * w)
    py = float(np.clip(pred_xy[1], 0.0, 1.0) * h)
    gx = float(np.clip(gt_xy[0], 0.0, 1.0) * w)
    gy = float(np.clip(gt_xy[1], 0.0, 1.0) * h)
    return float(np.hypot(px - gx, py - gy))


def _read_split_rows(dataset_root: Path, split: str) -> list[dict]:
    path = dataset_root / "labels" / f"{split}.jsonl"
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split_asset_leakage(dataset_root: Path) -> dict[str, int]:
    def _sets(rows: list[dict], key: str) -> set[str]:
        out: set[str] = set()
        for row in rows:
            if key == "distractor_asset_ids":
                out.update(str(x) for x in row.get(key, []))
            else:
                out.add(str(row.get(key, "")))
        return out

    tr = _read_split_rows(dataset_root, "train")
    va = _read_split_rows(dataset_root, "val")
    te = _read_split_rows(dataset_root, "test")
    out: dict[str, int] = {}
    for name, key in [("background", "background_asset_id"), ("target", "target_asset_id"), ("distractor", "distractor_asset_ids")]:
        a, b, c = _sets(tr, key), _sets(va, key), _sets(te, key)
        out[name] = len(a & b) + len(a & c) + len(b & c)
    return out


def main() -> None:
    args = parse_args()
    torch, DataLoader, Dataset, Subset = _import_torch()
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    if int(args.num_interop_threads) > 0:
        torch.set_num_interop_threads(int(args.num_interop_threads))
    if int(args.num_workers) > 0:
        cv2.setNumThreads(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    rng = np.random.default_rng(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else str(args.device)
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    dataset_root = Path(args.dataset_root).resolve()
    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    leak = _split_asset_leakage(dataset_root)
    has_leak = any(v > 0 for v in leak.values())
    if bool(args.strict_no_leak) and has_leak:
        raise RuntimeError(f"Asset leakage detected: {leak}")

    train_ds = build_stage2_rendered_dataset(
        root=dataset_root,
        split=args.train_split,
        project_root=project_root,
        max_samples=args.max_train_samples,
    )
    val_ds = build_stage2_rendered_dataset(
        root=dataset_root,
        split=args.val_split,
        project_root=project_root,
        max_samples=args.max_val_samples,
    )

    class _TorchDataset(Dataset):
        def __init__(self, ds):
            self.ds = ds

        def __len__(self) -> int:
            return int(len(self.ds))

        def __getitem__(self, idx: int):
            sample = self.ds[int(idx)]
            x = _to_tensor_image(sample.image, int(args.input_size))
            y = _target_from_sample(sample)
            return torch.from_numpy(x), torch.from_numpy(y)

    train_torch_ds = _TorchDataset(train_ds)
    val_torch_ds = _TorchDataset(val_ds)
    num_workers = max(0, int(args.num_workers))

    def _loader(ds, *, batch_size: int, shuffle: bool):
        kwargs = {
            "batch_size": max(1, int(batch_size)),
            "shuffle": bool(shuffle),
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": device == "cuda",
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
            kwargs["persistent_workers"] = bool(args.persistent_workers)
        return DataLoader(ds, **kwargs)

    train_loader = _loader(train_torch_ds, batch_size=int(args.batch_size), shuffle=True)
    train_eval_ds = train_torch_ds
    train_eval_count = int(len(train_torch_ds))
    if int(args.train_eval_max_samples) > 0 and len(train_torch_ds) > int(args.train_eval_max_samples):
        idx = rng.choice(len(train_torch_ds), size=int(args.train_eval_max_samples), replace=False)
        train_eval_ds = Subset(train_torch_ds, [int(i) for i in idx.tolist()])
        train_eval_count = int(args.train_eval_max_samples)
    val_eval_ds = val_torch_ds
    val_eval_count = int(len(val_torch_ds))
    if int(args.val_eval_max_samples) > 0 and len(val_torch_ds) > int(args.val_eval_max_samples):
        idx = rng.choice(len(val_torch_ds), size=int(args.val_eval_max_samples), replace=False)
        val_eval_ds = Subset(val_torch_ds, [int(i) for i in idx.tolist()])
        val_eval_count = int(args.val_eval_max_samples)
    train_eval_loader = _loader(train_eval_ds, batch_size=int(args.eval_batch_size), shuffle=False)
    val_eval_loader = _loader(val_eval_ds, batch_size=int(args.eval_batch_size), shuffle=False)

    model = HeatmapCNN(width=int(args.width), arch=str(args.cnn_arch)).to(device)
    init_weights = str(args.init_weights or "").strip()
    if init_weights:
        init_path = Path(init_weights).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Missing init weights: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device)
        if init_ckpt.get("model_type") == "cnn_heatmap":
            state_dict = init_ckpt.get("state_dict", init_ckpt)
            model.load_state_dict(state_dict, strict=True)
        else:
            raise RuntimeError(f"Expected cnn_heatmap checkpoint, got {init_ckpt.get('model_type', 'unknown')}")
    if bool(args.channels_last) and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    amp_enabled = device == "cuda" and str(args.amp) != "none"
    amp_dtype = torch.float16 if str(args.amp) == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and str(args.amp) == "fp16"))

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

    def _eval_loader(loader) -> dict[str, float | int]:
        model.eval()
        losses: list[float] = []
        px_errors: list[float] = []
        center_errors: list[float] = []
        unique_pred: set[tuple[int, int]] = set()
        count = 0
        with torch.inference_mode():
            for xb, yb in loader:
                xb = xb.to(device, non_blocking=(device == "cuda"))
                yb = yb.to(device, non_blocking=(device == "cuda"))
                if bool(args.channels_last) and device == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                    outputs = model(xb)
                    loss, _ = _loss(outputs, yb)
                losses.append(float(loss.detach().cpu().item()))
                if str(args.decode_method) == "softargmax":
                    pred_xy = soft_argmax_2d(
                        outputs["heatmap_logits"],
                        temperature=float(args.softargmax_temperature),
                    ).detach().cpu().numpy()
                else:
                    pred_xy = peak_argmax_2d(outputs["heatmap_logits"]).detach().cpu().numpy()
                gt = yb[:, :2].detach().cpu().numpy()
                for pred, target in zip(pred_xy, gt):
                    err = _pixel_error_norm(pred, target, int(args.input_size), int(args.input_size))
                    center = _pixel_error_norm(np.array([0.5, 0.5], dtype=np.float32), target, int(args.input_size), int(args.input_size))
                    px_errors.append(err)
                    center_errors.append(center)
                    unique_pred.add((int(round(pred[0] * int(args.input_size))), int(round(pred[1] * int(args.input_size)))))
                    count += 1
        px_mean = float(np.mean(px_errors)) if px_errors else 0.0
        center_mean = float(np.mean(center_errors)) if center_errors else 0.0
        return {
            "loss": float(np.mean(losses)) if losses else 0.0,
            "samples": int(count),
            "pixel_error_mean": px_mean,
            "pixel_error_p90": float(np.percentile(px_errors, 90)) if px_errors else 0.0,
            "center_baseline_pixel_error_mean": center_mean,
            "center_baseline_improve_ratio": float((center_mean - px_mean) / max(center_mean, 1e-12)) if center_errors else 0.0,
            "rounded_unique_pred_xy": int(len(unique_pred)),
        }

    with torch.inference_mode():
        train_initial = _eval_loader(train_eval_loader)
        val_initial = _eval_loader(val_eval_loader)

    def _selection_score(metrics: dict[str, float | int]) -> float:
        metric = str(args.selection_metric)
        if metric == "val_loss":
            return float(metrics["loss"])
        if metric == "val_center_improve":
            return -float(metrics["center_baseline_improve_ratio"])
        return float(metrics["pixel_error_mean"])

    best_score = float("inf")
    best_val_loss = float("inf")
    best_path = out_dir / "model_best.pth"
    last_path = out_dir / "model_last.pth"
    loss_trace: list[dict] = []
    epoch_times: list[float] = []
    eval_interval = max(1, int(args.eval_interval))
    train_loop_sec_total = 0.0
    val_eval_sec_total = 0.0

    for ep in range(max(1, int(args.epochs))):
        t0 = time.perf_counter()
        model.train()
        batch_losses: list[float] = []
        t_train = time.perf_counter()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=(device == "cuda"))
            yb = yb.to(device, non_blocking=(device == "cuda"))
            if bool(args.channels_last) and device == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                outputs = model(xb)
                loss, _ = _loss(outputs, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.detach().cpu().item()))
        train_loop_sec_total += float(time.perf_counter() - t_train)

        do_eval = ((ep + 1) % eval_interval == 0) or (ep + 1 == max(1, int(args.epochs)))
        val_metrics = {}
        val_loss = float("nan")
        if do_eval:
            t_eval = time.perf_counter()
            val_metrics = _eval_loader(val_eval_loader)
            val_loss = float(val_metrics["loss"])
            val_eval_sec_total += float(time.perf_counter() - t_eval)
            selection_score = _selection_score(val_metrics)
            if selection_score < best_score:
                best_score = float(selection_score)
                best_val_loss = float(val_loss)
                _save_checkpoint(best_path, model, args, dataset_root, train_ds, val_ds, best_val_loss, ep + 1, best_score)
        else:
            selection_score = float("nan")
        _save_checkpoint(last_path, model, args, dataset_root, train_ds, val_ds, best_val_loss, ep + 1, best_score)
        loss_trace.append(
            {
                "epoch": int(ep + 1),
                "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                "val_loss": val_loss,
                "selection_metric": str(args.selection_metric),
                "selection_score": float(selection_score),
                "val_metrics": val_metrics,
            }
        )
        (out_dir / "loss_trace.json").write_text(json.dumps(loss_trace, ensure_ascii=False, indent=2), encoding="utf-8")
        epoch_sec = float(time.perf_counter() - t0)
        epoch_times.append(epoch_sec)
        val_msg = f"{val_loss:.6f}" if do_eval else "skip"
        print(
            f"[CNN-HEATMAP] epoch={ep+1:03d}/{max(1, int(args.epochs))} "
            f"train_loss={loss_trace[-1]['train_loss']:.6f} "
            f"val_loss={val_msg} epoch_sec={epoch_sec:.1f}",
            flush=True,
        )

    ckpt = torch.load(best_path, map_location=device) if best_path.exists() else torch.load(last_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    train_final = _eval_loader(train_eval_loader)
    val_final = _eval_loader(val_eval_loader)

    report = {
        "task": "train_stage2_pilot_cnn_heatmap",
        "purpose": "formal_cnn_heatmap_baseline",
        "dataset": {
            "name": "stage2_rendered",
            "root": str(dataset_root),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "num_train": int(len(train_ds)),
            "num_val": int(len(val_ds)),
        },
        "device": str(device),
        "model": {"type": "cnn_heatmap", "arch": str(args.cnn_arch), "width": int(args.width), "output": "heatmap + confidence"},
        "hyperparams": {
            "batch_size": int(args.batch_size),
            "epochs": int(max(1, int(args.epochs))),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "input_size": int(args.input_size),
            "heatmap_size": int(args.heatmap_size),
            "heatmap_sigma": float(args.heatmap_sigma),
            "heatmap_weight": float(args.heatmap_weight),
            "coord_weight": float(args.coord_weight),
            "conf_weight": float(args.conf_weight),
            "softargmax_temperature": float(args.softargmax_temperature),
            "decode_method": str(args.decode_method),
            "selection_metric": str(args.selection_metric),
            "eval_interval": int(eval_interval),
            "eval_batch_size": int(args.eval_batch_size),
            "train_eval_max_samples": int(train_eval_count),
            "val_eval_max_samples": int(val_eval_count),
            "num_workers": int(num_workers),
            "amp": str(args.amp),
            "channels_last": bool(args.channels_last),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "timing": {
            "train_loop_sec_total": float(train_loop_sec_total),
            "val_eval_sec_total": float(val_eval_sec_total),
            "epoch_sec_mean": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "epoch_sec_p90": float(np.percentile(epoch_times, 90)) if epoch_times else 0.0,
            "cpu_count_os": int(os.cpu_count() or 0),
        },
        "leakage_check": {
            "split_asset_leakage": leak,
            "has_leakage": bool(has_leak),
            "strict_no_leak": bool(args.strict_no_leak),
        },
        "metrics": {
            "train_initial": train_initial,
            "train_final": train_final,
            "val_initial": val_initial,
            "val_final": val_final,
        },
        "success_criteria": {
            "train_final_lt_initial": bool(float(train_final["loss"]) < float(train_initial["loss"])),
            "val_final_lt_initial": bool(float(val_final["loss"]) < float(val_initial["loss"])),
            "val_beats_center_baseline": bool(float(val_final["center_baseline_improve_ratio"]) > 0.0),
            "val_prediction_not_constant": bool(int(val_final["rounded_unique_pred_xy"]) > 5),
        },
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "loss_trace_path": str(out_dir / "loss_trace.json"),
            "best_weights_path": str(best_path),
            "last_weights_path": str(last_path),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _save_checkpoint(
    path: Path,
    model,
    args: argparse.Namespace,
    dataset_root: Path,
    train_ds,
    val_ds,
    best_val: float,
    epoch: int,
    best_score: float,
) -> None:
    import torch

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "cnn_heatmap",
            "seed": int(args.seed),
            "dataset_root": str(dataset_root),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "num_train": int(len(train_ds)),
            "num_val": int(len(val_ds)),
            "input_size": int(args.input_size),
            "heatmap_size": int(args.heatmap_size),
            "heatmap_sigma": float(args.heatmap_sigma),
            "heatmap_weight": float(args.heatmap_weight),
            "coord_weight": float(args.coord_weight),
            "conf_weight": float(args.conf_weight),
            "softargmax_temperature": float(args.softargmax_temperature),
            "decode_method": str(args.decode_method),
            "width": int(args.width),
            "cnn_arch": str(args.cnn_arch),
            "init_weights": init_weights,
            "epoch": int(epoch),
            "selection_metric": str(args.selection_metric),
            "best_selection_score": float(best_score),
            "best_val_loss": float(best_val),
        },
        path,
    )


if __name__ == "__main__":
    main()
