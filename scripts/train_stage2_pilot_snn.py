from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_pilot_v6")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--train-split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-steps", type=int, default=12)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Resize input frames to NxN before training/eval in this script.",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="If >0, set torch CPU intra-op thread count.",
    )
    p.add_argument(
        "--num-interop-threads",
        type=int,
        default=0,
        help="If >0, set torch CPU inter-op thread count.",
    )
    p.add_argument(
        "--train-encoding",
        type=str,
        default="rate",
        choices=["rate", "direct"],
        help="Input encoding for training forward pass.",
    )
    p.add_argument(
        "--eval-encoding",
        type=str,
        default="direct",
        choices=["rate", "direct"],
        help="Input encoding for eval/validation forward pass.",
    )
    p.add_argument(
        "--eval-interval",
        type=int,
        default=2,
        help="Run validation every N epochs (always runs on last epoch).",
    )
    p.add_argument(
        "--val-eval-max-samples",
        type=int,
        default=512,
        help="Cap validation samples for in-training checkpoint evaluation.",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation loss computation.",
    )
    p.add_argument("--strict-no-leak", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "stage2_pilot_baselines" / "snn_fit"),
    )
    return p.parse_args()


def _import_torch_and_snn():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:
        raise RuntimeError("PyTorch is required.") from e
    try:
        import snntorch as snn
        from snntorch import surrogate
    except Exception as e:
        raise RuntimeError("snntorch is required. Install: pip install snntorch") from e
    return torch, nn, F, DataLoader, Dataset, snn, surrogate


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


def _make_visual(img_bgr: np.ndarray, pred: np.ndarray, gt: np.ndarray, tag: str) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    pred_x = int(np.clip(pred[0], 0.0, 1.0) * w_img)
    pred_y = int(np.clip(pred[1], 0.0, 1.0) * h_img)
    gt_x = int(np.clip(gt[0], 0.0, 1.0) * w_img)
    gt_y = int(np.clip(gt[1], 0.0, 1.0) * h_img)
    vis = img_bgr.copy()
    cv2.circle(vis, (gt_x, gt_y), 4, (0, 255, 0), -1)
    cv2.circle(vis, (pred_x, pred_y), 4, (0, 0, 255), -1)
    cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)
    cv2.putText(vis, f"{tag} gt=({gt_x},{gt_y}) pred=({pred_x},{pred_y})", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _read_split_rows(dataset_root: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    p = dataset_root / "labels" / f"{split}.jsonl"
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _split_asset_leakage(dataset_root: Path) -> dict[str, int]:
    def _sets(rows: list[dict], key: str) -> set[str]:
        out: set[str] = set()
        for r in rows:
            if key == "distractor_asset_ids":
                for d in r.get(key, []):
                    out.add(str(d))
            else:
                out.add(str(r.get(key, "")))
        return out

    tr = _read_split_rows(dataset_root, "train")
    va = _read_split_rows(dataset_root, "val")
    te = _read_split_rows(dataset_root, "test")
    out: dict[str, int] = {}
    for name, key in [("background", "background_asset_id"), ("target", "target_asset_id"), ("distractor", "distractor_asset_ids")]:
        a, b, c = _sets(tr, key), _sets(va, key), _sets(te, key)
        out[name] = len(a.intersection(b)) + len(a.intersection(c)) + len(b.intersection(c))
    return out


def main() -> None:
    args = parse_args()
    torch, nn, F, DataLoader, Dataset, snn, surrogate = _import_torch_and_snn()
    if int(args.num_threads) > 0:
        torch.set_num_threads(int(args.num_threads))
    if int(args.num_interop_threads) > 0:
        torch.set_num_interop_threads(int(args.num_interop_threads))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    leak = _split_asset_leakage(dataset_root)
    has_leak = any(v > 0 for v in leak.values())
    if args.strict_no_leak and has_leak:
        raise RuntimeError(f"Asset leakage detected: {leak}")

    train_ds = build_stage2_rendered_dataset(root=dataset_root, split=args.train_split, project_root=project_root, max_samples=args.max_train_samples)
    val_ds = build_stage2_rendered_dataset(root=dataset_root, split=args.val_split, project_root=project_root, max_samples=args.max_val_samples)

    t_materialize_begin = time.perf_counter()

    def _materialize(ds):
        xs = []
        ys = []
        for i in range(len(ds)):
            sample = ds[i]
            xs.append(_to_tensor_image(sample.image, input_size=int(args.input_size)))
            ys.append(_target_from_sample(sample))
        x_np = np.stack(xs, axis=0).astype(np.float32)
        y_np = np.stack(ys, axis=0).astype(np.float32)
        return x_np, y_np

    x_train, y_train = _materialize(train_ds)
    x_val, y_val = _materialize(val_ds)
    materialize_sec = float(time.perf_counter() - t_materialize_begin)

    class _NumpyDataset(Dataset):
        def __init__(self, x: np.ndarray, y: np.ndarray):
            self.x = x
            self.y = y
        def __len__(self) -> int:
            return int(self.x.shape[0])
        def __getitem__(self, idx: int):
            return torch.from_numpy(self.x[idx]), torch.from_numpy(self.y[idx])

    loader = DataLoader(_NumpyDataset(x_train, y_train), batch_size=max(1, int(args.batch_size)), shuffle=True, drop_last=False)

    spike_grad = surrogate.fast_sigmoid(slope=25.0)

    class _SmallSNN(nn.Module):
        def __init__(self, beta: float, num_steps: int, train_encoding: str, eval_encoding: str):
            super().__init__()
            self.num_steps = int(num_steps)
            self.train_encoding = str(train_encoding)
            self.eval_encoding = str(eval_encoding)
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(32, 3)
            self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad)

        def _rate_encode(self, x):
            return torch.rand_like(x).le(x).float()

        def _encode(self, x, stochastic: bool):
            mode = self.train_encoding if stochastic else self.eval_encoding
            if mode == "rate":
                return self._rate_encode(x)
            return x

        def forward(self, x, stochastic: bool = True):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            mem_out = self.lif_out.init_leaky()
            spk_out_list = []
            for _ in range(self.num_steps):
                x_t = self._encode(x, stochastic=stochastic)
                cur1 = self.conv1(x_t)
                spk1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.conv2(spk1)
                spk2, mem2 = self.lif2(cur2, mem2)
                z = self.pool(spk2).flatten(1)
                cur_out = self.fc(z)
                spk_out, mem_out = self.lif_out(cur_out, mem_out)
                spk_out_list.append(spk_out)
            return torch.stack(spk_out_list, dim=0).mean(dim=0)

    model = _SmallSNN(
        beta=float(args.beta),
        num_steps=int(args.num_steps),
        train_encoding=str(args.train_encoding),
        eval_encoding=str(args.eval_encoding),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    def _loss(pred, target):
        return F.smooth_l1_loss(pred[:, :2], target[:, :2]) + F.binary_cross_entropy(pred[:, 2], target[:, 2])

    eval_batch_size = max(1, int(args.eval_batch_size))

    def _eval_loss_numpy(x_np: np.ndarray, y_np: np.ndarray, stochastic: bool = False) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for s in range(0, int(x_np.shape[0]), eval_batch_size):
                e = min(s + eval_batch_size, int(x_np.shape[0]))
                xb = torch.from_numpy(x_np[s:e]).to(device)
                yb = torch.from_numpy(y_np[s:e]).to(device)
                pred = model(xb, stochastic=stochastic)
                v = float(_loss(pred, yb).item())
                n = int(e - s)
                total += v * n
                count += n
        return total / max(1, count)

    with torch.no_grad():
        train_initial_loss = float(_eval_loss_numpy(x_train, y_train, stochastic=False))
        val_initial_loss = float(_eval_loss_numpy(x_val, y_val, stochastic=False))

    val_eval_x = x_val
    val_eval_y = y_val
    val_eval_count = int(x_val.shape[0])
    max_val_eval_samples = max(1, int(args.val_eval_max_samples))
    if x_val.shape[0] > max_val_eval_samples:
        idx = rng.choice(int(x_val.shape[0]), size=max_val_eval_samples, replace=False)
        val_eval_x = x_val[idx]
        val_eval_y = y_val[idx]
        val_eval_count = int(max_val_eval_samples)

    best_val = float("inf")
    best_path = out_dir / "model_best.pth"
    loss_trace: list[dict] = []
    epochs = max(1, int(args.epochs))
    eval_interval = max(1, int(args.eval_interval))
    epoch_time_sec: list[float] = []
    train_loop_sec_total = 0.0
    val_eval_sec_total = 0.0
    for ep in range(epochs):
        t_epoch_begin = time.perf_counter()
        model.train()
        batch_losses: list[float] = []
        t_train_loop_begin = time.perf_counter()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, stochastic=True)
            loss = _loss(pred, yb)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().item()))
        train_loop_sec_total += float(time.perf_counter() - t_train_loop_begin)

        do_eval = ((ep + 1) % eval_interval == 0) or (ep + 1 == epochs)
        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss = float("nan")
        if do_eval:
            t_eval_begin = time.perf_counter()
            val_loss = float(_eval_loss_numpy(val_eval_x, val_eval_y, stochastic=False))
            val_eval_sec_total += float(time.perf_counter() - t_eval_begin)

        loss_trace.append({"epoch": ep + 1, "train_loss": train_loss, "val_loss": val_loss})
        if do_eval and val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": int(args.seed),
                    "dataset_root": str(dataset_root),
                    "train_split": str(args.train_split),
                    "val_split": str(args.val_split),
                    "num_train": int(len(train_ds)),
                    "num_val": int(len(val_ds)),
                    "num_steps": int(args.num_steps),
                    "beta": float(args.beta),
                    "train_encoding": str(args.train_encoding),
                    "eval_encoding": str(args.eval_encoding),
                },
                best_path,
            )
        val_msg = f"{val_loss:.6f}" if do_eval else "skip"
        epoch_sec = float(time.perf_counter() - t_epoch_begin)
        epoch_time_sec.append(epoch_sec)
        print(f"[SNN-FIT] epoch={ep+1:03d}/{epochs} train_loss={train_loss:.6f} val_loss={val_msg}")

    last_path = out_dir / "model_last.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "seed": int(args.seed),
            "dataset_root": str(dataset_root),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "num_train": int(len(train_ds)),
            "num_val": int(len(val_ds)),
            "num_steps": int(args.num_steps),
            "beta": float(args.beta),
            "train_encoding": str(args.train_encoding),
            "eval_encoding": str(args.eval_encoding),
        },
        last_path,
    )

    if not best_path.exists():
        torch.save(
            {
                "state_dict": model.state_dict(),
                "seed": int(args.seed),
                "dataset_root": str(dataset_root),
                "train_split": str(args.train_split),
                "val_split": str(args.val_split),
                "num_train": int(len(train_ds)),
                "num_val": int(len(val_ds)),
                "num_steps": int(args.num_steps),
                "beta": float(args.beta),
                "train_encoding": str(args.train_encoding),
                "eval_encoding": str(args.eval_encoding),
            },
            best_path,
        )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    with torch.no_grad():
        train_final_loss = float(_eval_loss_numpy(x_train, y_train, stochastic=False))
        val_final_loss = float(_eval_loss_numpy(x_val, y_val, stochastic=False))

    train_improve_ratio = (train_initial_loss - train_final_loss) / max(train_initial_loss, 1e-12)
    val_improve_ratio = (val_initial_loss - val_final_loss) / max(val_initial_loss, 1e-12)

    for tag, ds, n_vis in [("train", train_ds, 6), ("val", val_ds, 6)]:
        if len(ds) <= 0:
            continue
        idxs = rng.choice(len(ds), size=min(n_vis, len(ds)), replace=False)
        with torch.no_grad():
            model.eval()
            for rank, idx in enumerate(idxs):
                s = ds[int(idx)]
                x = torch.from_numpy(_to_tensor_image(s.image, input_size=int(args.input_size))).unsqueeze(0).to(device)
                pred = model(x, stochastic=False)[0].detach().cpu().numpy()
                gt = _target_from_sample(s)
                vis = _make_visual(s.image, pred, gt, tag=tag)
                cv2.imwrite(str(vis_dir / f"{tag}_sample_{rank:02d}.jpg"), vis)

    report = {
        "task": "train_stage2_pilot_snn",
        "purpose": "fit_check",
        "dataset": {
            "name": "stage2_rendered",
            "root": str(dataset_root),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "num_train": int(len(train_ds)),
            "num_val": int(len(val_ds)),
        },
        "device": str(device),
        "hyperparams": {
            "batch_size": int(args.batch_size),
            "epochs": int(epochs),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "num_steps": int(args.num_steps),
            "beta": float(args.beta),
            "train_encoding": str(args.train_encoding),
            "eval_encoding": str(args.eval_encoding),
            "eval_interval": int(eval_interval),
            "val_eval_max_samples": int(val_eval_count),
            "eval_batch_size": int(eval_batch_size),
            "input_size": int(args.input_size),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "timing": {
            "materialize_sec": float(materialize_sec),
            "train_loop_sec_total": float(train_loop_sec_total),
            "val_eval_sec_total": float(val_eval_sec_total),
            "epoch_sec_mean": float(np.mean(epoch_time_sec)) if epoch_time_sec else 0.0,
            "epoch_sec_p90": float(np.percentile(epoch_time_sec, 90)) if epoch_time_sec else 0.0,
            "total_epochs": int(epochs),
            "cpu_count_os": int(os.cpu_count() or 0),
        },
        "leakage_check": {
            "split_asset_leakage": leak,
            "has_leakage": bool(has_leak),
            "strict_no_leak": bool(args.strict_no_leak),
        },
        "metrics": {
            "train_initial_loss": float(train_initial_loss),
            "train_final_loss": float(train_final_loss),
            "train_improve_ratio": float(train_improve_ratio),
            "val_initial_loss": float(val_initial_loss),
            "val_final_loss": float(val_final_loss),
            "val_improve_ratio": float(val_improve_ratio),
        },
        "success_criteria": {
            "train_final_lt_initial": bool(train_final_loss < train_initial_loss),
            "train_improve_ratio_gt_0": bool(train_improve_ratio > 0.0),
            "val_final_lt_initial": bool(val_final_loss < val_initial_loss),
            "val_improve_ratio_gt_0": bool(val_improve_ratio > 0.0),
        },
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "loss_trace_path": str(out_dir / "loss_trace.json"),
            "best_weights_path": str(best_path),
            "last_weights_path": str(last_path),
            "visual_dir": str(vis_dir),
        },
    }
    (out_dir / "loss_trace.json").write_text(json.dumps(loss_trace, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
