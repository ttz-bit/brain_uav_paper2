from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset
from paper2.models.cnn_heatmap import HeatmapCNN
from paper2.models.snn_heatmap import HeatmapSNN, soft_argmax_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Estimate dense MACs and SNN spike-gated SynOps proxies for Phase3 vision models. "
            "This is a software-side operation-count analysis, not measured hardware energy."
        )
    )
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--snn-weights", type=str, required=True)
    p.add_argument("--cnn-weights", type=str, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", type=str, default="outputs/reports/phase3_vision_synops")
    return p.parse_args()


def _import_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required.") from exc
    return torch, nn


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return str(requested)


def _to_tensor_image(img_bgr: np.ndarray, input_size: int) -> np.ndarray:
    if int(input_size) > 0:
        h, w = img_bgr.shape[:2]
        if h != int(input_size) or w != int(input_size):
            img_bgr = cv2.resize(img_bgr, (int(input_size), int(input_size)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def _load_ckpt(torch: Any, path_text: str, device: str) -> dict[str, Any]:
    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise KeyError(f"Checkpoint must contain state_dict: {path}")
    ckpt["_resolved_path"] = str(path)
    return ckpt


def _build_model(ckpt: dict[str, Any], device: str):
    model_type = str(ckpt.get("model_type", "snn_heatmap"))
    if model_type == "snn_heatmap":
        model = HeatmapSNN(
            beta=float(ckpt.get("beta", 0.95)),
            num_steps=int(ckpt.get("num_steps", 12)),
            train_encoding=str(ckpt.get("train_encoding", "rate")),
            eval_encoding=str(ckpt.get("eval_encoding", "direct")),
            arch=str(ckpt.get("snn_arch", ckpt.get("arch", "enhanced"))),
        )
    elif model_type == "cnn_heatmap":
        model = HeatmapCNN(
            width=int(ckpt.get("width", 32)),
            arch=str(ckpt.get("cnn_arch", ckpt.get("arch", "enhanced"))),
        )
    else:
        raise ValueError(f"Unsupported model_type={model_type}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _make_batches(torch: Any, ds: Any, *, input_size: int, max_samples: int, batch_size: int, device: str) -> list[Any]:
    n = min(len(ds), max(1, int(max_samples)))
    batches: list[Any] = []
    current: list[np.ndarray] = []
    for idx in range(n):
        current.append(_to_tensor_image(ds[idx].image, input_size=input_size))
        if len(current) >= int(batch_size):
            batches.append(torch.from_numpy(np.stack(current, axis=0)).to(device))
            current = []
    if current:
        batches.append(torch.from_numpy(np.stack(current, axis=0)).to(device))
    return batches


def _conv_macs(module: Any, output: Any) -> float:
    out = output
    if not hasattr(out, "shape") or len(out.shape) != 4:
        return 0.0
    b, out_ch, out_h, out_w = [int(v) for v in out.shape]
    k_h, k_w = module.kernel_size
    in_ch = int(module.in_channels)
    groups = int(module.groups)
    return float(b * out_ch * out_h * out_w * (in_ch // groups) * int(k_h) * int(k_w))


def _linear_macs(module: Any, output: Any) -> float:
    if not hasattr(output, "shape"):
        return 0.0
    batch = int(output.shape[0]) if len(output.shape) > 0 else 1
    return float(batch * int(module.in_features) * int(module.out_features))


def _activity(tensor: Any) -> float:
    try:
        return float((tensor.detach() != 0).float().mean().cpu().item())
    except Exception:
        return 1.0


def _is_snn_sparse_input_layer(name: str) -> bool:
    return name in {"conv2", "conv3", "conv4", "conv5", "conv6", "skip2"}


def _profile_ops(torch: Any, nn: Any, model: Any, ckpt: dict[str, Any], batches: list[Any]) -> dict[str, Any]:
    model_type = str(ckpt.get("model_type", ""))
    records: dict[str, dict[str, float]] = {}
    handles = []

    def add_record(name: str, dense_macs: float, proxy_synops: float, input_activity: float) -> None:
        row = records.setdefault(name, {"dense_macs": 0.0, "proxy_synops": 0.0, "input_activity_sum": 0.0, "calls": 0.0})
        row["dense_macs"] += float(dense_macs)
        row["proxy_synops"] += float(proxy_synops)
        row["input_activity_sum"] += float(input_activity)
        row["calls"] += 1.0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            def hook(mod, inputs, output, layer_name=name):
                dense = _conv_macs(mod, output)
                input_tensor = inputs[0] if inputs else None
                act = _activity(input_tensor) if input_tensor is not None else 1.0
                if model_type == "snn_heatmap" and _is_snn_sparse_input_layer(layer_name):
                    proxy = dense * act
                else:
                    proxy = dense
                add_record(layer_name, dense, proxy, act)

            handles.append(module.register_forward_hook(hook))
        elif isinstance(module, nn.Linear):
            def hook(mod, inputs, output, layer_name=name):
                dense = _linear_macs(mod, output)
                input_tensor = inputs[0] if inputs else None
                act = _activity(input_tensor) if input_tensor is not None else 1.0
                add_record(layer_name, dense, dense, act)

            handles.append(module.register_forward_hook(hook))

    try:
        with torch.inference_mode():
            for batch in batches:
                if model_type == "snn_heatmap":
                    outputs = model(batch, stochastic=False, return_diagnostics=True)
                else:
                    outputs = model(batch)
                _ = soft_argmax_2d(
                    outputs["heatmap_logits"],
                    temperature=float(ckpt.get("softargmax_temperature", 20.0)),
                )
    finally:
        for handle in handles:
            handle.remove()

    total_images = int(sum(int(batch.shape[0]) for batch in batches))
    dense_total = float(sum(row["dense_macs"] for row in records.values()))
    proxy_total = float(sum(row["proxy_synops"] for row in records.values()))
    layer_rows = []
    for name, row in sorted(records.items()):
        calls = max(float(row["calls"]), 1.0)
        layer_rows.append(
            {
                "layer": name,
                "calls": int(row["calls"]),
                "dense_macs_total": float(row["dense_macs"]),
                "proxy_synops_total": float(row["proxy_synops"]),
                "dense_macs_per_image": float(row["dense_macs"] / max(1, total_images)),
                "proxy_synops_per_image": float(row["proxy_synops"] / max(1, total_images)),
                "mean_input_activity": float(row["input_activity_sum"] / calls),
            }
        )

    return {
        "checkpoint_path": str(ckpt.get("_resolved_path", "")),
        "model_type": model_type,
        "arch": str(ckpt.get("snn_arch", ckpt.get("cnn_arch", ckpt.get("arch", "")))),
        "num_steps": int(ckpt.get("num_steps", 1)),
        "total_images": int(total_images),
        "dense_macs_total": dense_total,
        "proxy_synops_total": proxy_total,
        "dense_macs_per_image": float(dense_total / max(1, total_images)),
        "proxy_synops_per_image": float(proxy_total / max(1, total_images)),
        "proxy_synops_to_dense_macs_ratio": float(proxy_total / max(dense_total, 1.0e-12)),
        "layer_rows": layer_rows,
    }


def main() -> None:
    args = parse_args()
    torch, nn = _import_torch()
    device = _resolve_device(torch, str(args.device))
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    ds = build_stage2_rendered_dataset(
        root=dataset_root,
        split=str(args.eval_split),
        project_root=project_root,
        max_samples=int(args.max_samples),
        load_water_mask=False,
    )
    snn_ckpt = _load_ckpt(torch, args.snn_weights, device)
    cnn_ckpt = _load_ckpt(torch, args.cnn_weights, device)
    input_size = int(snn_ckpt.get("input_size", cnn_ckpt.get("input_size", 256)))
    batches = _make_batches(
        torch,
        ds,
        input_size=input_size,
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        device=device,
    )
    snn_model = _build_model(snn_ckpt, device)
    cnn_model = _build_model(cnn_ckpt, device)
    snn_ops = _profile_ops(torch, nn, snn_model, snn_ckpt, batches)
    cnn_ops = _profile_ops(torch, nn, cnn_model, cnn_ckpt, batches)
    report = {
        "task": "estimate_phase3_vision_synops",
        "dataset_root": str(dataset_root),
        "eval_split": str(args.eval_split),
        "device": device,
        "max_samples": int(args.max_samples),
        "batch_size": int(args.batch_size),
        "results": {
            "SNN-enhanced": snn_ops,
            "CNN-enhanced": cnn_ops,
        },
        "interpretation": {
            "dense_macs": "Conventional dense multiply-accumulate operation count observed during the PyTorch forward pass.",
            "proxy_synops": (
                "SNN neuromorphic proxy count. Layers receiving spike tensors are scaled by measured presynaptic "
                "nonzero activity; analog first layer and non-spiking readout heads are counted as dense operations."
            ),
            "caveat": (
                "This is not measured hardware energy. It should be reported as a SynOps/MACs proxy and used only "
                "to support sparsity/neuromorphic-potential claims."
            ),
        },
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase3_vision_synops_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "summary": report["results"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
