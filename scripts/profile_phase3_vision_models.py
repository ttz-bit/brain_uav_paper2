from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset
from paper2.models.cnn_heatmap import HeatmapCNN
from paper2.models.snn_heatmap import HeatmapSNN, peak_argmax_2d, soft_argmax_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile Phase3 SNN/CNN heatmap models on the same visual samples.")
    p.add_argument("--dataset-root", type=str, required=True)
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--snn-weights", type=str, required=True)
    p.add_argument("--cnn-weights", type=str, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-samples", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument("--timed-batches", type=int, default=30)
    p.add_argument("--out-dir", type=str, default="outputs/reports/phase3_vision_profile")
    return p.parse_args()


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for profiling.") from exc
    return torch


def _resolve_device(torch: Any, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return str(device)


def _to_tensor_image(img_bgr: np.ndarray, input_size: int) -> np.ndarray:
    if int(input_size) > 0:
        h, w = img_bgr.shape[:2]
        if h != int(input_size) or w != int(input_size):
            img_bgr = cv2.resize(img_bgr, (int(input_size), int(input_size)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))


def _load_checkpoint(torch: Any, path_text: str, device: str) -> dict[str, Any]:
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
    if model_type == "cnn_heatmap":
        model = HeatmapCNN(
            width=int(ckpt.get("width", 32)),
            arch=str(ckpt.get("cnn_arch", ckpt.get("arch", "enhanced"))),
        )
    elif model_type == "snn_heatmap":
        model = HeatmapSNN(
            beta=float(ckpt.get("beta", 0.95)),
            num_steps=int(ckpt.get("num_steps", 12)),
            train_encoding=str(ckpt.get("train_encoding", "rate")),
            eval_encoding=str(ckpt.get("eval_encoding", "direct")),
            arch=str(ckpt.get("snn_arch", ckpt.get("arch", "enhanced"))),
        )
    else:
        raise ValueError(f"Unsupported model_type={model_type}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _num_params(model: Any) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def _make_batches(torch: Any, ds: Any, *, input_size: int, max_samples: int, batch_size: int, device: str) -> list[Any]:
    n = min(len(ds), max(1, int(max_samples)))
    batches: list[Any] = []
    current: list[np.ndarray] = []
    for idx in range(n):
        current.append(_to_tensor_image(ds[idx].image, input_size=input_size))
        if len(current) >= int(batch_size):
            arr = np.stack(current, axis=0)
            batches.append(torch.from_numpy(arr).to(device))
            current = []
    if current:
        arr = np.stack(current, axis=0)
        batches.append(torch.from_numpy(arr).to(device))
    return batches


def _decode(outputs: dict[str, Any], ckpt: dict[str, Any]):
    decode_method = str(ckpt.get("decode_method", "softargmax"))
    if decode_method == "heatmap_argmax":
        decode_method = "argmax"
    if decode_method == "argmax":
        return peak_argmax_2d(outputs["heatmap_logits"])
    return soft_argmax_2d(
        outputs["heatmap_logits"],
        temperature=float(ckpt.get("softargmax_temperature", 20.0)),
    )


def _forward(model: Any, ckpt: dict[str, Any], batch: Any, *, diagnostics: bool = False):
    model_type = str(ckpt.get("model_type", "snn_heatmap"))
    if model_type == "snn_heatmap":
        return model(batch, stochastic=False, return_diagnostics=diagnostics)
    return model(batch)


def _sync(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _profile_one(
    *,
    torch: Any,
    label: str,
    model: Any,
    ckpt: dict[str, Any],
    batches: list[Any],
    device: str,
    warmup_batches: int,
    timed_batches: int,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("No batches available for profiling.")
    with torch.inference_mode():
        for idx in range(max(0, int(warmup_batches))):
            batch = batches[idx % len(batches)]
            outputs = _forward(model, ckpt, batch)
            _ = _decode(outputs, ckpt)
        _sync(torch, device)

        elapsed: list[float] = []
        images = 0
        total_batches = max(1, int(timed_batches))
        for idx in range(total_batches):
            batch = batches[idx % len(batches)]
            _sync(torch, device)
            t0 = time.perf_counter()
            outputs = _forward(model, ckpt, batch)
            _ = _decode(outputs, ckpt)
            _sync(torch, device)
            elapsed.append(time.perf_counter() - t0)
            images += int(batch.shape[0])

        diagnostics: dict[str, float] = {}
        if str(ckpt.get("model_type", "")) == "snn_heatmap":
            diag_batch = batches[0]
            outputs = _forward(model, ckpt, diag_batch, diagnostics=True)
            raw_diag = outputs.get("diagnostics", {})
            if isinstance(raw_diag, dict):
                for key, value in raw_diag.items():
                    if str(key).startswith("spike_rate"):
                        try:
                            diagnostics[str(key)] = float(value.detach().cpu().item())
                        except Exception:
                            pass

    total_time = float(sum(elapsed))
    latency_ms_per_batch = np.asarray(elapsed, dtype=float) * 1000.0
    return {
        "label": label,
        "checkpoint_path": str(ckpt.get("_resolved_path", "")),
        "model_type": str(ckpt.get("model_type", "")),
        "arch": str(ckpt.get("snn_arch", ckpt.get("cnn_arch", ckpt.get("arch", "")))),
        "num_steps": int(ckpt.get("num_steps", 1)),
        "decode_method": str(ckpt.get("decode_method", "softargmax")),
        "params": _num_params(model),
        "timed_batches": int(total_batches),
        "timed_images": int(images),
        "latency_ms_per_batch_mean": float(latency_ms_per_batch.mean()),
        "latency_ms_per_batch_p90": float(np.percentile(latency_ms_per_batch, 90)),
        "latency_ms_per_image_mean": float(total_time * 1000.0 / max(1, images)),
        "throughput_images_per_s": float(images / max(total_time, 1.0e-12)),
        "spike_diagnostics": diagnostics,
    }


def main() -> None:
    args = parse_args()
    torch = _import_torch()
    device = _resolve_device(torch, str(args.device))
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    ds = build_stage2_rendered_dataset(
        root=dataset_root,
        split=str(args.eval_split),
        project_root=project_root,
        max_samples=int(args.max_samples),
        load_water_mask=False,
    )

    snn_ckpt = _load_checkpoint(torch, args.snn_weights, device)
    cnn_ckpt = _load_checkpoint(torch, args.cnn_weights, device)
    snn_model = _build_model(snn_ckpt, device)
    cnn_model = _build_model(cnn_ckpt, device)
    input_size = int(snn_ckpt.get("input_size", cnn_ckpt.get("input_size", 256)))
    batches = _make_batches(
        torch,
        ds,
        input_size=input_size,
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        device=device,
    )

    rows = [
        _profile_one(
            torch=torch,
            label="SNN-enhanced",
            model=snn_model,
            ckpt=snn_ckpt,
            batches=batches,
            device=device,
            warmup_batches=int(args.warmup_batches),
            timed_batches=int(args.timed_batches),
        ),
        _profile_one(
            torch=torch,
            label="CNN-enhanced",
            model=cnn_model,
            ckpt=cnn_ckpt,
            batches=batches,
            device=device,
            warmup_batches=int(args.warmup_batches),
            timed_batches=int(args.timed_batches),
        ),
    ]
    report = {
        "task": "profile_phase3_vision_models",
        "dataset_root": str(dataset_root),
        "eval_split": str(args.eval_split),
        "device": device,
        "max_samples": int(args.max_samples),
        "batch_size": int(args.batch_size),
        "warmup_batches": int(args.warmup_batches),
        "timed_batches": int(args.timed_batches),
        "results": rows,
        "notes": [
            "Latency includes model forward pass and heatmap decoding.",
            "GPU latency is wall-clock latency with torch.cuda.synchronize around timed regions.",
            "SNN energy claims should be phrased as sparsity/neuromorphic potential unless measured on neuromorphic hardware.",
        ],
    }
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase3_vision_model_profile.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "results": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
