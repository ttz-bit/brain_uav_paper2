from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from paper2.datasets.seadronessee_dataset import build_seadronessee_dataset
from paper2.models.cnn_heatmap import HeatmapCNN
from paper2.models.snn_heatmap import HeatmapSNN, peak_argmax_2d, soft_argmax_2d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Zero-shot/frozen evaluation of the formal Phase3 heatmap SNN/CNN on processed SeaDronesSee crops. "
            "The checkpoint is loaded read-only; no training or checkpoint writing is performed."
        )
    )
    p.add_argument("--root", type=str, required=True, help="Processed SeaDronesSee root.")
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--weights", type=str, required=True, help="Formal Phase3 heatmap checkpoint.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--input-size", type=int, default=0)
    p.add_argument("--decode-method", choices=["auto", "argmax", "softargmax"], default="auto")
    p.add_argument("--visual-audit-count", type=int, default=24)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required.") from exc
    return torch


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


def _target_from_sample(sample: Any) -> np.ndarray:
    h, w = sample.image.shape[:2]
    cx = float(sample.target_center[0]) / max(1.0, float(w))
    cy = float(sample.target_center[1]) / max(1.0, float(h))
    conf = 1.0 if bool(sample.valid) else 0.0
    return np.array([cx, cy, conf], dtype=np.float32)


def _pixel_error(pred_xy: np.ndarray, gt_xy: np.ndarray, h: int, w: int) -> float:
    px = float(np.clip(pred_xy[0], 0.0, 1.0) * w)
    py = float(np.clip(pred_xy[1], 0.0, 1.0) * h)
    gx = float(np.clip(gt_xy[0], 0.0, 1.0) * w)
    gy = float(np.clip(gt_xy[1], 0.0, 1.0) * h)
    return float(np.hypot(px - gx, py - gy))


def _center_error(gt_xy: np.ndarray, h: int, w: int) -> float:
    return _pixel_error(np.array([0.5, 0.5], dtype=np.float32), gt_xy, h, w)


def _make_visual(img_bgr: np.ndarray, pred_xy: np.ndarray, gt_xy: np.ndarray, *, err: float, label: str) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    pred_x = int(np.clip(pred_xy[0], 0.0, 1.0) * w)
    pred_y = int(np.clip(pred_xy[1], 0.0, 1.0) * h)
    gt_x = int(np.clip(gt_xy[0], 0.0, 1.0) * w)
    gt_y = int(np.clip(gt_xy[1], 0.0, 1.0) * h)
    vis = img_bgr.copy()
    cv2.circle(vis, (gt_x, gt_y), 4, (0, 255, 0), -1)
    cv2.circle(vis, (pred_x, pred_y), 4, (0, 0, 255), -1)
    cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)
    cv2.putText(
        vis,
        f"{label} gt=({gt_x},{gt_y}) pred=({pred_x},{pred_y}) err={err:.1f}px",
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


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
        raise ValueError(f"Unsupported checkpoint model_type={model_type}")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model


def _decode(outputs: dict[str, Any], ckpt: dict[str, Any], decode_method: str):
    method = decode_method
    if method == "auto":
        method = str(ckpt.get("decode_method", "softargmax"))
    if method == "heatmap_argmax":
        method = "argmax"
    if method == "argmax":
        return peak_argmax_2d(outputs["heatmap_logits"])
    return soft_argmax_2d(
        outputs["heatmap_logits"],
        temperature=float(ckpt.get("softargmax_temperature", 20.0)),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows if np.isfinite(float(r[key]))], dtype=float)


def main() -> None:
    args = parse_args()
    torch = _import_torch()
    device = _resolve_device(torch, str(args.device))
    root = Path(args.root).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    audit_dir = out_dir / "visual_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weights}")
    ckpt = torch.load(weights, map_location=device)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise KeyError("Checkpoint must contain state_dict.")
    model = _build_model(ckpt, device)
    model_type = str(ckpt.get("model_type", "unknown"))
    input_size = int(args.input_size) if int(args.input_size) > 0 else int(ckpt.get("input_size", 256))
    ds = build_seadronessee_dataset(
        root=root,
        split=str(args.split),
        project_root=project_root,
        max_samples=args.max_samples,
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for idx in range(len(ds)):
            sample = ds[idx]
            x_np = _to_tensor_image(sample.image, input_size=input_size)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            outputs = model(x, stochastic=False) if model_type == "snn_heatmap" else model(x)
            pred_xy = _decode(outputs, ckpt, str(args.decode_method))[0].detach().cpu().numpy()
            conf = float(torch.sigmoid(outputs["conf_logits"])[0].detach().cpu().item())
            gt = _target_from_sample(sample)
            h, w = sample.image.shape[:2]
            err = _pixel_error(pred_xy, gt[:2], h, w)
            ctr_err = _center_error(gt[:2], h, w)
            row = {
                "sample_index": int(idx),
                "sequence_id": str(sample.sequence_id),
                "frame_id": str(sample.frame_id),
                "valid": bool(sample.valid),
                "pixel_error": float(err),
                "center_baseline_pixel_error": float(ctr_err),
                "center_baseline_delta": float(ctr_err - err),
                "pred_x_norm": float(pred_xy[0]),
                "pred_y_norm": float(pred_xy[1]),
                "gt_x_norm": float(gt[0]),
                "gt_y_norm": float(gt[1]),
                "pred_conf": float(conf),
            }
            rows.append(row)
            if idx < int(args.visual_audit_count):
                vis = _make_visual(sample.image, pred_xy, gt[:2], err=err, label=model_type)
                cv2.imwrite(str(audit_dir / f"{idx:04d}_{sample.sequence_id}_{sample.frame_id}.jpg"), vis)

    px = _metric_array(rows, "pixel_error")
    center = _metric_array(rows, "center_baseline_pixel_error")
    valid_px = np.asarray([float(r["pixel_error"]) for r in rows if bool(r["valid"])], dtype=float)
    report = {
        "task": "eval_phase3_heatmap_on_seadronessee",
        "purpose": "frozen_external_validation",
        "dataset": {
            "name": "SeaDronesSee",
            "root": str(root),
            "split": str(args.split),
            "num_eval": int(len(rows)),
            "max_samples": args.max_samples,
        },
        "checkpoint": {
            "weights": str(weights),
            "model_type": model_type,
            "arch": str(ckpt.get("snn_arch", ckpt.get("cnn_arch", ckpt.get("arch", "unknown")))),
            "input_size": int(input_size),
            "decode_method": str(args.decode_method),
        },
        "metrics": {
            "pixel_error_mean": float(px.mean()) if px.size else 0.0,
            "pixel_error_median": float(np.median(px)) if px.size else 0.0,
            "pixel_error_p90": float(np.percentile(px, 90)) if px.size else 0.0,
            "valid_pixel_error_mean": float(valid_px.mean()) if valid_px.size else 0.0,
            "valid_pixel_error_p90": float(np.percentile(valid_px, 90)) if valid_px.size else 0.0,
            "center_baseline_pixel_error_mean": float(center.mean()) if center.size else 0.0,
            "center_baseline_improve_ratio": float(1.0 - px.mean() / max(center.mean(), 1.0e-12))
            if px.size and center.size
            else 0.0,
            "valid_count": int(sum(1 for r in rows if bool(r["valid"]))),
        },
        "caveats": [
            "This is frozen zero-shot evaluation of the formal Phase3 heatmap checkpoint; no training is performed.",
            "SeaDronesSee crops do not provide Phase3 world coordinates or GSD, so metrics are reported in crop pixels.",
            "If crops are centered on the object, center-baseline comparison should be interpreted cautiously.",
        ],
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "sample_errors_csv": str(out_dir / "sample_errors.csv"),
            "visual_audit_dir": str(audit_dir),
        },
    }
    _write_csv(out_dir / "sample_errors.csv", rows)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
