from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset
try:
    from paper2.models.cnn_heatmap import HeatmapCNN
    from paper2.models.snn_heatmap import HeatmapSNN, heatmap_loss, peak_argmax_2d, soft_argmax_2d
except Exception as e:
    raise RuntimeError(
        "Failed to import the SNN model stack. If you see a GLIBC_2.27 error, "
        "activate the paper2_torch_gpu environment on the server and rerun."
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=str, default="data/rendered/stage2_pilot_v6")
    p.add_argument("--project-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    p.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--input-size", type=int, default=0)
    p.add_argument("--eval-encoding", type=str, default="auto", choices=["auto", "rate", "direct"])
    p.add_argument("--decode-method", type=str, default="auto", choices=["auto", "argmax", "softargmax"])
    p.add_argument("--visual-audit-count", type=int, default=16)
    p.add_argument("--offcenter-threshold-px", type=float, default=20.0)
    p.add_argument("--min-center-improve-ratio", type=float, default=0.10)
    p.add_argument("--min-offcenter-improve-ratio", type=float, default=0.10)
    p.add_argument("--min-unique-pred-xy", type=int, default=16)
    p.add_argument("--water-interior-erode-px", type=int, default=6)
    p.add_argument("--water-interior-erode-far-px", type=int, default=10)
    p.add_argument("--water-interior-erode-mid-px", type=int, default=6)
    p.add_argument("--water-interior-erode-terminal-px", type=int, default=4)
    p.add_argument(
        "--weights",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "outputs"
            / "stage2_pre_baselines"
            / "snn_heatmap_fit_v2"
            / "model_best.pth"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "stage2_pre_baselines" / "snn_heatmap_eval_v2"),
    )
    return p.parse_args()


def _import_torch():
    try:
        import torch
    except Exception as e:
        raise RuntimeError("PyTorch is required.") from e
    return torch


def _to_tensor_image(img_bgr: np.ndarray, input_size: int = 0) -> np.ndarray:
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


def _nearest_distractor_error_px(pred_xy: np.ndarray, sample) -> tuple[float | None, int]:
    h, w = sample.image.shape[:2]
    px = float(np.clip(pred_xy[0], 0.0, 1.0) * w)
    py = float(np.clip(pred_xy[1], 0.0, 1.0) * h)
    errors: list[float] = []
    for box in list((sample.meta or {}).get("distractor_bboxes_xywh", [])):
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x, y, bw, bh = [float(v) for v in box[:4]]
        if bw <= 0.0 or bh <= 0.0:
            continue
        errors.append(float(np.hypot(px - (x + 0.5 * bw), py - (y + 0.5 * bh))))
    if not errors:
        return None, 0
    return float(min(errors)), int(len(errors))


def _pred_on_land(pred_xy: np.ndarray, sample) -> bool | None:
    water_mask = getattr(sample, "water_mask", None)
    if water_mask is None:
        return None
    h, w = sample.image.shape[:2]
    mask = np.asarray(water_mask)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    px = int(np.clip(pred_xy[0], 0.0, 1.0) * w)
    py = int(np.clip(pred_xy[1], 0.0, 1.0) * h)
    px = min(max(px, 0), w - 1)
    py = min(max(py, 0), h - 1)
    return bool(mask[py, px] <= 0)


def _water_mask_from_sample(sample, *, input_size: int, interior_erode_px: int = 0) -> np.ndarray:
    sz = int(input_size)
    h, w = sample.image.shape[:2]
    out_h = sz if sz > 0 else h
    out_w = sz if sz > 0 else w
    if getattr(sample, "water_mask", None) is None:
        return np.ones((1, out_h, out_w), dtype=np.float32)
    water_mask = np.asarray(sample.water_mask)
    if water_mask.shape[:2] != (h, w):
        water_mask = cv2.resize(water_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    water = (water_mask > 0).astype(np.uint8)
    if int(interior_erode_px) > 0 and int(water.sum()) > 0:
        k = int(interior_erode_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(water, kernel, iterations=1)
        if int(eroded.sum()) > 0:
            water = eroded
    if sz > 0 and (h != sz or w != sz):
        water = cv2.resize(water, (sz, sz), interpolation=cv2.INTER_NEAREST)
    return water.astype(np.float32, copy=False)[None, :, :]


def _water_interior_erode_px_for_sample(sample, args) -> int:
    stage = str((sample.meta or {}).get("perception_stage", "")).lower()
    if stage == "far":
        return int(args.water_interior_erode_far_px)
    if stage == "mid":
        return int(args.water_interior_erode_mid_px)
    if stage == "terminal":
        return int(args.water_interior_erode_terminal_px)
    return int(args.water_interior_erode_px)


def _land_mask_from_sample(sample, *, input_size: int, dilate_px: int) -> np.ndarray:
    sz = int(input_size)
    h, w = sample.image.shape[:2]
    out_h = sz if sz > 0 else h
    out_w = sz if sz > 0 else w
    if getattr(sample, "water_mask", None) is None:
        return np.zeros((1, out_h, out_w), dtype=np.float32)
    water_mask = np.asarray(sample.water_mask)
    if water_mask.shape[:2] != (h, w):
        water_mask = cv2.resize(water_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    land = (water_mask <= 0).astype(np.uint8)
    if int(dilate_px) > 0 and int(land.sum()) > 0:
        k = int(dilate_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        land = cv2.dilate(land, kernel, iterations=1)
    if sz > 0 and (h != sz or w != sz):
        land = cv2.resize(land, (sz, sz), interpolation=cv2.INTER_NEAREST)
    return land.astype(np.float32, copy=False)[None, :, :]


def _distractor_boxes_from_sample(sample) -> list[list[float]]:
    boxes: list[list[float]] = []
    for box in list((sample.meta or {}).get("distractor_bboxes_xywh", [])):
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x, y, w, h = [float(v) for v in box[:4]]
        if w <= 0.0 or h <= 0.0:
            continue
        boxes.append([x, y, w, h])
    return boxes


def _draw_distractor_boxes(vis: np.ndarray, boxes: list[list[float]]) -> None:
    for box in boxes:
        x, y, w, h = [int(round(float(v))) for v in box[:4]]
        cv2.rectangle(vis, (x, y), (x + max(1, w), y + max(1, h)), (0, 255, 255), 1, cv2.LINE_AA)


def _image_to_world(pred_xy: np.ndarray, sample) -> tuple[list[float] | None, list[float] | None, float | None]:
    meta = dict(sample.meta or {})
    crop = meta.get("crop_center_world")
    gsd = meta.get("gsd_km_per_px", meta.get("gsd", meta.get("gsd_m_per_px")))
    truth = meta.get("target_state_world")
    if crop is None or gsd is None:
        return None, None, None
    h, w = sample.image.shape[:2]
    u = float(np.clip(pred_xy[0], 0.0, 1.0) * w)
    v = float(np.clip(pred_xy[1], 0.0, 1.0) * h)
    cx, cy = float(crop[0]), float(crop[1])
    g = float(gsd)
    if "gsd_m_per_px" in meta and "gsd_km_per_px" not in meta and "gsd" not in meta:
        g = g / 1000.0
    xw = cx + (u - w * 0.5) * g
    yw = cy + (h * 0.5 - v) * g
    pred_world = [float(xw), float(yw)]
    if isinstance(truth, dict) and "x" in truth and "y" in truth:
        gt_world = [float(truth["x"]), float(truth["y"])]
    else:
        gt_world = None
    err = None
    if gt_world is not None:
        err = float(np.hypot(pred_world[0] - gt_world[0], pred_world[1] - gt_world[1]) * 1000.0)
    return pred_world, gt_world, err


def _make_visual(
    img_bgr: np.ndarray,
    pred_xy: np.ndarray,
    gt: np.ndarray,
    *,
    pixel_error: float,
    stage: str,
    background: str,
    world_error: float | None,
    distractor_boxes: list[list[float]] | None = None,
) -> np.ndarray:
    h_img, w_img = img_bgr.shape[:2]
    pred_x = int(np.clip(pred_xy[0], 0.0, 1.0) * w_img)
    pred_y = int(np.clip(pred_xy[1], 0.0, 1.0) * h_img)
    gt_x = int(np.clip(gt[0], 0.0, 1.0) * w_img)
    gt_y = int(np.clip(gt[1], 0.0, 1.0) * h_img)
    vis = img_bgr.copy()
    _draw_distractor_boxes(vis, list(distractor_boxes or []))
    cv2.circle(vis, (gt_x, gt_y), 4, (0, 255, 0), -1)
    cv2.circle(vis, (pred_x, pred_y), 4, (0, 0, 255), -1)
    cv2.line(vis, (gt_x, gt_y), (pred_x, pred_y), (255, 255, 0), 1)
    title = f"gt=({gt_x},{gt_y}) pred=({pred_x},{pred_y}) err={pixel_error:.1f}px {stage} {background}"
    if world_error is not None:
        title += f" world={world_error:.1f}m"
    cv2.putText(vis, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def _select_median(records: list[dict], count: int) -> list[dict]:
    if not records or count <= 0:
        return []
    ordered = sorted(records, key=lambda r: float(r["pixel_error"]))
    mid = len(ordered) // 2
    half = count // 2
    start = max(0, mid - half)
    end = min(len(ordered), start + count)
    start = max(0, end - count)
    return ordered[start:end]


def _save_visual_bucket(*, ds, records: list[dict], bucket_dir: Path, count: int) -> list[str]:
    bucket_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for rank, rec in enumerate(records[: max(0, int(count))]):
        sample = ds[int(rec["sample_index"])]
        pred_xy = np.array(rec["pred_xy"], dtype=np.float32)
        gt = np.array(rec["gt"], dtype=np.float32)
        vis = _make_visual(
            sample.image,
            pred_xy,
            gt,
            pixel_error=float(rec["pixel_error"]),
            stage=str(rec["stage"]),
            background=str(rec["background_category"]),
            world_error=rec.get("world_error_m"),
            distractor_boxes=_distractor_boxes_from_sample(sample),
        )
        name = (
            f"{rank:02d}_err{float(rec['pixel_error']):06.2f}_"
            f"{rec['stage']}_{rec['sequence_id']}_f{rec['frame_id']}.jpg"
        )
        out_path = bucket_dir / name
        cv2.imwrite(str(out_path), vis)
        saved.append(str(out_path))
    return saved


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _p90(vals: list[float]) -> float:
    return float(np.percentile(vals, 90)) if vals else 0.0


def main() -> None:
    args = parse_args()
    torch = _import_torch()
    device = "cpu"
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = "cuda"
    elif args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

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
        load_water_mask=True,
    )
    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights: {weights}")
    ckpt = torch.load(weights, map_location=device)
    ckpt_model_type = str(ckpt.get("model_type", "snn_heatmap"))
    if ckpt_model_type not in {"snn_heatmap", "cnn_heatmap"}:
        raise RuntimeError(f"Expected heatmap checkpoint, got model_type={ckpt_model_type}")

    input_size = int(args.input_size) if int(args.input_size) > 0 else int(ckpt.get("input_size", 256))
    heatmap_size = int(ckpt.get("heatmap_size", 64))
    heatmap_sigma = float(ckpt.get("heatmap_sigma", 1.5))
    heatmap_weight = float(ckpt.get("heatmap_weight", 1.0))
    coord_weight = float(ckpt.get("coord_weight", 5.0))
    conf_weight = float(ckpt.get("conf_weight", 0.2))
    land_penalty_weight = float(ckpt.get("land_penalty_weight", 0.0))
    land_penalty_dilate_px = int(ckpt.get("land_penalty_dilate_px", 0))
    water_logit_constraint = bool(ckpt.get("water_logit_constraint", False))
    water_interior_erode_px = int(ckpt.get("water_interior_erode_px", args.water_interior_erode_px))
    softargmax_temperature = float(ckpt.get("softargmax_temperature", 20.0))
    ckpt_train_encoding = str(ckpt.get("train_encoding", "rate"))
    ckpt_eval_encoding = str(ckpt.get("eval_encoding", "direct"))
    eval_encoding = ckpt_eval_encoding if args.eval_encoding == "auto" else str(args.eval_encoding)
    ckpt_decode_method = str(ckpt.get("decode_method", "softargmax"))
    if ckpt_decode_method == "heatmap_argmax":
        ckpt_decode_method = "argmax"
    decode_method = ckpt_decode_method if args.decode_method == "auto" else str(args.decode_method)

    if ckpt_model_type == "cnn_heatmap":
        model = HeatmapCNN(width=int(ckpt.get("width", 32))).to(device)
    else:
        snn_arch = str(ckpt.get("snn_arch", ckpt.get("arch", "legacy")))
        model = HeatmapSNN(
            beta=float(ckpt.get("beta", 0.95)),
            num_steps=int(ckpt.get("num_steps", 12)),
            train_encoding=ckpt_train_encoding,
            eval_encoding=eval_encoding,
            arch=snn_arch,
        ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    losses: list[float] = []
    px_errors: list[float] = []
    argmax_px_errors: list[float] = []
    softargmax_px_errors: list[float] = []
    center_errors: list[float] = []
    offcenter_px_errors: list[float] = []
    offcenter_center_errors: list[float] = []
    world_errors: list[float] = []
    nearest_distractor_errors: list[float] = []
    pred_closer_to_distractor_count = 0
    frames_with_distractors = 0
    pred_on_land_count = 0
    frames_with_water_mask = 0
    pred_x_px: list[float] = []
    pred_y_px: list[float] = []
    gt_x_px: list[float] = []
    gt_y_px: list[float] = []
    per_stage: dict[str, list[float]] = {"far": [], "mid": [], "terminal": []}
    per_background: dict[str, list[float]] = {}
    rows_head: list[dict] = []
    all_rows: list[dict] = []

    with torch.inference_mode():
        for i in range(len(ds)):
            s = ds[i]
            x_np = _to_tensor_image(s.image, input_size=input_size)
            gt = _target_from_sample(s)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            y = torch.from_numpy(gt.reshape(1, 3)).to(device)
            water = torch.from_numpy(
                _water_mask_from_sample(
                    s,
                    input_size=input_size,
                    interior_erode_px=_water_interior_erode_px_for_sample(s, args),
                )
            ).unsqueeze(0).to(device)
            land = torch.from_numpy(
                _land_mask_from_sample(
                    s,
                    input_size=input_size,
                    dilate_px=land_penalty_dilate_px,
                )
            ).unsqueeze(0).to(device)
            outputs = model(x) if ckpt_model_type == "cnn_heatmap" else model(x, stochastic=False)
            loss, _ = heatmap_loss(
                outputs,
                y,
                heatmap_size=heatmap_size,
                sigma=heatmap_sigma,
                coord_weight=coord_weight,
                heatmap_weight=heatmap_weight,
                conf_weight=conf_weight,
                softargmax_temperature=softargmax_temperature,
                land_mask=land,
                land_weight=land_penalty_weight,
                valid_mask=water if water_logit_constraint else None,
            )
            losses.append(float(loss.item()))
            valid_mask = water if water_logit_constraint else None
            argmax_xy = peak_argmax_2d(outputs["heatmap_logits"], valid_mask=valid_mask)[0].detach().cpu().numpy()
            soft_xy = soft_argmax_2d(
                outputs["heatmap_logits"],
                temperature=softargmax_temperature,
                valid_mask=valid_mask,
            )[0].detach().cpu().numpy()
            pred_xy = soft_xy if decode_method == "softargmax" else argmax_xy
            pred_conf = float(torch.sigmoid(outputs["conf_logits"])[0].detach().cpu().item())

            h, w = s.image.shape[:2]
            err = _pixel_error_norm(pred_xy, gt[:2], h, w)
            arg_err = _pixel_error_norm(argmax_xy, gt[:2], h, w)
            soft_err = _pixel_error_norm(soft_xy, gt[:2], h, w)
            center_err = _pixel_error_norm(np.array([0.5, 0.5], dtype=np.float32), gt[:2], h, w)
            px_errors.append(err)
            argmax_px_errors.append(arg_err)
            softargmax_px_errors.append(soft_err)
            center_errors.append(center_err)
            if center_err >= float(args.offcenter_threshold_px):
                offcenter_px_errors.append(err)
                offcenter_center_errors.append(center_err)
            px = float(np.clip(pred_xy[0], 0.0, 1.0) * w)
            py = float(np.clip(pred_xy[1], 0.0, 1.0) * h)
            gx = float(np.clip(gt[0], 0.0, 1.0) * w)
            gy = float(np.clip(gt[1], 0.0, 1.0) * h)
            pred_x_px.append(px)
            pred_y_px.append(py)
            gt_x_px.append(gx)
            gt_y_px.append(gy)
            pred_world, gt_world, world_err = _image_to_world(pred_xy, s)
            if world_err is not None:
                world_errors.append(float(world_err))
            nearest_distractor_err, distractor_count = _nearest_distractor_error_px(pred_xy, s)
            if nearest_distractor_err is not None:
                nearest_distractor_errors.append(float(nearest_distractor_err))
                frames_with_distractors += 1
                if nearest_distractor_err < err:
                    pred_closer_to_distractor_count += 1
            pred_land = _pred_on_land(pred_xy, s)
            if pred_land is not None:
                frames_with_water_mask += 1
                if pred_land:
                    pred_on_land_count += 1

            st = str(s.meta.get("perception_stage", "unknown"))
            bg = str(s.meta.get("background_category", "unknown")).lower()
            if st in per_stage:
                per_stage[st].append(err)
            per_background.setdefault(bg, []).append(err)

            row = {
                "sample_index": int(i),
                "sequence_id": str(s.sequence_id),
                "frame_id": str(s.frame_id),
                "stage": st,
                "background_category": bg,
                "pixel_error": err,
                "argmax_pixel_error": arg_err,
                "softargmax_pixel_error": soft_err,
                "center_baseline_pixel_error": center_err,
                "center_baseline_delta": center_err - err,
                "pred_xy": [float(v) for v in pred_xy.tolist()],
                "gt": [float(v) for v in gt.tolist()],
                "pred_xy_px": [px, py],
                "gt_xy_px": [gx, gy],
                "pred_conf": pred_conf,
                "gt_conf": float(gt[2]),
                "pred_world": pred_world,
                "gt_world": gt_world,
                "world_error_m": world_err,
                "distractor_count": int(distractor_count),
                "nearest_distractor_error_px": nearest_distractor_err,
                "pred_closer_to_distractor": bool(nearest_distractor_err is not None and nearest_distractor_err < err),
                "pred_on_land": pred_land,
            }
            all_rows.append(row)
            if i < 200:
                rows_head.append(row)
            if i < 16:
                vis = _make_visual(
                    s.image,
                    pred_xy,
                    gt,
                    pixel_error=err,
                    stage=st,
                    background=bg,
                    world_error=world_err,
                    distractor_boxes=_distractor_boxes_from_sample(s),
                )
                cv2.imwrite(str(vis_dir / f"eval_sample_{i:02d}.jpg"), vis)

    center_mean = _mean(center_errors)
    px_mean = _mean(px_errors)
    center_improve_ratio = float((center_mean - px_mean) / max(center_mean, 1e-12)) if center_errors else 0.0
    offcenter_center_mean = _mean(offcenter_center_errors)
    offcenter_px_mean = _mean(offcenter_px_errors)
    offcenter_improve_ratio = (
        float((offcenter_center_mean - offcenter_px_mean) / max(offcenter_center_mean, 1e-12))
        if offcenter_center_errors
        else 0.0
    )
    rounded_pred_xy = {(int(round(x)), int(round(y))) for x, y in zip(pred_x_px, pred_y_px)}
    warnings: list[str] = []
    if center_improve_ratio < float(args.min_center_improve_ratio):
        warnings.append("snn_heatmap_not_enough_better_than_fixed_center_baseline")
    if offcenter_center_errors and offcenter_improve_ratio < float(args.min_offcenter_improve_ratio):
        warnings.append("snn_heatmap_not_enough_better_on_offcenter_subset")
    if len(rounded_pred_xy) < int(args.min_unique_pred_xy):
        warnings.append("prediction_diversity_too_low")

    audit_count = max(0, int(args.visual_audit_count))
    audit_dir = vis_dir / "audit"
    best = sorted(all_rows, key=lambda r: float(r["pixel_error"]))[:audit_count]
    worst = sorted(all_rows, key=lambda r: float(r["pixel_error"]), reverse=True)[:audit_count]
    median = _select_median(all_rows, audit_count)
    visual_audit = {
        "best": _save_visual_bucket(ds=ds, records=best, bucket_dir=audit_dir / "best", count=audit_count),
        "median": _save_visual_bucket(ds=ds, records=median, bucket_dir=audit_dir / "median", count=audit_count),
        "worst": _save_visual_bucket(ds=ds, records=worst, bucket_dir=audit_dir / "worst", count=audit_count),
        "offcenter_worst": _save_visual_bucket(
            ds=ds,
            records=sorted(
                [r for r in all_rows if float(r["center_baseline_pixel_error"]) >= float(args.offcenter_threshold_px)],
                key=lambda r: float(r["pixel_error"]),
                reverse=True,
            )[:audit_count],
            bucket_dir=audit_dir / "offcenter_worst",
            count=audit_count,
        ),
    }

    report = {
        "task": "eval_stage2_pilot_heatmap",
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
        "eval_config": {
            "input_size": int(input_size),
            "heatmap_size": int(heatmap_size),
            "heatmap_sigma": float(heatmap_sigma),
            "heatmap_weight": float(heatmap_weight),
            "coord_weight": float(coord_weight),
            "conf_weight": float(conf_weight),
            "land_penalty_weight": float(land_penalty_weight),
            "land_penalty_dilate_px": int(land_penalty_dilate_px),
            "water_interior_erode_px": int(water_interior_erode_px),
            "water_interior_erode_far_px": int(args.water_interior_erode_far_px),
            "water_interior_erode_mid_px": int(args.water_interior_erode_mid_px),
            "water_interior_erode_terminal_px": int(args.water_interior_erode_terminal_px),
            "water_logit_constraint": bool(water_logit_constraint),
            "water_constrained_decode": bool(water_logit_constraint),
            "softargmax_temperature": float(softargmax_temperature),
            "eval_encoding": str(eval_encoding),
            "checkpoint_train_encoding": str(ckpt_train_encoding),
            "checkpoint_eval_encoding": str(ckpt_eval_encoding),
            "checkpoint_decode_method": str(ckpt_decode_method),
            "decode_method": str(decode_method),
            "loss_kind": str(ckpt.get("loss_kind", "unknown")),
            "checkpoint_model_type": str(ckpt_model_type),
            "checkpoint_snn_arch": str(ckpt.get("snn_arch", ckpt.get("arch", ""))),
        },
        "metrics": {
            "eval_loss_mean": _mean(losses),
            "eval_loss_p90": _p90(losses),
            "pixel_error_mean": px_mean,
            "pixel_error_p90": _p90(px_errors),
            "argmax_pixel_error_mean": _mean(argmax_px_errors),
            "argmax_pixel_error_p90": _p90(argmax_px_errors),
            "softargmax_pixel_error_mean": _mean(softargmax_px_errors),
            "softargmax_pixel_error_p90": _p90(softargmax_px_errors),
            "center_baseline_pixel_error_mean": center_mean,
            "center_baseline_pixel_error_p90": _p90(center_errors),
            "center_baseline_improve_ratio": center_improve_ratio,
            "offcenter_threshold_px": float(args.offcenter_threshold_px),
            "offcenter_count": int(len(offcenter_px_errors)),
            "offcenter_pixel_error_mean": offcenter_px_mean,
            "offcenter_center_baseline_pixel_error_mean": offcenter_center_mean,
            "offcenter_center_baseline_improve_ratio": offcenter_improve_ratio,
            "world_error_mean_m": _mean(world_errors),
            "world_error_p90_m": _p90(world_errors),
            "nearest_distractor_error_mean_px": _mean(nearest_distractor_errors),
            "frames_with_distractors": int(frames_with_distractors),
            "pred_closer_to_distractor_count": int(pred_closer_to_distractor_count),
            "pred_closer_to_distractor_ratio": float(pred_closer_to_distractor_count / max(1, frames_with_distractors)),
            "frames_with_water_mask": int(frames_with_water_mask),
            "pred_on_land_count": int(pred_on_land_count),
            "pred_on_land_ratio": float(pred_on_land_count / max(1, frames_with_water_mask)),
            "stage_pixel_error_mean": {k: _mean(v) for k, v in per_stage.items()},
            "background_pixel_error_mean": {k: _mean(v) for k, v in per_background.items()},
            "prediction_stats": {
                "pred_x_mean_px": _mean(pred_x_px),
                "pred_y_mean_px": _mean(pred_y_px),
                "pred_x_std_px": float(np.std(pred_x_px)) if pred_x_px else 0.0,
                "pred_y_std_px": float(np.std(pred_y_px)) if pred_y_px else 0.0,
                "gt_x_mean_px": _mean(gt_x_px),
                "gt_y_mean_px": _mean(gt_y_px),
                "gt_x_std_px": float(np.std(gt_x_px)) if gt_x_px else 0.0,
                "gt_y_std_px": float(np.std(gt_y_px)) if gt_y_px else 0.0,
                "rounded_unique_pred_xy": int(len(rounded_pred_xy)),
            },
            "diagnostics": {
                "suspect_center_collapse": bool(warnings),
                "warnings": warnings,
                "thresholds": {
                    "min_center_improve_ratio": float(args.min_center_improve_ratio),
                    "min_offcenter_improve_ratio": float(args.min_offcenter_improve_ratio),
                    "min_unique_pred_xy": int(args.min_unique_pred_xy),
                },
            },
        },
        "artifacts": {
            "report_path": str(out_dir / "report.json"),
            "sample_errors_path": str(out_dir / "sample_errors_head.json"),
            "sample_errors_all_path": str(out_dir / "sample_errors_all.json"),
            "visual_dir": str(vis_dir),
            "visual_audit_dir": str(audit_dir),
            "visual_audit": visual_audit,
        },
    }
    (out_dir / "sample_errors_head.json").write_text(json.dumps(rows_head, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "sample_errors_all.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
