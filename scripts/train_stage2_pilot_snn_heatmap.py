from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from paper2.datasets.stage2_rendered_dataset import build_stage2_rendered_dataset
try:
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
    p.add_argument("--train-split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--val-split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-steps", type=int, default=12)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--input-size", type=int, default=256)
    p.add_argument("--heatmap-size", type=int, default=64)
    p.add_argument("--heatmap-sigma", type=float, default=1.5)
    p.add_argument("--heatmap-weight", type=float, default=1.0)
    p.add_argument("--coord-weight", type=float, default=5.0)
    p.add_argument("--conf-weight", type=float, default=0.2)
    p.add_argument("--distractor-repel-weight", type=float, default=0.0)
    p.add_argument("--distractor-repel-sigma", type=float, default=2.0)
    p.add_argument("--max-distractors", type=int, default=4)
    p.add_argument("--land-penalty-weight", type=float, default=0.0)
    p.add_argument("--land-penalty-dilate-px", type=int, default=4)
    p.add_argument("--water-logit-constraint", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--water-interior-erode-px", type=int, default=6)
    p.add_argument("--water-interior-erode-far-px", type=int, default=10)
    p.add_argument("--water-interior-erode-mid-px", type=int, default=6)
    p.add_argument("--water-interior-erode-terminal-px", type=int, default=4)
    p.add_argument("--softargmax-temperature", type=float, default=20.0)
    p.add_argument("--decode-method", type=str, default="softargmax", choices=["argmax", "softargmax"])
    p.add_argument("--train-encoding", type=str, default="direct", choices=["rate", "direct"])
    p.add_argument("--eval-encoding", type=str, default="direct", choices=["rate", "direct"])
    p.add_argument("--init-weights", type=str, default="")
    p.add_argument("--eval-interval", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument(
        "--selection-metric",
        type=str,
        default="val_pixel_error",
        choices=["val_loss", "val_pixel_error", "val_argmax_pixel_error", "val_softargmax_pixel_error", "val_center_improve"],
    )
    p.add_argument("--train-eval-max-samples", type=int, default=2048)
    p.add_argument("--val-eval-max-samples", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--persistent-workers", action="store_true")
    p.add_argument("--amp", type=str, default="none", choices=["none", "fp16", "bf16"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--save-last-each-epoch", action="store_true")
    p.add_argument("--num-threads", type=int, default=0)
    p.add_argument("--num-interop-threads", type=int, default=0)
    p.add_argument("--strict-no-leak", action="store_true")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "stage2_pre_baselines" / "snn_heatmap_fit_v2"),
    )
    return p.parse_args()


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except Exception as e:
        raise RuntimeError("PyTorch is required.") from e
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


def _distractors_from_sample(sample, *, max_distractors: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = sample.image.shape[:2]
    centers = np.zeros((max(0, int(max_distractors)), 2), dtype=np.float32)
    mask = np.zeros((max(0, int(max_distractors)),), dtype=np.float32)
    if max_distractors <= 0:
        return centers, mask
    raw_boxes = list((sample.meta or {}).get("distractor_bboxes_xywh", []))
    for i, box in enumerate(raw_boxes[: int(max_distractors)]):
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            continue
        x, y, bw, bh = [float(v) for v in box[:4]]
        if bw <= 0.0 or bh <= 0.0:
            continue
        centers[i, 0] = np.float32((x + 0.5 * bw) / max(1.0, float(w)))
        centers[i, 1] = np.float32((y + 0.5 * bh) / max(1.0, float(h)))
        mask[i] = np.float32(1.0)
    return centers, mask


def _crop_origin_from_meta(meta: dict) -> list[float] | None:
    for key in ("crop_origin_bg_px", "crop_origin_xy", "crop_bg_xy", "crop_top_left"):
        value = meta.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return [float(value[0]), float(value[1])]
    return None


def _resolve_optional_path(path_raw: object, project_root: Path) -> Path | None:
    if path_raw is None:
        return None
    path = Path(str(path_raw))
    if path.exists():
        return path
    if not path.is_absolute():
        candidate = (project_root / path).resolve()
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def _metadata_coverage(ds, *, project_root: Path) -> dict[str, float | int]:
    rows = list(getattr(ds, "_rows", []))
    total = len(rows)
    crop_ready = 0
    water_ready = 0
    distractor_asset_rows = 0
    distractor_bbox_rows = 0
    for row in rows:
        meta = dict(row.get("meta", {}))
        if _crop_origin_from_meta(meta) is not None:
            crop_ready += 1
        crop_mask_path = _resolve_optional_path(meta.get("water_mask_crop_path"), project_root)
        full_mask_path = _resolve_optional_path(meta.get("water_mask_path"), project_root)
        if (crop_mask_path is not None and crop_mask_path.exists()) or (
            _crop_origin_from_meta(meta) is not None and full_mask_path is not None and full_mask_path.exists()
        ):
            water_ready += 1
        if len(list(row.get("distractor_asset_ids", []))) > 0:
            distractor_asset_rows += 1
        if len(list(meta.get("distractor_bboxes_xywh", []))) > 0:
            distractor_bbox_rows += 1
    return {
        "total_rows": int(total),
        "crop_origin_rows": int(crop_ready),
        "crop_origin_missing": int(total - crop_ready),
        "water_mask_rows": int(water_ready),
        "water_mask_missing": int(total - water_ready),
        "distractor_asset_rows": int(distractor_asset_rows),
        "distractor_bbox_rows": int(distractor_bbox_rows),
        "distractor_rows": int(distractor_bbox_rows),
        "crop_origin_coverage": float(crop_ready / max(1, total)),
        "water_mask_coverage": float(water_ready / max(1, total)),
        "distractor_asset_row_ratio": float(distractor_asset_rows / max(1, total)),
        "distractor_bbox_row_ratio": float(distractor_bbox_rows / max(1, total)),
        "distractor_row_ratio": float(distractor_bbox_rows / max(1, total)),
    }


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


def _pixel_error_norm(pred_xy: np.ndarray, gt_xy: np.ndarray, h: int, w: int) -> float:
    px = float(np.clip(pred_xy[0], 0.0, 1.0) * w)
    py = float(np.clip(pred_xy[1], 0.0, 1.0) * h)
    gx = float(np.clip(gt_xy[0], 0.0, 1.0) * w)
    gy = float(np.clip(gt_xy[1], 0.0, 1.0) * h)
    return float(np.hypot(px - gx, py - gy))


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


def _make_visual(
    img_bgr: np.ndarray,
    pred_xy: np.ndarray,
    gt: np.ndarray,
    tag: str,
    err: float,
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
    cv2.putText(
        vis,
        f"{tag} gt=({gt_x},{gt_y}) pred=({pred_x},{pred_y}) err={err:.1f}px",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
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
    torch, DataLoader, Dataset = _import_torch()
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

    water_constraint = bool(args.water_logit_constraint) or float(args.land_penalty_weight) > 0.0
    load_water_mask = bool(water_constraint)
    train_ds = build_stage2_rendered_dataset(
        root=dataset_root,
        split=args.train_split,
        project_root=project_root,
        max_samples=args.max_train_samples,
        load_water_mask=load_water_mask,
    )
    val_ds = build_stage2_rendered_dataset(
        root=dataset_root,
        split=args.val_split,
        project_root=project_root,
        max_samples=args.max_val_samples,
        load_water_mask=load_water_mask,
    )

    train_meta = _metadata_coverage(train_ds, project_root=project_root)
    val_meta = _metadata_coverage(val_ds, project_root=project_root)
    if load_water_mask and (train_meta["water_mask_missing"] > 0 or val_meta["water_mask_missing"] > 0):
        raise RuntimeError(
            "Water-mask supervision requested but crop/mask metadata is incomplete. "
            f"train_missing={train_meta['water_mask_missing']}, val_missing={val_meta['water_mask_missing']}. "
            "Re-render the dataset with water_mask_crop_path populated, or with crop_origin_bg_px / water_mask_path populated."
        )
    if float(args.distractor_repel_weight) > 0.0 and (
        train_meta["distractor_bbox_rows"] <= 0 or val_meta["distractor_bbox_rows"] <= 0
    ):
        raise RuntimeError(
            "Distractor repel supervision requested but distractor bbox metadata is unavailable. "
            f"train_bbox_rows={train_meta['distractor_bbox_rows']}, val_bbox_rows={val_meta['distractor_bbox_rows']}. "
            "Re-render the dataset with distractor_bboxes_xywh populated, or rerun with --distractor-repel-weight 0."
        )

    class _RenderedTorchDataset(Dataset):
        def __init__(self, ds, indices: np.ndarray | None = None):
            self.ds = ds
            self.indices = indices

        def __len__(self) -> int:
            return int(len(self.ds) if self.indices is None else len(self.indices))

        def __getitem__(self, idx: int):
            src_idx = int(idx if self.indices is None else self.indices[int(idx)])
            sample = self.ds[src_idx]
            x = _to_tensor_image(sample.image, input_size=int(args.input_size))
            y = _target_from_sample(sample)
            d_xy, d_mask = _distractors_from_sample(sample, max_distractors=int(args.max_distractors))
            land = _land_mask_from_sample(
                sample,
                input_size=int(args.input_size),
                dilate_px=int(args.land_penalty_dilate_px),
            )
            water = _water_mask_from_sample(
                sample,
                input_size=int(args.input_size),
                interior_erode_px=_water_interior_erode_px_for_sample(sample, args),
            )
            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(d_xy),
                torch.from_numpy(d_mask),
                torch.from_numpy(land),
                torch.from_numpy(water),
            )

    num_workers = max(0, int(args.num_workers))
    def _loader(ds, *, batch_size: int, shuffle: bool):
        kwargs = {
            "batch_size": max(1, int(batch_size)),
            "shuffle": bool(shuffle),
            "drop_last": False,
            "num_workers": num_workers,
            "pin_memory": (device == "cuda"),
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
            kwargs["persistent_workers"] = bool(args.persistent_workers)
        return DataLoader(ds, **kwargs)

    loader = _loader(_RenderedTorchDataset(train_ds), batch_size=int(args.batch_size), shuffle=True)
    model = HeatmapSNN(
        beta=float(args.beta),
        num_steps=int(args.num_steps),
        train_encoding=str(args.train_encoding),
        eval_encoding=str(args.eval_encoding),
    ).to(device)
    if bool(args.channels_last) and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    init_weights = str(args.init_weights or "").strip()
    if init_weights:
        init_path = Path(init_weights).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Missing init weights: {init_path}")
        init_ckpt = torch.load(init_path, map_location=device)
        state_dict = init_ckpt.get("state_dict", init_ckpt)
        model.load_state_dict(state_dict, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    amp_enabled = device == "cuda" and str(args.amp) != "none"
    amp_dtype = torch.float16 if str(args.amp) == "fp16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=bool(amp_enabled and str(args.amp) == "fp16"))

    def _loss(outputs, targets, distractor_centers=None, distractor_mask=None, land_mask=None, water_mask=None):
        return heatmap_loss(
            outputs,
            targets,
            heatmap_size=int(args.heatmap_size),
            sigma=float(args.heatmap_sigma),
            coord_weight=float(args.coord_weight),
            heatmap_weight=float(args.heatmap_weight),
            conf_weight=float(args.conf_weight),
            softargmax_temperature=float(args.softargmax_temperature),
            distractor_centers=distractor_centers,
            distractor_mask=distractor_mask,
            distractor_weight=float(args.distractor_repel_weight),
            distractor_sigma=float(args.distractor_repel_sigma),
            land_mask=land_mask,
            land_weight=float(args.land_penalty_weight),
            valid_mask=water_mask if water_constraint else None,
        )

    eval_batch_size = max(1, int(args.eval_batch_size))

    def _sample_indices(n: int, max_samples: int) -> np.ndarray | None:
        if int(max_samples) <= 0 or int(n) <= int(max_samples):
            return None
        return rng.choice(int(n), size=int(max_samples), replace=False).astype(np.int64)

    train_eval_indices = _sample_indices(len(train_ds), int(args.train_eval_max_samples))
    val_eval_indices = _sample_indices(len(val_ds), int(args.val_eval_max_samples))
    train_eval_count = int(len(train_ds) if train_eval_indices is None else len(train_eval_indices))
    val_eval_count = int(len(val_ds) if val_eval_indices is None else len(val_eval_indices))
    train_eval_loader = _loader(_RenderedTorchDataset(train_ds, train_eval_indices), batch_size=int(args.eval_batch_size), shuffle=False)
    val_eval_loader = _loader(_RenderedTorchDataset(val_ds, val_eval_indices), batch_size=int(args.eval_batch_size), shuffle=False)

    def _eval_loader(eval_loader) -> dict[str, float]:
        model.eval()
        total = 0.0
        count = 0
        argmax_px_errors: list[float] = []
        softargmax_px_errors: list[float] = []
        center_errors: list[float] = []
        pred_x_px: list[float] = []
        pred_y_px: list[float] = []
        with torch.inference_mode():
            for xb, yb, db, dmask, land, water in eval_loader:
                xb = xb.to(device, non_blocking=(device == "cuda"))
                yb = yb.to(device, non_blocking=(device == "cuda"))
                db = db.to(device, non_blocking=(device == "cuda"))
                dmask = dmask.to(device, non_blocking=(device == "cuda"))
                land = land.to(device, non_blocking=(device == "cuda"))
                water = water.to(device, non_blocking=(device == "cuda"))
                if bool(args.channels_last) and device == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                    outputs = model(xb, stochastic=False)
                    loss, _ = _loss(outputs, yb, db, dmask, land, water)
                n = int(xb.shape[0])
                total += float(loss.item()) * n
                count += n
                valid_mask = water if water_constraint else None
                argmax_xy = peak_argmax_2d(outputs["heatmap_logits"], valid_mask=valid_mask).detach().cpu().numpy()
                soft_xy = soft_argmax_2d(
                    outputs["heatmap_logits"],
                    temperature=float(args.softargmax_temperature),
                    valid_mask=valid_mask,
                ).detach().cpu().numpy()
                pred_xy = soft_xy if str(args.decode_method) == "softargmax" else argmax_xy
                gt_xy = yb[:, :2].detach().cpu().numpy()
                for arg_xy, soft_pred_xy, pxy, gxy in zip(argmax_xy, soft_xy, pred_xy, gt_xy):
                    arg_err = _pixel_error_norm(arg_xy, gxy, int(args.input_size), int(args.input_size))
                    soft_err = _pixel_error_norm(soft_pred_xy, gxy, int(args.input_size), int(args.input_size))
                    center_err = _pixel_error_norm(np.array([0.5, 0.5], dtype=np.float32), gxy, int(args.input_size), int(args.input_size))
                    argmax_px_errors.append(arg_err)
                    softargmax_px_errors.append(soft_err)
                    center_errors.append(center_err)
                    pred_x_px.append(float(np.clip(pxy[0], 0.0, 1.0) * int(args.input_size)))
                    pred_y_px.append(float(np.clip(pxy[1], 0.0, 1.0) * int(args.input_size)))
        px_errors = softargmax_px_errors if str(args.decode_method) == "softargmax" else argmax_px_errors
        px_mean = float(np.mean(px_errors)) if px_errors else 0.0
        center_mean = float(np.mean(center_errors)) if center_errors else 0.0
        rounded_unique = {(int(round(x)), int(round(y))) for x, y in zip(pred_x_px, pred_y_px)}
        return {
            "loss": total / max(1, count),
            "pixel_error_mean": px_mean,
            "pixel_error_p90": float(np.percentile(px_errors, 90)) if px_errors else 0.0,
            "argmax_pixel_error_mean": float(np.mean(argmax_px_errors)) if argmax_px_errors else 0.0,
            "argmax_pixel_error_p90": float(np.percentile(argmax_px_errors, 90)) if argmax_px_errors else 0.0,
            "softargmax_pixel_error_mean": float(np.mean(softargmax_px_errors)) if softargmax_px_errors else 0.0,
            "softargmax_pixel_error_p90": float(np.percentile(softargmax_px_errors, 90)) if softargmax_px_errors else 0.0,
            "center_baseline_pixel_error_mean": center_mean,
            "center_baseline_improve_ratio": float((center_mean - px_mean) / max(center_mean, 1e-12)) if center_errors else 0.0,
            "pred_x_std_px": float(np.std(pred_x_px)) if pred_x_px else 0.0,
            "pred_y_std_px": float(np.std(pred_y_px)) if pred_y_px else 0.0,
            "rounded_unique_pred_xy": int(len(rounded_unique)),
        }

    def _selection_score(metrics: dict[str, float]) -> float:
        metric = str(args.selection_metric)
        if metric == "val_loss":
            return float(metrics["loss"])
        if metric == "val_argmax_pixel_error":
            return float(metrics["argmax_pixel_error_mean"])
        if metric == "val_softargmax_pixel_error":
            return float(metrics["softargmax_pixel_error_mean"])
        if metric == "val_center_improve":
            return -float(metrics["center_baseline_improve_ratio"])
        return float(metrics["pixel_error_mean"])

    with torch.inference_mode():
        train_initial = _eval_loader(train_eval_loader)
        val_initial = _eval_loader(val_eval_loader)

    best_score = float("inf")
    best_val_loss = float("inf")
    best_path = out_dir / "model_best.pth"
    last_path = out_dir / "model_last.pth"
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
        batch_parts: list[dict[str, float]] = []
        t_train_loop_begin = time.perf_counter()
        for xb, yb, db, dmask, land, water in loader:
            xb = xb.to(device, non_blocking=(device == "cuda"))
            yb = yb.to(device, non_blocking=(device == "cuda"))
            db = db.to(device, non_blocking=(device == "cuda"))
            dmask = dmask.to(device, non_blocking=(device == "cuda"))
            land = land.to(device, non_blocking=(device == "cuda"))
            water = water.to(device, non_blocking=(device == "cuda"))
            if bool(args.channels_last) and device == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                outputs = model(xb, stochastic=True)
                loss, parts = _loss(outputs, yb, db, dmask, land, water)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_losses.append(float(loss.detach().item()))
            batch_parts.append(parts)
        train_loop_sec_total += float(time.perf_counter() - t_train_loop_begin)

        do_eval = ((ep + 1) % eval_interval == 0) or (ep + 1 == epochs)
        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_loss = float("nan")
        if do_eval:
            t_eval_begin = time.perf_counter()
            val_metrics = _eval_loader(val_eval_loader)
            val_loss = float(val_metrics["loss"])
            val_eval_sec_total += float(time.perf_counter() - t_eval_begin)
        else:
            val_metrics = {}
        selection_score = _selection_score(val_metrics) if do_eval else float("nan")
        loss_trace.append(
            {
                "epoch": int(ep + 1),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "selection_metric": str(args.selection_metric),
                "selection_score": float(selection_score),
                "train_parts_mean": {
                    k: float(np.mean([p[k] for p in batch_parts])) if batch_parts else 0.0
                    for k in ("heatmap_loss", "coord_loss", "conf_loss", "distractor_loss", "land_loss")
                },
                "val_metrics": val_metrics,
            }
        )
        if do_eval and selection_score < best_score:
            best_score = float(selection_score)
            best_val_loss = float(val_loss)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": "snn_heatmap",
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
                    "distractor_repel_weight": float(args.distractor_repel_weight),
                    "distractor_repel_sigma": float(args.distractor_repel_sigma),
                    "max_distractors": int(args.max_distractors),
                    "land_penalty_weight": float(args.land_penalty_weight),
                    "land_penalty_dilate_px": int(args.land_penalty_dilate_px),
                    "water_interior_erode_px": int(args.water_interior_erode_px),
                    "water_interior_erode_far_px": int(args.water_interior_erode_far_px),
                    "water_interior_erode_mid_px": int(args.water_interior_erode_mid_px),
                    "water_interior_erode_terminal_px": int(args.water_interior_erode_terminal_px),
                    "water_logit_constraint": bool(water_constraint),
                    "softargmax_temperature": float(args.softargmax_temperature),
                    "loss_kind": "spatial_cross_entropy_plus_coordinate_plus_distractor_repel_plus_land_penalty_plus_water_logit_constraint",
                    "num_steps": int(args.num_steps),
                    "beta": float(args.beta),
                    "train_encoding": str(args.train_encoding),
                    "eval_encoding": str(args.eval_encoding),
                    "init_weights": init_weights,
                    "decode_method": str(args.decode_method),
                    "epoch": int(ep + 1),
                    "selection_metric": str(args.selection_metric),
                    "best_selection_score": float(best_score),
                    "best_val_loss": float(best_val_loss),
                    "optimizer_state": optimizer.state_dict(),
                },
                best_path,
            )
        if bool(args.save_last_each_epoch):
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": "snn_heatmap",
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
                    "distractor_repel_weight": float(args.distractor_repel_weight),
                    "distractor_repel_sigma": float(args.distractor_repel_sigma),
                    "max_distractors": int(args.max_distractors),
                    "land_penalty_weight": float(args.land_penalty_weight),
                    "land_penalty_dilate_px": int(args.land_penalty_dilate_px),
                    "water_interior_erode_px": int(args.water_interior_erode_px),
                    "water_interior_erode_far_px": int(args.water_interior_erode_far_px),
                    "water_interior_erode_mid_px": int(args.water_interior_erode_mid_px),
                    "water_interior_erode_terminal_px": int(args.water_interior_erode_terminal_px),
                    "water_logit_constraint": bool(water_constraint),
                    "softargmax_temperature": float(args.softargmax_temperature),
                    "loss_kind": "spatial_cross_entropy_plus_coordinate_plus_distractor_repel_plus_land_penalty_plus_water_logit_constraint",
                    "num_steps": int(args.num_steps),
                    "beta": float(args.beta),
                    "train_encoding": str(args.train_encoding),
                    "eval_encoding": str(args.eval_encoding),
                    "init_weights": init_weights,
                    "decode_method": str(args.decode_method),
                    "epoch": int(ep + 1),
                    "selection_metric": str(args.selection_metric),
                    "best_selection_score": float(best_score),
                    "best_val_loss": float(best_val_loss),
                    "optimizer_state": optimizer.state_dict(),
                },
                last_path,
            )
        (out_dir / "loss_trace.json").write_text(json.dumps(loss_trace, ensure_ascii=False, indent=2), encoding="utf-8")
        epoch_sec = float(time.perf_counter() - t_epoch_begin)
        epoch_time_sec.append(epoch_sec)
        val_msg = f"{val_loss:.6f}" if do_eval else "skip"
        print(
            f"[SNN-HEATMAP-FIT] epoch={ep+1:03d}/{epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_msg} "
            f"selection={selection_score:.6f} epoch_sec={epoch_sec:.1f}",
            flush=True,
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "snn_heatmap",
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
            "distractor_repel_weight": float(args.distractor_repel_weight),
            "distractor_repel_sigma": float(args.distractor_repel_sigma),
            "max_distractors": int(args.max_distractors),
            "land_penalty_weight": float(args.land_penalty_weight),
            "land_penalty_dilate_px": int(args.land_penalty_dilate_px),
            "water_interior_erode_px": int(args.water_interior_erode_px),
            "water_interior_erode_far_px": int(args.water_interior_erode_far_px),
            "water_interior_erode_mid_px": int(args.water_interior_erode_mid_px),
            "water_interior_erode_terminal_px": int(args.water_interior_erode_terminal_px),
            "water_logit_constraint": bool(water_constraint),
            "softargmax_temperature": float(args.softargmax_temperature),
            "loss_kind": "spatial_cross_entropy_plus_coordinate_plus_distractor_repel_plus_land_penalty_plus_water_logit_constraint",
            "num_steps": int(args.num_steps),
            "beta": float(args.beta),
            "train_encoding": str(args.train_encoding),
            "eval_encoding": str(args.eval_encoding),
            "init_weights": init_weights,
            "decode_method": str(args.decode_method),
            "epoch": int(epochs),
            "selection_metric": str(args.selection_metric),
            "best_selection_score": float(best_score),
            "best_val_loss": float(best_val_loss),
            "optimizer_state": optimizer.state_dict(),
        },
        last_path,
    )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    train_final = _eval_loader(train_eval_loader)
    val_final = _eval_loader(val_eval_loader)
    train_improve_ratio = (float(train_initial["loss"]) - float(train_final["loss"])) / max(float(train_initial["loss"]), 1e-12)
    val_improve_ratio = (float(val_initial["loss"]) - float(val_final["loss"])) / max(float(val_initial["loss"]), 1e-12)

    for tag, ds, n_vis in [("train", train_ds, 8), ("val", val_ds, 8)]:
        idxs = rng.choice(len(ds), size=min(n_vis, len(ds)), replace=False)
        with torch.inference_mode():
            model.eval()
            for rank, idx in enumerate(idxs):
                s = ds[int(idx)]
                x = torch.from_numpy(_to_tensor_image(s.image, input_size=int(args.input_size))).unsqueeze(0).to(device)
                water = torch.from_numpy(_water_mask_from_sample(s, input_size=int(args.input_size))).unsqueeze(0).to(device)
                if bool(args.channels_last) and device == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=bool(amp_enabled)):
                    outputs = model(x, stochastic=False)
                if str(args.decode_method) == "softargmax":
                    pred_xy = soft_argmax_2d(
                        outputs["heatmap_logits"],
                        temperature=float(args.softargmax_temperature),
                        valid_mask=water if water_constraint else None,
                    )[0].detach().cpu().numpy()
                else:
                    pred_xy = peak_argmax_2d(
                        outputs["heatmap_logits"],
                        valid_mask=water if water_constraint else None,
                    )[0].detach().cpu().numpy()
                gt = _target_from_sample(s)
                err = _pixel_error_norm(pred_xy, gt[:2], s.image.shape[0], s.image.shape[1])
                vis = _make_visual(
                    s.image,
                    pred_xy,
                    gt,
                    tag=tag,
                    err=err,
                    distractor_boxes=_distractor_boxes_from_sample(s),
                )
                cv2.imwrite(str(vis_dir / f"{tag}_sample_{rank:02d}.jpg"), vis)

    report = {
        "task": "train_stage2_pilot_snn_heatmap",
        "purpose": "fit_check",
        "dataset": {
            "name": "stage2_rendered",
            "root": str(dataset_root),
            "train_split": str(args.train_split),
            "val_split": str(args.val_split),
            "num_train": int(len(train_ds)),
            "num_val": int(len(val_ds)),
        },
        "data_quality": {
            "train": train_meta,
            "val": val_meta,
            "water_mask_supervision_enabled": bool(load_water_mask),
            "water_logit_constraint_enabled": bool(water_constraint),
            "water_mask_metadata_complete": bool(
                train_meta["water_mask_missing"] == 0 and val_meta["water_mask_missing"] == 0
            ),
            "distractor_bbox_metadata_available": bool(
                train_meta["distractor_bbox_rows"] > 0 and val_meta["distractor_bbox_rows"] > 0
            ),
        },
        "device": str(device),
        "model": {
            "type": "snn_heatmap",
            "output": "64x64 heatmap + confidence by default",
            "decode_method": str(args.decode_method),
            "water_constrained_decode": bool(water_constraint),
        },
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
            "init_weights": init_weights,
            "input_size": int(args.input_size),
            "heatmap_size": int(args.heatmap_size),
            "heatmap_sigma": float(args.heatmap_sigma),
            "heatmap_weight": float(args.heatmap_weight),
            "coord_weight": float(args.coord_weight),
            "conf_weight": float(args.conf_weight),
            "distractor_repel_weight": float(args.distractor_repel_weight),
            "distractor_repel_sigma": float(args.distractor_repel_sigma),
            "max_distractors": int(args.max_distractors),
            "land_penalty_weight": float(args.land_penalty_weight),
            "land_penalty_dilate_px": int(args.land_penalty_dilate_px),
            "water_interior_erode_px": int(args.water_interior_erode_px),
            "water_interior_erode_far_px": int(args.water_interior_erode_far_px),
            "water_interior_erode_mid_px": int(args.water_interior_erode_mid_px),
            "water_interior_erode_terminal_px": int(args.water_interior_erode_terminal_px),
            "water_logit_constraint": bool(water_constraint),
            "selection_metric": str(args.selection_metric),
            "softargmax_temperature": float(args.softargmax_temperature),
            "decode_method": str(args.decode_method),
            "eval_interval": int(eval_interval),
            "eval_batch_size": int(eval_batch_size),
            "train_eval_max_samples": int(train_eval_count),
            "val_eval_max_samples": int(val_eval_count),
            "num_workers": int(num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "persistent_workers": bool(args.persistent_workers),
            "amp": str(args.amp),
            "channels_last": bool(args.channels_last),
            "save_last_each_epoch": bool(args.save_last_each_epoch),
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "timing": {
            "materialize_sec": 0.0,
            "streaming_dataset": True,
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
            "train_initial_loss": float(train_initial["loss"]),
            "train_initial_pixel_error_mean": float(train_initial["pixel_error_mean"]),
            "train_initial_center_baseline_improve_ratio": float(train_initial["center_baseline_improve_ratio"]),
            "train_final_loss": float(train_final["loss"]),
            "train_improve_ratio": float(train_improve_ratio),
            "val_initial_loss": float(val_initial["loss"]),
            "val_initial_pixel_error_mean": float(val_initial["pixel_error_mean"]),
            "val_initial_center_baseline_improve_ratio": float(val_initial["center_baseline_improve_ratio"]),
            "val_final_loss": float(val_final["loss"]),
            "val_improve_ratio": float(val_improve_ratio),
            "train_final_pixel_error_mean": float(train_final["pixel_error_mean"]),
            "val_final_pixel_error_mean": float(val_final["pixel_error_mean"]),
            "train_final_argmax_pixel_error_mean": float(train_final["argmax_pixel_error_mean"]),
            "val_final_argmax_pixel_error_mean": float(val_final["argmax_pixel_error_mean"]),
            "train_final_softargmax_pixel_error_mean": float(train_final["softargmax_pixel_error_mean"]),
            "val_final_softargmax_pixel_error_mean": float(val_final["softargmax_pixel_error_mean"]),
            "train_center_baseline_improve_ratio": float(train_final["center_baseline_improve_ratio"]),
            "val_center_baseline_improve_ratio": float(val_final["center_baseline_improve_ratio"]),
            "val_rounded_unique_pred_xy": int(val_final["rounded_unique_pred_xy"]),
        },
        "success_criteria": {
            "train_final_lt_initial": bool(train_final["loss"] < train_initial["loss"]),
            "train_improve_ratio_gt_0": bool(train_improve_ratio > 0.0),
            "val_final_lt_initial": bool(val_final["loss"] < val_initial["loss"]),
            "val_improve_ratio_gt_0": bool(val_improve_ratio > 0.0),
            "val_beats_center_baseline": bool(val_final["center_baseline_improve_ratio"] > 0.0),
            "val_prediction_not_constant": bool(val_final["rounded_unique_pred_xy"] > 5),
            "water_mask_metadata_complete": bool(
                train_meta["water_mask_missing"] == 0 and val_meta["water_mask_missing"] == 0
            ),
            "requested_supervision_metadata_available": bool(
                ((not water_constraint) or (train_meta["water_mask_missing"] == 0 and val_meta["water_mask_missing"] == 0))
                and (
                    float(args.distractor_repel_weight) <= 0.0
                    or (train_meta["distractor_bbox_rows"] > 0 and val_meta["distractor_bbox_rows"] > 0)
                )
            ),
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
