from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_bgra(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError(f"Expected BGRA image, got shape={img.shape} path={path}")
    return img


def resize_bgra_with_scale(img_bgra: np.ndarray, scale: float, image_size: int) -> np.ndarray:
    h, w = img_bgra.shape[:2]
    # scale is interpreted as a fraction of the output image size.
    long_side = max(1, int(round(float(image_size) * float(scale))))
    long_side = min(long_side, int(image_size * 0.9))
    ratio = long_side / max(1, float(max(h, w)))
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    return cv2.resize(img_bgra, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def rotate_bgra(img_bgra: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_bgra.shape[:2]
    if h <= 0 or w <= 0:
        return img_bgra
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cos = abs(m[0, 0])
    sin = abs(m[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    m[0, 2] += (new_w / 2.0) - center[0]
    m[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(
        img_bgra,
        m,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def trim_bgra_to_alpha_bbox(img_bgra: np.ndarray, *, alpha_threshold: int = 10) -> np.ndarray:
    if img_bgra.ndim != 3 or img_bgra.shape[2] != 4:
        raise ValueError(f"Expected BGRA image, got shape={img_bgra.shape}")
    alpha = img_bgra[:, :, 3] > int(alpha_threshold)
    ys, xs = np.where(alpha)
    if len(xs) <= 0:
        return img_bgra
    x1 = int(xs.min())
    x2 = int(xs.max()) + 1
    y1 = int(ys.min())
    y2 = int(ys.max()) + 1
    return img_bgra[y1:y2, x1:x2].copy()


def alpha_blend_center(
    canvas_bgr: np.ndarray,
    overlay_bgra: np.ndarray,
    center_x: float,
    center_y: float,
) -> tuple[tuple[int, int, int, int], float]:
    h, w = canvas_bgr.shape[:2]
    oh, ow = overlay_bgra.shape[:2]

    x1 = int(round(center_x - ow / 2))
    y1 = int(round(center_y - oh / 2))
    x2 = x1 + ow
    y2 = y1 + oh

    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(w, x2)
    iy2 = min(h, y2)
    total_alpha_pixels = int((overlay_bgra[:, :, 3] > 10).sum())
    if ix1 >= ix2 or iy1 >= iy2:
        return (x1, y1, ow, oh), 0.0

    ox1 = ix1 - x1
    oy1 = iy1 - y1
    ox2 = ox1 + (ix2 - ix1)
    oy2 = oy1 + (iy2 - iy1)

    patch = overlay_bgra[oy1:oy2, ox1:ox2]
    alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
    fg = patch[:, :, :3].astype(np.float32)
    bg = canvas_bgr[iy1:iy2, ix1:ix2].astype(np.float32)
    out = alpha * fg + (1.0 - alpha) * bg
    canvas_bgr[iy1:iy2, ix1:ix2] = out.astype(np.uint8)

    visible_alpha = patch[:, :, 3] > 10
    visible_alpha_pixels = int(visible_alpha.sum())
    if total_alpha_pixels <= 0:
        visibility = 0.0
    else:
        visibility = float(visible_alpha_pixels / float(total_alpha_pixels))
    if visible_alpha_pixels <= 0:
        bbox = (ix1, iy1, ix2 - ix1, iy2 - iy1)
    else:
        vy, vx = np.where(visible_alpha)
        bx1 = ix1 + int(vx.min())
        by1 = iy1 + int(vy.min())
        bx2 = ix1 + int(vx.max()) + 1
        by2 = iy1 + int(vy.max()) + 1
        bbox = (bx1, by1, bx2 - bx1, by2 - by1)
    return bbox, visibility
