from __future__ import annotations

import cv2
import numpy as np


def apply_perturbations(image_bgr: np.ndarray, cfg: dict, rng) -> np.ndarray:
    out = image_bgr.astype(np.float32)

    if "brightness_factor" in cfg:
        out = out * float(cfg["brightness_factor"])
    else:
        b_jitter = float(cfg.get("brightness_jitter", 0.0))
        if b_jitter > 0:
            out = out * float(rng.uniform(1.0 - b_jitter, 1.0 + b_jitter))

    if "contrast_factor" in cfg:
        mean = np.mean(out, axis=(0, 1), keepdims=True)
        out = (out - mean) * float(cfg["contrast_factor"]) + mean
    else:
        c_jitter = float(cfg.get("contrast_jitter", 0.0))
        if c_jitter > 0:
            mean = np.mean(out, axis=(0, 1), keepdims=True)
            out = (out - mean) * float(rng.uniform(1.0 - c_jitter, 1.0 + c_jitter)) + mean

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    blur_kernel = int(cfg.get("blur_kernel", 0))
    if blur_kernel >= 3 and blur_kernel % 2 == 1:
        out = cv2.GaussianBlur(out, (blur_kernel, blur_kernel), sigmaX=0.0)
    elif float(rng.random()) < float(cfg.get("blur_prob", 0.0)):
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0.0)

    if "haze_alpha" in cfg:
        haze_strength = float(cfg.get("haze_alpha", 0.0))
        if haze_strength > 0:
            haze = np.full_like(out, 220)
            out = cv2.addWeighted(out, 1.0 - haze_strength, haze, haze_strength, 0.0)
    elif float(rng.random()) < float(cfg.get("haze_prob", 0.0)):
        haze_strength = float(rng.uniform(0.08, 0.22))
        haze = np.full_like(out, 220)
        out = cv2.addWeighted(out, 1.0 - haze_strength, haze, haze_strength, 0.0)

    if "cloud_alpha" in cfg:
        alpha = float(cfg.get("cloud_alpha", 0.0))
        if alpha > 0:
            cloud = np.full_like(out, 255)
            out = cv2.addWeighted(out, 1.0 - alpha, cloud, alpha, 0.0)
    elif float(rng.random()) < float(cfg.get("light_cloud_prob", 0.0)):
        alpha = float(rng.uniform(0.06, 0.14))
        cloud = np.full_like(out, 255)
        out = cv2.addWeighted(out, 1.0 - alpha, cloud, alpha, 0.0)

    jpeg_quality = int(cfg.get("compression_quality", 0))
    if 1 <= jpeg_quality <= 100:
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec
    elif float(rng.random()) < float(cfg.get("compression_prob", 0.0)):
        quality = int(rng.integers(55, 86))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec

    return out
