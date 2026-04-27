from __future__ import annotations

import cv2
import numpy as np


def apply_perturbations(image_bgr: np.ndarray, cfg: dict, rng) -> np.ndarray:
    out = image_bgr.astype(np.float32)

    b_jitter = float(cfg.get("brightness_jitter", 0.0))
    c_jitter = float(cfg.get("contrast_jitter", 0.0))
    if b_jitter > 0:
        out = out * float(rng.uniform(1.0 - b_jitter, 1.0 + b_jitter))
    if c_jitter > 0:
        mean = np.mean(out, axis=(0, 1), keepdims=True)
        out = (out - mean) * float(rng.uniform(1.0 - c_jitter, 1.0 + c_jitter)) + mean

    out = np.clip(out, 0.0, 255.0).astype(np.uint8)

    if float(rng.random()) < float(cfg.get("blur_prob", 0.0)):
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0.0)

    if float(rng.random()) < float(cfg.get("haze_prob", 0.0)):
        haze_strength = float(rng.uniform(0.08, 0.22))
        haze = np.full_like(out, 220)
        out = cv2.addWeighted(out, 1.0 - haze_strength, haze, haze_strength, 0.0)

    if float(rng.random()) < float(cfg.get("light_cloud_prob", 0.0)):
        alpha = float(rng.uniform(0.06, 0.14))
        cloud = np.full_like(out, 255)
        out = cv2.addWeighted(out, 1.0 - alpha, cloud, alpha, 0.0)

    if float(rng.random()) < float(cfg.get("compression_prob", 0.0)):
        quality = int(rng.integers(55, 86))
        ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is not None:
                out = dec

    return out
