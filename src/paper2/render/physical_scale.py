from __future__ import annotations

from typing import Any


DEFAULT_TARGET_LENGTH_KM = 0.2
DEFAULT_TARGET_WIDTH_KM = 0.04


def target_size_km(stage_cfg: dict[str, Any]) -> tuple[float, float]:
    size_cfg = dict(stage_cfg.get("target_size", {}))
    length = float(size_cfg.get("length_km", DEFAULT_TARGET_LENGTH_KM))
    width = float(size_cfg.get("width_km", DEFAULT_TARGET_WIDTH_KM))
    if length <= 0.0 or width <= 0.0:
        raise ValueError("target_size length_km and width_km must be positive.")
    return length, width


def target_dimensions_px_from_km(
    *,
    gsd_km_per_px: float,
    stage_cfg: dict[str, Any],
    image_size: int,
) -> tuple[float, float]:
    length_km, width_km = target_size_km(stage_cfg)
    return _clamped_dimensions_px(length_km / float(gsd_km_per_px), width_km / float(gsd_km_per_px), image_size)


def target_dimensions_px_from_m(
    *,
    gsd_m_per_px: float,
    target_cfg: dict[str, Any],
    image_size: int,
) -> tuple[float, float]:
    size_cfg = dict(target_cfg.get("physical_size_m", {}))
    length_m = float(size_cfg.get("length_m", DEFAULT_TARGET_LENGTH_KM * 1000.0))
    width_m = float(size_cfg.get("width_m", DEFAULT_TARGET_WIDTH_KM * 1000.0))
    if length_m <= 0.0 or width_m <= 0.0:
        raise ValueError("target.physical_size_m length_m and width_m must be positive.")
    return _clamped_dimensions_px(length_m / float(gsd_m_per_px), width_m / float(gsd_m_per_px), image_size)


def target_scale_fraction_from_m(
    *,
    gsd_m_per_px: float,
    target_cfg: dict[str, Any],
    image_size: int,
) -> float:
    length_px, _ = target_dimensions_px_from_m(
        gsd_m_per_px=gsd_m_per_px,
        target_cfg=target_cfg,
        image_size=image_size,
    )
    return float(length_px / max(1.0, float(image_size)))


def _clamped_dimensions_px(length_px: float, width_px: float, image_size: int) -> tuple[float, float]:
    length = float(length_px)
    width = float(width_px)
    if length <= 0.0 or width <= 0.0:
        raise ValueError("Physical target dimensions must map to positive pixel dimensions.")
    max_long = float(image_size) * 0.85
    if length > max_long:
        scale = max_long / length
        length *= scale
        width *= scale
    return max(1.0, length), max(1.0, width)
