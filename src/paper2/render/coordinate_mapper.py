from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorldState:
    x: float
    y: float
    vx: float
    vy: float
    heading: float


def world_to_image(
    target_x: float,
    target_y: float,
    crop_x: float,
    crop_y: float,
    gsd_m_per_px: float,
    image_size: int,
) -> tuple[float, float]:
    half = float(image_size) * 0.5
    u = half + (target_x - crop_x) / max(1e-6, float(gsd_m_per_px))
    v = half - (target_y - crop_y) / max(1e-6, float(gsd_m_per_px))
    return u, v


def world_to_background_px(x_world: float, y_world: float, world_size_m: float, bg_w: int, bg_h: int) -> tuple[float, float]:
    x_norm = min(1.0, max(0.0, x_world / max(1e-6, float(world_size_m))))
    y_norm = min(1.0, max(0.0, y_world / max(1e-6, float(world_size_m))))
    return x_norm * float(bg_w - 1), (1.0 - y_norm) * float(bg_h - 1)


def background_px_to_world(x_px: float, y_px: float, world_size_m: float, bg_w: int, bg_h: int) -> tuple[float, float]:
    x_norm = min(1.0, max(0.0, float(x_px) / max(1e-6, float(bg_w - 1))))
    y_norm = min(1.0, max(0.0, float(y_px) / max(1e-6, float(bg_h - 1))))
    return x_norm * float(world_size_m), (1.0 - y_norm) * float(world_size_m)
