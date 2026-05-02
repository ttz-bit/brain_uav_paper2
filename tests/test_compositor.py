from __future__ import annotations

import numpy as np

from paper2.render.compositor import alpha_blend_center, trim_bgra_to_alpha_bbox


def test_trim_bgra_to_alpha_bbox_removes_transparent_padding():
    img = np.zeros((20, 30, 4), dtype=np.uint8)
    img[7:11, 19:25, :3] = 255
    img[7:11, 19:25, 3] = 255

    trimmed = trim_bgra_to_alpha_bbox(img)

    assert trimmed.shape[:2] == (4, 6)
    assert int((trimmed[:, :, 3] > 10).sum()) == 24


def test_alpha_blend_center_returns_visible_alpha_bbox():
    canvas = np.zeros((32, 32, 3), dtype=np.uint8)
    overlay = np.zeros((12, 12, 4), dtype=np.uint8)
    overlay[4:8, 6:10, :3] = 255
    overlay[4:8, 6:10, 3] = 255

    bbox, visibility = alpha_blend_center(canvas, overlay, 16.0, 16.0)

    assert bbox == (16, 14, 4, 4)
    assert visibility == 1.0
