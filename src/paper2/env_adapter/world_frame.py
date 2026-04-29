from __future__ import annotations

import numpy as np


def paper1_xy_to_paper2_xy(xy: np.ndarray, world_size_m: float = 4000.0) -> np.ndarray:
    xy_arr = np.asarray(xy, dtype=float).reshape(-1)
    if xy_arr.size < 2:
        raise ValueError("xy must contain at least two values.")
    offset = 0.5 * float(world_size_m)
    return xy_arr[:2] + np.array([offset, offset], dtype=float)


def paper2_xy_to_paper1_xy(xy: np.ndarray, world_size_m: float = 4000.0) -> np.ndarray:
    xy_arr = np.asarray(xy, dtype=float).reshape(-1)
    if xy_arr.size < 2:
        raise ValueError("xy must contain at least two values.")
    offset = 0.5 * float(world_size_m)
    return xy_arr[:2] - np.array([offset, offset], dtype=float)


def paper1_xyz_to_paper2_xyz(xyz: np.ndarray, world_size_m: float = 4000.0) -> np.ndarray:
    xyz_arr = np.asarray(xyz, dtype=float).reshape(-1)
    if xyz_arr.size < 3:
        raise ValueError("xyz must contain three values.")
    xy = paper1_xy_to_paper2_xy(xyz_arr[:2], world_size_m=world_size_m)
    return np.array([xy[0], xy[1], xyz_arr[2]], dtype=float)


def paper2_xyz_to_paper1_xyz(xyz: np.ndarray, world_size_m: float = 4000.0) -> np.ndarray:
    xyz_arr = np.asarray(xyz, dtype=float).reshape(-1)
    if xyz_arr.size < 3:
        raise ValueError("xyz must contain three values.")
    xy = paper2_xy_to_paper1_xy(xyz_arr[:2], world_size_m=world_size_m)
    return np.array([xy[0], xy[1], xyz_arr[2]], dtype=float)
