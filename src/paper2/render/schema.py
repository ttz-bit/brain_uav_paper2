from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RenderedFrameRecord:
    image_path: str
    split: str
    sequence_id: str
    frame_id: str
    stage: str
    observation_source: str
    gsd_m_per_px: float
    target_center_px: list[float]
    bbox_xywh: list[float]
    visibility: float
    background_asset_id: str
    target_asset_id: str
    distractor_asset_ids: list[str]
    motion_mode: str
    land_overlap_ratio: float
    shore_buffer_overlap_ratio: float
    scale_px: float
    angle_deg: float
    obs_valid: bool
    meta: dict

    def to_dict(self) -> dict:
        return asdict(self)
