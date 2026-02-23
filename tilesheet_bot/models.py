from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TilesheetParams:
    tile_width: int
    tile_height: int
    grid_cols: int
    grid_rows: int
    padding: int = 0
    order: str = "time"
    bg_mode: str = "remove"
    alpha_threshold: int = 16
    margin_bottom: int = 8
    clamp_min: float = 0.85
    clamp_max: float = 1.15
    resample: str = "bicubic"


@dataclass
class StoredImage:
    path: Path
    filename: str
    order_index: int
    frame_count: int = 1
