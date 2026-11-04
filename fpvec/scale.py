"""Helpers for turning the user's calibration into a mm-per-pixel scale."""
from __future__ import annotations

import math
from typing import Tuple


def mm_per_px_from_reference(p1: Tuple[float, float], p2: Tuple[float, float], length_mm: float) -> float:
    """User clicked two points on the image and typed the real distance."""
    d = math.dist(p1, p2)
    if d <= 0:
        raise ValueError("reference points must differ")
    return length_mm / d


def mm_per_px_from_wall(wall_thickness_px: float, wall_thickness_mm: float = 200.0) -> float:
    """Rough scale from the detected wall thickness (e.g. 200 or 240 mm walls)."""
    return wall_thickness_mm / wall_thickness_px
