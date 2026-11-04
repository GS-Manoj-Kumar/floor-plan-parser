"""Plain data classes for the vectorized floor plan.

All coordinates are in image pixels (origin top-left, y down) unless a field
name ends in ``_mm``. Millimetre values are filled in only when a scale is known.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Wall:
    id: int
    start: Point                 # centerline start (px)
    end: Point                   # centerline end (px)
    thickness: float             # px
    orientation: str             # "h", "v" or "d" (diagonal)
    exterior: bool = False

    @property
    def length(self) -> float:
        return ((self.end[0] - self.start[0]) ** 2 + (self.end[1] - self.start[1]) ** 2) ** 0.5


@dataclass
class Opening:
    id: int
    kind: str                    # "door", "window" or "opening" (plain gap / passage)
    start: Point                 # where the opening starts on the wall axis (px)
    end: Point                   # where it ends (px)
    thickness: float             # wall thickness at the opening (px)
    orientation: str             # "h" or "v"
    host_wall_ids: List[int] = field(default_factory=list)
    confidence: float = 0.0
    hinge: Optional[Point] = None         # doors: hinge point (px)
    swing_side: Optional[str] = None      # doors: "up"/"down" (h) or "left"/"right" (v)
    double: bool = False                  # doors: double leaf

    @property
    def width(self) -> float:
        return ((self.end[0] - self.start[0]) ** 2 + (self.end[1] - self.start[1]) ** 2) ** 0.5

    @property
    def center(self) -> Point:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)


@dataclass
class FloorPlan:
    image_size: Tuple[int, int]           # (width, height) px
    walls: List[Wall]
    openings: List[Opening]
    mm_per_px: Optional[float] = None
    wall_thickness_px: float = 0.0
    line_width_px: float = 0.0

    # ------------------------------------------------------------------ export
    def _mm(self, v: float) -> Optional[float]:
        return None if self.mm_per_px is None else round(v * self.mm_per_px, 1)

    def _pt_mm(self, p: Point):
        return None if self.mm_per_px is None else [self._mm(p[0]), self._mm(p[1])]

    def to_dict(self) -> dict:
        walls = []
        for w in self.walls:
            walls.append({
                "id": w.id,
                "orientation": w.orientation,
                "exterior": w.exterior,
                "start_px": [round(w.start[0], 1), round(w.start[1], 1)],
                "end_px": [round(w.end[0], 1), round(w.end[1], 1)],
                "length_px": round(w.length, 1),
                "thickness_px": round(w.thickness, 1),
                "start_mm": self._pt_mm(w.start),
                "end_mm": self._pt_mm(w.end),
                "length_mm": self._mm(w.length),
                "thickness_mm": self._mm(w.thickness),
            })
        openings = []
        for o in self.openings:
            d = {
                "id": o.id,
                "type": o.kind,
                "orientation": o.orientation,
                "host_wall_ids": o.host_wall_ids,
                "confidence": round(o.confidence, 2),
                "start_px": [round(o.start[0], 1), round(o.start[1], 1)],
                "end_px": [round(o.end[0], 1), round(o.end[1], 1)],
                "center_px": [round(o.center[0], 1), round(o.center[1], 1)],
                "width_px": round(o.width, 1),
                "thickness_px": round(o.thickness, 1),
                "start_mm": self._pt_mm(o.start),
                "end_mm": self._pt_mm(o.end),
                "center_mm": self._pt_mm(o.center),
                "width_mm": self._mm(o.width),
            }
            if o.host_wall_ids:
                hw = next((w for w in self.walls if w.id == o.host_wall_ids[0]), None)
                if hw is not None:
                    off = ((o.start[0] - hw.start[0]) ** 2 + (o.start[1] - hw.start[1]) ** 2) ** 0.5
                    d["offset_on_host_px"] = round(off, 1)
                    d["offset_on_host_mm"] = self._mm(off)
            if o.kind == "door":
                d["hinge_px"] = None if o.hinge is None else [round(o.hinge[0], 1), round(o.hinge[1], 1)]
                d["hinge_mm"] = None if o.hinge is None else self._pt_mm(o.hinge)
                d["swing_side"] = o.swing_side
                d["double"] = o.double
            openings.append(d)
        return {
            "image_size_px": list(self.image_size),
            "mm_per_px": self.mm_per_px,
            "wall_thickness_px": round(self.wall_thickness_px, 2),
            "line_width_px": round(self.line_width_px, 2),
            "walls": walls,
            "openings": openings,
        }


__all__ = ["Point", "Wall", "Opening", "FloorPlan", "asdict"]
