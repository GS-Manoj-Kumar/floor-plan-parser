"""Find openings (gaps in walls) and classify them as door / window / opening.

A gap is found by walking from every wall end along the wall axis until the
walk hits wall ink again. The content of the gap then decides its type:

* **door**   - a quarter-circle arc (and usually a leaf line) hinged at one
               end of the gap, or two half-width arcs (double door)
* **window** - thin lines running parallel to the wall across the gap
* **opening**- nothing drawn: a plain passage (only kept if short)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .walls import Seg


@dataclass
class Gap:
    o: str
    c: float
    a: float
    b: float
    t: float
    kind: str = "opening"
    score: float = 0.0
    hinge: Optional[Tuple[float, float]] = None
    swing: Optional[str] = None
    double: bool = False
    walls: List[int] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.b - self.a


def _xy(o: str, along: float, across: float) -> Tuple[float, float]:
    return (along, across) if o == "h" else (across, along)


def find_gaps(segs: List[Seg], barrier: np.ndarray, wall_t: float, min_w: float, max_w: float) -> List[Gap]:
    H, W = barrier.shape
    gaps: List[Gap] = []
    for idx, s in enumerate(segs):
        half = max(1, int(round(min(s.t, wall_t) * 0.3)))
        c = int(round(s.c))
        for end, step in ((s.raw_b, 1), (s.raw_a, -1)):
            start = int(round(end)) + (0 if step > 0 else -1)
            hit = None
            for d in range(1, int(max_w) + 2):
                p = start + step * d
                if s.o == "h":
                    if p < 0 or p >= W:
                        break
                    col = barrier[max(0, c - half):c + half + 1, p]
                else:
                    if p < 0 or p >= H:
                        break
                    col = barrier[p, max(0, c - half):c + half + 1]
                if col.any():
                    hit = p
                    break
            if hit is None:
                continue
            if step > 0:
                a, b = float(int(round(end))), float(hit)
            else:
                a, b = float(hit + 1), float(int(round(end)))
            if min_w <= b - a <= max_w:
                gaps.append(Gap(s.o, s.c, a, b, s.t, walls=[idx]))
    return _dedupe(gaps, wall_t)


def _dedupe(gaps: List[Gap], wall_t: float) -> List[Gap]:
    out: List[Gap] = []
    for g in sorted(gaps, key=lambda g: (g.o, g.c, g.a)):
        for q in out:
            if q.o == g.o and abs(q.c - g.c) < wall_t * 0.6 and g.a < q.b - 1 and g.b > q.a + 1:
                # same opening found from both sides: keep the tighter extent
                q.a, q.b = max(q.a, g.a), min(q.b, g.b)
                q.walls = sorted(set(q.walls) | set(g.walls))
                break
        else:
            out.append(g)
    return out


# ----------------------------------------------------------------- classify
def _sample(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < W) & (pts[:, 1] >= 0) & (pts[:, 1] < H)
    return mask[y, x].astype(bool) & inside


def _arc_score(near: np.ndarray, o: str, c: float, hinge_along: float, toward: int, side: int,
               r: float, face: float) -> float:
    """Score of a quarter arc hinged at ``hinge_along`` on the wall face.

    ``toward`` (+1/-1): direction along the wall from the hinge to the latch.
    ``side`` (+1/-1): which side of the wall the door swings into.
    The arc goes from the latch point (on the wall line) to the open-leaf tip
    (perpendicular to the wall). The leaf is the straight line hinge->tip.
    """
    across0 = c + side * face
    ang = np.linspace(0.12, math.pi / 2 - 0.12, 24)
    along = hinge_along + toward * r * np.cos(ang)
    across = across0 + side * r * np.sin(ang)
    pts = np.stack(_xy(o, along, across), axis=1)
    arc = _sample(near, pts).mean()
    lt = np.linspace(0.15, 0.95, 12)
    l_along = np.full_like(lt, hinge_along)
    l_across = across0 + side * r * lt
    lpts = np.stack(_xy(o, l_along, l_across), axis=1)
    leaf = _sample(near, lpts).mean()
    return 0.75 * arc + 0.25 * leaf


def classify_gap(g: Gap, thin_near: np.ndarray, thin: np.ndarray, wall_t: float, line_w: float) -> None:
    w = g.width
    face = g.t / 2
    best = (0.0, None)
    for side in (1, -1):
        for hinge_end in ("a", "b"):
            h_along = g.a if hinge_end == "a" else g.b
            toward = 1 if hinge_end == "a" else -1
            for rf in (1.0, 0.92, 0.85):
                sc = _arc_score(thin_near, g.o, g.c, h_along, toward, side, w * rf, face)
                if sc > best[0]:
                    best = (sc, (hinge_end, side, False))
        # double door: two half-width leaves on the same side
        for rf in (1.0, 0.9):
            s1 = _arc_score(thin_near, g.o, g.c, g.a, 1, side, w / 2 * rf, face)
            s2 = _arc_score(thin_near, g.o, g.c, g.b, -1, side, w / 2 * rf, face)
            sc = min(s1, s2) * 1.02
            if sc > best[0]:
                best = (sc, ("ab", side, True))

    # window evidence: thin lines parallel to the wall, across the gap, inside the wall band
    half = max(2, int(round(g.t * 0.75)))
    c = int(round(g.c))
    a, b = int(round(g.a + w * 0.1)), int(round(g.b - w * 0.1))
    if g.o == "h":
        band = thin[max(0, c - half):c + half + 1, a:max(a + 1, b)]
    else:
        band = thin[a:max(a + 1, b), max(0, c - half):c + half + 1].T
    row_cov = band.mean(axis=1) if band.size else np.zeros(1)
    n_lines = int((_runs_count(row_cov > 0.7)))
    win_score = min(1.0, n_lines / 2.0) if n_lines else 0.0

    door_score, cfg = best
    if door_score >= 0.55 and door_score >= win_score * 0.8:
        g.kind, g.score = "door", float(door_score)
        hinge_end, side, double = cfg
        g.double = double
        if not double:
            ha = g.a if hinge_end == "a" else g.b
            g.hinge = _xy(g.o, ha, g.c)
        if g.o == "h":
            g.swing = "down" if side > 0 else "up"
        else:
            g.swing = "right" if side > 0 else "left"
    elif n_lines >= 1:
        g.kind, g.score = "window", float(max(0.5, win_score))
    else:
        g.kind, g.score = "opening", float(1.0 - door_score)


def _runs_count(flags: np.ndarray) -> int:
    if not flags.any():
        return 0
    f = flags.astype(np.int8)
    return int(((np.diff(np.concatenate(([0], f))) == 1)).sum())
