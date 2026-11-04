"""Wall mask and wall segment extraction.

Pipeline
--------
1. ``wall_mask``: keep only ink that is at least ~half a wall thick
   (morphological opening with a square kernel). This removes text, furniture,
   door arcs and window glazing lines, which are all thin. Hollow walls (drawn
   as two parallel thin lines) are optionally filled first with a closing.
2. ``extract_segments``: open the wall mask with long 1-px kernels to split it
   into horizontal and vertical bars, then turn each connected bar into a
   centerline segment with a measured thickness.
3. ``merge_collinear`` + ``snap_junctions``: join pieces of the same wall and
   pull wall ends onto the centerline of the wall they run into, so the result
   is a connected wall graph that a CAD/drawing tool can consume directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class Seg:
    """Axis-aligned wall piece in working-image pixels.

    For ``o == "h"``: ``c`` is the y of the centerline, ``a..b`` the x extent.
    For ``o == "v"``: ``c`` is the x of the centerline, ``a..b`` the y extent.
    ``raw_a/raw_b`` keep the ink extent before snapping (used by gap search).
    """
    o: str
    c: float
    a: float
    b: float
    t: float
    raw_a: float = 0.0
    raw_b: float = 0.0

    def __post_init__(self):
        if not self.raw_a and not self.raw_b:
            self.raw_a, self.raw_b = self.a, self.b

    @property
    def length(self) -> float:
        return self.b - self.a


def _square(k: int) -> np.ndarray:
    k = max(1, int(round(k)))
    return np.ones((k, k), np.uint8)


def wall_mask(ink: np.ndarray, wall_t: float, fill_hollow: bool = True, line_w: float = 1.0) -> np.ndarray:
    """Binary mask (0/1) of wall pixels.

    The square opening kernel is ~0.4x the main wall thickness so that
    interior walls (often half as thick as exterior ones) survive, but always
    wider than the thin drawing lines."""
    k_open = max(2, int(round(line_w * 2 + 1)), int(round(wall_t * 0.4)))
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, _square(k_open))
    if not fill_hollow:
        return solid
    # Hollow walls: two thin parallel lines ~wall_t apart. Close each axis
    # separately so that only line *pairs* get bridged, not arbitrary clutter.
    k_close = max(3, int(round(wall_t * 0.9)))
    closed_h = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((k_close, 1), np.uint8))
    closed_v = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, k_close), np.uint8))
    L = int(round(wall_t * 4))
    hollow_h = cv2.morphologyEx(closed_h, cv2.MORPH_OPEN, np.ones((k_open, L), np.uint8))
    hollow_v = cv2.morphologyEx(closed_v, cv2.MORPH_OPEN, np.ones((L, k_open), np.uint8))
    return (solid | hollow_h | hollow_v).astype(np.uint8)


def _runs(flags: np.ndarray, max_gap: int = 0):
    """Contiguous True runs of a 1-D bool array as (start, end) half-open,
    bridging gaps of up to ``max_gap`` False values."""
    idx = np.flatnonzero(flags)
    if not len(idx):
        return []
    out = []
    s = p = idx[0]
    for i in idx[1:]:
        if i - p - 1 > max_gap:
            out.append((s, p + 1))
            s = i
        p = i
    out.append((s, p + 1))
    return out


def _component_to_segs(comp: np.ndarray, x0: int, y0: int, wall_t: float, min_len: float) -> List[Seg]:
    """Turn one horizontal-bar component into one or more wall segments.

    A component may hold several stacked walls joined by a perpendicular
    wall, and may have furniture blobs or columns glued to it. Rows are
    therefore picked by *relative* coverage: first the rows that carry most of
    the component (the main wall), then whatever long bands remain.
    """
    fill = comp.sum(axis=1).astype(float)
    segs: List[Seg] = []
    for _ in range(6):
        if fill.max() < min_len:
            break
        thr = max(min_len, 0.35 * fill.max())
        for r0, r1 in _runs(fill >= thr, max_gap=1):
            band = comp[r0:r1]
            t = r1 - r0
            col = band.sum(axis=0) >= max(1, 0.5 * t)
            for c0, c1 in _runs(col, max_gap=int(wall_t * 0.3)):
                length = c1 - c0
                if length < min_len or length < 1.5 * t:
                    continue
                if t > 2.5 * wall_t and length < 4 * t:
                    continue                      # filled block / column, not a wall
                segs.append(Seg("h", y0 + (r0 + r1) / 2, float(x0 + c0), float(x0 + c1), float(t)))
            fill[r0:r1] = 0
    return segs


def _bars(mask: np.ndarray, o: str, wall_t: float, min_len: float) -> List[Seg]:
    L = max(3, int(round(min_len)))
    kernel = np.ones((1, L), np.uint8) if o == "h" else np.ones((L, 1), np.uint8)
    bars = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    work = bars if o == "h" else np.ascontiguousarray(bars.T)   # make every bar horizontal
    n, lab, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=4)
    segs: List[Seg] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w < min_len:
            continue
        comp = (lab[y:y + h, x:x + w] == i)
        for sg in _component_to_segs(comp, x, y, wall_t, min_len):
            if sg.t >= wall_t * 0.4:
                sg.o = o
                segs.append(sg)
    return segs


def extract_segments(mask: np.ndarray, wall_t: float, min_len_factor: float = 1.6) -> List[Seg]:
    min_len = wall_t * min_len_factor
    segs = _bars(mask, "h", wall_t, min_len) + _bars(mask, "v", wall_t, min_len)
    # A bar shorter than ~2x its thickness that sits inside a perpendicular
    # wall is just the end/corner of that wall seen from the other axis.
    keep = []
    for s in segs:
        if s.length < 2.2 * s.t and any(
            p.o != s.o and abs(p.c - (s.a + s.b) / 2) < p.t and p.a - p.t / 2 <= s.c <= p.b + p.t / 2
            for p in segs
        ):
            continue
        keep.append(s)
    return keep


def merge_collinear(segs: List[Seg], wall_t: float) -> List[Seg]:
    """Merge segments on the same axis line that overlap or touch."""
    out: List[Seg] = []
    for o in ("h", "v"):
        group = sorted([s for s in segs if s.o == o], key=lambda s: (s.c, s.a))
        used = [False] * len(group)
        for i, s in enumerate(group):
            if used[i]:
                continue
            cur = Seg(o, s.c, s.a, s.b, s.t, s.raw_a, s.raw_b)
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j, q in enumerate(group):
                    if used[j]:
                        continue
                    if abs(q.c - cur.c) <= max(cur.t, q.t) * 0.5 and q.a <= cur.b + 1 and q.b >= cur.a - 1:
                        wa, wb = cur.length, q.length
                        cur.c = (cur.c * wa + q.c * wb) / (wa + wb)
                        cur.t = (cur.t * wa + q.t * wb) / (wa + wb)
                        cur.a, cur.b = min(cur.a, q.a), max(cur.b, q.b)
                        cur.raw_a, cur.raw_b = min(cur.raw_a, q.raw_a), max(cur.raw_b, q.raw_b)
                        used[j] = True
                        changed = True
            out.append(cur)
    return out


def snap_junctions(segs: List[Seg], wall_t: float) -> None:
    """Move each wall end onto the centerline of a perpendicular wall it runs
    into (L/T junctions), in place. Also aligns nearly-equal centerlines."""
    # 1. align parallel walls whose centerlines differ by < 0.35 t (drawing noise)
    for o in ("h", "v"):
        group = sorted([s for s in segs if s.o == o], key=lambda s: s.c)
        i = 0
        while i < len(group):
            j = i
            while j + 1 < len(group) and group[j + 1].c - group[i].c < wall_t * 0.35:
                j += 1
            if j > i:
                cl = group[i:j + 1]
                c = sum(s.c * s.length for s in cl) / sum(s.length for s in cl)
                for s in cl:
                    s.c = c
            i = j + 1

    # 2. snap ends to perpendicular centerlines
    tol = wall_t * 1.0
    for s in segs:
        perp = [p for p in segs if p.o != s.o]
        for end in ("a", "b"):
            v = getattr(s, end)
            best, best_d = None, tol
            for p in perp:
                # p must span s's centerline
                if not (p.a - p.t <= s.c <= p.b + p.t):
                    continue
                d = abs(p.c - v)
                # the end must be within the perpendicular wall's body or just short of it
                if d < best_d + 1e-6 and d <= max(tol, p.t * 0.75):
                    best, best_d = p, d
            if best is not None:
                setattr(s, end, best.c)
                # extend the perpendicular wall too if s lands just past its end
                if s.c < best.a and best.a - s.c <= tol:
                    best.a = s.c
                elif s.c > best.b and s.c - best.b <= tol:
                    best.b = s.c
