"""Detect walls and windows that are drawn with thin lines.

Many floor plans (especially Chinese/Asian sales plans) draw only the load
bearing walls solid black. Partition walls are drawn as two thin parallel
lines ("hollow" walls) and windows as 3-4 thin parallel lines inside the wall
band. This module finds groups of parallel thin lines whose total spread is
about one wall thickness and labels each stretch:

* 2 lines, empty between them            -> hollow wall
* 2 lines with more line(s) in between    -> window
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from .walls import Seg, _runs


@dataclass
class ThinLine:
    c: float      # across-axis position (y for h, x for v)
    a: int        # along-axis extent
    b: int
    w: float      # stroke width


def _thin_lines(thin: np.ndarray, min_len: int, max_w: float) -> List[ThinLine]:
    """Long horizontal thin lines in ``thin`` (lines must be horizontal)."""
    bars = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((1, min_len), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bars, connectivity=8)
    out: List[ThinLine] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        comp = lab[y:y + h, x:x + w] == i
        fill = comp.sum(axis=1)
        # a component can be two lines touching at a crossing: split by rows
        for r0, r1 in _runs(fill >= min_len * 0.6):
            band = comp[r0:r1]
            t = band.sum(axis=0)
            cols = np.flatnonzero(t > 0)
            width = float(np.median(t[cols])) if len(cols) else 0
            if width > max_w:
                continue
            for c0, c1 in _runs(t > 0, max_gap=2):
                if c1 - c0 >= min_len:
                    rows = np.arange(r0, r1)
                    rf = band.sum(axis=1)
                    cy = float((rows * rf).sum() / max(1, rf.sum())) + 0.5
                    out.append(ThinLine(y + cy, x + c0, x + c1, width))
    return out


def detect_line_bands(ink: np.ndarray, solid: np.ndarray, wall_t: float, line_w: float
                      ) -> Tuple[List[Seg], List[Seg]]:
    """Return (hollow_walls, window_bands) as Seg lists in image pixels."""
    k = max(1, int(round(line_w)))
    thin = (ink & (1 - cv2.dilate(solid, np.ones((2 * k + 1, 2 * k + 1), np.uint8)))).astype(np.uint8)
    walls: List[Seg] = []
    windows: List[Seg] = []
    min_len = max(5, int(round(wall_t * 1.5)))
    for o in ("h", "v"):
        img = thin if o == "h" else np.ascontiguousarray(thin.T)
        lines = _thin_lines(img, min_len, max_w=max(2.0, wall_t * 0.45))
        lines.sort(key=lambda l: l.c)
        claimed = np.zeros(img.shape, np.uint8)
        pairs = []
        for i, l1 in enumerate(lines):
            for l2 in lines[i + 1:]:
                d = l2.c - l1.c
                if d > wall_t * 1.45:
                    break
                if d < wall_t * 0.35:
                    continue
                a, b = max(l1.a, l2.a), min(l1.b, l2.b)
                if b - a >= min_len:
                    pairs.append((d, l1, l2, a, b))
        pairs.sort(key=lambda p: -p[0])   # widest band first so it absorbs inner pairs
        for d, l1, l2, a, b in pairs:
            y1, y2 = int(round(l1.c)), int(round(l2.c))
            if claimed[(y1 + y2) // 2, a:b].mean() > 0.5:
                continue
            inner0 = int(np.ceil(l1.c + l1.w / 2 + 1))
            inner1 = int(np.floor(l2.c - l2.w / 2 - 1))
            if inner1 <= inner0:
                continue
            strip = img[inner0:inner1, a:b]
            # a column "has a middle line" if any interior row is inked there;
            # only count rows that are inked along a long stretch (true lines,
            # not perpendicular crossings or text)
            row_cov = strip.mean(axis=1)
            mid_rows = row_cov > 0.25
            col_mid = strip[mid_rows].any(axis=0) if mid_rows.any() else np.zeros(b - a, bool)
            col_any = strip.any(axis=0)
            t = d + (l1.w + l2.w) / 2
            c = (l1.c + l2.c) / 2
            for c0, c1 in _runs(col_mid, max_gap=int(wall_t * 0.5)):
                if c1 - c0 >= wall_t * 1.5 and not _texture(img, c, t, a + c0, a + c1):
                    windows.append(Seg(o, c, float(a + c0), float(a + c1), t))
            for c0, c1 in _runs(~col_any, max_gap=int(wall_t * 0.4)):
                if c1 - c0 >= wall_t * 2.0 and not _texture(img, c, t, a + c0, a + c1):
                    walls.append(Seg(o, c, float(a + c0), float(a + c1), t))
            claimed[y1:y2 + 1, a:b] = 1
    return walls, windows


def _texture(img: np.ndarray, c: float, t: float, a: int, b: int) -> bool:
    """Parallel lines that continue beyond the band on BOTH sides are a
    pattern (floor planks, tiles, hatching), not a wall or a window."""
    H = img.shape[0]
    hits = 0
    for sgn in (-1, 1):
        r0 = int(round(c + sgn * (t * 0.5 + 2)))
        r1 = int(round(c + sgn * (t * 1.6 + 4)))
        lo, hi = max(0, min(r0, r1)), min(H, max(r0, r1) + 1)
        band = img[lo:hi, a:b]
        if band.size and (band.mean(axis=1) > 0.6).any():
            hits += 1
    return hits == 2


def keep_connected(cands: List[Seg], anchors: List[Seg], wall_t: float, both_ends: bool = False) -> List[Seg]:
    """Keep candidate segments that touch the wall network (anchors), growing
    the network as candidates are accepted. Removes furniture outlines."""
    def touches(s: Seg, p: Seg, end: float) -> bool:
        tol = wall_t * 1.2 + (s.t + p.t) / 2
        if p.o == s.o:
            return abs(p.c - s.c) < tol * 0.6 and (p.a - tol <= end <= p.b + tol)
        return abs(p.c - end) < tol and (p.a - tol <= s.c <= p.b + tol)

    net = list(anchors)
    kept: List[Seg] = []
    pending = list(cands)
    changed = True
    while changed:
        changed = False
        rest = []
        for s in pending:
            ta = any(touches(s, p, s.a) for p in net)
            tb = any(touches(s, p, s.b) for p in net)
            if (ta and tb) if both_ends else (ta or tb):
                kept.append(s)
                net.append(s)
                changed = True
            else:
                rest.append(s)
        pending = rest
    return kept
