"""Direct door detection from swing arcs.

Swing doors are drawn as a quarter circle centred on the hinge, plus (usually)
a straight leaf line. Detecting the arc gives the hinge, the door width (the
radius), the wall direction and the swing side, without needing the wall gap
to have been found first. This catches doors standing between two parallel
wall ends (e.g. an entrance between two wall stubs) that a walk along a wall
axis would never reach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DoorArc:
    hinge: Tuple[float, float]
    latch: Tuple[float, float]      # arc end on the wall line
    tip: Tuple[float, float]        # arc end at the open leaf
    r: float
    o: str                          # orientation of the wall the door sits in
    score: float
    jambs: int = 0                  # how many of hinge/latch touch wall ink


def _fit_circle(pts: np.ndarray) -> Tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit. Returns cx, cy, r, rms residual."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    r = math.sqrt(max(1e-9, sol[2] + cx * cx + cy * cy))
    res = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    return cx, cy, r, float(np.sqrt(np.mean(res ** 2)))


def _line_ink(mask: np.ndarray, p: Tuple[float, float], q: Tuple[float, float], n: int = 20) -> float:
    H, W = mask.shape
    t = np.linspace(0.12, 0.92, n)
    xs = np.clip(np.round(p[0] + (q[0] - p[0]) * t).astype(int), 0, W - 1)
    ys = np.clip(np.round(p[1] + (q[1] - p[1]) * t).astype(int), 0, H - 1)
    return float(mask[ys, xs].mean())


def detect_door_arcs(thin: np.ndarray, wall_t: float, line_w: float, ink: Optional[np.ndarray] = None,
                     min_r: Optional[float] = None, max_r: Optional[float] = None) -> List[DoorArc]:
    min_r = min_r or wall_t * 2.0
    max_r = max_r or wall_t * 9.0
    # remove long straight horizontal / vertical strokes so arcs separate from
    # the walls, leaf lines and furniture edges they touch
    k = max(5, int(round(wall_t * 2.5)))
    straight = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((1, k), np.uint8)) | \
        cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((k, 1), np.uint8))
    curved = (thin & (1 - straight)).astype(np.uint8)
    r = max(1, int(round(line_w)))
    near = cv2.dilate(thin, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    # faint / anti-aliased arcs break into fragments: bridge small breaks
    kb = max(3, int(round(line_w * 1.5)) | 1)
    joined = cv2.morphologyEx(curved, cv2.MORPH_CLOSE, np.ones((kb, kb), np.uint8))

    any_ink = cv2.dilate(ink if ink is not None else thin, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    H, W = near.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    arcs: List[DoorArc] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min(w, h) < min_r * 0.6 or max(w, h) > max_r * 1.1:
            continue
        if not (0.55 <= w / h <= 1.8):
            continue
        ys, xs = np.nonzero((lab[y:y + h, x:x + w] == i) & (curved[y:y + h, x:x + w] > 0))
        if len(xs) < 10:
            continue
        pts = np.stack([xs + x + 0.5, ys + y + 0.5], axis=1)
        # A door arc is a quarter circle: its hinge is one corner of the
        # arc's bounding box. Try all four corners, keep the best fit.
        best = None
        for sx in (0, 1):
            for sy in (0, 1):
                cx, cy = x + sx * w, y + sy * h
                d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
                rad = float(np.median(d))
                if not (min_r <= rad <= max_r):
                    continue
                err = float(np.mean(np.abs(d - rad)) / rad)
                # the quarter opens away from the hinge corner
                dx, dy = (1 if sx == 0 else -1), (1 if sy == 0 else -1)
                tt = np.linspace(0.08, math.pi / 2 - 0.08, 30)
                px = np.clip(np.round(cx + dx * rad * np.cos(tt)).astype(int), 0, W - 1)
                py = np.clip(np.round(cy + dy * rad * np.sin(tt)).astype(int), 0, H - 1)
                cover = float(near[py, px].mean())
                score = cover - 2 * err
                if best is None or score > best[0]:
                    best = (score, cx, cy, rad, err, cover, dx, dy)
        if best is None:
            continue
        _, cx, cy, rad, err, cover, dx, dy = best
        if err > 0.12 or cover < 0.65:
            continue
        hinge = (float(cx), float(cy))
        eh = (cx + dx * rad, cy)          # arc end along the horizontal axis
        ev = (cx, cy + dy * rad)          # arc end along the vertical axis
        lh, lv = _line_ink(near, hinge, eh), _line_ink(near, hinge, ev)
        if abs(lh - lv) < 0.3:
            # no clear leaf line: the latch end is the one that lands on
            # existing structure (the other door jamb)
            bh = _beyond(any_ink, hinge, eh, rad)
            bv = _beyond(any_ink, hinge, ev, rad)
            lh, lv = lh + 0.5 * bv, lv + 0.5 * bh
        if lh >= lv:
            tip, latch = eh, ev
            leaf = lh
        else:
            tip, latch = ev, eh
            leaf = lv
        o = "h" if abs(latch[0] - hinge[0]) > abs(latch[1] - hinge[1]) else "v"
        score = min(1.0, 0.7 * cover + 0.3 * min(1.0, leaf) + 0.1)
        arcs.append(DoorArc(hinge, latch, tip, rad, o, score))
    return arcs


def _beyond(mask: np.ndarray, hinge, end, rad: float) -> float:
    """Ink just past ``end`` (continuing away from the hinge)."""
    H, W = mask.shape
    ux, uy = (end[0] - hinge[0]) / rad, (end[1] - hinge[1]) / rad
    t = np.linspace(1.05, 1.3, 8) * rad
    xs = np.clip(np.round(hinge[0] + ux * t).astype(int), 0, W - 1)
    ys = np.clip(np.round(hinge[1] + uy * t).astype(int), 0, H - 1)
    return float(mask[ys, xs].mean())


def soft_thin_mask(gray: np.ndarray, solid: np.ndarray, line_w: float, dilate: bool = True) -> np.ndarray:
    """Thin-line mask with a lenient threshold: door arcs and window glazing
    are often drawn in light grey and break apart under the ink threshold."""
    bg = float(np.percentile(gray, 90))
    fg = float(np.percentile(gray, 1))
    th = bg - 0.18 * (bg - fg)
    m = ((gray < th) & (cv2.dilate(solid, np.ones((3, 3), np.uint8)) == 0)).astype(np.uint8)
    if not dilate:
        return m
    r = max(1, int(round(line_w)))
    return cv2.dilate(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8))


def detect_doors_at_wall_ends(segs, near: np.ndarray, ink: np.ndarray, wall_t: float,
                              min_r: Optional[float] = None, max_r: Optional[float] = None,
                              min_cover: float = 0.72, min_contrast: float = 0.5) -> List[DoorArc]:
    """Search for quarter-circle swing arcs hinged at wall ends.

    For every wall end and every way a door could hang there (closing along
    the wall axis or across it, swinging to either side) the radius is
    scanned and the arc coverage measured on ``near`` (dilated thin ink).
    """
    min_r = min_r or wall_t * 2.2
    max_r = max_r or wall_t * 8.0
    H, W = near.shape
    tt = np.linspace(0.1, math.pi / 2 - 0.1, 32)
    cos_t, sin_t = np.cos(tt), np.sin(tt)
    radii = np.arange(min_r, max_r, max(1.0, wall_t * 0.12))
    cands: List[DoorArc] = []

    def cover(hx, hy, ux, uy, vx, vy, r):
        # arc from latch direction u (t=0) to tip direction v (t=pi/2)
        px = hx + r * (ux * cos_t + vx * sin_t)
        py = hy + r * (uy * cos_t + vy * sin_t)
        xi, yi = np.round(px).astype(int), np.round(py).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        if ok.mean() < 0.9:
            return 0.0
        return float(near[yi[ok], xi[ok]].mean())

    for s in segs:
        for end, out in ((s.raw_a, -1), (s.raw_b, 1)):
            # end point on the centreline, axis unit vector pointing out of the wall
            if s.o == "h":
                ex, ey, ax, ay, nx, ny = end, s.c, out, 0, 0, 1
            else:
                ex, ey, ax, ay, nx, ny = s.c, end, 0, out, 1, 0
            # each config: (hinge_fn(r) -> (hx, hy), latch dir u, tip dir v, door orientation)
            configs = []
            for side in (1, -1):
                fx, fy = nx * side * s.t / 2, ny * side * s.t / 2
                # door continues the wall, hinged at this end
                configs.append((lambda r, fx=fx, fy=fy: (ex + fx, ey + fy), (ax, ay), (nx * side, ny * side), s.o))
                # door continues the wall, latching at this end (hinge one radius further on)
                configs.append((lambda r, fx=fx, fy=fy: (ex + ax * r + fx, ey + ay * r + fy), (-ax, -ay),
                                (nx * side, ny * side), s.o))
                # door across the wall end (between two wall stubs)
                for lat in (1, -1):
                    hx2, hy2 = ex - ax * s.t / 2 + nx * lat * s.t / 2, ey - ay * s.t / 2 + ny * lat * s.t / 2
                    configs.append((lambda r, hx2=hx2, hy2=hy2: (hx2, hy2), (nx * lat, ny * lat),
                                    (ax * side, ay * side), "v" if s.o == "h" else "h"))
            for hinge_fn, (ux, uy), (vx, vy), o in configs:
                best_r, best_c, best_s = None, 0.0, -1.0
                for r in radii:
                    hx, hy = hinge_fn(r)
                    c = cover(hx, hy, ux, uy, vx, vy, r)
                    if c < min_cover:
                        continue
                    ring = max(cover(hx, hy, ux, uy, vx, vy, r * 0.78),
                               0.5 * cover(hx, hy, ux, uy, vx, vy, r * 1.22))
                    sc = c - ring
                    if sc > best_s + 1e-9:
                        best_r, best_c, best_s = r, c, sc
                if best_r is None or best_s < min_contrast:
                    continue
                r = float(best_r)
                hx, hy = hinge_fn(r)
                latch = (hx + ux * r, hy + uy * r)
                tip = (hx + vx * r, hy + vy * r)
                cands.append(DoorArc((hx, hy), latch, tip, r, o, float(best_s)))
    # non-maximum suppression: one door per arc
    cands.sort(key=lambda a: -a.score)
    out_arcs: List[DoorArc] = []
    for a in cands:
        if not any(same_arc(a, b) for b in out_arcs):
            out_arcs.append(a)
    return out_arcs


QUADS = [((1, 0), (0, 1)), ((1, 0), (0, -1)), ((-1, 0), (0, 1)), ((-1, 0), (0, -1))]


def _arc_cover(near, hx, hy, ux, uy, vx, vy, r, n=32):
    H, W = near.shape
    tt = np.linspace(0.1, math.pi / 2 - 0.1, n)
    px = hx + r * (ux * np.cos(tt) + vx * np.sin(tt))
    py = hy + r * (uy * np.cos(tt) + vy * np.sin(tt))
    xi, yi = np.round(px).astype(int), np.round(py).astype(int)
    ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if ok.mean() < 0.9:
        return 0.0
    return float(near[yi[ok], xi[ok]].mean())


def refine_arc(hinge, r0: float, near: np.ndarray, wall_t: float, span_frac: float = 0.35, quads=None):
    """Local search around a rough arc estimate for the best quarter circle.
    Returns (score, hx, hy, (ux,uy), (vx,vy), r) or None."""
    step = max(1.0, wall_t * 0.25)
    span = r0 * span_frac
    offs = np.arange(-span, span + 1e-6, step) if span >= step else np.array([0.0])
    lo, hi = (0.75, 1.4) if span_frac > 0.2 else (0.92, 1.08)
    radii = np.arange(r0 * lo, r0 * hi, max(1.0, wall_t * 0.15))
    best = None
    for dx in offs:
        for dy in offs:
            hx, hy = hinge[0] + dx, hinge[1] + dy
            for (ux, uy), (vx, vy) in (quads or QUADS):
                for r in radii:
                    c = _arc_cover(near, hx, hy, ux, uy, vx, vy, r, 24)
                    if c < 0.7:
                        continue
                    ring = max(_arc_cover(near, hx, hy, ux, uy, vx, vy, r * 0.78, 24),
                               0.5 * _arc_cover(near, hx, hy, ux, uy, vx, vy, r * 1.22, 24))
                    sc = c - ring
                    if best is None or sc > best[0]:
                        best = (sc, hx, hy, (ux, uy), (vx, vy), r)
    return best


def orient_arc(hx, hy, u, v, r, near: np.ndarray, barrier: np.ndarray, wall_t: float):
    """Decide which arc end is the latch (on the wall line) and which the
    open-leaf tip. Uses the leaf line and which axis has wall behind the hinge."""
    e1 = (hx + u[0] * r, hy + u[1] * r)
    e2 = (hx + v[0] * r, hy + v[1] * r)
    H, W = barrier.shape

    def wall_behind(d):
        # wall ink on the far side of the hinge, along the closed-door axis
        pts = [(hx - d[0] * wall_t * k, hy - d[1] * wall_t * k) for k in (0.6, 1.0, 1.5)]
        vals = []
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            vals.append(barrier[yi, xi] if 0 <= xi < W and 0 <= yi < H else 0)
        return float(np.mean(vals))

    s1 = _line_ink(near, (hx, hy), e2) + wall_behind(u) * 0.8   # e1 latch => leaf along e2
    s2 = _line_ink(near, (hx, hy), e1) + wall_behind(v) * 0.8   # e2 latch => leaf along e1
    if s1 >= s2:
        latch, tip, d = e1, e2, u
    else:
        latch, tip, d = e2, e1, v
    o = "h" if d[1] == 0 else "v"
    return latch, tip, o, max(s1, s2)


def find_doors(thin: np.ndarray, gray: np.ndarray, solid: np.ndarray, barrier: np.ndarray,
               ink: np.ndarray, segs, wall_t: float, line_w: float) -> List[DoorArc]:
    """Combined door detector: arc-shape seeds + wall-end seeds, each refined
    by local search, then de-duplicated."""
    near = soft_thin_mask(gray, solid, line_w)
    # circle-fit seeds are rough: search widely. Wall-end seeds are already
    # precise: only polish them, so they cannot drift onto a neighbouring arc.
    seeds = [(a.hinge, a.r, 0.35, None) for a in detect_door_arcs(thin, wall_t, line_w, None)]
    for a in detect_doors_at_wall_ends(segs, near, ink, wall_t):
        u = (int(np.sign(round(a.latch[0] - a.hinge[0]))), int(np.sign(round(a.latch[1] - a.hinge[1]))))
        v = (int(np.sign(round(a.tip[0] - a.hinge[0]))), int(np.sign(round(a.tip[1] - a.hinge[1]))))
        seeds.append((a.hinge, a.r, 0.08, [(u, v), (v, u)]))
    cands: List[DoorArc] = []
    for hinge, r, span, quads in seeds:
        b = refine_arc(hinge, r, near, wall_t, span, quads)
        if b is None or b[0] < 0.45:
            continue
        sc, hx, hy, u, v, rr = b
        latch, tip, o, leafish = orient_arc(hx, hy, u, v, rr, near, barrier, wall_t)
        # a door hangs between two jambs: wall ink near the hinge and the latch
        # one jamb on a detected wall, the other at least on some ink (a short
        # wall stub may be too small to have become a wall)
        wb = _near_wall(barrier, (hx, hy), wall_t * 1.3) + _near_wall(barrier, latch, wall_t * 1.3)
        wi = _near_wall(ink, (hx, hy), wall_t) + _near_wall(ink, latch, wall_t)
        jamb = 2.0 if (wb >= 1 and wi + wb >= 3) or wb == 2 else wb
        # furniture (toilets, chairs) has ink inside the swept quarter
        clutter = _sector_ink(ink, hx, hy, u, v, rr)
        score = sc + 0.15 * jamb - 1.5 * clutter
        # the closed leaf lies in a wall line: wall ink continues behind the
        # hinge or past the latch along the door's axis (rejects furniture)
        if not _in_wall_line(barrier, (hx, hy), latch, rr, wall_t):
            continue
        if score < 0.5:
            continue
        cands.append(DoorArc((hx, hy), latch, tip, rr, o, float(min(1.0, score)), int(jamb)))
    cands.sort(key=lambda a: -a.score)
    out: List[DoorArc] = []
    for a in cands:
        if not any(same_arc(a, b) for b in out):
            out.append(a)
    return out


def _near_wall(barrier: np.ndarray, p, wall_t: float) -> float:
    H, W = barrier.shape
    k = int(round(wall_t * 0.8))
    x, y = int(round(p[0])), int(round(p[1]))
    win = barrier[max(0, y - k):min(H, y + k + 1), max(0, x - k):min(W, x + k + 1)]
    return 1.0 if win.size and win.any() else 0.0


def _sector_ink(ink: np.ndarray, hx, hy, u, v, r) -> float:
    """Fraction of inked pixels inside the swept quarter disc (excluding the rim)."""
    H, W = ink.shape
    rr = np.linspace(0.2, 0.8, 7) * r
    tt = np.linspace(0.15, math.pi / 2 - 0.15, 9)
    R, Tt = np.meshgrid(rr, tt)
    px = hx + R * (u[0] * np.cos(Tt) + v[0] * np.sin(Tt))
    py = hy + R * (u[1] * np.cos(Tt) + v[1] * np.sin(Tt))
    xi = np.clip(np.round(px).astype(int), 0, W - 1)
    yi = np.clip(np.round(py).astype(int), 0, H - 1)
    return float(ink[yi, xi].mean())


def same_arc(a: DoorArc, b: DoorArc, tol: float = 0.3) -> bool:
    """Two detections describe the same drawn arc (latch/tip may be swapped)."""
    t = max(a.r, b.r) * tol
    if math.dist(a.hinge, b.hinge) > t:
        return False
    ends_a, ends_b = (a.latch, a.tip), (b.latch, b.tip)
    return (math.dist(ends_a[0], ends_b[0]) < t and math.dist(ends_a[1], ends_b[1]) < t) or \
           (math.dist(ends_a[0], ends_b[1]) < t and math.dist(ends_a[1], ends_b[0]) < t)


def _in_wall_line(barrier: np.ndarray, hinge, latch, r: float, wall_t: float) -> bool:
    H, W = barrier.shape
    ux, uy = (latch[0] - hinge[0]) / r, (latch[1] - hinge[1]) / r
    pts = [(hinge[0] - ux * wall_t * k, hinge[1] - uy * wall_t * k) for k in (0.4, 0.8, 1.2)]
    pts += [(latch[0] + ux * wall_t * k, latch[1] + uy * wall_t * k) for k in (0.4, 0.8, 1.2)]
    k = max(1, int(round(wall_t * 0.35)))
    for x, y in pts:
        xi, yi = int(round(x)), int(round(y))
        if barrier[max(0, yi - k):min(H, yi + k + 1), max(0, xi - k):min(W, xi + k + 1)].any():
            return True
    return False
