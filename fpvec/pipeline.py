"""End-to-end raster floor plan -> vector walls / doors / windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .models import FloorPlan, Opening, Wall
from .doors import DoorArc, find_doors, soft_thin_mask
from .openings import Gap, classify_gap, find_gaps
from .preprocess import ImageLike, binarize, estimate_stroke_widths, load_gray
from .thinlines import detect_line_bands, keep_connected
from .walls import Seg, extract_segments, merge_collinear, snap_junctions, wall_mask


@dataclass
class Params:
    """Tuning knobs. Lengths ending in ``_t`` are multiples of wall thickness."""
    target_size: int = 1600            # long side of the working image (px)
    wall_thickness_px: Optional[float] = None   # override auto estimate (original px)
    ink_threshold: Optional[int] = None
    detect_thin_walls: bool = True     # hollow (double-line) walls + window bands
    min_opening_t: float = 2.0         # smallest door/window, in wall thicknesses
    max_opening_t: float = 26.0        # widest door/window
    max_plain_opening_t: float = 8.0   # widest gap kept when nothing is drawn in it
    door_score: float = 0.6
    merge_across_openings: bool = True  # one wall object per wall run, openings hosted on it
    max_opening_mm: Optional[float] = 4500.0  # used instead of max_opening_t when scale is known
    min_opening_mm: Optional[float] = 450.0


def _remove_border_frame(ink: np.ndarray) -> None:
    """Erase frame lines along the image border (screenshots, scans)."""
    H, W = ink.shape
    mh, mw = max(2, int(H * 0.02)), max(2, int(W * 0.02))
    for r in list(range(mh)) + list(range(H - mh, H)):
        if ink[r].mean() > 0.6:
            ink[r] = 0
    for c in list(range(mw)) + list(range(W - mw, W)):
        if ink[:, c].mean() > 0.6:
            ink[:, c] = 0


def _raster(segs: List[Seg], shape, extra: float = 0.0) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for s in segs:
        h = s.t / 2 + extra
        if s.o == "h":
            p1, p2 = (s.a, s.c - h), (s.b, s.c + h)
        else:
            p1, p2 = (s.c - h, s.a), (s.c + h, s.b)
        cv2.rectangle(m, (int(round(p1[0])), int(round(p1[1]))), (int(round(p2[0])) - 1, int(round(p2[1]))), 1, -1)
    return m


def _footprint(barrier: np.ndarray, wall_t: float) -> np.ndarray:
    k = int(round(wall_t * 3)) | 1
    closed = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    H, W = closed.shape
    flood = np.pad(closed, 1).copy()
    mask = np.zeros((H + 4, W + 4), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 2)
    return (flood[1:-1, 1:-1] != 2).astype(np.uint8)


def _outside_fraction(s, footprint: np.ndarray, wall_t: float) -> float:
    H, W = footprint.shape
    best = 0.0
    for side in (-1, 1):
        off = s.c + side * (s.t / 2 + wall_t * 1.0)
        along = np.linspace(s.a + 0.15 * (s.b - s.a), s.b - 0.15 * (s.b - s.a), 9)
        if s.o == "h":
            xs, ys = along, np.full_like(along, off)
        else:
            xs, ys = np.full_like(along, off), along
        outside = []
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            outside.append(not (0 <= xi < W and 0 <= yi < H) or footprint[yi, xi] == 0)
        best = max(best, float(np.mean(outside)))
    return best


def vectorize(image: ImageLike, mm_per_px: Optional[float] = None, params: Optional[Params] = None,
              debug: Optional[dict] = None) -> FloorPlan:
    """Vectorize a floor plan image.

    Parameters
    ----------
    image: path or numpy array (BGR / gray / BGRA).
    mm_per_px: real-world scale of the *original* image. If given, all
        results also come out in millimetres and opening size limits use mm.
    params: tuning knobs (see :class:`Params`).
    debug: optional dict that receives intermediate masks for inspection.
    """
    p = params or Params()
    gray0 = load_gray(image)
    H0, W0 = gray0.shape
    s = p.target_size / max(H0, W0)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    gray = cv2.resize(gray0, None, fx=s, fy=s, interpolation=interp) if abs(s - 1) > 1e-3 else gray0

    ink = binarize(gray, p.ink_threshold)
    _remove_border_frame(ink)
    line_w, T = estimate_stroke_widths(ink)
    if p.wall_thickness_px:
        T = p.wall_thickness_px * s

    # ---------------------------------------------------------------- walls
    solid = wall_mask(ink, T, fill_hollow=False, line_w=line_w)
    segs = extract_segments(solid, T)
    hollow: List[Seg] = []
    hollow_c: List[Seg] = []
    win_bands: List[Seg] = []
    if p.detect_thin_walls:
        hollow_c, win_bands = detect_line_bands(ink, solid, T, line_w)
        conn = keep_connected(hollow_c, segs, T)
        hollow = conn + [h for h in hollow_c if h not in conn and h.length >= 12 * T]
    walls = merge_collinear(segs + hollow, T)
    barrier = (solid | _raster(hollow, solid.shape)).astype(np.uint8)

    # ------------------------------------------------------------- openings
    px_mm = (mm_per_px / s) if mm_per_px else None     # mm per working px
    min_w = p.min_opening_t * T
    max_w = p.max_opening_t * T
    if px_mm:
        if p.min_opening_mm:
            min_w = p.min_opening_mm / px_mm
        if p.max_opening_mm:
            max_w = p.max_opening_mm / px_mm
    gaps = find_gaps(walls, barrier, T, min_w, max_w)

    thin = (ink & (1 - cv2.dilate(solid, np.ones((3, 3), np.uint8)))).astype(np.uint8)
    r = max(1, int(round(line_w)))
    thin_near = cv2.dilate(thin, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    soft_thin = soft_thin_mask(gray, solid, line_w, dilate=False)
    arcs = [a for a in find_doors(thin, gray, solid, barrier, ink, walls + hollow_c, T, line_w) if a.score >= p.door_score]
    used_arcs = set()
    kept: List[Gap] = []
    for g in gaps:
        arc_i = _match_arc(g, arcs, T)
        if arc_i is not None:
            _apply_arc(g, arcs[arc_i])
            used_arcs.add(arc_i)
            kept.append(g)
            continue
        classify_gap(g, thin_near, soft_thin, T, line_w)
        if g.kind == "door":          # doors come from the arc detector only
            g.kind, g.score = "opening", 0.5
        if g.kind == "opening" and g.width > p.max_plain_opening_t * T:
            continue
        kept.append(g)
    # arcs whose wall gap was not found (e.g. a door between two wall stubs)
    for i, arc in enumerate(arcs):
        if i in used_arcs or arc.score < p.door_score or arc.jambs < 2:
            continue
        g = _gap_from_arc(arc, walls, T)
        if any(q.o == g.o and abs(q.c - g.c) < T * 1.5 and g.a < q.b and g.b > q.a for q in kept):
            continue
        kept.append(g)

    # window bands that are not inside a found gap (e.g. a whole wall stretch
    # drawn as a window) become windows on their own if they touch the walls
    wb = keep_connected(win_bands, walls, T)
    for w in wb:
        if any(g.o == w.o and abs(g.c - w.c) < T and w.a < g.b and w.b > g.a for g in kept):
            continue
        kept.append(Gap(w.o, w.c, w.a, w.b, w.t, kind="window", score=0.6))

    walls, kept = _main_clusters(walls, kept, T)

    # ----------------------------------------------- tidy wall graph + exterior
    snap_junctions(walls, T)
    full_barrier = (barrier | _raster([Seg(g.o, g.c, g.a, g.b, g.t) for g in kept], barrier.shape)).astype(np.uint8)
    footprint = _footprint(full_barrier, T)
    if debug is not None:
        debug.update(dict(gray=gray, ink=ink, solid=solid, barrier=barrier, thin=thin,
                          footprint=footprint, scale=s, wall_t=T, line_w=line_w))

    if p.merge_across_openings:
        walls = _merge_across(walls, kept, T)

    # --------------------------------------------------------------- output
    inv = 1.0 / s
    out_walls: List[Wall] = []
    for i, w in enumerate(walls):
        if w.length < T * 1.0:
            continue
        ext = _outside_fraction(w, footprint, T) >= 0.5
        if w.o == "h":
            a, b = (w.a, w.c), (w.b, w.c)
        else:
            a, b = (w.c, w.a), (w.c, w.b)
        out_walls.append(Wall(len(out_walls), (a[0] * inv, a[1] * inv), (b[0] * inv, b[1] * inv),
                              w.t * inv, w.o, ext))
        w._out_id = out_walls[-1].id  # type: ignore[attr-defined]

    out_open: List[Opening] = []
    for g in sorted(kept, key=lambda g: (g.o, g.c, g.a)):
        if g.o == "h":
            a, b = (g.a, g.c), (g.b, g.c)
        else:
            a, b = (g.c, g.a), (g.c, g.b)
        hosts = [getattr(w, "_out_id") for w in walls if hasattr(w, "_out_id") and w.o == g.o
                 and abs(w.c - g.c) < max(T, w.t) * 0.75 and w.a - T <= g.b and w.b + T >= g.a]
        out_open.append(Opening(
            id=len(out_open), kind=g.kind,
            start=(a[0] * inv, a[1] * inv), end=(b[0] * inv, b[1] * inv),
            thickness=g.t * inv, orientation=g.o, host_wall_ids=hosts, confidence=g.score,
            hinge=None if g.hinge is None else (g.hinge[0] * inv, g.hinge[1] * inv),
            swing_side=g.swing, double=g.double))

    return FloorPlan(image_size=(W0, H0), walls=out_walls, openings=out_open,
                     mm_per_px=mm_per_px, wall_thickness_px=T * inv, line_width_px=line_w * inv)


def _merge_across(walls: List[Seg], gaps: List[Gap], wall_t: float) -> List[Seg]:
    """Join collinear wall pieces separated by an opening into one wall run."""
    walls = [Seg(w.o, w.c, w.a, w.b, w.t, w.raw_a, w.raw_b) for w in walls]
    changed = True
    while changed:
        changed = False
        for g in gaps:
            same = [w for w in walls if w.o == g.o and abs(w.c - g.c) < max(w.t, wall_t) * 0.6]
            left = [w for w in same if abs(w.b - g.a) <= wall_t * 1.2]
            right = [w for w in same if abs(w.a - g.b) <= wall_t * 1.2]
            if left and right:
                l, r = left[0], right[0]
                if l is r:
                    continue
                L1, L2 = l.length, r.length
                m = Seg(g.o, (l.c * L1 + r.c * L2) / (L1 + L2), min(l.a, r.a), max(l.b, r.b),
                        (l.t * L1 + r.t * L2) / (L1 + L2))
                walls = [w for w in walls if w is not l and w is not r] + [m]
                changed = True
    return walls


def _match_arc(g: Gap, arcs: List[DoorArc], wall_t: float) -> Optional[int]:
    best, best_d = None, 1e9
    for i, a in enumerate(arcs):
        if a.o != g.o:
            continue
        along, across = (a.hinge[0], a.hinge[1]) if g.o == "h" else (a.hinge[1], a.hinge[0])
        if abs(across - g.c) > max(g.t, wall_t) * 1.2:
            continue
        d = min(abs(along - g.a), abs(along - g.b))
        if a.r > g.width * 1.35 or a.r < g.width * 0.3:
            continue
        if d <= max(wall_t * 1.2, g.width * 0.25) and d < best_d:
            # the latch must point into the gap
            lat = a.latch[0] if g.o == "h" else a.latch[1]
            if min(g.a, g.b) - wall_t <= lat <= max(g.a, g.b) + wall_t or g.a - wall_t <= (along + lat) / 2 <= g.b + wall_t:
                best, best_d = i, d
    return best


def _apply_arc(g: Gap, a: DoorArc) -> None:
    along = a.hinge[0] if g.o == "h" else a.hinge[1]
    hinge_along = g.a if abs(along - g.a) <= abs(along - g.b) else g.b
    if g.width > a.r * 1.3:
        # gap is wider than the door leaf (e.g. a short wall stub beside the
        # door was too small to detect): the door spans hinge -> leaf length
        if hinge_along == g.a:
            g.b = g.a + a.r
        else:
            g.a = g.b - a.r
    g.kind, g.score = "door", float(a.score)
    g.hinge = (hinge_along, g.c) if g.o == "h" else (g.c, hinge_along)
    tip_across = a.tip[1] - a.hinge[1] if g.o == "h" else a.tip[0] - a.hinge[0]
    if g.o == "h":
        g.swing = "down" if tip_across > 0 else "up"
    else:
        g.swing = "right" if tip_across > 0 else "left"


def _gap_from_arc(a: DoorArc, walls: List[Seg], wall_t: float) -> Gap:
    o = a.o
    h_al, h_ac = (a.hinge[0], a.hinge[1]) if o == "h" else (a.hinge[1], a.hinge[0])
    l_al = a.latch[0] if o == "h" else a.latch[1]
    # put the door on the centreline of the nearest parallel wall, if close
    c, t = h_ac, wall_t
    near = [w for w in walls if w.o == o and abs(w.c - h_ac) < wall_t * 1.5]
    if near:
        w = min(near, key=lambda w: abs(w.c - h_ac))
        c, t = w.c, w.t
    else:
        perp = [w for w in walls if w.o != o and w.a - wall_t <= h_ac <= w.b + wall_t
                and min(abs(w.c - h_al), abs(w.c - l_al)) < wall_t * 2]
        if perp:
            # door between wall stubs: hinge sits on the stub's face, move to its axis
            t = min(perp, key=lambda w: abs(w.c - h_al)).t
            c = h_ac - (t / 2 if a.tip[1 if o == "h" else 0] > a.hinge[1 if o == "h" else 0] else -t / 2)
    g = Gap(o, c, min(h_al, l_al), max(h_al, l_al), t, kind="door", score=a.score)
    g.hinge = (h_al, c) if o == "h" else (c, h_al)
    tip_across = (a.tip[1] - a.hinge[1]) if o == "h" else (a.tip[0] - a.hinge[0])
    g.swing = ("down" if tip_across > 0 else "up") if o == "h" else ("right" if tip_across > 0 else "left")
    return g


def _main_clusters(walls: List[Seg], gaps: List[Gap], wall_t: float, keep_frac: float = 0.15):
    """Drop small groups of walls/openings not connected to the main plan
    (legends, key plans, north arrows, title blocks). Openings count as
    connections, so a wall separated from the rest only by a window stays."""
    items = list(walls) + list(gaps)
    n = len(items)
    if not walls:
        return walls, gaps
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def box(s):
        h = s.t / 2
        return (s.a, s.c - h, s.b, s.c + h) if s.o == "h" else (s.c - h, s.a, s.c + h, s.b)

    tol = wall_t * 1.5
    boxes = [box(s) for s in items]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a[0] - tol <= b[2] and b[0] - tol <= a[2] and a[1] - tol <= b[3] and b[1] - tol <= a[3]:
                parent[find(i)] = find(j)
    total: dict = {}
    for i, s in enumerate(items):
        total[find(i)] = total.get(find(i), 0) + (s.b - s.a)
    best_root = max(total, key=total.get)
    best = total[best_root]
    mb = [min(boxes[i][0] for i in range(n) if find(i) == best_root) - 2 * wall_t,
          min(boxes[i][1] for i in range(n) if find(i) == best_root) - 2 * wall_t,
          max(boxes[i][2] for i in range(n) if find(i) == best_root) + 2 * wall_t,
          max(boxes[i][3] for i in range(n) if find(i) == best_root) + 2 * wall_t]
    # a cluster survives if it is big, or if it lies inside the main plan's
    # bounding box (an interior room whose link opening was missed)
    centre: dict = {}
    for i in range(n):
        r = find(i)
        b = boxes[i]
        cx, cy, wsum = centre.get(r, (0.0, 0.0, 0.0))
        L = items[i].b - items[i].a
        centre[r] = (cx + (b[0] + b[2]) / 2 * L, cy + (b[1] + b[3]) / 2 * L, wsum + L)
    def inside(r):
        cx, cy, wsum = centre[r]
        cx, cy = cx / max(wsum, 1e-9), cy / max(wsum, 1e-9)
        return mb[0] <= cx <= mb[2] and mb[1] <= cy <= mb[3]
    keep = [total[find(i)] >= keep_frac * best or inside(find(i)) for i in range(n)]
    nw = len(walls)
    return ([w for i, w in enumerate(walls) if keep[i]],
            [g for i, g in enumerate(gaps) if keep[nw + i]])
