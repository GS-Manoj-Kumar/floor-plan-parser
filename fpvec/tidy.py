"""Tidy a traced plan: make walls orthogonal and join wall ends.

    tidy(plan, ortho=True, snap=True, snap_dist=3, snap_unit="t")

* **ortho**: every wall becomes exactly horizontal or vertical, and walls
  whose centrelines are nearly on one line (within half a wall thickness)
  are put on exactly that line.
* **snap** (join wall ends): a wall end that does not meet another wall is
  extended to the nearest wall within ``snap_dist``:
    - a wall straight ahead on the same line (the two become one wall,
      closing the gap - e.g. a doorway when only walls are detected);
    - a perpendicular wall it runs into (T junction);
    - the end of a perpendicular wall nearby (both are extended to meet:
      an L corner).
  ``snap_unit``: "t" = multiples of the wall thickness, "pct" = percent of
  the plan's larger side, "mm" = millimetres (needs a scale; without one a
  200 mm wall is assumed).
  Overshooting ends (a wall poking past the wall it meets by up to one
  thickness) are trimmed back to it.

Wall thickness: ``match_thickness(plan, "100-250")`` clamps every wall into a
range, ``match_thickness(plan, "100, 150, 230")`` sets each to the nearest
listed value (mm).

Openings stay on their wall: after tidying they are re-hosted on the wall
they now sit on.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .models import FloorPlan, Opening, Wall, _hosts


@dataclass
class _S:
    o: str          # "h" | "v" | "d"
    c: float        # y for h, x for v
    a: float        # start along the axis (a < b)
    b: float
    t: float
    ext: bool
    raw: Optional[Wall] = None


def _to_seg(w: Wall, force: bool) -> _S:
    (x0, y0), (x1, y1) = w.start, w.end
    dx, dy = x1 - x0, y1 - y0
    if not force and abs(dx) > 1e-6 and abs(dy) > 1e-6 and min(abs(dx), abs(dy)) / max(abs(dx), abs(dy)) > 0.02:
        return _S("d", 0, 0, 0, w.thickness, w.exterior, w)
    if abs(dx) >= abs(dy):
        return _S("h", (y0 + y1) / 2, min(x0, x1), max(x0, x1), w.thickness, w.exterior)
    return _S("v", (x0 + x1) / 2, min(y0, y1), max(y0, y1), w.thickness, w.exterior)


def snap_distance(plan: FloorPlan, snap_dist: float, snap_unit: str) -> float:
    """The join distance in image px."""
    ts = [w.thickness for w in plan.walls] or [plan.wall_thickness_px or 10.0]
    t = float(np.median(ts))
    if snap_unit == "pct":
        xs = [p[0] for w in plan.walls for p in (w.start, w.end)]
        ys = [p[1] for w in plan.walls for p in (w.start, w.end)]
        size = max(max(xs) - min(xs), max(ys) - min(ys)) if xs else 0.0
        return snap_dist / 100.0 * size
    if snap_unit == "mm":
        k = plan.mm_per_px or (200.0 / t)
        return snap_dist / k
    return snap_dist * t


def _align(segs: List[_S], tol: float) -> None:
    for o in ("h", "v"):
        g = sorted([s for s in segs if s.o == o], key=lambda s: s.c)
        i = 0
        while i < len(g):
            j = i
            while j + 1 < len(g) and g[j + 1].c - g[j].c <= tol:
                j += 1
            if j > i:
                grp = g[i:j + 1]
                c = sum(s.c * (s.b - s.a + 1) for s in grp) / sum(s.b - s.a + 1 for s in grp)
                for s in grp:
                    s.c = c
            i = j + 1


def _connected(s: _S, end: str, segs: List[_S]) -> bool:
    e = s.a if end == "a" else s.b
    for p in segs:
        if p is s or p.o == "d":
            continue
        tol = max(s.t, p.t) * 0.6
        if p.o != s.o:
            if abs(p.c - e) <= tol and p.a - tol <= s.c <= p.b + tol:
                return True
        elif abs(p.c - s.c) <= tol and (abs(p.a - e) <= tol or abs(p.b - e) <= tol or p.a < e < p.b):
            return True
    return False


def _join(segs: List[_S], D: float) -> None:
    for _ in range(3):
        changed = False
        for s in segs:
            if s.o == "d":
                continue
            for end, sgn in (("a", -1), ("b", 1)):
                e = s.a if end == "a" else s.b
                best = None                      # (distance, kind, target, new end, target end)
                for p in segs:
                    if p is s or p.o == "d":
                        continue
                    if p.o != s.o:
                        d = (p.c - e) * sgn      # >0: ahead of this end
                        if p.a - p.t / 2 <= s.c <= p.b + p.t / 2:
                            # it runs into p (or pokes past it by up to one thickness)
                            if -max(s.t, p.t) <= d <= D and abs(d) > 1e-6:
                                cand = (abs(d), "t", p, p.c, None)
                                best = cand if best is None or cand[0] < best[0] else best
                        else:
                            # an L corner: p ends near this wall's line
                            for pe, psg in ((p.a, -1), (p.b, 1)):
                                dd = (s.c - pe) * psg
                                if 0 <= dd <= D and 0 <= d <= D and not _connected(p, "a" if psg < 0 else "b", segs):
                                    cand = (max(d, dd), "l", p, p.c, "a" if psg < 0 else "b")
                                    best = cand if best is None or cand[0] < best[0] else best
                    else:
                        if abs(p.c - s.c) > max(s.t, p.t) * 0.6:
                            continue
                        gap = (p.a - e) if sgn > 0 else (e - p.b)
                        if 0 < gap <= D:
                            cand = (gap, "c", p, p.a if sgn > 0 else p.b, None)
                            best = cand if best is None or cand[0] < best[0] else best
                if best is None:
                    continue
                if _connected(s, end, segs) and best[1] != "t":
                    continue
                if _connected(s, end, segs) and best[0] > max(s.t, best[2].t):
                    continue
                _, kind, p, ne, pend = best
                if kind == "c":
                    # one wall from the far end of s to the far end of p
                    s.a, s.b = min(s.a, p.a), max(s.b, p.b)
                    s.t = max(s.t, p.t) if abs(s.t - p.t) < 0.3 * max(s.t, p.t) else (s.t + p.t) / 2
                    s.ext = s.ext or p.ext
                    p.o = "x"                    # removed
                else:
                    if end == "a":
                        s.a = ne
                    else:
                        s.b = ne
                    if kind == "l":
                        if pend == "a":
                            p.a = s.c
                        else:
                            p.b = s.c
                changed = True
        segs[:] = [s for s in segs if s.o != "x"]
        if not changed:
            break


def _merge(segs: List[_S]) -> List[_S]:
    out: List[_S] = []
    for o in ("h", "v"):
        g = sorted([s for s in segs if s.o == o], key=lambda s: (s.c, s.a))
        for s in g:
            tol = max(1.0, 0.25 * s.t)            # pieces touching or almost touching on one line
            m = next((q for q in out if q.o == o and abs(q.c - s.c) < tol and s.a <= q.b + tol and s.b >= q.a - tol), None)
            if m:
                m.a, m.b, m.t, m.ext = min(m.a, s.a), max(m.b, s.b), max(m.t, s.t), m.ext or s.ext
            else:
                out.append(s)
    return out + [s for s in segs if s.o == "d"]


def tidy(plan: FloorPlan, ortho: bool = True, snap: bool = True, snap_dist: float = 3.0,
         snap_unit: str = "t") -> FloorPlan:
    if not plan.walls:
        return plan
    segs = [_to_seg(w, ortho) for w in plan.walls]
    t = float(np.median([s.t for s in segs]))
    if ortho:
        _align(segs, t * 0.5)
    if snap and snap_dist > 0:
        _join(segs, snap_distance(plan, snap_dist, snap_unit))
    segs = _merge(segs)
    walls: List[Wall] = []
    for s in segs:
        if s.o == "d":
            r = s.raw
            walls.append(Wall(len(walls), r.start, r.end, r.thickness, "d", r.exterior))
        elif s.b - s.a > 0.5:
            a, b = ((s.a, s.c), (s.b, s.c)) if s.o == "h" else ((s.c, s.a), (s.c, s.b))
            walls.append(Wall(len(walls), a, b, s.t, s.o, s.ext))
    ops: List[Opening] = []
    for o in plan.openings:
        o2 = Opening(len(ops), o.kind, o.start, o.end, o.thickness, o.orientation, [], o.confidence, o.hinge,
                     o.swing_side, o.double)
        if ortho:                                   # put the opening exactly on its (aligned) wall
            o2.host_wall_ids = _hosts(o2, walls)
            if o2.host_wall_ids:
                w = walls[o2.host_wall_ids[0]]
                if w.orientation == "h":
                    o2.start, o2.end = (o2.start[0], w.start[1]), (o2.end[0], w.start[1])
                elif w.orientation == "v":
                    o2.start, o2.end = (w.start[0], o2.start[1]), (w.start[0], o2.end[1])
        o2.host_wall_ids = _hosts(o2, walls)
        if o2.host_wall_ids:
            o2.thickness = walls[o2.host_wall_ids[0]].thickness
        ops.append(o2)
    return FloorPlan(plan.image_size, walls, ops, plan.mm_per_px, plan.wall_thickness_px, plan.line_width_px)


def loose_ends(plan: FloorPlan) -> int:
    """How many wall ends touch no other wall (diagonal walls are skipped)."""
    segs = [s for s in (_to_seg(w, False) for w in plan.walls) if s.o != "d"]
    return sum(not _connected(s, e, segs) for s in segs for e in ("a", "b"))


def parse_thicknesses(spec) -> Optional[Tuple[str, List[float]]]:
    """Allowed wall thicknesses (mm) from the user's input.

    "100-250" / "100 to 250" / "100..250" -> ("range", [100, 250]);
    "100, 150, 230" / "100 150 230" / [100, 150] -> ("list", [100, 150, 230]).
    Empty or unreadable -> None."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        vals = sorted(float(v) for v in spec if float(v) > 0)
        return ("list", vals) if vals else None
    s = str(spec).strip().lower().replace("mm", "")
    if not s:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:-|–|to|\.\.)\s*(\d+(?:\.\d+)?)\s*", s)
    if m:
        lo, hi = sorted((float(m.group(1)), float(m.group(2))))
        return ("range", [lo, hi]) if hi > 0 else None
    vals = sorted({float(v) for v in re.findall(r"\d+(?:\.\d+)?", s) if float(v) > 0})
    return ("list", vals) if vals else None


def _fit_mm(ts, mm: float) -> float:
    kind, vals = ts
    return min(max(mm, vals[0]), vals[1]) if kind == "range" else min(vals, key=lambda v: abs(v - mm))


def assumed_wall_mm(spec, wall_mm: float = 200.0) -> float:
    """The main wall thickness to assume when there is no scale: 200 mm, or
    the allowed value nearest to it."""
    ts = parse_thicknesses(spec)
    return _fit_mm(ts, wall_mm) if ts else wall_mm


def match_thickness(plan: FloorPlan, spec) -> FloorPlan:
    """Set every wall's thickness to the nearest allowed value: the nearest
    one in a list, or clamped into a range (mm; without a scale a 200 mm
    main wall is assumed). Openings take their wall's new thickness."""
    ts = parse_thicknesses(spec)
    if ts is None or not plan.walls:
        return plan
    t_main = plan.wall_thickness_px or float(np.median([w.thickness for w in plan.walls]))
    k = plan.mm_per_px or (assumed_wall_mm(spec) / t_main)

    def fit(t_px: float) -> float:
        return _fit_mm(ts, t_px * k) / k

    walls = [Wall(w.id, w.start, w.end, fit(w.thickness), w.orientation, w.exterior) for w in plan.walls]
    ops = []
    for o in plan.openings:
        t = walls[o.host_wall_ids[0]].thickness if o.host_wall_ids and o.host_wall_ids[0] < len(walls) else o.thickness
        ops.append(Opening(o.id, o.kind, o.start, o.end, t, o.orientation, list(o.host_wall_ids), o.confidence,
                           o.hinge, o.swing_side, o.double))
    return FloorPlan(plan.image_size, walls, ops, plan.mm_per_px, plan.wall_thickness_px, plan.line_width_px)
