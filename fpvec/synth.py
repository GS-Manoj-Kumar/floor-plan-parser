"""Synthetic floor plan generator with known ground truth (for tests/benchmarks).

Draws a plan the way sales/CAD floor plans usually look: solid black outer
walls, interior walls either solid or hollow (two thin lines), swing doors
(leaf + quarter arc) and windows (3 thin lines across a wall gap). Units are
millimetres, rendered at ``px_per_mm``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _wall_rect(img, a, b, t, color=0, hollow=False, lw=1, fill=None):
    (x1, y1), (x2, y2) = a, b
    if y1 == y2:
        p, q = (min(x1, x2), y1 - t / 2), (max(x1, x2), y1 + t / 2)
    else:
        p, q = (x1 - t / 2, min(y1, y2)), (x1 + t / 2, max(y1, y2))
    p = (int(round(p[0])), int(round(p[1])))
    q = (int(round(q[0])), int(round(q[1])))
    if hollow:
        if fill is not None:
            cv2.rectangle(img, p, q, fill, -1)
        cv2.rectangle(img, p, q, color, lw)
    else:
        cv2.rectangle(img, p, q, color, -1)


def make_plan(px_per_mm: float = 0.1, seed: int = 0, hollow_interior: bool = True, noise: bool = False,
              grey_interior: bool = False) -> Tuple[np.ndarray, Dict]:
    """Return (BGR image, ground truth dict in px).

    Layout (mm): 10 000 x 8 000 apartment, 240 mm exterior walls, 120 mm
    interior walls, two bedrooms, a bathroom and a living room.
    """
    rng = np.random.default_rng(seed)
    S = px_per_mm
    margin = 1500
    W, H = int((10000 + 2 * margin) * S), int((8000 + 2 * margin) * S)
    img = np.full((H, W), 255, np.uint8)
    P = lambda x, y: ((x + margin) * S, (y + margin) * S)
    Te, Ti = 240 * S, 120 * S
    lw = max(1, int(round(10 * S)))

    walls: List[dict] = []
    openings: List[dict] = []

    def wall(x1, y1, x2, y2, t, hollow=False, ops=()):
        """Wall from (x1,y1) to (x2,y2) mm with openings [(kind, start_mm, width_mm, extra)]."""
        horiz = y1 == y2
        L = abs(x2 - x1) if horiz else abs(y2 - y1)
        walls.append(dict(start=P(x1, y1), end=P(x2, y2), t=t, o="h" if horiz else "v"))
        cuts = sorted(ops, key=lambda o: o[1])
        pos = 0
        for kind, s0, w, extra in cuts + [(None, L, 0, None)]:
            if s0 > pos:
                a = (x1 + pos, y1) if horiz else (x1, y1 + pos)
                b = (x1 + s0, y1) if horiz else (x1, y1 + s0)
                if hollow and grey_interior:
                    # sales-plan style partitions: mid-grey outline, light-grey fill
                    _wall_rect(img, P(*a), P(*b), t, color=175, hollow=True, lw=lw, fill=222)
                else:
                    _wall_rect(img, P(*a), P(*b), t, hollow=hollow, lw=lw)
            if kind is None:
                break
            a = (x1 + s0, y1) if horiz else (x1, y1 + s0)
            b = (x1 + s0 + w, y1) if horiz else (x1, y1 + s0 + w)
            pa, pb = P(*a), P(*b)
            if kind == "window":
                for f in (-0.5, -0.17, 0.17, 0.5):
                    off = f * t
                    if horiz:
                        cv2.line(img, (int(pa[0]), int(pa[1] + off)), (int(pb[0]), int(pb[1] + off)), 0, lw)
                    else:
                        cv2.line(img, (int(pa[0] + off), int(pa[1])), (int(pb[0] + off), int(pb[1])), 0, lw)
                openings.append(dict(kind="window", start=pa, end=pb, o="h" if horiz else "v"))
            elif kind == "door":
                hinge_at_start, side = extra
                hinge = pa if hinge_at_start else pb
                latch = pb if hinge_at_start else pa
                r = w * S
                # hinge sits on the wall face of the swing side
                if horiz:
                    hf = (hinge[0], hinge[1] + side * t / 2)
                    tip = (hf[0], hf[1] + side * r)
                    lf = (latch[0], latch[1] + side * t / 2)
                else:
                    hf = (hinge[0] + side * t / 2, hinge[1])
                    tip = (hf[0] + side * r, hf[1])
                    lf = (latch[0] + side * t / 2, latch[1])
                cv2.line(img, tuple(map(int, hf)), tuple(map(int, tip)), 0, lw + 1)
                a0 = math.degrees(math.atan2(lf[1] - hf[1], lf[0] - hf[0]))
                a1 = math.degrees(math.atan2(tip[1] - hf[1], tip[0] - hf[0]))
                lo, hi = sorted((a0, a1))
                if hi - lo > 180:
                    lo, hi = hi, lo + 360
                cv2.ellipse(img, tuple(map(int, hf)), (int(r), int(r)), 0, lo, hi, 0, lw)
                openings.append(dict(kind="door", start=pa, end=pb, o="h" if horiz else "v",
                                     hinge=hinge, swing=side))
            pos = s0 + w

    # exterior (solid)
    wall(0, 0, 10000, 0, Te, ops=[("window", 1200, 1800, None), ("window", 6500, 2400, None)])
    wall(10000, 0, 10000, 8000, Te, ops=[("window", 4500, 2000, None)])
    wall(10000, 8000, 0, 8000, Te) if False else wall(0, 8000, 10000, 8000, Te,
                                                       ops=[("door", 7000, 1000, (True, -1)), ("window", 1500, 1500, None)])
    wall(0, 0, 0, 8000, Te, ops=[("window", 1500, 1500, None)])
    # interior
    hi = hollow_interior
    wall(4000, 0, 4000, 4000, Ti, hollow=hi, ops=[("door", 3000, 800, (False, 1))])
    wall(0, 4000, 6000, 4000, Ti, hollow=hi, ops=[("door", 4400, 900, (True, -1))])
    wall(6000, 4000, 6000, 8000, Ti, hollow=hi, ops=[("door", 300, 900, (True, -1))])
    wall(0, 6000, 2500, 6000, Ti, hollow=hi, ops=[("door", 1500, 700, (False, -1))])

    # furniture clutter: a bed, a table with chairs, text
    bx, by = P(800, 800)
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 1600 * S), int(by + 2000 * S)), 0, lw)
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 1600 * S), int(by + 300 * S)), 0, lw)
    tx, ty = P(7500, 1500)
    cv2.circle(img, (int(tx), int(ty)), int(500 * S), 0, lw)
    cv2.putText(img, "LIVING", tuple(map(int, P(7000, 6000))), cv2.FONT_HERSHEY_SIMPLEX, 60 * S / 10, 0, lw)
    if noise:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    gt = dict(walls=walls, openings=openings, mm_per_px=1 / S, ext_t=Te, int_t=Ti)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), gt
