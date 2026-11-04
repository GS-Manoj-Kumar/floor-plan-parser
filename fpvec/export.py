"""Write a FloorPlan as JSON, SVG, DXF or a debug overlay PNG."""
from __future__ import annotations

import json
import math
from typing import Optional

import cv2
import numpy as np

from .models import FloorPlan
from .preprocess import ImageLike, load_gray

# BGR colours for the overlay, mirroring a typical CAD look
COLORS = {"wall": (0, 0, 230), "door": (0, 200, 255), "window": (230, 120, 0), "opening": (0, 170, 0)}


def to_json(plan: FloorPlan, path: Optional[str] = None, indent: int = 2) -> str:
    s = json.dumps(plan.to_dict(), indent=indent, ensure_ascii=False)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    return s


def draw_overlay(plan: FloorPlan, image: ImageLike, path: Optional[str] = None, scale: float = 2.0) -> np.ndarray:
    """Draw detected walls and openings over a faded copy of the image."""
    g = load_gray(image)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    vis = cv2.cvtColor((g * 0.35 + 165).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    S = lambda p: (int(round(p[0] * scale)), int(round(p[1] * scale)))
    for w in plan.walls:
        th = max(2, int(round(w.thickness * scale * 0.35)))
        cv2.line(vis, S(w.start), S(w.end), (60, 60, 60) if w.exterior else COLORS["wall"], th)
        cv2.line(vis, S(w.start), S(w.end), COLORS["wall"], max(1, th // 2))
    for o in plan.openings:
        col = COLORS[o.kind]
        half = o.thickness * 0.6
        if o.orientation == "h":
            p1, p2 = (o.start[0], o.start[1] - half), (o.end[0], o.end[1] + half)
        else:
            p1, p2 = (o.start[0] - half, o.start[1]), (o.end[0] + half, o.end[1])
        cv2.rectangle(vis, S(p1), S(p2), col, -1)
        cv2.rectangle(vis, S(p1), S(p2), (0, 0, 0), 1)
        if o.kind == "door" and o.hinge is not None:
            r = o.width
            hx, hy = o.hinge
            sgn = 1 if o.swing_side in ("down", "right") else -1
            if o.orientation == "h":
                toward = 1 if abs(hx - o.start[0]) < abs(hx - o.end[0]) else -1
                tip = (hx, hy + sgn * r)
                a0 = 0 if toward > 0 else 180
                a1 = 90 * sgn if toward > 0 else 180 - 90 * sgn
            else:
                toward = 1 if abs(hy - o.start[1]) < abs(hy - o.end[1]) else -1
                tip = (hx + sgn * r, hy)
                a0 = 90 if toward > 0 else 270
                a1 = (0 if sgn > 0 else 180) if toward > 0 else (360 if sgn > 0 else 180)
            cv2.line(vis, S(o.hinge), S(tip), col, 2)
            cv2.ellipse(vis, S(o.hinge), (int(r * scale), int(r * scale)), 0, min(a0, a1), max(a0, a1), col, 2)
        cx, cy = S(o.center)
        cv2.putText(vis, f"{o.kind[0].upper()}{o.id}", (cx + 4, cy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    for w in plan.walls:
        mx, my = S(((w.start[0] + w.end[0]) / 2, (w.start[1] + w.end[1]) / 2))
        label = f"{w.length * plan.mm_per_px:.0f}" if plan.mm_per_px else f"{w.length:.0f}px"
        cv2.putText(vis, label, (mx + 3, my + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 0, 90), 1, cv2.LINE_AA)
    if path:
        cv2.imwrite(path, vis)
    return vis


def to_svg(plan: FloorPlan, path: Optional[str] = None, units: str = "px") -> str:
    """SVG with one <line> per wall centreline and one <rect> per opening.
    ``units="mm"`` writes millimetre coordinates (requires a scale)."""
    k = plan.mm_per_px if units == "mm" and plan.mm_per_px else 1.0
    W, H = plan.image_size[0] * k, plan.image_size[1] * k
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.1f} {H:.1f}" width="{W:.0f}" height="{H:.0f}">',
           '<g id="walls" stroke="#d00" stroke-linecap="square" fill="none">']
    for w in plan.walls:
        out.append(f'<line data-id="{w.id}" data-exterior="{str(w.exterior).lower()}" x1="{w.start[0]*k:.1f}" y1="{w.start[1]*k:.1f}" '
                   f'x2="{w.end[0]*k:.1f}" y2="{w.end[1]*k:.1f}" stroke-width="{w.thickness*k:.1f}"/>')
    out.append('</g><g id="openings">')
    fill = {"door": "#ffc800", "window": "#0078e6", "opening": "#00aa00"}
    for o in plan.openings:
        half = o.thickness * 0.6
        if o.orientation == "h":
            x, y, w, h = o.start[0], o.start[1] - half, o.width, 2 * half
        else:
            x, y, w, h = o.start[0] - half, o.start[1], 2 * half, o.width
        out.append(f'<rect data-id="{o.id}" data-type="{o.kind}" x="{x*k:.1f}" y="{y*k:.1f}" width="{w*k:.1f}" height="{h*k:.1f}" fill="{fill[o.kind]}"/>')
    out.append('</g></svg>')
    s = "\n".join(out)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    return s


def to_dxf(plan: FloorPlan, path: str) -> None:
    """Minimal ASCII DXF (R12) with layers WALLS, DOORS, WINDOWS, OPENINGS.

    Coordinates are millimetres when a scale is known, otherwise pixels. The
    y axis is flipped so the drawing is upright in CAD.
    """
    k = plan.mm_per_px or 1.0
    H = plan.image_size[1]
    P = lambda p: (p[0] * k, (H - p[1]) * k)
    ents = []

    def line(layer, a, b):
        ents.append(f"0\nLINE\n8\n{layer}\n10\n{a[0]:.2f}\n20\n{a[1]:.2f}\n30\n0\n11\n{b[0]:.2f}\n21\n{b[1]:.2f}\n31\n0")

    def arc(layer, c, r, a0, a1):
        ents.append(f"0\nARC\n8\n{layer}\n10\n{c[0]:.2f}\n20\n{c[1]:.2f}\n30\n0\n40\n{r:.2f}\n50\n{a0:.2f}\n51\n{a1:.2f}")

    for w in plan.walls:
        # two face lines + centreline-free ends: draw the wall as a rectangle
        dx, dy = w.end[0] - w.start[0], w.end[1] - w.start[1]
        L = math.hypot(dx, dy) or 1
        nx, ny = -dy / L * w.thickness / 2, dx / L * w.thickness / 2
        c = [(w.start[0] + nx, w.start[1] + ny), (w.end[0] + nx, w.end[1] + ny),
             (w.end[0] - nx, w.end[1] - ny), (w.start[0] - nx, w.start[1] - ny)]
        for i in range(4):
            line("WALLS", P(c[i]), P(c[(i + 1) % 4]))
    layer = {"door": "DOORS", "window": "WINDOWS", "opening": "OPENINGS"}
    for o in plan.openings:
        half = o.thickness / 2
        if o.orientation == "h":
            c = [(o.start[0], o.start[1] - half), (o.end[0], o.end[1] - half), (o.end[0], o.end[1] + half), (o.start[0], o.start[1] + half)]
        else:
            c = [(o.start[0] - half, o.start[1]), (o.end[0] - half, o.end[1]), (o.end[0] + half, o.end[1]), (o.start[0] + half, o.start[1])]
        for i in range(4):
            line(layer[o.kind], P(c[i]), P(c[(i + 1) % 4]))
        if o.kind == "window":
            line("WINDOWS", P(o.start), P(o.end))
        if o.kind == "door" and o.hinge is not None:
            hx, hy = o.hinge
            sgn = 1 if o.swing_side in ("down", "right") else -1
            tip = (hx, hy + sgn * o.width) if o.orientation == "h" else (hx + sgn * o.width, hy)
            line("DOORS", P(o.hinge), P(tip))
            latch = o.end if (abs(hx - o.start[0]) + abs(hy - o.start[1])) < 1e-6 or \
                math.hypot(hx - o.start[0], hy - o.start[1]) < math.hypot(hx - o.end[0], hy - o.end[1]) else o.start
            hc = P(o.hinge)
            a0 = math.degrees(math.atan2(P(latch)[1] - hc[1], P(latch)[0] - hc[0])) % 360
            a1 = math.degrees(math.atan2(P(tip)[1] - hc[1], P(tip)[0] - hc[0])) % 360
            # DXF arcs run counter-clockwise from start to end angle
            if (a1 - a0) % 360 > 180:
                a0, a1 = a1, a0
            arc("DOORS", hc, o.width * k, a0, a1)

    layers = ["WALLS", "DOORS", "WINDOWS", "OPENINGS"]
    colors = {"WALLS": 1, "DOORS": 2, "WINDOWS": 5, "OPENINGS": 3}
    tbl = "\n".join(f"0\nLAYER\n2\n{n}\n70\n0\n62\n{colors[n]}\n6\nCONTINUOUS" for n in layers)
    dxf = (f"0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n{4 if plan.mm_per_px else 0}\n0\nENDSEC\n"
           f"0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n70\n{len(layers)}\n{tbl}\n0\nENDTAB\n0\nENDSEC\n"
           "0\nSECTION\n2\nENTITIES\n" + "\n".join(ents) + "\n0\nENDSEC\n0\nEOF\n")
    with open(path, "w", encoding="ascii") as f:
        f.write(dxf)
