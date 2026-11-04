"""Export to the Floor Plan editor's document format (floor_plan.html / SketchUp).

The editor stores millimetres with **y up**; fpvec works in image pixels with
y down. The result is a document the editor's *Open* button loads directly
(its ``fixDoc`` fills in settings, layers and levels):

* ``wall``    {a, b, bg, thk, h, just:'C', top:'next'}  centreline in mm
* ``opening`` {wall, kind door|window|cut, s0, w, sill, h, shape:'rect',
               style, hinge L|R, side ±1, tpl}
* ``image``   the source image as a 50 % underlay on the Images layer, placed
              so the traced walls sit exactly on it.
"""
from __future__ import annotations

import base64
import json
import math
import mimetypes
from typing import List, Optional

from .models import FloorPlan

# editor defaults (defSettings in floor_plan.html), in mm
WALL_H = 3048.0
DOOR = dict(h=2133.6, sill=0.0)
WINDOW = dict(h=1219.2, sill=914.4)


def _image_data_url(image_path: str) -> str:
    mt = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        return f"data:{mt};base64," + base64.b64encode(f.read()).decode("ascii")


def to_floorplan_doc(plan: FloorPlan, image_path: Optional[str] = None, image_data_url: Optional[str] = None,
                     mm_per_px: Optional[float] = None, level: str = "V1", wall_h: float = WALL_H) -> dict:
    """Build the editor document. ``mm_per_px`` overrides ``plan.mm_per_px``;
    without any scale the main wall is assumed to be 200 mm thick."""
    k = mm_per_px or plan.mm_per_px or (200.0 / plan.wall_thickness_px if plan.wall_thickness_px else 10.0)
    H = plan.image_size[1]

    def P(p):  # image px (y down) -> editor mm (y up)
        return {"x": round(p[0] * k, 1), "y": round((H - p[1]) * k, 1)}

    ents: List[dict] = []
    nid = [1]

    def add(e: dict) -> dict:
        e["id"] = f"e{nid[0]}"
        nid[0] += 1
        e["lv"] = level
        ents.append(e)
        return e

    if image_path or image_data_url:
        src = image_data_url or _image_data_url(image_path)
        W0, H0 = plan.image_size
        add({"t": "image", "src": src, "p": {"x": 0.0, "y": 0.0}, "w": round(W0 * k, 1), "h": round(H0 * k, 1),
             "rot": 0, "op": 0.5, "name": "traced image", "layer": "LI"})

    wall_ent = {}
    for w in plan.walls:
        e = add({"t": "wall", "a": P(w.start), "b": P(w.end), "bg": 0, "thk": round(w.thickness * k, 1),
                 "h": wall_h, "just": "C", "top": "next", "layer": "LW"})
        wall_ent[w.id] = (e, w)

    for o in plan.openings:
        host = None
        for hid in o.host_wall_ids:
            if hid in wall_ent:
                host = wall_ent[hid]
                break
        if host is None:
            # an opening found without a wall around it (e.g. a door between two
            # wall stubs): give it its own short wall, which the editor joins
            e = add({"t": "wall", "a": P(o.start), "b": P(o.end), "bg": 0, "thk": round(o.thickness * k, 1),
                     "h": wall_h, "just": "C", "top": "next", "layer": "LW"})
            host = (e, None)
        e, _ = host
        a, b = e["a"], e["b"]
        L = math.hypot(b["x"] - a["x"], b["y"] - a["y"]) or 1.0
        ux, uy = (b["x"] - a["x"]) / L, (b["y"] - a["y"]) / L
        ps, pe = P(o.start), P(o.end)
        s_a = (ps["x"] - a["x"]) * ux + (ps["y"] - a["y"]) * uy
        s_b = (pe["x"] - a["x"]) * ux + (pe["y"] - a["y"]) * uy
        lo, hi = min(s_a, s_b), max(s_a, s_b)
        if lo < 0 or hi > L:
            # the opening sits just past the wall's end (a door next to a corner):
            # run the wall on through it, as the editor models it
            if lo < 0:
                a = e["a"] = {"x": round(a["x"] + ux * lo, 1), "y": round(a["y"] + uy * lo, 1)}
                hi, lo, L = hi - lo, 0.0, L - lo
            if hi > L:
                b = e["b"] = {"x": round(a["x"] + ux * hi, 1), "y": round(a["y"] + uy * hi, 1)}
                L = hi
        s0, s1 = lo, hi
        if s1 - s0 < 1:
            continue
        rec = {"t": "opening", "wall": e["id"], "kind": "cut" if o.kind == "opening" else o.kind,
               "s0": round(s0, 1), "w": round(s1 - s0, 1), "shape": "rect", "tpl": "", "layer": "LO",
               "hinge": "L", "side": 1}
        if o.kind == "door":
            rec.update(DOOR, style="double" if o.double else "single")
            if o.hinge is not None:
                ph = P(o.hinge)
                sh = (ph["x"] - a["x"]) * ux + (ph["y"] - a["y"]) * uy
                rec["hinge"] = "L" if abs(sh - s0) <= abs(sh - s1) else "R"
            if o.swing_side:
                # swing direction in editor coords (image "up" = +y)
                d = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}[o.swing_side]
                nx, ny = -uy, ux                       # the editor's wall normal, vperp(u)
                rec["side"] = 1 if d[0] * nx + d[1] * ny >= 0 else -1
        elif o.kind == "window":
            rec.update(WINDOW, style="std")
        else:
            rec.update(DOOR, style="none")
        add(rec)

    return {"app": "floorplan", "v": 1, "nextId": nid[0], "curLevel": level, "ents": ents,
            "source": {"tool": "fpvec", "mm_per_px": k}}


def save_floorplan_doc(plan: FloorPlan, path: str, image_path: Optional[str] = None, **kw) -> dict:
    doc = to_floorplan_doc(plan, image_path=image_path, **kw)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return doc
