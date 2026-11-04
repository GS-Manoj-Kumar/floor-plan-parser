"""Score a vectorized plan against ground truth (synthetic plans).

All comparisons are in millimetres. The prediction is shifted so its
exterior bounding box starts where the ground truth's does (photos and scans
are re-framed by the correction, so pixel positions cannot be compared).
Scale errors are therefore *not* hidden: a wrong mm/px makes the plan the
wrong size and walls / openings stop matching.

Metrics
* size error      exterior width / height, % off
* wall recall     share of true wall centreline length with a predicted wall
                  (same direction) within ``tol`` mm
* wall precision  share of predicted wall length that is near a true wall
* door / window recall and precision: centre within ``open_tol`` mm, same kind
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .models import FloorPlan


def _segs_gt(gt) -> List[dict]:
    k = gt["mm_per_px"]
    out = []
    for w in gt["walls"]:
        (x0, y0), (x1, y1) = w["start"], w["end"]
        ext = w["exterior"] if "exterior" in w else w.get("t", 0) >= gt.get("ext_t", 1e9) - 1e-6
        out.append(dict(o=w["o"], a=(x0 * k, y0 * k), b=(x1 * k, y1 * k), ext=ext))
    return out


def _segs_pred(plan: FloorPlan) -> List[dict]:
    k = plan.mm_per_px or 1.0
    return [dict(o=w.orientation, a=(w.start[0] * k, w.start[1] * k), b=(w.end[0] * k, w.end[1] * k),
                 ext=w.exterior) for w in plan.walls]


def _bbox(segs, ext_only=True):
    s = [x for x in segs if x["ext"]] if ext_only and any(x["ext"] for x in segs) else segs
    if not s:
        return None
    xs = [p[0] for x in s for p in (x["a"], x["b"])]
    ys = [p[1] for x in s for p in (x["a"], x["b"])]
    return min(xs), min(ys), max(xs), max(ys)


def _sample(seg, step=50.0):
    (x0, y0), (x1, y1) = seg["a"], seg["b"]
    L = float(np.hypot(x1 - x0, y1 - y0))
    n = max(2, int(L / step))
    t = np.linspace(0, 1, n)
    return np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1), L


def _covered(samples, segs, o, tol):
    if not len(samples):
        return np.zeros(0, bool)
    hit = np.zeros(len(samples), bool)
    for s in segs:
        if s["o"] != o:
            continue
        (x0, y0), (x1, y1) = s["a"], s["b"]
        if o == "h":
            c, a, b = (y0 + y1) / 2, min(x0, x1), max(x0, x1)
            hit |= (np.abs(samples[:, 1] - c) <= tol) & (samples[:, 0] >= a - tol) & (samples[:, 0] <= b + tol)
        else:
            c, a, b = (x0 + x1) / 2, min(y0, y1), max(y0, y1)
            hit |= (np.abs(samples[:, 0] - c) <= tol) & (samples[:, 1] >= a - tol) & (samples[:, 1] <= b + tol)
    return hit


def score(plan: FloorPlan, gt: Dict, tol: float = 150.0, open_tol: float = 350.0) -> Dict:
    G, Pd = _segs_gt(gt), _segs_pred(plan)
    bg, bp = _bbox(G), _bbox(Pd)
    res: Dict = {"mm_per_px": plan.mm_per_px}
    if not bp:
        return dict(res, wall_recall=0.0, wall_precision=0.0, size_err_pct=None)
    dx, dy = bg[0] - bp[0], bg[1] - bp[1]
    for s in Pd:
        s["a"] = (s["a"][0] + dx, s["a"][1] + dy)
        s["b"] = (s["b"][0] + dx, s["b"][1] + dy)
    gw, gh = bg[2] - bg[0], bg[3] - bg[1]
    pw, ph = bp[2] - bp[0], bp[3] - bp[1]
    res["size_err_pct"] = round(100 * max(abs(pw - gw) / gw, abs(ph - gh) / gh), 2)

    def cover(src, dst):
        tot = hit = 0.0
        for s in src:
            pts, L = _sample(s)
            h = _covered(pts, dst, s["o"], tol)
            tot += L
            hit += L * (h.mean() if len(h) else 0)
        return hit / tot if tot else 0.0

    # shape-only score: also fit the size (per axis), so a wrong scale does
    # not hide how well the walls themselves were found
    sx, sy = gw / pw if pw else 1.0, gh / ph if ph else 1.0
    Ps = [dict(s, a=(bg[0] + (s["a"][0] - bg[0]) * sx, bg[1] + (s["a"][1] - bg[1]) * sy),
               b=(bg[0] + (s["b"][0] - bg[0]) * sx, bg[1] + (s["b"][1] - bg[1]) * sy)) for s in Pd]
    res["shape_wall_recall"] = round(cover(G, Ps), 3)
    res["shape_wall_precision"] = round(cover(Ps, G), 3)
    res["wall_recall"] = round(cover(G, Pd), 3)
    res["wall_precision"] = round(cover(Pd, G), 3)
    res["int_wall_recall"] = round(cover([g for g in G if not g["ext"]], Pd), 3) if any(not g["ext"] for g in G) else None

    k = gt["mm_per_px"]
    kp = plan.mm_per_px or 1.0
    gto = [dict(kind=o["kind"], c=((o["start"][0] + o["end"][0]) / 2 * k, (o["start"][1] + o["end"][1]) / 2 * k))
           for o in gt["openings"]]
    pro = [dict(kind=o.kind, c=(o.center[0] * kp + dx, o.center[1] * kp + dy)) for o in plan.openings]
    for kind in ("door", "window"):
        g = [o for o in gto if o["kind"] == kind]
        p = [o for o in pro if o["kind"] == kind]
        used, tp = set(), 0
        for o in g:
            best, bd = None, open_tol
            for i, q in enumerate(p):
                d = float(np.hypot(o["c"][0] - q["c"][0], o["c"][1] - q["c"][1]))
                if i not in used and d < bd:
                    best, bd = i, d
            if best is not None:
                used.add(best)
                tp += 1
        res[f"{kind}s_true"] = len(g)
        res[f"{kind}s_found"] = tp
        res[f"{kind}s_false"] = len(p) - tp
    return res
