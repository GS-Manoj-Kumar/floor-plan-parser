"""Accuracy tests on generated plans with known ground truth."""
import math

import numpy as np
import pytest

from fpvec import vectorize
from fpvec.synth import make_plan


def _match(found, truth, tol_px):
    """Greedy match of openings by centre distance; returns (matches, misses)."""
    used, hits = set(), []
    for t in truth:
        tc = ((t["start"][0] + t["end"][0]) / 2, (t["start"][1] + t["end"][1]) / 2)
        best, bd = None, tol_px
        for i, o in enumerate(found):
            if i in used or o.kind != t["kind"]:
                continue
            d = math.dist(tc, o.center)
            if d < bd:
                best, bd = i, d
        if best is not None:
            used.add(best)
            hits.append((t, found[best]))
    return hits, len(truth) - len(hits)


@pytest.mark.parametrize("hollow,noise", [(False, False), (False, True), (True, False)])
def test_openings_found(hollow, noise):
    img, gt = make_plan(hollow_interior=hollow, noise=noise)
    plan = vectorize(img, gt["mm_per_px"])
    tol = 300 / gt["mm_per_px"]                     # centre within 300 mm
    hits, misses = _match(plan.openings, gt["openings"], tol)
    n_doors = sum(o["kind"] == "door" for o in gt["openings"])
    n_win = len(gt["openings"]) - n_doors
    door_hits = sum(t["kind"] == "door" for t, _ in hits)
    win_hits = len(hits) - door_hits
    assert win_hits == n_win, f"windows {win_hits}/{n_win}"
    assert door_hits >= n_doors - (1 if hollow else 0), f"doors {door_hits}/{n_doors}"
    # widths within 150 mm
    for t, o in hits:
        tw = math.dist(t["start"], t["end"]) * gt["mm_per_px"]
        assert abs(o.width * gt["mm_per_px"] - tw) < 150, (t["kind"], tw, o.width * gt["mm_per_px"])


def test_exterior_walls_and_thickness():
    img, gt = make_plan(hollow_interior=False)
    plan = vectorize(img, gt["mm_per_px"])
    ext = [w for w in plan.walls if w.exterior]
    assert len(ext) == 4
    for w in ext:
        assert abs(w.length * gt["mm_per_px"] - (10000 if w.orientation == "h" else 8000)) < 150
    # main wall thickness ~240 mm
    assert abs(plan.wall_thickness_px * gt["mm_per_px"] - 240) < 40


def test_doors_have_hinge_and_swing():
    img, gt = make_plan(hollow_interior=False)
    plan = vectorize(img, gt["mm_per_px"])
    doors = [o for o in plan.openings if o.kind == "door"]
    assert doors
    for d in doors:
        assert d.hinge is not None
        assert d.swing_side in ("up", "down", "left", "right")


def test_scale_free_output_and_json():
    img, gt = make_plan()
    plan = vectorize(img)                           # no scale: px only
    d = plan.to_dict()
    assert d["mm_per_px"] is None
    assert d["walls"] and d["walls"][0]["length_mm"] is None
    assert {"start_px", "end_px", "thickness_px", "exterior"} <= set(d["walls"][0])


def test_resolution_invariance():
    """The same plan rendered at two resolutions gives the same mm answer."""
    res = []
    for s in (0.08, 0.15):
        img, gt = make_plan(px_per_mm=s, hollow_interior=False)
        plan = vectorize(img, gt["mm_per_px"])
        res.append(sorted(round(o.width * gt["mm_per_px"], -2) for o in plan.openings if o.kind == "window"))
    assert res[0] == res[1]


def test_blank_image_does_not_crash():
    img = np.full((400, 500, 3), 255, np.uint8)
    plan = vectorize(img)
    assert plan.walls == [] and plan.openings == []
