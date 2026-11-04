"""The Floor Plan editor document (floor_plan.html) export."""
import math

from fpvec import to_floorplan_doc, vectorize
from fpvec.synth import make_plan


def test_doc_shape_and_units():
    img, gt = make_plan(hollow_interior=False)
    plan = vectorize(img, gt["mm_per_px"])
    doc = to_floorplan_doc(plan)
    assert doc["app"] == "floorplan" and doc["v"] == 1
    walls = [e for e in doc["ents"] if e["t"] == "wall"]
    opens = [e for e in doc["ents"] if e["t"] == "opening"]
    assert len(walls) >= 8 and len(opens) == len(plan.openings)
    ids = {e["id"] for e in doc["ents"]}
    assert len(ids) == len(doc["ents"]) and doc["nextId"] == len(doc["ents"]) + 1
    # every opening lies inside its wall
    by_id = {w["id"]: w for w in walls}
    for o in opens:
        w = by_id[o["wall"]]
        L = math.hypot(w["b"]["x"] - w["a"]["x"], w["b"]["y"] - w["a"]["y"])
        assert -0.5 <= o["s0"] and o["s0"] + o["w"] <= L + 0.5
        assert o["kind"] in ("door", "window", "cut") and o["hinge"] in ("L", "R") and o["side"] in (1, -1)
    # exterior walls: 10 m x 8 m, y up (the bottom wall has the smallest y)
    long_h = sorted((w for w in walls if abs(w["a"]["y"] - w["b"]["y"]) < 1), key=lambda w: w["a"]["y"])
    assert abs(abs(long_h[0]["b"]["x"] - long_h[0]["a"]["x"]) - 10000) < 150
    assert long_h[-1]["a"]["y"] - long_h[0]["a"]["y"] > 7500


def test_door_hinge_and_swing_match_drawing():
    """The front door (bottom wall, hinge at its left end, swings into the room = up in the image)."""
    img, gt = make_plan(hollow_interior=False)
    plan = vectorize(img, gt["mm_per_px"])
    doc = to_floorplan_doc(plan)
    walls = {e["id"]: e for e in doc["ents"] if e["t"] == "wall"}
    front = None
    for o in doc["ents"]:
        if o["t"] == "opening" and o["kind"] == "door":
            w = walls[o["wall"]]
            if abs(w["a"]["y"] - w["b"]["y"]) < 1 and w["a"]["y"] < 2000:  # bottom wall (y up)
                front = (o, w)
    assert front, "front door not exported"
    o, w = front
    ux = 1 if w["b"]["x"] > w["a"]["x"] else -1
    hinge_x = w["a"]["x"] + ux * (o["s0"] if o["hinge"] == "L" else o["s0"] + o["w"])
    assert abs(hinge_x - (7000 + 1500)) < 150          # synth: door at 7000 mm, hinge at its start
    n_y = ux                                            # editor normal vperp(u) = (-uy, ux)
    assert o["side"] * n_y > 0                          # swings towards +y (into the room)
