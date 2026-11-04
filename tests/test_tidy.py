"""Tidy: ortho walls and joining wall ends."""
import json

from fastapi.testclient import TestClient

from fpvec.api import app
from fpvec.models import FloorPlan, Opening, Wall
from fpvec.tidy import loose_ends, snap_distance, tidy


def W(i, a, b, t=10.0):
    o = "h" if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else "v"
    return Wall(i, a, b, t, o)


def plan(walls, openings=(), mm=None):
    return FloorPlan((1000, 1000), list(walls), list(openings), mm, 10.0, 2.0)


def test_ortho_straightens_and_aligns():
    p = plan([W(0, (100, 100), (400, 103)), W(1, (420, 98), (700, 99))])
    out = tidy(p, ortho=True, snap=False)
    ys = {w.start[1] for w in out.walls} | {w.end[1] for w in out.walls}
    assert len(ys) == 1                            # both on one exact line


def test_ortho_off_keeps_diagonals():
    p = plan([Wall(0, (100, 100), (400, 300), 10, "d")])
    assert tidy(p, ortho=False, snap=False).walls[0].orientation == "d"
    assert tidy(p, ortho=True, snap=False).walls[0].orientation in ("h", "v")


def test_gap_closed_within_distance_only():
    walls = [W(0, (100, 100), (400, 100)), W(1, (440, 100), (700, 100))]   # 40 px gap = 4 t
    assert len(tidy(plan(walls), snap_dist=3, snap_unit="t").walls) == 2
    out = tidy(plan(walls), snap_dist=5, snap_unit="t")
    assert len(out.walls) == 1 and out.walls[0].start[0] == 100 and out.walls[0].end[0] == 700


def test_units_pct_and_mm():
    p = plan([W(0, (100, 100), (400, 100)), W(1, (440, 100), (700, 100))], mm=20.0)
    assert snap_distance(p, 10, "pct") == 60.0     # 10 % of 600 px
    assert snap_distance(p, 1000, "mm") == 50.0    # 1000 mm / 20 mm per px
    assert len(tidy(p, snap_dist=10, snap_unit="pct").walls) == 1
    assert len(tidy(p, snap_dist=500, snap_unit="mm").walls) == 2


def test_t_junction_and_l_corner():
    # a vertical wall stopping short of a horizontal one (T), and two walls that almost make a corner (L)
    walls = [W(0, (100, 100), (600, 100)), W(1, (300, 125), (300, 400)),
             W(2, (620, 130), (620, 400))]
    p = plan(walls)
    assert loose_ends(p) > 0
    out = tidy(p, snap_dist=3)
    v = [w for w in out.walls if w.orientation == "v"]
    assert min(min(w.start[1], w.end[1]) for w in v if abs(w.start[0] - 300) < 1) == 100
    h = [w for w in out.walls if w.orientation == "h"][0]
    corner = [w for w in v if abs(w.start[0] - 620) < 1][0]
    assert max(h.start[0], h.end[0]) == 620 and min(corner.start[1], corner.end[1]) == 100


def test_openings_rehosted():
    walls = [W(0, (100, 100), (400, 102)), W(1, (400, 102), (700, 102))]
    op = Opening(0, "door", (200, 104), (280, 104), 10, "h")
    out = tidy(plan(walls, [op]))
    assert out.openings[0].host_wall_ids
    w = out.walls[out.openings[0].host_wall_ids[0]]
    assert out.openings[0].start[1] == w.start[1]


def test_api_tidy():
    p = plan([W(0, (100, 100), (400, 100)), W(1, (440, 100), (700, 100))])
    c = TestClient(app)
    r = c.post("/api/tidy", data={"plan": json.dumps(p.to_dict()), "snap_dist": "5", "snap_unit": "t"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert len(j["plan"]["walls"]) == 1 and j["loose_after"] <= j["loose_before"]
    assert c.post("/api/tidy", data={"plan": json.dumps(p.to_dict()), "snap_unit": "ft"}).status_code == 400


def test_tuning_has_tidy_keys():
    from fpvec.presets import tuning_params
    assert tuning_params({"ortho": False, "snap_ends": True, "snap_dist": "4", "snap_unit": "pct"}) == \
        {"ortho": False, "snap_ends": True, "snap_dist": 4.0, "snap_unit": "pct"}


def test_parse_thicknesses():
    from fpvec.tidy import parse_thicknesses
    assert parse_thicknesses("100-250") == ("range", [100.0, 250.0])
    assert parse_thicknesses("250 to 100 mm") == ("range", [100.0, 250.0])
    assert parse_thicknesses("230, 100,150") == ("list", [100.0, 150.0, 230.0])
    assert parse_thicknesses("100 150") == ("list", [100.0, 150.0])
    assert parse_thicknesses("") is None and parse_thicknesses("abc") is None


def test_match_thickness_list_and_range():
    from fpvec.tidy import match_thickness
    walls = [W(0, (100, 100), (400, 100), t=4.0), W(1, (100, 100), (100, 400), t=9.0), W(2, (400, 100), (400, 400), t=30.0)]
    op = Opening(0, "door", (100, 200), (100, 260), 9.0, "v", [1])
    p = plan(walls, [op], mm=20.0)                      # 80, 180, 600 mm
    out = match_thickness(p, "100, 150, 230")
    assert [round(w.thickness * 20) for w in out.walls] == [100, 150, 230]
    assert round(out.openings[0].thickness * 20) == 150
    out = match_thickness(p, "100-250")
    assert [round(w.thickness * 20) for w in out.walls] == [100, 180, 250]
    assert match_thickness(p, "") is p


def test_thickness_in_params_and_api():
    from fpvec.presets import tuning_params
    assert tuning_params({"wall_thicknesses": " 100-250 "}) == {"wall_thicknesses": "100-250"}
    assert tuning_params({"wall_thicknesses": "abc"}) == {}
    p = plan([W(0, (100, 100), (400, 100), t=4.0)], mm=20.0)
    c = TestClient(app)
    r = c.post("/api/tidy", data={"plan": json.dumps(p.to_dict()), "wall_thicknesses": "100, 200"})
    assert r.status_code == 200 and round(r.json()["plan"]["walls"][0]["thickness_px"] * 20) == 100
    assert c.post("/api/tidy", data={"plan": json.dumps(p.to_dict()), "wall_thicknesses": "x"}).status_code == 400


def test_process_with_thicknesses():
    import cv2
    from fpvec.pipeline import Params
    from fpvec.process import process
    img = cv2.imread("examples/sample_plan.png")
    res = process(img, auto_scale=False, params=Params(items="walls", wall_thicknesses="100, 230"))
    vals = {round(w.thickness * res.plan.mm_per_px) for w in res.plan.walls}
    assert vals <= {100, 230}, vals
