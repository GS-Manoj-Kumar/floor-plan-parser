"""Wall marks, plain-opening rule, presets, edited-plan export and the session API."""
import json

import cv2
import numpy as np
import pytest

from fpvec import process
from fpvec.evaluate import score
from fpvec.models import plan_from_dict
from fpvec.pipeline import Params
from fpvec.presets import DEFAULT_PRESET, PRESETS, tuning_params
from fpvec.synth import make_plan


def _P(gt):
    k = gt["mm_per_px"]
    return lambda x, y: ((x + 1500) / k, (y + 1500) / k)


def test_wall_marks_raise_interior_recall():
    img, gt = make_plan(hollow_interior=True)
    P = _P(gt)
    hints = [[P(4000, 0), P(4000, 4000)], [P(0, 4000), P(6000, 4000)], [P(6000, 4000), P(6000, 8000)],
             [P(0, 6000), P(2500, 6000)]]
    base = score(process(img, mm_per_px=gt["mm_per_px"], auto_scale=False).plan, gt)
    marked = score(process(img, mm_per_px=gt["mm_per_px"], auto_scale=False, params=Params(hints=hints)).plan, gt)
    assert marked["int_wall_recall"] > base["int_wall_recall"]
    assert marked["int_wall_recall"] > 0.9
    # doors in the marked walls are still found
    assert marked["doors_found"] >= base["doors_found"]


def test_wall_mark_on_blank_paper_is_kept_as_drawn():
    img = np.full((1100, 1300, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (1200, 1000), (0, 0, 0), 20)
    r = process(img, mm_per_px=10, auto_scale=False, params=Params(hints=[[(100, 550), (1200, 550)]]))
    mids = [w for w in r.plan.walls if w.orientation == "h" and abs(w.start[1] - 550) < 3]
    assert mids and abs(mids[0].length - 1100) < 30


def test_plain_openings_setting():
    img, gt = make_plan(hollow_interior=False)
    k = gt["mm_per_px"]
    n = {}
    for mode in ("off", "between_walls", "all"):
        r = process(img, mm_per_px=k, auto_scale=False, params=Params(plain_openings=mode))
        n[mode] = sum(o.kind == "opening" for o in r.plan.openings)
    assert n["off"] == 0 and n["between_walls"] <= n["all"]


def test_presets_are_valid():
    assert DEFAULT_PRESET in PRESETS
    for name, p in PRESETS.items():
        assert tuning_params(p["tuning"]) == p["tuning"], name
        Params(**p["tuning"])
    assert tuning_params({"door_score": "0.7", "nope": 1, "sensitivity": "bad"}) == {"door_score": 0.7}


def test_edited_plan_round_trip_rehosts_openings():
    img, gt = make_plan(hollow_interior=False)
    r = process(img, mm_per_px=gt["mm_per_px"], auto_scale=False)
    d = r.plan.to_dict()
    # move the first door 40 px along its wall, as the editor would
    door = next(o for o in d["openings"] if o["type"] == "door")
    horiz = abs(door["end_px"][0] - door["start_px"][0]) > abs(door["end_px"][1] - door["start_px"][1])
    for key in ("start_px", "end_px"):
        door[key] = [door[key][0] + (40 if horiz else 0), door[key][1] + (0 if horiz else 40)]
    fp = plan_from_dict(d)
    assert len(fp.walls) == len(d["walls"]) and len(fp.openings) == len(d["openings"])
    assert all(o.host_wall_ids for o in fp.openings)


def test_session_api_flow():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from fpvec.api import app
    c = TestClient(app)
    img, gt = make_plan(hollow_interior=False, dims=True)
    ok, png = cv2.imencode(".png", img)
    r = c.post("/api/load", files={"file": ("p.png", png.tobytes(), "image/png")}, data={"photo": "auto"}).json()
    tok = r["token"]
    assert r["image"].startswith("data:image/")
    prep = json.dumps(PRESETS[DEFAULT_PRESET]["prep"])
    assert c.post("/api/prepare", data={"token": tok, "prep": prep}).json()["image"].startswith("data:image/")
    det = c.post("/api/detect", data={"token": tok, "prep": prep, "auto_scale": "1",
                                      "tuning": json.dumps({"door_score": 0.6})}).json()
    assert det["scale"]["method"] == "dimensions" and abs(det["plan"]["mm_per_px"] - gt["mm_per_px"]) < 0.1
    plan = det["plan"]
    plan["openings"] = plan["openings"][:-1]                        # an edit: delete one opening
    ex = c.post("/api/export", data={"plan": json.dumps(plan), "token": tok}).json()
    assert len(ex["plan"]["openings"]) == len(plan["openings"])
    assert ex["floorplan"]["app"] == "floorplan" and any(e["t"] == "image" for e in ex["floorplan"]["ents"])
    assert c.post("/api/prepare", data={"token": "nope"}).status_code == 404
    assert "bw" in c.get("/api/presets").json()["presets"]
