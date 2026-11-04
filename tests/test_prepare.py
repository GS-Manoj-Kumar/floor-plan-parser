"""Preparation step: grey levels and removed / blurred areas, after correction."""
import json

import cv2
import numpy as np
import pytest

from fpvec import apply_prep, process
from fpvec.prepare import describe, is_identity, levels_lut, normalize
from fpvec.synth_color import make_color_plan


def test_identity_and_normalize():
    assert is_identity(None) and is_identity({}) and is_identity({"masks": []})
    s = normalize({"black": 300, "white": 10})
    assert s["black"] == 254 and s["white"] == 255
    img = np.full((20, 20, 3), 128, np.uint8)
    assert np.array_equal(apply_prep(img, None), img)


def test_levels_darken_and_whiten():
    lut = levels_lut(100, 200, False)
    assert lut[100] == 0 and lut[50] == 0 and lut[200] == 255 and lut[230] == 255 and 120 < lut[150] < 135
    b = levels_lut(100, 200, True)
    assert b[149] == 0 and b[151] == 255
    img = np.zeros((10, 30, 3), np.uint8)
    img[:, :10] = 90       # grey line   -> black
    img[:, 10:20] = 160    # mid grey    -> stays grey
    img[:, 20:] = 220      # floor tint  -> white
    out = apply_prep(img, {"grayscale": True, "black": 100, "white": 200})
    assert out[0, 5, 0] == 0 and out[0, 25, 0] == 255 and 0 < out[0, 15, 0] < 255


def test_grayscale_makes_colour_grey():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:] = (200, 50, 20)
    out = apply_prep(img, {"grayscale": True})
    assert out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2]


def test_masks_white_and_blur():
    img = np.zeros((100, 100, 3), np.uint8)
    img[::4] = 255                                       # stripes
    spec = {"masks": [
        {"type": "rect", "points": [[0, 0], [40, 40]], "mode": "white"},
        {"type": "poly", "points": [[60, 60], [99, 60], [99, 99], [60, 99]], "mode": "blur", "blur": 6},
        {"type": "brush", "points": [[10, 80], [40, 80]], "size": 10, "mode": "white"},
    ]}
    out = apply_prep(img, spec)
    assert (out[0:40, 0:40] == 255).all()                # removed
    assert (out[78:83, 15:35] == 255).all()              # brush stroke
    blurred = out[70:90, 70:90, 0]
    assert blurred.std() < img[70:90, 70:90, 0].std() / 3  # stripes smoothed
    assert np.array_equal(out[45:55, 45:55], img[45:55, 45:55])  # untouched elsewhere
    assert "2 area(s) removed" in describe(spec) and "1 blurred" in describe(spec)


def test_process_uses_prep_and_removes_clutter():
    """Removing everything but the plan's own frame removes all walls inside the cleared area."""
    img, gt = make_color_plan(0)
    H, W = img.shape[:2]
    full = process(img)
    cleared = process(img, prep={"masks": [{"type": "rect", "points": [[0, 0], [W, H]], "mode": "white"}]})
    assert len(full.plan.walls) > 4 and len(cleared.plan.walls) == 0
    assert any("preparation applied" in n for n in cleared.notes)


def test_bw_prep_keeps_scale_and_walls():
    img, gt = make_color_plan(3)
    r = process(img, prep={"grayscale": True, "black": 110, "white": 200})
    assert r.scale_method == "dimensions"
    assert abs(r.plan.mm_per_px / gt["mm_per_px"] - 1) < 0.01


def test_api_prep():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from fpvec.api import app
    img, gt = make_color_plan(0)
    ok, png = cv2.imencode(".png", img)
    c = TestClient(app)
    spec = {"grayscale": True, "black": 100, "masks": [{"type": "rect", "points": [[0, 0], [50, 50]], "mode": "white"}]}
    r = c.post("/api/vectorize", files={"file": ("p.png", png.tobytes(), "image/png")},
               data={"auto_scale": "1", "prep": json.dumps(spec)})
    assert r.status_code == 200
    assert any("preparation applied" in n for n in r.json()["notes"])
    bad = c.post("/api/vectorize", files={"file": ("p.png", png.tobytes(), "image/png")}, data={"prep": "{nope"})
    assert bad.status_code == 400
