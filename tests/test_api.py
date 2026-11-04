import cv2
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from fpvec.api import app  # noqa: E402
from fpvec.synth import make_plan  # noqa: E402


def test_demo_page_and_vectorize():
    c = TestClient(app)
    assert "fpvec" in c.get("/").text
    img, gt = make_plan(hollow_interior=False)
    ok, png = cv2.imencode(".png", img)
    r = c.post("/api/vectorize", files={"file": ("p.png", png.tobytes(), "image/png")},
               data={"mm_per_px": str(gt["mm_per_px"])})
    assert r.status_code == 200
    j = r.json()
    assert j["plan"]["walls"] and j["floorplan"]["app"] == "floorplan"
    assert "ENTITIES" in j["dxf"] and j["svg"].startswith("<svg")


def test_bad_image():
    r = TestClient(app).post("/api/vectorize", files={"file": ("x.png", b"nope", "image/png")})
    assert r.status_code == 400
