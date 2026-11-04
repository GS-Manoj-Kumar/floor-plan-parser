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


def test_preview_and_auto_scale_pdf(tmp_path):
    pytest.importorskip("pypdfium2")
    from fpvec.autoscale import ocr_available
    from fpvec.synth import dim_text_layer, make_pdf
    img, gt = make_plan(hollow_interior=False, dims=True)
    f = tmp_path / "p.pdf"
    make_pdf(img, str(f), dim_text_layer(gt), dpi=150)
    c = TestClient(app)
    data = f.read_bytes()
    p = c.post("/api/preview", files={"file": ("p.pdf", data, "application/pdf")}, data={"dpi": "300"}).json()
    assert p["kind"] == "pdf" and p["pages"] == 1 and p["image"].startswith("data:image/")
    r = c.post("/api/vectorize", files={"file": ("p.pdf", data, "application/pdf")},
               data={"auto_scale": "1", "dpi": "300", "photo": "auto"}).json()
    assert r["scale"]["method"] == "dimensions" and abs(r["plan"]["mm_per_px"] - 5.0) < 0.05
    assert r["input"]["kind"] == "pdf" and r["image"].startswith("data:image/")
    assert any(d["used"] for d in r["scale"]["readings"])
