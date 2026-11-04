"""Auto scale from dimension text, PDF input and photo correction."""
import cv2
import pytest

from fpvec import process
from fpvec.autoscale import TextBox, estimate_scale, ocr_available, parse_length
from fpvec.inputs import load_input, pdf_available
from fpvec.synth import dim_text_layer, make_pdf, make_photo, make_plan

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="pip install fpvec[ocr]")
needs_pdf = pytest.mark.skipif(not pdf_available(), reason="pip install fpvec[pdf]")


@pytest.mark.parametrize("text,mm", [
    ("3600", 3600), ("3600mm", 3600), ("3.60", 3600), ("360cm", 3600), (".1000", 1000),
    ("12'-6\"", 3810), ("12'6\"", 3810), ("10'", 3048), ("150\"", 3810), ("2,40", 2400),
])
def test_parse_length(text, mm):
    assert abs(parse_length(text) - mm) < 0.5


@pytest.mark.parametrize("text", ["LIVING", "12", "O3", "", "99999999"])
def test_parse_rejects(text):
    assert parse_length(text) is None


def _boxes_from_gt(gt):
    """Text boxes where the synthetic dimension numbers were drawn (no OCR needed)."""
    out = []
    S = 1 / gt["mm_per_px"]
    for d in gt["dims"]:
        (ax, ay), (bx, by) = d["a"], d["b"]
        h = 220 * S
        w = len(d["text"]) * h * 0.55
        if d["horiz"]:
            cx = (ax + bx) / 2
            out.append(TextBox(d["text"], (cx - w / 2, ay - 60 * S - h, cx + w / 2, ay - 60 * S)))
        else:
            cy = (ay + by) / 2
            out.append(TextBox(d["text"], (ax - 60 * S - h, cy - w / 2, ax - 60 * S, cy + w / 2)))
    return out


def test_scale_from_dimension_lines_without_ocr():
    img, gt = make_plan(hollow_interior=False, dims=True)
    r = estimate_scale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), _boxes_from_gt(gt))
    assert r.method == "dimensions" and r.votes >= 8
    assert abs(r.mm_per_px - gt["mm_per_px"]) / gt["mm_per_px"] < 0.01


@needs_ocr
@pytest.mark.parametrize("units", ["mm", "ftin"])
def test_scale_by_ocr(units):
    img, gt = make_plan(hollow_interior=False, dims=True, dim_units=units)
    r = process(img)
    assert r.scale_method == "dimensions"
    assert abs(r.plan.mm_per_px - gt["mm_per_px"]) / gt["mm_per_px"] < 0.01


@needs_pdf
def test_pdf_text_layer(tmp_path):
    img, gt = make_plan(hollow_interior=False, dims=True)
    f = tmp_path / "p.pdf"
    make_pdf(img, str(f), dim_text_layer(gt), dpi=150)
    li = load_input(str(f), dpi=300)
    assert li.kind == "pdf" and li.pages == 1 and li.image.shape[1] == 2600
    assert sum(t.text.isdigit() for t in li.text) >= 10
    r = process(str(f), dpi=300)
    assert r.scale_method == "dimensions"
    assert abs(r.plan.mm_per_px - 5.0) < 0.05          # 10 mm/px drawn at 150 dpi, rendered at 300
    assert sum(o.kind == "window" for o in r.plan.openings) == 5


@needs_ocr
@pytest.mark.parametrize("seed,tilt,rot", [(1, 0.12, 7), (2, 0.18, -12), (3, 0.08, 25)])
def test_photo_corrected_and_scaled(seed, tilt, rot):
    img, gt = make_plan(hollow_interior=False, dims=True)
    ph = make_photo(img, seed=seed, tilt=tilt, rot_deg=rot)
    r = process(ph, photo="auto")
    assert "perspective" in r.rectify.applied
    k = r.plan.mm_per_px
    ext = [w for w in r.plan.walls if w.exterior]
    assert len(ext) == 4
    for w in ext:
        true = 10000 if w.orientation == "h" else 8000
        assert abs(w.length * k - true) / true < 0.015, (w.orientation, w.length * k)
    assert sum(o.kind == "window" for o in r.plan.openings) == 5
    assert sum(o.kind == "door" for o in r.plan.openings) >= 4


def test_clean_image_is_not_photo_corrected():
    img, gt = make_plan(hollow_interior=False)
    r = process(img, mm_per_px=10)
    assert r.rectify.applied == [] and r.scale_method == "given"
