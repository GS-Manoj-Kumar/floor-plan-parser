"""Coloured sample plans: room-area scale, the scorer, and one end-to-end run."""
import pytest

from fpvec import process
from fpvec.autoscale import ocr_available, parse_area
from fpvec.evaluate import score
from fpvec.synth import make_photo
from fpvec.synth_color import make_color_plan

needs_ocr = pytest.mark.skipif(not ocr_available(), reason="pip install fpvec[ocr]")


@pytest.mark.parametrize("text,m2", [("38.4 m2", 38.4), ("38.4m²", 38.4), ("12 sqm", 12), ("13.9m2", 13.9),
                                     ("220 sq ft", 20.44), ("LIVING", None), ("9999 m2", None)])
def test_parse_area(text, m2):
    v = parse_area(text)
    assert (v is None) if m2 is None else abs(v / 1e6 - m2) < 0.05


def test_score_perfect_and_wrong_scale():
    img, gt = make_color_plan(0)
    r = process(img, mm_per_px=gt["mm_per_px"], auto_scale=False)
    s = score(r.plan, gt)
    assert s["wall_recall"] > 0.95 and s["size_err_pct"] < 1
    r.plan.mm_per_px *= 1.1                       # 10 % off: size error shows, shape score does not care
    s2 = score(r.plan, gt)
    assert s2["size_err_pct"] > 8 and s2["shape_wall_recall"] > 0.95


@needs_ocr
@pytest.mark.parametrize("seed", [7, 12])
def test_scale_from_room_areas(seed):
    """Plans without dimensions: scale from the room-area labels."""
    img, gt = make_color_plan(seed)
    assert not gt["dims"]
    r = process(img)
    assert r.scale_method == "room areas"
    assert abs(r.plan.mm_per_px / gt["mm_per_px"] - 1) < 0.02


@needs_ocr
def test_coloured_photo_end_to_end():
    img, gt = make_color_plan(2)
    r = process(make_photo(img, seed=2, tilt=0.1, rot_deg=-5))
    s = score(r.plan, gt)
    assert r.scale_method == "dimensions" and s["size_err_pct"] < 2.5
    assert s["shape_wall_recall"] > 0.95 and s["windows_found"] == s["windows_true"]
