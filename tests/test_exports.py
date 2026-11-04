import json

from fpvec import draw_overlay, to_dxf, to_json, to_svg, vectorize
from fpvec.scale import mm_per_px_from_reference
from fpvec.synth import make_plan


def test_exports(tmp_path):
    img, gt = make_plan(hollow_interior=False)
    plan = vectorize(img, gt["mm_per_px"])
    j = json.loads(to_json(plan, str(tmp_path / "p.json")))
    assert len(j["walls"]) == len(plan.walls)
    assert "<svg" in to_svg(plan, str(tmp_path / "p.svg"), units="mm")
    to_dxf(plan, str(tmp_path / "p.dxf"))
    txt = (tmp_path / "p.dxf").read_text()
    assert "WALLS" in txt and "DOORS" in txt and txt.strip().endswith("EOF")
    vis = draw_overlay(plan, img, str(tmp_path / "p.png"))
    assert vis.shape[0] > 0


def test_reference_scale():
    assert mm_per_px_from_reference((0, 0), (100, 0), 5000) == 50
