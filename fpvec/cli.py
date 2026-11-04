"""Command line.

    fpvec plan.png --mm-per-px 12.5 -o out/
    fpvec plan.pdf --page 2 -o out/                 # scale read from the dimension text
    fpvec photo.jpg --photo on -o out/              # photo taken at an angle
"""
from __future__ import annotations

import argparse
import base64
import os

import cv2

from .export import draw_overlay, to_dxf, to_json, to_svg
from .floorplan_doc import to_floorplan_doc
from .pipeline import Params
from .process import process
from .scale import mm_per_px_from_reference


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Extract walls, doors and windows from a floor plan image or PDF.")
    ap.add_argument("image", help="PNG / JPG / TIFF / ... or PDF")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--mm-per-px", type=float, help="scale of the image (of the traced image, after corrections)")
    sc.add_argument("--ref", nargs=5, type=float, metavar=("X1", "Y1", "X2", "Y2", "MM"),
                    help="two image points and the real distance between them in mm")
    ap.add_argument("--no-auto-scale", action="store_true",
                    help="do not read the scale from the dimension text (OCR / PDF text)")
    ap.add_argument("--wall-mm", type=float, default=200.0,
                    help="fallback scale: assume the main wall is this thick (mm), default 200; 0 = none")
    ap.add_argument("--units", default="auto", choices=["auto", "mm", "cm", "m", "ftin"],
                    help="how plain dimension numbers are read (default auto: 3600 = mm, 3.60 = m, 12'-6\" = ft-in)")
    ap.add_argument("--photo", default="auto", choices=["auto", "on", "off"],
                    help="photo correction: perspective, lighting, skew (default auto)")
    ap.add_argument("--page", type=int, default=1, help="PDF page (1-based)")
    ap.add_argument("--dpi", type=float, default=200, help="PDF render resolution")
    ap.add_argument("--wall-px", type=float, help="override wall thickness estimate (px)")
    ap.add_argument("--no-thin-walls", action="store_true", help="only detect solid (filled) walls")
    ap.add_argument("--formats", default="json,svg,dxf,png,floorplan",
                    help="comma list of json, svg, dxf, png, floorplan (the Floor Plan editor document)")
    a = ap.parse_args(argv)

    mm = a.mm_per_px
    if a.ref:
        mm = mm_per_px_from_reference((a.ref[0], a.ref[1]), (a.ref[2], a.ref[3]), a.ref[4])
    r = process(a.image, mm_per_px=mm, auto_scale=not a.no_auto_scale, photo=a.photo, page=a.page - 1,
                dpi=a.dpi, units=a.units, wall_mm=a.wall_mm or None,
                params=Params(wall_thickness_px=a.wall_px, detect_thin_walls=not a.no_thin_walls))
    plan = r.plan

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    fm = set(a.formats.split(","))
    if "json" in fm:
        to_json(plan, os.path.join(a.out, stem + ".json"))
    if "svg" in fm:
        to_svg(plan, os.path.join(a.out, stem + ".svg"))
    if "dxf" in fm:
        to_dxf(plan, os.path.join(a.out, stem + ".dxf"))
    traced = os.path.join(a.out, stem + "_traced.png")
    if r.rectify.applied or r.input_kind == "pdf":
        cv2.imwrite(traced, r.image)          # the image the coordinates refer to
    if "floorplan" in fm:
        ok, png = cv2.imencode(".png", r.image)
        url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")
        import json
        with open(os.path.join(a.out, stem + "_floor_plan.json"), "w", encoding="utf-8") as f:
            json.dump(to_floorplan_doc(plan, image_data_url=url), f, indent=1)
    if "png" in fm:
        draw_overlay(plan, r.image, os.path.join(a.out, stem + "_overlay.png"))
    n = {k: sum(o.kind == k for o in plan.openings) for k in ("door", "window", "opening")}
    sc_txt = {"given": "given", "dimensions": f"from {r.scale.votes}/{r.scale.total} dimensions",
              "wall": f"assumed {a.wall_mm:g} mm wall", "none": "none"}[r.scale_method]
    print(f"walls={len(plan.walls)} doors={n['door']} windows={n['window']} openings={n['opening']} "
          f"wall_t={plan.wall_thickness_px:.1f}px mm/px={plan.mm_per_px and round(plan.mm_per_px, 4)} ({sc_txt})"
          + (f" photo={','.join(r.rectify.applied)}" if r.rectify.applied else ""))
    for note in r.notes:
        print("  -", note)


if __name__ == "__main__":
    main()
