"""Command line: ``fpvec plan.png --mm-per-px 12.5 -o out/``"""
from __future__ import annotations

import argparse
import os

from .export import draw_overlay, to_dxf, to_json, to_svg
from .floorplan_doc import save_floorplan_doc
from .pipeline import Params, vectorize
from .scale import mm_per_px_from_reference


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Extract walls, doors and windows from a floor plan image.")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    sc = ap.add_mutually_exclusive_group()
    sc.add_argument("--mm-per-px", type=float, help="scale of the image")
    sc.add_argument("--ref", nargs=5, type=float, metavar=("X1", "Y1", "X2", "Y2", "MM"),
                    help="two image points and the real distance between them in mm")
    sc.add_argument("--wall-mm", type=float,
                    help="assume the detected (main) wall thickness is this many mm, e.g. 200")
    ap.add_argument("--wall-px", type=float, help="override wall thickness estimate (px)")
    ap.add_argument("--no-thin-walls", action="store_true", help="only detect solid (filled) walls")
    ap.add_argument("--formats", default="json,svg,dxf,png,floorplan",
                    help="comma list of json, svg, dxf, png, floorplan (the Floor Plan editor document)")
    a = ap.parse_args(argv)

    params = Params(wall_thickness_px=a.wall_px, detect_thin_walls=not a.no_thin_walls)
    mm = a.mm_per_px
    if a.ref:
        mm = mm_per_px_from_reference((a.ref[0], a.ref[1]), (a.ref[2], a.ref[3]), a.ref[4])
    plan = vectorize(a.image, mm, params)
    if a.wall_mm and not mm:
        plan.mm_per_px = a.wall_mm / plan.wall_thickness_px

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.image))[0]
    fm = set(a.formats.split(","))
    if "json" in fm:
        to_json(plan, os.path.join(a.out, stem + ".json"))
    if "svg" in fm:
        to_svg(plan, os.path.join(a.out, stem + ".svg"))
    if "dxf" in fm:
        to_dxf(plan, os.path.join(a.out, stem + ".dxf"))
    if "floorplan" in fm:
        save_floorplan_doc(plan, os.path.join(a.out, stem + "_floor_plan.json"), image_path=a.image)
    if "png" in fm:
        draw_overlay(plan, a.image, os.path.join(a.out, stem + "_overlay.png"))
    n = {k: sum(o.kind == k for o in plan.openings) for k in ("door", "window", "opening")}
    print(f"walls={len(plan.walls)} doors={n['door']} windows={n['window']} openings={n['opening']} "
          f"wall_t={plan.wall_thickness_px:.1f}px mm/px={plan.mm_per_px}")


if __name__ == "__main__":
    main()
