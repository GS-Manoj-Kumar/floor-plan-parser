"""Build the sample set and run fpvec on it.

    python samples/run_samples.py            # 15 plans x (clean + distorted) -> samples/
Writes samples/input/, samples/output/, samples/results.json, samples/RESULTS.md
"""
import json
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from fpvec import draw_overlay, process, to_floorplan_doc  # noqa: E402
from fpvec.evaluate import score  # noqa: E402
from fpvec.synth import make_photo, make_scan  # noqa: E402
from fpvec.synth_color import make_color_plan  # noqa: E402

N = int(os.environ.get("N_SAMPLES", 15))
IN, OUT = os.path.join(HERE, "input"), os.path.join(HERE, "output")


def main():
    os.makedirs(IN, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for i in range(N):
        img, gt = make_color_plan(i)
        name = f"{i + 1:02d}_{gt['style']}"
        # distortion: photos for even ids, crooked scans for odd ones
        if i % 2 == 0:
            dist = make_photo(img, seed=i, tilt=0.06 + 0.02 * (i % 5), rot_deg=[-14, 9, -5, 18, 6][i % 5])
            dkind = "photo"
        else:
            dist = make_scan(img, seed=i, rot_deg=[2.5, -3.5, 1.5, -2.0, 4.0][i % 5])
            dkind = "scan"
        cv2.imwrite(os.path.join(IN, name + "_clean.png"), img)
        cv2.imwrite(os.path.join(IN, name + f"_{dkind}.jpg"), dist, [cv2.IMWRITE_JPEG_QUALITY, 88])
        g = {k: v for k, v in gt.items()}
        with open(os.path.join(IN, name + "_truth.json"), "w") as f:
            json.dump(g, f, default=float)
        for variant, im in (("clean", img), (dkind, dist)):
            t = time.time()
            r = process(im, auto_scale=True, photo="auto", wall_mm=200)
            secs = time.time() - t
            sc = score(r.plan, gt)
            stem = f"{name}_{variant}"
            ov = draw_overlay(r.plan, r.image, None, scale=1.0)
            cv2.imwrite(os.path.join(OUT, stem + "_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
            with open(os.path.join(OUT, stem + "_floor_plan.json"), "w") as f:
                json.dump(to_floorplan_doc(r.plan), f)
            row = dict(sample=name, variant=variant, style=gt["style"], dims=gt["dims"], units=gt["units"],
                       scale_method=r.scale_method, votes=r.scale.votes, rectify=r.rectify.applied,
                       true_mm_per_px=round(gt["mm_per_px"], 4), seconds=round(secs, 1), **sc)
            rows.append(row)
            print(f"{stem:32s} {r.scale_method:10s} size {sc['size_err_pct']}%  shape R{sc['shape_wall_recall']} P{sc['shape_wall_precision']}  walls R{sc['wall_recall']} "
                  f"P{sc['wall_precision']}  doors {sc['doors_found']}/{sc['doors_true']}(+{sc['doors_false']})  "
                  f"win {sc['windows_found']}/{sc['windows_true']}(+{sc['windows_false']})  {secs:.1f}s", flush=True)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    write_md(rows)


def write_md(rows):
    def agg(rs):
        n = len(rs)
        dt = sum(r["doors_true"] for r in rs)
        wt = sum(r["windows_true"] for r in rs)
        return dict(
            n=n,
            size=sorted(r["size_err_pct"] for r in rs if r["size_err_pct"] is not None),
            wr=sum(r["wall_recall"] for r in rs) / n, wp=sum(r["wall_precision"] for r in rs) / n,
            swr=sum(r["shape_wall_recall"] for r in rs) / n, swp=sum(r["shape_wall_precision"] for r in rs) / n,
            iwr=sum(r["int_wall_recall"] or 0 for r in rs) / n,
            dr=sum(r["doors_found"] for r in rs) / max(1, dt), df=sum(r["doors_false"] for r in rs),
            wnr=sum(r["windows_found"] for r in rs) / max(1, wt), wnf=sum(r["windows_false"] for r in rs),
            dims=sum(r["scale_method"] == "dimensions" for r in rs),
            areas=sum(r["scale_method"] == "room areas" for r in rs),
            auto=sum(r["scale_method"] in ("dimensions", "room areas") for r in rs))
    L = ["# Sample set results", "",
         "15 generated coloured floor plans (5 drawing styles), each as a **clean** image and a **distorted** one",
         "(a phone photo at an angle for odd numbers, a crooked yellowed scan for even numbers). Run with",
         "`process(image, auto_scale=True, photo=\"auto\", wall_mm=200)`, scored by `fpvec/evaluate.py` against the",
         "generator's ground truth. Regenerate with `python samples/run_samples.py`.", "",
         "| Set | Plans | Scale found automatically | Median size error | Wall recall | Wall precision | Walls, shape only (R / P) | Interior wall recall | Doors found | False doors | Windows found | False windows |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label, rs in (("clean", [r for r in rows if r["variant"] == "clean"]),
                      ("distorted", [r for r in rows if r["variant"] != "clean"])):
        if not rs:
            continue
        a = agg(rs)
        med = a["size"][len(a["size"]) // 2] if a["size"] else None
        L.append(f"| {label} | {a['n']} | {a['auto']} / {a['n']} ({a['dims']} from dimensions, {a['areas']} from room areas) | {med} % | "
                 f"{a['wr']:.0%} | {a['wp']:.0%} | {a['swr']:.0%} / {a['swp']:.0%} | {a['iwr']:.0%} | {a['dr']:.0%} | {a['df']} | {a['wnr']:.0%} | {a['wnf']} |")
    L += ["", "* **Wall recall / precision**: share of true wall length matched by a predicted wall within 150 mm, and vice versa,",
          "  in millimetres with the scale fallback included, so a wrong scale lowers it.",
          "* **Walls, shape only**: the same after fitting the plan's overall size, which shows detection quality without the scale.",
          "* **Doors / windows**: centre within 350 mm of a true one of the same kind.",
          "* **Scale**: dimension text first, then room-area labels (\"38.4 m2\") against the rooms the walls enclose,",
          "  then a 200 mm main wall is assumed (the true exterior walls are 200 or 240 mm).", "",
          "| Sample | Variant | Style | Dims | Scale | Size err | Walls R / P | Doors | Windows | Correction |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dims = (r["units"] if r["dims"] else "–")
        L.append(f"| [{r['sample']}](output/{r['sample']}_{r['variant']}_overlay.jpg) | {r['variant']} | {r['style']} | {dims} | "
                 f"{r['scale_method']} ({r['votes']}) | {r['size_err_pct']} % | {r['wall_recall']:.0%} / {r['wall_precision']:.0%} | "
                 f"{r['doors_found']}/{r['doors_true']} (+{r['doors_false']}) | {r['windows_found']}/{r['windows_true']} (+{r['windows_false']}) | "
                 f"{', '.join(r['rectify']) or '–'} |")
    with open(os.path.join(HERE, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
