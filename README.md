# fpvec: floor plan raster → vector

Turn a floor plan **image** into **vector walls, doors and windows** that a drawing/CAD tool can use directly.

* Walls come out as **centrelines with thickness**, flagged **exterior / interior**, with lengths in px and mm.
* Doors come out with **hinge point, width, swing side**, and the wall that hosts them.
* Windows and plain openings come out with **start / end / centre / width** and **offset along the host wall**.
* Export formats: **JSON**, **SVG**, **DXF** (layers WALLS / DOORS / WINDOWS / OPENINGS), a debug **overlay PNG**, and a
  **`floor_plan.json` that the Floor Plan editor (`floor_plan.html` / SketchUp dialog) opens directly**: walls, doors and
  windows as real editor objects, with the source image as a 50 % underlay.
* A **demo page** (upload → scale → vectorize → inspect → download), runnable in one click as a **GitHub Codespace**.

Pure Python + OpenCV + NumPy. No GPU and no model weights.

| Input | Output overlay |
|---|---|
| <img src="examples/synth_solid.png" width="380"> | <img src="examples/output/synth_solid_overlay.png" width="380"> |
| <img src="examples/sample_plan.png" width="250"> | <img src="examples/output/sample_plan_overlay.png" width="250"> |

Red = wall centrelines (dark outline = exterior), yellow = doors with swing arc, blue = windows, green = openings. Numbers are wall lengths in mm.

## Try it in a Codespace

**Code → Codespaces → Create codespace on main.** The container installs everything and starts the demo server.
Port 8000 opens in the browser (if it doesn't, open the **Ports** tab and click the globe icon next to 8000).

<img src="docs/demo.png" width="760">

The demo page:

1. **Open image…** (or drag a file in, or pick one of the **Examples**).
2. **Scale.** Choose one:
   * *Assume wall thickness*: e.g. 200 mm for the main wall.
   * *Two points + distance*: click two points, then type the real distance (`3600`, `3.6m` or `12'6"`).
   * *mm per pixel*.
3. **Vectorize.** Walls, doors (with swing) and windows are drawn over the image. Click one, or its table row, to highlight it.
4. **Download** `floor_plan.json` and open it with the editor's **Open** button. The image comes along as an underlay, so you can check and fix the trace there.

Locally, the same demo runs with:

```bash
pip install -e ".[api]"
fpvec-demo            # http://localhost:8000
```

## Install

```bash
pip install -r requirements.txt        # numpy + opencv-python-headless
pip install -e .                       # optional: installs the `fpvec` command
```

## Use

### Command line

```bash
# you know the scale (mm per image pixel), e.g. from your app's "scale" step
fpvec plan.png --mm-per-px 12.5 -o out/

# or: two points on the image and the real distance between them (mm)
fpvec plan.png --ref 120 340 980 340 8400 -o out/

# or: no scale, just assume the main wall is 200 mm thick
fpvec plan.png --wall-mm 200 -o out/
```

This writes `out/plan.json`, `plan.svg`, `plan.dxf`, `plan_overlay.png` and **`plan_floor_plan.json`** (the editor document).

### Python

```python
from fpvec import vectorize, mm_per_px_from_reference, to_json, to_dxf

scale = mm_per_px_from_reference((120, 340), (980, 340), 8400)   # user's scale step
plan = vectorize("plan.png", mm_per_px=scale)

for w in plan.walls:
    print(w.id, w.start, w.end, w.thickness, w.exterior)
for o in plan.openings:
    print(o.kind, o.start, o.end, o.width, o.hinge, o.swing_side, o.host_wall_ids)

to_json(plan, "plan.json")
to_dxf(plan, "plan.dxf")

from fpvec import save_floorplan_doc            # Floor Plan editor document
save_floorplan_doc(plan, "plan_floor_plan.json", image_path="plan.png")
```

### HTTP (for a web/JS front end)

```bash
pip install -e ".[api]"
fpvec-demo --port 8000
curl -F file=@plan.png -F mm_per_px=12.5 http://localhost:8000/api/vectorize
```

`/api/vectorize` takes `file` plus one of `mm_per_px` / `wall_mm`, and optionally `wall_px` and `thin_walls=0|1`.
It returns `{plan, floorplan, dxf, svg, seconds}`: `floorplan` is the editor document, ready for the
editor's `loadDoc(fixDoc(...))`.

## Floor Plan editor document

`to_floorplan_doc(plan)` / `--formats floorplan` writes the editor's own format (`app: "floorplan"`, mm, **y up**):

| fpvec | editor entity |
|---|---|
| wall (centreline, thickness) | `wall {a, b, bg: 0, thk, h: 3048, just: "C", top: "next"}` on layer `LW` |
| door | `opening {wall, kind: "door", s0, w, h: 2133.6, sill: 0, style: "single"\|"double", hinge: "L"\|"R", side: ±1}` |
| window | `opening {kind: "window", style: "std", h: 1219.2, sill: 914.4}` |
| plain opening | `opening {kind: "cut", style: "none"}` |
| source image | `image {src, p: {0,0}, w, h, op: 0.5}` on layer `LI`, so the trace sits exactly on it |

* `s0` is measured along the wall's centreline from `a`.
* `hinge`/`side` follow the editor's convention: local x along the wall from `s0`, swing towards `vperp(u)` when `side = 1`.
* An opening that sits just past a wall's end (a door next to a corner) makes that wall run on through it.
* An opening with no wall around it gets its own short wall.

Heights and sills are the editor defaults. Change them in the editor afterwards, or pass `wall_h=` to `to_floorplan_doc`.
The export was checked by loading it in `floor_plan.html` (headless Chromium): every wall builds geometry, every
opening resolves on its wall, and there are no errors.

## Output (JSON)

All coordinates are in **original image pixels** (origin top-left, y down). The `_mm` fields are filled in when a scale is given.

```jsonc
{
  "image_size_px": [1240, 1754],
  "mm_per_px": 12.5,
  "wall_thickness_px": 19.2,          // detected main wall thickness
  "walls": [{
    "id": 0, "orientation": "h", "exterior": true,
    "start_px": [102.0, 88.5], "end_px": [1130.0, 88.5],
    "length_px": 1028.0, "thickness_px": 19.6,
    "start_mm": [1275.0, 1106.3], "end_mm": [14125.0, 1106.3],
    "length_mm": 12850.0, "thickness_mm": 245.0
  }],
  "openings": [{
    "id": 3, "type": "door",             // "door" | "window" | "opening"
    "orientation": "h", "host_wall_ids": [0],
    "start_px": [...], "end_px": [...], "center_px": [...], "width_px": 72.0,
    "width_mm": 900.0, "offset_on_host_mm": 3100.0,
    "hinge_px": [...], "swing_side": "down", "double": false,
    "confidence": 0.93
  }]
}
```

Walls are **continuous runs**: a wall with a door in it is one wall, and the door references it through `host_wall_ids` and `offset_on_host_*`. Set `Params(merge_across_openings=False)` to get the wall pieces between openings instead.

## How it works

```
image ─► normalise (long side 1600 px) ─► ink mask ─► stroke widths ─► walls ─► openings ─► doors ─► tidy ─► export
```

1. **Stroke widths** (`preprocess.py`). The image is opened with long 1-px kernels so only long straight strokes remain. Their thickness is histogrammed (weighted by ink area): thin drawing lines form one peak, walls the dominant thicker peak. This gives the **wall thickness T** automatically, so every other threshold is a multiple of T and the method is resolution-independent.
2. **Solid walls** (`walls.py`). A square morphological opening (≈0.4 T) deletes everything thinner than a wall: text, furniture, arcs and glazing lines. The remaining mask is split into horizontal and vertical bars. Each bar becomes a centreline segment with measured thickness. Furniture blobs glued to walls are rejected by *relative* row coverage.
3. **Grey & thin-line walls** (`thinlines.py`, `pipeline.py`). Sales plans often draw only structural walls black, and partitions
   in **mid/light grey**: two grey lines, often with a light-grey fill. Those are invisible to a global ink threshold, so:
   * thin lines are found by **local contrast** (a black-hat filter: darker than their surroundings, whatever their grey level);
   * **grey bands** of wall width (grey outline plus fill) are found with a wider black-hat followed by an opening. Because furniture
     outlines look the same, a band only counts as a wall if it touches the wall network at one end and, at the other end, touches
     it too or faces a wall across a door-sized gap;
   * a thin double line that continues a thicker solid wall on the same axis is **glazing** (a window), not a partition;
   * of several parallel candidates side by side (a bookshelf against a partition), only one is kept.

   Classic line pairs are handled as before: Many plans draw partitions as two thin parallel lines and windows as 3–4 lines. Pairs of long parallel thin lines about one wall thickness apart are found. A pair with empty space between is a **hollow wall**, and one with line(s) between is a **window band**. Hollow walls are kept only if they connect to the wall network, which removes beds and wardrobes.
4. **Openings** (`openings.py`). From each wall end, walk along the wall axis until wall ink is hit again. Each gap is classified by its content: parallel glazing lines → **window**, nothing → **opening**. Window glazing is often light grey, so a lenient threshold is used for this test.
5. **Doors** (`doors.py`). A door must be anchored: either its arc sits in a detected wall gap, or its closed leaf lies on a wall line (a wall on the door's axis ends at the hinge or latch, or there is wall on both sides). This removes beds, chairs and toilets. Door swing arcs are found two ways: circle fits of curved strokes, and a search from every wall end over radius × swing direction. Candidates are refined by local search. Each is scored by arc coverage, a blank swept area (rejects toilets and chairs), and wall ink at both jambs. An arc labels the gap it sits in as a door, or creates a door when no gap was found (e.g. an entrance between two wall stubs). Result: hinge, width (radius), swing side.
6. **Tidy** (`pipeline.py`). Collinear pieces are merged, wall ends snap to perpendicular centrelines (clean L/T junctions), and small detached clusters outside the main plan (key plans, legends) are dropped. Walls are flagged exterior by testing both sides against the filled building footprint.

## Accuracy

`fpvec/synth.py` renders plans with known ground truth (solid or hollow interior walls, swing doors, 4-line windows, furniture, text, optional blur + noise). From `pytest tests/`:

| Plan | Windows | Doors | Width error |
|---|---|---|---|
| Solid walls (clean, and with blur + noise) | 5 / 5 | 5 / 5 | < 15 mm |
| Hollow (double-line) interior walls | 5 / 5 (+1 false 0.5 m window at a junction) | 4 / 5 | < 150 mm |
| Grey partitions (grey outline + light fill) | 5 / 5 (+3 false short windows) | 4 / 5 | all 4 partitions found; 120 mm walls read as 138 mm |
| Exterior wall length | 4 / 4 | | < 5 mm; thickness 240 mm ± 10 |

On the bundled `examples/sample_plan.png`, v0.2 finds the grey partitions that v0.1 missed: bedroom/bath, kitchen,
study, the master bath and wardrobe, and the inner face of the bay-window wall. It finds 46 walls, against 25 in v0.1, plus
the entrance, kitchen, bathroom, master-bedroom and master-bath doors. Some furniture next to walls (bedside tables, a bed
edge) still comes through as short walls or windows. That image is a 412 px wide screenshot (≈27 mm per pixel), where a door arc is only ~25 px across. Expect better results from source images **≥ 1500 px** on the long side.

## Limitations and tuning

* Walls are assumed **axis-aligned** (horizontal/vertical), which covers almost all residential plans. Diagonal and curved walls are not extracted yet.
* **Sliding doors** are drawn like windows and come out as `window`. **Bay windows** (飘窗) come out as a window plus short walls.
* Heavily stylised marketing plans (coloured floors, 3D furniture, photos) may need `Params(ink_threshold=...)`, or a learned wall mask (see `docs/reference_repos.md`).
* Main knobs (`fpvec.Params`): `wall_thickness_px` (override the estimate), `detect_thin_walls`, `min_opening_mm` / `max_opening_mm`, `door_score`, `merge_across_openings`.

Debug: `vectorize(img, debug=d)` fills `d` with the intermediate masks (`ink`, `solid`, `thin`, `barrier`, `footprint`) and the working scale.

## Layout

```
fpvec/
  preprocess.py  load, binarise, stroke-width estimation
  walls.py       wall mask, bar → segment extraction, merging, junction snapping
  thinlines.py   hollow walls and window bands from parallel thin lines
  openings.py    gap search along walls + window/opening classification
  doors.py       swing-arc door detection (hinge, width, swing)
  pipeline.py    vectorize() + Params
  models.py      Wall / Opening / FloorPlan + JSON dict
  export.py      JSON, SVG, DXF, overlay PNG
  scale.py       mm-per-px from a reference distance or wall thickness
  floorplan_doc.py  export to the Floor Plan editor document (floor_plan.json)
  web/index.html    demo page (served by api.py)
  synth.py       synthetic plans with ground truth
  cli.py         command line
  api.py         FastAPI: demo page + /api/vectorize (`fpvec-demo`)
.devcontainer/   Codespace: installs, starts the demo on port 8000
docs/reference_repos.md   review of the six reference repos
```
