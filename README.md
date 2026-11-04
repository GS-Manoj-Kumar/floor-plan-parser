# fpvec: floor plan raster → vector

Turn a floor plan **image** into **vector walls, doors and windows** that a drawing/CAD tool can use directly.

* Walls come out as **centrelines with thickness**, flagged **exterior / interior**, with lengths in px and mm.
* Doors come out with **hinge point, width, swing side**, and the wall that hosts them.
* Windows and plain openings come out with **start / end / centre / width** and **offset along the host wall**.
* Export formats: **JSON**, **SVG**, **DXF** (layers WALLS / DOORS / WINDOWS / OPENINGS), and a debug **overlay PNG**.

Pure Python + OpenCV + NumPy. No GPU and no model weights.

| Input | Output overlay |
|---|---|
| <img src="examples/synth_solid.png" width="380"> | <img src="examples/output/synth_solid_overlay.png" width="380"> |
| <img src="examples/sample_plan.png" width="250"> | <img src="examples/output/sample_plan_overlay.png" width="250"> |

Red = wall centrelines (dark outline = exterior), yellow = doors with swing arc, blue = windows, green = openings. Numbers are wall lengths in mm.

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

This writes `out/plan.json`, `plan.svg`, `plan.dxf` and `plan_overlay.png`.

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
```

### HTTP (for a web/JS front end)

```bash
pip install -e ".[api]"
uvicorn fpvec.api:app --port 8000
curl -F file=@plan.png -F mm_per_px=12.5 http://localhost:8000/vectorize
```

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
3. **Thin-line walls & windows** (`thinlines.py`). Many plans draw partitions as two thin parallel lines and windows as 3–4 lines. Pairs of long parallel thin lines about one wall thickness apart are found. A pair with empty space between is a **hollow wall**, and one with line(s) between is a **window band**. Hollow walls are kept only if they connect to the wall network, which removes beds and wardrobes.
4. **Openings** (`openings.py`). From each wall end, walk along the wall axis until wall ink is hit again. Each gap is classified by its content: parallel glazing lines → **window**, nothing → **opening**. Window glazing is often light grey, so a lenient threshold is used for this test.
5. **Doors** (`doors.py`). Door swing arcs are found two ways: circle fits of curved strokes, and a search from every wall end over radius × swing direction. Candidates are refined by local search. Each is scored by arc coverage, a blank swept area (rejects toilets and chairs), and wall ink at both jambs. An arc labels the gap it sits in as a door, or creates a door when no gap was found (e.g. an entrance between two wall stubs). Result: hinge, width (radius), swing side.
6. **Tidy** (`pipeline.py`). Collinear pieces are merged, wall ends snap to perpendicular centrelines (clean L/T junctions), and small detached clusters outside the main plan (key plans, legends) are dropped. Walls are flagged exterior by testing both sides against the filled building footprint.

## Accuracy

`fpvec/synth.py` renders plans with known ground truth (solid or hollow interior walls, swing doors, 4-line windows, furniture, text, optional blur + noise). From `pytest tests/`:

| Plan | Windows | Doors | Width error |
|---|---|---|---|
| Solid walls (clean, and with blur + noise) | 5 / 5 | 5 / 5 | < 15 mm |
| Hollow (double-line) interior walls | 5 / 5 (+1 false 0.5 m window at a junction) | 4 / 5 | < 150 mm |
| Exterior wall length | 4 / 4 | | < 5 mm; thickness 240 mm ± 10 |

On the bundled `examples/sample_plan.png`, the main walls, the big living-room window and most doors are found. That image is a 412 px wide screenshot (≈27 mm per pixel), where a door arc is only ~25 px across. Expect better results from source images **≥ 1500 px** on the long side.

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
  synth.py       synthetic plans with ground truth
  cli.py, api.py
docs/reference_repos.md   review of the six reference repos
```
