"""HTTP API + demo page.

Run:  pip install -e ".[api]"  &&  fpvec-demo          (or: uvicorn fpvec.api:app --port 8000)
Open: http://localhost:8000

POST /api/vectorize   multipart: file=<image or PDF>, and optionally
                      mm_per_px=<float>   scale of the (traced) image, or
                      auto_scale=<0|1>    read the scale from the dimension text (OCR / PDF text)
                      wall_mm=<float>     fallback: assume the main wall is this thick (mm)
                      photo=off|auto|on   perspective / lighting / skew correction
                      page=<n>, dpi=<f>   PDF page (1-based) and render resolution
                      units=auto|mm|cm|m|ftin   how plain dimension numbers are read
                      wall_px, thin_walls
                      prep=<JSON>         preparation recipe applied after photo correction
                                          (black & white, line darkness, areas removed / blurred;
                                          see fpvec/prepare.py)
  -> {"plan", "floorplan" (editor document), "scale" (method + every dimension
      read), "rectify", "input", "notes", "image" (traced image), "dxf", "svg"}
POST /api/preview     file, page, dpi, photo -> the image that will be traced
"""
from __future__ import annotations

import base64
import json
import uuid
from collections import OrderedDict
import os
import tempfile
import time
from typing import Optional

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse
except ImportError as e:  # pragma: no cover
    raise ImportError("install the API extra: pip install fpvec[api]") from e

from .autoscale import ocr_available
from .export import to_dxf, to_svg
from .floorplan_doc import to_floorplan_doc
from .inputs import load_input, pdf_available
from .models import plan_from_dict
from .prepare import apply_prep, is_identity
from .presets import DEFAULT_PRESET, PRESETS, TUNING, tuning_params
from .pipeline import Params, vectorize
from .process import process
from .rectify import correct_photo
from .tidy import loose_ends, tidy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="fpvec", version="0.5.0")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "web", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/examples/{name}")
def example(name: str):
    path = os.path.join(EXAMPLES, os.path.basename(name))
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")) or not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/examples")
def examples():
    if not os.path.isdir(EXAMPLES):
        return []
    return sorted(f for f in os.listdir(EXAMPLES) if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf")))


def _plain(o):
    """numpy scalars / tuples -> plain JSON types (recursively)."""
    if isinstance(o, dict):
        return {k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def _img_url(img: np.ndarray) -> str:
    """PNG for drawings (sharp lines), JPEG if that is much smaller."""
    ok, png = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    data, mt = png.tobytes(), "image/png"
    if len(data) > 2_500_000:
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        data, mt = jpg.tobytes(), "image/jpeg"
    return f"data:{mt};base64," + base64.b64encode(data).decode("ascii")


async def _read(file: UploadFile) -> bytes:
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "file larger than 25 MB")
    return raw


@app.post("/api/preview")
async def api_preview(file: UploadFile = File(...), page: int = Form(1), dpi: float = Form(200),
                      photo: str = Form("auto")):
    """The image that will be traced (PDF page rendered, photo corrected), so
    the page can show it and the user can pick scale points on it."""
    raw = await _read(file)
    try:
        li = load_input(raw, page=max(0, page - 1), dpi=dpi)
    except (ValueError, ImportError) as e:
        raise HTTPException(400, str(e))
    img, info = correct_photo(li.image, photo)
    return _plain({"image": _img_url(img), "kind": li.kind, "page": li.page + 1, "pages": li.pages,
                   "size_px": [int(img.shape[1]), int(img.shape[0])], "rectify": info.to_dict()})


@app.post("/api/vectorize")
async def api_vectorize(file: UploadFile = File(...), mm_per_px: Optional[float] = Form(None),
                        wall_mm: Optional[float] = Form(None), wall_px: Optional[float] = Form(None),
                        thin_walls: int = Form(1), auto_scale: int = Form(0), photo: str = Form("off"),
                        page: int = Form(1), dpi: float = Form(200), units: str = Form("auto"),
                        prep: Optional[str] = Form(None)):
    raw = await _read(file)
    t0 = time.time()
    try:
        recipe = json.loads(prep) if prep else None
    except ValueError:
        raise HTTPException(400, "prep must be JSON (see fpvec/prepare.py)")
    try:
        r = process(raw, mm_per_px=mm_per_px or None, auto_scale=bool(auto_scale), photo=photo,
                    page=max(0, page - 1), dpi=dpi, units=units, wall_mm=wall_mm or None,
                    params=Params(wall_thickness_px=wall_px or None, detect_thin_walls=bool(thin_walls)),
                    prep=recipe)
    except (ValueError, ImportError) as e:
        raise HTTPException(400, str(e))
    secs = time.time() - t0
    plan = r.plan
    url = _img_url(r.image)
    doc = to_floorplan_doc(plan, image_data_url=url)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "plan.dxf")
        to_dxf(plan, p)
        with open(p, encoding="ascii") as f:
            dxf = f.read()
    out = r.to_dict()
    out.update({"floorplan": doc, "dxf": dxf, "svg": to_svg(plan, units="mm" if plan.mm_per_px else "px"),
                "image": url, "seconds": round(secs, 2)})
    return _plain(out)


@app.get("/api/capabilities")
def capabilities():
    return {"ocr": ocr_available(), "pdf": pdf_available(), "editor": _editor_path() is not None}


# ------------------------------------------------------------------ v0.5: session API
# load once (render PDF page, correct the photo), then preview the
# preparation, detect and export by token without re-uploading.
_SESS: "OrderedDict[str, dict]" = OrderedDict()
_SESS_MAX = 12


def _sess(token: str) -> dict:
    s = _SESS.get(token)
    if s is None:
        raise HTTPException(404, "image expired - open it again")
    _SESS.move_to_end(token)
    return s


def _json_form(v: Optional[str], what: str):
    if not v:
        return None
    try:
        return json.loads(v)
    except ValueError:
        raise HTTPException(400, f"{what} must be JSON")


@app.post("/api/load")
async def api_load(file: UploadFile = File(...), page: int = Form(1), dpi: float = Form(200),
                   photo: str = Form("auto")):
    raw = await _read(file)
    try:
        li = load_input(raw, page=max(0, page - 1), dpi=dpi)
    except (ValueError, ImportError) as e:
        raise HTTPException(400, str(e))
    img, info = correct_photo(li.image, photo)
    moved = any(a in info.applied for a in ("perspective", "rotate", "shear"))
    token = uuid.uuid4().hex
    _SESS[token] = dict(image=img, text=None if moved else li.text, kind=li.kind, page=li.page, pages=li.pages,
                        rectify=info, name=file.filename or "plan")
    while len(_SESS) > _SESS_MAX:
        _SESS.popitem(last=False)
    return _plain({"token": token, "image": _img_url(img), "kind": li.kind, "page": li.page + 1, "pages": li.pages,
                   "size_px": [int(img.shape[1]), int(img.shape[0])], "rectify": info.to_dict()})


@app.post("/api/prepare")
async def api_prepare(token: str = Form(...), prep: Optional[str] = Form(None)):
    s = _sess(token)
    recipe = _json_form(prep, "prep")
    img = apply_prep(s["image"], recipe) if recipe and not is_identity(recipe) else s["image"]
    return {"image": _img_url(img)}


@app.post("/api/detect")
async def api_detect(token: str = Form(...), prep: Optional[str] = Form(None), tuning: Optional[str] = Form(None),
                     hints: Optional[str] = Form(None), mm_per_px: Optional[float] = Form(None),
                     auto_scale: int = Form(1), wall_mm: Optional[float] = Form(200), units: str = Form("auto")):
    s = _sess(token)
    recipe = _json_form(prep, "prep")
    t = tuning_params(_json_form(tuning, "tuning") or {})
    marks = _json_form(hints, "hints") or None
    t0 = time.time()
    params = Params(**t)
    params.hints = marks
    r = process(s["image"], mm_per_px=mm_per_px or None, auto_scale=bool(auto_scale), photo="off", units=units,
                wall_mm=wall_mm or None, params=params, prep=recipe, text=s["text"], rectify=s["rectify"])
    s["base"] = r.base
    out = r.to_dict()
    out["input"].update({"kind": s["kind"], "page": s["page"] + 1, "pages": s["pages"]})
    out.update(_exports(r.plan, s))
    out.update({"image": _img_url(r.image), "base": _img_url(r.base) if r.base is not r.image else None,
                "seconds": round(time.time() - t0, 2)})
    return _plain(out)


def _exports(plan, s: Optional[dict]) -> dict:
    under = s.get("base") if s else None
    doc = to_floorplan_doc(plan, image_data_url=_img_url(under) if under is not None else None)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "plan.dxf")
        to_dxf(plan, p)
        with open(p, encoding="ascii") as f:
            dxf = f.read()
    return {"floorplan": doc, "dxf": dxf, "svg": to_svg(plan, units="mm" if plan.mm_per_px else "px")}


@app.post("/api/export")
async def api_export(plan: str = Form(...), token: Optional[str] = Form(None)):
    """Export a plan the user edited in the demo (fpvec JSON, px) to the
    Floor Plan editor document, DXF and SVG. Openings are re-hosted on the
    walls they now sit on."""
    d = _json_form(plan, "plan")
    if not isinstance(d, dict) or "walls" not in d:
        raise HTTPException(400, "plan must be fpvec JSON with walls and openings")
    fp = plan_from_dict(d)
    s = _SESS.get(token) if token else None
    out = _exports(fp, s)
    out["plan"] = fp.to_dict()
    return _plain(out)


@app.post("/api/tidy")
async def api_tidy(plan: str = Form(...), ortho: bool = Form(True), snap: bool = Form(True),
                   snap_dist: float = Form(3.0), snap_unit: str = Form("t")):
    """Tidy an (edited) plan: ortho walls and join wall ends within
    ``snap_dist`` (unit "t" = wall thicknesses, "pct" = % of the plan size,
    "mm")."""
    d = _json_form(plan, "plan")
    if not isinstance(d, dict) or "walls" not in d:
        raise HTTPException(400, "plan must be fpvec JSON with walls and openings")
    if snap_unit not in ("t", "pct", "mm"):
        raise HTTPException(400, "snap_unit must be t, pct or mm")
    fp = plan_from_dict(d)
    out = tidy(fp, ortho=ortho, snap=snap, snap_dist=max(0.0, snap_dist), snap_unit=snap_unit)
    return _plain({"plan": out.to_dict(), "loose_before": loose_ends(fp), "loose_after": loose_ends(out)})


# ------------------------------------------------------------------ projects
# Floor Plan drawings saved from the embedded editor, kept on this machine.
def _proj_dir() -> str:
    d = os.environ.get("FPVEC_PROJECTS") or os.path.join(ROOT, "projects")
    os.makedirs(d, exist_ok=True)
    return d


def _proj_path(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in " ._-()").strip().rstrip(".")
    if not safe:
        raise HTTPException(400, "give the drawing a name")
    if not safe.lower().endswith(".json"):
        safe += ".json"
    return os.path.join(_proj_dir(), safe)


@app.get("/api/projects")
def projects():
    d = _proj_dir()
    out = []
    for f in sorted(os.listdir(d)):
        if f.lower().endswith(".json"):
            st = os.stat(os.path.join(d, f))
            out.append({"name": f[:-5], "modified": int(st.st_mtime), "bytes": st.st_size})
    out.sort(key=lambda x: -x["modified"])
    return out


@app.get("/api/projects/{name}")
def project_get(name: str):
    p = _proj_path(name)
    if not os.path.isfile(p):
        raise HTTPException(404, "no such drawing")
    return FileResponse(p, media_type="application/json")


@app.post("/api/projects")
async def project_save(name: str = Form(...), doc: str = Form(...)):
    d = _json_form(doc, "doc")
    if not isinstance(d, dict) or d.get("app") != "floorplan":
        raise HTTPException(400, "not a Floor Plan drawing")
    p = _proj_path(name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)
    return {"saved": os.path.basename(p)[:-5], "path": p}


@app.delete("/api/projects/{name}")
def project_delete(name: str):
    p = _proj_path(name)
    if os.path.isfile(p):
        os.remove(p)
    return {"deleted": name}


@app.get("/api/presets")
def api_presets():
    return {"presets": PRESETS, "default": DEFAULT_PRESET, "tuning": TUNING}


def _editor_path() -> Optional[str]:
    for p in (os.environ.get("FPVEC_EDITOR"), os.path.join(ROOT, "editor", "floor_plan.html")):
        if p and os.path.isfile(p):
            return p
    return None


@app.get("/editor", response_class=HTMLResponse)
def editor():
    """Serves your Floor Plan editor (floor_plan.html) on the same address as
    the demo, so "Promote" can hand it the traced plan."""
    p = _editor_path()
    if not p:
        raise HTTPException(404, "Floor Plan editor not found: copy floor_plan.html to editor/floor_plan.html "
                                 "(or set FPVEC_EDITOR to its path) and restart")
    with open(p, encoding="utf-8") as f:
        return f.read()


# backwards compatible endpoint (v0.1)
@app.post("/vectorize")
async def vectorize_endpoint(file: UploadFile = File(...), mm_per_px: Optional[float] = Form(None),
                             wall_thickness_px: Optional[float] = Form(None)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "could not decode image")
    return vectorize(img, mm_per_px, Params(wall_thickness_px=wall_thickness_px)).to_dict()


def main():  # pragma: no cover
    import argparse
    import uvicorn
    ap = argparse.ArgumentParser(description="fpvec demo server")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":  # pragma: no cover
    main()
