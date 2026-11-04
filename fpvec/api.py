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
  -> {"plan", "floorplan" (editor document), "scale" (method + every dimension
      read), "rectify", "input", "notes", "image" (traced image), "dxf", "svg"}
POST /api/preview     file, page, dpi, photo -> the image that will be traced
"""
from __future__ import annotations

import base64
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
from .pipeline import Params, vectorize
from .process import process
from .rectify import correct_photo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="fpvec", version="0.3.0")


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
                        page: int = Form(1), dpi: float = Form(200), units: str = Form("auto")):
    raw = await _read(file)
    t0 = time.time()
    try:
        r = process(raw, mm_per_px=mm_per_px or None, auto_scale=bool(auto_scale), photo=photo,
                    page=max(0, page - 1), dpi=dpi, units=units, wall_mm=wall_mm or None,
                    params=Params(wall_thickness_px=wall_px or None, detect_thin_walls=bool(thin_walls)))
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
    return {"ocr": ocr_available(), "pdf": pdf_available()}


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
