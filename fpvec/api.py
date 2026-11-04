"""HTTP API + demo page.

Run:  pip install -e ".[api]"  &&  fpvec-demo          (or: uvicorn fpvec.api:app --port 8000)
Open: http://localhost:8000

POST /api/vectorize   multipart: file=<image>, and optionally
                      mm_per_px=<float>   scale of the image, or
                      wall_mm=<float>     assume the main wall is this thick (mm)
                      wall_px=<float>     override the detected wall thickness (px)
                      thin_walls=<0|1>    detect double-line / grey partitions (default 1)
  -> {"plan": fpvec JSON, "floorplan": Floor Plan editor document,
      "dxf": text, "svg": text, "seconds": float}
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

from .export import to_dxf, to_svg
from .floorplan_doc import to_floorplan_doc
from .pipeline import Params, vectorize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples")
MAX_BYTES = 25 * 1024 * 1024

app = FastAPI(title="fpvec", version="0.2.0")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "web", "index.html"), encoding="utf-8") as f:
        return f.read()


@app.get("/examples/{name}")
def example(name: str):
    path = os.path.join(EXAMPLES, os.path.basename(name))
    if not name.lower().endswith((".png", ".jpg", ".jpeg")) or not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/examples")
def examples():
    if not os.path.isdir(EXAMPLES):
        return []
    return sorted(f for f in os.listdir(EXAMPLES) if f.lower().endswith((".png", ".jpg", ".jpeg")))


@app.post("/api/vectorize")
async def api_vectorize(file: UploadFile = File(...), mm_per_px: Optional[float] = Form(None),
                        wall_mm: Optional[float] = Form(None), wall_px: Optional[float] = Form(None),
                        thin_walls: int = Form(1)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        raise HTTPException(413, "image larger than 25 MB")
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(400, "could not decode the image (PNG / JPG / BMP / TIFF)")
    t0 = time.time()
    plan = vectorize(img, mm_per_px or None, Params(wall_thickness_px=wall_px or None,
                                                     detect_thin_walls=bool(thin_walls)))
    if not mm_per_px and wall_mm and plan.wall_thickness_px:
        plan.mm_per_px = wall_mm / plan.wall_thickness_px
    secs = time.time() - t0

    mt = file.content_type or "image/png"
    doc = to_floorplan_doc(plan, image_data_url=f"data:{mt};base64," + base64.b64encode(raw).decode("ascii"))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "plan.dxf")
        to_dxf(plan, p)
        with open(p, encoding="ascii") as f:
            dxf = f.read()
    return {"plan": plan.to_dict(), "floorplan": doc, "dxf": dxf, "svg": to_svg(plan, units="mm" if plan.mm_per_px else "px"),
            "seconds": round(secs, 2)}


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
