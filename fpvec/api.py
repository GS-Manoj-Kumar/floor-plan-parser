"""Optional HTTP API so a JS/web drawing app can call the Python vectorizer.

Run:  pip install fpvec[api]  &&  uvicorn fpvec.api:app --port 8000
POST /vectorize  (multipart: file=<image>, mm_per_px=<float, optional>)
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    from fastapi import FastAPI, File, Form, UploadFile
except ImportError as e:  # pragma: no cover
    raise ImportError("install the API extra: pip install fpvec[api]") from e

from .pipeline import Params, vectorize

app = FastAPI(title="fpvec", version="0.1.0")


@app.post("/vectorize")
async def vectorize_endpoint(file: UploadFile = File(...), mm_per_px: Optional[float] = Form(None),
                             wall_thickness_px: Optional[float] = Form(None)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return {"error": "could not decode image"}
    plan = vectorize(img, mm_per_px, Params(wall_thickness_px=wall_thickness_px))
    return plan.to_dict()
