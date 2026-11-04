"""One call from any input to a scaled vector plan.

    from fpvec import process
    r = process("plan.pdf", auto_scale=True, photo="auto")
    r.plan            # FloorPlan (coordinates in r.image pixels, mm filled in)
    r.image           # the image that was traced (after photo correction)
    r.scale           # how the scale was found, and every dimension read
    r.rectify         # what the photo correction did

Steps
1. load: image file / bytes / array, or a PDF page (rendered at ``dpi``;
   its text layer is kept)
2. photo correction (``photo``: "off" | "auto" | "on"): perspective, lighting,
   rotation / shear (see rectify.py)
2b. preparation (``prep``, see prepare.py): black & white, line darkness,
   background whitening, and areas painted white or blurred
3. scale, first that applies:
   * ``mm_per_px`` given                         -> "given"
   * ``auto_scale``: dimension text (PDF text layer, else OCR) measured
     against its dimension lines                 -> "dimensions"
     (different x / y scales -> the image is stretched to square pixels)
   * ``auto_scale``: room-area labels ("38.4 m2") against the rooms the
     walls enclose                               -> "room areas"
   * ``wall_mm``: assume the main wall thickness -> "wall"
4. vectorize
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .autoscale import ScaleResult, TextBox, estimate_scale, ocr_available, run_ocr, scale_from_areas
from .inputs import LoadedInput, Source, load_input
from .models import FloorPlan
from .pipeline import Params, vectorize
from .prepare import apply_prep, describe, is_identity
from .rectify import RectifyInfo, correct_photo, fix_aspect


@dataclass
class ProcessResult:
    plan: FloorPlan
    image: np.ndarray
    scale: ScaleResult
    scale_method: str
    rectify: RectifyInfo
    input_kind: str = "image"
    page: int = 0
    pages: int = 1
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "scale": dict(self.scale.to_dict(), method=self.scale_method),
            "rectify": self.rectify.to_dict(),
            "input": {"kind": self.input_kind, "page": self.page, "pages": self.pages,
                      "size_px": [int(self.image.shape[1]), int(self.image.shape[0])]},
            "notes": self.notes,
        }


def _colourful(img: np.ndarray) -> bool:
    if img.ndim != 3:
        return False
    small = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    return bool((sat > 40).mean() > 0.15)


def _numeric(boxes: List[TextBox]) -> int:
    return sum(1 for b in boxes if any(c.isdigit() for c in b.text))


def process(src: Source, mm_per_px: Optional[float] = None, auto_scale: bool = True, photo: str = "auto",
            page: int = 0, dpi: float = 300.0, units: str = "auto", wall_mm: Optional[float] = 200.0,
            params: Optional[Params] = None, prep: Optional[dict] = None) -> ProcessResult:
    notes: List[str] = []
    li: LoadedInput = load_input(src, page=page, dpi=dpi)
    img, rinfo = correct_photo(li.image, photo)
    moved = any(a in rinfo.applied for a in ("perspective", "rotate", "shear"))
    colourful = _colourful(li.image)
    if prep and not is_identity(prep):
        # preparation works on the corrected image, before scale and tracing
        img = apply_prep(img, prep)
        colourful = _colourful(img)
        notes.append("preparation applied: " + describe(prep))

    scale = ScaleResult(None)
    method = "none"
    boxes_for_areas: List[TextBox] = []
    if mm_per_px:
        method = "given"
    elif auto_scale:
        boxes: List[TextBox] = []
        if li.text and not moved and _numeric(li.text) >= 2:
            boxes = li.text
            notes.append(f"dimensions read from the PDF text layer ({_numeric(li.text)} numbers)")
        elif ocr_available():
            boxes = run_ocr(img)
            notes.append(f"OCR found {len(boxes)} text boxes")
        else:
            notes.append("auto scale needs OCR: pip install fpvec[ocr]")
        boxes_for_areas = boxes
        if boxes:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scale = estimate_scale(gray, boxes, units)
            if scale.mm_per_px:
                method = "dimensions"
                mm_per_px = scale.mm_per_px
                sx, sy = scale.mm_per_px_x, scale.mm_per_px_y
                if sx and sy:
                    img2, f = fix_aspect(img, sx, sy, rinfo)
                    if f != 1.0:
                        img = img2
                        mm_per_px = sx
                        scale.mm_per_px = sx
                        notes.append(f"x and y scales differed ({sx:.4f} vs {sy:.4f} mm/px): image stretched {f:.3f}x in y")
                        for r in scale.readings:      # keep the readings on the new image
                            r.box = (r.box[0], r.box[1] * f, r.box[2], r.box[3] * f)
                            r.p0 = (r.p0[0], r.p0[1] * f)
                            r.p1 = (r.p1[0], r.p1[1] * f)
            else:
                notes.append("no consistent dimensions found")

    params = Params(**vars(params)) if params else Params()
    # sensitive thin-line detection for photos / scans and for colour plans
    # (lines on coloured floors have less contrast); strict for clean
    # black-and-white drawings, where it keeps anti-aliasing out
    params.degraded = params.degraded or bool(rinfo.applied) or colourful
    plan = vectorize(img, mm_per_px, params)
    if not plan.mm_per_px and auto_scale and boxes_for_areas:
        ar = scale_from_areas(plan, img.shape, boxes_for_areas)
        if ar.mm_per_px:
            plan.mm_per_px = ar.mm_per_px
            scale = ar
            method = "room areas"
            notes.append(f"scale from {ar.votes} room-area label(s)")
    if not plan.mm_per_px and wall_mm and plan.wall_thickness_px:
        plan.mm_per_px = wall_mm / plan.wall_thickness_px
        method = "wall"
        notes.append(f"scale assumed from a {wall_mm:g} mm main wall")
    scale.mm_per_px = plan.mm_per_px
    return ProcessResult(plan, img, scale, method, rinfo, li.kind, li.page, li.pages, notes)
