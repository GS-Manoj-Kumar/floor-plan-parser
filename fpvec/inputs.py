"""Load any supported input: image files / bytes / arrays, and PDF pages.

PDFs are rendered with pypdfium2 (Apache-2.0 / BSD, PDFium). For vector
PDFs (CAD exports) the text layer is read too: its words come with exact
positions, so dimension numbers need no OCR.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Union

import cv2
import numpy as np

from .autoscale import TextBox

Source = Union[str, bytes, bytearray, np.ndarray]


@dataclass
class LoadedInput:
    image: np.ndarray                    # BGR
    kind: str                            # "image" | "pdf"
    page: int = 0
    pages: int = 1
    dpi: Optional[float] = None
    text: List[TextBox] = field(default_factory=list)   # PDF text layer (px of the rendered page)
    pdf_mm_per_px: Optional[float] = None               # paper mm per rendered px (not the drawing scale)


def pdf_available() -> bool:
    try:
        import pypdfium2  # noqa: F401
        return True
    except Exception:
        return False


def _is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def load_input(src: Source, page: int = 0, dpi: float = 300.0, max_side: int = 7000) -> LoadedInput:
    if isinstance(src, np.ndarray):
        img = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            img = _flatten_alpha(img)
        return LoadedInput(img, "image")
    if isinstance(src, str):
        with open(src, "rb") as f:
            data = f.read()
    else:
        data = bytes(src)
    if _is_pdf(data):
        return load_pdf(data, page, dpi, max_side)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode the image (PNG / JPG / BMP / TIFF / WEBP or PDF)")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = _flatten_alpha(img)
    return LoadedInput(img, "image")


def _flatten_alpha(img: np.ndarray) -> np.ndarray:
    a = img[:, :, 3:4].astype(np.float32) / 255.0
    return (img[:, :, :3].astype(np.float32) * a + 255 * (1 - a)).astype(np.uint8)


def load_pdf(data: bytes, page: int = 0, dpi: float = 300.0, max_side: int = 7000) -> LoadedInput:
    if not pdf_available():
        raise ImportError("PDF input needs pypdfium2: pip install fpvec[pdf]")
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(data)
    n = len(doc)
    if not 0 <= page < n:
        raise ValueError(f"page {page + 1} does not exist (the PDF has {n})")
    pg = doc[page]
    w_pt, h_pt = pg.get_size()
    scale = dpi / 72.0
    if max(w_pt, h_pt) * scale > max_side:
        scale = max_side / max(w_pt, h_pt)
    bmp = pg.render(scale=scale, fill_color=(255, 255, 255, 255))
    img = bmp.to_numpy()
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    # pdfium renders BGR(x) by default in to_numpy for BGRx formats; normalise to BGR
    mode = getattr(bmp, "mode", "BGR")
    if mode in ("RGB", "RGBA", "RGBX"):
        img = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
    img = np.ascontiguousarray(img)
    text = _pdf_words(pg, scale, h_pt)
    doc.close()
    return LoadedInput(img, "pdf", page, n, scale * 72.0, text, pdf_mm_per_px=25.4 / (scale * 72.0))


def _pdf_words(pg, scale: float, h_pt: float) -> List[TextBox]:
    """Words of the page's text layer with their boxes in rendered pixels."""
    out: List[TextBox] = []
    try:
        tp = pg.get_textpage()
    except Exception:
        return out
    try:
        n = tp.count_chars()
        cur, box = [], None

        def flush():
            if cur and box is not None:
                x0, y0, x1, y1 = box
                out.append(TextBox("".join(cur), (x0 * scale, (h_pt - y1) * scale, x1 * scale, (h_pt - y0) * scale),
                                   1.0, "pdf"))

        for i in range(n):
            ch = tp.get_text_range(i, 1)
            if not ch or ch.isspace():
                flush()
                cur, box = [], None
                continue
            l, b, r, t = tp.get_charbox(i)
            if box is not None:
                # a new word when the char is far from the current box
                gap = max(l - box[2], box[0] - r, b - box[3], box[1] - t)
                size = max(box[3] - box[1], box[2] - box[0], 1)
                if gap > 0.6 * min(size, max(t - b, r - l, 1)) * 1.5:
                    flush()
                    cur, box = [], None
            cur.append(ch)
            box = (l, b, r, t) if box is None else (min(box[0], l), min(box[1], b), max(box[2], r), max(box[3], t))
        flush()
    finally:
        tp.close()
    return out


def page_count(src: Source) -> int:
    data = open(src, "rb").read() if isinstance(src, str) and os.path.isfile(src) else bytes(src)
    if not _is_pdf(data) or not pdf_available():
        return 1
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(data)
    n = len(doc)
    doc.close()
    return n
