"""Automatic scale from the dimension text printed on the plan.

For every number found (by OCR, or read straight from a PDF's text layer):

1. parse it as a length (``3600`` mm, ``3.60`` m, ``12'-6"`` ft-in, ``3600mm`` ...);
2. find the **dimension line** it labels: a long thin line parallel to the
   text, just above / below (or beside, for vertical text) and running past
   both ends of it;
3. find the **ticks** on that line (45 degree slashes, arrows, dots or the
   extension lines crossing it) on either side of the text: the stretch
   between the nearest tick left and right is what the number measures;
4. ``mm per pixel = value / stretch length``.

Every dimension votes; the scale with most votes (within 2.5 %) wins. The
horizontal and vertical dimensions are also solved separately, so an image
that is stretched in one direction (a photo, a bad scan) is reported with
``mm_per_px_x`` != ``mm_per_px_y``.

OCR uses RapidOCR (Apache-2.0, ONNX models bundled, pip-only) when installed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------- text input


@dataclass
class TextBox:
    text: str
    box: Tuple[float, float, float, float]   # x0, y0, x1, y1 (image px)
    conf: float = 1.0
    source: str = "ocr"                      # "ocr" | "pdf"

    @property
    def vertical(self) -> bool:
        x0, y0, x1, y1 = self.box
        return (y1 - y0) > (x1 - x0) * 1.15


_OCR = None


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def run_ocr(img: np.ndarray, max_side: int = 2400) -> List[TextBox]:
    """Text boxes from RapidOCR (empty list if it is not installed)."""
    global _OCR
    if not ocr_available():
        return []
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    h, w = img.shape[:2]
    k = min(1.0, max_side / max(h, w))
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    res, _ = _OCR(small)
    out: List[TextBox] = []
    for box, txt, conf in res or []:
        pts = np.asarray(box, float) / k
        out.append(TextBox(str(txt), (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()),
                           float(conf), "ocr"))
    return out


# --------------------------------------------------------------- parse value

_FTIN = re.compile(r"^(\d+)\s*(?:'|’|ft)\s*-?\s*(?:(\d+(?:\.\d+)?)(?:\s+(\d+)/(\d+))?\s*(?:\"|”|in)?)?$")
_INCH = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:\"|”|in)$")
_NUM = re.compile(r"^(\d+(?:[.,]\d+)?)\s*(mm|cm|m)?$")


def parse_length(text: str, units: str = "auto") -> Optional[float]:
    """Length in mm, or None if the text is not a plausible dimension.

    units: "auto", "mm", "cm", "m" or "ftin" (how plain numbers are read)."""
    t = text.strip().lower().replace(" ", " ")
    t = re.sub(r"^[^\d]+", "", t)               # OCR junk in front: ".1000", "~3600"
    t = re.sub(r"[^\d'\"’”a-z./\s-]+$", "", t).strip()
    t = t.replace("o", "0") if re.fullmatch(r"[\do]+", t) else t
    if not t:
        return None
    m = _FTIN.match(t)
    if m and ("'" in t or "’" in t or "ft" in t):
        ft = float(m.group(1))
        inch = float(m.group(2) or 0) + (float(m.group(3)) / float(m.group(4)) if m.group(3) else 0)
        v = (ft * 12 + inch) * 25.4
        return v if 150 <= v <= 60000 else None
    m = _INCH.match(t)
    if m:
        v = float(m.group(1)) * 25.4
        return v if 150 <= v <= 60000 else None
    m = _NUM.match(t)
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    unit = m.group(2)
    if unit == "m" or (unit is None and units == "m"):
        v = num * 1000
    elif unit == "cm" or (unit is None and units == "cm"):
        v = num * 10
    elif unit == "mm" or units == "mm":
        v = num
    elif units == "ftin":
        v = num * 25.4                          # bare number = inches
    else:                                       # auto
        if "." in m.group(1) or "," in m.group(1):
            v = num * 1000 if num < 100 else num   # 3.60 -> metres
        else:
            v = num if num >= 100 else None        # 3600 -> mm; "12" alone is ambiguous
            if v is None:
                return None
    return v if 150 <= v <= 60000 else None


# --------------------------------------------------------- measure the lines


@dataclass
class DimReading:
    text: str
    value_mm: float
    box: Tuple[float, float, float, float]
    vertical: bool
    p0: Tuple[float, float] = (0, 0)          # measured stretch on the dimension line (px)
    p1: Tuple[float, float] = (0, 0)
    length_px: float = 0.0
    mm_per_px: float = 0.0
    used: bool = False
    note: str = ""


def _line_mask(gray: np.ndarray) -> np.ndarray:
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = (gray < min(max(t, 90), 200)).astype(np.uint8)
    # black-hat keeps faint thin lines too
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return (ink | (bh > 40)).astype(np.uint8)


def _measure_h(ink: np.ndarray, box, text_h: float) -> Optional[Tuple[float, float, float, str]]:
    """Horizontal dimension: returns (y, x_left, x_right, note) of the stretch."""
    H, W = ink.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    tw = max(1, x1 - x0)
    th = max(2.0, text_h)
    cx = (x0 + x1) / 2
    # long horizontal strokes only (text strokes are short)
    L = max(5, int(tw * 0.6))
    lines = cv2.morphologyEx(ink[max(0, y0 - int(3 * th)):min(H, y1 + int(3 * th)), :],
                             cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    off = max(0, y0 - int(3 * th))
    best = None
    for r in range(lines.shape[0]):
        y = r + off
        if y0 + 1 < y < y1 - 1:
            continue                                  # inside the text
        row = lines[r]
        xi = int(round(cx))
        if not (0 <= xi < W) or not row[max(0, xi - 1):xi + 2].any():
            continue
        gap = int(max(3, th * 0.6))       # ticks / arrows can break the line
        a = xi
        while a > 0 and row[max(0, a - gap):a].any():
            a -= 1
        b = xi
        while b < W - 1 and row[b + 1:b + 1 + gap].any():
            b += 1
        if b - a < tw * 1.05:
            continue
        dist = min(abs(y - y0), abs(y - y1))
        if best is None or dist < best[0]:
            best = (dist, y, a, b)
    if best is None:
        return None
    _, y, a, b = best
    # the dimension line's own thickness at this point
    col = ink[:, int(round(cx))]
    yt, yb = y, y
    while yt > 0 and col[yt - 1] and y - yt < th:
        yt -= 1
    while yb < H - 1 and col[yb + 1] and yb - y < th:
        yb += 1
    # ticks / arrows / dots / extension lines cross the line: ink on BOTH sides
    # (text sits on one side only, so it can never pass as a tick)
    r = int(max(4, th * 0.8))
    ext = int(th)                          # look a little past the line's ends
    a, b = max(0, a - ext), min(W - 1, b + ext)
    up = ink[max(0, yt - r):max(0, yt - 1), a:b + 1]
    dn = ink[min(H, yb + 2):min(H, yb + r + 1), a:b + 1]
    above = up.sum(axis=0) >= 2 if up.size else np.zeros(b - a + 1, bool)
    below = dn.sum(axis=0) >= 2 if dn.size else np.zeros(b - a + 1, bool)
    cols = np.arange(a, b + 1)
    xs = cols[above & below]
    groups = []
    for x in xs:
        if groups and x - groups[-1][-1] <= 3:
            groups[-1].append(x)
        else:
            groups.append([x])
    ticks = [float(np.mean(g)) for g in groups if len(g) <= max(8, th * 1.5)]
    left = [t for t in ticks if t < x0 - 1]
    right = [t for t in ticks if t > x1 + 1]
    note = "ticks"
    xl = max(left) if left else None
    xr = min(right) if right else None
    if xl is None:
        xl, note = float(a), "line end"
    if xr is None:
        xr, note = float(b), "line end"
    return float(y), xl, xr, note


def measure_dimensions(gray: np.ndarray, boxes: Sequence[TextBox], units: str = "auto") -> List[DimReading]:
    ink = _line_mask(gray)
    inkT = np.ascontiguousarray(ink.T)
    out: List[DimReading] = []
    for tb in boxes:
        v = parse_length(tb.text, units)
        if v is None:
            continue
        d = DimReading(tb.text, v, tuple(float(x) for x in tb.box), tb.vertical)
        x0, y0, x1, y1 = tb.box
        if not tb.vertical:
            m = _measure_h(ink, (x0, y0, x1, y1), y1 - y0)
            if m:
                y, xl, xr, d.note = m
                d.p0, d.p1, d.length_px = (xl, y), (xr, y), xr - xl
        else:
            m = _measure_h(inkT, (y0, x0, y1, x1), x1 - x0)
            if m:
                x, yt, yb, d.note = m
                d.p0, d.p1, d.length_px = (x, yt), (x, yb), yb - yt
        if d.length_px > 0:
            d.mm_per_px = v / d.length_px
        else:
            d.note = "no dimension line"
        out.append(d)
    return out


# ------------------------------------------------------------------ consensus


@dataclass
class ScaleResult:
    mm_per_px: Optional[float]
    mm_per_px_x: Optional[float] = None
    mm_per_px_y: Optional[float] = None
    votes: int = 0
    total: int = 0
    confidence: float = 0.0
    method: str = "none"                  # "dimensions" | "none"
    readings: List[DimReading] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["readings"] = [asdict(r) for r in self.readings]
        return d


def _consensus(vals: List[float], tol: float = 0.025) -> Tuple[Optional[float], List[int]]:
    if not vals:
        return None, []
    best: List[int] = []
    for i, v in enumerate(vals):
        grp = [j for j, w in enumerate(vals) if abs(w - v) <= tol * v]
        if len(grp) > len(best):
            best = grp
    return float(np.median([vals[j] for j in best])), best


def estimate_scale(gray: np.ndarray, boxes: Sequence[TextBox], units: str = "auto") -> ScaleResult:
    reads = [r for r in measure_dimensions(gray, boxes, units)]
    good = [r for r in reads if r.mm_per_px > 0]
    res = ScaleResult(None, total=len(reads), readings=reads)
    if not good:
        return res
    k, idx = _consensus([r.mm_per_px for r in good])
    if k is None or (len(idx) < 2 and len(good) > 1):
        return res
    for j in idx:
        good[j].used = True
    res.mm_per_px, res.votes, res.method = k, len(idx), "dimensions"
    res.confidence = round(len(idx) / max(1, len(good)), 2)
    # per axis (detects images stretched in one direction)
    for axis, vert in (("x", False), ("y", True)):
        sub = [r for r in good if r.vertical == vert]
        ka, ia = _consensus([r.mm_per_px for r in sub])
        if ka is not None and len(ia) >= 2:
            setattr(res, f"mm_per_px_{axis}", ka)
            for j in ia:
                sub[j].used = True
    return res
