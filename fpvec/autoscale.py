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


def _trace(lines: np.ndarray, x: int, y: int, step: int, gap: int) -> Tuple[int, dict]:
    """Follow a near-horizontal line from (x, y) in direction ``step``,
    allowing it to drift a row at a time (slight residual slant) and to be
    broken by up to ``gap`` px (ticks, arrows). Returns (last x, {x: y})."""
    H, W = lines.shape
    ys = {}
    last, miss, yc = x, 0, y
    while 0 <= x < W and miss <= gap:
        found = None
        for dy in (0, -1, 1):
            yy = yc + dy
            if 0 <= yy < H and lines[yy, x]:
                found = yy
                break
        if found is not None:
            yc, last, miss = found, x, 0
        else:
            miss += 1
        ys[x] = yc
        x += step
    return last, ys


def _measure_h(ink: np.ndarray, box, text_h: float) -> Optional[Tuple[float, float, float, str]]:
    """Horizontal dimension: returns (y, x_left, x_right, note) of the stretch."""
    H, W = ink.shape
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    tw = max(1, x1 - x0)
    th = max(2.0, text_h)
    cx = int(round((x0 + x1) / 2))
    if not 0 <= cx < W:
        return None
    # long, near-horizontal strokes (text strokes are short)
    L = max(5, int(min(tw * 0.5, th * 2.5)))
    r0, r1 = max(0, y0 - int(3 * th)), min(H, y1 + int(3 * th))
    lines = np.zeros_like(ink)
    lines[r0:r1] = cv2.morphologyEx(ink[r0:r1], cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    gap = int(max(3, th * 0.6))
    best = None
    for y in range(r0, r1):
        if y0 + 1 < y < y1 - 1 or not lines[y, max(0, cx - 1):cx + 2].any():
            continue
        a, ys_l = _trace(lines, cx, y, -1, gap)
        b, ys_r = _trace(lines, cx, y, 1, gap)
        if b - a < tw * 1.05:
            continue
        dist = min(abs(y - y0), abs(y - y1))
        if best is None or dist < best[0]:
            best = (dist, y, a, b, {**ys_l, **ys_r})
    if best is None:
        return None
    _, y, a, b, ys = best
    # line thickness at the text centre
    col = ink[:, cx]
    yt, yb = y, y
    while yt > 0 and col[yt - 1] and y - yt < th:
        yt -= 1
    while yb < H - 1 and col[yb + 1] and yb - y < th:
        yb += 1
    t_up, t_dn = y - yt, yb - y
    # ticks / arrows / dots / extension lines cross the line: ink on BOTH
    # sides (text sits on one side only, so it never passes as a tick)
    r = int(max(4, th * 0.8))
    ext = int(th)                          # look a little past the line's ends
    lo, hi = max(0, a - ext), min(W - 1, b + ext)
    xs = []
    for x in range(lo, hi + 1):
        yc = ys.get(x, ys.get(a if x < a else b, y))
        up = ink[max(0, yc - t_up - r):max(0, yc - t_up - 1), x]
        dn = ink[min(H, yc + t_dn + 2):min(H, yc + t_dn + r + 1), x]
        if up.sum() >= 2 and dn.sum() >= 2:
            xs.append(x)
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
    # stretches measured tick-to-tick are trustworthy; "line end" ones are a
    # fallback only (a broken line gives a wrong length)
    ticked = [r for r in good if r.note == "ticks"]
    pool = ticked if len(ticked) >= 2 else good
    if not pool:
        return res
    k, idx = _consensus([r.mm_per_px for r in pool])
    need = 2 if len(pool) <= 3 else max(2, int(np.ceil(len(pool) * 0.4)))
    if k is None or len(idx) < need:
        return res
    for j in idx:
        pool[j].used = True
    res.mm_per_px, res.votes, res.method = k, len(idx), "dimensions"
    res.confidence = round(len(idx) / max(1, len(pool)), 2)
    # per axis (detects images stretched in one direction)
    for axis, vert in (("x", False), ("y", True)):
        sub = [r for r in pool if r.vertical == vert]
        ka, ia = _consensus([r.mm_per_px for r in sub])
        if ka is not None and len(ia) >= 2:
            setattr(res, f"mm_per_px_{axis}", ka)
            for j in ia:
                sub[j].used = True
    return res


# ------------------------------------------------------------ room areas

_AREA = re.compile(r"(\d+(?:[.,]\d+)?)\s*(m2|m²|㎡|sqm|sq\.?\s*m\b|m\s*2|sq\.?\s*ft|sqft|sf|ft2|ft²)", re.I)


def parse_area(text: str) -> Optional[float]:
    """Room area label -> mm². ``38.4 m2``, ``38.4m²``, ``12 sqm``, ``220 sq ft``."""
    m = _AREA.search(text.replace(" ", " "))
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower().replace(" ", "").replace(".", "")
    if unit in ("sqft", "sf", "ft2", "ft²"):
        v *= 0.09290304
    return v * 1e6 if 1.0 <= v <= 500.0 else None


def scale_from_areas(plan, shape, boxes: Sequence[TextBox], tol: float = 0.04) -> ScaleResult:
    """Scale from room-area labels: each label's room is the free space
    enclosed by the detected walls (openings closed) that contains it, measured
    to the wall centrelines. mm per px = sqrt(area mm² / area px²); the value
    most rooms agree on (within ``tol``) wins."""
    labels = [(b, parse_area(b.text)) for b in boxes]
    labels = [(b, v) for b, v in labels if v]
    res = ScaleResult(None, total=len(labels))
    if not labels or not plan.walls:
        return res
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    for w in plan.walls:
        cv2.line(m, tuple(int(round(v)) for v in w.start), tuple(int(round(v)) for v in w.end), 255,
                 max(1, int(round(w.thickness))))
    for o in plan.openings:     # close doorways so each room is its own region
        cv2.line(m, tuple(int(round(v)) for v in o.start), tuple(int(round(v)) for v in o.end), 255,
                 max(1, int(round(o.thickness))))
    n, lab, stats, _ = cv2.connectedComponentsWithStats((m == 0).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    reads: List[DimReading] = []
    for b, area in labels:
        cx, cy = int((b.box[0] + b.box[2]) / 2), int((b.box[1] + b.box[3]) / 2)
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        i = int(lab[cy, cx])
        d = DimReading(b.text, area, tuple(float(v) for v in b.box), False, note="room area")
        if i == 0 or i in border:
            d.note = "label not inside a closed room"
            reads.append(d)
            continue
        comp = (lab == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = sum(cv2.arcLength(c, True) for c in cnts)
        t = float(np.median([w.thickness for w in plan.walls]))
        px_area = float(comp.sum()) + per * t / 2 + t * t       # grow to the wall centrelines
        d.length_px = float(np.sqrt(px_area))
        d.mm_per_px = float(np.sqrt(area / px_area))
        x, y, ww, hh, _ = stats[i]
        d.p0, d.p1 = (float(x), float(y)), (float(x + ww), float(y + hh))
        reads.append(d)
    res.readings = reads
    good = [r for r in reads if r.mm_per_px > 0]
    k, idx = _consensus([r.mm_per_px for r in good], tol)
    need = 1 if len(good) == 1 else 2
    if k is None or len(idx) < need:
        return res
    for j in idx:
        good[j].used = True
    res.mm_per_px, res.votes, res.method = k, len(idx), "room areas"
    res.confidence = round(len(idx) / len(good), 2)
    return res
