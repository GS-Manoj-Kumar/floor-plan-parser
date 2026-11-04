"""Photo correction: perspective, lighting, rotation and shear.

A plan photographed at an angle is a sheet of paper seen in perspective, on
some background, with uneven light. ``correct_photo`` (all OpenCV):

1. **Sheet outline.** The paper is the large bright region. Its outline is
   reduced to 4 corners and warped to a rectangle (perspective removed). The
   output size comes from the sheet's own edge lengths.
2. **Lighting.** Divides the image by a heavily smoothed copy of itself,
   which flattens shadows and gradients so the ink threshold works
   everywhere.
3. **Straighten.** The drawing's line directions are histogrammed (Hough).
   The two dominant families are rotated / sheared back to exactly
   horizontal and vertical.

The remaining unknown is the sheet's true aspect ratio (a perspective photo
does not fully determine it). When the plan has dimension text, the
auto-scale step measures the x and y scales separately and the pipeline
stretches the image to make its pixels square.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RectifyInfo:
    applied: List[str] = field(default_factory=list)   # "perspective", "lighting", "rotate", "shear", "aspect"
    quad: Optional[List[Tuple[float, float]]] = None    # sheet corners found in the input (px)
    rotation_deg: float = 0.0
    shear_deg: float = 0.0
    aspect_fix: float = 1.0                             # y stretch applied from per-axis scales
    size: Tuple[int, int] = (0, 0)                      # output (w, h)

    def to_dict(self) -> dict:
        return asdict(self)


def _order(pts: np.ndarray) -> np.ndarray:
    """tl, tr, br, bl."""
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_sheet(img: np.ndarray, min_area: float = 0.2) -> Optional[np.ndarray]:
    """Four corners of the paper sheet (float32 tl,tr,br,bl), or None when the
    sheet fills the frame / cannot be found (a scan or screenshot)."""
    h, w = img.shape[:2]
    k = 900 / max(h, w)
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img.copy()
    k = min(k, 1.0)
    g = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    # paper = bright and low-saturation; close the drawing's dark lines away
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    H, W = g.shape
    if area < min_area * H * W:
        return None
    hull = cv2.convexHull(c)
    quad = None
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float32)
            break
    if quad is None:
        return None
    # the sheet touching the frame on all sides = nothing to correct
    margin = 0.01 * max(H, W)
    xs, ys = quad[:, 0], quad[:, 1]
    if xs.min() <= margin and ys.min() <= margin and xs.max() >= W - 1 - margin and ys.max() >= H - 1 - margin:
        return None
    return _order(quad / k)


def warp_sheet(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def flatten_light(gray: np.ndarray) -> np.ndarray:
    """Remove shadows / light gradients: divide by the smoothed background."""
    k = max(15, (max(gray.shape) // 25) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    out = cv2.divide(gray.astype(np.float32), np.maximum(bg.astype(np.float32), 1), scale=255.0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # stretch so paper is white and ink dark again
    lo, hi = np.percentile(out, 0.5), np.percentile(out, 60)
    if hi - lo > 10:
        out = np.clip((out.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return out


def dominant_angles(gray: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """Angles (deg, image coords) of the near-horizontal and near-vertical
    line families, weighted by length."""
    edges = cv2.Canny(gray, 50, 150)
    L = max(20, int(min(gray.shape) * 0.04))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60, minLineLength=L, maxLineGap=4)
    if lines is None:
        return None, None
    h_ang, h_w, v_ang, v_w = [], [], [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        ln = math.hypot(x2 - x1, y2 - y1)
        a = (a + 90) % 180 - 90                     # (-90, 90]
        if abs(a) < 30:
            h_ang.append(a)
            h_w.append(ln)
        elif abs(a) > 60:
            v_ang.append(a - 90 if a > 0 else a + 90)   # deviation from vertical
            v_w.append(ln)

    def mode(angles, wts):
        if len(angles) < 3:
            return None
        hist, edges_ = np.histogram(angles, bins=240, range=(-30, 30), weights=wts)
        hist = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")
        i = int(np.argmax(hist))
        c = (edges_[i] + edges_[i + 1]) / 2
        sel = [(a, w) for a, w in zip(angles, wts) if abs(a - c) < 0.75]
        return float(np.average([a for a, _ in sel], weights=[w for _, w in sel])) if sel else c

    return mode(h_ang, h_w), mode(v_ang, v_w)


def straighten(img: np.ndarray, gray: np.ndarray, info: RectifyInfo, min_deg: float = 0.15):
    """Rotate + shear so the two line families become exactly horizontal and vertical."""
    ah, av = dominant_angles(gray)
    if ah is None and av is None:
        return img
    rot = ah if ah is not None else av
    shear = (av - ah) if (ah is not None and av is not None) else 0.0
    if abs(rot) < min_deg and abs(shear) < min_deg:
        return img
    h, w = img.shape[:2]
    # affine: rotate by -rot about the centre, then shear x so verticals stand up
    R = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
    R3 = np.vstack([R, [0, 0, 1]])
    sh = math.tan(math.radians(shear)) if abs(shear) >= min_deg else 0.0
    Sh = np.array([[1, sh, -sh * h / 2], [0, 1, 0], [0, 0, 1]])
    A = (Sh @ R3)[:2]
    # grow the canvas to keep the corners
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float).T
    t = A @ corners
    x0, y0, x1, y1 = t[0].min(), t[1].min(), t[0].max(), t[1].max()
    A[0, 2] -= x0
    A[1, 2] -= y0
    out = cv2.warpAffine(img, A, (int(math.ceil(x1 - x0)), int(math.ceil(y1 - y0))),
                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    if abs(rot) >= min_deg:
        info.applied.append("rotate")
        info.rotation_deg = round(rot, 3)
    if sh:
        info.applied.append("shear")
        info.shear_deg = round(shear, 3)
    return out


def is_photo(img: np.ndarray) -> bool:
    """Heuristic for "auto": a visible sheet edge, or strongly uneven paper."""
    if find_sheet(img) is not None:
        return True
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    bright = small[small > np.percentile(small, 40)]
    return bool(bright.size and (np.percentile(bright, 95) - np.percentile(bright, 5)) > 45)


def correct_photo(img: np.ndarray, mode: str = "auto") -> Tuple[np.ndarray, RectifyInfo]:
    """mode: "off" | "auto" (only when it looks like a photo) | "on"."""
    info = RectifyInfo()
    if mode == "off" or (mode == "auto" and not is_photo(img)):
        info.size = (img.shape[1], img.shape[0])
        return img, info
    out = img
    quad = find_sheet(img)
    if quad is not None:
        out = warp_sheet(img, quad)
        info.applied.append("perspective")
        info.quad = [(round(float(x), 1), round(float(y), 1)) for x, y in quad]
    g = out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    g = flatten_light(g)
    info.applied.append("lighting")
    g = straighten(g, g, info)
    info.size = (g.shape[1], g.shape[0])
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), info


def fix_aspect(img: np.ndarray, mm_x: float, mm_y: float, info: Optional[RectifyInfo] = None,
               tol: float = 0.015) -> Tuple[np.ndarray, float]:
    """Stretch y so both axes share mm_x mm per pixel. Returns (image, factor)."""
    f = mm_y / mm_x
    if abs(f - 1) < tol:
        return img, 1.0
    out = cv2.resize(img, (img.shape[1], int(round(img.shape[0] * f))), interpolation=cv2.INTER_CUBIC)
    if info is not None:
        info.applied.append("aspect")
        info.aspect_fix = round(f, 4)
        info.size = (out.shape[1], out.shape[0])
    return out, f
