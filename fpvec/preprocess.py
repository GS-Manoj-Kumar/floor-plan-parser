"""Image loading, binarisation and stroke-width estimation."""
from __future__ import annotations

from collections import Counter
from typing import Tuple, Union

import cv2
import numpy as np

ImageLike = Union[str, np.ndarray]


def load_gray(image: ImageLike) -> np.ndarray:
    """Load a path or array as an 8-bit grayscale image."""
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(image)
    else:
        img = image
    if img.ndim == 3 and img.shape[2] == 4:
        # composite transparent PNGs on white
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * alpha + 255.0 * (1 - alpha)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def binarize(gray: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """Return a 0/1 mask of ink (dark) pixels.

    Otsu, clamped so that light-grey fills are not ink, AND clearly darker
    than the local surroundings: on plans with coloured / wooden floors a
    mid-tone floor can fall under a global threshold, but it is not darker
    than itself, while walls and lines are much darker than the floor
    around them."""
    if threshold is None:
        t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(min(max(t, 90), 170))
    ink = gray < threshold
    k = max(41, (max(gray.shape) // 25) | 1)
    small = 4 if max(gray.shape) > 1200 else 1        # do the large closing at reduced size
    g = cv2.resize(gray, None, fx=1 / small, fy=1 / small, interpolation=cv2.INTER_AREA) if small > 1 else gray
    ks = max(5, (k // small) | 1)
    local = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    if small > 1:
        local = cv2.resize(local, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    contrast = local.astype(np.int16) - gray.astype(np.int16)
    return (ink & (contrast > 40)).astype(np.uint8)


def _run_lengths(mask: np.ndarray) -> Counter:
    """Histogram of horizontal + vertical run lengths of ink pixels."""
    c: Counter = Counter()
    for m in (mask, mask.T):
        padded = np.pad(m, ((0, 0), (1, 1)))
        diff = np.diff(padded.astype(np.int8), axis=1)
        starts = np.argwhere(diff == 1)
        ends = np.argwhere(diff == -1)
        # argwhere returns row-major order, so starts/ends pair up row by row
        lengths = ends[:, 1] - starts[:, 1]
        c.update(lengths.tolist())
    return c


def _perp_run_area(mask: np.ndarray) -> np.ndarray:
    """Histogram of run lengths across the stroke, weighted by run length."""
    padded = np.pad(mask, ((0, 0), (1, 1)))
    diff = np.diff(padded.astype(np.int8), axis=1)
    starts = np.argwhere(diff == 1)
    ends = np.argwhere(diff == -1)
    lengths = ends[:, 1] - starts[:, 1]
    return np.bincount(lengths, weights=lengths) if len(lengths) else np.zeros(1)


def estimate_stroke_widths(ink: np.ndarray) -> Tuple[float, float]:
    """Estimate (thin line width, wall thickness) in pixels.

    Only *long straight* strokes are measured: the ink is opened with a long
    1-px wide kernel (horizontal and vertical), then the stroke thickness
    across those bars is histogrammed, weighted by ink area. Thin lines
    (furniture, arcs, window glazing, text) form a peak at 1-3 px; walls,
    which hold most of the long ink, form the dominant peak above that.
    """
    runs = _run_lengths(ink)
    if not runs:
        return 1.0, 6.0
    line_hist = np.zeros(7)
    for k in range(1, 7):
        line_hist[k] = runs.get(k, 0)
    line_w = float(np.argmax(line_hist[1:]) + 1)

    L = max(9, int(min(ink.shape) * 0.03))
    h = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, L), np.uint8))
    v = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8))
    a, b = _perp_run_area(h.T), _perp_run_area(v)
    n = max(len(a), len(b))
    hist = np.zeros(n)
    hist[:len(a)] += a
    hist[:len(b)] += b

    lo = int(max(4, np.ceil(line_w * 2)))
    hi = int(min(n - 1, max(lo + 2, min(ink.shape) * 0.06)))
    if lo >= hi:
        return line_w, float(max(lo, 6))
    # weight by thickness once more: a wall of thickness t holds t times the
    # ink of a thin line of the same length, and we want the dominant *wall*
    smooth = np.convolve(hist * np.arange(n), [1, 1, 1], mode="same")
    wall_t = int(np.argmax(smooth[lo:hi + 1]) + lo)
    # refine with the area-weighted mean of the 3-bin window
    ks = np.arange(wall_t - 1, wall_t + 2)
    wts = hist[ks]
    if wts.sum() > 0:
        wall_t = float((ks * wts).sum() / wts.sum())
    return line_w, float(wall_t)
