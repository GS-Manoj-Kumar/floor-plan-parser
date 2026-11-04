"""Preparation step: clean the (already corrected) image before tracing.

Runs after photo correction (perspective / rotation / lighting) and before
the scale is read and the plan is traced, so every coordinate refers to the
corrected image. A preparation is a small JSON-able recipe:

    {
      "grayscale": true,          # colour -> black & white (grey levels)
      "black": 120,               # darken lines: grey at or below this becomes pure black (0..254)
      "white": 225,               # whiten background: grey at or above this becomes pure white (1..255)
      "binary": false,            # pure black / white at the midpoint between the two
      "masks": [                  # areas that are not needed, in corrected-image px
        {"type": "brush", "points": [[x, y], ...], "size": 40, "mode": "white"},
        {"type": "rect",  "points": [[x0, y0], [x1, y1]],     "mode": "blur", "blur": 12},
        {"type": "poly",  "points": [[x, y], ...],            "mode": "white"}
      ]
    }

Order: grey levels first (grayscale, black / white points, binary), then the
masks in the order drawn ("white" paints the area white, "blur" blurs it).
The demo page previews exactly this order.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

DEFAULT = {"grayscale": False, "black": 0, "white": 255, "binary": False, "masks": []}


def normalize(spec: Optional[dict]) -> dict:
    s = dict(DEFAULT)
    if spec:
        s.update({k: v for k, v in spec.items() if k in DEFAULT})
    s["black"] = int(np.clip(s.get("black") or 0, 0, 254))
    s["white"] = int(np.clip(s.get("white") if s.get("white") is not None else 255, s["black"] + 1, 255))
    s["masks"] = [m for m in (s.get("masks") or []) if isinstance(m, dict) and m.get("points")]
    return s


def is_identity(spec: Optional[dict]) -> bool:
    s = normalize(spec)
    return not s["grayscale"] and s["black"] == 0 and s["white"] == 255 and not s["binary"] and not s["masks"]


def levels_lut(black: int, white: int, binary: bool) -> np.ndarray:
    v = np.arange(256, dtype=np.float32)
    if binary:
        mid = (black + white) / 2.0
        return np.where(v <= mid, 0, 255).astype(np.uint8)
    out = (v - black) * 255.0 / max(1.0, white - black)
    return np.clip(out, 0, 255).astype(np.uint8)


def mask_of(m: dict, shape) -> np.ndarray:
    """0/255 mask for one shape."""
    H, W = shape[:2]
    out = np.zeros((H, W), np.uint8)
    pts = np.round(np.asarray(m["points"], np.float64)).astype(np.int32)
    t = m.get("type", "brush")
    if t == "rect" and len(pts) >= 2:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
    elif t == "poly" and len(pts) >= 3:
        cv2.fillPoly(out, [pts.reshape(-1, 1, 2)], 255)
    else:                                            # brush stroke
        size = max(1, int(round(float(m.get("size", 20)))))
        if len(pts) == 1:
            cv2.circle(out, tuple(int(v) for v in pts[0]), max(1, size // 2), 255, -1)
        else:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, 255, size, cv2.LINE_AA)
            for p in (pts[0], pts[-1]):              # round caps
                cv2.circle(out, tuple(int(v) for v in p), max(1, size // 2), 255, -1)
        out = np.where(out > 0, 255, 0).astype(np.uint8)
    return out


def apply_prep(img: np.ndarray, spec: Optional[dict]) -> np.ndarray:
    """Apply a preparation recipe to a BGR image (returns a new BGR image)."""
    s = normalize(spec)
    out = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if s["grayscale"] or s["binary"]:
        g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if s["black"] > 0 or s["white"] < 255 or s["binary"]:
        out = cv2.LUT(out, levels_lut(s["black"], s["white"], s["binary"]))
    else:
        out = out.copy()
    for m in s["masks"]:
        mk = mask_of(m, out.shape)
        if not mk.any():
            continue
        if m.get("mode", "white") == "blur":
            sigma = max(1.0, float(m.get("blur", 10)))
            blurred = cv2.GaussianBlur(out, (0, 0), sigma)
            out[mk > 0] = blurred[mk > 0]
        else:
            out[mk > 0] = 255
    return out


def describe(spec: Optional[dict]) -> str:
    s = normalize(spec)
    parts = []
    if s["grayscale"]:
        parts.append("black & white")
    if s["binary"]:
        parts.append("pure black / white")
    if s["black"] > 0:
        parts.append(f"lines darkened (black at {s['black']})")
    if s["white"] < 255:
        parts.append(f"background whitened (white at {s['white']})")
    if s["masks"]:
        n_w = sum(m.get("mode", "white") != "blur" for m in s["masks"])
        n_b = len(s["masks"]) - n_w
        parts.append(", ".join(x for x in (f"{n_w} area(s) removed" if n_w else "", f"{n_b} blurred" if n_b else "") if x))
    return "; ".join(parts) or "none"
