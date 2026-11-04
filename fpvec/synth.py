"""Synthetic floor plan generator with known ground truth (for tests/benchmarks).

Draws a plan the way sales/CAD floor plans usually look: solid black outer
walls, interior walls either solid or hollow (two thin lines), swing doors
(leaf + quarter arc) and windows (3 thin lines across a wall gap). Units are
millimetres, rendered at ``px_per_mm``.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _wall_rect(img, a, b, t, color=0, hollow=False, lw=1, fill=None):
    (x1, y1), (x2, y2) = a, b
    if y1 == y2:
        p, q = (min(x1, x2), y1 - t / 2), (max(x1, x2), y1 + t / 2)
    else:
        p, q = (x1 - t / 2, min(y1, y2)), (x1 + t / 2, max(y1, y2))
    p = (int(round(p[0])), int(round(p[1])))
    q = (int(round(q[0])), int(round(q[1])))
    if hollow:
        if fill is not None:
            cv2.rectangle(img, p, q, fill, -1)
        cv2.rectangle(img, p, q, color, lw)
    else:
        cv2.rectangle(img, p, q, color, -1)


def make_plan(px_per_mm: float = 0.1, seed: int = 0, hollow_interior: bool = True, noise: bool = False,
              grey_interior: bool = False, dims: bool = False, dim_units: str = "mm") -> Tuple[np.ndarray, Dict]:
    """Return (BGR image, ground truth dict in px).

    Layout (mm): 10 000 x 8 000 apartment, 240 mm exterior walls, 120 mm
    interior walls, two bedrooms, a bathroom and a living room.
    """
    rng = np.random.default_rng(seed)
    S = px_per_mm
    margin = 1500
    W, H = int((10000 + 2 * margin) * S), int((8000 + 2 * margin) * S)
    img = np.full((H, W), 255, np.uint8)
    P = lambda x, y: ((x + margin) * S, (y + margin) * S)
    Te, Ti = 240 * S, 120 * S
    lw = max(1, int(round(10 * S)))

    walls: List[dict] = []
    openings: List[dict] = []

    def wall(x1, y1, x2, y2, t, hollow=False, ops=()):
        """Wall from (x1,y1) to (x2,y2) mm with openings [(kind, start_mm, width_mm, extra)]."""
        horiz = y1 == y2
        L = abs(x2 - x1) if horiz else abs(y2 - y1)
        walls.append(dict(start=P(x1, y1), end=P(x2, y2), t=t, o="h" if horiz else "v"))
        cuts = sorted(ops, key=lambda o: o[1])
        pos = 0
        for kind, s0, w, extra in cuts + [(None, L, 0, None)]:
            if s0 > pos:
                a = (x1 + pos, y1) if horiz else (x1, y1 + pos)
                b = (x1 + s0, y1) if horiz else (x1, y1 + s0)
                if hollow and grey_interior:
                    # sales-plan style partitions: mid-grey outline, light-grey fill
                    _wall_rect(img, P(*a), P(*b), t, color=175, hollow=True, lw=lw, fill=222)
                else:
                    _wall_rect(img, P(*a), P(*b), t, hollow=hollow, lw=lw)
            if kind is None:
                break
            a = (x1 + s0, y1) if horiz else (x1, y1 + s0)
            b = (x1 + s0 + w, y1) if horiz else (x1, y1 + s0 + w)
            pa, pb = P(*a), P(*b)
            if kind == "window":
                for f in (-0.5, -0.17, 0.17, 0.5):
                    off = f * t
                    if horiz:
                        cv2.line(img, (int(pa[0]), int(pa[1] + off)), (int(pb[0]), int(pb[1] + off)), 0, lw)
                    else:
                        cv2.line(img, (int(pa[0] + off), int(pa[1])), (int(pb[0] + off), int(pb[1])), 0, lw)
                openings.append(dict(kind="window", start=pa, end=pb, o="h" if horiz else "v"))
            elif kind == "door":
                hinge_at_start, side = extra
                hinge = pa if hinge_at_start else pb
                latch = pb if hinge_at_start else pa
                r = w * S
                # hinge sits on the wall face of the swing side
                if horiz:
                    hf = (hinge[0], hinge[1] + side * t / 2)
                    tip = (hf[0], hf[1] + side * r)
                    lf = (latch[0], latch[1] + side * t / 2)
                else:
                    hf = (hinge[0] + side * t / 2, hinge[1])
                    tip = (hf[0] + side * r, hf[1])
                    lf = (latch[0] + side * t / 2, latch[1])
                cv2.line(img, tuple(map(int, hf)), tuple(map(int, tip)), 0, lw + 1)
                a0 = math.degrees(math.atan2(lf[1] - hf[1], lf[0] - hf[0]))
                a1 = math.degrees(math.atan2(tip[1] - hf[1], tip[0] - hf[0]))
                lo, hi = sorted((a0, a1))
                if hi - lo > 180:
                    lo, hi = hi, lo + 360
                cv2.ellipse(img, tuple(map(int, hf)), (int(r), int(r)), 0, lo, hi, 0, lw)
                openings.append(dict(kind="door", start=pa, end=pb, o="h" if horiz else "v",
                                     hinge=hinge, swing=side))
            pos = s0 + w

    # exterior (solid)
    wall(0, 0, 10000, 0, Te, ops=[("window", 1200, 1800, None), ("window", 6500, 2400, None)])
    wall(10000, 0, 10000, 8000, Te, ops=[("window", 4500, 2000, None)])
    wall(10000, 8000, 0, 8000, Te) if False else wall(0, 8000, 10000, 8000, Te,
                                                       ops=[("door", 7000, 1000, (True, -1)), ("window", 1500, 1500, None)])
    wall(0, 0, 0, 8000, Te, ops=[("window", 1500, 1500, None)])
    # interior
    hi = hollow_interior
    wall(4000, 0, 4000, 4000, Ti, hollow=hi, ops=[("door", 3000, 800, (False, 1))])
    wall(0, 4000, 6000, 4000, Ti, hollow=hi, ops=[("door", 4400, 900, (True, -1))])
    wall(6000, 4000, 6000, 8000, Ti, hollow=hi, ops=[("door", 300, 900, (True, -1))])
    wall(0, 6000, 2500, 6000, Ti, hollow=hi, ops=[("door", 1500, 700, (False, -1))])

    dim_list: List[dict] = []
    if dims:
        dim_list = _draw_dims(img, P, S, lw, dim_units)

    # furniture clutter: a bed, a table with chairs, text
    bx, by = P(800, 800)
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 1600 * S), int(by + 2000 * S)), 0, lw)
    cv2.rectangle(img, (int(bx), int(by)), (int(bx + 1600 * S), int(by + 300 * S)), 0, lw)
    tx, ty = P(7500, 1500)
    cv2.circle(img, (int(tx), int(ty)), int(500 * S), 0, lw)
    cv2.putText(img, "LIVING", tuple(map(int, P(7000, 6000))), cv2.FONT_HERSHEY_SIMPLEX, 60 * S / 10, 0, lw)
    if noise:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = np.clip(img.astype(np.int16) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    gt = dict(walls=walls, openings=openings, mm_per_px=1 / S, ext_t=Te, int_t=Ti, dims=dim_list)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), gt


def _fmt_dim(mm: float, units: str) -> str:
    if units == "ftin":
        inches = mm / 25.4
        ft, inch = int(inches // 12), int(round(inches - 12 * (inches // 12)))
        if inch == 12:
            ft, inch = ft + 1, 0
        return f"{ft}'-{inch}\""
    if units == "m":
        return f"{mm / 1000:.2f}"
    return f"{int(round(mm))}"


def _draw_dims(img, P, S, lw, units) -> List[dict]:
    """Dimension chains outside the plan: extension lines, 45 degree ticks,
    and the value centred above each segment (as on sales / CAD plans)."""
    out = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = max(0.4, 220 * S / 22)          # ~220 mm high digits
    th = max(1, int(round(lw)))

    def tick(x, y):
        k = int(round(90 * S))
        cv2.line(img, (int(x - k), int(y + k)), (int(x + k), int(y - k)), 0, th + 1)

    def chain(stops, horiz, off):
        # stops in mm along the edge; off = mm away from the wall line (outside)
        for i, v in enumerate(stops):
            if horiz:
                x, y = P(v, -off)
                _, y0 = P(v, -150)
                cv2.line(img, (int(x), int(y0)), (int(x), int(y - 120 * S)), 0, lw)
                tick(x, y)
            else:
                x, y = P(-off, v)
                x0, _ = P(-150, v)
                cv2.line(img, (int(x0), int(y)), (int(x - 120 * S), int(y)), 0, lw)
                tick(x, y)
        a = P(stops[0], -off) if horiz else P(-off, stops[0])
        b = P(stops[-1], -off) if horiz else P(-off, stops[-1])
        cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), 0, lw)
        for s0, s1 in zip(stops, stops[1:]):
            txt = _fmt_dim(s1 - s0, units)
            (tw, tht), _ = cv2.getTextSize(txt, font, fs, th)
            if horiz:
                cx, cy = P((s0 + s1) / 2, -off)
                org = (int(cx - tw / 2), int(cy - 60 * S))
                cv2.putText(img, txt, org, font, fs, 0, th, cv2.LINE_AA)
            else:
                cx, cy = P(-off, (s0 + s1) / 2)
                # vertical text: draw on a patch and rotate 90 degrees
                patch = np.full((tht + 10, tw + 6), 255, np.uint8)
                cv2.putText(patch, txt, (3, tht + 3), font, fs, 0, th, cv2.LINE_AA)
                patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ph, pw = patch.shape
                x1 = int(cx - 60 * S) - pw
                y1 = int(cy - ph / 2)
                if x1 >= 0 and y1 >= 0:
                    region = img[y1:y1 + ph, x1:x1 + pw]
                    np.minimum(region, patch, out=region)
            out.append(dict(value_mm=s1 - s0, text=txt, horiz=horiz,
                            a=(P(s0, -off) if horiz else P(-off, s0)), b=(P(s1, -off) if horiz else P(-off, s1))))

    chain([0, 1200, 3000, 4000, 6500, 8900, 10000], True, 700)
    chain([0, 10000], True, 1200)
    chain([0, 1500, 3000, 4000, 6000, 8000], False, 700)
    return out


def make_photo(plan_img: np.ndarray, seed: int = 0, tilt: float = 0.12, rot_deg: float = 7.0) -> np.ndarray:
    """Fake a phone photo of a printed plan: the sheet on a desk, seen at an
    angle (perspective), slightly rotated, with uneven light, blur, noise and
    JPEG compression."""
    rng = np.random.default_rng(seed)
    h, w = plan_img.shape[:2]
    # paper margin around the drawing
    m = int(0.06 * max(h, w))
    sheet = cv2.copyMakeBorder(plan_img, m, m, m, m, cv2.BORDER_CONSTANT, value=(250, 250, 248))
    sh, sw = sheet.shape[:2]
    W, H = int(sw * 1.5), int(sh * 1.5)
    # desk: brownish with texture
    desk = np.zeros((H, W, 3), np.float32)
    desk[:] = (60, 85, 120)
    desk += rng.normal(0, 12, (H, W, 1)).astype(np.float32)
    desk = cv2.GaussianBlur(desk, (0, 0), 3)
    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    cx, cy = W / 2, H / 2
    ang = math.radians(rot_deg)
    pts = []
    for (x, y) in src:
        x, y = x - sw / 2, y - sh / 2
        x, y = x * math.cos(ang) - y * math.sin(ang), x * math.sin(ang) + y * math.cos(ang)
        pts.append([cx + x, cy + y])
    dst = np.array(pts, np.float32)
    # perspective: the far edge (top) shrinks, plus random jitter
    dst[0, 0] += tilt * sw * 0.5
    dst[1, 0] -= tilt * sw * 0.5
    dst[0, 1] += tilt * sh * 0.3
    dst[1, 1] += tilt * sh * 0.3
    dst += rng.uniform(-0.015, 0.015, dst.shape).astype(np.float32) * max(sw, sh)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), M, (W, H))
    out = np.where(mask[..., None] > 0, warped.astype(np.float32), desk)
    # light: bright spot top-left, shadow bottom-right
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.05 - 0.35 * ((xx / W) * 0.6 + (yy / H) * 0.6)
    light += 0.12 * np.exp(-(((xx - 0.3 * W) ** 2 + (yy - 0.3 * H) ** 2) / (2 * (0.25 * W) ** 2)))
    out = out * light[..., None] * np.array([0.98, 1.0, 1.04], np.float32)   # warm cast
    out = cv2.GaussianBlur(out, (0, 0), 0.8)
    out += rng.normal(0, 4, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def make_pdf(img: np.ndarray, path: str, texts=None, dpi: float = 150.0) -> None:
    """Write ``img`` as a one-page PDF (JPEG image filling the page).

    ``texts``: optional list of (text, x_px, y_px_baseline, height_px, vertical)
    written as an invisible text layer (render mode 3), the way CAD exports
    and OCR'd scans carry selectable text."""
    h, w = img.shape[:2]
    k = 72.0 / dpi                                   # pt per px
    W, H = w * k, h * k
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    jpg = jpg.tobytes()
    content = [f"q {W:.2f} 0 0 {H:.2f} 0 0 cm /Im0 Do Q"]
    for (t, x, y, hh, vert) in texts or []:
        fs = hh * k * 1.35
        px, py = x * k, H - y * k
        if vert:     # rotated 90 degrees counter-clockwise
            content.append(f"BT 3 Tr /F1 {fs:.2f} Tf 0 1 -1 0 {px:.2f} {py:.2f} Tm ({t}) Tj ET")
        else:
            content.append(f"BT 3 Tr /F1 {fs:.2f} Tf 1 0 0 1 {px:.2f} {py:.2f} Tm ({t}) Tj ET")
    stream = "\n".join(content).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.2f} {H:.2f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> /Font << /F1 6 0 R >> >> /Contents 5 0 R >>").encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
         f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpg)} >>\nstream\n").encode() + jpg + b"\nendstream",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(out)


def dim_text_layer(gt: dict, px_per_mm: float = 0.1):
    """Text-layer entries for make_pdf from make_plan(dims=True) ground truth."""
    S = px_per_mm
    out = []
    for d in gt["dims"]:
        (ax, ay), (bx, by) = d["a"], d["b"]
        hh = 220 * S * 0.9
        if d["horiz"]:
            cx = (ax + bx) / 2
            out.append((d["text"], cx - len(d["text"]) * hh * 0.28, ay - 60 * S, hh, False))
        else:
            cy = (ay + by) / 2
            out.append((d["text"], ax - 60 * S, cy + len(d["text"]) * hh * 0.28, hh, True))
    return out
