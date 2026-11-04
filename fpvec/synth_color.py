"""Coloured floor plan generator (sales / marketing style) with ground truth.

Each seed gives a different apartment: a random room layout (recursive
splits), exterior and interior walls in one of several drawing styles,
coloured / textured floors per room type, swing doors, windows, furniture
blocks, room labels with areas and (often) dimension chains in mm or ft-in.

    img, gt = make_color_plan(seed)
    gt = {"mm_per_px", "walls": [{start, end, t, o, exterior}],  # px of img
          "openings": [{kind, start, end, o}], "style", "dims": bool, "units"}
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np

STYLES = ["sales", "wood", "blue", "grey_partitions", "brown_walls"]

FLOORS = {   # BGR
    "living": [(205, 230, 245), (170, 205, 230), (140, 180, 215)],   # beige / wood tones
    "bed": [(200, 215, 235), (215, 200, 230), (190, 225, 215)],
    "bath": [(235, 225, 200), (225, 225, 225), (230, 215, 190)],     # bluish / grey tiles
    "kitchen": [(200, 230, 230), (215, 215, 200)],
}
LABELS = {"living": "LIVING", "bed": "BEDROOM", "bath": "BATH", "kitchen": "KITCHEN"}


def _split(rect, rng, depth, min_size, out):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    if depth == 0 or (w < 2 * min_size and h < 2 * min_size):
        out.append(rect)
        return []
    vertical = w > h if abs(w - h) > 800 else rng.random() < 0.5
    if vertical and w < 2 * min_size:
        vertical = False
    if not vertical and h < 2 * min_size:
        vertical = True
    if vertical:
        s = round(rng.uniform(x0 + min_size, x1 - min_size) / 100) * 100
        lines = [("v", s, y0, y1)]
        lines += _split((x0, y0, s, y1), rng, depth - 1, min_size, out)
        lines += _split((s, y0, x1, y1), rng, depth - 1, min_size, out)
    else:
        s = round(rng.uniform(y0 + min_size, y1 - min_size) / 100) * 100
        lines = [("h", s, x0, x1)]
        lines += _split((x0, y0, x1, s), rng, depth - 1, min_size, out)
        lines += _split((x0, s, x1, y1), rng, depth - 1, min_size, out)
    return lines


def _wood(img, x0, y0, x1, y1, base, rng):
    """Plank texture."""
    cv2.rectangle(img, (x0, y0), (x1, y1), base, -1)
    step = max(4, int((y1 - y0) / max(1, (y1 - y0) // 9)))
    for y in range(y0, y1, step):
        d = int(rng.integers(-12, 12))
        col = tuple(int(np.clip(c + d, 0, 255)) for c in base)
        cv2.rectangle(img, (x0, y), (x1, min(y1, y + step - 1)), col, -1)
        cv2.line(img, (x0, y), (x1, y), tuple(int(c * 0.85) for c in base), 1)


def _tiles(img, x0, y0, x1, y1, base):
    cv2.rectangle(img, (x0, y0), (x1, y1), base, -1)
    k = max(6, int(min(x1 - x0, y1 - y0) / 8))
    line = tuple(int(c * 0.88) for c in base)
    for x in range(x0, x1, k):
        cv2.line(img, (x, y0), (x, y1), line, 1)
    for y in range(y0, y1, k):
        cv2.line(img, (x0, y), (x1, y), line, 1)


def make_color_plan(seed: int = 0, px_per_mm: float = None, style: str = None, dims: bool = None,
                    units: str = None) -> Tuple[np.ndarray, Dict]:
    rng = np.random.default_rng(1000 + seed)
    style = style or STYLES[seed % len(STYLES)]
    S = px_per_mm or float(rng.uniform(0.08, 0.12))
    dims = bool(rng.random() < 0.7) if dims is None else dims
    units = units or ("ftin" if rng.random() < 0.3 else "mm")
    W_mm = round(rng.uniform(9000, 14000) / 100) * 100
    H_mm = round(rng.uniform(7000, 11000) / 100) * 100
    Te = float(rng.choice([200, 240]))
    Ti = float(rng.choice([100, 120, 150]))
    margin = 1800 if dims else 900
    Wp, Hp = int((W_mm + 2 * margin) * S), int((H_mm + 2 * margin) * S)
    bg = (255, 255, 255) if style != "blue" else (250, 246, 240)
    img = np.full((Hp, Wp, 3), bg, np.uint8)
    P = lambda x, y: (int(round((x + margin) * S)), int(round((y + margin) * S)))
    wall_col = {"sales": (25, 25, 25), "wood": (60, 60, 60), "blue": (110, 60, 20),
                "grey_partitions": (20, 20, 20), "brown_walls": (40, 55, 95)}[style]

    # ---------------------------------------------------------------- layout
    rooms: List[tuple] = []
    splits = _split((0, 0, W_mm, H_mm), rng, int(rng.integers(2, 4)), 2400, rooms)
    kinds = ["living"] + list(rng.choice(["bed", "bed", "bath", "kitchen"], len(rooms) - 1))
    order = sorted(range(len(rooms)), key=lambda i: -(rooms[i][2] - rooms[i][0]) * (rooms[i][3] - rooms[i][1]))
    room_kind = {order[i]: kinds[i] for i in range(len(rooms))}

    # floors
    for i, (x0, y0, x1, y1) in enumerate(rooms):
        k = room_kind[i]
        base = FLOORS[k][int(rng.integers(len(FLOORS[k])))]
        a, b = P(x0, y0), P(x1, y1)
        if style in ("wood", "brown_walls") and k in ("living", "bed"):
            _wood(img, a[0], a[1], b[0], b[1], base, rng)
        elif k in ("bath", "kitchen") and style != "blue":
            _tiles(img, a[0], a[1], b[0], b[1], base)
        else:
            cv2.rectangle(img, a, b, base, -1)

    walls: List[dict] = []
    openings: List[dict] = []

    def add_wall(o, c, a, b, t, ext, ops):
        """Wall on axis o at c (mm) from a to b with openings [(kind, s0, w, hinge_at_start, side)]."""
        # record centreline to centreline (exterior walls are drawn past the corner by t/2)
        ra, rb = (max(a, 0.0), min(b, W_mm if o == "h" else H_mm)) if ext else (a, b)
        walls.append(dict(o=o, start=P(ra, c) if o == "h" else P(c, ra), end=P(rb, c) if o == "h" else P(c, rb),
                          t=t * S, exterior=ext))
        pos = a
        for kind, s0, w, hs, side in sorted(ops, key=lambda q: q[1]) + [(None, b, 0, 0, 0)]:
            if s0 > pos:
                p, q = (P(pos, c - t / 2), P(s0, c + t / 2)) if o == "h" else (P(c - t / 2, pos), P(c + t / 2, s0))
                if style == "grey_partitions" and not ext:
                    cv2.rectangle(img, p, q, (215, 215, 215), -1)
                    cv2.rectangle(img, p, q, (150, 150, 150), 1)
                else:
                    cv2.rectangle(img, p, q, wall_col, -1)
            if kind is None:
                break
            a0, a1 = (P(s0, c), P(s0 + w, c)) if o == "h" else (P(c, s0), P(c, s0 + w))
            if kind == "window":
                p, q = (P(s0, c - t / 2), P(s0 + w, c + t / 2)) if o == "h" else (P(c - t / 2, s0), P(c + t / 2, s0 + w))
                cv2.rectangle(img, p, q, (255, 255, 255), -1)
                cv2.rectangle(img, p, q, wall_col, 1)
                if o == "h":
                    cv2.line(img, (p[0], (p[1] + q[1]) // 2), (q[0], (p[1] + q[1]) // 2), wall_col, 1)
                else:
                    cv2.line(img, ((p[0] + q[0]) // 2, p[1]), ((p[0] + q[0]) // 2, q[1]), wall_col, 1)
            else:
                hinge_mm = s0 if hs else s0 + w
                latch_mm = s0 + w if hs else s0
                if o == "h":
                    hx, hy = P(hinge_mm, c + side * t / 2)
                    lx, ly = P(latch_mm, c + side * t / 2)
                    tip = (hx, int(round(hy + side * w * S)))
                else:
                    hx, hy = P(c + side * t / 2, hinge_mm)
                    lx, ly = P(c + side * t / 2, latch_mm)
                    tip = (int(round(hx + side * w * S)), hy)
                cv2.line(img, (hx, hy), tip, (60, 60, 60), 2)
                a0_ = math.degrees(math.atan2(ly - hy, lx - hx))
                a1_ = math.degrees(math.atan2(tip[1] - hy, tip[0] - hx))
                lo, hi = sorted((a0_, a1_))
                if hi - lo > 180:
                    lo, hi = hi, lo + 360
                cv2.ellipse(img, (hx, hy), (int(w * S), int(w * S)), 0, lo, hi, (90, 90, 90), 1, cv2.LINE_AA)
            openings.append(dict(kind=kind, start=a0, end=a1, o=o))
            pos = s0 + w

    # exterior walls with windows; front door on the bottom wall
    def ext_ops(length, n_win, door=False):
        ops, used = [], []
        if door:
            s = round(rng.uniform(800, length - 1900) / 100) * 100
            ops.append(("door", s, 1000, True, -1))
            used.append((s - 400, s + 1400))
        for _ in range(n_win):
            for _try in range(20):
                w = float(rng.choice([1200, 1500, 1800, 2100]))
                s = round(rng.uniform(600, length - w - 600) / 100) * 100
                if all(s + w + 400 < u0 or s - 400 > u1 for u0, u1 in used):
                    ops.append(("window", s, w, True, 1))
                    used.append((s, s + w))
                    break
        return ops

    # windows must not cross interior wall junctions
    def free_ops(o, c, length, n_win, door=False):
        ops = ext_ops(length, n_win, door)
        cuts = [sp[1] for sp in splits if sp[0] != o and sp[2] <= (c if c > 0 else 0) + 1 and sp[3] >= c - 1]
        cuts = [x for x in cuts if 0 < x < length]
        return [op for op in ops if all(not (op[1] - 300 < x < op[1] + op[2] + 300) for x in cuts)]

    add_wall("h", 0, -Te / 2, W_mm + Te / 2, Te, True, free_ops("h", 0, W_mm, 2))
    add_wall("h", H_mm, -Te / 2, W_mm + Te / 2, Te, True, free_ops("h", H_mm, W_mm, 1, door=True))
    add_wall("v", 0, Te / 2, H_mm - Te / 2, Te, True, free_ops("v", 0, H_mm, 1))
    add_wall("v", W_mm, Te / 2, H_mm - Te / 2, Te, True, free_ops("v", W_mm, H_mm, 1))

    # interior walls: each split line, cut where it crosses later split lines
    for o, c, a, b in splits:
        a0 = a + (Te / 2 if a == 0 else Ti / 2)
        b0 = b - (Te / 2 if b in (W_mm, H_mm) else Ti / 2)
        length = b0 - a0
        ops = []
        if length > 1600:
            w = float(rng.choice([700, 800, 900]))
            crosses = sorted(sp[1] for sp in splits if sp[0] != o and sp[2] < c < sp[3] and a0 < sp[1] < b0)
            # place the door in a stretch between crossings
            stretches = [(x, y) for x, y in zip([a0] + crosses, crosses + [b0]) if y - x > w + 500]
            if stretches:
                x, y = stretches[int(rng.integers(len(stretches)))]
                s = round(rng.uniform(x + 200, y - w - 200) / 50) * 50
                ops.append(("door", s, w, bool(rng.random() < 0.5), int(rng.choice([-1, 1]))))
        # draw from a0 to b0 (relative positions) using absolute coords
        add_wall(o, c, a0, b0, Ti, False, ops)

    # furniture blocks (coloured, inside rooms, away from doors)
    fcols = [(140, 160, 190), (120, 140, 120), (170, 150, 200), (100, 120, 150)]
    for i, (x0, y0, x1, y1) in enumerate(rooms):
        k = room_kind[i]
        if k in ("bed", "living") and (x1 - x0) > 3000 and (y1 - y0) > 3000:
            fw, fh = (1600, 2000) if k == "bed" else (2200, 900)
            fx, fy = (x0 + x1) / 2 - fw / 2, y0 + Te + 400
            a, b = P(fx, fy), P(fx + fw, fy + fh)
            col = fcols[int(rng.integers(len(fcols)))]
            cv2.rectangle(img, a, b, col, -1)
            cv2.rectangle(img, a, b, (70, 70, 70), 1)
            if k == "bed":
                cv2.rectangle(img, a, (b[0], a[1] + int(350 * S)), (240, 240, 240), -1)
        # label + area
        cx, cy = P((x0 + x1) / 2, (y0 + y1) / 2 + 300)
        fs = max(0.35, 180 * S / 22)
        lab = LABELS[k]
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(img, lab, (cx - tw // 2, cy), cv2.FONT_HERSHEY_SIMPLEX, fs, (60, 60, 60), 1, cv2.LINE_AA)
        area = (x1 - x0) * (y1 - y0) / 1e6
        t2 = f"{area:.1f} m2"
        (tw, _), _ = cv2.getTextSize(t2, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, 1)
        cv2.putText(img, t2, (cx - tw // 2, cy + int(th * 1.8)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (90, 90, 90), 1, cv2.LINE_AA)

    # dimension chains (top: wall stops; left: wall stops)
    if dims:
        from .synth import _fmt_dim
        font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 220 * S / 22), 1
        ink = (40, 40, 40)

        def chain(stops, horiz, off):
            for v in stops:
                x, y = P(v, -off) if horiz else P(-off, v)
                if horiz:
                    cv2.line(img, (x, P(0, -Te)[1]), (x, y - int(120 * S)), ink, 1)
                else:
                    cv2.line(img, (P(-Te, 0)[0], y), (x - int(120 * S), y), ink, 1)
                k = int(90 * S)
                cv2.line(img, (x - k, y + k), (x + k, y - k), ink, 2)
            a = P(stops[0], -off) if horiz else P(-off, stops[0])
            b = P(stops[-1], -off) if horiz else P(-off, stops[-1])
            cv2.line(img, a, b, ink, 1)
            for s0, s1 in zip(stops, stops[1:]):
                txt = _fmt_dim(s1 - s0, units)
                (tw, tht), _ = cv2.getTextSize(txt, font, fs, th)
                if (s1 - s0) * S < tw * 1.15:
                    continue
                if horiz:
                    cx, cy = P((s0 + s1) / 2, -off)
                    cv2.putText(img, txt, (cx - tw // 2, cy - int(60 * S)), font, fs, ink, th, cv2.LINE_AA)
                else:
                    cx, cy = P(-off, (s0 + s1) / 2)
                    patch = np.full((tht + 10, tw + 6, 3), 255, np.uint8)
                    cv2.putText(patch, txt, (3, tht + 3), font, fs, ink, th, cv2.LINE_AA)
                    patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ph, pw = patch.shape[:2]
                    x1_, y1_ = cx - int(60 * S) - pw, cy - ph // 2
                    if x1_ >= 0 and y1_ >= 0:
                        reg = img[y1_:y1_ + ph, x1_:x1_ + pw]
                        np.minimum(reg, patch, out=reg)

        xs = sorted({0, W_mm} | {sp[1] for sp in splits if sp[0] == "v" and sp[2] == 0})
        ys = sorted({0, H_mm} | {sp[1] for sp in splits if sp[0] == "h" and sp[2] == 0})
        chain(xs, True, 800)
        chain([0, W_mm], True, 1400)
        chain(ys, False, 800)

    gt = dict(mm_per_px=1 / S, walls=walls, openings=openings, style=style, dims=dims, units=units,
              size_mm=(W_mm, H_mm), ext_t=Te, int_t=Ti)
    return img, gt
