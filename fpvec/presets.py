"""Ready-made settings: a preparation recipe + detection tuning.

Each preset = {"label", "help", "prep": {...prepare.py recipe...},
"tuning": {...Params fields...}}. The demo page lists them; the CLI takes
``--preset NAME``. Values were chosen on the sample set (see samples/).
"""
from __future__ import annotations

from typing import Dict

PRESETS: Dict[str, dict] = {
    "bw": {
        "label": "Black & white (recommended)",
        "help": "Black & white, lines darkened, background whitened. Best overall on the sample set: "
                "fewest false doors and windows.",
        "prep": {"grayscale": True, "black": 110, "white": 210, "lines": 0},
        "tuning": {"sensitivity": "auto", "plain_openings": "between_walls"},
    },
    "faint": {
        "label": "Faint / grey lines",
        "help": "Pulls light-grey walls, door arcs and window glazing to black. For washed-out or very light drawings.",
        "prep": {"grayscale": True, "black": 130, "white": 225, "lines": 15},
        "tuning": {"sensitivity": "sensitive", "plain_openings": "between_walls"},
    },
    "colour": {
        "label": "Colour plan, textured floors",
        "help": "Whitens coloured and textured floors (wood, tiles) harder, so only the drawing's lines remain.",
        "prep": {"grayscale": True, "black": 110, "white": 195, "lines": 0},
        "tuning": {"sensitivity": "auto", "plain_openings": "between_walls"},
    },
    "clean": {
        "label": "Clean black & white drawing",
        "help": "For crisp CAD output: black & white only, nothing else changed.",
        "prep": {"grayscale": True, "black": 0, "white": 255, "lines": 0},
        "tuning": {"sensitivity": "strict", "plain_openings": "between_walls"},
    },
    "original": {
        "label": "Original (no preparation)",
        "help": "Uses the corrected image as it is, in colour.",
        "prep": {},
        "tuning": {"sensitivity": "auto", "plain_openings": "between_walls"},
    },
}

DEFAULT_PRESET = "bw"

# detection settings the user may change, with their range and meaning (for the UI)
TUNING = {
    "items": {"label": "Detect", "type": "choice", "choices": ["all", "walls_doors_windows", "walls_openings", "walls"],
              "default": "all",
              "help": "Everything; walls, doors and windows (no plain openings); walls and openings (every gap as a "
                      "plain opening, no door / window types); or walls only (walls run through the gaps)."},
    "sensitivity": {"label": "Thin-line sensitivity", "type": "choice", "choices": ["auto", "strict", "sensitive"],
                    "default": "auto",
                    "help": "How faint a line may be and still count. Strict ignores anti-aliasing and textures; "
                            "sensitive finds light-grey partitions and glazing."},
    "plain_openings": {"label": "Plain openings", "type": "choice", "choices": ["between_walls", "all", "off"],
                       "default": "between_walls",
                       "help": "Gaps with no door or window drawn: only where the wall continues on both sides, "
                               "any gap, or never."},
    "detect_thin_walls": {"label": "Double-line / grey walls", "type": "bool", "default": True,
                          "help": "Find walls drawn as two thin lines or as a grey band, not only filled ones."},
    "door_score": {"label": "Door certainty", "type": "range", "min": 0.4, "max": 0.9, "step": 0.05, "default": 0.6,
                   "help": "Lower finds more doors (and more false ones)."},
    "min_opening_mm": {"label": "Smallest opening (mm)", "type": "range", "min": 300, "max": 1200, "step": 50,
                       "default": 450, "help": "Narrower gaps in a wall are ignored."},
    "max_opening_mm": {"label": "Widest opening (mm)", "type": "range", "min": 1500, "max": 8000, "step": 100,
                       "default": 4500, "help": "Wider gaps are not treated as one door / window."},
    "min_wall_t": {"label": "Shortest wall (× thickness)", "type": "range", "min": 1.0, "max": 4.0, "step": 0.1,
                   "default": 1.6, "help": "Shorter solid pieces are ignored (removes blobs and stubs)."},
    "wall_thickness_px": {"label": "Wall thickness (px)", "type": "number", "default": None,
                          "help": "Leave empty to measure it automatically."},
    "wall_thicknesses": {"label": "Wall thicknesses (mm)", "type": "text", "default": "",
                         "placeholder": "e.g. 100-250 or 100, 150, 230",
                         "help": "Allowed wall thicknesses: a range (every wall is kept inside it) or a list (every "
                                 "wall gets the nearest value). Empty = as measured."},
    "ortho": {"label": "Ortho walls", "type": "bool", "default": True,
              "help": "Make every wall exactly horizontal or vertical and put nearly-aligned walls on one line."},
    "snap_ends": {"label": "Join wall ends", "type": "bool", "default": True,
                  "help": "Extend a loose wall end to the nearest wall within the join distance: straight on "
                          "across a gap (walls only closes doorways this way), into a wall it runs at (T), or "
                          "to meet a nearby wall end (L corner)."},
    "snap_dist": {"label": "Join distance", "type": "number", "default": 3.0,
                  "help": "How far a wall end may be extended. Unit below."},
    "snap_unit": {"label": "Join distance unit", "type": "choice", "choices": ["t", "pct", "mm"], "default": "t",
                  "help": "t = multiples of the wall thickness, pct = percent of the plan size, mm = millimetres "
                          "(uses the scale)."},
}


def spec_ok(k: str, v) -> bool:
    if k == "wall_thicknesses":
        from .tidy import parse_thicknesses
        return parse_thicknesses(v) is not None
    return True


def tuning_params(t: dict) -> dict:
    """Keep only known tuning keys with sensible types."""
    out = {}
    for k, v in (t or {}).items():
        spec = TUNING.get(k)
        if spec is None or v is None or v == "":
            continue
        if spec["type"] == "bool":
            out[k] = bool(v)
        elif spec["type"] == "text":
            if spec_ok(k, v):
                out[k] = str(v).strip()
        elif spec["type"] == "choice":
            if v in spec["choices"]:
                out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out
