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
}


def tuning_params(t: dict) -> dict:
    """Keep only known tuning keys with sensible types."""
    out = {}
    for k, v in (t or {}).items():
        spec = TUNING.get(k)
        if spec is None or v is None or v == "":
            continue
        if spec["type"] == "bool":
            out[k] = bool(v)
        elif spec["type"] == "choice":
            if v in spec["choices"]:
                out[k] = v
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                pass
    return out
