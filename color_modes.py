"""
color_modes.py
--------------
Sequence-derivable color modes for the Variant-Design Workbench (Stage 2). PURE —
no Qt, no ChimeraX; fully unit-testable. The SINGLE SOURCE OF TRUTH for "what color
is this residue under mode M": one ``aa (1-letter) -> hex | None`` function per mode,
used by BOTH the Qt panel (cell backgrounds) AND the 3D color-over-REST push
(`variant_model.build_color_commands`), so the panel and the structure cannot drift.

These modes derive purely from residue IDENTITY — no tool runs (Stage 2 scope). They
are a sequence-PROPERTY view; coloring the 3D by them on the shared backbone is a
PREVIEW (color-by-identity), not a claim the structure was remodeled (S4 folds).

``color_for`` returns ``None`` for "no opinion" (gap / unknown residue / the neutral
baseline), which downstream renders as the reset color (white). Mode ``"none"`` is the
explicit OFF switch (color_for is None) — the panel clears its overlay and the 3D is
left untouched (non-destructive; we do not know the pre-overlay coloring to restore).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

# Standard-20 set (the codes a sequence row can hold); anything else → no opinion.
_STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ── hydrophobicity (Kyte-Doolittle) → green shades ─────────────────────────────────
# Higher KD = more hydrophobic. We render hydrophobic = deep green, hydrophilic = white
# (interpolated), so the green INTENSITY reads as the hydrophobicity.
_KD = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8,
    "G": -0.4, "T": -0.7, "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6,
    "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}
_KD_LO, _KD_HI = -4.5, 4.5
_GREEN = (27, 122, 27)        # deep green = most hydrophobic
_WHITE = (255, 255, 255)      # white = most hydrophilic


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def _hydrophobicity(aa: str) -> Optional[str]:
    kd = _KD.get(aa)
    if kd is None:
        return None
    t = (kd - _KD_LO) / (_KD_HI - _KD_LO)          # 0 (hydrophilic) … 1 (hydrophobic)
    rgb = tuple(_WHITE[i] + t * (_GREEN[i] - _WHITE[i]) for i in range(3))
    return _hex(rgb)


# ── charge (His broken out — only weakly/partly cationic at physiological pH) ───────
_POS_STRONG = {"K", "R"}      # firmly cationic → blue
_POS_WEAK   = {"H"}           # borderline (~10% protonated at pH 7.4) → its OWN faint blue
_NEG        = {"D", "E"}      # anionic → red
_C_BLUE      = "#2a6fdb"
_C_BLUE_WEAK = "#9db8e8"
_C_RED       = "#e23b3b"


def _charge(aa: str) -> Optional[str]:
    if aa in _POS_STRONG:
        return _C_BLUE
    if aa in _POS_WEAK:
        return _C_BLUE_WEAK
    if aa in _NEG:
        return _C_RED
    return None                                     # neutral → reset (white)


# ── cysteine ────────────────────────────────────────────────────────────────────
def _cysteine(aa: str) -> Optional[str]:
    return "#ffd700" if aa == "C" else None         # gold-yellow


# ── aromatic (the large aromatic rings; His left out — small/borderline) ───────────
def _aromatic(aa: str) -> Optional[str]:
    return "#9b59b6" if aa in {"F", "W", "Y"} else None   # purple


# ── result-backed scales (Stage 4): per-residue VALUE → hex (not aa-keyed) ──────────
# These differ from the modes above: they color by a per-residue computed VALUE (ddG,
# pLDDT, deviation) rather than residue identity, so they live as value→hex scales used
# by the panel cells AND the 3D push for a variant that has results. Kept here so the two
# surfaces share one source (the S2 sync invariant, extended to result modes).
_DDG_NEUTRAL = 1.0          # |ddG| within ~1 kcal/mol ≈ method noise → white
_DDG_SAT     = 2.5          # saturates to full blue/red beyond ±2.5 kcal/mol
_DDG_BLUE = (42, 111, 219)  # stabilizing (ddG < 0)
_DDG_RED  = (226, 59, 59)   # destabilizing (ddG > 0)


def ddg_color(ddg: Optional[float]) -> Optional[str]:
    """Diverging blue–white–red for a SIGNED ddG (system convention: positive =
    destabilizing). None → no data (renders as reset). Inside the neutral band
    (|ddG| ≤ 1 kcal/mol ≈ method noise) → white; magnitude saturates the hue by ±2.5."""
    if ddg is None:
        return None
    if abs(ddg) <= _DDG_NEUTRAL:
        return "#ffffff"
    t = min(1.0, (abs(ddg) - _DDG_NEUTRAL) / (_DDG_SAT - _DDG_NEUTRAL))
    end = _DDG_RED if ddg > 0 else _DDG_BLUE
    rgb = tuple(_WHITE[i] + t * (end[i] - _WHITE[i]) for i in range(3))
    return _hex(rgb)


@dataclass(frozen=True)
class ColorMode:
    key:   str                              # stable id
    label: str                              # combo display
    fn:    Optional[Callable[[str], Optional[str]]]   # aa -> hex|None; None for "none"

    def color_for(self, aa: Optional[str]) -> Optional[str]:
        """Hex color for a residue, or None (gap/unknown/neutral → reset). The OFF
        mode ('none', fn is None) always returns None."""
        if self.fn is None or not aa or aa not in _STD_AA:
            return None
        return self.fn(aa)


# Registry — order = combo order. "none" first (the default OFF state).
_MODES: List[ColorMode] = [
    ColorMode("none",           "None",            None),
    ColorMode("hydrophobicity", "Hydrophobicity",  _hydrophobicity),
    ColorMode("charge",         "Charge",          _charge),
    ColorMode("cysteine",       "Cysteine",        _cysteine),
    ColorMode("aromatic",       "Aromatic",        _aromatic),
]
COLOR_MODES: Dict[str, ColorMode] = {m.key: m for m in _MODES}


def all_modes() -> List[ColorMode]:
    """Modes in display order (the combo population order)."""
    return list(_MODES)


def get_mode(key: str) -> ColorMode:
    """Look up a mode by key; the OFF mode ('none') for any unknown key."""
    return COLOR_MODES.get(key, COLOR_MODES["none"])
