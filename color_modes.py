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


# pLDDT confidence → hex (S4b result-backed scale; per-residue VALUE, not aa-keyed).
# Banded to the canonical AlphaFold/ESMFold palette so the panel mirrors the 3D
# `color byattribute bfactor … palette alphafold` push (panel↔3D sync invariant).
_PLDDT_VERY_HIGH = "#0053d6"   # > 90  very high  (dark blue)
_PLDDT_HIGH      = "#65cbf3"   # 70–90 confident  (light blue)
_PLDDT_LOW       = "#ffdb13"   # 50–70 low        (yellow)
_PLDDT_VERY_LOW  = "#ff7d45"   # < 50  very low   (orange)


def plddt_color(plddt: Optional[float]) -> Optional[str]:
    """Per-residue pLDDT (0–100) → AlphaFold-palette hex. None → no data (reset).
    Banded (>90 / 70–90 / 50–70 / <50) to match ChimeraX's native `palette alphafold`
    so the workbench panel color equals the 3D color for the same residue."""
    if plddt is None:
        return None
    if plddt > 90:
        return _PLDDT_VERY_HIGH
    if plddt > 70:
        return _PLDDT_HIGH
    if plddt > 50:
        return _PLDDT_LOW
    return _PLDDT_VERY_LOW


# ── variant-vs-WT Cα deviation (S4c result-backed scale; per-residue VALUE in Å) ────
# Floor-GATED magnitude ramp: a residue whose deviation does NOT clear its noise floor
# renders NEUTRAL (None → reset), so we never paint sampling noise as a result (§0 honest
# rendering). Residues that DO clear the floor are coloured by ABSOLUTE deviation on a
# fixed-Å cool→hot ramp (small real motion = cool, large = red). The thresholds are the
# same fixed-Å bands as tool_router's `_DEVIATION_BUCKETS` (0.5 / 1.0 / 2.0 / 3.5 Å) —
# NOT adaptive percentiles — so the panel cell colour equals the 3D model colour for the
# same residue (the S2 panel↔3D sync invariant, one source for both).
_DEV_BANDS = (
    (0.5, "#5b8def"),   # just cleared the floor — small real motion (cool blue)
    (1.0, "#8fd0e8"),   # cyan
    (2.0, "#ffd166"),   # yellow
    (3.5, "#f3953b"),   # orange
    (float("inf"), "#e23b3b"),   # ≥3.5 Å — large displacement (red)
)


def deviation_color(deviation: Optional[float],
                    floor: float = 0.0) -> Optional[str]:
    """Per-residue variant-vs-WT Cα deviation (Å) → fixed-Å hex, FLOOR-GATED.

    None → no data (reset). ``deviation <= floor`` → None (NEUTRAL: the residue moved
    no more than the same-sequence WT moves across seeds at that position, so the shift
    is not distinguishable from sampling noise — do not colour it as signal). Above the
    floor → a fixed-Å cool→hot magnitude band (small real motion = cool, large = red).
    *floor* is the residue's effective noise floor (≈0 + a global minimum for a
    deterministic engine; the measured cross-seed WT max for a stochastic one)."""
    if deviation is None:
        return None
    if deviation <= floor:
        return None
    for hi, hexc in _DEV_BANDS:
        if deviation < hi:
            return hexc
    return _DEV_BANDS[-1][1]


# ── disulfide-ENGINEERABILITY heatmap (Mode D; per-residue best-partner score in [0,1]) ──────
# A residue's colour = its BEST available partner's engineerability score. The heatmap is a
# NAVIGATIONAL INDEX (a glowing residue points you to the ranked pair-list, which is the data) —
# it does NOT promise the mutation is tolerated or the bond will form (caveat rides on the mode).
_SS_SCAN_MIN = 0.05                 # below this = not engineerable → neutral (no colour)
_SS_PALE = (0xe8, 0xe2, 0xc0)       # faint gold (a just-viable site)
_SS_GOLD = (0xf2, 0xa6, 0x1c)       # saturated gold (a best-fit site glows strongest)


def disulfide_compat_color(score: Optional[float]) -> Optional[str]:
    """Per-residue disulfide-engineerability (best-partner score 0–1) → pale→gold hex. None when
    *score* is None or below the surfacing threshold (not engineerable → no colour, never painted
    as a result). Best-fit sites saturate the gold."""
    if score is None or score < _SS_SCAN_MIN:
        return None
    t = min(1.0, (score - _SS_SCAN_MIN) / (1.0 - _SS_SCAN_MIN))
    rgb = tuple(_SS_PALE[i] + t * (_SS_GOLD[i] - _SS_PALE[i]) for i in range(3))
    return _hex(rgb)


# ── proline-stabilization heatmap (per-residue X→Pro favourability score in [0,1]) ────────────────
# A residue's colour = its proline-substitution score (φ/ψ proline-compatibility × H-bond-donor
# penalty). NAVIGATIONAL INDEX into the ranked candidate list (the data); the caveat rides on the
# mode. Pale→magenta (proline's traditional highlight hue, distinct from the disulfide gold).
_PRO_SCAN_MIN = 0.05                 # below this = not proline-favourable → neutral (no colour)
_PRO_PALE = (0xe8, 0xd0, 0xe8)       # faint magenta (a just-favourable site)
_PRO_MAGENTA = (0xc0, 0x1c, 0xc0)    # saturated magenta (a best proline site glows strongest)


def proline_compat_color(score: Optional[float]) -> Optional[str]:
    """Per-residue proline-stabilization favourability (score 0–1) → pale→magenta hex. None when
    *score* is None or below the surfacing threshold (not favourable → no colour, never painted as a
    result). The best φ/ψ-compatible non-H-bond-donor sites saturate the magenta."""
    if score is None or score < _PRO_SCAN_MIN:
        return None
    t = min(1.0, (score - _PRO_SCAN_MIN) / (1.0 - _PRO_SCAN_MIN))
    rgb = tuple(_PRO_PALE[i] + t * (_PRO_MAGENTA[i] - _PRO_PALE[i]) for i in range(3))
    return _hex(rgb)


# ── cavity-filling heatmap (per-residue best small→larger FILL score in [0,1]) ────────────────────
# A cavity-lining residue's colour = its best fill score (void-fill fraction × rotamer reach-into-void,
# soft-demoted on clash). NAVIGATIONAL INDEX into the ranked candidate list (the data); the caveat
# rides on the mode. Teal (a just-viable lining fill) → gold (a best fill glows strongest) — distinct
# from the disulfide gold-only ramp and the proline magenta. The threshold matches the scan's surfacing
# floor so a fill that didn't surface is never painted.
_CAV_SCAN_MIN = 0.02                 # below this = no viable surfaced fill → neutral (no colour)
_CAV_TEAL = (0x2c, 0xa0, 0xa0)       # teal (a just-viable lining fill)
_CAV_GOLD = (0xf2, 0xa6, 0x1c)       # saturated gold (a best fill glows strongest)


def cavity_compat_color(score: Optional[float]) -> Optional[str]:
    """Per-residue cavity-fill favourability (best fill score 0–1) → teal→gold hex. None when *score*
    is None or below the surfacing threshold (no viable fill → no colour, never painted as a result).
    The best void-filling sites saturate the gold."""
    if score is None or score < _CAV_SCAN_MIN:
        return None
    t = min(1.0, (score - _CAV_SCAN_MIN) / (1.0 - _CAV_SCAN_MIN))
    rgb = tuple(_CAV_TEAL[i] + t * (_CAV_GOLD[i] - _CAV_TEAL[i]) for i in range(3))
    return _hex(rgb)


# ── salt-bridge heatmap (per-residue best charge-pair score in [0,1]) ──────────────────────────────
# A residue's colour = its best salt-bridge score (existing-pair quality OR novel charge-pair
# engineerability — geometry × burial). NAVIGATIONAL INDEX into the ranked tables (the data); the
# context-dependent desolvation caveat rides on the mode. Pale → saturated ELECTRIC BLUE — reads as
# "charge / electrostatic", distinct from the disulfide/cavity gold, the proline magenta, the cavity
# teal. The threshold matches the scan's surfacing floor so a sub-threshold residue is never painted.
_SB_SCAN_MIN = 0.05                  # below this = not salt-bridge-favourable → neutral (no colour)
_SB_PALE = (0xc8, 0xd8, 0xf2)        # pale blue (a just-favourable site)
_SB_BLUE = (0x1e, 0x46, 0xd8)        # saturated electric blue (a best charge-pair site glows strongest)


def saltbridge_compat_color(score: Optional[str]) -> Optional[str]:
    """Per-residue salt-bridge favourability (best charge-pair score 0–1) → pale→electric-blue hex.
    None when *score* is None or below the surfacing threshold (not favourable → no colour, never
    painted as a result). The best buried, clash-free, ideal-geometry sites saturate the blue."""
    if score is None or score < _SB_SCAN_MIN:
        return None
    t = min(1.0, (score - _SB_SCAN_MIN) / (1.0 - _SB_SCAN_MIN))
    rgb = tuple(_SB_PALE[i] + t * (_SB_BLUE[i] - _SB_PALE[i]) for i in range(3))
    return _hex(rgb)


# Cα-lDDT disruption ramp — by lDDT VALUE (1 = locally conserved … 0 = fully changed), so the
# scale is INVERTED vs deviation: LOWER lDDT = MORE disrupted = warmer. Floor-gated like the
# deviation ramp (lDDT ≥ floor → neutral: changed no more than the WT does across seeds).
_LDDT_BANDS = (
    (0.50, "#e23b3b"),   # < 0.50 — severe local change (red)
    (0.65, "#f3953b"),   # orange
    (0.80, "#ffd166"),   # yellow
    (0.90, "#8fd0e8"),   # cyan — mild
)


# dRMSD (all-pairs distance-RMSD, Å) ramp — the PAINTED disruption signal. Magnitude in Å like
# the old deviation ramp (cool→hot, floor-gated), but the value is SUPERPOSITION-FREE so it is
# honest for rigidly-displaced-but-intact structure (lights up) and a whole-body move (stays
# neutral). Bands are wider than the Cα ramp because dRMSD aggregates many pairwise changes.
_DDM_BANDS = (
    (1.0, "#5b8def"),   # just cleared — small relative change (cool blue)
    (2.5, "#8fd0e8"),   # cyan
    (5.0, "#ffd166"),   # yellow
    (8.0, "#f3953b"),   # orange
    (float("inf"), "#e23b3b"),   # ≥8 Å — large rearrangement (red)
)


def ddm_color(ddm: Optional[float], floor: float = 0.5) -> Optional[str]:
    """Per-residue dRMSD (Å, superposition-free all-pairs distance-RMSD) → fixed-Å hex,
    FLOOR-GATED. None → no data. ``ddm <= floor`` → None (NEUTRAL: changed no more than the WT
    does across seeds). Above the floor → cool→hot by magnitude. Captures rigid DISPLACEMENT of
    intact structure (which lDDT misses) without the anchor/lever-arm blow-up of the Cα-RMSD
    ramp; a whole-body rigid move preserves all distances → 0 → neutral."""
    if ddm is None:
        return None
    if ddm <= floor:
        return None
    for hi, hexc in _DDM_BANDS:
        if ddm < hi:
            return hexc
    return _DDM_BANDS[-1][1]


def lddt_disruption_color(lddt: Optional[float],
                          floor: float = 0.9) -> Optional[str]:
    """Per-residue Cα-lDDT (variant-vs-WT, 1=locally conserved) → fixed hex, FLOOR-GATED.

    None → no data (reset). ``lddt >= floor`` → None (NEUTRAL: the local structure changed no
    more than the same-sequence WT changes across seeds → not distinguishable from sampling
    noise). Below the floor → a warm-by-severity band (lower lDDT = redder). *floor* is the
    residue's cross-seed lDDT noise floor (capped at the neutral cap; a deterministic engine
    has no cross-seed spread so the cap alone gates). SUPERPOSITION-FREE → a re-orienting domain
    stays conserved (high lDDT) instead of the lever-arm blow-up the Cα-RMSD ramp produced."""
    if lddt is None:
        return None
    if lddt >= floor:
        return None
    for hi, hexc in _LDDT_BANDS:
        if lddt < hi:
            return hexc
    return _LDDT_BANDS[-1][1]


# "diverges from the template but WITHIN the WT reference's own cross-seed noise" — a muted slate
# grey, distinct from the white baseline (confident tight fit) AND the cool→hot disruption ramp.
_UNCERTAIN_HEX = "#8a8f99"


def combined_disruption_color(ddm: Optional[float], floor_ddm: float,
                              lddt: Optional[float], floor_lddt: float,
                              min_a: float = 0.5) -> Optional[str]:
    """The PAINTED 'deviation vs WT' colour — a 3-TIER encoding of magnitude × confidence:

      • CONFIDENT disruption — beyond the WT's cross-seed noise in EITHER global position
        (dRMSD > floor_ddm) OR local structure (lDDT < floor_lddt) → the cool→hot ramp by dRMSD
        magnitude (a residue shown only via lDDT still colours by its displacement so the scale
        stays one thing). This is a real variant-driven change.
      • DISTINCT-BUT-UNCERTAIN — diverges from the template (dRMSD > min_a) but within that noise
        → a muted GREY: 'the variant differs here, but the WT reference is itself this variable —
        not a confident effect.'
      • ALIGNED — dRMSD ≤ min_a → None (white baseline: a tight, confident fit / rigid region).

    The single source for the panel cells and the 3D push."""
    confident = (ddm is not None and ddm > floor_ddm) or (lddt is not None and lddt < floor_lddt)
    if confident:
        if ddm is not None:
            return ddm_color(ddm, 0.0) or "#5b8def"  # magnitude by displacement (gate passed)
        return lddt_disruption_color(lddt, 1.0)      # dRMSD missing → lDDT severity
    if ddm is not None and ddm > min_a:
        return _UNCERTAIN_HEX                         # distinct from template, within WT noise
    return None                                      # aligned / rigid → neutral (white)


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
