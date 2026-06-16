"""
tests/test_color_modes.py
-------------------------
The sequence-derivable color registry (the SINGLE source for "what color is this
residue"): each mode's per-residue opinion, the His-broken-out-from-K/R charge nuance,
and the OFF mode + gap/unknown safety (color_for → None).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from color_modes import (get_mode, all_modes, COLOR_MODES, plddt_color,
                         deviation_color)


class TestDeviationColor:
    """S4c floor-gated variant-vs-WT Cα deviation (Å) → fixed-Å hex (one source for
    the panel AND the predicted-model 3D push; never paints sub-floor sampling noise)."""

    def test_none_is_no_data(self):
        assert deviation_color(None, 0.25) is None

    def test_within_floor_is_neutral(self):
        # at-or-below the floor → None (NEUTRAL), regardless of being a non-zero shift
        assert deviation_color(0.3, 0.5) is None
        assert deviation_color(0.5, 0.5) is None          # exactly on the floor → neutral

    def test_above_floor_colours_by_magnitude(self):
        assert deviation_color(0.6, 0.5) == "#8fd0e8"     # cleared 0.5 floor; 0.5–1.0 → cyan
        assert deviation_color(0.4, 0.25) == "#5b8def"    # cleared 0.25 floor, <0.5 → cool blue
        assert deviation_color(1.5, 0.25) == "#ffd166"    # 1.0–2.0 → yellow
        assert deviation_color(3.0, 0.25) == "#f3953b"    # 2.0–3.5 → orange
        assert deviation_color(4.0, 0.25) == "#e23b3b"    # ≥3.5 → red

    def test_fixed_A_bands_not_percentile(self):
        # absolute-Å thresholds (0.5/1.0/2.0/3.5) — independent of any distribution, so the
        # panel cell colour equals the 3D model colour for the same residue (sync invariant)
        assert deviation_color(0.49, 0.0) == "#5b8def"
        assert deviation_color(0.99, 0.0) == "#8fd0e8"
        assert deviation_color(1.99, 0.0) == "#ffd166"

    def test_default_floor_zero_still_gates_zero_shift(self):
        assert deviation_color(0.0, 0.0) is None           # no motion → neutral


class TestPlddtColor:
    """S4b per-residue pLDDT → AlphaFold-palette hex (banded, mirrors `palette alphafold`)."""

    def test_none_is_no_data(self):
        assert plddt_color(None) is None

    def test_bands(self):
        assert plddt_color(95) == "#0053d6"   # >90  very high (dark blue)
        assert plddt_color(80) == "#65cbf3"   # 70-90 confident (light blue)
        assert plddt_color(60) == "#ffdb13"   # 50-70 low (yellow)
        assert plddt_color(30) == "#ff7d45"   # <50  very low (orange)

    def test_band_edges_are_strict_lower_bounds(self):
        # exactly on a band edge falls to the lower band (>, not >=)
        assert plddt_color(90) == "#65cbf3"
        assert plddt_color(70) == "#ffdb13"
        assert plddt_color(50) == "#ff7d45"


class TestRegistry:
    def test_none_is_first_and_off(self):
        assert all_modes()[0].key == "none"
        none = get_mode("none")
        assert none.fn is None
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            assert none.color_for(aa) is None

    def test_unknown_key_falls_back_to_none(self):
        assert get_mode("nope").key == "none"

    def test_gap_and_unknown_residue_have_no_color(self):
        charge = get_mode("charge")
        assert charge.color_for(None) is None      # gap
        assert charge.color_for("-") is None       # gap glyph
        assert charge.color_for("X") is None        # non-standard
        assert charge.color_for("") is None


class TestCharge:
    def test_strong_cation_blue_weak_his_distinct(self):
        c = get_mode("charge")
        assert c.color_for("K") == c.color_for("R")            # K, R same strong blue
        assert c.color_for("H") not in (None, c.color_for("K"))  # His its OWN (weak) shade
        assert c.color_for("D") == c.color_for("E")            # D, E same red
        assert c.color_for("D") != c.color_for("K")            # red != blue
        assert c.color_for("A") is None                         # neutral → reset


class TestOtherModes:
    def test_cysteine_only_cys(self):
        cys = get_mode("cysteine")
        assert cys.color_for("C") is not None
        assert all(cys.color_for(a) is None for a in "AKRDE")

    def test_aromatic_large_rings_only(self):
        ar = get_mode("aromatic")
        assert ar.color_for("F") == ar.color_for("W") == ar.color_for("Y")
        assert ar.color_for("F") is not None
        assert ar.color_for("H") is None          # His left out (small/borderline)
        assert ar.color_for("A") is None

    def test_hydrophobicity_scales_hydrophobic_darker(self):
        h = get_mode("hydrophobicity")
        # every standard residue gets a green; the most hydrophobic (I) is darker than
        # the most hydrophilic (R) — lower summed RGB = darker.
        def lum(hexc):
            return int(hexc[1:3], 16) + int(hexc[3:5], 16) + int(hexc[5:7], 16)
        assert lum(h.color_for("I")) < lum(h.color_for("R"))
        assert h.color_for("R").lower() in ("#ffffff",) or lum(h.color_for("R")) > 700


class TestRegistryShape:
    def test_keys_match(self):
        assert set(COLOR_MODES) == {"none", "hydrophobicity", "charge", "cysteine", "aromatic"}
