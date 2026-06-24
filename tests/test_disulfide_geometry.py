"""
tests/test_disulfide_geometry.py
--------------------------------
The SHARED disulfide geometry core (disulfide_geometry): the lifted primitives, the χSS
dihedral PINNED against the reference convention (0 / ±90 / 180° cases), canonical-window
acceptance of an ideal disulfide + rejection of a far-apart pair, and the `_atom_site` CYS
parser. Plain-Python, no GPU/IO beyond a tmp CIF.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import disulfide_geometry as g


# ── primitives are the SAME object the existing tool re-exports ──────────────────────
def test_primitives_are_lifted_and_reexported():
    import disulfide_bridge
    assert disulfide_bridge.calc_distance is g.calc_distance
    assert disulfide_bridge.calc_dihedral is g.calc_dihedral


def test_calc_distance():
    assert g.calc_distance((0, 0, 0), (3, 4, 0)) == 5.0


# ── χSS pinned against the reference dihedral convention (Cβ–Sγ–Sγ–Cβ) ────────────────
def test_chi_ss_plus_90():
    # mirror the verified +90° arrangement: a1-b1 +y, b1-b2 +x, b2-a2 +z
    chi = g.chi_ss((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    assert abs(chi - 90.0) < 0.01


def test_chi_ss_trans_180():
    chi = g.chi_ss((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, -1.0, 0.0))
    assert abs(abs(chi) - 180.0) < 0.01


def test_chi_ss_cis_0():
    chi = g.chi_ss((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    assert abs(chi) < 0.01


def test_chi_ss_sign_is_handedness():
    # mirror image flips the sign (left- vs right-handed disulfide)
    pos = g.chi_ss((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 1.0))
    neg = g.chi_ss((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, -1.0))
    assert pos > 0 and neg < 0 and abs(pos + neg) < 0.01


# ── canonical-window acceptance / rejection ──────────────────────────────────────────
def _ideal_pair():
    # SG–SG ≈ 2.05 (in 1.8–2.5), Cβ–Cβ ≈ 3.8 (in 3.0–4.5), Cα–Cα ≈ 5.5 (in 4.5–7.5),
    # χSS ≈ +90° (in 60–120). Built along axes so each distance is exact.
    a = {"CA": (0.0, 0.0, 0.0),  "CB": (0.0, 1.5, 0.0),  "SG": (0.0, 2.5, 0.0)}
    b = {"CA": (5.5, 0.0, 0.0),  "CB": (3.8, 1.5, 0.0),  "SG": (2.05, 2.5, 0.0)}
    return a, b


def test_ideal_disulfide_accepted_all_windows():
    pg = g.pair_geometry(*_ideal_pair())
    assert pg["sg_sg"] == 2.05 and pg["bonding_compatible"] is True
    assert pg["windows"]["sg_sg"] is True and pg["windows"]["cb_cb"] is True
    assert pg["windows"]["ca_ca"] is True


def test_far_apart_pair_rejected():
    a = {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.0, 0.0), "SG": (0.0, 2.0, 0.0)}
    b = {"CA": (20.0, 0.0, 0.0), "CB": (20.0, 1.0, 0.0), "SG": (20.0, 2.0, 0.0)}
    pg = g.pair_geometry(a, b)
    assert pg["sg_sg"] == 20.0 and pg["bonding_compatible"] is False
    assert pg["windows"]["sg_sg"] is False and pg["all_windows"] is False


def test_missing_atom_yields_none_not_zero():
    # a residue with no SG (e.g. not yet built) → SG metrics None, never a fabricated 0
    a = {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.0, 0.0)}              # no SG
    b = {"CA": (5.5, 0.0, 0.0), "CB": (3.8, 1.0, 0.0), "SG": (2.05, 1.0, 0.0)}
    pg = g.pair_geometry(a, b)
    assert pg["sg_sg"] is None and pg["chi_ss"] is None
    assert pg["windows"]["sg_sg"] is None and pg["bonding_compatible"] is False
    assert pg["cb_cb"] is not None                                  # Cβ–Cβ still measured


# ── mmCIF CYS parser (the `_atom_site` loop) ─────────────────────────────────────────
_FAKE_CIF = """data_model
loop_
_atom_site.group_PDB
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM CA CYS A 12 0.0 0.0 0.0
ATOM CB CYS A 12 0.0 1.5 0.0
ATOM SG CYS A 12 0.0 2.5 0.0
ATOM CA ALA A 13 4.0 0.0 0.0
ATOM CA CYS A 45 5.5 0.0 0.0
ATOM CB CYS A 45 3.8 1.5 0.0
ATOM SG CYS A 45 2.05 2.5 0.0
#
"""


# ── author-resnum → 1-based chain index (the constraint correctness hinge) ───────────
def test_resnum_to_chain_index_one_start():
    ordered = list(range(1, 21))                                   # 1..20
    assert g.resnum_to_chain_index(ordered, 1) == 1
    assert g.resnum_to_chain_index(ordered, 12) == 12


def test_resnum_to_chain_index_non_one_start():
    # the load-bearing case: author resnums that DON'T start at 1 → index ≠ resnum
    ordered = list(range(10, 51))                                  # author resnums 10..50
    assert g.resnum_to_chain_index(ordered, 10) == 1               # first → index 1
    assert g.resnum_to_chain_index(ordered, 12) == 3               # NOT 12
    assert g.resnum_to_chain_index(ordered, 45) == 36
    assert g.resnum_to_chain_index(ordered, 99) is None            # absent → None


def test_resnum_to_chain_index_with_gaps_in_numbering():
    ordered = [5, 6, 9, 10, 11]                                    # a numbering gap (7,8 missing)
    assert g.resnum_to_chain_index(ordered, 9) == 3                # position, not value
    assert g.resnum_to_chain_index(ordered, 7) is None


def test_bond_constraint_shape():
    c = g.bond_constraint("A", 3, 36)
    assert c == {"atom1": ["A", 3, "SG"], "atom2": ["A", 36, "SG"]}


def test_parse_cys_atoms_only_cysteines(tmp_path):
    p = tmp_path / "fold.cif"; p.write_text(_FAKE_CIF)
    cys = g.parse_cys_atoms(str(p))
    assert set(cys["A"]) == {12, 45}                               # ALA 13 excluded
    assert cys["A"][12]["SG"] == (0.0, 2.5, 0.0)
    # and the parsed atoms feed pair_geometry end-to-end
    pg = g.pair_geometry(cys["A"][12], cys["A"][45])
    assert pg["sg_sg"] == 2.05 and pg["bonding_compatible"] is True
