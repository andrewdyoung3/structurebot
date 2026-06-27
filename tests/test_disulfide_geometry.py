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
    # SAME-chain (both atoms chain A) must emit EXACTLY what it did before the per-atom-chain split
    c = g.bond_constraint("A", 3, "A", 36)
    assert c == {"atom1": ["A", 3, "SG"], "atom2": ["A", 36, "SG"]}


def test_bond_constraint_inter_chain():
    # the step-2 capability: a chain PER ATOM → an INTER-chain bond is representable (atom1 on A,
    # atom2 on B). Nothing wires a cross-chain pair through this yet (that's step 3).
    c = g.bond_constraint("A", 3, "B", 36)
    assert c == {"atom1": ["A", 3, "SG"], "atom2": ["B", 36, "SG"]}


def test_parse_cys_atoms_only_cysteines(tmp_path):
    p = tmp_path / "fold.cif"; p.write_text(_FAKE_CIF)
    cys = g.parse_cys_atoms(str(p))
    assert set(cys["A"]) == {12, 45}                               # ALA 13 excluded
    assert cys["A"][12]["SG"] == (0.0, 2.5, 0.0)
    # and the parsed atoms feed pair_geometry end-to-end
    pg = g.pair_geometry(cys["A"][12], cys["A"][45])
    assert pg["sg_sg"] == 2.05 and pg["bonding_compatible"] is True


# ── Mode D: backbone engineering scan (residue-agnostic, NO χSS, soft-graded, prefiltered) ──
def test_backbone_scoring_soft_windows_pinned():
    # the soft-window distance scores (graded, never hard cutoffs) pinned at their ideal points
    assert g._gauss_score(5.5, g.CA_CA_SCORE_IDEAL, g.CA_CA_SCORE_SIGMA) == 1.0   # Cα–Cα ideal
    assert abs(g._gauss_score(3.8, g.CB_CB_IDEAL, g.CB_CB_SCORE_SIGMA) - 1.0) < 1e-9  # Cβ–Cβ ideal
    # the dead ORIENT_CUTOFF_D branch was REMOVED — orientation no longer drives the score
    assert not hasattr(g, "_orient_score")
    assert not hasattr(g, "ORIENT_CUTOFF_D")


def test_placement_primitive_reproduces_internal_coords():
    # place_from_internal must reproduce the requested bond / angle / dihedral EXACTLY
    import math
    N, CA, CB = (0.0, 0.0, 0.0), (1.46, 0.0, 0.0), (1.95, 1.42, 0.0)
    for chi1 in (-65.0, 60.0, 180.0):
        sg = g.place_from_internal(N, CA, CB, g.CYS_CB_SG_BOND, g.CYS_CA_CB_SG_ANGLE, chi1)
        assert abs(g.calc_distance(CB, sg) - g.CYS_CB_SG_BOND) < 1e-6        # CB–SG bond
        assert abs(g.calc_dihedral(N, CA, CB, sg) - chi1) < 1e-4             # N–CA–CB–SG dihedral = χ1
    # χSS from two placed Sγ matches chi_ss(cb,sg,sg,cb)
    sg1 = g.place_from_internal(N, CA, CB, g.CYS_CB_SG_BOND, g.CYS_CA_CB_SG_ANGLE, -60.0)
    N2, CA2, CB2 = (5.0, 0.0, 0.0), (4.0, 1.0, 0.0), (3.5, 2.0, 0.0)
    sg2 = g.place_from_internal(N2, CA2, CB2, g.CYS_CB_SG_BOND, g.CYS_CA_CB_SG_ANGLE, 60.0)
    assert abs(g.chi_ss(CB, sg1, sg2, CB2) - g.calc_dihedral(CB, sg1, sg2, CB2)) < 1e-9


def test_reachability_fine_sweep_beats_three_staggered():
    # a geometry where the REACHABLE rotamer sits BETWEEN the staggered χ1 values: a fine sweep
    # finds it, a 3-staggered set (−60/60/180) would miss it (the honesty hinge).
    a = {"N": (0.0, 0.0, 0.0), "CA": (1.46, 0.0, 0.0), "CB": (1.95, 1.42, 0.0)}
    sgs_a = g.sg_rotamers(a["N"], a["CA"], a["CB"])
    # place B's backbone so the IDEAL Sγ–Sγ is hit at a non-staggered χ1 for A
    b = {"N": (1.0, 5.5, 0.0), "CA": (2.0, 4.8, 0.0), "CB": (2.6, 3.6, 0.0)}
    sgs_b = g.sg_rotamers(b["N"], b["CA"], b["CB"])
    fine = g.sg_reachability(a["CB"], sgs_a, b["CB"], sgs_b)
    # the 3-staggered approximation: only −60/60/180 on each side
    stag = [-60.0, 60.0, 180.0]
    sa = [g._place_sg(a["N"], a["CA"], a["CB"], c) for c in stag]
    sb = [g._place_sg(b["N"], b["CA"], b["CB"], c) for c in stag]
    coarse = g.sg_reachability(a["CB"], sa, b["CB"], sb)
    assert fine["reach_score"] >= coarse["reach_score"]      # fine never worse
    assert fine["best_sg_sg"] is not None


def test_backbone_pair_score_graded_near_beats_off():
    # graded: a near-ideal geometry scores higher than an off-distance one, BUT both surface
    # (nothing hard-eliminated — a suggestion surface, not a filter)
    a = {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.5, 0.0)}
    near = g.backbone_pair_score(a, {"CA": (5.5, 0.0, 0.0), "CB": (3.8, 1.5, 0.0)})
    off = g.backbone_pair_score(a, {"CA": (6.5, 0.0, 0.0), "CB": (4.5, 1.5, 0.0)})
    assert near["ca_score"] == 1.0 and near["cb_score"] == 1.0
    assert near["score"] > off["score"] > 0      # graded — near ranks higher, off still surfaces


def test_backbone_pair_score_far_is_zero():
    a = {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.5, 0.0)}
    b = {"CA": (20.0, 0.0, 0.0), "CB": (20.0, 1.5, 0.0)}
    assert g.backbone_pair_score(a, b)["score"] < 0.001     # too far → ~0 (Cα-dominant)


def test_backbone_pair_score_no_chi_ss():
    # Mode D is BACKBONE-ONLY — it must not compute χSS (no Sγ pre-mutation)
    a = {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.5, 0.0)}
    b = {"CA": (5.5, 0.0, 0.0), "CB": (3.8, 1.5, 0.0)}
    assert "chi_ss" not in g.backbone_pair_score(a, b)


def test_backbone_pair_score_glycine_no_n_falls_back():
    # Gly with NO N/C to build a Cβ → last-resort CA pseudo-Cβ; orientation undefined, reachability
    # falls back to neutral (Sγ unplaceable), never a crash
    gly = {"CA": (0.0, 0.0, 0.0)}                           # no CB, no N, no C
    b = {"CA": (5.5, 0.0, 0.0), "CB": (3.8, 1.5, 0.0)}
    pg = g.backbone_pair_score(gly, b)
    assert pg is not None and pg["orientation"] is None
    assert pg["reach_score"] == g.REACH_NEUTRAL and pg["best_sg_sg"] is None
    assert "orient_score" not in pg                        # the orientation term is gone from the score


def test_backbone_pair_score_glycine_builds_cb_and_scores():
    # Gly WITH N+C → Cβ is reconstructed (build_cb) so the Gly→Cys site is SCORED via reachability,
    # not silently dropped. orientation stays None (built Cβ isn't a measured backbone feature).
    gly = {"N": (0.0, 1.46, 0.0), "CA": (0.0, 0.0, 0.0), "C": (1.4, -0.5, 0.0)}   # no CB → build it
    b = {"N": (4.6, 5.0, 0.0), "CA": (5.0, 4.0, 0.0), "CB": (4.2, 3.0, 0.0)}
    pg = g.backbone_pair_score(gly, b)
    assert pg is not None
    assert pg["orientation"] is None                       # built Cβ → orientation not "measured"
    assert pg["best_sg_sg"] is not None                    # but reachability WAS computed (Cβ built)
    assert pg["reach_score"] != g.REACH_NEUTRAL            # a real reach value, not the fallback


def test_build_cb_matches_real_residue():
    # the reconstructed Cβ lands close to a real Cβ — validated against a real fold's residues
    import os
    cif = os.path.join(os.path.dirname(__file__), "..", "cache", "1C9O.cif")
    if not os.path.isfile(cif):
        return                                             # cache not present → skip silently
    atoms = g._parse_atom_site(cif, ("N", "CA", "C", "CB"), comp_id=None)
    checked = 0
    for ch, residues in atoms.items():
        for rn, a in residues.items():
            if all(k in a for k in ("N", "CA", "C", "CB")):
                built = g.build_cb(a["N"], a["CA"], a["C"])
                assert g.calc_distance(built, a["CB"]) < 0.3   # within 0.3 Å of the real Cβ
                checked += 1
    assert checked > 10


def _scan_atoms():
    return {"A": {
        1: {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.5, 0.0)},
        2: {"CA": (2.0, 0.0, 0.0), "CB": (2.0, 1.5, 0.0)},
        5: {"CA": (5.5, 0.0, 0.0), "CB": (3.8, 1.5, 0.0)},   # an engineerable partner of res 1
        9: {"CA": (40.0, 0.0, 0.0), "CB": (40.0, 1.5, 0.0)}, # far from everything
    }}


def test_scan_prefilter_output_identical_to_full_scan():
    # THE prefilter invariant: the Cα gate changes SPEED, not OUTPUT (gated-out pairs are sub-
    # threshold, so a full scan would drop them too).
    atoms = _scan_atoms()
    ranked_gated, best_gated = g.scan_engineerable_sites(atoms)               # prefilter ON
    ranked_full, best_full = g.scan_engineerable_sites(atoms, ca_gate=None)   # full scan
    assert ranked_gated == ranked_full and best_gated == best_full


def test_scan_excludes_sequence_adjacent_and_far():
    atoms = _scan_atoms()
    ranked, best = g.scan_engineerable_sites(atoms)
    pairs = {(p["resnum_a"], p["resnum_b"]) for p in ranked}
    assert (1, 2) not in pairs                              # sequence-adjacent → excluded
    assert all(9 not in (a, b) for a, b in pairs)           # res 9 too far → never surfaces


def test_scan_best_partner_map():
    ranked, best = g.scan_engineerable_sites(_scan_atoms())
    # each surfaced residue's best-partner score = the max score of any pair it's in
    assert best.get(("A", 1)) is not None and best[("A", 1)] == best[("A", 5)]
    assert ("A", 9) not in best                             # far residue never gets a partner


# ── reachability replaces orientation in the RANKING + the clash check (tier b) ──────────────
def _reach_atoms():
    """Two residues whose Sγ CAN reach a disulfide (close, well-oriented) and one whose backbone is
    oriented so the sulfurs can't reach a good χSS at the same Cα/Cβ distances. Real N/CA/CB so the
    rotamer sweep runs."""
    return {"A": {
        1:  {"N": (0.0, 1.46, 0.0),  "CA": (0.0, 0.0, 0.0),  "CB": (0.5, -1.2, 0.6)},
        5:  {"N": (4.4, 1.4, 0.0),   "CA": (4.6, 0.0, 0.0),  "CB": (4.1, -1.2, 0.6)},   # faces res 1
        9:  {"N": (4.4, 1.4, 5.0),   "CA": (4.6, 0.0, 5.0),  "CB": (5.3, 1.0, 5.6)},    # CB faces AWAY
    }}


def test_reachability_drives_score_not_orientation():
    # a reachability-good pair must rank via reach_score; the score = ca*cb*reach (no orient term)
    atoms = _reach_atoms()
    pg = g.backbone_pair_score(atoms["A"][1], atoms["A"][5])
    assert pg["best_sg_sg"] is not None and pg["reach_score"] > 0
    expected = pg["ca_score"] * pg["cb_score"] * pg["reach_score"]
    assert abs(pg["score"] - expected) < 2e-4             # score is ca×cb×reach (within 4-dp rounding)
    assert "orientation" in pg                             # orientation still MEASURED + displayed
    assert pg["orientation"] is not None


def test_reachable_pair_outscores_unreachable_at_similar_distance():
    # res1↔res5 (sulfurs reach) should reach-score higher than res5↔res9 (Cβ points away) — the
    # signal orientation only weakly proxied
    atoms = _reach_atoms()
    reachable = g.backbone_pair_score(atoms["A"][1], atoms["A"][5])
    away = g.backbone_pair_score(atoms["A"][5], atoms["A"][9])
    assert reachable["reach_score"] > away["reach_score"]


def test_clash_flag_set_when_sg_overlaps_neighbor():
    # the placed Sγ of a candidate sitting ON a neighbour heavy atom → clash flag True + demotion;
    # a clear candidate → clash False, undemoted
    atoms = _reach_atoms()
    # neighbour atoms FAR from everything → no clash
    clear_grid = g.ClashGrid([("A", 100, "C", (50.0, 50.0, 50.0))])
    rc, _ = g.scan_engineerable_sites(atoms, clash_grid=clear_grid)
    top_clear = rc[0]
    assert top_clear["clash"] is False
    # now drop a carbon RIGHT where the best-reach Sγ lands (recover it from a no-grid run)
    pg = g.backbone_pair_score(atoms["A"][1], atoms["A"][5])  # _placed carries the Sγ coords
    sg = pg["_placed"]["sg_a"]
    clash_grid = g.ClashGrid([("A", 100, "C", sg)])           # a C atom ON the Sγ (residue 100 ≠ 1/5)
    rk, _ = g.scan_engineerable_sites(atoms, clash_grid=clash_grid)
    hit = next(p for p in rk if {p["resnum_a"], p["resnum_b"]} == {1, 5})
    assert hit["clash"] is True
    assert hit["score"] < top_clear["score"]                  # clashing site is softly DEMOTED


def test_clash_excludes_the_two_mutated_residues():
    # the candidate's OWN residue atoms must not count as a clash (their sidechains become the Cys)
    atoms = _reach_atoms()
    # a grid containing ONLY the two candidate residues' backbone atoms → must NOT clash
    grid = g.ClashGrid([("A", 1, "C", atoms["A"][1]["CA"]), ("A", 5, "C", atoms["A"][5]["CA"])])
    rk, _ = g.scan_engineerable_sites(atoms, clash_grid=grid)
    hit = next(p for p in rk if {p["resnum_a"], p["resnum_b"]} == {1, 5})
    assert hit["clash"] is False                              # excluded → no self-clash


# ── pair-shape reshape: two-chain container, intra-chain only (no cross-chain leakage) ───
def _scan_atoms_two_chains():
    # chain A and chain B EACH carry an internal engineerable pair (res 1↔5). B:1 also sits within
    # disulfide range of A:5 — a cross-chain pair that WOULD be engineerable if enumerated; the
    # per-chain scan must NOT surface it (no cross-chain enumeration until step 4).
    return {
        "A": {1: {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.5, 0.0)},
              5: {"CA": (5.5, 0.0, 0.0), "CB": (3.8, 1.5, 0.0)}},
        "B": {1: {"CA": (0.5, 0.0, 0.0), "CB": (0.5, 1.5, 0.0)},   # ~5 Å from A:5 → would-be partner
              5: {"CA": (6.0, 0.0, 0.0), "CB": (4.3, 1.5, 0.0)}},
    }


def test_scan_emits_chain_per_member_intrachain():
    ranked, _ = g.scan_engineerable_sites(_scan_atoms())
    assert ranked
    for p in ranked:
        assert "chain_a" in p and "chain_b" in p            # reshaped container (chain per member)
        assert p["chain_a"] == p["chain_b"] == "A"          # intra-chain: both members on one chain


def test_scan_no_cross_chain_leakage():
    # container widened, BEHAVIOR unchanged: a multi-chain fold yields ONLY intra-chain pairs, even
    # though a cross-chain pair here would be geometrically engineerable. Don't surface what Mode C
    # / the interface mode can't yet act on.
    ranked, best = g.scan_engineerable_sites(_scan_atoms_two_chains())
    assert ranked
    assert all(p["chain_a"] == p["chain_b"] for p in ranked)    # NO pair spans two chains
    assert {p["chain_a"] for p in ranked} == {"A", "B"}         # both chains scanned (intra each)
    assert all(isinstance(k, tuple) and k[0] in ("A", "B") for k in best)   # best keyed per member


# ── pair-shape helpers: reshaped + legacy back-compat + display collapse ──────────────
def test_pair_chains_reshaped():
    assert g.pair_chains({"chain_a": "A", "chain_b": "B"}) == ("A", "B")
    assert g.pair_chains({"chain_a": "A", "chain_b": "A"}) == ("A", "A")


def test_pair_chains_legacy_single_chain():
    # the back-compat hinge: an OLD saved scan carries only `chain` → reads as same-chain
    assert g.pair_chains({"chain": "A", "resnum_a": 5, "resnum_b": 9}) == ("A", "A")


def test_pair_label_same_chain_is_bare():
    # SAME-chain display is visually UNCHANGED (bare resnums) — the reshape is container-only here
    assert g.pair_label({"chain_a": "A", "chain_b": "A", "resnum_a": 5, "resnum_b": 9}) == "5–9"
    assert g.pair_label({"chain": "A", "resnum_a": 5, "resnum_b": 9}) == "5–9"      # legacy → bare
    assert g.pair_label({"chain_a": "A", "chain_b": "A", "resnum_a": 5, "resnum_b": 9},
                        cys=True) == "Cys5–Cys9"


def test_pair_label_cross_chain_shows_both_chains():
    # CROSS-chain MUST show the chain on BOTH members (ambiguous otherwise)
    assert g.pair_label({"chain_a": "A", "chain_b": "B", "resnum_a": 140, "resnum_b": 88}) \
        == "A:140 ↔ B:88"
    assert g.pair_label({"chain_a": "A", "chain_b": "B", "resnum_a": 140, "resnum_b": 88},
                        cys=True) == "Cys A:140 ↔ Cys B:88"


# ── interface (inter-chain) scan: cross-chain only, interface-bounded, shared scoring ─────────
def _interface_atoms():
    # chain A and chain B with ONE engineerable interface pair (A:5 ↔ B:5); the other residues are
    # far from the OTHER chain (non-interface) so they must not surface.
    return {
        "A": {5: {"CA": (0.0, 0.0, 0.0), "CB": (0.0, 1.5, 0.0)},
              9: {"CA": (40.0, 0.0, 0.0), "CB": (40.0, 1.5, 0.0)}},
        "B": {5: {"CA": (5.5, 0.0, 0.0), "CB": (3.8, 1.5, 0.0)},      # ~ideal partner of A:5
              9: {"CA": (80.0, 0.0, 0.0), "CB": (80.0, 1.5, 0.0)}},
    }


def test_interface_scan_emits_only_cross_chain():
    ranked, _ = g.scan_interface_sites(_interface_atoms())
    assert ranked
    assert all(p["chain_a"] != p["chain_b"] for p in ranked)          # every pair spans two chains
    top = ranked[0]
    assert {(top["chain_a"], top["resnum_a"]), (top["chain_b"], top["resnum_b"])} == {("A", 5), ("B", 5)}


def test_interface_scan_is_interface_bounded():
    # the Cα gate IS the interface bound — a residue far from the other chain never surfaces
    ranked, _ = g.scan_interface_sites(_interface_atoms())
    surfaced = {(p["chain_a"], p["resnum_a"]) for p in ranked} \
        | {(p["chain_b"], p["resnum_b"]) for p in ranked}
    assert ("A", 9) not in surfaced and ("B", 9) not in surfaced      # non-interface → excluded


def test_interface_scan_reuses_backbone_pair_score():
    # the convergence point: the SAME scoring as intra Mode D (no new geometry loop)
    atoms = _interface_atoms()
    top = g.scan_interface_sites(atoms)[0][0]
    bp = g.backbone_pair_score(atoms["A"][5], atoms["B"][5])
    assert top["score"] == bp["score"] and top["ca_ca"] == bp["ca_ca"] and top["cb_cb"] == bp["cb_cb"]


def test_interface_scan_empty_on_monomer():
    ranked, best = g.scan_interface_sites({"A": _interface_atoms()["A"]})
    assert ranked == [] and best == {}                               # one chain → no cross pairs
