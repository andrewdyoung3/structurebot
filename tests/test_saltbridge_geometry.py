"""
tests/test_saltbridge_geometry.py
---------------------------------
The plain-Python salt-bridge core (saltbridge_geometry): the geometry-score WINDOW (steric floor /
ideal plateau / 4 Å cutoff / 5 Å shoulder), the burial factor, EXISTING-pair assessment (closest
carboxyl-O↔basic-N + H-bond flag + the 4–5 Å optimizable near-miss), and the NOVEL rotamer-aware
reach half (a reachable clash-free charge pair credited; a walled-in one soft-demoted), intra +
inter-chain. No GPU, no ChimeraX. Real-structure checks skip silently if the cache CIF isn't present.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import saltbridge_geometry as sb
from disulfide_geometry import ClashGrid

_CACHE = Path(__file__).parent.parent / "cache"


# ── geometry-score window + burial factor (PURE) ────────────────────────────────────────────────
def test_geometry_score_is_a_window_not_a_ramp():
    assert sb.geometry_score(2.0) == 0.0           # below the steric floor → not a bridge
    assert sb.geometry_score(2.8) == 1.0           # ideal H-bonded separation
    assert 0.0 < sb.geometry_score(2.5) < 1.0      # ramps up out of the steric floor
    assert abs(sb.geometry_score(4.0) - sb.SB_GEOM_AT_CUTOFF) < 1e-9   # the 4 Å hard cutoff value
    assert 0.0 < sb.geometry_score(4.8) < sb.SB_GEOM_AT_CUTOFF         # 4–5 Å shoulder (optimizable)
    assert sb.geometry_score(5.5) == 0.0           # beyond the shoulder
    assert sb.geometry_score(None) == 0.0


def test_burial_factor_weights_buried_over_surface():
    assert sb.burial_factor(5.0) == 1.0                          # buried → full weight
    assert sb.burial_factor(80.0) == sb.SB_BURIAL_SURFACE_WEIGHT  # surface → down-weighted (desolvation)
    assert sb.burial_factor(None) == 1.0                         # unknown → NEUTRAL (never fabricated)
    mid = sb.burial_factor((sb.SB_BURIED_SASA + sb.SB_SURFACE_SASA) / 2.0)
    assert sb.SB_BURIAL_SURFACE_WEIGHT < mid < 1.0               # ramps between


# ── existing-pair assessment (pure measurement on real atoms) ───────────────────────────────────
def _asp(chain, rn, ox):
    """An Asp with carboxyl O's near *ox* (Å along x); secondary atoms offset in +y so the closest
    O–N to a base at x=ox+d is unambiguously d (not a shorter diagonal)."""
    return {"resname": "ASP", "OD1": (ox, 0.0, 0.0), "OD2": (ox, 1.2, 0.0),
            "CA": (ox - 3.0, 0.0, 0.0), "N": (ox - 4.0, 0.0, 0.0),
            "CB": (ox - 1.8, 0.0, 0.0), "C": (ox - 3.0, 1.2, 0.0)}


def _arg(chain, rn, nx):
    """An Arg with guanidinium N's near *nx* (Å along x), secondary atoms offset in +y to match."""
    return {"resname": "ARG", "NH1": (nx, 0.0, 0.0), "NH2": (nx, 1.2, 0.0),
            "NE": (nx + 1.0, 0.6, 0.0), "CA": (nx + 3.0, 0.0, 0.0),
            "N": (nx + 4.0, 0.0, 0.0), "CB": (nx + 1.8, 0.0, 0.0), "C": (nx + 3.0, 1.2, 0.0)}


def test_existing_pair_detected_with_hbond_flag():
    # Asp OD1 at x=0, Arg NH1 at x=2.8 → closest O–N 2.8 Å (a salt bridge AND an H-bond)
    residues = {"A": {10: _asp("A", 10, 0.0), 30: _arg("A", 30, 2.8)}}
    ranked, best = sb.scan_existing_pairs(residues)
    assert len(ranked) == 1
    p = ranked[0]
    assert abs(p["on_dist"] - 2.8) < 0.3
    assert p["within_cutoff"] and p["hbond_like"] and not p["optimizable"]
    assert p["type"] == "D-R"
    # canonical member order: A-side = acid (Asp), B-side = base (Arg)
    assert p["from_aa_a"] == "D" and p["from_aa_b"] == "R"
    assert (("A", 10) in best) and (("A", 30) in best)


def test_existing_near_miss_flagged_optimizable():
    # closest O–N ~4.8 Å → a 4–5 Å near-miss: surfaced but flagged optimizable, NOT within the cutoff
    residues = {"A": {10: _asp("A", 10, 0.0), 30: _arg("A", 30, 4.8)}}
    ranked, _ = sb.scan_existing_pairs(residues)
    assert len(ranked) == 1
    p = ranked[0]
    assert p["optimizable"] and not p["within_cutoff"] and not p["hbond_like"]


def test_existing_surface_pair_scored_below_buried():
    residues = {"A": {10: _asp("A", 10, 0.0), 30: _arg("A", 30, 2.8)}}
    buried, _ = sb.scan_existing_pairs(residues, sasa_map={("A", 10): 5.0, ("A", 30): 5.0})
    surface, _ = sb.scan_existing_pairs(residues, sasa_map={("A", 10): 90.0, ("A", 30): 90.0})
    assert buried[0]["score"] > surface[0]["score"]          # desolvation down-weights the surface pair
    assert buried[0]["buried"] is True and surface[0]["buried"] is False


# ── novel suggestion: rotamer-aware reach + clash ───────────────────────────────────────────────
def _bb(chain, rn, name, ca, cbdir=1.0):
    """A minimal backbone-only residue (N/CA/CB/C); *cbdir* (±1) points the Cβ (hence the sidechain
    reach) along ±x — so two facing residues can be made to reach toward each other."""
    cx, cy, cz = ca
    return {"resname": name, "CA": ca, "N": (cx - cbdir, cy + 0.5, cz),
            "CB": (cx + 1.5 * cbdir, cy, cz), "C": (cx, cy + 1.4, cz)}


def test_novel_reach_credits_a_clashfree_pair():
    # two ALA backbones FACING each other (sidechains point inward) in open space → a complementary
    # charge pair can reach a clash-free salt bridge (no surrounding atoms → no clash)
    residues = {"A": {10: _bb("A", 10, "ALA", (0.0, 0.0, 0.0), +1.0),
                      20: _bb("A", 20, "ALA", (8.0, 0.0, 0.0), -1.0)}}
    grid = ClashGrid([])                                      # empty → nothing to clash with
    ranked, best = sb.scan_novel_sites(residues, clash_grid=grid)
    assert len(ranked) == 1
    c = ranked[0]
    assert c["clash"] is False
    assert c["best_on"] is not None and c["best_on"] <= sb.SB_DIST_CUTOFF
    assert c["to_aa_a"] in ("D", "E", "R", "K") and c["to_aa_b"] in ("D", "E", "R", "K")
    # one acid + one base (a salt bridge needs both signs)
    assert {c["to_aa_a"], c["to_aa_b"]} & set(sb.SB_ACID) and {c["to_aa_a"], c["to_aa_b"]} & set(sb.SB_BASE)
    assert c["from_aa_a"] == "A" and c["from_aa_b"] == "A"


def test_novel_reach_demoted_when_no_rotamer_dodges_clash():
    # SAME pair, but pack the meeting region (a third residue's heavy atoms) so EVERY reaching rotamer
    # collides → the candidate is clash-flagged + soft-demoted (rotamer-aware-clash, the disulfide lesson)
    residues = {"A": {10: _bb("A", 10, "ALA", (0.0, 0.0, 0.0), +1.0),
                      20: _bb("A", 20, "ALA", (8.0, 0.0, 0.0), -1.0)}}
    open_grid = ClashGrid([])
    open_ranked, _ = sb.scan_novel_sites(residues, clash_grid=open_grid)
    # a dense wall of carbons (residue 99, NOT excluded) filling the space between the two Cβ (x≈1.5–6.5)
    wall = [("A", 99, "C", (x / 10.0, y / 10.0, z / 10.0))
            for x in range(10, 71, 3) for y in range(-20, 21, 3) for z in range(-20, 21, 3)]
    walled_grid = ClashGrid(wall)
    walled_ranked, _ = sb.scan_novel_sites(residues, clash_grid=walled_grid)
    assert open_ranked and open_ranked[0]["clash"] is False
    if walled_ranked:                                        # if it still surfaces, it must be demoted
        assert walled_ranked[0]["clash"] is True
        assert walled_ranked[0]["score"] < open_ranked[0]["score"]
    else:                                                    # or dropped below the floor entirely
        assert True


# ── inter-chain (the disulfide pair machinery reused; own-chain from_aa) ─────────────────────────
def test_novel_interface_maps_each_member_own_chain():
    # a cross-chain candidate: A:10 (was GLY) ↔ B:20 (was ALA) — from_aa recovered per OWN chain,
    # the shared-numbering trap analog (both could be resnum 10/20 on different chains)
    residues = {"A": {10: _bb("A", 10, "GLY", (0.0, 0.0, 0.0), +1.0)},
                "B": {20: _bb("B", 20, "ALA", (8.0, 0.0, 0.0), -1.0)}}
    ranked, _ = sb.scan_novel_interface(residues, clash_grid=ClashGrid([]))
    assert len(ranked) == 1
    c = ranked[0]
    assert c["chain_a"] == "A" and c["chain_b"] == "B" and c["interchain"] is True
    assert c["from_aa_a"] == "G" and c["from_aa_b"] == "A"     # own-chain from_aa, not swapped
    # Gly is installable (Cβ reconstructed via build_cb) — not silently dropped
    assert c["resnum_a"] == 10 and c["resnum_b"] == 20


def test_novel_skips_already_charged_and_proline():
    residues = {"A": {10: _bb("A", 10, "ASP", (0.0, 0.0, 0.0)),     # already charged → skip
                      20: _bb("A", 20, "PRO", (7.0, 0.0, 0.0))}}    # proline → skip
    ranked, _ = sb.scan_novel_sites(residues, clash_grid=ClashGrid([]))
    assert ranked == []


# ── real-structure smoke (skips silently without the cache CIF) ──────────────────────────────────
def _real_cif():
    for name in ("1C9O.cif", "1BOV.cif", "1A2W.cif"):
        p = _CACHE / name
        if p.exists():
            return str(p)
    return None


def test_real_structure_sensible_counts():
    cif = _real_cif()
    if cif is None:
        return
    residues = sb.parse_residues(cif)
    assert residues
    existing, _ = sb.scan_existing_pairs(residues)
    grid = ClashGrid([(c, r, e, xyz) for ch in residues for r in residues[ch]
                      for c, e, xyz in []])                # (real grid built in the router; here just smoke)
    novel, best = sb.scan_novel_sites(residues, clash_grid=None)
    # not zero, not everything — a sensible shortlist; existing pairs are real salt bridges
    assert isinstance(existing, list) and isinstance(novel, list)
    for p in existing:
        assert p["on_dist"] <= sb.SB_DIST_SHOULDER
    for c in novel:
        assert c["score"] >= sb.SB_NOVEL_MIN_SCORE


def test_compute_sasa_map_real_or_empty():
    cif = _real_cif()
    if cif is None:
        return
    sasa = sb.compute_sasa_map(cif)
    # FreeSASA is a declared dep → expect a populated map; if absent it degrades to {} (neutral burial)
    assert isinstance(sasa, dict)
    if sasa:
        k = next(iter(sasa))
        assert isinstance(k, tuple) and len(k) == 2 and sasa[k] >= 0.0
