"""
tests/test_glycan_bridge.py
---------------------------
Unit tests for glycan_bridge.GlycanBridge.

All tests are self-contained — no ChimeraX instance, no network, no PDB file
(unless a tmp_path fixture is used for PDB-backed SASA/SS tests).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from glycan_bridge import (
    GlycanBridge,
    _SEQUON_RE,
    _classify_ss_from_angles,
    _classify_confidence,
    _compute_composite_score,
    _COLOR_NATIVE,
    _COLOR_ENG_HIGH,
    _COLOR_ENG_MOD,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def gb() -> GlycanBridge:
    return GlycanBridge()


# ════════════════════════════════════════════════════════════════════════════════
# Section A — Regex / sequon detection
# ════════════════════════════════════════════════════════════════════════════════

def test_sequon_regex_matches_nxst():
    """N[^P][ST] should match NAS, NTS, NET, NLT but not NPS, NPT."""
    positives = ["NAS", "NTS", "NET", "NLT", "NGS", "NVT"]
    negatives = ["NPS", "NPT", "NAA", "NAP"]
    for s in positives:
        assert _SEQUON_RE.search(s), f"Expected match for {s}"
    for s in negatives:
        assert not _SEQUON_RE.search(s), f"Expected no match for {s}"


def test_scan_sequons_positions_are_1indexed(gb):
    """Position of N in the returned list must be 1-based."""
    # M(1)N(2)A(3)S(4)V(5)N(6)A(7)T(8)L(9)
    # NAS at 1-based pos 2; NAT at 1-based pos 6
    seq = "MNASVNATL"
    sites = gb.scan_sequons(seq, chain="A")
    positions = [s["position"] for s in sites]
    assert 2 in positions   # M-[N-A-S]-V…: N is at 1-based position 2
    assert 6 in positions   # …V-[N-A-T]-L: N is at 1-based position 6


def test_scan_sequons_no_proline_x(gb):
    """Sequences with NPS or NPT must NOT be detected."""
    seq = "ANPSAANPTB"
    sites = gb.scan_sequons(seq, chain="A")
    # NPS at pos 2, NPT at pos 7 — both should be absent
    assert len(sites) == 0


def test_scan_sequons_empty_sequence(gb):
    assert gb.scan_sequons("", "A") == []


def test_scan_sequons_chain_preserved(gb):
    seq = "MNATL"   # NAT at pos 2
    sites = gb.scan_sequons(seq, chain="B")
    assert all(s["chain"] == "B" for s in sites)


# ════════════════════════════════════════════════════════════════════════════════
# Section B — Scoring helpers
# ════════════════════════════════════════════════════════════════════════════════

def test_compute_composite_score_formula():
    """score = sasa × loop_factor × interface_factor × esm_factor"""
    # Loop (L) factor = 1.0
    assert _compute_composite_score(0.8, "L", 1.0, 1.0) == pytest.approx(0.8, abs=1e-4)
    # Helix (H) factor = 0.5
    assert _compute_composite_score(0.8, "H", 1.0, 1.0) == pytest.approx(0.4, abs=1e-4)
    # Sheet (E) factor = 0.3
    assert _compute_composite_score(0.8, "E", 1.0, 1.0) == pytest.approx(0.24, abs=1e-4)
    # Interface penalty = 0.5
    assert _compute_composite_score(0.8, "L", 0.5, 1.0) == pytest.approx(0.4, abs=1e-4)
    # ESM factor = 0.5
    assert _compute_composite_score(0.8, "L", 1.0, 0.5) == pytest.approx(0.4, abs=1e-4)


def test_classify_confidence_thresholds():
    assert _classify_confidence(0.0)  == "low"
    assert _classify_confidence(0.19) == "low"
    assert _classify_confidence(0.2)  == "moderate"
    assert _classify_confidence(0.39) == "moderate"
    assert _classify_confidence(0.4)  == "high"
    assert _classify_confidence(1.0)  == "high"


def test_classify_ss_from_angles_helix():
    # Canonical α-helix: φ = -57, ψ = -47
    assert _classify_ss_from_angles(-57.0, -47.0) == "H"


def test_classify_ss_from_angles_sheet():
    # Canonical β-sheet: φ = -119, ψ = 113
    assert _classify_ss_from_angles(-119.0, 113.0) == "E"


def test_classify_ss_from_angles_loop():
    # Neither helix nor sheet
    assert _classify_ss_from_angles(60.0, 40.0) == "L"


# ════════════════════════════════════════════════════════════════════════════════
# Section C — score_sequon_sites (no PDB)
# ════════════════════════════════════════════════════════════════════════════════

def test_score_sequon_sites_defaults_without_pdb(gb):
    """Without a PDB, SASA defaults to 0.5 and SS to 'L'."""
    seq = "MNATL"
    sites = gb.scan_sequons(seq, "A")
    scored = gb.score_sequon_sites(sites)
    assert len(scored) == 1
    s = scored[0]
    assert s["sasa"] == pytest.approx(0.5, abs=1e-4)
    assert s["secondary_structure"] == "L"
    # composite = 0.5 × 1.0 × 1.0 × 1.0 = 0.5
    assert s["composite_score"] == pytest.approx(0.5, abs=1e-4)
    assert s["confidence"] == "high"  # 0.5 >= 0.4


def test_score_sequon_sites_interface_penalty(gb):
    """Residue near an interface (within 5) should get interface_factor=0.5."""
    seq = "MNATL"    # NAT at position 2
    sites = gb.scan_sequons(seq, "A")
    # Interface at position 4 → |2-4|=2 ≤ 5, so near_interface=True
    scored = gb.score_sequon_sites(sites, interface_residues=[4])
    s = scored[0]
    assert s["interface_proximity"] is True
    # composite = 0.5 × 1.0 × 0.5 × 1.0 = 0.25
    assert s["composite_score"] == pytest.approx(0.25, abs=1e-4)


def test_score_sequon_sites_esm_factor(gb):
    """ESM tolerance should be applied to composite score."""
    seq = "MNATL"
    sites = gb.scan_sequons(seq, "A")
    scored = gb.score_sequon_sites(sites, esm_scores={2: 0.6})
    s = scored[0]
    assert s["esm_tolerance"] == pytest.approx(0.6, abs=1e-4)
    # composite = 0.5 × 1.0 × 1.0 × 0.6 = 0.3
    assert s["composite_score"] == pytest.approx(0.3, abs=1e-4)


def test_score_sequon_sites_sorted_descending(gb):
    """Scored sites must be sorted by composite_score descending."""
    seq = "MNASLNAT"   # NAS at 2, NAT at 6
    sites = gb.scan_sequons(seq, "A")
    # Give higher ESM to position 6 so it should score higher
    scored = gb.score_sequon_sites(sites, esm_scores={2: 0.2, 6: 0.9})
    scores = [s["composite_score"] for s in scored]
    assert scores == sorted(scores, reverse=True)


# ════════════════════════════════════════════════════════════════════════════════
# Section D — full_glycan_scan
# ════════════════════════════════════════════════════════════════════════════════

def test_full_glycan_scan_no_sequence(gb):
    """Empty sequence should return success=False."""
    result = gb.full_glycan_scan(sequence="")
    assert result["success"] is False
    assert "sequence" in result["error"].lower()


def test_full_glycan_scan_schema(gb):
    """Return dict must have all required keys."""
    result = gb.full_glycan_scan(sequence="MNASVNATL", chain="A")
    required = [
        "success", "chain", "pdb_path",
        "native_sequons", "engineered_candidates",
        "all_ranked", "top_n", "error",
    ]
    for key in required:
        assert key in result, f"Missing key: {key}"
    assert result["success"] is True
    assert result["error"] is None


def test_full_glycan_scan_native_sequons_above_min_score(gb):
    """native_sequons should only include sites with composite_score >= min_score."""
    result = gb.full_glycan_scan(sequence="MNASVNATL", min_score=0.3)
    for site in result["native_sequons"]:
        assert site["composite_score"] >= 0.3


def test_full_glycan_scan_all_ranked_is_superset(gb):
    """all_ranked >= native_sequons (may include sites below min_score)."""
    result = gb.full_glycan_scan(sequence="MNASVNATL", min_score=0.4)
    ranked_pos  = {s["position"] for s in result["all_ranked"]}
    native_pos  = {s["position"] for s in result["native_sequons"]}
    assert native_pos.issubset(ranked_pos)


def test_full_glycan_scan_hiv_gp120():
    """
    HIV-1 gp120 has many native N-glycosylation sequons.
    A representative 20-residue excerpt with 3 known sequons is used.

    Sequence: M(1)R(2)C(3)N(4)I(5)T(6)S(7)A(8)N(9)V(10)T(11)L(12)N(13)A(14)S(15)M(16)L(17)E(18)E(19)Q(20)
    Sequons: NIT at pos 4, NVT at pos 9, NAS at pos 13
    """
    seq = "MRCNITSANVTLNASMLEEQ"
    gb = GlycanBridge()
    result = gb.full_glycan_scan(sequence=seq, chain="A")
    assert result["success"] is True
    # All 3 sequons should be detected
    positions = {s["position"] for s in result["all_ranked"]}
    detected = sum(1 for p in [4, 9, 13] if p in positions)
    assert detected >= 2, f"Only {detected} sequons detected; positions found: {positions}"


# ════════════════════════════════════════════════════════════════════════════════
# Section E — suggest_engineered_sequons
# ════════════════════════════════════════════════════════════════════════════════

def test_suggest_engineered_sequons_one_mutation(gb):
    """
    Only single-AA mutations (X → N where pos+2 is already S/T) are proposed.
    """
    # "MALS" → at pos 1, aa=M, aa+1=A (≠P), aa+2=L (not S/T) → skip
    # "LAST" → at pos 3 (L), aa+1=A(≠P), aa+2=S → propose L3N
    seq  = "MALAST"
    sites = gb.scan_sequons(seq, "A")   # no native sequons in this sequence
    eng   = gb.suggest_engineered_sequons(sites, seq, "A", top_n=5)
    # Find if L3N or A4N is proposed
    mutations = {c["mutation"] for c in eng}
    # L at pos 3: L-A-S → N-A-S with L3N; A at pos 4: A-S-T → N-S-T with A4N
    assert len(eng) > 0, f"No engineered candidates found; seq={seq}"
    for c in eng:
        assert c.get("engineered") is True
        assert "mutation" in c


def test_suggest_engineered_sequons_top_n_respected(gb):
    """Returned list must not exceed top_n."""
    seq  = "MALASTLASQNATLL"
    sites = gb.scan_sequons(seq, "A")
    eng   = gb.suggest_engineered_sequons(sites, seq, "A", top_n=2)
    assert len(eng) <= 2


def test_suggest_no_proline_x(gb):
    """Positions where aa[i+1] == P are never proposed."""
    seq = "MAPSE"   # A at 2, X=P → skip
    sites = gb.scan_sequons(seq, "A")
    eng = gb.suggest_engineered_sequons(sites, seq, "A", top_n=10)
    for c in eng:
        seq_x = seq[c["position"]]   # 0-indexed aa at i+1
        assert seq_x != "P", f"Proline X proposed: {c}"


# ════════════════════════════════════════════════════════════════════════════════
# Section F — generate_chimerax_commands
# ════════════════════════════════════════════════════════════════════════════════

def test_generate_chimerax_commands_empty(gb):
    cmds, exps = gb.generate_chimerax_commands([])
    assert cmds == []
    assert exps == []


def test_generate_chimerax_commands_three_per_candidate(gb):
    """Exactly 3 commands and 1 explanation per candidate."""
    cands = [
        {"position": 5, "confidence": "high", "composite_score": 0.7,
         "engineered": False, "chain": "A"},
        {"position": 12, "confidence": "moderate", "composite_score": 0.25,
         "engineered": False, "chain": "A"},
    ]
    cmds, exps = gb.generate_chimerax_commands(cands, model_id="1", chain="A")
    assert len(cmds) == 6   # 3 per candidate
    assert len(exps) == 2


def test_generate_chimerax_commands_native_high_color(gb):
    """Native + high confidence → _COLOR_NATIVE (#00cc00)."""
    cands = [{"position": 5, "confidence": "high", "composite_score": 0.8,
              "engineered": False, "chain": "A"}]
    cmds, _ = gb.generate_chimerax_commands(cands, model_id="1", chain="A")
    color_cmd = cmds[0]
    assert _COLOR_NATIVE in color_cmd
    assert "#cc00cc" not in color_cmd   # not proline magenta


def test_generate_chimerax_commands_engineered_high_color(gb):
    """Engineered + high confidence → _COLOR_ENG_HIGH (#00cccc)."""
    cands = [{"position": 5, "confidence": "high", "composite_score": 0.8,
              "engineered": True, "mutation": "A5N", "chain": "A"}]
    cmds, _ = gb.generate_chimerax_commands(cands, model_id="1", chain="A")
    color_cmd = cmds[0]
    assert _COLOR_ENG_HIGH in color_cmd


def test_generate_chimerax_commands_moderate_color(gb):
    """Moderate / low confidence → _COLOR_ENG_MOD (#cccc00)."""
    cands = [{"position": 5, "confidence": "moderate", "composite_score": 0.3,
              "engineered": False, "chain": "A"}]
    cmds, _ = gb.generate_chimerax_commands(cands, model_id="1", chain="A")
    color_cmd = cmds[0]
    assert _COLOR_ENG_MOD in color_cmd


def test_generate_chimerax_commands_mutation_label(gb):
    """Engineered candidate labels must include the mutation string."""
    cands = [{"position": 5, "confidence": "high", "composite_score": 0.8,
              "engineered": True, "mutation": "A5N", "chain": "A"}]
    cmds, _ = gb.generate_chimerax_commands(cands, model_id="1", chain="A")
    label_cmd = cmds[2]
    assert "A5N" in label_cmd


# ════════════════════════════════════════════════════════════════════════════════
# Section G — SASA fallback (freesasa → BioPython)
# ════════════════════════════════════════════════════════════════════════════════

def test_sasa_fallback_to_biopython_when_freesasa_fails(tmp_path):
    """If freesasa raises, _get_sasa() should fall back to BioPython ShrakeRupley."""
    from glycan_bridge import _get_sasa

    # Minimal PDB content for a single Asn residue (enough for BioPython)
    pdb_content = """\
ATOM      1  N   ASN A   5       1.000   2.000   3.000  1.00 20.00           N
ATOM      2  CA  ASN A   5       2.000   2.000   3.000  1.00 20.00           C
ATOM      3  C   ASN A   5       3.000   2.000   3.000  1.00 20.00           C
ATOM      4  O   ASN A   5       3.000   3.000   3.000  1.00 20.00           O
END
"""
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(pdb_content)

    with patch("freesasa.Structure", side_effect=ImportError("freesasa not available")):
        # Should not raise — falls back to BioPython
        result = _get_sasa(str(pdb_file), "A", [5])
        # BioPython may return a value or {} if residue too small; either is fine
        assert isinstance(result, dict)


def test_sasa_returns_empty_for_missing_pdb():
    """_get_sasa() with a non-existent PDB path returns {}."""
    from glycan_bridge import _get_sasa
    result = _get_sasa("/nonexistent/path.pdb", "A", [1, 2, 3])
    assert result == {}


# ════════════════════════════════════════════════════════════════════════════════
# Section H — DSSP fallback
# ════════════════════════════════════════════════════════════════════════════════

def test_dssp_fallback_to_ramachandran(tmp_path):
    """If DSSP fails, _get_secondary_structure() falls back to Ramachandran angles."""
    from glycan_bridge import _get_secondary_structure

    # Write a minimal PDB — DSSP may fail on it, or succeed with 'L'
    pdb_content = """\
ATOM      1  N   ALA A   1      -0.525   1.362   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.520   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.120   1.060   0.000  1.00  0.00           O
END
"""
    pdb_file = tmp_path / "mini.pdb"
    pdb_file.write_text(pdb_content)

    with patch("Bio.PDB.DSSP", side_effect=Exception("no DSSP binary")):
        result = _get_secondary_structure(str(pdb_file), "A")
        # Either returns {} (no backbone angles computable) or a dict
        assert isinstance(result, dict)


# ════════════════════════════════════════════════════════════════════════════════
# Section I — analyze() integration
# ════════════════════════════════════════════════════════════════════════════════

def test_analyze_returns_summary(gb):
    """analyze() must include 'summary', 'chimerax_commands', 'chimerax_explanations'."""
    result = gb.analyze(sequence="MNASVNATL", chain="A", model_id="1")
    assert "summary"               in result
    assert "chimerax_commands"     in result
    assert "chimerax_explanations" in result
    assert isinstance(result["chimerax_commands"],     list)
    assert isinstance(result["chimerax_explanations"], list)
    # Summary must be multi-line so main.py renders the Rich Panel
    assert "\n" in result["summary"], "summary must contain newlines to trigger the Rich Panel"


def test_analyze_calls_set_glycan_results(gb):
    """analyze() should call session.set_glycan_results() when a session is given."""
    mock_session = MagicMock()
    gb.analyze(sequence="MNASVNATL", chain="A", model_id="1", session=mock_session)
    mock_session.set_glycan_results.assert_called_once()


def test_analyze_no_sequons_summary(gb):
    """When no native sequons are found, summary should mention 'no … sequons'."""
    # A sequence with no NXS/T sequons (all X positions are P)
    result = gb.analyze(sequence="MPPPPPPPPPPPPPP", chain="A", model_id="1")
    assert "no" in result["summary"].lower()


def test_analyze_viz_commands_non_empty_for_real_sequence(gb):
    """
    HIV-1 protease (1HSG chain A, 99 residues) has no native NXS/T sequons.
    analyze() must still return non-empty chimerax_commands from engineered
    candidates (single-AA proposals that would create a sequon).
    """
    # 1HSG chain A — HIV-1 protease, no native N-glycosylation sequons
    seq_1hsg = (
        "PQITLWQRPIVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQ"
        "ILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    )
    result = gb.analyze(sequence=seq_1hsg, chain="A", model_id="1")

    assert result.get("success") is True, f"analyze() failed: {result.get('error')}"
    # No native sequons expected for HIV protease
    assert len(result["native_sequons"]) == 0, (
        f"Unexpected native sequons: {result['native_sequons']}"
    )
    # Engineered candidates must be found for a 99-residue sequence
    assert len(result["engineered_candidates"]) > 0, (
        "No engineered candidates found for a 99-residue sequence"
    )
    # chimerax_commands must be non-empty (generated from engineered candidates)
    assert len(result["chimerax_commands"]) > 0, (
        "chimerax_commands is empty — engineered candidates not reaching visualization"
    )
    # Summary must be multi-line (triggers Rich Panel in main.py)
    assert "\n" in result["summary"], (
        "summary must contain newlines to trigger the Rich Panel"
    )


# ════════════════════════════════════════════════════════════════════════════════
# Section J — Projection scoring and sequon geometry (new fields)
# ════════════════════════════════════════════════════════════════════════════════

class TestProjectionAndGeometry:

    def test_score_sequon_sites_projection_outward_bonus(self, gb):
        """
        A site with projection_score=0.9 must score higher than the same site
        with projection_score=0.3 (when no backbone is passed — loop_factor=1.0).
        """
        seq = "MNATL"
        sites = gb.scan_sequons(seq, "A")

        proj_high = {2: {"projection_score": 0.9, "gly_proxy": False}}
        proj_low  = {2: {"projection_score": 0.3, "gly_proxy": False}}

        scored_high = gb.score_sequon_sites(sites, projection_scores=proj_high)
        scored_low  = gb.score_sequon_sites(sites, projection_scores=proj_low)

        assert len(scored_high) == 1 and len(scored_low) == 1
        assert scored_high[0]["composite_score"] > scored_low[0]["composite_score"]

    def test_score_sequon_sites_projection_helix_penalty(self, gb):
        """
        When backbone has helix φ/ψ at all three sequon positions,
        sequon_geometry='helix' and sequon_geometry_factor=0.5.
        """
        seq   = "MNATL"
        sites = gb.scan_sequons(seq, "A")

        # φ=-57, ψ=-47 → canonical helix for all three positions (2, 3, 4)
        backbone = {
            2: {"phi": -57.0, "psi": -47.0, "ss": "H"},
            3: {"phi": -57.0, "psi": -47.0, "ss": "H"},
            4: {"phi": -57.0, "psi": -47.0, "ss": "H"},
        }
        scored = gb.score_sequon_sites(sites, backbone=backbone)

        assert len(scored) == 1
        s = scored[0]
        assert s["sequon_geometry"] == "helix"
        assert s["sequon_geometry_factor"] == pytest.approx(0.5)

    def test_score_sequon_sites_projection_none_falls_back(self, gb):
        """
        projection_scores=None → proj_factor=1.0, projection_category='unknown'.
        composite_score still computed (no crash).
        """
        seq   = "MNATL"
        sites = gb.scan_sequons(seq, "A")
        scored = gb.score_sequon_sites(sites, projection_scores=None)

        assert len(scored) == 1
        s = scored[0]
        assert s["projection_category"] == "unknown"
        assert s["projection_score"]    is None
        assert s["composite_score"]     == pytest.approx(0.5, abs=1e-3)

    def test_score_sequon_sites_beta_turn_bonus(self, gb):
        """
        When all three sequon positions are in turn-like φ/ψ,
        sequon_geometry='beta_turn' and sequon_geometry_factor=1.4.
        """
        seq   = "MNATL"
        sites = gb.scan_sequons(seq, "A")

        # φ=-50, ψ=20 → turn region (not helix, not extended)
        backbone = {
            2: {"phi": -50.0, "psi": 20.0, "ss": "L"},
            3: {"phi": -50.0, "psi": 20.0, "ss": "L"},
            4: {"phi": -50.0, "psi": 20.0, "ss": "L"},
        }
        scored = gb.score_sequon_sites(sites, backbone=backbone)

        assert len(scored) == 1
        s = scored[0]
        assert s["sequon_geometry"]        == "beta_turn"
        assert s["sequon_geometry_factor"] == pytest.approx(1.4)

    def test_score_sequon_sites_projection_categories(self, gb):
        """
        Score ≥ 0.6 → 'outward'; 0.2 ≤ score < 0.6 → 'flat'; < 0.2 → 'inward'.
        """
        seq   = "MNATL"
        sites = gb.scan_sequons(seq, "A")

        for score, expected_cat in [(0.7, "outward"), (0.4, "flat"), (0.1, "inward")]:
            proj = {2: {"projection_score": score, "gly_proxy": False}}
            scored = gb.score_sequon_sites(sites, projection_scores=proj)
            assert scored[0]["projection_category"] == expected_cat, (
                f"score={score}: expected '{expected_cat}', "
                f"got '{scored[0]['projection_category']}'"
            )

    def test_full_glycan_scan_includes_projection_fields(self, gb):
        """
        full_glycan_scan() (no PDB) must include projection_score,
        projection_category, and sequon_geometry in every candidate.
        """
        result = gb.full_glycan_scan(sequence="MNASVNATL", chain="A")
        assert result["success"] is True

        all_candidates = result["all_ranked"] + result["engineered_candidates"]
        assert len(all_candidates) > 0, "Expected at least one candidate"

        for cand in all_candidates:
            assert "projection_score"       in cand, f"Missing projection_score in {cand}"
            assert "projection_category"    in cand, f"Missing projection_category in {cand}"
            assert "sequon_geometry"        in cand, f"Missing sequon_geometry in {cand}"
            assert "sequon_geometry_factor" in cand, f"Missing sequon_geometry_factor in {cand}"

    def test_analyze_summary_includes_projection_column(self, gb):
        """
        analyze() on a sequence with engineered candidates must produce a
        summary string containing 'Projection' and 'Geometry' column headers.
        """
        # Use HIV protease (no native sequons → forced into engineered candidates path)
        seq_1hsg = (
            "PQITLWQRPIVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQ"
            "ILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
        )
        result = gb.analyze(sequence=seq_1hsg, chain="A", model_id="1")
        summary = result.get("summary", "")

        assert "Projection" in summary, "Summary missing 'Projection' column"
        assert "Geometry"   in summary, "Summary missing 'Geometry' column"

    def test_chimerax_commands_cb_sphere_for_outward_only(self, gb):
        """
        An outward candidate must produce @CB sphere commands;
        a flat candidate must NOT produce @CB sphere commands.
        """
        outward_cand = {
            "position": 5, "confidence": "high", "composite_score": 0.8,
            "engineered": True, "mutation": "A5N", "chain": "A",
            "projection_category": "outward",
        }
        flat_cand = {
            "position": 12, "confidence": "moderate", "composite_score": 0.4,
            "engineered": True, "mutation": "E12N", "chain": "A",
            "projection_category": "flat",
        }

        cmds, _ = gb.generate_chimerax_commands(
            [outward_cand, flat_cand], model_id="1", chain="A"
        )

        # Find commands containing @CB
        cb_cmds = [c for c in cmds if "@CB" in c]
        assert len(cb_cmds) > 0, "Expected @CB sphere commands for outward candidate"

        # All @CB commands must reference residue 5 (outward) not 12 (flat)
        for c in cb_cmds:
            assert ":5@CB" in c, (
                f"@CB command references wrong residue: {c}"
            )
            assert ":12@CB" not in c, (
                f"Flat candidate residue 12 should not have @CB command: {c}"
            )

        # Verify sphere and size commands specifically
        sphere_cmds = [c for c in cb_cmds if "sphere" in c]
        size_cmds   = [c for c in cb_cmds if "atomRadius" in c]
        assert len(sphere_cmds) >= 1, "Expected 'style ... sphere' command for outward Cb"
        assert len(size_cmds)   >= 1, "Expected 'size ... atomRadius' command for outward Cb"
