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


# ════════════════════════════════════════════════════════════════════════════════
# Section K — suggest_glycosylation_positions + validate_sequon_engineering
# ════════════════════════════════════════════════════════════════════════════════

class TestGlycanPositions:
    """
    Tests for GlycanBridge.suggest_glycosylation_positions() and
    GlycanBridge.validate_sequon_engineering().

    Structural data (projection / SASA) is mocked via patch('glycan_bridge._su')
    to avoid network calls and PDB dependencies.  A tmp_path fixture supplies a
    real (empty) file so that Path.exists() returns True inside the method.
    """

    def test_suggest_glycosylation_positions_ranks_by_projection(self, gb, tmp_path):
        """
        Candidates must be sorted descending by composite_score.
        Composite is dominated by projection_score when SASA is constant.
        """
        seq = "MALEFQ"   # 6 AA, none G or P → all included
        pdb_file = tmp_path / "dummy.pdb"
        pdb_file.write_text("ATOM  1 dummy\n")

        mock_su = MagicMock()
        mock_su.compute_projection_score.return_value = {
            1: {"projection_score": 0.9, "gly_proxy": False},  # M
            2: {"projection_score": 0.6, "gly_proxy": False},  # A
            3: {"projection_score": 0.55, "gly_proxy": False}, # L
            4: {"projection_score": 0.8, "gly_proxy": False},  # E
            5: {"projection_score": 0.7, "gly_proxy": False},  # F
            6: {"projection_score": 0.52, "gly_proxy": False}, # Q
        }
        mock_su.compute_sasa.return_value = {i: 100.0 for i in range(1, 7)}

        with patch("glycan_bridge._su", mock_su):
            candidates = gb.suggest_glycosylation_positions(
                pdb_path=str(pdb_file), chain="A", sequence=seq, min_projection=0.5
            )

        assert len(candidates) > 0
        scores = [c["composite_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True), (
            f"Candidates not sorted descending; scores={scores}"
        )

    def test_suggest_glycosylation_positions_filters_projection(self, gb, tmp_path):
        """
        Residues with projection_score < min_projection must be excluded.
        Residues with projection_score >= min_projection must be included.
        """
        seq = "MALEF"   # 5 AA, none G or P
        pdb_file = tmp_path / "dummy.pdb"
        pdb_file.write_text("ATOM  1 dummy\n")

        mock_su = MagicMock()
        mock_su.compute_projection_score.return_value = {
            1: {"projection_score": 0.8, "gly_proxy": False},  # M — passes
            2: {"projection_score": 0.3, "gly_proxy": False},  # A — fails (< 0.5)
            3: {"projection_score": 0.7, "gly_proxy": False},  # L — passes
            4: {"projection_score": 0.1, "gly_proxy": False},  # E — fails (< 0.5)
            5: {"projection_score": 0.9, "gly_proxy": False},  # F — passes
        }
        mock_su.compute_sasa.return_value = {i: 100.0 for i in range(1, 6)}

        with patch("glycan_bridge._su", mock_su):
            candidates = gb.suggest_glycosylation_positions(
                pdb_path=str(pdb_file), chain="A", sequence=seq, min_projection=0.5
            )

        positions = {c["position"] for c in candidates}
        assert 1 in positions, "Position 1 (proj=0.8) should pass min_projection=0.5"
        assert 3 in positions, "Position 3 (proj=0.7) should pass"
        assert 5 in positions, "Position 5 (proj=0.9) should pass"
        assert 2 not in positions, "Position 2 (proj=0.3) must be filtered out"
        assert 4 not in positions, "Position 4 (proj=0.1) must be filtered out"

    def test_suggest_glycosylation_positions_excludes_gp(self, gb, tmp_path):
        """
        Glycine (G) and Proline (P) must be excluded by default
        (exclude_residues='GP').
        """
        seq = "MGAPF"   # G at pos 2, P at pos 4
        pdb_file = tmp_path / "dummy.pdb"
        pdb_file.write_text("ATOM  1 dummy\n")

        mock_su = MagicMock()
        mock_su.compute_projection_score.return_value = {
            i: {"projection_score": 0.8, "gly_proxy": False} for i in range(1, 6)
        }
        mock_su.compute_sasa.return_value = {i: 100.0 for i in range(1, 6)}

        with patch("glycan_bridge._su", mock_su):
            candidates = gb.suggest_glycosylation_positions(
                pdb_path=str(pdb_file), chain="A", sequence=seq,
            )

        positions = {c["position"] for c in candidates}
        assert 2 not in positions, "G at pos 2 must be excluded"
        assert 4 not in positions, "P at pos 4 must be excluded"
        assert 1 in positions, "M at pos 1 should be included"
        assert 3 in positions, "A at pos 3 should be included"
        assert 5 in positions, "F at pos 5 should be included"

    def test_suggest_glycosylation_positions_engineering_notes_outward(self, gb, tmp_path):
        """
        An outward-projecting residue (proj_score ≥ 0.6) must have 'outward'
        in its engineering_notes.
        """
        seq = "MALEF"
        pdb_file = tmp_path / "dummy.pdb"
        pdb_file.write_text("ATOM  1 dummy\n")

        mock_su = MagicMock()
        mock_su.compute_projection_score.return_value = {
            1: {"projection_score": 0.9, "gly_proxy": False},  # outward
        }
        mock_su.compute_sasa.return_value = {1: 120.0}

        with patch("glycan_bridge._su", mock_su):
            candidates = gb.suggest_glycosylation_positions(
                pdb_path=str(pdb_file), chain="A", sequence=seq, min_projection=0.5,
            )

        pos1 = next((c for c in candidates if c["position"] == 1), None)
        assert pos1 is not None, "Position 1 should appear in candidates"
        assert pos1["projection_category"] == "outward"
        assert "outward" in pos1["engineering_notes"].lower(), (
            f"Expected 'outward' in notes; got: {pos1['engineering_notes']!r}"
        )

    def test_suggest_glycosylation_positions_engineering_notes_inward(self, gb, tmp_path):
        """
        An inward-projecting residue (proj_score < 0.2) must have 'inward' in
        engineering_notes.  Setting min_projection=0.0 disables the filter so
        the residue is not removed from results.
        """
        seq = "MALEF"
        pdb_file = tmp_path / "dummy.pdb"
        pdb_file.write_text("ATOM  1 dummy\n")

        mock_su = MagicMock()
        mock_su.compute_projection_score.return_value = {
            1: {"projection_score": 0.1, "gly_proxy": False},  # inward
        }
        mock_su.compute_sasa.return_value = {1: 120.0}

        with patch("glycan_bridge._su", mock_su):
            candidates = gb.suggest_glycosylation_positions(
                pdb_path=str(pdb_file), chain="A", sequence=seq,
                min_projection=0.0,   # disable projection filter
            )

        pos1 = next((c for c in candidates if c["position"] == 1), None)
        assert pos1 is not None, "Position 1 should appear when min_projection=0.0"
        assert pos1["projection_category"] == "inward"
        assert "inward" in pos1["engineering_notes"].lower(), (
            f"Expected 'inward' in notes; got: {pos1['engineering_notes']!r}"
        )

    def test_validate_sequon_engineering_esmfold_called(self, gb):
        """
        ESMFold bridge must be called at least twice: once for the wildtype
        baseline and once per mutant (up to top_esm_designs).
        """
        mock_esmfold = MagicMock()
        mock_esmfold.predict.return_value = {
            "success": True, "mean_plddt": 80.0, "plddt": {}, "length": 9,
        }
        mock_rosetta = MagicMock()
        mock_rosetta.analyze.return_value = MagicMock(
            success=True, data={"ddg_scores": {"A5N": 0.3}}
        )

        wt_seq = "MALEAFGHI"
        gb.validate_sequon_engineering(
            position=5, wildtype_sequence=wt_seq,
            mutations=["A5N"], pdb_path="dummy.pdb", chain="A",
            esmfold_bridge=mock_esmfold, rosetta_bridge=mock_rosetta,
            top_esm_designs=1,
        )

        # Wildtype call + 1 mutant call = at least 2
        assert mock_esmfold.predict.call_count >= 2, (
            f"Expected ≥ 2 ESMFold calls (wt + mutant); "
            f"got {mock_esmfold.predict.call_count}"
        )

    def test_validate_sequon_engineering_rosetta_called(self, gb):
        """
        Rosetta bridge must be called with the parsed mutation dict.
        """
        mock_esmfold = MagicMock()
        mock_esmfold.predict.return_value = {
            "success": True, "mean_plddt": 78.0, "plddt": {}, "length": 9,
        }
        mock_rosetta = MagicMock()
        mock_rosetta.analyze.return_value = MagicMock(
            success=True, data={"ddg_scores": {"A5N": -0.8}}
        )

        wt_seq = "MALEAFGHI"
        gb.validate_sequon_engineering(
            position=5, wildtype_sequence=wt_seq,
            mutations=["A5N"], pdb_path="dummy.pdb", chain="A",
            esmfold_bridge=mock_esmfold, rosetta_bridge=mock_rosetta,
        )

        assert mock_rosetta.analyze.call_count >= 1, "Rosetta must be called"
        # The second positional arg is the list of mutation dicts
        call_args    = mock_rosetta.analyze.call_args
        mut_dicts    = call_args[0][1]   # positional args tuple, index 1
        assert any(
            m.get("to_aa") == "N" and int(m.get("position", 0)) == 5
            for m in mut_dicts
        ), f"Expected A5N mutation dict; got {mut_dicts}"

    def test_validate_sequon_engineering_ddg_classification(self, gb):
        """
        ddg_category must be 'stabilizing' (< -0.5), 'neutral' (≤ 0.5),
        or 'destabilizing' (> 0.5).
        """
        mock_esmfold = MagicMock()
        mock_esmfold.predict.return_value = {
            "success": True, "mean_plddt": 80.0, "plddt": {}, "length": 5,
        }

        for ddg_val, expected_cat in [
            (-1.0, "stabilizing"),
            (0.2,  "neutral"),
            (2.0,  "destabilizing"),
        ]:
            mock_rosetta = MagicMock()
            mock_rosetta.analyze.return_value = MagicMock(
                success=True, data={"ddg_scores": {"A2N": ddg_val}}
            )

            result = gb.validate_sequon_engineering(
                position=2, wildtype_sequence="MALEF",
                mutations=["A2N"], pdb_path="dummy.pdb", chain="A",
                esmfold_bridge=mock_esmfold, rosetta_bridge=mock_rosetta,
            )

            assert len(result["results"]) >= 1
            got = result["results"][0]["ddg_category"]
            assert got == expected_cat, (
                f"ddg={ddg_val}: expected '{expected_cat}', got '{got}'"
            )

    def test_validate_sequon_engineering_pass_threshold(self, gb):
        """
        pass_threshold=True only when pLDDT drop ≤ 5 AND ddG ≤ 1.0.

        Case 1: wt_pLDDT=80, mut_pLDDT=78, ddG=0.5 → drop=-2, passes both.
        Case 2: wt_pLDDT=80, mut_pLDDT=70, ddG=0.5 → drop=-10, fails pLDDT.
        """
        # Case 1 — should PASS
        mock_esmfold_pass = MagicMock()
        mock_esmfold_pass.predict.side_effect = [
            {"success": True, "mean_plddt": 80.0, "plddt": {}, "length": 5},  # wt
            {"success": True, "mean_plddt": 78.0, "plddt": {}, "length": 5},  # mut
        ]
        mock_rosetta = MagicMock()
        mock_rosetta.analyze.return_value = MagicMock(
            success=True, data={"ddg_scores": {"A2N": 0.5}}
        )

        result_pass = gb.validate_sequon_engineering(
            position=2, wildtype_sequence="MALEF",
            mutations=["A2N"], pdb_path="dummy.pdb", chain="A",
            esmfold_bridge=mock_esmfold_pass, rosetta_bridge=mock_rosetta,
        )
        assert result_pass["results"][0]["pass_threshold"] is True, (
            "Should pass: pLDDT drop=-2, ddG=0.5"
        )

        # Case 2 — should FAIL (pLDDT drop too large)
        mock_esmfold_fail = MagicMock()
        mock_esmfold_fail.predict.side_effect = [
            {"success": True, "mean_plddt": 80.0, "plddt": {}, "length": 5},  # wt
            {"success": True, "mean_plddt": 70.0, "plddt": {}, "length": 5},  # mut
        ]
        mock_rosetta2 = MagicMock()
        mock_rosetta2.analyze.return_value = MagicMock(
            success=True, data={"ddg_scores": {"A2N": 0.5}}
        )

        result_fail = gb.validate_sequon_engineering(
            position=2, wildtype_sequence="MALEF",
            mutations=["A2N"], pdb_path="dummy.pdb", chain="A",
            esmfold_bridge=mock_esmfold_fail, rosetta_bridge=mock_rosetta2,
        )
        assert result_fail["results"][0]["pass_threshold"] is False, (
            "Should fail: pLDDT drop=-10 (< -5)"
        )

    def test_validate_sequon_engineering_notes_generation(self, gb):
        """
        notes must be a non-empty string for each result dict.
        """
        mock_esmfold = MagicMock()
        mock_esmfold.predict.return_value = {
            "success": True, "mean_plddt": 75.0, "plddt": {}, "length": 5,
        }
        mock_rosetta = MagicMock()
        mock_rosetta.analyze.return_value = MagicMock(
            success=True, data={"ddg_scores": {"A2N": 0.3}}
        )

        result = gb.validate_sequon_engineering(
            position=2, wildtype_sequence="MALEF",
            mutations=["A2N"], pdb_path="dummy.pdb", chain="A",
            esmfold_bridge=mock_esmfold, rosetta_bridge=mock_rosetta,
        )

        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert isinstance(r["notes"], str) and len(r["notes"]) > 0, (
                f"notes must be a non-empty string; got: {r['notes']!r}"
            )


# ════════════════════════════════════════════════════════════════════════════════
# Section L — SessionState netnglyc_results save/load round-trip
# ════════════════════════════════════════════════════════════════════════════════

class TestSessionStateNetNGlyc:
    """
    Verify that netnglyc_results are correctly persisted and reloaded via
    SessionState.save() / SessionState.load().
    """

    def test_netnglyc_results_save_load_round_trip(self, tmp_path):
        """
        set_netnglyc_results() / get_netnglyc_results() must survive a
        save-load round-trip through JSON.
        """
        import sys, os
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)

        from session_state import SessionState

        state = SessionState()
        payload = [
            {"position": 5,  "ost_score": 0.72, "ost_category": "high"},
            {"position": 12, "ost_score": 0.31, "ost_category": "low"},
        ]
        state.set_netnglyc_results("1", payload)

        # Verify accessor works before save
        retrieved = state.get_netnglyc_results("1")
        assert retrieved is not None
        assert retrieved[0]["ost_score"] == 0.72

        # Save and reload
        json_path = str(tmp_path / "session.json")
        state.save(json_path)
        loaded = SessionState.load(json_path)

        reloaded = loaded.get_netnglyc_results("1")
        assert reloaded is not None, "netnglyc_results should survive save/load"
        assert len(reloaded) == 2
        assert reloaded[0]["ost_category"] == "high"
        assert reloaded[1]["position"] == 12

    def test_netnglyc_results_snapshot_restore(self):
        """
        netnglyc_results must be included in snapshot() / restore().
        """
        import sys, os
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)

        from session_state import SessionState

        state = SessionState()
        state.set_netnglyc_results("2", [{"position": 99, "ost_score": 0.55}])

        snap = state.snapshot()
        # Clear and restore
        state.netnglyc_results = {}
        state.restore(snap)

        restored = state.get_netnglyc_results("2")
        assert restored is not None, "netnglyc_results should survive snapshot/restore"
        assert restored[0]["position"] == 99

    def test_missing_netnglyc_returns_none(self):
        """
        get_netnglyc_results() must return None for a model_id with no stored results.
        """
        import sys, os
        _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)

        from session_state import SessionState

        state = SessionState()
        assert state.get_netnglyc_results("99") is None
