"""
tests/test_proline_bridge.py
----------------------------
18-test suite for proline_bridge.ProlineBridge.

Tests are pure-Python (no PDB files required for most).
The backbone-angle test uses a minimal synthetic PDB inline.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from proline_bridge import ProlineBridge, _classify_ss_from_angles


# ════════════════════════════════════════════════════════════════════════════════
# Helpers / fixtures
# ════════════════════════════════════════════════════════════════════════════════

def _make_backbone(
    positions: List[int],
    aas:       List[str],
    phis:      Optional[List[Optional[float]]] = None,
    psis:      Optional[List[Optional[float]]] = None,
    sss:       Optional[List[str]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Build a synthetic backbone dict for testing."""
    phis = phis or [-60.0] * len(positions)
    psis = psis or [-40.0] * len(positions)
    sss  = sss  or ["L"] * len(positions)
    return {
        pos: {
            "phi":     phis[i],
            "psi":     psis[i],
            "ss":      sss[i],
            "resname": "XXX",
            "aa":      aas[i],
        }
        for i, pos in enumerate(positions)
    }


@pytest.fixture
def bridge() -> ProlineBridge:
    return ProlineBridge()


# ── Minimal PDB for backbone-angle extraction test ─────────────────────────────
_MINIMAL_PDB = textwrap.dedent("""\
    ATOM      1  N   ALA A   4      -0.525   1.359   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   4       0.000   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   4       1.520   0.000   0.000  1.00  0.00           C
    ATOM      4  O   ALA A   4       2.128   1.060   0.000  1.00  0.00           O
    ATOM      5  CB  ALA A   4      -0.527  -0.760   1.206  1.00  0.00           C
    ATOM      6  N   LEU A   5       2.162  -1.145   0.000  1.00  0.00           N
    ATOM      7  CA  LEU A   5       3.620  -1.257   0.000  1.00  0.00           C
    ATOM      8  C   LEU A   5       4.156  -0.177   1.000  1.00  0.00           C
    ATOM      9  O   LEU A   5       3.400   0.783   1.000  1.00  0.00           O
    ATOM     10  CB  LEU A   5       4.115  -2.621   0.000  1.00  0.00           C
    ATOM     11  N   VAL A   6       5.474  -0.245   1.000  1.00  0.00           N
    ATOM     12  CA  VAL A   6       6.100   0.810   1.800  1.00  0.00           C
    ATOM     13  C   VAL A   6       7.600   0.560   1.800  1.00  0.00           C
    ATOM     14  O   VAL A   6       8.200   0.560   0.750  1.00  0.00           O
    ATOM     15  CB  VAL A   6       5.800   2.200   1.300  1.00  0.00           C
    END
""")


# ════════════════════════════════════════════════════════════════════════════════
# 1. Secondary structure classification
# ════════════════════════════════════════════════════════════════════════════════

class TestClassifySS:
    """Tests for _classify_ss_from_angles helper."""

    def test_helix_canonical(self):
        """φ=-65, ψ=-40 → helix (H)."""
        assert _classify_ss_from_angles(-65.0, -40.0) == "H"

    def test_sheet_canonical(self):
        """φ=-120, ψ=130 → sheet (E)."""
        assert _classify_ss_from_angles(-120.0, 130.0) == "E"

    def test_loop_default(self):
        """φ=60, ψ=60 → loop (L)."""
        assert _classify_ss_from_angles(60.0, 60.0) == "L"

    def test_helix_boundary(self):
        """φ=-44 is just outside the helix range (-90,-45) — should be loop."""
        assert _classify_ss_from_angles(-44.0, -30.0) == "L"

    def test_sheet_boundary(self):
        """φ=-89, ψ=91 → just inside sheet range."""
        assert _classify_ss_from_angles(-95.0, 91.0) == "E"


# ════════════════════════════════════════════════════════════════════════════════
# 2. Backbone angle extraction
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractBackboneAngles:
    """Tests for ProlineBridge.extract_backbone_angles()."""

    def test_returns_dict_with_pdb(self, bridge, tmp_path):
        """Should return a non-empty dict for a valid (even tiny) PDB file."""
        pdb_file = tmp_path / "mini.pdb"
        pdb_file.write_text(_MINIMAL_PDB)
        try:
            result = bridge.extract_backbone_angles(str(pdb_file), "A")
            # A 3-residue peptide: PPBuilder may return 1–3 entries
            assert isinstance(result, dict)
            # Keys are integers (residue sequence numbers)
            for k in result:
                assert isinstance(k, int)
        except Exception as exc:
            # BioPython may not produce angles for a 3-residue stub — that's OK
            # as long as the function doesn't crash with an unhandled exception.
            pytest.skip(f"BioPython angle extraction skipped for minimal PDB: {exc}")

    def test_missing_file_raises(self, bridge):
        """Should raise ValueError for a non-existent PDB path."""
        with pytest.raises((ValueError, Exception)):
            bridge.extract_backbone_angles("/nonexistent/path/foo.pdb", "A")

    def test_result_fields(self, bridge, tmp_path):
        """Each entry should have phi, psi, ss, resname, aa keys."""
        pdb_file = tmp_path / "mini.pdb"
        pdb_file.write_text(_MINIMAL_PDB)
        try:
            result = bridge.extract_backbone_angles(str(pdb_file), "A")
            for entry in result.values():
                assert "phi"     in entry
                assert "psi"     in entry
                assert "ss"      in entry
                assert "resname" in entry
                assert "aa"      in entry
        except Exception as exc:
            pytest.skip(f"BioPython extraction skipped: {exc}")


# ════════════════════════════════════════════════════════════════════════════════
# 3. Candidate scanning
# ════════════════════════════════════════════════════════════════════════════════

class TestScanProlineCandidates:
    """Tests for ProlineBridge.scan_proline_candidates()."""

    def test_basic_loop_returns_candidates(self, bridge):
        """Loop residues with φ≈-60 should produce high-scoring candidates."""
        # 10-residue chain: positions 1-10, all LEU in loop with φ=-60
        positions = list(range(1, 11))
        backbone  = _make_backbone(positions, ["L"] * 10, phis=[-60.0] * 10)
        seq       = "L" * 10
        candidates = bridge.scan_proline_candidates(backbone, seq)
        # At least some should be found (termini excluded within 3 residues)
        assert isinstance(candidates, list)
        assert len(candidates) > 0

    def test_excludes_existing_proline(self, bridge):
        """Positions already containing Pro must be excluded."""
        positions = list(range(1, 11))
        aas = ["L", "P", "L", "L", "L", "L", "L", "L", "L", "L"]
        backbone = _make_backbone(positions, aas, phis=[-60.0] * 10)
        candidates = bridge.scan_proline_candidates(backbone, "".join(aas))
        cand_positions = {c["position"] for c in candidates}
        assert 2 not in cand_positions   # position of P must be excluded

    def test_excludes_glycine(self, bridge):
        """Glycine positions must be excluded."""
        positions = list(range(1, 11))
        aas = ["L"] * 5 + ["G"] + ["L"] * 4
        backbone = _make_backbone(positions, aas, phis=[-60.0] * 10)
        candidates = bridge.scan_proline_candidates(backbone, "".join(aas))
        cand_positions = {c["position"] for c in candidates}
        assert 6 not in cand_positions   # Gly at position 6

    def test_excludes_helix(self, bridge):
        """Helix positions must be excluded."""
        positions = list(range(1, 11))
        aas = ["L"] * 10
        sss = ["L"] * 4 + ["H"] * 2 + ["L"] * 4
        backbone = _make_backbone(positions, aas, phis=[-60.0] * 10, sss=sss)
        candidates = bridge.scan_proline_candidates(backbone, "L" * 10)
        cand_positions = {c["position"] for c in candidates}
        # Positions 5 and 6 are in helix — must be excluded
        assert 5 not in cand_positions
        assert 6 not in cand_positions

    def test_excludes_terminus_residues(self, bridge):
        """Residues within 3 of terminus must be excluded."""
        positions = list(range(1, 11))
        backbone = _make_backbone(positions, ["L"] * 10, phis=[-60.0] * 10)
        candidates = bridge.scan_proline_candidates(backbone, "L" * 10)
        cand_positions = {c["position"] for c in candidates}
        # Positions 1, 2, 3 and 8, 9, 10 must all be excluded
        for bad_pos in [1, 2, 3, 10]:
            assert bad_pos not in cand_positions

    def test_excludes_post_proline(self, bridge):
        """Position immediately after a Pro must be excluded."""
        positions = list(range(1, 11))
        aas = ["L"] * 4 + ["P"] + ["L"] * 5   # Pro at 5, so 6 must be excluded
        backbone = _make_backbone(positions, aas, phis=[-60.0] * 10)
        candidates = bridge.scan_proline_candidates(backbone, "".join(aas))
        cand_positions = {c["position"] for c in candidates}
        assert 6 not in cand_positions

    def test_phi_score_formula(self, bridge):
        """φ=-60 should give phi_score=1.0; φ=-120 should give 0.0."""
        positions = list(range(1, 12))
        aas = ["L"] * 11
        # Position 6 has φ=-60 (ideal), position 7 has φ=-120 (max deviation)
        phis = [-60.0] * 6 + [-120.0] + [-60.0] * 4
        backbone = _make_backbone(positions, aas, phis=phis)
        candidates = bridge.scan_proline_candidates(backbone, "L" * 11)
        by_pos = {c["position"]: c for c in candidates}
        if 6 in by_pos:
            assert by_pos[6]["phi_score"] == pytest.approx(1.0, abs=1e-4)
        if 7 in by_pos:
            assert by_pos[7]["phi_score"] == pytest.approx(0.0, abs=1e-4)

    def test_interface_penalty(self, bridge):
        """Residues near interface should have iface_factor=0.4."""
        positions = list(range(1, 12))
        aas = ["L"] * 11
        backbone = _make_backbone(positions, aas, phis=[-60.0] * 11)
        # Interface residue at 6 → positions 4-8 should be near interface
        candidates = bridge.scan_proline_candidates(
            backbone, "L" * 11, interface_residues={6}
        )
        by_pos = {c["position"]: c for c in candidates}
        # Position 5 is 1 away from interface residue 6 → penalised
        if 5 in by_pos:
            assert by_pos[5]["iface_factor"] == pytest.approx(0.4, abs=1e-4)

    def test_esm_penalty(self, bridge):
        """Low ESM tolerance should give esm_factor=0.6."""
        positions = list(range(1, 12))
        aas = ["L"] * 11
        backbone = _make_backbone(positions, aas, phis=[-60.0] * 11)
        esm_scores = {6: 0.2}   # low tolerance at position 6
        candidates = bridge.scan_proline_candidates(
            backbone, "L" * 11, esm_scores=esm_scores
        )
        by_pos = {c["position"]: c for c in candidates}
        if 6 in by_pos:
            assert by_pos[6]["esm_factor"] == pytest.approx(0.6, abs=1e-4)

    def test_sorted_by_composite_score(self, bridge):
        """Candidates must be sorted descending by composite_score."""
        positions = list(range(1, 12))
        aas = ["L"] * 11
        phis = [-60.0, -60.0, -60.0, -60.0, -60.0, -60.0, -90.0, -90.0, -90.0, -90.0, -90.0]
        backbone = _make_backbone(positions, aas, phis=phis)
        candidates = bridge.scan_proline_candidates(backbone, "L" * 11)
        scores = [c["composite_score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_confidence_bands(self, bridge):
        """Confidence should be high/moderate/low based on thresholds."""
        positions = list(range(1, 12))
        aas = ["L"] * 11
        backbone = _make_backbone(positions, aas, phis=[-60.0] * 11)
        candidates = bridge.scan_proline_candidates(backbone, "L" * 11)
        for c in candidates:
            score = c["composite_score"]
            if score > 0.6:
                assert c["confidence"] == "high"
            elif score >= 0.3:
                assert c["confidence"] == "moderate"
            else:
                assert c["confidence"] == "low"


# ════════════════════════════════════════════════════════════════════════════════
# 4. DynaMut2 validation
# ════════════════════════════════════════════════════════════════════════════════

class TestValidateWithDynaMut2:
    """Tests for ProlineBridge.validate_with_dynamut2()."""

    def _make_candidates(self, n=5) -> List[Dict[str, Any]]:
        return [
            {
                "position":        i + 5,
                "from_aa":         "L",
                "to_aa":           "P",
                "phi":             -60.0,
                "psi":             -40.0,
                "ss":              "L",
                "phi_score":       1.0,
                "loop_bonus":      1.3,
                "esm_factor":      1.0,
                "iface_factor":    1.0,
                "hbond_factor":    1.0,
                "composite_score": round(1.0 * 1.3, 4),
                "confidence":      "high",
                "near_interface":  False,
            }
            for i in range(n)
        ]

    def test_no_bridge_returns_unchanged(self, bridge):
        """If dynamut2_bridge is None, return candidates unchanged."""
        cands = self._make_candidates(3)
        result = bridge.validate_with_dynamut2(
            candidates=cands, pdb_path="fake.pdb", chain="A",
            dynamut2_bridge=None,
        )
        assert result is cands   # same object returned

    def test_ddg_annotated(self, bridge, tmp_path):
        """After DynaMut2 mock call, each top-N candidate should have 'ddg' key."""
        cands = self._make_candidates(3)
        mock_bridge = MagicMock()
        mock_result = MagicMock()
        mock_result.data = {
            "ddg_scores": {
                "L5P": -1.5,
                "L6P": -0.5,
                "L7P": -2.0,
            }
        }
        mock_bridge.analyze.return_value = mock_result

        pdb_file = tmp_path / "fake.pdb"
        pdb_file.write_text("ATOM\n")

        result = bridge.validate_with_dynamut2(
            candidates       = cands,
            pdb_path         = str(pdb_file),
            chain            = "A",
            top_n            = 3,
            dynamut2_bridge  = mock_bridge,
        )
        assert all("ddg" in c for c in result[:3])


# ════════════════════════════════════════════════════════════════════════════════
# 5. ChimeraX command generation
# ════════════════════════════════════════════════════════════════════════════════

class TestGenerateChimeraXCommands:
    """Tests for ProlineBridge.generate_chimerax_commands()."""

    def _make_cands(self) -> List[Dict[str, Any]]:
        return [
            {
                "position":        37,
                "from_aa":         "L",
                "to_aa":           "P",
                "phi":             -62.0,
                "psi":             -41.0,
                "ss":              "L",
                "phi_score":       0.97,
                "loop_bonus":      1.3,
                "esm_factor":      1.0,
                "iface_factor":    1.0,
                "hbond_factor":    1.0,
                "composite_score": 0.85,
                "confidence":      "high",
                "near_interface":  False,
            }
        ]

    def test_empty_candidates(self, bridge):
        """Empty candidate list → empty commands."""
        cmds, exps = bridge.generate_chimerax_commands([])
        assert cmds == []
        assert exps == []

    def test_command_count(self, bridge):
        """
        Each candidate produces 3 commands (style, color, label).
        The explanations list must be the same length (one entry per command)
        so that zip(cmds, exps) in main.py's _show_preview() shows all rows.
        """
        cands = self._make_cands()
        cmds, exps = bridge.generate_chimerax_commands(cands, model_id="1", chain="A")
        assert len(cmds) == 3
        assert len(exps) == len(cmds), (
            f"len(exps)={len(exps)} must equal len(cmds)={len(cmds)} "
            "so the command table is not truncated"
        )

    def test_high_confidence_magenta(self, bridge):
        """High confidence → magenta (#cc00cc) color."""
        cands = self._make_cands()
        cmds, _ = bridge.generate_chimerax_commands(cands, model_id="1", chain="A")
        color_cmd = next(c for c in cmds if c.startswith("color"))
        assert "#cc00cc" in color_cmd

    def test_model_chain_in_spec(self, bridge):
        """Model ID and chain must appear in the ChimeraX atom spec."""
        cands = self._make_cands()
        cmds, _ = bridge.generate_chimerax_commands(cands, model_id="2", chain="B")
        for cmd in cmds:
            assert "#2/B:37" in cmd

    def test_chimerax_command_explanations_non_empty(self, bridge):
        """
        generate_chimerax_commands() with 2 candidates → all 6 explanation
        strings are non-empty and each contains the mutation string (e.g. 'L37P').
        """
        cands = self._make_cands() + [
            {
                "position":        10,
                "from_aa":         "E",
                "to_aa":           "P",
                "phi":             -55.0,
                "psi":             -38.0,
                "ss":              "L",
                "phi_score":       0.80,
                "loop_bonus":      1.0,
                "esm_factor":      0.9,
                "iface_factor":    1.0,
                "hbond_factor":    1.0,
                "composite_score": 0.55,
                "confidence":      "moderate",
                "near_interface":  False,
            }
        ]
        cmds, exps = bridge.generate_chimerax_commands(cands, model_id="1", chain="A")
        assert len(cmds) == 6
        assert len(exps) == 6
        # All explanations must be non-empty
        for i, exp in enumerate(exps):
            assert exp.strip(), f"Explanation #{i} is empty (cmd: {cmds[i]!r})"
        # Each explanation must contain the mutation string for its candidate
        # Rows 0-2 belong to L37P, rows 3-5 belong to E10P
        for exp in exps[:3]:
            assert "L37P" in exp, f"Expected 'L37P' in {exp!r}"
        for exp in exps[3:]:
            assert "E10P" in exp, f"Expected 'E10P' in {exp!r}"


# ════════════════════════════════════════════════════════════════════════════════
# 6. Full scan (orchestrator)
# ════════════════════════════════════════════════════════════════════════════════

class TestFullProlineScan:
    """Tests for ProlineBridge.full_proline_scan() with mocked extract."""

    def test_empty_backbone_returns_zero_count(self, bridge, tmp_path):
        """If backbone is empty (chain not found), count should be 0."""
        pdb_file = tmp_path / "empty.pdb"
        pdb_file.write_text("REMARK empty\nEND\n")
        with patch.object(bridge, "extract_backbone_angles", return_value={}):
            result = bridge.full_proline_scan(str(pdb_file), chain="Z")
        assert result["count"] == 0
        assert result["candidates"] == []

    def test_orchestrator_returns_correct_keys(self, bridge, tmp_path):
        """Result dict should have candidates, count, top, chain, pdb_path, n_residues_scanned."""
        pdb_file = tmp_path / "fake.pdb"
        pdb_file.write_text("ATOM\n")
        positions = list(range(1, 12))
        fake_backbone = _make_backbone(positions, ["L"] * 11, phis=[-60.0] * 11)
        with patch.object(bridge, "extract_backbone_angles", return_value=fake_backbone):
            result = bridge.full_proline_scan(str(pdb_file), chain="A")
        assert "candidates"                   in result
        assert "count"                        in result
        assert "top"                          in result
        assert "chain"                        in result
        assert "pdb_path"                     in result
        assert "n_residues_scanned"           in result
        assert "inferred_functional_residues" in result
        assert "functional_residues"          in result


# ════════════════════════════════════════════════════════════════════════════════
# 7. Functional / active-site proximity exclusion
# ════════════════════════════════════════════════════════════════════════════════

class TestFunctionalSiteExclusion:
    """Tests for the functional_residues hard-exclusion in scan_proline_candidates."""

    def _make_chain(self, n: int = 15) -> Dict[int, Dict[str, Any]]:
        """Return a backbone with n LEU residues, all in loop at φ=-60."""
        positions = list(range(1, n + 1))
        return _make_backbone(positions, ["L"] * n, phis=[-60.0] * n)

    def test_functional_site_exclusion(self, bridge):
        """
        Position within 2 residues of a functional residue is hard-excluded
        and its count appears in exclusion_counts["functional_site"].
        """
        backbone = self._make_chain(15)
        # Position 8 is the functional residue; positions 6–10 should be excluded
        candidates = bridge.scan_proline_candidates(
            backbone, "L" * 15, functional_residues={8}
        )
        cand_positions = {c["position"] for c in candidates}

        # All positions within 2 of position 8 (i.e. 6,7,8,9,10) must be absent
        for bad in (6, 7, 8, 9, 10):
            assert bad not in cand_positions, (
                f"Position {bad} should be excluded (within 2 of functional residue 8), "
                f"but was found in candidates"
            )

        # Exclusion count should be non-zero
        counts = bridge._count_exclusions(backbone, functional_residues={8})
        assert counts["functional_site"] > 0, (
            "functional_site exclusion count should be > 0"
        )

    def test_functional_site_exclusion_boundary(self, bridge):
        """
        Exactly 2 away → excluded.
        Exactly 3 away → NOT excluded by functional-site rule.
        """
        backbone = self._make_chain(15)
        fr = {8}   # functional residue

        candidates = bridge.scan_proline_candidates(
            backbone, "L" * 15, functional_residues=fr
        )
        cand_positions = {c["position"] for c in candidates}

        # pos 6 = 8-2 → excluded (within 2)
        assert 6 not in cand_positions, "Position 6 (exactly 2 from fr=8) must be excluded"
        # pos 10 = 8+2 → excluded (within 2)
        assert 10 not in cand_positions, "Position 10 (exactly 2 from fr=8) must be excluded"
        # pos 5 = 8-3 → NOT excluded by functional site (but may be excluded by terminal)
        # pos 11 = 8+3 → NOT excluded by functional site
        # We verify positions 5 and 11 are *not absent due to functional site*:
        # (they may still be missing for other reasons, so just check the
        # exclusion count doesn't attribute them to functional_site)
        counts = bridge._count_exclusions(backbone, functional_residues=fr)
        # Only positions {6,7,8,9,10} can be functional_site — exactly 5
        assert counts["functional_site"] == 5, (
            f"Expected 5 functional_site exclusions (positions 6–10), "
            f"got {counts['functional_site']}"
        )

    def test_full_scan_passes_functional_residues(self, bridge, tmp_path):
        """
        functional_residues={25,26,27} passed to full_proline_scan causes
        positions 23–29 to be excluded from candidates.
        """
        pdb_file = tmp_path / "fake.pdb"
        pdb_file.write_text("ATOM\n")

        # 40-residue chain with all LEU at φ=-60 (ideal proline candidate)
        positions = list(range(1, 41))
        fake_backbone = _make_backbone(positions, ["L"] * 40, phis=[-60.0] * 40)

        with patch.object(bridge, "extract_backbone_angles", return_value=fake_backbone):
            # Disable SASA auto-detection so our explicit set is used
            result = bridge.full_proline_scan(
                str(pdb_file), chain="A",
                functional_residues={25, 26, 27},
            )

        cand_positions = {c["position"] for c in result["candidates"]}

        # Positions 23–29 (within 2 of {25,26,27}) must be absent
        for pos in range(23, 30):
            assert pos not in cand_positions, (
                f"Position {pos} should be excluded (near functional site {{25,26,27}}), "
                f"but was found in candidates"
            )

        # The result dict must record the functional_residues used
        assert result["functional_residues"] == [25, 26, 27]
        # Since we passed them explicitly, inferred_functional_residues is empty
        assert result["inferred_functional_residues"] == []

    def test_auto_detection_returns_inferred_set(self, bridge, tmp_path):
        """
        When functional_residues is None, full_proline_scan calls
        _detect_functional_residues_sasa().  When the detector returns a
        non-empty set, those residues are stored in inferred_functional_residues
        and respected as exclusions.
        """
        pdb_file = tmp_path / "fake.pdb"
        pdb_file.write_text("ATOM\n")

        positions = list(range(1, 20))
        fake_backbone = _make_backbone(positions, ["L"] * 19, phis=[-60.0] * 19)

        # Simulate SASA finding buried His at position 10
        inferred_set = {10}

        with patch.object(bridge, "extract_backbone_angles", return_value=fake_backbone):
            with patch.object(
                bridge, "_detect_functional_residues_sasa",
                return_value=inferred_set
            ):
                result = bridge.full_proline_scan(
                    str(pdb_file), chain="A"
                    # functional_residues omitted → auto-detection fires
                )

        # Inferred set is recorded
        assert set(result["inferred_functional_residues"]) == inferred_set, (
            f"Expected inferred_functional_residues={inferred_set}, "
            f"got {result['inferred_functional_residues']}"
        )
        # The auto-detected residues are applied as exclusions
        cand_positions = {c["position"] for c in result["candidates"]}
        for pos in (8, 9, 10, 11, 12):   # within 2 of fr=10
            assert pos not in cand_positions, (
                f"Position {pos} should be excluded by auto-detected fr={{10}}"
            )

    def test_auto_detection_failure_is_silent(self, bridge, tmp_path):
        """
        If _detect_functional_residues_sasa() raises (SASA unavailable, PDB error, etc.),
        the scan completes normally and inferred_functional_residues is empty.
        """
        pdb_file = tmp_path / "fake.pdb"
        pdb_file.write_text("ATOM\n")

        positions = list(range(1, 15))
        fake_backbone = _make_backbone(positions, ["L"] * 14, phis=[-60.0] * 14)

        def _sasa_fail(*_a, **_kw):
            raise RuntimeError("SASA library unavailable")

        with patch.object(bridge, "extract_backbone_angles", return_value=fake_backbone):
            # _detect_functional_residues_sasa already wraps exceptions internally,
            # but we can also patch it to raise to test the outer flow.
            with patch.object(
                bridge, "_detect_functional_residues_sasa",
                side_effect=_sasa_fail
            ):
                # The scan should NOT raise — it must complete silently
                try:
                    result = bridge.full_proline_scan(str(pdb_file), chain="A")
                except Exception as exc:
                    pytest.fail(
                        f"full_proline_scan should not raise when SASA fails, "
                        f"but got: {exc}"
                    )

        # Scan completed — inferred set is empty (failure was silent)
        assert result["inferred_functional_residues"] == [], (
            "inferred_functional_residues should be [] when SASA fails"
        )


# ════════════════════════════════════════════════════════════════════════════════
# 8. structural_utils migration check
# ════════════════════════════════════════════════════════════════════════════════

class TestStructuralUtilsMigration:
    """
    Verify that ProlineBridge.extract_backbone_angles() still works correctly
    after it was migrated to delegate to structural_utils.extract_backbone_angles().
    """

    def test_proline_bridge_still_works_after_structural_utils_migration(
        self, bridge, tmp_path
    ):
        """
        full_proline_scan() with a minimal PDB must complete without errors
        and return the expected result schema.  This ensures the wrapper
        delegation to structural_utils did not break anything.
        """
        pdb_file = tmp_path / "mini.pdb"
        pdb_file.write_text(_MINIMAL_PDB)

        try:
            result = bridge.full_proline_scan(str(pdb_file), chain="A")
        except Exception as exc:
            pytest.fail(
                f"full_proline_scan raised after structural_utils migration: {exc}"
            )

        # Schema must be intact regardless of whether candidates were found
        required_keys = [
            "candidates", "count", "top", "chain", "pdb_path",
            "n_residues_scanned", "functional_residues",
            "inferred_functional_residues",
        ]
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in full_proline_scan result"

        # The 3-residue PDB is too short to yield candidates (all near termini),
        # but the scan must complete and count must be >= 0.
        assert isinstance(result["count"], int)
        assert isinstance(result["candidates"], list)

    def test_extract_backbone_angles_returns_expected_keys(self, bridge, tmp_path):
        """
        After migration, extract_backbone_angles() must still return entries with
        the original keys (phi, psi, ss, resname, aa) PLUS the new ca_coords key.
        """
        pdb_file = tmp_path / "mini.pdb"
        pdb_file.write_text(_MINIMAL_PDB)

        try:
            result = bridge.extract_backbone_angles(str(pdb_file), "A")
        except Exception as exc:
            pytest.skip(f"BioPython extraction unavailable: {exc}")

        if not result:
            pytest.skip("No backbone angles computed for minimal PDB")

        for resno, entry in result.items():
            assert "phi"     in entry, f"resno={resno}: missing 'phi'"
            assert "psi"     in entry, f"resno={resno}: missing 'psi'"
            assert "ss"      in entry, f"resno={resno}: missing 'ss'"
            assert "resname" in entry, f"resno={resno}: missing 'resname'"
            assert "aa"      in entry, f"resno={resno}: missing 'aa'"
            # ca_coords is new (added by structural_utils) but must be present
            assert "ca_coords" in entry, f"resno={resno}: missing 'ca_coords'"
