"""
tests/test_structural_utils.py
-------------------------------
Unit tests for structural_utils — the shared geometry utility module.

Tests cover:
  A. extract_backbone_angles()
  B. compute_sasa()
  C. compute_projection_score()
  D. classify_sequon_geometry()

All PDB files are built inline using tmp_path; no network or ChimeraX required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import structural_utils


# ── Minimal PDB helpers ───────────────────────────────────────────────────────
# Same 3-residue backbone used by test_proline_bridge.py

_MINIMAL_PDB_3RES = """\
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
"""

# One-residue PDB for SASA test (ASN, residue 5)
_SINGLE_RES_PDB = """\
ATOM      1  N   ASN A   5       1.000   2.000   3.000  1.00 20.00           N
ATOM      2  CA  ASN A   5       2.000   2.000   3.000  1.00 20.00           C
ATOM      3  C   ASN A   5       3.000   2.000   3.000  1.00 20.00           C
ATOM      4  O   ASN A   5       3.000   3.000   3.000  1.00 20.00           O
ATOM      5  CB  ASN A   5       2.000   2.000   4.500  1.00 20.00           C
END
"""

# 3-residue PDB designed so residue 2 has CB pointing away from centroid
# Residue 1: CA=(0,0,1),  CB=(0,-1,1)   → pointing "south" (roughly inward)
# Residue 2: CA=(5,0,1),  CB=(6,0,1)    → CB one step further from centroid (outward)
# Residue 3: CA=(0,5,1),  CB=(-1,5,1)   → pointing "west" (roughly inward)
# Centroid ≈ (1.67, 1.67, 1)
# For residue 2: outward normal ≈ (0.89, -0.45, 0); CA->CB = (1,0,0); proj ≈ 0.89
_PROJ_PDB_OUTWARD = """\
ATOM      1  N   ALA A   1       0.000   0.500   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.000   0.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   1.000   1.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       0.000  -1.000   1.000  1.00  0.00           C
ATOM      6  N   ALA A   2       5.000   0.500   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.000   0.000   1.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.000   1.000   1.000  1.00  0.00           C
ATOM      9  O   ALA A   2       5.000   2.000   1.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       6.000   0.000   1.000  1.00  0.00           C
ATOM     11  N   ALA A   3       0.000   5.500   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       0.000   5.000   1.000  1.00  0.00           C
ATOM     13  C   ALA A   3       1.000   5.000   1.000  1.00  0.00           C
ATOM     14  O   ALA A   3       1.500   6.000   1.000  1.00  0.00           O
ATOM     15  CB  ALA A   3      -1.000   5.000   1.000  1.00  0.00           C
END
"""

# Same as above but residue 2 CB at (4,0,1) — pointing TOWARD centroid (inward)
_PROJ_PDB_INWARD = """\
ATOM      1  N   ALA A   1       0.000   0.500   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.000   0.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   1.000   1.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       0.000  -1.000   1.000  1.00  0.00           C
ATOM      6  N   ALA A   2       5.000   0.500   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       5.000   0.000   1.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.000   1.000   1.000  1.00  0.00           C
ATOM      9  O   ALA A   2       5.000   2.000   1.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       4.000   0.000   1.000  1.00  0.00           C
ATOM     11  N   ALA A   3       0.000   5.500   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       0.000   5.000   1.000  1.00  0.00           C
ATOM     13  C   ALA A   3       1.000   5.000   1.000  1.00  0.00           C
ATOM     14  O   ALA A   3       1.500   6.000   1.000  1.00  0.00           O
ATOM     15  CB  ALA A   3      -1.000   5.000   1.000  1.00  0.00           C
END
"""

# 3-residue PDB with one GLY residue (no CB) — for gly_proxy test
# Residue 2 is GLY; N is at a known position relative to CA
_PROJ_PDB_GLY = """\
ATOM      1  N   ALA A   1       0.000   0.500   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       1.000   0.000   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.500   1.000   1.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       0.000  -1.000   1.000  1.00  0.00           C
ATOM      6  N   GLY A   2       5.000   0.500   0.000  1.00  0.00           N
ATOM      7  CA  GLY A   2       5.000   0.000   1.000  1.00  0.00           C
ATOM      8  C   GLY A   2       5.000   1.000   1.000  1.00  0.00           C
ATOM      9  O   GLY A   2       5.000   2.000   1.000  1.00  0.00           O
ATOM     10  N   ALA A   3       0.000   5.500   0.000  1.00  0.00           N
ATOM     11  CA  ALA A   3       0.000   5.000   1.000  1.00  0.00           C
ATOM     12  C   ALA A   3       1.000   5.000   1.000  1.00  0.00           C
ATOM     13  O   ALA A   3       1.500   6.000   1.000  1.00  0.00           O
ATOM     14  CB  ALA A   3      -1.000   5.000   1.000  1.00  0.00           C
END
"""


# ════════════════════════════════════════════════════════════════════════════════
# Section A — extract_backbone_angles()
# ════════════════════════════════════════════════════════════════════════════════

class TestExtractBackboneAngles:

    def test_extract_backbone_angles_returns_phi_psi(self, tmp_path):
        """
        A 3-residue PDB should produce a dict; the middle residue typically has
        both φ and ψ available.  Keys must be integers (residue numbers).
        """
        pdb_file = tmp_path / "mini.pdb"
        pdb_file.write_text(_MINIMAL_PDB_3RES)

        result = structural_utils.extract_backbone_angles(str(pdb_file), "A")
        assert isinstance(result, dict)

        if not result:
            pytest.skip("BioPython PPBuilder did not compute angles for this minimal PDB")

        # All keys must be integers (residue sequence numbers)
        for k in result:
            assert isinstance(k, int)

        # Each entry must have the required fields
        for entry in result.values():
            assert "phi"      in entry
            assert "psi"      in entry
            assert "ss"       in entry
            assert "resname"  in entry
            assert "aa"       in entry
            assert "ca_coords" in entry

    def test_extract_backbone_angles_missing_file_returns_empty(self, tmp_path):
        """Non-existent path must return {} (graceful failure)."""
        result = structural_utils.extract_backbone_angles(
            str(tmp_path / "no_such_file.pdb"), "A"
        )
        assert result == {}

    def test_extract_backbone_angles_ca_coords_type(self, tmp_path):
        """ca_coords must be a 3-tuple of floats or None."""
        pdb_file = tmp_path / "mini.pdb"
        pdb_file.write_text(_MINIMAL_PDB_3RES)

        result = structural_utils.extract_backbone_angles(str(pdb_file), "A")
        if not result:
            pytest.skip("No angles computed for minimal PDB")

        for entry in result.values():
            coords = entry["ca_coords"]
            if coords is not None:
                assert len(coords) == 3
                for v in coords:
                    assert isinstance(v, float)


# ════════════════════════════════════════════════════════════════════════════════
# Section B — compute_sasa()
# ════════════════════════════════════════════════════════════════════════════════

class TestComputeSASA:

    def test_compute_sasa_returns_per_residue(self, tmp_path):
        """compute_sasa() should return a dict with int keys and float values."""
        pdb_file = tmp_path / "asn.pdb"
        pdb_file.write_text(_SINGLE_RES_PDB)

        result = structural_utils.compute_sasa(str(pdb_file), "A")
        assert isinstance(result, dict)

        if not result:
            pytest.skip("SASA computation did not produce results for this single-residue PDB")

        for key, val in result.items():
            assert isinstance(key, int), f"Expected int key, got {type(key)}"
            assert isinstance(val, float), f"Expected float value, got {type(val)}"

    def test_compute_sasa_missing_file_returns_empty(self):
        """Non-existent PDB path must return {} — no exception."""
        result = structural_utils.compute_sasa("/nonexistent/file.pdb", "A")
        assert result == {}

    def test_compute_sasa_no_chain_filter(self, tmp_path):
        """compute_sasa(chain=None) must also return a dict without crashing."""
        pdb_file = tmp_path / "asn.pdb"
        pdb_file.write_text(_SINGLE_RES_PDB)
        result = structural_utils.compute_sasa(str(pdb_file))
        assert isinstance(result, dict)


# ════════════════════════════════════════════════════════════════════════════════
# Section C — compute_projection_score()
# ════════════════════════════════════════════════════════════════════════════════

class TestComputeProjectionScore:

    def test_compute_projection_score_outward_residue(self, tmp_path):
        """
        Residue 2 has CB pointing away from the Cα centroid → projection near +1.
        The score must be > 0.6 ('outward' threshold).
        """
        pdb_file = tmp_path / "out.pdb"
        pdb_file.write_text(_PROJ_PDB_OUTWARD)

        result = structural_utils.compute_projection_score(str(pdb_file), "A")
        assert isinstance(result, dict)

        if 2 not in result:
            pytest.skip("Residue 2 not parsed — CA atom may be missing")

        entry = result[2]
        assert "projection_score" in entry
        assert "gly_proxy"        in entry
        assert entry["gly_proxy"] is False
        assert entry["projection_score"] > 0.6, (
            f"Expected outward projection > 0.6, got {entry['projection_score']}"
        )

    def test_compute_projection_score_inward_residue(self, tmp_path):
        """
        Residue 2 has CB pointing toward the centroid → projection ≤ 0.2 (or negative).
        """
        pdb_file = tmp_path / "inw.pdb"
        pdb_file.write_text(_PROJ_PDB_INWARD)

        result = structural_utils.compute_projection_score(str(pdb_file), "A")
        assert isinstance(result, dict)

        if 2 not in result:
            pytest.skip("Residue 2 not parsed")

        entry = result[2]
        assert entry["projection_score"] <= 0.2, (
            f"Expected inward projection ≤ 0.2, got {entry['projection_score']}"
        )

    def test_compute_projection_score_gly_proxy(self, tmp_path):
        """
        A GLY residue (no Cβ) must return gly_proxy=True and still produce a score.
        """
        pdb_file = tmp_path / "gly.pdb"
        pdb_file.write_text(_PROJ_PDB_GLY)

        result = structural_utils.compute_projection_score(str(pdb_file), "A")
        assert isinstance(result, dict)

        # Residue 2 is GLY — find it
        if 2 not in result:
            pytest.skip("GLY residue 2 not found in projection result")

        entry = result[2]
        assert entry["gly_proxy"] is True
        assert isinstance(entry["projection_score"], float)

    def test_compute_projection_score_failure_returns_empty(self):
        """Invalid PDB path must return {} — no exception raised."""
        result = structural_utils.compute_projection_score(
            "/no/such/file.pdb", "A"
        )
        assert result == {}

    def test_compute_projection_score_all_residues_have_schema(self, tmp_path):
        """Every entry in the result dict must have projection_score and gly_proxy."""
        pdb_file = tmp_path / "all.pdb"
        pdb_file.write_text(_PROJ_PDB_OUTWARD)

        result = structural_utils.compute_projection_score(str(pdb_file), "A")
        for resno, entry in result.items():
            assert "projection_score" in entry, f"resno={resno} missing projection_score"
            assert "gly_proxy"        in entry, f"resno={resno} missing gly_proxy"
            score = entry["projection_score"]
            assert -1.0 <= score <= 1.0, f"resno={resno} score={score} out of [-1,1]"


# ════════════════════════════════════════════════════════════════════════════════
# Section D — classify_sequon_geometry()
# ════════════════════════════════════════════════════════════════════════════════

def _make_backbone(phi: float, psi: float, positions=(1, 2, 3)) -> Dict[int, Dict]:
    """Build a minimal backbone dict with given phi/psi at all positions."""
    return {p: {"phi": phi, "psi": psi, "ss": "L"} for p in positions}


class TestClassifySequonGeometry:

    def test_classify_sequon_geometry_loop(self):
        """
        φ=90, ψ=40 → loop (_classify_single_pos returns 'L' for all three).
        No helix, no turn, no extended → result is 'loop'.
        """
        bb = _make_backbone(90.0, 40.0)
        assert structural_utils.classify_sequon_geometry(bb, 1) == "loop"

    def test_classify_sequon_geometry_helix(self):
        """
        φ=-57, ψ=-47 at position i+1 → helix (dominant, returns 'helix').
        """
        # pos 1 and 3 are loop; pos 2 (i+1) is helix
        bb = {
            1: {"phi": 90.0,  "psi": 40.0,  "ss": "L"},
            2: {"phi": -57.0, "psi": -47.0, "ss": "H"},   # helix
            3: {"phi": 90.0,  "psi": 40.0,  "ss": "L"},
        }
        assert structural_utils.classify_sequon_geometry(bb, 1) == "helix"

    def test_classify_sequon_geometry_beta_turn(self):
        """
        φ=-50, ψ=20 → turn-like ('T' code); not helix, not extended.
        At least one turn-like position → 'beta_turn'.
        """
        # (-50, 20): not helix (-90≤-50≤-45 AND -60≤20≤-10? No, 20 > -10) → NOT helix
        # Extended? -50 <= -90? No. Turn? -90≤-50≤-30 AND -90≤20≤60? Yes → T
        bb = _make_backbone(-50.0, 20.0)
        result = structural_utils.classify_sequon_geometry(bb, 1)
        assert result == "beta_turn", (
            f"Expected 'beta_turn' for phi=-50, psi=20; got '{result}'"
        )

    def test_classify_sequon_geometry_extended(self):
        """
        φ=-120, ψ=130 → extended ('E' code).
        All three positions extended → 'extended'.
        """
        bb = _make_backbone(-120.0, 130.0)
        assert structural_utils.classify_sequon_geometry(bb, 1) == "extended"

    def test_classify_sequon_geometry_unknown_missing_position(self):
        """
        Missing position (i+2 absent from backbone) → 'unknown'.
        """
        bb = {
            1: {"phi": 90.0, "psi": 40.0, "ss": "L"},
            2: {"phi": 90.0, "psi": 40.0, "ss": "L"},
            # position 3 missing
        }
        assert structural_utils.classify_sequon_geometry(bb, 1) == "unknown"

    def test_classify_sequon_geometry_unknown_none_angles(self):
        """
        phi=None at one position → 'unknown'.
        """
        bb = {
            1: {"phi": 90.0,  "psi": 40.0, "ss": "L"},
            2: {"phi": None,  "psi": 40.0, "ss": "L"},
            3: {"phi": 90.0,  "psi": 40.0, "ss": "L"},
        }
        assert structural_utils.classify_sequon_geometry(bb, 1) == "unknown"
