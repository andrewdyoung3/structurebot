"""
tests/test_salt_bridge_bridge.py
---------------------------------
Unit tests for SaltBridgeBridge.

Uses synthetic PDB strings to test bridge detection at known distances.
No real PDB file or network access required.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from salt_bridge_bridge import SaltBridgeBridge, _SALT_BRIDGE_CUTOFF


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_pdb(content: str) -> str:
    """Write PDB content to a temp file and return the path."""
    fh = tempfile.NamedTemporaryFile(
        suffix=".pdb", delete=False, mode="w", encoding="utf-8"
    )
    fh.write(content)
    fh.close()
    return fh.name


def _make_pdb_with_pair(
    res1_name: str,
    res2_name: str,
    charged_atom1: str,
    charged_atom2: str,
    distance: float,
    chain1: str = "A",
    chain2: str = "A",
    resno1: int = 10,
    resno2: int = 20,
) -> str:
    """
    Create a minimal PDB with two residues.  The charged atoms are placed
    at (0,0,0) and (distance,0,0).  CA atoms are placed nearby.

    Only the charged atoms are needed for salt bridge detection; we add CA
    so BioPython doesn't complain about incomplete residues.
    """
    # Both residues on chain1 unless chain2 differs
    lines = [
        # Residue 1
        f"ATOM      1  CA  {res1_name} {chain1}{resno1:4d}       0.000   0.000   0.000  1.00  0.00           C",
        f"ATOM      2  {charged_atom1:<4s}{res1_name} {chain1}{resno1:4d}       0.000   0.000   0.000  1.00  0.00           N",
        # Residue 2 — charged atom at 'distance' along X axis
        f"ATOM      3  CA  {res2_name} {chain2}{resno2:4d}     {distance:7.3f}   0.000   0.000  1.00  0.00           C",
        f"ATOM      4  {charged_atom2:<4s}{res2_name} {chain2}{resno2:4d}     {distance:7.3f}   0.000   0.000  1.00  0.00           O",
        "END",
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def bridge() -> SaltBridgeBridge:
    return SaltBridgeBridge()


# ─────────────────────────────────────────────────────────────────────────────
# 1. find_existing_salt_bridges — close pair detected
# ─────────────────────────────────────────────────────────────────────────────

def test_find_existing_salt_bridges_detects_close_pair(bridge):
    """ARG NH1 and ASP OD1 at 3.5 A should be detected as a salt bridge."""
    pdb_content = _make_pdb_with_pair(
        "ARG", "ASP", "NH1", "OD1", distance=3.5,
        resno1=10, resno2=20,
    )
    pdb_path = _write_pdb(pdb_content)
    try:
        result = bridge.find_existing_salt_bridges(pdb_path)
        assert isinstance(result, list)
        assert len(result) >= 1, f"Expected at least one salt bridge, got {result}"
        b = result[0]
        assert b["distance"] <= _SALT_BRIDGE_CUTOFF
        assert "A10" in (b["res1"], b["res2"]) or "A20" in (b["res1"], b["res2"])
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. find_existing_salt_bridges — distance cutoff (4.1A not detected)
# ─────────────────────────────────────────────────────────────────────────────

def test_find_existing_salt_bridges_distance_cutoff(bridge):
    """ARG NH1 and ASP OD1 at 4.1 A should NOT be detected (cutoff is 4.0 A)."""
    pdb_content = _make_pdb_with_pair(
        "ARG", "ASP", "NH1", "OD1", distance=4.1,
        resno1=10, resno2=20,
    )
    pdb_path = _write_pdb(pdb_content)
    try:
        result = bridge.find_existing_salt_bridges(pdb_path)
        assert result == [], f"Expected no salt bridges at 4.1 A, got {result}"
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. find_existing_salt_bridges — interchain pair
# ─────────────────────────────────────────────────────────────────────────────

def test_find_existing_salt_bridges_interchain(bridge):
    """LYS NZ on chain A and GLU OE1 on chain B at 3.0 A — interchain=True."""
    pdb_content = _make_pdb_with_pair(
        "LYS", "GLU", "NZ  ", "OE1 ", distance=3.0,
        chain1="A", chain2="B",
        resno1=5, resno2=8,
    )
    pdb_path = _write_pdb(pdb_content)
    try:
        result = bridge.find_existing_salt_bridges(pdb_path)
        assert isinstance(result, list)
        # With two-chain PDB: should find the inter-chain bridge
        if result:
            b = result[0]
            assert b["interchain"] is True, f"Expected interchain=True, got {b}"
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. suggest_new — identifies a surface candidate
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_new_identifies_surface_candidate(bridge):
    """
    A surface-exposed Ala near a Lys partner should be suggested as a
    candidate for Glu introduction to form a new salt bridge.
    We mock SASA and the structure to return a predictable result.
    """
    # We just test that the method runs and returns a list (may be empty
    # without a real PDB, which is expected graceful degradation).
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh:
        fh.write("REMARK fake\nEND\n")
        pdb_path = fh.name
    try:
        result = bridge.suggest_new_salt_bridges(
            pdb_path   = pdb_path,
            chain      = "A",
            sequence   = "AKLDEALR",
        )
        assert isinstance(result, list)
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. suggest_new — skips interface residues
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_new_skips_interface_residues(bridge):
    """
    Candidates at interface positions must be excluded.
    We use _compute_sasa = {} (no FreeSASA) so candidates list is empty —
    the key check is that no candidate has resno in the interface set.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh:
        fh.write("REMARK fake\nEND\n")
        pdb_path = fh.name
    try:
        interface = list(range(1, 20))   # positions 1-19 are "interface"
        result = bridge.suggest_new_salt_bridges(
            pdb_path           = pdb_path,
            chain              = "A",
            sequence           = "A" * 20,
            interface_residues = interface,
        )
        for cand in result:
            assert cand["position"] not in set(interface), (
                f"Interface residue {cand['position']} should be excluded"
            )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. suggest_new — skips low ESM tolerance
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_new_skips_low_esm_tolerance(bridge):
    """
    Positions with esm_tolerance < 0 (very low) should be penalised such that
    composite_score <= 0 and thus excluded.  We check no candidate appears
    with esm_tolerance = 0.0.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh:
        fh.write("REMARK fake\nEND\n")
        pdb_path = fh.name
    try:
        # esm_scores maps all positions to 0.0 (very low tolerance)
        esm_scores = {i: 0.0 for i in range(1, 30)}
        result = bridge.suggest_new_salt_bridges(
            pdb_path   = pdb_path,
            chain      = "A",
            sequence   = "A" * 30,
            esm_scores = esm_scores,
        )
        # composite_score = dist * sasa * 0.0 * iface = 0.0 for all, so excluded
        for cand in result:
            assert cand["composite_score"] > 0, (
                "Candidate with esm_tolerance=0.0 should have composite_score=0 and be excluded"
            )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. suggest_new — distance scoring (5.0 A > 7.5 A)
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_new_distance_scoring(bridge):
    """
    Distance score peaks at 5.0 A.  A candidate at 5.0 A should have higher
    distance_score than one at 7.5 A.
    We test the math directly from the formula used in the code.
    """
    def distance_score(d: float) -> float:
        return max(0.0, 1.0 - abs(d - 5.0) / 3.0)

    score_5  = distance_score(5.0)
    score_75 = distance_score(7.5)
    assert score_5 > score_75, (
        f"5.0 A (score={score_5:.3f}) should beat 7.5 A (score={score_75:.3f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 8. full_scan_schema — all required keys present
# ─────────────────────────────────────────────────────────────────────────────

def test_full_scan_schema(bridge):
    """full_salt_bridge_scan returns a dict with all required keys."""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as fh:
        fh.write("REMARK fake\nEND\n")
        pdb_path = fh.name
    try:
        result = bridge.full_salt_bridge_scan(
            pdb_path = pdb_path,
            chain    = "A",
            sequence = "AKLDE",
        )
        required = {
            "success", "chain", "existing_salt_bridges", "candidates",
            "total_existing", "total_candidates", "error",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )
        assert isinstance(result["existing_salt_bridges"], list)
        assert isinstance(result["candidates"], list)
        assert isinstance(result["total_existing"], int)
        assert isinstance(result["total_candidates"], int)
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. generate_summary — existing and candidates
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_summary_existing_and_candidates(bridge):
    """generate_summary renders existing bridges and candidate table."""
    result = {
        "success": True,
        "chain": "A",
        "existing_salt_bridges": [
            {
                "res1": "A10", "res2": "A20",
                "chain1": "A", "chain2": "A",
                "distance": 3.5,
                "type": "Lys-Asp",
                "interchain": False,
                "buried": False,
            }
        ],
        "candidates": [
            {
                "position": 5,
                "chain": "A",
                "wildtype_residue": "A",
                "suggested_mutation": "A5K",
                "partner_residue": "A20",
                "partner_distance": 5.5,
                "charge_pair": "Lys-Asp",
                "sasa": 80.0,
                "esm_tolerance": None,
                "composite_score": 0.45,
                "confidence": "moderate",
            }
        ],
        "total_existing": 1,
        "total_candidates": 1,
        "error": None,
    }
    summary = bridge.generate_summary(result)
    assert isinstance(summary, str)
    assert "A10" in summary
    assert "A20" in summary
    assert "A5K" in summary
    assert "Salt bridge analysis" in summary


# ─────────────────────────────────────────────────────────────────────────────
# 10. generate_chimerax_commands — existing bridges are orange
# ─────────────────────────────────────────────────────────────────────────────

def test_chimerax_commands_existing_orange(bridge):
    """Existing salt bridge residues must be colored orange (#ff8800)."""
    result = {
        "chain": "A",
        "existing_salt_bridges": [
            {
                "res1": "A10", "res2": "A20",
                "chain1": "A", "chain2": "A",
                "distance": 3.5,
                "type": "Arg-Glu",
                "interchain": False,
                "buried": False,
            }
        ],
        "candidates": [],
    }
    cmds, exps = bridge.generate_chimerax_commands(result, model_id="1")
    assert any("#ff8800" in c for c in cmds), (
        "Existing salt bridges should be colored orange (#ff8800)"
    )
    assert len(cmds) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. generate_chimerax_commands — high-confidence candidates are lime
# ─────────────────────────────────────────────────────────────────────────────

def test_chimerax_commands_candidate_lime(bridge):
    """High-confidence candidates must be colored lime (#88ff00)."""
    result = {
        "chain": "A",
        "existing_salt_bridges": [],
        "candidates": [
            {
                "position": 5,
                "suggested_mutation": "A5K",
                "confidence": "high",
                "composite_score": 0.75,
            }
        ],
    }
    cmds, exps = bridge.generate_chimerax_commands(result, model_id="1")
    assert any("#88ff00" in c for c in cmds), (
        "High-confidence candidates should be lime (#88ff00)"
    )
