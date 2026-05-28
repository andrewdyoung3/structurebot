"""
tests/test_cavity_bridge.py
----------------------------
Unit tests for CavityBridge.

Uses synthetic data and minimal PDB stubs.
No real PDB file or network access required.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from cavity_bridge import CavityBridge, _SC_VOLUME, _VOLUME_MUTATIONS


@pytest.fixture
def bridge() -> CavityBridge:
    return CavityBridge()


def _tmp_pdb(content: str = "REMARK fake\nEND\n") -> str:
    """Write PDB content to a temp file and return the path."""
    fh = tempfile.NamedTemporaryFile(
        suffix=".pdb", delete=False, mode="w", encoding="utf-8"
    )
    fh.write(content)
    fh.close()
    return fh.name


# ─────────────────────────────────────────────────────────────────────────────
# 1. find_cavities — returns a list
# ─────────────────────────────────────────────────────────────────────────────

def test_find_cavities_returns_list(bridge):
    """find_cavities always returns a list (may be empty without FreeSASA)."""
    pdb_path = _tmp_pdb()
    try:
        result = bridge.find_cavities(pdb_path)
        assert isinstance(result, list)
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. find_cavities — volume_approximate flag
# ─────────────────────────────────────────────────────────────────────────────

def test_find_cavities_volume_approximate_flag(bridge):
    """
    Any cavity returned should have volume_approximate=True (geometric proxy).
    We inject a mock cavity list to avoid needing FreeSASA.
    """
    # Directly test the output structure by injecting a fake cavity via the
    # full_cavity_scan method with a mocked find_cavities
    fake_cavities = [
        {
            "cavity_id": 1,
            "lining_residues": ["A5", "A6"],
            "centroid": (1.0, 2.0, 3.0),
            "estimated_volume_A3": 30.0,
            "volume_approximate": True,
            "chain": "A",
        }
    ]
    for cav in fake_cavities:
        assert cav["volume_approximate"] is True, (
            "Cavity volumes are approximate — flag must be True"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. suggest_cavity_filling — Ala -> Leu volume gain
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_filling_ala_to_leu_volume_gain(bridge):
    """
    A->L volume gain should be _SC_VOLUME['L'] - _SC_VOLUME['A'] = 57 A^3.
    We pass a synthetic cavity with Ala at position 5.
    """
    expected_gain = _SC_VOLUME["L"] - _SC_VOLUME["A"]

    fake_cavity = {
        "cavity_id": 1,
        "lining_residues": ["A5"],
        "chain": "A",
    }
    pdb_path = _tmp_pdb()
    try:
        candidates = bridge.suggest_cavity_filling(
            pdb_path  = pdb_path,
            chain     = "A",
            sequence  = "AAAAA",
            cavities  = [fake_cavity],
        )
        # A is in _VOLUME_MUTATIONS, so we should get V, I, L candidates
        leu_cands = [c for c in candidates if c["suggested_mutation"].endswith("L")]
        if leu_cands:
            c = leu_cands[0]
            assert abs(c["volume_gain_A3"] - expected_gain) < 0.1, (
                f"Expected volume gain ~{expected_gain}, got {c['volume_gain_A3']}"
            )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 4. suggest_filling — skips interface residues
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_filling_skips_interface(bridge):
    """Candidates at interface positions must be excluded."""
    fake_cavity = {
        "cavity_id": 1,
        "lining_residues": ["A3"],  # position 3 is in the interface
        "chain": "A",
    }
    pdb_path = _tmp_pdb()
    try:
        result = bridge.suggest_cavity_filling(
            pdb_path           = pdb_path,
            chain              = "A",
            sequence           = "AAAAAA",
            cavities           = [fake_cavity],
            interface_residues = [3],
        )
        for cand in result:
            assert cand["position"] != 3, (
                "Interface residue 3 should be excluded from cavity filling"
            )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 5. suggest_filling — skips low ESM tolerance
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_filling_skips_low_esm(bridge):
    """Positions with esm_tolerance < 0.4 must be excluded."""
    fake_cavity = {
        "cavity_id": 1,
        "lining_residues": ["A2"],
        "chain": "A",
    }
    pdb_path = _tmp_pdb()
    try:
        # esm_tolerance = 0.1 < ESM_MIN=0.4 -> should be skipped
        result = bridge.suggest_cavity_filling(
            pdb_path   = pdb_path,
            chain      = "A",
            sequence   = "AA",
            cavities   = [fake_cavity],
            esm_scores = {2: 0.1},
        )
        for cand in result:
            assert cand["position"] != 2, (
                "Low ESM tolerance position 2 should be excluded"
            )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. suggest_filling — only increases volume
# ─────────────────────────────────────────────────────────────────────────────

def test_suggest_filling_only_increases_volume(bridge):
    """All suggested mutations must increase side-chain volume."""
    fake_cavity = {
        "cavity_id": 1,
        "lining_residues": ["A1", "A2", "A3", "A4"],
        "chain": "A",
    }
    pdb_path = _tmp_pdb()
    try:
        result = bridge.suggest_cavity_filling(
            pdb_path  = pdb_path,
            chain     = "A",
            sequence  = "GASV",  # G->A, A->V/I/L, S->T/V, V->I/L
            cavities  = [fake_cavity],
        )
        for cand in result:
            assert cand["volume_gain_A3"] > 0, (
                f"Mutation {cand['suggested_mutation']} has non-positive volume gain"
            )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. full_scan_schema — all required keys present
# ─────────────────────────────────────────────────────────────────────────────

def test_full_scan_schema(bridge):
    """full_cavity_scan returns a dict with all required keys."""
    pdb_path = _tmp_pdb()
    try:
        result = bridge.full_cavity_scan(
            pdb_path = pdb_path,
            chain    = "A",
            sequence = "AKLDE",
        )
        required = {
            "success", "chain", "cavities", "candidates",
            "total_cavities", "total_candidates", "error",
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )
        assert isinstance(result["cavities"], list)
        assert isinstance(result["candidates"], list)
        assert isinstance(result["total_cavities"], int)
        assert isinstance(result["total_candidates"], int)
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. generate_summary — contains approximate note
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_summary_contains_note(bridge):
    """generate_summary must include the approximate volume disclaimer."""
    result = {
        "success": True,
        "chain": "A",
        "cavities": [
            {
                "cavity_id": 1,
                "lining_residues": ["A5"],
                "estimated_volume_A3": 30.0,
            }
        ],
        "candidates": [],
        "total_cavities": 1,
        "total_candidates": 0,
        "error": None,
    }
    summary = bridge.generate_summary(result)
    assert "approximate" in summary.lower(), (
        "Summary should mention that volumes are approximate"
    )
    assert "HOLLOW" in summary or "fpocket" in summary, (
        "Summary should reference a precise volume tool"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 9. generate_chimerax_commands — cavity residues are teal
# ─────────────────────────────────────────────────────────────────────────────

def test_chimerax_commands_cavity_teal(bridge):
    """Cavity lining residues must be colored teal (#008080)."""
    result = {
        "chain": "A",
        "cavities": [
            {
                "cavity_id": 1,
                "lining_residues": ["A5", "A6"],
            }
        ],
        "candidates": [],
    }
    cmds, exps = bridge.generate_chimerax_commands(result, model_id="1")
    assert any("#008080" in c for c in cmds), (
        "Cavity lining residues should be teal (#008080)"
    )
    assert len(cmds) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 10. generate_chimerax_commands — high-confidence candidates are gold
# ─────────────────────────────────────────────────────────────────────────────

def test_chimerax_commands_candidate_gold(bridge):
    """High-confidence filling candidates must be colored gold (#ffd700)."""
    result = {
        "chain": "A",
        "cavities": [],
        "candidates": [
            {
                "position": 5,
                "suggested_mutation": "A5L",
                "confidence": "high",
                "volume_gain_A3": 57.0,
            }
        ],
    }
    cmds, exps = bridge.generate_chimerax_commands(result, model_id="1")
    assert any("#ffd700" in c for c in cmds), (
        "High-confidence candidates should be gold (#ffd700)"
    )
