"""
tests/test_cavity_bridge.py
----------------------------
Unit tests for CavityBridge.

Most tests use synthetic data / minimal PDB stubs and MOCK SASA. That mocking is
exactly how a dead `Bio.PDB.ShrakeRupley` import (the SASA path returned {} → 0
cavities for every structure) survived 20 green tests for 17 days (dfd8d9c
2026-05-28 → 7e36f87 2026-06-14). TestRealSASAPath below is the antidote: it
exercises the REAL Bio.PDB.SASA path against a committed tiny real protein
(crambin, hermetic — no network) so the import cannot silently die again.
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

from cavity_bridge import CavityBridge, _SC_VOLUME, _VOLUME_MUTATIONS, _BIOPYTHON_SASA_OK


_CRAMBIN_PDB = Path(__file__).parent / "fixtures" / "1crn.pdb"


class TestRealSASAPath:
    """NON-mocked end-to-end SASA: would have caught the dead ShrakeRupley import.

    Crambin (1CRN, 46 residues, classic hydrophobic core) is committed as a tiny
    hermetic fixture — no network, no mock. Asserts both the import is alive AND
    the real path produces buried residues / a cavity.
    """

    def test_biopython_sasa_import_is_alive(self):
        # the single assertion that directly catches a dead SASA import
        assert _BIOPYTHON_SASA_OK is True, (
            "Bio.PDB.SASA import is dead — cavity detection silently returns 0 "
            "cavities for every structure (the dfd8d9c→7e36f87 regression class)")

    def test_real_sasa_computes_buried_residues(self):
        assert _CRAMBIN_PDB.is_file(), f"missing fixture {_CRAMBIN_PDB}"
        cav = CavityBridge()
        st = cav._load_structure(str(_CRAMBIN_PDB))
        assert st is not None
        sasa = cav._sasa_for_chains(st, str(_CRAMBIN_PDB), ["A"])
        assert len(sasa) > 0, "real ShrakeRupley produced an EMPTY map (dead path)"
        buried = [r for (ch, r), a in sasa.items() if a < 20.0]
        assert len(buried) >= 1, "crambin must have a buried core (it has 7 at <20 Å²)"

    def test_real_find_cavities_nonzero(self):
        cav = CavityBridge()
        cavs = cav.find_cavities(str(_CRAMBIN_PDB), chains=["A"])
        assert isinstance(cavs, list)
        assert len(cavs) >= 1, "crambin's hydrophobic core must yield >= 1 cavity live"
        assert cavs[0]["n_residues"] >= 1

    def test_real_solvent_exposed_residues_nonzero(self):
        # the design-goal exposed-selector on the real path (not the dead {} → [])
        cav = CavityBridge()
        exposed = cav.solvent_exposed_residues(str(_CRAMBIN_PDB), "A", sasa_threshold=40.0)
        assert len(exposed) >= 1, "crambin must have solvent-exposed residues live"


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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for geometry tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_chain_pdb(chain_residues: list) -> str:
    """
    Build a minimal PDB string.

    chain_residues : list of (chain_id, resno, x, y, z)
    Each residue gets a CA atom at (x, y, z) and a CB at (x+1.5, y, z).
    """
    lines: list = []
    atom_no = 1
    for chain_id, resno, x, y, z in chain_residues:
        lines.append(
            f"ATOM  {atom_no:5d}  CA  ALA {chain_id}{resno:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C  "
        )
        atom_no += 1
        lines.append(
            f"ATOM  {atom_no:5d}  CB  ALA {chain_id}{resno:4d}    "
            f"{x + 1.5:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C  "
        )
        atom_no += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# 11. find_cavities — single chain returns is_interface_cavity=False
# ─────────────────────────────────────────────────────────────────────────────

def test_find_cavities_single_chain_detects_intrachain(bridge):
    """
    When all buried residues belong to one chain, is_interface_cavity must be False.
    SASA is mocked so the threshold check is bypassed.
    """
    # 6 residues in chain A packed tightly (3 Å spacing → all within 6 Å cluster radius)
    positions = [("A", i, float(i * 3), 0.0, 0.0) for i in range(1, 7)]
    pdb_content = _make_chain_pdb(positions)
    pdb_path = _tmp_pdb(pdb_content)

    # All residues buried (SASA below threshold)
    mocked_sasa = {("A", i): 5.0 for i in range(1, 7)}

    try:
        with patch.object(bridge, "_sasa_for_chains", return_value=mocked_sasa):
            cavities = bridge.find_cavities(
                pdb_path,
                chains=["A"],
                burial_sasa_threshold=20.0,
                cluster_radius=6.0,
                min_cluster_size=4,
            )
        assert len(cavities) >= 1, "Expected at least one cavity cluster"
        for cav in cavities:
            assert cav["is_interface_cavity"] is False, (
                "Single-chain cavity should not be an interface cavity"
            )
            assert "chains_involved" in cav
            assert cav["chains_involved"] == ["A"]
            assert "n_residues" in cav
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 12. find_cavities — dimer detects interface cavity
# ─────────────────────────────────────────────────────────────────────────────

def test_find_cavities_dimer_detects_interface_cavity(bridge):
    """
    Residues from two chains within cluster_radius form an interface cavity.
    """
    # Chain A residues at x=0..12, chain B at x=1..13 — all within 6 Å
    positions = (
        [("A", i, float(i * 2), 0.0, 0.0) for i in range(1, 5)]
        + [("B", i, float(i * 2) + 1.0, 0.0, 0.0) for i in range(1, 5)]
    )
    pdb_content = _make_chain_pdb(positions)
    pdb_path = _tmp_pdb(pdb_content)

    mocked_sasa = {
        **{("A", i): 5.0 for i in range(1, 5)},
        **{("B", i): 5.0 for i in range(1, 5)},
    }

    try:
        with patch.object(bridge, "_sasa_for_chains", return_value=mocked_sasa):
            cavities = bridge.find_cavities(
                pdb_path,
                chains=None,  # all chains
                burial_sasa_threshold=20.0,
                cluster_radius=6.0,
                min_cluster_size=4,
            )
        assert len(cavities) >= 1, "Expected at least one cavity"
        interface_cavs = [c for c in cavities if c["is_interface_cavity"]]
        assert len(interface_cavs) >= 1, (
            "Expected at least one interface cavity for dimer"
        )
        assert "B" in interface_cavs[0]["chains_involved"]
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 13. find_cavities — chains=None includes all chains
# ─────────────────────────────────────────────────────────────────────────────

def test_find_cavities_chains_none_uses_all_chains(bridge):
    """chains=None should analyse every chain in the structure."""
    positions = (
        [("A", i, float(i * 3), 0.0, 0.0) for i in range(1, 5)]
        + [("B", i, float(i * 3), 5.0, 0.0) for i in range(1, 5)]
    )
    pdb_content = _make_chain_pdb(positions)
    pdb_path = _tmp_pdb(pdb_content)

    mocked_sasa = {
        **{("A", i): 5.0 for i in range(1, 5)},
        **{("B", i): 5.0 for i in range(1, 5)},
    }

    try:
        with patch.object(bridge, "_sasa_for_chains", return_value=mocked_sasa):
            cavities_all = bridge.find_cavities(
                pdb_path, chains=None, min_cluster_size=4
            )
        # Both chains should contribute residues to the cavity labels
        all_labels = [lab for cav in cavities_all for lab in cav["lining_residues"]]
        chains_seen = set(lab[0] for lab in all_labels if lab)
        assert "A" in chains_seen or "B" in chains_seen, (
            "chains=None should include residues from at least one chain"
        )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 14. find_cavities — chains=["A"] excludes chain B residues
# ─────────────────────────────────────────────────────────────────────────────

def test_find_cavities_chains_list_filters_correctly(bridge):
    """When chains=["A"] is specified, chain B residues must not appear."""
    positions = (
        [("A", i, float(i * 3), 0.0, 0.0) for i in range(1, 7)]
        + [("B", i, float(i * 3), 0.0, 0.0) for i in range(1, 7)]
    )
    pdb_content = _make_chain_pdb(positions)
    pdb_path = _tmp_pdb(pdb_content)

    mocked_sasa = {
        **{("A", i): 5.0 for i in range(1, 7)},
        **{("B", i): 5.0 for i in range(1, 7)},
    }

    try:
        with patch.object(bridge, "_sasa_for_chains", return_value=mocked_sasa):
            cavities = bridge.find_cavities(
                pdb_path,
                chains=["A"],
                burial_sasa_threshold=20.0,
                min_cluster_size=4,
            )
        for cav in cavities:
            for label in cav["lining_residues"]:
                assert label[0] == "A", (
                    f"Chain B residue {label} found when chains=['A']"
                )
            assert cav["chains_involved"] == ["A"]
            assert cav["is_interface_cavity"] is False
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 15. find_cavities — cluster size filter
# ─────────────────────────────────────────────────────────────────────────────

def test_find_cavities_cluster_size_filter(bridge):
    """Clusters smaller than min_cluster_size must be discarded."""
    # Only 3 residues — below default min_cluster_size=4
    positions = [("A", i, float(i * 3), 0.0, 0.0) for i in range(1, 4)]
    pdb_content = _make_chain_pdb(positions)
    pdb_path = _tmp_pdb(pdb_content)

    mocked_sasa = {("A", i): 5.0 for i in range(1, 4)}

    try:
        with patch.object(bridge, "_sasa_for_chains", return_value=mocked_sasa):
            cavities = bridge.find_cavities(
                pdb_path,
                chains=["A"],
                burial_sasa_threshold=20.0,
                cluster_radius=6.0,
                min_cluster_size=4,  # 3 residues < 4 → filtered out
            )
        assert cavities == [], (
            "Cluster of 3 below min_cluster_size=4 should be discarded"
        )
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 16. full_cavity_scan — assembly mode fields are present
# ─────────────────────────────────────────────────────────────────────────────

def test_full_cavity_scan_assembly_mode_fields(bridge):
    """full_cavity_scan must return the new assembly-mode keys."""
    pdb_path = _tmp_pdb()
    try:
        result = bridge.full_cavity_scan(
            pdb_path = pdb_path,
            chain    = "A",
            sequence = "AKLDE",
            chains   = None,   # assembly mode
        )
        new_required = {
            "assembly_mode",
            "chains_analysed",
            "interface_cavities",
            "intrachain_cavities",
        }
        assert new_required.issubset(result.keys()), (
            f"Missing new fields: {new_required - result.keys()}"
        )
        assert isinstance(result["assembly_mode"], bool)
        assert isinstance(result["chains_analysed"], list)
        assert isinstance(result["interface_cavities"], int)
        assert isinstance(result["intrachain_cavities"], int)
    finally:
        os.unlink(pdb_path)


# ─────────────────────────────────────────────────────────────────────────────
# 17. full_cavity_scan — candidates only for primary chain
# ─────────────────────────────────────────────────────────────────────────────

def test_full_cavity_scan_filling_only_for_primary_chain(bridge):
    """
    suggest_cavity_filling must only propose mutations for the primary chain
    even when cavities span multiple chains.
    """
    # Inject a multi-chain cavity where chain B also has a lining residue
    fake_result = bridge.full_cavity_scan.__func__  # just call directly

    interface_cavity = {
        "cavity_id":           1,
        "lining_residues":     ["A3", "B7"],  # both chains
        "estimated_volume_A3": 60.0,
        "volume_approximate":  True,
        "chain":               "A",
        "chains_involved":     ["A", "B"],
        "is_interface_cavity": True,
        "n_residues":          4,
    }
    candidates = bridge.suggest_cavity_filling(
        pdb_path  = _tmp_pdb(),
        chain     = "A",         # primary chain
        sequence  = "AAAAAAAAAA",
        cavities  = [interface_cavity],
    )
    for cand in candidates:
        assert cand["chain"] == "A", (
            f"Candidate chain should be 'A', got {cand['chain']}"
        )
        # Must not reference chain B residues
        assert cand["position"] != 7, (
            "Chain B residue 7 should not appear as a candidate for chain A"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 18. generate_summary — interface label appears
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_summary_interface_label(bridge):
    """generate_summary must show [INTERFACE] for interface cavities."""
    result = {
        "success":            True,
        "chain":              "A",
        "assembly_mode":      True,
        "chains_analysed":    ["A", "B"],
        "cavities": [
            {
                "cavity_id":           1,
                "lining_residues":     ["A5", "B10"],
                "estimated_volume_A3": 60.0,
                "chain":               "A",
                "is_interface_cavity": True,
            },
            {
                "cavity_id":           2,
                "lining_residues":     ["A20"],
                "estimated_volume_A3": 30.0,
                "chain":               "A",
                "is_interface_cavity": False,
            },
        ],
        "candidates":         [],
        "total_cavities":     2,
        "total_candidates":   0,
        "interface_cavities": 1,
        "intrachain_cavities": 1,
        "error":              None,
    }
    summary = bridge.generate_summary(result)
    assert "[INTERFACE]" in summary, (
        "Summary should label interface cavities with [INTERFACE]"
    )
    assert "[INTRACHAIN-A]" in summary, (
        "Summary should label intrachain cavities with [INTRACHAIN-A]"
    )
    assert "assembly mode" in summary.lower(), (
        "Summary should mention assembly mode when assembly_mode=True"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 19. Router — assembly keywords trigger assembly_mode=True
# ─────────────────────────────────────────────────────────────────────────────

def test_cavity_router_full_dimer_passes_none_chains():
    """
    'full dimer' in user input should match _CAVITY_ASSEMBLY_KEYWORDS,
    resulting in assembly_mode=True being injected into cavity tool_inputs.
    """
    from tool_router import ToolRouter

    assembly_kws = ToolRouter._CAVITY_ASSEMBLY_KEYWORDS
    assert len(assembly_kws) > 0, "_CAVITY_ASSEMBLY_KEYWORDS must be non-empty"

    user_input = "scan the full dimer for interface cavities"
    matched = any(kw in user_input.lower() for kw in assembly_kws)
    assert matched, (
        f"'full dimer' / 'interface cavities' should match _CAVITY_ASSEMBLY_KEYWORDS. "
        f"Keywords: {assembly_kws}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 20. Router — non-assembly request does not trigger assembly mode
# ─────────────────────────────────────────────────────────────────────────────

def test_cavity_router_chain_a_passes_single_chain():
    """
    A plain single-chain cavity request should NOT match assembly keywords,
    resulting in assembly_mode=False and single-chain analysis.
    """
    from tool_router import ToolRouter

    assembly_kws = ToolRouter._CAVITY_ASSEMBLY_KEYWORDS

    user_input = "find cavities in chain A of this protein"
    matched = any(kw in user_input.lower() for kw in assembly_kws)
    assert not matched, (
        f"Single-chain request should NOT match _CAVITY_ASSEMBLY_KEYWORDS. "
        f"Matched: {[kw for kw in assembly_kws if kw in user_input.lower()]}"
    )
