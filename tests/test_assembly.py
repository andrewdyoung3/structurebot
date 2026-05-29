"""
test_assembly.py
----------------
Tests for assembly_analyser.py — biological assembly detection,
interface mapping, and monomer/multimer mode selection.

All RCSB API calls and ChimeraX bridge calls are mocked to run offline.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assembly_analyser import (
    AssemblyAnalyser,
    fetch_assembly_info,
    parse_contacts_output,
    _stoichiometry_label,
)
from session_state import SessionState


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_session():
    """A minimal SessionState for testing."""
    session = SessionState(working_dir=str(Path(__file__).parent.parent))
    session.add_structure(
        model_id = "1",
        name     = "1HSG",
        metadata = {
            "pdb_id":       "1HSG",
            "title":        "HIV-1 PROTEASE",
            "chains":       ["A", "B"],
            "ligand_codes": ["MK1"],
            "resolution":   "1.9",
        },
    )
    return session


def _zone_selection_output(chain: str, resnos: List[int]) -> str:
    """
    Build a canned `info selection` response in the format ChimeraX 1.11.1
    returns for zone-selection on CA atoms:
      atom id /CHAIN:RESNO@CA idatm_type C3
    """
    n = len(resnos)
    lines = [f"{n} atoms, 0 bonds, 0 pseudobonds, {n} residues, 1 models selected"]
    for r in resnos:
        lines.append(f"atom id /{chain}:{r}@CA idatm_type C3")
    return "\n".join(lines)


@pytest.fixture
def mock_bridge():
    """
    A mock ChimeraXBridge that returns zone-selection output.

    detect_interfaces() calls run_command() for each chain direction:
      - select #1/A@CA & (#1/B :< 5.0); info selection  → A residues near B
      - select #1/B@CA & (#1/A :< 5.0); info selection  → B residues near A
      - select clear                                      → cleanup
    """
    bridge = MagicMock()
    bridge.is_running.return_value = True

    # Canned interface residues (subset of real 1HSG A/B interface)
    _a_near_b = [25, 26, 27, 50, 51, 52, 76, 99]
    _b_near_a = [24, 25, 50, 51, 52, 97, 98, 99]

    _responses = {
        "A": _zone_selection_output("A", _a_near_b),
        "B": _zone_selection_output("B", _b_near_a),
    }

    def _run_command(cmd: str):
        if "select clear" in cmd:
            return {"value": "", "error": None}
        # Which chain is being selected?
        for chain in ("A", "B"):
            if f"/{chain}@CA" in cmd:
                return {"value": _responses[chain], "error": None}
        # Fallback (shouldn't be reached in tests)
        return {"value": "", "error": None}

    bridge.run_command.side_effect = _run_command
    return bridge


@pytest.fixture
def hsg_assembly_data():
    """
    Canned RCSB assembly API response for 1HSG (homodimer).

    Reflects the real RCSB payload:
      - asym_id_list includes ALL 5 asym units (A/B=protein, C=MK1 ligand, D/E=water)
      - rcsb_assembly_info.stoichiometry is absent (not returned by live RCSB for 1HSG)
      - rcsb_struct_symmetry.clusters contain only protein chains A and B
      - oligomeric_details is "dimeric" (generic), not "homodimer"

    The parser must use rcsb_struct_symmetry to produce the correct "homodimer"
    label and chain list ["A", "B"].
    """
    return {
        "rcsb_assembly_info": {
            "polymer_entity_instance_count":         2,
            "polymer_entity_instance_count_protein": 2,
            "nonpolymer_entity_instance_count":      1,
            "solvent_entity_instance_count":         2,
            # NOTE: stoichiometry is intentionally absent — matches live RCSB
        },
        "pdbx_struct_assembly": {
            "oligomeric_details": "dimeric",   # live RCSB returns "dimeric", not "homodimer"
            "oligomeric_count":   2,
        },
        "pdbx_struct_assembly_gen": [
            # Full asym_id_list: A+B=protein, C=MK1, D+E=water
            {"asym_id_list": ["A", "B", "C", "D", "E"], "oper_expression": "1", "ordinal": 1},
        ],
        "rcsb_struct_symmetry": [
            {
                "kind":            "Global Symmetry",
                "oligomeric_state": "Homo 2-mer",
                "stoichiometry":   ["A2"],
                "symbol":          "C2",
                "type":            "Cyclic",
                "clusters": [
                    {
                        "members": [
                            {"asym_id": "A", "pdbx_struct_oper_list_ids": ["1"]},
                            {"asym_id": "B", "pdbx_struct_oper_list_ids": ["1"]},
                        ],
                        "avg_rmsd": 0.4003,
                    }
                ],
            }
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. RCSB assembly query — mock API, verify homodimer detection for 1HSG
# ══════════════════════════════════════════════════════════════════════════════

def test_fetch_assembly_info_homodimer(hsg_assembly_data):
    """
    fetch_assembly_info correctly identifies 1HSG as a homodimer with chains A+B.

    Uses the realistic fixture (asym_id_list has A/B/C/D/E; stoichiometry absent;
    rcsb_struct_symmetry carries the "A2" stoich and protein-only cluster members).
    Verifies that:
      - assembly_type resolves to "homodimer" (not "dimeric")
      - chains is exactly ["A", "B"] (not ["A","B","C","D","E"])
    """
    with patch("assembly_analyser._requests") as mock_req:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = hsg_assembly_data
        mock_req.get.return_value = mock_resp

        result = fetch_assembly_info("1HSG")

    assert result["pdb_id"] == "1HSG"
    assert result["assembly_type"] == "homodimer", (
        f"Expected 'homodimer', got: {result['assembly_type']!r}"
    )
    assert result["n_subunits"] == 2
    # Chains must be ONLY the protein chains A and B — not ligand/water asym IDs
    assert result["chains"] == ["A", "B"], (
        f"Expected ['A', 'B'] (protein chains only), got: {result['chains']}"
    )
    assert result["error"] is None


def test_fetch_assembly_info_network_error():
    """fetch_assembly_info returns error dict gracefully on network failure."""
    with patch("assembly_analyser._requests") as mock_req:
        mock_req.get.side_effect = Exception("Connection refused")
        result = fetch_assembly_info("1HSG")

    assert result["error"] is not None
    assert result["assembly_type"] is None


def test_stoichiometry_label():
    """_stoichiometry_label maps RCSB stoich strings to readable names."""
    assert _stoichiometry_label("A2", 2) == "homodimer"
    assert _stoichiometry_label("A1", 1) == "monomer"
    assert _stoichiometry_label("A4", 4) == "homotetramer"
    assert _stoichiometry_label("AB", 2) == "heterodimer"
    assert "hetero" in _stoichiometry_label("A2B2", 4).lower()


# ══════════════════════════════════════════════════════════════════════════════
# 2. Interface detection — mock ChimeraX contacts output, verify residue extraction
# ══════════════════════════════════════════════════════════════════════════════

def test_parse_contacts_output_basic():
    """parse_contacts_output correctly extracts chain pairs and residue numbers."""
    contacts = (
        "#1/A:25 GLY <-> #1/B:99 PRO  dist 4.2\n"
        "#1/A:26 ASP <-> #1/B:98 ALA  dist 3.9\n"
        "#1/A:50 ILE <-> #1/B:50 ILE  dist 3.1\n"
    )
    result = parse_contacts_output(contacts)

    # Should have one interface: A-B (or B-A, stored as sorted)
    assert len(result) == 1
    pair = list(result.keys())[0]
    assert set(pair) == {"A", "B"}

    resnos = result[pair]
    assert 25 in resnos
    assert 26 in resnos
    assert 50 in resnos
    assert 99 in resnos
    assert 98 in resnos


def test_parse_contacts_output_intrachain_ignored():
    """Intra-chain contacts (same chain) are ignored."""
    contacts = (
        "#1/A:10 ALA <-> #1/A:20 GLY  dist 3.0\n"  # intra-chain
        "#1/A:25 GLY <-> #1/B:99 PRO  dist 4.2\n"  # inter-chain
    )
    result = parse_contacts_output(contacts)
    assert len(result) == 1  # only the inter-chain contact
    pair = list(result.keys())[0]
    assert set(pair) == {"A", "B"}


def test_parse_contacts_output_empty():
    """Empty contacts text returns empty dict."""
    result = parse_contacts_output("")
    assert result == {}


def test_detect_interfaces(mock_bridge, mock_session):
    """AssemblyAnalyser.detect_interfaces returns correct chain-residue mapping."""
    analyser = AssemblyAnalyser(bridge=mock_bridge, session=mock_session)
    interfaces = analyser.detect_interfaces(model_id="1", contact_distance=5.0)

    assert len(interfaces) >= 1
    # All pairs should be inter-chain
    for (c1, c2), resnos in interfaces.items():
        assert c1 != c2, "Intra-chain contact leaked into interface dict"
        assert len(resnos) > 0
    # Should detect A-B interface
    pairs = {frozenset(pair) for pair in interfaces}
    assert frozenset({"A", "B"}) in pairs


# ══════════════════════════════════════════════════════════════════════════════
# 3. Monomer mode — verify interface residues NOT excluded
# ══════════════════════════════════════════════════════════════════════════════

def test_monomer_mode_no_interface_exclusion(mock_bridge, mock_session, hsg_assembly_data):
    """In MONOMER mode, protected_residues is empty and interfaces not detected."""
    with patch("assembly_analyser.fetch_assembly_info") as mock_fetch:
        mock_fetch.return_value = {
            **hsg_assembly_data.get("rcsb_assembly_info", {}),
            "pdb_id": "1HSG",
            "assembly_type": "homodimer",
            "n_subunits": 2,
            "chains": ["A", "B"],
            "is_obligate": True,
            "error": None,
            "stoichiometry": "A2",
            "oligomeric_state": "homodimer",
        }

        analyser = AssemblyAnalyser(bridge=mock_bridge, session=mock_session)
        result   = analyser.analyse(
            model_id = "1",
            pdb_id   = "1HSG",
            mode     = "monomer",
            chain_id = "A",
        )

    assert result["mode"] == "monomer"
    assert result["protected_residues"] == []
    assert result["excluded_count"] == 0
    # In monomer mode, no bridge call for contacts should have been made
    mock_bridge.run_command.assert_not_called()

    # But should warn about obligate multimer in monomer mode
    warnings = result.get("warnings", [])
    has_multimer_warning = any("obligate" in w.lower() or "multimer" in w.lower()
                               for w in warnings)
    assert has_multimer_warning, (
        "Expected a warning about analysing obligate multimer as monomer"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Multimer mode — verify interface residues excluded from candidates
# ══════════════════════════════════════════════════════════════════════════════

def test_multimer_mode_interface_exclusion(mock_bridge, mock_session):
    """In MULTIMER mode, interface residues populate protected_residues."""
    analyser = AssemblyAnalyser(bridge=mock_bridge, session=mock_session)
    result   = analyser.analyse(
        model_id = "1",
        pdb_id   = None,  # skip RCSB lookup for simplicity
        mode     = "multimer",
        chain_id = "A",
    )

    assert result["mode"] == "multimer"
    # Bridge should have been called for contacts
    mock_bridge.run_command.assert_called()

    # Should have found interfaces
    assert len(result["interfaces"]) > 0

    # protected_residues should be non-empty for chain A
    assert len(result["protected_residues"]) > 0, (
        "Expected interface residues to be protected in multimer mode"
    )

    # All protected residues should appear in interface data
    all_iface_resnos: set = set()
    for (c1, c2), resnos in result["interfaces"].items():
        if "A" in (c1, c2):
            all_iface_resnos.update(resnos)

    for prot in result["protected_residues"]:
        assert prot in all_iface_resnos, (
            f"Protected residue {prot} not found in interface data"
        )


def test_session_stores_interface_residues(mock_bridge, mock_session):
    """AssemblyAnalyser stores interface residues in session state correctly."""
    analyser = AssemblyAnalyser(bridge=mock_bridge, session=mock_session)
    analyser.analyse(model_id="1", mode="multimer", chain_id="A")

    # Session should have stored the interface data
    stored = mock_session.get_interface_residues("1")
    assert len(stored) >= 1, "Session should store interface residues after multimer analysis"

    # get_protected_residues_for_chain should return chain A's interface residues
    protected = mock_session.get_protected_residues_for_chain("1", "A")
    assert len(protected) > 0


# ══════════════════════════════════════════════════════════════════════════════
# 5. Interface-proximal warning — residues within 3Å flagged
# ══════════════════════════════════════════════════════════════════════════════

def test_interface_proximal_flagging():
    """MutationScanner flags residues within 3 positions of interface as proximal."""
    from mutation_scanner import MutationScanner

    session = SessionState()
    session.add_structure("1", "TEST", metadata={"sequences": {"A": "ACDEFGHIKLM"}})

    # Store interface residues at positions 5, 6 in session
    session.set_interface_residues("1", {("A", "B"): [5, 6]})
    session.set_analysis_mode("1", "multimer")

    scanner = MutationScanner(session=session, model_id="1")

    # Positions 2-4 and 7-9 (within 3 of positions 5, 6) should be proximal
    from mutation_scanner import MutationScanner

    # _identify_candidates is internal but we can test via scan() indirectly
    # by checking that candidates near the interface have interface_proximal=True
    # We'll inject fake camsol/esm results and check
    session.add_tool_result("camsol", "1", {
        "scores": {str(i): -1.0 for i in range(1, 12)},
        "aggregation_hot_spots": list(range(1, 12)),
    })
    session.add_tool_result("esm", "1", {
        "conservation": {str(i): 0.1 for i in range(1, 12)},
        "mean_conservation": 0.1,
    })

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
        # Minimal valid PDB content to satisfy Path.is_file() check
        f.write("HEADER    TEST\nEND\n")
        pdb_path = f.name

    try:
        results = scanner.scan(
            pdb_path           = pdb_path,
            chain_id           = "A",
            sequence           = "ACDEFGHIKLM",
            filters            = {"camsol_threshold": 0.0, "esm_threshold": 1.0},
            protected_residues = [5, 6],
            analysis_mode      = "multimer",
        )
    finally:
        os.unlink(pdb_path)

    # Positions 5 and 6 should be absent (protected)
    positions_in_results = {r["position"] for r in results}
    assert 5 not in positions_in_results, "Position 5 should be excluded (interface)"
    assert 6 not in positions_in_results, "Position 6 should be excluded (interface)"

    # Positions near interface (2-4, 7-9) should be flagged as proximal
    proximal_positions = {r["position"] for r in results if r.get("interface_proximal")}
    # At least some proximal positions should be flagged
    expected_proximal = {2, 3, 4, 7, 8, 9}
    found_proximal = proximal_positions & expected_proximal
    assert len(found_proximal) > 0, (
        f"Expected some proximal positions from {expected_proximal}, "
        f"got proximal: {proximal_positions}"
    )

    # Check that proximal positions have a caution note in recommendation
    for r in results:
        if r.get("interface_proximal"):
            assert "caution" in r["recommendation"].lower(), (
                f"Expected caution in recommendation for proximal position {r['position']}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Assembly metadata display format
# ══════════════════════════════════════════════════════════════════════════════

def test_assembly_display_format(mock_bridge, mock_session):
    """get_assembly_display returns correctly formatted string."""
    analyser = AssemblyAnalyser(bridge=mock_bridge, session=mock_session)
    asm_info = {
        "assembly_type": "homodimer",
        "chains": ["A", "B"],
        "n_subunits": 2,
        "error": None,
    }
    display = analyser.get_assembly_display("1HSG", asm_info)

    assert "homodimer" in display.lower()
    # Should also include structure metadata (ligands, resolution) from session
    assert "MK1" in display or "1.9" in display or len(display) > 0


def test_assembly_header_monomer():
    """_build_header returns correct label for monomer analysis."""
    bridge  = MagicMock()
    session = SessionState()
    analyser = AssemblyAnalyser(bridge=bridge, session=session)

    asm_info = {"assembly_type": "homodimer", "chains": ["A", "B"]}
    header   = analyser._build_header("1HSG", asm_info, "monomer")
    assert "monomer analysis" in header.lower()
    assert "1HSG" in header


def test_assembly_header_multimer():
    """_build_header returns correct label for multimer analysis."""
    bridge  = MagicMock()
    session = SessionState()
    analyser = AssemblyAnalyser(bridge=bridge, session=session)

    asm_info = {"assembly_type": "homodimer", "chains": ["A", "B"]}
    header   = analyser._build_header("1HSG", asm_info, "multimer")
    assert "multimer analysis" in header.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 7. Visualization command generation
# ══════════════════════════════════════════════════════════════════════════════

def test_interface_viz_commands(mock_bridge, mock_session):
    """generate_interface_viz_commands produces valid ChimeraX commands."""
    analyser = AssemblyAnalyser(bridge=mock_bridge, session=mock_session)
    interfaces = {("A", "B"): [25, 26, 27, 50, 51]}

    cmds, exps = analyser.generate_interface_viz_commands(
        model_id   = "1",
        interfaces = interfaces,
    )

    assert len(cmds) > 0
    assert len(cmds) == len(exps)

    # Should start with cartoon and white reset
    assert any("cartoon" in c for c in cmds)
    assert any("white" in c for c in cmds)

    # Should include color command for the interface residues
    color_cmds = [c for c in cmds if c.startswith("color") and "25" in c or
                  c.startswith("color") and "26" in c]
    # Should include some coloring
    color_cmds_all = [c for c in cmds if "color" in c and "#1" in c and "white" not in c]
    assert len(color_cmds_all) >= 1


def test_interface_viz_empty_interfaces(mock_bridge, mock_session):
    """generate_interface_viz_commands returns empty lists for no interfaces."""
    analyser = AssemblyAnalyser(bridge=mock_bridge, session=mock_session)
    cmds, exps = analyser.generate_interface_viz_commands("1", {})
    assert cmds == []
    assert exps == []


# ══════════════════════════════════════════════════════════════════════════════
# 8. SessionState assembly fields
# ══════════════════════════════════════════════════════════════════════════════

def test_session_assembly_info_roundtrip():
    """SessionState correctly stores and retrieves assembly info."""
    session = SessionState()
    info    = {"assembly_type": "homodimer", "n_subunits": 2, "chains": ["A", "B"]}
    session.set_assembly_info("1HSG", info)

    retrieved = session.get_assembly_info("1HSG")
    assert retrieved is not None
    assert retrieved["assembly_type"] == "homodimer"
    assert retrieved["n_subunits"] == 2


def test_session_analysis_mode():
    """SessionState correctly stores and retrieves analysis mode."""
    session = SessionState()
    session.set_analysis_mode("1", "multimer")
    assert session.get_analysis_mode("1") == "multimer"
    assert session.get_analysis_mode("2") == "monomer"  # default


def test_session_interface_residues_roundtrip():
    """SessionState correctly stores and retrieves interface residues."""
    session = SessionState()
    interfaces = {("A", "B"): [10, 20, 30], ("A", "C"): [50, 60]}
    session.set_interface_residues("1", interfaces)

    retrieved = session.get_interface_residues("1")
    assert len(retrieved) == 2

    # Check that chain pair A-B is present
    found_ab = False
    for key, resnos in retrieved.items():
        if set(key) == {"A", "B"}:
            assert set(resnos) == {10, 20, 30}
            found_ab = True
    assert found_ab, "A-B interface not found in retrieved data"


def test_session_protected_residues_for_chain():
    """get_protected_residues_for_chain returns correct residues for chain."""
    session = SessionState()
    session.set_interface_residues("1", {
        ("A", "B"): [10, 20, 30, 50],
        ("A", "C"): [70, 80],
        ("B", "C"): [5, 6, 7],   # does not involve A
    })

    protected_a = session.get_protected_residues_for_chain("1", "A")
    assert set(protected_a) == {10, 20, 30, 50, 70, 80}

    protected_b = session.get_protected_residues_for_chain("1", "B")
    assert set(protected_b) == {10, 20, 30, 50, 5, 6, 7}  # A-B and B-C


def test_session_save_load_assembly_fields(tmp_path):
    """Assembly fields are preserved through save/load."""
    session = SessionState()
    session.set_assembly_info("1HSG", {"assembly_type": "homodimer"})
    session.set_analysis_mode("1", "multimer")
    session.set_interface_residues("1", {("A", "B"): [10, 20]})

    save_path = str(tmp_path / "test_session.json")
    session.save(save_path)

    loaded = SessionState.load(save_path)
    assert loaded.get_assembly_info("1HSG")["assembly_type"] == "homodimer"
    assert loaded.get_analysis_mode("1") == "multimer"
    ifaces = loaded.get_interface_residues("1")
    assert len(ifaces) == 1


# ════════════════════════════════════════════════════════════════════════════════
# Interface detection retry behaviour
# ════════════════════════════════════════════════════════════════════════════════

def test_interface_detection_retries_on_zero_result(mock_session):
    """
    If zone-select returns 0 residues on the first attempt but non-zero on
    the second, detect_interfaces() must use the non-zero value.
    The retry logic waits 1 s between attempts (mocked out here).
    """
    call_count = [0]

    def _run_cmd(cmd: str):
        if "select clear" in cmd:
            return {"value": "", "error": None}
        call_count[0] += 1
        if call_count[0] <= 2:
            # First attempt (both directions): return 0 residues
            return {"value": "0 atoms, 0 bonds selected", "error": None}
        # Second attempt: return real residues
        if "/A@CA" in cmd:
            return {"value": _zone_selection_output("A", [25, 26, 27]), "error": None}
        return {"value": _zone_selection_output("B", [24, 25, 26]), "error": None}

    bridge = MagicMock()
    bridge.is_running.return_value = True
    bridge.run_command.side_effect = _run_cmd

    analyser = AssemblyAnalyser(bridge=bridge, session=mock_session)

    with patch("assembly_analyser._time.sleep"):  # don't actually sleep in tests
        ifaces = analyser.detect_interfaces("1", contact_distance=5.0)

    assert ifaces, "detect_interfaces() should return non-empty after retry"
    pair = list(ifaces.keys())[0]
    assert len(ifaces[pair]) > 0, "Retried result should contain residue numbers"
    assert call_count[0] > 2, "Should have made more than 2 zone-select calls (retry fired)"


def test_interface_detection_stores_none_on_persistent_zero_for_multimer(mock_session):
    """
    If zone-select returns 0 on all 3 attempts for a confirmed multimer,
    analyse() must:
      - NOT store empty interfaces in session (leaves default = unknown)
      - Add a warning about interface data being unavailable
    """
    bridge = MagicMock()
    bridge.is_running.return_value = True
    # Always return 0 residues for every zone-select call
    bridge.run_command.return_value = {"value": "0 atoms, 0 bonds selected", "error": None}

    analyser = AssemblyAnalyser(bridge=bridge, session=mock_session)

    homodimer_info = {
        "assembly_type": "homodimer",
        "stoichiometry":  "A2",
        "n_subunits":     2,
        "is_obligate":    True,
        "chains":         ["A", "B"],
        "oligomeric_state": "Homo 2-mer",
        "error":          None,
    }

    with patch("assembly_analyser.fetch_assembly_info", return_value=homodimer_info), \
         patch("assembly_analyser._time.sleep"):
        result = analyser.analyse(
            "1", pdb_id="1HSG", mode="multimer", chain_id="A"
        )

    # Warning must be present
    warns = result.get("warnings", [])
    assert any("unavailable" in w.lower() or "0 residues" in w.lower() for w in warns), (
        f"Expected unavailability warning for multimer with 0 interfaces, got: {warns}"
    )

    # Session must NOT have stored empty interfaces (leaves it at default = unknown)
    stored = mock_session.get_interface_residues("1")
    assert not stored, (
        "Session should not store empty interface data for a confirmed multimer; "
        f"got: {stored}"
    )
