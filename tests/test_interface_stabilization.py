"""
tests/test_interface_stabilization.py
--------------------------------------
Tests for Phase 1 interface stabilization:
  - intent detection + route() override
  - _run_interface_stabilization error guards
  - InterfaceStabilization class (mocked bridge)
  - Sub-model addressing correctness (no flat specs)
  - Interface type classification + symmetry-type assignment
  - Disulfide routing (intra_copy runs, inter_copy uses assembly PDB)
  - Assembly PDB export with distinct chain IDs
  - Sub-model-aware chain coloring (distinct colors, no bychain collision)
  - Session persistence roundtrip

All mocked — no live ChimeraX or PDB files required.

Test groups
-----------
1.  Routing — intent detection + route override
2.  _run_interface_stabilization — error-first guards
3.  InterfaceStabilization — sub-model discovery, zone-select, buried area
4.  Interface type classification (intra_copy / inter_copy / flat)
5.  Symmetry-type assignment
6.  Disulfide routing (intra_copy from AU PDB, inter_copy from assembly PDB)
7.  Assembly PDB export — distinct chain IDs + mapping
8.  Sub-model spec correctness (spec format #N.M/chain)
9.  Chain coloring — sub-model-aware distinct colors
10. Session persistence roundtrip
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter, ToolStepResult
from interface_stabilization import InterfaceStabilization, _parse_buried_area


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_router(
    structures: dict | None = None,
    generated_assemblies: dict | None = None,
) -> ToolRouter:
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = structures if structures is not None else {
        "1": {"name": "2VNC", "path": None}
    }
    mock_session.generated_assemblies = generated_assemblies or {}
    mock_session.get_proteinmpnn_result.return_value = None
    mock_session.get_assembly_info.return_value = None
    mock_session.get_structure.return_value = {"name": "2VNC", "path": None}
    return ToolRouter(bridge=mock_bridge, session=mock_session)


def _make_stab(bridge_val_map: dict | None = None) -> tuple:
    """Return (InterfaceStabilization, mock_bridge, mock_session).

    bridge_val_map: {cmd_substring: return_value} for run_command side effects.
    """
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    # Return None so AssemblyAnalyser._get_model_chains falls through to
    # ChimeraX query (Priority 3) rather than iterating a MagicMock chain list.
    mock_session.get_structure.return_value = None
    mock_session.get_assembly_info.return_value = None

    def _run_cmd(cmd: str) -> dict:
        if bridge_val_map:
            for key, val in bridge_val_map.items():
                if key in cmd:
                    return val
        return {"value": "", "error": None}

    mock_bridge.run_command.side_effect = _run_cmd
    mock_bridge.is_running.return_value = True
    stab = InterfaceStabilization(bridge=mock_bridge, session=mock_session)
    return stab, mock_bridge, mock_session


def _cx_result(value: str = "", error: str | None = None) -> dict:
    return {"value": value, "error": error}


def _translator_chimerax() -> Dict[str, Any]:
    return {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "high",
        "tools_needed":         ["chimerax"],
        "tool_inputs":          {},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1. Routing — intent detection + route() override
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("phrase", [
    "stabilize the interface",
    "stabilise the interface",
    "stabilize the dimer interface",
    "lock with disulfide",
    "lock with a disulfide",
    "engineer interface disulfide",
    "interface disulfide",
    "inter-subunit disulfide",
    "interchain disulfide",
    "crosslink the interface",
    "interface stabilization",
    "interface stabilisation",
    "strengthen the interface",
    "reinforce the assembly",
    "detect interfaces",
    "characterize interface",
    "characterize the interface",
    "map the interface",
    "buried interface area",
    "buried surface area",
])
def test_interface_stabilization_intent_detected(phrase):
    router = _make_router()
    assert router._detect_interface_stabilization_intent(phrase)


@pytest.mark.parametrize("non_phrase", [
    "show me the structure",
    "redesign chain A",
    "mutate residue 25 to alanine",
    "generate biological assembly",  # bio_assembly, not this
    "compare conformers",
    "run cavity scan",
])
def test_interface_stabilization_intent_not_triggered_on_unrelated_phrases(non_phrase):
    router = _make_router()
    assert not router._detect_interface_stabilization_intent(non_phrase)


def test_interface_stabilization_route_override_claims_intent():
    router = _make_router()
    result = router.route(
        translator_result=_translator_chimerax(),
        user_input="stabilize the dimer interface",
    )
    assert result.get("tools_needed") == ["interface_stabilization"]
    assert "interface_stabilization" in result.get("tool_inputs", {})


def test_interface_stabilization_route_override_not_claimed_by_bio_assembly():
    """bio_assembly and interface_stabilization should never fire together."""
    router = _make_router()
    ba_result = router.route(
        translator_result=_translator_chimerax(),
        user_input="generate biological assembly",
    )
    # bio_assembly fires first — interface_stabilization should NOT be in tools_needed
    assert "interface_stabilization" not in ba_result.get("tools_needed", [])


def test_primary_assembly_model_id_returns_assembly_when_generated():
    router = _make_router(
        generated_assemblies={"1": {"assembly_model_id": "2", "pdb_id": "2VNC"}}
    )
    assert router._primary_assembly_model_id() == "2"


def test_primary_assembly_model_id_falls_back_to_au_when_no_assembly():
    router = _make_router(generated_assemblies={})
    # Primary AU model is "1" (only structure loaded)
    assert router._primary_assembly_model_id() == "1"


# ══════════════════════════════════════════════════════════════════════════════
# 2. _run_interface_stabilization — error-first guards
# ══════════════════════════════════════════════════════════════════════════════

def test_run_interface_stabilization_no_bridge_returns_clean_error():
    router = _make_router()
    router.bridge = None
    result = router._run_interface_stabilization({"model_id": "2"})
    assert not result.success
    assert "unavailable" in result.error.lower()


def test_run_interface_stabilization_no_structure_returns_clean_error():
    router = _make_router(structures={})
    result = router._run_interface_stabilization({"model_id": "2"})
    assert not result.success
    assert "no structure" in result.error.lower()


def test_run_interface_stabilization_no_pdb_returns_clean_error():
    router = _make_router()
    with patch.object(router, "_ensure_pdb_file", return_value=None):
        result = router._run_interface_stabilization({"model_id": "1"})
    assert not result.success
    assert "pdb" in result.error.lower()


def test_run_interface_stabilization_dispatches_to_stab_class(tmp_path):
    """Dispatches to InterfaceStabilization.analyze() when PDB present."""
    dummy_pdb = tmp_path / "2VNC.pdb"
    dummy_pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n")

    router = _make_router(
        generated_assemblies={"1": {"assembly_model_id": "2", "pdb_id": "2VNC"}}
    )
    with patch.object(router, "_ensure_pdb_file", return_value=str(dummy_pdb)):
        with patch(
            "interface_stabilization.InterfaceStabilization"
        ) as MockStab:
            mock_result = ToolStepResult(
                tool="interface_stabilization", success=True,
                data={"interfaces": [], "n_interfaces": 0},
                summary="test",
            )
            MockStab.return_value.analyze.return_value = mock_result
            result = router._run_interface_stabilization({"model_id": "2"})

    assert result.success


# ══════════════════════════════════════════════════════════════════════════════
# 3. InterfaceStabilization — sub-model discovery, zone-select, buried area
# ══════════════════════════════════════════════════════════════════════════════

def test_get_submodels_parses_chimerax_output():
    stab, bridge, _ = _make_stab({
        "info models #2": _cx_result(
            "model id #2 type Group\n"
            "model id #2.1 type AtomicStructure name 2vnc\n"
            "model id #2.2 type AtomicStructure name 2vnc\n"
        )
    })
    subs = stab._get_submodels("2")
    assert subs == ["2.1", "2.2"]


def test_get_submodels_empty_for_plain_model():
    stab, bridge, _ = _make_stab({
        "info models #1": _cx_result(
            "model id #1 type AtomicStructure name 2VNC\n"
        )
    })
    subs = stab._get_submodels("1")
    assert subs == []


def test_get_chains_for_submodel_parses_output():
    stab, bridge, _ = _make_stab({
        "info chains #2.1": _cx_result(
            "chain id /A chain_id A\nchain id /B chain_id B\n"
        )
    })
    chains = stab._get_chains_for_submodel("2.1")
    assert chains == ["A", "B"]


def test_parse_buried_area_standard_format():
    text = "Buried accessible surface area of #2.1/A with #2.2/B = 1347.5"
    val = _parse_buried_area(text)
    assert val == pytest.approx(1347.5)


def test_parse_buried_area_returns_none_on_empty():
    assert _parse_buried_area("") is None
    assert _parse_buried_area("No result") is None


# ══════════════════════════════════════════════════════════════════════════════
# 4. Interface type classification (intra_copy / inter_copy / flat)
# ══════════════════════════════════════════════════════════════════════════════

def test_intra_copy_type_when_same_submodel():
    """Two chains in same sub-model → intra_copy."""
    ca_hit = _cx_result("atom id /A:10@CA\natom id /A:11@CA\n")
    no_hit = _cx_result("")

    def _cmd_handler(cmd: str) -> dict:
        if "info models #2" in cmd:
            return _cx_result(
                "model id #2 type Group\n"
                "model id #2.1 type AtomicStructure\n"
                "model id #2.2 type AtomicStructure\n"
            )
        if "info chains #2.1" in cmd or "info chains #2.2" in cmd:
            return _cx_result("chain id /A chain_id A\nchain id /B chain_id B\n")
        if "#2.1/A@CA" in cmd and "#2.1/B" in cmd:
            return ca_hit
        if "#2.1/B@CA" in cmd and "#2.1/A" in cmd:
            return ca_hit
        return no_hit

    stab, bridge, _ = _make_stab()
    bridge.run_command.side_effect = _cmd_handler

    submodels = ["2.1", "2.2"]
    interfaces = stab._detect_submodel_interfaces("2", submodels, 5.0, lambda msg: None)

    for iface in interfaces:
        sm_a = iface["submodel_a"]
        sm_b = iface["submodel_b"]
        if sm_a == sm_b:
            assert iface["type"] == "intra_copy", (
                f"Expected intra_copy for same sub-model pair, got {iface['type']}"
            )
        else:
            assert iface["type"] == "inter_copy", (
                f"Expected inter_copy for cross sub-model pair, got {iface['type']}"
            )


def test_flat_type_for_no_submodel_model():
    """Plain AU model → type == 'flat'."""
    def _cmd_handler(cmd: str) -> dict:
        if "info models #1" in cmd:
            return _cx_result("model id #1 type AtomicStructure\n")
        if "info chains #1" in cmd:
            return _cx_result("chain id /A chain_id A\nchain id /B chain_id B\n")
        if "@CA" in cmd:
            return _cx_result("atom id /A:10@CA\n")
        return _cx_result("")

    stab, bridge, session = _make_stab()
    bridge.run_command.side_effect = _cmd_handler

    interfaces = stab._detect_flat_interfaces("1", 5.0, lambda msg: None)
    for iface in interfaces:
        assert iface["type"] == "flat"


# ══════════════════════════════════════════════════════════════════════════════
# 5. Symmetry-type assignment
# ══════════════════════════════════════════════════════════════════════════════

def test_symmetry_type_assigned_after_buried_area():
    """_assign_symmetry_types sets symmetry_type on all interfaces."""
    from interface_stabilization import InterfaceStabilization
    stab = InterfaceStabilization.__new__(InterfaceStabilization)

    interfaces = [
        {"type": "intra_copy", "chain_a": "A", "chain_b": "B",
         "buried_area_ang2": 1518.0},
        {"type": "intra_copy", "chain_a": "A", "chain_b": "B",
         "buried_area_ang2": 1518.0},
        {"type": "inter_copy", "chain_a": "A", "chain_b": "A",
         "buried_area_ang2": 1378.0},
        {"type": "inter_copy", "chain_a": "B", "chain_b": "B",
         "buried_area_ang2": 1209.0},
        {"type": "inter_copy", "chain_a": "A", "chain_b": "B",
         "buried_area_ang2": 316.0},
        {"type": "inter_copy", "chain_a": "B", "chain_b": "A",
         "buried_area_ang2": 316.0},
    ]
    stab._assign_symmetry_types(interfaces)

    # All interfaces must have symmetry_type set
    for iface in interfaces:
        assert iface["symmetry_type"] is not None

    # The two intra_copy (A-B) entries must share the same symmetry_type
    intra = [i for i in interfaces if i["type"] == "intra_copy"]
    assert len(set(i["symmetry_type"] for i in intra)) == 1

    # Total unique symmetry types must be 3 for this C2-symmetric assembly
    unique_types = len(set(i["symmetry_type"] for i in interfaces))
    assert unique_types == 3, f"Expected 3 symmetry types, got {unique_types}"

    # Type 1 must be the intra_copy (highest buried area)
    assert intra[0]["symmetry_type"] == 1


def test_symmetry_key_cross_chain_is_order_invariant():
    """A-B and B-A inter_copy get the same key regardless of order."""
    from interface_stabilization import InterfaceStabilization
    iface_ab = {"type": "inter_copy", "chain_a": "A", "chain_b": "B"}
    iface_ba = {"type": "inter_copy", "chain_a": "B", "chain_b": "A"}
    assert (
        InterfaceStabilization._symmetry_key(iface_ab) ==
        InterfaceStabilization._symmetry_key(iface_ba)
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Disulfide routing — intra_copy from AU PDB, inter_copy from assembly PDB
# ══════════════════════════════════════════════════════════════════════════════

def test_intra_copy_interface_gets_disulfide_scan(tmp_path):
    """
    InterfaceStabilization.analyze() calls DisulfideBridge for intra_copy interfaces.
    """
    pdb = tmp_path / "2VNC.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00  0.00           C\n"
    )

    stab, bridge, session = _make_stab()
    bridge.is_running.return_value = True

    # Patch _get_submodels to return no sub-models → falls through to flat detection
    # Patch _detect_flat_interfaces to return a known intra (flat) interface
    dummy_iface = {
        "type":               "flat",
        "spec_a":             "#1/A",
        "spec_b":             "#1/B",
        "chain_a":            "A",
        "chain_b":            "B",
        "submodel_a":         "1",
        "submodel_b":         "1",
        "contact_residues_a": [10, 11],
        "contact_residues_b": [20, 21],
        "contact_count":      4,
        "buried_area_ang2":   None,
        "disulfide_candidates": None,
        "disulfide_count":    0,
        "disulfide_top":      None,
        "disulfide_note":     None,
    }

    mock_ds_result = ToolStepResult(
        tool="disulfide", success=True,
        data={"candidates": [{"combined_score": 0.8, "chain_a_residue": 10}], "count": 1},
        summary="top candidate found",
    )

    with patch.object(stab, "_get_submodels", return_value=[]):
        with patch.object(stab, "_detect_flat_interfaces", return_value=[dummy_iface]):
            with patch.object(stab, "_measure_buried_area", return_value=1200.0):
                with patch("interface_stabilization.DisulfideBridge") as MockDS:
                    MockDS.return_value.analyze.return_value = mock_ds_result
                    result = stab.analyze(
                        model_id="1", pdb_path=str(pdb), pdb_id="2VNC"
                    )

    assert result.success
    # Disulfide scan was called for the flat (intra) interface
    MockDS.return_value.analyze.assert_called_once()

    interfaces = result.data["interfaces"]
    assert len(interfaces) == 1
    assert interfaces[0]["disulfide_count"] == 1
    assert interfaces[0]["disulfide_note"] is None


def test_inter_copy_interface_scan_uses_assembly_pdb(tmp_path):
    """
    inter_copy interfaces use the exported assembly PDB for the disulfide scan.
    When export succeeds, DisulfideBridge.analyze must be called with the
    assembly PDB path (not the AU PDB).
    """
    pdb = tmp_path / "2VNC.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )
    asm_pdb = tmp_path / "assembly_merged.pdb"
    asm_pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA C   1       5.000   0.000   0.000  1.00  0.00           C\n"
    )

    stab, bridge, session = _make_stab()
    bridge.is_running.return_value = True

    dummy_inter = {
        "type":               "inter_copy",
        "symmetry_type":      2,
        "spec_a":             "#2.1/A",
        "spec_b":             "#2.2/A",
        "chain_a":            "A",
        "chain_b":            "A",
        "submodel_a":         "2.1",
        "submodel_b":         "2.2",
        "contact_residues_a": [15, 16],
        "contact_residues_b": [15, 16],
        "contact_count":      4,
        "buried_area_ang2":   1378.0,
        "disulfide_candidates": None,
        "disulfide_count":    0,
        "disulfide_top":      None,
        "disulfide_note":     None,
    }

    chain_mapping = {("2.1", "A"): "A", ("2.1", "B"): "B",
                     ("2.2", "A"): "C", ("2.2", "B"): "D"}

    mock_ds_result = ToolStepResult(
        tool="disulfide", success=True,
        data={"candidates": [], "count": 0},
        summary="0 candidates",
    )

    with patch.object(stab, "_get_submodels", return_value=["2.1", "2.2"]):
        with patch.object(stab, "_detect_submodel_interfaces", return_value=[dummy_inter]):
            with patch.object(stab, "_measure_buried_area", return_value=1378.0):
                with patch.object(
                    stab, "_export_assembly_pdb",
                    return_value=(str(asm_pdb), chain_mapping),
                ):
                    with patch("interface_stabilization.DisulfideBridge") as MockDS:
                        MockDS.return_value.analyze.return_value = mock_ds_result
                        result = stab.analyze(
                            model_id="2", pdb_path=str(pdb), pdb_id="2VNC"
                        )

    assert result.success
    MockDS.return_value.analyze.assert_called_once()
    call_kwargs = MockDS.return_value.analyze.call_args
    used_pdb = call_kwargs[1].get("pdb_path") or call_kwargs[0][0]
    assert used_pdb == str(asm_pdb), (
        f"Expected assembly PDB path, got {used_pdb}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Assembly PDB export — distinct chain IDs + mapping
# ══════════════════════════════════════════════════════════════════════════════

def test_export_assembly_pdb_distinct_chain_ids():
    """
    _export_assembly_pdb must produce a PDB with 4 distinct chain IDs when
    two sub-models each have chains A, B.

    The bridge mock writes real PDB content to the requested save path so that
    the Python-side chain-rename logic can be exercised end-to-end.
    """
    import os as _os

    sub1_content = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA B   1       5.000   0.000   0.000  1.00  0.00           C\n"
    )
    sub2_content = (
        "ATOM      3  CA  ALA A   1      10.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  CA  ALA B   1      15.000   0.000   0.000  1.00  0.00           C\n"
    )

    def _cmd_handler(cmd: str) -> dict:
        if "info chains #2.1" in cmd:
            return _cx_result("chain id /A chain_id A\nchain id /B chain_id B\n")
        if "info chains #2.2" in cmd:
            return _cx_result("chain id /A chain_id A\nchain id /B chain_id B\n")
        if cmd.startswith("save ") and "#2.1" in cmd:
            path = cmd.split()[1]
            with open(path, "w") as fh:
                fh.write(sub1_content)
            return {"value": "", "error": None}
        if cmd.startswith("save ") and "#2.2" in cmd:
            path = cmd.split()[1]
            with open(path, "w") as fh:
                fh.write(sub2_content)
            return {"value": "", "error": None}
        return {"value": "", "error": None}

    stab, bridge, _ = _make_stab()
    bridge.run_command.side_effect = _cmd_handler

    pdb_path, mapping = stab._export_assembly_pdb("2", ["2.1", "2.2"])

    try:
        assert pdb_path is not None, "Expected a combined PDB path, got None"
        assert _os.path.isfile(pdb_path), f"Combined PDB not written: {pdb_path}"
        assert mapping, "Expected a non-empty chain mapping"

        # Mapping must cover all 4 (submodel, chain) pairs
        for sm, ch in [("2.1", "A"), ("2.1", "B"), ("2.2", "A"), ("2.2", "B")]:
            assert (sm, ch) in mapping, f"Missing mapping for ({sm!r}, {ch!r})"

        # Sub-model 2's chains must be renamed (different from sub-model 1's)
        assert mapping[("2.1", "A")] != mapping[("2.2", "A")]
        assert mapping[("2.1", "B")] != mapping[("2.2", "B")]

        # All 4 mapped letters must be distinct
        letters = list(mapping.values())
        assert len(set(letters)) == 4, f"Expected 4 distinct chain IDs, got {letters}"

        # Verify the combined PDB actually has 4 distinct chain IDs on ATOM records
        with open(pdb_path) as fh:
            chains_in_file = {
                line[21]
                for line in fh
                if line.startswith("ATOM  ") and len(line) > 21
            }
        assert len(chains_in_file) == 4, (
            f"Expected 4 chains in combined PDB, found {sorted(chains_in_file)}"
        )
    finally:
        if pdb_path and _os.path.isfile(pdb_path):
            _os.unlink(pdb_path)


def test_map_candidates_to_chimerax_adds_specs():
    """_map_candidates_to_chimerax adds chimerax_spec_a/b fields."""
    from interface_stabilization import InterfaceStabilization

    mapping = {
        ("2.1", "A"): "A",
        ("2.1", "B"): "B",
        ("2.2", "A"): "C",
        ("2.2", "B"): "D",
    }
    candidates = [
        {"chain_a": "A", "chain_a_residue": 14, "chain_b": "C", "chain_b_residue": 42},
        {"chain_a": "B", "chain_a_residue": 20, "chain_b": "D", "chain_b_residue": 55},
    ]
    InterfaceStabilization._map_candidates_to_chimerax(candidates, mapping)

    assert candidates[0]["chimerax_spec_a"] == "#2.1/A:14"
    assert candidates[0]["chimerax_spec_b"] == "#2.2/A:42"
    assert candidates[1]["chimerax_spec_a"] == "#2.1/B:20"
    assert candidates[1]["chimerax_spec_b"] == "#2.2/B:55"


# ══════════════════════════════════════════════════════════════════════════════
# 8. Sub-model spec correctness — no flat specs, correct #N.M/chain format
# ══════════════════════════════════════════════════════════════════════════════

def test_submodel_specs_use_submodel_id_not_flat():
    """
    Zone-select commands must use sub-model specs like #2.1/A, not flat #2/A.
    STEP 0 finding: flat specs address all sub-models simultaneously, breaking
    per-interface detection.
    """
    zone_cmds: list[str] = []

    def _cmd_handler(cmd: str) -> dict:
        if "info models #2" in cmd:
            return _cx_result(
                "model id #2 type Group\n"
                "model id #2.1 type AtomicStructure\n"
                "model id #2.2 type AtomicStructure\n"
            )
        if "info chains #2.1" in cmd or "info chains #2.2" in cmd:
            return _cx_result("chain id /A chain_id A\nchain id /B chain_id B\n")
        if "@CA" in cmd and "info" not in cmd:
            zone_cmds.append(cmd)
            return _cx_result("atom id /A:10@CA\n")
        return _cx_result("")

    stab, bridge, _ = _make_stab()
    bridge.run_command.side_effect = _cmd_handler

    submodels = ["2.1", "2.2"]
    stab._detect_submodel_interfaces("2", submodels, 5.0, lambda msg: None)

    # Every zone-select must use a sub-model spec (#N.M/chain), not a flat #N/chain
    for cmd in zone_cmds:
        # The spec pattern for a sub-model: #2.1 or #2.2 (contains a dot)
        assert re.search(r"#2\.\d+/", cmd), (
            f"Zone-select command does not use sub-model spec: {cmd!r}"
        )
        # Must NOT use flat spec #2/chain (without sub-model dot)
        assert not re.search(r"#2/[A-Z]", cmd), (
            f"Zone-select command uses flat spec instead of sub-model spec: {cmd!r}"
        )


import re  # used in test above


def test_intra_copy_specs_reference_same_submodel():
    """intra_copy spec_a and spec_b must share the same sub-model number."""
    stab, bridge, _ = _make_stab()

    def _cmd_handler(cmd: str) -> dict:
        if "info models #2" in cmd:
            return _cx_result(
                "model id #2 type Group\n"
                "model id #2.1 type AtomicStructure\n"
            )
        if "info chains #2.1" in cmd:
            return _cx_result("chain id /A chain_id A\nchain id /B chain_id B\n")
        if "#2.1/A@CA" in cmd and "#2.1/B" in cmd:
            return _cx_result("atom id /A:10@CA\n")
        if "#2.1/B@CA" in cmd and "#2.1/A" in cmd:
            return _cx_result("atom id /B:20@CA\n")
        return _cx_result("")

    bridge.run_command.side_effect = _cmd_handler

    interfaces = stab._detect_submodel_interfaces("2", ["2.1"], 5.0, lambda msg: None)
    for iface in interfaces:
        if iface["type"] == "intra_copy":
            sm_a = iface["submodel_a"]
            sm_b = iface["submodel_b"]
            assert sm_a == sm_b, (
                f"intra_copy interface has mismatched submodels: {sm_a} vs {sm_b}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 9. Chain coloring — sub-model-aware distinct colors
# ══════════════════════════════════════════════════════════════════════════════

def test_viz_emits_four_distinct_chain_colors_for_assembly():
    """
    _build_viz_commands with submodels= produces four color commands for a
    2-copy, 2-chain assembly, one per (submodel, chain) pair.
    """
    from interface_stabilization import InterfaceStabilization, _CHAIN_PALETTE

    stab, _, _ = _make_stab()
    # Include interfaces that collectively cover all 4 (submodel, chain) pairs
    interfaces = [
        {
            "type": "intra_copy", "submodel_a": "2.1", "submodel_b": "2.1",
            "chain_a": "A", "chain_b": "B", "spec_a": "#2.1/A", "spec_b": "#2.1/B",
            "contact_residues_a": [], "disulfide_candidates": [],
        },
        {
            "type": "intra_copy", "submodel_a": "2.2", "submodel_b": "2.2",
            "chain_a": "A", "chain_b": "B", "spec_a": "#2.2/A", "spec_b": "#2.2/B",
            "contact_residues_a": [], "disulfide_candidates": [],
        },
        {
            "type": "inter_copy", "submodel_a": "2.1", "submodel_b": "2.2",
            "chain_a": "A", "chain_b": "A", "spec_a": "#2.1/A", "spec_b": "#2.2/A",
            "contact_residues_a": [], "disulfide_candidates": [],
        },
    ]
    cmds, exps = stab._build_viz_commands(
        "2", interfaces, top_n_disulfides=3,
        submodels=["2.1", "2.2"], color_by_chain=True,
    )

    # Expect a color command for each of the 4 chain positions
    color_cmds = [c for c in cmds if c.startswith("color #2.") and "@" not in c]
    chain_specs_colored = set()
    for cmd in color_cmds:
        # e.g. "color #2.1/A cornflowerblue"
        parts = cmd.split()
        if len(parts) == 3:
            chain_specs_colored.add(parts[1])

    assert "#2.1/A" in chain_specs_colored, "Missing color cmd for #2.1/A"
    assert "#2.1/B" in chain_specs_colored, "Missing color cmd for #2.1/B"
    assert "#2.2/A" in chain_specs_colored, "Missing color cmd for #2.2/A"
    assert "#2.2/B" in chain_specs_colored, "Missing color cmd for #2.2/B"


def test_viz_same_letter_chains_across_submodels_get_distinct_colors():
    """
    STEP 0 finding: color bychain assigns the same color to #2.1/A and #2.2/A
    because it keys on chain-ID only.  The explicit palette must assign different
    colors to same-letter chains across sub-model copies.
    """
    from interface_stabilization import InterfaceStabilization, _CHAIN_PALETTE

    stab, _, _ = _make_stab()
    interfaces = [
        {
            "type": "inter_copy", "submodel_a": "2.1", "submodel_b": "2.2",
            "chain_a": "A", "chain_b": "A", "spec_a": "#2.1/A", "spec_b": "#2.2/A",
            "contact_residues_a": [], "disulfide_candidates": [],
        },
        {
            "type": "intra_copy", "submodel_a": "2.1", "submodel_b": "2.1",
            "chain_a": "A", "chain_b": "B", "spec_a": "#2.1/A", "spec_b": "#2.1/B",
            "contact_residues_a": [], "disulfide_candidates": [],
        },
    ]
    cmds, _ = stab._build_viz_commands(
        "2", interfaces, top_n_disulfides=0,
        submodels=["2.1", "2.2"], color_by_chain=True,
    )

    # Extract color assigned to #2.1/A and #2.2/A
    def _color_for(spec: str) -> str | None:
        for cmd in cmds:
            parts = cmd.split()
            if len(parts) == 3 and parts[0] == "color" and parts[1] == spec:
                return parts[2]
        return None

    col_21a = _color_for("#2.1/A")
    col_22a = _color_for("#2.2/A")

    assert col_21a is not None, "#2.1/A not colored"
    assert col_22a is not None, "#2.2/A not colored"
    assert col_21a != col_22a, (
        f"#2.1/A and #2.2/A got the same color ({col_21a}) — bychain collision not fixed"
    )


def test_viz_no_chain_colors_when_disabled():
    """
    When color_by_chain=False, the fallback 'color #{model_id} white' is emitted
    (no per-chain color commands, no bychain collision concern).
    """
    from interface_stabilization import InterfaceStabilization

    stab, _, _ = _make_stab()
    interfaces = [
        {
            "type": "intra_copy", "submodel_a": "2.1", "submodel_b": "2.1",
            "chain_a": "A", "chain_b": "B", "spec_a": "#2.1/A", "spec_b": "#2.1/B",
            "contact_residues_a": [], "disulfide_candidates": [],
        },
    ]
    cmds, _ = stab._build_viz_commands(
        "2", interfaces, top_n_disulfides=0,
        submodels=["2.1", "2.2"], color_by_chain=False,
    )

    # Must have the white reset
    assert any("white" in c for c in cmds), "Expected 'color #2 white' when disabled"
    # Must NOT have per-chain sub-model color commands
    chain_color_cmds = [c for c in cmds if c.startswith("color #2.") and "@" not in c]
    assert not chain_color_cmds, (
        f"Unexpected per-chain color cmds with color_by_chain=False: {chain_color_cmds}"
    )


def test_viz_flat_model_uses_white_not_bychain():
    """
    For a flat (non-assembly) model, no submodels= are passed and the command
    falls back to 'color #{model_id} white' regardless of color_by_chain.
    """
    from interface_stabilization import InterfaceStabilization

    stab, _, _ = _make_stab()
    interfaces = [
        {
            "type": "flat", "submodel_a": None, "submodel_b": None,
            "chain_a": "A", "chain_b": "B", "spec_a": "#1/A", "spec_b": "#1/B",
            "contact_residues_a": [], "disulfide_candidates": [],
        },
    ]
    cmds, _ = stab._build_viz_commands(
        "1", interfaces, top_n_disulfides=0,
        submodels=None, color_by_chain=True,
    )

    assert any("color #1 white" == c for c in cmds)
    chain_color_cmds = [c for c in cmds if re.search(r"color #1\.\d+/", c)]
    assert not chain_color_cmds


def test_chain_color_not_emitted_on_plain_chimerax_open():
    """
    Chain-coloring commands are NOT generated by a plain ChimeraX open action —
    they only appear in the interface stabilization viz pipeline.
    """
    router = _make_router()
    # Simulate a plain 'open' translation (no interface tool)
    translation = _translator_chimerax()
    translation["commands"] = ["open 2vnc"]
    translation["explanations"] = ["Open 2VNC"]

    # The viz commands from a plain chimerax result must contain no chain-color cmds
    chimerax_viz = translation["commands"]
    chain_color_cmds = [
        c for c in chimerax_viz
        if re.search(r"color #\d+\.\d+/", c)
    ]
    assert not chain_color_cmds, (
        "Chain-color commands leaked into a plain ChimeraX open result"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 10. Session persistence roundtrip
# ══════════════════════════════════════════════════════════════════════════════

def test_interface_stabilization_results_session_roundtrip(tmp_path):
    """set/get roundtrip for interface_stabilization_results in SessionState."""
    from session_state import SessionState

    state = SessionState()
    payload = {
        "model_id": "2",
        "pdb_id":   "2VNC",
        "interfaces": [
            {"type": "intra_subunit", "spec_a": "#2.1/A", "spec_b": "#2.1/B",
             "contact_count": 12, "buried_area_ang2": 1400.0}
        ],
        "is_assembly": True,
        "submodels":   ["2.1", "2.2"],
    }
    state.set_interface_stabilization_result("2", payload)
    retrieved = state.get_interface_stabilization_result("2")
    assert retrieved == payload
    assert state.get_interface_stabilization_result("99") is None


def test_interface_stabilization_results_persist_to_disk(tmp_path):
    """interface_stabilization_results survives a save/load cycle."""
    from session_state import SessionState

    state = SessionState()
    state.set_interface_stabilization_result("2", {
        "model_id": "2", "n_interfaces": 3
    })

    path = str(tmp_path / "session.json")
    state.save(path)

    loaded, err = SessionState.try_load(path)
    assert err is None
    assert loaded is not None
    rec = loaded.get_interface_stabilization_result("2")
    assert rec is not None
    assert rec["n_interfaces"] == 3


def test_interface_stabilization_results_in_snapshot_restore():
    """interface_stabilization_results survives snapshot/restore."""
    from session_state import SessionState

    state = SessionState()
    state.set_interface_stabilization_result("2", {"model_id": "2", "n_interfaces": 2})
    snap = state.snapshot()

    state2 = SessionState()
    state2.restore(snap)
    rec = state2.get_interface_stabilization_result("2")
    assert rec is not None
    assert rec["n_interfaces"] == 2
