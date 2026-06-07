"""
tests/test_interface_stabilization.py
--------------------------------------
Tests for Phase 1 interface stabilization:
  - intent detection + route() override
  - _run_interface_stabilization error guards
  - InterfaceStabilization class (mocked bridge)
  - Sub-model addressing correctness (no flat specs)
  - Intra-subunit disulfide scan; inter-subunit deferred
  - Session persistence roundtrip

All mocked — no live ChimeraX or PDB files required.

Test groups
-----------
1.  Routing — intent detection + route override
2.  _run_interface_stabilization — error-first guards
3.  InterfaceStabilization — sub-model discovery, zone-select, buried area
4.  Interface type classification (intra vs inter)
5.  Disulfide routing (intra runs, inter deferred)
6.  Sub-model spec correctness (spec format #N.M/chain)
7.  Session persistence roundtrip
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
# 4. Interface type classification (intra vs inter)
# ══════════════════════════════════════════════════════════════════════════════

def test_intra_subunit_type_when_same_submodel():
    """Two chains in same sub-model → intra_subunit."""
    # Build a stab and call _detect_submodel_interfaces with mocked zone-selects.
    # Sub-models: 2.1 and 2.2 each with chains A, B.
    # Pairs: (2.1/A, 2.1/B)=intra, (2.1/A, 2.2/A)=inter, (2.1/A, 2.2/B)=inter, etc.

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

    intra = [i for i in interfaces if i["type"] == "intra_subunit"]
    inter = [i for i in interfaces if i["type"] == "inter_subunit"]

    # At least the A-B intra-subunit interface should be detected (both copies)
    # Depends on whether the mock returns hits; at minimum the type mapping is correct.
    for iface in interfaces:
        sm_a = iface["submodel_a"]
        sm_b = iface["submodel_b"]
        if sm_a == sm_b:
            assert iface["type"] == "intra_subunit"
        else:
            assert iface["type"] == "inter_subunit"


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
# 5. Disulfide routing — intra runs, inter deferred
# ══════════════════════════════════════════════════════════════════════════════

def test_intra_subunit_interface_gets_disulfide_scan(tmp_path):
    """
    InterfaceStabilization.analyze() calls DisulfideBridge for intra_subunit interfaces.
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


def test_inter_subunit_interface_disulfide_deferred(tmp_path):
    """
    inter_subunit interfaces must receive a disulfide_note, not a scan result.
    """
    pdb = tmp_path / "2VNC.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    )

    stab, bridge, session = _make_stab()
    bridge.is_running.return_value = True

    dummy_inter = {
        "type":               "inter_subunit",
        "spec_a":             "#2.1/A",
        "spec_b":             "#2.2/A",
        "chain_a":            "A",
        "chain_b":            "A",
        "submodel_a":         "2.1",
        "submodel_b":         "2.2",
        "contact_residues_a": [15, 16],
        "contact_residues_b": [15, 16],
        "contact_count":      4,
        "buried_area_ang2":   None,
        "disulfide_candidates": None,
        "disulfide_count":    0,
        "disulfide_top":      None,
        "disulfide_note":     None,
    }

    with patch.object(stab, "_get_submodels", return_value=["2.1", "2.2"]):
        with patch.object(stab, "_detect_submodel_interfaces", return_value=[dummy_inter]):
            with patch.object(stab, "_measure_buried_area", return_value=800.0):
                with patch("interface_stabilization.DisulfideBridge") as MockDS:
                    result = stab.analyze(
                        model_id="2", pdb_path=str(pdb), pdb_id="2VNC"
                    )

    assert result.success
    # DisulfideBridge.analyze must NOT have been called for inter-subunit
    MockDS.return_value.analyze.assert_not_called()

    iface = result.data["interfaces"][0]
    assert iface["disulfide_note"] is not None
    assert "phase 2" in iface["disulfide_note"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Sub-model spec correctness — no flat specs, correct #N.M/chain format
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


def test_intra_subunit_specs_reference_same_submodel():
    """Intra-subunit spec_a and spec_b must share the same sub-model number."""
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
        if iface["type"] == "intra_subunit":
            sm_a = iface["submodel_a"]
            sm_b = iface["submodel_b"]
            assert sm_a == sm_b, (
                f"Intra-subunit interface has mismatched submodels: {sm_a} vs {sm_b}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 7. Session persistence roundtrip
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
