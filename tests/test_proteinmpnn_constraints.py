"""
End-to-end constraint/scope path for ProteinMPNN (translator tool_inputs →
tool_router._run_proteinmpnn → bridge.analyze → _resolve_design_constraints →
ProteinMPNN flags). Asserts a CORRECT request is HONOURED deterministically:
exclude_cys → omit_AAs C, solubility → hydrophilic bias, scope=20-30 →
design only those (fix the complement, never whole-chain), and a restricted
design that resolves to nothing ERRORS. Subprocess/bridge mocked in CI; the
constraint resolver runs against a real small PDB fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_harness as eh
from tool_router import ToolRouter, ToolStepResult, _HYDROPHILIC_AAS
import proteinmpnn_bridge as pb


# ── a real (tiny) single-chain PDB: chain A residues 1..40, a CYS at 10 ──────────
@pytest.fixture
def mini_pdb(tmp_path) -> str:
    lines = []
    serial = 1
    for resnum in range(1, 41):
        resname = "CYS" if resnum == 10 else "ALA"
        x = 10.0 + resnum * 3.8       # strung-out CA trace
        lines.append(
            f"ATOM  {serial:>5}  CA  {resname} A{resnum:>4}    "
            f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C"
        )
        serial += 1
    lines.append("TER")
    lines.append("END")
    p = tmp_path / "mini.pdb"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _make_router(mini_pdb: str, selected: List[int] = None) -> ToolRouter:
    bridge = MagicMock()
    session = MagicMock()
    session.structures = {"1": {"name": "MINI", "path": mini_pdb}}
    session.get_interface_residues.return_value = None
    session.get_protected_residues_for_chain.return_value = []
    router = ToolRouter(bridge=bridge, session=session)
    router._ensure_pdb_file = lambda *a, **k: mini_pdb
    router._first_model_id = lambda *a, **k: "1"
    router._read_selected_residues = lambda *a, **k: list(selected or [])
    return router


def _capture_analyze(router):
    """Patch the ProteinMPNN bridge so analyze() captures full_inputs instead of
    running inference; returns the captured-inputs holder."""
    holder: Dict[str, Any] = {}

    class _SpyBridge:
        def analyze(self, full_inputs, session=None):
            holder.update(full_inputs)
            return ToolStepResult(tool="proteinmpnn", success=True,
                                  data={"sequences": []}, summary="ok")

    router._get_proteinmpnn_bridge = lambda: _SpyBridge()
    return holder


# ════════════════════════════════════════════════════════════════════════════════
#  DISPATCH boundary: translator tool_inputs → bridge full_inputs (field mapping)
# ════════════════════════════════════════════════════════════════════════════════
def test_exclude_cys_maps_to_omit_C(mini_pdb):
    router = _make_router(mini_pdb)
    holder = _capture_analyze(router)
    router._run_proteinmpnn({"model_id": "1", "chain": "A",
                             "exclude_amino_acids": ["C"]})
    assert holder["omit_aas"] == "C"


def test_exclude_pro_maps_to_omit_P(mini_pdb):
    router = _make_router(mini_pdb)
    holder = _capture_analyze(router)
    router._run_proteinmpnn({"model_id": "1", "chain": "A",
                             "exclude_amino_acids": ["P"]})
    assert holder["omit_aas"] == "P"


def test_solubility_bias_toward_soluble_is_honoured(mini_pdb):
    # the corpus's own wording: bias_toward:"soluble" (NOT the word "hydrophilic")
    router = _make_router(mini_pdb)
    holder = _capture_analyze(router)
    router._run_proteinmpnn({"model_id": "1", "chain": "A",
                             "bias_toward": "soluble"})
    assert set(holder["bias_aas"]) == set(_HYDROPHILIC_AAS), \
        "solubility must map to the polar/charged hydrophilic bias set"


def test_explicit_bias_amino_acids_passthrough(mini_pdb):
    router = _make_router(mini_pdb)
    holder = _capture_analyze(router)
    router._run_proteinmpnn({"model_id": "1", "chain": "A",
                             "bias_amino_acids": ["D", "E", "K"]})
    assert set(holder["bias_aas"]) == {"D", "E", "K"}


def test_scope_design_positions_passthrough(mini_pdb):
    router = _make_router(mini_pdb)
    holder = _capture_analyze(router)
    router._run_proteinmpnn({"model_id": "1", "chain": "A",
                             "design_scope": "selected",
                             "design_positions": list(range(20, 31))})
    assert set(holder["design_positions"]) == set(range(20, 31))
    assert holder["chain_id"] == "A"


# ════════════════════════════════════════════════════════════════════════════════
#  CONSTRAINT RESOLVER (real PDB): scope → fix the complement; errors honestly
# ════════════════════════════════════════════════════════════════════════════════
def test_resolve_scope_fixes_complement(mini_pdb):
    bridge = pb.ProteinMPNNBridge()
    fixed, omit = bridge._resolve_design_constraints(
        mini_pdb, "A",
        {"design_positions": list(range(20, 31)), "omit_aas": "C"},
    )
    # 40 residues, design 20..30 (11), preserve native CYS@10 → fixed = everything
    # except 20..30 (fixed indices are 1-based chain order == resnum here).
    assert omit == "C"
    assert set(range(20, 31)).isdisjoint(set(fixed)), "designable range must NOT be fixed"
    assert 10 in fixed, "native cysteine must be held fixed"
    assert len(fixed) == 40 - 11           # complement of the designable set


def test_resolve_empty_scope_errors_not_whole_chain(mini_pdb):
    bridge = pb.ProteinMPNNBridge()
    with pytest.raises(ValueError):
        # residues 100-110 don't exist on this 40-residue chain → nothing designable
        bridge._resolve_design_constraints(
            mini_pdb, "A", {"design_positions": [100, 101, 102]},
        )


def test_resolve_interface_no_partner_errors(mini_pdb):
    bridge = pb.ProteinMPNNBridge()
    with pytest.raises(ValueError):
        # interface design on a single-chain PDB → no interface positions
        bridge._resolve_design_constraints(
            mini_pdb, "A", {"interface_design": True},
        )


# ════════════════════════════════════════════════════════════════════════════════
#  INTERFACE without a live selection → BioPython fallback, never whole-chain
# ════════════════════════════════════════════════════════════════════════════════
def test_interface_request_without_selection_falls_back_not_whole_chain(mini_pdb):
    # design_scope "interface" but the live selection is empty → must set
    # interface_design (deterministic coordinate computation), NOT whole-chain.
    router = _make_router(mini_pdb, selected=[])
    holder = _capture_analyze(router)
    router._run_proteinmpnn({"model_id": "1", "chain": "A",
                             "design_scope": "interface", "partner_chain": "B"})
    assert holder["interface_design"] is True
    assert holder["design_positions"] in (None, [], )  # resolved downstream, not whole-chain


# ════════════════════════════════════════════════════════════════════════════════
#  END-TO-END vs the eval-harness dispatch GOLDS (s3 scope, s4 soluble/no-cys)
# ════════════════════════════════════════════════════════════════════════════════
def _model_output(tool_inputs: Dict[str, Any]) -> Dict[str, Any]:
    return {"commands": [], "explanations": [], "warnings": [],
            "clarification_needed": None, "confidence": "high",
            "tools_needed": ["proteinmpnn"], "tool_inputs": {"proteinmpnn": tool_inputs}}


def test_matches_sample_dispatch_golds(mini_pdb):
    cases = {c.id: c for c in eh.SAMPLE_CASES}

    # s3: scope 20-30 — the model emits design_positions; the dispatch gold passes
    # AND the bridge receives the scoped set.
    s3_out = _model_output({"chain": "A", "design_scope": "selected",
                            "design_positions": list(range(20, 31))})
    assert eh.score_functionality(cases["s3_sel_scope"], s3_out).passed
    assert eh.score_accuracy(cases["s3_sel_scope"], s3_out).passed
    router = _make_router(mini_pdb)
    holder = _capture_analyze(router)
    router._run_proteinmpnn(s3_out["tool_inputs"]["proteinmpnn"])
    assert set(holder["design_positions"]) == set(range(20, 31))

    # s4: exclude_cys + solubility — bridge honours both
    s4_out = _model_output({"chain": "A", "exclude_amino_acids": ["C"],
                            "bias_toward": "soluble"})
    assert eh.score_accuracy(cases["s4_mpnn_soluble"], s4_out).passed
    holder2 = _capture_analyze(router)
    router._run_proteinmpnn(s4_out["tool_inputs"]["proteinmpnn"])
    assert holder2["omit_aas"] == "C"
    assert set(holder2["bias_aas"]) == set(_HYDROPHILIC_AAS)
