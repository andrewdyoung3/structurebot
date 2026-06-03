"""
Pinned intent-override PRECEDENCE matrix for ToolRouter.route().

Deterministic router behaviour (no LLM): when a single prompt legitimately
triggers two intent-override keyword sets, the HIGHER-precedence override must
win; a tool keyword mentioned only in passing (a distractor) must NOT capture
routing from the dominant intent; and a compound request must keep its primary
tool. These are PINNED so a future keyword edit cannot silently reorder the
documented §2 chain:

  validate_design → validate_ddg → colabfold → proline → mpnn_esmfold →
  glycan_positions → netnglyc → glycan → salt_bridge → cavity →
  double_mutant → mutation_scan (fallback, last).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tool_router import ToolRouter


def _make_router(session_has_mpnn: bool = False) -> ToolRouter:
    bridge = MagicMock()
    session = MagicMock()
    session.structures = {"1": {"name": "1HSG", "path": None}}
    session.get_proteinmpnn_result.return_value = (
        {"designs": []} if session_has_mpnn else None
    )
    return ToolRouter(bridge=bridge, session=session)


def _tr(tools: List[str], chain: str = "A") -> Dict[str, Any]:
    return {
        "commands": [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "high",
        "tools_needed": list(tools),
        "tool_inputs": {t: {"model_id": "1", "chain": chain} for t in tools},
    }


def _route(translator_tools: List[str], user_input: str,
           session_has_mpnn: bool = False) -> List[str]:
    router = _make_router(session_has_mpnn=session_has_mpnn)
    return router.route(_tr(translator_tools), user_input=user_input)["tools_needed"]


# ════════════════════════════════════════════════════════════════════════════════
#  (1) ORDERED PRECEDENCE MATRIX — higher wins on a dual-trigger prompt
#  (prompt, translator_tools, higher_tool, lower_tool, session_has_mpnn)
# ════════════════════════════════════════════════════════════════════════════════
PRECEDENCE = [
    ("Thoroughly validate the design and confirm stability for chain A.",
     ["chimerax"], "validate_design", "validate_ddg", False),
    ("Run a multi-trajectory ddG for chain A and fold it with AlphaFold.",
     ["chimerax"], "validate_ddg", "colabfold", False),
    ("Fold chain A with ColabFold and then suggest proline mutations.",
     ["mutation_scan"], "colabfold", "proline", False),
    ("Fold chain A with AlphaFold and look for buried cavities.",
     ["chimerax"], "colabfold", "cavity", False),
    ("Suggest proline mutations for chain A and scan for glycans.",
     ["mutation_scan"], "proline", "glycan", False),
    ("Proline mutations for chain A and also fill internal cavities.",
     ["mutation_scan"], "proline", "cavity", False),
    ("Suggest proline mutations to improve solubility of chain A.",
     ["mutation_scan"], "proline", "mutation_scan", False),
    ("Fold design for chain A and find salt bridges.",
     ["proteinmpnn"], "mpnn_esmfold", "salt_bridge", False),
    ("Run esmfold mpnn on the design and check for cavities.",
     ["proteinmpnn"], "mpnn_esmfold", "cavity", False),
    ("Find glycan candidates and the glycosylation on chain A.",
     ["chimerax"], "glycan_positions", "glycan", False),
    ("Give the OST recognition score and the N-glycans on chain A.",
     ["chimerax"], "netnglyc", "glycan", False),
    ("Find N-glycans and salt bridges on chain A.",
     ["chimerax"], "glycan", "salt_bridge", False),
    ("Find salt bridges and buried cavities on chain A.",
     ["chimerax"], "salt_bridge", "cavity", False),
    ("Find internal cavities and suggest double mutant combinations.",
     ["chimerax"], "cavity", "double_mutant", False),
    ("Suggest double mutant combinations to improve solubility of chain A.",
     ["chimerax"], "double_mutant", "mutation_scan", False),
]


@pytest.mark.parametrize("prompt,trans,higher,lower,has_mpnn", PRECEDENCE)
def test_precedence_higher_wins(prompt, trans, higher, lower, has_mpnn):
    tools = _route(trans, prompt, session_has_mpnn=has_mpnn)
    assert higher in tools, f"{prompt!r} → {tools}, expected higher {higher}"
    assert lower not in tools, f"{prompt!r} → {tools}, lower {lower} must not capture"


# ════════════════════════════════════════════════════════════════════════════════
#  (2) DISTRACTOR NON-CAPTURE — a keyword for X appears but the intent is Y
#  (prompt, translator_tools, must_be_present, must_be_absent)
# ════════════════════════════════════════════════════════════════════════════════
DISTRACTORS = [
    # solubility/aggregation wording must NOT pull a real proteinmpnn redesign
    # into camsol/mutation_scan (the collision the new eval corpus forbids).
    ("Redesign chain A to improve its solubility with no cysteines.",
     ["proteinmpnn"], "proteinmpnn", "mutation_scan"),
    ("Redesign chain A to reduce aggregation.",
     ["proteinmpnn"], "proteinmpnn", "mutation_scan"),
    # a passing "proline" mention in a pure viz command must not route proline
    ("Color the proline residues on chain A red.",
     ["chimerax"], "chimerax", "proline"),
    # "avoid disulfides" in a redesign must not route the disulfide analyser
    ("Redesign chain A while avoiding new disulfides.",
     ["proteinmpnn"], "proteinmpnn", "disulfide"),
    # "binding pocket" was deliberately removed from the cavity keywords — a
    # binding-pocket viz request must not capture cavity detection.
    ("Color the binding pocket residues on chain A.",
     ["chimerax"], "chimerax", "cavity"),
    # "avoiding" must never trigger cavity (the classic bare-"void" trap)
    ("Redesign chain A avoiding buried positions.",
     ["proteinmpnn"], "proteinmpnn", "cavity"),
]


@pytest.mark.parametrize("prompt,trans,present,absent", DISTRACTORS)
def test_distractor_non_capture(prompt, trans, present, absent):
    tools = _route(trans, prompt)
    assert present in tools, f"{prompt!r} → {tools}, expected {present}"
    assert absent not in tools, f"{prompt!r} → {tools}, distractor {absent} captured"


# ════════════════════════════════════════════════════════════════════════════════
#  (3) COMPOUND multi_tool — the primary tools survive (no override collapses them)
# ════════════════════════════════════════════════════════════════════════════════
COMPOUND = [
    ("Run a CamSol solubility scan on chain A and also score it for ESM conservation.",
     ["camsol", "esm"], ["camsol", "esm"]),
    ("Detect the biological assembly and find disulfide candidates.",
     ["assembly_analyser", "disulfide"], ["assembly_analyser", "disulfide"]),
]


@pytest.mark.parametrize("prompt,trans,expected", COMPOUND)
def test_compound_primary_tools_survive(prompt, trans, expected):
    tools = _route(trans, prompt)
    for t in expected:
        assert t in tools, f"{prompt!r} → {tools}, expected {t} preserved"
