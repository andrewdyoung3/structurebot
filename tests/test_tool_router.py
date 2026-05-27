"""
tests/test_tool_router.py
-------------------------
Routing tests for ToolRouter — specifically verifying that proline intent
phrases are correctly intercepted and dispatched to ProlineBridge rather
than the generic mutation_scan pipeline.

Four tests:
  1. test_proline_phrase_routes_to_proline_bridge
  2. test_proline_keyword_in_mutation_scan_redirects
  3. test_stabilise_proline_routes_correctly
  4. test_generic_mutation_scan_not_affected
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter, ToolStepResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_router() -> ToolRouter:
    """Return a ToolRouter with mocked bridge and session (no real ChimeraX needed)."""
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = {"1": {"name": "1HSG", "path": None}}
    return ToolRouter(bridge=mock_bridge, session=mock_session)


def _mutation_scan_translator_result(chain: str = "A") -> Dict[str, Any]:
    """Fake translator result for a generic mutation scan (what the LLM returns)."""
    return {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "high",
        "tools_needed":         ["mutation_scan"],
        "tool_inputs":          {
            "mutation_scan": {
                "model_id": "1",
                "chain":    chain,
                "focus":    "solubility",
            }
        },
    }


def _ok_proline_result() -> ToolStepResult:
    """Minimal successful ToolStepResult from ProlineBridge."""
    return ToolStepResult(
        tool    = "proline",
        success = True,
        data    = {"candidates": [], "count": 0},
        summary = "Proline scan chain A: 0 candidates.",
    )


def _ok_scan_result() -> ToolStepResult:
    """Minimal successful ToolStepResult from mutation_scan."""
    return ToolStepResult(
        tool    = "mutation_scan",
        success = True,
        data    = {"candidates": [], "count": 0},
        summary = "Mutation scan: 0 candidates.",
    )


# ════════════════════════════════════════════════════════════════════════════════
# 1. route() rewrites mutation_scan → proline on proline phrase
# ════════════════════════════════════════════════════════════════════════════════

def test_proline_phrase_routes_to_proline_bridge():
    """
    When user_input contains 'proline' and the translator returned
    tools_needed=['mutation_scan'], route() must rewrite to ['proline'].
    """
    router        = _make_router()
    translator_r  = _mutation_scan_translator_result()
    user_input    = "suggest proline mutations to stabilise chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "proline" in routed["tools_needed"], (
        f"Expected 'proline' in tools_needed, got {routed['tools_needed']}"
    )
    assert "mutation_scan" not in routed["tools_needed"], (
        f"mutation_scan should have been replaced, got {routed['tools_needed']}"
    )
    # Proline inputs should carry over model_id and chain from mutation_scan
    proline_inp = routed["tool_inputs"].get("proline", {})
    assert proline_inp.get("model_id") == "1"
    assert proline_inp.get("chain")    == "A"


# ════════════════════════════════════════════════════════════════════════════════
# 2. _dispatch_tool() guard redirects mutation_scan → proline
# ════════════════════════════════════════════════════════════════════════════════

def test_proline_keyword_in_mutation_scan_redirects():
    """
    Even if route() was not called with user_input, the _dispatch_tool()
    guard must redirect mutation_scan → _run_proline when the user_input
    contains a proline keyword.
    """
    router = _make_router()

    # Replace both runners with mocks so we can verify which was called
    mock_proline = MagicMock(return_value=_ok_proline_result())
    mock_scan    = MagicMock(return_value=_ok_scan_result())
    router._run_proline       = mock_proline
    router._run_mutation_scan = mock_scan

    inputs     = {"model_id": "1", "chain": "A"}
    user_input = "suggest proline mutations"

    result = router._dispatch_tool("mutation_scan", inputs, user_input=user_input)

    assert mock_proline.called, "_run_proline should have been called"
    assert not mock_scan.called, "_run_mutation_scan should NOT have been called"
    assert result.tool == "proline"


# ════════════════════════════════════════════════════════════════════════════════
# 3. Various stabilising-proline phrases all route correctly
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("user_input", [
    "suggest stabilising proline substitutions on chain A",
    "which residues should I change to proline for backbone stabilisation?",
    "scan for entropic stabilisation candidates",
    "rigidify the loop region with proline mutations",
    "phi angle analysis for proline scan",
])
def test_stabilise_proline_routes_correctly(user_input):
    """
    A variety of proline/backbone-stabilisation phrasings all cause route()
    to replace mutation_scan with proline.
    """
    router       = _make_router()
    translator_r = _mutation_scan_translator_result()

    routed = router.route(translator_r, user_input=user_input)

    assert "proline" in routed["tools_needed"], (
        f"Phrase {user_input!r} should route to proline; "
        f"got {routed['tools_needed']}"
    )
    assert "mutation_scan" not in routed["tools_needed"]


# ════════════════════════════════════════════════════════════════════════════════
# 4. Generic mutation scan is unaffected (no proline keywords)
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("user_input", [
    "suggest mutations to improve solubility",
    "what mutations would help reduce aggregation of chain A?",
    "run a mutation scan on this protein",
    "engineering candidates for better thermostability",
    "",   # empty string (e.g. route() called without user_input)
])
def test_generic_mutation_scan_not_affected(user_input):
    """
    Non-proline mutation requests must still route to mutation_scan,
    not to the proline bridge.
    """
    router       = _make_router()
    translator_r = _mutation_scan_translator_result()

    routed = router.route(translator_r, user_input=user_input)

    assert "mutation_scan" in routed["tools_needed"], (
        f"Phrase {user_input!r} should route to mutation_scan; "
        f"got {routed['tools_needed']}"
    )
    assert "proline" not in routed["tools_needed"], (
        f"Phrase {user_input!r} should NOT route to proline; "
        f"got {routed['tools_needed']}"
    )


# ════════════════════════════════════════════════════════════════════════════════
# 5. Glycan intent routes directly to the glycan bridge
# ════════════════════════════════════════════════════════════════════════════════

def _chimerax_translator_result(chain: str = "A") -> Dict[str, Any]:
    """
    Fake translator result when the LLM returns only chimerax (no special tool).
    This represents the typical wrong-routing case for glycan queries.
    """
    return {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "medium",
        "tools_needed":         ["chimerax"],
        "tool_inputs":          {},
    }


def _clarification_translator_result() -> Dict[str, Any]:
    """
    Fake translator result that includes a clarification question.
    Represents the crash-triggering case: translator is unsure and asks
    a follow-up whose answer would be re-sent to translate(), causing
    stop_reason='refusal'.
    """
    return {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": "What type of glycosylation analysis would you like?",
        "confidence":           "low",
        "tools_needed":         ["chimerax"],
        "tool_inputs":          {},
    }


def test_glycan_phrase_routes_to_glycan_bridge():
    """
    'suggest glycosylation sites on chain A' must route to the glycan tool,
    not trigger a clarification question.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "suggest glycosylation sites on chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "glycan" in routed["tools_needed"], (
        f"Expected 'glycan' in tools_needed, got {routed['tools_needed']}"
    )
    assert "chimerax" not in routed["tools_needed"], (
        f"'chimerax' should be replaced by 'glycan', got {routed['tools_needed']}"
    )
    # Clarification must be suppressed so main.py never asks the user a question
    assert routed.get("clarification_needed") is None, (
        "clarification_needed must be None for glycan intent"
    )
    # Glycan tool_inputs must be populated
    glycan_inp = routed["tool_inputs"].get("glycan", {})
    assert glycan_inp.get("model_id") is not None


def test_glycan_sequon_phrase_routes_correctly():
    """
    'identify NXS sequons on chain A' must route to the glycan tool.
    The translator might emit mutation_scan for this phrase; the glycan
    guard should redirect it.
    """
    router       = _make_router()
    translator_r = _mutation_scan_translator_result()
    user_input   = "identify NXS sequons on chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "glycan" in routed["tools_needed"], (
        f"NXS sequon phrase should route to glycan; got {routed['tools_needed']}"
    )
    assert "mutation_scan" not in routed["tools_needed"]
    assert routed.get("clarification_needed") is None


def test_glycan_engineering_phrase_routes_correctly():
    """
    'suggest glycoengineering positions on chain A' must route to the
    glycan tool.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "suggest glycoengineering positions on chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "glycan" in routed["tools_needed"], (
        f"Glycoengineering phrase should route to glycan; "
        f"got {routed['tools_needed']}"
    )
    assert routed.get("clarification_needed") is None


# ════════════════════════════════════════════════════════════════════════════════
# 6. Clarification refusal crash is prevented by clearing clarification_needed
# ════════════════════════════════════════════════════════════════════════════════

def test_clarification_refusal_handled_gracefully():
    """
    When the translator returns clarification_needed for a glycan phrase,
    route() must:
      (a) clear the flag (so main.py never enters the clarification loop), AND
      (b) rewrite tools_needed to ['glycan'].

    This is the primary safeguard against the stop_reason='refusal' crash:
    if clarification_needed is None, the retranslation call that raises
    ValueError("stop_reason='refusal'") is never made.
    """
    router       = _make_router()
    translator_r = _clarification_translator_result()
    user_input   = "suggest glycosylation sites on chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert routed.get("clarification_needed") is None, (
        "route() must clear clarification_needed for glycan intent; "
        "a non-None value triggers the clarification loop, which can crash "
        "with ValueError(stop_reason='refusal') when the answer is retranslated"
    )
    assert "glycan" in routed["tools_needed"], (
        f"tools_needed must contain 'glycan', got {routed['tools_needed']}"
    )
    # Also verify _detect_glycan_intent works for the canonical phrase
    assert ToolRouter._detect_glycan_intent(user_input), (
        "_detect_glycan_intent must return True for glycan site suggestions"
    )


# ════════════════════════════════════════════════════════════════════════════════
# 7. Generic solubility mutation scan still unaffected by glycan routing
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("user_input", [
    "suggest mutations to improve solubility of chain A",
    "what mutations would reduce aggregation?",
    "run a full mutation scan on chain B",
])
def test_solubility_mutation_scan_unaffected_by_glycan_routing(user_input):
    """
    'suggest mutations to improve solubility' and similar non-glycan phrases
    must still route to mutation_scan — glycan detection must not fire.
    """
    router       = _make_router()
    translator_r = _mutation_scan_translator_result()

    routed = router.route(translator_r, user_input=user_input)

    assert "mutation_scan" in routed["tools_needed"], (
        f"Phrase {user_input!r} should route to mutation_scan; "
        f"got {routed['tools_needed']}"
    )
    assert "glycan" not in routed["tools_needed"], (
        f"Phrase {user_input!r} should NOT trigger glycan routing; "
        f"got {routed['tools_needed']}"
    )
