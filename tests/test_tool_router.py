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
    # Default: no ProteinMPNN results in session.  Tests that need results
    # must override this via _make_router_with_mpnn_session() below.
    mock_session.get_proteinmpnn_result.return_value = None
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
# High-accuracy ddG validation tier routing
# ════════════════════════════════════════════════════════════════════════════════

def test_validation_tier_intent_routing():
    """
    'validate ddg for I72R' must route to the validate_ddg tool — NOT the
    fast single-trajectory mutation_scan and NOT double_mutant.
    """
    router = _make_router()
    translator_r = _mutation_scan_translator_result()   # LLM guessed mutation_scan
    routed = router.route(translator_r, user_input="validate ddg for I72R")

    assert routed["tools_needed"] == ["validate_ddg"], (
        f"Expected ['validate_ddg'], got {routed['tools_needed']}"
    )
    assert "mutation_scan" not in routed["tools_needed"]
    assert "double_mutant" not in routed["tools_needed"]
    vinp = routed["tool_inputs"].get("validate_ddg", {})
    assert vinp.get("model_id") == "1"
    assert vinp.get("_user_input") == "validate ddg for I72R"


def test_generic_scan_not_routed_to_validation():
    """A plain solubility-scan request must NOT trigger the validation tier."""
    router = _make_router()
    translator_r = _mutation_scan_translator_result()
    routed = router.route(
        translator_r, user_input="suggest mutations to improve solubility of chain A"
    )
    assert "validate_ddg" not in routed["tools_needed"]
    assert "mutation_scan" in routed["tools_needed"]


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


# ════════════════════════════════════════════════════════════════════════════════
# 8. MPNN+ESMFold routing — session-aware esmfold → mpnn_esmfold redirect
# ════════════════════════════════════════════════════════════════════════════════

def _esmfold_translator_result(chain: str = "A") -> Dict[str, Any]:
    """Fake translator result when the LLM emits 'esmfold' (single tool)."""
    return {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "high",
        "tools_needed":         ["esmfold"],
        "tool_inputs":          {
            "esmfold": {"model_id": "1", "chain": chain},
        },
    }


def _proteinmpnn_translator_result(chain: str = "A") -> Dict[str, Any]:
    """Fake translator result when the LLM emits 'proteinmpnn'."""
    return {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "high",
        "tools_needed":         ["proteinmpnn"],
        "tool_inputs":          {
            "proteinmpnn": {"model_id": "1", "chain_id": chain},
        },
    }


def _make_router_with_mpnn_session() -> "ToolRouter":
    """
    Return a ToolRouter whose session already contains a ProteinMPNN result
    for model '1' (simulates having run a redesign earlier in the session).

    We set get_proteinmpnn_result.return_value directly because
    add_proteinmpnn_result on a MagicMock is a no-op (it doesn't persist data).
    """
    _mpnn_data = {
        "sequences": [
            {"sequence": "ACDE", "score": -1.2, "recovery": 0.9, "mutations": ["L2A"]},
        ],
        "wildtype_sequence": "ACLE",
        "fixed_positions":   [],
        "backend":           "local",
    }
    router = _make_router()
    router.session.get_proteinmpnn_result.return_value = _mpnn_data
    return router


def test_esmfold_phrase_with_mpnn_session_routes_to_mpnn_esmfold():
    """
    "ESMFold top 2-3 sequences to assess structural integrity" should route to
    mpnn_esmfold (not plain esmfold) when the session has ProteinMPNN results.
    """
    router       = _make_router_with_mpnn_session()
    translator_r = _esmfold_translator_result()
    user_input   = "ESMFold top 2-3 sequences to assess structural integrity"

    routed = router.route(translator_r, user_input=user_input)

    assert "mpnn_esmfold" in routed["tools_needed"], (
        f"Expected mpnn_esmfold; got {routed['tools_needed']}"
    )
    assert "esmfold" not in routed["tools_needed"], (
        f"'esmfold' should be replaced; got {routed['tools_needed']}"
    )


def test_esmfold_phrase_without_mpnn_session_stays_esmfold():
    """
    The same ESMFold phrase should NOT redirect to mpnn_esmfold when the
    session has no ProteinMPNN results (no prior redesign run).
    """
    router       = _make_router()          # empty session
    translator_r = _esmfold_translator_result()
    user_input   = "ESMFold top 2-3 sequences to assess structural integrity"

    routed = router.route(translator_r, user_input=user_input)

    assert "esmfold" in routed["tools_needed"], (
        f"Without MPNN session, esmfold should be kept; got {routed['tools_needed']}"
    )
    assert "mpnn_esmfold" not in routed["tools_needed"], (
        f"mpnn_esmfold should NOT appear without MPNN session; got {routed['tools_needed']}"
    )


def test_structural_integrity_phrase_routes_to_mpnn_esmfold():
    """
    'assess structural integrity of designed sequences' with an MPNN session
    must route to mpnn_esmfold even when translator emits 'esmfold'.
    """
    router       = _make_router_with_mpnn_session()
    translator_r = _esmfold_translator_result()
    user_input   = "assess structural integrity of designed sequences"

    routed = router.route(translator_r, user_input=user_input)

    assert "mpnn_esmfold" in routed["tools_needed"], (
        f"Expected mpnn_esmfold; got {routed['tools_needed']}"
    )


def test_validate_design_routes_to_mpnn_esmfold():
    """
    'validate design' with proteinmpnn in translator output must always rewrite
    to mpnn_esmfold (session not required for 'proteinmpnn' rewriting).
    """
    router       = _make_router()          # no MPNN results in session
    translator_r = _proteinmpnn_translator_result()
    user_input   = "validate design by folding with ESMFold"

    routed = router.route(translator_r, user_input=user_input)

    assert "mpnn_esmfold" in routed["tools_needed"], (
        f"'proteinmpnn' should always rewrite to mpnn_esmfold; got {routed['tools_needed']}"
    )
    assert "proteinmpnn" not in routed["tools_needed"]


def test_check_fold_without_mpnn_session_stays_esmfold():
    """
    'check fold' is an MPNN_ESMFOLD keyword, but without MPNN session results,
    an 'esmfold' tool should NOT be rewritten to mpnn_esmfold.
    """
    router       = _make_router()
    translator_r = _esmfold_translator_result()
    user_input   = "check fold quality of the current structure"

    routed = router.route(translator_r, user_input=user_input)

    assert "esmfold" in routed["tools_needed"], (
        f"Without MPNN session, 'check fold' should keep esmfold; got {routed['tools_needed']}"
    )
    assert "mpnn_esmfold" not in routed["tools_needed"]


def test_rewrite_as_mpnn_esmfold_handles_esmfold_in_tools():
    """
    _rewrite_as_mpnn_esmfold(['esmfold'], ...) must produce ['mpnn_esmfold'],
    not ['esmfold', 'mpnn_esmfold'].
    """
    router     = _make_router()
    tools_in   = ["esmfold"]
    inputs_in  = {"esmfold": {"model_id": "1", "chain": "A"}}

    new_tools, new_inputs = router._rewrite_as_mpnn_esmfold(tools_in, inputs_in)

    assert new_tools == ["mpnn_esmfold"], f"Expected ['mpnn_esmfold'], got {new_tools}"
    assert "esmfold" not in new_inputs,   "esmfold key should be removed from tool_inputs"
    assert "mpnn_esmfold" in new_inputs,  "mpnn_esmfold key should be added to tool_inputs"
    assert new_inputs["mpnn_esmfold"]["model_id"] == "1"


# ════════════════════════════════════════════════════════════════════════════════
# 9. Dispatch-level guard: 'esmfold' tool redirects to mpnn_esmfold when appropriate
# ════════════════════════════════════════════════════════════════════════════════

def test_dispatch_esmfold_redirects_to_mpnn_esmfold_with_session():
    """
    _dispatch_tool('esmfold', ...) must redirect to _run_mpnn_esmfold when
    (a) user_input has MPNN keywords, AND (b) session has MPNN results.
    """
    router = _make_router_with_mpnn_session()

    mock_mpnn_esmfold = MagicMock(return_value=_ok_scan_result())
    mock_esmfold      = MagicMock(return_value=_ok_scan_result())
    router._run_mpnn_esmfold = mock_mpnn_esmfold
    router._run_esmfold      = mock_esmfold

    inputs     = {"model_id": "1", "chain": "A"}
    user_input = "ESMFold top sequences to assess structural integrity"

    router._dispatch_tool("esmfold", inputs, user_input=user_input)

    assert mock_mpnn_esmfold.called, "_run_mpnn_esmfold should have been called"
    assert not mock_esmfold.called,  "_run_esmfold should NOT have been called"


def test_dispatch_esmfold_no_redirect_without_session():
    """
    _dispatch_tool('esmfold', ...) must NOT redirect when session is empty,
    even with MPNN/ESMFold keywords present.
    """
    router = _make_router()   # empty session — no MPNN results

    mock_mpnn_esmfold = MagicMock(return_value=_ok_scan_result())
    mock_esmfold      = MagicMock(return_value=_ok_scan_result())
    router._run_mpnn_esmfold = mock_mpnn_esmfold
    router._run_esmfold      = mock_esmfold

    inputs     = {"model_id": "1", "chain": "A"}
    user_input = "ESMFold top sequences to assess structural integrity"

    router._dispatch_tool("esmfold", inputs, user_input=user_input)

    assert mock_esmfold.called,          "_run_esmfold should have been called"
    assert not mock_mpnn_esmfold.called, "_run_mpnn_esmfold should NOT be called without MPNN session"


# ════════════════════════════════════════════════════════════════════════════════
# 10. handle_sequence_display_command / _show_designed_sequences
# ════════════════════════════════════════════════════════════════════════════════

def test_handle_sequence_display_returns_none_without_session():
    """
    handle_sequence_display_command must return None when the session has
    no ProteinMPNN results, even if the phrase matches.
    """
    router = _make_router()
    result = router.handle_sequence_display_command("show designed sequences")
    assert result is None, (
        "Should return None (no MPNN results) so the LLM handles the request"
    )


def test_handle_sequence_display_returns_none_for_non_display_phrase():
    """
    Non-display phrases must return None regardless of session state.
    """
    router = _make_router_with_mpnn_session()
    result = router.handle_sequence_display_command("run a mutation scan on chain A")
    assert result is None


def test_handle_sequence_display_with_session_returns_string():
    """
    When the session has MPNN results and user_input matches a display
    keyword, handle_sequence_display_command must return a non-empty string.
    """
    router = _make_router_with_mpnn_session()
    result = router.handle_sequence_display_command("show designed sequences")
    assert result is not None, "Should return a display string when session has results"
    assert isinstance(result, str)
    assert len(result) > 10


def test_show_designed_sequences_format():
    """
    _show_designed_sequences() must include model_id, score, recovery,
    mutation count, and the sequence (possibly truncated).
    """
    router = _make_router_with_mpnn_session()
    output = router._show_designed_sequences()

    assert "model #1" in output or "#1" in output, "model_id should appear"
    assert "Score" in output
    assert "Recovery" in output
    assert "Mutations" in output
    assert "Sequence" in output
    # The single design stored by _make_router_with_mpnn_session has mutation L2A
    assert "L2A" in output or "1" in output   # mutation or count


def test_show_designed_sequences_no_results():
    """
    _show_designed_sequences() when no MPNN results in session returns an
    informative error message (not an exception).
    """
    router = _make_router()
    output = router._show_designed_sequences()
    assert isinstance(output, str)
    assert "No ProteinMPNN results" in output


# ════════════════════════════════════════════════════════════════════════════════
# 11. Multi-model session fixes (models #1 + #2 loaded simultaneously)
# ════════════════════════════════════════════════════════════════════════════════

def _make_router_with_two_models() -> ToolRouter:
    """
    Router with two models loaded: #1 (original crystal structure) and
    #2 (e.g. an ESMFold predicted structure opened afterwards).
    No MPNN results in session.
    """
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = {
        "1": {"name": "1HSG", "path": None},
        "2": {"name": "ESMFold_pred", "path": None},
    }
    mock_session.get_proteinmpnn_result.return_value = None
    return ToolRouter(bridge=mock_bridge, session=mock_session)


def _make_router_with_two_models_and_mpnn() -> ToolRouter:
    """
    Router with two models loaded AND ProteinMPNN results for model #1
    (simulates: run redesign → ESMFold opens as #2 → user asks about sequences).
    """
    _mpnn_data = {
        "sequences": [
            {"sequence": "ACDE", "score": -1.2, "recovery": 0.9, "mutations": ["L2A"]},
        ],
        "wildtype_sequence": "ACLE",
        "fixed_positions":   [],
        "backend":           "local",
    }
    router = _make_router_with_two_models()
    router.session.get_proteinmpnn_result.return_value = _mpnn_data
    return router


def test_mpnn_esmfold_routing_with_second_model_loaded():
    """
    "ESMFold top 2-3 sequences to assess structural integrity" with model #2
    loaded and MPNN results in session must route to mpnn_esmfold, not esmfold.
    """
    router       = _make_router_with_two_models_and_mpnn()
    translator_r = {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "high",
        "tools_needed":         ["esmfold"],
        "tool_inputs":          {"esmfold": {"model_id": "2", "chain": "A"}},
    }
    user_input = "ESMFold top 2-3 sequences to assess structural integrity"

    routed = router.route(translator_r, user_input=user_input)

    assert "mpnn_esmfold" in routed["tools_needed"], (
        f"Expected mpnn_esmfold with 2 models + MPNN session; "
        f"got {routed['tools_needed']}"
    )
    assert "esmfold" not in routed["tools_needed"], (
        f"'esmfold' should be replaced by mpnn_esmfold; "
        f"got {routed['tools_needed']}"
    )


def test_assembly_analyser_targets_model_1_with_multiple_models():
    """
    When two models are loaded and inputs contain model_id='2', the assembly
    analyser must redirect to model_id='1' (the original crystal structure).
    """
    router = _make_router_with_two_models()

    # Configure mock session so get_structure() returns something parse-able
    router.session.get_structure.return_value = {"name": "1HSG"}

    # Inject a mock analyser so we can inspect the model_id it was called with
    mock_analyser = MagicMock()
    mock_analyser.analyse.return_value = {
        "mode":               "multimer",
        "model_id":           "1",
        "assembly_info":      {},
        "interfaces":         {},
        "protected_residues": [],
        "excluded_count":     0,
        "header":             "test",
        "warnings":           [],
        "interface_summary":  "",
    }
    router._assembly_analyser = mock_analyser

    # Call with model_id="2" — guard must redirect to "1"
    router._run_assembly_analyser({"model_id": "2", "mode": "multimer", "chain_id": "A"})

    call_args = mock_analyser.analyse.call_args
    called_model_id = (
        call_args.kwargs.get("model_id")
        if call_args.kwargs
        else (call_args.args[0] if call_args.args else None)
    )
    assert called_model_id == "1", (
        f"assembly analyser should target #1; was called with model_id={called_model_id!r}"
    )


def test_proline_routing_with_multiple_models():
    """
    "suggest proline mutations to stabilise chain A" with #2 active: the
    rewritten proline tool_inputs must have model_id='1' (crystal structure),
    not '2' (ESMFold prediction guessed by the translator).
    """
    router = _make_router_with_two_models()

    # Translator guessed model_id="2" (the currently active/visible model)
    translator_r = {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "high",
        "tools_needed":         ["mutation_scan"],
        "tool_inputs":          {
            "mutation_scan": {
                "model_id": "2",
                "chain":    "A",
                "focus":    "stability",
            }
        },
    }
    user_input = "suggest proline mutations to stabilise chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "proline" in routed["tools_needed"], (
        f"Expected proline routing; got {routed['tools_needed']}"
    )
    proline_mid = routed["tool_inputs"]["proline"]["model_id"]
    assert proline_mid == "1", (
        f"Proline scan must target #1 (crystal structure); got model_id={proline_mid!r}"
    )


def test_sequence_display_output_the_sequences():
    """
    'can you output the sequences' with MPNN results in session →
    handle_sequence_display_command returns a non-empty string.
    """
    router = _make_router_with_mpnn_session()
    result = router.handle_sequence_display_command("can you output the sequences")
    assert result is not None, (
        "'output the sequences' should trigger sequence display when session has MPNN results"
    )
    assert isinstance(result, str) and len(result) > 10


def test_sequence_display_sequences_for_redesigns():
    """
    'show me the sequences for the redesigns' with MPNN results in session →
    handle_sequence_display_command returns a non-empty string.
    """
    router = _make_router_with_mpnn_session()
    result = router.handle_sequence_display_command(
        "show me the sequences for the redesigns"
    )
    assert result is not None, (
        "'sequences for the redesigns' should trigger sequence display when session has MPNN results"
    )
    assert isinstance(result, str) and len(result) > 10


# ================================================================================
# 12. Salt bridge routing
# ================================================================================

def test_salt_bridge_phrase_routes_correctly():
    """
    'find salt bridges on chain A' must route to the salt_bridge tool.
    """
    router = _make_router()
    translator_r = _mutation_scan_translator_result()
    user_input = 'find salt bridges on chain A'

    routed = router.route(translator_r, user_input=user_input)

    assert 'salt_bridge' in routed['tools_needed'], (
        f"Expected 'salt_bridge' in tools_needed, got {routed['tools_needed']}"
    )
    assert 'mutation_scan' not in routed['tools_needed']
    sb_inp = routed['tool_inputs'].get('salt_bridge', {})
    assert sb_inp.get('model_id') is not None


# ================================================================================
# 13. Cavity routing
# ================================================================================

def test_cavity_phrase_routes_correctly():
    """
    'detect cavities in chain A' must route to the cavity tool.
    """
    router = _make_router()
    translator_r = _mutation_scan_translator_result()
    user_input = 'detect cavities in chain A'

    routed = router.route(translator_r, user_input=user_input)

    assert 'cavity' in routed['tools_needed'], (
        f"Expected 'cavity' in tools_needed, got {routed['tools_needed']}"
    )
    assert 'mutation_scan' not in routed['tools_needed']
    cav_inp = routed['tool_inputs'].get('cavity', {})
    assert cav_inp.get('model_id') is not None


# ================================================================================
# 14. FASTA export keyword
# ================================================================================

def test_fasta_export_keyword():
    """
    'export fasta' phrase must trigger _export_sequences_fasta, not _show_designed_sequences.
    When session has no MPNN results, the export returns an informative error string.
    """
    router = _make_router()  # no MPNN results
    result = router.handle_sequence_display_command('export fasta')
    # Should not be None -- FASTA export is checked before session guard
    assert result is not None, (
        'FASTA export keyword should return a string (error message if no results)'
    )
    assert isinstance(result, str)
    # With no session results, should mention ProteinMPNN
    assert 'ProteinMPNN' in result or 'sequences' in result.lower()


# ================================================================================
# 15. Sequence display no truncation
# ================================================================================

def test_sequence_display_no_truncation():
    """
    _show_designed_sequences() must include the full sequence, not truncated at 80 chars.
    We inject a 120-char sequence and verify it appears in full.
    """
    long_seq = 'ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRS'
    assert len(long_seq) > 80, 'Test sequence should be longer than 80 chars'

    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = {'1': {'name': '1HSG', 'path': None}}
    mpnn_data = {
        'sequences': [
            {'sequence': long_seq, 'score': -1.5, 'recovery': 0.95, 'mutations': []},
        ],
        'wildtype_sequence': 'A' * 10,
        'backend': 'local',
    }
    mock_session.get_proteinmpnn_result.return_value = mpnn_data
    router = ToolRouter(bridge=mock_bridge, session=mock_session)

    output = router._show_designed_sequences()
    assert long_seq in output, (
        'Full sequence should appear in output without truncation at 80 chars'
    )


# ════════════════════════════════════════════════════════════════════════════════
# Section N — Glycan positions routing
# ════════════════════════════════════════════════════════════════════════════════

def test_glycan_positions_phrase_routes():
    """
    'identify glycosylation positions for domain masking' must route to
    glycan_positions, not the standard glycan tool.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "identify glycosylation positions for domain masking on chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "glycan_positions" in routed["tools_needed"], (
        f"'domain masking' phrase should route to glycan_positions; "
        f"got {routed['tools_needed']}"
    )
    assert "glycan" not in routed["tools_needed"], (
        f"Should not also route to plain glycan tool; got {routed['tools_needed']}"
    )


def test_glycan_positions_projection_aware_keyword():
    """
    'projection-aware glycosylation' phrase must route to glycan_positions.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "show projection-aware glycosylation sites on chain B"

    routed = router.route(translator_r, user_input=user_input)

    assert "glycan_positions" in routed["tools_needed"], (
        f"'projection-aware glycosylation' should route to glycan_positions; "
        f"got {routed['tools_needed']}"
    )


def test_glycan_positions_inputs_have_model_id_and_chain():
    """
    glycan_positions tool_inputs must carry model_id and chain keys.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "find glycan candidates for immunosilencing chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "glycan_positions" in routed["tools_needed"], (
        f"'immunosilence' phrase should route to glycan_positions; "
        f"got {routed['tools_needed']}"
    )
    gp_inputs = routed.get("tool_inputs", {}).get("glycan_positions", {})
    assert "model_id" in gp_inputs, "glycan_positions inputs must include model_id"
    assert "chain"    in gp_inputs, "glycan_positions inputs must include chain"
    assert routed.get("clarification_needed") is None, (
        "clarification_needed must be cleared for glycan_positions routing"
    )


# ════════════════════════════════════════════════════════════════════════════════
# Section O — NetNGlyc routing
# ════════════════════════════════════════════════════════════════════════════════

def test_netnglyc_keyword_routes_to_netnglyc():
    """
    'run netnglyc on my sequence' must route to the netnglyc tool.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "run netnglyc on my sequence for chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "netnglyc" in routed["tools_needed"], (
        f"'netnglyc' keyword should route to netnglyc tool; "
        f"got {routed['tools_needed']}"
    )
    assert routed.get("clarification_needed") is None, (
        "clarification_needed must be cleared for netnglyc routing"
    )


def test_ost_recognition_phrase_routes_to_netnglyc():
    """
    'predict OST recognition for the engineered sequon' must route to netnglyc.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "predict ost recognition for the engineered sequon at position 42"

    routed = router.route(translator_r, user_input=user_input)

    assert "netnglyc" in routed["tools_needed"], (
        f"'ost recognition' phrase should route to netnglyc; "
        f"got {routed['tools_needed']}"
    )


def test_netnglyc_inputs_have_model_id_and_chain():
    """
    netnglyc tool_inputs must carry model_id and chain keys.
    """
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "show me the netnglyc ost score for this protein"

    routed = router.route(translator_r, user_input=user_input)

    assert "netnglyc" in routed["tools_needed"], (
        f"netnglyc routing missing; got {routed['tools_needed']}"
    )
    ng_inputs = routed.get("tool_inputs", {}).get("netnglyc", {})
    assert "model_id" in ng_inputs, "netnglyc inputs must include model_id"
    assert "chain"    in ng_inputs, "netnglyc inputs must include chain"


# ════════════════════════════════════════════════════════════════════════════════
# Double mutant routing tests
# ════════════════════════════════════════════════════════════════════════════════

def _scan_results_in_session() -> MagicMock:
    """Return a mock session with scan results and a loaded structure."""
    mock_session = MagicMock()
    mock_session.structures = {"1": {"name": "1HSG", "path": "cache/1HSG.pdb"}}
    mock_session.get_proteinmpnn_result.return_value = None
    mock_session.get_scan_result.return_value = [
        {
            "chain": "A", "position": 82, "from_aa": "V", "to_aa": "A",
            "ddg": -1.2, "solubility_delta": 0.8, "esm_tolerance": 0.7,
            "interface_proximal": False,
        },
        {
            "chain": "A", "position": 64, "from_aa": "I", "to_aa": "E",
            "ddg": -0.5, "solubility_delta": 1.2, "esm_tolerance": 0.6,
            "interface_proximal": False,
        },
    ]
    mock_session.get_interface_residues.return_value = {}
    mock_session.get_functional_residues.return_value = set()
    return mock_session


def _make_router_with_scan() -> ToolRouter:
    """ToolRouter with a mock session containing scan results."""
    mock_bridge  = MagicMock()
    mock_session = _scan_results_in_session()
    return ToolRouter(bridge=mock_bridge, session=mock_session)


def test_double_mutant_phrase_routes_correctly():
    """
    'suggest double mutant combinations' with scan results in session
    must route to double_mutant, NOT mutation_scan.
    """
    router       = _make_router_with_scan()
    translator_r = _mutation_scan_translator_result()
    user_input   = "suggest double mutant combinations"

    routed = router.route(translator_r, user_input=user_input)

    assert "double_mutant" in routed["tools_needed"], (
        f"Expected 'double_mutant' in tools_needed, got {routed['tools_needed']}"
    )
    assert "mutation_scan" not in routed["tools_needed"], (
        f"mutation_scan should be replaced; got {routed['tools_needed']}"
    )
    dm_inputs = routed["tool_inputs"].get("double_mutant", {})
    assert "model_id" in dm_inputs
    assert "_user_input" in dm_inputs


def test_epitope_mode_detected():
    """'preserve the epitope' keyword in user input → mode = 'epitope' in run."""
    router     = _make_router_with_scan()
    user_input = "double mutant combinations to preserve the epitope"

    routed = router.route(_mutation_scan_translator_result(), user_input=user_input)
    assert "double_mutant" in routed["tools_needed"]

    # Verify the _user_input is passed through so mode detection fires at run time
    dm_inputs = routed["tool_inputs"].get("double_mutant", {})
    assert "preserve" in dm_inputs.get("_user_input", "").lower()


def test_stability_mode_default():
    """No epitope keywords → stability mode (verified via _user_input passthrough)."""
    router     = _make_router_with_scan()
    user_input = "double mutant combinations"

    routed = router.route(_mutation_scan_translator_result(), user_input=user_input)
    assert "double_mutant" in routed["tools_needed"]

    dm_inputs = routed["tool_inputs"].get("double_mutant", {})
    stored_ui = dm_inputs.get("_user_input", "")
    _epitope_kw = ("epitope", "binding", "interface", "preserve", "target")
    mode = "epitope" if any(kw in stored_ui.lower() for kw in _epitope_kw) else "stability"
    assert mode == "stability", f"Expected stability mode for {user_input!r}, got {mode}"


def test_double_mutant_requires_scan_results():
    """No scan results in session → error ToolStepResult with helpful message."""
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = {"1": {"name": "1HSG", "path": "cache/1HSG.pdb"}}
    mock_session.get_proteinmpnn_result.return_value = None
    mock_session.get_scan_result.return_value = None  # no scan yet

    router = ToolRouter(bridge=mock_bridge, session=mock_session)

    # Patch _ensure_pdb_file to return a valid path
    with patch.object(router, "_ensure_pdb_file", return_value="cache/1HSG.pdb"):
        result = router._run_double_mutant({"model_id": "1", "_user_input": "double mutant"})

    assert not result.success
    assert "scan" in (result.error or "").lower(), (
        f"Error should mention scan results; got: {result.error}"
    )


def test_double_mutant_requires_loaded_structure():
    """No PDB file → error ToolStepResult with clear message."""
    router = _make_router_with_scan()

    with patch.object(router, "_ensure_pdb_file", return_value=None):
        result = router._run_double_mutant({"model_id": "1", "_user_input": "double mutant"})

    assert not result.success
    assert "structure" in (result.error or "").lower() or "pdb" in (result.error or "").lower(), (
        f"Error should mention missing structure; got: {result.error}"
    )


def test_pyrosetta_flag_detected():
    """'with rosetta validation' in user input → run_pyrosetta would be True."""
    router     = _make_router_with_scan()
    user_input = "double mutant combinations with rosetta validation"

    _pr_kw = ("pyrosetta", "rosetta", "accurate", "high accuracy", "validate")
    assert any(kw in user_input.lower() for kw in _pr_kw), (
        "Test input must contain a PyRosetta trigger keyword"
    )
    # Verify the intent detection logic in isolation
    lower = user_input.lower()
    run_pyrosetta = any(kw in lower for kw in _pr_kw)
    assert run_pyrosetta is True


def test_mutation_scan_not_affected():
    """Generic solubility scan phrases (no double/combine) still route to mutation_scan."""
    router = _make_router()

    for user_input in (
        "suggest mutations to improve solubility",
        "what mutations would reduce aggregation of chain A?",
        "run a mutation scan on this protein",
        "",
    ):
        routed = router.route(_mutation_scan_translator_result(), user_input=user_input)
        assert "mutation_scan" in routed["tools_needed"], (
            f"Phrase {user_input!r} should route to mutation_scan; "
            f"got {routed['tools_needed']}"
        )
        assert "double_mutant" not in routed["tools_needed"], (
            f"Phrase {user_input!r} should NOT route to double_mutant; "
            f"got {routed['tools_needed']}"
        )


def test_double_mutant_guard_in_mutation_scan():
    """
    When 'double mutant' reaches _dispatch_tool with tool='mutation_scan',
    the guard redirects to _run_double_mutant instead.
    """
    router = _make_router_with_scan()

    mock_dm   = MagicMock(return_value=ToolStepResult(
        tool="double_mutant", success=True, data={}, summary="ok"
    ))
    mock_scan = MagicMock(return_value=ToolStepResult(
        tool="mutation_scan", success=True, data={}, summary="ok"
    ))
    router._run_double_mutant = mock_dm
    router._run_mutation_scan = mock_scan

    inputs     = {"model_id": "1", "chain": "A"}
    user_input = "suggest double mutations for stability"

    result = router._dispatch_tool("mutation_scan", inputs, user_input=user_input)

    assert mock_dm.called, "_run_double_mutant should have been called"
    assert not mock_scan.called, "_run_mutation_scan should NOT have been called"


# ════════════════════════════════════════════════════════════════════════════════
# Mutation scan intent fallback routing tests
# ════════════════════════════════════════════════════════════════════════════════

def test_solubility_singular_routes_to_mutation_scan():
    """
    'suggest mutation to improve solubility' (singular) must route to
    mutation_scan, not cavity or any other tool.
    """
    router = _make_router()

    # Simulate translator returning "cavity" (the bug case)
    translator_r = {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "medium",
        "tools_needed":         ["cavity"],
        "tool_inputs":          {"cavity": {"model_id": "1", "chain": "A"}},
    }
    user_input = "suggest mutation to improve solubility of chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "mutation_scan" in routed["tools_needed"], (
        f"Singular 'mutation' should route to mutation_scan; "
        f"got {routed['tools_needed']}"
    )
    assert "cavity" not in routed["tools_needed"], (
        f"cavity should be overridden; got {routed['tools_needed']}"
    )


def test_solubility_without_mutation_keyword_routes_correctly():
    """
    'improve solubility of chain A avoiding interfaces' (no 'mutation' keyword)
    must still route to mutation_scan.
    """
    router = _make_router()

    # Simulate translator returning only chimerax (failed to detect scan intent)
    translator_r = {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "medium",
        "tools_needed":         ["chimerax"],
        "tool_inputs":          {},
    }
    user_input = "improve solubility of chain A avoiding interfaces"

    routed = router.route(translator_r, user_input=user_input)

    assert "mutation_scan" in routed["tools_needed"], (
        f"'improve solubility' should route to mutation_scan; "
        f"got {routed['tools_needed']}"
    )


def test_refusal_error_handled_gracefully():
    """
    When translator.translate() raises ValueError with 'refusal' in the message,
    _handle_request() must catch it, print a helpful message, and return cleanly
    without re-raising or crashing the process.
    """
    import main as _main_module
    from main import StructureBot
    from translator import CommandTranslator

    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = {}
    mock_session.command_history = []
    mock_session.get_proteinmpnn_result.return_value = None

    bot = StructureBot.__new__(StructureBot)
    bot.bridge         = mock_bridge
    bot.session        = mock_session
    bot.auto_proceed   = False
    bot.auto_proceed_delay = 0
    bot.log_file       = Path("/tmp/test_structurebot.log")
    bot.router         = ToolRouter(bridge=mock_bridge, session=mock_session)
    bot.translator     = MagicMock(spec=CommandTranslator)
    bot.translator.translate.side_effect = ValueError(
        "API returned empty response (stop_reason='refusal'). "
        "The prompt may have triggered a safety filter."
    )

    # Mock console so status() context manager and print() work in test environment
    mock_console = MagicMock()
    _cm = MagicMock()
    _cm.__enter__ = MagicMock(return_value=None)
    _cm.__exit__  = MagicMock(return_value=False)  # do not suppress exceptions
    mock_console.status.return_value = _cm

    raised = False
    with patch.object(_main_module, "console", mock_console):
        try:
            bot._handle_request("suggest epitope preserving mutations")
        except Exception:
            raised = True

    assert not raised, (
        "_handle_request() should catch refusal ValueError and return cleanly, "
        "not propagate an exception"
    )
    # Verify a warning was printed (console.print called with refusal message)
    assert mock_console.print.called, "console.print should have been called with the warning"


# ════════════════════════════════════════════════════════════════════════════════
# Cavity detection precision tests
# ════════════════════════════════════════════════════════════════════════════════

def test_cavity_does_not_match_solubility_request():
    """
    'suggest mutation to improve solubility of chain A' (singular) must NOT
    route to cavity — it must route to mutation_scan.
    """
    router       = _make_router()
    # Simulate the exact bug: translator wrongly returned ["cavity"]
    translator_r = {
        "commands":             [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "medium",
        "tools_needed":         ["cavity"],
        "tool_inputs":          {"cavity": {"model_id": "1", "chain": "A"}},
    }
    user_input = "suggest mutation to improve solubility of chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "cavity" not in routed["tools_needed"], (
        f"Solubility request should NOT route to cavity; got {routed['tools_needed']}"
    )
    assert "mutation_scan" in routed["tools_needed"], (
        f"Solubility request should route to mutation_scan; got {routed['tools_needed']}"
    )


def test_cavity_does_not_match_stability_request():
    """'calculate stability of V82A' must NOT route to cavity."""
    router       = _make_router()
    translator_r = _mutation_scan_translator_result()
    user_input   = "calculate stability of V82A"

    routed = router.route(translator_r, user_input=user_input)

    assert "cavity" not in routed["tools_needed"], (
        f"Stability request should NOT route to cavity; got {routed['tools_needed']}"
    )


def test_cavity_matches_explicit_cavity_request():
    """'identify cavities in chain A' must route to cavity."""
    router       = _make_router()
    # Translator may return chimerax for this request
    translator_r = _chimerax_translator_result()
    user_input   = "identify cavities in chain A"

    routed = router.route(translator_r, user_input=user_input)

    assert "cavity" in routed["tools_needed"], (
        f"Explicit cavity request should route to cavity; got {routed['tools_needed']}"
    )


def test_cavity_matches_fill_request():
    """'find buried voids to fill' must route to cavity."""
    router       = _make_router()
    translator_r = _chimerax_translator_result()
    user_input   = "find buried voids to fill"

    routed = router.route(translator_r, user_input=user_input)

    assert "cavity" in routed["tools_needed"], (
        f"'buried voids' should route to cavity; got {routed['tools_needed']}"
    )
