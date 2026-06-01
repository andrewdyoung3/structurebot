"""
tests/test_translator.py
------------------------
Tests for Translator internals that don't require a live API key.

A. _pre_screen()  -- rfdiffusion keyword detection, no-API-call path
B. _call_api()    -- empty response guard (IndexError fix)

Usage
-----
  cd structurebot
  python -m pytest tests/test_translator.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from translator import (
    CommandTranslator as Translator,
    RefusalError,
    _sanitize_zone_syntax,
)

# -- Helpers -------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"

_results = {"pass": 0, "fail": 0}


def _ok(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {PASS} {name}{suffix}")
    _results["pass"] += 1


def _fail(name: str, reason: str) -> None:
    print(f"  {FAIL} {name}: {reason}")
    _results["fail"] += 1


def _assert(cond: bool, name: str, msg: str = "") -> bool:
    if cond:
        _ok(name)
        return True
    _fail(name, msg or "assertion failed")
    return False


def _make_translator() -> Translator:
    """Return a Translator with a fake API key (no live calls)."""
    return Translator(api_key="sk-ant-test-key-does-not-call-api")


# -- A. _pre_screen() ----------------------------------------------------------

def test_pre_screen_passes_through_normal_request() -> None:
    print("\n=== A. _pre_screen() ===")
    t = _make_translator()
    result = t._pre_screen("open 1HSG and color by chain")
    _assert(result is None, "normal request -> None (no pre-screen)")


def test_pre_screen_rfdiffusion_keywords() -> None:
    """All rfdiffusion keyword variants trigger the pre-screen."""
    t = _make_translator()
    inputs = [
        "design a binder for 1HSG",
        "I want binder design for the active site",
        "Use RFdiffusion to make something",
        "run rf diffusion",
        "de novo backbone design",
        "de-novo backbone generation",
        "scaffold a motif from chain A",
        "motif scaffold design",
        "partial diffusion of the structure",
        "design symmetric oligomer C3",
        "backbone generation request",
    ]
    for phrase in inputs:
        result = t._pre_screen(phrase)
        _assert(
            result is not None,
            f"rfdiffusion keyword detected: {phrase!r}",
            f"got None (should be a dict)",
        )


def test_pre_screen_returns_valid_result_shape() -> None:
    """Pre-screen result has required keys and rfdiffusion in tools_needed."""
    t = _make_translator()
    result = t._pre_screen("design a protein binder for the active site")
    _assert(result is not None, "pre-screen triggered")
    if result is None:
        return
    for key in ("commands", "explanations", "warnings",
                "confidence", "tools_needed", "tool_inputs"):
        _assert(key in result, f"result has '{key}' key")
    _assert("rfdiffusion" in result["tools_needed"],
            "tools_needed contains rfdiffusion",
            f"got {result['tools_needed']}")
    _assert(result["commands"] == [], "commands is empty list")
    _assert(result["confidence"] == "high", "confidence is high")


def test_pre_screen_case_insensitive() -> None:
    """Keyword matching is case-insensitive."""
    t = _make_translator()
    for variant in ("DESIGN A BINDER", "Design A Binder", "Design a BINDER"):
        result = t._pre_screen(variant)
        _assert(result is not None,
                f"case-insensitive match for {variant!r}")


def test_translate_rfdiffusion_skips_api() -> None:
    """translate() with an rfdiffusion request never calls the API."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."

    # If the API were called, it would raise (fake key).
    # We verify _call_api is NOT invoked.
    with patch.object(t, "_call_api") as mock_api:
        result = t.translate("design a binder for chain A", session)

    mock_api.assert_not_called()
    _assert("rfdiffusion" in result.get("tools_needed", []),
            "pre-screened result routes to rfdiffusion",
            f"got tools_needed={result.get('tools_needed')}")


# -- B. _call_api() empty response guard ---------------------------------------

def test_call_api_empty_content_raises_value_error() -> None:
    print("\n=== B. _call_api() empty response guard ===")
    t = _make_translator()

    # Simulate an API response with an empty content list
    mock_response = MagicMock()
    mock_response.content = []
    mock_response.stop_reason = "end_turn"

    with patch.object(t, "client") as mock_client:
        mock_client.messages.create.return_value = mock_response
        try:
            t._call_api([{"type": "text", "text": "system"}])
            _fail("empty content raises ValueError", "no exception raised")
        except ValueError as exc:
            _assert("empty" in str(exc).lower() or "stop_reason" in str(exc),
                    "ValueError message mentions empty response",
                    f"got: {exc}")
        except IndexError:
            _fail("empty content raises ValueError",
                  "got IndexError instead (guard not applied)")


def test_call_api_non_empty_content_returns_text() -> None:
    """Normal response returns stripped text without raising."""
    t = _make_translator()

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="  hello world  ")]
    mock_response.stop_reason = "end_turn"

    with patch.object(t, "client") as mock_client:
        mock_client.messages.create.return_value = mock_response
        result = t._call_api([{"type": "text", "text": "system"}])

    _assert(result == "hello world", "non-empty content returns stripped text",
            f"got {result!r}")


def test_call_api_stop_reason_in_error() -> None:
    """stop_reason is included in the ValueError message."""
    t = _make_translator()

    mock_response = MagicMock()
    mock_response.content = []
    mock_response.stop_reason = "max_tokens"

    with patch.object(t, "client") as mock_client:
        mock_client.messages.create.return_value = mock_response
        try:
            t._call_api([])
            _fail("stop_reason in error", "no exception raised")
        except ValueError as exc:
            _assert("max_tokens" in str(exc),
                    "stop_reason='max_tokens' in error message",
                    f"got: {exc}")


# -- C. Over-eager refusal handling (false positives on benign design) ---------

_BENIGN = "redesign the dimer interface residues to be hydrophilic, no cysteines"

_VALID_JSON = (
    '{"commands": ["select #1/A"], "explanations": ["x"], "warnings": [], '
    '"clarification_needed": null, "confidence": "high", '
    '"tools_needed": ["proteinmpnn"], "tool_inputs": {}}'
)


def test_benign_design_request_not_flagged() -> None:
    """
    A benign protein-engineering request that gets ONE transient empty/declined
    response must NOT be flagged: the automatic retry succeeds and a normal result
    is returned (no RefusalError).
    """
    print("\n=== C. over-eager refusal handling ===")
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."

    empty = MagicMock(content=[], stop_reason="refusal")
    good  = MagicMock(content=[MagicMock(text=_VALID_JSON)], stop_reason="end_turn")

    with patch.object(t, "client") as mock_client:
        mock_client.messages.create.side_effect = [empty, good]   # 1st empty, retry OK
        result = t.translate(_BENIGN, session)

    assert result.get("commands") == ["select #1/A"], (
        f"benign request should translate after retry, got {result!r}")
    assert mock_client.messages.create.call_count == 2, "should retry exactly once"
    _ok("benign design request not flagged (retry rescued it)")


def test_call_api_retries_once_then_refuses() -> None:
    """Two consecutive empty responses → RefusalError, after exactly 2 calls."""
    t = _make_translator()
    with patch.object(t, "client") as mock_client:
        mock_client.messages.create.return_value = MagicMock(
            content=[], stop_reason="refusal")
        try:
            t._call_api([{"type": "text", "text": "system"}])
            _fail("retries then refuses", "no exception raised")
            return
        except RefusalError as exc:
            _assert(mock_client.messages.create.call_count == 2,
                    "retried exactly once before refusing",
                    f"got {mock_client.messages.create.call_count} calls")
            # Transparent: shows the REAL stop_reason, does NOT claim a safety filter.
            _assert("stop_reason" in str(exc) and "refusal" in str(exc),
                    "refusal message surfaces the real stop_reason", f"got: {exc}")
            _assert("safety filter" not in str(exc).lower(),
                    "refusal message does NOT assume a safety filter", f"got: {exc}")


def test_system_prompt_frames_routine_design_as_legitimate() -> None:
    """The static prompt frames routine protein-engineering as standard, so the
    model doesn't over-refuse it."""
    prompt = _make_translator()._static_block.lower()
    assert "standard computational structural biology" in prompt
    assert "do not decline" in prompt or "do not refuse" in prompt
    for term in ("interface", "hydrophilic", "cysteine", "epitope"):
        assert term in prompt, f"scope note should mention {term!r}"
    _ok("system prompt frames routine design operations as legitimate")


# -- D. Chimera-1 zone-syntax guard (invalid in ChimeraX 1.11) -----------------

_ZONE_JSON = (
    '{"commands": ["select #1/A & (zone #1/B 4.5)", "info residues sel"], '
    '"explanations": ["interface", "list"], "warnings": [], '
    '"clarification_needed": null, "confidence": "high", '
    '"tools_needed": ["chimerax"], "tool_inputs": {}}'
)


def test_zone_guard_rewrites_bare_and_parenthesised() -> None:
    print("\n=== D. zone-syntax guard ===")
    # bare form
    cmds, _, notes = _sanitize_zone_syntax(["select zone #1/B 4.5 & #1/A"], [""])
    _assert(cmds == ["select #1/B :<4.5 & #1/A"],
            "bare zone rewritten to :<", f"got {cmds}")
    # parenthesised form (the regression seen live)
    cmds2, _, notes2 = _sanitize_zone_syntax(["select #1/A & (zone #1/B 4.5)"], [""])
    _assert(cmds2 == ["select #1/A & (#1/B :<4.5)"],
            "parenthesised (zone …) rewritten, parens kept", f"got {cmds2}")
    _assert(all("zone" not in c for c in cmds + cmds2), "no `zone` keyword remains")
    _assert(bool(notes) and bool(notes2), "both rewrites logged in notes")


def test_zone_guard_leaves_valid_and_volume_zone() -> None:
    valid = ["select #1/B :<4.5 & #1/A", "info residues sel"]
    cmds, _, notes = _sanitize_zone_syntax(valid, ["", ""])
    _assert(cmds == valid and notes == [], "valid :< left untouched", f"got {cmds}")
    vz, _, vnotes = _sanitize_zone_syntax(["volume zone #1 4.5"], [""])
    _assert(vz == ["volume zone #1 4.5"] and vnotes == [],
            "volume zone preserved", f"got {vz}")


def test_zone_guard_drops_unrewritable() -> None:
    cmds, exps, notes = _sanitize_zone_syntax(
        ["select zone sel", "info residues sel"], ["bad", "list"])
    _assert(cmds == ["info residues sel"], "unrewritable zone dropped", f"got {cmds}")
    _assert(exps == ["list"], "dropped command's explanation removed")
    _assert(any("Dropped" in n for n in notes), "drop logged", f"got {notes}")


def test_translate_rewrites_zone_end_to_end() -> None:
    """A model emitting the parenthesised Chimera-1 zone is corrected before
    return: interface request → `:<` + `& #1/A`, `info residues sel` survives."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "1IL8 loaded as #1 (chains A, B)."
    resp = MagicMock(content=[MagicMock(text=_ZONE_JSON)], stop_reason="end_turn")
    with patch.object(t, "client") as mock_client:
        mock_client.messages.create.return_value = resp
        result = t.translate(
            "select the residues at the dimer interface on chain A", session)
    cmds = result.get("commands", [])
    _assert(cmds[0] == "select #1/A & (#1/B :<4.5)",
            "interface select uses :< not zone", f"got {cmds}")
    _assert(all("zone" not in c for c in cmds), "no `zone` survives")
    _assert("info residues sel" in cmds, "`info residues sel` preserved")
    _assert(any("Rewrote" in w for w in result.get("warnings", [])),
            "rewrite surfaced as a warning")


def test_system_prompt_documents_zone_and_hide() -> None:
    prompt = _make_translator()._static_block
    low = prompt.lower()
    _assert(":<" in prompt and "@<" in prompt, "prompt documents :< / @< operators")
    _assert("info residues sel" in prompt, "prompt lists via `info residues sel`")
    _assert("never" in low and "zone" in low, "prompt forbids Chimera-1 `zone`")
    # BUG 2: hide/show must target the representation, not bare atoms.
    _assert("hide #1/b cartoon" in low or "hide #1/b target ac" in low,
            "prompt shows hide targeting the cartoon/representation")


def test_hide_chain_targets_representation() -> None:
    """'hide chain B' must target the cartoon (not a bare `hide #1/B`, which
    only hides atoms and leaves the ribbon visible)."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "1IL8 loaded as #1 (chains A, B)."
    hide_json = (
        '{"commands": ["hide #1/B cartoon"], "explanations": ["hide B ribbon"], '
        '"warnings": [], "clarification_needed": null, "confidence": "high", '
        '"tools_needed": ["chimerax"], "tool_inputs": {}}'
    )
    resp = MagicMock(content=[MagicMock(text=hide_json)], stop_reason="end_turn")
    with patch.object(t, "client") as mock_client:
        mock_client.messages.create.return_value = resp
        result = t.translate("hide chain B", session)
    cmds = result.get("commands", [])
    joined = " ".join(cmds).lower()
    _assert(cmds == ["hide #1/B cartoon"], "hide targets cartoon", f"got {cmds}")
    _assert("cartoon" in joined or "target a" in joined,
            "representation named (cartoon / target ac), not bare hide")
    _assert(cmds != ["hide #1/B"], "not a bare atoms-only hide")


# -- Runner --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("tests/test_translator.py -- Translator Internals Tests")
    print("=" * 60)

    # A. _pre_screen()
    test_pre_screen_passes_through_normal_request()
    test_pre_screen_rfdiffusion_keywords()
    test_pre_screen_returns_valid_result_shape()
    test_pre_screen_case_insensitive()
    test_translate_rfdiffusion_skips_api()

    # B. _call_api() guard
    test_call_api_empty_content_raises_value_error()
    test_call_api_non_empty_content_returns_text()
    test_call_api_stop_reason_in_error()

    # C. over-eager refusal handling
    test_benign_design_request_not_flagged()
    test_call_api_retries_once_then_refuses()
    test_system_prompt_frames_routine_design_as_legitimate()

    # D. zone-syntax guard + hide/show representation
    test_zone_guard_rewrites_bare_and_parenthesised()
    test_zone_guard_leaves_valid_and_volume_zone()
    test_zone_guard_drops_unrewritable()
    test_translate_rewrites_zone_end_to_end()
    test_system_prompt_documents_zone_and_hide()
    test_hide_chain_targets_representation()

    print()
    print("=" * 60)
    print(
        f"Results: {_results['pass']} passed, {_results['fail']} failed"
    )
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
