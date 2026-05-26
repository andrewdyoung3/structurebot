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

from translator import CommandTranslator as Translator

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

    print()
    print("=" * 60)
    print(
        f"Results: {_results['pass']} passed, {_results['fail']} failed"
    )
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
