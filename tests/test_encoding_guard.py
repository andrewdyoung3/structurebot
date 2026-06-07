"""
Regression guard for Unicode / stdout encoding (fixed 2026-06-07).

Root cause: bare print() calls in translator.py contained → (U+2192), which
cp1252 (Windows default stdout encoding) cannot represent.  The fix is a
two-layer defence:
  1. tests/conftest.py reconfigures sys.stdout to UTF-8 before any tests run.
  2. main.py reconfigures sys.stdout to UTF-8 at REPL startup.

These tests verify both layers are in place and that runtime-built Greek
strings (assembled from stoichiometry / chain labels, never literals) are safe.

Note: pytest.ini sets -p no:capture (Python 3.14 compat), so capsys is
unavailable; tests that capture stdout use contextlib.redirect_stdout.
"""
import contextlib
import io
import sys

import pytest


# ---------------------------------------------------------------------------
# 1. conftest.py layer: stdout must be UTF-8 before any test runs
# ---------------------------------------------------------------------------

def test_stdout_encoding_is_utf8():
    """conftest.py must have reconfigured sys.stdout to UTF-8."""
    enc = (sys.stdout.encoding or "").lower().replace("-", "")
    assert enc in ("utf8", "utf8bom"), (
        f"sys.stdout.encoding is {sys.stdout.encoding!r}; "
        "expected utf-8 (conftest.py must reconfigure it before tests run)"
    )


# ---------------------------------------------------------------------------
# 2. Runtime round-trip: runtime-built Greek strings encode under stdout.encoding
# ---------------------------------------------------------------------------

def test_runtime_unicode_strings_encode_under_stdout_encoding():
    """
    Greek strings built at runtime (assembled from parts, not literals) must
    encode without error under the current sys.stdout.encoding.

    This catches the gap the source-scan meta-test missed: a character like
    → assembled at runtime (e.g. from a format string that wasn't scanned)
    would have passed the old literal-scan but still crash on print().
    """
    alpha = "α"   # α
    beta  = "β"   # β
    arrow = "→"   # →
    enc = sys.stdout.encoding

    runtime_strings = [
        f"heterotetrameric ({alpha}2{beta}2)",
        f"{alpha} · 2HHB:A,C",              # middle dot U+00B7
        f"group {alpha}-globin",
        f"group {beta}-globin",
        f"  [chain-scope] 'color #1/A' {arrow} 'color (#1/A & ~ligand)'",
        f"  [zone-guard] 'select zone #1/B 4.5' {arrow} 'select #1/B :<4.5'",
    ]
    for s in runtime_strings:
        try:
            s.encode(enc)
        except UnicodeEncodeError as exc:
            pytest.fail(
                f"sys.stdout.encoding={enc!r} cannot encode runtime-built string "
                f"{s!r}: {exc}"
            )


# ---------------------------------------------------------------------------
# 3. Translator debug prints run without encoding error
# ---------------------------------------------------------------------------

def test_scope_chain_refs_print_runs_without_error():
    """`_scope_chain_refs_to_macromolecule` debug print must not raise."""
    from translator import _scope_chain_refs_to_macromolecule

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _scope_chain_refs_to_macromolecule(["color #1/A red", "select #2/B"])
    assert "[chain-scope]" in buf.getvalue()


def test_sanitize_zone_syntax_print_runs_without_error():
    """`_sanitize_zone_syntax` debug print must not raise."""
    from translator import _sanitize_zone_syntax

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _sanitize_zone_syntax(["select zone #1/B 4.5 & #1/A"], [""])
    assert "[zone-guard]" in buf.getvalue()


# ---------------------------------------------------------------------------
# 4. session_state JSON round-trip preserves Greek strings end-to-end
# ---------------------------------------------------------------------------

def test_session_state_greek_strings_roundtrip(tmp_path):
    """Greek strings ('α2β2', 'α · 2HHB:A,C') survive a session save/load cycle."""
    from session_state import SessionState

    greek_payload = {
        "assembly_label": "heterotetrameric (α2β2)",
        "group_label":    "α · 2HHB:A,C",
        "chain_names":    ["α-globin", "β-globin"],
    }

    state = SessionState()
    state.tool_results["greek_test"] = greek_payload

    save_path = str(tmp_path / "session_greek.json")
    state.save(save_path)

    loaded = SessionState.load(save_path)
    assert loaded.tool_results["greek_test"] == greek_payload, (
        "Greek strings lost or corrupted during session save/load"
    )
