"""
tests/test_ollama_backend.py
----------------------------
Ollama local backend: constrained-JSON request build + parse, backend-agnostic
guards, one-directional fallback routing, the VRAM-unload invariant, and config
plumbing. Everything mocked — no live Ollama, no live API, no live ChimeraX.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
import httpx
import requests as _requests

import config
import translator as T
from translator import (
    CommandTranslator, OllamaBackend, ClaudeBackend,
    ensure_translator_unloaded, make_backend, TRANSLATION_JSON_SCHEMA,
)
from tool_router import ToolRouter, ToolStepResult

PASS, FAIL = "[PASS]", "[FAIL]"
_results = {"pass": 0, "fail": 0}
def _ok(n): print(f"  {PASS} {n}"); _results["pass"] += 1
def _fail(n, w): print(f"  {FAIL} {n}: {w}"); _results["fail"] += 1
def _assert(c, n, m=""):
    (_ok(n) if c else _fail(n, m or "assertion failed")); return c


_VALID = {
    "commands": ["cartoon #1", "color #1/A red", "view"],
    "explanations": ["a", "b", "c"], "warnings": [], "clarification_needed": None,
    "confidence": "high", "tools_needed": ["chimerax"], "tool_inputs": {},
}
# translate() scopes a bare "chain A" ref to the macromolecule, so the parsed
# commands come back with `color #1/A` → `color (#1/A & ~ligand & ~solvent & ~ions)`
# (`cartoon #1` is a whole-MODEL ref, not a chain ref → untouched).
_VALID_SCOPED = ["cartoon #1", "color (#1/A & ~ligand & ~solvent & ~ions) red", "view"]


def _translator() -> CommandTranslator:
    return CommandTranslator(api_key="sk-ant-test-key")


def _ollama_translator() -> CommandTranslator:
    t = _translator(); t._backend = OllamaBackend(); return t


def _ollama_resp(payload: dict) -> MagicMock:
    r = MagicMock()
    r.json.return_value = {"message": {"content": json.dumps(payload)}}
    return r


def _session():
    s = MagicMock(); s.get_context_summary.return_value = "No structures loaded."
    return s


def _make_router() -> ToolRouter:
    return ToolRouter(bridge=MagicMock(), session=MagicMock())


# -- A. request build + parse --------------------------------------------------

def test_ollama_builds_constrained_request_and_parses() -> None:
    print("\n=== A. request build + parse ===")
    t = _ollama_translator()
    cap = {}
    def fake_post(url, json=None, timeout=None):
        cap["url"] = url; cap["payload"] = json; cap["timeout"] = timeout
        return _ollama_resp(_VALID)
    with patch("requests.post", side_effect=fake_post):
        result = t.translate("color chain A red", _session())
    p = cap["payload"]
    _assert(cap["url"].endswith("/api/chat"), "POSTs to /api/chat")
    _assert(p["model"] == config.OLLAMA_MODEL, "uses config.OLLAMA_MODEL")
    _assert(p["format"] == TRANSLATION_JSON_SCHEMA,
            "format == the 7-key JSON schema (constrained output, not tool-calls)")
    _assert(p["options"]["temperature"] == 0, "temperature=0 (deterministic)")
    _assert(p["options"]["num_ctx"] == config.OLLAMA_NUM_CTX, "num_ctx from config")
    _assert(p["keep_alive"] == config.OLLAMA_KEEP_ALIVE, "keep_alive from config")
    _assert(p["stream"] is False, "non-streaming")
    _assert(p["messages"][0]["role"] == "system" and "ChimeraX" in p["messages"][0]["content"],
            "shared system prompt reused (not forked)")
    _assert(any(m["role"] == "user" for m in p["messages"]), "user message present")
    for k in ("commands", "explanations", "warnings", "clarification_needed",
              "confidence", "tools_needed", "tool_inputs"):
        _assert(k in result, f"normalized result has '{k}'")
    _assert(result["commands"] == _VALID_SCOPED, "parsed commands match (chain-scoped)")


# -- B. guards apply to Ollama output (backend-agnostic) -----------------------

def test_guard_applies_to_ollama_output() -> None:
    print("\n=== B. guards apply to Ollama output ===")
    t = _ollama_translator()
    bad = dict(_VALID, commands=["select #1/A & (zone #1/B 4.5)", "info residues sel"],
               explanations=["x", "y"])
    with patch("requests.post", return_value=_ollama_resp(bad)):
        result = t.translate("interface residues within 4.5 of B", _session())
    _assert(result["commands"][0] == "select (#1/A & ~ligand & ~solvent & ~ions) & "
            "((#1/B & ~ligand & ~solvent & ~ions) :<4.5)",
            "_sanitize_zone_syntax rewrote the Ollama output, then chain-scoped (same guard path)",
            f"got {result['commands'][0]!r}")
    _assert(any("Rewrote" in w for w in result["warnings"]), "rewrite surfaced as warning")


# -- C. one-directional fallback routing ---------------------------------------

def test_claude_error_falls_back_to_ollama() -> None:
    print("\n=== C. fallback routing ===")
    t = _translator()   # default claude backend
    conn = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
    with patch.object(t, "client") as mc, \
         patch("requests.post", return_value=_ollama_resp(_VALID)) as mpost:
        mc.messages.create.side_effect = conn
        result = t.translate("color chain A red", _session())
    _assert(mpost.called, "Claude connectivity error + fallback on → Ollama called")
    _assert(result["commands"] == _VALID_SCOPED, "result came from Ollama (chain-scoped)")


def test_claude_success_does_not_call_ollama() -> None:
    t = _translator()
    good = MagicMock(content=[MagicMock(text=json.dumps(_VALID))], stop_reason="end_turn")
    with patch.object(t, "client") as mc, patch("requests.post") as mpost:
        mc.messages.create.return_value = good
        t.translate("color chain A red", _session())
    _assert(not mpost.called, "Claude success → Ollama NEVER called")


def test_fallback_disabled_reraises() -> None:
    t = _translator()
    conn = anthropic.APIConnectionError(request=httpx.Request("POST", "http://x"))
    with patch.object(config, "TRANSLATOR_FALLBACK", False), \
         patch.object(t, "client") as mc, patch("requests.post") as mpost:
        mc.messages.create.side_effect = conn
        raised = False
        try:
            t.translate("color chain A red", _session())
        except anthropic.APIConnectionError:
            raised = True
    _assert(raised, "fallback OFF → Claude error propagates")
    _assert(not mpost.called, "fallback OFF → Ollama NOT called")


def test_forced_ollama_error_surfaces_claude_never_called() -> None:
    t = _ollama_translator()
    with patch.object(t, "client") as mc, \
         patch("requests.post", side_effect=_requests.exceptions.ConnectionError("down")):
        surfaced = False
        try:
            t.translate("color chain A red", _session())
        except _requests.exceptions.ConnectionError:
            surfaced = True
    _assert(surfaced, "forced ollama + ollama down → error surfaces (benchmark honesty)")
    _assert(not mc.messages.create.called, "forced ollama → Claude is NEVER called")


def test_refusal_is_not_a_fallback_trigger() -> None:
    """An empty/declined-but-successful Claude response (RefusalError) is NOT a
    fallback trigger — it propagates as today."""
    from translator import RefusalError
    t = _translator()
    empty = MagicMock(content=[], stop_reason="refusal")
    with patch.object(t, "client") as mc, patch("requests.post") as mpost:
        mc.messages.create.return_value = empty   # both attempts empty → RefusalError
        raised = False
        try:
            t.translate("color chain A red", _session())
        except RefusalError:
            raised = True
    _assert(raised, "RefusalError still raised (not swallowed by fallback)")
    _assert(not mpost.called, "RefusalError does NOT fall back to Ollama")


# -- C2. usage-cap (400) fallback ----------------------------------------------

def _usage_cap_error() -> anthropic.BadRequestError:
    """A Claude USAGE/SPEND-CAP rejection: HTTP 400 invalid_request_error whose
    message reports a usage limit (NOT a 429 RateLimitError / 401 AuthError)."""
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    msg = ("You have reached your specified API usage limits. "
           "You will regain access on 2026-07-01 at 00:00 UTC.")
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": msg}}
    return anthropic.BadRequestError(msg, response=resp, body=body)


def _malformed_400() -> anthropic.BadRequestError:
    """A genuinely malformed request — also a 400, must NOT be treated as a cap."""
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    msg = "messages: at least one message is required"
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": msg}}
    return anthropic.BadRequestError(msg, response=resp, body=body)


def test_is_usage_cap_error_classification() -> None:
    print("\n=== C2. usage-cap fallback ===")
    _assert(T.is_usage_cap_error(_usage_cap_error()), "usage-cap 400 classified as a cap")
    _assert(not T.is_usage_cap_error(_malformed_400()), "malformed 400 is NOT a cap")
    _assert(not T.is_usage_cap_error(
        anthropic.RateLimitError(
            "rate", response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
            body=None)), "429 RateLimitError is NOT a cap")
    _assert(not T.is_usage_cap_error(ValueError("nope")), "non-API error is NOT a cap")
    # forced 'ollama' never falls back, so a cap can't reroute it
    _assert(not _ollama_translator()._may_fall_back(), "forced ollama → _may_fall_back is False")
    _assert(_translator()._may_fall_back(), "active claude + fallback on → _may_fall_back is True")


def test_usage_cap_falls_back_to_ollama() -> None:
    t = _translator()   # active = claude
    with patch.object(t, "client") as mc, \
         patch("requests.post", return_value=_ollama_resp(_VALID)) as mpost:
        mc.messages.create.side_effect = _usage_cap_error()
        result = t.translate("color chain A red", _session())
    _assert(mpost.called, "usage-cap + fallback on → Ollama called")
    _assert(result["commands"] == _VALID_SCOPED, "result came from Ollama (normal translation)")


def test_usage_cap_fallback_disabled_reraises() -> None:
    t = _translator()
    with patch.object(config, "TRANSLATOR_FALLBACK", False), \
         patch.object(t, "client") as mc, patch("requests.post") as mpost:
        mc.messages.create.side_effect = _usage_cap_error()
        raised = False
        try:
            t.translate("color chain A red", _session())
        except anthropic.BadRequestError:
            raised = True
    _assert(raised, "usage-cap + fallback OFF → surfaces (no silent swallow)")
    _assert(not mpost.called, "fallback OFF → Ollama NOT called")


def test_non_usage_cap_400_does_not_fall_back() -> None:
    t = _translator()
    with patch.object(t, "client") as mc, patch("requests.post") as mpost:
        mc.messages.create.side_effect = _malformed_400()
        raised = False
        try:
            t.translate("color chain A red", _session())
        except anthropic.BadRequestError:
            raised = True
    _assert(raised, "malformed 400 → surfaces as an error (not a cap)")
    _assert(not mpost.called, "malformed 400 → does NOT reroute to Ollama")


# -- D. VRAM unload invariant --------------------------------------------------

def test_ensure_unloaded_noop_when_nothing_loaded(monkeypatch) -> None:
    print("\n=== D. ensure_translator_unloaded ===")
    monkeypatch.setattr(T, "_OLLAMA_MAY_BE_LOADED", False)
    with patch("requests.post") as mp:
        ensure_translator_unloaded()
    _assert(not mp.called, "no-op when no Ollama model loaded (e.g. Claude-only)")


def test_ensure_unloaded_fires_keepalive_zero(monkeypatch) -> None:
    monkeypatch.setattr(T, "_OLLAMA_MAY_BE_LOADED", True)
    cap = {}
    def fake_post(url, json=None, timeout=None):
        cap["url"] = url; cap["json"] = json; return MagicMock()
    with patch("requests.post", side_effect=fake_post):
        ensure_translator_unloaded()
    _assert(cap.get("json", {}).get("keep_alive") == 0, "unloads with keep_alive=0")
    _assert(cap["json"]["model"] == config.OLLAMA_MODEL, "targets the configured model")
    _assert(T._OLLAMA_MAY_BE_LOADED is False, "loaded-flag cleared after unload")


def test_gpu_dispatch_fires_unload() -> None:
    r = _make_router()
    with patch("translator.ensure_translator_unloaded") as munload, \
         patch.object(r, "_run_proteinmpnn",
                      return_value=ToolStepResult(tool="proteinmpnn", success=True)):
        r._dispatch_tool("proteinmpnn", {})
    _assert(munload.called, "GPU bridge dispatch (proteinmpnn) calls ensure_translator_unloaded")


def test_non_gpu_dispatch_does_not_unload() -> None:
    r = _make_router()
    with patch("translator.ensure_translator_unloaded") as munload, \
         patch.object(r, "_run_camsol",
                      return_value=ToolStepResult(tool="camsol", success=True)):
        r._dispatch_tool("camsol", {})
    _assert(not munload.called, "non-GPU dispatch (camsol) does NOT unload")


# -- E. config plumbing + factory ----------------------------------------------

def test_config_and_factory() -> None:
    print("\n=== E. config + factory ===")
    for k in ("TRANSLATOR_BACKEND", "TRANSLATOR_FALLBACK", "OLLAMA_BASE_URL",
              "OLLAMA_MODEL", "OLLAMA_NUM_CTX", "OLLAMA_KEEP_ALIVE", "OLLAMA_TIMEOUT",
              "TRANSLATOR_TOOL_NAMES"):
        _assert(hasattr(config, k), f"config.{k} present")
    _assert(config.TRANSLATOR_BACKEND == "claude", "default backend UNCHANGED (claude)")
    _assert(isinstance(make_backend("ollama"), OllamaBackend), "factory builds OllamaBackend")
    _assert("ollama" in T._BACKENDS, "ollama registered in _BACKENDS")


# -- F. enum-constrained tools + targeted few-shot -----------------------------

def test_schema_enum_constrains_tools() -> None:
    print("\n=== F. enum + few-shot ===")
    enum = TRANSLATION_JSON_SCHEMA["properties"]["tools_needed"]["items"].get("enum")
    _assert(enum == list(config.TRANSLATOR_TOOL_NAMES),
            "tools_needed items ENUM == config.TRANSLATOR_TOOL_NAMES")
    _assert(all(t == t.lower() for t in enum), "all enum tool names lowercase (router literals)")
    for t in ("camsol", "esm", "proteinmpnn", "mutation_scan", "disulfide", "chimerax"):
        _assert(t in enum, f"valid tool {t!r} in enum")
    _assert("CamSol" not in enum and "frobnicate" not in enum,
            "mis-cased / hallucinated names are NOT selectable (constrained decoding)")
    # The enum must match the REAL router registry (parse tool_router source).
    import re as _re
    src = (Path(__file__).parent.parent / "tool_router.py").read_text(encoding="utf-8")
    dispatch = set(_re.findall(r'tool == "([a-z_]+)"', src))
    _assert(dispatch.issubset(set(enum)),
            "every router-dispatched tool is in the enum (no router tool omitted)",
            f"missing {dispatch - set(enum)}")


def test_ollama_request_includes_targeted_fewshot() -> None:
    t = _ollama_translator()
    cap = {}
    def fake_post(url, json=None, timeout=None):
        cap["payload"] = json; return _ollama_resp(_VALID)
    with patch("requests.post", side_effect=fake_post):
        t.translate("color chain A red", _session())
    msgs = cap["payload"]["messages"]
    _assert(msgs[0]["role"] == "system", "system prompt first")
    # few-shot demo pairs (from EXAMPLE_POOL) precede the real user turn
    import translator_corpus as _tc, json as _json
    fs = _tc.few_shot_messages()
    _assert(len(fs) >= 8 and msgs[1:1+len(fs)] == fs,
            "targeted few-shot (camsol/mpnn/sel/esm) injected between system and the request")
    asst = [_json.loads(m["content"]) for m in fs[1::2]]
    routed = {tn for a in asst for tn in a["tools_needed"]}
    _assert({"camsol", "proteinmpnn", "esm"}.issubset(routed),
            "few-shot demonstrates the failing-category routings")


def main() -> int:
    print("=" * 60); print("tests/test_ollama_backend.py"); print("=" * 60)
    test_ollama_builds_constrained_request_and_parses()
    test_guard_applies_to_ollama_output()
    test_claude_error_falls_back_to_ollama()
    test_claude_success_does_not_call_ollama()
    test_fallback_disabled_reraises()
    test_forced_ollama_error_surfaces_claude_never_called()
    test_refusal_is_not_a_fallback_trigger()
    test_is_usage_cap_error_classification()
    test_usage_cap_falls_back_to_ollama()
    test_usage_cap_fallback_disabled_reraises()
    test_non_usage_cap_400_does_not_fall_back()
    print("\n(ensure-unloaded tests need pytest's monkeypatch)")
    test_gpu_dispatch_fires_unload()
    test_non_gpu_dispatch_does_not_unload()
    test_config_and_factory()
    test_schema_enum_constrains_tools()
    test_ollama_request_includes_targeted_fewshot()
    print(f"\nResults: {_results['pass']} passed, {_results['fail']} failed")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
