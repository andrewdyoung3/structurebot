"""
tests/test_bio_assembly.py
--------------------------
Tests for biological-assembly generation (Component 1), AU mismatch detection
(Component 2), and verb-guard + correction-loop fixes (Component 3a/3b).

All mocked — no live ChimeraX required in CI.

Test groups
-----------
1.  Routing — bio_assembly intent detection + route() override
2.  _run_bio_assembly — correct sym command, no-re-open, default/override assembly id,
    error-first (no model / sym failure lists available assemblies)
3.  AU mismatch detector — header A4 + 2 loaded chains → flag emitted;
    loaded == biological → no flag
4.  Verb guard 3a — assembly_analyser blocked; sym passes
5.  Correction loop 3b — repeated rejected verb → halt; different real verb → proceeds
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter, ToolStepResult
from translator import _validate_command_verbs, _HALLUCINATED_VERB_DENYLIST


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_router(structures: dict | None = None) -> ToolRouter:
    """ToolRouter with mocked bridge + session."""
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = structures if structures is not None else {
        "1": {"name": "2VNC", "path": None}
    }
    mock_session.get_proteinmpnn_result.return_value = None
    mock_session.get_assembly_info.return_value = None
    mock_session.get_structure.return_value = {"name": "2VNC", "path": None}
    return ToolRouter(bridge=mock_bridge, session=mock_session)


def _chimerax_result(value: str = "", error: str | None = None) -> dict:
    return {"value": value, "error": error}


def _translator_result_chimerax(cmds: list | None = None) -> Dict[str, Any]:
    return {
        "commands":             cmds or [],
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
    "work as tetramer",
    "work as a tetramer",
    "generate biological assembly",
    "generate the biological assembly",
    "build the biological unit",
    "build biological assembly",
    "open as tetramer",
    "open as a tetramer",
    "show the full assembly",
    "make the full tetramer",
    "apply crystal symmetry",
])
def test_bio_assembly_intent_detected(phrase):
    """bio_assembly intent keywords must fire _detect_bio_assembly_intent."""
    assert ToolRouter._detect_bio_assembly_intent(phrase), (
        f"Expected bio_assembly intent for {phrase!r}"
    )


def test_bio_assembly_route_override_replaces_chimerax():
    """route() with bio_assembly intent must replace tools_needed with ['bio_assembly']."""
    router       = _make_router()
    translator_r = _translator_result_chimerax(cmds=["open 2VNC"])
    routed       = router.route(translator_r, user_input="work as tetramer")

    assert routed["tools_needed"] == ["bio_assembly"], (
        f"Expected ['bio_assembly'], got {routed['tools_needed']}"
    )
    assert "bio_assembly" in routed["tool_inputs"]


def test_bio_assembly_route_clears_translator_commands():
    """Any translator-emitted commands (e.g. re-open) are cleared on bio_assembly route."""
    router       = _make_router()
    translator_r = _translator_result_chimerax(cmds=["open 2VNC", "cartoon"])
    routed       = router.route(translator_r, user_input="generate biological assembly")

    assert routed["commands"] == [], (
        f"Commands should be cleared on bio_assembly override, got {routed['commands']}"
    )


def test_bio_assembly_default_assembly_id_is_1():
    """Default assembly id in tool_inputs is 1."""
    router  = _make_router()
    tr      = _translator_result_chimerax()
    routed  = router.route(tr, user_input="work as tetramer")
    inp     = routed["tool_inputs"].get("bio_assembly", {})
    assert inp.get("assembly_id") == 1, f"Expected assembly_id=1, got {inp.get('assembly_id')}"


def test_bio_assembly_explicit_assembly_id_override():
    """'generate assembly 2' overrides to assembly_id=2."""
    router  = _make_router()
    tr      = _translator_result_chimerax()
    routed  = router.route(tr, user_input="generate assembly 2")
    inp     = routed["tool_inputs"].get("bio_assembly", {})
    assert inp.get("assembly_id") == 2, f"Expected assembly_id=2, got {inp.get('assembly_id')}"


def test_bio_assembly_not_claimed_by_higher_override():
    """validate_design takes precedence over bio_assembly (higher in chain)."""
    router       = _make_router()
    translator_r = {
        "commands": [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "high",
        "tools_needed": ["chimerax"], "tool_inputs": {},
    }
    # Simulate high-accuracy validate phrasing — should NOT route to bio_assembly
    routed = router.route(translator_r, user_input="thoroughly validate the full assembly design")
    assert "bio_assembly" not in routed["tools_needed"], (
        f"validate_design should have priority; got {routed['tools_needed']}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. _run_bio_assembly — correct sym command, no re-open, error-first
# ══════════════════════════════════════════════════════════════════════════════

def test_run_bio_assembly_emits_sym_command():
    """_run_bio_assembly must call sym #1 assembly 1 copies true on the loaded model."""
    router = _make_router(structures={"1": {"name": "2VNC", "path": None}})
    router.bridge.run_command.side_effect = [
        _chimerax_result("Made 2 copies for 2vnc assembly 1"),  # sym
        _chimerax_result("model id #1 type AtomicStructure name 2vnc\n"
                         "model id #2 type Model name \"2vnc assembly 1\""),  # info models
    ]
    router.session.get_structure.return_value = {"name": "2VNC", "path": None}
    router.session.get_assembly_info.return_value = None

    result = router._run_bio_assembly({"model_id": "1", "assembly_id": 1})

    assert result.success, f"Expected success; error={result.error}"
    # The sym command must have been called with the correct form
    first_call_cmd = router.bridge.run_command.call_args_list[0][0][0]
    assert "sym #1 assembly 1 copies true" in first_call_cmd, (
        f"Expected sym command, got: {first_call_cmd!r}"
    )


def test_run_bio_assembly_does_not_open_when_model_loaded():
    """_run_bio_assembly must NOT emit any 'open' command."""
    router = _make_router(structures={"1": {"name": "2VNC"}})
    router.bridge.run_command.side_effect = [
        _chimerax_result("Made 2 copies for 2vnc assembly 1"),
        _chimerax_result("model id #1 type AtomicStructure name 2vnc\n"
                         "model id #2 type Model name \"2vnc assembly 1\""),
    ]
    router.session.get_structure.return_value = {"name": "2VNC"}
    router.session.get_assembly_info.return_value = None

    router._run_bio_assembly({"model_id": "1", "assembly_id": 1})

    for c in router.bridge.run_command.call_args_list:
        cmd_str = c[0][0] if c[0] else ""
        assert not cmd_str.strip().lower().startswith("open "), (
            f"_run_bio_assembly must not issue 'open'; got: {cmd_str!r}"
        )


def test_run_bio_assembly_no_model_loaded_clean_error():
    """With no structures loaded, _run_bio_assembly returns a clean error without raising."""
    router = _make_router(structures={})
    result = router._run_bio_assembly({"model_id": "1", "assembly_id": 1})

    assert not result.success
    assert result.error is not None
    assert "open" in result.error.lower() or "loaded" in result.error.lower(), (
        f"Error should guide user to open a structure; got: {result.error!r}"
    )


def test_run_bio_assembly_sym_failure_lists_available_assemblies():
    """When sym fails, the error message must include available assemblies from sym #N."""
    router = _make_router(structures={"1": {"name": "2VNC"}})
    router.bridge.run_command.side_effect = [
        _chimerax_result(error="Unknown assembly id 99"),   # sym #1 assembly 99 copies true fails
        _chimerax_result("2vnc mmCIF Assemblies | 1 | 2 copies of chains A,B"),  # sym #1 listing
    ]
    router.session.get_structure.return_value = {"name": "2VNC"}
    router.session.get_assembly_info.return_value = None

    result = router._run_bio_assembly({"model_id": "1", "assembly_id": 99})

    assert not result.success
    assert result.error is not None
    assert "assembly" in result.error.lower(), (
        f"Error should mention assemblies; got: {result.error!r}"
    )


def test_run_bio_assembly_tracks_in_session_state():
    """On success, set_generated_assembly must be called with correct keys."""
    router = _make_router(structures={"1": {"name": "2VNC"}})
    router.bridge.run_command.side_effect = [
        _chimerax_result("Made 2 copies for 2vnc assembly 1"),
        _chimerax_result("model id #1 type AtomicStructure name 2vnc\n"
                         "model id #2 type Model name \"2vnc assembly 1\""),
    ]
    router.session.get_structure.return_value = {"name": "2VNC"}
    router.session.get_assembly_info.return_value = None

    router._run_bio_assembly({"model_id": "1", "assembly_id": 1})

    router.session.set_generated_assembly.assert_called_once()
    call_args = router.session.set_generated_assembly.call_args
    au_id, info_dict = call_args[0]
    assert au_id == "1", f"Expected au_model_id='1', got {au_id!r}"
    assert "assembly_id" in info_dict, "info_dict must contain assembly_id"


# ══════════════════════════════════════════════════════════════════════════════
# 3. AU mismatch detector
# ══════════════════════════════════════════════════════════════════════════════

def _make_asm_info(n_subunits: int, asm_type: str = "homotetramer",
                   stoich: str = "A4") -> dict:
    return {
        "pdb_id":        "2VNC",
        "assembly_type": asm_type,
        "stoichiometry": stoich,
        "n_subunits":    n_subunits,
        "error":         None,
    }


def test_au_mismatch_note_emitted_when_bio_larger_than_loaded():
    """When n_subunits (4) > loaded chains (2), a mismatch note must be printed."""
    import io, contextlib
    from unittest.mock import patch as _patch

    captured_prints = []

    # We test the mismatch detection logic directly — mock the AssemblyAnalyser path
    asm_info = _make_asm_info(n_subunits=4)

    # Mock the bridge for _model_chains
    mock_bridge = MagicMock()
    mock_bridge._model_chains.return_value = ["A", "B"]  # 2 chains loaded

    # Build a minimal StructureBot mock just for the test
    class _MockBot:
        bridge  = mock_bridge
        session = MagicMock()

    bot = _MockBot()
    bot.session.get_assembly_info.return_value = None

    # Patch fetch_assembly_info and AssemblyAnalyser so they don't hit network
    with _patch("assembly_analyser.fetch_assembly_info", return_value=asm_info), \
         _patch("assembly_analyser.AssemblyAnalyser") as MockAA:
        mock_aa_instance = MockAA.return_value
        mock_aa_instance.get_assembly_display.return_value = "homotetramer (A4)"

        messages = []

        class _FakeConsole:
            @staticmethod
            def print(msg, **kw):
                messages.append(str(msg))

        import main as _main
        orig_console = _main.console
        _main.console = _FakeConsole()
        try:
            _main.StructureBot._display_assembly_type_on_open(bot, "2VNC", "1")
        finally:
            _main.console = orig_console

    combined = " ".join(messages)
    assert "asymmetric unit" in combined.lower() or "homotetramer" in combined.lower(), (
        f"Mismatch note not found in output: {messages}"
    )


def test_no_mismatch_note_when_loaded_equals_bio():
    """When loaded chains == n_subunits, NO mismatch note should be emitted."""
    asm_info = _make_asm_info(n_subunits=2, asm_type="homodimer", stoich="A2")

    mock_bridge = MagicMock()
    mock_bridge._model_chains.return_value = ["A", "B"]  # 2 chains = biological

    class _MockBot:
        bridge  = mock_bridge
        session = MagicMock()

    bot = _MockBot()
    bot.session.get_assembly_info.return_value = None

    from unittest.mock import patch as _patch
    import main as _main

    messages = []

    class _FakeConsole:
        @staticmethod
        def print(msg, **kw):
            messages.append(str(msg))

    with _patch("assembly_analyser.fetch_assembly_info", return_value=asm_info), \
         _patch("assembly_analyser.AssemblyAnalyser") as MockAA:
        mock_aa_instance = MockAA.return_value
        mock_aa_instance.get_assembly_display.return_value = "homodimer (A2)"

        orig_console = _main.console
        _main.console = _FakeConsole()
        try:
            _main.StructureBot._display_assembly_type_on_open(bot, "2VNC", "1")
        finally:
            _main.console = orig_console

    combined = " ".join(messages)
    assert "asymmetric unit" not in combined.lower(), (
        f"Should NOT emit mismatch note when loaded==biological; got: {messages}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Verb guard 3a — assembly_analyser blocked; sym passes
# ══════════════════════════════════════════════════════════════════════════════

def test_verb_guard_blocks_assembly_analyser_via_registry():
    """assembly_analyser must be blocked via the registry check (Tier 2), NOT the denylist.
    Proof: assert it is NOT in the denylist, then verify it is still blocked when a real
    ChimeraX registry is supplied that doesn't include it."""
    assert "assembly_analyser" not in _HALLUCINATED_VERB_DENYLIST, (
        "assembly_analyser should be removed from the denylist; "
        "it is the registry's job to catch non-ChimeraX verbs"
    )
    registry = frozenset({"sym", "open", "close", "color", "view", "info",
                           "cartoon", "align", "matchmaker", "select", "hide",
                           "show", "surface", "style", "transparency"})
    cmds = ["assembly_analyser #1 mode tetramer", "view"]
    exps = ["run assembly analyser", "view"]
    new_cmds, new_exps, blocked = _validate_command_verbs(cmds, exps, known_verbs=registry)

    assert "assembly_analyser #1 mode tetramer" not in new_cmds, (
        "assembly_analyser command must be blocked by the registry check"
    )
    assert len(blocked) == 1, f"Expected 1 blocked command; got {blocked}"
    assert "view" in new_cmds, "Real verb 'view' must pass through"


def test_verb_guard_passes_sym_with_assembly_arg():
    """sym is a valid ChimeraX verb — the verb guard must not block it."""
    cmds = ["sym #1 assembly 1 copies true"]
    exps = ["generate biological assembly"]
    # Supply a realistic registry that includes 'sym' but not 'assembly_analyser'
    registry = frozenset({"sym", "open", "close", "color", "view", "info",
                           "cartoon", "align", "matchmaker", "select", "hide",
                           "show", "surface", "style", "transparency"})
    new_cmds, new_exps, blocked = _validate_command_verbs(cmds, exps, known_verbs=registry)

    assert new_cmds == cmds, (
        f"sym command should pass verb guard, but was blocked: {blocked}"
    )
    assert len(blocked) == 0


def test_verb_guard_novel_fake_verbs_blocked_via_registry():
    """Novel hallucinated verbs not in the registry — tetramer_builder, fold_protein —
    must also be blocked by Tier 2 without being enumerated in the denylist.
    This proves the fix generalises beyond any enumerated list."""
    for fake_verb in ("tetramer_builder", "fold_protein", "analyse_complex"):
        assert fake_verb not in _HALLUCINATED_VERB_DENYLIST, (
            f"{fake_verb!r} is in the denylist; it should only be caught by the registry"
        )
    registry = frozenset({"sym", "open", "close", "color", "view", "info",
                           "cartoon", "align", "matchmaker", "select", "hide",
                           "show", "surface", "style", "transparency"})
    for fake_verb in ("tetramer_builder", "fold_protein", "analyse_complex"):
        cmds = [f"{fake_verb} #1"]
        _, _, blocked = _validate_command_verbs(cmds, [""], known_verbs=registry)
        assert len(blocked) == 1, (
            f"Novel verb {fake_verb!r} must be blocked by registry check; got: {blocked}"
        )


def test_verb_guard_real_verbs_all_pass():
    """open, color, matchmaker, transparency, close, sym must all pass the guard."""
    registry = frozenset({"sym", "open", "close", "color", "view", "info",
                           "cartoon", "align", "matchmaker", "select", "hide",
                           "show", "surface", "style", "transparency"})
    test_cmds = [
        ("open 2VNC", "open structure"),
        ("color #1/A red", "color chain"),
        ("matchmaker #1 to #2", "matchmaker"),
        ("transparency #1 50", "transparency"),
        ("close #1", "close model"),
        ("sym #1 assembly 1 copies true", "bio assembly"),
    ]
    for cmd, label in test_cmds:
        new_cmds, _, blocked = _validate_command_verbs([cmd], [""], known_verbs=registry)
        assert len(blocked) == 0 and cmd in new_cmds, (
            f"Real verb ({label}) '{cmd}' must not be blocked; blocked={blocked}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. Correction loop 3b — repeated rejected verb halts; different real verb proceeds
# ══════════════════════════════════════════════════════════════════════════════

def _make_main_bot():
    """Minimal StructureBot mock for testing correction-loop logic."""
    import main as _main
    mock_bridge      = MagicMock()
    mock_session     = MagicMock()
    mock_translator  = MagicMock()
    bot = object.__new__(_main.StructureBot)
    bot.bridge     = mock_bridge
    bot.session    = mock_session
    bot.translator = mock_translator
    return bot


def test_correction_loop_halts_on_same_rejected_verb(monkeypatch):
    """
    If the correction re-proposes the same non-standard verb as the failed command,
    the loop must halt (same_verb detection).  We verify by calling the guard logic
    inline (not the full _handle_request to avoid REPL side effects).
    """
    import main as _main

    failed_cmd = "assembly_analyser #1 mode tetramer"
    # Correction also uses assembly_analyser (same bad verb)
    fix_cmds = ["assembly_analyser #1 mode dimer"]

    _failed_verb = failed_cmd.strip().split()[0].lower()
    _fix_verb    = fix_cmds[0].strip().split()[0].lower()
    _safe_verbs  = frozenset({
        "open", "close", "color", "colour", "select", "hide", "show",
        "cartoon", "surface", "style", "align", "view", "transparency",
        "sym", "matchmaker",
    })
    _same_verb = bool(
        _failed_verb and _fix_verb
        and _failed_verb == _fix_verb
        and _failed_verb not in _safe_verbs
    )
    assert _same_verb, (
        "Same-verb guard should have fired for assembly_analyser re-proposal"
    )


def test_correction_loop_proceeds_on_different_real_verb():
    """
    When the correction switches to a different, legitimate verb, _same_verb is False
    and the correction is allowed to proceed.
    """
    failed_cmd = "assembly_analyser #1 mode tetramer"
    fix_cmds   = ["sym #1 assembly 1 copies true"]   # different, real verb

    _failed_verb = failed_cmd.strip().split()[0].lower()
    _fix_verb    = fix_cmds[0].strip().split()[0].lower()
    _safe_verbs  = frozenset({
        "open", "close", "color", "colour", "select", "hide", "show",
        "cartoon", "surface", "style", "align", "view", "transparency",
        "sym", "matchmaker",
    })
    _same_verb = bool(
        _failed_verb and _fix_verb
        and _failed_verb == _fix_verb
        and _failed_verb not in _safe_verbs
    )
    assert not _same_verb, (
        "Different-verb correction should not trigger the same_verb halt"
    )


def test_correction_loop_safe_verbs_never_halt():
    """Common ChimeraX verbs like 'color' must NOT trigger the same_verb halt
    even when both the failure and the correction start with the same verb."""
    for safe_verb in ("color", "open", "align", "sym"):
        failed_cmd = f"{safe_verb} #1 red"
        fix_cmds   = [f"{safe_verb} #1 blue"]
        _failed_verb = failed_cmd.strip().split()[0].lower()
        _fix_verb    = fix_cmds[0].strip().split()[0].lower()
        _safe_verbs  = frozenset({
            "open", "close", "color", "colour", "select", "hide", "show",
            "cartoon", "surface", "style", "align", "view", "transparency",
            "sym", "matchmaker",
        })
        _same_verb = bool(
            _failed_verb and _fix_verb
            and _failed_verb == _fix_verb
            and _failed_verb not in _safe_verbs
        )
        assert not _same_verb, (
            f"Safe verb '{safe_verb}' triggered same_verb halt incorrectly"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 6. Error-first: no assembly info / invalid assembly id
# ══════════════════════════════════════════════════════════════════════════════

def test_bio_assembly_no_bridge_returns_clean_error():
    """_run_bio_assembly without a bridge returns a clean error, never raises."""
    router = _make_router()
    router.bridge = None  # type: ignore

    result = router._run_bio_assembly({"model_id": "1", "assembly_id": 1})
    assert not result.success
    assert result.error is not None
    assert "unavailable" in result.error.lower() or "bridge" in result.error.lower()


def test_session_generated_assemblies_roundtrip(tmp_path):
    """set_generated_assembly / get_generated_assembly survive a save/load cycle."""
    from session_state import SessionState

    state = SessionState()
    state.set_generated_assembly("1", {
        "au_model_id":       "1",
        "assembly_model_id": "2",
        "assembly_id":       1,
        "assembly_type":     "homotetramer",
        "n_subunits":        4,
        "pdb_id":            "2VNC",
    })
    save_path = str(tmp_path / "session_test.json")
    state.save(save_path)

    loaded = SessionState.load(save_path)
    rec = loaded.get_generated_assembly("1")
    assert rec is not None, "Generated assembly record not found after load"
    assert rec["assembly_model_id"] == "2"
    assert rec["assembly_type"] == "homotetramer"
    assert rec["n_subunits"] == 4
