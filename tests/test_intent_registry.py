"""
tests/test_intent_registry.py
-----------------------------
Tests for the Intent/Render separation framework (intent_registry.py) and its
integration with the routing + error-detection pipeline.

Test groups
-----------
1.  IntentRegistry — alias resolution (tier a)
2.  IntentRegistry — LLM constrained classifier (tier b): returns label only
3.  IntentRegistry — graceful miss (tier c)
4.  Render layer — each viewer intent → correct command sequence
5.  Render: "remove spheres" → hide atoms (NOT hide sphere)
6.  Render: "just cartoon" → hide atoms + show cartoons
7.  Category detection for pre-translate interception
8.  Routing integration — covered phrase → tools_needed=["representation"]
9.  Routing integration — uncovered phrase → chimerax (falls through)
10. Error detection — "Expected…" response → marked FAILED, not ✓
11. Sub-model spec handling (#2.1/A)
12. Verify guard — post-command probe
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from intent_registry import (
    IntentDef,
    IntentRegistry,
    VIEWER_REGISTRY,
    _probe_atom_count,
    is_representation_shaped,
    make_llm_classify_fn,
)
from tool_router import ToolRouter, ToolStepResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_router(structures: dict | None = None) -> ToolRouter:
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = structures if structures is not None else {
        "1": {"name": "2d31", "path": None}
    }
    mock_session.get_proteinmpnn_result.return_value = None
    mock_session.get_assembly_info.return_value = None
    mock_session.get_structure.return_value = {"name": "2d31", "path": None}
    return ToolRouter(bridge=mock_bridge, session=mock_session)


def _translator_stub(cmds: list | None = None) -> Dict[str, Any]:
    return {
        "commands":             cmds or [],
        "explanations":         [],
        "warnings":             [],
        "clarification_needed": None,
        "confidence":           "high",
        "tools_needed":         ["chimerax"],
        "tool_inputs":          {},
    }


def _cx_ok(value: str = "") -> dict:
    return {"value": value, "error": None}


def _cx_err(error: str) -> dict:
    return {"value": None, "error": error}


# ── 1. Alias resolution ────────────────────────────────────────────────────────

class TestAliasResolution:

    def test_exact_alias_cartoon_only(self):
        assert VIEWER_REGISTRY.resolve_alias("change to cartoon mode") == "view.cartoon_only"

    def test_exact_alias_cartoon_keyword(self):
        assert VIEWER_REGISTRY.resolve_alias("show as cartoon") == "view.cartoon_only"

    def test_exact_alias_ribbon(self):
        assert VIEWER_REGISTRY.resolve_alias("ribbon mode") == "view.cartoon_only"

    def test_exact_alias_just_cartoon(self):
        assert VIEWER_REGISTRY.resolve_alias("just cartoon") == "view.cartoon_only"

    def test_exact_alias_spacefill(self):
        assert VIEWER_REGISTRY.resolve_alias("show as spacefill") == "view.spacefill"

    def test_exact_alias_spheres(self):
        assert VIEWER_REGISTRY.resolve_alias("sphere mode") == "view.spacefill"

    def test_exact_alias_sticks(self):
        assert VIEWER_REGISTRY.resolve_alias("show as sticks") == "view.sticks"

    def test_exact_alias_ball_and_stick(self):
        assert VIEWER_REGISTRY.resolve_alias("balls and sticks") == "view.ball_and_stick"

    def test_exact_alias_surface(self):
        assert VIEWER_REGISTRY.resolve_alias("show as surface") == "view.surface"

    def test_exact_alias_hide_atoms(self):
        assert VIEWER_REGISTRY.resolve_alias("hide atoms") == "view.hide_atoms"

    def test_exact_alias_remove_spheres_maps_to_hide_atoms(self):
        # Critical: "remove spheres" must map to hide_atoms (not hide_sphere)
        assert VIEWER_REGISTRY.resolve_alias("remove spheres") == "view.hide_atoms"

    def test_exact_alias_hide_spheres_maps_to_hide_atoms(self):
        assert VIEWER_REGISTRY.resolve_alias("hide spheres") == "view.hide_atoms"

    def test_exact_alias_hide_cartoon(self):
        assert VIEWER_REGISTRY.resolve_alias("hide cartoon") == "view.hide_cartoon"

    def test_exact_alias_hide_ribbon(self):
        assert VIEWER_REGISTRY.resolve_alias("hide ribbon") == "view.hide_cartoon"

    def test_exact_alias_hide_surface(self):
        assert VIEWER_REGISTRY.resolve_alias("hide surface") == "view.hide_surface"

    def test_exact_alias_show_atoms(self):
        assert VIEWER_REGISTRY.resolve_alias("show atoms") == "view.show_atoms"

    def test_case_insensitive(self):
        assert VIEWER_REGISTRY.resolve_alias("CARTOON MODE") == "view.cartoon_only"
        assert VIEWER_REGISTRY.resolve_alias("Show As Cartoon") == "view.cartoon_only"

    def test_no_match_returns_none(self):
        assert VIEWER_REGISTRY.resolve_alias("mutate chain A") is None
        assert VIEWER_REGISTRY.resolve_alias("") is None
        assert VIEWER_REGISTRY.resolve_alias("open 2d31") is None

    def test_paraphrase_not_in_alias_returns_none(self):
        # A true paraphrase with no alias substring match → None; goes to LLM tier
        assert VIEWER_REGISTRY.resolve_alias("display the protein as a ribbon diagram") is None
        assert VIEWER_REGISTRY.resolve_alias("make the molecules invisible except backbone") is None

    def test_strip_back_to_ribbon_resolves_via_just_the_ribbon_alias(self):
        # "strip it back to just the ribbon" contains "just the ribbon" (alias)
        # → deterministic alias resolution, no LLM needed
        key = VIEWER_REGISTRY.resolve_alias("strip it back to just the ribbon")
        assert key == "view.cartoon_only"

    def test_just_the_ribbon_resolves(self):
        assert VIEWER_REGISTRY.resolve_alias("just the ribbon") == "view.cartoon_only"

    def test_back_to_cartoon_resolves(self):
        assert VIEWER_REGISTRY.resolve_alias("back to cartoon") == "view.cartoon_only"


# ── 2. LLM constrained classifier ─────────────────────────────────────────────

class TestLLMClassifier:

    def test_llm_returns_valid_label(self):
        """LLM fn returning a valid label → accepted for a true paraphrase."""
        def mock_llm(text, labels):
            return "view.cartoon_only"

        # "display the protein as a ribbon diagram" — no alias match, goes to LLM tier
        key, method = VIEWER_REGISTRY.resolve(
            "display the protein as a ribbon diagram",
            llm_classify_fn=mock_llm,
        )
        assert key == "view.cartoon_only"
        assert method == "llm"

    def test_llm_must_return_label_not_syntax(self):
        """LLM fn returning ChimeraX syntax → rejected (not a valid label key)."""
        def bad_llm(text, labels):
            return "hide #1 atoms"   # syntax, not a label

        key, method = VIEWER_REGISTRY.resolve(
            "display the protein as a ribbon diagram",
            llm_classify_fn=bad_llm,
        )
        # "hide #1 atoms" is not in the intent registry → treated as miss
        assert key is None
        assert method == "miss"

    def test_llm_returning_none_falls_through_to_miss(self):
        """LLM fn returning None → graceful miss."""
        def no_match_llm(text, labels):
            return None

        key, method = VIEWER_REGISTRY.resolve(
            "make it sparkly",
            llm_classify_fn=no_match_llm,
        )
        assert key is None
        assert method == "miss"

    def test_llm_exception_falls_through_to_miss(self):
        """LLM fn raising exception → graceful miss (no crash)."""
        def crashing_llm(text, labels):
            raise RuntimeError("API down")

        key, method = VIEWER_REGISTRY.resolve(
            "do something with the display",
            llm_classify_fn=crashing_llm,
        )
        assert key is None
        assert method == "miss"

    def test_alias_takes_precedence_over_llm(self):
        """Alias match fires first; LLM is never called for listed phrases."""
        called = []

        def spy_llm(text, labels):
            called.append(text)
            return "view.spacefill"

        key, method = VIEWER_REGISTRY.resolve("cartoon mode", llm_classify_fn=spy_llm)
        assert key == "view.cartoon_only"
        assert method == "alias"
        assert called == []   # LLM was NOT invoked

    def test_llm_receives_label_list_not_syntax(self):
        """The labels passed to LLM classifier are intent keys, not ChimeraX syntax."""
        received_labels: List[str] = []

        def capture_llm(text, labels):
            received_labels.extend(labels)
            return labels[0] if labels else None

        VIEWER_REGISTRY.resolve("something ambiguous", llm_classify_fn=capture_llm)
        # All labels must be intent keys (contain "."), not ChimeraX commands
        for lbl in received_labels:
            assert "." in lbl, f"Label {lbl!r} looks like syntax, not an intent key"
            assert " " not in lbl, f"Label {lbl!r} should be a single token key"


# ── 3. Graceful miss ──────────────────────────────────────────────────────────

class TestGracefulMiss:

    def test_miss_when_no_alias_no_llm(self):
        key, method = VIEWER_REGISTRY.resolve("make it prettier")
        assert key is None
        assert method == "miss"

    def test_graceful_miss_message_contains_intents(self):
        msg = VIEWER_REGISTRY.graceful_miss_message("make it sparkly", "view")
        assert "make it sparkly" in msg
        assert "view.cartoon_only" in msg
        assert "view.spacefill"    in msg
        assert "view.surface"      in msg
        assert "rephrase" in msg.lower()

    def test_graceful_miss_message_does_not_contain_chimerax_syntax(self):
        msg = VIEWER_REGISTRY.graceful_miss_message("??", "view")
        # Should not contain raw ChimeraX commands
        assert "hide #" not in msg
        assert "show #" not in msg
        assert "style #" not in msg


# ── 4. Render layer — command sequences ───────────────────────────────────────

class TestRenderLayer:

    def test_cartoon_only_commands(self):
        cmds = VIEWER_REGISTRY.render("view.cartoon_only", "#1")
        assert "hide #1 atoms"    in cmds
        assert "show #1 cartoons" in cmds
        assert "~surface #1"      in cmds

    def test_spacefill_commands(self):
        cmds = VIEWER_REGISTRY.render("view.spacefill", "#1")
        assert "show #1 atoms"    in cmds
        assert "style #1 sphere"  in cmds

    def test_sticks_commands(self):
        cmds = VIEWER_REGISTRY.render("view.sticks", "#1")
        assert "show #1 atoms"   in cmds
        assert "style #1 stick"  in cmds

    def test_ball_and_stick_commands(self):
        cmds = VIEWER_REGISTRY.render("view.ball_and_stick", "#1")
        assert "show #1 atoms"  in cmds
        assert "style #1 ball"  in cmds

    def test_surface_commands(self):
        cmds = VIEWER_REGISTRY.render("view.surface", "#1")
        assert "surface #1" in cmds

    def test_hide_atoms_commands(self):
        cmds = VIEWER_REGISTRY.render("view.hide_atoms", "#1")
        assert "hide #1 atoms" in cmds

    def test_hide_cartoon_commands(self):
        cmds = VIEWER_REGISTRY.render("view.hide_cartoon", "#1")
        assert "hide #1 cartoons" in cmds

    def test_hide_surface_commands(self):
        cmds = VIEWER_REGISTRY.render("view.hide_surface", "#1")
        assert "~surface #1" in cmds

    def test_show_atoms_commands(self):
        cmds = VIEWER_REGISTRY.render("view.show_atoms", "#1")
        assert "show #1 atoms" in cmds

    def test_unknown_intent_raises_key_error(self):
        with pytest.raises(KeyError):
            VIEWER_REGISTRY.render("view.does_not_exist", "#1")


# ── 5. "remove spheres" → hide atoms (NOT hide sphere) ───────────────────────

class TestRemoveSpheres:

    def test_remove_spheres_alias_maps_to_hide_atoms(self):
        key = VIEWER_REGISTRY.resolve_alias("remove spheres")
        assert key == "view.hide_atoms"

    def test_remove_spheres_render_is_hide_atoms_not_sphere(self):
        key = VIEWER_REGISTRY.resolve_alias("remove spheres")
        cmds = VIEWER_REGISTRY.render(key, "#1")
        assert "hide #1 atoms" in cmds
        # Must NOT contain "hide #1 sphere" (sphere is a style, not a hide target)
        assert "hide #1 sphere" not in cmds

    def test_hide_spheres_does_not_emit_hide_sphere(self):
        cmds = VIEWER_REGISTRY.render("view.hide_atoms", "#1")
        for cmd in cmds:
            assert "sphere" not in cmd, (
                f"Command {cmd!r} references 'sphere' — sphere is a STYLE target, "
                "not a valid hide target"
            )


# ── 6. "just cartoon" → hide atoms + show cartoons ───────────────────────────

class TestJustCartoon:

    def test_just_cartoon_resolves_to_cartoon_only(self):
        key = VIEWER_REGISTRY.resolve_alias("just cartoon")
        assert key == "view.cartoon_only"

    def test_cartoon_only_hides_atoms_and_shows_cartoons(self):
        cmds = VIEWER_REGISTRY.render("view.cartoon_only", "#1")
        assert "hide #1 atoms"    in cmds
        assert "show #1 cartoons" in cmds

    def test_cartoon_command_alone_is_not_used(self):
        """The broken 'cartoon #1' pattern (leaves atoms visible) is NOT emitted."""
        cmds = VIEWER_REGISTRY.render("view.cartoon_only", "#1")
        # No bare "cartoon #1" — that doesn't hide atoms
        assert "cartoon #1" not in cmds
        assert cmds[0].startswith("hide"), (
            "cartoon_only must FIRST hide atoms, not just show cartoons"
        )


# ── 7. Category detection ─────────────────────────────────────────────────────

class TestCategoryDetection:

    def test_detects_cartoon(self):
        assert VIEWER_REGISTRY.detect_category_phrase("change to cartoon mode") is True

    def test_detects_ribbon(self):
        assert VIEWER_REGISTRY.detect_category_phrase("strip it back to just the ribbon") is True

    def test_detects_spheres(self):
        assert VIEWER_REGISTRY.detect_category_phrase("remove spheres") is True

    def test_detects_show_surface(self):
        assert VIEWER_REGISTRY.detect_category_phrase("show as surface") is True

    def test_detects_hide_atoms(self):
        assert VIEWER_REGISTRY.detect_category_phrase("hide atoms") is True

    def test_detects_spacefill(self):
        assert VIEWER_REGISTRY.detect_category_phrase("spacefill") is True

    def test_does_not_detect_mutation_request(self):
        assert VIEWER_REGISTRY.detect_category_phrase("mutate chain A") is False

    def test_does_not_detect_open_request(self):
        assert VIEWER_REGISTRY.detect_category_phrase("open 2d31") is False

    def test_does_not_detect_interface_request(self):
        assert VIEWER_REGISTRY.detect_category_phrase("analyse the interface contacts") is False

    def test_surface_area_request_is_now_flagged(self):
        # verb "show" + noun "surface" → True (surface-area queries route via rep tier)
        assert VIEWER_REGISTRY.detect_category_phrase("show the surface area contacts") is True

    def test_does_not_detect_esm_request(self):
        assert VIEWER_REGISTRY.detect_category_phrase("run esm conservation") is False

    def test_is_covered_phrase_true_for_alias(self):
        assert VIEWER_REGISTRY.is_covered_phrase("cartoon mode") is True

    def test_is_covered_phrase_false_for_non_repr(self):
        assert VIEWER_REGISTRY.is_covered_phrase("run a mutation scan") is False


# ── 8. Routing integration — covered phrase ───────────────────────────────────

class TestRoutingIntegration:

    def test_covered_phrase_routes_to_representation_tool(self):
        router = _make_router()
        stub   = _translator_stub(cmds=["cartoon #1"])   # wrong command from translator
        result = router.route(stub, user_input="change to cartoon mode")
        assert "representation" in result.get("tools_needed", [])
        assert result.get("has_extra_tools") is True

    def test_representation_tool_clears_translator_commands(self):
        """Translator-emitted commands are discarded for covered intents."""
        router = _make_router()
        stub   = _translator_stub(cmds=["hide #1 sphere", "cartoon #1"])
        result = router.route(stub, user_input="remove spheres")
        assert result.get("commands", []) == []
        assert "representation" in result.get("tools_needed", [])

    def test_intent_key_populated_for_alias_match(self):
        router = _make_router()
        stub   = _translator_stub()
        result = router.route(stub, user_input="show as sticks")
        tinputs = result.get("tool_inputs", {})
        assert "representation" in tinputs
        assert tinputs["representation"]["intent_key"] == "view.sticks"

    def test_intent_key_none_for_paraphrase_needing_llm(self):
        """Paraphrase not in alias list → intent_key=None; LLM fires in execute()."""
        router = _make_router()
        stub   = _translator_stub()
        # "display the protein as a ribbon diagram" → contains "ribbon" → category detected
        # but no specific alias matches → intent_key=None (LLM tier in execute)
        result = router.route(stub, user_input="display the protein as a ribbon diagram")
        tinputs = result.get("tool_inputs", {})
        assert "representation" in result.get("tools_needed", [])
        assert tinputs.get("representation", {}).get("intent_key") is None


# ── 9. Routing integration — uncovered phrase falls through ───────────────────

class TestUncoveredFallthrough:

    def test_mutation_request_does_not_intercept(self):
        router = _make_router()
        stub   = _translator_stub(cmds=["mutation_scan"])
        result = router.route(stub, user_input="suggest mutations to improve solubility")
        assert "representation" not in result.get("tools_needed", [])

    def test_open_request_does_not_intercept(self):
        router = _make_router()
        stub   = _translator_stub(cmds=["open 2d31"])
        result = router.route(stub, user_input="open 2d31")
        assert "representation" not in result.get("tools_needed", [])

    def test_interface_request_does_not_intercept(self):
        router = _make_router()
        stub   = _translator_stub()
        stub["tools_needed"] = ["interface_stabilization"]
        result = router.route(stub, user_input="stabilise the interface")
        assert "representation" not in result.get("tools_needed", [])


# ── 10. Error detection ───────────────────────────────────────────────────────

class TestErrorDetection:

    def _bridge_with_response(self, body: str) -> "ChimeraXBridge":
        """Create a properly initialized bridge with a mocked REST response."""
        from chimerax_bridge import ChimeraXBridge
        import requests as _requests

        bridge = ChimeraXBridge(chimerax_path="X", port=60001)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = body
        mock_resp.json.side_effect = ValueError("not JSON")

        return bridge, mock_resp

    def test_expected_error_prefix_is_detected(self):
        """ChimeraX 'Expected …' responses must be treated as errors, not successes."""
        from chimerax_bridge import ChimeraXBridge
        import requests as _req

        error_text = (
            "Expected a collection of one of 'atoms', 'bonds', 'cartoons', "
            "'models', 'pbonds', 'pseudobonds', 'ribbons', or 'surfaces' or a keyword"
        )
        bridge, mock_resp = self._bridge_with_response(error_text)

        with patch.object(bridge, "is_running", return_value=True), \
             patch("requests.get", return_value=mock_resp):
            result = bridge._run_command_once("hide #1 sphere")

        assert result.get("error") is not None, (
            "'Expected …' response must set error, not pass as success"
        )
        assert "Expected" in result["error"]

    def test_hide_sphere_produces_expected_error(self):
        """
        'hide #1 sphere' is a common mistaken command (sphere is a style, not a
        hide target).  ChimeraX returns 'Expected a collection …' — must be FAILED.
        """
        from chimerax_bridge import ChimeraXBridge

        error_text = (
            "Expected a collection of one of 'atoms', 'bonds', 'cartoons', "
            "'models', 'pbonds', 'pseudobonds', 'ribbons', or 'surfaces' or a keyword"
        )
        bridge, mock_resp = self._bridge_with_response(error_text)

        with patch.object(bridge, "is_running", return_value=True), \
             patch("requests.get", return_value=mock_resp):
            result = bridge._run_command_once("hide #1 sphere")

        assert result.get("error") is not None
        assert "Expected" in result["error"]

    def test_unknown_command_still_detected(self):
        """'Unknown command:' prefix remains detected after the Expected fix."""
        from chimerax_bridge import ChimeraXBridge

        bridge, mock_resp = self._bridge_with_response(
            "Unknown command: totally_fake_command #1"
        )

        with patch.object(bridge, "is_running", return_value=True), \
             patch("requests.get", return_value=mock_resp):
            result = bridge._run_command_once("totally_fake_command #1")

        assert result.get("error") is not None


# ── 11. Sub-model spec handling ───────────────────────────────────────────────

class TestSubModelSpec:

    def test_render_with_submodel_spec(self):
        """Commands must use the exact spec including sub-model / chain."""
        cmds = VIEWER_REGISTRY.render("view.cartoon_only", "#2.1/A")
        assert "hide #2.1/A atoms"    in cmds
        assert "show #2.1/A cartoons" in cmds
        assert "~surface #2.1/A"      in cmds

    def test_render_with_assembly_model(self):
        cmds = VIEWER_REGISTRY.render("view.spacefill", "#2")
        assert "show #2 atoms"   in cmds
        assert "style #2 sphere" in cmds

    def test_route_uses_primary_model_id(self):
        """route() populates the spec from session's primary model."""
        router = _make_router(structures={"1": {"name": "2d31"}, "2": {"name": "asm"}})
        stub   = _translator_stub()
        result = router.route(stub, user_input="show as sticks")
        assert "representation" in result.get("tools_needed", [])


# ── 12. Verify guard ──────────────────────────────────────────────────────────

class TestVerifyGuard:

    def test_verify_atoms_hidden_after_cartoon_only(self):
        mock_bridge = MagicMock()
        # Simulate: atoms_shown=0 → correct after hide atoms
        mock_bridge.run_command.return_value = {"value": "0", "error": None}

        result = VIEWER_REGISTRY.verify("view.cartoon_only", "#1", mock_bridge)
        assert result is True

    def test_verify_atoms_still_shown_after_cartoon_only_is_false(self):
        mock_bridge = MagicMock()
        # Simulate: atoms_shown=6124 → state did NOT change
        mock_bridge.run_command.return_value = {"value": "6124", "error": None}

        result = VIEWER_REGISTRY.verify("view.cartoon_only", "#1", mock_bridge)
        assert result is False

    def test_verify_atoms_shown_after_spacefill(self):
        mock_bridge = MagicMock()
        mock_bridge.run_command.return_value = {"value": "6124", "error": None}

        result = VIEWER_REGISTRY.verify("view.spacefill", "#1", mock_bridge)
        assert result is True

    def test_verify_none_for_intent_without_verify(self):
        result = VIEWER_REGISTRY.verify("view.sticks", "#1", MagicMock())
        assert result is None   # sticks has no verify_fn registered

    def test_verify_none_on_probe_failure(self):
        mock_bridge = MagicMock()
        mock_bridge.run_command.side_effect = RuntimeError("bridge down")

        result = VIEWER_REGISTRY.verify("view.cartoon_only", "#1", mock_bridge)
        assert result is None   # probe failure → None, not crash


# ── 13. _run_representation integration ──────────────────────────────────────

class TestRunRepresentation:

    def setup_method(self):
        import tool_router
        tool_router._repr_classify_fn = None

    def _router_with_bridge(self, run_command_results: list) -> ToolRouter:
        mock_bridge  = MagicMock()
        mock_bridge.run_command.side_effect = run_command_results
        mock_session = MagicMock()
        mock_session.structures = {"1": {"name": "2d31"}}
        mock_session.get_structure.return_value = {"name": "2d31"}
        return ToolRouter(bridge=mock_bridge, session=mock_session)

    def test_alias_resolved_commands_execute(self):
        """cartoon_only → 3 commands execute, all return ok."""
        ok   = _cx_ok("")
        # 3 commands + 1 runscript verify probe
        router = self._router_with_bridge([ok, ok, ok, _cx_ok("0")])

        result = router._run_representation(
            {"_user_input": "cartoon mode", "intent_key": "view.cartoon_only"},
        )
        assert result.success is True
        assert "cartoon" in result.summary.lower()

    def test_failed_command_returns_failed_result(self):
        """If a command returns an error, the tool returns failure."""
        err    = _cx_err("Expected a collection of …")
        # _cx_ok("") is consumed by the pre-render snapshot probe
        router = self._router_with_bridge([_cx_ok(""), err])

        result = router._run_representation(
            {"_user_input": "hide spheres", "intent_key": "view.hide_atoms"},
        )
        # view.hide_atoms renders to "hide #1 atoms" (not sphere),
        # so this simulates an unexpected error response
        assert result.success is False
        assert result.error is not None

    def test_llm_fallback_called_when_no_alias(self):
        """When intent_key=None, LLM classifier is invoked."""
        ok     = _cx_ok("")
        router = self._router_with_bridge([ok, ok, ok, _cx_ok("0")])

        with patch("intent_registry.make_llm_classify_fn") as mock_factory:
            mock_factory.return_value = lambda t, ls: "view.cartoon_only"
            result = router._run_representation(
                {"_user_input": "display the protein as a ribbon diagram", "intent_key": None},
            )
        assert result.success is True
        assert result.data.get("resolution") == "llm"

    def test_graceful_miss_when_llm_returns_none(self):
        """LLM returning None → graceful miss, success=False."""
        router = self._router_with_bridge([])

        with patch("intent_registry.make_llm_classify_fn") as mock_factory:
            mock_factory.return_value = lambda t, ls: None
            result = router._run_representation(
                {"_user_input": "make it look sparkly", "intent_key": None},
            )
        assert result.success is False
        assert "available" in result.error.lower()

    def test_no_bridge_returns_error(self):
        mock_session = MagicMock()
        mock_session.structures = {"1": {"name": "2d31"}}
        router = ToolRouter(bridge=None, session=mock_session)

        result = router._run_representation(
            {"_user_input": "cartoon mode", "intent_key": "view.cartoon_only"},
        )
        assert result.success is False
        assert "bridge" in result.error.lower()


# ── 14. Classifier backend — Ollama path, think flag, fallback ────────────────

class TestClassifierBackend:

    def test_ollama_path_puts_think_false_at_top_level(self):
        """Ollama request must have `think: false` at top level, NOT inside options."""
        captured: list = []

        def fake_post(url, json=None, **kw):
            captured.append(json)
            resp = MagicMock()
            resp.json.return_value = {"response": "view.cartoon_only"}
            return resp

        with patch("requests.post", side_effect=fake_post):
            fn = make_llm_classify_fn(backend_name="ollama")
            labels = VIEWER_REGISTRY.list_intent_keys("view")
            fn("change all chain views to cartoon", labels)

        assert captured, "requests.post was never called"
        body = captured[0]
        assert body.get("think") is False, (
            "`think: false` must be at the TOP LEVEL of the Ollama request JSON, "
            "not inside options — qwen3 ignores it otherwise"
        )
        options = body.get("options", {})
        assert "think" not in options, (
            "`think` must not appear inside options{} — only at the top level"
        )

    def test_ollama_path_uses_num_predict_60(self):
        """num_predict must be >= 60 to avoid truncation on longer intent keys."""
        captured: list = []

        def fake_post(url, json=None, **kw):
            captured.append(json)
            resp = MagicMock()
            resp.json.return_value = {"response": "view.hide_atoms"}
            return resp

        with patch("requests.post", side_effect=fake_post):
            fn = make_llm_classify_fn(backend_name="ollama")
            fn("remove all spheres", VIEWER_REGISTRY.list_intent_keys("view"))

        body = captured[0]
        assert body["options"]["num_predict"] >= 60

    def test_classifier_falls_back_to_ollama_when_claude_raises(self):
        """If the Claude path raises (e.g. API cap), Ollama is tried as fallback."""
        ollama_calls: list = []

        def fake_post(url, json=None, **kw):
            ollama_calls.append(url)
            resp = MagicMock()
            resp.json.return_value = {"response": "view.hide_atoms"}
            return resp

        with patch("anthropic.Anthropic") as mock_anthropic, \
             patch("requests.post", side_effect=fake_post):
            mock_anthropic.return_value.messages.create.side_effect = RuntimeError(
                "monthly usage limit reached"
            )
            fn = make_llm_classify_fn(backend_name="claude")
            labels = VIEWER_REGISTRY.list_intent_keys("view")
            result = fn("remove all spheres", labels)

        assert ollama_calls, "Ollama fallback was never called after Claude failure"
        assert result == "view.hide_atoms"

    def test_uncovered_response_yields_graceful_miss(self):
        """LLM returning 'uncovered' means representation-category but no match → miss."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "uncovered"}
            fn = make_llm_classify_fn(backend_name="ollama")
            result = fn("make the residues flicker", VIEWER_REGISTRY.list_intent_keys("view"))
        # "uncovered" is NOT a valid intent key → resolve() treats it as miss
        key, method = VIEWER_REGISTRY.resolve(
            "make the residues flicker",
            llm_classify_fn=lambda t, ls: result,
        )
        assert key is None
        assert method == "miss"

    def test_none_response_yields_graceful_miss(self):
        """LLM returning 'none' (not a representation request) → miss."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "none"}
            fn = make_llm_classify_fn(backend_name="ollama")
            result = fn("calculate molecular weight", VIEWER_REGISTRY.list_intent_keys("view"))
        key, method = VIEWER_REGISTRY.resolve(
            "calculate molecular weight",
            llm_classify_fn=lambda t, ls: result,
        )
        assert key is None
        assert method == "miss"


# ── 15. Category detection — expanded triggers ────────────────────────────────

class TestCategoryDetectionExpanded:

    def test_detects_remove_all_spheres_atoms(self):
        """'remove all spheres / atoms' must be caught by the category gate."""
        assert VIEWER_REGISTRY.detect_category_phrase("remove all spheres / atoms") is True

    def test_detects_all_spheres(self):
        assert VIEWER_REGISTRY.detect_category_phrase("hide all spheres now") is True

    def test_detects_all_atoms(self):
        assert VIEWER_REGISTRY.detect_category_phrase("remove all atoms") is True

    def test_detects_strip_it_back(self):
        """'strip it back' (bare undo phrase) must be caught."""
        assert VIEWER_REGISTRY.detect_category_phrase("strip it back") is True

    def test_detects_undo_that(self):
        assert VIEWER_REGISTRY.detect_category_phrase("undo that") is True

    def test_detects_put_it_back(self):
        assert VIEWER_REGISTRY.detect_category_phrase("put it back") is True

    def test_does_not_detect_molecular_weight(self):
        assert VIEWER_REGISTRY.detect_category_phrase("calculate molecular weight") is False

    def test_surface_area_request_flagged_after_expansion(self):
        """verb 'show' + noun 'surface' → True after FIX 2 (verb-required tier)."""
        assert VIEWER_REGISTRY.detect_category_phrase("show the surface area contacts") is True


# ── 16. Undo/revert representation ────────────────────────────────────────────

class TestUndoRepresentation:

    def _router_with_bridge(self, run_command_results: list) -> ToolRouter:
        mock_bridge  = MagicMock()
        mock_bridge.run_command.side_effect = run_command_results
        mock_session = MagicMock()
        mock_session.structures = {"1": {"name": "2d31"}}
        mock_session.get_structure.return_value = {"name": "2d31"}
        return ToolRouter(bridge=mock_bridge, session=mock_session)

    def test_undo_intent_registered(self):
        """view.undo_representation must exist in the registry."""
        defn = VIEWER_REGISTRY.get_defn("view.undo_representation")
        assert defn is not None
        assert defn.category == "view"

    def test_undo_aliases_include_strip_it_back(self):
        key = VIEWER_REGISTRY.resolve_alias("strip it back")
        assert key == "view.undo_representation"

    def test_undo_aliases_include_undo_that(self):
        key = VIEWER_REGISTRY.resolve_alias("undo that")
        assert key == "view.undo_representation"

    def test_undo_aliases_include_put_it_back(self):
        key = VIEWER_REGISTRY.resolve_alias("put it back")
        assert key == "view.undo_representation"

    def test_undo_no_prior_state_returns_graceful_miss(self):
        """Undo with no snapshot recorded → success=False, not a crash."""
        router = self._router_with_bridge([])
        result = router._run_representation(
            {"_user_input": "strip it back", "intent_key": "view.undo_representation"},
        )
        assert result.success is False
        assert "no prior" in result.error.lower()

    def test_undo_with_snapshot_restores_and_clears(self):
        """After a render, undo executes the restore commands and clears the snapshot."""
        ok = _cx_ok("")
        # snapshot probe (returns atom/cartoon state), render cmd, verify probe
        # then undo restore cmds (2: hide atoms + hide cartoons)
        snapshot_response = _cx_ok("False,True,1")   # atoms hidden, cartoon shown, stick
        router = self._router_with_bridge([
            snapshot_response,   # pre-render snapshot probe
            ok,                  # hide #1 atoms (render cmd)
            _cx_ok("0"),         # verify probe
            ok,                  # hide #1 atoms (restore cmd 1 from snapshot)
            ok,                  # hide #1 cartoons (restore cmd 2 from snapshot)
        ])

        # First: render a change that triggers a snapshot
        result1 = router._run_representation(
            {"_user_input": "hide atoms", "intent_key": "view.hide_atoms"},
        )
        assert result1.success is True

        # Snapshot should be stored for "#1"
        assert "#1" in router._repr_snapshots
        restore_cmds = router._repr_snapshots["#1"]
        # atoms were False → restore is "hide #1 atoms"; cartoon was True → "show #1 cartoons"
        assert any("hide" in c and "atoms" in c for c in restore_cmds)

        # Now: undo
        result2 = router._run_representation(
            {"_user_input": "strip it back", "intent_key": "view.undo_representation"},
        )
        assert result2.success is True
        assert "reverted" in result2.summary.lower()

        # Snapshot cleared after undo
        assert "#1" not in router._repr_snapshots

    def test_snapshot_repr_parses_probe_output(self):
        """_snapshot_repr returns correct restore commands from probe output."""
        mock_bridge = MagicMock()
        # atoms shown (True), cartoon shown (True), draw mode 0 (sphere)
        mock_bridge.run_command.return_value = _cx_ok("True,True,0")

        from tool_router import ToolRouter
        mock_session = MagicMock()
        mock_session.structures = {"1": {}}
        router = ToolRouter(bridge=mock_bridge, session=mock_session)
        cmds = router._snapshot_repr("#1", mock_bridge)

        assert any("show #1 atoms" in c for c in cmds)
        assert any("sphere" in c for c in cmds)   # style sphere preserved
        assert any("show #1 cartoons" in c for c in cmds)

    def test_snapshot_repr_returns_empty_on_bad_probe(self):
        """_snapshot_repr silently returns [] if probe output is unparseable."""
        mock_bridge = MagicMock()
        mock_bridge.run_command.return_value = _cx_ok("unparseable garbage")
        mock_session = MagicMock()
        mock_session.structures = {"1": {}}
        router = ToolRouter(bridge=mock_bridge, session=mock_session)
        cmds = router._snapshot_repr("#1", mock_bridge)
        assert cmds == []

    def test_snapshot_repr_returns_empty_when_bridge_none(self):
        mock_session = MagicMock()
        mock_session.structures = {"1": {}}
        router = ToolRouter(bridge=None, session=mock_session)
        cmds = router._snapshot_repr("#1", None)
        assert cmds == []


# ── 17. Structural guard — render layer never emits malformed commands ─────────

class TestStructuralGuard:

    def test_no_intent_renders_hide_sphere(self):
        """No registered intent may render 'hide ... sphere(s)' — sphere is a style target."""
        for key in VIEWER_REGISTRY.list_intent_keys("view"):
            cmds = VIEWER_REGISTRY.render(key, "#1")
            for cmd in cmds:
                lower = cmd.lower()
                assert "hide #1 sphere" not in lower, (
                    f"Intent {key!r} emits {cmd!r} — 'sphere' is a STYLE target, "
                    "not a valid collection for hide/show"
                )

    def test_hide_atoms_never_emits_hide_sphere(self):
        """Critical regression guard: view.hide_atoms must emit 'hide #1 atoms'."""
        cmds = VIEWER_REGISTRY.render("view.hide_atoms", "#1")
        assert "hide #1 atoms" in cmds
        assert not any("sphere" in c for c in cmds)

    def test_undo_render_fn_returns_empty(self):
        """Undo render_fn returns [] — the actual execution is in _run_representation."""
        cmds = VIEWER_REGISTRY.render("view.undo_representation", "#1")
        assert cmds == []

    def test_remove_all_spheres_atoms_routes_to_representation(self):
        """'remove all spheres / atoms' must be intercepted as a representation request."""
        router = _make_router()
        stub   = _translator_stub(cmds=["hide #1 spheres"])  # malformed from free-translation
        result = router.route(stub, user_input="remove all spheres / atoms")
        # Must be caught by category gate, NOT fall through to free-translation
        assert "representation" in result.get("tools_needed", [])
        assert result.get("commands", []) == []   # malformed translator cmd discarded


# ── 18. Noun-floor category detection ────────────────────────────────────────

class TestNounFloorCategoryDetection:
    """
    Verify the structural noun-floor catches informal representation phrases that
    the old literal-phrase list missed.  These are the user's failing test cases.
    Also verifies that non-representation phrases are NOT over-blocked.
    """

    # -- Formerly failing phrases that the noun-floor now catches ----------------

    def test_get_rid_of_balls(self):
        """'balls' is an unambiguous rep noun → True even without a standard verb."""
        assert VIEWER_REGISTRY.detect_category_phrase("get rid of the balls on chain A") is True

    def test_lose_the_sticks(self):
        """'sticks' is an unambiguous rep noun → True."""
        assert VIEWER_REGISTRY.detect_category_phrase("lose the sticks") is True

    def test_ditch_the_surface(self):
        """'surface' with no analysis-context qualifier → True."""
        assert VIEWER_REGISTRY.detect_category_phrase("ditch the surface") is True

    # -- Authoritative alias match (1a) overrides noun-floor --------------------

    def test_alias_match_is_authoritative_strip_it_back(self):
        """Alias match fires even if the phrase has no rep noun ('strip it back')."""
        assert VIEWER_REGISTRY.detect_category_phrase("strip it back") is True

    def test_alias_match_is_authoritative_undo_that(self):
        assert VIEWER_REGISTRY.detect_category_phrase("undo that") is True

    # -- Over-block guard: non-representation phrases must reach free-translation

    def test_fold_not_blocked(self):
        assert VIEWER_REGISTRY.detect_category_phrase("fold the top design") is False

    def test_open_not_blocked(self):
        assert VIEWER_REGISTRY.detect_category_phrase("open 2hhb") is False

    def test_atom_count_not_blocked(self):
        """Ambiguous noun 'atoms' without a display verb → not flagged."""
        assert VIEWER_REGISTRY.detect_category_phrase("how many atoms are in chain A") is False

    def test_surface_area_flagged_verb_present(self):
        """verb 'show' + noun 'surface' → True after FIX 2 (verb-required tier)."""
        assert VIEWER_REGISTRY.detect_category_phrase("show the surface area contacts") is True

    # ── FIX 2: surface false-positive regression suite ────────────────────────

    def test_binding_surface_no_verb_not_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("binding surface") is False

    def test_surface_of_dimer_no_verb_not_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("the surface of the dimer") is False

    def test_hydrophobic_surface_patch_not_flagged(self):
        """'drop' is a substring of 'hydrophobic'; word-boundary check must prevent match."""
        assert VIEWER_REGISTRY.detect_category_phrase("hydrophobic surface patch") is False

    def test_surface_contacts_between_chains_not_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("surface contacts between chains") is False

    def test_closely_related_surface_residues_not_flagged(self):
        """'lose' is a substring of 'closely'; word-boundary check must prevent match."""
        assert VIEWER_REGISTRY.detect_category_phrase("closely related surface residues") is False

    def test_show_surface_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("show the surface") is True

    def test_hide_surface_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("hide the surface") is True

    def test_add_surface_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("add a surface") is True

    def test_ditch_surface_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("ditch the surface") is True

    def test_no_more_surface_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("no more surface") is True

    def test_bare_surface_token_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("surface") is True

    def test_bare_surfaces_token_flagged(self):
        assert VIEWER_REGISTRY.detect_category_phrase("surfaces") is True


# ── 19. Emission guard — is_representation_shaped() ──────────────────────────

class TestEmissionGuard:
    """
    Verify that is_representation_shaped() catches commands that must only
    originate from the render layer, and does NOT block unrelated commands.
    """

    def test_hide_atoms_shaped(self):
        assert is_representation_shaped("hide #1 atoms") is True

    def test_hide_spheres_shaped(self):
        assert is_representation_shaped("hide #1 spheres") is True

    def test_show_cartoons_shaped(self):
        assert is_representation_shaped("show #1 cartoons") is True

    def test_style_sphere_shaped(self):
        assert is_representation_shaped("style #1 sphere") is True

    def test_style_stick_shaped(self):
        assert is_representation_shaped("style #1/A stick") is True

    def test_surface_shaped(self):
        assert is_representation_shaped("surface #1") is True

    def test_tilde_surface_shaped(self):
        assert is_representation_shaped("~surface #1") is True

    def test_cartoon_with_spec_shaped(self):
        assert is_representation_shaped("cartoon #1") is True

    def test_hide_solvent_not_shaped(self):
        """'hide solvent atoms' has no #spec — not representation-shaped."""
        assert is_representation_shaped("hide solvent atoms") is False

    def test_hide_bare_model_not_shaped(self):
        """'hide #1' (no collection) — not blocked."""
        assert is_representation_shaped("hide #1") is False

    def test_color_not_shaped(self):
        assert is_representation_shaped("color #1 red") is False

    def test_open_not_shaped(self):
        assert is_representation_shaped("open 2hhb") is False

    def test_runscript_not_shaped(self):
        assert is_representation_shaped("runscript /tmp/probe.py") is False

    def test_chain_spec_atoms_shaped(self):
        """Chain-qualified spec still matches — must come from render layer."""
        assert is_representation_shaped("hide #1/A atoms") is True


# ── 20. Integration — main._handle_request() (real REPL entry point) ─────────

class TestIntegrationHandleRequest:
    """
    Drive the actual REPL entry point (main._handle_request) with a mock bridge
    and a mock translator.  Asserts call/no-call on translate() and execution.

    setup_method resets the module-level classifier cache so each test gets a
    fresh make_llm_classify_fn() call (required for mocking to take effect).
    """

    def setup_method(self):
        import tool_router
        tool_router._repr_classify_fn = None

    @staticmethod
    def _make_bot(bridge=None, translate_side_effect=None):
        """Minimal StructureBot for testing — no network, no __init__ side-effects."""
        from main import StructureBot
        from session_state import SessionState
        from tool_router import ToolRouter

        if bridge is None:
            bridge = MagicMock()
            bridge.run_command.return_value  = {"value": "", "error": None}
            bridge.run_commands.return_value = []

        mock_translator = MagicMock()
        if translate_side_effect is not None:
            mock_translator.translate.side_effect = translate_side_effect
        else:
            mock_translator.translate.return_value = {
                "commands": [], "tools_needed": [], "tool_inputs": {},
                "explanations": [], "warnings": [],
                "clarification_needed": None, "confidence": "high",
            }

        session = SessionState()
        session.structures = {"1": {"name": "test", "path": None}}

        bot = object.__new__(StructureBot)
        # translate_error_fix must return an empty fix so the error-correction
        # path doesn't try to Prompt.ask in a headless test
        mock_translator.translate_error_fix.return_value = {
            "commands": [], "explanations": [], "warnings": [],
            "clarification_needed": None, "confidence": "high",
        }

        bot = object.__new__(StructureBot)
        bot.bridge             = bridge
        bot.translator         = mock_translator
        bot.session            = session
        bot.router             = ToolRouter(bridge, session)
        bot.auto_proceed       = True   # skip confirmation prompt
        bot.auto_proceed_delay = 0      # no countdown delay
        bot._log_exchange      = MagicMock()   # avoid log_file write
        return bot

    def test_a_rep_phrase_routes_to_render_translate_not_called(self):
        """
        (a) A phrase that triggers noun-floor goes through the render layer;
        translate() is never called.
        """
        bridge = MagicMock()
        bridge.run_command.return_value  = {"value": "", "error": None}
        bridge.run_commands.return_value = []

        # Patch make_llm_classify_fn where tool_router imports it
        classifier = MagicMock(return_value="view.hide_atoms")
        with patch("intent_registry.make_llm_classify_fn", return_value=classifier), \
             patch("main.probe_chimerax_verbs"):
            bot = self._make_bot(bridge)
            bot._handle_request("remove all spheres")

        # translate() must NOT have been called
        bot.translator.translate.assert_not_called()

        # bridge.run_command must have been called with "hide #1 atoms" (trusted=True unused here)
        executed = [c.args[0] for c in bridge.run_command.call_args_list]
        assert any("hide" in cmd and "atoms" in cmd for cmd in executed), (
            f"Expected 'hide ... atoms' in executed commands; got: {executed}"
        )

    def test_b_rep_phrase_classifier_miss_no_command_executed(self):
        """
        (b) Noun-floor triggers (wireframe is unambiguous, no alias match),
        LLM returns None → graceful miss, no rep bridge call.
        """
        bridge = MagicMock()
        bridge.run_command.return_value  = {"value": "", "error": None}
        bridge.run_commands.return_value = []

        # "switch to the wireframe rendering mode" → noun-floor True, alias None
        # Classifier returns None → graceful miss
        classifier = MagicMock(return_value=None)
        with patch("intent_registry.make_llm_classify_fn", return_value=classifier), \
             patch("main.probe_chimerax_verbs"):
            bot = self._make_bot(bridge)
            bot._handle_request("switch to the wireframe rendering mode")

        # No representation command should have reached the bridge
        rep_calls = [
            c.args[0] for c in bridge.run_command.call_args_list
            if is_representation_shaped(c.args[0])
        ]
        assert rep_calls == [], f"Unexpected rep commands executed: {rep_calls}"
        # translate() also not called (noun-floor intercepted before translate)
        bot.translator.translate.assert_not_called()

    def test_c_non_rep_phrase_reaches_free_translation(self):
        """
        (c) A non-representation phrase must bypass the noun-floor and reach
        translate() (the free-translation path).
        """
        with patch("main.probe_chimerax_verbs"):
            bot = self._make_bot()
            bot._handle_request("fold the top design")

        bot.translator.translate.assert_called_once()
        call_text = bot.translator.translate.call_args[0][0]
        assert "fold" in call_text.lower()

    def test_d_forced_rep_command_blocked_at_emission_guard(self):
        """
        (d) Free-translation producing a rep-shaped command is blocked by the
        emission guard in _execute_commands(); bridge.run_commands never called with it.
        """
        bad_cmd = "hide #1 spheres"
        bridge = MagicMock()
        bridge.run_command.return_value  = {"value": "", "error": None}
        bridge.run_commands.return_value = []

        def _mock_translate(text, session):
            return {
                "commands": [bad_cmd], "tools_needed": [], "tool_inputs": {},
                "explanations": [], "warnings": [],
                "clarification_needed": None, "confidence": "high",
            }

        with patch("main.probe_chimerax_verbs"):
            bot = self._make_bot(bridge, translate_side_effect=_mock_translate)
            bot._handle_request("make it glow with spheres everywhere")

        # The emission guard must have prevented run_commands from seeing the bad command
        for c in bridge.run_commands.call_args_list:
            cmds = c.args[0] if c.args else []
            assert bad_cmd not in cmds, (
                f"Blocked command {bad_cmd!r} reached bridge.run_commands: {cmds}"
            )
