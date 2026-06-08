"""
tests/test_color_intent.py
--------------------------
Tests for the color op-class (Intent/Render migration, Priority 0.5) in
intent_registry.py + its routing/handler integration in tool_router.py / main.py.

Mirrors tests/test_intent_registry.py (the viewer op-class).

Test groups
-----------
1.  Alias resolution — color scheme intents
2.  Named-color extraction (single + multi-word)
3.  Category floor + detect_category_phrase("color") — conservative gate
4.  Render layer — each scheme → probe-verified (spec-first) command
5.  Routing integration — covered color phrase → tools_needed=["color"]
6.  Routing integration — uncovered phrase falls through
7.  _run_color — scheme / solid / graceful-miss / no-bridge
8.  Chain-scope guard — chain colors never bleed (the payoff bug)
9.  LLM classifier — color registry + task block
10. Integration — main._handle_request() (real REPL entry point)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from intent_registry import (
    COLOR_REGISTRY,
    extract_named_color,
    _color_category_floor,
    make_llm_classify_fn,
    _COLOR_TASK_BLOCK,
)
from tool_router import ToolRouter, ToolStepResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_router(structures: dict | None = None) -> ToolRouter:
    mock_bridge  = MagicMock()
    mock_session = MagicMock()
    mock_session.structures = structures if structures is not None else {
        "1": {"name": "1hsg", "path": None}
    }
    mock_session.get_proteinmpnn_result.return_value = None
    mock_session.get_assembly_info.return_value = None
    mock_session.get_structure.return_value = {"name": "1hsg", "path": None}
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


def _router_with_bridge(run_command_results: list) -> ToolRouter:
    mock_bridge  = MagicMock()
    mock_bridge.run_command.side_effect = run_command_results
    mock_session = MagicMock()
    mock_session.structures = {"1": {"name": "1hsg"}}
    mock_session.get_structure.return_value = {"name": "1hsg"}
    return ToolRouter(bridge=mock_bridge, session=mock_session)


# ── 1. Alias resolution ────────────────────────────────────────────────────────

class TestAliasResolution:

    def test_by_chain(self):
        assert COLOR_REGISTRY.resolve_alias("color by chain") == "color.by_chain"

    def test_by_chain_compact(self):
        assert COLOR_REGISTRY.resolve_alias("color bychain") == "color.by_chain"

    def test_by_element(self):
        assert COLOR_REGISTRY.resolve_alias("color by element") == "color.by_element"

    def test_by_heteroatom(self):
        assert COLOR_REGISTRY.resolve_alias("color heteroatoms") == "color.by_heteroatom"

    def test_rainbow(self):
        assert COLOR_REGISTRY.resolve_alias("rainbow") == "color.rainbow"

    def test_by_bfactor(self):
        assert COLOR_REGISTRY.resolve_alias("color by bfactor") == "color.by_attribute"

    def test_by_temperature_factor(self):
        assert COLOR_REGISTRY.resolve_alias("color by temperature factor") == "color.by_attribute"

    def test_case_insensitive(self):
        assert COLOR_REGISTRY.resolve_alias("COLOR BY CHAIN") == "color.by_chain"

    def test_solid_has_no_alias(self):
        # color.solid is resolved via extract_named_color, never alias
        assert COLOR_REGISTRY.resolve_alias("color chain A red") is None

    def test_non_color_returns_none(self):
        assert COLOR_REGISTRY.resolve_alias("mutate chain A") is None
        assert COLOR_REGISTRY.resolve_alias("") is None


# ── 2. Named-color extraction ──────────────────────────────────────────────────

class TestNamedColorExtraction:

    def test_single_word(self):
        assert extract_named_color("color chain A red") == "red"

    def test_single_word_blue(self):
        assert extract_named_color("make it blue") == "blue"

    def test_multi_word_precedence(self):
        # "cornflower blue" must win over bare "blue"
        assert extract_named_color("color it cornflower blue") == "cornflower blue"

    def test_multi_word_forest_green(self):
        assert extract_named_color("color chain B forest green") == "forest green"

    def test_no_color(self):
        assert extract_named_color("color by chain") is None

    def test_no_color_in_scheme_phrase(self):
        assert extract_named_color("color by element") is None

    def test_grey_variant(self):
        assert extract_named_color("color the model grey") == "grey"


# ── 3. Category floor + detect_category_phrase ─────────────────────────────────

class TestCategoryFloor:

    def test_floor_color_verb(self):
        assert _color_category_floor("color chain a red") is True

    def test_floor_recolor(self):
        assert _color_category_floor("recolor by chain") is True

    def test_floor_colour_british(self):
        assert _color_category_floor("colour the surface blue") is True

    def test_floor_rainbow(self):
        assert _color_category_floor("rainbow the model") is True

    def test_floor_no_color_verb(self):
        # scheme phrase WITHOUT a color verb must NOT gate (conservative)
        assert _color_category_floor("by chain alignment") is False

    def test_floor_bfactor_distribution_no_verb(self):
        assert _color_category_floor("the b-factor distribution") is False

    def test_detect_covered_alias(self):
        assert COLOR_REGISTRY.detect_category_phrase("color by chain", "color") is True

    def test_detect_solid_via_floor(self):
        assert COLOR_REGISTRY.detect_category_phrase("color chain A red", "color") is True

    def test_detect_rainbow(self):
        assert COLOR_REGISTRY.detect_category_phrase("rainbow it", "color") is True

    def test_detect_non_color_false(self):
        assert COLOR_REGISTRY.detect_category_phrase("mutate chain A", "color") is False
        assert COLOR_REGISTRY.detect_category_phrase("show as cartoon", "color") is False
        assert COLOR_REGISTRY.detect_category_phrase("open 1hsg", "color") is False

    # -- Complex/narrow selections must NOT gate (fall through to free-translation)
    def test_floor_proline_residues_not_gated(self):
        # op-class can't select residue types — must fall through (pins the
        # test_router_precedence distractor case)
        assert _color_category_floor("color the proline residues on chain a red") is False

    def test_floor_binding_pocket_not_gated(self):
        assert _color_category_floor("color the binding pocket residues on chain a") is False

    def test_floor_resnum_range_not_gated(self):
        assert _color_category_floor("color residues 20-30 red") is False

    def test_floor_active_site_not_gated(self):
        assert _color_category_floor("color the active site blue") is False

    def test_floor_hydrophobic_not_gated(self):
        assert _color_category_floor("color hydrophobic patches orange") is False

    def test_detect_proline_residues_false(self):
        assert COLOR_REGISTRY.detect_category_phrase(
            "color the proline residues on chain A red", "color") is False


# ── 4. Render layer ────────────────────────────────────────────────────────────

class TestRenderLayer:

    def test_by_chain_render(self):
        # spec-FIRST is the only valid form (scheme-first errors in ChimeraX)
        assert COLOR_REGISTRY.render("color.by_chain", "#1") == ["color #1 bychain"]

    def test_by_element_render(self):
        assert COLOR_REGISTRY.render("color.by_element", "#1") == ["color #1 byelement"]

    def test_by_heteroatom_render(self):
        assert COLOR_REGISTRY.render("color.by_heteroatom", "#1") == ["color #1 byhetero"]

    def test_rainbow_render(self):
        assert COLOR_REGISTRY.render("color.rainbow", "#1") == ["rainbow #1"]

    def test_by_attribute_render(self):
        cmds = COLOR_REGISTRY.render("color.by_attribute", "#1")
        assert cmds == ["color byattribute bfactor #1 palette blue:white:red"]

    def test_solid_render_is_placeholder(self):
        # color.solid render is overridden in _run_color (needs the color value)
        assert COLOR_REGISTRY.render("color.solid", "#1") == []

    def test_submodel_spec(self):
        assert COLOR_REGISTRY.render("color.by_chain", "#2.1/A") == ["color #2.1/A bychain"]


# ── 5. Routing integration — covered ───────────────────────────────────────────

class TestRoutingIntegration:

    def test_covered_color_routes_to_color_tool(self):
        router = _make_router()
        stub   = _translator_stub(cmds=["color #1 red"])  # translator guess discarded
        result = router.route(stub, user_input="color by chain")
        assert "color" in result.get("tools_needed", [])
        assert result.get("has_extra_tools") is True

    def test_color_tool_clears_translator_commands(self):
        router = _make_router()
        stub   = _translator_stub(cmds=["color /B red"])
        result = router.route(stub, user_input="color chain A red")
        assert result.get("commands", []) == []
        assert "color" in result.get("tools_needed", [])

    def test_intent_key_populated_for_scheme_alias(self):
        router = _make_router()
        result = router.route(_translator_stub(), user_input="color by chain")
        tinputs = result.get("tool_inputs", {})
        assert tinputs["color"]["intent_key"] == "color.by_chain"

    def test_intent_key_none_for_solid_phrase(self):
        # "color chain A red" → no scheme alias → intent_key None (solid in execute)
        router = _make_router()
        result = router.route(_translator_stub(), user_input="color chain A red")
        tinputs = result.get("tool_inputs", {})
        assert "color" in result.get("tools_needed", [])
        assert tinputs.get("color", {}).get("intent_key") is None

    def test_representation_wins_over_color_when_both(self):
        # A pure representation phrase must not be hijacked by the color override
        router = _make_router()
        result = router.route(_translator_stub(), user_input="show as cartoon")
        assert "representation" in result.get("tools_needed", [])
        assert "color" not in result.get("tools_needed", [])


# ── 6. Routing integration — uncovered falls through ───────────────────────────

class TestUncoveredFallthrough:

    def test_mutation_request_not_intercepted(self):
        router = _make_router()
        stub   = _translator_stub(cmds=["mutation_scan"])
        result = router.route(stub, user_input="suggest mutations to improve solubility")
        assert "color" not in result.get("tools_needed", [])

    def test_by_chain_alignment_not_intercepted(self):
        # "by chain" without a color verb must NOT gate to color
        router = _make_router()
        result = router.route(_translator_stub(), user_input="align the chains by chain order")
        assert "color" not in result.get("tools_needed", [])

    def test_open_not_intercepted(self):
        router = _make_router()
        result = router.route(_translator_stub(cmds=["open 1hsg"]), user_input="open 1hsg")
        assert "color" not in result.get("tools_needed", [])


# ── 7. _run_color handler ──────────────────────────────────────────────────────

class TestRunColor:

    def setup_method(self):
        import tool_router
        tool_router._color_classify_fn = None

    def test_scheme_alias_executes(self):
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "color by chain",
                                    "intent_key": "color.by_chain"})
        assert result.success is True
        assert result.data["commands"] == ["color #1 bychain"]

    def test_solid_color_whole_model(self):
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "color everything blue",
                                    "intent_key": None})
        assert result.success is True
        assert result.data["intent_key"] == "color.solid"
        assert result.data["color_name"] == "blue"
        # whole model → not scoped
        assert result.data["commands"] == ["color #1 blue"]

    def test_rainbow_executes(self):
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "rainbow the model",
                                    "intent_key": "color.rainbow"})
        assert result.success is True
        assert result.data["commands"] == ["rainbow #1"]

    def test_graceful_miss_when_no_color_resolved(self):
        router = _router_with_bridge([])
        with patch("intent_registry.make_llm_classify_fn") as factory:
            factory.return_value = lambda t, ls: None
            result = router._run_color({"_user_input": "color it somehow vaguely",
                                        "intent_key": None})
        assert result.success is False
        assert "available" in result.error.lower()

    def test_solid_without_color_name_asks_which(self):
        router = _router_with_bridge([])
        with patch("intent_registry.make_llm_classify_fn") as factory:
            factory.return_value = lambda t, ls: "color.solid"
            result = router._run_color({"_user_input": "give it a solid color",
                                        "intent_key": None})
        assert result.success is False
        assert "which color" in result.error.lower()

    def test_no_bridge_returns_error(self):
        mock_session = MagicMock()
        mock_session.structures = {"1": {"name": "1hsg"}}
        router = ToolRouter(bridge=None, session=mock_session)
        result = router._run_color({"_user_input": "color by chain",
                                    "intent_key": "color.by_chain"})
        assert result.success is False
        assert "bridge" in result.error.lower()

    def test_failed_command_returns_failure(self):
        router = _router_with_bridge([_cx_err("Expected a collection of …")])
        result = router._run_color({"_user_input": "color by chain",
                                    "intent_key": "color.by_chain"})
        assert result.success is False
        assert result.error is not None


# ── 8. Chain-scope guard — the payoff bug ──────────────────────────────────────

class TestChainScopeGuard:
    """
    'color chain A red' MUST be scoped to exclude ligand/solvent/ions so the
    color never bleeds.  Verified live on 1HSG (solvent_red=0, ligand_red=0);
    these assert the emitted command carries the scope.
    """

    def setup_method(self):
        import tool_router
        tool_router._color_classify_fn = None

    def test_solid_chain_color_is_scoped(self):
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "color chain A red",
                                    "intent_key": None})
        assert result.success is True
        cmd = result.data["commands"][0]
        assert cmd == "color (#1/A & ~ligand & ~solvent & ~ions) red", cmd
        # the bare (bleeding) form must NOT be emitted
        assert "color #1/A red" not in result.data["commands"]

    def test_scheme_chain_color_is_scoped(self):
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "color chain B by element",
                                    "intent_key": "color.by_element"})
        assert result.success is True
        cmd = result.data["commands"][0]
        assert cmd == "color (#1/B & ~ligand & ~solvent & ~ions) byelement", cmd

    def test_whole_model_solid_not_scoped(self):
        # coloring "everything" is explicit — must NOT be scoped
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "color the whole model red",
                                    "intent_key": None})
        assert result.data["commands"] == ["color #1 red"]

    def test_slash_chain_spec_is_scoped(self):
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "color /A green",
                                    "intent_key": None})
        cmd = result.data["commands"][0]
        assert cmd == "color (#1/A & ~ligand & ~solvent & ~ions) green", cmd

    def test_each_chain_is_whole_model_not_article_a(self):
        # "give each chain a separate shade" — the article 'a' must NOT be read as
        # chain A; "each chain" → whole model (regression from live verify)
        router = _router_with_bridge([_cx_ok("")])
        result = router._run_color({"_user_input": "give each chain a separate shade",
                                    "intent_key": "color.by_chain"})
        assert result.data["commands"] == ["color #1 bychain"], result.data["commands"]
        assert result.data["chain"] is None


# ── 9. LLM classifier — color registry ─────────────────────────────────────────

class TestColorClassifier:

    def test_classifier_uses_color_registry_labels(self):
        captured: list = []

        def fake_post(url, json=None, **kw):
            captured.append(json)
            resp = MagicMock()
            resp.json.return_value = {"response": "color.by_chain"}
            return resp

        with patch("requests.post", side_effect=fake_post):
            fn = make_llm_classify_fn(
                backend_name="ollama",
                registry=COLOR_REGISTRY,
                task_block=_COLOR_TASK_BLOCK,
            )
            labels = COLOR_REGISTRY.list_intent_keys("color")
            result = fn("give every chain its own color", labels)

        assert result == "color.by_chain"
        body = captured[0]
        assert body.get("think") is False
        # prompt must reference the color intents, not viewer ones
        assert "color.by_chain" in body["prompt"]
        assert "view.cartoon_only" not in body["prompt"]

    def test_classifier_think_not_in_options(self):
        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = {"response": "color.rainbow"}
            fn = make_llm_classify_fn(backend_name="ollama",
                                      registry=COLOR_REGISTRY,
                                      task_block=_COLOR_TASK_BLOCK)
            fn("spectrum colors", COLOR_REGISTRY.list_intent_keys("color"))
            body = mock_post.call_args.kwargs["json"]
        assert "think" not in body.get("options", {})


# ── 10. Integration — main._handle_request() ──────────────────────────────────

class TestIntegrationHandleRequest:
    """Drive the real REPL entry point (main._handle_request)."""

    def setup_method(self):
        import tool_router
        tool_router._color_classify_fn = None
        tool_router._repr_classify_fn = None

    @staticmethod
    def _make_bot(bridge=None):
        from main import StructureBot
        from session_state import SessionState
        from tool_router import ToolRouter

        if bridge is None:
            bridge = MagicMock()
            bridge.run_command.return_value  = {"value": "", "error": None}
            bridge.run_commands.return_value = []

        mock_translator = MagicMock()
        mock_translator.translate.return_value = {
            "commands": [], "tools_needed": [], "tool_inputs": {},
            "explanations": [], "warnings": [],
            "clarification_needed": None, "confidence": "high",
        }
        mock_translator.translate_error_fix.return_value = {
            "commands": [], "explanations": [], "warnings": [],
            "clarification_needed": None, "confidence": "high",
        }

        session = SessionState()
        session.structures = {"1": {"name": "1hsg", "path": None}}

        bot = object.__new__(StructureBot)
        bot.bridge             = bridge
        bot.translator         = mock_translator
        bot.session            = session
        bot.router             = ToolRouter(bridge, session)
        bot.auto_proceed       = True
        bot.auto_proceed_delay = 0
        bot._log_exchange      = MagicMock()
        return bot

    def test_color_chain_red_scoped_translate_not_called(self):
        """'color chain A red' → render layer (scoped); translate() never called."""
        bridge = MagicMock()
        bridge.run_command.return_value  = {"value": "", "error": None}
        bridge.run_commands.return_value = []

        with patch("main.probe_chimerax_verbs"):
            bot = self._make_bot(bridge)
            bot._handle_request("color chain A red")

        bot.translator.translate.assert_not_called()
        executed = [c.args[0] for c in bridge.run_command.call_args_list]
        assert "color (#1/A & ~ligand & ~solvent & ~ions) red" in executed, executed
        # the bleeding bare form must not have been executed
        assert "color #1/A red" not in executed

    def test_color_by_chain_routes_to_scheme(self):
        bridge = MagicMock()
        bridge.run_command.return_value  = {"value": "", "error": None}
        bridge.run_commands.return_value = []

        with patch("main.probe_chimerax_verbs"):
            bot = self._make_bot(bridge)
            bot._handle_request("color by chain")

        bot.translator.translate.assert_not_called()
        executed = [c.args[0] for c in bridge.run_command.call_args_list]
        assert "color #1 bychain" in executed, executed

    def test_non_color_phrase_reaches_translation(self):
        with patch("main.probe_chimerax_verbs"):
            bot = self._make_bot()
            bot._handle_request("fold the top design")
        bot.translator.translate.assert_called_once()

    def test_color_classifier_miss_no_command_executed(self):
        """A color-gated phrase the classifier can't resolve → no color command."""
        bridge = MagicMock()
        bridge.run_command.return_value  = {"value": "", "error": None}
        bridge.run_commands.return_value = []

        classifier = MagicMock(return_value=None)
        with patch("intent_registry.make_llm_classify_fn", return_value=classifier), \
             patch("main.probe_chimerax_verbs"):
            bot = self._make_bot(bridge)
            bot._handle_request("color it in some artistic way")

        bot.translator.translate.assert_not_called()
        executed = [c.args[0] for c in bridge.run_command.call_args_list]
        color_cmds = [c for c in executed if c.startswith(("color ", "rainbow "))]
        assert color_cmds == [], color_cmds
