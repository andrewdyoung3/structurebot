"""
tests/test_design_goal.py
-------------------------
Design-intent op-class (Stage 1: "solubility") — the THIRD op-class after
viewer/color. Same §0 intent/render discipline; render codomain = a tool-invocation
profile (tool + scope + params + ranking).

Groups
------
1.  Registry resolution — alias / LLM tier / over-attraction MISS
2.  Category floor + precedence (redesign+goal only; bare redesign & "suggest
    mutations" excluded)
3.  Exposed-position accessor (cavity_bridge.solvent_exposed_residues)
4.  CamSol comparable ranking scalar (camsol_solubility_score)
5.  Route supersession — single source of truth, no double-route
6.  _run_design_goal — profile realisation, ranking, fold-guard, miss-handback,
    refuse-on-empty, transparency
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import intent_registry as ir
from intent_registry import DESIGN_GOAL_REGISTRY, DESIGN_PROFILES, _design_goal_category_floor
from camsol_bridge import camsol_solubility_score, camsol_score
from cavity_bridge import CavityBridge
from tool_router import ToolRouter, ToolStepResult


def _make_router(structures=None) -> ToolRouter:
    bridge = MagicMock()
    session = MagicMock()
    session.structures = structures if structures is not None else {"1": {"name": "1hsg"}}
    session.get_structure.return_value = {"name": "1hsg"}
    return ToolRouter(bridge=bridge, session=session)


def _translator_stub(cmds=None):
    return {"commands": cmds or [], "explanations": [], "warnings": [],
            "clarification_needed": None, "confidence": "high",
            "tools_needed": ["chimerax"], "tool_inputs": {}}


# ── 1. Registry resolution ──────────────────────────────────────────────────────

class TestRegistryResolution:

    def test_alias_direct(self):
        assert DESIGN_GOAL_REGISTRY.resolve_alias("redesign for solubility") == "design.solubility"
        assert DESIGN_GOAL_REGISTRY.resolve_alias("redesign to reduce aggregation") == "design.solubility"

    def test_llm_tier_resolves_paraphrase(self):
        # alias misses, LLM returns the label
        key, method = DESIGN_GOAL_REGISTRY.resolve(
            "rework the surface so it stops clumping",
            llm_classify_fn=lambda t, ls: "design.solubility")
        assert key == "design.solubility" and method == "llm"

    def test_over_attraction_alias_does_not_force_fit(self):
        # the deterministic guard: a non-solubility goal must NOT alias to solubility
        assert DESIGN_GOAL_REGISTRY.resolve_alias("redesign to be more thermostable") is None

    def test_over_attraction_llm_none_is_miss(self):
        # a correct classifier returns None for thermostability → MISS (not force-fit)
        key, method = DESIGN_GOAL_REGISTRY.resolve(
            "redesign chain A to be more thermostable",
            llm_classify_fn=lambda t, ls: None)
        assert key is None and method == "miss"

    def test_profile_present_for_solubility(self):
        prof = DESIGN_PROFILES["design.solubility"]
        assert prof.tool == "proteinmpnn"
        assert prof.designable == "solvent_exposed"
        assert prof.bias == "soluble"
        assert "C" in prof.omit
        assert prof.ranking == ("camsol", "esmfold")


# ── 2. Category floor + precedence ──────────────────────────────────────────────

class TestCategoryFloor:

    def test_redesign_plus_goal_enters(self):
        assert _design_goal_category_floor("redesign chain a for solubility") is True
        assert _design_goal_category_floor("redesign to reduce aggregation") is True

    def test_bare_redesign_does_not_enter(self):
        # no objective → never enters the op-class (plain redesign stays plain)
        assert _design_goal_category_floor("redesign chain a") is False

    def test_suggest_mutations_does_not_enter(self):
        # no redesign verb → stays mutation_scan
        assert _design_goal_category_floor("suggest mutations to improve solubility") is False

    def test_non_solubility_goal_still_enters(self):
        # thermostable trips the floor (so it reaches resolution → MISS-handback)
        assert _design_goal_category_floor("redesign chain a to be more thermostable") is True


# ── 3. Exposed-position accessor ─────────────────────────────────────────────────

class TestExposedSelector:

    def test_returns_only_exposed_resnums(self):
        cav = CavityBridge()
        sasa = {("A", 10): 5.0, ("A", 11): 80.0, ("A", 12): 45.0, ("B", 13): 90.0}
        with patch.object(cav, "_load_structure", return_value=object()), \
             patch.object(cav, "_sasa_for_chains", return_value=sasa):
            exposed = cav.solvent_exposed_residues("x.pdb", "A", sasa_threshold=40.0)
        assert exposed == [11, 12]            # 10 buried (5<40); B excluded (other chain)

    def test_empty_when_sasa_unavailable(self):
        cav = CavityBridge()
        with patch.object(cav, "_load_structure", return_value=None):
            assert cav.solvent_exposed_residues("x.pdb", "A") == []


# ── 4. CamSol comparable ranking scalar ─────────────────────────────────────────

class TestCamsolRankingScalar:

    def test_charged_more_soluble_than_hydrophobic(self):
        assert camsol_solubility_score("DEKRDEKRD") > camsol_solubility_score("FFFFFFFFF")

    def test_normalized_profile_mean_is_zero(self):
        # the per-residue z-profile cannot rank (mean ≈ 0 for any sequence)
        seq = "DEKRDEKRDFFFFFWWWW"
        assert abs(sum(camsol_score(seq)) / len(seq)) < 1e-6

    def test_comparable_scalar_differs_between_sequences(self):
        assert camsol_solubility_score("AAAA") != camsol_solubility_score("DDDD")


# ── 5. Route supersession ───────────────────────────────────────────────────────

class TestRouteSupersession:

    def test_redesign_for_solubility_routes_to_design_goal(self):
        r = _make_router()
        out = r.route(_translator_stub(cmds=["something"]),
                      user_input="redesign chain A for solubility")
        assert out["tools_needed"] == ["design_goal"]
        assert out["commands"] == []          # translator commands cleared (no double-route)
        assert out["tool_inputs"]["design_goal"]["intent_key"] == "design.solubility"

    def test_bare_redesign_not_design_goal(self):
        r = _make_router()
        out = r.route(_translator_stub(cmds=["x"]), user_input="redesign chain A")
        assert out["tools_needed"] != ["design_goal"]

    def test_suggest_mutations_solubility_not_design_goal(self):
        # different verb — must NOT be stolen by the design op-class
        r = _make_router()
        out = r.route(_translator_stub(), user_input="suggest mutations to improve solubility")
        assert "design_goal" not in out["tools_needed"]


# ── 6. _run_design_goal handler ─────────────────────────────────────────────────

def _designs():
    # two designs; the FIRST is less soluble so ranking must reorder it second
    return [
        {"sequence": "FFFFFFFFFF", "recovery": 0.9},   # hydrophobic → low CamSol
        {"sequence": "DEKRDEKRDE", "recovery": 0.5},   # charged → high CamSol
    ]


def _mpnn_result():
    return ToolStepResult(
        tool="proteinmpnn", success=True,
        data={"sequences": _designs(), "wildtype_sequence": "AFAFAFAFAF"},
        viz_commands=[], viz_explanations=[], summary="raw mpnn")


class TestRunDesignGoal:

    def setup_method(self):
        import tool_router
        tool_router._design_classify_fn = None

    def _router_for_handler(self):
        r = _make_router()
        cav = MagicMock()
        cav.solvent_exposed_residues.return_value = [3, 5, 7]
        r._get_cavity_bridge = MagicMock(return_value=cav)
        r._ensure_pdb_file = MagicMock(return_value="1hsg.pdb")
        r._first_model_id = MagicMock(return_value="1")
        return r

    def test_resolved_profile_passes_exposed_positions_to_mpnn(self):
        r = self._router_for_handler()
        captured = {}
        def _fake_mpnn(inp):
            captured.update(inp)
            return _mpnn_result()
        r._run_proteinmpnn = _fake_mpnn
        r._get_esmfold_bridge = MagicMock(side_effect=Exception("no esmfold"))
        res = r._run_design_goal({"intent_key": "design.solubility", "chain": "A",
                                  "model_id": "1"}, user_input="redesign chain A for solubility")
        assert res.success
        assert captured["design_positions"] == [3, 5, 7]    # exposed-only, not whole chain
        assert captured["_resolved_profile"] == "design.solubility"
        assert captured["exclude_amino_acids"] == ["C"]     # Cys omitted
        assert captured["bias_toward"] == "soluble"

    def test_ranks_by_camsol_descending(self):
        r = self._router_for_handler()
        r._run_proteinmpnn = MagicMock(return_value=_mpnn_result())
        r._get_esmfold_bridge = MagicMock(return_value=MagicMock(
            predict=MagicMock(return_value={"mean_plddt": 85.0})))
        res = r._run_design_goal({"intent_key": "design.solubility", "chain": "A"},
                                 user_input="redesign chain A for solubility")
        seqs = res.data["sequences"]
        assert seqs[0]["sequence"] == "DEKRDEKRDE"          # charged design ranked first
        assert seqs[0]["camsol"] > seqs[1]["camsol"]
        assert res.data["wt_camsol"] is not None
        assert seqs[0]["camsol_gain"] is not None
        assert res.data["ranking"] == "camsol+esmfold"

    def test_fold_guard_flags_low_plddt(self):
        r = self._router_for_handler()
        r._run_proteinmpnn = MagicMock(return_value=_mpnn_result())
        r._get_esmfold_bridge = MagicMock(return_value=MagicMock(
            predict=MagicMock(return_value={"mean_plddt": 40.0})))   # below floor
        res = r._run_design_goal({"intent_key": "design.solubility", "chain": "A"},
                                 user_input="redesign chain A for solubility")
        assert any("fold_flag" in d for d in res.data["sequences"])

    def test_esmfold_unavailable_ranks_camsol_only(self):
        r = self._router_for_handler()
        r._run_proteinmpnn = MagicMock(return_value=_mpnn_result())
        r._get_esmfold_bridge = MagicMock(return_value=MagicMock(
            predict=MagicMock(side_effect=Exception("venv312 down"))))
        res = r._run_design_goal({"intent_key": "design.solubility", "chain": "A"},
                                 user_input="redesign chain A for solubility")
        assert res.success
        assert "CamSol only" in res.summary

    def test_miss_hands_back_to_plain_proteinmpnn(self):
        # non-solubility goal → classifier None → MISS → plain redesign, never widen/error
        r = self._router_for_handler()
        handback = MagicMock(return_value=ToolStepResult(
            tool="proteinmpnn", success=True, data={}, summary="plain"))
        r._run_proteinmpnn = handback
        import tool_router
        tool_router._design_classify_fn = lambda t, ls: None     # correct: not force-fit
        res = r._run_design_goal({"intent_key": None, "chain": "A"},
                                 user_input="redesign chain A to be more thermostable")
        assert res.success
        handback.assert_called_once()
        # handed back WITHOUT a resolved profile (plain path)
        assert "_resolved_profile" not in handback.call_args.args[0]

    def test_refuse_on_empty_exposed_never_widens(self):
        r = self._router_for_handler()
        r._get_cavity_bridge().solvent_exposed_residues.return_value = []   # SASA unavailable
        r._run_proteinmpnn = MagicMock()
        res = r._run_design_goal({"intent_key": "design.solubility", "chain": "A"},
                                 user_input="redesign chain A for solubility")
        assert res.success is False
        assert "whole chain" in res.error.lower()
        r._run_proteinmpnn.assert_not_called()       # never reaches the redesign

    def test_transparency_states_profile(self):
        r = self._router_for_handler()
        r._run_proteinmpnn = MagicMock(return_value=_mpnn_result())
        r._get_esmfold_bridge = MagicMock(return_value=MagicMock(
            predict=MagicMock(return_value={"mean_plddt": 85.0})))
        res = r._run_design_goal({"intent_key": "design.solubility", "chain": "A"},
                                 user_input="redesign chain A for solubility")
        s = res.summary.lower()
        assert "design profile: solubility" in s
        assert "solvent-exposed" in s
        assert "cys omitted" in s
        assert "camsol" in s
        assert "wt camsol baseline" in s
