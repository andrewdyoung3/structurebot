"""
Unit tests for the model-independent 3-dimension eval harness (eval_harness.py).
Scorers are exercised on the 8 SAMPLE_CASES with hand-authored GOOD/BAD model
outputs (model-independent gold — the outputs here stand in for any backend).
ChimeraX is mocked; the live `effect` path is validated separately by a script.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import eval_harness as eh
import translator_corpus as tc

CASES = {c.id: c for c in eh.SAMPLE_CASES}


# ── a mock ChimeraX probe (records commands; canned query answers) ───────────────
class MockProbe:
    def __init__(self, sel_output="", color_output="rgb(255,0,0)"):
        self.ran = []
        self.sel_output = sel_output
        self.color_output = color_output

    def __call__(self, command: str) -> str:
        self.ran.append(command)
        c = command.lower()
        if "info residues sel" in c:
            return self.sel_output
        if "color" in c:
            return self.color_output
        return ""


def _tr(tools=None, commands=None, tool_inputs=None, clarify=None):
    return {
        "commands": commands or [], "explanations": [], "warnings": [],
        "clarification_needed": clarify, "confidence": "high",
        "tools_needed": tools or [], "tool_inputs": tool_inputs or {},
    }


# ════════════════════════════════════════════════════════════════════════════════
#  ACCURACY — strict + partial, forbidden, constraint-in-tool_inputs, scope
# ════════════════════════════════════════════════════════════════════════════════
def test_accuracy_chimerax_args_from_commands_strict():
    tr = _tr(tools=["chimerax"], commands=["color #1/A red", "view"])
    r = eh.score_accuracy(CASES["s1_viz_color"], tr)
    assert r.applicable and r.passed and r.partial == 1.0


def test_accuracy_wrong_color_partial_not_strict():
    tr = _tr(tools=["chimerax"], commands=["color #1/A blue"])
    r = eh.score_accuracy(CASES["s1_viz_color"], tr)
    assert not r.passed                       # strict fails (color wrong)
    assert 0.0 < r.partial < 1.0              # but tool + chain still credited
    assert any("color(red)" in m for m in r.detail["missed"])


def test_accuracy_scope_checked_in_tool_inputs():
    # THE point of the new harness: scope=20-30 finally gets checked.
    good = _tr(tools=["proteinmpnn"],
               tool_inputs={"proteinmpnn": {"chain": "A", "design_positions": list(range(20, 31))}})
    assert eh.score_accuracy(CASES["s3_sel_scope"], good).passed

    whole = _tr(tools=["proteinmpnn"], tool_inputs={"proteinmpnn": {"chain": "A"}})
    r = eh.score_accuracy(CASES["s3_sel_scope"], whole)
    assert not r.passed                       # no scope → fails scope + whole-chain
    missed = " ".join(r.detail["missed"])
    assert "scope(20-30)" in missed and "whole-chain" in missed


def test_accuracy_forbidden_tool_blocks():
    # mis-route to mutation_scan trips both tool() and forbidden(mutation_scan)
    tr = _tr(tools=["mutation_scan"], tool_inputs={"mutation_scan": {"chain": "A"}})
    r = eh.score_accuracy(CASES["s3_sel_scope"], tr)
    assert not r.passed
    assert any("forbidden_clear(mutation_scan)" in m for m in r.detail["missed"])


def test_accuracy_constraint_exclude_cys_and_solubility():
    good = _tr(tools=["proteinmpnn"],
               tool_inputs={"proteinmpnn": {"chain": "A",
                                            "exclude_amino_acids": ["C"], "bias_toward": "soluble"}})
    assert eh.score_accuracy(CASES["s4_mpnn_soluble"], good).passed

    # routes to camsol (the collision) → tool wrong + forbidden(camsol) + constraints missed
    bad = _tr(tools=["camsol"], tool_inputs={"camsol": {"chain": "A"}})
    r = eh.score_accuracy(CASES["s4_mpnn_soluble"], bad)
    assert not r.passed
    assert any("forbidden_clear(camsol)" in m for m in r.detail["missed"])
    assert any("constraint(exclude_cys)" in m for m in r.detail["missed"])


def test_accuracy_any_of_tool_group():
    for t in ("mutation_scan", "proline", "rosetta"):
        tr = _tr(tools=[t], tool_inputs={t: {"chain": "A"}})
        assert eh.score_accuracy(CASES["s5_proline"], tr).passed, t
    bad = _tr(tools=["disulfide"], tool_inputs={"disulfide": {"chain": "A"}})
    assert not eh.score_accuracy(CASES["s5_proline"], bad).passed


def test_accuracy_forbidden_zone_keyword():
    ok = _tr(tools=["chimerax"], commands=["select #1/A & ~(#1/A :< 8)"])
    assert eh.score_accuracy(CASES["s6_zone_beyond"], ok).passed
    bad = _tr(tools=["chimerax"], commands=["zone #1/A 8"])
    assert not eh.score_accuracy(CASES["s6_zone_beyond"], bad).passed


# ════════════════════════════════════════════════════════════════════════════════
#  USABILITY — execute / clarify / refuse + hallucination
# ════════════════════════════════════════════════════════════════════════════════
def test_usability_execute():
    tr = _tr(tools=["camsol"], tool_inputs={"camsol": {"chain": "A"}})
    assert eh.score_usability(CASES["s2_camsol"], tr).passed


def test_usability_clarify_passes_only_when_asking():
    asking = _tr(clarify="Which chain do you mean?")
    assert eh.score_usability(CASES["s7_ambiguous"], asking).passed
    # a confident whole-structure call on an ambiguous prompt MUST fail
    confident = _tr(tools=["proteinmpnn"], tool_inputs={"proteinmpnn": {"chain": "A"}})
    assert not eh.score_usability(CASES["s7_ambiguous"], confident).passed


def test_usability_refuse_fails_on_any_action():
    declined = _tr(tools=[], commands=[])
    assert eh.score_usability(CASES["s8_unsafe"], declined).passed
    acted = _tr(commands=["delete ~/*"])
    assert not eh.score_usability(CASES["s8_unsafe"], acted).passed
    tooled = _tr(tools=["chimerax"], commands=[])
    assert not eh.score_usability(CASES["s8_unsafe"], tooled).passed


def test_usability_hallucinated_tool_fails_execute():
    tr = _tr(tools=["totally_fake_tool"], tool_inputs={})
    r = eh.score_usability(CASES["s2_camsol"], tr)
    assert not r.passed
    assert r.detail["hallucinated"] == ["totally_fake_tool"]


def test_hallucinated_tools_helper():
    assert eh.hallucinated_tools(_tr(tools=["camsol", "nope"])) == ["nope"]
    assert eh.hallucinated_tools(_tr(tools=["camsol", "esm"])) == []


# ════════════════════════════════════════════════════════════════════════════════
#  FUNCTIONALITY — dispatch (static) + effect (mock probe)
# ════════════════════════════════════════════════════════════════════════════════
def test_functionality_dispatch_inputs():
    good = _tr(tools=["proteinmpnn"],
               tool_inputs={"proteinmpnn": {"chain": "A", "design_positions": list(range(20, 31))}})
    r = eh.score_functionality(CASES["s3_sel_scope"], good)
    assert r.applicable and r.passed and r.detail["mode"] == "dispatch"

    wrong_chain = _tr(tools=["proteinmpnn"],
                      tool_inputs={"proteinmpnn": {"chain": "B", "design_positions": list(range(20, 31))}})
    assert not eh.score_functionality(CASES["s3_sel_scope"], wrong_chain).passed

    not_dispatched = _tr(tools=["camsol"], tool_inputs={"camsol": {"chain": "A"}})
    assert not eh.score_functionality(CASES["s3_sel_scope"], not_dispatched).passed


def test_functionality_effect_with_mock_probe():
    tr = _tr(tools=["chimerax"], commands=["select #1/A & ~(#1/A :< 8)", "info residues sel"])
    probe = MockProbe(sel_output="")          # no residues selected → expected []
    r = eh.score_functionality(CASES["s6_zone_beyond"], tr, probe=probe)
    assert r.applicable and r.passed and r.detail["probe"] == "selection_resnums"
    assert any("info residues sel" in c.lower() for c in probe.ran)


def test_functionality_effect_no_probe_is_not_passed():
    tr = _tr(tools=["chimerax"], commands=["color #1/A red"])
    r = eh.score_functionality(CASES["s1_viz_color"], tr, probe=None)
    assert r.applicable and not r.passed      # CI without ChimeraX: honest non-pass, not skipped-as-pass


# ════════════════════════════════════════════════════════════════════════════════
#  AGGREGATE math (weighted mean + strict fully-correct)
# ════════════════════════════════════════════════════════════════════════════════
def test_aggregate_weighted_mean_and_fully_correct():
    probe = MockProbe()
    # all-correct outputs for every sample case
    good = {
        "s1_viz_color": _tr(tools=["chimerax"], commands=["color #1/A red", "info residues sel"]),
        "s2_camsol":    _tr(tools=["camsol"], tool_inputs={"camsol": {"chain": "A"}}),
        "s3_sel_scope": _tr(tools=["proteinmpnn"],
                            tool_inputs={"proteinmpnn": {"chain": "A", "design_positions": list(range(20, 31))}}),
        "s4_mpnn_soluble": _tr(tools=["proteinmpnn"],
                               tool_inputs={"proteinmpnn": {"chain": "A", "exclude_amino_acids": ["C"],
                                                            "bias_toward": "soluble"}}),
        "s5_proline":   _tr(tools=["mutation_scan"], tool_inputs={"mutation_scan": {"chain": "A"}}),
        "s6_zone_beyond": _tr(tools=["chimerax"], commands=["select #1/A & ~(#1/A :< 8)", "info residues sel"]),
        "s7_ambiguous": _tr(clarify="Which chain — A or B?"),
        "s8_unsafe":    _tr(tools=[], commands=[]),
    }
    scores = [eh.score_case(CASES[cid], tr, probe=probe) for cid, tr in good.items()]
    for s in scores:
        assert s.fully_correct, s.id
        assert s.aggregate == pytest.approx(1.0)

    agg = eh.aggregate_scores(scores, eh.SAMPLE_CASES)
    assert agg["fully_correct_rate"] == pytest.approx(1.0)
    assert agg["aggregate_weighted_mean"] == pytest.approx(1.0)
    # refuse case (s8) has only the usability dimension applicable
    assert agg["dimensions"]["usability"] == pytest.approx(1.0)


def test_aggregate_weights_normalised_over_applicable_dims():
    # a refuse case: only usability applies → aggregate == usability score (1.0)
    s = eh.score_case(CASES["s8_unsafe"], _tr(tools=[], commands=[]))
    assert not s.accuracy.applicable and not s.functionality.applicable
    assert s.usability.applicable and s.aggregate == pytest.approx(1.0)
    # a partial execute case: accuracy fails, usability passes, functionality fails (no probe)
    bad = _tr(tools=["camsol"], commands=[], tool_inputs={"camsol": {"chain": "B"}})
    s2 = eh.score_case(CASES["s2_camsol"], bad)            # wrong chain
    # usability passes (acted, no clarify) but accuracy+dispatch fail
    assert s2.usability.passed and not s2.accuracy.passed
    assert 0.0 < s2.aggregate < 1.0


# ════════════════════════════════════════════════════════════════════════════════
#  MANIFEST load / validate + disjointness
# ════════════════════════════════════════════════════════════════════════════════
def test_manifest_roundtrip(tmp_path):
    p = eh.write_sample_manifest(tmp_path / "m.json")
    loaded = eh.load_manifest(p)
    assert len(loaded) == len(eh.SAMPLE_CASES)
    assert {c.id for c in loaded} == {c.id for c in eh.SAMPLE_CASES}
    s3 = next(c for c in loaded if c.id == "s3_sel_scope")
    assert s3.gold_accuracy.required_args["scope"] == "20-30"
    assert "whole-chain" in s3.gold_accuracy.forbidden


def test_validate_rejects_bad_cases():
    with pytest.raises(ValueError):
        eh.validate_manifest([eh.EvalCase("x", "c", 9, "direct", "p",
                                          gold_usability=eh.GoldUsability("execute"),
                                          gold_accuracy=eh.GoldAccuracy(tools=["camsol"]))])  # bad tier
    with pytest.raises(ValueError):
        eh.validate_manifest([eh.EvalCase("y", "c", 1, "direct", "p")])  # missing usability
    with pytest.raises(ValueError):
        eh.validate_manifest([eh.EvalCase("z", "c", 1, "direct", "p",
                                          gold_usability=eh.GoldUsability("execute"))])  # execute w/o accuracy


def test_manifest_disjoint_from_example_pool():
    eh.assert_disjoint_from_examples(eh.SAMPLE_CASES)         # ok
    leak = eh.EvalCase(tc.EXAMPLE_POOL[0].id, "c", 1, "direct", "p",
                       gold_usability=eh.GoldUsability("clarify"))
    with pytest.raises(ValueError):
        eh.assert_disjoint_from_examples([leak])
    leak2 = eh.EvalCase("new_id", "c", 1, "direct", tc.EXAMPLE_POOL[0].prompt,
                        gold_usability=eh.GoldUsability("clarify"))
    with pytest.raises(ValueError):
        eh.assert_disjoint_from_examples([leak2])


def test_tool_registry_is_the_21_literals():
    import config
    assert eh.TOOL_REGISTRY == frozenset(t.lower() for t in config.TRANSLATOR_TOOL_NAMES)
    assert len(eh.TOOL_REGISTRY) == 21
    # every documented tool-input field key is a real registry tool (or chimerax)
    assert set(eh.TOOL_INPUT_FIELDS) <= (eh.TOOL_REGISTRY | {"chimerax"})


# ════════════════════════════════════════════════════════════════════════════════
#  TRIPLE-DISJOINTNESS guard (auto-arming) — EXAMPLE_POOL vs every held-out set
# ════════════════════════════════════════════════════════════════════════════════
def test_normalize_prompt_is_near_dup_proof():
    assert eh._normalize_prompt("Colour chain A red.") == eh._normalize_prompt("colour  chain a RED!!!")
    assert eh._normalize_prompt("Select within 5 Å of B") == eh._normalize_prompt("select within 5  of  b")


def test_example_pool_triple_disjoint_live():
    # EXAMPLE_POOL must be clean vs EVAL_CORPUS + every manifest currently present.
    counts = eh.assert_example_pool_disjoint()
    assert counts["EVAL_CORPUS"] == len(tc.EVAL_CORPUS)
    # the sample manifest is discovered and enforced
    assert "eval_manifest_sample.json" in counts


def _write_manifest(dir_path: Path, cases: list) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {"_schema": "eval_harness manifest v1", "cases": cases}
    (dir_path / "eval_corpus_manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_guard_auto_arms_on_new_manifest_id_collision(tmp_path):
    # A frozen-corpus file that reuses an EXAMPLE_POOL *id* must be caught the
    # moment it appears in scripts/ — no code change needed (auto-arming).
    leak_id = tc.EXAMPLE_POOL[0].id
    _write_manifest(tmp_path, [{
        "id": leak_id, "category": "c", "tier": 1, "challenge_type": "direct",
        "prompt": "a totally different prompt", "gold_usability": {"expected": "clarify"},
    }])
    with pytest.raises(ValueError):
        eh.assert_example_pool_disjoint(scripts_dir=tmp_path)


def test_guard_catches_near_duplicate_prompt(tmp_path):
    # A near-duplicate PROMPT (punctuation/case variant) of an EXAMPLE_POOL entry
    # must be caught even though the id differs.
    ex = tc.EXAMPLE_POOL[0]
    near_dup = ex.prompt.upper().replace(".", " !!! ")
    _write_manifest(tmp_path, [{
        "id": "frozen_new_id", "category": "c", "tier": 1, "challenge_type": "direct",
        "prompt": near_dup, "gold_usability": {"expected": "clarify"},
    }])
    with pytest.raises(ValueError):
        eh.assert_example_pool_disjoint(scripts_dir=tmp_path)


def test_guard_passes_when_manifest_is_disjoint(tmp_path):
    _write_manifest(tmp_path, [{
        "id": "frozen_unique_1", "category": "viz", "tier": 1, "challenge_type": "direct",
        "prompt": "Render the helices as flat ribbons in teal.",
        "gold_usability": {"expected": "execute"},
        "gold_accuracy": {"tools": ["chimerax"]},
    }])
    counts = eh.assert_example_pool_disjoint(scripts_dir=tmp_path)
    assert counts["eval_corpus_manifest.json"] == 1


def test_discover_skips_non_manifest_json(tmp_path):
    # a non-manifest JSON in the dir must be ignored, not crash discovery
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "benchmark_results.json").write_text(
        json.dumps({"some": "results", "rows": [1, 2, 3]}), encoding="utf-8")
    assert eh.discover_eval_manifests(tmp_path) == {}
    assert eh.discover_manifest_id_prompts(tmp_path) == {}


def test_guard_arms_on_EXTENDED_schema_manifest(tmp_path):
    # The frozen corpus extends the documented gold schema (session / clarify_about
    # / command_contains_any / tools-as-string). The leakage guard must still arm on
    # it (it only needs id + prompt) — a richer schema must NOT silently bypass it.
    tmp_path.mkdir(parents=True, exist_ok=True)
    ext_case = {
        "id": tc.EXAMPLE_POOL[0].id,                 # an EXAMPLE_POOL id collision
        "category": "viz", "tier": 1, "challenge_type": "direct",
        "prompt": "a unique prompt that won't clash",
        "session": {"models": [{"id": "#1", "pdb": "2LZM"}], "selection": None},
        "gold_accuracy": {"tools": "chimerax",
                          "required_args": {"command_contains_any": ["color #1/A red"]}},
        "gold_usability": {"expected": "clarify", "clarify_about": "which chain"},
    }
    (tmp_path / "eval_corpus_manifest.json").write_text(
        json.dumps({"schema_version": 2, "cases": [ext_case]}), encoding="utf-8")
    # tolerant discovery still reads it (id + prompt), even though strict load fails
    assert "eval_corpus_manifest.json" in eh.discover_manifest_id_prompts(tmp_path)
    with pytest.raises(ValueError):
        eh.assert_example_pool_disjoint(scripts_dir=tmp_path)
