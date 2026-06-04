"""
tests/test_translator_benchmark_logic.py
----------------------------------------
CI tests for the translator benchmark harness — corpus loader, scorer (incl.
**router case-sensitivity**), FULL-vs-RAW guard split, schema, single-pass
bucketing, **N-run aggregation math**, **eval/example disjointness**, **per-
category balance**, forced-backend + fallback-rescue EXCLUSION. All synthetic/
mocked: no live Ollama, no live API, no live ChimeraX. The benchmark RUN is
opt-in (tests/test_translator_benchmark.py).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests as _requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import translator_corpus as corpus
import translator_benchmark as bm
from translator import CommandTranslator
from tool_router import ToolRouter, ToolStepResult

PASS, FAIL = "[PASS]", "[FAIL]"
_results = {"pass": 0, "fail": 0}
def _ok(n): print(f"  {PASS} {n}"); _results["pass"] += 1
def _fail(n, w): print(f"  {FAIL} {n}: {w}"); _results["fail"] += 1
def _assert(c, n, m=""):
    (_ok(n) if c else _fail(n, m or "assertion failed")); return c


def _result(commands=None, tools=None, **over):
    r = {"commands": commands or [], "explanations": [], "warnings": [],
         "clarification_needed": None, "confidence": "high",
         "tools_needed": tools or ["chimerax"], "tool_inputs": {}}
    r.update(over); return r


# -- A. corpus loader + balance + disjointness ---------------------------------

def test_corpus_loads_and_balanced() -> None:
    print("\n=== A. corpus / balance / disjointness ===")
    counts = corpus.category_counts(corpus.EVAL_CORPUS)
    _assert(len(corpus.EVAL_CORPUS) >= 88, "eval corpus ~90+ cases",
            f"n={len(corpus.EVAL_CORPUS)}")
    _assert(all(c.prompt and c.category and c.checks for c in corpus.EVAL_CORPUS),
            "every eval case has prompt + category + checks")
    under = {k: v for k, v in counts.items() if v < corpus.MIN_PER_CATEGORY}
    _assert(not under, f"every category has ≥{corpus.MIN_PER_CATEGORY} eval cases",
            f"under-filled: {under}")
    for emph in ("camsol", "mpnn", "selection_scope", "hide_show", "zone"):
        _assert(counts.get(emph, 0) >= corpus.MIN_PER_CATEGORY,
                f"emphasised category {emph} ≥{corpus.MIN_PER_CATEGORY}", f"got {counts.get(emph)}")


def test_eval_example_disjoint() -> None:
    eval_ids = {c.id for c in corpus.EVAL_CORPUS}
    ex_ids   = {c.id for c in corpus.EXAMPLE_POOL}
    eval_p   = {c.prompt.strip().lower() for c in corpus.EVAL_CORPUS}
    ex_p     = {c.prompt.strip().lower() for c in corpus.EXAMPLE_POOL}
    _assert(eval_ids.isdisjoint(ex_ids), "eval/example ids disjoint",
            f"overlap {eval_ids & ex_ids}")
    _assert(eval_p.isdisjoint(ex_p), "eval/example PROMPTS disjoint (no leakage)",
            f"overlap {eval_p & ex_p}")
    _assert(len(corpus.EXAMPLE_POOL) >= 10, "example pool is non-trivial",
            f"n={len(corpus.EXAMPLE_POOL)}")


# -- B. scorer + ROUTER CASE-SENSITIVITY ---------------------------------------

def test_scorer_case_insensitive_matches_router() -> None:
    print("\n=== B. scorer / case-INSENSITIVITY (matches the router) ===")
    case = next(c for c in corpus.EVAL_CORPUS if c.id == "camsol_scan")  # tools_any camsol
    _assert(corpus.score_case(case, _result(tools=["chimerax", "camsol"]))[0],
            "'camsol' → pass")
    _assert(corpus.score_case(case, _result(tools=["CamSol"]))[0],
            "'CamSol' → PASS (scorer is case-insensitive, in lockstep with the router)")
    _assert(corpus.score_case(case, _result(tools=["ESM"]))[0] is False,
            "wrong tool ('ESM') still fails the camsol check")
    _assert(not corpus.score_case(case, _result(tools=["chimerax"]))[0],
            "missing camsol → fail")


def test_router_and_scorer_case_consistent() -> None:
    """The honest-scorer invariant: scorer matches the real router. Both must
    resolve a wrong-cased tool name."""
    r = ToolRouter(bridge=MagicMock(), session=MagicMock())
    with patch.object(r, "_run_camsol",
                      return_value=ToolStepResult(tool="camsol", success=True)) as m:
        r._dispatch_tool("CamSol", {})       # wrong case
    _assert(m.called, "router._dispatch_tool routes 'CamSol' → camsol (case-insensitive)")
    case = next(c for c in corpus.EVAL_CORPUS if c.id == "camsol_scan")
    _assert(corpus.score_case(case, _result(tools=["CamSol"]))[0],
            "scorer ALSO accepts 'CamSol' (router + scorer consistent)")


def test_few_shot_example_pool_only() -> None:
    """Few-shot demos come ONLY from EXAMPLE_POOL (zero EVAL_CORPUS leakage) and
    target only the failing categories; each output is a verified gold label."""
    eval_p = {c.prompt.strip().lower() for c in corpus.EVAL_CORPUS}
    ex_by_prompt = {c.prompt.strip().lower(): c for c in corpus.EXAMPLE_POOL}
    pairs = corpus.few_shot_pairs()
    _assert(len(pairs) >= 8, "few-shot has demos", f"n={len(pairs)}")
    for prompt, output in pairs:
        p = prompt.strip().lower()
        _assert(p in ex_by_prompt, f"few-shot prompt sourced from EXAMPLE_POOL: {prompt!r}")
        _assert(p not in eval_p, f"few-shot prompt NOT in EVAL_CORPUS (no leakage): {prompt!r}")
        c = ex_by_prompt[p]
        _assert(c.category in corpus.FEW_SHOT_CATEGORIES,
                f"few-shot category is a failing one: {c.category}")
        _assert(corpus.score_case(c, output)[0],
                f"few-shot OUTPUT is a verified correct label for {c.id}")


def test_few_shot_message_assembly() -> None:
    import json as _json
    msgs = corpus.few_shot_messages()
    _assert(len(msgs) == 2 * len(corpus.few_shot_pairs()), "alternating user/assistant pairs")
    _assert(msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", "user then assistant")
    for m in msgs[1::2]:
        obj = _json.loads(m["content"])
        _assert("tools_needed" in obj and isinstance(obj["tools_needed"], list),
                "assistant content is a valid translation JSON")


def test_check_kinds() -> None:
    r = _result(commands=["select #1/B :<4.5 & #1/A", "info residues sel"])
    _assert(corpus.Check("cmd_re", r":<").evaluate(r), "cmd_re matches")
    _assert(corpus.Check("no_cmd_re", r"\bzone\b").evaluate(r), "no_cmd_re passes when absent")
    _assert(not corpus.Check("no_cmd_re", r"info").evaluate(r), "no_cmd_re fails when present")
    # behaviour check kinds (clarify / refuse exemplars)
    clar = _result(tools=[], clarification_needed="Which chain — A or B?")
    _assert(corpus.Check("clar_set").evaluate(clar), "clar_set true when a question is asked")
    _assert(corpus.Check("no_action").evaluate(clar), "no_action true with no tool & no command")
    _assert(not corpus.Check("clar_set").evaluate(_result()), "clar_set false on a normal action")
    ref = _result(tools=[], refused=True)
    _assert(corpus.Check("refused").evaluate(ref), "refused true when declined")
    _assert(not corpus.Check("refused").evaluate(_result()), "refused false on a normal action")


def test_few_shot_pool_v1_is_mostly_execute_with_one_clarify_and_refuse() -> None:
    """FEW_SHOT_POOL_V1 is a FIXED, named intervention: mostly execute, with at
    least one CLARIFY and one REFUSE exemplar so the execute-heavy pool doesn't bias
    the model away from clarifying (T4) or erode refuse behaviour. Reproducible."""
    pairs = corpus.few_shot_pairs()
    ex = cl = rf = 0
    for _prompt, out in pairs:
        if out.get("refused"):
            rf += 1
        elif out.get("clarification_needed"):
            cl += 1
        else:
            ex += 1
    _assert(cl >= 1, "pool has a CLARIFY exemplar", f"clarify={cl}")
    _assert(rf >= 1, "pool has a REFUSE exemplar", f"refuse={rf}")
    _assert(ex > (cl + rf), "pool is MOSTLY execute (not a thumb on the scale)",
            f"execute={ex} vs clarify+refuse={cl + rf}")
    # the targeted weak-spot categories are actually wired in
    cats = {c.category for c in corpus.EXAMPLE_POOL
            if c.id in corpus.FEW_SHOT_OUTPUTS}
    for need in ("selection_scope", "zone", "mutation_scan", "multi_tool", "clarify", "safety"):
        _assert(need in cats, f"weak-spot category {need!r} is in the few-shot pool")


def test_few_shot_pool_v2_mutation_scan_boundary_contrasts() -> None:
    """FEW_SHOT_POOL_V2 adds CONTRAST exemplars that teach mutation_scan's boundaries
    (correcting the V1 over-attraction). Each routes to the CORRECT neighbour, NOT
    mutation_scan — and the V1 mutation_scan wins (proline, inferential) are RETAINED."""
    out = corpus.FEW_SHOT_OUTPUTS
    def _tools(cid):
        return {t.lower() for t in out[cid]["tools_needed"]}
    # the three boundaries route away from mutation_scan, to the right tool
    _assert(_tools("ex_esm_3") == {"esm"}, "esm boundary -> esm (not mutation_scan)")
    _assert(_tools("ex_mpnn_3") == {"proteinmpnn"}, "mpnn boundary -> proteinmpnn (not mutation_scan)")
    _assert(_tools("ex_multi_3") == {"camsol", "esm"}, "compound boundary -> camsol+esm (not mutation_scan)")
    for cid in ("ex_esm_3", "ex_mpnn_3", "ex_multi_3"):
        _assert("mutation_scan" not in _tools(cid), f"{cid} does NOT route mutation_scan")
    # RETAIN V1: the mutation_scan exemplars are still present (boundaries didn't delete the wins)
    _assert(_tools("ex_pro_1") == {"mutation_scan"}, "V1 proline route retained")
    _assert(_tools("ex_infer_1") == {"mutation_scan"}, "V1 inferential-stability route retained")


# -- C. FULL vs RAW (guard split) ----------------------------------------------

def test_full_vs_raw_guard_split() -> None:
    print("\n=== C. full vs raw ===")
    case = next(c for c in corpus.EVAL_CORPUS if c.id == "zone_iface_list")
    raw = _result(commands=["select #1/A & (zone #1/B 4.5)", "info residues sel"])
    _assert(not corpus.score_case(case, raw)[0], "RAW fails — Chimera-1 `zone`, no `:<`")
    full = bm._apply_guard(raw)
    _assert(corpus.score_case(case, full)[0], "FULL passes — guard rewrote zone→:< (guard-rescue)")


# -- D. schema validity --------------------------------------------------------

def test_schema_validity() -> None:
    print("\n=== D. schema ===")
    _assert(corpus.is_schema_valid(_result()), "well-formed 7-key dict valid")
    _assert(not corpus.is_schema_valid({"commands": []}), "missing keys invalid")
    _assert(not corpus.is_schema_valid(_result(confidence="bananas")), "bad confidence invalid")
    _assert(not corpus.is_schema_valid(_result(tools_needed=[])), "empty tools invalid")


# -- E. single-pass bucketing + N-run aggregation math -------------------------

def _row(cat, full, raw, routing=True, lat=0.5):
    return {"id": cat, "category": cat, "prompt": "", "raw_pass": raw, "full_pass": full,
            "schema_valid": True, "routing_ok": routing, "latency_s": lat,
            "error": None, "tools_needed": [], "full_commands": []}

def test_single_pass_bucketing() -> None:
    print("\n=== E. aggregation ===")
    rows = [_row("camsol", True, True), _row("camsol", False, False),
            _row("zone", True, False)]
    s = bm.aggregate(rows)
    _assert(s["by_category"]["camsol"] == {"n": 2, "full": 1, "raw": 1}, "camsol bucket",
            f"got {s['by_category']['camsol']}")
    _assert(s["full_pass"] == 2 and s["raw_pass"] == 1, "overall counts")


def test_nrun_aggregation_math() -> None:
    # idx 0,1 = camsol; 2,3 = mpnn. run1 all pass; run2 = [T,F,T,F].
    def rows(passes):
        out = []
        for i, p in enumerate(passes):
            out.append(_row("camsol" if i < 2 else "mpnn", p, p, routing=p))
        return out
    run1 = rows([True, True, True, True])     # rate 1.0
    run2 = rows([True, False, True, False])   # rate 0.5
    s = bm.aggregate_runs([run1, run2])
    _assert(s["n_runs"] == 2 and s["n_cases"] == 4, "n_runs / n_cases")
    _assert(abs(s["full"]["mean"] - 0.75) < 1e-9 and s["full"]["min"] == 0.5 and s["full"]["max"] == 1.0,
            "overall FULL mean 0.75 [0.5–1.0]", f"got {s['full']}")
    cm = s["by_category"]["camsol"]["full"]
    _assert(abs(cm["mean"] - 0.75) < 1e-9 and cm["min"] == 0.5 and cm["max"] == 1.0,
            "camsol per-category FULL mean 0.75 [0.5–1.0]", f"got {cm}")
    _assert(abs(s["routing"]["mean"] - 0.75) < 1e-9, "routing mean averaged over runs",
            f"got {s['routing']}")


# -- F. forced backend + fallback EXCLUSION ------------------------------------

class _StubBackend:
    name = "ollama"
    def __init__(self, result=None, raises=False):
        self.result, self.raises = result, raises
    def translate(self, translator, user_input, session):
        if self.raises:
            raise _requests.exceptions.ConnectionError("ollama down")
        return dict(self.result)


def test_forced_backend_no_claude(monkeypatch) -> None:
    print("\n=== F. forced backend / no fallback ===")
    monkeypatch.setattr(bm, "make_backend", lambda n: _StubBackend(result=_result(tools=["camsol"])))
    t = CommandTranslator(api_key="sk-ant-test"); t.client = MagicMock()
    rows = bm.run_backend("ollama", cases=corpus.EVAL_CORPUS[:3], translator=t)
    _assert(len(rows) == 3, "ran the forced backend over the cases")
    _assert(not t.client.messages.create.called, "Claude client NEVER called for ollama")


def test_failing_backend_not_rescued(monkeypatch) -> None:
    monkeypatch.setattr(bm, "make_backend", lambda n: _StubBackend(raises=True))
    t = CommandTranslator(api_key="sk-ant-test"); t.client = MagicMock()
    t.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"commands":["x"],"tools_needed":["camsol"]}')], stop_reason="end_turn")
    rows = bm.run_backend("ollama", cases=corpus.EVAL_CORPUS[:3], translator=t)
    _assert(all(r["error"] for r in rows), "down backend → every case errors")
    _assert(all(not r["full_pass"] and not r["raw_pass"] for r in rows),
            "errored cases count as failures (no smoothing)")
    _assert(not t.client.messages.create.called, "Claude NOT consulted to rescue local backend")


def test_markdown_report_builds() -> None:
    print("\n=== G. report ===")
    rows = [_row("camsol", True, True), _row("mpnn", False, False)]
    comp = {"claude": {"runs": [rows], "summary": bm.aggregate_runs([rows])},
            "ollama": {"runs": [rows], "summary": bm.aggregate_runs([rows])}}
    md = bm.build_markdown(comp, model_label="qwen3:8b")
    _assert("Translator backend benchmark" in md and "Per-category" in md, "markdown sections")
    _assert("N = 1 runs" in md and "mean [min–max]" in md, "reports N-run mean/range")


def main() -> int:
    print("=" * 60); print("tests/test_translator_benchmark_logic.py"); print("=" * 60)
    test_corpus_loads_and_balanced(); test_eval_example_disjoint()
    test_scorer_case_insensitive_matches_router(); test_router_and_scorer_case_consistent()
    test_few_shot_example_pool_only(); test_few_shot_message_assembly()
    test_check_kinds()
    test_full_vs_raw_guard_split(); test_schema_validity()
    test_single_pass_bucketing(); test_nrun_aggregation_math()
    test_markdown_report_builds()
    print("\n(forced-backend tests need pytest monkeypatch)")
    print(f"\nResults: {_results['pass']} passed, {_results['fail']} failed")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
