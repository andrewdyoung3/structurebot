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
from unittest.mock import MagicMock

import requests as _requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import translator_corpus as corpus
import translator_benchmark as bm
from translator import CommandTranslator

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

def test_scorer_router_case_sensitivity() -> None:
    print("\n=== B. scorer / case-sensitivity ===")
    case = next(c for c in corpus.EVAL_CORPUS if c.id == "camsol_scan")  # tools_any camsol
    _assert(corpus.score_case(case, _result(tools=["chimerax", "camsol"]))[0],
            "exact 'camsol' → pass (matches the router literal)")
    _assert(not corpus.score_case(case, _result(tools=["CamSol"]))[0],
            "'CamSol' (wrong case) → FAIL (router is case-sensitive; would not route)")
    _assert(not corpus.score_case(case, _result(tools=["chimerax"]))[0],
            "missing camsol → fail")


def test_check_kinds() -> None:
    r = _result(commands=["select #1/B :<4.5 & #1/A", "info residues sel"])
    _assert(corpus.Check("cmd_re", r":<").evaluate(r), "cmd_re matches")
    _assert(corpus.Check("no_cmd_re", r"\bzone\b").evaluate(r), "no_cmd_re passes when absent")
    _assert(not corpus.Check("no_cmd_re", r"info").evaluate(r), "no_cmd_re fails when present")


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
    test_scorer_router_case_sensitivity(); test_check_kinds()
    test_full_vs_raw_guard_split(); test_schema_validity()
    test_single_pass_bucketing(); test_nrun_aggregation_math()
    test_markdown_report_builds()
    print("\n(forced-backend tests need pytest monkeypatch)")
    print(f"\nResults: {_results['pass']} passed, {_results['fail']} failed")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
