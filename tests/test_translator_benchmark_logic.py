"""
tests/test_translator_benchmark_logic.py
----------------------------------------
CI tests for the translator benchmark harness logic — corpus loader, scorer,
schema validity, per-category bucketing, FULL-vs-RAW (guard) split, forced-
backend selection, and fallback-rescue EXCLUSION. All synthetic/mocked: no live
Ollama, no live API, no live ChimeraX. The actual benchmark RUN is opt-in
(tests/test_translator_benchmark.py).
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
    r = {
        "commands": commands or [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "high",
        "tools_needed": tools or ["chimerax"], "tool_inputs": {},
    }
    r.update(over); return r


# -- A. corpus loader ----------------------------------------------------------

def test_corpus_loads() -> None:
    print("\n=== A. corpus ===")
    _assert(len(corpus.CORPUS) >= 12, "corpus has cases", f"n={len(corpus.CORPUS)}")
    _assert(all(c.prompt and c.category and c.checks for c in corpus.CORPUS),
            "every case has prompt + category + checks")
    ids = [c.id for c in corpus.CORPUS]
    _assert(len(ids) == len(set(ids)), "case ids unique")
    cats = set(corpus.categories())
    _assert({"zone", "hide_show", "mpnn", "camsol", "esm", "disulfide"} <= cats,
            "core rule categories present", f"got {sorted(cats)}")


# -- B. scorer classifies matching vs non-matching -----------------------------

def test_scorer_pass_and_fail() -> None:
    print("\n=== B. scorer ===")
    case = next(c for c in corpus.CORPUS if c.id == "camsol_scan")   # tools_any camsol
    _assert(corpus.score_case(case, _result(tools=["chimerax", "camsol"]))[0],
            "matching tools → pass")
    _assert(not corpus.score_case(case, _result(tools=["chimerax"]))[0],
            "missing camsol → fail")


def test_check_kinds() -> None:
    r = _result(commands=["select #1/B :<4.5 & #1/A", "info residues sel"])
    _assert(corpus.Check("cmd_re", r":<").evaluate(r), "cmd_re matches")
    _assert(corpus.Check("no_cmd_re", r"\bzone\b").evaluate(r), "no_cmd_re passes when absent")
    _assert(not corpus.Check("no_cmd_re", r"info").evaluate(r), "no_cmd_re fails when present")
    _assert(corpus.Check("tools_all", ["chimerax"]).evaluate(r), "tools_all")
    _assert(corpus.Check("clar_none").evaluate(r), "clar_none")


# -- C. FULL vs RAW (guard split) ----------------------------------------------

def test_full_vs_raw_guard_split() -> None:
    print("\n=== C. full vs raw ===")
    case = next(c for c in corpus.CORPUS if c.id == "zone_interface")
    raw = _result(commands=["select #1/A & (zone #1/B 4.5)", "info residues sel"])
    _assert(not corpus.score_case(case, raw)[0],
            "RAW (pre-guard) fails — has Chimera-1 `zone`, no `:<`")
    full = bm._apply_guard(raw)
    _assert(corpus.score_case(case, full)[0],
            "FULL (post-guard) passes — guard rewrote zone→:< (guard-rescue)")


# -- D. schema validity --------------------------------------------------------

def test_schema_validity() -> None:
    print("\n=== D. schema ===")
    _assert(corpus.is_schema_valid(_result()), "well-formed 7-key dict is valid")
    _assert(not corpus.is_schema_valid({"commands": []}), "missing keys → invalid")
    _assert(not corpus.is_schema_valid(_result(confidence="bananas")), "bad confidence → invalid")
    _assert(not corpus.is_schema_valid(_result(tools_needed=[])), "empty tools_needed → invalid")
    _assert(not corpus.is_schema_valid("not a dict"), "non-dict → invalid")


# -- E. aggregation / bucketing ------------------------------------------------

def _row(cat, raw=True, full=True, schema=True, routing=True, lat=0.5, err=None):
    return {"id": cat, "category": cat, "prompt": "", "raw_pass": raw, "full_pass": full,
            "schema_valid": schema, "routing_ok": routing, "latency_s": lat,
            "error": err, "tools_needed": [], "full_commands": []}

def test_aggregate_buckets() -> None:
    print("\n=== E. aggregate ===")
    rows = [_row("camsol", full=True, raw=True), _row("camsol", full=False, raw=False),
            _row("zone", full=True, raw=False), _row("zone", routing=None, full=True, raw=True)]
    s = bm.aggregate(rows)
    _assert(s["n"] == 4, "counts rows")
    _assert(s["by_category"]["camsol"] == {"n": 2, "full": 1, "raw": 1}, "camsol bucket",
            f"got {s['by_category']['camsol']}")
    _assert(s["by_category"]["zone"]["full"] == 2 and s["by_category"]["zone"]["raw"] == 1,
            "zone bucket full=2 raw=1")
    _assert(s["full_pass"] == 3 and s["raw_pass"] == 2, "overall full/raw counts")
    _assert(s["routing_n"] == 3, "routing denominator excludes None")


# -- F. forced backend + fallback EXCLUSION ------------------------------------

class _StubBackend:
    name = "ollama"
    def __init__(self, result=None, raises=False):
        self.result, self.raises = result, raises
    def translate(self, translator, user_input, session):
        if self.raises:
            raise _requests.exceptions.ConnectionError("ollama down")
        return dict(self.result)


def test_forced_backend_runs_directly(monkeypatch) -> None:
    print("\n=== F. forced backend / no fallback ===")
    stub = _StubBackend(result=_result(tools=["chimerax", "camsol"]))
    monkeypatch.setattr(bm, "make_backend", lambda n: stub)
    t = CommandTranslator(api_key="sk-ant-test")
    t.client = MagicMock()
    rows = bm.run_backend("ollama", cases=corpus.CORPUS[:3], translator=t)
    _assert(len(rows) == 3, "ran the forced backend over the cases")
    _assert(not t.client.messages.create.called,
            "Claude client NEVER called when benchmarking ollama")


def test_failing_backend_not_rescued_by_claude(monkeypatch) -> None:
    """A down under-test backend yields FAILURES — a Claude fallback must NOT
    rescue the numbers (the harness calls the backend directly)."""
    monkeypatch.setattr(bm, "make_backend", lambda n: _StubBackend(raises=True))
    t = CommandTranslator(api_key="sk-ant-test")
    t.client = MagicMock()
    # make Claude "succeed" if it were (wrongly) consulted
    t.client.messages.create.return_value = MagicMock(
        content=[MagicMock(text='{"commands":["x"],"tools_needed":["chimerax"]}')],
        stop_reason="end_turn")
    rows = bm.run_backend("ollama", cases=corpus.CORPUS[:3], translator=t)
    _assert(all(r["error"] for r in rows), "down backend → every case errors")
    _assert(all(not r["full_pass"] and not r["raw_pass"] for r in rows),
            "errored cases count as failures (no smoothing)")
    _assert(not t.client.messages.create.called,
            "Claude was NOT consulted to rescue the local backend")


def test_markdown_report_builds() -> None:
    print("\n=== G. report ===")
    comp = {
        "claude": {"rows": [_row("camsol")], "summary": bm.aggregate([_row("camsol")])},
        "ollama": {"rows": [_row("camsol", full=False)], "summary": bm.aggregate([_row("camsol", full=False)])},
    }
    md = bm.build_markdown(comp, model_label="qwen3:8b")
    _assert("Translator backend benchmark" in md and "Per-category" in md, "markdown has sections")
    _assert("claude" in md and "ollama" in md, "both backends in the table")


def main() -> int:
    print("=" * 60); print("tests/test_translator_benchmark_logic.py"); print("=" * 60)
    test_corpus_loads(); test_scorer_pass_and_fail(); test_check_kinds()
    test_full_vs_raw_guard_split(); test_schema_validity(); test_aggregate_buckets()
    test_markdown_report_builds()
    print("\n(forced-backend tests need pytest monkeypatch)")
    print(f"\nResults: {_results['pass']} passed, {_results['fail']} failed")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
