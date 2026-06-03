"""
translator_benchmark.py
-----------------------
Translator eval/benchmark harness — measure local (Ollama) vs Claude translation
quality on StructureBot's real corpus (``translator_corpus.CORPUS``), per rule
category. A MEASUREMENT tool for model selection / the fallback bar, NOT a CI gate.

Two scored columns per item (the locked split):
  • FULL   — `CommandTranslator.translate()` output, i.e. backend.translate() THEN
             the backend-agnostic `_sanitize_zone_syntax` guard. This is the
             headline number (the actual experience downstream).
  • RAW    — the pre-guard backend.translate() output. RAW vs FULL localises
             model-quality vs guard-rescue → tells you whether a fix belongs in
             the model (few-shot/fine-tune) or in the guards.

Honesty: deterministic (temperature 0); each backend is forced explicitly and
run DIRECTLY via make_backend(name).translate() — no Claude fallback can rescue
the local model's numbers; raw counts reported, no smoothing.

Metrics per backend: pass-rate (full + raw), per-category breakdown, schema-
validity rate, tool-routing accuracy, latency.
"""

from __future__ import annotations

import copy
import csv
import datetime as _dt
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
import translator_corpus as corpus
from translator import CommandTranslator, make_backend, _sanitize_zone_syntax

# Fixed deterministic session context (1HSG — the §7 reference structure) so the
# corpus prompts resolve to a known model + chains without any network/ChimeraX.
_BENCH_CONTEXT = (
    "Loaded structures:\n"
    "  #1: 1HSG — HIV-1 protease (homodimer, chains A and B; ligand MK1)\n"
)

_ARTIFACT_DIR = Path(__file__).parent / "scripts"


class _FixedSession:
    """Minimal SessionState stand-in: a constant context summary."""
    def get_context_summary(self) -> str:
        return _BENCH_CONTEXT


def _apply_guard(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Replicate CommandTranslator.translate()'s post-step: the backend-agnostic
    guard. FULL == _apply_guard(backend.translate(...)) by construction (translate
    does exactly backend.translate + this guard; no fallback for a forced backend)."""
    full = copy.deepcopy(raw)
    nc, ne, notes = _sanitize_zone_syntax(
        full.get("commands") or [], full.get("explanations") or [])
    if notes:
        full["commands"]     = nc
        full["explanations"] = ne
        full["warnings"]     = (full.get("warnings") or []) + notes
    return full


def run_backend(name: str,
                cases: Optional[List[corpus.CorpusCase]] = None,
                translator: Optional[CommandTranslator] = None) -> List[Dict[str, Any]]:
    """
    Run every corpus case through the FORCED backend *name*, directly via
    `make_backend(name).translate()` (no fallback path). Returns per-case rows
    with raw/full pass, schema validity, routing, latency.
    """
    cases = cases if cases is not None else corpus.CORPUS
    t = translator or CommandTranslator(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or "sk-ant-benchmark-dummy")
    # Force the backend and call it DIRECTLY (not via t.translate()): the one-way
    # fallback lives in CommandTranslator.translate, so calling the backend
    # directly makes a Claude rescue of the local model STRUCTURALLY impossible —
    # honest local numbers. (No need to touch the global config.TRANSLATOR_BACKEND.)
    t._backend = make_backend(name)
    session = _FixedSession()

    rows: List[Dict[str, Any]] = []
    for case in cases:
        t.reset_conversation()
        t0 = time.perf_counter()
        try:
            raw = t._backend.translate(t, case.prompt, session)
            err = None
        except Exception as exc:                # benchmark honesty: a down backend = failures
            raw, err = {}, f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - t0
        full = _apply_guard(raw) if isinstance(raw, dict) and raw else {}

        rows.append({
            "id":           case.id,
            "category":     case.category,
            "prompt":       case.prompt,
            "raw_pass":     corpus.score_case(case, raw)[0] if raw else False,
            "full_pass":    corpus.score_case(case, full)[0] if full else False,
            "schema_valid": corpus.is_schema_valid(raw),
            "routing_ok":   (corpus.tool_routing_ok(case, full)
                             if case.has_tool_expectation and full else
                             (None if not case.has_tool_expectation else False)),
            "latency_s":    round(dt, 2),
            "error":        err,
            "tools_needed": raw.get("tools_needed") if isinstance(raw, dict) else None,
            "full_commands": full.get("commands") if isinstance(full, dict) else None,
        })
    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-backend rows into the metric summary (+ per-category buckets)."""
    n = len(rows) or 1
    routing_rows = [r for r in rows if r["routing_ok"] is not None]
    lat = [r["latency_s"] for r in rows] or [0.0]
    cats: Dict[str, Dict[str, int]] = {}
    for r in rows:
        c = cats.setdefault(r["category"], {"n": 0, "full": 0, "raw": 0})
        c["n"] += 1
        c["full"] += int(r["full_pass"])
        c["raw"]  += int(r["raw_pass"])
    return {
        "n":              len(rows),
        "full_pass":      sum(r["full_pass"] for r in rows),
        "raw_pass":       sum(r["raw_pass"] for r in rows),
        "full_rate":      sum(r["full_pass"] for r in rows) / n,
        "raw_rate":       sum(r["raw_pass"] for r in rows) / n,
        "schema_rate":    sum(r["schema_valid"] for r in rows) / n,
        "routing_n":      len(routing_rows),
        "routing_ok":     sum(r["routing_ok"] for r in routing_rows),
        "routing_rate":   (sum(r["routing_ok"] for r in routing_rows) / len(routing_rows))
                          if routing_rows else 0.0,
        "latency_mean":   statistics.mean(lat),
        "latency_median": statistics.median(lat),
        "latency_max":    max(lat),
        "n_errors":       sum(1 for r in rows if r["error"]),
        "by_category":    cats,
    }


def _mm(vals: List[float]) -> Dict[str, float]:
    """mean / min / max of a per-run rate list."""
    vals = list(vals) or [0.0]
    return {"mean": statistics.mean(vals), "min": min(vals), "max": max(vals)}


def aggregate_runs(run_rows: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Aggregate N runs (each a full pass over the corpus) into the HONEST summary:
    per-backend **mean pass-rate with min/max range over the N runs**, plus
    per-category mean±range. The pass bar is the MEAN over N≥5 runs — single-shot
    numbers are never used for comparisons (the local model is non-deterministic
    even at temperature 0).
    """
    run_rows = [rs for rs in run_rows if rs]
    n_cases = len(run_rows[0]) if run_rows else 0

    def _per_run_rate(rows, key):
        return (sum(r[key] for r in rows) / len(rows)) if rows else 0.0

    def _per_run_routing(rows):
        rr = [r for r in rows if r["routing_ok"] is not None]
        return (sum(r["routing_ok"] for r in rr) / len(rr)) if rr else 0.0

    full = _mm([_per_run_rate(rs, "full_pass") for rs in run_rows])
    raw  = _mm([_per_run_rate(rs, "raw_pass")  for rs in run_rows])
    schema = _mm([_per_run_rate(rs, "schema_valid") for rs in run_rows])
    routing = _mm([_per_run_routing(rs) for rs in run_rows])
    lat = [r["latency_s"] for rs in run_rows for r in rs] or [0.0]

    cat_names = sorted({r["category"] for rs in run_rows for r in rs})
    by_category: Dict[str, Any] = {}
    for cat in cat_names:
        full_cr, raw_cr = [], []
        for rs in run_rows:
            crows = [r for r in rs if r["category"] == cat]
            cn = len(crows) or 1
            full_cr.append(sum(r["full_pass"] for r in crows) / cn)
            raw_cr.append(sum(r["raw_pass"] for r in crows) / cn)
        by_category[cat] = {
            "n":    len([r for r in run_rows[0] if r["category"] == cat]) if run_rows else 0,
            "full": _mm(full_cr),
            "raw":  _mm(raw_cr),
        }

    return {
        "n_runs": len(run_rows), "n_cases": n_cases,
        "full": full, "raw": raw, "schema": schema, "routing": routing,
        "latency_mean": statistics.mean(lat), "latency_median": statistics.median(lat),
        "latency_max": max(lat),
        "n_errors": sum(1 for rs in run_rows for r in rs if r["error"]),
        "by_category": by_category,
    }


def run_comparison(backends=("claude", "ollama"), runs: int = 1,
                   cases: Optional[List[corpus.CorpusCase]] = None,
                   translator: Optional[CommandTranslator] = None) -> Dict[str, Any]:
    """
    Run each backend over the corpus *runs* times; return
    {backend: {"runs": [rows, …], "summary": aggregate_runs(...)}}.
    The pass bar = mean over N≥5 runs on the held-out EVAL_CORPUS.
    """
    cases = cases if cases is not None else corpus.EVAL_CORPUS
    t = translator or CommandTranslator(
        api_key=os.environ.get("ANTHROPIC_API_KEY") or "sk-ant-benchmark-dummy")
    out: Dict[str, Any] = {}
    for b in backends:
        run_rows = [run_backend(b, cases=cases, translator=t) for _ in range(max(1, runs))]
        out[b] = {"runs": run_rows, "summary": aggregate_runs(run_rows)}
    return out


# ── Reporting (mean ± range over N runs) ─────────────────────────────────────────
def _pct_range(d: Dict[str, float]) -> str:
    return f"{d['mean']*100:.0f}% [{d['min']*100:.0f}–{d['max']*100:.0f}]"


def build_markdown(comparison: Dict[str, Any], model_label: str = "") -> str:
    backends = list(comparison)
    S = {b: comparison[b]["summary"] for b in backends}
    n_runs = S[backends[0]]["n_runs"] if backends else 0
    n_cases = S[backends[0]]["n_cases"] if backends else 0
    L: List[str] = [f"# Translator backend benchmark — {_dt.date.today().isoformat()}"]
    L.append("\n> **SUPERSEDED** by the model-independent 3-dimension harness "
             "(`eval_harness.py`: accuracy / functionality / usability, gold human-defined, "
             "Claude scored as a contestant). This routing+syntax \"% of Claude\" artifact is "
             "kept for history only — do not read it as end-to-end correctness.")
    L.append(f"\n_Ollama model: **{model_label or config.OLLAMA_MODEL}**_  ·  "
             f"eval corpus: {n_cases} cases × {len(corpus.categories())} categories  ·  "
             f"**N = {n_runs} runs** (mean [min–max]). FULL = post-guard (`translate()`); "
             f"RAW = pre-guard (`backend.translate()`). Bar = mean over N≥5.")
    L.append("\n> **Scope of this score: ROUTING + COMMAND SYNTAX only.** A case passes on "
             "correct tool routing and well-formed ChimeraX commands; **`tool_inputs` "
             "(arguments / scope) are NOT checked**, so this is *not* an end-to-end "
             "correctness number. The argument/scope-checking upgrade is a separate item.")
    L.append("\n## Overall (mean [min–max] over N runs)\n")
    L.append("| Metric | " + " | ".join(backends) + " |")
    L.append("|--------|" + "|".join(["---"] * len(backends)) + "|")
    L.append("| Pass-rate FULL (post-guard) | " + " | ".join(_pct_range(S[b]["full"]) for b in backends) + " |")
    L.append("| Pass-rate RAW (pre-guard) | "  + " | ".join(_pct_range(S[b]["raw"])  for b in backends) + " |")
    L.append("| Schema-valid | "               + " | ".join(_pct_range(S[b]["schema"]) for b in backends) + " |")
    L.append("| Tool-routing accuracy | "      + " | ".join(_pct_range(S[b]["routing"]) for b in backends) + " |")
    L.append("| Latency median (s) | "         + " | ".join(f"{S[b]['latency_median']:.1f}" for b in backends) + " |")
    L.append("| Latency max (s) | "            + " | ".join(f"{S[b]['latency_max']:.1f}" for b in backends) + " |")
    L.append("| Errors (total) | "             + " | ".join(str(S[b]["n_errors"]) for b in backends) + " |")
    L.append("\n## Per-category — FULL mean [min–max] · RAW mean [min–max]\n")
    L.append("| Category (n) | " + " | ".join(backends) + " |")
    L.append("|----------|" + "|".join(["---"] * len(backends)) + "|")
    for cat in corpus.categories():
        cells = []
        for b in backends:
            c = S[b]["by_category"].get(cat)
            cells.append(f"{_pct_range(c['full'])} · {_pct_range(c['raw'])}" if c else "—")
        n = S[backends[0]]["by_category"].get(cat, {}).get("n", 0)
        L.append(f"| {cat} ({n}) | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def write_artifacts(comparison: Dict[str, Any], model_label: str = "",
                    out_dir: Optional[Path] = None) -> Dict[str, Path]:
    out_dir = Path(out_dir or _ARTIFACT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path  = out_dir / "translator_benchmark_results.md"
    csv_path = out_dir / "translator_benchmark_results.csv"
    md_path.write_text(build_markdown(comparison, model_label), encoding="utf-8")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "run", "id", "category", "raw_pass", "full_pass",
                    "schema_valid", "routing_ok", "latency_s", "error",
                    "tools_needed", "routed_tool"])
        for b, data in comparison.items():
            for ri, rows in enumerate(data["runs"]):
                for r in rows:
                    tn = r.get("tools_needed") or []
                    tn = tn if isinstance(tn, list) else [tn]
                    tools_needed = ";".join(str(t) for t in tn)
                    routed_tool = str(tn[0]) if tn else ""   # primary routed tool
                    w.writerow([b, ri, r["id"], r["category"], r["raw_pass"], r["full_pass"],
                                r["schema_valid"], r["routing_ok"], r["latency_s"], r["error"] or "",
                                tools_needed, routed_tool])
    return {"md": md_path, "csv": csv_path}


def print_rich(comparison: Dict[str, Any], model_label: str = "") -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        print(build_markdown(comparison, model_label)); return
    con = Console()
    backends = list(comparison)
    S = {b: comparison[b]["summary"] for b in backends}
    n_runs = S[backends[0]]["n_runs"] if backends else 0
    t = Table(title=f"Translator benchmark — {model_label or config.OLLAMA_MODEL}, N={n_runs} (mean [min–max])")
    t.add_column("Metric")
    for b in backends:
        t.add_column(b, justify="right")
    t.add_row("Pass FULL (post-guard)", *[_pct_range(S[b]["full"]) for b in backends])
    t.add_row("Pass RAW (pre-guard)",   *[_pct_range(S[b]["raw"]) for b in backends])
    t.add_row("Schema-valid",           *[_pct_range(S[b]["schema"]) for b in backends])
    t.add_row("Tool-routing",           *[_pct_range(S[b]["routing"]) for b in backends])
    t.add_row("Latency median (s)",     *[f"{S[b]['latency_median']:.1f}" for b in backends])
    con.print(t)
    ct = Table(title="Per-category — FULL · RAW (mean [min–max])")
    ct.add_column("Category")
    for b in backends:
        ct.add_column(b)
    for cat in corpus.categories():
        cells = []
        for b in backends:
            c = S[b]["by_category"].get(cat)
            cells.append(f"{_pct_range(c['full'])} · {_pct_range(c['raw'])}" if c else "—")
        ct.add_row(cat, *cells)
    con.print(ct)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Translator backend benchmark (Claude vs Ollama).")
    ap.add_argument("--backends", default="claude,ollama",
                    help="comma list of registered backends to compare")
    ap.add_argument("--runs", type=int, default=5,
                    help="runs per case (mean over N>=5 is the comparison bar)")
    ap.add_argument("--model", default=None, help="override OLLAMA_MODEL for this run")
    args = ap.parse_args()
    if args.model:
        config.OLLAMA_MODEL = args.model
    comp = run_comparison(tuple(b.strip() for b in args.backends.split(",") if b.strip()),
                          runs=args.runs)
    print_rich(comp, model_label=config.OLLAMA_MODEL)
    paths = write_artifacts(comp, model_label=config.OLLAMA_MODEL)
    print(f"\nArtifacts: {paths['md']} · {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
