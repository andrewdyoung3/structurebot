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


def run_comparison(backends=("claude", "ollama"),
                   translator: Optional[CommandTranslator] = None) -> Dict[str, Any]:
    """Run each backend over the corpus; return {backend: {rows, summary}}."""
    out: Dict[str, Any] = {}
    for b in backends:
        rows = run_backend(b, translator=translator)
        out[b] = {"rows": rows, "summary": aggregate(rows)}
    return out


# ── Reporting ───────────────────────────────────────────────────────────────────
def build_markdown(comparison: Dict[str, Any], model_label: str = "") -> str:
    backends = list(comparison)
    L: List[str] = []
    L.append(f"# Translator backend benchmark — {_dt.date.today().isoformat()}")
    if model_label:
        L.append(f"\n_Ollama model: **{model_label}**_  ·  corpus: {len(corpus.CORPUS)} cases, "
                 f"{len(corpus.categories())} categories. FULL = post-guard "
                 f"(`translate()`); RAW = pre-guard (`backend.translate()`).")
    L.append("\n## Overall\n")
    L.append("| Metric | " + " | ".join(backends) + " |")
    L.append("|--------|" + "|".join(["---"] * len(backends)) + "|")
    def _row(label, fn):
        return "| " + label + " | " + " | ".join(fn(comparison[b]["summary"]) for b in backends) + " |"
    L.append(_row("Pass-rate (FULL, post-guard)",
                  lambda s: f"{s['full_rate']*100:.0f}% ({s['full_pass']}/{s['n']})"))
    L.append(_row("Pass-rate (RAW, pre-guard)",
                  lambda s: f"{s['raw_rate']*100:.0f}% ({s['raw_pass']}/{s['n']})"))
    L.append(_row("Schema-valid",
                  lambda s: f"{s['schema_rate']*100:.0f}%"))
    L.append(_row("Tool-routing accuracy",
                  lambda s: f"{s['routing_rate']*100:.0f}% ({s['routing_ok']}/{s['routing_n']})"))
    L.append(_row("Latency median (s)", lambda s: f"{s['latency_median']:.1f}"))
    L.append(_row("Latency max (s)",    lambda s: f"{s['latency_max']:.1f}"))
    L.append(_row("Errors",             lambda s: str(s["n_errors"])))

    L.append("\n## Per-category (FULL · RAW pass)\n")
    L.append("| Category | " + " | ".join(backends) + " |")
    L.append("|----------|" + "|".join(["---"] * len(backends)) + "|")
    for cat in corpus.categories():
        cells = []
        for b in backends:
            c = comparison[b]["summary"]["by_category"].get(cat, {"n": 0, "full": 0, "raw": 0})
            cells.append(f"{c['full']}/{c['n']} · {c['raw']}/{c['n']}")
        L.append(f"| {cat} | " + " | ".join(cells) + " |")
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
        w.writerow(["backend", "id", "category", "raw_pass", "full_pass",
                    "schema_valid", "routing_ok", "latency_s", "error"])
        for b, data in comparison.items():
            for r in data["rows"]:
                w.writerow([b, r["id"], r["category"], r["raw_pass"], r["full_pass"],
                            r["schema_valid"], r["routing_ok"], r["latency_s"], r["error"] or ""])
    return {"md": md_path, "csv": csv_path}


def print_rich(comparison: Dict[str, Any], model_label: str = "") -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except Exception:
        print(build_markdown(comparison, model_label)); return
    con = Console()
    backends = list(comparison)
    t = Table(title=f"Translator backend benchmark (Ollama: {model_label or config.OLLAMA_MODEL})")
    t.add_column("Metric")
    for b in backends:
        t.add_column(b, justify="right")
    S = {b: comparison[b]["summary"] for b in backends}
    t.add_row("Pass FULL (post-guard)", *[f"{S[b]['full_rate']*100:.0f}% ({S[b]['full_pass']}/{S[b]['n']})" for b in backends])
    t.add_row("Pass RAW (pre-guard)",   *[f"{S[b]['raw_rate']*100:.0f}% ({S[b]['raw_pass']}/{S[b]['n']})" for b in backends])
    t.add_row("Schema-valid",           *[f"{S[b]['schema_rate']*100:.0f}%" for b in backends])
    t.add_row("Tool-routing",           *[f"{S[b]['routing_rate']*100:.0f}% ({S[b]['routing_ok']}/{S[b]['routing_n']})" for b in backends])
    t.add_row("Latency median (s)",     *[f"{S[b]['latency_median']:.1f}" for b in backends])
    con.print(t)
    ct = Table(title="Per-category (FULL/total · RAW/total)")
    ct.add_column("Category")
    for b in backends:
        ct.add_column(b)
    for cat in corpus.categories():
        cells = []
        for b in backends:
            c = S[b]["by_category"].get(cat, {"n": 0, "full": 0, "raw": 0})
            cells.append(f"{c['full']}/{c['n']} · {c['raw']}/{c['n']}")
        ct.add_row(cat, *cells)
    con.print(ct)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Translator backend benchmark (Claude vs Ollama).")
    ap.add_argument("--backends", default="claude,ollama",
                    help="comma list of registered backends to compare")
    ap.add_argument("--model", default=None, help="override OLLAMA_MODEL for this run")
    args = ap.parse_args()
    if args.model:
        config.OLLAMA_MODEL = args.model
    comp = run_comparison(tuple(b.strip() for b in args.backends.split(",") if b.strip()))
    print_rich(comp, model_label=config.OLLAMA_MODEL)
    paths = write_artifacts(comp, model_label=config.OLLAMA_MODEL)
    print(f"\nArtifacts: {paths['md']} · {paths['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
