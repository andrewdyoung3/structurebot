"""
eval_runner.py — benchmark-grade runner for the model-INDEPENDENT 3-dimension eval
(eval_harness). Drives BOTH backends (Claude + Ollama) over the frozen corpus,
scores accuracy/functionality/usability byte-identically, and emits an auditable,
provenance-stamped report. NEVER runs heavy tools (dispatch = static parsed-inputs
assertion; effect = live ChimeraX commands only) and NEVER scores on unfrozen gold
(assert_no_pending_gold gates every run).

Backends are injected as CALLERS — `caller(case) -> (translation_dict, meta_dict)` —
so the runner is unit-testable with mocked backends. `make_claude_caller` /
`make_ollama_caller` wire the real backends (forced + called DIRECTLY, no fallback,
so a Claude rescue of local numbers is structurally impossible — honest numbers).

TRUNCATION INSTRUMENTATION (honesty guard — a silently truncated prompt scored as a
model failure was the original num_ctx bug): per Ollama case we log
prompt_eval_count + done_reason, estimate prompt+output vs num_ctx, and FLAG any case
that is near the ceiling or whose done_reason == "length".
"""
from __future__ import annotations

import csv
import hashlib
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
import eval_harness as eh

# Reuse the post-guard the production translate() applies, so FULL == guard(raw).
try:
    from translator_benchmark import _apply_guard
except Exception:                                   # pragma: no cover
    def _apply_guard(raw):                          # type: ignore
        return raw

Caller = Callable[[eh.EvalCase], Tuple[Dict[str, Any], Dict[str, Any]]]
_NEAR_CEILING = 0.90                                # warn at ≥90% of num_ctx


# ════════════════════════════════════════════════════════════════════════════════
#  Per-case loaded-state session (so ambiguity cases are well-defined)
# ════════════════════════════════════════════════════════════════════════════════
def session_summary(case_session: Optional[Dict[str, Any]]) -> str:
    models = (case_session or {}).get("models") or []
    if not models:
        return "No structure is currently open."
    parts = [f"{m.get('id', '#?')}: {m.get('pdb', '?')} "
             f"(chains {', '.join(m.get('chains') or []) or '?'})" for m in models]
    sel = (case_session or {}).get("selection")
    tail = "" if sel in (None, {}) else "  A residue selection is active."
    return "Open models — " + "; ".join(parts) + "." + tail


class EvalSession:
    """Minimal SessionState stand-in built from a case's `session` (the backend
    only needs get_context_summary())."""
    def __init__(self, case_session: Optional[Dict[str, Any]]):
        self._summary = session_summary(case_session)

    def get_context_summary(self) -> str:
        return self._summary


# ════════════════════════════════════════════════════════════════════════════════
#  Result rows
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class CaseRun:
    backend: str
    run_idx: int
    case_id: str
    category: str
    tier: int
    challenge: str
    score: eh.CaseScore
    tools_needed: List[str]
    routed_tool: str
    # truncation instrumentation (Ollama)
    prompt_eval_count: Optional[int] = None
    done_reason: Optional[str] = None
    est_prompt_tokens: Optional[int] = None
    num_ctx: Optional[int] = None
    num_predict: Optional[int] = None
    near_ceiling: bool = False
    length_truncated: bool = False
    # output-capture honesty: an error string if the call failed (after retries), and
    # whether the (parsed) translation was effectively empty (no tool/command/clarify/
    # refuse) — a swallowed failure that previously looked like a silent 0-score row.
    error: Optional[str] = None
    output_empty: bool = False


def _truncation(meta: Dict[str, Any]) -> Dict[str, Any]:
    pec = meta.get("prompt_eval_count")
    num_ctx = meta.get("num_ctx")
    num_predict = meta.get("num_predict") or 0
    done = meta.get("done_reason")
    near = bool(pec and num_ctx and (pec + num_predict) >= _NEAR_CEILING * num_ctx)
    return {
        "prompt_eval_count": pec,
        "done_reason": done,
        "est_prompt_tokens": pec,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "near_ceiling": near,
        "length_truncated": (done == "length"),
    }


def score_translation(case: eh.EvalCase, translation: Dict[str, Any],
                      weights: Dict[str, float], probe) -> eh.CaseScore:
    full = _apply_guard(translation) if translation else {}
    return eh.score_case(case, full, probe=probe, weights=weights)


# ════════════════════════════════════════════════════════════════════════════════
#  The run loop
# ════════════════════════════════════════════════════════════════════════════════
def run_corpus(callers: Dict[str, Caller], cases: List[eh.EvalCase],
               runs: int = 5, weights: Dict[str, float] = eh.WEIGHTS,
               probe=None) -> Dict[str, List[List[CaseRun]]]:
    """Run each backend over *cases* `runs` times. Returns {backend: [run0, run1, …]}
    where each run is a list of CaseRun. Honesty gate: refuses to score on unfrozen
    gold."""
    eh.assert_no_pending_gold(cases)                # never run on PENDING_FREEZE gold
    out: Dict[str, List[List[CaseRun]]] = {b: [] for b in callers}
    for backend, caller in callers.items():
        for r in range(max(1, runs)):
            run_rows: List[CaseRun] = []
            for case in cases:
                translation, meta = caller(case)
                meta = meta or {}
                sc = score_translation(case, translation, weights, probe)
                tn = [t for t in (translation.get("tools_needed") or []) if isinstance(t, str)] \
                    if isinstance(translation, dict) else []
                tr = _truncation(meta)
                run_rows.append(CaseRun(
                    backend=backend, run_idx=r, case_id=case.id, category=case.category,
                    tier=case.tier, challenge=case.challenge_type, score=sc,
                    tools_needed=tn, routed_tool=(tn[0] if tn else ""),
                    error=meta.get("error"), output_empty=_is_empty_output(translation), **tr))
            out[backend].append(run_rows)
    return out


def _is_empty_output(tr: Dict[str, Any]) -> bool:
    """A translation that does NOTHING — no tool, no command, no clarification, no
    refusal. The signature of a swallowed failure (vs a legitimate clarify/refuse)."""
    if not isinstance(tr, dict):
        return True
    cmds = [c for c in (tr.get("commands") or []) if isinstance(c, str) and c.strip()]
    return not (tr.get("tools_needed") or cmds or tr.get("clarification_needed") or tr.get("refused"))


def assert_capture_rate(all_runs: Dict[str, List[List[CaseRun]]],
                        threshold: float = 0.10) -> Dict[str, float]:
    """ABORT (raise) if any backend produced no usable output for more than
    *threshold* of its rows (errored OR empty). Same honesty principle as the
    truncation guard: a backend returning empty must FAIL the run, not yield a
    fictitious low-score 'result'. Returns the per-backend miss-rate."""
    rates: Dict[str, float] = {}
    problems: List[str] = []
    for backend, run_list in all_runs.items():
        rows = [r for run in run_list for r in run]
        n = len(rows) or 1
        missed = [r for r in rows if r.error or r.output_empty]
        rates[backend] = len(missed) / n
        if rates[backend] > threshold:
            kinds = sorted({(r.error or "empty-output").split(":")[0] for r in missed})
            sample = next((r.error for r in missed if r.error), "empty-output")
            problems.append(f"{backend}: {len(missed)}/{n} ({rates[backend]*100:.0f}%) rows "
                            f"empty/errored, over the {threshold*100:.0f}% threshold — kinds: "
                            f"{kinds}; e.g.: {sample[:240]}")
    if problems:
        raise RuntimeError(
            "HOLLOW RUN — a backend failed to produce output for too many cases; the "
            "scored result would be fiction. Inspect results.csv (error column).\n  "
            + "\n  ".join(problems))
    return rates


# ════════════════════════════════════════════════════════════════════════════════
#  Aggregation — discard the cold run, mean over the rest
# ════════════════════════════════════════════════════════════════════════════════
def _rate(scores, get) -> Optional[float]:
    applic = [s for s in scores if get(s).applicable]
    return (sum(1 for s in applic if get(s).passed) / len(applic)) if applic else None


def _jaccard_pairs(got, want) -> float:
    """Jaccard overlap of two (chain,resnum) selection sets given as ';'-style lists
    of 'A:25' tokens. Two empty sets are a perfect 1.0; otherwise |∩|/|∪|."""
    def to_set(seq):
        s = set()
        for x in (seq or []):
            x = str(x).strip()
            if ":" in x:
                c, _, n = x.partition(":")
                s.add((c, n))
            elif x:
                s.add(x)
        return s
    g, w = to_set(got), to_set(want)
    if not g and not w:
        return 1.0
    return len(g & w) / len(g | w) if (g | w) else 1.0


def _graded_functionality(s: eh.CaseScore) -> Optional[float]:
    """GRADED functionality (companion to strict): for a selection_resnums effect
    case, the Jaccard overlap of the actual vs gold (chain,resnum) sets; for every
    OTHER applicable case it equals the strict 0/1 (so graded == strict everywhere
    except selection-effect — isolating exactly what grading changes). None when
    functionality is not applicable."""
    f = s.functionality
    if not f.applicable:
        return None
    d = f.detail or {}
    if d.get("probe") == "selection_resnums":
        return _jaccard_pairs(d.get("got"), d.get("want"))
    return 1.0 if f.passed else 0.0


def _case_graded_aggregate(s: eh.CaseScore, weights=eh.WEIGHTS) -> float:
    """The companion weighted aggregate, using GRADED functionality (strict A and U)."""
    fg = _graded_functionality(s)
    num = den = 0.0
    if s.accuracy.applicable:
        num += weights["accuracy"] * (1.0 if s.accuracy.passed else 0.0); den += weights["accuracy"]
    if s.functionality.applicable:
        num += weights["functionality"] * (fg if fg is not None else 0.0); den += weights["functionality"]
    if s.usability.applicable:
        num += weights["usability"] * (1.0 if s.usability.passed else 0.0); den += weights["usability"]
    return num / den if den else 0.0


def _dim_breakdown(pairs: List[Tuple[eh.EvalCase, eh.CaseScore]],
                   keyfn) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[eh.CaseScore]] = {}
    for case, sc in pairs:
        groups.setdefault(str(keyfn(case)), []).append(sc)
    out: Dict[str, Dict[str, Any]] = {}
    for k, scs in sorted(groups.items()):
        fg = [_graded_functionality(s) for s in scs if s.functionality.applicable]
        out[k] = {
            "n": len(scs),
            "accuracy":      _rate(scs, lambda s: s.accuracy),
            "functionality": _rate(scs, lambda s: s.functionality),
            "functionality_graded": (statistics.mean(fg) if fg else None),   # companion
            "usability":     _rate(scs, lambda s: s.usability),
            "aggregate":     statistics.mean(s.aggregate for s in scs),       # STRICT (primary)
            "aggregate_graded": statistics.mean(_case_graded_aggregate(s) for s in scs),
            "fully_correct": sum(1 for s in scs if s.fully_correct) / len(scs),
        }
    return out


def aggregate(all_runs: Dict[str, List[List[CaseRun]]],
              cases: List[eh.EvalCase]) -> Dict[str, Any]:
    """Discard each backend's COLD run (run 0) when runs>1, then mean over the rest
    by treating every (case, scored-run) as a sample. Returns per-backend
    per-dimension × overall/category/tier/challenge + aggregate + fully-correct, plus
    a truncation summary."""
    by_id = {c.id: c for c in cases}
    report: Dict[str, Any] = {}
    for backend, run_list in all_runs.items():
        scored = run_list[1:] if len(run_list) > 1 else run_list   # DISCARD cold run
        n_runs_used = len(scored)
        pairs = [(by_id[row.case_id], row.score) for run in scored for row in run]
        # truncation roll-up (over the scored runs)
        trunc_rows = [row for run in scored for row in run]
        near = [r.case_id for r in trunc_rows if r.near_ceiling]
        length = [r.case_id for r in trunc_rows if r.length_truncated]
        pecs = [r.prompt_eval_count for r in trunc_rows if r.prompt_eval_count is not None]
        report[backend] = {
            "n_runs_total": len(run_list),
            "n_runs_scored": n_runs_used,
            "n_cases": len(cases),
            "overall": _dim_breakdown(pairs, lambda c: "ALL")["ALL"],
            "by_category": _dim_breakdown(pairs, lambda c: c.category),
            "by_tier": _dim_breakdown(pairs, lambda c: c.tier),
            "by_challenge": _dim_breakdown(pairs, lambda c: c.challenge_type),
            "truncation": {
                "near_ceiling_cases": sorted(set(near)),
                "length_truncated_cases": sorted(set(length)),
                "max_prompt_eval_count": (max(pecs) if pecs else None),
                "instrumented": bool(pecs),
            },
        }
    return report


# ════════════════════════════════════════════════════════════════════════════════
#  Provenance
# ════════════════════════════════════════════════════════════════════════════════
def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=str(Path(__file__).parent)).decode().strip()
    except Exception:
        return "nogit"


def _content_sha(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
    except Exception:
        return "missing"


def provenance(corpus_path: Path, runs: int, weights: Dict[str, float]) -> Dict[str, Any]:
    return {
        "corpus_sha": _content_sha(Path(corpus_path)),
        "harness_sha": _git_head()[:12],
        "claude_model": getattr(config, "CLAUDE_MODEL", getattr(config, "ANTHROPIC_MODEL", "claude")),
        "ollama_model": config.OLLAMA_MODEL,
        "ollama_num_ctx": int(config.OLLAMA_NUM_CTX),
        "ollama_num_predict": int(config.OLLAMA_NUM_PREDICT),
        "seed": 0,
        "runs": runs,
        "weights": dict(weights),
    }


def header_text(prov: Dict[str, Any]) -> str:
    return (
        "# Translator 3-dimension benchmark (model-independent)\n\n"
        f"_corpus_sha **{prov['corpus_sha']}** · harness_sha **{prov['harness_sha']}** · "
        f"Claude **{prov['claude_model']}** vs Ollama **{prov['ollama_model']}** · "
        f"num_ctx {prov['ollama_num_ctx']} · num_predict {prov['ollama_num_predict']} · "
        f"seed {prov['seed']} · N={prov['runs']} (cold run discarded) · "
        f"weights A{prov['weights']['accuracy']}/F{prov['weights']['functionality']}/"
        f"U{prov['weights']['usability']}_\n"
    )


# ════════════════════════════════════════════════════════════════════════════════
#  CSV (per backend × run × case) + tools_needed + routed_tool + truncation
# ════════════════════════════════════════════════════════════════════════════════
def write_csv(all_runs: Dict[str, List[List[CaseRun]]], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["backend", "run", "id", "category", "tier", "challenge",
                    "accuracy", "functionality", "functionality_graded", "usability", "aggregate",
                    "fully_correct", "tools_needed", "routed_tool",
                    "prompt_eval_count", "done_reason", "near_ceiling", "length_truncated",
                    # output-capture honesty: the captured call error (if any) and
                    # whether the parsed output was empty (a swallowed failure).
                    "error", "output_empty",
                    # the ACTUAL effect sets, persisted so graded overlap can be
                    # computed post-hoc WITHOUT re-running: chain-qualified residue
                    # lists for selection_resnums ("A:25;B:25;…"), or got/want RGB for
                    # residue_color. Empty for dispatch / usability-only cases.
                    "effect_got", "effect_want"])
        for backend, run_list in all_runs.items():
            for r, run in enumerate(run_list):
                for row in run:
                    s = row.score
                    def cell(d):
                        return "" if not d.applicable else ("1" if d.passed else "0")
                    got, want = _effect_sets(s.functionality)
                    fg = _graded_functionality(s)
                    w.writerow([
                        backend, r, row.case_id, row.category, row.tier, row.challenge,
                        cell(s.accuracy), cell(s.functionality),
                        ("" if fg is None else f"{fg:.3f}"), cell(s.usability),
                        f"{s.aggregate:.3f}", int(s.fully_correct),
                        ";".join(row.tools_needed), row.routed_tool,
                        row.prompt_eval_count if row.prompt_eval_count is not None else "",
                        row.done_reason or "", int(row.near_ceiling), int(row.length_truncated),
                        row.error or "", int(row.output_empty),
                        got, want,
                    ])
    return path


def _effect_sets(fun: eh.DimResult) -> Tuple[str, str]:
    """Extract the model's resulting set (got) and the gold (want) from an effect
    DimResult, as `;`-joined strings — the FULL chain-qualified lists for
    selection_resnums (e.g. 'A:25;B:25'), or the RGB triples for residue_color."""
    d = fun.detail or {}
    got = d.get("got", d.get("got_rgb"))
    want = d.get("want", d.get("want_rgb"))
    def _j(v):
        if v is None:
            return ""
        if isinstance(v, (list, tuple)):
            return ";".join(str(x) for x in v)
        return str(v)
    return _j(got), _j(want)


# ════════════════════════════════════════════════════════════════════════════════
#  Production backend callers (forced + direct, no fallback)
# ════════════════════════════════════════════════════════════════════════════════
# Transient API conditions worth a retry+backoff (rate-limit / overload / network).
# Over a 900-call run these WILL occur; without a retry+capture they get swallowed
# into empty rows and a hollow "0.08" result (the bug this fixes).
_TRANSIENT_MARKERS = ("rate", "ratelimit", "overload", "timeout", "timed out",
                      "connection", "temporarily", "529", "503", "502", "429", "500")

def _empty_translation(reason: str) -> Dict[str, Any]:
    return {"commands": [], "explanations": [], "warnings": [reason],
            "clarification_needed": None, "confidence": "low",
            "tools_needed": [], "tool_inputs": {}, "refused": False}

def _translate_capturing(backend, translator, case, retries: int = 4,
                         base_delay: float = 2.0):
    """Run backend.translate EXACTLY as the smoke does (reset → forced backend →
    translate), but retry transient API failures with exponential backoff and, on
    final failure, CAPTURE the error (return it) instead of silently emptying."""
    import time
    last = None
    for attempt in range(max(1, retries)):
        try:
            translator.reset_conversation()
            translator._backend = backend
            sess = EvalSession(case.session)
            return backend.translate(translator, case.prompt, sess), None
        except Exception as exc:                       # noqa: BLE001 — captured, not hidden
            last = exc
            msg = f"{type(exc).__name__}: {exc}".lower()
            if any(m in msg for m in _TRANSIENT_MARKERS) and attempt < retries - 1:
                time.sleep(min(base_delay * (2 ** attempt), 30.0))
                continue
            break
    err = f"{type(last).__name__}: {last}" if last is not None else "empty"
    return _empty_translation(err), err


def make_claude_caller(translator) -> Caller:
    from translator import make_backend
    backend = make_backend("claude")

    def call(case):
        tr, err = _translate_capturing(backend, translator, case)
        return tr, ({"error": err} if err else {})
    return call


def make_ollama_caller(translator) -> Caller:
    from translator import make_backend, OllamaBackend
    backend = make_backend("ollama")

    def call(case):
        tr, err = _translate_capturing(backend, translator, case)
        meta = OllamaBackend.last_meta()
        if err:
            meta = dict(meta); meta["error"] = err
        return tr, meta
    return call


def make_chimerax_probe(base_url: str = "http://localhost:60001"):
    """A `command -> str` probe over the ChimeraX REST API (same interface the effect
    scorer drives; same one freeze_zone_gold.py uses)."""
    import urllib.parse
    import urllib.request

    def probe(command: str) -> str:
        url = base_url.rstrip("/") + "/run?command=" + urllib.parse.quote(command)
        with urllib.request.urlopen(url, timeout=20) as r:
            return r.read().decode("utf-8", "replace")
    return probe


# ════════════════════════════════════════════════════════════════════════════════
#  Report markdown + artifacts + CLI
# ════════════════════════════════════════════════════════════════════════════════
def _pct(x) -> str:
    return "n/a" if x is None or x != x else f"{x*100:.0f}%"   # NaN-safe


def build_report_md(all_runs: Dict[str, List[List["CaseRun"]]],
                    cases: List[eh.EvalCase], prov: Dict[str, Any]) -> str:
    rep = aggregate(all_runs, cases)
    backends = list(all_runs)
    L = [header_text(prov).rstrip(), ""]
    # overall (per-dimension + aggregate + fully-correct), both backends
    L.append("## Overall (mean over scored runs; cold run discarded)\n")
    L.append("| Metric | " + " | ".join(backends) + " |")
    L.append("|--------|" + "|".join(["---"] * len(backends)) + "|")
    for label, key in [("Accuracy", "accuracy"), ("Functionality (strict)", "functionality")]:
        L.append(f"| {label} | " + " | ".join(_pct(rep[b]["overall"][key]) for b in backends) + " |")
    L.append("| _Functionality (graded, companion)_ | " +
             " | ".join(f"_{_pct(rep[b]['overall']['functionality_graded'])}_" for b in backends) + " |")
    L.append("| Usability | " + " | ".join(_pct(rep[b]["overall"]["usability"]) for b in backends) + " |")
    L.append("| **Aggregate (strict, PRIMARY)** (A.50/F.35/U.15) | " +
             " | ".join(f"**{rep[b]['overall']['aggregate']*100:.0f}%**" for b in backends) + " |")
    L.append("| _Graded aggregate (companion)_ | " +
             " | ".join(f"_{rep[b]['overall']['aggregate_graded']*100:.0f}%_" for b in backends) + " |")
    L.append("| **Fully-correct (strict)** | " +
             " | ".join(f"**{_pct(rep[b]['overall']['fully_correct'])}**" for b in backends) + " |")
    L.append("| scored runs (of total) | " +
             " | ".join(f"{rep[b]['n_runs_scored']}/{rep[b]['n_runs_total']}" for b in backends) + " |")
    L.append("\n_Strict exact-set functionality is the PRIMARY scored value (feeds the "
             "aggregate + fully-correct, unchanged). **Graded** is an additive companion: "
             "selection_resnums effect cases use the Jaccard overlap of the actual vs gold "
             "(chain,resnum) sets; every other case's graded == its strict value._\n")

    for axis, title in [("by_category", "Per-category"), ("by_tier", "Per-tier"),
                        ("by_challenge", "Per-challenge")]:
        L.append(f"\n## {title} — A · F(strict)[F graded] · U · agg(strict)[agg graded] · fully\n")
        L.append("| Group (n) | " + " | ".join(backends) + " |")
        L.append("|-----------|" + "|".join(["---"] * len(backends)) + "|")
        keys = sorted(set().union(*[set(rep[b][axis]) for b in backends]))
        for k in keys:
            cells = []
            n = 0
            for b in backends:
                g = rep[b][axis].get(k)
                if not g:
                    cells.append("—"); continue
                n = g["n"]
                cells.append(f"{_pct(g['accuracy'])} · {_pct(g['functionality'])}"
                             f"[{_pct(g['functionality_graded'])}] · {_pct(g['usability'])} · "
                             f"{g['aggregate']*100:.0f}%[{g['aggregate_graded']*100:.0f}%] · "
                             f"{_pct(g['fully_correct'])}")
            L.append(f"| {k} ({n}) | " + " | ".join(cells) + " |")

    L.append("\n## Truncation (Ollama honesty guard)\n")
    for b in backends:
        t = rep[b]["truncation"]
        L.append(f"- **{b}**: instrumented={t['instrumented']} · "
                 f"max_prompt_eval_count={t['max_prompt_eval_count']} · "
                 f"near_ceiling={t['near_ceiling_cases']} · length_truncated={t['length_truncated_cases']}")
    L.append("\n_Per-case rows (incl. the full chain-qualified effect_got/effect_want "
             "sets) are in `results.csv`._\n")
    return "\n".join(L) + "\n"


def _runs_from_csv(csv_path, by_id: Dict[str, eh.EvalCase]) -> Dict[str, List[List["CaseRun"]]]:
    """Reconstruct an `all_runs` structure from a saved results.csv so the report
    (strict + graded) can be RESTATED post-hoc WITHOUT re-running. Graded is
    recomputed from the persisted effect_got/effect_want + each case's probe type."""
    import csv as _csv
    from collections import defaultdict
    rows = list(_csv.DictReader(Path(csv_path).read_text(encoding="utf-8").splitlines()))
    grouped: Dict[str, Dict[int, List[CaseRun]]] = defaultdict(lambda: defaultdict(list))

    def _dim(cell: str, detail=None) -> eh.DimResult:
        return eh.DimResult(applicable=(cell != ""), passed=(cell == "1"),
                            partial=(1.0 if cell == "1" else 0.0), detail=(detail or {}))

    for r in rows:
        case = by_id.get(r["id"])
        probe = (case.gold_functionality.assertion or {}).get("probe") \
            if (case and case.gold_functionality) else None
        got = [x for x in (r.get("effect_got") or "").split(";") if x]
        want = [x for x in (r.get("effect_want") or "").split(";") if x]
        sc = eh.CaseScore(
            id=r["id"], accuracy=_dim(r["accuracy"]),
            functionality=_dim(r["functionality"], {"probe": probe, "got": got, "want": want}),
            usability=_dim(r["usability"]),
            aggregate=(float(r["aggregate"]) if r.get("aggregate") else 0.0),
            fully_correct=(r.get("fully_correct") == "1"))
        cr = CaseRun(
            backend=r["backend"], run_idx=int(r["run"]), case_id=r["id"], category=r["category"],
            tier=int(r["tier"]), challenge=r["challenge"], score=sc,
            tools_needed=[t for t in (r.get("tools_needed") or "").split(";") if t],
            routed_tool=(r.get("routed_tool") or ""),
            prompt_eval_count=(int(r["prompt_eval_count"]) if r.get("prompt_eval_count") else None),
            done_reason=(r.get("done_reason") or None),
            near_ceiling=(r.get("near_ceiling") == "1"),
            length_truncated=(r.get("length_truncated") == "1"),
            error=(r.get("error") or None),
            output_empty=(r.get("output_empty") == "1"))
        grouped[r["backend"]][int(r["run"])].append(cr)
    return {b: [runs[i] for i in sorted(runs)] for b, runs in grouped.items()}


def report_from_csv(csv_path, manifest_path: str = "scripts/eval_corpus_manifest.json",
                    out_dir=None) -> Dict[str, Any]:
    """Restate a saved benchmark in BOTH rulers (strict + graded) WITHOUT re-running.
    Writes `<csv_dir>/report_restated.md`; returns the per-backend capture rate so a
    hollow backend can be flagged (not silently presented as a baseline)."""
    cases = eh.load_manifest(manifest_path)
    by_id = {c.id: c for c in cases}
    all_runs = _runs_from_csv(csv_path, by_id)
    out = Path(out_dir or Path(csv_path).parent)
    out.mkdir(parents=True, exist_ok=True)
    runs_total = max((len(rl) for rl in all_runs.values()), default=0)
    prov = provenance(manifest_path, runs=runs_total, weights=eh.WEIGHTS)
    md = out / "report_restated.md"
    md.write_text(build_report_md(all_runs, cases, prov), encoding="utf-8")
    rates = {}
    for b, rl in all_runs.items():
        rows = [r for run in rl for r in run]
        rates[b] = (sum(1 for r in rows if r.error or r.output_empty) / len(rows)) if rows else 0.0
    return {"report": md, "rates": rates, "all_runs": all_runs, "cases": cases}


def write_artifacts(all_runs: Dict[str, List[List["CaseRun"]]], cases: List[eh.EvalCase],
                    out_dir, prov: Dict[str, Any]) -> Dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "results.csv"
    md_path = out / "report.md"
    write_csv(all_runs, csv_path)
    md_path.write_text(build_report_md(all_runs, cases, prov), encoding="utf-8")
    return {"report": md_path, "csv": csv_path}


def main(argv=None) -> Dict[str, Path]:
    import argparse
    import os
    ap = argparse.ArgumentParser(description="Run the model-independent 3-dimension translator benchmark.")
    ap.add_argument("--manifest", default="scripts/eval_corpus_manifest.json",
                    help="frozen corpus JSON")
    ap.add_argument("--backends", default="ollama",
                    help="comma list; only 'ollama' runs (the 'claude' arm is retired — "
                         "translation is local-only; requesting it raises)")
    ap.add_argument("--runs", type=int, default=6,
                    help="TOTAL runs per backend; the cold run is discarded, so scored = runs-1 "
                         "(use 6 for 5 scored runs)")
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded in provenance; the Ollama backend itself is fixed at seed 0")
    ap.add_argument("--out", default="scripts/eval_3dim_results",
                    help="output DIRECTORY for report.md + results.csv")
    ap.add_argument("--chimerax-url", default="http://localhost:60001")
    ap.add_argument("--from-csv", default=None,
                    help="RESTATE a saved results.csv in strict+graded (no run, no live deps)")
    args = ap.parse_args(argv)

    if args.from_csv:                                   # post-hoc restatement, no run
        res = report_from_csv(args.from_csv, args.manifest, args.out)
        print(f"restated {args.from_csv} -> {res['report']} - capture rates "
              f"{ {b: f'{(1-r)*100:.0f}%' for b, r in res['rates'].items()} }")
        return {"report": res["report"]}

    import config
    config.load_env_file()
    from translator import CommandTranslator

    cases = eh.load_manifest(args.manifest)
    eh.assert_no_pending_gold(cases)                 # never run on unfrozen gold
    t = CommandTranslator(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    factory = {"claude": make_claude_caller, "ollama": make_ollama_caller}
    callers = {n: factory[n](t) for n in (x.strip() for x in args.backends.split(",")) if n in factory}
    probe = make_chimerax_probe(args.chimerax_url)

    all_runs = run_corpus(callers, cases, runs=args.runs, probe=probe)
    prov = provenance(args.manifest, runs=args.runs, weights=eh.WEIGHTS)

    # Always write the CSV first (errors are inspectable), THEN gate: a hollow run
    # (a backend that produced empty/errored output for >10% of cases) ABORTS before
    # a clean report.md is written — it must fail loudly, not present a 0.08 result.
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    csv_path = write_csv(all_runs, out / "results.csv")
    print(f"wrote {csv_path}")
    rates = assert_capture_rate(all_runs, threshold=0.10)     # raises on a hollow run
    md_path = out / "report.md"
    md_path.write_text(build_report_md(all_runs, cases, prov), encoding="utf-8")
    print(f"scored {prov['runs']-1} of {prov['runs']} runs · capture rates "
          f"{ {b: f'{(1-r)*100:.0f}%' for b, r in rates.items()} } · wrote {md_path}")
    return {"report": md_path, "csv": csv_path}


if __name__ == "__main__":
    main()
