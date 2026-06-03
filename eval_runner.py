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
                sc = score_translation(case, translation, weights, probe)
                tn = [t for t in (translation.get("tools_needed") or []) if isinstance(t, str)] \
                    if isinstance(translation, dict) else []
                tr = _truncation(meta or {})
                run_rows.append(CaseRun(
                    backend=backend, run_idx=r, case_id=case.id, category=case.category,
                    tier=case.tier, challenge=case.challenge_type, score=sc,
                    tools_needed=tn, routed_tool=(tn[0] if tn else ""), **tr))
            out[backend].append(run_rows)
    return out


# ════════════════════════════════════════════════════════════════════════════════
#  Aggregation — discard the cold run, mean over the rest
# ════════════════════════════════════════════════════════════════════════════════
def _rate(scores, get) -> Optional[float]:
    applic = [s for s in scores if get(s).applicable]
    return (sum(1 for s in applic if get(s).passed) / len(applic)) if applic else None


def _dim_breakdown(pairs: List[Tuple[eh.EvalCase, eh.CaseScore]],
                   keyfn) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[eh.CaseScore]] = {}
    for case, sc in pairs:
        groups.setdefault(str(keyfn(case)), []).append(sc)
    out: Dict[str, Dict[str, Any]] = {}
    for k, scs in sorted(groups.items()):
        out[k] = {
            "n": len(scs),
            "accuracy":      _rate(scs, lambda s: s.accuracy),
            "functionality": _rate(scs, lambda s: s.functionality),
            "usability":     _rate(scs, lambda s: s.usability),
            "aggregate":     statistics.mean(s.aggregate for s in scs),
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
                    "accuracy", "functionality", "usability", "aggregate",
                    "fully_correct", "tools_needed", "routed_tool",
                    "prompt_eval_count", "done_reason", "near_ceiling", "length_truncated"])
        for backend, run_list in all_runs.items():
            for r, run in enumerate(run_list):
                for row in run:
                    s = row.score
                    def cell(d):
                        return "" if not d.applicable else ("1" if d.passed else "0")
                    w.writerow([
                        backend, r, row.case_id, row.category, row.tier, row.challenge,
                        cell(s.accuracy), cell(s.functionality), cell(s.usability),
                        f"{s.aggregate:.3f}", int(s.fully_correct),
                        ";".join(row.tools_needed), row.routed_tool,
                        row.prompt_eval_count if row.prompt_eval_count is not None else "",
                        row.done_reason or "", int(row.near_ceiling), int(row.length_truncated),
                    ])
    return path


# ════════════════════════════════════════════════════════════════════════════════
#  Production backend callers (forced + direct, no fallback)
# ════════════════════════════════════════════════════════════════════════════════
def make_claude_caller(translator) -> Caller:
    from translator import make_backend
    backend = make_backend("claude")

    def call(case):
        translator.reset_conversation()
        translator._backend = backend
        sess = EvalSession(case.session)
        try:
            tr = backend.translate(translator, case.prompt, sess)
        except Exception as exc:
            tr = {"commands": [], "explanations": [], "warnings": [str(exc)],
                  "clarification_needed": None, "confidence": "low",
                  "tools_needed": [], "tool_inputs": {}}
        return tr, {}
    return call


def make_ollama_caller(translator) -> Caller:
    from translator import make_backend, OllamaBackend
    backend = make_backend("ollama")

    def call(case):
        translator.reset_conversation()
        translator._backend = backend
        sess = EvalSession(case.session)
        try:
            tr = backend.translate(translator, case.prompt, sess)
        except Exception as exc:
            tr = {"commands": [], "explanations": [], "warnings": [str(exc)],
                  "clarification_needed": None, "confidence": "low",
                  "tools_needed": [], "tool_inputs": {}}
        return tr, OllamaBackend.last_meta()
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
