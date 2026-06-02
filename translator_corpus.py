"""
translator_corpus.py
--------------------
Shared gold-standard corpus + scorer for the translator eval/benchmark harness
(``translator_benchmark.py``) and its tests. Single source of truth — the corpus
is NOT duplicated in the tests.

Each `CorpusCase` is (prompt → declarative `Check`s) tagged with a rule
`category`. A case PASSES when ALL its checks pass against a normalized
translation dict (the A-interface 7-key object). Checks are robust predicates
(substring/regex/tool-set), not brittle exact-dict matches, so multiple valid
phrasings of a correct translation pass. Claude is the reference: the checks
encode behaviour Claude exemplifies; the harness reports how close a local model
gets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

REQUIRED_KEYS = {
    "commands", "explanations", "warnings", "clarification_needed",
    "confidence", "tools_needed", "tool_inputs",
}


@dataclass
class Check:
    """A single predicate on a normalized translation dict."""
    kind: str
    arg: Any = None

    def evaluate(self, result: Dict[str, Any]) -> bool:
        cmds  = [c for c in (result.get("commands") or []) if isinstance(c, str)]
        tools = {str(t).lower() for t in (result.get("tools_needed") or [])}
        k = self.kind
        if k == "tools_any":
            return any(str(a).lower() in tools for a in self.arg)
        if k == "tools_all":
            return all(str(a).lower() in tools for a in self.arg)
        if k == "cmd_re":
            return any(re.search(self.arg, c, re.I) for c in cmds)
        if k == "no_cmd_re":
            return not any(re.search(self.arg, c, re.I) for c in cmds)
        if k == "clar_none":
            return result.get("clarification_needed") in (None, "", "null")
        if k == "cmds_nonempty":
            return len(cmds) > 0
        raise ValueError(f"unknown check kind {k!r}")


@dataclass
class CorpusCase:
    id: str
    category: str
    prompt: str
    checks: List[Check] = field(default_factory=list)

    @property
    def has_tool_expectation(self) -> bool:
        return any(c.kind in ("tools_any", "tools_all") for c in self.checks)


# ── Check constructors (concise corpus authoring) ───────────────────────────────
def _tools_any(*names) -> Check:  return Check("tools_any", list(names))
def _tools_all(*names) -> Check:  return Check("tools_all", list(names))
def _cmd(rgx: str) -> Check:      return Check("cmd_re", rgx)
def _no_cmd(rgx: str) -> Check:   return Check("no_cmd_re", rgx)
def _clar_none() -> Check:        return Check("clar_none")
def _cmds() -> Check:             return Check("cmds_nonempty")


# ── The corpus ──────────────────────────────────────────────────────────────────
# Categories mirror the translator rule families. Checks are calibrated so the
# reference backend (Claude) passes; the benchmark reports how a local model fares.
CORPUS: List[CorpusCase] = [
    # ── zone selection (rule 16): operators :< / @<, never Chimera-1 `zone` ──
    CorpusCase("zone_interface", "zone",
               "Select the chain A residues within 4.5 Å of chain B and list them.",
               [_cmd(r":<"), _no_cmd(r"\bzone\b"), _cmd(r"info\s+residues\s+sel"),
                _tools_any("chimerax")]),
    CorpusCase("zone_ligand", "zone",
               "Select the atoms within 4 Å of the MK1 ligand.",
               [_cmd(r"[:@]<"), _no_cmd(r"\bzone\b")]),

    # ── hide/show representation (rule 17): target the cartoon, not bare atoms ──
    CorpusCase("hide_chain", "hide_show",
               "Hide chain B.",
               [_cmd(r"hide\b[^\n]*\bB\b[^\n]*(cartoon|target)"), _no_cmd(r"\bzone\b")]),
    CorpusCase("show_cartoon", "hide_show",
               "Show chain A as a cartoon.",
               [_cmd(r"cartoon")]),

    # ── MPNN schema/routing ─────────────────────────────────────────────────────
    CorpusCase("mpnn_interface", "mpnn",
               "Redesign the interface between chain A and chain B.",
               [_tools_any("proteinmpnn")]),
    CorpusCase("mpnn_soluble", "mpnn",
               "Redesign chain A to be more soluble with no cysteines.",
               [_tools_any("proteinmpnn")]),

    # ── selection scope ─────────────────────────────────────────────────────────
    CorpusCase("sel_redesign", "selection_scope",
               "Redesign only the residues I have selected on chain A.",
               [_tools_any("proteinmpnn")]),
    CorpusCase("sel_interface_only", "selection_scope",
               "Redesign just the dimer-interface residues on chain A, keep the backbone fixed.",
               [_tools_any("proteinmpnn")]),

    # ── multi-tool routing ──────────────────────────────────────────────────────
    CorpusCase("multi_camsol_esm", "multi_tool",
               "Run a CamSol solubility scan on chain A and also score it for ESM conservation.",
               [_tools_all("camsol", "esm")]),
    CorpusCase("multi_assembly_disulfide", "multi_tool",
               "Map the interface between chains A and B and predict disulfide candidates there.",
               [_tools_any("disulfide")]),

    # ── CamSol ──────────────────────────────────────────────────────────────────
    CorpusCase("camsol_scan", "camsol",
               "Run a CamSol solubility scan on chain A.",
               [_tools_any("camsol")]),
    CorpusCase("camsol_color", "camsol",
               "Colour chain A by aggregation propensity.",
               [_tools_any("camsol")]),

    # ── ESM ─────────────────────────────────────────────────────────────────────
    CorpusCase("esm_conservation", "esm",
               "Score chain A for evolutionary conservation with ESM.",
               [_tools_any("esm")]),

    # ── disulfide ───────────────────────────────────────────────────────────────
    CorpusCase("disulfide_pairs", "disulfide",
               "Suggest interchain disulfide bond candidates between chains A and B.",
               [_tools_any("disulfide")]),

    # ── proline (routes via the engineering/mutation-scan path) ─────────────────
    CorpusCase("proline_stabilise", "proline",
               "Suggest proline mutations to stabilise chain A.",
               [_tools_any("mutation_scan", "proline", "rosetta")]),

    # ── plain ChimeraX visualisation ────────────────────────────────────────────
    CorpusCase("viz_open_color", "viz",
               "Open 1HSG and colour it by chain.",
               [_cmd(r"\bopen\b"), _cmd(r"bychain"), _tools_any("chimerax")]),

    # ── rfdiffusion (pre-screen short-circuit — identical for both backends) ────
    CorpusCase("rfd_binder", "rfdiffusion",
               "Design a binder for chain A.",
               [_tools_any("rfdiffusion")]),
]


# ── Scoring ─────────────────────────────────────────────────────────────────────
def score_case(case: CorpusCase, result: Dict[str, Any]) -> Tuple[bool, List[Tuple[Check, bool]]]:
    """Return (passed_all, [(check, passed), …]) for *case* against *result*."""
    per = [(c, bool(c.evaluate(result))) for c in case.checks]
    return all(p for _, p in per), per


def is_schema_valid(result: Any) -> bool:
    """True iff *result* is a well-formed normalized 7-key translation dict."""
    if not isinstance(result, dict) or not REQUIRED_KEYS.issubset(result):
        return False
    if not all(isinstance(result.get(k), list) for k in ("commands", "explanations", "warnings")):
        return False
    if not isinstance(result.get("tools_needed"), list) or not result["tools_needed"]:
        return False
    if not isinstance(result.get("tool_inputs"), dict):
        return False
    if result.get("confidence") not in ("high", "medium", "low"):
        return False
    cn = result.get("clarification_needed")
    return cn is None or isinstance(cn, str)


def tool_routing_ok(case: CorpusCase, result: Dict[str, Any]) -> bool:
    """Did *result* route to the expected tool(s)? (only meaningful when the case
    has a tool expectation — check `case.has_tool_expectation` first)."""
    tool_checks = [c for c in case.checks if c.kind in ("tools_any", "tools_all")]
    return all(c.evaluate(result) for c in tool_checks)


def categories() -> List[str]:
    return sorted({c.category for c in CORPUS})
