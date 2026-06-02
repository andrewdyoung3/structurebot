"""
translator_corpus.py
--------------------
Shared gold-standard corpus + scorer for the translator eval/benchmark harness
(``translator_benchmark.py``) and its tests. Single source of truth.

Two DISJOINT pools (leakage discipline — overlap would make a 90% bar fiction):
  • EVAL_CORPUS    — the held-out evaluation set scored by the benchmark.
  • EXAMPLE_POOL   — a separate pool for few-shot demos / future fine-tuning.
A test (`tests/test_translator_benchmark_logic.py`) enforces ZERO prompt/id
overlap between them.

Each `CorpusCase` is (prompt → declarative `Check`s) tagged with a rule
`category`. A case PASSES when ALL its checks pass against a normalized
translation dict (the A-interface 7-key object). Checks are robust predicates
(substring/regex/exact tool-set), not brittle exact-dict matches.

VERIFIED GOLD LABELS — how each expected label was checked (NOT "whatever a model
emitted"):
  • tool-routing categories (camsol/esm/proteinmpnn/disulfide/proline/multi_tool/
    selection_scope/mpnn/rfdiffusion): the gold tool name is the EXACT literal the
    real router dispatches on (`tool_router._dispatch_tool`: `if tool == "camsol"`
    …) and that the translator prompt enumerates. Scoring is **case-SENSITIVE**
    to match the router exactly (`"CamSol"` does NOT route → fails). Cross-checked
    by the Claude reference scoring ~100% on EVAL_CORPUS — a check Claude fails is
    a mis-specified check, fixed there, not a model excuse.
  • command-syntax categories (zone/hide_show/viz): the gold patterns match
    commands hand-verified to run clean in ChimeraX 1.11.1 over REST — zone
    operators `:<`/`@<`/`:>` (never Chimera-1 `zone`), representation-targeted
    hide/show (`cartoon`/`surface`/`target ac`, never bare `hide #1/B`),
    `bgColor`/`byelement`/`byattribute bfactor`/`save …png`, and the forbidden
    `preset publication` excluded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

REQUIRED_KEYS = {
    "commands", "explanations", "warnings", "clarification_needed",
    "confidence", "tools_needed", "tool_inputs",
}

MIN_PER_CATEGORY = 8   # balance floor for the eval set


@dataclass
class Check:
    """A single predicate on a normalized translation dict."""
    kind: str
    arg: Any = None

    def evaluate(self, result: Dict[str, Any]) -> bool:
        cmds  = [c for c in (result.get("commands") or []) if isinstance(c, str)]
        # EXACT tool names — the real router matches `tool == "camsol"` (no
        # lowercasing), so "CamSol"/"ChimeraX" must NOT count as a match.
        tools = {t for t in (result.get("tools_needed") or []) if isinstance(t, str)}
        k = self.kind
        if k == "tools_any":
            return any(a in tools for a in self.arg)
        if k == "tools_all":
            return all(a in tools for a in self.arg)
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


# ── Check constructors ──────────────────────────────────────────────────────────
def _any(*names) -> Check:  return Check("tools_any", list(names))
def _all(*names) -> Check:  return Check("tools_all", list(names))
def _cmd(rgx: str) -> Check:    return Check("cmd_re", rgx)
def _nocmd(rgx: str) -> Check:  return Check("no_cmd_re", rgx)
def _C(cid, cat, prompt, *checks) -> CorpusCase:
    return CorpusCase(cid, cat, prompt, list(checks))

_ZONE = r"\bzone\b"          # forbidden Chimera-1 keyword
_PUBPRESET = r"preset\s+publication"


# ── EVAL CORPUS (held-out; scored) ──────────────────────────────────────────────
EVAL_CORPUS: List[CorpusCase] = [
    # ─────────────── zone (operators :< / @< / :>, never `zone`) ───────────────
    _C("zone_iface_list", "zone",
       "Select the chain A residues within 4.5 Å of chain B and list them.",
       _cmd(r":<"), _nocmd(_ZONE), _cmd(r"info\s+residues\s+sel"), _any("chimerax")),
    _C("zone_ligand_atoms", "zone",
       "Select the atoms within 4 Å of the MK1 ligand.",
       _cmd(r"[:@]<"), _nocmd(_ZONE)),
    _C("zone_b_near_ligand", "zone",
       "Select chain B residues within 5 Å of the MK1 ligand.",
       _cmd(r":<"), _nocmd(_ZONE)),
    _C("zone_which_iface", "zone",
       "Which chain A residues are within 6 Å of chain B?",
       _cmd(r":<"), _nocmd(_ZONE)),
    _C("zone_near_sel", "zone",
       "Select everything within 3.5 Å of the current selection.",
       _cmd(r"[:@]<"), _nocmd(_ZONE)),
    _C("zone_highlight_atoms", "zone",
       "Highlight the chain A atoms within 4 Å of chain B.",
       _cmd(r"[:@]<"), _nocmd(_ZONE)),
    _C("zone_beyond", "zone",
       "Select chain A residues that are more than 8 Å away from chain B.",
       # "beyond" = the `:>` operator OR the equivalent negated within-zone
       # `& ~(… :<N)` — both verified to run clean in ChimeraX 1.11.
       _cmd(r":>|~[^\n]*:<"), _nocmd(_ZONE)),
    _C("zone_report", "zone",
       "Select chain A residues within 4.5 Å of the ligand and report which ones.",
       _cmd(r":<"), _cmd(r"info\s+residues\s+sel"), _nocmd(_ZONE)),
    _C("zone_protein_near_lig", "zone",
       "Select the protein atoms within 5 Å of MK1.",
       _cmd(r"[:@]<"), _nocmd(_ZONE)),

    # ─────────────── hide_show (target the representation) ─────────────────────
    _C("hide_chainB", "hide_show", "Hide chain B.",
       _cmd(r"hide\b[^\n]*\bB\b[^\n]*(cartoon|target)"), _nocmd(_ZONE)),
    _C("show_chainA_cartoon", "hide_show", "Show chain A as a cartoon.",
       _cmd(r"cartoon")),
    _C("hide_chainA_cartoon", "hide_show", "Hide the cartoon for chain A.",
       _cmd(r"hide\b[^\n]*\bA\b[^\n]*(cartoon|target)")),
    _C("show_chainB_surface", "hide_show", "Show chain B as a molecular surface.",
       _cmd(r"surface")),
    _C("show_ligand_stick", "hide_show", "Show the MK1 ligand as sticks.",
       _cmd(r"(style|show)[^\n]*MK1[^\n]*stick|MK1[^\n]*stick")),
    _C("display_B_surface", "hide_show", "Display chain B with a surface representation.",
       _cmd(r"surface")),
    _C("hide_A_ribbon", "hide_show", "Hide chain A's ribbon.",
       _cmd(r"hide\b[^\n]*\bA\b[^\n]*(cartoon|target)")),
    _C("show_A_ribbons", "hide_show", "Show chain A as ribbons.",
       _cmd(r"cartoon")),
    _C("hide_B_cartoon_explicit", "hide_show", "Hide the chain B cartoon.",
       _cmd(r"hide\b[^\n]*\bB\b[^\n]*(cartoon|target)")),

    # ─────────────── mpnn (fixed-backbone redesign → proteinmpnn) ──────────────
    _C("mpnn_iface", "mpnn", "Redesign the interface between chain A and chain B.",
       _any("proteinmpnn")),
    _C("mpnn_soluble", "mpnn", "Redesign chain A to be more soluble with no cysteines.",
       _any("proteinmpnn")),
    _C("mpnn_use", "mpnn", "Use ProteinMPNN to redesign chain A.", _any("proteinmpnn")),
    _C("mpnn_surface_hydrophilic", "mpnn",
       "Redesign the surface residues of chain A to be more hydrophilic.", _any("proteinmpnn")),
    _C("mpnn_fixedbb", "mpnn",
       "Generate new sequences for chain A keeping the backbone fixed.", _any("proteinmpnn")),
    _C("mpnn_deaggregate", "mpnn",
       "Use ProteinMPNN to redesign chain B and reduce its aggregation propensity.",
       _any("proteinmpnn")),
    _C("mpnn_fixed_backbone2", "mpnn", "Do a fixed-backbone redesign of chain A.",
       _any("proteinmpnn")),
    _C("mpnn_charged_iface", "mpnn",
       "Redesign the dimer interface to be more charged.", _any("proteinmpnn")),
    _C("mpnn_exclude_pro", "mpnn",
       "Run ProteinMPNN on chain A and avoid introducing prolines.", _any("proteinmpnn")),

    # ─────────────── selection_scope (redesign a subset → proteinmpnn) ─────────
    _C("sel_selected", "selection_scope",
       "Redesign only the residues I have selected on chain A.", _any("proteinmpnn")),
    _C("sel_iface_only", "selection_scope",
       "Redesign just the dimer-interface residues on chain A, keep the backbone fixed.",
       _any("proteinmpnn")),
    _C("sel_highlighted", "selection_scope",
       "Redesign the residues I've highlighted in the viewer.", _any("proteinmpnn")),
    _C("sel_positions", "selection_scope",
       "Redesign only the selected positions.", _any("proteinmpnn")),
    _C("sel_active_site", "selection_scope",
       "Redesign just the selected binding-site residues on chain A.", _any("proteinmpnn")),
    _C("sel_core", "selection_scope",
       "Redesign just the buried core of chain A.", _any("proteinmpnn")),
    _C("sel_loop", "selection_scope",
       "Redesign the selected loop on chain A.", _any("proteinmpnn")),
    _C("sel_range", "selection_scope",
       "Redesign only residues 20 to 30 of chain A.", _any("proteinmpnn")),
    _C("sel_iface_nothing_else", "selection_scope",
       "Redesign the interface residues only, nothing else.", _any("proteinmpnn")),

    # ─────────────── camsol (solubility / aggregation → camsol) ────────────────
    _C("camsol_scan", "camsol", "Run a CamSol solubility scan on chain A.", _any("camsol")),
    _C("camsol_color_agg", "camsol", "Colour chain A by aggregation propensity.", _any("camsol")),
    _C("camsol_where_agg", "camsol",
       "Where are the aggregation-prone residues on chain A?", _any("camsol")),
    _C("camsol_profile", "camsol", "Compute the solubility profile of chain A.", _any("camsol")),
    _C("camsol_score", "camsol", "Score chain A for solubility.", _any("camsol")),
    _C("camsol_sticky", "camsol", "Highlight the sticky patches on chain A.", _any("camsol")),
    _C("camsol_chainB", "camsol", "Run CamSol on chain B.", _any("camsol")),
    _C("camsol_hotspots", "camsol", "Show me the aggregation hot spots on chain A.", _any("camsol")),
    _C("camsol_map", "camsol", "Map solubility onto chain A.", _any("camsol")),

    # ─────────────── esm (conservation → esm) ──────────────────────────────────
    _C("esm_conservation", "esm",
       "Score chain A for evolutionary conservation with ESM.", _any("esm")),
    _C("esm_most_conserved", "esm",
       "Which residues on chain A are the most conserved?", _any("esm")),
    _C("esm_run", "esm", "Run an ESM conservation analysis on chain A.", _any("esm")),
    _C("esm_color", "esm", "Colour chain A by conservation.", _any("esm")),
    _C("esm_per_residue", "esm", "Compute per-residue conservation for chain A.", _any("esm")),
    _C("esm_positions", "esm", "Where are the conserved positions in chain A?", _any("esm")),
    _C("esm_esm2", "esm", "Run ESM-2 on chain A.", _any("esm")),
    _C("esm_chainB", "esm", "Show evolutionary conservation for chain B.", _any("esm")),

    # ─────────────── disulfide (interchain SS candidates → disulfide) ──────────
    _C("ss_iface", "disulfide",
       "Suggest interchain disulfide bond candidates between chains A and B.", _any("disulfide")),
    _C("ss_where", "disulfide",
       "Where could I add a disulfide between chains A and B?", _any("disulfide")),
    _C("ss_dimer", "disulfide", "Predict disulfide bond candidates for the dimer.", _any("disulfide")),
    _C("ss_cys_pairs", "disulfide",
       "Find cysteine pairs that could form a disulfide across the interface.", _any("disulfide")),
    _C("ss_stabilising", "disulfide",
       "Suggest stabilising disulfides between chains A and B.", _any("disulfide")),
    _C("ss_engineering", "disulfide",
       "Identify disulfide engineering sites in the dimer.", _any("disulfide")),
    _C("ss_mutate_cys", "disulfide",
       "Which residue pairs could be mutated to cysteines for an interchain disulfide?",
       _any("disulfide")),
    _C("ss_recommend", "disulfide", "Recommend interchain SS bonds for this dimer.", _any("disulfide")),

    # ─────────────── proline (→ mutation_scan engineering path) ────────────────
    _C("pro_stabilise", "proline",
       "Suggest proline mutations to stabilise chain A.", _any("mutation_scan", "proline", "rosetta")),
    _C("pro_rigidify", "proline",
       "Where can I add prolines to rigidify chain A?", _any("mutation_scan", "proline", "rosetta")),
    _C("pro_substitutions", "proline",
       "Recommend proline substitutions for chain A.", _any("mutation_scan", "proline", "rosetta")),
    _C("pro_sites", "proline",
       "Find good proline mutation sites on chain A.", _any("mutation_scan", "proline", "rosetta")),
    _C("pro_loops", "proline",
       "Which loops on chain A could take a proline?", _any("mutation_scan", "proline", "rosetta")),
    _C("pro_backbone", "proline",
       "Suggest backbone-rigidifying proline mutations on chain A.", _any("mutation_scan", "proline", "rosetta")),
    _C("pro_scan", "proline", "Run a proline scan of chain A.", _any("mutation_scan", "proline", "rosetta")),
    _C("pro_candidates", "proline",
       "Identify proline engineering candidates on chain A.", _any("mutation_scan", "proline", "rosetta")),

    # ─────────────── multi_tool (combined routing) ─────────────────────────────
    _C("multi_camsol_esm", "multi_tool",
       "Run a CamSol solubility scan on chain A and also score it for ESM conservation.",
       _all("camsol", "esm")),
    _C("multi_assembly_disulfide", "multi_tool",
       "Map the interface between chains A and B and predict disulfide candidates there.",
       _any("disulfide")),
    _C("multi_sol_and_conservation", "multi_tool",
       "Score chain A for both solubility and conservation.", _all("camsol", "esm")),
    _C("multi_assembly_map", "multi_tool",
       "Detect the biological assembly and map the chain A–B interface.",
       _any("assembly_analyser")),
    _C("multi_disulfide_assembly", "multi_tool",
       "Find disulfide candidates and analyse the biological assembly.",
       _all("disulfide", "assembly_analyser")),
    _C("multi_full_scan", "multi_tool",
       "Run a full engineering mutation scan on chain A.", _any("mutation_scan")),
    _C("multi_esmfold", "multi_tool",
       "Predict the foldability of chain A with ESMFold.", _any("esmfold")),
    _C("multi_conservation_solubility", "multi_tool",
       "Show me chain A conservation and its solubility profile.", _all("esm", "camsol")),

    # ─────────────── viz (plain ChimeraX visualisation) ────────────────────────
    _C("viz_open_color", "viz", "Open 1HSG and colour it by chain.",
       _cmd(r"\bopen\b"), _cmd(r"bychain"), _any("chimerax")),
    _C("viz_color_red", "viz", "Colour chain A red.", _cmd(r"color[^\n]*\bred\b")),
    _C("viz_bg_black", "viz", "Set the background to black.",
       _cmd(r"bgColor\s+black"), _nocmd(r"background\s+color")),
    _C("viz_byelement", "viz", "Colour the structure by element.", _cmd(r"byelement")),
    _C("viz_pub_image", "viz", "Make a publication-quality image of the dimer.",
       _cmd(r"save[^\n]*\.png")),
    _C("viz_rainbow", "viz", "Colour the structure by rainbow.", _cmd(r"rainbow")),
    _C("viz_spheres", "viz", "Show the structure as spheres.", _cmd(r"sphere")),
    _C("viz_bfactor", "viz", "Colour chain A by B-factor.", _cmd(r"bfactor")),

    # ─────────────── rfdiffusion (pre-screen → rfdiffusion) ────────────────────
    _C("rfd_binder", "rfdiffusion", "Design a binder for chain A.", _any("rfdiffusion")),
    _C("rfd_binder_design", "rfdiffusion", "I want binder design for the interface.", _any("rfdiffusion")),
    _C("rfd_protein_binder", "rfdiffusion", "Make a protein binder against chain B.", _any("rfdiffusion")),
    _C("rfd_scaffold_motif", "rfdiffusion", "Scaffold a motif from chain A.", _any("rfdiffusion")),
    _C("rfd_motif_scaffold", "rfdiffusion", "Run a motif scaffold design.", _any("rfdiffusion")),
    _C("rfd_partial", "rfdiffusion", "Do partial diffusion on the structure.", _any("rfdiffusion")),
    _C("rfd_denovo_bb", "rfdiffusion", "De novo backbone design for a new fold.", _any("rfdiffusion")),
    _C("rfd_use", "rfdiffusion", "Use RFdiffusion to generate a backbone.", _any("rfdiffusion")),
]


# ── EXAMPLE POOL (few-shot / training; DISJOINT from EVAL_CORPUS) ────────────────
EXAMPLE_POOL: List[CorpusCase] = [
    _C("ex_zone_1", "zone", "Select chain A residues within 5 Å of the bound inhibitor.",
       _cmd(r":<"), _nocmd(_ZONE)),
    _C("ex_zone_2", "zone", "Pick the atoms within 4.5 Å of chain A.", _cmd(r"[:@]<"), _nocmd(_ZONE)),
    _C("ex_hide_1", "hide_show", "Hide chain A.",
       _cmd(r"hide\b[^\n]*\bA\b[^\n]*(cartoon|target)")),
    _C("ex_hide_2", "hide_show", "Show chain B as a cartoon.", _cmd(r"cartoon")),
    _C("ex_mpnn_1", "mpnn", "Redesign chain B with ProteinMPNN.", _any("proteinmpnn")),
    _C("ex_mpnn_2", "mpnn", "Optimise the sequence of chain A on a fixed backbone.", _any("proteinmpnn")),
    _C("ex_sel_1", "selection_scope", "Redesign the residues currently selected.", _any("proteinmpnn")),
    _C("ex_sel_2", "selection_scope", "Redesign only the flap loop on chain A.", _any("proteinmpnn")),
    _C("ex_camsol_1", "camsol", "Assess the solubility of chain B.", _any("camsol")),
    _C("ex_camsol_2", "camsol", "Find the aggregation-prone stretches on chain B.", _any("camsol")),
    _C("ex_esm_1", "esm", "How conserved is chain B?", _any("esm")),
    _C("ex_esm_2", "esm", "Run a conservation scan on chain B with ESM-2.", _any("esm")),
    _C("ex_ss_1", "disulfide", "Propose a disulfide staple across the dimer interface.", _any("disulfide")),
    _C("ex_ss_2", "disulfide", "Where can interchain cysteine bridges go?", _any("disulfide")),
    _C("ex_pro_1", "proline", "Add prolines to stiffen the loops of chain B.",
       _any("mutation_scan", "proline", "rosetta")),
    _C("ex_pro_2", "proline", "Proline engineering candidates for chain B.", _any("mutation_scan", "proline", "rosetta")),
    _C("ex_multi_1", "multi_tool", "Give me solubility and conservation for chain B.", _all("camsol", "esm")),
    _C("ex_multi_2", "multi_tool", "Analyse the assembly and its interface.", _any("assembly_analyser")),
    _C("ex_viz_1", "viz", "Colour chain B blue.", _cmd(r"color[^\n]*\bblue\b")),
    _C("ex_viz_2", "viz", "Set the background to white.",
       _cmd(r"bgColor\s+white"), _nocmd(r"background\s+color")),
    _C("ex_rfd_1", "rfdiffusion", "Design a binder targeting chain B.", _any("rfdiffusion")),
    _C("ex_rfd_2", "rfdiffusion", "Scaffold a motif into a new backbone.", _any("rfdiffusion")),
]

# Back-compat alias (the benchmark + tests reference the eval set).
CORPUS = EVAL_CORPUS


# ── Scoring ─────────────────────────────────────────────────────────────────────
def score_case(case: CorpusCase, result: Dict[str, Any]) -> Tuple[bool, List[Tuple[Check, bool]]]:
    per = [(c, bool(c.evaluate(result))) for c in case.checks]
    return all(p for _, p in per), per


def is_schema_valid(result: Any) -> bool:
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
    tool_checks = [c for c in case.checks if c.kind in ("tools_any", "tools_all")]
    return all(c.evaluate(result) for c in tool_checks)


def categories(cases: List[CorpusCase] = None) -> List[str]:
    cases = EVAL_CORPUS if cases is None else cases
    return sorted({c.category for c in cases})


def category_counts(cases: List[CorpusCase] = None) -> Dict[str, int]:
    cases = EVAL_CORPUS if cases is None else cases
    out: Dict[str, int] = {}
    for c in cases:
        out[c.category] = out.get(c.category, 0) + 1
    return out
