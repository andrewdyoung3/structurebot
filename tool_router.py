"""
tool_router.py
--------------
Routes translator output through the appropriate computational tools.

Dispatcher for the StructureBot tool pipeline:
  chimerax     -> ChimeraXBridge  (visualization — always handled by main.py)
  camsol       -> CamsolBridge    (per-residue solubility scoring)
  esm          -> EsmBridge       (evolutionary conservation via ESM-2)
  proteinmpnn  -> ProteinMPNNBridge (fixed-backbone sequence redesign)
  rfdiffusion  -> RFdiffusionBridge (de novo backbone diffusion — stub)

The translator emits a 'tools_needed' list such as:
  ["chimerax"]                      - visualization only (default)
  ["camsol"]                        - solubility + auto-visualization
  ["esm"]                           - conservation + auto-visualization
  ["camsol", "esm"]                 - both analyses, then visualize
  ["chimerax", "camsol"]            - open/setup, then solubility

Workflow
--------
1.  route(translator_result)
      Augments the translator dict with tool routing metadata.
      Safe to call before user confirmation; no tool runs.

2.  execute(routed_result, status_callback=...)
      Runs the non-chimerax tools in order.
      Returns the same dict augmented with step results and viz_commands.
      main.py then runs those viz_commands through ChimeraXBridge.

Each bridge's analyze() method returns a ToolStepResult with:
  - data          : tool-specific output (dict)
  - viz_commands  : ChimeraX commands to visualize results
  - viz_explanations : human-readable explanations for each viz command
  - summary       : one-line result description
  - error         : error string or None
"""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from chimerax_bridge import ChimeraXBridge
    from session_state import SessionState


# ── Tool step result ──────────────────────────────────────────────────────────

class ToolStepResult:
    """Encapsulates the result of one non-chimerax tool step."""

    def __init__(
        self,
        tool:              str,
        success:           bool,
        data:              Optional[Dict[str, Any]] = None,
        viz_commands:      Optional[List[str]] = None,
        viz_explanations:  Optional[List[str]] = None,
        summary:           str = "",
        error:             Optional[str] = None,
        elapsed_ms:        float = 0.0,
    ):
        self.tool             = tool
        self.success          = success
        self.data             = data or {}
        self.viz_commands     = viz_commands or []
        self.viz_explanations = viz_explanations or []
        self.summary          = summary
        self.error            = error
        self.elapsed_ms       = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool":             self.tool,
            "success":          self.success,
            "data":             self.data,
            "viz_commands":     self.viz_commands,
            "viz_explanations": self.viz_explanations,
            "summary":          self.summary,
            "error":            self.error,
            "elapsed_ms":       round(self.elapsed_ms, 1),
        }


# ── Router ─────────────────────────────────────────────────────────────────────

# Hydrophilic residues to bias toward when the request asks for a hydrophilic
# (more soluble / more polar) redesign — used when the translator emits
# `bias_toward: "hydrophilic"` instead of an explicit `bias_amino_acids` list.
_HYDROPHILIC_AAS = "DENQHKRST"

# GPU-heavy bridges. Before dispatching any of these, the local-LLM translator is
# freed from VRAM (translator.ensure_translator_unloaded) so it never contends
# with a fold/design run for GPU memory. No-op under the Claude backend.
_GPU_BRIDGE_TOOLS = frozenset({
    "proteinmpnn", "mpnn_esmfold", "rfdiffusion", "colabfold",
    "validate_design", "esmfold", "esm",
})


class ToolRouter:
    """
    Routes translator output through the correct computational tools.

    Usage::

        router = ToolRouter(bridge, session)

        # After translation — augments result, no execution yet:
        routed = router.route(translator_result)

        # After user confirms — runs computational tools:
        routed = router.execute(routed, status_callback=console.status)
        # routed["all_viz_commands"] and routed["all_viz_explanations"]
        # are now populated; main.py runs them through the bridge.
    """

    _TOOL_ICONS: Dict[str, str] = {
        "chimerax":          "🎨",
        "camsol":            "💧",
        "esm":               "🧬",
        "esmfold":           "🔮",
        "proteinmpnn":       "🔬",
        "mpnn_esmfold":      "🔬🔮",
        "rfdiffusion":       "🌀",
        "rosetta":           "⚗️",
        "mutation_scan":     "🔬⚗️",
        "assembly_analyser": "🔗",
        "disulfide":         "🔗⚗️",
        "proline":           "🧪",
        "glycan":            "🍬",
        "glycan_positions":  "🍬🔮",
        "netnglyc":          "🔬🍬",
        "salt_bridge":       "⚡",
        "cavity":            "🕳",
        "double_mutant":     "⚗️🔗",
        "validate_ddg":      "⚗️✅",
        "colabfold":         "🧬🔮",
        "validate_design":   "🧬✅",
    }

    # Validate-design meta-tool: the HEAVY ColabFold+Rosetta orchestrator. It
    # fires ONLY on explicit high-accuracy phrasing — BARE "validate design" /
    # "check this design" etc. deliberately fall through to the light
    # mpnn_esmfold / ESMFold fast screen (its prior behaviour). The "... with
    # colabfold/alphafold" case is handled separately in _detect (compound).
    # Checked BEFORE the bare match in route(), so a QUALIFIED phrase like
    # "high-accuracy validate design" hits the meta-tool while bare
    # "validate design" does not.
    _VALIDATE_DESIGN_KEYWORDS: tuple = (
        "full validation",
        "full design validation",
        "high-accuracy validation", "high accuracy validation",
        "high-confidence validation", "high confidence validation",
        "high-accuracy validate", "high accuracy validate",
        "high-confidence validate", "high confidence validate",
        "thoroughly validate", "thorough validation",
    )

    # Explicit ColabFold (AF2) folding keywords — the high-accuracy structure-
    # prediction path. Kept tight so it neither swallows nor is swallowed by the
    # esmfold / mpnn_esmfold pipeline (see route()).
    _COLABFOLD_KEYWORDS: tuple = (
        "colabfold", "alphafold", "alpha fold", "af2",
    )

    # Oligomer words → homo-oligomer copy count (used for both intent + parsing).
    _OLIGOMER_COPIES: Dict[str, int] = {
        "monomer": 1, "monomeric": 1,
        "dimer": 2, "homodimer": 2, "dimeric": 2,
        "trimer": 3, "homotrimer": 3, "trimeric": 3,
        "tetramer": 4, "homotetramer": 4, "tetrameric": 4,
        "pentamer": 5, "pentameric": 5,
        "hexamer": 6, "hexameric": 6,
    }

    # Keywords that signal a ProteinMPNN + ESMFold validation request.
    # Checked case-insensitively; any match replaces 'proteinmpnn' (or 'esmfold'
    # when session has MPNN results) in the pipeline.
    _MPNN_ESMFOLD_KEYWORDS: tuple = (
        "validate design",
        "validate sequence",
        "fold design",
        "fold sequence",
        "esmfold mpnn",
        "esmfold proteinmpnn",
        "mpnn esmfold",
        "proteinmpnn esmfold",
        "fold redesign",
        "structural integrity",
        "assess fold",
        "check fold",
        "validate mpnn",
        "esmfold top",
        "fold top",
        "sequences to assess",
        "designed sequences",
        "redesigned sequences",
        # Phrases that arise when the user refers to "top N sequences" after redesign
        "top 2-3 sequences",
        "top 2–3 sequences",   # en-dash variant (U+2013)
        "top 3 sequences",
        "top sequences",
        "assess structural integrity",
        "structural fidelity",
    )

    # Contextual keywords: when present alongside a session with MPNN results,
    # an 'esmfold' dispatch is redirected to _run_mpnn_esmfold().
    _MPNN_CONTEXT_KEYWORDS: tuple = (
        "sequence",
        "design",
        "top",
        "integrity",
        "redesign",
    )

    # Keywords that trigger the "show designed sequences" fast-path handler
    # (bypasses LLM when session already has ProteinMPNN results).
    _SEQUENCE_DISPLAY_KEYWORDS: tuple = (
        "show designed sequences",
        "show design sequences",
        "show sequences",
        "list designed sequences",
        "list sequences",
        "print designed sequences",
        "print sequences",
        "display designed sequences",
        "display sequences",
        "output sequences",
        "output the sequences",
        "what are the designed sequences",
        "what are the sequences",
        "show mpnn sequences",
        "what sequences did",
        "show me the sequences",
        "sequences for the redesigns",
        "redesign sequences",
    )

    # Verbs used in fuzzy sequence-display matching (see handle_sequence_display_command).
    _SEQUENCE_DISPLAY_VERBS: tuple = (
        "show", "output", "print", "list", "display", "what",
    )

    # Phrasings that mean "RUN/RE-RUN a redesign" — a display request must never
    # match these (re-running is stochastic and overwrites the design).
    _MPNN_RUN_TRIGGERS: tuple = (
        "with proteinmpnn", "with protein mpnn", "with mpnn", "with ligandmpnn",
        "using proteinmpnn", "using mpnn", "run proteinmpnn", "run mpnn",
        "redesign chain", "redesign the chain", "redesign again", "rerun", "re-run",
        "another design", "new design", "design again", "redesign the dimer",
        "redesign the interface", "redesign selected", "make it hydrophilic",
        "make it hydrophobic",
    )
    _MPNN_DISPLAY_VERBS: tuple = (
        "show", "output", "print", "display", "list", "what", "see", "view",
        "give", "tell", "get",
    )
    # Phrasings that ask for the WT-vs-redesign ALIGNMENT / change set.
    _MPNN_ALIGNMENT_KEYWORDS: tuple = (
        "alignment", "wt vs", "wildtype vs", "vs wildtype", "vs wt",
        "compare to wildtype", "compared to wildtype", "compare to wt",
        "what changed", "what positions changed", "which positions changed",
        "show the changes", "the changes", "show the redesign", "redesign view",
        "what mutations", "mutations made", "side by side", "side-by-side",
    )

    _FASTA_EXPORT_KEYWORDS: tuple = (
        "export sequences",
        "save sequences",
        "save fasta",
        "export fasta",
        "write sequences",
        "download sequences",
    )

    _SALT_BRIDGE_KEYWORDS: tuple = (
        "salt bridge",
        "salt-bridge",
        "electrostatic interaction",   # narrowed from "electrostatic" — avoids matching
        "ionic interaction",           # "show electrostatic surface" (visualization)
        "charge pair",
        "engineer salt",
        "new salt bridge",
    )

    # Cavity keyword list — does NOT include bare "void" (matches "avoiding").
    # Bare "void" / "voids" is handled with word-boundary regex in
    # _detect_cavity_intent().  Only explicit cavity-specific compound forms here.
    # Removed: "hydrophobic core" (too broad — matches mutation scan requests)
    _CAVITY_KEYWORDS: tuple = (
        "cavit",           # cavity, cavities, cavity-filling
        "internal void",   # compound — "internal voids" also matches via substring
        "buried void",     # compound — "buried voids" also matches via substring
        "fill void",
        "cavity fill",
        "fill cavity",
        "packing defect",
        "buried pocket",   # pocket + burial context
        "internal pocket",
        "fill pocket",
        "hollow",          # hollow protein core
        "fpocket",         # explicit cavity-detection tool name
        "tunnel",          # protein tunnel / channel
    )

    # Assembly / dimer keywords trigger chains=None (all chains) in find_cavities.
    # Removed: "binding pocket" — too broad; a binding-pocket analysis request is
    # not the same as asking for dimer cavity detection.
    _CAVITY_ASSEMBLY_KEYWORDS: tuple = (
        "full dimer",
        "full assembly",
        "both chains",
        "all chains",
        "dimer cavity",
        "oligomer",
        "assembly cavity",
        "interface cavity",
        "interface pocket",
        "active site cavity",
    )

    # Keywords that signal an N-glycosylation site scan request.
    # Checked case-insensitively; any match rewrites tools_needed to ["glycan"]
    # and clears any translator clarification_needed flag.
    _GLYCAN_KEYWORDS: tuple = (
        "glycan",
        "glycosylat",        # glycosylation, glycosylated, glycosylate, …
        "n-linked",
        "nxs",               # N-X-S sequon notation
        "nxt",               # N-X-T sequon notation
        "sequon",
        "glycoengineering",
        "glycosylation site",
        "n-glycan",
        "sugar",
    )

    # Keywords that signal a projection-aware glycosylation position scan
    # (distinct from the classic NXS/T sequon scan).  Checked BEFORE the
    # general glycan block so these phrases are never swallowed by _GLYCAN_KEYWORDS.
    _GLYCAN_POSITIONS_KEYWORDS: tuple = (
        "glycosylation positions",
        "glycan candidates",
        "glycan sites",
        "domain masking",
        "immunosilence",
        "glycan engineering candidates",
        "surface glycosylation",
        "projection-aware glycosylation",
    )

    # Keywords that trigger a standalone NetNGlyc OST recognition prediction.
    # These fire when the user explicitly requests OST/NetNGlyc scoring
    # without going through glycan_positions (which auto-calls NetNGlyc on
    # top-5 candidates already).
    _NETNGLYC_KEYWORDS: tuple = (
        "netnglyc",
        "ost recognition",
        "ost score",
        "oligosaccharyltransferase",
        "ost prediction",
        "glycosylation efficiency",
        "sequon recognition",
        "glycan ost",
        "ost glycosylation",
    )

    # Keywords that signal a mutation scan / engineering scan request.
    # Used as a fallback when the translator returns an unexpected tool
    # (e.g. cavity) for a clear solubility / mutation engineering request.
    # Matches both "mutation" (singular) and "mutations" (plural) via "mutati".
    _MUTATION_SCAN_KEYWORDS: tuple = (
        "suggest mutation",       # singular
        "suggest mutations",      # plural
        "improve solubility",     # solubility improvement without explicit mutation
        "reduce aggregation",     # aggregation engineering
        "mutation scan",          # explicit scan name
        "solubility scan",        # explicit scan name
        "mutation to improve",    # singular + context
        "mutations to improve",   # plural + context
        "engineering candidate",  # engineering language
        "stabilising mutation",   # stability mutation (British)
        "stabilizing mutation",   # stability mutation (American)
        "stability mutation",
    )

    # Keywords that trigger double mutant pair scoring.
    # Checked BEFORE mutation_scan dispatch so these phrases are never
    # sent through the single-point scan pipeline.
    _DOUBLE_MUTANT_KEYWORDS: tuple = (
        "double mutant",
        "combine mutations",
        "combined mutations",
        "synergistic",
        "epistasis",
        "pair mutations",
        "two mutations",
        "mutation combination",
        "epitope preserv",
        "preserve epitope",
        "scaffold stabili",
    )

    # Keywords that signal the high-accuracy ddG VALIDATION tier (multi-trajectory
    # median ddG on a small explicit set of mutations / top scan candidates).
    # Distinct from the fast single-trajectory mutation_scan path. Checked
    # case-insensitively; checked BEFORE mutation_scan / double_mutant overrides.
    _VALIDATE_DDG_KEYWORDS: tuple = (
        "validate ddg",
        "validate the ddg",
        "high accuracy ddg",
        "high-accuracy ddg",
        "high confidence stability",
        "high-confidence stability",
        "confirm ddg",
        "confirm the ddg",
        "confirm stability",
        "validate stability",
        "high accuracy stability",
        "multi-trajectory",
        "multitrajectory",
    )

    # Keywords that signal a proline substitution scan request.
    # Checked case-insensitively; any match overrides generic mutation_scan routing.
    _PROLINE_KEYWORDS: tuple = (
        "proline",           # catches "proline mutations", "proline scan", etc.
        "pro substitut",     # "pro substitution(s)"
        "pro mutation",      # "pro mutation(s)"
        "backbone stabili",  # "backbone stabilisation / stabilise"
        "entropic stabili",  # "entropic stabilisation"
        "phi angle",         # explicit backbone geometry reference
        "φ angle",      # φ angle (Unicode phi)
        "rigidif",           # "rigidify", "rigidification"
        "proline scan",      # explicit tool name
        "proline candidate", # explicit candidate language
    )

    def __init__(
        self,
        bridge:  "ChimeraXBridge",
        session: "SessionState",
    ):
        self.bridge  = bridge
        self.session = session

        # Bridges are instantiated lazily on first use
        self._camsol_bridge:           Optional[Any] = None
        self._esm_bridge:              Optional[Any] = None
        self._esmfold_bridge:          Optional[Any] = None
        self._proteinmpnn_bridge:      Optional[Any] = None
        self._mpnn_esmfold_pipeline:   Optional[Any] = None
        self._rfdiffusion_bridge:      Optional[Any] = None
        self._rosetta_bridge:          Optional[Any] = None
        self._mutation_scanner:        Optional[Any] = None
        self._assembly_analyser:       Optional[Any] = None
        self._disulfide_bridge:        Optional[Any] = None
        self._proline_bridge:          Optional[Any] = None
        self._glycan_bridge:           Optional[Any] = None
        self._netnglyc_bridge:         Optional[Any] = None
        self._salt_bridge_bridge:      Optional[Any] = None
        self._cavity_bridge:           Optional[Any] = None
        self._double_mutant_bridge:    Optional[Any] = None
        self._colabfold_bridge:        Optional[Any] = None

    # ── Phase 1: Route (no execution) ─────────────────────────────────────────

    # ── Cavity intent helpers ─────────────────────────────────────────────────

    @classmethod
    def _detect_cavity_intent(cls, text: str) -> bool:
        """
        Return True if *text* signals a cavity detection / filling request.

        "void" / "voids" are matched with a word-boundary regex (\\bvoids?\\b) so
        that "avoiding" does NOT trigger cavity routing.  All other cavity keywords
        are matched as substrings (they are long enough to be unambiguous).
        """
        lower = text.lower()
        if re.search(r'\bvoids?\b', lower):
            return True
        return any(kw in lower for kw in cls._CAVITY_KEYWORDS)

    # ── Mutation scan intent helpers ──────────────────────────────────────────

    @classmethod
    def _detect_mutation_scan_intent(cls, text: str) -> bool:
        """
        Return True if *text* signals a mutation scan / engineering request.

        Matches 'mutation' (singular) and 'mutations' (plural) via shared
        substring, plus standalone solubility / aggregation engineering phrases.
        Used as a routing fallback when the translator returns an unexpected tool.
        """
        lower = text.lower()
        # Substring check covers mutation / mutations / mutational
        if "mutati" in lower and any(
            kw in lower for kw in ("solubility", "aggregation", "stabili", "engineer",
                                   "improve", "suggest", "scan", "candidate")
        ):
            return True
        return any(kw in lower for kw in cls._MUTATION_SCAN_KEYWORDS)

    # ── Double mutant intent helpers ──────────────────────────────────────────

    @classmethod
    def _detect_double_mutant_intent(cls, text: str) -> bool:
        """Return True if *text* signals a double mutant pair scoring request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._DOUBLE_MUTANT_KEYWORDS)

    # ── Proline intent helpers ────────────────────────────────────────────────

    @classmethod
    def _detect_validate_ddg_intent(cls, text: str) -> bool:
        """Return True if *text* requests the high-accuracy ddG validation tier."""
        lower = (text or "").lower()
        return any(kw in lower for kw in cls._VALIDATE_DDG_KEYWORDS)

    @classmethod
    def _detect_proline_intent(cls, text: str) -> bool:
        """Return True if *text* signals a proline substitution scan request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._PROLINE_KEYWORDS)

    def _rewrite_as_proline(
        self,
        tools_needed: List[str],
        tool_inputs:  Dict[str, Any],
    ) -> tuple:
        """
        Replace 'mutation_scan' with 'proline' in the tool pipeline.

        Builds proline tool_inputs from the mutation_scan inputs so model_id
        and chain are preserved.  Other tools (assembly_analyser, disulfide,
        chimerax) are kept unchanged.

        Returns (new_tools_needed, new_tool_inputs).
        """
        new_tools  = []
        new_inputs = dict(tool_inputs)

        for tool in tools_needed:
            if tool == "mutation_scan":
                new_tools.append("proline")
                ms = tool_inputs.get("mutation_scan", {})
                # Always target the primary (crystal) structure — not a
                # secondary ESMFold prediction that the translator may have
                # guessed because it was the active model in ChimeraX.
                new_inputs["proline"] = {
                    "model_id": self._primary_model_id(),
                    "chain":    ms.get("chain", "A"),
                    "pdb_path": ms.get("pdb_path"),
                    "top_n":    5,
                }
                new_inputs.pop("mutation_scan", None)
            else:
                new_tools.append(tool)

        # If mutation_scan wasn't in the list, still ensure proline is added
        if "proline" not in new_tools:
            new_tools.append("proline")
            new_inputs["proline"] = {
                "model_id": self._primary_model_id(),
                "chain":    "A",
                "top_n":    5,
            }

        return new_tools, new_inputs

    # ── Glycan intent helpers ─────────────────────────────────────────────────

    @classmethod
    def _detect_glycan_intent(cls, text: str) -> bool:
        """Return True if *text* signals an N-glycosylation site scan request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._GLYCAN_KEYWORDS)

    def _rewrite_as_glycan(
        self,
        tools_needed: List[str],
        tool_inputs:  Dict[str, Any],
    ) -> tuple:
        """
        Replace the current tool pipeline with a single 'glycan' step.

        Inherits model_id and chain from any existing tool_inputs so that
        context set by the translator (chain letter, model number) is preserved.
        Initial ChimeraX commands in ``result["commands"]`` are unaffected.

        Returns (new_tools_needed, new_tool_inputs).
        """
        new_inputs = dict(tool_inputs)

        # Inherit model_id / chain from whatever the translator already parsed
        model_id: Optional[str] = None
        chain:    str           = "A"
        for inp in tool_inputs.values():
            if isinstance(inp, dict):
                if inp.get("model_id") and not model_id:
                    model_id = str(inp["model_id"])
                if inp.get("chain"):
                    chain = inp["chain"]

        model_id = model_id or self._first_model_id()

        new_inputs["glycan"] = {
            "model_id": model_id,
            "chain":    chain,
            "top_n":    3,
        }
        # Drop other analysis tool inputs; keep chimerax passthrough untouched
        for drop in ("mutation_scan", "proline", "camsol", "esm"):
            new_inputs.pop(drop, None)

        return ["glycan"], new_inputs

    @classmethod
    def _detect_glycan_positions_intent(cls, text: str) -> bool:
        """
        Return True if *text* signals a projection-aware glycosylation
        position scan (distinct from the classic NXS/T sequon scan).
        """
        lower = text.lower()
        return any(kw in lower for kw in cls._GLYCAN_POSITIONS_KEYWORDS)

    @classmethod
    def _detect_netnglyc_intent(cls, text: str) -> bool:
        """Return True if *text* signals a standalone NetNGlyc OST prediction request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._NETNGLYC_KEYWORDS)

    # ── MPNN+ESMFold intent helpers ───────────────────────────────────────────

    @classmethod
    def _detect_mpnn_esmfold_intent(cls, text: str) -> bool:
        """Return True if *text* signals a ProteinMPNN + ESMFold validation request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._MPNN_ESMFOLD_KEYWORDS)

    def _rewrite_as_mpnn_esmfold(
        self,
        tools_needed: List[str],
        tool_inputs:  Dict[str, Any],
    ) -> tuple:
        """
        Replace ``'proteinmpnn'`` with ``'mpnn_esmfold'`` in the tool pipeline.

        Builds mpnn_esmfold tool_inputs from the proteinmpnn inputs so that
        model_id and chain are preserved.  Other tools are kept unchanged.

        Returns (new_tools_needed, new_tool_inputs).
        """
        new_tools  = []
        new_inputs = dict(tool_inputs)

        for tool in tools_needed:
            if tool == "proteinmpnn":
                new_tools.append("mpnn_esmfold")
                pi = tool_inputs.get("proteinmpnn", {})
                new_inputs["mpnn_esmfold"] = {
                    "model_id": pi.get("model_id") or self._first_model_id(),
                    "chain_id": pi.get("chain_id") or pi.get("chain", "A"),
                }
                new_inputs.pop("proteinmpnn", None)
            elif tool == "esmfold":
                # Redirect standalone ESMFold → mpnn_esmfold when session has
                # MPNN results (route() gate already checked this condition).
                new_tools.append("mpnn_esmfold")
                ei = tool_inputs.get("esmfold", {})
                new_inputs["mpnn_esmfold"] = {
                    "model_id": ei.get("model_id") or self._first_model_id(),
                    "chain_id": ei.get("chain_id") or ei.get("chain", "A"),
                }
                new_inputs.pop("esmfold", None)
            else:
                new_tools.append(tool)

        # If neither proteinmpnn nor esmfold was in the list, still add mpnn_esmfold
        if "mpnn_esmfold" not in new_tools:
            new_tools.append("mpnn_esmfold")
            new_inputs["mpnn_esmfold"] = {
                "model_id": self._first_model_id(),
                "chain_id": "A",
            }

        return new_tools, new_inputs

    # ── Phase 1: Route (no execution) ─────────────────────────────────────────

    def route(
        self,
        translator_result: Dict[str, Any],
        user_input:        str = "",
    ) -> Dict[str, Any]:
        """
        Augment a translator result with tool routing metadata.

        Adds the following keys (safe to call before user confirmation):
          "tools_needed"    — list of tools from translator (default ["chimerax"])
          "tool_steps_info" — list of {tool, icon, description} for each step
          "has_extra_tools" — True if any non-chimerax tools are present
          "_user_input"     — original user text, used by execute() for proline guard

        If *user_input* contains proline intent keywords and 'mutation_scan' is
        in tools_needed, the pipeline is rewritten to use 'proline' instead.
        """
        tools_needed = translator_result.get("tools_needed") or ["chimerax"]
        tool_inputs  = translator_result.get("tool_inputs") or {}

        # ── Validate-design meta-tool override ─────────────────────────────────
        # Checked FIRST: "validate design" appears in _MPNN_ESMFOLD_KEYWORDS too,
        # so this must win and replace tools_needed before that override runs.
        # Distinct from bare `colabfold` (no fold/oligomer/template trigger) and
        # from `validate_ddg` (its keywords are about ddG/stability, not design).
        _vd_intent = bool(
            user_input
            and self._detect_validate_design_intent(user_input)
            and "validate_design" not in tools_needed
        )
        if _vd_intent:
            _vd_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _vd_chain = _inp["chain"]
                    break
            vd_inputs: Dict[str, Any] = {
                "model_id":    self._primary_model_id(),
                "chain":       _vd_chain,
                "_user_input": user_input,
            }
            vd_inputs.update(self._parse_colabfold_options(user_input))  # seq/copies/template/quick
            vd_inputs.update(self._parse_validate_design_options(user_input))
            tools_needed = ["validate_design"]
            tool_inputs  = {"validate_design": vd_inputs}

        # ── High-accuracy ddG validation tier override ─────────────────────────
        # Checked FIRST so "validate ddg" / "high-accuracy stability" route to the
        # multi-trajectory validation tier, NOT the fast single-trajectory scan or
        # double-mutant pipeline. Operates on an explicit mutation list (parsed in
        # _run_validate_ddg) or the top candidates from existing scan_results.
        _vddg_intent = bool(user_input and self._detect_validate_ddg_intent(user_input))
        if _vddg_intent and "validate_ddg" not in tools_needed:
            _v_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _v_chain = _inp["chain"]
                    break
            tools_needed = ["validate_ddg"]
            tool_inputs  = {
                "validate_ddg": {
                    "model_id":    self._primary_model_id(),
                    "chain":       _v_chain,
                    "_user_input": user_input,
                }
            }

        # ── ColabFold intent override ──────────────────────────────────────────
        # Explicit high-accuracy folding path (AF2 via the WSL2 ColabFold env).
        # Fires only on explicit keywords (colabfold/alphafold, "fold … as a
        # <oligomer>", or "use PDB XXXX as template to fold"), so it neither
        # swallows nor is swallowed by the esmfold / mpnn_esmfold pipeline.
        _cf_intent = bool(
            user_input
            and self._detect_colabfold_intent(user_input)
            and "colabfold" not in tools_needed
            and "validate_design" not in tools_needed  # meta-tool already claimed it
        )
        if _cf_intent:
            _cf_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _cf_chain = _inp["chain"]
                    break
            cf_inputs: Dict[str, Any] = {
                "model_id":    self._primary_model_id(),
                "chain":       _cf_chain,
                "_user_input": user_input,
            }
            cf_inputs.update(self._parse_colabfold_options(user_input))
            tools_needed = ["colabfold"]
            tool_inputs  = {"colabfold": cf_inputs}

        # ── Proline intent override ────────────────────────────────────────────
        # Check BEFORE building step_info so the icon/description are correct.
        if user_input and self._detect_proline_intent(user_input):
            if "mutation_scan" in tools_needed:
                tools_needed, tool_inputs = self._rewrite_as_proline(
                    tools_needed, tool_inputs
                )

        # ── MPNN+ESMFold intent override ───────────────────────────────────────
        # Always rewrite if 'proteinmpnn' is in the pipeline.
        # Only rewrite 'esmfold' → 'mpnn_esmfold' when the session already holds
        # ProteinMPNN results (prevents "check fold mutation I64E" from hijacking
        # a plain ESMFold foldability request on a structure with no designs).
        if user_input and self._detect_mpnn_esmfold_intent(user_input):
            _session_has_mpnn = (
                self.session.get_proteinmpnn_result(self._first_model_id()) is not None
            )
            if "proteinmpnn" in tools_needed or (
                _session_has_mpnn and "esmfold" in tools_needed
            ):
                tools_needed, tool_inputs = self._rewrite_as_mpnn_esmfold(
                    tools_needed, tool_inputs
                )

        # ── Glycan positions intent override ──────────────────────────────────
        # Must fire BEFORE the general glycan check so that phrases like
        # "glycan candidates" / "domain masking" are not swallowed by the
        # broader glycan keyword set.
        _glycan_positions_intent = bool(
            user_input and self._detect_glycan_positions_intent(user_input)
        )
        if _glycan_positions_intent and "glycan_positions" not in tools_needed:
            _gp_model_id = self._first_model_id()
            _gp_chain    = "A"
            for _inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _gp_chain = _inp["chain"]
                    break
            tools_needed = ["glycan_positions"]
            tool_inputs  = {
                "glycan_positions": {
                    "model_id":    _gp_model_id,
                    "chain":       _gp_chain,
                    "top_n":       20,
                    "_user_input": user_input,
                }
            }

        # ── NetNGlyc intent override ───────────────────────────────────────────
        # Fires for explicit OST recognition requests (e.g. "run NetNGlyc on
        # my sequence" / "what is the OST score for position 42?").
        _netnglyc_intent = bool(
            user_input and self._detect_netnglyc_intent(user_input)
        )
        if _netnglyc_intent and "netnglyc" not in tools_needed:
            _ng_model_id = self._first_model_id()
            _ng_chain    = "A"
            for _inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _ng_chain = _inp["chain"]
                    break
            tools_needed = ["netnglyc"]
            tool_inputs  = {
                "netnglyc": {
                    "model_id":    _ng_model_id,
                    "chain":       _ng_chain,
                    "_user_input": user_input,
                }
            }

        # ── Glycan intent override ─────────────────────────────────────────
        # Fires when glycan keywords are present and the translator did NOT
        # already emit "glycan" in tools_needed (wrong routing or unclear query).
        # ALSO clears any clarification_needed flag from the translator — this
        # prevents the clarification retry loop in main.py from asking the user a
        # question whose answer would be re-sent to translate(), which crashes with
        # stop_reason='refusal' when the short answer ("chain A") has no prior
        # context the model can work with.
        _glycan_intent = bool(user_input and self._detect_glycan_intent(user_input))
        if _glycan_intent and "glycan" not in tools_needed and not _glycan_positions_intent and not _netnglyc_intent:
            tools_needed, tool_inputs = self._rewrite_as_glycan(
                tools_needed, tool_inputs
            )

        # ── Salt bridge intent override ────────────────────────────────────────
        _sb_intent = bool(user_input and any(kw in user_input.lower() for kw in self._SALT_BRIDGE_KEYWORDS))
        if _sb_intent and "salt_bridge" not in tools_needed:
            # Rewrite to salt_bridge tool
            tools_needed = ["salt_bridge"]
            tool_inputs = {"salt_bridge": {"model_id": self._primary_model_id(), "chain": "A"}}
            # Extract chain from any existing tool_input
            for inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(inp, dict) and inp.get("chain"):
                    tool_inputs["salt_bridge"]["chain"] = inp["chain"]
                    break

        # ── Cavity intent override ──────────────────────────────────────────────
        _cav_intent = bool(user_input and self._detect_cavity_intent(user_input))
        if _cav_intent and "cavity" not in tools_needed:
            tools_needed = ["cavity"]
            tool_inputs = {"cavity": {"model_id": self._primary_model_id(), "chain": "A"}}
            for inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(inp, dict) and inp.get("chain"):
                    tool_inputs["cavity"]["chain"] = inp["chain"]
                    break

        # ── Cavity assembly mode detection ──────────────────────────────────────
        # When assembly keywords are present, find_cavities() uses chains=None
        # (all chains), enabling interface cavity detection for dimers/oligomers.
        if "cavity" in tool_inputs:
            _cav_assembly = bool(
                user_input and any(
                    kw in user_input.lower()
                    for kw in self._CAVITY_ASSEMBLY_KEYWORDS
                )
            )
            tool_inputs["cavity"]["assembly_mode"] = _cav_assembly

        # ── Double mutant intent override ─────────────────────────────────────
        # Fires when double mutant keywords are detected AND the tool is not
        # already set.  Runs AFTER other overrides so proline/glycan/etc. take
        # priority when their more-specific keywords also appear.
        _dm_intent = bool(
            user_input
            and self._detect_double_mutant_intent(user_input)
            and "double_mutant" not in tools_needed
        )
        if _dm_intent:
            _dm_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _dm_chain = _inp["chain"]
                    break
            tools_needed = ["double_mutant"]
            tool_inputs  = {
                "double_mutant": {
                    "model_id":    self._primary_model_id(),
                    "chain":       _dm_chain,
                    "_user_input": user_input,
                }
            }

        # ── Mutation scan intent fallback ──────────────────────────────────────
        # Fires when user clearly wants a mutation scan but the translator returned
        # a different tool (e.g. cavity, chimerax).  Handles singular "mutation"
        # as well as "improve solubility" without the mutation keyword.
        # Runs LAST: all more-specific overrides (proline, glycan, double_mutant)
        # have already fired and take priority.
        # Blocks on ACTUAL user keywords (not translator output) so that cavity
        # and salt-bridge are only respected when the user's words match those tools.
        _lower_ui_ms = user_input.lower() if user_input else ""
        _ms_intent = bool(
            user_input
            and self._detect_mutation_scan_intent(user_input)
            and "mutation_scan" not in tools_needed
            and not self._detect_proline_intent(user_input)
            and not self._detect_glycan_intent(user_input)
            and not self._detect_double_mutant_intent(user_input)
            and not any(t in tools_needed for t in (
                "proline", "glycan", "glycan_positions", "double_mutant", "disulfide",
            ))
            # Only skip if the user's words genuinely match cavity / salt-bridge tools.
            and not self._detect_cavity_intent(user_input)
            and not any(kw in _lower_ui_ms for kw in self._SALT_BRIDGE_KEYWORDS)
        )
        if _ms_intent:
            _ms_chain    = "A"
            _ms_model_id = self._primary_model_id()
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict):
                    if _inp.get("chain"):
                        _ms_chain = _inp["chain"]
                    if _inp.get("model_id"):
                        _ms_model_id = str(_inp["model_id"])
            tools_needed = ["mutation_scan"]
            tool_inputs  = {
                "mutation_scan": {
                    "model_id": _ms_model_id,
                    "chain":    _ms_chain,
                    "focus":    "solubility",
                }
            }

        result = dict(translator_result)
        result["tools_needed"] = tools_needed
        result["tool_inputs"]  = tool_inputs
        result["_user_input"]  = user_input   # passed to execute() for guard

        # Suppress translator clarification when we've resolved the intent ourselves
        if _glycan_intent or _glycan_positions_intent or _netnglyc_intent:
            result["clarification_needed"] = None

        step_info: List[Dict[str, Any]] = []
        for tool in tools_needed:
            icon = self._TOOL_ICONS.get(tool, "⚙️")
            step_info.append({
                "tool":        tool,
                "icon":        icon,
                "description": self._step_description(tool, tool_inputs, result),
            })

        result["tool_steps_info"] = step_info
        result["has_extra_tools"] = any(t != "chimerax" for t in tools_needed)
        return result

    def _step_description(
        self,
        tool:        str,
        tool_inputs: Dict[str, Any],
        result:      Dict[str, Any],
    ) -> str:
        if tool == "chimerax":
            n = len(result.get("commands", []))
            return f"Execute {n} ChimeraX command(s)"
        if tool == "camsol":
            inp   = tool_inputs.get("camsol", {})
            chain = inp.get("chain", "all chains")
            mid   = inp.get("model_id") or self._first_model_id()
            return f"CamSol solubility analysis — #{mid} chain {chain}"
        if tool == "esm":
            inp = tool_inputs.get("esm", {})
            mid = inp.get("model_id") or self._first_model_id()
            return f"ESM-2 evolutionary conservation — #{mid}"
        if tool == "esmfold":
            inp = tool_inputs.get("esmfold", {})
            mid = inp.get("model_id") or self._first_model_id()
            return f"ESMFold foldability prediction — #{mid} (ESM Atlas API)"
        if tool == "colabfold":
            inp     = tool_inputs.get("colabfold", {})
            copies  = inp.get("copies", 1)
            shape   = "monomer" if copies <= 1 else f"{copies}-copy homo-oligomer"
            tmpl    = " (templated)" if inp.get("template") else ""
            return f"ColabFold AF2 structure prediction — {shape}{tmpl} (remote MSA)"
        if tool == "validate_design":
            return ("Validate design — ColabFold confidence + matchmaker RMSD + "
                    "Rosetta folding-energy sanity (evidence-rich report)")
        if tool == "proteinmpnn":
            return "ProteinMPNN fixed-backbone sequence redesign"
        if tool == "mpnn_esmfold":
            inp   = tool_inputs.get("mpnn_esmfold", {})
            mid   = inp.get("model_id") or self._first_model_id()
            top_n = inp.get("top_n", 3)
            return (
                f"MPNN + ESMFold validation — #{mid} "
                f"top {top_n} design(s) (pLDDT confidence)"
            )
        if tool == "rfdiffusion":
            inp  = tool_inputs.get("rfdiffusion", {})
            mode = inp.get("mode", "binder")
            return f"RFdiffusion de novo backbone generation (mode: {mode})"
        if tool == "rosetta":
            inp  = tool_inputs.get("rosetta", {})
            mid  = inp.get("model_id") or self._first_model_id()
            muts = inp.get("mutations", [])
            return (
                f"Rosetta ddG — #{mid}, "
                f"{len(muts)} mutation(s)"
                if muts else f"Rosetta ddG — #{mid}"
            )
        if tool == "mutation_scan":
            inp   = tool_inputs.get("mutation_scan", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            focus = inp.get("focus", "solubility")
            mode  = inp.get("analysis_mode", "monomer")
            return f"CamSol + ESM + Rosetta scan — #{mid} chain {chain} [{focus}] [{mode} mode]"
        if tool == "assembly_analyser":
            inp  = tool_inputs.get("assembly_analyser", {})
            mid  = inp.get("model_id") or self._first_model_id()
            mode = inp.get("mode", "multimer")
            ch   = inp.get("chain_id", "")
            return (
                f"Assembly analysis — #{mid} [{mode} mode]"
                + (f", chain {ch}" if ch else "")
            )
        if tool == "disulfide":
            inp  = tool_inputs.get("disulfide", {})
            mid  = inp.get("model_id") or self._first_model_id()
            ca   = inp.get("chain_a", "A")
            cb   = inp.get("chain_b", "B")
            return (
                f"Disulfide candidate prediction — #{mid} chains {ca}/{cb} "
                "(geometry + ESM + DynaMut2)"
            )
        if tool == "proline":
            inp   = tool_inputs.get("proline", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"Proline substitution scan — #{mid} chain {chain} "
                "(backbone φ/ψ + ESM tolerance)"
            )
        if tool == "glycan":
            inp   = tool_inputs.get("glycan", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"N-glycosylation site scan — #{mid} chain {chain} "
                "(NXS/T sequons, SASA + SS + ESM)"
            )
        if tool == "glycan_positions":
            inp   = tool_inputs.get("glycan_positions", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"Projection-aware glycosylation position scan — #{mid} chain {chain} "
                "(all residues, SASA + projection + ESM)"
            )
        if tool == "netnglyc":
            inp   = tool_inputs.get("netnglyc", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"NetNGlyc OST recognition prediction — #{mid} chain {chain} "
                "(DTU Health Tech API)"
            )
        if tool == "salt_bridge":
            inp   = tool_inputs.get("salt_bridge", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return f"Salt bridge analysis -- #{mid} chain {chain}"
        if tool == "cavity":
            inp   = tool_inputs.get("cavity", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return f"Cavity detection and filling -- #{mid} chain {chain}"
        if tool == "double_mutant":
            inp  = tool_inputs.get("double_mutant", {})
            mid  = inp.get("model_id") or self._first_model_id()
            ui   = inp.get("_user_input", "")
            mode = "epitope" if any(kw in ui.lower() for kw in ("epitope", "binding", "preserve")) else "stability"
            return (
                f"Double mutant pair scoring — #{mid} [{mode} mode] "
                "(DynaMut2 prediction_mm + epistasis)"
            )
        if tool == "validate_ddg":
            inp = tool_inputs.get("validate_ddg", {})
            mid = inp.get("model_id") or self._first_model_id()
            import config as _cfg
            n   = getattr(_cfg, "ROSETTA_VALIDATION_TRAJECTORIES", 5)
            cyc = getattr(_cfg, "ROSETTA_VALIDATION_CYCLES", 8)
            return (
                f"High-accuracy ddG validation — #{mid} "
                f"({n} trajectories x {cyc} cycles, median + confidence)"
            )
        return f"Unknown tool: {tool}"

    # ── Phase 2: Execute (non-chimerax tools) ─────────────────────────────────

    def execute(
        self,
        routed_result:   Dict[str, Any],
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run all non-chimerax tools in the pipeline.

        Adds to the returned dict:
          "tool_step_results"    — list of ToolStepResult.to_dict() per non-cx step
          "all_viz_commands"     — ChimeraX commands from all tools combined
          "all_viz_explanations" — corresponding human-readable explanations
          "tool_summaries"       — {tool_name: summary_string}
          "pipeline_success"     — True if no tool step failed
          "pipeline_error"       — first error message, or None
        """
        tools_needed = routed_result.get("tools_needed") or ["chimerax"]
        # Case-insensitive tool_inputs lookup (mirrors the case-insensitive
        # dispatch) so "CamSol"-keyed inputs still reach the camsol bridge.
        tool_inputs  = {str(k).strip().lower(): v
                        for k, v in (routed_result.get("tool_inputs") or {}).items()}
        # user_input stored by route() for the proline guard
        user_input   = routed_result.get("_user_input", "")

        step_results:   List[Dict[str, Any]] = []
        all_viz_cmds:   List[str] = []
        all_viz_exps:   List[str] = []
        tool_summaries: Dict[str, str] = {}
        pipeline_error: Optional[str] = None

        for _raw_tool in tools_needed:
            tool = str(_raw_tool).strip().lower()   # case-insensitive routing
            if tool == "chimerax":
                # ChimeraX execution is handled by main.py; skip here
                step_results.append({
                    "tool":    "chimerax",
                    "skipped": True,
                    "note":    "executed by main.py",
                })
                continue

            icon = self._TOOL_ICONS.get(tool, "⚙️")
            if status_callback:
                status_callback(f"{icon} Running {tool}…")

            t0   = time.perf_counter()
            step = self._dispatch_tool(tool, tool_inputs.get(tool) or {}, user_input=user_input)
            step.elapsed_ms = (time.perf_counter() - t0) * 1000

            step_results.append(step.to_dict())
            tool_summaries[tool] = step.summary

            if step.success:
                # Cache results in session state for later use / display
                mid = (tool_inputs.get(tool) or {}).get("model_id") or self._first_model_id()
                self.session.add_tool_result(tool, mid, step.data)

                if step.viz_commands:
                    all_viz_cmds.extend(step.viz_commands)
                    all_viz_exps.extend(step.viz_explanations)
            else:
                pipeline_error = pipeline_error or step.error

        result = dict(routed_result)
        result["tool_step_results"]    = step_results
        result["all_viz_commands"]     = all_viz_cmds
        result["all_viz_explanations"] = all_viz_exps
        result["tool_summaries"]       = tool_summaries
        result["pipeline_success"]     = pipeline_error is None
        result["pipeline_error"]       = pipeline_error
        return result

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def _dispatch_tool(
        self,
        tool:       str,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Route to the correct bridge; return ToolStepResult.

        Secondary guards:
          - Proline: if tool is 'mutation_scan' but user_input contains proline
            keywords, redirect to _run_proline() (backbone φ-angle scanner).
          - Glycan: if tool is 'mutation_scan' but user_input contains glycan
            keywords, redirect to _run_glycan() (N-glycosylation site scan).
        These guards fire if route() was not called with user_input.
        """
        # Case-INSENSITIVE tool resolution: the registry literals below are
        # lowercase, so a local model emitting "CamSol"/"ESM" still routes. The
        # benchmark scorer matches case-insensitively too (kept in lockstep).
        tool = (tool or "").strip().lower()

        # VRAM invariant: before any GPU-heavy bridge, free the local LLM
        # translator from VRAM (no-op under Claude / when nothing is loaded).
        # The explicit guard — not the idle timer — is the contract that prevents
        # a mid-run OOM from the translator contending with a fold.
        if tool in _GPU_BRIDGE_TOOLS:
            try:
                from translator import ensure_translator_unloaded
                ensure_translator_unloaded()
            except Exception:
                pass

        try:
            if tool == "camsol":
                return self._run_camsol(inputs)
            if tool == "esm":
                return self._run_esm(inputs)
            if tool == "esmfold":
                # ── MPNN+ESMFold guard (secondary safety net) ────────────────
                # If the user's request contains MPNN/ESMFold validation keywords
                # *or* general context keywords (design, sequence, top, …) and
                # the session already holds ProteinMPNN results, redirect to the
                # combined pipeline instead of the bare ESMFold tool.
                if user_input:
                    _lower_ui         = user_input.lower()
                    _has_mpnn_kw      = self._detect_mpnn_esmfold_intent(user_input)
                    _has_ctx_kw       = any(kw in _lower_ui for kw in self._MPNN_CONTEXT_KEYWORDS)
                    _session_has_mpnn = (
                        self.session.get_proteinmpnn_result(self._first_model_id()) is not None
                    )
                    if (_has_mpnn_kw or _has_ctx_kw) and _session_has_mpnn:
                        return self._run_mpnn_esmfold({
                            "model_id": inputs.get("model_id") or self._first_model_id(),
                            "chain_id": inputs.get("chain_id") or inputs.get("chain", "A"),
                        })
                return self._run_esmfold(inputs)
            if tool == "proteinmpnn":
                return self._run_proteinmpnn(inputs)
            if tool == "mpnn_esmfold":
                return self._run_mpnn_esmfold(inputs)
            if tool == "rfdiffusion":
                return self._run_rfdiffusion(inputs)
            if tool == "rosetta":
                return self._run_rosetta(inputs)
            if tool == "mutation_scan":
                # ── Double mutant guard (highest-priority secondary net) ──────
                # Fires when user_input contains "double" or "combine" regardless
                # of the full keyword list — covers "double mutations", "combine
                # these mutations", etc. that route() may not have intercepted.
                _lower_ui = user_input.lower() if user_input else ""
                if _lower_ui and ("double" in _lower_ui or "combine" in _lower_ui):
                    dm_inputs = {
                        "model_id":    self._primary_model_id(),
                        "chain":       inputs.get("chain", "A"),
                        "_user_input": user_input,
                    }
                    return self._run_double_mutant(dm_inputs)
                # ── Proline guard (secondary safety net) ─────────────────────
                # If the user asked about proline and route() didn't catch it
                # (e.g. user_input was not passed to route()), redirect here.
                if user_input and self._detect_proline_intent(user_input):
                    proline_inputs = {
                        "model_id": self._primary_model_id(),
                        "chain":    inputs.get("chain", "A"),
                        "pdb_path": inputs.get("pdb_path"),
                        "top_n":    5,
                    }
                    return self._run_proline(proline_inputs)
                # ── Glycan guard (secondary safety net) ──────────────────────
                # If the user asked about glycans and route() didn't catch it,
                # redirect here.
                if user_input and self._detect_glycan_intent(user_input):
                    glycan_inputs = {
                        "model_id": inputs.get("model_id") or self._first_model_id(),
                        "chain":    inputs.get("chain", "A"),
                        "top_n":    3,
                    }
                    return self._run_glycan(glycan_inputs)
                return self._run_mutation_scan(inputs)
            if tool == "double_mutant":
                return self._run_double_mutant(inputs, user_input=user_input)
            if tool == "validate_ddg":
                return self._run_validate_ddg(inputs, user_input=user_input)
            if tool == "colabfold":
                return self._run_colabfold(inputs, user_input=user_input)
            if tool == "validate_design":
                return self._run_validate_design(inputs, user_input=user_input)
            if tool == "assembly_analyser":
                return self._run_assembly_analyser(inputs)
            if tool == "disulfide":
                return self._run_disulfide(inputs)
            if tool == "proline":
                return self._run_proline(inputs)
            if tool == "glycan":
                return self._run_glycan(inputs)
            if tool == "glycan_positions":
                return self._run_glycan_positions(inputs)
            if tool == "netnglyc":
                return self._run_netnglyc(inputs)
            if tool == "salt_bridge":
                return self._run_salt_bridge(inputs)
            if tool == "cavity":
                return self._run_cavity(inputs)
            return ToolStepResult(
                tool=tool, success=False,
                error=(
                    f"Unknown tool '{tool}'. "
                    "Available: chimerax, camsol, esm, esmfold, proteinmpnn, "
                    "mpnn_esmfold, rfdiffusion, rosetta, mutation_scan, "
                    "assembly_analyser, disulfide, proline, glycan, "
                    "glycan_positions, netnglyc, salt_bridge, cavity, "
                    "double_mutant, validate_ddg, colabfold, validate_design."
                ),
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool=tool, success=False,
                error=f"{tool} raised an unexpected error: {exc}",
            )

    def _run_disulfide(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run interchain disulfide bond candidate prediction."""
        bridge   = self._get_disulfide_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain_a  = inputs.get("chain_a", "A")
        chain_b  = inputs.get("chain_b", "B")

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="disulfide", success=False,
                error=(
                    "Disulfide prediction requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        # Pull any interface residues the assembly analyser already found
        binding_site: Optional[List[int]] = None
        interface_data = self.session.get_interface_residues(model_id)
        if interface_data:
            # Collect binding-site residues for chain_a
            binding_site = self.session.get_protected_residues_for_chain(
                model_id, chain_a
            ) or None

        import time as _time
        t0 = _time.perf_counter()

        result = bridge.analyze(
            pdb_path              = pdb_path,
            chain_a               = chain_a,
            chain_b               = chain_b,
            session               = self.session,
            model_id              = model_id,
            binding_site_residues = binding_site,
        )
        result.elapsed_ms = (_time.perf_counter() - t0) * 1000

        if result.success and result.data.get("candidates"):
            self.session.set_disulfide_candidates(
                model_id, chain_a, chain_b, result.data["candidates"]
            )

        return result

    def _run_proline(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run proline substitution scan on a chain."""
        import time as _time
        bridge   = self._get_proline_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 5))

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="proline", success=False,
                error=(
                    "Proline scan requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        # Pull ESM tolerance scores from session if already computed
        esm_scores: Optional[Dict[int, float]] = None
        esm_data = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_data, dict):
            raw = esm_data.get("per_residue_tolerance") or esm_data.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        # Pull interface residues from session if already computed
        iface_set: Optional[set] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                iface_set = set(protected)

        # Optional DynaMut2 validation
        dynamut2_bridge: Optional[Any] = None
        use_dynamut2 = inputs.get("use_dynamut2", False)
        if use_dynamut2:
            try:
                dynamut2_bridge = self._get_rosetta_bridge()
            except Exception:
                pass   # proceed without DynaMut2 if unavailable

        # Pull user-declared active-site residues from session
        # (None → trigger SASA auto-detection in proline_bridge)
        session_fr = self.session.get_functional_residues()
        functional_residues: Optional[set] = session_fr if session_fr else None

        t0 = _time.perf_counter()
        try:
            result = bridge.full_proline_scan(
                pdb_path             = pdb_path,
                chain                = chain,
                interface_residues   = iface_set,
                esm_scores           = esm_scores,
                top_n                = top_n,
                dynamut2_bridge      = dynamut2_bridge,
                functional_residues  = functional_residues,
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool="proline", success=False,
                error=f"Proline scan failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        candidates = result.get("candidates", [])
        if result.get("count", 0) == 0:
            return ToolStepResult(
                tool       = "proline",
                success    = True,
                data       = result,
                summary    = f"Proline scan chain {chain}: no candidates found.",
                elapsed_ms = elapsed_ms,
            )

        # Store in session
        self.session.set_proline_results(model_id, chain, result)

        # Visualization
        viz_cmds, viz_exps = bridge.generate_chimerax_commands(
            candidates[:top_n], model_id=model_id, chain=chain
        )

        summary = bridge._generate_summary(result)

        return ToolStepResult(
            tool             = "proline",
            success          = True,
            data             = result,
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_glycan(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run N-glycosylation site prediction on a chain."""
        import time as _time
        bridge   = self._get_glycan_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 3))
        min_score= float(inputs.get("min_score", 0.05))

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)
        if not sequence:
            return ToolStepResult(
                tool="glycan", success=False,
                error=(
                    "Glycan scan requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)

        # Pull ESM tolerance scores from session if available
        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        # Pull interface residues from session if available
        interface_residues: Optional[List[int]] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = protected

        t0 = _time.perf_counter()
        try:
            result = bridge.analyze(
                pdb_path           = pdb_path,
                chain              = chain,
                sequence           = sequence,
                model_id           = model_id,
                session            = self.session,
                interface_residues = interface_residues,
                esm_scores         = esm_scores,
                min_score          = min_score,
                top_n              = top_n,
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool="glycan", success=False,
                error=f"Glycan scan failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="glycan", success=False,
                error=result.get("error", "Glycan scan failed"),
                elapsed_ms=elapsed_ms,
            )

        cx_cmds = result.get("chimerax_commands", [])
        cx_exps = result.get("chimerax_explanations", [])

        if not cx_cmds:
            # Defensive: if analyze() returned nothing, log clearly so it's visible
            n_native = len(result.get("native_sequons", []))
            n_eng    = len(result.get("engineered_candidates", []))
            print(
                f"  [glycan] WARNING: no ChimeraX commands generated "
                f"(native={n_native}, engineered={n_eng})"
            )

        return ToolStepResult(
            tool             = "glycan",
            success          = True,
            data             = {k: v for k, v in result.items()
                                if k not in ("chimerax_commands", "chimerax_explanations", "summary")},
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = result.get("summary", "Glycan scan complete."),
            elapsed_ms       = elapsed_ms,
        )

    def _run_glycan_positions(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """
        Run projection-aware glycosylation position scan (all residues).

        Unlike _run_glycan() which detects existing NXS/T sequons, this method
        scans every surface-exposed, outward-projecting residue and ranks them
        as engineering targets for de-novo glycan attachment.
        """
        import time as _time
        bridge   = self._get_glycan_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 20))

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)
        if not sequence:
            return ToolStepResult(
                tool="glycan_positions", success=False,
                error=(
                    "Glycan position scan requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)

        # Pull ESM tolerance scores from session if available
        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        # Pull interface residues from session if available
        interface_residues: Optional[set] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = set(protected)

        t0 = _time.perf_counter()
        try:
            candidates = bridge.suggest_glycosylation_positions(
                pdb_path          = pdb_path,
                chain             = chain,
                sequence          = sequence,
                interface_residues= interface_residues,
                esm_scores        = esm_scores,
                top_n             = top_n,
            )
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            return ToolStepResult(
                tool="glycan_positions", success=False,
                error=f"Glycan position scan failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        # ── Auto-call NetNGlyc on top N candidates (when enabled) ───────────────
        netnglyc_annotated: bool = False
        try:
            import config as _cfg
            _netnglyc_enabled = getattr(_cfg, "NETNGLYC_ENABLED", True)
            _netnglyc_top_n   = getattr(_cfg, "NETNGLYC_TOP_N", 5)
        except ImportError:
            _netnglyc_enabled = True
            _netnglyc_top_n   = 5

        if _netnglyc_enabled and candidates:
            try:
                ng_bridge = self._get_netnglyc_bridge()
                top_cands = candidates[:_netnglyc_top_n]
                # Build an engineered sequence: apply the first candidate's
                # mutation to the sequence so NxS/T sequons appear at each pos.
                # We use the raw sequence as proxy (NetNGlyc scans all N positions).
                annotated = ng_bridge.integrate_with_glycan_candidates(
                    candidates          = top_cands,
                    engineered_sequence = sequence,
                )
                # Merge ost_score, ost_category, combined_confidence back into candidates
                annotated_map = {c.get("position"): c for c in annotated}
                for cand in candidates:
                    pos = cand.get("position")
                    if pos in annotated_map:
                        ann = annotated_map[pos]
                        cand["ost_score"]          = ann.get("ost_score")
                        cand["ost_category"]       = ann.get("ost_category")
                        cand["ost_prediction"]     = ann.get("ost_prediction")
                        cand["ost_confidence"]     = ann.get("ost_confidence")
                        cand["combined_confidence"] = ann.get("combined_confidence")
                # Store in session
                try:
                    self.session.set_netnglyc_results(model_id, annotated)
                except Exception:
                    pass
                netnglyc_annotated = True
            except Exception:
                pass   # NetNGlyc failure is non-fatal — proceed without OST scores

        cx_cmds, cx_exps = bridge.generate_positions_chimerax_commands(
            candidates, model_id=model_id, chain=chain, top_n=top_n
        )
        summary = bridge.generate_positions_summary(
            candidates, chain=chain, top_n=top_n
        )
        if netnglyc_annotated:
            summary += "\n  [OST scores from NetNGlyc 1.0 added to top candidates]"

        return ToolStepResult(
            tool             = "glycan_positions",
            success          = True,
            data             = {
                "candidates":         candidates,
                "count":              len(candidates),
                "netnglyc_annotated": netnglyc_annotated,
            },
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_netnglyc(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """
        Run NetNGlyc 1.0 OST recognition prediction for a sequence.

        Fetches the sequence from session, submits to the NetNGlyc REST API,
        and returns per-sequon OST scores annotated with confidence categories.
        Stores annotated results in session.netnglyc_results.
        """
        import time as _time
        try:
            import config as _cfg
            if not getattr(_cfg, "NETNGLYC_ENABLED", True):
                return ToolStepResult(
                    tool="netnglyc", success=False,
                    error="NetNGlyc is disabled (NETNGLYC_ENABLED=false in config).",
                )
        except ImportError:
            pass

        bridge   = self._get_netnglyc_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        if not sequence:
            return ToolStepResult(
                tool="netnglyc", success=False,
                error=(
                    "NetNGlyc requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        # Use the position hint from user input if present (score_engineered_sequon mode)
        sequon_position: Optional[int] = inputs.get("sequon_position")
        engineered_seq:  Optional[str] = inputs.get("engineered_sequence")
        wildtype_seq:    Optional[str] = inputs.get("wildtype_sequence") or sequence

        t0 = _time.perf_counter()
        if sequon_position and engineered_seq:
            result = bridge.score_engineered_sequon(
                sequon_position      = sequon_position,
                engineered_sequence  = engineered_seq,
                wildtype_sequence    = wildtype_seq,
            )
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if not result.get("success"):
                return ToolStepResult(
                    tool="netnglyc", success=False,
                    error=result.get("error", "NetNGlyc prediction failed"),
                    elapsed_ms=elapsed_ms,
                )
            summary = (
                f"NetNGlyc — position {sequon_position}: "
                f"OST score={result['ost_score']:.3f} ({result['ost_category']}). "
                f"{result.get('notes', '')}"
            )
            data = result
        else:
            result = bridge.predict_glycosylation(
                sequence = sequence,
                name     = f"model{model_id}_chain{chain}",
            )
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if not result.get("success"):
                return ToolStepResult(
                    tool="netnglyc", success=False,
                    error=result.get("error", "NetNGlyc prediction failed"),
                    elapsed_ms=elapsed_ms,
                )
            n_found = result["n_sequons_found"]
            n_high  = result["n_sequons_high_score"]
            summary = (
                f"NetNGlyc — {n_found} sequon(s) found, "
                f"{n_high} with high OST score (>0.7)."
            )
            data = result

        # Store in session
        try:
            self.session.set_netnglyc_results(model_id, data)
        except Exception:
            pass

        return ToolStepResult(
            tool       = "netnglyc",
            success    = True,
            data       = data,
            summary    = summary,
            elapsed_ms = elapsed_ms,
        )

    @staticmethod
    def _parse_glycan_validation_request(user_input: str) -> Optional[Dict[str, Any]]:
        """
        Parse a glycan validation sub-request from user text.

        Returns a dict with position if found, {"validate": True} if "validate"
        is present but no position, or None if the input is not a validation request.
        """
        if not user_input:
            return None
        lower = user_input.lower()
        if "validate" not in lower and "validation" not in lower:
            return None
        m = re.search(r"\b(?:position|pos|at)\s+(\d+)\b", lower)
        if m:
            return {"position": int(m.group(1))}
        return {"validate": True}

    def _run_salt_bridge(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run salt bridge analysis."""
        import time as _time
        bridge   = self._get_salt_bridge_bridge()
        model_id = inputs.get("model_id") or self._primary_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 10))

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="salt_bridge", success=False,
                error=(
                    "Salt bridge analysis requires a local PDB file.\n"
                    "  Load a structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download from RCSB."
                ),
            )

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        # Pull ESM scores and interface residues from session
        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        interface_residues: Optional[List[int]] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = protected

        t0 = _time.perf_counter()
        try:
            result = bridge.full_salt_bridge_scan(
                pdb_path           = pdb_path,
                chain              = chain,
                sequence           = sequence or "",
                interface_residues = interface_residues,
                esm_scores         = esm_scores,
                top_n              = top_n,
            )
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            return ToolStepResult(
                tool="salt_bridge", success=False,
                error=f"Salt bridge analysis failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="salt_bridge", success=False,
                error=result.get("error", "Salt bridge analysis failed"),
                elapsed_ms=elapsed_ms,
            )

        self.session.set_salt_bridge_results(model_id, result)

        cx_cmds, cx_exps = bridge.generate_chimerax_commands(result, model_id=model_id)
        summary = bridge.generate_summary(result)

        return ToolStepResult(
            tool             = "salt_bridge",
            success          = True,
            data             = result,
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_cavity(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run cavity detection and filling suggestion."""
        import time as _time
        bridge        = self._get_cavity_bridge()
        model_id      = inputs.get("model_id") or self._primary_model_id()
        chain         = inputs.get("chain", "A")
        top_n         = int(inputs.get("top_n", 10))
        assembly_mode = bool(inputs.get("assembly_mode", False))
        # chains=None → all chains (assembly mode); chains=[chain] → single chain
        chains_for_cavity = None if assembly_mode else [chain]

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="cavity", success=False,
                error=(
                    "Cavity detection requires a local PDB file.\n"
                    "  Load a structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download from RCSB."
                ),
            )

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        interface_residues: Optional[List[int]] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = protected

        t0 = _time.perf_counter()
        try:
            result = bridge.full_cavity_scan(
                pdb_path           = pdb_path,
                chain              = chain,
                sequence           = sequence or "",
                interface_residues = interface_residues,
                esm_scores         = esm_scores,
                top_n              = top_n,
                chains             = chains_for_cavity,
            )
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            return ToolStepResult(
                tool="cavity", success=False,
                error=f"Cavity analysis failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="cavity", success=False,
                error=result.get("error", "Cavity analysis failed"),
                elapsed_ms=elapsed_ms,
            )

        self.session.set_cavity_results(model_id, result)

        cx_cmds, cx_exps = bridge.generate_chimerax_commands(result, model_id=model_id)
        summary = bridge.generate_summary(result)

        return ToolStepResult(
            tool             = "cavity",
            success          = True,
            data             = result,
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_camsol(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge   = self._get_camsol_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain")
        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        if not sequence:
            return ToolStepResult(
                tool="camsol", success=False,
                error=(
                    "No amino-acid sequence available. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )
        return bridge.analyze(
            sequence,
            model_id=model_id,
            chain=chain,
            session=self.session,
        )

    def _run_esm(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge   = self._get_esm_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain")
        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        if not sequence:
            return ToolStepResult(
                tool="esm", success=False,
                error=(
                    "No amino-acid sequence available. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )
        return bridge.analyze(sequence, model_id=model_id, session=self.session)

    def _run_esmfold(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run ESMFold foldability prediction via ESM Atlas API."""
        import time as _time
        bridge   = self._get_esmfold_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()

        # Resolve sequence
        sequence = inputs.get("sequence") or self._fetch_sequence(
            model_id, inputs.get("chain")
        )
        if not sequence:
            return ToolStepResult(
                tool="esmfold", success=False,
                error=(
                    "ESMFold requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        mutation_positions: List[int] = inputs.get("mutation_positions") or []

        t0 = _time.perf_counter()
        if mutation_positions:
            # Compare wildtype vs mutant sequences at specified positions
            mut_sequence = inputs.get("mut_sequence", "")
            if not mut_sequence:
                # Single-sequence foldability prediction (no mutant provided)
                result = bridge.predict(sequence, label=f"#{model_id}")
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                if not result["success"]:
                    return ToolStepResult(
                        tool="esmfold", success=False,
                        error=f"ESMFold prediction failed: {result.get('error')}",
                        elapsed_ms=elapsed_ms,
                    )
                mean_plddt = result["mean_plddt"]
                summary = (
                    f"ESMFold: model #{model_id} — mean pLDDT {mean_plddt:.1f} "
                    f"({'high' if mean_plddt > 70 else 'low'} confidence)."
                )
                return ToolStepResult(
                    tool="esmfold", success=True,
                    data={
                        "mean_plddt":  mean_plddt,
                        "plddt":       result["plddt"],
                        "length":      result["length"],
                        "source":      result.get("source"),
                    },
                    summary=summary,
                    elapsed_ms=elapsed_ms,
                )
            else:
                cmp = bridge.compare_to_wildtype(sequence, mut_sequence, mutation_positions)
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                if not cmp["success"]:
                    return ToolStepResult(
                        tool="esmfold", success=False,
                        error=f"ESMFold comparison failed: {cmp.get('error')}",
                        elapsed_ms=elapsed_ms,
                    )
                risk = cmp["foldability_risk"]
                drop = cmp["plddt_drop"]
                summary = (
                    f"ESMFold: foldability risk = {risk} "
                    f"(mean pLDDT drop {drop:.1f} at mutation positions)."
                )
                if cmp.get("warning"):
                    summary += f"\n  {cmp['warning']}"
                return ToolStepResult(
                    tool="esmfold", success=True,
                    data=cmp,
                    summary=summary,
                    elapsed_ms=elapsed_ms,
                )
        else:
            # Plain foldability check — no mutation comparison
            result = bridge.predict(sequence, label=f"#{model_id}")
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if not result["success"]:
                return ToolStepResult(
                    tool="esmfold", success=False,
                    error=f"ESMFold prediction failed: {result.get('error')}",
                    elapsed_ms=elapsed_ms,
                )
            mean_plddt = result["mean_plddt"]
            conf       = "high" if mean_plddt > 70 else "medium" if mean_plddt > 50 else "low"
            summary    = (
                f"ESMFold: model #{model_id} — mean pLDDT {mean_plddt:.1f} "
                f"({conf} confidence, {result['length']} residues)."
            )
            return ToolStepResult(
                tool="esmfold", success=True,
                data={
                    "mean_plddt": mean_plddt,
                    "plddt":      result["plddt"],
                    "length":     result["length"],
                    "source":     result.get("source"),
                },
                summary=summary,
                elapsed_ms=elapsed_ms,
            )

    def _read_selected_residues(self, model_id: str, chain_id: str) -> List[int]:
        """
        Read the LIVE ChimeraX selection and return the selected residue numbers
        on *chain_id* of *model_id*.

        Uses ``info residues sel & #<model>/<chain>`` whose output lines look like
        ``residue id /A:8 name GLN index 7`` — we extract the PDB residue numbers.
        Returns [] on any failure (caller decides the fallback).
        """
        try:
            cmd = f"info residues sel & #{model_id}/{chain_id}"
            res = self.bridge.run_command(cmd)
            text = (res.get("value") or "") if isinstance(res, dict) else ""
            nums = [
                int(m.group(1))
                for m in re.finditer(rf"/{re.escape(chain_id)}:(-?\d+)\b", text)
            ]
            return sorted(set(nums))
        except Exception:
            return []

    def _run_proteinmpnn(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """
        Run ProteinMPNN fixed-backbone sequence redesign.

        Consumes the STRUCTURED constraint fields the translator emits
        (``exclude_amino_acids``, ``bias_amino_acids``, ``design_mode``,
        ``use_selection``) — no natural-language re-parsing — and resolves the
        designable set from the LIVE ChimeraX selection when the request is
        selection-scoped, falling back to a direct interface computation, and
        NEVER silently to a whole-chain redesign.
        """
        bridge   = self._get_proteinmpnn_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain_id = inputs.get("chain_id") or inputs.get("chain", "A")

        # ── Resolve PDB path ──────────────────────────────────────────────────
        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool    = "proteinmpnn",
                success = False,
                error   = (
                    "ProteinMPNN requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        # ── Structured constraint fields (straight from the translator) ────────
        # The model is not perfectly consistent about field names, so accept the
        # observed synonyms rather than re-parsing the NL text.
        exclude_aas = (inputs.get("exclude_amino_acids")
                       or inputs.get("omit_amino_acids") or [])
        omit_aas = "".join(sorted({
            ch.upper() for aa in exclude_aas for ch in str(aa) if ch.isalpha()
        }))

        bias_aas = [str(a).upper() for a in (inputs.get("bias_amino_acids") or [])
                    if str(a).strip()]
        bias_toward = str(inputs.get("bias_toward") or "").lower()
        if not bias_aas and "hydrophil" in bias_toward:
            bias_aas = list(_HYDROPHILIC_AAS)   # D E N Q H K R S T

        # design_scope is the canonical key (see translator prompt); design_mode
        # and the boolean flags below are tolerated synonyms.
        design_mode = (str(inputs.get("design_scope") or "")
                       + " " + str(inputs.get("design_mode") or "")).lower()
        partner_chain = (inputs.get("partner_chain")
                         or inputs.get("interface_partner_chain"))
        design_positions = inputs.get("design_positions")
        # A SCOPED (selection / interface-only) redesign — detected robustly,
        # since the model is inconsistent about exact field names.  Any boolean
        # "design only the selection/interface" flag, an interface/selection
        # design_mode, an explicit position list, or a named partner chain all
        # mean "do NOT redesign the whole chain".
        scoped = bool(
            design_positions
            or partner_chain
            or inputs.get("use_selection")
            or inputs.get("design_only_interface")
            or inputs.get("redesign_selected")
            or inputs.get("selected_only")
            or "interface" in design_mode
            or "select" in design_mode
        )

        # ── Designable set: explicit > live selection > interface > (error) ────
        interface_design = bool(inputs.get("interface_design"))
        if scoped and not design_positions:
            # The user's live ChimeraX selection is the ground truth for "the
            # selected residues"; if none was captured, compute the interface
            # directly from coordinates (deterministic, never whole-chain).
            design_positions = self._read_selected_residues(model_id, chain_id)
            if not design_positions:
                interface_design = True

        # ── Legacy "protect cached interface residues" path ────────────────────
        # Only for an UNRESTRICTED design (no selection / interface / explicit set);
        # restricted paths fix the complement themselves.
        fixed_positions: List[int] = inputs.get("fixed_positions") or []
        if (not fixed_positions and not scoped
                and not design_positions and not interface_design):
            interface_data = self.session.get_interface_residues(model_id)
            if interface_data:
                fixed_positions = (
                    self.session.get_protected_residues_for_chain(model_id, chain_id)
                    or []
                )

        full_inputs: Dict[str, Any] = {
            "model_id":         model_id,
            "pdb_path":         pdb_path,
            "chain_id":         chain_id,
            "fixed_positions":  fixed_positions,
            "num_sequences":    int(inputs.get("num_sequences", 8)),
            "temperature":      float(inputs.get("temperature", 0.1)),
            # Design-scope + AA constraints (structured fields, consumed directly)
            "design_positions": design_positions,
            "interface_design": interface_design,
            "partner_chain":    partner_chain,
            "omit_aas":         omit_aas,
            "bias_aas":         bias_aas,
        }
        return bridge.analyze(full_inputs, session=self.session)

    def _run_mpnn_esmfold(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run ProteinMPNN + ESMFold combined redesign and validation pipeline."""
        import time as _time
        import config as _cfg

        pipeline  = self._get_mpnn_esmfold_pipeline()
        model_id  = inputs.get("model_id") or self._first_model_id()
        chain_id  = inputs.get("chain_id") or inputs.get("chain", "A")
        top_n     = int(inputs.get("top_n", _cfg.MPNN_ESMFOLD_TOP_N))
        include_wt = bool(inputs.get("include_wildtype", _cfg.MPNN_ESMFOLD_INCLUDE_WT))
        plddt_threshold = float(inputs.get("plddt_threshold", 70.0))

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)

        t0 = _time.perf_counter()
        try:
            result = pipeline.run(
                model_id         = model_id,
                pdb_path         = pdb_path,
                chain_id         = chain_id,
                session          = self.session,
                top_n            = top_n,
                plddt_threshold  = plddt_threshold,
                include_wildtype = include_wt,
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool    = "mpnn_esmfold",
                success = False,
                error   = f"MPNN+ESMFold pipeline failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool       = "mpnn_esmfold",
                success    = False,
                error      = result.get("error", "Pipeline failed"),
                elapsed_ms = elapsed_ms,
            )

        # Store in session (pdb_str is stripped automatically on save)
        self.session.set_mpnn_esmfold_results(model_id, result)

        cx_cmds = result.get("chimerax_commands", [])
        cx_exps = result.get("chimerax_explanations", [])

        return ToolStepResult(
            tool             = "mpnn_esmfold",
            success          = True,
            data             = {
                k: v for k, v in result.items()
                if k not in ("chimerax_commands", "chimerax_explanations", "summary")
            },
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = result.get("summary", "MPNN+ESMFold validation complete."),
            elapsed_ms       = elapsed_ms,
        )

    def _run_rfdiffusion(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge = self._get_rfdiffusion_bridge()
        return bridge.analyze(inputs, session=self.session)

    def _run_rosetta(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge    = self._get_rosetta_bridge()
        model_id  = inputs.get("model_id") or self._first_model_id()
        mutations = inputs.get("mutations", [])
        chain     = inputs.get("chain")

        if not mutations:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "No mutations supplied to rosetta tool.\n"
                    "  Provide tool_inputs: {\"rosetta\": {\"mutations\": "
                    "[{\"chain\": \"A\", \"position\": 82, \"from_aa\": \"V\", "
                    "\"to_aa\": \"A\"}]}}"
                ),
            )

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "Rosetta requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        return bridge.analyze(
            pdb_path  = pdb_path,
            mutations = mutations,
            session   = self.session,
            model_id  = model_id,
            chain     = chain,
        )

    def _run_validate_ddg(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        High-accuracy ddG validation tier: multi-trajectory MEDIAN ddG + spread
        + confidence on a SMALL explicit set of mutations (named in the request)
        or the top candidates from existing scan_results. NOT a full re-scan.
        """
        import re as _re
        user_input = user_input or inputs.get("_user_input", "")
        model_id   = inputs.get("model_id") or self._primary_model_id()
        chain      = inputs.get("chain", "A")

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="validate_ddg", success=False,
                error=(
                    "No structure loaded or PDB file unavailable. "
                    "Run 'open 1HSG' (or your structure) first."
                ),
            )

        # 1) Explicit mutations named in the request, e.g. "I72R", "V82A".
        mutations: List[Dict[str, Any]] = []
        seen: set = set()
        for m in _re.finditer(
            r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})([ACDEFGHIKLMNPQRSTVWY])\b",
            user_input or "",
        ):
            frm, pos, to = m.group(1), int(m.group(2)), m.group(3)
            key = f"{frm}{pos}{to}"
            if key in seen:
                continue
            seen.add(key)
            mutations.append({"chain": chain, "position": pos,
                              "from_aa": frm, "to_aa": to})

        source_note = ""
        if mutations:
            source_note = f"{len(mutations)} mutation(s) named in request"
        else:
            # 2) Fall back to the top candidates from a prior scan.
            scan_data = self.session.get_scan_result(model_id)
            if not scan_data:
                return ToolStepResult(
                    tool="validate_ddg", success=False,
                    error=(
                        "No mutations named and no prior scan found. "
                        "Name the mutations (e.g. 'validate ddg for I72R and V82A'), "
                        "or run a mutation scan first then ask to validate the top hits."
                    ),
                )
            top = sorted(
                scan_data, key=lambda c: c.get("combined_score", 0.0), reverse=True
            )[:3]
            for c in top:
                mutations.append({
                    "chain":    c.get("chain", chain),
                    "position": c.get("position"),
                    "from_aa":  c.get("from_aa"),
                    "to_aa":    c.get("to_aa"),
                })
            source_note = f"top {len(mutations)} candidate(s) from prior scan"

        if not mutations:
            return ToolStepResult(
                tool="validate_ddg", success=False,
                error="Could not determine any mutations to validate.",
            )

        def _status(msg: str) -> None:
            print(msg, flush=True)

        bridge = self._get_rosetta_bridge()
        result = bridge.validate_ddg(
            pdb_path          = pdb_path,
            mutations         = mutations,
            session           = self.session,
            model_id          = model_id,
            chain             = chain,
            progress_callback = _status,
        )

        if not result.success:
            return ToolStepResult(
                tool="validate_ddg", success=False,
                error=result.error or "Validation-tier ddG failed.",
            )

        data    = result.data or {}
        scores  = data.get("ddg_scores", {})
        spreads = data.get("ddg_spread", {})
        confs   = data.get("ddg_confidence", {})
        srcs    = data.get("ddg_source", {})
        n_traj  = data.get("n_trajectories", "?")

        # Build a confidence table (multi-line → shown in a Rich Panel).
        lines: List[str] = [
            f"=== High-Accuracy ddG Validation ({n_traj} trajectories, "
            f"median + MAD spread) ===",
            f"Mutations: {source_note}",
            "",
            f"  {'Mutation':<9} {'ddG(med)':>9} {'spread':>7} {'confidence':<12} {'source'}",
            "  " + "-" * 56,
        ]
        for key in scores:
            ddg = scores.get(key)
            sp  = spreads.get(key)
            cf  = confs.get(key, "?")
            sr  = srcs.get(key, "?")
            ddg_s = f"{ddg:+.3f}" if ddg is not None else "  N/A"
            sp_s  = f"{sp:.2f}" if sp is not None else "  —"
            lines.append(f"  {key:<9} {ddg_s:>9} {sp_s:>7} {cf:<12} {sr}")
        lines.append("")
        for w in data.get("warnings", []) or []:
            lines.append(f"[!] {w}")
        summary = "\n".join(lines)

        return ToolStepResult(
            tool             = "validate_ddg",
            success          = True,
            data             = data,
            viz_commands     = result.viz_commands,
            viz_explanations = result.viz_explanations,
            summary          = summary,
        )

    def _run_assembly_analyser(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run biological assembly analysis (MONOMER or MULTIMER mode)."""
        import time as _time
        t0 = _time.perf_counter()

        model_id = inputs.get("model_id") or self._first_model_id()

        # Guard: when multiple models are loaded (e.g. original structure #1 +
        # ESMFold prediction #2), the translator may target #2 because it is
        # the most recently opened / currently active model.  Interface analysis
        # must always run on the original crystal structure (#1).
        _n_models = len(self.session.structures)
        if _n_models > 1 and model_id != "1" and "1" in self.session.structures:
            print(
                f"  [WARN] Multiple models loaded -- interface analysis targeting #1 "
                f"(original structure), not #{model_id}",
                flush=True,
            )
            model_id = "1"

        mode     = inputs.get("mode", "multimer")
        chain_id = inputs.get("chain_id") or inputs.get("chain")
        visualize= inputs.get("visualize", False)
        dist     = float(inputs.get("contact_distance", 5.0))

        # Determine PDB ID for RCSB assembly query
        pdb_id: Optional[str] = None
        info = self.session.get_structure(model_id)
        if info:
            name = info.get("name", "")
            if re.match(r"^[A-Za-z0-9]{4}$", name):
                pdb_id = name.upper()

        analyser = self._get_assembly_analyser()
        result   = analyser.analyse(
            model_id          = model_id,
            pdb_id            = pdb_id,
            mode              = mode,
            chain_id          = chain_id,
            contact_distance  = dist,
        )

        elapsed_ms = (_time.perf_counter() - t0) * 1000

        # Build summary string
        asm_info   = result.get("assembly_info", {})
        asm_type   = asm_info.get("assembly_type", "unknown")
        n_ifaces   = len(result.get("interfaces", {}))
        n_excluded = result.get("excluded_count", 0)
        mode_str   = result.get("mode", mode)
        header     = result.get("header", "")

        summary_parts = [f"Assembly: {asm_type} [{mode_str} mode]"]
        if mode_str == "multimer":
            summary_parts.append(f"{n_ifaces} interface(s) detected")
            if n_excluded:
                ch_label = chain_id or "chain"
                summary_parts.append(
                    f"{n_excluded} positions excluded from scan (chain {ch_label} interface)"
                )
        summary = " — ".join(summary_parts)

        # Warnings
        warnings_out = result.get("warnings", [])

        # Visualization commands (if requested and interfaces detected)
        viz_cmds: List[str] = []
        viz_exps: List[str] = []
        if visualize and mode_str == "multimer" and result.get("interfaces"):
            viz_cmds, viz_exps = analyser.generate_interface_viz_commands(
                model_id   = model_id,
                interfaces = result["interfaces"],
            )

        return ToolStepResult(
            tool             = "assembly_analyser",
            success          = True,
            data             = {
                "mode":               mode_str,
                "assembly_type":      asm_type,
                "assembly_info":      asm_info,
                "interfaces":         {
                    f"{k[0]}:{k[1]}": v
                    for k, v in result.get("interfaces", {}).items()
                },
                "protected_residues": result.get("protected_residues", []),
                "excluded_count":     n_excluded,
                "header":             header,
                "interface_summary":  result.get("interface_summary", ""),
                "warnings":           warnings_out,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_mutation_scan(self, inputs: Dict[str, Any]) -> ToolStepResult:
        model_id       = inputs.get("model_id") or self._first_model_id()
        chain          = inputs.get("chain", "A")
        focus          = inputs.get("focus", "solubility")
        analysis_mode  = inputs.get("analysis_mode", "monomer")
        sequence       = inputs.get("sequence") or self._fetch_sequence(model_id, chain)
        pdb_path       = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)

        if not sequence:
            return ToolStepResult(
                tool="mutation_scan", success=False,
                error=(
                    "No amino-acid sequence available for mutation scan.\n"
                    "  Load a structure first, or pass a sequence explicitly."
                ),
            )
        if not pdb_path:
            return ToolStepResult(
                tool="mutation_scan", success=False,
                error=(
                    "Mutation scan requires a local PDB file for Rosetta ddG.\n"
                    "  StructureBot will attempt to download from RCSB if the\n"
                    "  structure has a 4-letter PDB ID and internet is available."
                ),
            )

        # Build filters from inputs
        filters: Dict[str, Any] = {}
        for key in ("camsol_threshold", "esm_threshold", "max_candidates",
                    "candidates_per_pos", "binding_site_residues",
                    "w_ddg", "w_sol", "w_tol"):
            if key in inputs:
                filters[key] = inputs[key]
        filters["focus"] = focus

        # In multimer mode, pull interface residues from session state
        protected_residues: List[int] = []
        if analysis_mode == "multimer":
            protected_residues = self.session.get_protected_residues_for_chain(
                model_id, chain
            )
            if not protected_residues:
                # Interface data not yet computed — it should have been run
                # via assembly_analyser first, but we proceed without it
                pass

        from mutation_scanner import MutationScanner

        def _progress(msg: str) -> None:
            pass  # progress shown by main.py via status_callback

        scanner = MutationScanner(
            session           = self.session,
            model_id          = model_id,
            progress_callback = _progress,
        )

        import time as _time
        t0      = _time.perf_counter()
        results = scanner.scan(
            pdb_path           = pdb_path,
            chain_id           = chain,
            sequence           = sequence,
            filters            = filters,
            protected_residues = protected_residues,
            analysis_mode      = analysis_mode,
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not results:
            excluded_note = ""
            if protected_residues:
                excluded_note = (
                    f" ({len(protected_residues)} positions excluded "
                    f"due to interface contacts)"
                )
            return ToolStepResult(
                tool      = "mutation_scan",
                success   = True,
                data      = {"candidates": [], "count": 0,
                             "excluded_count": len(protected_residues)},
                summary   = f"Mutation scan complete — no candidates met the criteria.{excluded_note}",
                elapsed_ms = elapsed_ms,
            )

        # Generate visualization for top 5
        viz_cmds, viz_exps = scanner.generate_chimerax_commands(results, top_n=5)

        top = results[0]
        excluded_note = ""
        if protected_residues:
            excluded_note = (
                f" [{len(protected_residues)} interface position(s) excluded]"
            )

        one_liner = (
            f"Mutation scan [{analysis_mode} mode]: {len(results)} candidate(s) found.{excluded_note} "
            f"Top: {top['from_aa']}{top['position']}{top['to_aa']} "
            f"(score={top['combined_score']:+.2f}, "
            f"ddG={top['ddg']:+.3f} kcal/mol [{top.get('ddg_source', '?')}], "
            f"solubility delta={top['solubility_delta']:+.2f})"
        )

        detailed_summary = scanner._generate_summary(results)

        return ToolStepResult(
            tool             = "mutation_scan",
            success          = True,
            data             = {
                "candidates":    results,
                "count":         len(results),
                "top":           top,
                "excluded_count": len(protected_residues),
                "analysis_mode": analysis_mode,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = detailed_summary,
            elapsed_ms       = elapsed_ms,
        )

    # ── Double mutant tool ────────────────────────────────────────────────────

    def _run_double_mutant(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Score double mutant pairs from existing scan results.

        Reads scan_results from session state, builds the mutations list,
        routes pairs via DynaMut2 prediction_mm (or PyRosetta for close pairs),
        and stores results in session_state.double_mutant_results.
        """
        import time as _time

        user_input = user_input or inputs.get("_user_input", "")
        model_id   = inputs.get("model_id") or self._primary_model_id()
        lower      = user_input.lower()

        # Step 1 — detect mode
        _epitope_kw = ("epitope", "binding", "interface", "preserve", "target")
        mode = "epitope" if any(kw in lower for kw in _epitope_kw) else "stability"
        print(f"[DoubleMutant] Mode: {mode}", flush=True)

        # Step 2 — prerequisites: PDB file
        pdb_path = self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=(
                    "No structure loaded or PDB file unavailable. "
                    "Run 'open 1HSG' (or your structure) first."
                ),
            )

        # Step 2 — prerequisites: scan results
        scan_data = self.session.get_scan_result(model_id)
        if not scan_data:
            # Diagnostic: show which session fields have data to aid debugging
            populated = [
                k for k in vars(self.session)
                if k not in ("session_start", "working_dir")
                and getattr(self.session, k, None)
            ]
            print(
                f"[DoubleMutant] Session state fields with data: {populated}",
                flush=True,
            )
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=(
                    "No single-point scan results found. Run a mutation scan first, e.g.\n"
                    "'suggest mutations to improve solubility of chain A',\n"
                    "then ask for double mutant combinations."
                ),
            )

        # Map scan result dicts → DoubleMutantBridge schema
        # mutation_scanner uses 'solubility_delta'; bridge expects 'camsol_delta'
        mutations: List[Dict[str, Any]] = []
        for m in scan_data:
            mutations.append({
                "chain":             m.get("chain", "A"),
                "position":          m.get("position"),
                "from_aa":           m.get("from_aa"),
                "to_aa":             m.get("to_aa"),
                "ddg":               m.get("ddg", 0.0),
                "camsol_delta":      m.get("solubility_delta") or m.get("camsol_delta", 0.0),
                "esm_tolerance":     m.get("esm_tolerance", 1.0),
                "interface_proximal": m.get("interface_proximal", False),
            })

        if len(mutations) < 2:
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=(
                    "Need at least 2 candidate mutations to generate pairs. "
                    "Run a wider mutation scan first."
                ),
            )

        # Step 3 — detect run_pyrosetta flag
        _pr_kw = ("pyrosetta", "rosetta", "accurate", "high accuracy", "validate")
        run_pyrosetta = any(kw in lower for kw in _pr_kw)
        if run_pyrosetta:
            print(
                "⚠ PyRosetta validation requested for close pairs — this adds ~30 min "
                "per close pair. Close pairs only.",
                flush=True,
            )

        # Step 4 — gather session context
        iface_dict = self.session.get_interface_residues(model_id)
        iface_set: Optional[set] = None
        if iface_dict:
            iface_set = set()
            for resnos in iface_dict.values():
                iface_set.update(resnos)

        func_set = self.session.get_functional_residues()
        func_residues: Optional[set] = func_set if func_set else None

        import config as _cfg_dm
        bridge_inputs: Dict[str, Any] = {
            "pdb_path":            pdb_path,
            "mutations":           mutations,
            "mode":                mode,
            "interface_residues":  iface_set,
            "functional_residues": func_residues,
            "top_n":               _cfg_dm.DOUBLE_MUTANT_TOP_N,
            "run_pyrosetta":       run_pyrosetta,
        }

        bridge = self._get_double_mutant_bridge()
        t0 = _time.perf_counter()
        result = bridge.analyze(bridge_inputs, self.session)
        result.elapsed_ms = (_time.perf_counter() - t0) * 1000

        # Step 5 — handle result
        if not result.success:
            return result

        self.session.set_double_mutant_results(model_id, result.data)

        # Generate ChimeraX visualization
        viz_cmds, viz_exps = self._build_double_mutant_viz(
            result.data.get("top_pairs", []), model_id
        )
        result.viz_commands     = viz_cmds
        result.viz_explanations = viz_exps

        return result

    def _build_double_mutant_viz(
        self,
        top_pairs: List[Dict[str, Any]],
        model_id:  str,
    ) -> tuple:
        """Generate ChimeraX commands to visualize top double mutant pairs."""
        if not top_pairs:
            return [], []

        _CONF_COLORS = {
            "high":     "#6495ed",  # cornflower blue
            "moderate": "#ffd700",  # gold
            "low":      "#c0c0c0",  # light grey
        }

        cmds: List[str] = []
        exps: List[str] = []

        chain = top_pairs[0]["mutation_a"].get("chain", "A")
        cmds.append(f"cartoon #{model_id}")
        exps.append("Switch to cartoon for double mutant visualization")
        cmds.append(f"color #{model_id}/{chain} white")
        exps.append(f"Reset chain {chain} to white before pair coloring")

        for pair in top_pairs[:5]:
            m_a      = pair["mutation_a"]
            m_b      = pair["mutation_b"]
            chain_a  = m_a.get("chain", "A")
            chain_b  = m_b.get("chain", "A")
            pos_a    = m_a["position"]
            pos_b    = m_b["position"]
            conf     = pair.get("confidence", "low")
            color    = _CONF_COLORS.get(conf, "#c0c0c0")
            pair_key = pair.get("pair_key", f"pos{pos_a}+pos{pos_b}")
            epistasis = pair.get("epistasis")
            ep_str   = f" (e={epistasis:+.1f})" if epistasis is not None else ""
            label    = f"{pair_key}{ep_str}"

            spec_a = f"#{model_id}/{chain_a}:{pos_a}"
            spec_b = f"#{model_id}/{chain_b}:{pos_b}"

            cmds.append(f"color {spec_a} {color}")
            exps.append(f"{pair_key}: color residue {pos_a} by confidence ({conf})")
            cmds.append(f"color {spec_b} {color}")
            exps.append(f"{pair_key}: color residue {pos_b} by confidence ({conf})")

            cmds.append(f"show {spec_a} atoms")
            exps.append(f"{pair_key}: show residue {pos_a} as atoms")
            cmds.append(f"style {spec_a} sphere")
            exps.append(f"{pair_key}: sphere style for residue {pos_a}")
            cmds.append(f"show {spec_b} atoms")
            exps.append(f"{pair_key}: show residue {pos_b} as atoms")
            cmds.append(f"style {spec_b} sphere")
            exps.append(f"{pair_key}: sphere style for residue {pos_b}")

            cmds.append(f"distance {spec_a}@CA {spec_b}@CA")
            exps.append(f"Draw Ca-Ca distance line for {pair_key}")

            cmds.append(f'label {spec_a} text "{label}" size 12 color white')
            exps.append(f"Label {pair_key} at residue {pos_a} with pair key and epistasis")

        cmds.append(f"view #{model_id}")
        exps.append("Fit structure in view to show all labeled pairs")

        return cmds, exps

    # ── ColabFold intent detection + option parsing ────────────────────────────

    def _detect_colabfold_intent(self, text: str) -> bool:
        """
        True only for an EXPLICIT high-accuracy folding request:
          * 'colabfold' / 'alphafold' / 'af2' anywhere, OR
          * 'fold … as a <oligomer>' (oligomer word present with a fold verb), OR
          * 'use PDB XXXX as template to fold …' (template + fold).
        Intentionally tight so generic 'fold sequence' / 'fold design' stays with
        the ESMFold / MPNN+ESMFold pipeline.
        """
        if not text:
            return False
        low = text.lower()
        if any(kw in low for kw in self._COLABFOLD_KEYWORDS):
            return True
        _has_fold = "fold" in low or "predict" in low or "structure" in low
        if _has_fold and any(w in low for w in self._OLIGOMER_COPIES):
            return True
        if "template" in low and "fold" in low:
            return True
        return False

    def _parse_colabfold_options(self, text: str) -> Dict[str, Any]:
        """
        Parse copies (oligomer word), an optional template PDB id, an optional
        explicit sequence, and a 'quick' preset flag from the user text.
        """
        opts: Dict[str, Any] = {}
        if not text:
            return opts
        low = text.lower()

        # copies ← first oligomer word found
        for word, n in self._OLIGOMER_COPIES.items():
            if re.search(rf"\b{word}\b", low):
                opts["copies"] = n
                break

        # template ← "use PDB XXXX as template" / "template XXXX" (4-char id)
        m = re.search(
            r"(?:pdb\s+)?\b([0-9][a-z0-9]{3})\b\s+as\s+(?:a\s+)?template", low
        ) or re.search(r"template[:\s]+(?:pdb\s+)?\b([0-9][a-z0-9]{3})\b", low)
        if m:
            opts["template"] = m.group(1).upper()

        # quick preset
        if "quick" in low or "fast fold" in low or "rough fold" in low:
            opts["quick"] = True

        # explicit sequence ← a long run of standard one-letter codes (>=20 aa)
        m = re.search(r"\b([ACDEFGHIKLMNPQRSTVWY]{20,})\b", text.upper())
        if m:
            opts["sequence"] = m.group(1)

        return opts

    # ── ColabFold dispatch ──────────────────────────────────────────────────────

    def _run_colabfold(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> "ToolStepResult":
        """
        Run AF2 structure prediction via the WSL2 ColabFold env and build viz.

        Resolves the sequence (explicit input/parsed sequence, else the loaded
        structure's chain — MPNN-result auto-pull is DEFERRED), folds via
        ColabFoldBridge, stores a trimmed result in session_state.colabfold_results,
        and builds ChimeraX confidence-map viz (open ranked PDB → native AlphaFold
        pLDDT palette → Sequence Viewer → optional matchmaker RMSD vs compare_to).
        Opens the PAE/pLDDT/coverage PNGs in the OS image viewer (best-effort).
        """
        import time as _time

        user_input = user_input or inputs.get("_user_input", "")
        model_id   = inputs.get("model_id") or self._first_model_id()
        copies     = int(inputs.get("copies", 1) or 1)
        template   = inputs.get("template")
        quick      = bool(inputs.get("quick", False))

        # ── Resolve sequence ──────────────────────────────────────────────────────
        sequence = inputs.get("sequence") or self._fetch_sequence(
            model_id, inputs.get("chain")
        )
        if not sequence:
            return ToolStepResult(
                tool="colabfold", success=False,
                error=(
                    "ColabFold needs an amino-acid sequence. Provide one explicitly "
                    "(e.g. 'fold MKT... as a dimer with colabfold'), or load a "
                    "structure first so its chain sequence can be used."
                ),
            )

        # ── Resolve optional template (PDB id → local file) ────────────────────────
        template_path: Optional[str] = None
        if template:
            if Path(str(template)).is_file():
                template_path = str(template)
            else:
                template_path = self._download_pdb_by_id(str(template))
                if not template_path:
                    return ToolStepResult(
                        tool="colabfold", success=False,
                        error=(
                            f"Could not obtain template '{template}'. Provide a 4-char "
                            "PDB id (downloadable from RCSB) or a local .pdb path."
                        ),
                    )

        bridge = self._get_colabfold_bridge()

        # num_models/num_recycle default to config inside the bridge when None;
        # quick=True overrides both to 1/1 there.
        t0 = _time.perf_counter()
        result = bridge.predict(
            sequence=sequence, copies=copies, template=template_path,
            num_models=None, num_recycle=None, quick=quick,
            label=f"model{model_id}",
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="colabfold", success=False,
                error=result.get("error", "ColabFold prediction failed"),
                data={"oom_risk": result.get("oom_risk", False)},
                elapsed_ms=elapsed_ms,
            )

        # ── Store a trimmed result in session (omit the heavy PAE matrix) ──────────
        trimmed = {
            k: result[k] for k in (
                "ranked_pdb", "mean_plddt", "ptm", "iptm", "length", "copies",
                "total_residues", "num_models", "num_recycle", "png_paths",
                "cached", "source",
            ) if k in result
        }
        self.session.set_colabfold_results(model_id, trimmed)

        # ── Open the confidence-map PNGs in the OS viewer (best-effort) ─────────────
        self._open_pngs(result.get("png_paths", {}))

        # ── Build ChimeraX viz (open ranked PDB → pLDDT palette → seq → matchmaker) ─
        viz_cmds, viz_exps = self._build_colabfold_viz(result, inputs, model_id)

        mean_plddt = result["mean_plddt"]
        conf = "very high" if mean_plddt > 90 else "high" if mean_plddt > 70 else \
               "low" if mean_plddt > 50 else "very low"
        ptm = result.get("ptm")
        iptm = result.get("iptm")
        _shape = "monomer" if result["copies"] <= 1 else f"{result['copies']}-mer"
        summary = (
            f"ColabFold ({_shape}): mean pLDDT {mean_plddt:.1f} ({conf} confidence)"
            + (f", pTM {ptm:.2f}" if isinstance(ptm, (int, float)) else "")
            + (f", ipTM {iptm:.2f}" if isinstance(iptm, (int, float)) else "")
            + (" [cached]" if result.get("cached") else "")
            + f".\n  Ranked model: {result.get('ranked_pdb', '?')}"
        )
        return ToolStepResult(
            tool="colabfold", success=True,
            data=result,
            viz_commands=viz_cmds,
            viz_explanations=viz_exps,
            summary=summary,
            elapsed_ms=elapsed_ms,
        )

    def _build_colabfold_viz(
        self,
        result:   Dict[str, Any],
        inputs:   Dict[str, Any],
        model_id: str,
    ) -> tuple:
        """
        ChimeraX commands: open the ranked PDB as a NEW model, colour it by the
        native AlphaFold pLDDT palette (B-factor holds pLDDT), open the Sequence
        Viewer, and — when a compare_to structure resolves — superpose with
        matchmaker. The new model id is taken from session.next_model_id(), which
        is exactly what main.py assigns to the open command it later state-tracks.
        """
        ranked = result.get("ranked_pdb", "")
        if not ranked:
            return [], []

        new_id = str(self.session.next_model_id())
        pdb_posix = Path(ranked).as_posix()

        cmds: List[str] = []
        exps: List[str] = []

        cmds.append(f'open "{pdb_posix}"')
        exps.append(f"Open the ColabFold ranked model as #{new_id}")
        # Native AlphaFold pLDDT colouring (canonical blue→orange palette over the
        # B-factor column, where ColabFold stores per-residue pLDDT).
        cmds.append(f"color byattribute bfactor #{new_id} palette alphafold")
        exps.append("Colour by pLDDT using the native AlphaFold palette (blue=confident)")
        cmds.append(f"cartoon #{new_id}")
        exps.append("Cartoon representation for the predicted model")
        cmds.append(f"sequence chain #{new_id}")
        exps.append("Open the Sequence Viewer for the predicted chain(s)")

        # ── Optional structural comparison (matchmaker RMSD) ────────────────────────
        ref_spec = self._resolve_colabfold_compare_to(inputs, exclude_model=new_id)
        if ref_spec:
            cmds.append(f"matchmaker #{new_id} to {ref_spec}")
            exps.append(
                f"Superpose the predicted model onto {ref_spec} (matchmaker RMSD "
                "reported in ChimeraX log)"
            )
        cmds.append(f"view #{new_id}")
        exps.append("Fit the predicted model in view")
        return cmds, exps

    def _resolve_colabfold_compare_to(
        self,
        inputs:        Dict[str, Any],
        exclude_model: str,
    ) -> Optional[str]:
        """
        Resolve the compare_to reference for matchmaker.

        Default: the currently-loaded primary model (if any), as a ``#id`` spec,
        with optional chain. Overridable via inputs['compare_to'] = a ``#id``/
        ``#id/chain`` spec or a model id. Returns None when nothing to compare to
        (e.g. the predicted model is the only structure).
        """
        compare_to = inputs.get("compare_to")
        if compare_to:
            spec = str(compare_to).strip()
            return spec if spec.startswith("#") else f"#{spec}"
        # Default = the existing primary model, if one is loaded and it isn't the
        # model we just predicted.
        try:
            primary = self._primary_model_id()
        except Exception:
            primary = None
        if primary and str(primary) != str(exclude_model) and self.session.get_structure(primary):
            chain = inputs.get("chain")
            return f"#{primary}/{chain}" if chain else f"#{primary}"
        return None

    def _download_pdb_by_id(self, pdb_id: str) -> Optional[str]:
        """Download a 4-char PDB id from RCSB into cache/ and return the local path."""
        if not re.match(r"^[A-Za-z0-9]{4}$", pdb_id):
            return None
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / f"{pdb_id.upper()}.pdb"
        if local.is_file():
            return str(local)
        try:
            import requests
            resp = requests.get(
                f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30
            )
            resp.raise_for_status()
            local.write_bytes(resp.content)
            return str(local)
        except Exception:
            return None

    @staticmethod
    def _open_pngs(png_paths: Dict[str, str]) -> None:
        """Open the confidence-map PNGs in the OS default image viewer (best-effort)."""
        import os as _os
        import sys as _sys
        import subprocess as _sub
        for _kind, _path in (png_paths or {}).items():
            if not _path or not Path(_path).is_file():
                continue
            try:
                if _sys.platform == "win32":
                    _os.startfile(_path)             # type: ignore[attr-defined]
                elif _sys.platform == "darwin":
                    _sub.Popen(["open", _path])
                else:
                    _sub.Popen(["xdg-open", _path])
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # Validate-design meta-tool (thin orchestrator — NOT a new bridge)
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_validate_design_intent(self, text: str) -> bool:
        """
        True ONLY for explicit high-accuracy validation phrasing:
          * an explicit qualifier (full / high-accuracy / high-confidence /
            thorough validation, or '<qualifier> validate'), OR
          * a 'validate ... with colabfold/alphafold' request.
        Bare 'validate design' / 'check this design' return False (they revert to
        the light mpnn_esmfold / ESMFold fast screen).
        """
        if not text:
            return False
        low = text.lower()
        if any(kw in low for kw in self._VALIDATE_DESIGN_KEYWORDS):
            return True
        # "validate ... with colabfold/alphafold" → explicit high-accuracy path.
        if "validat" in low and ("colabfold" in low or "alphafold" in low):
            return True
        return False

    def _parse_validate_design_options(self, text: str) -> Dict[str, Any]:
        """
        Parse the RMSD reference (fold preservation) and the energy reference
        (relative stability) — they are SEPARATE. Cross-topology is fine for the
        RMSD reference (matchmaker on the selected chain); the energy reference is
        explicit and only ever scored relative when topologies match.
        """
        opts: Dict[str, Any] = {}
        if not text:
            return opts
        low = text.lower()
        # RMSD reference: "vs / against / compare(d) to / relative to PDB XXXX [chain Y]"
        m = re.search(
            r"(?:vs\.?|versus|against|compared?\s+to|relative\s+to|compare\s+with|"
            r"superpose\s+(?:on|onto))\s+(?:pdb\s+)?([0-9][a-z0-9]{3})"
            r"(?:\s+chain\s+([a-z0-9]))?",
            low,
        )
        if m:
            opts["rmsd_ref"] = {
                "pdb": m.group(1).upper(),
                "chain": (m.group(2).upper() if m.group(2) else None),
            }
        # Explicit energy reference → implies a relative-stability request.
        me = re.search(r"energy\s+reference\s+(?:pdb\s+)?([0-9][a-z0-9]{3})", low)
        if me:
            opts["energy_ref"] = me.group(1).upper()
            opts["requested_relative"] = True
        # Relative-stability intent without an explicit ref id (decision will
        # still DECLINE unless a same-topology energy ref is actually resolved).
        if (("relative" in low and ("energy" in low or "stability" in low))
                or "energy delta" in low or "stability delta" in low
                or "compare energy" in low or "compare the energy" in low):
            opts["requested_relative"] = True
        return opts

    @staticmethod
    def _design_energy_decision(
        design_topology: Optional[tuple],
        ref_topology:    Optional[tuple],
        requested_relative: bool,
        ref_name: str = "the energy reference",
    ) -> Dict[str, Any]:
        """
        PURE honesty logic for the folding-energy axis (no I/O — unit-testable).

        Topology = ``(n_chains, tuple(sorted(per_chain_lengths)))``.

        Returns ``{"mode": ..., "reason": ...}`` where mode is:
          * "sanity"   — no relative number; report fold-plausibility only.
          * "relative" — topologies MATCH and a relative score was requested.
          * "declined" — relative was requested but topologies DON'T match;
                         NO relative number is emitted and a reason is given.
        """
        if not requested_relative or ref_topology is None:
            return {
                "mode": "sanity",
                "reason": (
                    "No explicit same-topology energy reference — the relaxed ref2015 "
                    "energy is reported as a fold-PLAUSIBILITY sanity signal (total REU + "
                    "per-residue density + clash check), NOT a stability-vs-reference claim."
                ),
            }
        if design_topology is not None and design_topology == ref_topology:
            return {
                "mode": "relative",
                "reason": (
                    f"Topologies match (same chain count and per-chain lengths) — relative "
                    f"ΔREU vs {ref_name} reported. Ranking-reliable; absolute magnitude "
                    f"approximate (~±2.7 kcal/mol, see PROJECT_CONTEXT §7 calibration verdict)."
                ),
            }
        d_n, d_lens = design_topology if design_topology else (0, ())
        r_n, r_lens = ref_topology
        return {
            "mode": "declined",
            "reason": (
                f"Relative stability vs {ref_name} isn't meaningful — {ref_name} is a "
                f"{r_n}-mer (chain lengths {list(r_lens)}), the design is a {d_n}-mer "
                f"(chain lengths {list(d_lens)}). ref2015 totals across different topologies "
                f"are not comparable; the per-residue energy density is given for the design "
                f"as a sanity signal instead."
            ),
        }

    @staticmethod
    def _topology_from_fold(fold: Dict[str, Any]) -> Optional[tuple]:
        """(n_chains, sorted per-chain lengths) for a ColabFold result (homo-oligomer)."""
        copies = int(fold.get("copies", 1) or 1)
        length = fold.get("length")
        if not length:
            plddt = fold.get("plddt") or {}
            length = len(plddt) // copies if plddt and copies else (len(plddt) or 0)
        if not length:
            return None
        return (copies, tuple([int(length)] * copies))

    @staticmethod
    def _topology_from_pdb(pdb_path: Optional[str]) -> Optional[tuple]:
        """(n_chains, sorted per-chain CA counts) parsed from a PDB file, or None."""
        if not pdb_path or not Path(pdb_path).is_file():
            return None
        per_chain: Dict[str, int] = {}
        seen: set = set()
        try:
            with open(pdb_path, errors="replace") as fh:
                for line in fh:
                    if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                        ch  = line[21]
                        key = (ch, line[22:27])
                        if key in seen:
                            continue
                        seen.add(key)
                        per_chain[ch] = per_chain.get(ch, 0) + 1
        except Exception:
            return None
        if not per_chain:
            return None
        lengths = sorted(per_chain.values())
        return (len(lengths), tuple(lengths))

    def _acquire_design_fold(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get the design's ColabFold fold, REUSING an existing result instead of
        re-folding when possible (guardrail). Priority:
          1. an explicit ``colabfold_result`` dict (chaining / tests);
          2. an in-session fold for this model (no re-fold) — enriched from the
             on-disk full result.json when present;
          3. fold the given sequence via the bridge (whose hash-cache itself
             reuses a prior fold → only folds if there genuinely isn't one).
        """
        if isinstance(inputs.get("colabfold_result"), dict):
            r = dict(inputs["colabfold_result"])
            r.setdefault("success", True)
            r.setdefault("fold_source", "provided")
            return r

        model_id = inputs.get("model_id") or self._first_model_id()
        sequence = inputs.get("sequence")

        if not sequence:
            sess = self.session.get_colabfold_results(model_id) if self.session else None
            if sess and sess.get("ranked_pdb"):
                r = dict(sess)
                r["success"] = True
                r["fold_source"] = "reused (session)"
                self._enrich_fold_from_disk(r)
                return r
            return {
                "success": False,
                "error": (
                    "No sequence provided and no in-session ColabFold result for this "
                    "model. Fold a sequence first (e.g. 'fold <seq> with colabfold'), or "
                    "give a sequence to validate."
                ),
            }

        bridge = self._get_colabfold_bridge()
        r = dict(bridge.predict(
            sequence=sequence, copies=int(inputs.get("copies", 1) or 1),
            template=inputs.get("template"), quick=bool(inputs.get("quick", False)),
            label=f"design{model_id}",
        ))
        r["fold_source"] = "reused (cache)" if r.get("cached") else "folded"
        return r

    @staticmethod
    def _enrich_fold_from_disk(r: Dict[str, Any]) -> None:
        """Backfill pae/per-residue pLDDT/pTM from the on-disk full result.json
        beside the ranked PDB (the session copy is trimmed). Best-effort."""
        try:
            rp = r.get("ranked_pdb")
            if not rp:
                return
            meta = Path(rp).parent / "result.json"
            if not meta.is_file():
                return
            import json as _j
            full = _j.loads(meta.read_text(encoding="utf-8"))
            for k in ("pae", "plddt", "ptm", "iptm", "mean_plddt"):
                if r.get(k) in (None, {}, []) and full.get(k) is not None:
                    r[k] = full[k]
        except Exception:
            pass

    @staticmethod
    def _parse_model_spec(resp: dict, fallback_id: Optional[str]) -> Optional[str]:
        """Parse '#N' from a ChimeraX open response; fall back to '#fallback_id'."""
        val = (resp or {}).get("value")
        if isinstance(val, str):
            m = re.search(r"#(\d+)", val)
            if m:
                return f"#{m.group(1)}"
        return f"#{fallback_id}" if fallback_id else None

    # ── Fold-preservation helpers (pure — unit-testable) ────────────────────────

    @staticmethod
    def _parse_matchmaker_rmsds(val: str) -> Optional[Dict[str, Any]]:
        """
        Parse BOTH RMSDs + pair counts from a ChimeraX matchmaker response:
          "RMSD between N pruned atom pairs is X angstroms; (across all M pairs: Y)"
        Returns {pruned_n, pruned_rmsd, total_n, all_pairs_rmsd}. If the
        "(across all M pairs: Y)" clause is absent (no pruning occurred), falls
        back to all-pairs == pruned (total_n = pruned_n, all_pairs_rmsd = pruned).
        Returns None if no RMSD could be parsed at all.
        """
        if not val:
            return None
        mp = re.search(r"RMSD between\s+(\d+)\s+(?:pruned\s+)?atom pairs is\s+([\d.]+)", val)
        if not mp:
            mf = re.search(r"RMSD[^0-9]*([\d.]+)\s*angstrom", val, re.IGNORECASE)
            if not mf:
                return None
            x = round(float(mf.group(1)), 3)
            return {"pruned_n": None, "pruned_rmsd": x, "total_n": None, "all_pairs_rmsd": x}
        pruned_n   = int(mp.group(1))
        pruned_rmsd = round(float(mp.group(2)), 3)
        ma = re.search(r"across all\s+(\d+)\s+pairs?:?\s*([\d.]+)", val)
        if ma:
            total_n = int(ma.group(1))
            all_pairs = round(float(ma.group(2)), 3)
        else:
            total_n = pruned_n
            all_pairs = pruned_rmsd
        return {"pruned_n": pruned_n, "pruned_rmsd": pruned_rmsd,
                "total_n": total_n, "all_pairs_rmsd": all_pairs}

    @staticmethod
    def _concentration_descriptor(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Free variance-concentration read from the matchmaker numbers. matchmaker's
        pruning already isolates the divergent subsection, so a big all-pairs−pruned
        gap from FEW pruned residues = "a lot of variation from a small subsection".
        Descriptive only — NO threshold pass/fail.
        """
        allp   = parsed.get("all_pairs_rmsd")
        pruned = parsed.get("pruned_rmsd")
        n      = parsed.get("pruned_n")     # core (kept) pairs
        m      = parsed.get("total_n")      # all pairs
        out: Dict[str, Any] = {
            "gap": None, "pruned_count": n, "total_count": m,
            "pruned_fraction": None, "descriptor": None,
        }
        if allp is None or pruned is None:
            out["descriptor"] = "single RMSD only (no pruned/all-pairs split available)."
            return out
        gap = round(allp - pruned, 3)
        out["gap"] = gap
        frac = None
        if n is not None and m and m > 0:
            frac = round((m - n) / m, 3)        # fraction of residues pruned out
            out["pruned_fraction"] = frac
        # Descriptive read.
        if gap < 0.3:
            out["descriptor"] = (
                f"divergence is low/uniform — all-pairs {allp:.2f} Å vs core {pruned:.2f} Å "
                f"(gap {gap:.2f} Å); the fit is even across the chain."
            )
        elif frac is not None and frac <= 0.25:
            pct = round(frac * 100)
            out["descriptor"] = (
                f"divergence is CONCENTRATED — all-pairs {allp:.2f} Å but core {pruned:.2f} Å; "
                f"~{pct}% of residues ({m - n}/{m}) drive most of the gap ({gap:.2f} Å), "
                f"i.e. a localized over-represented subsection."
            )
        else:
            out["descriptor"] = (
                f"broadly divergent — all-pairs {allp:.2f} Å vs core {pruned:.2f} Å "
                f"(gap {gap:.2f} Å) with many residues pruned"
                + (f" ({m - n}/{m})" if (n is not None and m) else "")
                + "; not a single localized pocket."
            )
        return out

    @staticmethod
    def _ca_coords(pdb_path: str, chain: Optional[str]) -> Dict[int, "Any"]:
        """{resno: (x,y,z)} for Cα atoms of *chain* (all chains if None). First
        occurrence per (chain,resno) wins (skips altlocs/duplicates)."""
        import numpy as _np
        out: Dict[int, Any] = {}
        seen: set = set()
        try:
            with open(pdb_path, errors="replace") as fh:
                for line in fh:
                    if not line.startswith(("ATOM", "HETATM")):
                        continue
                    if line[12:16].strip() != "CA":
                        continue
                    ch = line[21]
                    if chain and ch != chain:
                        continue
                    try:
                        resno = int(line[22:26])
                        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    except ValueError:
                        continue
                    key = (ch, resno)
                    if key in seen:
                        continue
                    seen.add(key)
                    out[resno] = _np.array(xyz, dtype=float)
        except Exception:
            return {}
        return out

    @classmethod
    def _per_residue_ca_deviation(
        cls,
        design_pdb:   str,
        design_chain: Optional[str],
        ref_pdb:      str,
        ref_chain:    Optional[str],
    ) -> Tuple[Optional[Dict[int, float]], Optional[float]]:
        """
        Per-residue Cα deviation (Å) of the design vs reference, over an
        independent Kabsch superposition of matched Cα (matched by residue
        number). Returns ({design_resno: deviation}, all_pairs_rmsd) or
        (None, None). The matchmaker superposition isn't exposed over REST, so
        this is an independent Cα fit — adequate for localizing the divergence.
        """
        import numpy as _np
        d = cls._ca_coords(design_pdb, design_chain)
        r = cls._ca_coords(ref_pdb, ref_chain)
        common = sorted(set(d) & set(r))
        if len(common) < 3:
            return None, None
        P = _np.array([d[i] for i in common])   # design
        Q = _np.array([r[i] for i in common])   # reference
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        # Kabsch: rotation fitting P onto Q.
        H = Pc.T @ Qc
        U, _S, Vt = _np.linalg.svd(H)
        dsign = _np.sign(_np.linalg.det(Vt.T @ U.T))
        D = _np.diag([1.0, 1.0, dsign])
        R = Vt.T @ D @ U.T
        P_fit = Pc @ R.T
        diff = P_fit - Qc
        per_res = _np.sqrt((diff ** 2).sum(1))
        dev = {resno: round(float(per_res[i]), 3) for i, resno in enumerate(common)}
        all_pairs = round(float(_np.sqrt((per_res ** 2).mean())), 3)
        return dev, all_pairs

    @staticmethod
    def _topk_deviant(dev: Dict[int, float], chain: Optional[str], k: int = 10) -> List[Dict[str, Any]]:
        """Top-K most-deviant residues (chain, resno, deviation Å), descending."""
        if not dev:
            return []
        top = sorted(dev.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [{"chain": chain or "?", "resno": rn, "deviation": dv} for rn, dv in top]

    # Deviation → colour buckets (blue=low … red=high). Documented scale; >=3.5 Å
    # caps at red so a few large outliers don't wash out the gradient.
    _DEVIATION_BUCKETS = (
        (0.5, "blue"),
        (1.0, "cornflower blue"),
        (2.0, "white"),
        (3.5, "orange"),
        (float("inf"), "red"),
    )

    @classmethod
    def _build_deviation_color_cmds(
        cls,
        dev:         Dict[int, float],
        design_spec: str,
        chain:       Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Colour the design model by per-residue Cα deviation, reusing the CamSol
        grouped-run idiom (consecutive same-colour residues → one `color` command).
        blue=low … red=high (>=3.5 Å). design_spec is the live '#N' model spec.
        """
        if not dev:
            return [], []
        def _bucket(v: float) -> str:
            for hi, col in cls._DEVIATION_BUCKETS:
                if v < hi:
                    return col
            return "red"
        chain_spec = f"/{chain}" if chain else ""
        base = design_spec if "/" in design_spec else f"{design_spec}{chain_spec}"
        cmds = [f"color {base} white"]
        exps = ["Reset to white before per-residue deviation colouring"]
        runs: List[Tuple[str, List[int]]] = []
        for resno in sorted(dev):
            col = _bucket(dev[resno])
            if runs and runs[-1][0] == col:
                runs[-1][1].append(resno)
            else:
                runs.append((col, [resno]))
        for col, resnos in runs:
            if col == "white":
                continue
            if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
                spec = f":{resnos[0]}-{resnos[-1]}"
            else:
                spec = ":" + ",".join(str(r) for r in resnos)
            cmds.append(f"color {design_spec}{spec} {col}")
            exps.append(f"Colour {spec} {col} (Cα deviation bucket)")
        return cmds, exps

    def _matchmaker_rmsd_live(
        self,
        fold:   Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Open the predicted model, apply ColabFold-style pLDDT viz, and superpose
        it onto the RMSD reference with matchmaker; parse and return Cα RMSD.

        RMSD reference: default = the loaded session primary model; overridable to
        a named PDB with chain selection (cross-topology is fine here). Degrades
        gracefully (rmsd=None + note) when ChimeraX or a reference is unavailable.
        Runs live during execute (the RMSD value needs a live run); the issued
        commands are returned for transparency.
        """
        out: Dict[str, Any] = {"rmsd_ca": None, "reference": None, "commands": [], "note": None}
        ranked = fold.get("ranked_pdb")
        if self.bridge is None:
            out["note"] = "ChimeraX bridge unavailable — RMSD skipped."
            return out
        if not ranked or not Path(str(ranked)).is_file():
            out["note"] = "No ranked PDB available — RMSD skipped."
            return out

        cmds: List[str] = []
        try:
            # Open + colour the predicted model (ColabFold viz, reused).
            design_guess = str(self.session.next_model_id()) if self.session else None
            open_cmd = f'open "{Path(ranked).as_posix()}"'
            r = self.bridge.run_command(open_cmd)
            cmds.append(open_cmd)
            design_spec = self._parse_model_spec(r, design_guess)
            if design_spec:
                for c in (f"color byattribute bfactor {design_spec} palette alphafold",
                          f"sequence chain {design_spec}"):
                    self.bridge.run_command(c)
                    cmds.append(c)

            # Resolve the reference spec AND a LOCAL reference PDB path (the
            # latter for the offline per-residue Cα Kabsch).
            rmsd_ref      = inputs.get("rmsd_ref")
            ref_spec      = None
            ref_pdb_path  = None
            ref_chain     = None
            design_chain  = inputs.get("design_chain", "A")  # ColabFold monomer → chain A
            if rmsd_ref and rmsd_ref.get("pdb"):
                ref_pdb_path = self._download_pdb_by_id(rmsd_ref["pdb"])
                if not ref_pdb_path:
                    out["note"] = f"Could not fetch RMSD reference {rmsd_ref['pdb']}; RMSD skipped."
                    out["commands"] = cmds
                    return out
                ropen = f'open "{Path(ref_pdb_path).as_posix()}"'
                rr = self.bridge.run_command(ropen)
                cmds.append(ropen)
                ref_model = self._parse_model_spec(rr, None)
                ref_chain = rmsd_ref.get("chain")
                ref_spec = (f"{ref_model}/{ref_chain}" if (ref_model and ref_chain) else ref_model)
                out["reference"] = (
                    f"{rmsd_ref['pdb']}" + (f" chain {ref_chain}" if ref_chain else "")
                )
            else:
                # Default: the loaded primary model (assumed already open).
                primary = self._primary_model_id()
                if primary and design_spec and str(primary) != design_spec.lstrip("#") \
                        and self.session and self.session.get_structure(primary):
                    ref_chain = inputs.get("chain")
                    ref_spec = f"#{primary}/{ref_chain}" if ref_chain else f"#{primary}"
                    out["reference"] = f"loaded model #{primary}" + (f" chain {ref_chain}" if ref_chain else "")
                    ref_pdb_path = self._ensure_pdb_file(primary)
                else:
                    out["note"] = (
                        "No RMSD reference (no loaded structure and none specified) — "
                        "fold preservation not assessed."
                    )

            if design_spec and ref_spec:
                mm = f"matchmaker {design_spec} to {ref_spec}"
                mr = self.bridge.run_command(mm)
                cmds.append(mm)
                val = (mr or {}).get("value") or ""
                parsed = self._parse_matchmaker_rmsds(val)
                if parsed:
                    # HEADLINE = all-pairs (honest "did the fold drift overall").
                    out["rmsd_ca"]        = parsed["all_pairs_rmsd"]
                    out["all_pairs_rmsd"] = parsed["all_pairs_rmsd"]
                    out["pruned_rmsd"]    = parsed["pruned_rmsd"]
                    out["pruned_n"]       = parsed["pruned_n"]
                    out["total_n"]        = parsed["total_n"]
                    conc = self._concentration_descriptor(parsed)
                    out["gap"]             = conc["gap"]
                    out["pruned_fraction"] = conc["pruned_fraction"]
                    out["concentration"]   = conc["descriptor"]
                else:
                    out["note"] = "matchmaker ran but RMSD could not be parsed from the response."

                # ── Localize: per-residue Cα deviation + colour the design model ──
                if ref_pdb_path and Path(str(ref_pdb_path)).is_file():
                    dev, dev_allpairs = self._per_residue_ca_deviation(
                        ranked, design_chain, ref_pdb_path, ref_chain
                    )
                    if dev:
                        out["per_residue_deviation"] = dev
                        out["per_residue_all_pairs_rmsd"] = dev_allpairs
                        out["top_deviant_residues"] = self._topk_deviant(dev, design_chain, k=10)
                        ccmds, _cexp = self._build_deviation_color_cmds(
                            dev, design_spec, design_chain
                        )
                        for c in ccmds:
                            self.bridge.run_command(c)
                            cmds.append(c)
                        out["colored_by_deviation"] = bool(ccmds)
                    else:
                        out["per_residue_note"] = (
                            "per-residue deviation unavailable (could not match >=3 Cα "
                            "between design and reference chains)."
                        )
                else:
                    out["per_residue_note"] = (
                        "per-residue deviation skipped (no local reference PDB)."
                    )

                self.bridge.run_command(f"view {design_spec}")
                cmds.append(f"view {design_spec}")
        except Exception as exc:
            out["note"] = f"RMSD step error: {exc}"
        out["commands"] = cmds
        return out

    def _run_validate_design(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> "ToolStepResult":
        """
        Thin orchestrator: fold confidence (ColabFold, reused if present) +
        fold preservation (matchmaker RMSD) + folding energy (Rosetta relax +
        ref2015, HONEST: sanity by default, relative only on same-topology, decline
        cross-topology). EVIDENCE-RICH report; no binary verdict. Stored in
        session_state.validate_design_results.
        """
        import time as _time
        user_input = user_input or inputs.get("_user_input", "")
        model_id   = inputs.get("model_id") or self._first_model_id()
        t0 = _time.perf_counter()

        # ── 1. Fold confidence (reuse-or-fold) ────────────────────────────────────
        fold = self._acquire_design_fold(inputs)
        if not fold.get("success"):
            return ToolStepResult(
                tool="validate_design", success=False,
                error=fold.get("error", "could not obtain a fold for the design"),
                data={"oom_risk": fold.get("oom_risk", False)},
            )

        plddt_map  = fold.get("plddt") or {}
        mean_plddt = fold.get("mean_plddt") or (
            round(sum(plddt_map.values()) / len(plddt_map), 2) if plddt_map else None
        )
        fold_flags: List[str] = []
        if isinstance(mean_plddt, (int, float)) and mean_plddt < 70:
            fold_flags.append(f"low mean pLDDT ({mean_plddt:.1f} < 70) — low-confidence fold")
        fold_conf = {
            "label":            "AF2 fold confidence (ColabFold)",
            "fold_source":      fold.get("fold_source"),
            "mean_plddt":       mean_plddt,
            "per_residue_plddt": plddt_map or None,
            "pae_available":    fold.get("pae") is not None,
            "ptm":              fold.get("ptm"),
            "iptm":             fold.get("iptm"),
            "ranked_pdb":       fold.get("ranked_pdb"),
            "flags":            fold_flags,
        }

        # ── 2. Fold preservation (matchmaker all-pairs RMSD + concentration + where) ─
        rmsd = self._matchmaker_rmsd_live(fold, inputs)
        rmsd_flags: List[str] = []
        if isinstance(rmsd.get("rmsd_ca"), (int, float)) and rmsd["rmsd_ca"] > 4.0:
            rmsd_flags.append(
                f"high all-pairs Cα RMSD ({rmsd['rmsd_ca']:.2f} Å > 4 Å) — fold may not be preserved"
            )
        _gap = rmsd.get("gap")
        _frac = rmsd.get("pruned_fraction")
        if isinstance(_gap, (int, float)) and _gap >= 0.3 and isinstance(_frac, (int, float)) and _frac <= 0.25:
            rmsd_flags.append(
                f"divergence concentrated in ~{round(_frac*100)}% of residues "
                f"(all-pairs−core gap {_gap:.2f} Å)"
            )
        fold_pres = {
            "label":               "Fold preservation — matchmaker Cα RMSD vs reference (headline = all-pairs)",
            "rmsd_ca":             rmsd.get("rmsd_ca"),            # all-pairs (headline)
            "all_pairs_rmsd":      rmsd.get("all_pairs_rmsd"),
            "pruned_rmsd":         rmsd.get("pruned_rmsd"),        # core (pruned) — flatters agreement
            "pruned_n":            rmsd.get("pruned_n"),
            "total_n":             rmsd.get("total_n"),
            "pruned_fraction":     rmsd.get("pruned_fraction"),
            "gap":                 rmsd.get("gap"),
            "concentration":       rmsd.get("concentration"),
            "top_deviant_residues": rmsd.get("top_deviant_residues"),
            "per_residue_deviation": rmsd.get("per_residue_deviation"),
            "colored_by_deviation": rmsd.get("colored_by_deviation", False),
            "reference":           rmsd.get("reference"),
            "note":                rmsd.get("note") or rmsd.get("per_residue_note"),
            "flags":               rmsd_flags,
        }

        # ── 3. Folding energy (HONEST) ────────────────────────────────────────────
        rosetta = self._get_rosetta_bridge()
        energy_flags: List[str] = []
        energy: Dict[str, Any] = {
            "label": "Folding energy (Rosetta FastRelax + ref2015)",
            "mode":  "sanity",
        }
        ranked_pdb = fold.get("ranked_pdb")
        score = rosetta.relax_and_score(ranked_pdb) if ranked_pdb else {"success": False, "error": "no ranked PDB"}

        # Decide sanity / relative / declined (pure logic).
        requested_relative = bool(inputs.get("requested_relative", False))
        energy_ref = inputs.get("energy_ref")
        ref_topo = None
        ref_name = str(energy_ref) if energy_ref else "the energy reference"
        energy_ref_pdb = None
        if requested_relative and energy_ref:
            energy_ref_pdb = (energy_ref if Path(str(energy_ref)).is_file()
                              else self._download_pdb_by_id(str(energy_ref)))
            ref_topo = self._topology_from_pdb(energy_ref_pdb)
        decision = self._design_energy_decision(
            self._topology_from_fold(fold), ref_topo, requested_relative, ref_name
        )
        energy["mode"]   = decision["mode"]
        energy["reason"] = decision["reason"]

        if not score.get("success"):
            energy["available"] = False
            energy["error"]     = score.get("error")
            energy_flags.append("Rosetta relax/score unavailable — folding-energy axis not assessed")
        else:
            energy.update({
                "available":           True,
                "total_reu":           score.get("total_reu"),
                "per_residue_density": score.get("per_residue_density"),
                "fa_rep":              score.get("fa_rep"),
                "clash_ok":            score.get("clash_ok"),
                "converged":           score.get("converged"),
                "relaxed_pdb":         score.get("relaxed_pdb"),
            })
            if score.get("clash_ok") is False:
                energy_flags.append("post-relax repulsive energy is high — possible clashes / strained packing")
            # RELATIVE number ONLY on the relative path (never on declined/sanity).
            if decision["mode"] == "relative" and energy_ref_pdb:
                ref_score = rosetta.relax_and_score(energy_ref_pdb)
                if ref_score.get("success"):
                    delta = round(score["total_reu"] - ref_score["total_reu"], 3)
                    energy["relative_delta_reu"]  = delta
                    energy["reference_total_reu"] = ref_score["total_reu"]
                    energy["energy_reference"]    = ref_name
                    energy["relative_note"] = (
                        "ΔREU = design − reference (negative = design lower-energy). "
                        "Ranking-reliable; magnitude approximate (~±2.7, §7)."
                    )
                    if delta > 5.0:
                        energy_flags.append(
                            f"design is much higher-energy than {ref_name} (ΔREU {delta:+.1f})"
                        )
                else:
                    # Could not score the reference → downgrade to sanity, stay honest.
                    energy["mode"] = "sanity"
                    energy["reason"] = (
                        f"Relative comparison requested and topologies matched, but {ref_name} "
                        f"could not be scored ({ref_score.get('error')}); reporting the design's "
                        "sanity signal only."
                    )
        energy["flags"] = energy_flags

        # ── Assemble evidence-rich report (no binary verdict) ─────────────────────
        report = {
            "success":          True,
            "design": {
                "sequence_length": fold.get("length"),
                "copies":          fold.get("copies", 1),
                "fold_source":     fold.get("fold_source"),
            },
            "fold_confidence":  fold_conf,
            "fold_preservation": fold_pres,
            "folding_energy":   energy,
            "artifacts": {
                "ranked_pdb":  fold.get("ranked_pdb"),
                "relaxed_pdb": energy.get("relaxed_pdb"),
                "png_paths":   fold.get("png_paths", {}),
            },
            "viz_applied": bool(rmsd.get("commands")),
            "commands":    rmsd.get("commands", []),
            "elapsed_s":   round(_time.perf_counter() - t0, 1),
        }
        if self.session:
            self.session.set_validate_design_results(model_id, report)

        # Auto-open the confidence-map PNGs (best-effort).
        self._open_pngs(fold.get("png_paths", {}))

        # ── Evidence-rich multi-line summary (honesty labels, NO pass/fail) ───────
        lines: List[str] = ["Validate-design report (evidence — not a verdict):"]
        _mp = f"{mean_plddt:.1f}" if isinstance(mean_plddt, (int, float)) else "n/a"
        _ptm = fold.get("ptm"); _iptm = fold.get("iptm")
        lines.append(
            f"  • Fold confidence [{fold.get('fold_source')}]: mean pLDDT {_mp}"
            + (f", pTM {_ptm:.2f}" if isinstance(_ptm, (int, float)) else "")
            + (f", ipTM {_iptm:.2f}" if isinstance(_iptm, (int, float)) else "")
            + (f", PAE matrix available" if fold.get("pae") is not None else "")
        )
        if fold_pres["rmsd_ca"] is not None:
            _pru = fold_pres.get("pruned_rmsd")
            _pru_s = f" / core(pruned) {_pru:.2f} Å" if isinstance(_pru, (int, float)) else ""
            lines.append(
                f"  • Fold preservation: all-pairs Cα RMSD {fold_pres['rmsd_ca']:.2f} Å"
                f"{_pru_s} vs {fold_pres['reference']}"
            )
            if fold_pres.get("concentration"):
                lines.append(f"      ↳ {fold_pres['concentration']}")
            _topk = fold_pres.get("top_deviant_residues") or []
            if _topk:
                _shown = ", ".join(f"{d['chain']}{d['resno']} {d['deviation']:.1f}Å" for d in _topk[:5])
                lines.append(f"      ↳ most-deviant residues: {_shown}"
                             + (" …" if len(_topk) > 5 else "")
                             + ("  [design coloured by per-residue deviation in ChimeraX]"
                                if fold_pres.get("colored_by_deviation") else ""))
        else:
            lines.append(f"  • Fold preservation: not assessed — {fold_pres['note']}")
        if energy.get("available"):
            if energy["mode"] == "relative" and "relative_delta_reu" in energy:
                lines.append(
                    f"  • Folding energy [RELATIVE]: ΔREU {energy['relative_delta_reu']:+.1f} vs "
                    f"{energy.get('energy_reference')} (total {energy['total_reu']:.1f} REU; "
                    f"ranking-reliable, magnitude approximate). {energy['reason']}"
                )
            elif energy["mode"] == "declined":
                lines.append(
                    f"  • Folding energy [SANITY — relative DECLINED]: total {energy['total_reu']:.1f} REU, "
                    f"density {energy['per_residue_density']:.2f} REU/res, "
                    f"clash_ok={energy['clash_ok']}. {energy['reason']}"
                )
            else:
                lines.append(
                    f"  • Folding energy [SANITY]: total {energy['total_reu']:.1f} REU, "
                    f"density {energy['per_residue_density']:.2f} REU/res, "
                    f"clash_ok={energy['clash_ok']}. {energy['reason']}"
                )
        else:
            lines.append(f"  • Folding energy: unavailable — {energy.get('error')}")
        _allflags = fold_flags + rmsd_flags + energy_flags
        if _allflags:
            lines.append("  • Flags: " + "; ".join(_allflags))
        summary = "\n".join(lines)

        return ToolStepResult(
            tool="validate_design", success=True,
            data=report,
            viz_commands=[],            # viz already applied live (with the RMSD run)
            viz_explanations=[],
            summary=summary,
            elapsed_ms=(_time.perf_counter() - t0) * 1000,
        )

    # ── Bridge accessors (lazy init) ───────────────────────────────────────────

    def _get_assembly_analyser(self):
        if self._assembly_analyser is None:
            from assembly_analyser import AssemblyAnalyser
            self._assembly_analyser = AssemblyAnalyser(
                bridge  = self.bridge,
                session = self.session,
            )
        return self._assembly_analyser

    def _get_disulfide_bridge(self):
        if self._disulfide_bridge is None:
            from disulfide_bridge import DisulfideBridge
            self._disulfide_bridge = DisulfideBridge(chimerax_bridge=self.bridge)
        return self._disulfide_bridge

    def _get_proline_bridge(self):
        if self._proline_bridge is None:
            from proline_bridge import ProlineBridge
            self._proline_bridge = ProlineBridge()
        return self._proline_bridge

    def _get_glycan_bridge(self):
        if self._glycan_bridge is None:
            from glycan_bridge import GlycanBridge
            self._glycan_bridge = GlycanBridge()
        return self._glycan_bridge

    def _get_netnglyc_bridge(self):
        if self._netnglyc_bridge is None:
            from netnglyc_bridge import NetNGlycBridge
            self._netnglyc_bridge = NetNGlycBridge()
        return self._netnglyc_bridge

    def _get_salt_bridge_bridge(self):
        if self._salt_bridge_bridge is None:
            from salt_bridge_bridge import SaltBridgeBridge
            self._salt_bridge_bridge = SaltBridgeBridge()
        return self._salt_bridge_bridge

    def _get_cavity_bridge(self):
        if self._cavity_bridge is None:
            from cavity_bridge import CavityBridge
            self._cavity_bridge = CavityBridge()
        return self._cavity_bridge

    def _get_double_mutant_bridge(self):
        if self._double_mutant_bridge is None:
            from double_mutant_bridge import DoubleMutantBridge
            self._double_mutant_bridge = DoubleMutantBridge()
        return self._double_mutant_bridge

    def _get_colabfold_bridge(self):
        if self._colabfold_bridge is None:
            from colabfold_bridge import ColabFoldBridge
            self._colabfold_bridge = ColabFoldBridge()
        return self._colabfold_bridge

    def _get_camsol_bridge(self):
        if self._camsol_bridge is None:
            from camsol_bridge import CamsolBridge
            self._camsol_bridge = CamsolBridge()
        return self._camsol_bridge

    def _get_esm_bridge(self):
        if self._esm_bridge is None:
            from esm_bridge import EsmBridge
            self._esm_bridge = EsmBridge()
        return self._esm_bridge

    def _get_esmfold_bridge(self):
        if self._esmfold_bridge is None:
            from esmfold_bridge import ESMFoldBridge
            self._esmfold_bridge = ESMFoldBridge()
        return self._esmfold_bridge

    def _get_proteinmpnn_bridge(self):
        if self._proteinmpnn_bridge is None:
            from proteinmpnn_bridge import ProteinMPNNBridge
            self._proteinmpnn_bridge = ProteinMPNNBridge()
        return self._proteinmpnn_bridge

    def _get_mpnn_esmfold_pipeline(self):
        if self._mpnn_esmfold_pipeline is None:
            from mpnn_esmfold_pipeline import MPNNESMFoldPipeline
            self._mpnn_esmfold_pipeline = MPNNESMFoldPipeline()
        return self._mpnn_esmfold_pipeline

    def _get_rfdiffusion_bridge(self):
        if self._rfdiffusion_bridge is None:
            from rfdiffusion_bridge import RFdiffusionBridge
            self._rfdiffusion_bridge = RFdiffusionBridge()
        return self._rfdiffusion_bridge

    def _get_rosetta_bridge(self):
        if self._rosetta_bridge is None:
            from rosetta_bridge import RosettaBridge
            self._rosetta_bridge = RosettaBridge()
        return self._rosetta_bridge

    # ── PDB file retrieval ────────────────────────────────────────────────────

    def _ensure_pdb_file(self, model_id: str) -> Optional[str]:
        """
        Get or download the PDB file for a loaded structure.

        Priority:
          1. session.structures[model_id]["path"] (if file exists locally)
          2. Download from RCSB if the structure name is a 4-char PDB ID
             → cached to cache/<ID>.pdb

        Returns the local path string, or None on failure.
        """
        info = self.session.get_structure(model_id)
        if not info:
            return None

        # Check for a cached local path
        path = info.get("path")
        if path and Path(path).is_file():
            return path

        # Try downloading from RCSB for known PDB IDs
        name = info.get("name", "")
        if not re.match(r"^[A-Za-z0-9]{4}$", name):
            return None

        cache_dir  = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / f"{name.upper()}.pdb"

        if local_path.is_file():
            info["path"] = str(local_path)
            return str(local_path)

        try:
            import requests
            url  = f"https://files.rcsb.org/download/{name.upper()}.pdb"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            info["path"] = str(local_path)
            return str(local_path)
        except Exception:
            return None

    # ── Sequence retrieval ─────────────────────────────────────────────────────

    def _fetch_sequence(
        self,
        model_id: str,
        chain:    Optional[str] = None,
    ) -> Optional[str]:
        """
        Attempt to get an amino-acid sequence for a loaded structure.

        Priority:
          1. Session state cache (sequences stored after first fetch)
          2. RCSB FASTA API (if the structure name is a 4-char PDB ID)
          3. ChimeraX 'sequence chain' command (may not return text via REST)
        """
        info = self.session.get_structure(model_id)
        if info:
            meta = info.get("metadata", {})
            # Check cached sequences dict
            if chain and isinstance(meta.get("sequences"), dict):
                seq = meta["sequences"].get(chain)
                if seq:
                    return seq
            elif meta.get("sequence"):
                return meta["sequence"]

            # Try RCSB for 4-letter PDB IDs
            name = info.get("name", "")
            if re.match(r"^[A-Za-z0-9]{4}$", name):
                seq = self._fetch_rcsb_fasta(name, chain)
                if seq:
                    # Cache for next time
                    if "sequences" not in meta:
                        meta["sequences"] = {}
                    key = chain or "A"
                    meta["sequences"][key] = seq
                    return seq

        # Last resort: ask ChimeraX
        return self._fetch_sequence_from_chimerax(model_id, chain)

    def _fetch_rcsb_fasta(
        self,
        pdb_id: str,
        chain:  Optional[str] = None,
    ) -> Optional[str]:
        """Fetch the amino-acid FASTA sequence from RCSB for a PDB ID + chain."""
        try:
            import requests
            url  = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/display"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            # Parse multi-entry FASTA; find the matching chain
            sequences = self._parse_fasta(resp.text)
            if not sequences:
                return None
            if chain:
                # RCSB FASTA headers contain the chain letter, e.g.:
                # >1HSG_1|Chain A|...
                for header, seq in sequences.items():
                    if f"|Chain {chain.upper()}|" in header or f"Chain {chain.upper()}" in header:
                        return seq
            # Fall back to first sequence
            return next(iter(sequences.values()))
        except Exception:
            return None

    @staticmethod
    def _parse_fasta(text: str) -> Dict[str, str]:
        """Parse a FASTA string into {header: sequence} dict."""
        sequences: Dict[str, str] = {}
        current_header = ""
        current_seq:    List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_seq:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq    = []
            else:
                current_seq.append(line)
        if current_header and current_seq:
            sequences[current_header] = "".join(current_seq)
        return sequences

    def _fetch_sequence_from_chimerax(
        self,
        model_id: str,
        chain:    Optional[str] = None,
    ) -> Optional[str]:
        """
        Ask ChimeraX for the sequence of a model/chain.
        Note: 'sequence chain' outputs to the GUI log, not always the REST response.
        This is a best-effort fallback.
        """
        if not self.bridge.is_running():
            return None
        spec = f"#{model_id}/{chain}" if chain else f"#{model_id}"
        result = self.bridge.run_command(f"sequence chain {spec}")
        if result.get("error") or not result.get("value"):
            return None
        text = result["value"]
        # Strip header line(s), join remaining text, keep only AA letters
        lines = text.strip().splitlines()
        seq_parts = []
        for line in lines:
            part = line.split(":", 1)[-1].strip()
            seq_parts.append(part)
        seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", "".join(seq_parts).upper())
        return seq if len(seq) >= 5 else None

    # ── Active-site command handler ────────────────────────────────────────────

    # Patterns for active-site management commands (matched case-insensitively).
    _ACTIVE_SITE_SET_RE   = re.compile(
        r"^\s*set\s+active[\s\-]?site\s+residues?\s+([\d\s,]+)\s*$", re.I
    )
    _ACTIVE_SITE_CLEAR_RE = re.compile(
        r"^\s*clear\s+active[\s\-]?site\s+residues?\s*$", re.I
    )
    _ACTIVE_SITE_SHOW_RE  = re.compile(
        r"^\s*show\s+active[\s\-]?site\s+residues?\s*$", re.I
    )

    def handle_active_site_command(self, user_input: str) -> Optional[str]:
        """
        Handle active-site residue management commands without LLM translation.

        Supported patterns:
          "set active site residues 25 26 27"  → stores {25, 26, 27} in session
          "clear active site residues"          → clears the stored set
          "show active site residues"           → prints current set

        Returns a human-readable result string if the command was handled,
        or None if the input is not an active-site command (caller should
        fall through to LLM translation).
        """
        text = user_input.strip()

        m = self._ACTIVE_SITE_SET_RE.match(text)
        if m:
            nums_str = m.group(1)
            nums = {
                int(x)
                for x in re.split(r"[\s,]+", nums_str.strip())
                if x.strip()
            }
            self.session.set_functional_residues(nums)
            return (
                f"Active-site residues set: {sorted(nums)}.\n"
                f"All subsequent proline scans will exclude positions "
                f"within 2 residues of these sites."
            )

        if self._ACTIVE_SITE_CLEAR_RE.match(text):
            self.session.set_functional_residues(set())
            return "Active-site residues cleared — proline scans will use SASA auto-detection."

        if self._ACTIVE_SITE_SHOW_RE.match(text):
            current = self.session.get_functional_residues()
            if current:
                return f"Active-site residues: {sorted(current)}."
            return "No active-site residues declared (SASA auto-detection will be used)."

        return None   # not an active-site command

    # ── Sequence display command handler ──────────────────────────────────────

    def _export_sequences_fasta(self) -> str:
        """Export ProteinMPNN designed sequences to a FASTA file on the desktop."""
        import config as _cfg
        model_id  = self._primary_model_id()
        mpnn_data = self.session.get_proteinmpnn_result(model_id)
        if not mpnn_data:
            return (
                "No ProteinMPNN sequences in session -- run "
                "'redesign chain A with ProteinMPNN' first."
            )
        sequences = mpnn_data.get("sequences", [])
        wildtype  = mpnn_data.get("wildtype_sequence", "")

        try:
            desktop = _cfg.desktop_path()
        except Exception:
            desktop = Path.home() / "Desktop"

        out_path = Path(desktop) / "proteinmpnn_designs.fasta"
        lines: list = []

        if wildtype:
            lines.append(f">wildtype length={len(wildtype)}")
            lines.append(wildtype)

        for i, entry in enumerate(sequences, 1):
            seq       = entry.get("sequence", "")
            score     = entry.get("score", 0.0)
            recovery  = entry.get("recovery", 0.0)
            mutations = entry.get("mutations", [])
            n_muts    = len(mutations)
            lines.append(
                f">design_{i} score={score:.3f} "
                f"recovery={recovery:.3f} mutations={n_muts}"
            )
            lines.append(seq)

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Sequences saved to {out_path}"

    # ── Live-selection commands (the ChimeraX selection as a first-class input) ─

    _SELECTION_REPORT_PHRASES = (
        "what's selected", "whats selected", "what is selected",
        "what residues are selected", "what's the selection", "whats the selection",
        "show the selection", "show selection", "describe the selection",
        "describe selection", "report the selection", "current selection",
        "list the selection", "the current selection",
    )
    _SELECTION_SCAN_PHRASES = (
        "scan the selection", "scan selection", "scan the selected",
        "analyze the selection", "analyse the selection",
        "analyze the selected", "analyse the selected",
        "solubility of the selection", "camsol the selection",
        "score the selection", "score the selected",
    )
    _SELECTION_REDESIGN_PHRASES = (
        "redesign the selection", "redesign the selected", "redesign selection",
        "design the selection", "design the selected residues",
        "mpnn the selection", "mpnn the selected",
    )

    def _detect_selection_intent(self, user_input: str) -> Optional[str]:
        """Return 'report' | 'scan' | 'redesign' for a selection-consuming phrasing,
        else None (the caller falls through to the LLM)."""
        low = user_input.lower().strip()
        if any(p in low for p in self._SELECTION_REPORT_PHRASES):
            return "report"
        if any(p in low for p in self._SELECTION_SCAN_PHRASES):
            return "scan"
        if any(p in low for p in self._SELECTION_REDESIGN_PHRASES):
            return "redesign"
        return None

    def handle_selection_command(self, user_input: str) -> Optional[str]:
        """
        Poll-on-command fast-path: when the user asks to act on "the selection",
        read the LIVE ChimeraX selection over REST and either report it or feed it
        into the matching tool. Returns a printable string, or None to fall through.

        Empty selection is a graceful, friendly message — never an error.
        """
        intent = self._detect_selection_intent(user_input)
        if intent is None:
            return None
        if getattr(self, "bridge", None) is None:
            return ("[warn]ChimeraX is not connected.[/warn] Open it, select residues, "
                    "then try again.")

        from selection import read_selection
        sel = read_selection(self.bridge.run_command)
        if sel.is_empty:
            return ("[warn]Nothing is selected in ChimeraX.[/warn]\n"
                    "[dim]Ctrl-click residues in the 3D view (or drag across them in the "
                    "Sequence Viewer), then run the command again.[/dim]")

        if intent == "report":
            return self._format_selection_report(sel)
        if intent == "scan":
            return self._scan_selection_camsol(sel)
        if intent == "redesign":
            return self._redesign_selection(sel)
        return None

    def _chain_resnums(self, model_id: str, chain: str) -> List[int]:
        """Ordered residue numbers of a loaded chain (via `info residues #M/CH`),
        parsed with the same selection parser. [] on failure."""
        try:
            from selection import parse_selection_text
            res = self.bridge.run_command(f"info residues #{model_id}/{chain}")
            text = res.get("value") if isinstance(res, dict) else ""
            refs = parse_selection_text(text or "", default_model=str(model_id))
            return sorted({r for _, c, r, _ in refs if c == chain})
        except Exception:
            return []

    def _format_selection_report(self, sel) -> str:
        by = sel.by_chain()
        lines = [f"[hi]Current ChimeraX selection[/hi] — [bold]{sel.count}[/bold] "
                 f"residue(s) across {len(by)} chain(s):"]
        for chain, rns in by.items():
            spec = self._compact_resspec(rns)
            lines.append(f"  chain [bold]{chain}[/bold]: {len(rns)} residue(s)  "
                         f"([dim]{spec}[/dim])")
        lines.append("[dim]→ 'scan the selection' (solubility) or 'redesign the "
                     "selection' (ProteinMPNN) act on exactly these residues.[/dim]")
        return "\n".join(lines)

    def _scan_selection_camsol(self, sel) -> str:
        """Run CamSol on the selected chain(s) and report/colour EXACTLY the
        selected residues (their per-residue solubility), nothing else."""
        from proteinmpnn_bridge import chain_resnum_to_seqpos
        from camsol_bridge import _assign_colour
        bridge = self._get_camsol_bridge()
        lines = ["[hi]CamSol solubility — selected residues only[/hi]:"]
        color_cmds: List[str] = []
        any_scored = False
        for chain, sel_resnums in sel.by_chain().items():
            model_id = (sel.models[0] if sel.models else None) or self._first_model_id()
            seq = self._fetch_sequence(model_id, chain)
            if not seq:
                lines.append(f"  chain {chain}: [warn]no sequence available[/warn]")
                continue
            ordered = self._chain_resnums(model_id, chain) or list(range(1, len(seq) + 1))
            pos1 = chain_resnum_to_seqpos(ordered)          # resnum -> 1-based position
            res = bridge.analyze(seq, model_id=model_id, chain=chain,
                                 start_resno=ordered[0], session=self.session)
            scores = res.data.get("scores", {}) if res.success else {}
            scores_list = [scores[k] for k in sorted(scores)]   # 0-indexed by position
            band: Dict[str, List[int]] = {}
            for r in sel_resnums:
                p = pos1.get(r)
                sc = scores_list[p - 1] if (p and 1 <= p <= len(scores_list)) else None
                if sc is None:
                    lines.append(f"  {chain}:{r}  (not scored)")
                    continue
                any_scored = True
                tag = "aggregation-prone" if sc < -0.5 else ("soluble" if sc > 0.5 else "neutral")
                lines.append(f"  {chain}:{r}  z={sc:+.2f}  [{tag}]")
                band.setdefault(_assign_colour(sc), []).append(r)
            for colour, rs in band.items():
                if colour == "white":
                    continue
                color_cmds.append(
                    f"color #{model_id}/{chain}:{self._compact_resspec(rs)} {colour}")
        if color_cmds:
            try:
                self.bridge.run_commands(color_cmds)
                lines.append("[dim]Selected residues coloured in 3D by solubility "
                             "(red=aggregation-prone … blue=soluble).[/dim]")
            except Exception:
                pass
        if not any_scored:
            lines.append("[dim](no residues could be scored)[/dim]")
        return "\n".join(lines)

    def _redesign_selection(self, sel) -> str:
        """Feed the live selection into ProteinMPNN as its designable set."""
        chain = sel.chains[0]
        model_id = (sel.models[0] if sel.models else None) or self._first_model_id()
        result = self._run_proteinmpnn({
            "model_id": model_id, "chain_id": chain,
            "design_positions": sel.resnums(chain),
        })
        if result.viz_commands:
            try:
                self.bridge.run_commands(result.viz_commands)
            except Exception:
                pass
        if not result.success:
            return f"[warn]{result.error or 'ProteinMPNN failed'}[/warn]"
        return (f"[ok]ProteinMPNN redesigned the {len(sel.resnums(chain))} selected "
                f"residue(s) on chain {chain}.[/ok]\n{result.summary or ''}")

    def handle_sequence_display_command(self, user_input: str) -> Optional[str]:
        """
        Handle "show designed sequences" and similar commands without LLM translation.

        Returns a human-readable string if BOTH conditions hold:
          1. *user_input* matches a display keyword OR fuzzy rule (see below), AND
          2. The session already holds ProteinMPNN results.

        Returns None if either condition is false — the caller should fall through
        to LLM translation so the model can answer naturally.

        Fuzzy rule: plural "sequences" + any display verb ("show", "output",
        "print", "list", "display", "what") also triggers the handler when the
        session has MPNN results, catching natural phrasings like
        "can you output the sequences" that aren't covered by explicit keywords.
        """
        lower = user_input.lower().strip()

        # FASTA export takes priority
        if any(kw in lower for kw in self._FASTA_EXPORT_KEYWORDS):
            return self._export_sequences_fasta()

        intent = self._detect_mpnn_display_intent(user_input)
        if intent is None:
            # Back-compat: the original plural-keyword / fuzzy rule still counts as
            # a 'sequence' display request.
            if any(kw in lower for kw in self._SEQUENCE_DISPLAY_KEYWORDS) or (
                "sequences" in lower and any(v in lower for v in self._SEQUENCE_DISPLAY_VERBS)
            ):
                intent = "sequence"

        if intent is None:
            return None   # not a display request — let the LLM handle it (incl. runs)

        model_id = self._first_model_id()
        mpnn_data, src = self._resolve_mpnn_data(model_id)
        if not mpnn_data:
            # A DISPLAY request with no design anywhere must NOT fall through to a
            # fresh MPNN run — inform instead.
            return ("No ProteinMPNN design found in this session or the cache "
                    "(cache/proteinmpnn/). Run a redesign first, e.g. "
                    "'redesign chain A with ProteinMPNN'.")

        if intent == "alignment":
            return self._show_mpnn_alignment(mpnn_data, model_id, src)
        return self._show_designed_sequences(mpnn_data=mpnn_data, model_id=model_id)

    def _detect_mpnn_display_intent(self, text: str) -> Optional[str]:
        """
        Classify a redesign-DISPLAY request (never a run): 'alignment', 'sequence',
        or None. Run/re-run phrasings ('redesign chain A with MPNN') return None so
        they fall through to the normal pipeline.
        """
        if not text:
            return None
        low = text.lower()
        if any(t in low for t in self._MPNN_RUN_TRIGGERS):
            return None
        if any(k in low for k in self._MPNN_ALIGNMENT_KEYWORDS):
            return "alignment"
        has_verb = any(v in low for v in self._MPNN_DISPLAY_VERBS)
        has_noun = any(n in low for n in (
            "sequence", "sequences", "redesign", "redesigned", "designed",
            "the design", "mpnn", "mutation",
        ))
        if has_verb and has_noun:
            return "sequence"
        return None

    def _resolve_mpnn_data(self, model_id: Optional[str] = None):
        """
        Return the most recent ProteinMPNN result WITHOUT re-running: prefer the
        in-session result, fall back to the latest persisted cache FASTA. Returns
        (mpnn_data | None, source_str).
        """
        model_id = model_id or self._first_model_id()
        data = self.session.get_proteinmpnn_result(model_id)
        if data:
            return data, "session"
        # Fall back to the on-disk design cache (survives an unsaved session).
        try:
            from proteinmpnn_bridge import latest_cached_fasta, read_designs_fasta
            fa = latest_cached_fasta(model_id) or latest_cached_fasta()
            if fa:
                return read_designs_fasta(fa), f"cache:{fa.name}"
        except Exception:
            pass
        return None, "none"

    def _show_designed_sequences(self, mpnn_data=None, model_id=None) -> str:
        """
        Build a human-readable display of ProteinMPNN designed sequences,
        retrieving (never re-running) from the session or the persistent cache.
        """
        model_id  = model_id or self._first_model_id()
        if mpnn_data is None:
            mpnn_data, _src = self._resolve_mpnn_data(model_id)
        if not mpnn_data:
            return (
                "No ProteinMPNN design found (session or cache). "
                "Run a sequence redesign first "
                "(e.g. 'redesign chain A with ProteinMPNN')."
            )

        sequences   = mpnn_data.get("sequences", [])
        wildtype    = mpnn_data.get("wildtype_sequence", "")
        backend     = mpnn_data.get("backend", "unknown")
        n_sequences = len(sequences)

        if not sequences:
            return "ProteinMPNN ran but produced no designed sequences."

        lines = [
            f"ProteinMPNN Designed Sequences — model #{model_id} ({backend})",
            f"  Wildtype length: {len(wildtype)} residues",
            f"  Designs: {n_sequences}",
            "",
        ]

        for i, seq_entry in enumerate(sequences, 1):
            seq       = seq_entry.get("sequence", "")
            score     = seq_entry.get("score", 0.0)
            recovery  = seq_entry.get("recovery", 0.0)
            mutations = list(seq_entry.get("mutations") or [])

            # Compute mutations from diff if not stored
            if not mutations and wildtype and seq:
                try:
                    from mpnn_esmfold_pipeline import _diff_sequences
                    mutations = _diff_sequences(wildtype, seq)
                except ImportError:
                    pass

            n_muts      = len(mutations)
            mut_display = ", ".join(mutations)   # full set — never hide the design

            lines.append(f"  Design {i}/{n_sequences}")
            lines.append(f"    Score:     {score:.3f}")
            lines.append(f"    Recovery:  {recovery:.1%}")
            lines.append(
                f"    Mutations: {n_muts} — {mut_display if mut_display else 'none (identical to WT)'}"
            )
            seq_display = seq
            lines.append(f"    Sequence:  {seq_display}")
            lines.append("")

        return "\n".join(lines)

    # ── WT-vs-redesign alignment (console + interactive ChimeraX) ────────────────

    def _show_mpnn_alignment(self, mpnn_data, model_id: str, src: str) -> str:
        """
        Render the numbered WT-vs-top-redesign console alignment AND (best-effort,
        if ChimeraX is up) open the interactive ChimeraX Sequence Viewer associated
        with the loaded chain. Retrieval only — never re-runs MPNN.
        """
        seqs = mpnn_data.get("sequences") or []
        wt   = mpnn_data.get("wildtype_sequence") or ""
        if not seqs or not wt:
            return "No ProteinMPNN design available to align."
        top   = seqs[0]
        chain = mpnn_data.get("chain", "A")
        console_view = self._build_mpnn_alignment_console(
            wt, top.get("sequence", ""), top, model_id, chain, src
        )
        viz_note = self._open_mpnn_sequence_viewer(
            wt, top.get("sequence", ""), model_id, chain
        )
        return console_view + ("\n" + viz_note if viz_note else "")

    @staticmethod
    def _alignment_rows(wt: str, design: str) -> List[int]:
        """1-based positions where design differs from WT (over the common length)."""
        return [i for i, (w, d) in enumerate(zip(wt, design), 1) if w != d]

    def _build_mpnn_alignment_console(
        self, wt: str, design: str, top: Dict[str, Any],
        model_id: str, chain: str, src: str, width: int = 50,
    ) -> str:
        """
        Numbered, Rich-marked WT-vs-redesign alignment in residue numbering
        (1-based, consistent with the recovery viz), in blocks of *width*, every
        changed column flagged. No truncation.
        """
        n = min(len(wt), len(design))
        changed = set(self._alignment_rows(wt, design))
        mutations = list(top.get("mutations") or [
            f"{wt[i-1]}{i}{design[i-1]}" for i in sorted(changed)
        ])
        rec = top.get("recovery")
        out: List[str] = [
            f"[bold]WT vs ProteinMPNN redesign[/bold] — model #{model_id} chain {chain}  "
            f"[dim](source: {src})[/dim]",
            f"  length {n} · [red]{len(changed)} changed[/red] · "
            + (f"recovery {rec:.1%}" if isinstance(rec, (int, float)) else "")
            + "   (numbering = 1-based residue position)",
            "",
        ]
        for start in range(0, n, width):
            end = min(start + width, n)
            ruler = "".join(
                ("|" if (p % 10 == 0) else (":" if (p % 5 == 0) else "."))
                for p in range(start + 1, end + 1)
            )
            wt_row, de_row, mark = [], [], []
            for p in range(start + 1, end + 1):
                w, d = wt[p - 1], design[p - 1]
                if p in changed:
                    wt_row.append(w)
                    de_row.append(f"[bold red]{d}[/bold red]")
                    mark.append("^")
                else:
                    wt_row.append(w)
                    de_row.append(d)
                    mark.append(" ")
            lbl = f"{start + 1:>5}"
            out.append(f"  pos {lbl} {ruler}")
            out.append(f"  WT      {''.join(wt_row)}")
            out.append(f"  design  {''.join(de_row)}")
            out.append(f"          {''.join(mark)}")
            out.append("")
        out.append(f"  [bold]Changes ({len(mutations)}):[/bold] " + ", ".join(mutations))
        return "\n".join(out)

    def _open_mpnn_sequence_viewer(
        self, wt: str, design: str, model_id: str, chain: str,
    ) -> Optional[str]:
        """
        Build a 2-sequence ungapped FASTA (WT + top redesign) and open it in
        ChimeraX's Sequence Viewer associated with the loaded chain, so selecting a
        column selects the 3D residue (and vice versa). Best-effort; returns a note.

        Association subtlety: the redesign is heavily changed (> ChimeraX's ~10%
        auto-association threshold) so the redesign row won't auto-associate, but
        the WT row (identical to chain A) does, and the ungapped 1:1 columns still
        map to the correct 3D residues through it.
        """
        if self.bridge is None:
            return ("  (ChimeraX not connected — run while ChimeraX is open to get the "
                    "interactive Sequence Viewer: changed columns highlighted; select a "
                    "column → highlights the 3D residue.)")
        try:
            from proteinmpnn_bridge import build_alignment_fasta
            import config as _cfg
            out = Path(_cfg.PROTEINMPNN_CACHE_DIR) / f"alignment_model{model_id}.fa"
            build_alignment_fasta(wt, design, out,
                                  wt_label="WT", design_label="redesign_top")
            posix = Path(out).as_posix()

            changed   = self._alignment_rows(wt, design)            # 1-based changed columns
            n         = min(len(wt), len(design))
            conserved = [i for i in range(1, n + 1) if i not in set(changed)]
            spec = f"#{model_id}/{chain}"

            cmds = [
                f'open "{posix}"',                       # opens the Sequence Viewer + auto-associates WT↔chain
                f"sequence associate {spec}",            # force-associate the chain to the alignment
                # ── Auto-decorate (one consistent story across sequence + 3D) ──
                # ChimeraX 1.11.1 has no `sequence region`/`sequence color` command,
                # so we (a) colour the 3D STRUCTURE with the MPNN convention
                # (changed=tomato, conserved=cornflower blue), and (b) SELECT the
                # changed residues — through the WT association the Sequence Viewer
                # mirrors this as a highlighted region over exactly the changed
                # columns. The auto-shown conservation header also marks them.
                f"cartoon {spec}",
                f"color {spec} white",
            ]
            if conserved:
                cmds.append(f"color {spec}:{self._compact_resspec(conserved)} cornflower blue")
            if changed:
                cmds.append(f"color {spec}:{self._compact_resspec(changed)} tomato")
                cmds.append(f"select {spec}:{self._compact_resspec(changed)}")

            ran_ok = True
            for c in cmds:
                r = self.bridge.run_command(c)
                if isinstance(r, dict) and r.get("error"):
                    ran_ok = False
            if ran_ok:
                return (f"  [green]Interactive alignment open in ChimeraX[/green] — "
                        f"{len(changed)} changed columns highlighted in the Sequence Viewer "
                        f"and coloured [red]tomato[/red] on chain {chain} in 3D (conserved = "
                        "cornflower blue). Select a column → highlights the 3D residue (and "
                        "vice versa); the WT row is associated with the structure.")
            return f"  (Alignment FASTA written to {out}; open it in ChimeraX to interact.)"
        except Exception as exc:
            return f"  (Could not open the ChimeraX Sequence Viewer: {exc})"

    @staticmethod
    def _compact_resspec(positions: List[int]) -> str:
        """Compact a sorted 1-based position list into a ChimeraX spec, e.g.
        [1,2,3,5,8,9] → '1-3,5,8-9'."""
        if not positions:
            return ""
        ps = sorted(set(positions))
        runs, start, prev = [], ps[0], ps[0]
        for p in ps[1:]:
            if p == prev + 1:
                prev = p
            else:
                runs.append((start, prev)); start = prev = p
        runs.append((start, prev))
        return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _first_model_id(self) -> str:
        """Return the first loaded structure's model ID, or '1'."""
        if self.session.structures:
            return next(iter(self.session.structures))
        return "1"

    def _primary_model_id(self) -> str:
        """
        Return the primary (crystal-structure) model ID.

        When multiple models are loaded — e.g. the original structure (#1)
        alongside an ESMFold prediction (#2) — tools that operate on the
        crystal structure (proline scan, assembly analysis, …) should always
        target #1.  This helper returns '1' whenever that model is in the
        session, falling back to the first loaded model otherwise.
        """
        structures = self.session.structures
        if not structures:
            return "1"
        if "1" in structures:
            return "1"
        return next(iter(structures))

    def available_tools(self) -> Dict[str, str]:
        """Return {tool_name: status_string} for all tools."""
        status: Dict[str, str] = {"chimerax": "active"}

        for name in ("camsol_bridge", "esm_bridge"):
            try:
                __import__(name)
                status[name.replace("_bridge", "")] = "active"
            except ImportError:
                status[name.replace("_bridge", "")] = "module not found"

        try:
            from proteinmpnn_bridge import ProteinMPNNBridge
            _b = ProteinMPNNBridge()
            status["proteinmpnn"] = (
                f"{_b._backend} — {_b._dir}" if _b._available
                else "not configured (set PROTEINMPNN_DIR)"
            )
        except ImportError:
            status["proteinmpnn"] = "module not found"

        try:
            from rfdiffusion_bridge import RFdiffusionBridge
            _rfd = RFdiffusionBridge()
            status["rfdiffusion"] = (
                f"rfdiffusion — {_rfd._dir}" if _rfd._available
                else "not configured (set RFDIFFUSION_DIR)"
            )
        except ImportError:
            status["rfdiffusion"] = "module not found"

        # Rosetta: report which backend is configured
        try:
            from rosetta_bridge import RosettaBridge, _select_backend
            backend = _select_backend()
            import os
            api_key = os.environ.get("ROBETTA_API_KEY", "").strip()
            if backend == "pyrosetta":
                status["rosetta"] = "PyRosetta (local) — active"
            elif api_key:
                status["rosetta"] = "Robetta web API — active"
            else:
                status["rosetta"] = "Robetta web API — set ROBETTA_API_KEY to enable"
        except ImportError:
            status["rosetta"] = "module not found"

        # mutation_scan depends on rosetta_bridge + mutation_scanner
        try:
            __import__("mutation_scanner")
            status["mutation_scan"] = "active (depends on rosetta)"
        except ImportError:
            status["mutation_scan"] = "module not found"

        # assembly_analyser
        try:
            __import__("assembly_analyser")
            status["assembly_analyser"] = "active"
        except ImportError:
            status["assembly_analyser"] = "module not found"

        # disulfide
        try:
            __import__("disulfide_bridge")
            status["disulfide"] = "active (geometry + ESM + DynaMut2)"
        except ImportError:
            status["disulfide"] = "module not found"

        # proline
        try:
            __import__("proline_bridge")
            status["proline"] = "active (backbone φ/ψ + ESM tolerance + optional DynaMut2)"
        except ImportError:
            status["proline"] = "module not found"

        # esmfold
        try:
            __import__("esmfold_bridge")
            status["esmfold"] = "active (ESM Atlas API — free, no auth)"
        except ImportError:
            status["esmfold"] = "module not found"

        # glycan
        try:
            __import__("glycan_bridge")
            status["glycan"] = "active (NXS/T sequon detection + SASA/SS/ESM scoring)"
            status["glycan_positions"] = "active (projection-aware all-residue scan)"
        except ImportError:
            status["glycan"] = "module not found"
            status["glycan_positions"] = "module not found"

        # salt_bridge
        try:
            __import__("salt_bridge_bridge")
            status["salt_bridge"] = "active (BioPython + FreeSASA)"
        except ImportError:
            status["salt_bridge"] = "module not found"

        # cavity
        try:
            __import__("cavity_bridge")
            status["cavity"] = "active (BioPython + FreeSASA dual-probe)"
        except ImportError:
            status["cavity"] = "module not found"

        # mpnn_esmfold
        try:
            __import__("mpnn_esmfold_pipeline")
            status["mpnn_esmfold"] = "active (ProteinMPNN + ESMFold combined validation)"
        except ImportError:
            status["mpnn_esmfold"] = "module not found"

        # rosetta_local via WSL2
        try:
            from wsl_bridge import WSLBridge
            wsl = WSLBridge()
            if wsl.is_available():
                if wsl.check_pyrosetta():
                    status["rosetta_local"] = "active (PyRosetta via WSL2)"
                else:
                    status["rosetta_local"] = "WSL2 available — PyRosetta not installed"
            else:
                status["rosetta_local"] = "WSL2 not installed"
        except ImportError:
            status["rosetta_local"] = "wsl_bridge module not found"

        return status

    def __repr__(self) -> str:
        extra = [t for t in (self._camsol_bridge, self._esm_bridge, self._proteinmpnn_bridge)
                 if t is not None]
        return (
            f"<ToolRouter bridge={self.bridge!r} "
            f"loaded_bridges={len(extra)}>"
        )
