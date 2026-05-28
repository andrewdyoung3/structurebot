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
        "electrostatic",
        "ionic interaction",
        "charge pair",
        "engineer salt",
        "new salt bridge",
    )

    _CAVITY_KEYWORDS: tuple = (
        "cavit",              # matches cavity, cavities, cavity-filling, etc.
        "void",
        "cavity fill",
        "fill cavity",
        "internal void",
        "packing defect",
        "hydrophobic core",
    )

    # Assembly / dimer keywords trigger chains=None (all chains) in find_cavities
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
        "binding pocket",
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

    # ── Phase 1: Route (no execution) ─────────────────────────────────────────

    # ── Double mutant intent helpers ──────────────────────────────────────────

    @classmethod
    def _detect_double_mutant_intent(cls, text: str) -> bool:
        """Return True if *text* signals a double mutant pair scoring request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._DOUBLE_MUTANT_KEYWORDS)

    # ── Proline intent helpers ────────────────────────────────────────────────

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
        _cav_intent = bool(user_input and any(kw in user_input.lower() for kw in self._CAVITY_KEYWORDS))
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
        tool_inputs  = routed_result.get("tool_inputs") or {}
        # user_input stored by route() for the proline guard
        user_input   = routed_result.get("_user_input", "")

        step_results:   List[Dict[str, Any]] = []
        all_viz_cmds:   List[str] = []
        all_viz_exps:   List[str] = []
        tool_summaries: Dict[str, str] = {}
        pipeline_error: Optional[str] = None

        for tool in tools_needed:
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
                    "double_mutant."
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

    def _run_proteinmpnn(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run ProteinMPNN fixed-backbone sequence redesign."""
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

        # ── Fixed positions: explicit > session interface residues ─────────────
        fixed_positions: List[int] = inputs.get("fixed_positions") or []
        if not fixed_positions:
            # Use interface / active-site residues cached by the assembly analyser
            interface_data = self.session.get_interface_residues(model_id)
            if interface_data:
                fixed_positions = (
                    self.session.get_protected_residues_for_chain(model_id, chain_id)
                    or []
                )

        full_inputs: Dict[str, Any] = {
            "model_id":        model_id,
            "pdb_path":        pdb_path,
            "chain_id":        chain_id,
            "fixed_positions": fixed_positions,
            "num_sequences":   int(inputs.get("num_sequences", 8)),
            "temperature":     float(inputs.get("temperature", 0.1)),
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
            f"ddG={top['ddg']:+.2f} kcal/mol, "
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

        # 1. Exact keyword match
        matched = any(kw in lower for kw in self._SEQUENCE_DISPLAY_KEYWORDS)

        # 2. Fuzzy match — plural "sequences" + a display verb
        if not matched:
            if "sequences" in lower and any(
                v in lower for v in self._SEQUENCE_DISPLAY_VERBS
            ):
                matched = True

        if not matched:
            return None

        model_id = self._primary_model_id()
        if self.session.get_proteinmpnn_result(model_id) is None:
            return None   # no results yet — let the LLM respond

        return self._show_designed_sequences()

    def _show_designed_sequences(self) -> str:
        """
        Build a human-readable display of ProteinMPNN designed sequences
        from the current session.

        Returns a multi-line string ready for console.print(), or an
        informative error message if no MPNN results are in the session.
        """
        model_id  = self._first_model_id()
        mpnn_data = self.session.get_proteinmpnn_result(model_id)
        if not mpnn_data:
            return (
                "No ProteinMPNN results in session. "
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
            mut_display = ", ".join(mutations[:8])
            if n_muts > 8:
                mut_display += f" (+{n_muts - 8} more)"

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
