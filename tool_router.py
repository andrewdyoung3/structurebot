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
    }

    # Keywords that signal a ProteinMPNN + ESMFold validation request.
    # Checked case-insensitively; any match replaces 'proteinmpnn' in the pipeline.
    _MPNN_ESMFOLD_KEYWORDS: tuple = (
        "validate mpnn",
        "validate design",
        "fold design",
        "esmfold mpnn",
        "esmfold design",
        "check foldabilit",   # "check foldability"
        "fold redesign",
        "validate redesign",
        "mpnn esmfold",
        "mpnn+esmfold",
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

    # ── Phase 1: Route (no execution) ─────────────────────────────────────────

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
                new_inputs["proline"] = {
                    "model_id": ms.get("model_id") or self._first_model_id(),
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
                "model_id": self._first_model_id(),
                "chain":    "A",
                "top_n":    5,
            }

        return new_tools, new_inputs

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
            else:
                new_tools.append(tool)

        # If proteinmpnn was not in the list, still add mpnn_esmfold
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
        if user_input and self._detect_mpnn_esmfold_intent(user_input):
            if "proteinmpnn" in tools_needed:
                tools_needed, tool_inputs = self._rewrite_as_mpnn_esmfold(
                    tools_needed, tool_inputs
                )

        result = dict(translator_result)
        result["tools_needed"] = tools_needed
        result["tool_inputs"]  = tool_inputs
        result["_user_input"]  = user_input   # passed to execute() for guard

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

        Proline guard: if tool is 'mutation_scan' but user_input contains proline
        intent keywords, redirect to _run_proline() so the backbone φ-angle
        scanner runs instead of the generic CamSol/ESM/Rosetta pipeline.
        """
        try:
            if tool == "camsol":
                return self._run_camsol(inputs)
            if tool == "esm":
                return self._run_esm(inputs)
            if tool == "esmfold":
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
                # ── Proline guard (secondary safety net) ─────────────────────
                # If the user asked about proline and route() didn't catch it
                # (e.g. user_input was not passed to route()), redirect here.
                if user_input and self._detect_proline_intent(user_input):
                    proline_inputs = {
                        "model_id": inputs.get("model_id") or self._first_model_id(),
                        "chain":    inputs.get("chain", "A"),
                        "pdb_path": inputs.get("pdb_path"),
                        "top_n":    5,
                    }
                    return self._run_proline(proline_inputs)
                return self._run_mutation_scan(inputs)
            if tool == "assembly_analyser":
                return self._run_assembly_analyser(inputs)
            if tool == "disulfide":
                return self._run_disulfide(inputs)
            if tool == "proline":
                return self._run_proline(inputs)
            if tool == "glycan":
                return self._run_glycan(inputs)
            return ToolStepResult(
                tool=tool, success=False,
                error=(
                    f"Unknown tool '{tool}'. "
                    "Available: chimerax, camsol, esm, esmfold, proteinmpnn, "
                    "mpnn_esmfold, rfdiffusion, rosetta, mutation_scan, "
                    "assembly_analyser, disulfide, proline, glycan."
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

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _first_model_id(self) -> str:
        """Return the first loaded structure's model ID, or '1'."""
        if self.session.structures:
            return next(iter(self.session.structures))
        return "1"

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
        except ImportError:
            status["glycan"] = "module not found"

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
