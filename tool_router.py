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
        "proteinmpnn":       "🔬",
        "rfdiffusion":       "🌀",
        "rosetta":           "⚗️",
        "mutation_scan":     "🔬⚗️",
        "assembly_analyser": "🔗",
        "disulfide":         "🔗⚗️",
    }

    def __init__(
        self,
        bridge:  "ChimeraXBridge",
        session: "SessionState",
    ):
        self.bridge  = bridge
        self.session = session

        # Bridges are instantiated lazily on first use
        self._camsol_bridge:        Optional[Any] = None
        self._esm_bridge:           Optional[Any] = None
        self._proteinmpnn_bridge:   Optional[Any] = None
        self._rfdiffusion_bridge:   Optional[Any] = None
        self._rosetta_bridge:       Optional[Any] = None
        self._mutation_scanner:     Optional[Any] = None
        self._assembly_analyser:    Optional[Any] = None
        self._disulfide_bridge:     Optional[Any] = None

    # ── Phase 1: Route (no execution) ─────────────────────────────────────────

    def route(self, translator_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Augment a translator result with tool routing metadata.

        Adds the following keys (safe to call before user confirmation):
          "tools_needed"    — list of tools from translator (default ["chimerax"])
          "tool_steps_info" — list of {tool, icon, description} for each step
          "has_extra_tools" — True if any non-chimerax tools are present
        """
        tools_needed = translator_result.get("tools_needed") or ["chimerax"]
        tool_inputs  = translator_result.get("tool_inputs") or {}

        result = dict(translator_result)
        result["tools_needed"] = tools_needed
        result["tool_inputs"]  = tool_inputs

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
        if tool == "proteinmpnn":
            return "ProteinMPNN fixed-backbone sequence redesign"
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
            step = self._dispatch_tool(tool, tool_inputs.get(tool) or {})
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
        tool:   str,
        inputs: Dict[str, Any],
    ) -> ToolStepResult:
        """Route to the correct bridge; return ToolStepResult."""
        try:
            if tool == "camsol":
                return self._run_camsol(inputs)
            if tool == "esm":
                return self._run_esm(inputs)
            if tool == "proteinmpnn":
                return self._run_proteinmpnn(inputs)
            if tool == "rfdiffusion":
                return self._run_rfdiffusion(inputs)
            if tool == "rosetta":
                return self._run_rosetta(inputs)
            if tool == "mutation_scan":
                return self._run_mutation_scan(inputs)
            if tool == "assembly_analyser":
                return self._run_assembly_analyser(inputs)
            if tool == "disulfide":
                return self._run_disulfide(inputs)
            return ToolStepResult(
                tool=tool, success=False,
                error=(
                    f"Unknown tool '{tool}'. "
                    "Available: chimerax, camsol, esm, proteinmpnn, rfdiffusion, "
                    "rosetta, mutation_scan, assembly_analyser, disulfide."
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

    def _run_proteinmpnn(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge = self._get_proteinmpnn_bridge()
        return bridge.analyze(inputs, session=self.session)

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

        summary = (
            f"Mutation scan [{analysis_mode} mode]: {len(results)} candidate(s) found.{excluded_note} "
            f"Top: {top['from_aa']}{top['position']}{top['to_aa']} "
            f"(score={top['combined_score']:+.2f}, "
            f"ddG={top['ddg']:+.2f} kcal/mol, "
            f"solubility Δ={top['solubility_delta']:+.2f})"
        )

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
            summary          = summary,
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

    def _get_proteinmpnn_bridge(self):
        if self._proteinmpnn_bridge is None:
            from proteinmpnn_bridge import ProteinMPNNBridge
            self._proteinmpnn_bridge = ProteinMPNNBridge()
        return self._proteinmpnn_bridge

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
