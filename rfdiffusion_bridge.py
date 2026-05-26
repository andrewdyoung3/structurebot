"""
rfdiffusion_bridge.py
---------------------
RFdiffusion integration for StructureBot.  Currently a documented stub.

RFdiffusion (Watson et al., 2023) is a deep-learning backbone diffusion model
for *de novo* protein design.  It can:
  - Design binders to a target protein (hotspot-guided)
  - Scaffold functional motifs (e.g. enzyme active sites)
  - Generate symmetric oligomers (cyclic, dihedral, tetrahedral)
  - Partial-diffusion (diversify) an existing structure

Unlike ProteinMPNN (sequence design given a fixed backbone), RFdiffusion
generates *new backbones* and is therefore a substantially heavier compute step.

Installation
------------
RFdiffusion requires:
  - Python 3.9-3.11 (NOT 3.12+ as of 2024; check repo for updates)
  - PyTorch >= 2.0
  - ~20 GB model weights (download script included in repo)

  git clone https://github.com/RosettaCommons/RFdiffusion
  cd RFdiffusion
  pip install -e .
  bash scripts/download_models.sh models/

  # In .env.local:
  RFDIFFUSION_DIR=C:/path/to/RFdiffusion

Interface
---------
analyze(inputs, session) -> ToolStepResult
  inputs keys:
    mode              : "binder" | "motif_scaffold" | "symmetric" | "partial_diffusion"
    pdb_path          : target structure (for binder/motif modes)
    chain_id          : target chain (binder mode)
    hotspot_residues  : list of residue numbers on target to bind near
                        e.g. [82, 83, 84, 119, 120]
    num_designs       : number of backbone samples to generate (default 4)
    num_steps         : diffusion steps (default 50; more = better, slower)
    symmetry          : "C2" | "C3" | ... | "D2" etc. (symmetric mode)
    partial_T         : noise level for partial-diffusion (0.0-1.0, default 0.2)
    contigs           : contig string for motif scaffolding
                        e.g. "A1-10/20-30/A50-80"

  Returns (when active):
    data["pdb_paths"]    : list of generated .pdb file paths
    data["mode"]         : the design mode used
    data["num_designs"]  : number of structures generated
    viz_commands         : ChimeraX commands to open all designs
    summary              : e.g. "RFdiffusion: 4 binder backbones generated"

Activation
----------
Set RFDIFFUSION_DIR in .env.local to the cloned RFdiffusion repo directory.
The bridge will auto-detect run_inference.py + the models/ directory.

References
----------
Watson JL, Juergens D, Bennett NR, et al. (2023).
"De novo design of protein structure and function with RFdiffusion."
Nature 620:1089-1100.  https://doi.org/10.1038/s41586-023-06415-8
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import config as _cfg
from tool_router import ToolStepResult

# ── Configuration ─────────────────────────────────────────────────────────────

_RFDIFFUSION_DIR: str = os.environ.get(
    "RFDIFFUSION_DIR",
    str(Path(_cfg.__file__).parent / "RFdiffusion"),
).strip()

_INSTALL_INSTRUCTIONS = """\
RFdiffusion is not yet configured.

To enable it:
  1. Clone the repo (Python 3.9-3.11 only; check repo for updates):
       git clone https://github.com/RosettaCommons/RFdiffusion

  2. Download model weights (~20 GB):
       cd RFdiffusion && bash scripts/download_models.sh models/

  3. Install:
       pip install -e .

  4. Add to .env.local:
       RFDIFFUSION_DIR=C:/path/to/RFdiffusion

  5. Restart StructureBot.

Reference: Watson et al. (2023) Nature 620:1089-1100
"""

# Windows: prevent child processes from inheriting/corrupting the parent console
_CREATE_NO_WINDOW: int = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


# ── Main class ────────────────────────────────────────────────────────────────

class RFdiffusionBridge:
    """
    RFdiffusion backbone diffusion bridge.

    When RFDIFFUSION_DIR is not set or invalid: returns a helpful stub error.
    When RFDIFFUSION_DIR is valid: runs run_inference.py via subprocess.

    All compute is delegated to a subprocess so RFdiffusion's Python 3.9-3.11
    environment stays isolated from the main StructureBot Python 3.14 runtime.
    """

    def __init__(self) -> None:
        self._dir:       Optional[Path] = Path(_RFDIFFUSION_DIR) if _RFDIFFUSION_DIR else None
        self._script:    Optional[Path] = None
        self._available: bool = self._check_available()

    def _check_available(self) -> bool:
        """
        Return True if RFdiffusion appears to be installed.
        Sets self._script on success.
        """
        if not self._dir or not self._dir.is_dir():
            return False

        script = self._dir / "run_inference.py"
        models = self._dir / "models"

        if script.is_file() and models.is_dir():
            self._script = script
            return True

        return False

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        inputs:  Dict[str, Any],
        session: Any = None,
    ) -> ToolStepResult:
        """
        Run RFdiffusion backbone generation.

        If not yet configured, returns a helpful error message with installation
        instructions.  When active, delegates to the appropriate mode handler.
        """
        if not self._available:
            return ToolStepResult(
                tool    = "rfdiffusion",
                success = False,
                error   = _INSTALL_INSTRUCTIONS,
            )

        mode = inputs.get("mode", "binder").lower()
        pdb_path = inputs.get("pdb_path", "")

        if mode in ("binder", "motif_scaffold") and (
            not pdb_path or not Path(pdb_path).is_file()
        ):
            return ToolStepResult(
                tool    = "rfdiffusion",
                success = False,
                error   = (
                    f"RFdiffusion mode '{mode}' requires a local PDB file.\n"
                    "  Provide pdb_path in tool_inputs, or load the structure first."
                ),
            )

        try:
            return self._run_inference(inputs, session)
        except Exception as exc:
            return ToolStepResult(
                tool    = "rfdiffusion",
                success = False,
                error   = f"RFdiffusion inference failed: {exc}",
            )

    def status(self) -> str:
        """Return a one-line status string for display."""
        if self._available:
            return f"rfdiffusion — {self._dir}"
        if self._dir and self._dir.is_dir():
            return f"directory found ({self._dir}) but run_inference.py/models/ missing"
        return "not configured (set RFDIFFUSION_DIR in .env.local)"

    # ── Internal inference ─────────────────────────────────────────────────────

    def _run_inference(
        self,
        inputs:  Dict[str, Any],
        session: Any,
    ) -> ToolStepResult:
        """Build command and run run_inference.py via subprocess."""
        import time
        t0 = time.perf_counter()

        mode           = inputs.get("mode", "binder").lower()
        pdb_path       = inputs.get("pdb_path", "")
        chain_id       = inputs.get("chain_id", "A")
        hotspots       = inputs.get("hotspot_residues", [])
        num_designs    = int(inputs.get("num_designs", 4))
        num_steps      = int(inputs.get("num_steps", 50))
        symmetry       = inputs.get("symmetry", "")
        partial_T      = float(inputs.get("partial_T", 0.2))
        contigs        = inputs.get("contigs", "")
        model_id       = str(inputs.get("model_id", "1"))

        with tempfile.TemporaryDirectory() as out_dir:
            out_path = Path(out_dir)

            cmd = self._build_cmd(
                mode=mode, pdb_path=pdb_path, chain_id=chain_id,
                hotspots=hotspots, num_designs=num_designs, num_steps=num_steps,
                symmetry=symmetry, partial_T=partial_T, contigs=contigs,
                out_path=out_path,
            )

            subprocess.run(
                cmd, check=True, capture_output=True,
                stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
                cwd=str(self._dir),
            )

            # Collect generated PDB files
            pdb_files = sorted(out_path.rglob("*.pdb"))
            pdb_paths = [str(p) for p in pdb_files]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Build ChimeraX open commands for all generated structures
        viz_cmds: List[str] = []
        viz_exps: List[str] = []
        for i, p in enumerate(pdb_paths, 1):
            viz_cmds.append(f"open {p!r}")
            viz_exps.append(f"Open RFdiffusion design {i}: {Path(p).stem}")

        if pdb_paths:
            viz_cmds.append("rainbow models")
            viz_exps.append("Rainbow-color each design for visual distinction")

        summary = (
            f"RFdiffusion ({mode}): {len(pdb_paths)} backbone(s) generated "
            f"in {elapsed_ms/1000:.1f}s."
        )

        return ToolStepResult(
            tool             = "rfdiffusion",
            success          = True,
            data             = {
                "pdb_paths":   pdb_paths,
                "mode":        mode,
                "num_designs": len(pdb_paths),
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _build_cmd(
        self,
        mode: str,
        pdb_path: str,
        chain_id: str,
        hotspots: List[int],
        num_designs: int,
        num_steps: int,
        symmetry: str,
        partial_T: float,
        contigs: str,
        out_path: Path,
    ) -> List[str]:
        """
        Build the run_inference.py command list for the requested mode.

        RFdiffusion uses Hydra (config) so arguments are passed as key=value
        overrides.  The exact flags depend on the mode.
        """
        # RFdiffusion needs a Python that has its deps installed.
        # Use venv312 if available (has torch), else sys.executable.
        python_exe = (
            _cfg.VENV312_PYTHON
            if Path(_cfg.VENV312_PYTHON).is_file()
            else sys.executable
        )

        cmd = [python_exe, str(self._script)]

        if mode == "binder":
            hs_str = ",".join(f"{chain_id}{r}" for r in hotspots)
            cmd += [
                f"inference.input_pdb={pdb_path}",
                f"inference.num_designs={num_designs}",
                f"diffuser.T={num_steps}",
                f"inference.output_prefix={out_path / 'binder'}",
                f"ppi.hotspot_res=[{hs_str}]",
            ]

        elif mode == "motif_scaffold":
            cmd += [
                f"inference.input_pdb={pdb_path}",
                f"inference.num_designs={num_designs}",
                f"diffuser.T={num_steps}",
                f"inference.output_prefix={out_path / 'scaffold'}",
                f"contigmap.contigs=[{contigs}]",
            ]

        elif mode == "symmetric":
            cmd += [
                f"inference.symmetry={symmetry}",
                f"inference.num_designs={num_designs}",
                f"diffuser.T={num_steps}",
                f"inference.output_prefix={out_path / 'symmetric'}",
            ]

        elif mode == "partial_diffusion":
            cmd += [
                f"inference.input_pdb={pdb_path}",
                f"inference.num_designs={num_designs}",
                f"diffuser.partial_T={int(partial_T * num_steps)}",
                f"inference.output_prefix={out_path / 'partial'}",
            ]

        else:
            raise ValueError(
                f"Unknown RFdiffusion mode '{mode}'. "
                "Valid modes: binder, motif_scaffold, symmetric, partial_diffusion"
            )

        return cmd

    def __repr__(self) -> str:
        return f"<RFdiffusionBridge available={self._available}>"
