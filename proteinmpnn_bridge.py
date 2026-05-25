"""
proteinmpnn_bridge.py
---------------------
ProteinMPNN integration stub for StructureBot.

ProteinMPNN (Dauparas et al., 2022) is a graph neural network that predicts
optimal amino-acid sequences for a given protein backbone.  It is the standard
tool for fixed-backbone sequence design in structural biology.

Current status: NOT YET CONFIGURED
-----------------------------------
This module is a documented stub. It returns a helpful error message that
guides the user through installation.

To activate full ProteinMPNN support:

1.  Install PyTorch (≥ 2.0):
      pip install torch --index-url https://download.pytorch.org/whl/cu118

2.  Clone the ProteinMPNN repository:
      git clone https://github.com/dauparas/ProteinMPNN
      cd ProteinMPNN

3.  Set the PROTEINMPNN_DIR environment variable in .env.local:
      PROTEINMPNN_DIR=C:/path/to/ProteinMPNN

4.  Download model weights (included in the repo under vanilla_model_weights/).

Once PROTEINMPNN_DIR is set and the weights exist, this bridge will
automatically activate and the analyze() method will run inference.

Interface (future)
------------------
analyze(inputs, session) → ToolStepResult
  inputs keys:
    model_id     : ChimeraX model number (e.g. "1")
    chain        : chain to redesign (e.g. "A") or None for all
    fixed_residues : list of residue numbers to keep fixed
    n_seqs       : number of sequences to generate (default 8)
    temperature  : sampling temperature (default 0.1 — lower = more conservative)

  Returns:
    data["sequences"]    : list of designed sequences
    data["scores"]       : negative log-likelihood per sequence (lower = better)
    data["recovery"]     : fraction of WT residues recovered per sequence
    viz_commands         : ChimeraX commands to colour by per-residue recovery
    summary              : e.g. "ProteinMPNN: 8 sequences, mean recovery 0.42"

Reference
---------
Dauparas J, Anishchenko I, Bennett N, et al. (2022).
"Robust deep learning–based protein sequence design using ProteinMPNN."
Science 378(6615):49–56.  https://doi.org/10.1126/science.add2187
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from tool_router import ToolStepResult

_PROTEINMPNN_DIR = os.environ.get("PROTEINMPNN_DIR", "").strip()

_NOT_CONFIGURED_MSG = """\
ProteinMPNN is not yet configured.

To enable it:
  1. pip install torch --index-url https://download.pytorch.org/whl/cu118
  2. git clone https://github.com/dauparas/ProteinMPNN
  3. Add to .env.local:  PROTEINMPNN_DIR=C:/path/to/ProteinMPNN
  4. Restart StructureBot

Reference: Dauparas et al. (2022) Science 378:49-56.
"""


class ProteinMPNNBridge:
    """
    ProteinMPNN sequence design bridge (stub).

    Returns a helpful 'not configured' message until PROTEINMPNN_DIR is set
    and the required weights are present.
    """

    def __init__(self) -> None:
        self._dir = Path(_PROTEINMPNN_DIR) if _PROTEINMPNN_DIR else None
        self._available = self._check_available()

    def _check_available(self) -> bool:
        """Return True if ProteinMPNN appears to be installed and configured."""
        if not self._dir:
            return False
        weights = self._dir / "vanilla_model_weights"
        return self._dir.is_dir() and weights.is_dir()

    def analyze(
        self,
        inputs:  Dict[str, Any],
        session: Any = None,
    ) -> ToolStepResult:
        """
        Run ProteinMPNN sequence design.

        If not yet configured, returns a helpful error message.
        If PROTEINMPNN_DIR is set and valid, runs inference (future implementation).
        """
        if not self._available:
            return ToolStepResult(
                tool    = "proteinmpnn",
                success = False,
                error   = _NOT_CONFIGURED_MSG,
            )

        # ── Full implementation goes here when PROTEINMPNN_DIR is set ────────
        # This section will be implemented in a future session.
        # For now, even with a valid dir, we return a "coming soon" message.
        return ToolStepResult(
            tool    = "proteinmpnn",
            success = False,
            error   = (
                f"ProteinMPNN directory found at {self._dir}, "
                "but full inference integration is not yet implemented in StructureBot. "
                "This will be added in a future update."
            ),
        )

    def status(self) -> str:
        """Return a one-line status string for display."""
        if self._available:
            return f"configured — {self._dir}"
        if self._dir:
            return f"directory set ({self._dir}) but weights not found"
        return "not configured (set PROTEINMPNN_DIR in .env.local)"

    def __repr__(self) -> str:
        return f"<ProteinMPNNBridge available={self._available}>"
