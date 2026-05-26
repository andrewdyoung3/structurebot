"""
proteinmpnn_bridge.py
---------------------
ProteinMPNN / LigandMPNN integration for StructureBot.

ProteinMPNN (Dauparas et al., 2022) is a graph neural network that predicts
optimal amino-acid sequences for a given protein backbone.  It is the standard
tool for fixed-backbone sequence design in structural biology.

LigandMPNN (Dauparas et al., 2023) extends ProteinMPNN to handle small-molecule
ligands — ideal for redesigning binding-site residues.

Installation
------------
ProteinMPNN is not pip-installable.  Clone the repo and set PROTEINMPNN_DIR:

  git clone https://github.com/dauparas/ProteinMPNN
  # In .env.local:
  PROTEINMPNN_DIR=C:/path/to/ProteinMPNN

OR for LigandMPNN (recommended if your structure has ligands):

  git clone https://github.com/dauparas/LigandMPNN
  PROTEINMPNN_DIR=C:/path/to/LigandMPNN

Both repos expect model weights in <repo>/vanilla_model_weights/ or
<repo>/model_params/ (LigandMPNN).

This bridge invokes the inference script via subprocess:
  python protein_mpnn_run.py --pdb_path ... --chain_id_jsonl ... --out_folder ...

Interface
---------
analyze(inputs, session) → ToolStepResult
  inputs keys:
    model_id         : ChimeraX model number (e.g. "1")
    pdb_path         : local .pdb file path
    chain_id         : chain to redesign (e.g. "A") — others are fixed
    fixed_positions  : list of residue numbers to keep unchanged
                       (active site, interface residues)
    num_sequences    : number of sequences to generate (default 8)
    temperature      : sampling temperature
                       0.1 = conservative, 0.5 = diverse (default 0.1)
    batch_size       : sequences per forward pass (default 1)

  Returns:
    data["sequences"]       : list of designed sequence dicts
    data["wildtype_sequence"]: WT sequence string
    data["fixed_positions"] : list of fixed residue numbers
    data["backend"]         : "proteinmpnn" or "ligandmpnn"
    viz_commands            : ChimeraX commands to colour by per-residue recovery
    summary                 : e.g. "ProteinMPNN: 8 sequences, mean recovery 0.82"

Output schema per sequence:
  {
    "sequence"  : "ACDEFG...",
    "score"     : -1.23,      # log-likelihood, lower = better
    "recovery"  : 0.87,       # fraction identical to wildtype
    "mutations" : ["P9E", "I50K", ...]  # vs wildtype
  }

References
----------
Dauparas J, Anishchenko I, Bennett N, et al. (2022).
"Robust deep learning–based protein sequence design using ProteinMPNN."
Science 378(6615):49–56.  https://doi.org/10.1126/science.add2187

Dauparas J et al. (2023). "Atomic context-conditioned protein sequence design
using LigandMPNN."  biorxiv.  https://doi.org/10.1101/2023.12.22.573103
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import config as _cfg
from tool_router import ToolStepResult

# ── Configuration ─────────────────────────────────────────────────────────────

_PROTEINMPNN_DIR = _cfg.PROTEINMPNN_DIR.strip()

# Use venv312 python (has PyTorch) for subprocess inference.
# Falls back to sys.executable if venv312 is missing.
_PYTHON_EXE: str = (
    _cfg.VENV312_PYTHON
    if Path(_cfg.VENV312_PYTHON).is_file()
    else sys.executable
)

# Windows: CREATE_NO_WINDOW prevents child process from corrupting parent console
_CREATE_NO_WINDOW: int = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

_INSTALL_INSTRUCTIONS = """\
ProteinMPNN is not yet configured.

To enable it:
  1. Clone the repo:
       git clone https://github.com/dauparas/ProteinMPNN
     OR for ligand-aware design:
       git clone https://github.com/dauparas/LigandMPNN

  2. Add to .env.local:
       PROTEINMPNN_DIR=C:/path/to/ProteinMPNN

  3. Download model weights (included in the repo under vanilla_model_weights/).

  4. Restart StructureBot.

References:
  ProteinMPNN: Dauparas et al. (2022) Science 378:49-56
  LigandMPNN:  Dauparas et al. (2023) biorxiv 2023.12.22
"""


# ── Wildtype comparison ───────────────────────────────────────────────────────

def _diff_sequences(wt: str, designed: str) -> List[str]:
    """
    Return list of mutations in format "W{pos}D" comparing wt to designed.
    Positions are 1-indexed.
    """
    mutations: List[str] = []
    for i, (w, d) in enumerate(zip(wt, designed), 1):
        if w != d:
            mutations.append(f"{w}{i}{d}")
    return mutations


def _sequence_recovery(wt: str, designed: str) -> float:
    """Fraction of designed residues identical to wildtype."""
    if not wt or not designed:
        return 0.0
    matches = sum(w == d for w, d in zip(wt, designed))
    return round(matches / max(len(wt), len(designed)), 4)


# ── Visualization ─────────────────────────────────────────────────────────────

def _build_recovery_viz(
    wt_sequence: str,
    top_sequence: str,
    model_id: str,
    chain_id: Optional[str],
) -> Tuple[List[str], List[str]]:
    """
    Colour residues by per-position recovery (conserved vs mutated).
    Blue = conserved (same as WT), red = mutated.
    """
    chain_spec = f"/{chain_id}" if chain_id else ""
    cmds = [
        f"cartoon #{model_id}",
        f"color #{model_id}{chain_spec} white",
    ]
    exps = [
        "Switch to cartoon representation",
        "Reset all residues to white before colouring by ProteinMPNN recovery",
    ]

    conserved: List[int] = []
    mutated:   List[int] = []
    for i, (w, d) in enumerate(zip(wt_sequence, top_sequence), 1):
        if w == d:
            conserved.append(i)
        else:
            mutated.append(i)

    if conserved:
        spec = ":" + ",".join(str(r) for r in conserved)
        cmds.append(f"color #{model_id}{chain_spec}{spec} cornflower blue")
        exps.append(f"Blue: {len(conserved)} conserved residues (same as WT)")

    if mutated:
        spec = ":" + ",".join(str(r) for r in mutated)
        cmds.append(f"color #{model_id}{chain_spec}{spec} tomato")
        exps.append(f"Red: {len(mutated)} redesigned residues (differ from WT)")

    cmds.append(f"view #{model_id}")
    exps.append("Fit structure in view")

    return cmds, exps


# ── Main class ────────────────────────────────────────────────────────────────

class ProteinMPNNBridge:
    """
    ProteinMPNN / LigandMPNN sequence design bridge.

    When PROTEINMPNN_DIR is not set: returns a helpful error message.
    When PROTEINMPNN_DIR is set and valid: runs inference via subprocess.
    """

    def __init__(self) -> None:
        self._dir:       Optional[Path] = Path(_PROTEINMPNN_DIR) if _PROTEINMPNN_DIR else None
        self._backend:   Optional[str] = None   # "proteinmpnn" | "ligandmpnn"
        self._script:    Optional[Path] = None
        self._available: bool = self._check_available()

    def _check_available(self) -> bool:
        """
        Return True if ProteinMPNN or LigandMPNN appears to be installed.
        Sets self._backend and self._script on success.
        """
        if not self._dir or not self._dir.is_dir():
            return False

        # ProteinMPNN: protein_mpnn_run.py + vanilla_model_weights/
        script_pmpnn  = self._dir / "protein_mpnn_run.py"
        weights_pmpnn = self._dir / "vanilla_model_weights"
        if script_pmpnn.is_file() and weights_pmpnn.is_dir():
            self._backend = "proteinmpnn"
            self._script  = script_pmpnn
            return True

        # LigandMPNN: run.py + model_params/
        script_lmpnn  = self._dir / "run.py"
        weights_lmpnn = self._dir / "model_params"
        if script_lmpnn.is_file() and weights_lmpnn.is_dir():
            self._backend = "ligandmpnn"
            self._script  = script_lmpnn
            return True

        return False

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        inputs:  Dict[str, Any],
        session: Any = None,
    ) -> ToolStepResult:
        """
        Run ProteinMPNN / LigandMPNN sequence design.

        If not yet configured, returns a helpful error message.
        """
        if not self._available:
            return ToolStepResult(
                tool    = "proteinmpnn",
                success = False,
                error   = _INSTALL_INSTRUCTIONS,
            )

        pdb_path = inputs.get("pdb_path", "")
        if not pdb_path or not Path(pdb_path).is_file():
            return ToolStepResult(
                tool    = "proteinmpnn",
                success = False,
                error   = (
                    "ProteinMPNN requires a local PDB file.\n"
                    "  Provide pdb_path in tool_inputs, or load the structure first."
                ),
            )

        chain_id        = inputs.get("chain_id") or inputs.get("chain", "A")
        fixed_positions = inputs.get("fixed_positions", [])
        num_sequences   = int(inputs.get("num_sequences", 8))
        temperature     = float(inputs.get("temperature", 0.1))

        try:
            return self._run_inference(
                pdb_path        = Path(pdb_path).resolve().as_posix(),  # absolute + POSIX
                chain_id        = chain_id,
                fixed_positions = fixed_positions,
                num_sequences   = num_sequences,
                temperature     = temperature,
                session         = session,
                model_id        = str(inputs.get("model_id", "1")),
            )
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace").strip()
            detail = stderr or stdout or "(no output captured)"
            return ToolStepResult(
                tool    = "proteinmpnn",
                success = False,
                error   = (
                    f"ProteinMPNN subprocess failed (exit {exc.returncode}):\n"
                    f"{detail}"
                ),
            )
        except Exception as exc:
            return ToolStepResult(
                tool    = "proteinmpnn",
                success = False,
                error   = f"ProteinMPNN inference failed: {exc}",
            )

    def status(self) -> str:
        """Return a one-line status string for display."""
        if self._available:
            return f"{self._backend} — {self._dir}"
        if self._dir:
            return f"directory set ({self._dir}) but weights/script not found"
        return "not configured (set PROTEINMPNN_DIR in .env.local)"

    # ── Internal inference ─────────────────────────────────────────────────────

    def _run_inference(
        self,
        pdb_path:        str,
        chain_id:        str,
        fixed_positions: List[int],
        num_sequences:   int,
        temperature:     float,
        session:         Any,
        model_id:        str,
    ) -> ToolStepResult:
        """Run ProteinMPNN via subprocess and parse the FASTA output."""
        import time
        t0 = time.perf_counter()

        with tempfile.TemporaryDirectory() as out_dir:
            out_path = Path(out_dir)

            if self._backend == "proteinmpnn":
                result_data = self._run_proteinmpnn(
                    pdb_path, chain_id, fixed_positions,
                    num_sequences, temperature, out_path,
                )
            else:
                result_data = self._run_ligandmpnn(
                    pdb_path, chain_id, fixed_positions,
                    num_sequences, temperature, out_path,
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        sequences  = result_data.get("sequences", [])
        wt_seq     = result_data.get("wildtype_sequence", "")
        n_seqs     = len(sequences)
        mean_rec   = (
            sum(s["recovery"] for s in sequences) / n_seqs
            if sequences else 0
        )

        # Cache in session
        if session is not None:
            try:
                session.add_proteinmpnn_result(model_id, result_data)
            except AttributeError:
                pass

        # Build visualization for top sequence
        viz_cmds: List[str] = []
        viz_exps: List[str] = []
        if sequences and wt_seq:
            top_seq = sequences[0]["sequence"]
            viz_cmds, viz_exps = _build_recovery_viz(wt_seq, top_seq, model_id, chain_id)

        summary = (
            f"{self._backend.replace('mpnn', 'MPNN')}: "
            f"{n_seqs} sequence(s) designed. "
            f"Mean WT recovery {mean_rec:.2f}. "
            f"Top score {sequences[0]['score']:.3f}." if sequences else
            f"{self._backend}: No sequences generated."
        )

        return ToolStepResult(
            tool             = "proteinmpnn",
            success          = True,
            data             = result_data,
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_proteinmpnn(
        self,
        pdb_path:        str,
        chain_id:        str,
        fixed_positions: List[int],
        num_sequences:   int,
        temperature:     float,
        out_path:        Path,
    ) -> Dict[str, Any]:
        """
        Call protein_mpnn_run.py and parse the FASTA output.

        ProteinMPNN writes results to {out_path}/seqs/{pdb_stem}.fa
        """
        # Resolve to absolute, then convert to POSIX forward-slash form.
        # ProteinMPNN uses rfind("/") to extract the stem; on Windows its path
        # detection breaks unless forward slashes are used throughout.
        pdb_path  = Path(pdb_path).resolve().as_posix()
        pdb_stem  = Path(pdb_path).stem
        seqs_dir  = out_path / "seqs"
        seqs_dir.mkdir(parents=True, exist_ok=True)

        # Build fixed-positions JSON
        fixed_json_path: Optional[str] = None
        if fixed_positions:
            # ProteinMPNN format: {"pdb_stem": {"chain": [pos1, pos2, ...]}}
            fixed_data = {
                pdb_stem: {chain_id: [int(p) for p in fixed_positions]}
            }
            fixed_json_path = (out_path / "fixed_positions.jsonl").as_posix()
            with open(fixed_json_path, "w") as fh:
                fh.write(json.dumps(fixed_data) + "\n")

        # ProteinMPNN auto-detects weights via rfind("/") which fails on Windows
        # (backslash paths → rfind returns -1 → truncates filename).
        # Supply --path_to_model_weights explicitly, with a trailing slash.
        weights_dir = (self._dir / "vanilla_model_weights").as_posix() + "/"

        cmd = [
            _PYTHON_EXE,
            str(self._script),
            "--pdb_path",              pdb_path,
            "--out_folder",            out_path.as_posix(),
            "--num_seq_per_target",    str(num_sequences),
            "--sampling_temp",         str(temperature),
            "--pdb_path_chains",       chain_id,   # ProteinMPNN flag (not --chains_to_design)
            "--path_to_model_weights", weights_dir,
        ]
        if fixed_json_path:
            cmd += ["--fixed_positions_jsonl", fixed_json_path]

        subprocess.run(
            cmd, check=True, capture_output=True,
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
            cwd=str(self._dir),
        )

        # Parse output FASTA
        fa_path = seqs_dir / f"{pdb_stem}.fa"
        if not fa_path.is_file():
            raise RuntimeError(f"ProteinMPNN output FASTA not found: {fa_path}")

        return self._parse_proteinmpnn_fasta(fa_path)

    def _run_ligandmpnn(
        self,
        pdb_path:        str,
        chain_id:        str,
        fixed_positions: List[int],
        num_sequences:   int,
        temperature:     float,
        out_path:        Path,
    ) -> Dict[str, Any]:
        """
        Call LigandMPNN run.py and parse the FASTA output.

        LigandMPNN writes results to {out_path}/seqs/{pdb_stem}.fa
        """
        cmd = [
            _PYTHON_EXE,
            str(self._script),
            "--pdb_path", pdb_path,
            "--out_folder", str(out_path),
            "--number_of_batches", str(num_sequences),
            "--temperature", str(temperature),
            "--chains_to_design", chain_id,
        ]

        subprocess.run(
            cmd, check=True, capture_output=True,
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
            cwd=str(self._dir),
        )

        pdb_stem = Path(pdb_path).stem
        seqs_dir = out_path / "seqs"
        fa_path  = seqs_dir / f"{pdb_stem}.fa"
        if not fa_path.is_file():
            raise RuntimeError(f"LigandMPNN output FASTA not found: {fa_path}")

        return self._parse_proteinmpnn_fasta(fa_path)

    def _parse_proteinmpnn_fasta(self, fa_path: Path) -> Dict[str, Any]:
        """
        Parse ProteinMPNN / LigandMPNN FASTA output.

        Format:
            >score={score}, global_score={gs}, ...
            SEQUENCE...

        First entry is the wildtype (score line starts with WT).
        """
        text = fa_path.read_text(encoding="utf-8")
        entries: List[Tuple[str, str]] = []  # [(header, sequence)]
        current_header = ""
        current_seq:   List[str] = []

        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_seq:
                    entries.append((current_header, "".join(current_seq)))
                current_header = line[1:]
                current_seq    = []
            else:
                current_seq.append(line)
        if current_header and current_seq:
            entries.append((current_header, "".join(current_seq)))

        if not entries:
            return {"sequences": [], "wildtype_sequence": "", "fixed_positions": []}

        # First entry is the wildtype
        wt_header, wt_sequence = entries[0]
        wt_sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", wt_sequence.upper())

        # Parse designed sequences
        parsed: List[Dict[str, Any]] = []
        for header, seq in entries[1:]:
            seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", seq.upper())
            # Extract score from header: score=-1.23
            score_match = re.search(r"score=([-\d.]+)", header)
            score = float(score_match.group(1)) if score_match else 0.0

            recovery = _sequence_recovery(wt_sequence, seq)
            mutations = _diff_sequences(wt_sequence, seq)

            parsed.append({
                "sequence":  seq,
                "score":     round(score, 4),
                "recovery":  recovery,
                "mutations": mutations,
            })

        # Sort by score ascending (lower = better log-likelihood)
        parsed.sort(key=lambda x: x["score"])

        return {
            "sequences":         parsed,
            "wildtype_sequence": wt_sequence,
            "fixed_positions":   [],  # would need to be passed through
            "backend":           self._backend or "proteinmpnn",
        }

    def __repr__(self) -> str:
        return (
            f"<ProteinMPNNBridge backend={self._backend!r} "
            f"available={self._available}>"
        )
