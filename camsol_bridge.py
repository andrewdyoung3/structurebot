"""
camsol_bridge.py
----------------
Per-residue solubility scoring based on the CamSol algorithm.

Reference
---------
Sormanni P, Aprile FA, Vendruscolo M (2015).
"The CamSol method of rational design of protein mutants with enhanced solubility."
J Mol Biol 427(2):478–490.

Algorithm summary
-----------------
For each residue i, compute a window-averaged score:

    CamSol(i) = (1/W) * Σ_j∈window [ β·|charge(j)| − hydrophobicity(j) ]

Where:
  - window  W = 9 residues centred at i (clamped at sequence ends)
  - charge(j)        = formal charge at pH 7.4 (D,E → −1; K,R → +1; H → 0.1)
  - hydrophobicity(j) = Kyte–Doolittle hydrophobicity (I=4.5 … R=−4.5)
  - β = 3.0  (weight on charge; empirically balances the two terms)

Raw scores are then mean-centred and σ-normalised so the output is in
units of standard deviations from the mean of this particular sequence:
  positive → more soluble than average
  negative → aggregation-prone

Visualization
-------------
Residues are mapped onto a 5-band colour scale and coloured in ChimeraX:
  deep blue     (> +1.5 σ)  — very soluble
  dodger blue   (+0.5 to +1.5 σ)
  white         (−0.5 to +0.5 σ)  — neutral
  tomato        (−1.5 to −0.5 σ)
  red           (< −1.5 σ)  — aggregation-prone

Web-API fallback
----------------
If a PROTEIN-SOL_URL environment variable is set, the bridge posts to the
Protein-Sol web server (https://protein-sol.manchester.ac.uk/) and parses
its JSON response instead of running the local algorithm.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_router import ToolStepResult


# ── Amino-acid property tables ────────────────────────────────────────────────

# Kyte–Doolittle hydrophobicity scale (original 1982 values)
_KD_HYDROPHOBICITY: Dict[str, float] = {
    "I":  4.5, "V":  4.2, "L":  3.8, "F":  2.8, "C":  2.5,
    "M":  1.9, "A":  1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "D": -3.5,
    "N": -3.5, "Q": -3.5, "E": -3.5, "K": -3.9, "R": -4.5,
}

# Formal charge at pH 7.4
_FORMAL_CHARGE: Dict[str, float] = {
    "D": -1.0, "E": -1.0,          # acidic (deprotonated)
    "K":  1.0, "R":  1.0,          # basic (protonated)
    "H":  0.1,                     # histidine ~10% protonated at pH 7.4
}

# 5-band colour scale (score in σ units → ChimeraX colour name)
_COLOUR_BANDS: List[Tuple[float, float, str]] = [
    (-999.0, -1.5, "red"),
    (-1.5,   -0.5, "tomato"),
    (-0.5,    0.5, "white"),
    (0.5,    1.5,  "dodger blue"),
    (1.5,   999.0, "blue"),
]

# RGB (0-255) for the band colour names, matching ChimeraX's named colours — used
# to mirror the structure colouring onto the Sequence Viewer via an .scf file.
_COLOUR_RGB: Dict[str, Tuple[int, int, int]] = {
    "red":         (255,   0,   0),
    "tomato":      (255,  99,  71),
    "white":       (255, 255, 255),
    "dodger blue": ( 30, 144, 255),
    "blue":        (  0,   0, 255),
}

_CHARGE_WEIGHT = 3.0   # β in the formula above
_WINDOW_SIZE   = 9


# ── Core algorithm ─────────────────────────────────────────────────────────────

def camsol_score(sequence: str, window: int = _WINDOW_SIZE) -> List[float]:
    """
    Compute the CamSol per-residue solubility profile for *sequence*.

    Returns a list of floats (z-scores, one per residue) where:
      positive → more soluble than the sequence average
      negative → aggregation-prone relative to the sequence average
    """
    n    = len(sequence)
    half = window // 2

    raw: List[float] = []
    for i in range(n):
        total = 0.0
        count = 0
        for j in range(max(0, i - half), min(n, i + half + 1)):
            aa      = sequence[j]
            hydro   = _KD_HYDROPHOBICITY.get(aa, 0.0)
            charge  = abs(_FORMAL_CHARGE.get(aa, 0.0))
            # Higher charge = better solubility; higher hydrophobicity = worse
            total  += _CHARGE_WEIGHT * charge - hydro
            count  += 1
        raw.append(total / count if count else 0.0)

    if not raw:
        return raw

    # Mean-centre
    mean    = sum(raw) / len(raw)
    centred = [r - mean for r in raw]

    # σ-normalise
    variance = sum(c * c for c in centred) / len(centred)
    std      = variance ** 0.5
    if std > 1e-9:
        centred = [c / std for c in centred]

    return centred


def _assign_colour(score: float) -> str:
    """Map a z-score to a ChimeraX colour name."""
    for lo, hi, colour in _COLOUR_BANDS:
        if lo <= score < hi:
            return colour
    return "white"


def _build_viz_commands(
    scores:      List[float],
    model_id:    str,
    chain:       Optional[str],
    start_resno: int = 1,
) -> Tuple[List[str], List[str]]:
    """
    Generate compact ChimeraX colour commands for the CamSol results.

    Groups consecutive residues of the same colour into one command, e.g.:
      color #1:1-12 white
      color #1:13,14,15 red
      ...

    Returns (commands, explanations).
    """
    if not scores:
        return [], []

    chain_spec = f"/{chain}" if chain else ""

    # Assign colour for each residue position
    coloured = [(start_resno + i, _assign_colour(s)) for i, s in enumerate(scores)]

    # Group consecutive same-colour runs
    runs: List[Tuple[str, List[int]]] = []   # (colour, [resno, ...])
    for resno, colour in coloured:
        if runs and runs[-1][0] == colour:
            runs[-1][1].append(resno)
        else:
            runs.append((colour, [resno]))

    cmds = [
        f"cartoon #{model_id}",
        f"color #{model_id}{chain_spec} white",   # reset base
    ]
    exps = [
        "Switch to cartoon representation",
        "Reset all residues to white before applying CamSol colours",
    ]

    for colour, resnos in runs:
        if colour == "white":
            continue   # already white from the reset above
        # Use range notation when possible (e.g. :1-50), otherwise comma list
        if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
            spec = f":{resnos[0]}-{resnos[-1]}"
        else:
            spec = ":" + ",".join(str(r) for r in resnos)
        full_spec = f"#{model_id}{chain_spec}{spec}"
        cmds.append(f"color {full_spec} {colour}")
        exps.append(
            f"Color residues {spec} {colour} "
            f"({'aggregation-prone' if colour in ('red','tomato') else 'soluble'})"
        )

    cmds.append(f"view #{model_id}")
    exps.append("Fit structure in view")

    return cmds, exps


# ── Bridge class ───────────────────────────────────────────────────────────────

class CamsolBridge:
    """
    Computes per-residue solubility scores and generates ChimeraX
    visualization commands.

    Two modes (selected automatically):
      - Local algorithm  : always available; pure Python; see camsol_score()
      - Protein-Sol API  : if PROTEIN_SOL_URL env var is set; sends HTTP POST
    """

    def analyze(
        self,
        sequence:  str,
        model_id:  str  = "1",
        chain:     Optional[str] = None,
        session:   Any  = None,
        start_resno: int = 1,
    ) -> ToolStepResult:
        """
        Score *sequence* and generate ChimeraX visualization commands.

        Parameters
        ----------
        sequence    : single-letter amino-acid string (standard AA letters only)
        model_id    : ChimeraX model number string (e.g. "1")
        chain       : chain identifier (e.g. "A") or None for all chains
        session     : SessionState (unused here, reserved for caching)
        start_resno : residue number of the first amino acid (default 1)

        Returns
        -------
        ToolStepResult with:
          data["scores"]         : {residue_number: score}
          data["aggregation_hot_spots"] : residue numbers where score < -1 σ
          data["highly_soluble"]        : residue numbers where score > +1 σ
          viz_commands / viz_explanations : ChimeraX colouring commands
          summary : one-line result description
        """
        # Clean and validate sequence
        sequence = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", sequence.upper())
        if len(sequence) < 5:
            return ToolStepResult(
                tool="camsol", success=False,
                error=(
                    f"Sequence too short ({len(sequence)} residues). "
                    "CamSol requires at least 5 amino acids."
                ),
            )

        # Try web API first if configured
        api_url = os.environ.get("PROTEIN_SOL_URL", "").strip()
        if api_url:
            result = self._run_web_api(sequence, api_url)
            if result is not None:
                scores = result
            else:
                scores = camsol_score(sequence)
        else:
            scores = camsol_score(sequence)

        # Build per-residue dict (1-indexed residue numbers)
        scores_dict = {
            start_resno + i: round(s, 4)
            for i, s in enumerate(scores)
        }

        hot_spots    = [r for r, s in scores_dict.items() if s < -1.0]
        very_soluble = [r for r, s in scores_dict.items() if s > 1.0]

        viz_cmds, viz_exps = _build_viz_commands(
            scores, model_id, chain, start_resno
        )

        # Mirror the same per-residue colouring onto the Sequence Viewer (.scf).
        # Error-first: if anything fails, the structure colouring above stands.
        seq_cmds, seq_exps = self._build_sequence_viewer_viz(
            scores_dict, model_id, chain
        )
        viz_cmds += seq_cmds
        viz_exps += seq_exps

        n_agg = len(hot_spots)
        pct   = 100.0 * n_agg / len(scores) if scores else 0
        summary = (
            f"CamSol: {len(scores)} residues scored. "
            f"{n_agg} aggregation hot-spots ({pct:.0f}%), "
            f"{len(very_soluble)} highly soluble."
        )

        return ToolStepResult(
            tool    = "camsol",
            success = True,
            data    = {
                "scores":              scores_dict,
                "aggregation_hot_spots": hot_spots,
                "highly_soluble":      very_soluble,
                "sequence_length":     len(sequence),
                "algorithm":           "local" if not api_url else "protein-sol-api",
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
        )

    # ── Sequence-Viewer mirror (.scf) ─────────────────────────────────────────
    @staticmethod
    def _build_sequence_viewer_viz(
        scores_dict: Dict[int, float],
        model_id:    str,
        chain:       Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Build commands that mirror the CamSol aggregation gradient onto the
        ChimeraX Sequence Viewer: open + associate the chain, then load an .scf
        of the SAME per-residue colours via a runscript.

        Returns ([commands], [explanations]); ([], []) on any failure so the
        structure colouring is never broken by a sequence-viewer issue.
        """
        try:
            import config as _cfg
            from sequence_viewer import (
                build_scf_file,
                build_scf_runscript,
                ensure_sequence_viewer_commands,
            )

            # Per-residue RGB (skip the neutral white band — nothing to mark).
            residue_colors: Dict[int, Tuple[int, int, int]] = {}
            for resno, score in scores_dict.items():
                name = _assign_colour(score)
                if name == "white":
                    continue
                residue_colors[resno] = _COLOUR_RGB.get(name, (255, 255, 255))
            if not residue_colors:
                return [], []

            ordered_resnums = sorted(scores_dict)        # contiguous chain order
            tag = f"model{model_id}_{chain or 'all'}"
            scf_path = Path(_cfg.SEQVIEW_CACHE_DIR) / f"camsol_{tag}.scf"
            py_path  = Path(_cfg.SEQVIEW_CACHE_DIR) / f"camsol_{tag}.scf.py"

            scf_posix = build_scf_file(
                residue_colors, ordered_resnums, scf_path,
                seq_index=0, region_name="CamSol aggregation",
            )
            run_cmd = build_scf_runscript(scf_posix, py_path)

            cmds = ensure_sequence_viewer_commands(
                model_id, [chain] if chain else None
            )
            cmds.append(run_cmd)
            exps = (
                [f"Open + associate the Sequence Viewer for chain {chain or 'all'}"]
                * len(cmds[:-1])
                + ["Mirror the CamSol colours onto the sequence (.scf regions)"]
            )
            return cmds, exps
        except Exception:
            return [], []

    # ── Protein-Sol web API (optional) ────────────────────────────────────────

    def _run_web_api(
        self,
        sequence: str,
        url:      str,
    ) -> Optional[List[float]]:
        """
        POST sequence to the Protein-Sol web API.
        Returns a list of per-residue scores, or None on any failure.

        The Protein-Sol API (https://protein-sol.manchester.ac.uk/) accepts
        a FASTA sequence and returns JSON with a "scores" array.
        """
        try:
            import requests
            payload  = {"sequence": f">query\n{sequence}"}
            resp     = requests.post(url, data=payload, timeout=30)
            if resp.status_code != 200:
                return None
            data     = resp.json()
            raw      = data.get("scores") or data.get("profile")
            if isinstance(raw, list) and len(raw) == len(sequence):
                return [float(x) for x in raw]
        except Exception:
            pass
        return None

    def __repr__(self) -> str:
        api = os.environ.get("PROTEIN_SOL_URL", "")
        mode = "protein-sol-api" if api else "local-algorithm"
        return f"<CamsolBridge mode={mode!r}>"
