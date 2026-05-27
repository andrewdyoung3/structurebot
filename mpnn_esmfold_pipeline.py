"""
mpnn_esmfold_pipeline.py
------------------------
ProteinMPNN + ESMFold combined redesign and validation pipeline.

Workflow
--------
1. Read ProteinMPNN results from session state (must have been run first).
2. For each of the top-N designed sequences (+ optionally the wildtype),
   run ESMFold to predict the folded structure and obtain per-residue pLDDT.
3. Classify each design as pass / warn / fail based on mean-pLDDT threshold.
4. Generate ChimeraX commands to color mutated positions by pLDDT confidence.
5. Generate a multi-line summary table (Rich Panel-ready).

Critical env rules (same as esmfold_bridge.py):
  - All subprocess calls must use stdin=subprocess.DEVNULL,
    creationflags=subprocess.CREATE_NO_WINDOW.
  - Path args must use Path(...).as_posix().
  - sys.stdout.reconfigure() MUST NOT be used.
  - esmfold_worker.py is NOT modified here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from esmfold_bridge import ESMFoldBridge

if TYPE_CHECKING:
    from session_state import SessionState

# ── pLDDT colour constants (matching AlphaFold conventions) ──────────────────

_PLDDT_HIGH: float = 70.0
_PLDDT_MEDIUM: float = 50.0

_COLOR_HIGH:    str = "#00cc00"   # green  — high confidence
_COLOR_MEDIUM:  str = "#cccc00"   # yellow — medium confidence
_COLOR_LOW:     str = "#cc0000"   # red    — low confidence


# ── Module-level helpers ──────────────────────────────────────────────────────

def _diff_sequences(wt: str, designed: str) -> List[str]:
    """
    Return a list of mutation strings ``'W{pos}D'`` (1-indexed) comparing
    *wt* and *designed*.  Stops at the shorter of the two sequences.
    """
    mutations: List[str] = []
    for i, (a, b) in enumerate(zip(wt, designed)):
        if a != b:
            mutations.append(f"{a}{i + 1}{b}")
    return mutations


def _plddt_confidence(mean_plddt: float) -> str:
    """Classify a mean pLDDT value as ``'high'`` / ``'medium'`` / ``'low'``."""
    if mean_plddt >= _PLDDT_HIGH:
        return "high"
    if mean_plddt >= _PLDDT_MEDIUM:
        return "medium"
    return "low"


# ── Pipeline class ────────────────────────────────────────────────────────────

class MPNNESMFoldPipeline:
    """
    Combined ProteinMPNN + ESMFold redesign and validation pipeline.

    This class is stateless — instantiate once and call ``run()`` as needed.
    All persistent state lives in *session* (a ``SessionState`` instance).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        model_id:         str,
        pdb_path:         Optional[str],
        chain_id:         str,
        session:          "SessionState",
        top_n:            int   = 3,
        plddt_threshold:  float = 70.0,
        include_wildtype: bool  = True,
    ) -> Dict[str, Any]:
        """
        Run ESMFold on the top ProteinMPNN-designed sequences and return
        validated results.

        Parameters
        ----------
        model_id         : ChimeraX model number (string).
        pdb_path         : Local PDB file path (used for context only; the
                           sequence is read from the MPNN result in session).
        chain_id         : Chain identifier (e.g. ``"A"``).
        session          : Active ``SessionState`` instance.
        top_n            : How many MPNN designs to validate (default 3).
        plddt_threshold  : Mean pLDDT ≥ this value → design passes (default 70).
        include_wildtype : If True, also fold the wildtype for baseline comparison.

        Returns
        -------
        Dict with keys:
          success, model_id, chain_id, wildtype_sequence,
          validated_designs, top_n, plddt_threshold,
          passed_count, failed_count, error,
          chimerax_commands, chimerax_explanations, summary.
        """

        # 1. Fetch ProteinMPNN results from session ---------------------------
        mpnn_result = session.get_proteinmpnn_result(model_id)
        if not mpnn_result:
            return {
                "success":            False,
                "error":              (
                    f"No ProteinMPNN results found for model #{model_id}. "
                    "Run ProteinMPNN first."
                ),
                "model_id":           model_id,
                "chain_id":           chain_id,
                "wildtype_sequence":  "",
                "validated_designs":  [],
                "top_n":              top_n,
                "plddt_threshold":    plddt_threshold,
                "passed_count":       0,
                "failed_count":       0,
                "chimerax_commands":  [],
                "chimerax_explanations": [],
                "summary":            "",
            }

        wildtype_sequence: str       = mpnn_result.get("wildtype_sequence", "")
        sequences:         List[Any] = mpnn_result.get("sequences", [])

        if not sequences:
            return {
                "success":            False,
                "error":              "ProteinMPNN results contain no designed sequences.",
                "model_id":           model_id,
                "chain_id":           chain_id,
                "wildtype_sequence":  wildtype_sequence,
                "validated_designs":  [],
                "top_n":              top_n,
                "plddt_threshold":    plddt_threshold,
                "passed_count":       0,
                "failed_count":       0,
                "chimerax_commands":  [],
                "chimerax_explanations": [],
                "summary":            "",
            }

        # 2. Build the list of sequences to fold ------------------------------
        to_fold: List[Dict[str, Any]] = []

        if include_wildtype and wildtype_sequence:
            to_fold.append({
                "rank":          0,
                "sequence":      wildtype_sequence,
                "mpnn_score":    0.0,
                "recovery":      1.0,
                "mutations":     [],
                "is_wildtype":   True,
                "esmfold_label": f"#{model_id}_wildtype",
            })

        for i, seq_entry in enumerate(sequences[:top_n]):
            seq = seq_entry.get("sequence", "")
            mutations = seq_entry.get("mutations", [])
            if not mutations and wildtype_sequence:
                mutations = _diff_sequences(wildtype_sequence, seq)
            to_fold.append({
                "rank":          i + 1,
                "sequence":      seq,
                "mpnn_score":    float(seq_entry.get("score", 0.0)),
                "recovery":      float(seq_entry.get("recovery", 0.0)),
                "mutations":     mutations,
                "is_wildtype":   False,
                "esmfold_label": f"#{model_id}_design_{i + 1}",
            })

        # 3. Run ESMFold on each sequence -------------------------------------
        esmfold = ESMFoldBridge()
        validated_designs: List[Dict[str, Any]] = []

        for entry in to_fold:
            seq   = entry["sequence"]
            label = entry["esmfold_label"]

            if not seq:
                validated_designs.append({
                    **entry,
                    "pdb_str":        "",
                    "mean_plddt":     0.0,
                    "plddt":          {},
                    "pass_threshold": False,
                    "confidence":     "low",
                    "error":          "Empty sequence",
                })
                continue

            fold = esmfold.predict(seq, label=label)

            if not fold.get("success"):
                validated_designs.append({
                    **entry,
                    "pdb_str":        "",
                    "mean_plddt":     0.0,
                    "plddt":          {},
                    "pass_threshold": False,
                    "confidence":     "low",
                    "error":          fold.get("error", "ESMFold prediction failed"),
                })
                continue

            mean_plddt: float            = float(fold.get("mean_plddt", 0.0))
            plddt:      Dict[int, float] = fold.get("plddt", {})

            validated_designs.append({
                **entry,
                "pdb_str":        fold.get("pdb_str", ""),
                "mean_plddt":     mean_plddt,
                "plddt":          plddt,
                "pass_threshold": mean_plddt >= plddt_threshold,
                "confidence":     _plddt_confidence(mean_plddt),
                "error":          None,
            })

        # 4. Aggregate pass / fail counts (designed sequences only) -----------
        designs_only   = [d for d in validated_designs if not d.get("is_wildtype")]
        passed_count   = sum(1 for d in designs_only if d.get("pass_threshold"))
        failed_count   = len(designs_only) - passed_count

        # 5. Generate ChimeraX commands and summary ---------------------------
        cx_cmds, cx_exps = self.generate_chimerax_commands(
            validated_designs, model_id=model_id, chain_id=chain_id
        )
        summary = self.generate_summary(validated_designs, model_id=model_id)

        return {
            "success":               True,
            "model_id":              model_id,
            "chain_id":              chain_id,
            "wildtype_sequence":     wildtype_sequence,
            "validated_designs":     validated_designs,
            "top_n":                 top_n,
            "plddt_threshold":       plddt_threshold,
            "passed_count":          passed_count,
            "failed_count":          failed_count,
            "error":                 None,
            "chimerax_commands":     cx_cmds,
            "chimerax_explanations": cx_exps,
            "summary":               summary,
        }

    # ------------------------------------------------------------------

    def generate_chimerax_commands(
        self,
        validated_designs: List[Dict[str, Any]],
        model_id:          str,
        chain_id:          str = "A",
    ) -> Tuple[List[str], List[str]]:
        """
        Generate ChimeraX commands to color mutated positions of the best
        passing design by ESMFold pLDDT confidence.

        Color scheme:
          pLDDT ≥ 70 → green  (#00cc00)
          pLDDT ≥ 50 → yellow (#cccc00)
          pLDDT < 50 → red    (#cc0000)

        Three commands per mutated residue:
          1. ``show <spec> atoms``
          2. ``color <spec> <hex>``
          3. ``label <spec> text "FromPosTo" height 0.5``

        Returns
        -------
        (commands, explanations) — parallel lists.
        """
        cmds: List[str] = []
        exps: List[str] = []

        designs_only = [d for d in validated_designs if not d.get("is_wildtype")]
        if not designs_only:
            return cmds, exps

        # Use the highest-ranked passing design; fall back to rank-1 design
        passing = [d for d in designs_only if d.get("pass_threshold")]
        best    = passing[0] if passing else designs_only[0]

        mutations: List[str]     = best.get("mutations", [])
        plddt:     Dict[int, float] = best.get("plddt", {})

        if not mutations:
            return cmds, exps

        _MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")

        for mut_str in mutations:
            m = _MUT_RE.match(mut_str)
            if not m:
                continue
            from_aa = m.group(1)
            pos     = int(m.group(2))
            to_aa   = m.group(3)

            pos_plddt = float(plddt.get(pos, 0.0))

            if pos_plddt >= _PLDDT_HIGH:
                color = _COLOR_HIGH
                conf  = "high"
            elif pos_plddt >= _PLDDT_MEDIUM:
                color = _COLOR_MEDIUM
                conf  = "medium"
            else:
                color = _COLOR_LOW
                conf  = "low"

            spec = f"#{model_id}/{chain_id}:{pos}"

            cmds.append(f"show {spec} atoms")
            exps.append(f"{mut_str} — show atoms")

            cmds.append(f"color {spec} {color}")
            exps.append(
                f"{mut_str} — color by pLDDT confidence "
                f"({conf}, pLDDT={pos_plddt:.1f})"
            )

            cmds.append(
                f'label {spec} text "{from_aa}{pos}{to_aa}" height 0.5'
            )
            exps.append(f"{mut_str} — label {from_aa}→{to_aa} at position {pos}")

        return cmds, exps

    # ------------------------------------------------------------------

    def generate_summary(
        self,
        validated_designs: List[Dict[str, Any]],
        model_id:          str,
    ) -> str:
        """
        Build a multi-line Rich-Panel-ready summary table.

        The first line is always the header, so ``"\\n" in summary`` is
        guaranteed when there is at least one design — which ensures
        ``main.py`` renders the full Rich Panel.
        """
        lines: List[str] = [
            f"MPNN + ESMFold pipeline — model #{model_id}",
            "",
        ]

        wt_entry  = next((d for d in validated_designs if d.get("is_wildtype")), None)
        designs   = [d for d in validated_designs if not d.get("is_wildtype")]

        if wt_entry and not wt_entry.get("error"):
            conf = wt_entry.get("confidence", "?")
            lines.append(
                f"Wildtype  pLDDT: {wt_entry['mean_plddt']:.1f}"
                f"  ({conf} confidence)"
            )
            lines.append("")

        if not designs:
            lines.append("No designed sequences to display.")
            return "\n".join(lines)

        # Column header
        lines.append(
            f"{'Rank':<5} {'pLDDT':>6} {'Conf':<8} "
            f"{'Score':>7} {'Recovery':>9}  {'Mutations':<38} {'Pass'}"
        )
        lines.append("─" * 85)

        for d in designs:
            rank     = str(d.get("rank", "?"))
            if d.get("error"):
                plddt_s = "ERROR"
                conf    = "—"
            else:
                plddt_s = f"{d['mean_plddt']:.1f}"
                conf    = d.get("confidence", "?")
            score    = f"{d.get('mpnn_score', 0.0):.3f}"
            recovery = f"{d.get('recovery', 0.0) * 100:.1f}%"
            muts     = d.get("mutations", [])
            muts_str = ", ".join(muts[:5])
            if len(muts) > 5:
                muts_str += f" +{len(muts) - 5} more"
            passed   = "✓" if d.get("pass_threshold") else "✗"

            lines.append(
                f"{rank:<5} {plddt_s:>6} {conf:<8} "
                f"{score:>7} {recovery:>9}  {muts_str:<38} {passed}"
            )

        passed_count = sum(1 for d in designs if d.get("pass_threshold"))
        lines.append("─" * 85)
        lines.append(f"{passed_count}/{len(designs)} design(s) passed pLDDT threshold")

        return "\n".join(lines)
