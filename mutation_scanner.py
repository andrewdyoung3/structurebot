"""
mutation_scanner.py
-------------------
Full CamSol + ESM + Rosetta mutation screening pipeline.

Given a loaded protein structure, MutationScanner:

  1. Retrieves or runs CamSol solubility scores
  2. Retrieves or runs ESM-2 evolutionary conservation scores
  3. Identifies candidate positions:
       CamSol score < threshold  (aggregation-prone region)
       ESM conservation < threshold  (evolutionarily tolerant — safe to mutate)
       Not within a protected binding-site residue list (if supplied)
  4. At each candidate position, generates amino-acid substitution candidates:
       — Prefers charged / hydrophilic replacements for solubility focus
       — Excludes Pro (helix-breaker) and Cys (disulfide risk)
       — Ranks up to 3 candidates per position by estimated physicochemical gain
  5. Runs Rosetta ddG on all (position, substitution) pairs
  6. Computes a combined engineering score:
         score = w_ddg × (−ddg) + w_sol × camsol_delta + w_tol × esm_tolerance
       where
         w_ddg = 0.5, w_sol = 0.3, w_tol = 0.2  (defaults, configurable)
         esm_tolerance = 1 − conservation_score
         camsol_delta  = estimated CamSol improvement from hydrophobicity change
  7. Returns ranked mutation candidates with full scores + ChimeraX commands

Top candidates are coloured on a gradient; top-5 positions get sphere markers
and text labels to stand out against the ribbon background.

Scoring reference
-----------------
Kyte-Doolittle hydrophobicity values used for camsol_delta estimation.
ddG interpretation: negative = stabilising, positive = destabilising.
Combined score: higher = better overall engineering outcome.
"""

from __future__ import annotations

import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Physicochemical tables ─────────────────────────────────────────────────────

# Kyte-Doolittle hydrophobicity (higher = more hydrophobic)
_HYDROPHOBICITY: Dict[str, float] = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Residues that improve surface solubility (ordered by preference)
_SOLUBILITY_DONORS: List[str] = [
    "K",  # lysine  — strongly positive, very soluble
    "R",  # arginine — strongly positive
    "E",  # glutamate — negative, very soluble
    "D",  # aspartate — negative, soluble
    "N",  # asparagine — polar uncharged
    "Q",  # glutamine — polar uncharged
    "S",  # serine — small polar
    "T",  # threonine — small polar
]

# Residues to always exclude from substitution candidates
_EXCLUDED_SUBSTITUTIONS: set = {
    "P",  # proline — helix breaker, introduces kink
    "C",  # cysteine — free SH may form unwanted disulfides
}

# Scaling factor to convert hydrophobicity delta → approximate CamSol delta
_HYDRO_TO_CAMSOL: float = 0.25

# Combined-score weights (must sum to 1.0)
_W_DDG = 0.50   # Rosetta ddG contribution
_W_SOL = 0.30   # solubility improvement contribution
_W_TOL = 0.20   # evolutionary tolerance contribution


# ── Score helpers ──────────────────────────────────────────────────────────────

def _estimate_camsol_delta(from_aa: str, to_aa: str) -> float:
    """
    Estimate the change in CamSol score for a single residue substitution.

    Uses hydrophobicity difference as a proxy:
      Δ = (hydro(from) − hydro(to)) × scale_factor

    Positive delta → mutation improves solubility.
    """
    h_from = _HYDROPHOBICITY.get(from_aa, 0.0)
    h_to   = _HYDROPHOBICITY.get(to_aa,   0.0)
    return round((h_from - h_to) * _HYDRO_TO_CAMSOL, 3)


def combined_score(
    ddg:            float,
    camsol_delta:   float,
    esm_tolerance:  float,
    w_ddg: float = _W_DDG,
    w_sol: float = _W_SOL,
    w_tol: float = _W_TOL,
) -> float:
    """
    Compute combined engineering score (higher = better).

    Parameters
    ----------
    ddg           : Rosetta ΔΔG in kcal/mol (negative = stabilising)
    camsol_delta  : estimated solubility improvement (positive = better)
    esm_tolerance : 1 − conservation_score  (higher = more tolerant)
    """
    return round(
        w_ddg * (-ddg)
        + w_sol * camsol_delta
        + w_tol * esm_tolerance,
        4,
    )


def effective_weights(
    use_ddg: bool,
    w_ddg: float = _W_DDG,
    w_sol: float = _W_SOL,
    w_tol: float = _W_TOL,
) -> Tuple[float, float, float]:
    """
    Return the weights to use for the combined score given which voters are active.

    Triage→validate tiering: the DEFAULT (fast) tier runs CamSol + ESM only and
    drops the ddG voter; its weight is RENORMALISED across the remaining voters so
    the score stays on the same scale (default 0.3/0.2 → 0.6/0.4).  When the ddG
    voter is active (opt-in Rosetta deep tier) the original weights are used.

    Structured so additional fast-tier voters (ThermoMPNN / RaSP) can join later:
    set ddG inactive, add their weights to the active set, and renormalise the same
    way.
    """
    if use_ddg:
        return w_ddg, w_sol, w_tol
    total = w_sol + w_tol
    if total <= 0:
        return 0.0, 0.0, 0.0
    return 0.0, round(w_sol / total, 4), round(w_tol / total, 4)


def _recommendation(
    ddg:           Optional[float],
    camsol_delta:  float,
    esm_tolerance: float,
    score:         float,
) -> str:
    """Generate a human-readable recommendation for a candidate mutation.

    *ddg* may be None (fast/default tier — Rosetta not run): the stability clause
    is omitted rather than fabricated.
    """
    parts: List[str] = []

    if ddg is None:
        parts.append("stability not computed (opt in with 'rosetta')")
    elif ddg <= -1.0:
        parts.append("strongly stabilising")
    elif ddg < 0:
        parts.append("mildly stabilising")
    elif ddg < 1.0:
        parts.append("neutral stability")
    else:
        parts.append("mildly destabilising")

    if camsol_delta >= 1.0:
        parts.append("significantly improves solubility")
    elif camsol_delta >= 0.3:
        parts.append("improves solubility")

    if esm_tolerance >= 0.8:
        parts.append("highly tolerated evolutionarily")
    elif esm_tolerance >= 0.6:
        parts.append("evolutionarily tolerated")
    else:
        parts.append("evolutionary tolerance moderate")

    if score >= 1.5:
        label = "⭐ Strong candidate"
    elif score >= 0.5:
        label = "Good candidate"
    else:
        label = "Weak candidate"

    return f"{label} — {', '.join(parts)}"


# ══════════════════════════════════════════════════════════════════════════════
# Main scanner class
# ══════════════════════════════════════════════════════════════════════════════

class MutationScanner:
    """
    Full CamSol + ESM + Rosetta mutation screening pipeline.

    Instantiate once and reuse across multiple scan() calls::

        scanner = MutationScanner(session=session_state)
        results = scanner.scan(pdb_path="cache/1HSG.pdb", chain_id="A")
        cmds    = scanner.generate_chimerax_commands(results, top_n=5)
    """

    def __init__(
        self,
        session:           Any,
        model_id:          str = "1",
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.session  = session
        self.model_id = model_id
        def _safe_print(msg: str) -> None:
            try:
                print(msg, flush=True)
            except UnicodeEncodeError:
                print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)

        self._progress = progress_callback or _safe_print

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(
        self,
        pdb_path:           str,
        chain_id:           Optional[str] = None,
        sequence:           Optional[str] = None,
        filters:            Optional[Dict[str, Any]] = None,
        protected_residues: Optional[List[int]] = None,
        analysis_mode:      str = "monomer",
        scan_timeout:       int = 600,      # overall wall-clock budget (seconds)
        include_positions:  Optional[List[int]] = None,
        run_rosetta:        bool = True,
        rosetta_shortlist_k: Optional[int] = None,
        ddg_basis:          str = "symmetric",
    ) -> List[Dict[str, Any]]:
        """
        Run the full CamSol → ESM → Rosetta pipeline.

        Parameters
        ----------
        pdb_path           : local PDB/CIF file (required for Rosetta)
        chain_id           : chain to analyse (None = use all / first available)
        sequence           : amino-acid sequence string (fetched from session if None)
        filters            : optional dict overriding default thresholds:
            camsol_threshold      float  default -0.5
            esm_threshold         float  default 0.3  (conservation, lower=tolerant)
            max_candidates        int    default 20
            candidates_per_pos    int    default 3
            focus                 str    "solubility" | "stability" | "both"
            binding_site_residues list   residue numbers to protect
            w_ddg / w_sol / w_tol float  weight overrides
        protected_residues : explicit list of residue IDs to exclude from scan
                             (e.g. interface residues from AssemblyAnalyser in
                             multimer mode).  Merged with binding_site_residues.
        analysis_mode      : "monomer" | "multimer" — used for reporting only.
        include_positions  : if given, scan ONLY these residue numbers (assignable
                             scope) — bypasses the CamSol/ESM pre-filter so every
                             named position is scanned; default None = whole chain
                             (current behaviour).  A non-None-but-empty scope means
                             the request resolved to nothing → returns [].
        run_rosetta        : when True (deep tier) run the Rosetta ddG voter; when
                             False (fast/default triage tier) skip it entirely (no
                             WSL2/PyRosetta spawned), report ddG as "not computed",
                             and renormalise the score over CamSol + ESM.
        rosetta_shortlist_k: deep-tier coverage.  None (DEFAULT) = FULL grid — run
                             Rosetta on every generated candidate (max data).  An
                             int = SHORTLIST — run Rosetta only on the top-K by
                             fast (CamSol+ESM) score; the rest are RETAINED with
                             ddg="not_computed" (never dropped, only deferred).
        ddg_basis          : "symmetric" (DEFAULT — per-mutation paired WT re-relax,
                             variance-reduced) or "asymmetric" (score against the
                             single cached WT; ~2× faster, noisier, labelled).  The
                             RESULT is fully LOSSLESS: every generated candidate is
                             retained with all per-measure fields + the tier/basis
                             that produced each (substrate for export + aggregation).

        Returns
        -------
        List of candidate dicts, sorted by combined_score descending.
        Each dict contains: position, chain, from_aa, to_aa, ddg,
        solubility_delta, esm_tolerance, combined_score, recommendation,
        and optionally interface_proximal (bool).
        """
        _scan_start    = time.perf_counter()
        _scan_deadline = _scan_start + scan_timeout

        filt               = filters or {}
        camsol_thr         = filt.get("camsol_threshold",   -0.5)
        esm_thr            = filt.get("esm_threshold",       0.3)
        # LOSSLESS / full-coverage default: None = no generation cap (retain every
        # candidate that passes the CamSol/ESM thresholds).  An explicit int still
        # caps generation if a caller wants it.  (Rosetta cost is governed by the
        # opt-in deep tier + shortlist, NOT by truncating the evaluated set.)
        max_candidates     = filt.get("max_candidates", None)
        cands_per_pos      = filt.get("candidates_per_pos",  3)
        protected          = set(filt.get("binding_site_residues", []))
        w_ddg              = filt.get("w_ddg", _W_DDG)
        w_sol              = filt.get("w_sol", _W_SOL)
        w_tol              = filt.get("w_tol", _W_TOL)

        # Merge in explicit protected_residues (e.g. interface residues)
        _interface_protected: set = set(protected_residues or [])
        protected = protected | _interface_protected
        self._analysis_mode = analysis_mode

        # ── Step 1: Fetch / compute CamSol scores ─────────────────────────────
        self._progress("Step 1/4: CamSol solubility scoring...")
        _t0 = time.perf_counter()
        camsol_scores = self._get_or_run_camsol(sequence, chain_id)
        if not camsol_scores:
            return self._error_result("CamSol scores unavailable.")
        self._progress(f"  CamSol: complete ({time.perf_counter() - _t0:.1f}s)")

        # ── Step 2: Fetch / compute ESM conservation scores ───────────────────
        self._progress("Step 2/4: ESM-2 conservation scoring...")
        _t0 = time.perf_counter()
        esm_scores = self._get_or_run_esm(sequence, chain_id, deadline=_scan_deadline)
        if not esm_scores:
            return self._error_result("ESM-2 scores unavailable.")
        self._progress(f"  ESM-2: complete ({time.perf_counter() - _t0:.1f}s)")

        # Retrieve actual sequence if not passed in
        if sequence is None:
            sequence = self._get_sequence(chain_id)
        if not sequence:
            return self._error_result("No amino-acid sequence available.")

        # ── Step 3: Identify candidates ───────────────────────────────────────
        self._progress("Step 3/4: Identifying candidate positions...")
        _t0 = time.perf_counter()

        # Report interface exclusions in multimer mode
        if _interface_protected:
            self._progress(
                f"  {len(_interface_protected)} position(s) excluded — "
                f"at chain interface (multimer mode)."
            )

        # Assignable scope: when an explicit position set is given, scan ONLY those
        # positions and raise the candidate cap so every scoped position is covered
        # (a scope of 16 positions must not be truncated by the default cap of 20).
        _include_set: Optional[set] = None
        if include_positions is not None:
            _include_set = {int(p) for p in include_positions}
            if not _include_set:
                self._progress(
                    "  Requested scan scope resolved to no positions — nothing to scan."
                )
                return []
            # (Full coverage is default — max_candidates None means no cap, so the
            # whole scope × cands_per_pos grid is generated; no cap-raise needed.)

        raw_candidates = self._identify_candidates(
            sequence           = sequence,
            camsol_scores      = camsol_scores,
            esm_scores         = esm_scores,
            camsol_thr         = camsol_thr,
            esm_thr            = esm_thr,
            protected          = protected,
            cands_per_pos      = cands_per_pos,
            max_candidates     = max_candidates,
            chain_id           = chain_id,
            interface_residues = _interface_protected,
            include_positions  = _include_set,
        )

        if not raw_candidates:
            self._progress(
                "  No positions meet both CamSol and ESM criteria. "
                "Consider relaxing thresholds."
            )
            return []

        n_positions = len({c["position"] for c in raw_candidates})
        self._progress(
            f"  {len(raw_candidates)} candidate mutations at {n_positions} positions."
        )
        self._progress(
            f"  Generating candidates: complete ({time.perf_counter() - _t0:.1f}s)"
        )

        # ── Step 4: Rosetta ddG (OPT-IN deep tier) ────────────────────────────
        # The full grid above is ALWAYS scored on CamSol + ESM (lossless).  The
        # deep tier runs Rosetta on EVERY candidate (full coverage, default) OR the
        # top-K by fast score (shortlist opt-in); non-Rosetta'd candidates are
        # RETAINED with ddg="not_computed" — shortlist DEFERS, never drops.  ddG is
        # decided PER CANDIDATE below (key present in ddg_scores), never a fake 0.0.

        # Fast (CamSol+ESM, renormalised) score for EVERY candidate — used for
        # ranking, shortlist selection, and as the retained score when ddG is off.
        _wf_ddg, _wf_sol, _wf_tol = effective_weights(False, w_ddg, w_sol, w_tol)
        for cand in raw_candidates:
            _etol = round(1.0 - esm_scores.get(cand["position"], 0.5), 4)
            cand["_esm_tol"]    = _etol
            cand["_fast_score"] = combined_score(
                0.0, cand["estimated_camsol_delta"], _etol, _wf_ddg, _wf_sol, _wf_tol
            )

        ddg_scores:     Dict[str, float] = {}
        ddg_source:     Dict[str, str]   = {}
        ddg_spread:     Dict[str, Any]   = {}
        ddg_confidence: Dict[str, str]   = {}

        if not run_rosetta:
            self._progress(
                f"  Fast tier (CamSol + ESM only) — Rosetta ddG skipped for all "
                f"{len(raw_candidates)} candidate(s); re-run with 'rosetta' for the "
                "deep ddG-validated tier."
            )
        else:
            _remaining = _scan_deadline - time.perf_counter()
            if _remaining <= 30:
                self._progress(
                    f"  Overall scan timeout ({scan_timeout}s) reached before ddG "
                    "scoring — candidates ranked by CamSol + ESM (ddG not computed)."
                )
            else:
                if rosetta_shortlist_k and rosetta_shortlist_k < len(raw_candidates):
                    rosetta_cands = sorted(
                        raw_candidates, key=lambda c: c["_fast_score"], reverse=True
                    )[:rosetta_shortlist_k]
                    self._progress(
                        f"Step 4/4: Rosetta ddG — SHORTLIST top {len(rosetta_cands)} "
                        f"of {len(raw_candidates)} by fast score ({ddg_basis} basis); "
                        "the rest retained as 'not computed'."
                    )
                else:
                    rosetta_cands = raw_candidates
                    self._progress(
                        f"Step 4/4: Rosetta ddG — FULL coverage, "
                        f"{len(rosetta_cands)} mutation(s) ({ddg_basis} basis)..."
                    )
                _t0 = time.perf_counter()
                ddg_scores, ddg_source, ddg_spread, ddg_confidence = self._run_rosetta_batch(
                    pdb_path, rosetta_cands, chain_id,
                    scan_deadline=_scan_deadline, ddg_basis=ddg_basis,
                )
                self._progress(f"  Rosetta: complete ({time.perf_counter() - _t0:.1f}s)")

        # ── Assemble (LOSSLESS — EVERY generated candidate retained) + rank ───
        # One structured record per candidate keyed by (from,pos,to), carrying all
        # per-measure values + the tier/basis that produced the ddG.  Deep-scored
        # candidates use the full ddG-inclusive score; the rest keep their fast
        # (CamSol+ESM) score and an explicit not_computed ddG.  Nothing truncated —
        # this is the substrate for the future export + cross-layer aggregation.
        results: List[Dict[str, Any]] = []
        for cand in raw_candidates:
            pos     = cand["position"]
            to_aa   = cand["to_aa"]
            from_aa = cand["from_aa"]
            key     = f"{from_aa}{pos}{to_aa}"
            camsol_delta = cand["estimated_camsol_delta"]
            esm_tol      = cand["_esm_tol"]

            if key in ddg_scores:                          # this candidate got Rosetta
                ddg      = ddg_scores[key]
                ddg_src  = ddg_source.get(key, "none")
                ddg_sprd = ddg_spread.get(key)
                ddg_conf = ddg_confidence.get(key, "single-trajectory")
                ddg_out  = round(ddg, 3)
                basis    = ddg_basis                       # "symmetric" | "asymmetric"
                tier     = "deep"
                score    = combined_score(ddg, camsol_delta, esm_tol,
                                          w_ddg, w_sol, w_tol)
            else:                                          # fast tier / deferred by shortlist
                ddg = ddg_out = None
                ddg_src  = "not_computed"
                ddg_sprd = None
                ddg_conf = "not_computed"
                basis    = "none"
                tier     = "fast"
                score    = cand["_fast_score"]

            is_proximal = cand.get("interface_proximal", False)
            rec = _recommendation(ddg, camsol_delta, esm_tol, score)
            if is_proximal:
                rec += " — ⚠ interface-proximal, mutate with caution"

            results.append({
                "key":               key,
                "position":          pos,
                "chain":             chain_id or "A",
                "from_aa":           from_aa,
                "to_aa":             to_aa,
                "ddg":               ddg_out,
                "ddg_source":        ddg_src,
                "ddg_spread":        ddg_sprd,
                "ddg_confidence":    ddg_conf,
                "ddg_basis":         basis,
                "tier":              tier,
                "solubility_delta":  camsol_delta,
                "esm_tolerance":     esm_tol,
                "camsol_score":      round(camsol_scores.get(pos, 0.0), 3),
                "fast_score":        cand["_fast_score"],
                "combined_score":    score,
                "recommendation":    rec,
                "interface_proximal": is_proximal,
                "analysis_mode":     getattr(self, "_analysis_mode", "monomer"),
            })

        results.sort(key=lambda r: r["combined_score"], reverse=True)

        # Cache results in session
        try:
            self.session.add_scan_result(self.model_id, results)
        except AttributeError:
            pass

        return results

    def generate_chimerax_commands(
        self,
        scan_results: List[Dict[str, Any]],
        top_n:        int = 5,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate ChimeraX commands to visualise the top-N scan candidates.

        Visualization scheme:
          • All residues: cartoon, coloured white baseline
          • Candidate positions: coloured by combined_score on a gradient
              score ≥ 1.5  → blue    (strong candidate)
              score ≥ 0.5  → cyan    (good candidate)
              score ≥ 0.0  → yellow  (marginal)
              score  < 0.0 → orange  (not recommended)
          • Top-N positions: shown as spheres + 2D label with mutation suggestion
        """
        if not scan_results:
            return [], []

        mid        = self.model_id
        chain_spec = f"/{scan_results[0]['chain']}" if scan_results else ""

        cmds = [
            f"cartoon #{mid}",
            f"color #{mid}{chain_spec} white",
        ]
        exps = [
            "Switch to cartoon representation",
            "Reset all residues to white before applying scan colours",
        ]

        # Colour all candidates by score
        for r in scan_results:
            pos   = r["position"]
            score = r["combined_score"]
            spec  = f"#{mid}{chain_spec}:{pos}"

            if score >= 1.5:
                colour = "blue"
            elif score >= 0.5:
                colour = "cyan"
            elif score >= 0.0:
                colour = "yellow"
            else:
                colour = "orange red"

            cmds.append(f"color {spec} {colour}")
            exps.append(
                f"Position {pos} ({r['from_aa']}→{r['to_aa']}): "
                f"score={score:.2f}"
            )

        # Top-N: spheres + labels
        for rank, r in enumerate(scan_results[:top_n], 1):
            pos   = r["position"]
            spec  = f"#{mid}{chain_spec}:{pos}"
            label = f"{r['from_aa']}{pos}{r['to_aa']} ({r['combined_score']:+.1f})"

            cmds.append(f"show {spec} atoms")
            exps.append(f"Show top-{rank} candidate (position {pos}) as atoms")

            cmds.append(f"style {spec} sphere")
            exps.append(f"Sphere style to highlight position {pos}")

            cmds.append(f"label {spec} text \"{label}\" size 14 color white")
            exps.append(f"Label position {pos}: {label}")

        cmds.append(f"view #{mid}")
        exps.append("Fit structure in view to show all labelled candidates")

        return cmds, exps

    # ── Internal steps ─────────────────────────────────────────────────────────

    def _get_or_run_camsol(
        self,
        sequence: Optional[str],
        chain_id: Optional[str],
    ) -> Optional[Dict[int, float]]:
        """
        Return CamSol scores {resno: score} from cache or by running the bridge.
        """
        cached = None
        try:
            cached = self.session.get_tool_result("camsol", self.model_id)
        except AttributeError:
            pass

        if cached and cached.get("scores"):
            return {int(k): float(v) for k, v in cached["scores"].items()}

        # Not cached — run it
        seq = sequence or self._get_sequence(chain_id)
        if not seq:
            return None

        from camsol_bridge import CamsolBridge
        bridge = CamsolBridge()
        result = bridge.analyze(
            seq,
            model_id = self.model_id,
            chain    = chain_id,
        )
        if result.success:
            try:
                self.session.add_tool_result("camsol", self.model_id, result.data)
            except AttributeError:
                pass
            return {int(k): float(v) for k, v in result.data["scores"].items()}
        return None

    def _get_or_run_esm(
        self,
        sequence: Optional[str],
        chain_id: Optional[str],
        deadline: Optional[float] = None,
    ) -> Optional[Dict[int, float]]:
        """
        Return ESM conservation scores {resno: score} from cache or by running.

        Parameters
        ----------
        deadline : optional ``time.perf_counter()`` value; if set, the
                   remaining seconds are used as the ESM inference timeout
                   (capped at 120 s).  If the scan deadline has already
                   passed the step is skipped immediately.
        """
        cached = None
        try:
            cached = self.session.get_tool_result("esm", self.model_id)
        except AttributeError:
            pass

        if cached and cached.get("conservation"):
            return {int(k): float(v) for k, v in cached["conservation"].items()}

        seq = sequence or self._get_sequence(chain_id)
        if not seq:
            return None

        # Compute per-sequence timeout from the overall scan deadline
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 5:
                # No time left — skip ESM entirely
                return None
            inference_timeout = max(10, min(120, int(remaining)))
        else:
            inference_timeout = 120

        from esm_bridge import EsmBridge
        bridge = EsmBridge()
        result = bridge.analyze(
            seq,
            model_id          = self.model_id,
            session           = self.session,
            inference_timeout = inference_timeout,
        )
        if result.success:
            try:
                self.session.add_tool_result("esm", self.model_id, result.data)
            except AttributeError:
                pass
            return {int(k): float(v) for k, v in result.data["conservation"].items()}
        return None

    def _get_sequence(self, chain_id: Optional[str]) -> Optional[str]:
        """Look up the amino-acid sequence from session state."""
        try:
            info = self.session.get_structure(self.model_id)
        except AttributeError:
            return None
        if not info:
            return None
        meta = info.get("metadata", {})
        if chain_id and isinstance(meta.get("sequences"), dict):
            return meta["sequences"].get(chain_id)
        return meta.get("sequence")

    def _identify_candidates(
        self,
        sequence:          str,
        camsol_scores:     Dict[int, float],
        esm_scores:        Dict[int, float],
        camsol_thr:        float,
        esm_thr:           float,
        protected:         set,
        cands_per_pos:     int,
        max_candidates:    int,
        chain_id:          Optional[str],
        interface_residues: Optional[set] = None,
        include_positions: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the list of (position, from_aa, to_aa, estimated_camsol_delta) dicts
        that pass both the CamSol and ESM thresholds.

        Positions in *interface_residues* are excluded (already in *protected*).
        Positions within 3 residues of an interface residue are flagged as
        interface_proximal=True — still included, but labelled "mutate with caution".

        *include_positions* (assignable scope): when given, ONLY these residue
        numbers are considered AND the CamSol/ESM pre-filter is bypassed — the user
        named the positions, so every one is scanned (still excluding *protected*
        and the Pro/Cys substitution rules in _substitution_candidates).
        """
        candidates: List[Dict[str, Any]] = []
        iface_set = interface_residues or set()

        # Pre-compute proximal positions: within 3 of any interface residue
        proximal_set: set = set()
        if iface_set:
            for ir in iface_set:
                for offset in range(-3, 4):
                    proximal_set.add(ir + offset)
            proximal_set -= iface_set  # exclude direct interface residues

        scoped = include_positions is not None

        for pos, from_aa in enumerate(sequence, 1):
            if scoped and pos not in include_positions:
                continue     # outside the assigned scope
            if pos in protected:
                continue

            # Threshold pre-filter applies ONLY to whole-chain scans.  An explicit
            # scope means the user named these positions → scan them all.
            if not scoped:
                camsol_val = camsol_scores.get(pos, 0.0)
                esm_val    = esm_scores.get(pos, 0.5)
                if camsol_val >= camsol_thr:
                    continue     # not aggregation-prone enough
                if esm_val >= esm_thr:
                    continue     # too conserved — risky to mutate

            is_proximal = pos in proximal_set

            # Generate substitution candidates for this position
            subs = self._substitution_candidates(from_aa, cands_per_pos)
            for to_aa, camsol_delta in subs:
                candidates.append({
                    "position":              pos,
                    "chain":                 chain_id or "A",
                    "from_aa":               from_aa,
                    "to_aa":                 to_aa,
                    "estimated_camsol_delta": camsol_delta,
                    "interface_proximal":     is_proximal,
                })
                if max_candidates is not None and len(candidates) >= max_candidates:
                    return candidates

        return candidates

    def _substitution_candidates(
        self,
        from_aa:       str,
        top_n:         int,
    ) -> List[Tuple[str, float]]:
        """
        Generate up to top_n substitution candidates for a given residue.

        Returns [(to_aa, estimated_camsol_delta), ...] sorted by delta descending.

        Strategy:
          1. Score every standard AA except self, Pro, Cys by predicted delta
          2. Prefer hydrophilic / charged substitutions (solubility bias)
          3. Return top_n
        """
        candidates: List[Tuple[str, float]] = []

        for to_aa in _SOLUBILITY_DONORS:
            if to_aa == from_aa or to_aa in _EXCLUDED_SUBSTITUTIONS:
                continue
            delta = _estimate_camsol_delta(from_aa, to_aa)
            candidates.append((to_aa, delta))

        # Sort descending by delta
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_n]

    def _run_rosetta_batch(
        self,
        pdb_path:      str,
        candidates:    List[Dict[str, Any]],
        chain_id:      Optional[str],
        scan_deadline: Optional[float] = None,
        ddg_basis:     str = "symmetric",
    ) -> Tuple[Dict[str, float], Dict[str, str], Dict[str, Any], Dict[str, str]]:
        """
        Run Rosetta ddG for all candidate mutations in one batch call.

        Returns (ddg_scores, ddg_source, ddg_spread, ddg_confidence):
          ddg_scores     : {mutation_key: ddg_kcal_mol}
          ddg_source     : {mutation_key: provenance} ("pyrosetta"/"empirical"/backend)
          ddg_spread     : {mutation_key: MAD across trajectories | None}
          ddg_confidence : {mutation_key: "high"/"moderate"/"low"/"single-trajectory"}
        Production scans are single-trajectory (ROSETTA_NUM_TRAJECTORIES=1), so
        spread is None and confidence is "single-trajectory".
        If Rosetta is unavailable, returns ({}, {}, {}, {}) so the pipeline can
        still rank by CamSol + ESM alone (with a user-visible warning).

        scan_deadline : time.perf_counter() deadline for the overall scan.
                        Forwarded to RosettaBridge so DynaMut2 can switch to
                        empirical fallback before the budget is exhausted.
        """
        if not Path(pdb_path).is_file():
            self._progress(
                "  PDB file not found — skipping Rosetta (ddG will be 0.0)."
            )
            return {}, {}, {}, {}

        from rosetta_bridge import RosettaBridge
        bridge = RosettaBridge()

        # Convert candidate dicts to the format RosettaBridge expects
        mutations = [
            {
                "chain":    c.get("chain", chain_id or "A"),
                "position": c["position"],
                "from_aa":  c["from_aa"],
                "to_aa":    c["to_aa"],
            }
            for c in candidates
        ]

        result = bridge.analyze(
            pdb_path          = pdb_path,
            mutations         = mutations,
            session           = self.session,
            model_id          = self.model_id,
            chain             = chain_id,
            progress_callback = self._progress,
            scan_deadline     = scan_deadline,
            ddg_basis         = ddg_basis,
        )

        if result.success:
            scores = result.data.get("ddg_scores", {})
            source = dict(result.data.get("ddg_source", {}))
            spread = dict(result.data.get("ddg_spread", {}))
            conf   = dict(result.data.get("ddg_confidence", {}))
            if not source:
                # Bridge did not provide per-mutation provenance (e.g. the
                # DynaMut2 path) — label every scored value with the backend.
                be = result.data.get("backend", "rosetta")
                source = {k: be for k in scores}
            return scores, source, spread, conf

        # Rosetta failed — warn and continue with zeros
        self._progress(
            f"  Rosetta ddG unavailable: {(result.error or '')[:100]}\n"
            "    Candidates ranked by CamSol + ESM only (ddG = 0.0)."
        )
        return {}, {}, {}, {}

    @staticmethod
    def _error_result(msg: str) -> List[Dict[str, Any]]:
        """Return an empty list with a print; callers check for empty."""
        print(f"[MutationScanner] Error: {msg}", flush=True)
        return []

    @staticmethod
    def _generate_summary(candidates: List[Dict[str, Any]]) -> str:
        """
        Generate a multi-line actionable summary string for the top candidates.

        Designed to be displayed in a Rich Panel after a mutation scan.
        Returns a fallback one-liner if candidates is empty or data is missing.
        """
        if not candidates:
            return "No candidates found. Try relaxing CamSol or ESM thresholds."

        lines: List[str] = []
        top3 = candidates[:3]

        lines.append(f"Ranked candidates  ({len(candidates)} total):")
        lines.append("-" * 48)

        for rank, c in enumerate(top3, 1):
            pos     = c.get("position", "?")
            chain   = c.get("chain", "A")
            from_aa = c.get("from_aa", "?")
            to_aa   = c.get("to_aa", "?")
            score   = c.get("combined_score", 0.0)
            ddg     = c.get("ddg")
            src     = c.get("ddg_source", "?")
            conf    = c.get("ddg_confidence", "single-trajectory")
            sprd    = c.get("ddg_spread")
            sol     = c.get("solubility_delta", 0.0)
            tol     = c.get("esm_tolerance", 0.0)
            # ddG honesty: None → "not computed (opt in)" rather than a fake 0.0.
            if ddg is None:
                ddg_str = "ddG=not computed (opt in with 'rosetta')"
            else:
                _prov = f"[{src}, {conf}" + (f", spread {sprd:.2f}]" if sprd is not None else "]")
                ddg_str = f"ddG={ddg:+.3f} kcal/mol {_prov}"
            lines.append(
                f"  #{rank}  {chain}{pos}: {from_aa} -> {to_aa}  "
                f"score={score:+.2f}  {ddg_str}  "
                f"solubility+={sol:+.2f}  ESM_tol={tol:.2f}"
            )

        if len(candidates) > 5:
            extra = len(candidates) - 3
            lines.append(f"  ... +{extra} more candidates (see scan results)")

        lines.append("")
        lines.append("Suggested next steps:")
        lines.append("  1. Validate top candidate with ESMFold structural prediction")
        lines.append("  2. Explore double-mutant combinations for additive effects")
        lines.append("  3. Run ProteinMPNN for global sequence redesign context")
        lines.append("  4. Order gene synthesis for top 1-3 candidates")

        # Enforce max 15 lines
        lines = lines[:15]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<MutationScanner model_id={self.model_id!r} "
            f"session={self.session!r}>"
        )
