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


def _recommendation(
    ddg:           float,
    camsol_delta:  float,
    esm_tolerance: float,
    score:         float,
) -> str:
    """Generate a human-readable recommendation for a candidate mutation."""
    parts: List[str] = []

    if ddg <= -1.0:
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
        max_candidates     = filt.get("max_candidates",      20)
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

        # ── Step 4: Rosetta ddG scoring ───────────────────────────────────────
        _remaining = _scan_deadline - time.perf_counter()
        if _remaining <= 30:
            self._progress(
                f"  Overall scan timeout ({scan_timeout}s) reached before ddG scoring. "
                "Candidates ranked by CamSol + ESM only (ddG = 0.0)."
            )
            ddg_scores:     Dict[str, float] = {}
            ddg_source:     Dict[str, str]   = {}
            ddg_spread:     Dict[str, Any]   = {}
            ddg_confidence: Dict[str, str]   = {}
        else:
            self._progress(
                f"Step 4/4: Rosetta ddG for {len(raw_candidates)} mutations..."
            )
            _t0 = time.perf_counter()
            ddg_scores, ddg_source, ddg_spread, ddg_confidence = self._run_rosetta_batch(
                pdb_path, raw_candidates, chain_id,
                scan_deadline=_scan_deadline,
            )
            self._progress(f"  DynaMut2: complete ({time.perf_counter() - _t0:.1f}s)")

        # ── Assemble and rank ─────────────────────────────────────────────────
        results: List[Dict[str, Any]] = []
        for cand in raw_candidates:
            pos     = cand["position"]
            to_aa   = cand["to_aa"]
            from_aa = cand["from_aa"]
            key     = f"{from_aa}{pos}{to_aa}"

            ddg           = ddg_scores.get(key, 0.0)
            # Provenance: "pyrosetta" / "empirical" / backend label, or "none"
            # when ddG was not computed (timeout/failure → defaulted to 0.0).
            ddg_src       = ddg_source.get(key, "none")
            ddg_sprd      = ddg_spread.get(key)            # MAD, or None (single-traj)
            ddg_conf      = ddg_confidence.get(key, "single-trajectory")
            camsol_delta  = cand["estimated_camsol_delta"]
            esm_cons      = esm_scores.get(pos, 0.5)
            esm_tol       = round(1.0 - esm_cons, 4)

            score = combined_score(ddg, camsol_delta, esm_tol, w_ddg, w_sol, w_tol)

            # Check interface-proximal: within 3 positions of an interface residue
            is_proximal = cand.get("interface_proximal", False)

            rec = _recommendation(ddg, camsol_delta, esm_tol, score)
            if is_proximal:
                rec += " — ⚠ interface-proximal, mutate with caution"

            results.append({
                "position":          pos,
                "chain":             chain_id or "A",
                "from_aa":           from_aa,
                "to_aa":             to_aa,
                "ddg":               round(ddg, 3),
                "ddg_source":        ddg_src,
                "ddg_spread":        ddg_sprd,
                "ddg_confidence":    ddg_conf,
                "solubility_delta":  camsol_delta,
                "esm_tolerance":     esm_tol,
                "combined_score":    score,
                "camsol_score":      round(camsol_scores.get(pos, 0.0), 3),
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
    ) -> List[Dict[str, Any]]:
        """
        Build the list of (position, from_aa, to_aa, estimated_camsol_delta) dicts
        that pass both the CamSol and ESM thresholds.

        Positions in *interface_residues* are excluded (already in *protected*).
        Positions within 3 residues of an interface residue are flagged as
        interface_proximal=True — still included, but labelled "mutate with caution".
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

        for pos, from_aa in enumerate(sequence, 1):
            if pos in protected:
                continue

            camsol_val = camsol_scores.get(pos, 0.0)
            esm_val    = esm_scores.get(pos, 0.5)

            # Apply both filters
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
                if len(candidates) >= max_candidates:
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
            ddg     = c.get("ddg", 0.0)
            src     = c.get("ddg_source", "?")
            conf    = c.get("ddg_confidence", "single-trajectory")
            sprd    = c.get("ddg_spread")
            sol     = c.get("solubility_delta", 0.0)
            tol     = c.get("esm_tolerance", 0.0)
            # Show provenance + confidence; include spread only when multi-trajectory.
            _prov = f"[{src}, {conf}" + (f", spread {sprd:.2f}]" if sprd is not None else "]")
            lines.append(
                f"  #{rank}  {chain}{pos}: {from_aa} -> {to_aa}  "
                f"score={score:+.2f}  ddG={ddg:+.3f} kcal/mol {_prov}  "
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
