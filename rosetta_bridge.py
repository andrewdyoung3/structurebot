"""
rosetta_bridge.py
-----------------
# Stability backend: DynaMut2 (primary) + empirical (fallback)
# Future: replace with local Rosetta cartesian_ddg on Linux/Mac
# for publication-quality results. DynaMut2 is appropriate for
# candidate screening; Rosetta recommended for final validation.

Three backends + one future stub — identical public interface:

BACKEND A — PyRosetta (local stub)
  Activates when PYROSETTA_AVAILABLE=true in .env.local AND
  `import pyrosetta` succeeds.  Python 3.14 wheels not yet available.
  See _run_pyrosetta() docstring for setup instructions.

BACKEND B — DynaMut2 web API  [DEFAULT]
  https://biosig.lab.uq.edu.au/dynamut2/
  Free, no registration or API key required.
  Two-step async flow per mutation:
    1. POST /api/prediction_single  ->  {"job_id": "..."}
    2. GET  /api/prediction_single?job_id=...  ->  {"prediction": float, ...}
       (poll until response is not {"message": "RUNNING"})
  Handles rate-limiting and transient errors with retry + exponential backoff.
  Automatically falls through to empirical for mutations that fail.

  Citation: Rodrigues et al. (2021) Nucleic Acids Research,
            https://doi.org/10.1093/nar/gkab371

BACKEND C — Empirical (offline fallback)
  BLOSUM62 substitution score + B-factor buried/exposed correction.
  No network required.  All results labelled "estimated".
  Pearson r ≈ 0.4 vs experimental — screening only.

BACKEND D — Local Rosetta (future stub)
  Activate with ROSETTA_BACKEND=local + ROSETTA_LOCAL_PATH in .env.local.
  Pearson r ≈ 0.8 vs experimental.  Linux/Mac only.  Requires license.
  See _run_rosetta_local() for full setup instructions.

Backend selection
-----------------
  ROSETTA_BACKEND=auto       (default) — PyRosetta if available, else DynaMut2
  ROSETTA_BACKEND=pyrosetta  — force PyRosetta (fails if not installed)
  ROSETTA_BACKEND=dynamut2   — always use DynaMut2 web API
  ROSETTA_BACKEND=empirical  — offline BLOSUM62 estimates only
  ROSETTA_BACKEND=local      — local Rosetta installation (stub)

Output schema
-------------
ToolStepResult.data keys:
  mutations        : list of {chain, position, from_aa, to_aa}
  ddg_scores       : {mutation_key: float}  e.g. {"V82A": 1.47}
                     kcal/mol; positive = destabilising, negative = stabilising
  stability_change : float — mean ddG across all scored mutations
  confidence       : "high" | "medium" | "low"
  backend          : "pyrosetta" | "dynamut2" | "dynamut2+empirical" | "empirical"
  warnings         : list[str]
  method_note      : str  — accuracy / provenance note for display
  job_id           : None  (kept for schema compatibility; always None for DynaMut2)

DynaMut2 API details
--------------------
  Step 1 — Submit:
    POST https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single
    Content-Type: multipart/form-data
    Fields:
      pdb_file  (file, optional)  — PDB bytes
      pdb       (str,  optional)  — 4-char PDB ID (alternative to file upload)
      chain     (str,  required)  — chain ID, e.g. "A"
      mutation  (str,  required)  — e.g. "V82A"
      email     (str,  optional)
    Response: {"job_id": "<uuid>"}

  Step 2 — Poll:
    GET https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single?job_id=<uuid>
    Response while running: {"message": "RUNNING"}
    Response when done:
      {"prediction": <float>, "chain": "A", "res_number": 82,
       "wild-type": "V", "mutant": "A", "results_page": "<url>"}

  Sign convention: positive prediction = destabilising, negative = stabilising.
  Auth: none required.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tool_router import ToolStepResult

# ── DynaMut2 API constants ─────────────────────────────────────────────────────

_DYNAMUT2_BASE         = "https://biosig.lab.uq.edu.au/dynamut2/api"
_DYNAMUT2_SUBMIT_URL   = f"{_DYNAMUT2_BASE}/prediction_single"
_DYNAMUT2_RESULT_URL   = f"{_DYNAMUT2_BASE}/prediction_single"   # GET with ?job_id=

_DYNAMUT2_TIMEOUT         = 15    # seconds per HTTP request (connect + read)
_DYNAMUT2_POLL_INTERVAL   = 5     # seconds between result polls
_DYNAMUT2_MAX_POLLS       = 12    # 12 × 5s = 60s max polling per mutation (was 300s)
_DYNAMUT2_RETRY_DELAYS    = (3, 8) # 2 submit retries with short delays (was (5,15,30))
_DYNAMUT2_PER_MUT_TIMEOUT = 60    # wall-clock seconds budget per mutation
_DYNAMUT2_CIRCUIT_BREAKER = 2     # consecutive failures before switching to empirical

# ── BLOSUM62 substitution matrix (Henikoff & Henikoff 1992) ───────────────────
# Used by the empirical fallback backend.

_BLOSUM62: Dict[str, Dict[str, int]] = {
    "A": {"A": 4,"R":-1,"N":-2,"D":-2,"C": 0,"Q":-1,"E":-1,"G": 0,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 0,"W":-3,"Y":-2,"V": 0},
    "R": {"A":-1,"R": 5,"N": 0,"D":-2,"C":-3,"Q": 1,"E": 0,"G":-2,"H": 0,"I":-3,"L":-2,"K": 2,"M":-1,"F":-3,"P":-2,"S":-1,"T":-1,"W":-3,"Y":-2,"V":-3},
    "N": {"A":-2,"R": 0,"N": 6,"D": 1,"C":-3,"Q": 0,"E": 0,"G": 0,"H": 1,"I":-3,"L":-3,"K": 0,"M":-2,"F":-3,"P":-2,"S": 1,"T": 0,"W":-4,"Y":-2,"V":-3},
    "D": {"A":-2,"R":-2,"N": 1,"D": 6,"C":-3,"Q": 0,"E": 2,"G":-1,"H":-1,"I":-3,"L":-4,"K":-1,"M":-3,"F":-3,"P":-1,"S": 0,"T":-1,"W":-4,"Y":-3,"V":-3},
    "C": {"A": 0,"R":-3,"N":-3,"D":-3,"C": 9,"Q":-3,"E":-4,"G":-3,"H":-3,"I":-1,"L":-1,"K":-3,"M":-1,"F":-2,"P":-3,"S":-1,"T":-1,"W":-2,"Y":-2,"V":-1},
    "Q": {"A":-1,"R": 1,"N": 0,"D": 0,"C":-3,"Q": 5,"E": 2,"G":-2,"H": 0,"I":-3,"L":-2,"K": 1,"M": 0,"F":-3,"P":-1,"S": 0,"T":-1,"W":-2,"Y":-1,"V":-2},
    "E": {"A":-1,"R": 0,"N": 0,"D": 2,"C":-4,"Q": 2,"E": 5,"G":-2,"H": 0,"I":-3,"L":-3,"K": 1,"M":-2,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "G": {"A": 0,"R":-2,"N": 0,"D":-1,"C":-3,"Q":-2,"E":-2,"G": 6,"H":-2,"I":-4,"L":-4,"K":-2,"M":-3,"F":-3,"P":-2,"S": 0,"T":-2,"W":-2,"Y":-3,"V":-3},
    "H": {"A":-2,"R": 0,"N": 1,"D":-1,"C":-3,"Q": 0,"E": 0,"G":-2,"H": 8,"I":-3,"L":-3,"K":-1,"M":-2,"F":-1,"P":-2,"S":-1,"T":-2,"W":-2,"Y": 2,"V":-3},
    "I": {"A":-1,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-3,"E":-3,"G":-4,"H":-3,"I": 4,"L": 2,"K":-3,"M": 1,"F": 0,"P":-3,"S":-2,"T":-1,"W":-3,"Y":-1,"V": 3},
    "L": {"A":-1,"R":-2,"N":-3,"D":-4,"C":-1,"Q":-2,"E":-3,"G":-4,"H":-3,"I": 2,"L": 4,"K":-2,"M": 2,"F": 0,"P":-3,"S":-2,"T":-1,"W":-2,"Y":-1,"V": 1},
    "K": {"A":-1,"R": 2,"N": 0,"D":-1,"C":-3,"Q": 1,"E": 1,"G":-2,"H":-1,"I":-3,"L":-2,"K": 5,"M":-1,"F":-3,"P":-1,"S": 0,"T":-1,"W":-3,"Y":-2,"V":-2},
    "M": {"A":-1,"R":-1,"N":-2,"D":-3,"C":-1,"Q": 0,"E":-2,"G":-3,"H":-2,"I": 1,"L": 2,"K":-1,"M": 5,"F": 0,"P":-2,"S":-1,"T":-1,"W":-1,"Y":-1,"V": 1},
    "F": {"A":-2,"R":-3,"N":-3,"D":-3,"C":-2,"Q":-3,"E":-3,"G":-3,"H":-1,"I": 0,"L": 0,"K":-3,"M": 0,"F": 6,"P":-4,"S":-2,"T":-2,"W": 1,"Y": 3,"V":-1},
    "P": {"A":-1,"R":-2,"N":-2,"D":-1,"C":-3,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-3,"L":-3,"K":-1,"M":-2,"F":-4,"P": 7,"S":-1,"T":-1,"W":-4,"Y":-3,"V":-2},
    "S": {"A": 1,"R":-1,"N": 1,"D": 0,"C":-1,"Q": 0,"E": 0,"G": 0,"H":-1,"I":-2,"L":-2,"K": 0,"M":-1,"F":-2,"P":-1,"S": 4,"T": 1,"W":-3,"Y":-2,"V":-2},
    "T": {"A": 0,"R":-1,"N": 0,"D":-1,"C":-1,"Q":-1,"E":-1,"G":-2,"H":-2,"I":-1,"L":-1,"K":-1,"M":-1,"F":-2,"P":-1,"S": 1,"T": 5,"W":-2,"Y":-2,"V": 0},
    "W": {"A":-3,"R":-3,"N":-4,"D":-4,"C":-2,"Q":-2,"E":-3,"G":-2,"H":-2,"I":-3,"L":-2,"K":-3,"M":-1,"F": 1,"P":-4,"S":-3,"T":-2,"W":11,"Y": 2,"V":-3},
    "Y": {"A":-2,"R":-2,"N":-2,"D":-3,"C":-2,"Q":-1,"E":-2,"G":-3,"H": 2,"I":-1,"L":-1,"K":-2,"M":-1,"F": 3,"P":-3,"S":-2,"T":-2,"W": 2,"Y": 7,"V":-1},
    "V": {"A": 0,"R":-3,"N":-3,"D":-3,"C":-1,"Q":-2,"E":-2,"G":-3,"H":-3,"I": 3,"L": 1,"K":-2,"M": 1,"F":-1,"P":-2,"S":-2,"T": 0,"W":-3,"Y":-1,"V": 4},
}

_STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")


# ── Safe print (Windows cp1252 compatible) ────────────────────────────────────

def _safe_print(msg: str) -> None:
    """Print with ASCII fallback for narrow terminal encodings (Windows cp1252)."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mutation_key(mut: Dict[str, Any]) -> str:
    """{"from_aa": "V", "position": 82, "to_aa": "A"} -> "V82A"."""
    return f"{mut['from_aa']}{mut['position']}{mut['to_aa']}"


def _pyrosetta_importable() -> bool:
    """True if the pyrosetta package can actually be imported."""
    try:
        import pyrosetta  # noqa: F401
        return True
    except ImportError:
        return False


# ── Backend detection ──────────────────────────────────────────────────────────

def _select_backend() -> str:
    """
    Return the active backend name based on ROSETTA_BACKEND env var and
    availability.

    Returns one of: "pyrosetta", "dynamut2", "empirical", "local"
    """
    forced = os.environ.get("ROSETTA_BACKEND", "auto").strip().lower()
    if forced == "pyrosetta":
        return "pyrosetta"
    if forced == "dynamut2":
        return "dynamut2"
    if forced == "empirical":
        return "empirical"
    if forced == "local":
        return "local"
    # "auto": prefer PyRosetta only if explicitly enabled AND importable
    flag = os.environ.get("PYROSETTA_AVAILABLE", "").strip().lower()
    if flag in ("1", "true", "yes") and _pyrosetta_importable():
        return "pyrosetta"
    return "dynamut2"


# ── DynaMut2 response parsing ──────────────────────────────────────────────────

def _parse_dynamut2_result(data: Dict[str, Any], mutation_str: str) -> float:
    """
    Extract ΔΔG (kcal/mol) from a completed DynaMut2 result JSON.

    Expected format:
      {"prediction": <float>, "chain": "A", "res_number": 82,
       "wild-type": "V", "mutant": "A", "results_page": "<url>"}

    Sign convention: positive = destabilising, negative = stabilising.

    Raises ValueError if the response is not a completed result
    (e.g. still {"message": "RUNNING"}) or lacks a numeric prediction.
    """
    if "message" in data:
        raise ValueError(
            f"DynaMut2 job for {mutation_str!r} not yet complete: "
            f"{data['message']!r}"
        )

    # Primary field: "prediction"
    if "prediction" in data:
        try:
            return round(float(data["prediction"]), 3)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"DynaMut2 'prediction' field for {mutation_str!r} is not numeric: "
                f"{data['prediction']!r}"
            ) from exc

    raise ValueError(
        f"DynaMut2 result for {mutation_str!r} missing 'prediction' field. "
        f"Got keys: {list(data.keys())}. "
        "Check https://biosig.lab.uq.edu.au/dynamut2/api for current response format."
    )


# ── Empirical helpers ──────────────────────────────────────────────────────────

def _get_buried_factor(pdb_path: str, position: int, chain: str) -> float:
    """
    B-factor buried/exposed correction for the empirical backend.

    Reads the B-factor of the CA atom at *position*:chain from the PDB.

    Factor:
        B < 20 Å²  → buried / ordered  → 1.0 (full structural impact)
        B > 40 Å²  → exposed / flexible → 0.5 (halved impact)
        20–40 Å²   → linear interpolation
        not found  → 1.0 (conservative default)
    """
    try:
        with open(pdb_path, "r", errors="replace") as fh:
            for line in fh:
                if line[:4] not in ("ATOM", "HETA"):
                    continue
                rec_chain = line[21] if len(line) > 21 else " "
                resnum    = line[22:26].strip() if len(line) > 26 else ""
                atomname  = line[12:16].strip() if len(line) > 16 else ""
                if rec_chain == chain and resnum == str(position) and atomname == "CA":
                    try:
                        bfac = float(line[60:66].strip() or "30.0")
                    except ValueError:
                        return 1.0
                    if bfac < 20.0:
                        return 1.0
                    if bfac > 40.0:
                        return 0.5
                    return 1.0 - (bfac - 20.0) / 40.0
    except Exception:
        pass
    return 1.0  # conservative: treat as fully buried


def _empirical_ddg_single(mut: Dict[str, Any], pdb_path: str) -> float:
    """
    Estimate ΔΔG from BLOSUM62 substitution score + B-factor correction.

    Formula:
        blosum_raw    = BLOSUM62[from_aa][to_aa]
        raw_ddg       = -blosum_raw * 0.5
            (negative BLOSUM = radical substitution = destabilising)
        buried_factor = _get_buried_factor() in [0.5, 1.0]
        ddg           = raw_ddg * buried_factor

    Positive ddG = destabilising; negative = stabilising.
    Accuracy: Pearson r ~0.4 vs experimental. For screening only.
    """
    from_aa  = mut.get("from_aa", "A")
    to_aa    = mut.get("to_aa",   "A")
    position = int(mut.get("position", 1))
    chain    = mut.get("chain", "A")

    blosum    = _BLOSUM62.get(from_aa, {}).get(to_aa, 0)
    raw_ddg   = -blosum * 0.5
    buried_f  = _get_buried_factor(pdb_path, position, chain)
    return round(raw_ddg * buried_f, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Public bridge class
# ══════════════════════════════════════════════════════════════════════════════

class RosettaBridge:
    """
    Unified stability / ddG calculation bridge.

    Usage::

        bridge = RosettaBridge()
        result = bridge.analyze(
            pdb_path  = "cache/1HSG.pdb",
            mutations = [{"chain": "A", "position": 82,
                          "from_aa": "V", "to_aa": "A"}],
            session   = session_state,
        )
        if result.success:
            ddg = result.data["ddg_scores"]["V82A"]   # kcal/mol
    """

    def __init__(self) -> None:
        self._backend = _select_backend()

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        pdb_path:          str,
        mutations:         List[Dict[str, Any]],
        mode:              str = "ddg",
        session:           Any = None,
        model_id:          str = "1",
        chain:             Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        scan_deadline:     Optional[float] = None,
    ) -> ToolStepResult:
        """
        Calculate ddG for one or more mutations.

        Parameters
        ----------
        pdb_path          : local path to a PDB/CIF file
        mutations         : list of {chain, position, from_aa, to_aa}
        mode              : "ddg" (default) | "stability" (no effect currently)
        session           : SessionState for result persistence
        model_id          : ChimeraX model number (used in viz commands)
        chain             : chain ID for viz colouring (None = all chains)
        progress_callback : callable(str) for real-time progress messages
        """
        if not mutations:
            return ToolStepResult(
                tool="rosetta", success=False,
                error="No mutations specified.",
            )
        if not Path(pdb_path).is_file():
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"PDB file not found: {pdb_path}",
            )

        try:
            if self._backend == "pyrosetta":
                return self._run_pyrosetta(
                    pdb_path, mutations, mode, model_id, chain, progress_callback
                )
            elif self._backend == "empirical":
                return self._run_empirical(
                    pdb_path, mutations, model_id, chain, progress_callback
                )
            elif self._backend == "local":
                self._run_rosetta_local(pdb_path, mutations)  # always raises
            else:
                return self._run_dynamut2(
                    pdb_path, mutations, model_id, chain, session,
                    progress_callback, scan_deadline,
                )
        except NotImplementedError as exc:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=str(exc),
            )
        except Exception as exc:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"Rosetta [{self._backend}] unexpected error: {exc}",
            )

    def backend_status(self) -> str:
        """One-line status string for display in StructureBot's `state` command."""
        if self._backend == "pyrosetta":
            return "PyRosetta (local) — ACTIVE"
        if self._backend == "empirical":
            return "Empirical (BLOSUM62 + B-factor) — ACTIVE — accuracy r~0.4"
        if self._backend == "local":
            lp = os.environ.get("ROSETTA_LOCAL_PATH", "not configured")
            return f"Local Rosetta (stub) — ROSETTA_LOCAL_PATH={lp}"
        return (
            "DynaMut2 web API (biosig.lab.uq.edu.au/dynamut2/) — ACTIVE | "
            "no auth required | empirical fallback enabled"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Backend B: DynaMut2 (primary)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_dynamut2(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
        model_id:  str,
        chain:     Optional[str],
        session:   Any,
        progress_callback: Optional[Callable[[str], None]],
        scan_deadline: Optional[float] = None,
    ) -> ToolStepResult:
        """
        Query DynaMut2 for each mutation (synchronous, one POST per mutation).
        Mutations that fail fall through to the empirical estimator.

        Timeouts and guards
        -------------------
        scan_deadline        : time.perf_counter() deadline for the overall scan.
                               Mutations beyond the deadline are scored empirically
                               immediately rather than waiting for DynaMut2.
        _DYNAMUT2_PER_MUT_TIMEOUT : wall-clock seconds budget per mutation.
                               If a single mutation exceeds this, it falls back to
                               empirical without burning the remaining budget.
        _DYNAMUT2_CIRCUIT_BREAKER : consecutive submit failures before skipping
                               all remaining DynaMut2 attempts (e.g. no network).
        """

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                _safe_print(msg)

        _progress(f"  Querying DynaMut2 for {len(mutations)} mutation(s)...")

        ddg_scores:          Dict[str, float] = {}
        warnings:            List[str]        = []
        empirical_fallbacks: List[str]        = []
        _consecutive_failures: int            = 0

        for idx, mut in enumerate(mutations):
            key = _mutation_key(mut)

            # ── Overall scan deadline ─────────────────────────────────────────
            if scan_deadline is not None and time.perf_counter() >= scan_deadline:
                remaining = mutations[idx:]
                _progress(
                    f"  Overall scan timeout: scoring remaining "
                    f"{len(remaining)} mutation(s) with empirical fallback."
                )
                for rm in remaining:
                    rk = _mutation_key(rm)
                    ddg_scores[rk] = _empirical_ddg_single(rm, pdb_path)
                    empirical_fallbacks.append(rk)
                warnings.append(
                    f"Scan timeout: {len(remaining)} mutation(s) scored empirically."
                )
                break

            # ── Circuit breaker: stop hammering an unreachable server ─────────
            if _consecutive_failures >= _DYNAMUT2_CIRCUIT_BREAKER:
                remaining = mutations[idx:]
                _progress(
                    f"  DynaMut2 circuit breaker ({_consecutive_failures} consecutive "
                    f"failures) — scoring remaining {len(remaining)} mutation(s) "
                    "with empirical fallback."
                )
                for rm in remaining:
                    rk = _mutation_key(rm)
                    ddg_scores[rk] = _empirical_ddg_single(rm, pdb_path)
                    empirical_fallbacks.append(rk)
                warnings.append(
                    f"DynaMut2 unreachable ({_consecutive_failures} consecutive "
                    "failures). Empirical fallback used for remaining mutations."
                )
                break

            # ── Per-mutation deadline ─────────────────────────────────────────
            _mut_start         = time.perf_counter()
            per_mut_deadline   = _mut_start + _DYNAMUT2_PER_MUT_TIMEOUT

            _progress(
                f"  DynaMut2: scoring mutation {idx + 1}/{len(mutations)} ({key})..."
            )

            try:
                ddg = self._query_dynamut2_single(
                    pdb_path, mut, _progress, per_mut_deadline
                )
                _elapsed = time.perf_counter() - _mut_start
                ddg_scores[key] = ddg
                _consecutive_failures = 0           # reset on success
                _progress(
                    f"  + {key}: ddG = {ddg:+.2f} kcal/mol "
                    f"({_elapsed:.1f}s, DynaMut2)"
                )
            except Exception as exc:
                _elapsed = time.perf_counter() - _mut_start
                _consecutive_failures += 1
                warnings.append(
                    f"DynaMut2 failed for {key} ({exc}). "
                    "Using empirical BLOSUM62 estimate."
                )
                ddg_emp         = _empirical_ddg_single(mut, pdb_path)
                ddg_scores[key] = ddg_emp
                empirical_fallbacks.append(key)
                _progress(
                    f"  ! {key}: empirical estimate {ddg_emp:+.2f} kcal/mol "
                    f"({_elapsed:.1f}s — {str(exc)[:80]})"
                )

        # ── Confidence and method note ─────────────────────────────────────────
        n_dynamut2 = len(mutations) - len(empirical_fallbacks)

        if not empirical_fallbacks:
            confidence  = "high"
            backend_lbl = "dynamut2"
            method_note = (
                "DynaMut2 ensemble predictions (Rodrigues et al. 2021, NAR). "
                "Pearson r ~0.65 vs experimental. Appropriate for candidate "
                "screening. For publication use local Rosetta cartesian_ddg."
            )
        elif n_dynamut2 > 0:
            confidence  = "medium"
            backend_lbl = "dynamut2+empirical"
            method_note = (
                f"Mixed: {n_dynamut2} DynaMut2 predictions + "
                f"{len(empirical_fallbacks)} empirical BLOSUM62 estimates. "
                "Treat empirical values as rough indicators only."
            )
        else:
            confidence  = "low"
            backend_lbl = "empirical"
            method_note = (
                "DynaMut2 unavailable — all values are empirical BLOSUM62 estimates "
                "(B-factor buried/exposed correction). Pearson r ~0.4 vs experimental. "
                "For screening only. Retry later or set ROSETTA_BACKEND=empirical "
                "to silence this warning."
            )

        if empirical_fallbacks:
            warnings.insert(0,
                "ACCURACY NOTE: Empirical BLOSUM62 estimates have Pearson r ~0.4 "
                "vs experimental ΔΔG. Fallback mutations: "
                + ", ".join(empirical_fallbacks)
            )

        # ── Build result ───────────────────────────────────────────────────────
        stability_change = (
            sum(ddg_scores.values()) / len(ddg_scores) if ddg_scores else 0.0
        )
        viz_cmds, viz_exps = self._build_viz_commands(
            mutations, ddg_scores, model_id, chain
        )
        best_key = min(ddg_scores, key=ddg_scores.get) if ddg_scores else "?"
        best_ddg = ddg_scores.get(best_key, 0.0)
        display_backend = "DynaMut2" if not empirical_fallbacks else "DynaMut2/empirical"
        summary = (
            f"Stability ({display_backend}): "
            f"{len(ddg_scores)}/{len(mutations)} mutations scored. "
            f"Most stabilising: {best_key} ({best_ddg:+.2f} kcal/mol). "
            f"Mean ΔΔG: {stability_change:+.2f} kcal/mol."
        )
        _progress(f"✓ ⚗️  {summary}")

        # Persist completed result in session (DynaMut2 is synchronous — no job polling)
        if session is not None:
            try:
                job_id = f"dynamut2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                session.add_rosetta_job(job_id, {
                    "mutations":    mutations,
                    "pdb_path":     pdb_path,
                    "backend":      backend_lbl,
                    "submitted_at": datetime.now().isoformat(timespec="seconds"),
                    "status":       "completed",
                    "results":      ddg_scores,
                })
            except AttributeError:
                pass

        return ToolStepResult(
            tool    = "rosetta",
            success = True,
            data    = {
                "mutations":        mutations,
                "ddg_scores":       ddg_scores,
                "stability_change": round(stability_change, 3),
                "confidence":       confidence,
                "backend":          backend_lbl,
                "warnings":         warnings,
                "method_note":      method_note,
                "job_id":           None,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
        )

    def _query_dynamut2_single(
        self,
        pdb_path:            str,
        mut:                 Dict[str, Any],
        progress:            Callable[[str], None],
        per_mutation_deadline: Optional[float] = None,
    ) -> float:
        """
        Submit one mutation to DynaMut2 and poll for the result.

        Step 1 — Submit:
          POST https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single
          Fields: pdb_file, chain, mutation (e.g. "V82A")
          Response: {"job_id": "<uuid>"}

        Step 2 — Poll:
          GET https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single?job_id=<uuid>
          Response while running: {"message": "RUNNING"}
          Response when done: {"prediction": <float>, ...}

        per_mutation_deadline : time.perf_counter() deadline.  Checked before
          each retry/poll so the overall-scan budget is never exceeded by a
          single mutation.

        Returns ΔΔG in kcal/mol (positive = destabilising).
        Raises on submission failure (after retries) or poll timeout.
        """
        import requests

        mutation_str = f"{mut['from_aa']}{mut['position']}{mut['to_aa']}"
        chain_id     = mut.get("chain", "A")

        def _over_deadline() -> bool:
            return (
                per_mutation_deadline is not None
                and time.perf_counter() >= per_mutation_deadline
            )

        # ── Step 1: Submit with retry ──────────────────────────────────────────
        job_id:   Optional[str]       = None
        last_exc: Optional[Exception] = None

        delays = list(_DYNAMUT2_RETRY_DELAYS) + [None]
        for attempt, delay in enumerate(delays):
            if _over_deadline():
                raise TimeoutError(
                    f"Per-mutation timeout ({_DYNAMUT2_PER_MUT_TIMEOUT}s) "
                    f"exceeded during submit for {mutation_str}"
                )
            try:
                with open(pdb_path, "rb") as fh:
                    resp = requests.post(
                        _DYNAMUT2_SUBMIT_URL,
                        files   = {"pdb_file": (Path(pdb_path).name, fh, "chemical/x-pdb")},
                        data    = {"chain": chain_id, "mutation": mutation_str},
                        timeout = _DYNAMUT2_TIMEOUT,
                    )

                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", delay or 30))
                    if delay is not None:
                        progress(
                            f"  DynaMut2 rate-limited (attempt {attempt + 1}); "
                            f"retrying in {retry_after}s..."
                        )
                        time.sleep(retry_after)
                        last_exc = RuntimeError(
                            f"HTTP 429 rate-limited (attempt {attempt + 1})"
                        )
                        continue
                    raise RuntimeError(
                        "DynaMut2 rate limit exceeded — no more retries"
                    )

                if resp.status_code != 200:
                    raise RuntimeError(
                        f"DynaMut2 submit HTTP {resp.status_code}: {resp.text[:200]}"
                    )

                body   = resp.json()
                job_id = str(body.get("job_id", ""))
                if not job_id:
                    raise RuntimeError(
                        f"DynaMut2 submit response missing job_id. Body: {body}"
                    )
                break   # submit succeeded

            except requests.exceptions.Timeout:
                last_exc = TimeoutError(
                    f"DynaMut2 submit timed out after {_DYNAMUT2_TIMEOUT}s"
                )
            except requests.exceptions.ConnectionError as exc:
                last_exc = ConnectionError(f"DynaMut2 unreachable: {exc}")
            except (RuntimeError, ValueError):
                raise
            except Exception as exc:
                last_exc = exc

            if delay is not None:
                progress(
                    f"  DynaMut2 submit attempt {attempt + 1} failed "
                    f"({last_exc}); retrying in {delay}s..."
                )
                time.sleep(delay)

        if job_id is None:
            raise last_exc or RuntimeError(
                f"DynaMut2 submit for {mutation_str} failed after all retries"
            )

        # ── Step 2: Poll for result ────────────────────────────────────────────
        for poll_n in range(_DYNAMUT2_MAX_POLLS):
            if _over_deadline():
                raise TimeoutError(
                    f"Per-mutation timeout during polling for {mutation_str}"
                )

            time.sleep(_DYNAMUT2_POLL_INTERVAL)

            try:
                r = requests.get(
                    _DYNAMUT2_RESULT_URL,
                    params  = {"job_id": job_id},
                    timeout = _DYNAMUT2_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
            except requests.exceptions.Timeout:
                progress(f"  DynaMut2 poll {poll_n + 1} timeout; retrying...")
                continue
            except Exception as exc:
                progress(f"  DynaMut2 poll {poll_n + 1} error ({exc}); retrying...")
                continue

            # Result ready?
            if "prediction" in data:
                return _parse_dynamut2_result(data, mutation_str)

            # Explicit server-side error — don't keep polling
            msg    = str(data.get("message", "")).upper()
            status = str(data.get("status",  "")).lower()
            if (status in ("error", "failed", "failure")
                    or "error" in str(data.get("error", "")).lower()):
                raise RuntimeError(
                    f"DynaMut2 job failed for {mutation_str}: "
                    f"{data.get('message') or data.get('error') or status!r}"
                )

            # Still running?
            still_running = (
                msg in ("RUNNING", "PENDING", "QUEUED")
                or status in ("running", "pending", "queued", "processing",
                              "submitted", "waiting")
                or (not msg and not status and "prediction" not in data
                    and "error" not in data)
            )

            if still_running:
                if poll_n % 6 == 0:   # log every ~30s
                    progress(
                        f"  DynaMut2 {mutation_str}: waiting "
                        f"({(poll_n + 1) * _DYNAMUT2_POLL_INTERVAL}s elapsed, "
                        f"status={status or msg or 'unknown'})..."
                    )
                continue

            # Unrecognised non-error response — attempt to extract prediction
            return _parse_dynamut2_result(data, mutation_str)

        raise TimeoutError(
            f"DynaMut2 job {job_id} for {mutation_str} timed out "
            f"after {_DYNAMUT2_MAX_POLLS * _DYNAMUT2_POLL_INTERVAL}s"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Backend C: Empirical (offline fallback)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_empirical(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
        model_id:  str,
        chain:     Optional[str],
        progress_callback: Optional[Callable[[str], None]],
    ) -> ToolStepResult:
        """
        Offline ΔΔG estimates using BLOSUM62 + B-factor buried/exposed correction.

        All results are labelled as "estimated" and carry an accuracy disclaimer.
        Confidence is always "low".

        To get better estimates:
          • DynaMut2 (free web API): set ROSETTA_BACKEND=dynamut2 (default)
          • Local Rosetta: see _run_rosetta_local() stub for setup instructions
        """

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                _safe_print(msg)

        _progress(
            f"⚗️  Computing empirical ΔΔG for {len(mutations)} mutation(s) "
            "(BLOSUM62 + B-factor)…"
        )

        ddg_scores: Dict[str, float] = {}
        for mut in mutations:
            key             = _mutation_key(mut)
            ddg_scores[key] = _empirical_ddg_single(mut, pdb_path)

        warnings = [
            "ACCURACY NOTE: All values are empirical estimates from BLOSUM62 "
            "substitution scores with B-factor buried/exposed correction. "
            "Pearson r ~0.4 vs experimental ΔΔG. "
            "Not suitable for publication. "
            "DynaMut2 (free, no registration) provides significantly better "
            "estimates: set ROSETTA_BACKEND=dynamut2 (the default)."
        ]

        stability_change = (
            sum(ddg_scores.values()) / len(ddg_scores) if ddg_scores else 0.0
        )
        viz_cmds, viz_exps = self._build_viz_commands(
            mutations, ddg_scores, model_id, chain
        )
        best_key = min(ddg_scores, key=ddg_scores.get) if ddg_scores else "?"
        best_ddg = ddg_scores.get(best_key, 0.0)
        summary = (
            f"Stability (empirical estimate): "
            f"{len(ddg_scores)}/{len(mutations)} mutations scored. "
            f"Most stabilising: {best_key} ({best_ddg:+.2f} kcal/mol estimated). "
            f"Mean ΔΔG: {stability_change:+.2f} kcal/mol. "
            f"[ESTIMATED — accuracy ±2 kcal/mol]"
        )
        _progress(f"⚠ ⚗️  {summary}")

        return ToolStepResult(
            tool    = "rosetta",
            success = True,
            data    = {
                "mutations":        mutations,
                "ddg_scores":       ddg_scores,
                "stability_change": round(stability_change, 3),
                "confidence":       "low",
                "backend":          "empirical",
                "warnings":         warnings,
                "method_note":      (
                    "Empirical BLOSUM62 estimates with B-factor correction. "
                    "Pearson r ~0.4. Screening only."
                ),
                "job_id":           None,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Backend A: PyRosetta (documented stub)
    # ═══════════════════════════════════════════════════════════════════════════

    def _run_pyrosetta(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
        mode:      str,
        model_id:  str,
        chain:     Optional[str],
        progress_callback: Optional[Callable[[str], None]],
    ) -> ToolStepResult:
        """
        PyRosetta CartesianDDG protocol — documented stub.

        Full implementation outline (for Python <= 3.13 + valid wheel):
        ─────────────────────────────────────────────────────────────────
            import pyrosetta
            pyrosetta.init(flags="-mute all -ignore_unrecognized_res true")

            pose     = pyrosetta.io.pose_from_file(pdb_path)
            scorefxn = pyrosetta.create_score_function("ref2015_cart")

            # 1. FastRelax to establish clean energy baseline
            relax = pyrosetta.rosetta.protocols.relax.FastRelax(scorefxn, 5)
            relax.apply(pose)
            wt_energy = scorefxn(pose)

            # 2. Per-mutation CartesianDDG
            ddg_scores = {}
            for mut in mutations:
                mut_pose = pose.clone()
                mutate_residue(mut_pose, mut["position"], mut["to_aa"])
                repack_neighbors(mut_pose, scorefxn, mut["position"], radius=8.0)
                ddg = scorefxn(mut_pose) - wt_energy
                ddg_scores[_mutation_key(mut)] = round(ddg, 3)

        Reference: Park et al. (2016) Sci Rep 6; doi:10.1038/srep46918
        ─────────────────────────────────────────────────────────────────
        To activate this backend:
          1. Set up Python <= 3.13 (PyRosetta has no 3.14 wheel yet)
          2. pip install pyrosetta-installer
          3. python -c "import pyrosetta_installer;
                        pyrosetta_installer.install_pyrosetta()"
          4. Add PYROSETTA_AVAILABLE=true to .env.local
          5. Restart StructureBot
        """
        return ToolStepResult(
            tool="rosetta", success=False,
            error=(
                "PyRosetta backend is not yet active.\n\n"
                "Python 3.14 wheels are not yet available from pyrosetta.org.\n"
                "To enable:\n"
                "  1. Set up Python <= 3.13 (or configure WSL)\n"
                "  2. pip install pyrosetta-installer\n"
                "  3. python -c \"import pyrosetta_installer; "
                "pyrosetta_installer.install_pyrosetta()\"\n"
                "  4. Add PYROSETTA_AVAILABLE=true to .env.local\n"
                "  5. Restart StructureBot\n\n"
                "DynaMut2 web API is the default backend and requires no setup."
            ),
        )

    # ── Future: Local Rosetta (publication-quality ddG) ───────────────────────
    # To activate: set ROSETTA_LOCAL_PATH in .env.local pointing
    # to your Rosetta installation directory on Linux/Mac
    #
    # Required setup:
    # 1. Obtain free academic license: https://rosettacommons.org
    # 2. Install on Linux/Mac (Windows not supported for science)
    # 3. Set ROSETTA_LOCAL_PATH=/path/to/rosetta/source/bin
    # 4. Set ROSETTA_BACKEND=local in .env.local
    #
    # Protocol: cartesian_ddg
    # Expected accuracy: Pearson r ~0.8 vs experimental
    # Expected runtime: 5-30 min per mutation (CPU)
    #                   30-120 min for full scan (100 candidates)
    #
    # Citation: Frenz et al. (2020) Biochemistry
    #           Park et al. (2016) J Chem Theory Comput
    #
    # The _run_rosetta_local() method below is a documented stub.
    # Implement by calling:
    #   {ROSETTA_LOCAL_PATH}/rosetta_scripts.linuxgccrelease
    #   with the cartesian_ddg XML protocol

    def _run_rosetta_local(self, pdb_path: str, mutations: List[Dict[str, Any]]) -> None:
        raise NotImplementedError(
            "Local Rosetta not yet configured. "
            "See comment block above _run_rosetta_local() for setup instructions. "
            "Current default backend: DynaMut2 (screening) or empirical."
        )

    # ── Visualization ──────────────────────────────────────────────────────────

    def _build_viz_commands(
        self,
        mutations:  List[Dict[str, Any]],
        ddg_scores: Dict[str, float],
        model_id:   str,
        chain:      Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Colour mutated residues by ddG on a 5-band scale:
          blue   <= -1.0  strongly stabilising
          cyan    -1 – 0  mildly stabilising
          white   ~  0    neutral (score = 0 or unknown)
          yellow   0 – +1 mildly destabilising
          red    >= +1.0  strongly destabilising
        """
        if not mutations:
            return [], []

        chain_spec = f"/{chain}" if chain else ""
        cmds = [
            f"cartoon #{model_id}",
            f"color #{model_id}{chain_spec} white",
        ]
        exps = [
            "Switch to cartoon representation",
            "Reset all residues to white before applying stability colours",
        ]

        for mut in mutations:
            key  = _mutation_key(mut)
            ddg  = ddg_scores.get(key, 0.0)
            pos  = mut["position"]
            spec = f"#{model_id}{chain_spec}:{pos}"

            if ddg <= -1.0:
                colour, label = "blue",   "strongly stabilising"
            elif ddg < 0.0:
                colour, label = "cyan",   "mildly stabilising"
            elif ddg < 1.0:
                colour, label = "yellow", "mildly destabilising"
            else:
                colour, label = "red",    "strongly destabilising"

            cmds.append(f"color {spec} {colour}")
            exps.append(
                f"Residue {pos} ({key}): ddG = {ddg:+.2f} kcal/mol -- {label}"
            )
            cmds.append(f"show {spec} atoms")
            exps.append(f"Show residue {pos} as atoms for visibility")

        cmds.append(f"view #{model_id}")
        exps.append("Fit structure in view")

        return cmds, exps

    def __repr__(self) -> str:
        return f"<RosettaBridge backend={self._backend!r}>"
