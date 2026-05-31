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
    1. POST /api/prediction_single (data: chain, mutation; files: pdb_file)
         ->  {"job_id": "..."}
    2. GET  /api/prediction_single (job_id)
         ->  while running: {"status": "RUNNING", "job_id": ...}
         ->  when done:     {"status": "DONE", "prediction": <float>, ...}
       (poll until status is DONE / a numeric "prediction" appears)
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
    GET https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single (job_id)
    Response while running: {"status": "RUNNING", "job_id": "<uuid>"}
    Response when done:
      {"status": "DONE", "prediction": <float>, "chain": "A",
       "wild-type": "ILE", "mutant": "ARG", "position": "72",
       "results_page": "<url>"}

  Sign convention: positive prediction = destabilising, negative = stabilising.
  Auth: none required.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import config as _cfg
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
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


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

    Current API format (prediction_single), confirmed live 2026-05:
      {"status": "DONE", "prediction": 0.789, "chain": "A",
       "wild-type": "ILE", "mutant": "ARG", "position": "72",
       "results_page": "<url>"}
    Legacy format (still accepted):
      {"prediction": "1.4", "chain": "A", "res_number": 82, ...}

    Sign convention: positive = destabilising, negative = stabilising
    (matches StructureBot's internal convention — no flip needed).

    Raises:
      RuntimeError if the job reports a server-side error (status=ERROR) —
        the caller must treat this as a failure, never as a real ddG of 0.0.
      ValueError if the response is not yet complete (status RUNNING/PENDING/
        QUEUED, or legacy {"message": "RUNNING"}) or lacks a numeric prediction.
    """
    status = str(data.get("status", "")).strip().lower()
    msg    = str(data.get("message", "")).strip().upper()

    # Server-side error — distinct from a valid 0.0 prediction.
    if (status in ("error", "failed", "failure")
            or "error" in str(data.get("error", "")).lower()):
        raise RuntimeError(
            f"DynaMut2 job reported an error for {mutation_str!r}: "
            f"{data.get('message') or data.get('error') or status!r}"
        )

    # Still running — not a completed result yet.
    if (status in ("running", "pending", "queued", "processing", "submitted", "waiting")
            or msg in ("RUNNING", "PENDING", "QUEUED")):
        raise ValueError(
            f"DynaMut2 job for {mutation_str!r} not yet complete "
            f"(status={status or msg or 'unknown'!r})"
        )

    # Completed result: extract "prediction" (cast robustly — the live API
    # returns a float, the legacy docs show a string).
    if data.get("prediction") is not None:
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


def _aggregate_ddg_trajectories(trajectories):
    """
    Aggregate per-trajectory ΔΔG values → (median, MAD spread).

    Canonical aggregation rule for the multi-trajectory ddG (the WSL2 worker
    mirrors this exact formula): the reported ddG is the **median** — NOT the
    mean (outlier-sensitive) and NOT the min (two-sided trajectory noise makes
    min invent fake stabilisers; see scripts/rosetta_validation_notes.md). The
    spread is the median absolute deviation (MAD), robust and consistent with
    the median. Returns (None, None) if there are no usable values; MAD is 0.0
    for a single trajectory.
    """
    import statistics as _stats
    vals = [float(x) for x in (trajectories or []) if x is not None]
    if not vals:
        return None, None
    med = _stats.median(vals)
    mad = _stats.median([abs(x - med) for x in vals]) if len(vals) > 1 else 0.0
    return round(med, 3), round(mad, 3)


def _ddg_confidence_label(spread: Optional[float], n_trajectories: Optional[int]) -> str:
    """
    Confidence label for a (multi-trajectory) PyRosetta ddG, from the
    median-absolute-deviation spread across trajectories.

      "single-trajectory"  n <= 1 (spread undefined — NOT labelled high)
      "high"               spread <= 1.5 kcal/mol
      "moderate"           spread <= ROSETTA_SPREAD_LOW_CONFIDENCE (default 3.0)
      "low"                spread >  ROSETTA_SPREAD_LOW_CONFIDENCE

    Note: even "high" confidence means trajectory agreement, NOT calibrated
    accuracy — magnitudes carry roughly 2-3 kcal/mol uncertainty across all
    structural categories (not just large cavities).
    """
    if not n_trajectories or n_trajectories <= 1 or spread is None:
        return "single-trajectory"
    try:
        import config as _cfg
        low = float(getattr(_cfg, "ROSETTA_SPREAD_LOW_CONFIDENCE", 3.0))
    except Exception:
        low = 3.0
    if spread <= 1.5:
        return "high"
    if spread <= low:
        return "moderate"
    return "low"


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
                return self._run_rosetta_local(
                    pdb_path, mutations, model_id, chain, progress_callback
                )
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
            try:
                from wsl_bridge import WSLBridge
                wsl = WSLBridge()
                if wsl.is_available():
                    has_py = wsl.check_pyrosetta()
                    if has_py:
                        return "Local Rosetta: PyRosetta via WSL2 — ACTIVE (publication quality)"
                    return "Local Rosetta: WSL2 available but PyRosetta not installed in WSL2"
                return (
                    "Local Rosetta: WSL2 not installed — "
                    "run `wsl --install -d Ubuntu-24.04` (Administrator)"
                )
            except ImportError:
                pass
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
        Query DynaMut2 for each mutation.
        Dispatches to parallel or sequential based on config.DYNAMUT2_MAX_WORKERS.

        Timeouts and guards
        -------------------
        scan_deadline             : time.perf_counter() deadline for the overall scan.
        _DYNAMUT2_PER_MUT_TIMEOUT : wall-clock seconds budget per mutation.
        _DYNAMUT2_CIRCUIT_BREAKER : consecutive failures before empirical fallback.
        """

        def _progress(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                _safe_print(msg)

        max_workers = getattr(_cfg, "DYNAMUT2_MAX_WORKERS", 4)

        # Use parallel execution when there are multiple mutations and workers > 1
        if max_workers > 1 and len(mutations) > 1:
            ddg_scores, empirical_fallbacks, warnings = self._score_mutations_parallel(
                pdb_path, mutations, max_workers, scan_deadline, _progress
            )
        else:
            ddg_scores, empirical_fallbacks, warnings = self._score_mutations_sequential(
                pdb_path, mutations, scan_deadline, _progress
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

    # ── Sequential scoring loop ────────────────────────────────────────────────

    def _score_mutations_sequential(
        self,
        pdb_path:     str,
        mutations:    List[Dict[str, Any]],
        scan_deadline: Optional[float],
        progress:     Callable[[str], None],
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """Score mutations one at a time (original behaviour)."""
        ddg_scores:          Dict[str, float] = {}
        empirical_fallbacks: List[str]        = []
        warnings:            List[str]        = []
        consecutive_failures: int             = 0

        progress(f"  Querying DynaMut2 for {len(mutations)} mutation(s)...")

        for idx, mut in enumerate(mutations):
            key = _mutation_key(mut)

            if scan_deadline is not None and time.perf_counter() >= scan_deadline:
                remaining = mutations[idx:]
                progress(
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

            if consecutive_failures >= _DYNAMUT2_CIRCUIT_BREAKER:
                remaining = mutations[idx:]
                progress(
                    f"  DynaMut2 circuit breaker ({consecutive_failures} consecutive "
                    f"failures) — scoring remaining {len(remaining)} mutation(s) "
                    "with empirical fallback."
                )
                for rm in remaining:
                    rk = _mutation_key(rm)
                    ddg_scores[rk] = _empirical_ddg_single(rm, pdb_path)
                    empirical_fallbacks.append(rk)
                warnings.append(
                    f"DynaMut2 unreachable ({consecutive_failures} consecutive "
                    "failures). Empirical fallback used for remaining mutations."
                )
                break

            mut_start        = time.perf_counter()
            per_mut_deadline = mut_start + _DYNAMUT2_PER_MUT_TIMEOUT
            progress(
                f"  DynaMut2: scoring mutation {idx + 1}/{len(mutations)} ({key})..."
            )

            try:
                ddg = self._query_dynamut2_single(
                    pdb_path, mut, progress, per_mut_deadline
                )
                elapsed = time.perf_counter() - mut_start
                ddg_scores[key] = ddg
                consecutive_failures = 0
                progress(
                    f"  + {key}: ddG = {ddg:+.2f} kcal/mol ({elapsed:.1f}s, DynaMut2)"
                )
            except Exception as exc:
                elapsed = time.perf_counter() - mut_start
                consecutive_failures += 1
                warnings.append(
                    f"DynaMut2 failed for {key} ({exc}). "
                    "Using empirical BLOSUM62 estimate."
                )
                ddg_emp = _empirical_ddg_single(mut, pdb_path)
                ddg_scores[key] = ddg_emp
                empirical_fallbacks.append(key)
                progress(
                    f"  ! {key}: empirical estimate {ddg_emp:+.2f} kcal/mol "
                    f"({elapsed:.1f}s — {str(exc)[:80]})"
                )

        return ddg_scores, empirical_fallbacks, warnings

    # ── Parallel scoring ────────────────────────────────────────────────────────

    def _score_mutations_parallel(
        self,
        pdb_path:     str,
        mutations:    List[Dict[str, Any]],
        max_workers:  int,
        scan_deadline: Optional[float],
        progress:     Callable[[str], None],
    ) -> Tuple[Dict[str, float], List[str], List[str]]:
        """
        Score all mutations concurrently using a thread pool.

        Thread safety
        -------------
        - Circuit breaker counter protected by _cb_lock (threading.Lock).
        - _circuit_broken (threading.Event) signals remaining workers to skip DynaMut2.
        - Progress output collected per-future in the main thread (no interleaving).
        - _query_dynamut2_single uses only local variables — no shared mutable state.
        - Each worker gets its own per-mutation deadline based on start time.
        """
        ddg_scores:          Dict[str, float] = {}
        empirical_fallbacks: List[str]        = []
        warnings:            List[str]        = []

        _cb_lock        = threading.Lock()
        _consec_fail    = [0]          # list so inner closure can mutate
        _circuit_broken = threading.Event()

        batch_start = time.perf_counter()
        progress(
            f"  DynaMut2: scoring {len(mutations)} mutation(s) "
            f"concurrently (max_workers={max_workers})..."
        )

        def _score_one(mut: Dict[str, Any]) -> Tuple[str, float, str, float]:
            """Worker function — returns (key, ddg, source, elapsed_s)."""
            key = _mutation_key(mut)

            # Fast exits: circuit broken or scan deadline already passed
            if _circuit_broken.is_set():
                return key, _empirical_ddg_single(mut, pdb_path), "circuit_breaker", 0.0

            if scan_deadline is not None and time.perf_counter() >= scan_deadline:
                return key, _empirical_ddg_single(mut, pdb_path), "deadline", 0.0

            per_mut_deadline = time.perf_counter() + _DYNAMUT2_PER_MUT_TIMEOUT
            start = time.perf_counter()

            try:
                # Pass a silent no-op for progress; completion reported by main thread
                ddg = self._query_dynamut2_single(
                    pdb_path, mut, lambda _: None, per_mut_deadline
                )
                with _cb_lock:
                    _consec_fail[0] = 0   # reset on success
                return key, ddg, "dynamut2", time.perf_counter() - start
            except Exception as exc:
                with _cb_lock:
                    _consec_fail[0] += 1
                    if _consec_fail[0] >= _DYNAMUT2_CIRCUIT_BREAKER:
                        _circuit_broken.set()
                elapsed = time.perf_counter() - start
                return key, _empirical_ddg_single(mut, pdb_path), str(exc), elapsed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_mut = {
                executor.submit(_score_one, mut): mut
                for mut in mutations
            }
            for future in as_completed(future_to_mut):
                mut = future_to_mut[future]
                key = _mutation_key(mut)
                try:
                    r_key, ddg, source, elapsed = future.result()
                    ddg_scores[r_key] = ddg
                    if source == "dynamut2":
                        progress(
                            f"  + {r_key}: ddG = {ddg:+.2f} kcal/mol "
                            f"({elapsed:.1f}s, DynaMut2)"
                        )
                    elif source == "circuit_breaker":
                        empirical_fallbacks.append(r_key)
                        progress(
                            f"  ! {r_key}: circuit breaker "
                            f"(empirical {ddg:+.2f} kcal/mol)"
                        )
                    elif source == "deadline":
                        empirical_fallbacks.append(r_key)
                        progress(
                            f"  ! {r_key}: scan deadline "
                            f"(empirical {ddg:+.2f} kcal/mol)"
                        )
                    else:
                        empirical_fallbacks.append(r_key)
                        warnings.append(
                            f"DynaMut2 failed for {r_key} ({source}). "
                            "Using empirical BLOSUM62 estimate."
                        )
                        progress(
                            f"  ! {r_key}: empirical {ddg:+.2f} kcal/mol "
                            f"({source[:60]})"
                        )
                except Exception as exc:
                    # Future itself raised (shouldn't happen; guard anyway)
                    ddg = _empirical_ddg_single(mut, pdb_path)
                    ddg_scores[key] = ddg
                    empirical_fallbacks.append(key)
                    progress(f"  ! {key}: unexpected error (empirical {ddg:+.2f})")

        wall = time.perf_counter() - batch_start
        progress(
            f"  DynaMut2: batch complete "
            f"({wall:.1f}s wall, {max_workers} concurrent)"
        )

        if _circuit_broken.is_set():
            warnings.append(
                f"DynaMut2 circuit breaker tripped after "
                f"{_DYNAMUT2_CIRCUIT_BREAKER} consecutive failures. "
                "Remaining mutations used empirical BLOSUM62 estimates."
            )

        return ddg_scores, empirical_fallbacks, warnings

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
          GET https://biosig.lab.uq.edu.au/dynamut2/api/prediction_single (job_id)
          Response while running: {"status": "RUNNING", "job_id": "<uuid>"}
          Response when done:     {"status": "DONE", "prediction": <float>, ...}

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
                # The live prediction_single endpoint reads job_id from the form
                # body (data=); send it as a query param too for backward compat.
                r = requests.get(
                    _DYNAMUT2_RESULT_URL,
                    params  = {"job_id": job_id},
                    data    = {"job_id": job_id},
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

            msg    = str(data.get("message", "")).strip().upper()
            status = str(data.get("status",  "")).strip().lower()

            # Explicit server-side error — don't keep polling. Raising here
            # surfaces a failure upstream (→ empirical fallback), keeping a real
            # ddG of 0.0 distinguishable from a failed lookup.
            if (status in ("error", "failed", "failure")
                    or "error" in str(data.get("error", "")).lower()):
                raise RuntimeError(
                    f"DynaMut2 job failed for {mutation_str}: "
                    f"{data.get('message') or data.get('error') or status!r}"
                )

            # Completed? Current API uses status=DONE; legacy responses carry a
            # top-level numeric "prediction" with no status.
            if (status in ("done", "finished", "complete", "completed", "success")
                    or data.get("prediction") is not None):
                return _parse_dynamut2_result(data, mutation_str)

            # Still running?
            still_running = (
                msg in ("RUNNING", "PENDING", "QUEUED")
                or status in ("running", "pending", "queued", "processing",
                              "submitted", "waiting")
                or (not msg and not status)   # empty/unknown → assume spinning up
            )

            if still_running:
                if poll_n % 6 == 0:   # log every ~30s
                    progress(
                        f"  DynaMut2 {mutation_str}: waiting "
                        f"({(poll_n + 1) * _DYNAMUT2_POLL_INTERVAL}s elapsed, "
                        f"status={status or msg or 'unknown'})..."
                    )
                continue

            # Unrecognised non-error, non-running response — try to parse;
            # if it isn't a usable result, keep polling rather than crash.
            try:
                return _parse_dynamut2_result(data, mutation_str)
            except ValueError:
                continue

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

    # ── Local Rosetta (publication-quality ddG via PyRosetta in WSL2) ─────────
    # ACTIVE implementation (not a stub).  Selected when ROSETTA_BACKEND=local.
    #
    # How it works:
    #   1. Copy the PDB into WSL2 /tmp (wsl_bridge.copy_to_wsl)
    #   2. Build a standalone PyRosetta worker script and run it under
    #      WSL2 (wsl_bridge.run_python_script → PYROSETTA_PYTHON)
    #   3. Worker: cleanATOM → FastRelax (cached by PDB hash) → per-mutation
    #      MutateResidue + FastRelax → ddG = score(mut) − score(wt)
    #   4. Copy the results JSON back and parse per-mutation ΔΔG
    #
    # Requirements: WSL2 (Ubuntu-24.04) with PyRosetta installed in
    #   PYROSETTA_PYTHON (see wsl_bridge.py).  Academic license required:
    #   https://rosettacommons.org
    #
    # Protocol: per-mutation FastRelax ddG (ref2015).  Pearson r ~0.8 vs
    # experimental.  Runtime: ~2–5 min per mutation on CPU.
    # Citation: Park et al. (2016) J Chem Theory Comput; Frenz et al. (2020).
    #
    # If the WSL2/PyRosetta run fails as a whole, this method falls back to
    # per-mutation empirical BLOSUM62 estimates (marked ddg_source="empirical")
    # rather than returning all-zero ddG.

    def _run_rosetta_local(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
        model_id:  str = "1",
        chain:     Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        relax_cycles: int = 3,
        num_trajectories: Optional[int] = None,
    ) -> "ToolStepResult":
        """
        PyRosetta cartesian_ddg protocol via WSL2.

        relax_cycles : FastRelax cycles for the per-mutation mutant relax AND
            the symmetric WT re-relax (default 3 = production behaviour; do not
            change for the production scan path). Exposed only so the validation
            tier / convergence diagnostics can sweep the cycle count; the cached
            WT baseline relax is unaffected.
        num_trajectories : independent relax+score trajectories per mutation;
            reported ddG is the MEDIAN, with a spread (MAD) for confidence.
            None → config.ROSETTA_NUM_TRAJECTORIES (default 1 = production,
            single trajectory, byte-for-byte unchanged behaviour).

        Requires
        --------
        1. WSL2 installed:  wsl --install -d Ubuntu-24.04  (Administrator)
        2. PyRosetta in WSL2:
             pip install pyrosetta-installer
             python3 -c "import pyrosetta_installer;
                         pyrosetta_installer.install_pyrosetta()"
        3. ROSETTA_BACKEND=local in .env.local

        Protocol
        --------
        1. Copy PDB to WSL2 /tmp
        2. FastRelax the structure (cache result by PDB hash in ROSETTA_RELAX_CACHE)
        3. CartesianDDG for each mutation
        4. Copy results back, parse ΔΔG values

        Pearson r ~0.8 vs experimental.  Runtime: 2–5 min per mutation on CPU.

        Citation: Park et al. (2016) J Chem Theory Comput; Frenz et al. (2020)
        """
        def _prog(msg: str) -> None:
            if progress_callback:
                progress_callback(msg)
            else:
                _safe_print(msg)

        # Per-mutation relax cycles (3 = production default) and trajectory count
        # (config default 1 = production). int-coerced so the values interpolated
        # into the worker script are always safe integers.
        import config as _cfg_traj
        relax_cycles = max(1, int(relax_cycles))
        if num_trajectories is None:
            num_trajectories = getattr(_cfg_traj, "ROSETTA_NUM_TRAJECTORIES", 1)
        num_trajectories = max(1, int(num_trajectories))

        try:
            from wsl_bridge import WSLBridge
        except ImportError:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "wsl_bridge module not found. "
                    "Ensure wsl_bridge.py is in the project directory."
                ),
            )

        wsl = WSLBridge()

        if not wsl.is_available():
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "Local Rosetta (PyRosetta via WSL2) is not configured.\n"
                    "WSL2 is not installed or Ubuntu-24.04 distribution is not found.\n\n"
                    "To enable (PowerShell as Administrator):\n"
                    "  1. wsl --install -d Ubuntu-24.04\n"
                    "  2. Reboot\n"
                    "  3. Inside WSL2:\n"
                    "       pip install pyrosetta-installer\n"
                    "       python3 -c \"import pyrosetta_installer; "
                    "pyrosetta_installer.install_pyrosetta()\"\n"
                    "  4. Restart StructureBot\n\n"
                    "DynaMut2 web API (default) requires no setup and is appropriate "
                    "for candidate screening."
                ),
            )

        if not wsl.check_pyrosetta():
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "WSL2 is available but PyRosetta is not installed inside it.\n"
                    "In your WSL2 terminal (Ubuntu-24.04), run:\n"
                    "  pip install pyrosetta-installer\n"
                    "  python3 -c \"import pyrosetta_installer; "
                    "pyrosetta_installer.install_pyrosetta()\"\n"
                    "This downloads ~1.5 GB and may take 20–30 minutes."
                ),
            )

        _prog("⚗️  [Rosetta] Copying PDB to WSL2...")

        # Copy PDB to WSL2 temp directory
        wsl_pdb = wsl.copy_to_wsl(pdb_path)
        if not wsl_pdb:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=f"Failed to copy {pdb_path} to WSL2 /tmp.",
            )

        # Build PyRosetta script to run cartesian_ddg
        import hashlib, json
        pdb_hash = hashlib.md5(open(pdb_path, "rb").read()).hexdigest()[:12]

        mut_list_json = json.dumps([
            {
                "chain":   m.get("chain", "A"),
                "pos":     m.get("position", 1),
                "from_aa": m.get("from_aa", "A"),
                "to_aa":   m.get("to_aa", "A"),
            }
            for m in mutations
        ])

        import config as _cfg
        relax_cache_wsl = wsl.translate_path(str(_cfg.ROSETTA_RELAX_CACHE))
        # Crystallographic-water handling (default: preserve). The worker below
        # re-appends HOH that cleanATOM strips. Namespace the relax cache key by
        # mode so preserved-water relaxed structures never reuse a previously
        # cached stripped-water one; the stripped-water key is byte-identical to
        # before, so the committed validation numbers stay valid for that path.
        _strip_waters = bool(getattr(_cfg, "ROSETTA_STRIP_WATERS", False))
        _strip_py = "True" if _strip_waters else "False"
        _relax_key = pdb_hash + ("" if _strip_waters else "_wat")
        wsl_results_path = f"/tmp/rosetta_ddg_{pdb_hash}.json"

        script = f"""
import json, sys, os
sys.path.insert(0, '/opt/conda/lib/python3.12/site-packages')  # common PyRosetta location

try:
    import pyrosetta
    from pyrosetta import init as rosetta_init, pose_from_file
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.toolbox import cleanATOM

    rosetta_init(options="-mute all -ex1 -ex2 -use_input_sc -ignore_unrecognized_res true")

    pdb_path  = {wsl_pdb!r}
    mutations = json.loads({mut_list_json!r})
    cache_dir = {relax_cache_wsl!r}
    hash_key  = {_relax_key!r}
    strip_waters = {_strip_py}
    relaxed   = os.path.join(cache_dir, f"{{hash_key}}_relaxed.pdb")

    os.makedirs(cache_dir, exist_ok=True)

    # cleanATOM + size guard: keep the cleaned file only if it is non-trivial.
    cleaned_path = pdb_path.replace('.pdb', '_clean.pdb')
    if cleaned_path == pdb_path:            # input had no .pdb extension
        cleaned_path = pdb_path + '_clean.pdb'
    cleanATOM(pdb_path, cleaned_path)
    if os.path.isfile(cleaned_path) and os.path.getsize(cleaned_path) > 100:
        pdb_path = cleaned_path
    print(f"[Rosetta] PDB cleaned → {{pdb_path}}", flush=True)

    # PRESERVE crystallographic waters (default): cleanATOM drops ALL HETATM,
    # HOH included. When strip_waters is False, re-append the HOH records from
    # the untouched original PDB so buried structural waters reach the pose
    # (targets wrong-sign buried ddG, e.g. T26A / G88V). The "_wat" cache-key
    # suffix keeps these relaxed structures separate from stripped-water ones.
    if not strip_waters and pdb_path == cleaned_path:
        try:
            _hoh = [ln for ln in open({wsl_pdb!r})
                    if ln[:6] in ("HETATM", "ATOM  ") and ln[17:20].strip() == "HOH"]
            if _hoh:
                with open(cleaned_path) as _fh:
                    _body = [ln for ln in _fh if not ln.startswith("END")]
                _wat_path = cleaned_path + ".wat.pdb"
                with open(_wat_path, "w") as _fh:
                    _fh.writelines(_body)
                    _fh.writelines(_hoh)
                    _fh.write("END\\n")
                pdb_path = _wat_path
                print(f"[Rosetta] Preserved {{len(_hoh)}} crystallographic water atom(s)", flush=True)
            else:
                print("[Rosetta] No crystallographic waters (HOH) in input", flush=True)
        except Exception as _we:
            print(f"[Rosetta] Water preservation skipped: {{_we}}", flush=True)

    # Validate before loading: pose_from_file infers type from extension/content,
    # so an empty/HTML file or a non-.pdb name raises "Cannot determine file type".
    if not os.path.isfile(pdb_path) or os.path.getsize(pdb_path) == 0:
        raise RuntimeError(f"PDB to load is missing/empty: {{pdb_path}}")
    _head = []
    with open(pdb_path, "r", errors="replace") as _fh:
        for _i, _ln in enumerate(_fh):
            if _i >= 5:
                break
            _head.append(_ln.rstrip("\\n"))
    _valid_starts = ("ATOM", "HETATM", "HEADER", "CRYST", "MODEL",
                     "REMARK", "TITLE", "SEQRES", "EXPDTA", "COMPND")
    if not any(any(_l.startswith(_p) for _p in _valid_starts) for _l in _head):
        raise RuntimeError(f"File does not look like PDB; first lines: {{_head}}")

    # Ensure the loaded path ends in .pdb so file-type inference succeeds.
    if not pdb_path.endswith(".pdb"):
        import shutil as _shutil
        _renamed = pdb_path + ".pdb"
        _shutil.copyfile(pdb_path, _renamed)
        pdb_path = _renamed

    try:
        pose = pose_from_file(pdb_path)
    except Exception:
        # Explicit PDB reader fallback when type inference fails.
        from pyrosetta import pose_from_pdb
        pose = pose_from_pdb(pdb_path)
    scorefxn = pyrosetta.create_score_function("ref2015")

    if os.path.isfile(relaxed):
        print(f"[Rosetta] Loading cached relaxed structure: {{relaxed}}", flush=True)
        wt_pose = pose_from_file(relaxed)
    else:
        print("[Rosetta] FastRelax (2-5 min)...", flush=True)
        relax = FastRelax(scorefxn, 5)
        wt_pose = pose.clone()
        relax.apply(wt_pose)
        wt_pose.dump_pdb(relaxed)
        print(f"[Rosetta] Relax complete → {{relaxed}}", flush=True)

    import statistics as _stats
    import random as _pyrng
    _n_traj = {num_trajectories}
    _base_seed = _pyrng.SystemRandom().randint(1, 2000000000)
    print(f"[Rosetta] Starting mutation scoring, {{len(mutations)}} mutation(s) "
          f"({{_n_traj}} trajectory/ies x {relax_cycles} relax cycles)", flush=True)
    scorefxn_std = pyrosetta.create_score_function("ref2015")
    results = {{}}
    for mut in mutations:
        chain_id = mut["chain"]
        pos      = int(mut["pos"])
        from_aa  = mut["from_aa"]
        to_aa    = mut["to_aa"]
        key      = f"{{from_aa}}{{pos}}{{to_aa}}"
        try:
            from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
            from pyrosetta.rosetta.protocols.relax import FastRelax
            _aa1to3 = {{'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE',
                        'G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU',
                        'M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG',
                        'S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR'}}
            to_aa3 = _aa1to3.get(to_aa, to_aa)
            res_num = wt_pose.pdb_info().pdb2pose(chain_id, pos)
            if res_num == 0:
                raise ValueError(f"Residue {{pos}}{{chain_id}} not found in pose")
            _traj = []
            for _t in range(_n_traj):
                # Independent random seed per trajectory (only when N>1, so the
                # N=1 production path is identical to the previous single run).
                if _n_traj > 1:
                    try:
                        from pyrosetta.rosetta.numeric.random import rg as _rg
                        _rg().set_seed(int((_base_seed + _t * 104729) % 2147483647))
                    except Exception:
                        pass  # seeding best-effort; the global RNG still advances
                mut_pose = wt_pose.clone()
                MutateResidue(target=res_num, new_res=to_aa3).apply(mut_pose)
                FastRelax(scorefxn_std, {relax_cycles}).apply(mut_pose)
                wt_re = wt_pose.clone()
                FastRelax(scorefxn_std, {relax_cycles}).apply(wt_re)
                _ddg_t = scorefxn_std(mut_pose) - scorefxn_std(wt_re)
                _traj.append(round(float(_ddg_t), 3))
                print(f"[Rosetta] {{key}} traj {{_t+1}}/{{_n_traj}}: "
                      f"ddG = {{_ddg_t:+.2f}} kcal/mol", flush=True)
            _median = _stats.median(_traj)
            # Median absolute deviation — robust spread, consistent with median.
            _mad = _stats.median([abs(x - _median) for x in _traj]) if len(_traj) > 1 else 0.0
            results[key] = {{"ddg": round(float(_median), 3),
                             "spread": round(float(_mad), 3),
                             "n": len(_traj),
                             "trajectories": _traj}}
            print(f"[Rosetta] {{key}}: median ddG = {{_median:+.2f}} kcal/mol "
                  f"(spread {{_mad:.2f}}, n={{len(_traj)}})", flush=True)
        except Exception as e:
            import traceback
            results[key] = None
            print(f"[Rosetta] {{key}} failed: {{e}}", flush=True)
            traceback.print_exc()

    with open({wsl_results_path!r}, "w") as f:
        json.dump(results, f)
    print(f"[Rosetta] Results written to {wsl_results_path!r}", flush=True)
    print("[Rosetta] Done.", flush=True)

except Exception as exc:
    import traceback
    print(f"[Rosetta] FATAL: {{exc}}", flush=True)
    traceback.print_exc()
    with open({wsl_results_path!r}, "w") as f:
        json.dump({{"error": str(exc)}}, f)
"""

        import tempfile, os
        debug_path = os.path.join(tempfile.gettempdir(), "structurebot_worker_debug.py")
        with open(debug_path, "w", encoding="utf-8") as _dbg:
            _dbg.write(script)

        # Defensive whole-batch fallback: if the PyRosetta WSL2 run fails as a
        # whole, score every mutation with the empirical BLOSUM62 estimate
        # (marked ddg_source="empirical") instead of returning all-zero ddG.
        def _empirical_result(reason: str) -> "ToolStepResult":
            _prog(f"  [Rosetta] {reason} — falling back to empirical BLOSUM62 estimates.")
            _scores: Dict[str, float] = {}
            _source: Dict[str, str]   = {}
            _spread: Dict[str, Optional[float]] = {}
            _conf:   Dict[str, str]   = {}
            for _m in mutations:
                _k = _mutation_key(_m)
                _scores[_k] = _empirical_ddg_single(_m, pdb_path)
                _source[_k] = "empirical"
                _spread[_k] = None
                _conf[_k]   = "empirical"
            _warns = [
                f"PyRosetta WSL2 failed ({reason}). All ddG values are empirical "
                "BLOSUM62 + B-factor estimates (Pearson r ~0.4). For screening only."
            ]
            _change = sum(_scores.values()) / len(_scores) if _scores else 0.0
            _vc, _ve = self._build_viz_commands(mutations, _scores, model_id, chain)
            _bk = min(_scores, key=_scores.get) if _scores else "?"
            _summary = (
                f"Stability (empirical fallback): {len(_scores)}/{len(mutations)} "
                f"mutations estimated. Most stabilising: {_bk} "
                f"({_scores.get(_bk, 0.0):+.2f} kcal/mol). Mean ΔΔG: {_change:+.2f} kcal/mol."
            )
            _prog(f"! ⚗️  {_summary}")
            return ToolStepResult(
                tool="rosetta", success=True,
                data={
                    "mutations":        mutations,
                    "ddg_scores":       _scores,
                    "ddg_source":       _source,
                    "ddg_spread":       _spread,
                    "ddg_confidence":   _conf,
                    "n_trajectories":   num_trajectories,
                    "stability_change": round(_change, 3),
                    "confidence":       "low",
                    "backend":          "empirical_fallback",
                    "warnings":         _warns,
                    "method_note": (
                        "PyRosetta WSL2 failed; empirical BLOSUM62 + B-factor "
                        "estimates (Pearson r ~0.4). For screening only."
                    ),
                    "job_id":           None,
                },
                viz_commands     = _vc,
                viz_explanations = _ve,
                summary          = _summary,
            )

        _prog("⚗️  [Rosetta] Running PyRosetta cartesian_ddg (may take 2–5 min per mutation)...")
        # Scale the WSL2 process timeout to the workload: ALL num_trajectories
        # trajectories run inside one worker process, so the budget must cover
        # the whole batch. N=1, cycles=3 (production) → 1800s, unchanged.
        _wsl_timeout = max(
            1800,
            int(len(mutations) * num_trajectories * (relax_cycles * 90 + 150) + 300),
        )
        result = wsl.run_python_script(script, timeout=_wsl_timeout)

        if result["stdout"]:
            for line in result["stdout"].splitlines():
                if line.strip():
                    _prog(f"  {line.strip()}")

        if not result["ok"]:
            # Surface the error field (e.g. "timed out after Ns") AND stderr so a
            # timeout is distinguishable from a crash.
            _why = (result.get("error") or "").strip() or str(result.get("stderr", "")).strip()
            return _empirical_result(f"WSL2 script failed: {_why[:200]}")

        # Copy results file back from WSL2
        import tempfile, json as _json
        win_results = str(
            Path(tempfile.gettempdir()) / f"rosetta_ddg_{pdb_hash}.json"
        )
        ok = wsl.copy_from_wsl(wsl_results_path, win_results)
        if not ok or not Path(win_results).is_file():
            return _empirical_result("no output file produced by worker")

        try:
            with open(win_results, "r") as fh:
                ddg_raw = _json.load(fh)
        except Exception as exc:
            return _empirical_result(f"could not parse worker output ({exc})")

        if "error" in ddg_raw:
            return _empirical_result(f"worker error: {str(ddg_raw['error'])[:200]}")

        # Parse per-mutation results. The worker emits a dict per key:
        # {"ddg": median, "spread": MAD, "n": n_trajectories, "trajectories":[...]}.
        # Missing/failed mutations fall back to an empirical estimate so real
        # PyRosetta ddG stays distinguishable (ddg_source) and confidence is
        # derived from the trajectory spread.
        ddg_scores: Dict[str, float]        = {}
        ddg_source: Dict[str, str]          = {}
        ddg_spread: Dict[str, Optional[float]] = {}
        ddg_confidence: Dict[str, str]      = {}
        warnings:   List[str]               = []
        for mut in mutations:
            key   = _mutation_key(mut)
            entry = ddg_raw.get(key)
            if isinstance(entry, dict) and entry.get("ddg") is not None:
                ddg_scores[key]     = float(entry["ddg"])
                ddg_source[key]     = "pyrosetta"
                _sp                 = entry.get("spread")
                ddg_spread[key]     = float(_sp) if _sp is not None else None
                ddg_confidence[key] = _ddg_confidence_label(
                    ddg_spread[key], entry.get("n", num_trajectories)
                )
            elif entry is not None and not isinstance(entry, dict):
                # Legacy flat-float schema (older worker) — accept defensively.
                ddg_scores[key]     = float(entry)
                ddg_source[key]     = "pyrosetta"
                ddg_spread[key]     = None
                ddg_confidence[key] = _ddg_confidence_label(None, num_trajectories)
            else:
                ddg_scores[key]     = _empirical_ddg_single(mut, pdb_path)
                ddg_source[key]     = "empirical"
                ddg_spread[key]     = None
                ddg_confidence[key] = "empirical"
                warnings.append(
                    f"PyRosetta failed for {key}; using empirical BLOSUM62 estimate."
                )

        stability_change = (
            sum(ddg_scores.values()) / len(ddg_scores) if ddg_scores else 0.0
        )
        viz_cmds, viz_exps = self._build_viz_commands(
            mutations, ddg_scores, model_id, chain
        )
        best_key = min(ddg_scores, key=ddg_scores.get) if ddg_scores else "?"
        best_ddg = ddg_scores.get(best_key, 0.0)
        _multi = num_trajectories > 1
        _traj_note = (
            f" Median of {num_trajectories} trajectories x {relax_cycles} cycles."
            if _multi else ""
        )
        summary = (
            f"Stability (PyRosetta/WSL2): "
            f"{len(ddg_scores)}/{len(mutations)} mutations scored. "
            f"Most stabilising: {best_key} ({best_ddg:+.2f} kcal/mol). "
            f"Mean ΔΔG: {stability_change:+.2f} kcal/mol.{_traj_note}"
        )
        _prog(f"✓ ⚗️  {summary}")

        return ToolStepResult(
            tool    = "rosetta",
            success = True,
            data    = {
                "mutations":        mutations,
                "ddg_scores":       ddg_scores,
                "ddg_source":       ddg_source,
                "ddg_spread":       ddg_spread,
                "ddg_confidence":   ddg_confidence,
                "n_trajectories":   num_trajectories,
                "stability_change": round(stability_change, 3),
                "confidence":       "high",
                "backend":          "pyrosetta_wsl2",
                "warnings":         warnings,
                "method_note":      (
                    "PyRosetta CartesianDDG via WSL2 (Park et al. 2016). "
                    + (f"Median of {num_trajectories} independent trajectories "
                       f"({relax_cycles} relax cycles); spread reported as MAD. "
                       "Ranking-grade — absolute magnitudes approximate."
                       if _multi else
                       "Single-trajectory screening. Pearson r ~0.8 vs experimental.")
                ),
                "job_id":           None,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
        )

    # ── Relax + score an arbitrary structure (no mutation) ──────────────────────

    def relax_and_score(
        self,
        pdb_path:          str,
        relax_cycles:      int = 3,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        FastRelax + ref2015-score an ARBITRARY structure (NO mutation).

        Reuses the SAME building blocks as the ddG path (`cleanATOM` →
        `pose_from_file` → `FastRelax(ref2015)` → score), minus the
        MutateResidue/ddG step — it is NOT a new protocol and does NOT touch the
        ddG path. Used by the validate-design orchestrator as a fold-PLAUSIBILITY
        sanity signal (total REU + per-residue density + a coarse clash flag),
        NOT a stability-vs-WT claim.

        Returns (never raises)::
            {
              "success":              bool,
              "total_reu":            float,      # full-pose ref2015 total
              "n_residues":           int,
              "per_residue_density":  float,      # total_reu / n_residues
              "per_residue":          [float,...],# per-residue total energies
              "fa_rep":               float,      # repulsive term (clash proxy)
              "clash_ok":             bool,       # fa_rep/residue below a loose bar
              "converged":            bool,       # relax completed
              "relaxed_pdb":          str | "",   # Windows path to relaxed model
              "backend":              "pyrosetta_wsl2",
              "error":                None | str,
            }
        """
        def _prog(msg: str) -> None:
            (progress_callback or _safe_print)(msg)

        relax_cycles = max(1, int(relax_cycles))

        def _err(msg: str) -> Dict[str, Any]:
            return {
                "success": False, "error": msg, "backend": "pyrosetta_wsl2",
                "total_reu": None, "n_residues": 0, "per_residue_density": None,
                "per_residue": [], "fa_rep": None, "clash_ok": None,
                "converged": False, "relaxed_pdb": "",
            }

        if not pdb_path or not Path(pdb_path).is_file():
            return _err(f"structure file not found: {pdb_path}")

        try:
            from wsl_bridge import WSLBridge
        except ImportError:
            return _err("wsl_bridge module not found")

        wsl = WSLBridge()
        if not wsl.is_available():
            return _err("WSL2 not available (PyRosetta relax/score runs in WSL2).")
        if not wsl.check_pyrosetta():
            return _err("PyRosetta not installed in WSL2.")

        import hashlib, json, tempfile
        wsl_pdb = wsl.copy_to_wsl(pdb_path)
        if not wsl_pdb:
            return _err(f"failed to copy {pdb_path} to WSL2 /tmp")

        h           = hashlib.md5(open(pdb_path, "rb").read()).hexdigest()[:12]
        wsl_result  = f"/tmp/relax_score_{h}.json"
        wsl_relaxed = f"/tmp/relax_score_{h}_relaxed.pdb"

        # Standalone worker: zero project imports, JSON-file I/O, writes a result
        # file even on exception, literal braces doubled.
        script = f"""
import json, os
result_path = {wsl_result!r}
def _write(d):
    with open(result_path, "w") as fh:
        json.dump(d, fh)
try:
    import pyrosetta
    from pyrosetta import init as rosetta_init, pose_from_file
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.scoring import ScoreType
    from pyrosetta.toolbox import cleanATOM

    rosetta_init(options="-mute all -ex1 -ex2 -use_input_sc -ignore_unrecognized_res true")

    pdb_path = {wsl_pdb!r}
    cleaned  = pdb_path.replace(".pdb", "_clean.pdb")
    if cleaned == pdb_path:
        cleaned = pdb_path + "_clean.pdb"
    cleanATOM(pdb_path, cleaned)
    if os.path.isfile(cleaned) and os.path.getsize(cleaned) > 100:
        pdb_path = cleaned

    pose = pose_from_file(pdb_path)
    sfx  = pyrosetta.create_score_function("ref2015")
    FastRelax(sfx, {relax_cycles}).apply(pose)

    total = float(sfx(pose))
    nres  = int(pose.total_residue())
    per_res = [round(float(pose.energies().residue_total_energy(i)), 3)
               for i in range(1, nres + 1)]
    fa_rep = float(pose.energies().total_energies()[ScoreType.fa_rep])
    pose.dump_pdb({wsl_relaxed!r})

    _write({{"success": True, "total_reu": round(total, 3), "n_residues": nres,
             "per_residue": per_res, "fa_rep": round(fa_rep, 3)}})
    print("[Rosetta] relax+score done: total %.2f REU, %d residues" % (total, nres), flush=True)
except Exception as exc:
    import traceback
    traceback.print_exc()
    try:
        _write({{"success": False, "error": str(exc)}})
    except Exception:
        pass
"""

        _prog(f"⚗️  [Rosetta] Relaxing + scoring structure ({relax_cycles} cycles)...")
        timeout = max(600, relax_cycles * 90 + 300)
        run = wsl.run_python_script(script, timeout=timeout)
        if run.get("stdout"):
            for line in run["stdout"].splitlines():
                if line.strip():
                    _prog(f"  {line.strip()}")
        if not run["ok"]:
            why = (run.get("error") or "").strip() or str(run.get("stderr", ""))[:200]
            return _err(f"WSL2 relax/score failed: {why}")

        win_result = str(Path(tempfile.gettempdir()) / f"relax_score_{h}.json")
        if not wsl.copy_from_wsl(wsl_result, win_result) or not Path(win_result).is_file():
            return _err("worker produced no result file")
        try:
            with open(win_result, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            return _err(f"could not parse worker output ({exc})")
        if not data.get("success"):
            return _err(f"worker error: {str(data.get('error'))[:200]}")

        total = float(data["total_reu"])
        nres  = int(data["n_residues"]) or 1
        fa_rep = float(data.get("fa_rep", 0.0))
        # Copy the relaxed model back (best-effort; not fatal if it fails).
        win_relaxed = str(Path(tempfile.gettempdir()) / f"relax_score_{h}_relaxed.pdb")
        relaxed_pdb = win_relaxed if wsl.copy_from_wsl(wsl_relaxed, win_relaxed) \
            and Path(win_relaxed).is_file() else ""

        # Coarse clash flag: post-relax repulsive energy per residue. This is a
        # loose sanity bar, NOT a calibrated threshold — surfaced as a signal.
        clash_ok = (fa_rep / nres) < 5.0

        return {
            "success":             True,
            "total_reu":           round(total, 3),
            "n_residues":          nres,
            "per_residue_density": round(total / nres, 4),
            "per_residue":         data.get("per_residue", []),
            "fa_rep":              round(fa_rep, 3),
            "clash_ok":            bool(clash_ok),
            "converged":           True,
            "relaxed_pdb":         relaxed_pdb,
            "backend":             "pyrosetta_wsl2",
            "error":               None,
        }

    # ── High-accuracy validation tier ───────────────────────────────────────────

    def validate_ddg(
        self,
        pdb_path:  str,
        mutations: List[Dict[str, Any]],
        session:   Any = None,
        model_id:  str = "1",
        chain:     Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> "ToolStepResult":
        """
        High-accuracy validation tier: multi-trajectory MEDIAN ddG on a SMALL
        explicit set of mutations, using ROSETTA_VALIDATION_TRAJECTORIES and
        ROSETTA_VALIDATION_CYCLES. Always uses the local PyRosetta/WSL2 path
        (returns a clear error if WSL2/PyRosetta is unavailable).

        NOT for full interactive scans — see _run_rosetta_local for the fast
        single-trajectory production path. Absolute magnitudes carry roughly
        2-3 kcal/mol uncertainty across all structural categories; ranking/sign
        is reliable.
        """
        import config as _cfg
        n   = max(1, int(getattr(_cfg, "ROSETTA_VALIDATION_TRAJECTORIES", 5)))
        cyc = max(1, int(getattr(_cfg, "ROSETTA_VALIDATION_CYCLES", 8)))

        def _prog(msg: str) -> None:
            (progress_callback or _safe_print)(msg)

        _per_min = max(1, round(n * cyc * 0.6))
        _prog(
            f"⚗️  High-accuracy validation: {n} trajectories x {cyc} cycles per "
            f"mutation, ~{_per_min} min per mutation. Magnitudes carry roughly "
            "2-3 kcal/mol uncertainty even at this tier — use for confidence "
            "ranking, validate critical predictions experimentally."
        )

        result = self._run_rosetta_local(
            pdb_path, mutations, model_id, chain, progress_callback,
            relax_cycles=cyc, num_trajectories=n,
        )

        # Honest disclosure + tier metadata (non-negotiable: never claim calibration).
        if result.success and isinstance(result.data, dict):
            result.data["tier"] = "validation"
            result.data.setdefault("warnings", []).append(
                "High-accuracy validation tier (median of "
                f"{n} trajectories x {cyc} relax cycles). Absolute ΔΔG remains "
                "APPROXIMATE — magnitudes carry roughly 2-3 kcal/mol uncertainty "
                "across all structural categories even at this tier (more cycles "
                "only partially close the gap). Ranking/sign is reliable; high spread = low "
                "confidence = low trust. NOT calibrated — confirm critical "
                "predictions experimentally."
            )
        return result

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
