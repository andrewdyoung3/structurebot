"""
config.py
---------
Central configuration for StructureBot.
All path and tuning constants live here.
Override any value by setting the matching environment variable or by editing
this file — never hard-code paths in other modules.
"""

import os
from pathlib import Path

# ── ChimeraX ──────────────────────────────────────────────────────────────────

CHIMERAX_PATH: str = os.environ.get(
    "CHIMERAX_PATH",
    r"C:\Users\andre\documents\ChimeraX 1.11.1\bin\ChimeraX.exe",
)

REST_HOST: str = os.environ.get("CHIMERAX_HOST", "127.0.0.1")
REST_PORT: int = int(os.environ.get("CHIMERAX_PORT", "60001"))
REST_TIMEOUT: int = int(os.environ.get("CHIMERAX_TIMEOUT", "10"))

# ── Anthropic ─────────────────────────────────────────────────────────────────

# claude-sonnet-4-6 is the current recommended model.
# Upgrade to claude-opus-4-7 for the hardest multi-step reasoning tasks.
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# Maximum number of user/assistant exchange *pairs* kept in rolling history.
# Each turn consumes input tokens; prompt caching absorbs the static block cost.
MAX_CONVERSATION_HISTORY: int = int(os.environ.get("MAX_CONVERSATION_HISTORY", "6"))

# ── UI behaviour ──────────────────────────────────────────────────────────────

# Seconds before auto-executing high/medium-confidence commands.
# Set to 0 to always require explicit confirmation.
AUTO_PROCEED_DELAY: int = int(os.environ.get("AUTO_PROCEED_DELAY", "2"))

# ── Directories ───────────────────────────────────────────────────────────────

_BASE = Path(__file__).parent

LOG_DIR: Path     = Path(os.environ.get("STRUCTUREBOT_LOG_DIR",     str(_BASE / "logs")))
SESSION_DIR: Path = Path(os.environ.get("STRUCTUREBOT_SESSION_DIR", str(_BASE / "sessions")))

# Create at import time so nothing else has to mkdir guard
LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# ── Desktop path helper ───────────────────────────────────────────────────────

def desktop_path() -> str:
    """Return the current user's Desktop path, always with forward slashes."""
    username = os.environ.get("USERNAME", os.environ.get("USER", "user"))
    p = Path(f"C:/Users/{username}/Desktop")
    return p.as_posix()  # forward slashes — ChimeraX save prefers them

# ── Stability / ddG ───────────────────────────────────────────────────────────

# Which stability backend to use:
#   "auto"       — PyRosetta if available, else DynaMut2 (default)
#   "dynamut2"   — DynaMut2 web API (free, no registration)
#   "empirical"  — offline BLOSUM62 estimates (no network required)
#   "pyrosetta"  — local PyRosetta (requires Python <= 3.13 + wheel)
#   "local"      — local Rosetta binary (requires ROSETTA_LOCAL_PATH)
ROSETTA_BACKEND: str = os.environ.get("ROSETTA_BACKEND", "auto").strip()

# Path to local Rosetta binary directory (ROSETTA_BACKEND=local only).
# Example: /path/to/rosetta/source/bin
# Linux/Mac only; see rosetta_bridge._run_rosetta_local() for setup.
ROSETTA_LOCAL_PATH: str = os.environ.get("ROSETTA_LOCAL_PATH", "").strip()

# Set PYROSETTA_AVAILABLE=true in .env.local when PyRosetta is installed.
# Python 3.14 wheels are not yet released; this is False by default.
# See rosetta_bridge._run_pyrosetta() docstring for installation instructions.
PYROSETTA_AVAILABLE: bool = (
    os.environ.get("PYROSETTA_AVAILABLE", "").strip().lower()
    in ("1", "true", "yes")
)

# ── WSL2 ──────────────────────────────────────────────────────────────────────

# Name of the WSL2 distribution to use for PyRosetta.
# Override with WSL_DISTRIBUTION env var.
WSL_DISTRIBUTION: str = os.environ.get("WSL_DISTRIBUTION", "Ubuntu-22.04").strip()

# ── Rosetta relax cache ────────────────────────────────────────────────────────

# Directory where FastRelax'd PDB structures are cached to avoid re-relaxing.
# Keyed by MD5 hash of the PDB content.  Populated by the local Rosetta backend.
ROSETTA_RELAX_CACHE: Path = Path(
    os.environ.get("ROSETTA_RELAX_CACHE", str(_BASE / "cache" / "rosetta_relaxed"))
)
ROSETTA_RELAX_CACHE.mkdir(parents=True, exist_ok=True)

# ── venv312 (Python 3.12 + CUDA torch 2.11.0+cu128) ─────────────────────────

# Absolute path to the Python 3.12 virtual-environment interpreter.
# venv312 ships torch 2.11.0+cu128, which has working CUDA on RTX 5070 Ti
# (sm_120 Blackwell).  The main venv (Python 3.14) has no working CUDA build
# for sm_120, so GPU inference is delegated to venv312 as a subprocess.
# Override with VENV312_PYTHON env var.
VENV312_PYTHON: str = os.environ.get(
    "VENV312_PYTHON",
    str(Path(__file__).parent / "venv312" / "Scripts" / "python.exe"),
)

# Path to ProteinMPNN repo directory.  Must contain protein_mpnn_run.py and
# vanilla_model_weights/.  Override with PROTEINMPNN_DIR env var.
# Default: <structurebot>/ProteinMPNN (if cloned alongside the project).
PROTEINMPNN_DIR: str = os.environ.get(
    "PROTEINMPNN_DIR",
    str(Path(__file__).parent / "ProteinMPNN"),
)

# Controls whether ESM-2 uses the venv312 GPU backend.
#   "auto"      — use venv312 if it exists and passes a CUDA smoke-test (default)
#   "true"/"1"  — always use venv312; raise if unavailable
#   "false"/"0" — always use current-venv CPU path (disables GPU delegation)
ESM_USE_VENV312: str = os.environ.get("ESM_USE_VENV312", "auto").strip()

# ── Parallel DynaMut2 ─────────────────────────────────────────────────────────

# Maximum concurrent DynaMut2 requests.
# 4 is conservative — server handles up to ~5.  Set to 1 to disable parallelism.
DYNAMUT2_MAX_WORKERS: int = int(os.environ.get("DYNAMUT2_MAX_WORKERS", "4"))

# ── Double mutant scoring ─────────────────────────────────────────────────────

# Cα-Cα distance threshold (Å) above which DynaMut2 is reliable for double mutants.
DOUBLE_MUTANT_DISTANCE_THRESHOLD_FAR: float = float(
    os.environ.get("DOUBLE_MUTANT_DISTANCE_THRESHOLD_FAR", "10.0")
)

# Cα-Cα distance threshold (Å) below which PyRosetta is required (DynaMut2 not used).
DOUBLE_MUTANT_DISTANCE_THRESHOLD_CLOSE: float = float(
    os.environ.get("DOUBLE_MUTANT_DISTANCE_THRESHOLD_CLOSE", "4.0")
)

# Maximum pairs to consider before applying distance-based routing.
# If the combination count exceeds this, take top-N by sum of |ddG|.
DOUBLE_MUTANT_MAX_PAIRS: int = int(os.environ.get("DOUBLE_MUTANT_MAX_PAIRS", "500"))

# Default number of top-ranked pairs to return from analyze().
DOUBLE_MUTANT_TOP_N: int = int(os.environ.get("DOUBLE_MUTANT_TOP_N", "10"))

# ── ESMFold ───────────────────────────────────────────────────────────────────

# Enable ESMFold foldability checking on top mutation/disulfide candidates.
ESMFOLD_ENABLED: bool = (
    os.environ.get("ESMFOLD_ENABLED", "true").strip().lower()
    in ("1", "true", "yes")
)

# How many top candidates to check with ESMFold after a mutation scan.
ESMFOLD_TOP_N: int = int(os.environ.get("ESMFOLD_TOP_N", "3"))

# Mean pLDDT drop (at mutation positions) above which we issue a warning.
ESMFOLD_PLDDT_WARNING_THRESHOLD: float = float(
    os.environ.get("ESMFOLD_PLDDT_WARNING_THRESHOLD", "10.0")
)

# Prefer local GPU inference (venv312 subprocess) over the Atlas API.
# Set ESMFOLD_USE_LOCAL=false to always use the Atlas API.
ESMFOLD_USE_LOCAL: bool = (
    os.environ.get("ESMFOLD_USE_LOCAL", "true").strip().lower()
    not in ("0", "false", "no")
)

# HuggingFace model ID for local ESMFold inference.
ESMFOLD_MODEL_NAME: str = os.environ.get(
    "ESMFOLD_MODEL_NAME", "facebook/esmfold_v1"
)

# Worker subprocess timeouts.
# Cold start (first run, weights not yet downloaded) needs up to 10 minutes
# to pull ~2.5 GB from HuggingFace.  Warm start (weights already cached)
# should complete within 2 minutes.  Override via env vars.
ESMFOLD_WORKER_TIMEOUT_COLD: int = int(
    os.environ.get("ESMFOLD_WORKER_TIMEOUT_COLD", "600")   # 10 min
)
ESMFOLD_WORKER_TIMEOUT_WARM: int = int(
    os.environ.get("ESMFOLD_WORKER_TIMEOUT_WARM", "120")   # 2 min
)

# Force the cold (600 s) timeout regardless of cache state.
# Set True as a one-time override when weights are not yet downloaded but
# _is_model_cached is returning True (e.g. partial previous download).
# Default is False — _is_model_cached() correctly detects real weight files
# (under snapshots/ and blobs/) and ignores HuggingFace .no_exist/ sentinels.
ESMFOLD_FORCE_COLD_TIMEOUT: bool = (
    os.environ.get("ESMFOLD_FORCE_COLD_TIMEOUT", "false").strip().lower()
    in ("1", "true", "yes")
)

# ── ProteinMPNN + ESMFold validation pipeline ─────────────────────────────────

# How many top ProteinMPNN designs to validate with ESMFold.
MPNN_ESMFOLD_TOP_N: int = int(os.environ.get("MPNN_ESMFOLD_TOP_N", "3"))

# Whether to include the wildtype sequence as a baseline in ESMFold validation.
MPNN_ESMFOLD_INCLUDE_WT: bool = (
    os.environ.get("MPNN_ESMFOLD_INCLUDE_WT", "true").strip().lower()
    not in ("0", "false", "no")
)

# ── NetNGlyc 1.0 (OST recognition prediction) ─────────────────────────────────

# REST API endpoint for DTU Health Tech NetNGlyc 1.0.
# Note: "SEQENCE" typo in POST data is intentional — server-side parameter name.
NETNGLYC_API_URL: str = os.environ.get(
    "NETNGLYC_API_URL",
    "https://services.healthtech.dtu.dk/service.php?NetNGlyc-1.0",
)

# HTTP timeout in seconds for NetNGlyc API calls.
NETNGLYC_TIMEOUT: int = int(os.environ.get("NETNGLYC_TIMEOUT", "30"))

# Set to False to skip all NetNGlyc API calls (e.g. in offline/CI environments).
NETNGLYC_ENABLED: bool = (
    os.environ.get("NETNGLYC_ENABLED", "true").strip().lower()
    not in ("0", "false", "no")
)

# How many top glycan-position candidates to annotate via NetNGlyc after a
# projection-aware scan (_run_glycan_positions in tool_router.py).
NETNGLYC_TOP_N: int = int(os.environ.get("NETNGLYC_TOP_N", "5"))

# ── .env.local loader ─────────────────────────────────────────────────────────
# Called by main.py at startup BEFORE any other imports that read env vars.

def load_env_file(path: str | Path | None = None) -> None:
    """Load key=value pairs from .env.local (or a given path) into os.environ."""
    env_file = Path(path) if path else Path(__file__).parent / ".env.local"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())
