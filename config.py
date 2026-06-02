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

# When a structure is opened, also open the Sequence Viewer for its chain(s) so
# loaded PDBs show their sequence by default (applies to ALL opens, not just
# ColabFold). Set CHIMERAX_SHOW_SEQUENCE_ON_OPEN=false to disable.
CHIMERAX_SHOW_SEQUENCE_ON_OPEN: bool = (
    os.environ.get("CHIMERAX_SHOW_SEQUENCE_ON_OPEN", "true").strip().lower()
    not in ("0", "false", "no", "off")
)

# ── Deterministic ChimeraX layout + presentation ──────────────────────────────
# Config-driven command lists applied by StructureBot (NOT LLM-generated, NOT the
# built-in `preset`). All tokens verified against ChimeraX 1.11.1.

# LEAN LAYOUT — applied ONCE per ChimeraX session (first open). Hides the Log,
# Command Line Interface and Toolbar panels for a clean window; KEEPS the menubar
# and title bar. The Sequence Viewer (opened via sequence_viewer.ensure_sequence_
# viewer_commands) docks at the top by default. REST command/runscript coloring +
# selection keep working with the CLI hidden (verified). Disable with
# CHIMERAX_LEAN_LAYOUT=false.
CHIMERAX_LEAN_LAYOUT: bool = (
    os.environ.get("CHIMERAX_LEAN_LAYOUT", "true").strip().lower()
    not in ("0", "false", "no", "off")
)
CHIMERAX_LEAN_LAYOUT_COMMANDS: list = [
    "tool hide Log",
    'tool hide "Command Line Interface"',
    "tool hide Toolbar",
]

# DEFAULT PRESENTATION — applied per structure open, AFTER load and BEFORE any
# analysis colouring (CamSol/ESM/MPNN override the by-chain baseline; SCF sequence
# regions still land). Disable with CHIMERAX_DEFAULT_PRESENTATION=false.
CHIMERAX_DEFAULT_PRESENTATION: bool = (
    os.environ.get("CHIMERAX_DEFAULT_PRESENTATION", "true").strip().lower()
    not in ("0", "false", "no", "off")
)
CHIMERAX_DEFAULT_PRESENTATION_COMMANDS: list = [
    "hide solvent atoms",
    "cartoon",
    "show ligand atoms",
    "style ligand stick",
    "color bychain",
    "color ligand byhetero",
    "set bgColor black",
    "lighting soft",
    "graphics silhouettes true",
    "view",
]

# ── Anthropic ─────────────────────────────────────────────────────────────────

# claude-sonnet-4-6 is the current recommended model.
# Upgrade to claude-opus-4-7 for the hardest multi-step reasoning tasks.
ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

# NL→ChimeraX translation backend (pluggable). "claude" (Anthropic API, default)
# or "ollama" (local model). Unknown values fall back to "claude".
TRANSLATOR_BACKEND: str = os.environ.get("TRANSLATOR_BACKEND", "claude").strip().lower()

# Canonical valid tool names — the EXACT (lowercase) literals the router
# dispatches on (`tool_router._dispatch_tool`). Used to ENUM-constrain the Ollama
# backend's `tools_needed` output (constrained decoding cannot then emit a
# misspelled / hallucinated / wrong-cased tool). A test asserts this matches the
# router registry. Keep sorted + in sync with `_dispatch_tool`.
TRANSLATOR_TOOL_NAMES: list = [
    "assembly_analyser", "camsol", "cavity", "chimerax", "colabfold",
    "disulfide", "double_mutant", "esm", "esmfold", "glycan",
    "glycan_positions", "mpnn_esmfold", "mutation_scan", "netnglyc",
    "proline", "proteinmpnn", "rfdiffusion", "rosetta", "salt_bridge",
    "validate_ddg", "validate_design",
]

# One-directional fallback: when TRANSLATOR_BACKEND="claude" and the Claude API
# fails with a REAL API-failure (connection-unreachable / timeout / auth /
# rate-limit), fall back to the local Ollama backend. A successful-but-imperfect
# Claude response is used as-is (never a fallback trigger). A forced
# TRANSLATOR_BACKEND="ollama" NEVER falls back to Claude (benchmark honesty).
TRANSLATOR_FALLBACK: bool = (
    os.environ.get("TRANSLATOR_FALLBACK", "true").strip().lower()
    not in ("0", "false", "no", "off")
)

# ── Ollama local LLM backend (benchmark + fallback; see translator.OllamaBackend)
OLLAMA_BASE_URL: str = os.environ.get(
    "OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
# Exact model+tag — pulled and verified to serve schema-constrained structured
# output. Qwen3 8B: strong tool-calling, native Ollama structured-output, fits
# the 16 GB RTX 5070 Ti comfortably.
OLLAMA_MODEL: str = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
# Context window. StructureBot's system prompt (role + rules + the full ChimeraX
# command reference) is ~7.5k tokens, plus the targeted few-shot (~0.5k) — so it
# must be ≥ ~10k or Ollama SILENTLY TRUNCATES the prompt (the model never sees the
# rules/few-shot → routing collapses). 16384 fits the prompt + few-shot + room to
# generate, and the 8B's KV cache at 16k still fits the 16 GB RTX 5070 Ti.
# ⚠ Over-sizing num_ctx BEYOND VRAM makes Ollama spill to CPU — slow AND degrades
# constrained-decoding reliability; tune to your GPU.
OLLAMA_NUM_CTX: int = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
# Output token cap. Generous so the full 7-key JSON (commands + tool_inputs)
# never truncates mid-generation; the `format` grammar stops it well before this.
OLLAMA_NUM_PREDICT: int = int(os.environ.get("OLLAMA_NUM_PREDICT", "1024"))
# Idle keep-alive before Ollama unloads the model from VRAM. SHORT by default so
# VRAM frees quickly for GPU bridges; the explicit
# translator.ensure_translator_unloaded() is the real contract — do NOT rely on
# this idle timer alone (a mid-run OOM is the failure mode to prevent).
OLLAMA_KEEP_ALIVE: str = os.environ.get("OLLAMA_KEEP_ALIVE", "30s")
# Per-request HTTP timeout (a cold model load on first call can be slow).
OLLAMA_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", "180"))

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

# ── Multi-trajectory ddG (PyRosetta local path) ─────────────────────────────────
# Number of independent relax+score trajectories per mutation; the reported ddG
# is the MEDIAN across trajectories (median chosen over min/mean — see
# scripts/rosetta_validation_notes.md: two-sided noise makes min invent fake
# stabilisers). Default 1 = fast single-trajectory production scan behaviour
# (UNCHANGED). The high-accuracy validation tier uses the *_VALIDATION_* values.
ROSETTA_NUM_TRAJECTORIES: int = int(os.environ.get("ROSETTA_NUM_TRAJECTORIES", "1"))

# High-accuracy validation tier: more trajectories + more relax cycles, run only
# on a small explicit set of candidate mutations (NOT on full interactive scans —
# 8+8-cycle trajectories are ~3-5 min each).
ROSETTA_VALIDATION_TRAJECTORIES: int = int(
    os.environ.get("ROSETTA_VALIDATION_TRAJECTORIES", "5")
)
ROSETTA_VALIDATION_CYCLES: int = int(os.environ.get("ROSETTA_VALIDATION_CYCLES", "8"))

# Median-absolute-deviation spread (kcal/mol) above which a multi-trajectory
# ddG prediction is flagged low-confidence.
ROSETTA_SPREAD_LOW_CONFIDENCE: float = float(
    os.environ.get("ROSETTA_SPREAD_LOW_CONFIDENCE", "3.0")
)

# Strip crystallographic waters (HOH) before PyRosetta scoring.
#   True (default) = STRIP waters: cleanATOM removes ALL HETATM (HOH included).
#       This is the validated, standard-Rosetta-practice baseline — the ref2015
#       lk_ball implicit solvent already models surface desolvation, and all
#       rigorous panel validation (commit cbe327a) was measured on this path.
#   False           = PRESERVE waters (opt-in): re-append crystallographic HOH
#       records so buried structural waters reach PyRosetta. Fixes wrong-sign
#       ddG on buried mutations near them (validated on T26A: +3.80 shift, sign
#       corrected). WARNING: preserve-ALL-static is a Rosetta anti-pattern
#       (static waters act as immovable spheres → clash-driven over-
#       destabilization on non-buried mutations); its panel-level behaviour is
#       unmeasured. Proper fix = selective buried-only / movable waters (backlog).
# NOTE: the relaxed-structure cache key is namespaced by this mode, so
# preserved-water runs never reuse a previously cached stripped-water structure.
ROSETTA_STRIP_WATERS: bool = (
    os.environ.get("ROSETTA_STRIP_WATERS", "true").strip().lower()
    not in ("0", "false", "no", "off")
)

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

# Persistent cache for ProteinMPNN designs. The bridge runs in a deleted
# tempfile.TemporaryDirectory(), and the interactive session is only saved on a
# clean quit (no per-turn autosave), so designs evaporated. Every run now also
# writes its full FASTA (all designs + WT, scores in headers) here so a design is
# never lost and can be retrieved/aligned without re-running (re-running is
# stochastic and overwrites the design).
PROTEINMPNN_CACHE_DIR: Path = Path(
    os.environ.get("PROTEINMPNN_CACHE_DIR", str(_BASE / "cache" / "proteinmpnn"))
)
PROTEINMPNN_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Sequence-Viewer integration scratch (.scf coloring files + their runscript
# loaders) — see sequence_viewer.py. Forward-slash paths only (ChimeraX).
SEQVIEW_CACHE_DIR: Path = Path(
    os.environ.get("SEQVIEW_CACHE_DIR", str(_BASE / "cache" / "seqview"))
)
SEQVIEW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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

# ΔΔG (kcal/mol) above which a single mutation is considered clearly
# destabilising. In stability mode a pair is dropped only when BOTH mutations
# exceed this threshold; ddg == 0.0 (DynaMut2 neutral/unknown) is never filtered.
DOUBLE_MUTANT_DESTABILISING_DDG: float = float(
    os.environ.get("DOUBLE_MUTANT_DESTABILISING_DDG", "2.0")
)

# When True (default), pairs that fail the DynaMut2 mm API fall back to
# additive ddG scoring (ddG_A + ddG_B) with epistasis set to 0.
# Set False to skip such pairs entirely rather than using estimates.
DOUBLE_MUTANT_ADDITIVE_FALLBACK: bool = (
    os.environ.get("DOUBLE_MUTANT_ADDITIVE_FALLBACK", "true").strip().lower()
    not in ("0", "false", "no")
)

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

# ── ColabFold (WSL2 ~/colabfold_env, AF2-quality folding) ───────────────────────
# v1 standalone bridge. Runs colabfold_batch inside the isolated WSL2 ColabFold
# env (wsl_bridge.COLABFOLD_PYTHON) using the REMOTE MSA server (no local DBs).

# Number of AF2 models to run (1-5). More models = better ranking, slower.
COLABFOLD_NUM_MODELS: int = int(os.environ.get("COLABFOLD_NUM_MODELS", "5"))

# Number of recycles per model. More recycles = better convergence, slower.
COLABFOLD_NUM_RECYCLE: int = int(os.environ.get("COLABFOLD_NUM_RECYCLE", "3"))

# MSA mode passed to colabfold_batch. Default uses the remote MMseqs2 server
# (no hundreds-of-GB local databases). "single_sequence" (DEFERRED) skips MSA.
COLABFOLD_MSA_MODE: str = os.environ.get("COLABFOLD_MSA_MODE", "mmseqs2_uniref_env").strip()

# Total-residue budget (len(sequence) x copies). AlphaFold attention memory
# scales ~ (total residues)^2, so oligomers OOM the laptop GPU quickly. Above
# this budget the bridge refuses to launch and returns an OOM-risk message
# rather than crashing. Conservative default; recalibrate empirically (see
# PROJECT_CONTEXT §9 ColabFold multimer caveat + the remote-GPU-handoff backlog).
COLABFOLD_MAX_TOTAL_RESIDUES: int = int(
    os.environ.get("COLABFOLD_MAX_TOTAL_RESIDUES", "1500")
)

# Base WSL2 process timeout (seconds). The bridge scales this up with
# total_residues x num_models x num_recycle; this is the floor (covers the
# one-time weight download + first sm_120 XLA compile).
COLABFOLD_TIMEOUT: int = int(os.environ.get("COLABFOLD_TIMEOUT", "1800"))

# Cache dir for completed folds, keyed by hash(seq+copies+template+models+recycle).
# A re-fold of an identical input returns the cached ranked PDB instantly.
COLABFOLD_CACHE_DIR: Path = Path(
    os.environ.get("COLABFOLD_CACHE_DIR", str(_BASE / "cache" / "colabfold"))
)
COLABFOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# JAX persistent compilation cache dir — a WSL2 ext4 path (NOT under /mnt/c; the
# Windows-boundary I/O penalty would defeat the point). The fold result cache
# (above) only saves IDENTICAL re-folds; this lets XLA reuse compiled executables
# across the fresh-per-fold worker processes for any matching input shape, so a
# second fold of a different sequence at the same length skips the ~10-min XLA
# compile. '~' is expanded inside the worker (it runs in WSL2). String, not Path.
COLABFOLD_JAX_COMPILE_CACHE_DIR: str = os.environ.get(
    "COLABFOLD_JAX_COMPILE_CACHE_DIR", "~/.cache/colabfold_jax_compile"
)

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
