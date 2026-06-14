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

# PER-CHAIN sequences: open one Sequence Viewer PER CHAIN (`sequence chain #N/A`,
# `#N/B`, …) instead of a single grouped viewer. ChimeraX otherwise collapses
# identical chains (e.g. a homodimer) into one "chains A,B" row, so a column
# selection hits BOTH chains; per-chain viewers let the user select residues in a
# SPECIFIC chain. Set CHIMERAX_SEQUENCE_PER_CHAIN=false to restore the grouped
# viewer. CHIMERAX_SEQUENCE_PER_CHAIN_MAX caps it: structures with MORE chains
# than this (e.g. a viral capsid) fall back to the single grouped viewer so we
# never open dozens of panels.
CHIMERAX_SEQUENCE_PER_CHAIN: bool = (
    os.environ.get("CHIMERAX_SEQUENCE_PER_CHAIN", "true").strip().lower()
    not in ("0", "false", "no", "off")
)
CHIMERAX_SEQUENCE_PER_CHAIN_MAX: int = int(
    os.environ.get("CHIMERAX_SEQUENCE_PER_CHAIN_MAX", "8")
)

# DOCK the Sequence Viewer(s) along the BOTTOM edge, stacked vertically (so each
# chain's sequence is visible at once). ChimeraX docks the viewer at the TOP by
# default and exposes no REST command to move it, so StructureBot drives the Qt
# dock widget directly (verified on 1.11.1). Set CHIMERAX_SEQUENCE_DOCK_BOTTOM=
# false to leave the viewer where ChimeraX puts it.
CHIMERAX_SEQUENCE_DOCK_BOTTOM: bool = (
    os.environ.get("CHIMERAX_SEQUENCE_DOCK_BOTTOM", "true").strip().lower()
    not in ("0", "false", "no", "off")
)

# NUMBERING: add a residue-number RULER to each per-chain Sequence Viewer, labelled
# every N residues. The labels are the ACTUAL PDB residue numbers (auth seq IDs),
# placed via proteinmpnn_bridge.chain_resnum_to_seqpos — so a chain that doesn't
# start at 1 (1IL8 chain A is 2..72) shows 2,12,22… NOT 1,11,21, consistent with the
# MPNN alignment numbering. ChimeraX 1.11.1's NATIVE numbering is position-based
# (`numbering_start + count`, a linear offset that can't honor gaps), so this ships
# a custom FIXED HEADER (`alignment.add_fixed_header`) instead. Set
# CHIMERAX_SEQUENCE_NUMBERING=false to disable.
CHIMERAX_SEQUENCE_NUMBERING: bool = (
    os.environ.get("CHIMERAX_SEQUENCE_NUMBERING", "true").strip().lower()
    not in ("0", "false", "no", "off")
)
CHIMERAX_SEQUENCE_NUMBER_INTERVAL: int = int(
    os.environ.get("CHIMERAX_SEQUENCE_NUMBER_INTERVAL", "10")
)

# CONSOLIDATION: when a structure has MORE than this many chains, collapse all
# chains with the same sequence into ONE alignment window (one row per structure ×
# unique sequence group) instead of opening N separate per-chain panels. Keeps
# per-chain addressability — `#N/A` targeting is unaffected. Set to 0 to always
# consolidate; set a large number to never consolidate (uses per-chain up to
# CHIMERAX_SEQUENCE_PER_CHAIN_MAX). Default 3: 1-3 chains → per-chain (unchanged);
# 4-8 chains → consolidated. Verified on ChimeraX 1.11.1: new_alignment auto-
# associates ALL chains with identical sequences → selecting a row selects all
# copies in 3D. Set CHIMERAX_SEQUENCE_CONSOLIDATE_THRESHOLD=0 to always consolidate.
CHIMERAX_SEQUENCE_CONSOLIDATE_THRESHOLD: int = int(
    os.environ.get("CHIMERAX_SEQUENCE_CONSOLIDATE_THRESHOLD", "3")
)

# ── Deterministic ChimeraX layout + presentation ──────────────────────────────
# Config-driven command lists applied by StructureBot (NOT LLM-generated, NOT the
# built-in `preset`). All tokens verified against ChimeraX 1.11.1.

# LEAN LAYOUT — applied ONCE per ChimeraX session (first open). Hides the Log,
# Command Line Interface and Toolbar panels for a clean window; KEEPS the menubar
# and title bar. The Sequence Viewer (opened via sequence_viewer.ensure_sequence_
# viewer_commands) is re-docked to the bottom by StructureBot. REST command/
# runscript coloring + selection keep working with the CLI hidden (verified). Disable with
# CHIMERAX_LEAN_LAYOUT=false. (The Sequence Viewer is re-docked to the BOTTOM,
# stacked per chain — see CHIMERAX_SEQUENCE_DOCK_BOTTOM above.)
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

# Opacity of the mobile conformer (model B) in a conformer comparison overlay.
# 0 = fully opaque; 100 = fully transparent.  Verified: ChimeraX 1.11.1 requires
# explicit "target c" — bare "transparency #N 50" silently has no effect on cartoons.
CONFORMER_B_TRANSPARENCY: int = int(
    os.environ.get("CONFORMER_B_TRANSPARENCY", "50")
)

# ── Design-intent profiles (op-class: goal → tool + params + ranking) ────────────
# "redesign for solubility" → ProteinMPNN on the SOLVENT-EXPOSED positions only
# (buried core fixed), soluble bias, Cys omitted; ranked by CamSol (+ ESMFold).
#
# Exposure gate: per-residue absolute SASA (Å², BioPython ShrakeRupley via
# cavity_bridge). NOTE this is an EXPOSURE cut, deliberately HIGHER than cavity's
# ~20 Å² BURIAL cutoff — at 20 it would grab partially-buried rim positions whose
# mutation to charge can still destabilise. Tunable without a rebuild; if the 1HSG
# live-verify still looks permissive, raise it or move to relative SASA (RSA).
DESIGN_EXPOSED_SASA_THRESHOLD: float = float(
    os.environ.get("DESIGN_EXPOSED_SASA_THRESHOLD", "40")
)
# Fold guard for ranking: ESMFold mean-pLDDT floor below which a design is flagged
# as a likely misfolder, and how many top-CamSol designs to fold-check (ESMFold is
# the GPU/venv312 cost, so only the shortlist is folded).
DESIGN_FOLD_PLDDT_FLOOR: float = float(
    os.environ.get("DESIGN_FOLD_PLDDT_FLOOR", "70")
)
DESIGN_FOLD_CHECK_TOP_K: int = int(
    os.environ.get("DESIGN_FOLD_CHECK_TOP_K", "3")
)

# ── Anthropic ─────────────────────────────────────────────────────────────────

# NL→ChimeraX translation is LOCAL-ONLY (the local Ollama model; see translator.py
# and §0). There is no Claude/Anthropic backend, no API key, and no backend-selection
# switch — TRANSLATOR_BACKEND / TRANSLATOR_FALLBACK / ANTHROPIC_MODEL were removed.

# Canonical valid tool names — the EXACT (lowercase) literals the router
# dispatches on (`tool_router._dispatch_tool`). Used to ENUM-constrain the Ollama
# backend's `tools_needed` output (constrained decoding cannot then emit a
# misspelled / hallucinated / wrong-cased tool). A test asserts this matches the
# router registry. Keep sorted + in sync with `_dispatch_tool`.
TRANSLATOR_TOOL_NAMES: list = [
    "assembly_analyser", "bio_assembly", "camsol", "cavity", "chimerax", "colabfold",
    "conformer_comparison",
    "disulfide", "double_mutant", "esm", "esmfold", "glycan",
    "glycan_positions", "interface_stabilization", "mpnn_esmfold", "mutation_scan",
    "netnglyc", "proline", "proteinmpnn", "rfdiffusion", "rosetta", "salt_bridge",
    "validate_ddg", "validate_design",
]


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
# Per-request HTTP read timeout. With Ollama on the GPU a translate is ~3.5 s (cold
# model-load a bit more); 120 s is generous for a cold load yet lets a REAL hang surface
# in ~2 min instead of 10. (Was 600 — a CPU-era band-aid; on CPU it merely masked a hang
# as slowness. Raise via env if you deliberately run translation on CPU.)
OLLAMA_TIMEOUT: int = int(os.environ.get("OLLAMA_TIMEOUT", "120"))

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

# ── Deep-tier (PyRosetta) parallelization ───────────────────────────────────────
# The per-mutation FastRelax units are independent → run them through a pool inside
# the single WSL2 worker (one wsl.exe spawn; amortises the PyRosetta import + the
# cached WT relax).  Pure speedup with IDENTICAL results: each mutation's RNG is
# seeded deterministically from (ROSETTA_BASE_SEED, mutation_key), independent of
# worker/order, so parallel output == serial output.
#
# Worker count is CAPPED at runtime (rosetta_bridge.resolve_rosetta_workers) to
# min(configured, physical_cores − 2 headroom, wsl_mem_budget / per_worker_footprint).
# DEFAULT 8 = the P-core count of the i9-14900HX (8 P + 16 E; E-cores ~½ speed for
# FastRelax, single-channel DDR5 is bandwidth-limited, 140 W laptop throttles
# sustained all-core) — NOT the 32 logical threads (that throttles/swaps).
ROSETTA_MAX_WORKERS: int = int(os.environ.get("ROSETTA_MAX_WORKERS", "8"))
# Physical cores for the CPU cap (default best-effort = logical count; on this box
# set ROSETTA_PHYSICAL_CORES=24).  The cap leaves 2 cores of host headroom.
ROSETTA_PHYSICAL_CORES: int = int(
    os.environ.get("ROSETTA_PHYSICAL_CORES", str(os.cpu_count() or 8))
)
# WSL2 RAM available to the pool (default WSL = ~50% host; ~16 GB on 32 GB box).
# Leave host headroom (Chrome/Windows share the 32 GB) → budget 12 GB by default.
ROSETTA_WSL_MEM_BUDGET_MB: int = int(
    os.environ.get("ROSETTA_WSL_MEM_BUDGET_MB", "12000")
)
# Measured per-worker PyRosetta footprint (~1 GB on the 2HHB tetramer) + margin.
ROSETTA_WORKER_FOOTPRINT_MB: int = int(
    os.environ.get("ROSETTA_WORKER_FOOTPRINT_MB", "1200")
)
# Fixed base seed → deterministic, reproducible ddG and the parallel==serial
# identical-results contract.  (Ranking/sign are seed-independent regardless.)
ROSETTA_BASE_SEED: int = int(os.environ.get("ROSETTA_BASE_SEED", "1"))

# ── Deep-tier runtime estimate + size-aware worker cap (estimate-honesty fix) ───
# The pre-launch estimate is the entire user-facing surface of the opt-in design,
# so it must NOT undershoot.  Per-mutation FastRelax cost scales SUPER-linearly
# with pose size (memory-bandwidth bound on single-channel DDR5).  Calibrated to
# two measured anchors (WT relax cached, solo): 1CRN 46 res ≈ 10 s/mutation;
# 2HHB 574 res ≳ 733 s/mutation (one core, killed before finish → a lower bound).
#   per_mut_sec(n) = BASE_SEC × (n / BASE_RES) ** EXPONENT
#   @46 → 10 s ; @574 → ~940 s (biased ABOVE the >733 s lower bound) ; @141 → ~76 s
# EXPONENT 1.8 is intentionally conservative (round up; never undershoot an anchor).
ROSETTA_PER_MUT_BASE_SEC:  float = float(os.environ.get("ROSETTA_PER_MUT_BASE_SEC", "10"))
ROSETTA_PER_MUT_BASE_RES:  int   = int(os.environ.get("ROSETTA_PER_MUT_BASE_RES", "46"))
ROSETTA_PER_MUT_EXPONENT:  float = float(os.environ.get("ROSETTA_PER_MUT_EXPONENT", "1.8"))

# Per-worker PyRosetta footprint now scales with the ACTUAL pose size so the
# worker cap shrinks for large complexes and never oversubscribes WSL into swap
# (0% disk must STAY 0%).  Calibrated to the 2HHB tetramer: 8 workers ≈ 1.75 GB
# each (574 res) → BASE + PER_RES×n ⇒ 500 + 2.2×574 ≈ 1763 MB ⇒ cap_mem = 6 (not
# the lucky 8).  46-res monomer ≈ 600 MB.  Replaces the flat ROSETTA_WORKER_
# FOOTPRINT_MB, which is kept only as the fallback when pose size is unknown.
ROSETTA_WORKER_BASE_MB:    int   = int(os.environ.get("ROSETTA_WORKER_BASE_MB", "500"))
ROSETTA_WORKER_MB_PER_RES: float = float(os.environ.get("ROSETTA_WORKER_MB_PER_RES", "2.2"))

# Deep-tier coverage: FULL grid (all scoped positions × candidates_per_pos) is the
# DEFAULT (max data — the user's stated bias; safe because Rosetta is opt-in + the
# estimate is honest).  SHORTLIST is an explicit speed opt-in that validates only
# the fast tier's top-K by combined score; the rest are RETAINED as "not_computed"
# (shortlist never silently drops candidates).  When the full-grid estimate exceeds
# OFFER_SEC, the confirm/tier surface presents BOTH options with their estimates.
ROSETTA_SHORTLIST_K:        int   = int(os.environ.get("ROSETTA_SHORTLIST_K", "15"))
ROSETTA_FULL_GRID_OFFER_SEC: int  = int(os.environ.get("ROSETTA_FULL_GRID_OFFER_SEC", "300"))

# ddG basis: SYMMETRIC (per-mutation paired WT re-relax — variance-reduced, the
# reason to opt into Rosetta) is the DEFAULT.  ASYMMETRIC (score against the single
# cached global-WT relax) ~halves deep cost but is a DIFFERENT, noisier ddG basis —
# explicit fast-validation opt-in only, always labelled; the two are never mixed.
ROSETTA_DDG_BASIS:         str   = os.environ.get("ROSETTA_DDG_BASIS", "symmetric").strip().lower()

# Estimate large-pose guard (BUILD 3): the /workers divisor assumes ~linear speedup,
# but single-channel DDR5 is bandwidth-bound so parallel efficiency falls for poses
# LARGER than the measured 2HHB anchor.  Above REF_RES the estimate's effective
# worker count is scaled by efficiency = REF_RES / n_res (≤1) so it stays BIASED
# HIGH for big complexes (never undershoots).  At/below the anchor → no change.
ROSETTA_PARALLEL_EFF_REF_RES: int = int(os.environ.get("ROSETTA_PARALLEL_EFF_REF_RES", "574"))

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

# ── ThermoMPNN (fast-tier local stability voter, venv312 GPU) ───────────────────
# ThermoMPNN (Kuhlman-Lab) — a GNN stability (ddG) predictor built ON ProteinMPNN's
# encoder.  Runs in venv312 (torch 2.11 + Lightning 2.6.5 confirmed compatible;
# install does NOT touch torch, so ESM stays intact).  Weights ship in-repo
# (models/thermoMPNN_default.pt + bundled vanilla_model_weights/).  GRACEFUL: if
# disabled/absent/failing the scan still works (fast tier renormalises without it).
THERMOMPNN_DIR: str = os.environ.get(
    "THERMOMPNN_DIR", str(Path(__file__).parent / "ThermoMPNN_repo")
)
# venv312 interpreter for ThermoMPNN inference (reuses the GPU env + ProteinMPNN).
THERMOMPNN_PYTHON: str = os.environ.get("THERMOMPNN_PYTHON", VENV312_PYTHON)
THERMOMPNN_MODEL: str = os.environ.get(
    "THERMOMPNN_MODEL", str(Path(THERMOMPNN_DIR) / "models" / "thermoMPNN_default.pt")
)
# Enable the fast-tier ThermoMPNN voter.  "auto" = use it if THERMOMPNN_DIR is a
# valid install, else skip gracefully.  "false"/"0" disables (scan == pre-ThermoMPNN).
THERMOMPNN_ENABLE: str = os.environ.get("THERMOMPNN_ENABLE", "auto").strip().lower()
# Sign normalisation: ThermoMPNN trains on NEGATED Megascale ddG (datasets.py:161),
# Megascale positive = stabilising → ThermoMPNN predicts negative = stabilising,
# which ALREADY matches the system convention (positive = destabilising).  So the
# default multiplier is +1 (no flip).  The live sign-guard test confirms this on a
# known stabiliser; flip to -1 here ONLY if that test ever shows the opposite.
THERMOMPNN_DDG_SIGN: int = int(os.environ.get("THERMOMPNN_DDG_SIGN", "1"))
# PROVISIONAL fast-tier weight — PENDING BENCHMARK CALIBRATION (do not bless this
# number).  ThermoMPNN is the highest fast-tier voter (stability is the primary
# goal).  Renormalised with CamSol (_W_SOL) + ESM (_W_TOL) over present voters;
# CamSol:ESM stays 3:2 so a ThermoMPNN-absent scan falls back EXACTLY to the
# pre-ThermoMPNN 0.6/0.4.  NOTE for calibration: ThermoMPNN and ESM are NOT fully
# independent (ThermoMPNN runs on the ProteinMPNN encoder → shares structural
# context with ESM), whereas CamSol's solubility axis is orthogonal — calibrated
# weights must account for voter redundancy, not just standalone accuracy.
THERMOMPNN_WEIGHT: float = float(os.environ.get("THERMOMPNN_WEIGHT", "0.45"))

# ── RaSP — fast-tier PHYSICS-PROXY ddG voter (Rosetta surrogate) ───────────────
# RaSP (KULL-Centre/_2022_ML-ddG-Blaabjerg) is a trained surrogate for Rosetta
# ddG, so it fills the PHYSICS axis as a fast proxy — it is NOT an independent
# voter: when real Rosetta ddG exists for a candidate it HANDS OFF (RaSP's score
# contribution → 0, value retained as a proxy-QC delta).  Runs in a dedicated
# WSL2 venv (`~/rasp_env`, modern torch-CPU + openmm + pdbfixer + compiled reduce;
# the README's 2022 conda stack does not fit), subprocessed like the WSL bridges.
# GRACEFUL: disabled/absent/failing → fast tier renormalises over present axes.
RASP_DIR: str = os.environ.get("RASP_DIR", "/home/andre/RaSP_repo")
# WSL2 interpreter for RaSP inference (the dedicated rasp_env venv).
RASP_PYTHON: str = os.environ.get("RASP_PYTHON", "/home/andre/rasp_env/bin/python")
RASP_WSL_DISTRO: str = os.environ.get("RASP_WSL_DISTRO", "Ubuntu-24.04")
# Enable the fast-tier RaSP voter.  "auto" = use it if WSL + rasp_env are present,
# else skip gracefully.  "false"/"0" disables (scan == pre-RaSP, byte-for-byte).
RASP_ENABLE: str = os.environ.get("RASP_ENABLE", "auto").strip().lower()
# Sign normalisation: RaSP trains on Rosetta ddG (positive = destabilising), so it
# ALREADY matches the system convention; default +1 (no flip).  Live-confirmed on
# 1PGA (positive=destabilising, Pearson 0.88 vs the shipped Rosetta reference).
RASP_DDG_SIGN: int = int(os.environ.get("RASP_DDG_SIGN", "1"))
# PROVISIONAL physics-axis weight — PENDING BENCHMARK CALIBRATION (do not bless
# this number).  RaSP fills the PHYSICS slot ONLY when real Rosetta is absent for a
# candidate (per-candidate handoff); it never votes alongside Rosetta.  The §9
# aggregate-weighting spec (independence × confidence, staged by round) calibrates
# the physics-axis weight later — RaSP is the low-confidence proxy, Rosetta the
# high-confidence real value.  Kept equal to the Rosetta deep weight for now so the
# fast tier's physics slot is on the same scale; calibration will lower the proxy.
RASP_WEIGHT: float = float(os.environ.get("RASP_WEIGHT", "0.50"))
# Cache-busting tag folded into the RaSP result cache key alongside (pdb-hash,
# chain).  Bump when the sign, model, or port changes so a config change busts the
# cache instead of silently serving stale ddG.
RASP_VERSION_TAG: str = os.environ.get(
    "RASP_VERSION_TAG", f"rasp-v1;sign={RASP_DDG_SIGN};ds10;port=openmm8+srSASA"
)
# Wall-clock budget for one RaSP worker run (clean+extract+CPU inference).
RASP_TIMEOUT: int = int(os.environ.get("RASP_TIMEOUT", "600"))
# Per-(pdb,chain,version) RaSP result cache so a re-scan is free.
RASP_CACHE_DIR: Path = Path(os.environ.get("RASP_CACHE_DIR", str(_BASE / "cache" / "rasp")))
RASP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Controls whether ESM-2 uses the venv312 GPU backend.
#   "auto"      — use venv312 if it exists and passes a CUDA smoke-test (default)
#   "true"/"1"  — always use venv312; raise if unavailable
#   "false"/"0" — always use current-venv CPU path (disables GPU delegation)
ESM_USE_VENV312: str = os.environ.get("ESM_USE_VENV312", "auto").strip()

# ── Parallel DynaMut2 ─────────────────────────────────────────────────────────

# Maximum concurrent DynaMut2 requests.
# 4 is conservative — server handles up to ~5.  Set to 1 to disable parallelism.
DYNAMUT2_MAX_WORKERS: int = int(os.environ.get("DYNAMUT2_MAX_WORKERS", "4"))

# ── DynaMut2 SIGN normalisation (defined ONCE; both the single-mutation parser and
# the double-mutant mm path route through normalize_dynamut2_ddg) ──────────────
# EMPIRICALLY VERIFIED OPPOSITE: DynaMut2 reports positive = STABILISING (mCSM
# family), the OPPOSITE of the system convention (positive = destabilising).
# Live anchor on 2LZM: L99A (canonical destabiliser, exp +5.0) → DynaMut2 −3.32;
# V149A (exp +2.0) → −2.06.  So the multiplier is −1 (flip to system convention).
# (The pre-2026-06-10 code used the raw value with NO flip → it shipped inverted
# DynaMut2 ddG for both the single-mutation backend AND the double-mutant path.)
DYNAMUT2_DDG_SIGN: int = int(os.environ.get("DYNAMUT2_DDG_SIGN", "-1"))
# The SINGLE-mutation sign (above) is empirically verified (L99A live anchor).  The
# MULTIPLE-mutation (prediction_mm) sign is only INFERRED from the single-mutation
# family — the mm endpoint chronically ERRORs, so it could not be live-reconfirmed.
# Until it is, mm output is tagged sign_unverified so the inferred sign is never
# trusted silently (the mm fixture sign-corrections are as provisional as this).
# Set True ONLY after a live anti-symmetry / known-mutation check on prediction_mm.
DYNAMUT2_MM_SIGN_VERIFIED: bool = (
    os.environ.get("DYNAMUT2_MM_SIGN_VERIFIED", "false").strip().lower() in ("1", "true", "yes")
)

# ── DynaMut2 fast-tier DYNAMICS-axis voter (shortlist / round-2) ───────────────
# DynaMut2 (normal-mode dynamics + graph signature, ML-trained on experimental
# ddG) is the DYNAMICS axis — an INDEPENDENT voter (unlike RaSP it does NOT hand
# off / collapse; it always counts when present).  REMOTE API → SPARSE: runs only
# on the deep/round-2 candidate set (rides the deep opt-in), capped.
DYNAMUT2_ENABLE: str = os.environ.get("DYNAMUT2_ENABLE", "auto").strip().lower()
# Max candidates DynaMut2 scores per deep run (remote API can't do hundreds).  Over
# the cap → cover the top-N by combined score, mark the rest dynamics not_computed.
DYNAMUT2_MAX_CANDIDATES: int = int(os.environ.get("DYNAMUT2_MAX_CANDIDATES", "25"))
# PROVISIONAL dynamics-axis weight — PENDING BENCHMARK.  DynaMut2 is its OWN axis
# (no handoff); always counted when present, renormalised over present axes.  Note
# for calibration: DynaMut2 shares experimental supervision with ThermoMPNN →
# partial correlation, to be handled in the §9 aggregate-weighting benchmark.
DYNAMUT2_WEIGHT: float = float(os.environ.get("DYNAMUT2_WEIGHT", "0.45"))

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

# ── Interface stabilization ────────────────────────────────────────────────────

# Apply per-chain distinct coloring when the interface workflow runs so that
# each (sub-model × chain) pair gets a unique color.  Prevents the bychain
# collision where same-letter chains across sub-models render identically.
# Set INTERFACE_COLOR_BY_CHAIN=false to preserve a user-set coloring instead.
INTERFACE_COLOR_BY_CHAIN: bool = (
    os.environ.get("INTERFACE_COLOR_BY_CHAIN", "true").strip().lower()
    not in ("0", "false", "no", "off")
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

# ── RFdiffusion (WSL2 ~/rfdiffusion_env, de novo backbone diffusion) ────────────
# Cache dir for completed backbone-generation runs, keyed by a content hash of the
# resolved inputs (mode + contigs + hotspots + designs + steps + seed). A re-run
# of an identical request returns the cached PDB set instantly — mirror of the
# ColabFold fold cache. The bridge writes generated PDBs here (a Windows path that
# is also WSL-visible via /mnt/c, so the WSL run_inference.py writes straight into
# it and the Windows side collects them without a copy-back).
RFDIFFUSION_CACHE_DIR: Path = Path(
    os.environ.get("RFDIFFUSION_CACHE_DIR", str(_BASE / "cache" / "rfdiffusion"))
)
RFDIFFUSION_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
