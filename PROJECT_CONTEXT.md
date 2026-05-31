# StructureBot — Project Context

<!-- Regenerate with: claude "Read PROJECT_CONTEXT.md for regeneration instructions,
then regenerate it in full by reading the entire codebase. Preserve the Changelog
section and append a new entry." -->

## Meta

| Field | Value |
|-------|-------|
| Generated | 2026-05-31 (full resync after ColabFold env + bridge v1 + JAX compile cache + stack merged to `main`; hand-curated §7/§8 analytical text and the full §13 changelog preserved verbatim) |
| Test count at generation | **572 collected / 559 passing / 13 skipped** (12 PyRosetta benchmark tests skip without `STRUCTUREBOT_RUN_LIVE_BENCHMARK=1`+WSL2+PyRosetta; 1 ColabFold live fold skips without `STRUCTUREBOT_RUN_LIVE_COLABFOLD=1`) |
| Regenerate with | `claude "Read PROJECT_CONTEXT.md for regeneration instructions, then regenerate it in full by reading the entire codebase. Preserve the Changelog section and append a new entry."` |

This file is the single source of truth for project state. It should be regenerated after every build session. All source of truth lives in the `.py` files, not here.

---

## 1. Project Overview

StructureBot is a Windows-native natural-language interface for UCSF ChimeraX 1.11.1, running on Python 3.14 (main venv) with a GPU/ML delegation layer in Python 3.12 (venv312). Users type free-text biology requests ("suggest mutations to improve solubility of chain A, avoiding interfaces") into a Rich-console REPL; the system translates those requests via the Anthropic Claude API (claude-sonnet-4-6 with prompt caching), routes them through a pipeline of computational bridges (CamSol, ESM-2, ProteinMPNN, PyRosetta, DynaMut2, disulfide/proline/glycan/cavity/salt-bridge/double-mutant analysers), and executes the resulting ChimeraX commands via the ChimeraX REST API on port 60001. The application also supports a `--script` batch mode and session persistence via `session.json`.

---

## 2. Architecture

### Data Flow

```
User input (Rich REPL or --script file)
  │
  ▼
main.py / StructureBot
  │  1. config.load_env_file()  ← loads .env.local FIRST
  │  2. CommandTranslator.translate()
  │     └─ Anthropic API (claude-sonnet-4-6)
  │        Block 1: STATIC prompt (cached, ephemeral cache_control)
  │        Block 2: DYNAMIC session context (uncached, changes per turn)
  │  3. ToolRouter.route()      ← augments result, no execution
  │     Intent overrides (in order): validate_ddg → colabfold → proline → mpnn_esmfold
  │     → glycan_positions → netnglyc → glycan → salt_bridge → cavity → double_mutant
  │     → mutation_scan (fallback, last)
  │  4. User confirmation / auto-proceed countdown
  │  5. ChimeraXBridge.run_commands()  ← initial viz commands
  │  6. ToolRouter.execute()    ← computational pipeline
  │     ├─ CamsolBridge         ← local algorithm, no network
  │     ├─ EsmBridge            ← delegates to venv312 subprocess
  │     ├─ ESMFoldBridge        ← delegates to venv312 subprocess
  │     ├─ RosettaBridge        ← DynaMut2 API or WSL2 PyRosetta
  │     ├─ MutationScanner      ← orchestrates CamSol+ESM+Rosetta
  │     ├─ DoubleMutantBridge   ← DynaMut2 prediction_mm + optional PyRosetta
  │     ├─ DisulfideBridge      ← BioPython + ESM + DynaMut2
  │     ├─ ProlineBridge        ← BioPython + ESM
  │     ├─ GlycanBridge         ← BioPython + ESM + NetNGlyc API
  │     ├─ AssemblyAnalyser     ← RCSB API + ChimeraX zone-select
  │     ├─ SaltBridgeBridge     ← BioPython + FreeSASA
  │     ├─ CavityBridge         ← BioPython ShrakeRupley
  │     ├─ ProteinMPNNBridge    ← subprocess: protein_mpnn_run.py
  │     ├─ ColabFoldBridge      ← WSL2 ~/colabfold_env worker (AF2, remote MSA)
  │     └─ RFdiffusionBridge    ← STUB (not activated)
  │  7. ChimeraXBridge.run_commands()  ← viz commands from tools
  │  8. SessionState.add_to_history()
  │
  ▼
ChimeraX REST API  (http://127.0.0.1:60001/run)
```

### Two-Venv Architecture

| Venv | Python | Purpose |
|------|--------|---------|
| `venv/` | 3.14 | Main process: Anthropic SDK, Rich, BioPython, requests, all bridges |
| `venv312/` | 3.12 | GPU/ML delegation: torch 2.11.0+cu128 (RTX 5070 Ti, sm_120), ESM-2 inference, ESMFold inference |

The main venv cannot run GPU inference because no PyTorch cu128 build exists for Python 3.14. ESM-related bridges spawn `venv312/Scripts/python.exe` as a subprocess with JSON I/O (`--input`, `--output` temp files). Worker scripts have **no project imports** — they only import `torch`, `transformers`, `esm`, and stdlib.

### WSL2 Layer (PyRosetta)

When `ROSETTA_BACKEND=local`, `RosettaBridge._run_rosetta_local()` builds a standalone Python worker script as an f-string, writes it to a Windows temp file, translates the path to `/mnt/c/...` form, and runs it via `WSLBridge.run_python_script()` → `wsl.exe --distribution Ubuntu-24.04 --exec bash -c "{PYROSETTA_PYTHON} '{wsl_path}'"`. Results are returned as a JSON file at `/tmp/rosetta_ddg_{hash}.json`, copied back to Windows via `wsl.exe`. Same pattern applies to `DoubleMutantBridge._run_pair_pyrosetta()` for close pairs.

**Worker script constraints** (critical, applies to ALL worker scripts):
- No project imports — completely standalone
- Communicates only via JSON files in `/tmp/`
- All subprocess calls in `wsl_bridge.py` use `stdin=subprocess.DEVNULL` and `creationflags=subprocess.CREATE_NO_WINDOW`

---

## 3. Module Registry

| File | Class / Key Functions | Status | Notes |
|------|----------------------|--------|-------|
| `main.py` | `StructureBot`, `_ElapsedTicker`, `_maybe_restore_session()`, `_reconnect_or_offer_reopen()`, `_cmd_clear_session()` | ✅ Complete | REPL + `--script` mode; startup sequence. **Session auto-restore on startup**: offers to restore a previous `session.json` (restore / keep / delete), then checks whether each restored structure is still in ChimeraX and offers a fast re-open if not; `clear session` / `new session` command wipes `session.json` and resets state. Script mode skips restore prompts (`_interactive=False`) |
| `config.py` | constants + `load_env_file()` | ✅ Complete | Called first in `main.py`; all env-var overrides centralised here |
| `translator.py` | `CommandTranslator` | ✅ Complete | Claude API; prompt caching (Block 1 static, Block 2 dynamic); rolling history `MAX_CONVERSATION_HISTORY=6` |
| `tool_router.py` | `ToolRouter`, `ToolStepResult` | ✅ Complete | Dispatches 15 tool types including `double_mutant` and `validate_ddg` (high-accuracy multi-trajectory ddG tier — intent: "validate ddg", "high-accuracy/high-confidence stability", "confirm ddg"; parses named mutations or pulls top-3 from scan_results); intent detection for all tools; MPNN+ESMFold pipeline; FASTA export; sequence display fast-path |
| `session_state.py` | `SessionState`, `try_load()`, `_from_dict()`, `restore_summary()`, `parse_pdb_header()`, `fetch_rcsb_metadata()` | ✅ Complete | Persists all tool results including `double_mutant_results`, `colabfold_results`, `validate_design_results`; save/load/snapshot/restore. **Auto-restore support**: `try_load()` returns `(state, error)` — `(None, None)` missing file, `(None, "msg")` corrupt/incompatible, `(state, None)` ok; `load()` delegates and never raises (fresh state on failure); `_from_dict()` shared builder; `restore_summary()` one-screen summary for the startup prompt |
| `chimerax_bridge.py` | `ChimeraXBridge`, `find_chimerax()`, `_run_command_once()`, `_try_reconnect()` | ✅ Complete | REST API on port 60001; blank-image post-save guard. **Auto-reconnect**: `run_command()` wraps `_run_command_once()`; on a dropped connection (`ConnectionError` at pre-check or mid-request) it calls `ensure_connected()` once and retries — succeeds silently or raises a clear "check ChimeraX is still open" error. `run_commands()` inherits this per-command |
| `wsl_bridge.py` | `WSLBridge`, `PYROSETTA_PYTHON`, `COLABFOLD_PYTHON`, `check_colabfold()` | ✅ Complete | `PYROSETTA_PYTHON="/home/andre/pyrosetta_env/bin/python"`; `COLABFOLD_PYTHON="/home/andre/colabfold_env/bin/python"`; default distro `Ubuntu-24.04`; `check_pyrosetta()`/`check_colabfold()` use `chr(79)+chr(75)`. `run_python_script(..., python_bin=PYROSETTA_PYTHON)` — backward-compatible param so callers can run inside the ColabFold env |
| `rosetta_bridge.py` | `RosettaBridge`, `_select_backend()`, `_run_command_*` (DynaMut2), `_run_rosetta_local()` | ⚠️ Magnitude bias | 4 backends (dynamut2/empirical/pyrosetta-stub/local-WSL2). **DynaMut2 single parser fixed** — handles current status-based API (`status` DONE/RUNNING/ERROR), robust float cast of `prediction`, raises (never silent 0.0) on ERROR. **Local PyRosetta path hardened**: `pose_from_file` validation + `.pdb`-extension/`pose_from_pdb` fallback; whole-batch failure → per-mutation **empirical fallback** (not all-zero); each value carries **`ddg_source`** (`pyrosetta`/`empirical`). **Tiered multi-trajectory ddG**: `_run_rosetta_local(num_trajectories=, relax_cycles=)` runs N independent relax+score trajectories per mutation (per-trajectory RNG seed when N>1), aggregates by **median** (`_aggregate_ddg_trajectories`), reports **MAD spread** + **`ddg_confidence`** (`_ddg_confidence_label`); WSL timeout scales by N×cycles. Defaults (N=1, cycles=3) = production single-trajectory, unchanged. **Protocol = manual symmetric FastRelax + torsion-space `ref2015` (`:1449-1464`), NOT the `cartesian_ddg` app / `ref2015_cart`; reported "kcal/mol" is a raw REU delta with no conversion — magnitudes are UNCALIBRATED (2.94 doesn't apply; ranking/sign reliable). See §7.** **`validate_ddg()`** = high-accuracy tier (ROSETTA_VALIDATION_TRAJECTORIES/CYCLES) with a non-calibrated disclosure. Path validated for ranking (T4 r≈+0.49, sign 100%); **validation-tier 2LZM panel (N=5 × 8+8 cycles, median, 2026-05-30): RMSE 2.73, MAE 2.59, r +0.50, sign 90% — meets all revised thresholds (r>0.3, RMSE<4.0, sign>=60%)**, roughly halving single-traj RMSE (5.23→2.73). Magnitudes now tighter but still approximate (~±2.7 kcal/mol; gain is in magnitude, not ranking). **Those panel numbers were measured with `ROSETTA_STRIP_WATERS=true` (strip), which — reverted 2026-05-31 — is again the default, so they describe the DEFAULT path. Preserve-waters (`False`) is now opt-in, validated only on buried-water target T26A (+3.80 shift, sign corrected); preserve-path panel behaviour is unmeasured.** See §7/§8. **`relax_and_score(pdb_path, relax_cycles=3)`** (added for validate-design): FastRelax + `ref2015` total/per-residue/`fa_rep` on an ARBITRARY structure (NO mutation) via the same WSL2 worker building blocks → `{total_reu, per_residue_density, fa_rep, clash_ok, converged, relaxed_pdb}`; returns a dict, never raises; **does NOT touch the ddG path** (backward-compatible). Live-validated (HP36: −88.4 REU, 36 res) |
| `double_mutant_bridge.py` | `DoubleMutantBridge`, `generate_pairs()`, `compute_ca_distance()`, `route_pairs()`, `score_pairs_dynamut2()`, `score_pairs_pyrosetta()`, `compute_composite_score()`, `generate_summary()`, `_apply_additive_fallback()` | ✅ Complete | Two-mode (stability/epitope) double mutant ΔΔG scoring; DynaMut2 `prediction_mm` for distant pairs (>10 Å), PyRosetta WSL2 for close pairs (<4 Å); real epistasis = ddG(double) − ddG(additive). Stability ddG filter excludes a pair only when **both** mutations are clearly destabilising (`ddg > DOUBLE_MUTANT_DESTABILISING_DDG`); `ddg=0.0` treated as neutral. Close pairs fall back to additive scoring when `run_pyrosetta=False` (no longer dropped). Shared `_apply_additive_fallback()` helper. Routing now logs a single concise summary (`Routed N pairs: X dynamut2, Y dynamut2_warned, Z pyrosetta_required`) — the verbose per-pair dict dumps were removed. Validated end-to-end (171→79 funnel confirmed live). 42 tests |
| `esm_bridge.py` | `EsmBridge` | ✅ Complete | ESM-2 `esm2_t6_8M_UR50D` default; delegates GPU inference to `esm_worker.py` via venv312 subprocess; disk cache `cache/esm_{hash}.json` |
| `esm_worker.py` | standalone subprocess script | ✅ Complete | No project imports; writes JSON result file; run by venv312 python |
| `esmfold_bridge.py` | `ESMFoldBridge` | ✅ Complete | Primary: venv312 GPU via `esmfold_worker.py`; fallback: ESM Atlas API; `compare_to_wildtype()`, `check_disulfide_foldability()` |
| `esmfold_worker.py` | standalone subprocess script | ✅ Complete | HuggingFace `facebook/esmfold_v1`; no project imports; pLDDT normalisation guard (×100 if mean < 2.0) |
| `mutation_scanner.py` | `MutationScanner` | ✅ Complete | CamSol+ESM+Rosetta pipeline; combined score `0.5×(-ddG) + 0.3×camsol_delta + 0.2×esm_tolerance`; Pro/Cys exclusion; interface protection. `_run_rosetta_batch` returns `(ddg_scores, ddg_source, ddg_spread, ddg_confidence)`; spread/confidence threaded into scan results + the summary table (scans are single-trajectory → confidence "single-trajectory") |
| `camsol_bridge.py` | `CamsolBridge` | ✅ Complete | Local CamSol algorithm; window=9, β=3.0; no network required |
| `disulfide_bridge.py` | `DisulfideBridge` | ✅ Complete | Cβ-Cβ geometry (4.5 Å cutoff) + ESM tolerance + DynaMut2 stability; combined score 0.4/0.3/0.3 |
| `proline_bridge.py` | `ProlineBridge` | ✅ Complete | φ-angle scoring; DSSP or Ramachandran fallback; functional residue exclusion |
| `glycan_bridge.py` | `GlycanBridge` | ✅ Complete | NXS/T sequon detection + SASA + SS + ESM + projection scoring; engineered sequon suggestion; NetNGlyc integration |
| `netnglyc_bridge.py` | `predict_glycosylation()`, `integrate_with_glycan_candidates()` | ✅ Complete | DTU NetNGlyc 1.0 REST API; OST recognition scoring |
| `assembly_analyser.py` | `AssemblyAnalyser`, `fetch_assembly_info()` | ✅ Complete | RCSB assembly API; ChimeraX zone-select for interface detection (5 Å CA); monomer/multimer mode |
| `salt_bridge_bridge.py` | `SaltBridgeBridge` | ✅ Complete | BioPython + FreeSASA; Asp/Glu↔Arg/Lys/His within 4 Å |
| `cavity_bridge.py` | `CavityBridge` | ✅ Complete | BFS clustering on buried Cα; SASA < threshold; approximate volume (n_residues × 15 Å³); assembly-aware |
| `structural_utils.py` | `extract_backbone_angles()`, `compute_sasa()`, `compute_projection_score()`, `classify_sequon_geometry()` | ✅ Complete | Shared geometry utilities; used by `glycan_bridge.py` and `proline_bridge.py` |
| `colabfold_bridge.py` | `ColabFoldBridge`, `predict()`, `estimate_runtime_s()`, `_build_worker()`, `is_available()` | ✅ Complete (v1) | AF2-quality sequence→structure via the WSL2 `~/colabfold_env` (run via `wsl_bridge.COLABFOLD_PYTHON`). f-string worker (zero project imports, JSON in `/tmp`, copy back), **remote MSA server** (no local DBs). copies→colon-joined homo-oligomer (multimer); optional custom template; num_models/num_recycle/`quick` preset. **Total-residue guard** `COLABFOLD_MAX_TOTAL_RESIDUES` (pre-launch + runtime CUDA-OOM catch → clear message, never a raw traceback). Returns ranked PDB + per-residue/mean pLDDT + PAE + pTM(/ipTM) + PNG paths; input-hash cache `cache/colabfold_{hash}/`. NEVER raises (error-first dict). **JAX persistent compilation cache** (`COLABFOLD_JAX_COMPILE_CACHE_DIR`, WSL2 ext4) set on the `colabfold_batch` process so XLA reuses compiled executables across the fresh-per-fold workers — a different sequence skips most of the ~10-min recompile (measured 36-aa: cold 538s → warm 287s, ~47%; partial because MSA-depth differs, see §8). DEFERRED: fused validate-design meta-tool, MPNN auto-pull, batch top-N, amber relax, single_sequence (see §9) |
| `rfdiffusion_bridge.py` | `RFdiffusionBridge` | 🔲 Stub | Documented stub; returns helpful error unless `RFDIFFUSION_DIR` configured; requires Python 3.9-3.11 |
| `log_analyser.py` | `display_stats()` | ✅ Complete | Parses JSONL session logs; `stats` command in REPL |
| `diag.py` | — | ✅ Complete | One-off diagnostic script; tests WSL2 availability + PyRosetta import |

---

## 4. Configuration Reference

### ChimeraX

| Constant | Default / Current value | Description |
|----------|------------------------|-------------|
| `CHIMERAX_PATH` | `C:\Users\andre\documents\ChimeraX 1.11.1\bin\ChimeraX.exe` | Path to ChimeraX executable |
| `REST_HOST` | `"127.0.0.1"` | REST server host |
| `REST_PORT` | `60001` | REST server port |
| `REST_TIMEOUT` | `10` | Seconds per HTTP request to ChimeraX |

### Anthropic / LLM

| Constant | Default / Current value | Description |
|----------|------------------------|-------------|
| `ANTHROPIC_MODEL` | `"claude-sonnet-4-6"` | Model used for translation |
| `MAX_CONVERSATION_HISTORY` | `6` | Rolling history pairs (turns) |
| `AUTO_PROCEED_DELAY` | `2` | Seconds before auto-executing (0 = always prompt) |

### Directories

| Constant | Default / Current value | Description |
|----------|------------------------|-------------|
| `LOG_DIR` | `<project>/logs/` | Session JSONL log files |
| `SESSION_DIR` | `<project>/sessions/` | Named session `.cxs` + `.json` files |

### Stability / ddG

| Constant | Default | `.env.local` override | Description |
|----------|---------|----------------------|-------------|
| `ROSETTA_BACKEND` | `"auto"` | **`"local"`** ← ACTIVE | Backend: `auto`/`dynamut2`/`empirical`/`pyrosetta`/`local` |
| `ROSETTA_LOCAL_PATH` | `""` | — | Path to Rosetta binary dir (unused; local backend uses PyRosetta via WSL2) |
| `PYROSETTA_AVAILABLE` | `False` | — | Legacy flag for direct PyRosetta import (not the WSL2 path) |
| `ROSETTA_RELAX_CACHE` | `<project>/cache/rosetta_relaxed/` | — | Cached FastRelax'd PDB files (keyed by MD5) |
| `ROSETTA_STRIP_WATERS` | `True` (strip) | `false` to preserve | Strip crystallographic HOH before PyRosetta (default; validated, standard practice). Set `False` to opt into **preserve** (re-append HOH; validated only on buried-water T26A). Preserve-all-static is a Rosetta anti-pattern — proper fix = selective buried-only/movable (Build Queue §9). Relax cache is namespaced by mode (`_wat` suffix for preserve) |
| `relax_cycles` (param, not env) | `3` | — | `_run_rosetta_local()` parameter: FastRelax cycles for the per-mutation mutant relax + symmetric WT re-relax. Default 3 = production protocol (unchanged); exposed only so the convergence diagnostic can sweep 3/5/8. Cached WT baseline relax unaffected |

### WSL2

| Constant | Default | Description |
|----------|---------|-------------|
| `WSL_DISTRIBUTION` | `"Ubuntu-24.04"` | WSL2 distro name |

### venv312 / ESM

| Constant | Default / Current value | Description |
|----------|------------------------|-------------|
| `VENV312_PYTHON` | `<project>/venv312/Scripts/python.exe` | Python 3.12 GPU interpreter |
| `PROTEINMPNN_DIR` | `<project>/ProteinMPNN` | `.env.local`: `C:\Users\andre\documents\structurebot\ProteinMPNN` |
| `ESM_USE_VENV312` | `"auto"` | `.env.local`: `"auto"` — use venv312 if CUDA smoke-test passes |

### ESMFold

| Constant | Default | Description |
|----------|---------|-------------|
| `ESMFOLD_ENABLED` | `True` | Enable ESMFold fold validation on top candidates |
| `ESMFOLD_TOP_N` | `3` | Top candidates to check after mutation scan |
| `ESMFOLD_PLDDT_WARNING_THRESHOLD` | `10.0` | pLDDT drop (at mutation site) that triggers "high" risk |
| `ESMFOLD_USE_LOCAL` | `True` | Prefer local GPU over Atlas API |
| `ESMFOLD_MODEL_NAME` | `"facebook/esmfold_v1"` | HuggingFace model ID |
| `ESMFOLD_WORKER_TIMEOUT_COLD` | `600` | Seconds — cold start (weights not cached) |
| `ESMFOLD_WORKER_TIMEOUT_WARM` | `120` | Seconds — warm start |
| `ESMFOLD_FORCE_COLD_TIMEOUT` | `False` | Force 600 s timeout regardless of cache state |

### DynaMut2 / Parallel

| Constant | Default | Description |
|----------|---------|-------------|
| `DYNAMUT2_MAX_WORKERS` | `4` | Concurrent DynaMut2 requests (set to 1 to disable) |

### Double Mutant Scoring

| Constant | Default | Description |
|----------|---------|-------------|
| `DOUBLE_MUTANT_DISTANCE_THRESHOLD_FAR` | `10.0` | Cα-Cα distance (Å) above which DynaMut2 is reliable for double mutants |
| `DOUBLE_MUTANT_DISTANCE_THRESHOLD_CLOSE` | `4.0` | Cα-Cα distance (Å) below which PyRosetta is required (close pairs) |
| `DOUBLE_MUTANT_MAX_PAIRS` | `500` | Max pairs to consider before distance-based routing |
| `DOUBLE_MUTANT_TOP_N` | `10` | Default number of top-ranked pairs to return |
| `DOUBLE_MUTANT_DESTABILISING_DDG` | `2.0` | ΔΔG (kcal/mol) above which a mutation is "clearly destabilising". Stability mode drops a pair only when **both** mutations exceed this; `ddg=0.0` (DynaMut2 neutral/unknown) is never filtered |
| `DOUBLE_MUTANT_ADDITIVE_FALLBACK` | `True` | When True, pairs with no epistasis-aware backend available (DynaMut2 mm API down, or close pair with PyRosetta disabled) are scored additively (ddG_A + ddG_B, epistasis=0) instead of dropped |

### NetNGlyc

| Constant | Default | Description |
|----------|---------|-------------|
| `NETNGLYC_API_URL` | `"https://services.healthtech.dtu.dk/service.php?NetNGlyc-1.0"` | NetNGlyc 1.0 REST endpoint |
| `NETNGLYC_TIMEOUT` | `30` | HTTP timeout in seconds |
| `NETNGLYC_ENABLED` | `True` | Set False to skip all NetNGlyc calls |
| `NETNGLYC_TOP_N` | `5` | Top glycan candidates to annotate |

### MPNN + ESMFold Pipeline

| Constant | Default | Description |
|----------|---------|-------------|
| `MPNN_ESMFOLD_TOP_N` | `3` | Top MPNN designs to validate with ESMFold |
| `MPNN_ESMFOLD_INCLUDE_WT` | `True` | Include wildtype as baseline in ESMFold validation |

---

## 5. Critical Conventions

### Subprocess rules (wsl_bridge.py, enforced everywhere)
All subprocess calls that interact with Windows console handles **must** use:
```python
stdin=subprocess.DEVNULL
creationflags=subprocess.CREATE_NO_WINDOW
```
Rationale: `wsl.exe` calls `SetConsoleMode()` on the inherited stdin handle, which permanently disables `ReadConsole()` for the rest of the process lifetime, breaking the Rich REPL.

### Path handling rule
All file paths passed to ChimeraX commands **must** use `.as_posix()` to produce forward slashes. ChimeraX on Windows rejects backslashes in `save "..."` and `open "..."` commands.
```python
cx_fwd = Path(some_path).as_posix()
result = bridge.run_command(f'save "{cx_fwd}"')
```

### Worker script rules (ESM worker, ESMFold worker, PyRosetta worker, double mutant PyRosetta worker)
Worker scripts that run in subprocesses or WSL2 **must**:
1. Have zero imports from the StructureBot project
2. Communicate exclusively via JSON files (never stdin/stdout for structured data)
3. Write a result file even on exception (`try/except` at the outermost level)
4. Use `flush=True` on all print calls
5. PyRosetta workers: use double-braces `{{...}}` throughout (embedded in Python f-string)

### ChimeraX selector ordering rule
In ChimeraX commands, specifiers must appear in this order to avoid syntax errors:
```
#{model}/{chain}:{residue}@{atom}
```
Example: `#1/A:82@CA` — model first, then chain, then residue, then atom.

### Primary model guard
Commands that should only act on the first loaded model must use `#1` explicitly. The translator's static system prompt instructs Claude to always emit `#1` for single-model operations.

### f-string double-brace rule in worker scripts
The PyRosetta and double-mutant worker scripts are embedded as Python f-strings. Any literal `{...}` that should appear in the worker code must be written as `{{...}}`. Only actual f-string interpolations from the outer scope use single braces (e.g., `{wsl_pdb!r}`, `{mut_list_json!r}`).

### Error-first return convention (all bridges)
All bridge `analyze()` methods return `ToolStepResult` — they **never raise**. On failure, `result.success = False` and `result.error` contains the message. Callers check `result.success` and degrade gracefully.

### Double mutant mode detection (in `_run_double_mutant`)
Epitope keywords: `"epitope"`, `"binding"`, `"interface"`, `"preserve"`, `"target"` → `mode = "epitope"`. Otherwise → `mode = "stability"`. PyRosetta keywords: `"pyrosetta"`, `"rosetta"`, `"accurate"`, `"high accuracy"`, `"validate"` → `run_pyrosetta = True`.

### Double mutant ddG / fallback semantics
- **`ddg=0.0` is neutral/unknown, not destabilising.** The stability-mode pair filter excludes a pair only when **both** mutations have `ddg > DOUBLE_MUTANT_DESTABILISING_DDG` (default +2.0 kcal/mol). A single `ddg=0.0` mutation never eliminates a pair. (Earlier logic required at least one *beneficial* mutation, which collapsed the candidate set to ~2 pairs whenever DynaMut2 returned all-zero ΔΔG.)
- **Additive fallback never drops a pair.** When no epistasis-aware backend is available — DynaMut2 mm API down, circuit breaker tripped, or a close pair with `run_pyrosetta=False` — `_apply_additive_fallback()` scores the pair as `ddG_A + ddG_B` with `epistasis=0.0` and `backend_used="additive_fallback"`, and appends a warning. Set `DOUBLE_MUTANT_ADDITIVE_FALLBACK=False` to drop such pairs instead.

---

## 6. Test Suite Summary

**Total: 606 collected | 592 passing | 14 skipped** (2026-06-01, real `pytest` run). Non-benchmark suite `pytest tests/ --ignore=tests/test_rosetta_benchmark.py -q` → **592 passed + 2 skipped** (~40s); the 2 skips are the opt-in live ColabFold fold and the opt-in live validate-design e2e. **`tests/test_validate_design.py` (34 collected = 33 + 1 opt-in live)** — now also covers the upgraded RMSD axis: all-pairs headline (real-string regression), pruned/all-pairs+counts parse incl. missing-`across all` fallback, concentration math/descriptor, top-K selection, per-residue Kabsch deviation (localizes a displaced residue; needs ≥3 matches), deviation colour-command construction: PURE energy-decision honesty logic (sanity / relative-on-match / **DECLINE cross-topology with no number emitted**), topology extraction (fold + PDB), fold REUSE-vs-fold, matchmaker RMSD parse, relax-score parse, full report assembly + session storage, energy-unavailable degradation, clash/low-pLDDT flags, intent + routing precedence (wins the `mpnn_esmfold` keyword collision; doesn't grab `colabfold`). All three sub-steps mocked. The 12 PyRosetta benchmark tests skip unless `STRUCTUREBOT_RUN_LIVE_BENCHMARK=1` + WSL2 + PyRosetta. **`tests/test_colabfold_bridge.py` (37 collected = 36 passing + 1 opt-in live skip)**: worker schema/compile + remote-MSA & template flags, **JAX compile-cache env wiring** (set before the subprocess; omitted when empty; predict passes the config dir), oligomer colon-join, total-residue guard (incl. copy counting), result parsing (pLDDT/PAE/pTM/ipTM), runtime-OOM + worker-error surfacing, cache hit/miss + key sensitivity, ETA scaling, intent detection (positive/negative, no esmfold hijack), routing + option parsing, viz construction (open/alphafold-palette/sequence/matchmaker), ChimeraX sequence-on-open hook, env-absent skip paths; one opt-in live monomer fold (`STRUCTUREBOT_RUN_LIVE_COLABFOLD=1`). All WSL/ChimeraX calls mocked — no live fold in CI.

Run commands:
```bash
# Full suite (excludes slow benchmarks)
pytest tests/ --ignore=tests/test_rosetta_benchmark.py -q

# Including benchmark collection check
pytest tests/ -q --collect-only

# PyRosetta benchmarks (slow, requires WSL2+PyRosetta)
pytest tests/test_rosetta_benchmark.py -m benchmark -v --timeout=1800 -s

# Double mutant tests only
pytest tests/test_double_mutant_bridge.py -v

# Single benchmark spot-check
pytest tests/test_rosetta_benchmark.py -m benchmark -v -s -k "t4_l99a"
```

**Note:** `pytest.ini` now registers `benchmark`, `slow`, and `timeout` as known markers — no more `PytestUnknownMarkWarning`.

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_glycan_bridge.py` | 56 | N-glycan sequon detection, SASA scoring, engineered sequon suggestion, NetNGlyc integration |
| `test_colabfold_bridge.py` | 37 | ColabFold v1: worker compile + remote-MSA/template/JAX-compile-cache flags, oligomer colon-join, total-residue guard, pLDDT/PAE/pTM/ipTM parsing, OOM/error surfacing, result-cache hit/miss + key sensitivity, ETA, intent + routing + option parsing (no esmfold hijack), viz construction, sequence-on-open hook, env-absent skips; 1 opt-in live monomer fold (skipped in CI) |
| `test_validate_design.py` | 34 | Validate-design meta-tool: PURE energy-decision honesty (sanity / relative-on-topology-match / DECLINE cross-topology emits NO number + a reason), topology from fold/PDB, fold reuse-vs-fold (no re-fold when in-session), matchmaker RMSD parse, relax-score parse, evidence-rich report assembly + session storage, energy-unavailable degradation, clash/low-pLDDT flags, intent + route precedence; 1 opt-in live e2e (reuses cached fold + real relax-score) |
| `test_tool_router.py` | 62 | Route dispatch, MPNN+ESMFold routing, FASTA export, active-site commands, double mutant routing, tool icon registry |
| `test_double_mutant_bridge.py` | 42 | Pair generation (stability and epitope modes), distance routing, DynaMut2 `prediction_mm` result parsing, PyRosetta worker schema, composite scoring formulas, epistasis sign convention, max-pairs cap, ddG-filter neutrality (`ddg=0.0` not excluded), additive fallback (API failure, circuit-breaker survival, close-pair-without-PyRosetta) |
| `test_proline_bridge.py` | 35 | φ-angle scoring, functional residue exclusion, DSSP fallback, BioPython parsing |
| `test_disulfide.py` | 35 | Cβ geometry, dihedral scoring, ESM tolerance, DynaMut2 mock, combined score |
| `test_mpnn_esmfold_pipeline.py` | 29 | MPNN+ESMFold combined pipeline, session routing, pLDDT comparison |
| `test_rosetta.py` | 31 | Backend detection, DynaMut2 HTTP mock + single-parser status format (DONE/RUNNING/string/ERROR), MutationScanner, combined scoring, session persistence |
| `test_proteinmpnn.py` | 26 | ProteinMPNN subprocess call, JSON output parsing, error paths |
| `test_esmfold.py` | 25 | ESMFold local/atlas paths, pLDDT normalisation, foldability risk thresholds |
| `test_tools.py` | 24 | Integration: CamSol, ESM, ChimeraX commands, DynaMut2 stub |
| `test_assembly.py` | 23 | RCSB assembly metadata, monomer/multimer mode, interface detection mock |
| `test_cavity_bridge.py` | 20 | BFS cavity clustering, SASA burial, volume estimation, interface flagging |
| `test_netnglyc_bridge.py` | 19 | OST score parsing, harmonic mean integration, API mock |
| `test_structural_utils.py` | 17 | `extract_backbone_angles()`, `compute_sasa()`, `compute_projection_score()` |
| `test_wsl.py` | 12 | `WSLBridge` availability, path translation, `run_command()` (skip if no WSL2) |
| `test_rosetta_benchmark.py` | 12 | PyRosetta ddG benchmarks vs ProThermDB (11 mutations + 1 correlation test); `@pytest.mark.benchmark` and `@pytest.mark.slow`; skipped without WSL2+PyRosetta |
| `test_rfdiffusion.py` | 12 | RFdiffusion stub error structure, route detection, directory validation |
| `test_salt_bridge_bridge.py` | 11 | Salt bridge geometry, charge classification, SASA burial scoring |
| `test_main.py` | 10 | `StructureBot` startup mocking, semicolon chaining, `--script` runner |
| `test_translator.py` | 8 | `CommandTranslator` API mock, prompt caching structure, history management |
| `test_session_state.py` | 7 | Auto-restore: `try_load`/`load` robustness (corrupt/missing/non-dict JSON), scan-result round-trip, `clear session` wipe, ChimeraX presence check + re-open offer |
| `test_integration.py` | 6 | End-to-end route + execute mock (no live ChimeraX or APIs) |
| `test_chimerax_bridge.py` | 4 | Auto-reconnect: dropped-connection retry succeeds silently, clear error when reconnect fails, no reconnect on success, `run_commands` mid-list recovery |

**Skip conditions:**
- `test_rosetta_benchmark.py`: entire module skipped if `WSLBridge().is_available()` is False **or** `WSLBridge().check_pyrosetta()` is False
- `test_wsl.py`: individual tests skip if WSL2 not installed
- `test_esmfold.py`: local-path tests skip if `venv312` python not found

---

## 7. Validated Live Results

The following results have been confirmed working against **1HSG** (HIV-1 protease, PDB ID 1HSG, homodimer, chains A+B, ligand MK1):

| Feature | Validated result |
|---------|-----------------|
| CamSol solubility scan | Chain A scored; aggregation-prone residues coloured red in ChimeraX |
| ESM-2 conservation | Chain A scored; conserved residues blue, variable red |
| Mutation scan (full pipeline) | Top candidate `I64E` (ΔΔG = −3.53 kcal/mol by DynaMut2); `V82A` reference used in tests |
| V82A ddG (DynaMut2) | ~+1.5 to +2.0 kcal/mol (Mahalingam et al.); used as benchmark reference |
| Assembly analysis | 1HSG detected as homodimer (A2 stoichiometry); interface residues between chains A+B stored |
| Disulfide candidates | Chain A↔B candidates with Cβ-Cβ < 4.5 Å ranked by geometry+ESM+stability |
| Proline scan | φ-angle candidates on chain A; functional residue exclusion when active-site set |
| Glycan scan | NXS/T sequons on chain A scored for surface exposure, SS content, ESM tolerance |
| ProteinMPNN redesign | Chain A sequences generated; 3 top designs validated with ESMFold |
| Salt bridge analysis | Asp/Glu↔Arg/Lys/His contacts within 4 Å on chain A |
| Cavity detection | Internal voids in chain A ranked by burial depth and size |
| Double mutant bridge | **Validated end-to-end** (live): scan → "suggest double mutant combinations" produces the 171→79 candidate funnel, routes/sco­res pairs, and prints the Rich Panel + ChimeraX viz. 42 unit tests pass; stability filter yields the full candidate set (~79 pairs vs the prior collapse to 2); close pairs scored additively when PyRosetta disabled |
| Validate-design fold-preservation (RMSD) axis | **Verified live against real ChimeraX 1.11.1** (2026-06-01). Parses BOTH RMSDs from `RMSD between N pruned atom pairs is X; (across all M pairs: Y)` → **headline = all-pairs Y** (honest overall drift; pruned X flatters by dropping the worst residues), keeps pruned X + counts N/M. **Concentration** (free): gap = Y−X, pruned_fraction = (M−N)/M → descriptor (low/uniform · concentrated · broadly divergent). **Localized**: per-residue Cα deviation via an independent numpy **Kabsch** fit on the local design+reference PDBs (ChimeraX exposes no per-residue RMSD attribute over REST), → top-K deviant residues + the design **coloured by deviation** (blue→red, reusing the CamSol grouped-`color` idiom) live during execute. Live check (1HSG chain A vs B): 99 matched, own-Kabsch all-pairs 0.4 (== matchmaker), deviations 0.07–1.06 Å peaking at the flap loops (res 49/72/44/63/73); 18 colour commands landed, 0 errors. Locked by real-string + Kabsch unit tests |
| ColabFold bridge v1 | **Validated end-to-end** (live, WSL2 GPU): villin HP36 (36 aa, remote MSA, GPU confirmed) → mean pLDDT 82.4, pTM 0.44, ranked PDB + PAE/pLDDT/coverage PNGs + scores JSON parsed by the real worker. **JAX compile cache** measured cold 538s → warm 287s (~47% faster, same-length different sequence; partial — shape-dependent, see §8). 37 unit tests (all mocked). Env: `~/colabfold_env` (Py 3.12, jax 0.5.3, colabfold 1.6.1) |

**PyRosetta Benchmark Run 1 (2026-05-29)** — 11 mutations from ProThermDB:

| Metric | Result |
|--------|--------|
| Sign accuracy | 6/10 = 60% (I64E excluded, no precise experimental value) |
| Within 2.0 kcal/mol | 4/10 = 40% |
| MAE | 3.823 kcal/mol |
| RMSE | 5.492 kcal/mol |
| Pearson r | −0.059 (driven negative by A98V outlier: +14.52 predicted vs −0.5 experimental) |

Protocol suitable for sign prediction on surface-exposed mutations (4/6 correct for non-buried, non-interface positions); not reliable for buried mutations or magnitude accuracy. Without the A98V outlier, r ≈ +0.46 and RMSE ≈ 2.58 kcal/mol. Full analysis in `scripts/rosetta_validation_notes.md`.

### PyRosetta ddG protocol status (current, standalone validation 2026-05-29)

The local PyRosetta path is confirmed **working for ranking** but **not for calibrated absolute magnitudes**:

- **1HSG control panel** (`test_pyrosetta_ddg_controls.py`): buried-charge ≫ surface separation is correct (e.g. L90K +12.1, surface I72R ≈ 0); sign behaviour sensible.
- **T4 lysozyme cross-check** (`test_pyrosetta_ddg_t4lysozyme.py`, 2LZM, 10 literature mutations): **sign accuracy 100%, Pearson r = +0.487** (vs Run 1's −0.059), but **MAE 3.92 / RMSE 5.23** — systematic **over-prediction**, worst on large cavities (L99A +12 vs exp +5; L133A +15 vs +2.7).
- **Aggregation diagnostic** (`test_pyrosetta_aggregation_diag.py`, 5 trajectories × 3 mutations): per-trajectory **spread ≈ 8 kcal/mol** (two-sided noise). Aggregator MAE: **median 2.79 < mean 3.51 < min 3.84** → **MEDIAN is the right aggregator**; MIN is refuted (drags noisy surface mutations to false stabilising values).
- **Convergence-vs-bias diagnostic** (`test_pyrosetta_convergence_diag.py`, completed 2026-05-29): swept relax_cycles 3/5/8 (median of 5 each). Verdict: **convergence-fixable (partial)**. More cycles move every median toward experiment and shrink spread (7.61→5.13 kcal/mol), but at 8+8 only ~half the gap closes — L99A +9.24→+7.30 (exp +5.0), V87M +7.24→+5.37 (exp +1.5; residual cavity-fill bias cycles don't clear), N116D −1.70→+0.64 (converges to ~0). Confirmed design: **median aggregation + more relax cycles for a validation tier** — magnitudes remain approximate, ranking reliable. Full notes in `scripts/rosetta_validation_notes.md`.

Bottom line: use PyRosetta ddG for **ranking + sign** (good), aggregate by **median**. The validation tier (median-of-N at higher cycle counts) roughly **halves** the magnitude error vs single-trajectory while leaving ranking unchanged — **see the confirmed 2LZM panel below**; absolute ddG is now tighter but should still be **disclosed as approximate (~±2.7 kcal/mol RMSE)**.

### Validation-tier 2LZM panel — CONFIRMED (2026-05-30)

The required full re-run is done: the 10-mutation 2LZM panel scored at the validation tier (`validate_ddg`, **N=5 trajectories × 8+8 relax cycles, median**) via `scripts/validate_2lzm_panel.py`. Authoritative numbers in `scripts/validate_2lzm_results.json` (gitignored); full table + conclusion in `scripts/rosetta_validation_notes.md`.

| metric | validation tier (N=5 × 8+8) | 1-traj 2LZM | 1-traj Run 1 |
|--------|----------------------------:|------------:|-------------:|
| Sign accuracy | **90%** (9/10) | 100% | 60% |
| MAE (kcal/mol) | **2.586** | 3.92 | 3.823 |
| RMSE (kcal/mol) | **2.729** | 5.23 | 5.492 |
| Pearson r | **+0.499** | +0.487 | −0.059 |

**Thresholds (revised): r > 0.30 ✅ (+0.499) · RMSE < 4.00 ✅ (2.729) · sign ≥ 60% ✅ (90%) — all PASS.** RMSE/MAE roughly halve vs single-trajectory while r is unchanged (~0.50): the gain is in **magnitude, not ranking**. Residual decomposes into a ~+1.3 kcal/mol systematic over-prediction plus ~2.4 kcal/mol scatter, and is **no longer large-cavity-specific** (errors spread across large-cavity/moderate/surface groups). Lone sign flip: S117V (−1.24 vs exp +0.9), a near-zero mutation. Absolute magnitudes remain approximate but now clear the revised thresholds.

> **NOTE — measured on the stripped-water path, which is the DEFAULT.** These numbers were
> produced with `ROSETTA_STRIP_WATERS=true` (strip). Strip was the default *until 2026-05-30*,
> was briefly switched to preserve (commit 3318bc9), and was **reverted to strip on 2026-05-31**
> as the validated, standard-Rosetta-practice baseline — so these panel numbers again describe
> the **default** path. **Preserve-waters** (`ROSETTA_STRIP_WATERS=False`) is now **opt-in**:
> it re-appends crystallographic HOH and is validated only on buried-water target **Barnase
> T26A** (strip −0.34 wrong-sign → preserve +3.47 correct, exp +1.3; a **+3.80 shift**, 216 HOH
> confirmed reaching the pose). Preserve is a documented Rosetta anti-pattern at the all-static
> level, and its **panel-level behaviour remains unmeasured**; the proper fix is selective
> buried-only / movable waters (Build Queue §9). The panel's surface/cavity mutations are not
> near buried waters, so the opt-in preserve path would be unlikely to change these specific
> metrics — but that is **unconfirmed**.

### Validation scripts (`scripts/`, read-only; require ROSETTA_BACKEND=local)

| Script | Validates |
|--------|-----------|
| `test_pyrosetta_single.py` | Single-mutation probe (I72R/1HSG): dumps worker debug + final ddg/ddg_source; confirmed the local path works |
| `test_pyrosetta_ddg_controls.py` | 1HSG buried→charged / moderate / surface panel — magnitude sanity (buried ≫ surface) |
| `test_pyrosetta_ddg_t4lysozyme.py` | 2LZM vs 10 published experimental ddG (sign/MAE/RMSE/Pearson r vs Benchmark Run 1) |
| `test_pyrosetta_aggregation_diag.py` | 5 trajectories × 3 mutations — which aggregator (min/median/mean) best matches experiment |
| `test_pyrosetta_convergence_diag.py` | 3 levels (3/5/8 relax cycles) × 5 trajectories — is large-cavity bias convergence-fixable or baked in |

Full pipeline validation script: `scripts/validate_full_pipeline.txt`

### Rosetta ddG methodology & known deviations

Context for the systematic over-prediction (panel mean signed error ≈ +1.3 kcal/mol; e.g. L99A +7.95 vs exp +5.0). Three axes, two of them deliberate/known deviations from the canonical protocol:

1. **Basis — actual implementation (modeled on, but deviating from, canonical cartesian_ddg).** ddG is computed by a **manual symmetric-relax protocol** (`rosetta_bridge.py:1449-1464`), **not** the Rosetta `cartesian_ddg` application: clone the cache-relaxed WT → `MutateResidue` → **torsion-space `FastRelax`(mutant)**; **independently `FastRelax` a re-cloned WT**; subtract **full-pose `ref2015` total scores** → `ddG = score(mut) − score(WT_re-relax)`; **median over N trajectories**. It is **MODELED ON** the canonical `cartesian_ddg` / `ref2015_cart` protocol (Park 2016; Frenz 2020) but **DEVIATES**: it uses **plain `ref2015` in TORSION space with FastRelax** (`:1401`, `:1420`) — not `cartesian_ddg`, not `ref2015_cart`, not cartesian minimisation. **Only the multi-trajectory + median noise-smoothing mirrors** the canonical 3–5 round averaging. Consequence: the published **2.94 REU→kcal/mol calibration does NOT apply** (see Calibration item); magnitudes are reported as **raw, uncalibrated REU**. ⚠ The stub docstring (`:1115`) and one `method_note` string (`:1654`) still say "CartesianDDG"/"ref2015_cart" — aspirational labels the running code has never matched; do not take them as the implemented protocol.

2. **Calibration (KNOWN LEVER — AUDIT COMPLETE 2026-05-30; VERDICT: leave uncalibrated).** ref2015 cartesian_ddg REU famously requires a **~2.94 scaling factor → kcal/mol** (Park 2016), and that factor is **protocol-specific**. Audit verdict:
   - The active ddG path is **NOT `cartesian_ddg`**. It is the manual protocol above (`rosetta_bridge.py:1449-1464`): clone WT → mutate → `FastRelax`(mutant) → independently `FastRelax` a re-cloned WT → subtract full-pose `ref2015` total scores; median over trajectories. **`ref2015` TORSION space**, not `ref2015_cart`, not cartesian minimisation.
   - Reported **"kcal/mol" = raw `ref2015` REU delta, with NO conversion** (implicit factor **1.0**; grep confirms no `2.94` / scaling anywhere in `rosetta_bridge.py`, the worker, or `config.py`).
   - **Park's 2.94 does NOT apply** (different protocol *and* score function). Worked check on the 2LZM panel (`scripts/validate_2lzm_results.json`, whose `predicted_median` values are the raw REU deltas): ÷2.94 lowers MAE **2.59 → 1.24** but **over-corrects** — the bias flips negative (mean signed error +1.28 → −0.94) and the large-cavity mutations are wrecked (L99A 7.95→2.71 vs exp 5.0). So magnitudes are **genuinely UNCALIBRATED**, and the "ranking reliable / magnitudes approximate (~±2.7 kcal/mol)" framing used elsewhere is **accurate, not a placeholder**. Scaling is **irrelevant to ranking/sign** (it is a monotonic transform).
   - **DECISION: do NOT add any scaling factor or fitted calibration.** 2.94 over-corrects, and a 10-point empirical fit would overfit. (Recorded explicitly so a future session does not add one.) If calibrated absolute magnitudes ever become a requirement, adopt the **canonical** `cartesian_ddg` (cartesian relax + `ref2015_cart` + averaging + 2.94) wholesale — see Build Queue backlog — rather than patching a factor onto the torsion-space delta.

3. **Water handling (KNOWN DEVIATION FROM BEST PRACTICE).** Established Rosetta practice is to **strip bulk/surface waters** — the implicit Lazaridis–Karplus / `lk_ball` solvent term in `ref2015` already models surface desolvation — and keep **only buried/bridging structural waters**, and **not as static HOH records**: static waters act as immovable spheres and cause **clash-driven over-destabilization**. The supported **movable**-water approach is **Rosetta-ICO/ECO** (Pavlovicz, Park & DiMaio 2020). Our current default (`ROSETTA_STRIP_WATERS=False` → **preserve ALL ~216 crystallographic waters, static**; commit 3318bc9) is a **deliberate deviation**: it fixes the buried-water case (**Barnase T26A validated**: strip −0.34 wrong-sign → preserve +3.47 correct, exp +1.3) but is **expected to degrade non-buried surface/cavity mutations**. **DECISION PENDING:** after the preserve-path 2LZM re-validation (Build Queue §9 item 1), reconsider the default — likely move to **selective buried-only** preservation, or revert the default to **strip** with preservation as opt-in.

---

## 8. Known Issues and Technical Debt

### From `scripts/rosetta_validation_notes.md` (PyRosetta protocol audit)

| Issue | Severity | Detail |
|-------|----------|--------|
| Single relax trajectory | ⚠️ Medium | 1 trajectory per mutation; adds ~1–2 kcal/mol stochastic noise; RMSE 5.49 in benchmark |
| cleanATOM removes crystallographic waters | ⚠️ High | Causes wrong-sign predictions for T26A (Barnase) and G88V (SNase) where buried waters contribute to stability; **one-line fix: preserve HOH records** |
| Cavity-filling mutation overestimation | ⚠️ High | A98V: +14.52 predicted vs −0.5 experimental; single trajectory can't resolve backbone strain for volume-increasing substitutions; fix: detect mutant > WT size, use more relax cycles |
| No large backbone flexibility | ⚠️ Medium | FastRelax can't model local unfolding; proline-insertion destabilisation underestimated |
| Single-chain scoring for oligomers | ⚠️ Low | 1HSG chains A+B both loaded; interface contacts from B on A mutations present but not explicitly managed |

### PyRosetta ddG benchmark failure modes (from run 1)

**Failure mode 1 — Systematic overestimation of destabilisation:**
I88V (+5.264 vs +0.6), L69A (+6.6 vs +2.4), L99A (+7.137 vs +4.0), A98V (+14.52 vs −0.5). Root cause: single trajectory landing in poor local minimum. **Quantified 2026-05-29** (aggregation diagnostic): per-trajectory spread ≈ **8 kcal/mol** (two-sided), and **median** is the best aggregator (MAE 2.79 < mean 3.51 < min 3.84) — MIN is refuted. The convergence diagnostic (2026-05-29) showed more relax cycles also reduce the *bias* (not just noise) — **partially**: at 8+8 cycles ~half the gap closes (L99A +9.24→+7.30, V87M +7.24→+5.37) and spread drops 7.61→5.13 kcal/mol, but magnitudes do not fully calibrate (V87M retains residual bias). Fix path: multi-trajectory **median** aggregation **+ more relax cycles for a validation tier**; absolute magnitudes stay approximate until a full re-run confirms RMSE.

**Failure mode 2 — Wrong sign on buried mutations:**
T26A (−2.437 predicted vs +1.3 exp), G88V (−0.25 vs +2.1), V82A (−0.06 vs +1.75). Root cause: `cleanATOM` removing crystallographic waters that stabilise buried positions in Barnase and SNase.

### Revised benchmark thresholds (updated based on run 1)
Realistic thresholds for the current single-trajectory protocol: `r > 0.30`, `RMSE < 4.00`, `sign_accuracy ≥ 60%`. **DONE 2026-05-31** — `test_rosetta_benchmark.py` re-scoped for honesty: the aggregate correlation test (`test_benchmark_correlation_acceptable`) is now THE gate at these thresholds (with a `MIN_BENCHMARK_ENTRIES=6` minimum-N guard that **skips, not fails**, below the floor); per-mutation tests only **record** to `benchmark_results.json` and assert **sign only** for `|experimental| > 1.5` (`SIGN_ASSERT_MIN_ABS_EXP`), never magnitude (the old ±2.0 kcal/mol tolerance gated on stochastic single-trajectory noise — T26A came back +1.41 vs +7.74 across two live runs). Also fixed the cp1252 `UnicodeEncodeError` (the `✓`/`✗` summary glyph → ASCII `[PASS]`/`[FAIL]` + best-effort UTF-8 stream reconfigure). Module still SKIPS cleanly (all 12 tests) without WSL2/PyRosetta. **Follow-up (also 2026-05-31, commit after 06f6478):** (a) **xfail buckets** — buried §8-failure-mode-2 wrong-sign cases under the default strip-water path (G88V, V82A) are now `@pytest.mark.xfail(strict=False, raises=AssertionError)` so a correct sign on the default path is an expected-failure, not a deterministic RED; T26A stays bucket-1 (record-only) since `|exp|=1.3 ≤ 1.5`; (b) **opt-in gate** — live run now requires `STRUCTUREBOT_RUN_LIVE_BENCHMARK=1` AND WSL2 AND PyRosetta (short-circuit `and`), so a dev box with PyRosetta installed no longer launches the hours-long suite by accident.

### From code review (TODO/FIXME/stub findings)

| Location | Issue |
|----------|-------|
| `rfdiffusion_bridge.py` | Entire module is a documented stub; requires Python 3.9-3.11 environment |
| `rosetta_bridge.py` — `_run_pyrosetta()` | Backend A (`pyrosetta` mode) is a documented stub; WSL2 path (`local` backend) works |
| `rosetta_bridge.py` Ubuntu strings | ✅ FIXED 2026-05-29 — all `Ubuntu-22.04` references in `_run_rosetta_local()` / `backend_status()` corrected to `Ubuntu-24.04`; the stale "documented stub" comment was also corrected |
| `main.py` line ~327 | WSL2 availability message still says `wsl --install -d Ubuntu-22.04` (display string only; not yet updated) |
| `diag.py` | One-off diagnostic script in project root; now gitignored / untracked |
| ColabFold JAX compile cache | Cross-sequence reuse is **partial**, not total: AF2 input feature shapes depend on MSA depth, so two same-length sequences with different MSA depths trigger a partial XLA recompile. Measured 36-aa cold 538s → warm 287s (~47% faster, 250s saved) — meaningful but not the full ~10-min compile elided. Identical-shape re-folds benefit most |

---

## 9. Build Queue

**Completed (2026-05-29 → 05-31):** ✅ Double-mutant live e2e · session auto-restore + `clear session` · ChimeraX auto-reconnect · DynaMut2 single-parser fix + `ddg_source` provenance (2026-05-29). ✅ **Multi-trajectory ddG** (median + MAD spread + `ddg_confidence`, tiered `validate_ddg`) and **2LZM validation** (N=5×8+8: RMSE 2.73, MAE 2.59, r +0.50, sign 90% — all revised thresholds PASS). ✅ **Benchmark thresholds** relaxed to r>0.3 / RMSE<4.0 / sign≥60% + ddG magnitude disclosure broadened (db3add2). ✅ **Crystal-water fix** (`ROSETTA_STRIP_WATERS` + preserve path, 3318bc9) → **water default RESOLVED to strip**, preserve opt-in (7907a20). ✅ **Methodology + calibration audit recorded** in §7 (manual torsion-`ref2015`, NOT cartesian_ddg; magnitudes uncalibrated, no 2.94 scaling; verdict = leave uncalibrated). ✅ **Benchmark-test honesty** (`test_rosetta_benchmark.py`, 2026-05-31): cp1252 `✓`/`✗` crash fixed (ASCII `[PASS]`/`[FAIL]` + UTF-8 reconfigure); per-mutation tests record-only + sign-only-for-`|exp|>1.5`, no magnitude gate; aggregate correlation is now THE gate with a `MIN_BENCHMARK_ENTRIES=6` skip-not-fail guard. Module still skips cleanly (12 tests) without WSL2/PyRosetta; non-benchmark suite green (523). ✅ **Benchmark-honesty follow-up (commit after 06f6478)**: (Fix A) xfail buckets for buried wrong-sign mutations (G88V, V82A) so correct-sign-on-default-path isn't a deterministic RED; (Fix B) `STRUCTUREBOT_RUN_LIVE_BENCHMARK=1` opt-in gate so the hours-long live suite never auto-runs on a PyRosetta-equipped dev box. Verified offline: 523 passed; benchmark module 12 skipped / 0 run. ✅ **ColabFold environment stood up + validated on GPU (2026-05-31, `feat/colabfold-env`)**: isolated WSL2 **Python-3.12** env `~/colabfold_env` (NOT venv310 — that plan is superseded), hermetic CUDA via pip (`jax[cuda12]`), GPU verified incl. **sm_120 (Blackwell RTX 5070 Ti)**, ColabFold 1.6.1 reconciled to **jax/jaxlib 0.5.3** (CUDA-plugin aligned to 0.5.3), end-to-end smoke fold on GPU (villin HP36, pLDDT 82.4) using the **remote MSA server** (no local DBs). RFdiffusion is kept a **SEPARATE future env** (its torch/SE3 stack is its own Blackwell problem — not coupled to ColabFold).

### Immediate next

**✅ DONE — ColabFold bridge v1** (`colabfold_bridge.py`, merged to `main` @ `b1cdb8d`): standalone AF2 folding via the WSL2 `~/colabfold_env`, remote MSA, copies/oligomer + multimer, optional template, total-residue guard, input-hash result cache, **JAX persistent compile cache** (~47% cold-fold speedup), ChimeraX confidence-map viz + matchmaker, 37 tests, live-validated (HP36 pLDDT 82.4). See §3/§7.

**✅ DONE — Validate-design meta-tool** (`feat/validate-design`, NOT yet merged): thin orchestrator in `tool_router._run_validate_design` (NOT a new bridge) fusing ColabFold confidence + matchmaker Cα RMSD + Rosetta `relax_and_score` (new public method) into one **evidence-rich** report (no binary verdict). Honest energy axis: **sanity by default**, **relative only on same-topology** (same chain count AND per-chain length), **DECLINE cross-topology** (no number emitted + reason). Reuses an in-session/cached fold (no re-fold). 24 tests (incl. the pure honesty-decision logic); `relax_and_score` live-validated (HP36 −88.4 REU). Stored in `session_state.validate_design_results`. See the design note + §3. **Intent is HEAVY-only** (re-scoped): fires ONLY on explicit qualifiers — *full / high-accuracy / high-confidence / thorough* validation, "`<qualifier>` validate", or "validate … with colabfold/alphafold". **Bare** "validate design" / "check this design" deliberately fall through to the light **mpnn_esmfold / ESMFold** fast screen (their pre-meta-tool behaviour). **RMSD axis upgraded**: headline is now **all-pairs** Cα RMSD (honest overall drift) with pruned/core kept; a **concentration** read (all-pairs−core gap + pruned fraction → low-uniform/concentrated/broadly-divergent) and **localization** (per-residue Kabsch deviation → top-K residues + design coloured blue→red by deviation). See §7.

| Priority | Item | Rationale |
|----------|------|-----------|
| 1 | **RFdiffusion activation** (stub at `rfdiffusion_bridge.py`) | De novo backbone generation; completes the design loop (RFdiffusion → ProteinMPNN → ColabFold → validate-design). **SEPARATE env** (do NOT couple to ColabFold): its own WSL2 env, torch/SE3-Transformer stack, ~20 GB weights; Blackwell sm_120 torch support is its own problem to solve at build time |
| 2 | **Validate-design refinements** | (a) **MPNN-result sequence auto-pull** so "validate the top design" needs no paste; (b) the **polished interpretive text summary** (deferred from v1 — currently evidence-only); (c) surface per-residue pLDDT/PAE plots in the report panel |

### Validate-design meta-tool — agreed design (DONE — kept for reference)

A **thin orchestrator** (a `tool_router` dispatch + result-assembly method, **not** a new bridge — it composes existing bridges). Inputs: a designed sequence (or a reused in-session ColabFold result) + a reference structure. Three evidence sources:

1. **ColabFold** — fold confidence (mean/per-residue pLDDT, pTM/ipTM, PAE). **Reuse an existing in-session `colabfold_results` fold instead of re-folding** when one matches; only fold if absent.
2. **ChimeraX matchmaker** — superpose the predicted model onto the reference; report RMSD (+ aligned-residue count).
3. **Rosetta** — FastRelax + `ref2015` total energy of the predicted model.

**Energy-reporting rule (important):**
- **Default = clash-free / low-energy SANITY check** — report the relaxed total energy as a "structure is physically reasonable / not clashing" signal, NOT a stability claim.
- **Like-for-like RELATIVE score** (design vs reference Δenergy) **only when topologies match** (matchmaker RMSD low / same fold) **or on explicit request**.
- **DECLINE a cross-topology absolute stability number** and **state why** (ref2015 deltas across different folds are not comparable; our ddG path is uncalibrated — see §7). Surface the sanity signal instead.

**Reports are EVIDENCE-RICH:** surface ALL underlying outputs (pLDDT map, PAE, RMSD, per-term energies, the per-source warnings) rather than collapsing to a verdict; the **interpretive natural-language summary is DEFERRED** to a later iteration. Mirrors the existing error-first / `ToolStepResult` + viz idiom; no new venv or external dependency.

### Backlog

| Item | Rationale |
|------|-----------|
| **Selective / movable buried-only water handling** (Rosetta-ECO) | Real fix for the water trade-off: keep **only buried/bridging** waters (the `ref2015` `lk_ball` implicit solvent already models surface desolvation) and treat them as **movable** (Rosetta-ICO/ECO; Pavlovicz, Park & DiMaio 2020) — captures the T26A buried-water benefit without the surface-mutation penalty that preserve-all-static causes. Would then warrant a selective-path 2LZM re-validation. See §7 Water handling |
| **Adopt canonical cartesian_ddg for calibrated magnitudes** | **Only if** absolute ΔΔG magnitudes ever become a requirement: cartesian relax + `ref2015_cart` + multi-round averaging + the published ~2.94 REU→kcal/mol factor. **Ranking/sign is unaffected by scaling**, so this is unnecessary for screening/ranking. Do **NOT** instead patch 2.94 onto the current torsion-space delta (it over-corrects). See §7 Calibration |
| **Conditional preserve-path 2LZM re-validation** | Required **only if** preserve is promoted to default or made selective. The committed panel (RMSE 2.73, r +0.50, sign 90%) describes the strip default, so no re-run is needed for current behaviour. ~4.5 h via `scripts/validate_2lzm_panel.py` (preserve uses a fresh `_wat` relax-cache namespace) |
| **Parallelize independent ddG work units** (process pool, `ROSETTA_MAX_WORKERS`) | *Drafted, not built.* Independent per-mutation / per-trajectory units run serially today; a process pool is a **pure speedup with identical results** (no scientific change). Add a `ROSETTA_MAX_WORKERS` config knob |
| **Mid-execution ESC / cancellation** | Long-running tools (PyRosetta ~15 min, ColabFold ~5 min) have no cancellation; needs a background-thread pattern |
| **Double mutant — real PyRosetta close-pair scoring** | Close pairs (<4 Å) get an *additive* estimate when `run_pyrosetta=False` (can't capture epistasis); real close-pair scoring is now worthwhile since the ddG protocol is settled |
| **GlycosuitDB lookup** | Expression-system glycan characterisation for engineered sequons (CHO, HEK293, E. coli, …) |
| **Remote/cloud or alternate-GPU ColabFold handoff** | For folds exceeding `COLABFOLD_MAX_TOTAL_RESIDUES` (e.g. large homo-tetramers that OOM the laptop GPU): hand off to a remote/cloud GPU or a larger local card. The total-residue budget guard is the intended trigger point; the exact boundary is to be found empirically via a later multimer stress test, then the budget recalibrated |
| **Warm persistent ColabFold worker** | The JAX compile cache makes cross-fold recompiles *partial* (shapes differ with MSA depth); a long-lived warm worker process (one JAX session serving multiple folds) would **fully elide** recompile within a session. Bigger lift than the on-disk cache; superfor repeated interactive folds |
| **ColabFold batch top-N folding** | Fold several candidate sequences in one invocation (e.g. top-N ProteinMPNN designs) rather than one predict() per call |
| **ColabFold MPNN-result sequence auto-pull** | v1 takes explicit sequences only; auto-pull the top design sequences from `session.proteinmpnn_results`/`mpnn_esmfold_results` so "fold the top design" needs no paste |
| **ColabFold `single_sequence` MSA mode** | Skip the remote MSA for speed on sequences where an MSA adds little (e.g. de novo designs); a `COLABFOLD_MSA_MODE=single_sequence` path |
| **ColabFold amber relax** (`--amber`) | Optional post-prediction Amber relaxation of the ranked model for cleaner geometry |

---

## 10. External Dependencies

### Main venv (`venv/`, Python 3.14)

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `anthropic` | `>=0.40.0` | Claude API (translator, prompt caching) |
| `requests` | `>=2.31.0` | DynaMut2 API, RCSB API, ChimeraX REST, ESM Atlas fallback |
| `rich` | `>=13.7.0` | Console REPL, tables, panels |
| `biopython` | [verify] | PDB parsing in disulfide/proline/cavity/salt-bridge/structural_utils |
| `freesasa` | [verify] | SASA computation in salt_bridge_bridge.py (optional — degrades gracefully) |

### venv312 (`venv312/`, Python 3.12)

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | `2.11.0+cu128` | GPU tensor ops (RTX 5070 Ti, Blackwell sm_120) |
| `transformers` | [verify] | ESMFold `facebook/esmfold_v1` via HuggingFace |
| `fair-esm` / `transformers` | [verify] | ESM-2 inference (either package works) |

### WSL2 (Ubuntu-24.04)

| Dependency | Location | Purpose |
|------------|----------|---------|
| PyRosetta | `/home/andre/pyrosetta_env/` | ddG calculations; Rosetta Commons **academic license required** |
| Python venv | `/home/andre/pyrosetta_env/bin/python` | PyRosetta interpreter |

**License note:** PyRosetta requires a free academic license from rosettacommons.org. Commercial use not permitted.

### External Services

| Service | URL | Auth | Usage |
|---------|-----|------|-------|
| Anthropic API | `api.anthropic.com` | `ANTHROPIC_API_KEY` | Translation (every request) |
| DynaMut2 (single) | `biosig.lab.uq.edu.au/dynamut2/api/prediction_single` | None | Single-point ddG scoring (default backend) |
| DynaMut2 (multi) | `biosig.lab.uq.edu.au/dynamut2/api/prediction_mm` | None | Double mutant ddG via `prediction_mm` endpoint |
| RCSB PDB | `data.rcsb.org/rest/v1/` | None | Assembly metadata, chain info |
| NetNGlyc 1.0 | `services.healthtech.dtu.dk` | None | OST recognition prediction |
| ESM Atlas | `esmatlas.com/api/fold` | None | ESMFold fallback (when local fails) |
| HuggingFace | `huggingface.co` | None | ESMFold model weights (`facebook/esmfold_v1`) |

### Not pip-installable (must clone and configure)

| Tool | Setup | Config |
|------|-------|--------|
| ProteinMPNN | `git clone https://github.com/dauparas/ProteinMPNN` | `PROTEINMPNN_DIR=C:\Users\andre\documents\structurebot\ProteinMPNN` |
| RFdiffusion | `git clone https://github.com/RosettaCommons/RFdiffusion` + weights | `RFDIFFUSION_DIR=<path>` (not yet set) |

### ColabFold env (INSTALLED 2026-05-31 — WSL2, GPU-validated)

**Decision change:** the old "venv310 (Python 3.10), shared with RFdiffusion" plan is **superseded**. ColabFold now lives in a dedicated **WSL2 Python-3.12** venv with **hermetic CUDA via pip**; RFdiffusion is a **separate** future env (decoupled).

| Package | Env | Version (exact, installed) | Purpose |
|---------|-----|----------------------------|---------|
| `colabfold` | `~/colabfold_env` (WSL2, Py 3.12.3) | **1.6.1** | AF2-quality folding; **remote MSA server** (no local MMseqs2 DBs) |
| `alphafold-colabfold` | `~/colabfold_env` | **2.3.13** | ColabFold's AlphaFold fork (pins `jax<0.6.0`) |
| `dm-haiku` | `~/colabfold_env` | **0.0.16** | AlphaFold NN modules |
| `jax` / `jaxlib` | `~/colabfold_env` | **0.5.3 / 0.5.3** | ColabFold runtime; the ColabFold-compatible jax that ALSO supports sm_120 |
| `jax-cuda12-plugin` / `-pjrt` | `~/colabfold_env` | **0.5.3 / 0.5.3** | hermetic CUDA backend (aligned to jaxlib 0.5.3) |
| NVIDIA CUDA wheels | `~/colabfold_env` | CUDA **12.9** (`nvidia-cuda-nvcc-cu12` 12.9.86), cuDNN **9.23.0.39** | pip-pulled hermetic CUDA — does NOT touch system CUDA or `pyrosetta_env` |

**GPU validated:** `jax.devices()` → `[CudaDevice(id=0)]`, `device_kind="NVIDIA GeForce RTX 5070 Ti Laptop GPU"`, platform `gpu`; 2048³ matmul runs with **no sm_120 ptxas error**. WSL2 driver **596.49 / CUDA 13.2** (`nvidia-smi`). **Reconciliation note:** `jax[cuda12]` latest is 0.10.1 (works on sm_120), but ColabFold caps `jax<0.6.0` → resolves to **0.5.3**; the cuda12 plugin must be pinned to match (`pip install "jax[cuda12]==0.5.3"`) or the PJRT C-API mismatches and aborts. 0.5.3 supports sm_120, so a clean intersection exists.

### Planned — separate future env

| Tool | Env | Purpose |
|------|-----|---------|
| RFdiffusion | **own** WSL2 env (NOT shared with ColabFold) | De novo backbone generation; torch/SE3-Transformer stack; Blackwell sm_120 torch support is its own build-time problem |

Note: JAX (ColabFold) and PyTorch (RFdiffusion, `venv312`) must remain in separate envs due to CUDA runtime conflicts; `pyrosetta_env` is untouched by all of the above.

---

## 11. File Locations Reference

| Resource | Path |
|----------|------|
| Project root | `C:\Users\andre\documents\structurebot\` |
| Main venv | `<root>\venv\` |
| venv312 (GPU) | `<root>\venv312\` |
| venv312 Python | `<root>\venv312\Scripts\python.exe` |
| WSL2 PyRosetta Python | `/home/andre/pyrosetta_env/bin/python` |
| WSL2 ColabFold env | `/home/andre/colabfold_env/` (Python 3.12, JAX cuda12 hermetic) |
| WSL2 ColabFold Python | `/home/andre/colabfold_env/bin/python` (= `wsl_bridge.COLABFOLD_PYTHON`) |
| ColabFold weights cache | `/home/andre/.cache/colabfold/` (~3.47 GB `alphafold2_ptm`; downloaded on first fold) |
| ColabFold smoke-test dir | `/home/andre/colabfold_test/` (`in.fasta`, `out/`, `run.log`) |
| ColabFold bridge cache | `<root>\cache\colabfold\colabfold_{hash}\` (ranked.pdb, pae/plddt/coverage.png, result.json) |
| ChimeraX executable | `C:\Users\andre\documents\ChimeraX 1.11.1\bin\ChimeraX.exe` |
| ESM disk cache | `<root>\cache\esm_{hash}.json` |
| Rosetta relax cache | `<root>\cache\rosetta_relaxed\` |
| PDB download cache | `<root>\cache\{PDBID}.pdb` |
| HuggingFace model cache | `~/.cache/huggingface/hub/models--facebook--esmfold_v1/` |
| Session JSONL logs | `<root>\logs\session_YYYYMMDD_HHMMSS.jsonl` |
| Named sessions | `<root>\sessions\{name}.cxs` + `{name}.json` |
| Live session | `<root>\session.json` |
| Worker debug dump | `%TEMP%\structurebot_worker_debug.py` (written each PyRosetta run) |
| Benchmark results | `<root>\scripts\benchmark_results.json` |
| Benchmark run log | `<root>\scripts\benchmark_run.log` |
| PyRosetta audit + benchmark | `<root>\scripts\rosetta_validation_notes.md` |
| Full pipeline script | `<root>\scripts\validate_full_pipeline.txt` |
| Context update script | `<root>\scripts\update_context.sh` |
| PyRosetta validation scripts | `<root>\scripts\test_pyrosetta_*.py` (single / ddg_controls / ddg_t4lysozyme / aggregation_diag / convergence_diag — see §7) |
| ProteinMPNN repo | `C:\Users\andre\documents\structurebot\ProteinMPNN\` |

### Repo hygiene

A `.gitignore` is now in place. **Untracked** (intentionally not committed): `cache/` (PDBs, ESM cache, `rosetta_relaxed/`), `logs/`, `sessions/`, `session.json`, `ProteinMPNN/`, the venvs (`venv/`, `venv312/`, `venv310/`), `__pycache__/`, `*.pyc`, `*.pdb`, `*.jsonl`, `.env` / `.env.local`, and `diag.py`. These were untracked from the repo in this session's hygiene commits. Source modules, tests, and `scripts/*.py` remain tracked.

---

## 12. Session Continuity Notes

**For starting a new co-pilot conversation:**

- **Test count (2026-05-31)**: 535 collected, **523 passing** (12 benchmark tests skip without WSL2+PyRosetta). Non-benchmark suite runs in ~36s (`pytest tests/ --ignore=tests/test_rosetta_benchmark.py -q`). Only pre-existing BioPython `RuntimeWarning`s; no unknown-marker warnings.
- **Last built/validated this session**: (1) double-mutant bridge validated **live end-to-end**; (2) **session auto-restore** on startup + `clear session` command (`session_state.try_load`, `main._maybe_restore_session`); (3) **ChimeraX auto-reconnect** on dropped REST connection; (4) **DynaMut2 single-parser fix** (status-based API) + `ddg_source` provenance + `relax_cycles` param + Ubuntu-24.04 corrections in `rosetta_bridge.py`; (5) **PyRosetta ddG standalone validation** (1HSG controls, T4/2LZM cross-check, aggregation + convergence diagnostics). All session work is committed; working tree clean.
- **Convergence diagnostic COMPLETE** (2026-05-29): large-cavity bias is **partially convergence-fixable** — more relax cycles move every median toward experiment and shrink spread (7.61→5.13 kcal/mol), but at 8+8 only ~half the gap closes. `rosetta_bridge.py` is now **free** (no PyRosetta running).
- **Multi-trajectory ddG + 2LZM confirmation COMPLETE** (2026-05-30): the tiered `validate_ddg` shipped and the full 2LZM panel was scored at the validation tier (N=5 × 8+8, median): **RMSE 2.73, MAE 2.59, r +0.50, sign 90% — all revised thresholds (r>0.3, RMSE<4.0, sign>=60%) PASS**. RMSE/MAE roughly halve vs single-trajectory while r is unchanged (gain in magnitude, not ranking). Build-queue item 1 is fully closed.
- **Crystal-water fix + benchmark thresholds DONE (2026-05-30)**: `ROSETTA_STRIP_WATERS` flag + preserve code path (commit 3318bc9) re-append crystallographic HOH; validated on Barnase T26A (strip −0.34 wrong-sign → preserve +3.47 correct, +3.80 shift, 216 HOH). Benchmark thresholds relaxed to r>0.3/RMSE<4.0/sign>=60% (commit db3add2).
- **Water-handling default RESOLVED (2026-05-31)**: `ROSETTA_STRIP_WATERS` default **reverted to `True` (strip)** — the validated, standard-Rosetta-practice baseline; preserve-all-static is a documented anti-pattern and only the strip path is panel-validated. **Preserve is opt-in** (`ROSETTA_STRIP_WATERS=false`), validated only on buried-water T26A. The committed 2LZM panel numbers (RMSE 2.73, r +0.50, sign 90%) therefore again describe the **default** path. The real future fix is **selective buried-only / movable** waters (Build Queue §9); preserve-path 2LZM re-validation is downgraded to *conditional* (only if preserve is promoted to default or made selective).
- **Benchmark-test honesty DONE (2026-05-31)**: `test_rosetta_benchmark.py` re-scoped (Build Queue §9 item closed). (a) cp1252 `UnicodeEncodeError` fixed — the `✓`/`✗` summary glyph is now ASCII `[PASS]`/`[FAIL]`, plus a best-effort UTF-8 stream reconfigure in `compute_benchmark_correlation` (same pattern as `validate_2lzm_panel.py`). (b) per-mutation tests no longer gate on noise: the ±2.0 kcal/mol magnitude asserts are **gone**, sign is asserted **only** for `|experimental| > 1.5` (`SIGN_ASSERT_MIN_ABS_EXP` + `_check_sign` helper) and near-neutral mutations are **recorded but not asserted**; every per-mutation test still writes to `benchmark_results.json`. (c) `test_benchmark_correlation_acceptable` is now THE gate (r>0.3 / RMSE<4.0 / sign≥60%) with a `MIN_BENCHMARK_ENTRIES=6` minimum-N guard that **skips, not fails**, below the floor. Verified: non-benchmark suite **523 passed**; benchmark module collects 12 + **skips all 12 cleanly** when WSL2/PyRosetta absent (simulated by forcing `wsl_bridge._WSL_AVAILABLE_CACHE=False`); cp1252 + logic exercised offline with synthetic data. **Did NOT run the hours-long live benchmark** (on this dev machine PyRosetta IS available, so a bare `pytest tests/test_rosetta_benchmark.py` would execute it live — guard accordingly). **Immediate next is now features**: ColabFold → RFdiffusion. *(Update: ColabFold env now stood up — see the ColabFold env bullet below; the `venv310` plan was superseded by a WSL2 Py-3.12 env.)*
- **Benchmark-honesty follow-up DONE (2026-05-31, recovered after a machine crash; committed after 06f6478)**: two further `test_rosetta_benchmark.py` fixes. **Fix A** — buried §8-failure-mode-2 wrong-sign mutations under the default strip-water path (**G88V, V82A**) are `xfail(strict=False, raises=AssertionError)`; T26A is bucket-1 (record-only, `|exp|=1.3`). **Fix B** — `STRUCTUREBOT_RUN_LIVE_BENCHMARK=1` opt-in gate (`run live iff opt-in AND WSL2 AND PyRosetta`, short-circuit) so the hours-long live suite never auto-fires on this PyRosetta-equipped dev box. Verified offline: non-benchmark **523 passed**; `pytest tests/test_rosetta_benchmark.py` → **12 skipped, 0 run, 0.04s**. **To run the live benchmark you must now explicitly set `STRUCTUREBOT_RUN_LIVE_BENCHMARK=1`** (a bare invocation no longer runs it).
- **Throwaway diagnostics (safe to delete, never commit)**: `ddg_audit.md` and `validate_2lzm_run.log` in the repo root are one-off diagnostic scratch from this session — delete freely. `scripts/validate_2lzm_results.json` is **validation data** (the authoritative 2LZM panel numbers; gitignored) — **keep it**. None of these three should be committed.
- **PyRosetta ddG status**: works for **ranking + sign** (T4/2LZM: r=+0.487 single-traj, +0.499 validation tier; sign 90–100%) and the validation tier now **roughly halves magnitude error** (RMSE 5.2→2.73). Aggregate by **median** (min refuted). Absolute ddG is tighter but still **approximate (~±2.7 kcal/mol)**.
- **ColabFold ENVIRONMENT DONE (2026-05-31, `feat/colabfold-env`)** — env stand-up + GPU validation only; **the bridge (`colabfold_bridge.py`) is the NEXT task, deliberately not built here**. Isolated WSL2 **Python-3.12** venv `~/colabfold_env` (the old "venv310 shared with RFdiffusion" plan is **superseded**). Hermetic CUDA via pip. **Exact installed versions**: python **3.12.3**, colabfold **1.6.1**, alphafold-colabfold **2.3.13**, dm-haiku **0.0.16**, jax **0.5.3**, jaxlib **0.5.3**, jax-cuda12-plugin/pjrt **0.5.3**, nvidia-cuda-nvcc-cu12 **12.9.86**, cuDNN **9.23.0.39**. WSL2 driver **596.49 / CUDA 13.2**, RTX 5070 Ti. **GPU+sm_120 verified** (2048³ matmul, no ptxas error). **Reconciliation gotcha**: `jax[cuda12]` latest (0.10.1) works on sm_120 but ColabFold caps `jax<0.6.0` → 0.5.3; you MUST realign the cuda plugin (`pip install "jax[cuda12]==0.5.3"`) or the PJRT C-API aborts. **Smoke fold** (villin HP36, remote MSA server, GPU, 1 model/1 recycle): **rank_001 pLDDT 82.4, pTM 0.44**, output `~/colabfold_test/out/`, wall 13:59 (mostly one-time 3.47 GB weight download + first sm_120 XLA compile). **RFdiffusion stays a SEPARATE env** — do not couple. `pyrosetta_env` untouched.
- **ColabFold BRIDGE v1 DONE (2026-05-31, `feat/colabfold-bridge`)** — standalone `colabfold_bridge.py` (Option A) wraps the env from `feat/colabfold-env`. `ColabFoldBridge.predict(sequence, copies, template, num_models, num_recycle, quick)` runs an f-string worker via `wsl_bridge.COLABFOLD_PYTHON` (remote MSA, no local DBs), returns ranked PDB + mean/per-residue pLDDT + PAE + pTM(/ipTM) + PNG paths, **input-hash cache** `cache/colabfold_{hash}/`. **Total-residue guard** `COLABFOLD_MAX_TOTAL_RESIDUES` (default 1500): over budget → no launch + clear OOM-risk message; runtime CUDA-OOM also caught. Router: explicit intent (`colabfold`/`alphafold`/"fold … as a `<oligomer>`"/template-fold) parses copies+template+quick+sequence, dispatches `_run_colabfold`, stores trimmed result in `session_state.colabfold_results`. **Viz**: open ranked PDB → `color byattribute bfactor … palette alphafold` (native pLDDT) → `sequence chain` → optional `matchmaker` vs compare_to (default = loaded primary model); model id from `session.next_model_id()` (matches main's open tracking); PAE/pLDDT/coverage PNGs auto-open in the OS viewer. New `CHIMERAX_SHOW_SEQUENCE_ON_OPEN` flag opens the Sequence Viewer for ALL structure opens (chimerax_bridge.run_commands, fire-and-forget). `_ElapsedTicker` extended with an approximate ETA. **Tests: 33** in `tests/test_colabfold_bridge.py` (all mocked); full non-benchmark suite **556 passed + 1 skipped**. **Optional live monomer fold RAN** through the real bridge (opt-in `STRUCTUREBOT_RUN_LIVE_COLABFOLD=1`): villin HP36, GPU confirmed, **mean pLDDT 82.4, pTM 0.44**, 616 s — validated the worker's real ColabFold-1.6.1 output-dir globs + scores-JSON parsing. **DEFERRED** (next): fused validate-design meta-tool, MPNN-result sequence auto-pull, batch top-N, amber relax, single_sequence, generalized ETA. **Multimer caveat**: budget guard provisional; large homo-tetramers expected to OOM → remote/alternate-GPU handoff is backlog.
- **Active configuration flags**:
  - `ROSETTA_BACKEND=local` in `.env.local` — PyRosetta via WSL2
  - `ESM_USE_VENV312=auto` — GPU inference if CUDA smoke-test passes
  - `ESMFOLD_FORCE_COLD_TIMEOUT=False` — correct, do not change
- **API key**: Set in `.env.local` as `ANTHROPIC_API_KEY` — do not commit this file.
- **No `venv310`** — that plan is superseded. ColabFold uses the WSL2 `~/colabfold_env` (above). RFdiffusion will get its own separate WSL2 env when activated.

---

## 13. Changelog

| Date | Tests | What changed |
|------|-------|-------------|
| 2026-06-01 | 606 (592✓/14s) | feat(validate-design): **upgrade the fold-preservation (RMSD) axis** (`feat/validate-design`; RMSD axis only — fold-confidence/energy unchanged). (1) **Headline = all-pairs** Cα RMSD (honest overall drift) instead of pruned — pruned drops the worst-fitting residues by construction and flatters agreement; pruned/core RMSD + pair counts N/M still captured. `_parse_matchmaker_rmsds` now captures BOTH RMSDs + both counts, robust to a missing "(across all M pairs: Y)" clause (falls back to all-pairs = pruned); the verified pruned capture is not regressed. (2) **Concentration indicator** (free, from the parsed numbers): gap = all-pairs−pruned, pruned_fraction = (M−N)/M → descriptive read (low/uniform · CONCENTRATED, i.e. a big gap from few pruned residues · broadly divergent) — no threshold pass/fail. (3) **Localization**: per-residue Cα deviation via an independent numpy **Kabsch** superposition on the local design+reference PDBs (ChimeraX exposes no per-residue RMSD attribute over REST), → **top-K deviant residues** + the design **coloured by deviation** (blue→red buckets, reusing the CamSol grouped-`color` idiom) issued live during execute on the opened/superposed design model. All surfaced in the evidence-rich report + `session_state.validate_design_results`; still no binary verdict. **Tests**: `test_validate_design.py` 27→34 — regression test updated to assert all-pairs headline (pruned still captured); new parse/fallback, concentration math+descriptor, top-K, Kabsch-localizes-displaced-residue, colour-command tests. Column-exact PDB test fixture so the Cα parser reads coords. Non-benchmark suite **592 passed + 2 skipped**. **Live-verified** the new per-residue mechanism (ChimeraX up): 1HSG chain A vs B → 99 matched, own-Kabsch all-pairs 0.4 (== matchmaker), deviations 0.07–1.06 Å peaking at the flap loops, 18 colour commands landed with 0 errors. §6/§7/§9 updated. NOT merged. |
| 2026-05-31 | 596 (582✓/14s) | refactor(validate-design): **re-scope the meta-tool intent to HEAVY-only** (`feat/validate-design`). Bare "validate design" / "validate this design" / "evaluate design" / "check this design" were stealing the request from the existing light **mpnn_esmfold / ESMFold** fast screen; they now revert to it (pre-meta-tool behaviour). The ColabFold+Rosetta meta-tool fires ONLY on explicit qualifiers — `_VALIDATE_DESIGN_KEYWORDS` = *full validation / full design validation / high-accuracy|high-confidence validation / `<qualifier>` validate / thoroughly validate / thorough validation* — plus the compound "validate … with colabfold/alphafold" (`_detect_validate_design_intent`). Containment handled by route() order (validate_design checked before the bare-match fall-through): "high-accuracy validate design" → meta-tool, bare "validate design" → mpnn_esmfold. Also guarded the colabfold override with `"validate_design" not in tools_needed` so "validate … with colabfold" isn't re-grabbed by the bare colabfold matcher. Tests: original `test_validate_design_routes_to_mpnn_esmfold` restored (bare → mpnn_esmfold); meta-tool routing test uses an explicit trigger; new anti-collision test (bare ↛ meta-tool; explicit ↛ mpnn_esmfold/colabfold). Non-benchmark suite **582 passed + 2 skipped**. |
| 2026-05-31 | 595 (581✓/14s) | feat(validate-design): **validate-design meta-tool** (`feat/validate-design`, NOT merged). A thin orchestrator (`tool_router._run_validate_design` — NOT a new bridge) fusing three evidence axes into one EVIDENCE-RICH report (no binary verdict): (1) **fold confidence** — reuses an in-session/cached ColabFold result (only folds if absent — guardrail), surfaces mean/per-residue pLDDT, PAE, pTM/ipTM; (2) **fold preservation** — ChimeraX `matchmaker` Cα RMSD vs an RMSD reference (default = loaded primary model; overridable to a named PDB + chain — cross-topology OK here, e.g. monomer vs chain A of a WT dimer); (3) **folding energy** — new `rosetta_bridge.relax_and_score(pdb)` (FastRelax + `ref2015` total/per-residue/`fa_rep` on an arbitrary structure, NO mutation; reuses existing building blocks, **does NOT touch the ddG path**). **Honest energy axis by construction**: **sanity** signal by default (total REU + per-residue density + clash flag, framed as fold-plausibility NOT stability); **relative** ΔREU ONLY when an explicit energy reference is given AND topologies match (same chain count AND per-chain length), labelled ranking-reliable/magnitude-approximate (~±2.7, §7); **DECLINE** cross-topology — emits NO relative number and states why. Router: new `validate_design` intent + dispatch + icon, placed FIRST in `route()` so it wins the `mpnn_esmfold` "validate design" keyword collision (that path still reachable via "fold design"/"validate sequence"; one existing test renamed accordingly) and doesn't grab bare `colabfold`. New `session_state.validate_design_results` (persisted); `main._ElapsedTicker` long-tools += validate_design. **Tests**: +`tests/test_validate_design.py` (23: pure sanity/relative/**decline-cross-topology** logic, topology extraction, fold reuse-no-refold, RMSD + relax-score parse, report assembly + storage, degradation/flags, route precedence). Non-benchmark suite **581 passed + 2 skipped**; benchmark still 12-skips. **Optional live e2e RAN** (opt-in): reused the cached HP36 fold + real `relax_and_score` → −88.4 REU/36 res (validates the new worker's real-output parse; matchmaker RMSD parse unit-tested only — ChimeraX not driven live). DEFERRED to next: MPNN auto-pull, polished interpretive text summary. §3/§6/§9 updated. |
| 2026-05-31 | 572 (559✓/13s) | chore(release)+docs: **ColabFold feature stack merged to `main` + full PROJECT_CONTEXT resync**. Consolidates this session's ColabFold work (each step has its own row below): (1) **env stand-up** — isolated WSL2 Py-3.12 `~/colabfold_env`, hermetic CUDA, jax 0.5.3 pinned for sm_120, GPU-validated (`3d03f39`); (2) **bridge v1** — standalone `colabfold_bridge.py` + router/viz/config/tests, remote MSA, residue guard, result cache, ChimeraX confidence-map viz + matchmaker, sequence-on-open flag (`07fe737`); (3) **JAX persistent compile cache** — ~47% cold-fold speedup, partial/shape-dependent (`b1cdb8d`). The linear stack (`fix/double-mutant-ddg-filter` → `feat/colabfold-env` → `feat/colabfold-bridge`) was **fast-forward-merged to `main`** (no squash, full history preserved) and the three feature branches deleted local+origin. Real suite at resync: **572 collected / 559 passing / 13 skipped** (12 benchmark + 1 opt-in live ColabFold). Doc-only resync of the meta header + §2/§3/§6/§7/§9/§10/§11/§12 code-derived sections; §7/§8 ddG/water/calibration/2LZM analytical text and all prior §13 rows preserved verbatim. **Next: validate-design meta-tool** (thin orchestrator — ColabFold confidence + matchmaker RMSD + Rosetta sanity energy; §9). |
| 2026-05-31 | 559 | perf(colabfold): **JAX persistent compilation cache** in the ColabFold worker. The bridge spawns a fresh WSL worker per fold, so every fold previously paid the full ~10-min XLA compile and the result cache only saved IDENTICAL re-folds. The worker now sets `JAX_COMPILATION_CACHE_DIR` (+ `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1`, version-robust) on the `colabfold_batch` process — a WSL2 ext4 dir (`COLABFOLD_JAX_COMPILE_CACHE_DIR`, default `~/.cache/colabfold_jax_compile`, NOT under /mnt/c) created if absent — so XLA reuses compiled executables across processes for matching input shapes. Env var set BEFORE the subprocess (child inherits it). **Live measured** (opt-in, two DIFFERENT 36-aa monomers, fresh processes, to isolate the compile cache from the result cache): seq1 **cold 538s** → seq2 same-length **warm 287s**, **Δ 250s (~47% faster)**. Partial (not the full compile elided) because AF2 feature shapes depend on MSA depth, which differs per sequence → partial recompile (recorded honestly in §8). Tests: +3 in `tests/test_colabfold_bridge.py` (worker sets the cache env before the subprocess; omitted when empty; predict passes the config value) → **559 passed + 1 skipped**. §3/§8/§13 updated. |
| 2026-05-31 | 556 | feat(colabfold): **ColabFold bridge v1 (standalone, Option A)** — `colabfold_bridge.py` + router/viz/config/tests (`feat/colabfold-bridge`). `ColabFoldBridge.predict()` runs an f-string worker via `wsl_bridge.COLABFOLD_PYTHON` in WSL2 (remote MSA, no local DBs; mirrors `rosetta_bridge._run_rosetta_local`): writes FASTA, runs `colabfold_batch`, parses the output dir, returns ranked PDB + mean/per-residue pLDDT + PAE + pTM(/ipTM) + PNG paths; **input-hash cache** `cache/colabfold_{hash}/`; **total-residue guard** `COLABFOLD_MAX_TOTAL_RESIDUES` (pre-launch refusal + runtime CUDA-OOM catch → clear message, never a raw traceback); copies→colon-joined homo-oligomer (multimer); optional custom template; `quick` 1/1 preset; rough ETA. **Router**: explicit `colabfold`/`alphafold`/"fold … as `<oligomer>`"/template-fold intent (won't swallow or be swallowed by esmfold/mpnn_esmfold), `_run_colabfold` dispatch, `session_state.colabfold_results` (persisted), icon 🧬🔮. **Viz**: open ranked PDB → native AlphaFold pLDDT palette (`color byattribute bfactor … palette alphafold`) → Sequence Viewer → optional `matchmaker` RMSD vs compare_to (default loaded primary), PNG auto-open; model id via `session.next_model_id()`. New `CHIMERAX_SHOW_SEQUENCE_ON_OPEN` (default on) opens the Sequence Viewer for ALL opens (chimerax_bridge.run_commands, fire-and-forget, contract-preserving). `wsl_bridge.run_python_script` gained a backward-compatible `python_bin` param + `check_colabfold()`; `_ElapsedTicker` gained an approximate `eta_s`. **Tests**: +`tests/test_colabfold_bridge.py` (33, all WSL/ChimeraX mocked — no CI fold); non-benchmark suite **556 passed + 1 skipped**; benchmark module still skips cleanly (12). **Optional live monomer fold RAN** end-to-end through the real bridge (opt-in): villin HP36, GPU confirmed, mean pLDDT 82.4, pTM 0.44, 616 s — validated the real output parsing. DEFERRED (see §9): fused validate-design meta-tool, MPNN auto-pull, batch top-N, amber relax, single_sequence, generalized ETA. §3/§6/§9/§10/§11/§12 updated. |
| 2026-05-31 | 523 | docs(colabfold): record ColabFold bridge v1 design note before build (STEP 0 of `feat/colabfold-bridge`; §9). Scope, DEFERRED items, laptop-GPU multimer caveat (`COLABFOLD_MAX_TOTAL_RESIDUES` budget; large tetramers expected to OOM gracefully; recalibrate empirically), and the remote/alternate-GPU handoff backlog item — committed durably before the long build. |
| 2026-05-31 | 523 | feat(env)+docs: **ColabFold environment stood up + GPU-validated in WSL2** (`feat/colabfold-env`; env + validation ONLY — no `colabfold_bridge.py` yet, that is the next task). Isolated WSL2 **Python-3.12** venv `~/colabfold_env`, **hermetic CUDA via pip** (`jax[cuda12]`) — `pyrosetta_env`/system CUDA/Windows venv312 all untouched. **Decision change recorded**: the old "venv310 (Py 3.10) shared with RFdiffusion" plan is **superseded** by this WSL2 Py-3.12 env; **RFdiffusion kept SEPARATE**. **Exact versions**: python 3.12.3, colabfold **1.6.1**, alphafold-colabfold 2.3.13, dm-haiku 0.0.16, jax/jaxlib **0.5.3**, jax-cuda12-plugin/pjrt **0.5.3**, nvidia-cuda-nvcc-cu12 12.9.86, cuDNN 9.23.0.39; WSL2 driver **596.49 / CUDA 13.2**, RTX 5070 Ti. **GPU + sm_120 verified** (`jax.devices()`→`[CudaDevice(id=0)]`, 2048³ matmul, no ptxas error). **JAX reconciliation**: latest `jax[cuda12]` 0.10.1 supports sm_120 but ColabFold caps `jax<0.6.0`→0.5.3; the cuda12 plugin had to be realigned to 0.5.3 (`pip install "jax[cuda12]==0.5.3"`) or the PJRT C-API aborts — clean intersection at 0.5.3, which still supports sm_120. **Smoke fold** via **remote MSA server** (no local DBs): villin HP36 (36 aa), 1 model / 1 recycle, **on GPU** → rank_001 **pLDDT 82.4, pTM 0.44**, outputs in `~/colabfold_test/out/`, wall 13:59 (one-time 3.47 GB weight download + first sm_120 XLA compile dominate). Added one constant `COLABFOLD_PYTHON` to `wsl_bridge.py` (mirrors `PYROSETTA_PYTHON`; no bridge logic). Docs: §9/§10/§11/§12 updated. |
| 2026-05-31 | 523 | test(rosetta): **benchmark-honesty follow-up to `06f6478`** (`tests/test_rosetta_benchmark.py`). Two fixes, both verified offline. **Fix A — xfail buckets for buried wrong-sign mutations.** The per-mutation sign policy now has three buckets: (1) `|exp| <= SIGN_ASSERT_MIN_ABS_EXP` (1.5) → record-only, no assert (single-traj noise = coin-flip); (2) `|exp| > 1.5` and not a known buried case → hard sign assert; (3) `|exp| > 1.5` AND a documented §8-failure-mode-2 buried wrong-sign case under the **default strip-water** path → `@pytest.mark.xfail(strict=False, raises=AssertionError)`. Without (3) these would deterministically go RED the first time the live benchmark runs correctly on the default path — the inverse of the noise-flapping `06f6478` removed. Marked: **G88V (2SNS, exp +2.1)** and **V82A (1HSG, exp +1.75)**; T26A (exp +1.3) is already bucket 1 so needs no marker. `_record(...)` still runs for the xfail cases (precedes the sign assert), so they still feed the aggregate. **Fix B — opt-in live-benchmark gate.** New `STRUCTUREBOT_RUN_LIVE_BENCHMARK=1` env requirement in `pytestmark`: live run iff `RUN_LIVE_BENCHMARK and WSL2 and PyRosetta`; the `and` short-circuits so a bare (non-opted-in) collection never spawns WSL to probe PyRosetta. Closes the footgun that a dev box with WSL2+PyRosetta installed would silently launch the hours-long live suite. Verified: non-benchmark suite **523 passed**; `pytest tests/test_rosetta_benchmark.py` → **12 skipped, 0 run, 0.04s** (opt-in unset). Live benchmark **not run** (intentional). |
| 2026-05-31 | 523 | test+docs: **benchmark-test honesty** (`tests/test_rosetta_benchmark.py`). (1) **cp1252 crash fixed** — the `✓`/`✗` glyph in the `compute_benchmark_correlation` summary (`UnicodeEncodeError` on the Windows cp1252 console, so the test never evaluated) is now ASCII `[PASS]`/`[FAIL]`, plus a best-effort `sys.stdout/stderr.reconfigure(encoding="utf-8")` mirroring `scripts/validate_2lzm_panel.py`; remaining `ΔΔG` in executable assert messages → `ddG` (docstrings/banners left as-is). (2) **per-mutation tests de-flaked** — the ±2.0 kcal/mol magnitude asserts (which gated on stochastic single-trajectory noise: T26A +1.41 vs +7.74 across two live runs) are removed; new `_check_sign` helper asserts **sign only** for `|experimental| > SIGN_ASSERT_MIN_ABS_EXP` (1.5), records-without-asserting for near-neutral mutations; all per-mutation tests still `_record(...)` to `benchmark_results.json`. (3) **aggregate is the gate** — `test_benchmark_correlation_acceptable` keeps the validated thresholds (r>0.3, RMSE<4.0, sign≥60%) and gains a `MIN_BENCHMARK_ENTRIES=6` minimum-N guard that **skips (not fails)** below the floor. Verified: non-benchmark suite **523 passed**; benchmark module collects 12 and **skips all 12 cleanly** without WSL2/PyRosetta; cp1252 + new logic exercised offline with synthetic results. Live PyRosetta benchmark **not run** (intentional). Build Queue §9 item 1 closed; §8 thresholds note + §12 updated. |
| 2026-05-31 | 523 | fix(config)+docs: **water-handling default RESOLVED — revert `ROSETTA_STRIP_WATERS` default to `True` (strip)**. Strip is the validated, standard-Rosetta-practice baseline (all rigorous panel validation, commit cbe327a, is on the strip path); preserve-ALL-static is a documented Rosetta anti-pattern (static waters → clash-driven over-destabilization) with unmeasured panel behaviour. The preserve code path is fully intact and now **opt-in** via `ROSETTA_STRIP_WATERS=false` (validated on buried-water T26A: +3.80 shift, sign corrected); T26A's buried-water benefit is thus still served. `config.py` default flipped `False`→`True` (logic now defaults to strip unless explicitly set falsy) + inline comment rewritten. Docs synced: §3/§4/§7/§12 + `scripts/rosetta_validation_notes.md` — the cbe327a panel numbers (RMSE 2.73, r +0.50, sign 90%) again describe the **default** path, "non-default" caveat softened. Build Queue: preserve-path 2LZM re-validation **downgraded to conditional** (only if preserve is promoted to default or made selective); **selective/movable buried-only water handling** kept as the real future fix. Non-benchmark suite green (**523 passed**). |
| 2026-05-30 | 535 | docs: **REU→kcal/mol scaling audit — verdict recorded (leave uncalibrated)**. §7 Calibration item updated from "under audit" to the verdict, mirrored into `scripts/rosetta_validation_notes.md`. Findings: the active ddG path is **NOT `cartesian_ddg`** but a manual `ref2015` **torsion-space** delta (`rosetta_bridge.py:1449-1464`); reported "kcal/mol" is the **raw REU delta with NO conversion** (factor 1.0; grep confirms no `2.94`/scaling). Park's 2.94 is cartesian_ddg-specific and **over-corrects** (÷2.94 on the 2LZM panel: MAE 2.59→1.24 but bias flips +1.28→−0.94 and large-cavity mutations wrecked), so magnitudes are **genuinely uncalibrated** — "ranking reliable / magnitudes approximate" is accurate, and scaling is monotonic ⇒ irrelevant to ranking/sign. **DECISION: add no scaling factor or fitted calibration** (recorded so a future session doesn't). Build Queue: the "verify/fix scaling (2.94)" item marked AUDIT COMPLETE; new **backlog** item "adopt canonical cartesian_ddg for calibrated magnitudes — only if absolute magnitudes become a requirement". Docs-only; no `.py` changes. |
| 2026-05-30 | 535 | docs: **Rosetta ddG methodology & known deviations** recorded (§7, new subsection). Captures (1) **Basis** — the active path is a manual symmetric-relax protocol (`rosetta_bridge.py:1449-1464`: clone WT → mutate → torsion-space FastRelax(mutant) → independently FastRelax a re-cloned WT → subtract full-pose `ref2015` total scores → median over trajectories), **modeled on but deviating from** canonical `cartesian_ddg`/`ref2015_cart` (plain `ref2015`, torsion not cartesian; only multi-traj+median mirrors the canonical averaging); the "CartesianDDG"/"ref2015_cart" labels in the stub docstring/`method_note` are aspirational, not the running code. (2) **Calibration** — the protocol-specific ~2.94 REU→kcal/mol factor (Park 2016); over-prediction may be partly a calibration artifact; **status: under audit**. (3) **Water handling** — default `ROSETTA_STRIP_WATERS=False` preserves all ~216 waters as static HOH, a deliberate deviation from best practice (strip bulk/surface, keep only buried/bridging, movably via Rosetta-ICO/ECO; Pavlovicz/Park/DiMaio 2020); decision pending. Build Queue: added "verify/fix REU→kcal/mol scaling (2.94)" and "selective/movable water handling — do not default to preserve-all-static"; preserve-path 2LZM re-validation item kept. Docs-only; no `.py` changes. |
| 2026-05-30 | 535 | feat+docs: **crystal-water preservation** (`ROSETTA_STRIP_WATERS`, default False = preserve; commit 3318bc9) + **benchmark thresholds/wording** (commit db3add2). cleanATOM was stripping ALL HETATM (crystallographic HOH included), the suspected cause of wrong-sign buried ddG; the worker now re-appends HOH and namespaces the relax cache by mode (`_wat`). Validated live on **Barnase T26A**: strip −0.34 (wrong sign) → preserve +3.47 (correct, exp +1.3), **+3.80 shift**, 216 HOH reached the pose. **⚠ Caveat recorded:** the committed validation-tier 2LZM numbers (RMSE 2.73, r +0.50, sign 90%) were measured on the stripped-water path (now non-default); a **preserve-path 2LZM re-validation is PENDING** (Build Queue §9 item 1) — the panel's surface/cavity mutations aren't near buried waters so the effect on them is unmeasured. Also: benchmark thresholds relaxed to r>0.3/RMSE<4.0/sign>=60% and the validate_ddg disclosure broadened from "large-cavity" to ~2-3 kcal/mol across all categories. (§3/§7/§9 caveats + this entry added; docs-only commit.) |
| 2026-05-30 | 535 | docs: **validation-tier 2LZM panel executed and confirmed**. Full 10-mutation panel scored via `scripts/validate_2lzm_panel.py` at the validation tier (N=5 trajectories × 8+8 relax cycles, median): **RMSE 2.729, MAE 2.586, Pearson r +0.499, sign 90%** (9/10; lone flip S117V −1.24 vs exp +0.9) — all three revised thresholds (r>0.3, RMSE<4.0, sign>=60%) **PASS**. RMSE/MAE roughly halve vs single-trajectory (5.23→2.73, 3.92→2.59) while r is unchanged (~0.50): gain in magnitude, not ranking; residual ~2.4 scatter over ~+1.3 systematic over-prediction, no longer large-cavity-specific. Build Queue item 1 fully closed (§3/§7/§9/§12 updated); numbers in `scripts/rosetta_validation_notes.md`. Also committed this session: the `validate_2lzm_panel.py` restartable runner + self-managing no-sleep helper. Docs-only; no `.py` changes in this commit. |
| 2026-05-29 | 526 | Session hardening + PyRosetta ddG validation. **main.py/session_state.py**: session auto-restore on startup (`try_load`→(state,error), `_from_dict`, `restore_summary`, `_maybe_restore_session`, `_reconnect_or_offer_reopen`) + `clear session`/`new session` command; +`tests/test_session_state.py` (7). **chimerax_bridge.py**: auto-reconnect + one-shot retry on dropped REST connection (`_run_command_once`/`_try_reconnect`); +`tests/test_chimerax_bridge.py` (4). **rosetta_bridge.py**: DynaMut2 single-parser fixed (status DONE/RUNNING/ERROR, robust float cast, raises not silent-0.0); local PyRosetta path hardened (pose_from_file validation + pose_from_pdb fallback, whole-batch empirical fallback, `ddg_source` provenance); optional `relax_cycles` param (default 3, production-unchanged); Ubuntu-24.04 string + stub-comment corrections; +4 DynaMut2 single tests. **mutation_scanner.py/tool_router.py**: ddG shown to 3 dp, `ddg_source` threaded into scan results + display. **double_mutant_bridge.py**: verbose per-pair routing dumps removed → concise summary; validated live e2e. **Repo hygiene**: added `.gitignore`, untracked cache/logs/sessions/ProteinMPNN/venvs/scratch. **Validation scripts (scripts/test_pyrosetta_*.py)**: single, ddg_controls (1HSG), ddg_t4lysozyme (2LZM r=+0.487, sign 100%, MAE 3.9), aggregation_diag (~8 kcal/mol spread; median best, min refuted), convergence_diag (running). PROJECT_CONTEXT.md regenerated |
| 2026-05-29 | 511 | fix(double_mutant): stability-mode ddG filter rewritten — `ddg=0.0` (DynaMut2 neutral/unknown) is no longer treated as non-beneficial; a pair is excluded only when **both** mutations are clearly destabilising (`ddg > DOUBLE_MUTANT_DESTABILISING_DDG`, new config constant, default +2.0). Restores the full candidate set (~79 pairs) that previously collapsed to 2 when all single-point ΔΔG were zero. Close (`pyrosetta_required`, <4 Å) pairs now scored with additive fallback (`ddG_A + ddG_B`, epistasis=0) when `run_pyrosetta=False` instead of being dropped; new `DOUBLE_MUTANT_ADDITIVE_FALLBACK` flag documented; shared `_apply_additive_fallback()` helper extracted. Tests: +`test_ddg_zero_not_filtered_in_stability_mode`, +`test_pyrosetta_pairs_get_additive_fallback_when_disabled`, replaced obsolete close-pair-skip test (double-mutant file 36→42). PROJECT_CONTEXT.md updated |
| 2026-05-29 | 496 | double_mutant_bridge.py built (36 tests): two-mode stability/epitope pair scoring, DynaMut2 multi-mutation API (prediction_mm), PyRosetta WSL2 close-pair validation, epistasis detection, distance-based backend routing; 4 new config constants. Tool router integration: _DOUBLE_MUTANT_KEYWORDS intent detection, _run_double_mutant() with full 5-step pipeline, _build_double_mutant_viz(), session_state.double_mutant_results field, 8 new routing tests, pytest.ini marker registrations. PyRosetta benchmark run 1 complete: sign accuracy 60%, RMSE 5.49, r=-0.059 (A98V outlier); rosetta_validation_notes.md updated with failure mode analysis, revised thresholds (r>0.30, RMSE<4.0), improvement roadmap. ColabFold+RFdiffusion added to build queue. PROJECT_CONTEXT.md regenerated |
| 2026-05-29 | 452 | PyRosetta WSL2 backend fully implemented: symmetric FastRelax ddG protocol, cleanATOM PDB prep, modern pdb2pose API, ref2015 score function, wsl_bridge PYROSETTA_PYTHON constant + Ubuntu-24.04 default; benchmark test suite (11 ProThermDB mutations) + rosetta_validation_notes.md created; PROJECT_CONTEXT.md generated |
| 2026-05-28 | 440 | feat(netnglyc): NetNGlyc 1.0 OST recognition integration |
| 2026-05-28 | [verify] | feat(glycan): projection-aware glycosylation position scan + fold validation |
| 2026-05-28 | [verify] | feat(cavity_bridge): dimer/oligomer-aware cavity detection with BFS clustering |
| 2026-05-28 | [verify] | feat(glycan): projection scoring, sequon geometry, structural_utils shared module |
| 2026-05-28 | [verify] | feat: add salt_bridge_bridge, cavity_bridge, FASTA export, sequence display fix |
| 2026-05-27 | [verify] | feat: semicolon chaining, --script flag, compact MPNN+ESMFold summary table |
| 2026-05-27 | [verify] | fix(router): multi-model session routing and sequence display bugs |
| 2026-05-27 | [verify] | feat(router): session-aware MPNN+ESMFold routing and sequence display command |
| 2026-05-27 | [verify] | fix(routing): prevent glycan clarification loop and refusal crash |
| 2026-05-27 | [verify] | feat(mpnn_esmfold): ProteinMPNN + ESMFold combined redesign and validation pipeline |
| 2026-05-27 | [verify] | feat(glycan): full N-glycosylation site prediction |
| 2026-05-26 | [verify] | feat(esmfold): local GPU inference via venv312 subprocess |
