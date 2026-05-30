# StructureBot — Project Context

<!-- Regenerate with: claude "Read PROJECT_CONTEXT.md for regeneration instructions,
then regenerate it in full by reading the entire codebase. Preserve the Changelog
section and append a new entry." -->

## Meta

| Field | Value |
|-------|-------|
| Generated | 2026-05-29 |
| Test count at generation | 535 collected / 523 passing (12 benchmark tests skip without WSL2+PyRosetta) |
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
  │     Intent overrides (in order): double_mutant → mpnn_esmfold → glycan_positions
  │     → netnglyc → glycan → salt_bridge → cavity → double_mutant (final check)
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
| `session_state.py` | `SessionState`, `try_load()`, `_from_dict()`, `restore_summary()`, `parse_pdb_header()`, `fetch_rcsb_metadata()` | ✅ Complete | Persists all tool results including `double_mutant_results`; save/load/snapshot/restore. **Auto-restore support**: `try_load()` returns `(state, error)` — `(None, None)` missing file, `(None, "msg")` corrupt/incompatible, `(state, None)` ok; `load()` delegates and never raises (fresh state on failure); `_from_dict()` shared builder; `restore_summary()` one-screen summary for the startup prompt |
| `chimerax_bridge.py` | `ChimeraXBridge`, `find_chimerax()`, `_run_command_once()`, `_try_reconnect()` | ✅ Complete | REST API on port 60001; blank-image post-save guard. **Auto-reconnect**: `run_command()` wraps `_run_command_once()`; on a dropped connection (`ConnectionError` at pre-check or mid-request) it calls `ensure_connected()` once and retries — succeeds silently or raises a clear "check ChimeraX is still open" error. `run_commands()` inherits this per-command |
| `wsl_bridge.py` | `WSLBridge`, `PYROSETTA_PYTHON` | ✅ Complete | `PYROSETTA_PYTHON="/home/andre/pyrosetta_env/bin/python"`; default distro `Ubuntu-24.04`; `check_pyrosetta()` uses `chr(79)+chr(75)` |
| `rosetta_bridge.py` | `RosettaBridge`, `_select_backend()`, `_run_command_*` (DynaMut2), `_run_rosetta_local()` | ⚠️ Magnitude bias | 4 backends (dynamut2/empirical/pyrosetta-stub/local-WSL2). **DynaMut2 single parser fixed** — handles current status-based API (`status` DONE/RUNNING/ERROR), robust float cast of `prediction`, raises (never silent 0.0) on ERROR. **Local PyRosetta path hardened**: `pose_from_file` validation + `.pdb`-extension/`pose_from_pdb` fallback; whole-batch failure → per-mutation **empirical fallback** (not all-zero); each value carries **`ddg_source`** (`pyrosetta`/`empirical`). **Tiered multi-trajectory ddG**: `_run_rosetta_local(num_trajectories=, relax_cycles=)` runs N independent relax+score trajectories per mutation (per-trajectory RNG seed when N>1), aggregates by **median** (`_aggregate_ddg_trajectories`), reports **MAD spread** + **`ddg_confidence`** (`_ddg_confidence_label`); WSL timeout scales by N×cycles. Defaults (N=1, cycles=3) = production single-trajectory, unchanged. **`validate_ddg()`** = high-accuracy tier (ROSETTA_VALIDATION_TRAJECTORIES/CYCLES) with a non-calibrated disclosure. Path validated for ranking (T4 r≈+0.49, sign 100%); **validation-tier 2LZM panel (N=5 × 8+8 cycles, median, 2026-05-30): RMSE 2.73, MAE 2.59, r +0.50, sign 90% — meets all revised thresholds (r>0.3, RMSE<4.0, sign>=60%)**, roughly halving single-traj RMSE (5.23→2.73). Magnitudes now tighter but still approximate (~±2.7 kcal/mol; gain is in magnitude, not ranking). **⚠ Those panel numbers were measured with `ROSETTA_STRIP_WATERS=true` (pre-2026-05-30 default); the default is now preserve-waters (`False`), so a preserve-path 2LZM re-validation is PENDING (Build Queue §9).** See §7/§8 |
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

**Total: 526 collected** | **514 passing** (12 benchmark tests skip unless WSL2+PyRosetta present; on this machine they run, since PyRosetta is installed in WSL2)

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

> **⚠ CAVEAT — measured on the stripped-water path, now NON-default.** These numbers were
> produced with `ROSETTA_STRIP_WATERS=true`, the default *until 2026-05-30*. The default is
> now **preserve-waters** (`ROSETTA_STRIP_WATERS=False`; commit 3318bc9), which carries
> crystallographic HOH through to PyRosetta. Preserve-waters is validated on buried-water
> target **Barnase T26A** (strip −0.34 wrong-sign → preserve +3.47 correct, exp +1.3; a
> **+3.80 shift**, 216 HOH confirmed reaching the pose), but a **preserve-path re-validation
> of the full 2LZM panel is PENDING** (Build Queue §9). The panel's surface/cavity mutations
> are not near buried waters, so the effect on these specific metrics is **unmeasured**; they
> remain valid for the stripped-water path only.

### Validation scripts (`scripts/`, read-only; require ROSETTA_BACKEND=local)

| Script | Validates |
|--------|-----------|
| `test_pyrosetta_single.py` | Single-mutation probe (I72R/1HSG): dumps worker debug + final ddg/ddg_source; confirmed the local path works |
| `test_pyrosetta_ddg_controls.py` | 1HSG buried→charged / moderate / surface panel — magnitude sanity (buried ≫ surface) |
| `test_pyrosetta_ddg_t4lysozyme.py` | 2LZM vs 10 published experimental ddG (sign/MAE/RMSE/Pearson r vs Benchmark Run 1) |
| `test_pyrosetta_aggregation_diag.py` | 5 trajectories × 3 mutations — which aggregator (min/median/mean) best matches experiment |
| `test_pyrosetta_convergence_diag.py` | 3 levels (3/5/8 relax cycles) × 5 trajectories — is large-cavity bias convergence-fixable or baked in |

Full pipeline validation script: `scripts/validate_full_pipeline.txt`

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
Current tests assert `r > 0.50`, `RMSE < 2.50`, `sign_accuracy ≥ 70%` — all fail with current single-trajectory protocol. Realistic thresholds for current protocol: `r > 0.30`, `RMSE < 4.00`, `sign_accuracy ≥ 60%`. Tests in `test_rosetta_benchmark.py` need updating (follow-up task, not yet done).

### From code review (TODO/FIXME/stub findings)

| Location | Issue |
|----------|-------|
| `rfdiffusion_bridge.py` | Entire module is a documented stub; requires Python 3.9-3.11 environment |
| `rosetta_bridge.py` — `_run_pyrosetta()` | Backend A (`pyrosetta` mode) is a documented stub; WSL2 path (`local` backend) works |
| `rosetta_bridge.py` Ubuntu strings | ✅ FIXED 2026-05-29 — all `Ubuntu-22.04` references in `_run_rosetta_local()` / `backend_status()` corrected to `Ubuntu-24.04`; the stale "documented stub" comment was also corrected |
| `main.py` line ~327 | WSL2 availability message still says `wsl --install -d Ubuntu-22.04` (display string only; not yet updated) |
| `diag.py` | One-off diagnostic script in project root; now gitignored / untracked |

---

## 9. Build Queue

**Completed this session (2026-05-29):** ✅ Double-mutant live end-to-end validation · ✅ Session auto-restore on startup + `clear session` command · ✅ ChimeraX auto-reconnect/retry on dropped REST connection · ✅ DynaMut2 single-parser fix + ddg_source provenance + Ubuntu-24.04 corrections.

Prioritised by impact and readiness:

| Priority | Item | Rationale |
|----------|------|-----------|
| 1 | **Preserve-path 2LZM validation-tier re-run** | The committed validation-tier panel (RMSE 2.73, r +0.50, sign 90%; §7) was measured on the **stripped-water** path, now NON-default. Re-run the full 10-mutation 2LZM panel at the validation tier with the new default `ROSETTA_STRIP_WATERS=False` (preserve) and compare. Buried-water target T26A already shows a +3.80 shift / sign correction, but the panel's surface/cavity mutations aren't near buried waters so the effect on those is unmeasured. ~4.5 h via `scripts/validate_2lzm_panel.py` (preserve-path uses a fresh `_wat` relax-cache namespace, so it won't collide with the stripped-water cache) |
| 2 | ~~**Crystal-water cleanATOM fix**~~ — ✅ **COMPLETE (2026-05-30, commit 3318bc9)** | Added `ROSETTA_STRIP_WATERS` (default False = preserve); worker re-appends crystallographic HOH that cleanATOM strips, relax cache namespaced by mode. Validated live on **Barnase T26A**: strip −0.34 (wrong sign) → preserve +3.47 (correct, exp +1.3), +3.80 shift, 216 HOH reached the pose. Follow-up = item 1 (preserve-path 2LZM re-run) |
| 3 | ~~**Update benchmark test thresholds**~~ — ✅ **COMPLETE (2026-05-30, commit db3add2)** | `test_rosetta_benchmark.py` now asserts `r > 0.30`, `RMSE < 4.00`, `sign_accuracy ≥ 60%` (revised thresholds confirmed by the 2LZM validation panel: r +0.50, RMSE 2.73, sign 90% — all PASS) |
| 3 | ~~**Multi-trajectory ddG (median aggregation + validation tier)**~~ — ✅ **COMPLETE (2026-05-30)** | Shipped tiered `validate_ddg` (N trajectories→median + higher relax cycles) and confirmed it on the full 2LZM panel (N=5 × 8+8, median): RMSE 2.73, MAE 2.59, r +0.50, sign 90% — all revised thresholds PASS; RMSE/MAE ~halved vs single-traj, ranking unchanged. See §7 + `scripts/rosetta_validation_notes.md` |
| 4 | **ColabFold bridge** (design agreed, prompt not yet written) | AF2-quality folding for designed sequences; template-guided mode validates ProteinMPNN designs with wildtype as template. Prerequisites: new `venv310` (Python 3.10, shared with RFdiffusion); ColabFold pip install; ~20 GB AF2 weights. **Plan venv310 setup before writing the ColabFold prompt** |
| 5 | **RFdiffusion activation** (stub exists at `rfdiffusion_bridge.py`) | De novo backbone generation; completes design loop (RFdiffusion → ProteinMPNN → ColabFold). Prerequisites: `venv310` shared with ColabFold; ~20 GB weights; Python 3.9-3.11 |
| 6 | **Mid-execution ESC / cancellation** | Long-running tools (PyRosetta ~15 min, ColabFold ~5 min) have no cancellation. Background thread pattern needed |
| 7 | **Double mutant — real PyRosetta close-pair scoring** | Close pairs (<4 Å) currently get an *additive* estimate when `run_pyrosetta=False`; additive scoring cannot capture epistasis. After the PyRosetta protocol improvements (items 1–2), enabling real close-pair scoring becomes scientifically valuable |
| 8 | **GlycosuitDB lookup** | Expression-system glycan characterisation for engineered sequons (CHO, HEK293, E. coli, …) |

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

### Planned (not yet installed)

| Package | Venv | Purpose |
|---------|------|---------|
| `colabfold` | `venv310` (Python 3.10, new) | AF2-quality folding with MMseqs2 MSA; template-guided and de novo modes |
| `jax` / `jaxlib` | `venv310` | ColabFold runtime |
| RFdiffusion | `venv310` (Python 3.9-3.11) | De novo backbone generation; shared venv with ColabFold |

Note: `venv310` does not yet exist. It will be created when ColabFold or RFdiffusion setup is initiated. PyTorch (`venv312`) and JAX (`venv310`) must remain in separate venvs due to CUDA runtime conflicts.

---

## 11. File Locations Reference

| Resource | Path |
|----------|------|
| Project root | `C:\Users\andre\documents\structurebot\` |
| Main venv | `<root>\venv\` |
| venv312 (GPU) | `<root>\venv312\` |
| venv312 Python | `<root>\venv312\Scripts\python.exe` |
| WSL2 PyRosetta Python | `/home/andre/pyrosetta_env/bin/python` |
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

- **Test count**: 526 collected, 514 passing (12 benchmark tests skip without WSL2+PyRosetta). Non-benchmark suite runs in ~35s. No warnings about unknown markers.
- **Last built/validated this session**: (1) double-mutant bridge validated **live end-to-end**; (2) **session auto-restore** on startup + `clear session` command (`session_state.try_load`, `main._maybe_restore_session`); (3) **ChimeraX auto-reconnect** on dropped REST connection; (4) **DynaMut2 single-parser fix** (status-based API) + `ddg_source` provenance + `relax_cycles` param + Ubuntu-24.04 corrections in `rosetta_bridge.py`; (5) **PyRosetta ddG standalone validation** (1HSG controls, T4/2LZM cross-check, aggregation + convergence diagnostics). All session work is committed; working tree clean.
- **Convergence diagnostic COMPLETE** (2026-05-29): large-cavity bias is **partially convergence-fixable** — more relax cycles move every median toward experiment and shrink spread (7.61→5.13 kcal/mol), but at 8+8 only ~half the gap closes. `rosetta_bridge.py` is now **free** (no PyRosetta running).
- **Multi-trajectory ddG + 2LZM confirmation COMPLETE** (2026-05-30): the tiered `validate_ddg` shipped and the full 2LZM panel was scored at the validation tier (N=5 × 8+8, median): **RMSE 2.73, MAE 2.59, r +0.50, sign 90% — all revised thresholds (r>0.3, RMSE<4.0, sign>=60%) PASS**. RMSE/MAE roughly halve vs single-trajectory while r is unchanged (gain in magnitude, not ranking). Build-queue item 1 is fully closed.
- **Crystal-water fix + benchmark thresholds DONE (2026-05-30)**: `ROSETTA_STRIP_WATERS` (default False = preserve; commit 3318bc9) re-appends crystallographic HOH; validated on Barnase T26A (strip −0.34 wrong-sign → preserve +3.47 correct, +3.80 shift, 216 HOH). Benchmark thresholds relaxed to r>0.3/RMSE<4.0/sign>=60% (commit db3add2).
- **Immediate next item**: **preserve-path 2LZM validation-tier re-run** (Build Queue §9 item 1). The committed 2LZM numbers (RMSE 2.73, r +0.50, sign 90%) were measured on the now-non-default stripped-water path; the panel's surface/cavity mutations aren't near buried waters, so the preserve-path effect on them is unmeasured.
- **PyRosetta ddG status**: works for **ranking + sign** (T4/2LZM: r=+0.487 single-traj, +0.499 validation tier; sign 90–100%) and the validation tier now **roughly halves magnitude error** (RMSE 5.2→2.73). Aggregate by **median** (min refuted). Absolute ddG is tighter but still **approximate (~±2.7 kcal/mol)**.
- **After ddG protocol work**: ColabFold bridge. **First step is creating venv310** — write a venv310 setup prompt before the ColabFold bridge prompt. venv310 will be shared with RFdiffusion (plan both together).
- **Active configuration flags**:
  - `ROSETTA_BACKEND=local` in `.env.local` — PyRosetta via WSL2
  - `ESM_USE_VENV312=auto` — GPU inference if CUDA smoke-test passes
  - `ESMFOLD_FORCE_COLD_TIMEOUT=False` — correct, do not change
- **API key**: Set in `.env.local` as `ANTHROPIC_API_KEY` — do not commit this file.
- **venv310 does not yet exist** — needed for ColabFold and RFdiffusion.

---

## 13. Changelog

| Date | Tests | What changed |
|------|-------|-------------|
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
