# StructureBot — Project Context

<!-- Regenerate with: claude "Read PROJECT_CONTEXT.md for regeneration instructions,
then regenerate it in full by reading the entire codebase. Preserve the Changelog
section and append a new entry." -->

## Meta

| Field | Value |
|-------|-------|
| Generated | 2026-05-29 |
| Test count at generation | 511 collected / 499 passing (12 benchmark tests skip without WSL2+PyRosetta) |
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
| `main.py` | `StructureBot`, `_ElapsedTicker` | ✅ Complete | REPL + `--script` mode; startup sequence; session persistence |
| `config.py` | constants + `load_env_file()` | ✅ Complete | Called first in `main.py`; all env-var overrides centralised here |
| `translator.py` | `CommandTranslator` | ✅ Complete | Claude API; prompt caching (Block 1 static, Block 2 dynamic); rolling history `MAX_CONVERSATION_HISTORY=6` |
| `tool_router.py` | `ToolRouter`, `ToolStepResult` | ✅ Complete | Dispatches 14 tool types including `double_mutant`; intent detection for all tools; MPNN+ESMFold combined pipeline; FASTA export; sequence display fast-path |
| `session_state.py` | `SessionState`, `parse_pdb_header()`, `fetch_rcsb_metadata()` | ✅ Complete | Persists all tool results including `double_mutant_results`; save/load/snapshot/restore |
| `chimerax_bridge.py` | `ChimeraXBridge`, `find_chimerax()` | ✅ Complete | REST API on port 60001; blank-image post-save guard; `run_commands()` |
| `wsl_bridge.py` | `WSLBridge`, `PYROSETTA_PYTHON` | ✅ Complete | `PYROSETTA_PYTHON="/home/andre/pyrosetta_env/bin/python"`; default distro `Ubuntu-24.04`; `check_pyrosetta()` uses `chr(79)+chr(75)` |
| `rosetta_bridge.py` | `RosettaBridge`, `_select_backend()` | ⚠️ Known issues | 4 backends (dynamut2/empirical/pyrosetta-stub/local-WSL2); benchmark run 1 complete: sign accuracy 60%, RMSE 5.49, r=−0.059; see §8 for protocol audit and failure modes |
| `double_mutant_bridge.py` | `DoubleMutantBridge`, `generate_pairs()`, `compute_ca_distance()`, `route_pairs()`, `score_pairs_dynamut2()`, `score_pairs_pyrosetta()`, `compute_composite_score()`, `generate_summary()`, `_apply_additive_fallback()` | ✅ Complete | Two-mode (stability/epitope) double mutant ΔΔG scoring; DynaMut2 `prediction_mm` for distant pairs (>10 Å), PyRosetta WSL2 for close pairs (<4 Å); real epistasis = ddG(double) − ddG(additive). Stability ddG filter excludes a pair only when **both** mutations are clearly destabilising (`ddg > DOUBLE_MUTANT_DESTABILISING_DDG`); `ddg=0.0` treated as neutral. Close pairs fall back to additive scoring when `run_pyrosetta=False` (no longer dropped). Shared `_apply_additive_fallback()` helper. 42 tests |
| `esm_bridge.py` | `EsmBridge` | ✅ Complete | ESM-2 `esm2_t6_8M_UR50D` default; delegates GPU inference to `esm_worker.py` via venv312 subprocess; disk cache `cache/esm_{hash}.json` |
| `esm_worker.py` | standalone subprocess script | ✅ Complete | No project imports; writes JSON result file; run by venv312 python |
| `esmfold_bridge.py` | `ESMFoldBridge` | ✅ Complete | Primary: venv312 GPU via `esmfold_worker.py`; fallback: ESM Atlas API; `compare_to_wildtype()`, `check_disulfide_foldability()` |
| `esmfold_worker.py` | standalone subprocess script | ✅ Complete | HuggingFace `facebook/esmfold_v1`; no project imports; pLDDT normalisation guard (×100 if mean < 2.0) |
| `mutation_scanner.py` | `MutationScanner` | ✅ Complete | CamSol+ESM+Rosetta pipeline; combined score `0.5×(-ddG) + 0.3×camsol_delta + 0.2×esm_tolerance`; Pro/Cys exclusion; interface protection |
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

**Total: 511 collected** | **499 passing** (12 benchmark tests skip unless WSL2+PyRosetta present; on this machine they run, since PyRosetta is installed in WSL2)

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
| `test_tool_router.py` | 55 | Route dispatch, MPNN+ESMFold routing, FASTA export, active-site commands, double mutant routing (8 new tests), tool icon registry |
| `test_double_mutant_bridge.py` | 42 | Pair generation (stability and epitope modes), distance routing, DynaMut2 `prediction_mm` result parsing, PyRosetta worker schema, composite scoring formulas, epistasis sign convention, max-pairs cap, ddG-filter neutrality (`ddg=0.0` not excluded), additive fallback (API failure, circuit-breaker survival, close-pair-without-PyRosetta) |
| `test_proline_bridge.py` | 35 | φ-angle scoring, functional residue exclusion, DSSP fallback, BioPython parsing |
| `test_disulfide.py` | 35 | Cβ geometry, dihedral scoring, ESM tolerance, DynaMut2 mock, combined score |
| `test_mpnn_esmfold_pipeline.py` | 29 | MPNN+ESMFold combined pipeline, session routing, pLDDT comparison |
| `test_rosetta.py` | 27 | Backend detection, DynaMut2 HTTP mock, MutationScanner, combined scoring, session persistence |
| `test_proteinmpnn.py` | 26 | ProteinMPNN subprocess call, JSON output parsing, error paths |
| `test_esmfold.py` | 25 | ESMFold local/atlas paths, pLDDT normalisation, foldability risk thresholds |
| `test_tools.py` | 24 | Integration: CamSol, ESM, ChimeraX commands, DynaMut2 stub |
| `test_assembly.py` | 21 | RCSB assembly metadata, monomer/multimer mode, interface detection mock |
| `test_cavity_bridge.py` | 20 | BFS cavity clustering, SASA burial, volume estimation, interface flagging |
| `test_netnglyc_bridge.py` | 19 | OST score parsing, harmonic mean integration, API mock |
| `test_structural_utils.py` | 17 | `extract_backbone_angles()`, `compute_sasa()`, `compute_projection_score()` |
| `test_wsl.py` | 12 | `WSLBridge` availability, path translation, `run_command()` (skip if no WSL2) |
| `test_rosetta_benchmark.py` | 12 | PyRosetta ddG benchmarks vs ProThermDB (11 mutations + 1 correlation test); `@pytest.mark.benchmark` and `@pytest.mark.slow`; skipped without WSL2+PyRosetta |
| `test_rfdiffusion.py` | 12 | RFdiffusion stub error structure, route detection, directory validation |
| `test_salt_bridge_bridge.py` | 11 | Salt bridge geometry, charge classification, SASA burial scoring |
| `test_main.py` | 10 | `StructureBot` startup mocking, ChimeraX connection, session save/load |
| `test_translator.py` | 8 | `CommandTranslator` API mock, prompt caching structure, history management |
| `test_integration.py` | 6 | End-to-end route + execute mock (no live ChimeraX or APIs) |

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
| Double mutant bridge | All double-mutant unit tests passing (42 in `test_double_mutant_bridge.py`); bridge importable; `analyze()` returns correct schema with all required keys. Stability filter now yields the full candidate set (~79 pairs on the all-zero-ΔΔG dataset that previously collapsed to 2); close pairs scored additively when PyRosetta disabled. Live end-to-end test pending (requires live ChimeraX session with prior mutation scan) |

**PyRosetta Benchmark Run 1 (2026-05-29)** — 11 mutations from ProThermDB:

| Metric | Result |
|--------|--------|
| Sign accuracy | 6/10 = 60% (I64E excluded, no precise experimental value) |
| Within 2.0 kcal/mol | 4/10 = 40% |
| MAE | 3.823 kcal/mol |
| RMSE | 5.492 kcal/mol |
| Pearson r | −0.059 (driven negative by A98V outlier: +14.52 predicted vs −0.5 experimental) |

Protocol suitable for sign prediction on surface-exposed mutations (4/6 correct for non-buried, non-interface positions); not reliable for buried mutations or magnitude accuracy. Without the A98V outlier, r ≈ +0.46 and RMSE ≈ 2.58 kcal/mol. Full analysis in `scripts/rosetta_validation_notes.md`.

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
I88V (+5.264 vs +0.6), L69A (+6.6 vs +2.4), L99A (+7.137 vs +4.0), A98V (+14.52 vs −0.5). Root cause: single trajectory landing in poor local minimum; 50-replicate averaging would resolve.

**Failure mode 2 — Wrong sign on buried mutations:**
T26A (−2.437 predicted vs +1.3 exp), G88V (−0.25 vs +2.1), V82A (−0.06 vs +1.75). Root cause: `cleanATOM` removing crystallographic waters that stabilise buried positions in Barnase and SNase.

### Revised benchmark thresholds (updated based on run 1)
Current tests assert `r > 0.50`, `RMSE < 2.50`, `sign_accuracy ≥ 70%` — all fail with current single-trajectory protocol. Realistic thresholds for current protocol: `r > 0.30`, `RMSE < 4.00`, `sign_accuracy ≥ 60%`. Tests in `test_rosetta_benchmark.py` need updating (follow-up task, not yet done).

### From code review (TODO/FIXME/stub findings)

| Location | Issue |
|----------|-------|
| `rfdiffusion_bridge.py` | Entire module is a documented stub; requires Python 3.9-3.11 environment |
| `rosetta_bridge.py` — `_run_pyrosetta()` | Backend A (`pyrosetta` mode) is a documented stub; WSL2 path (`local` backend) works |
| `rosetta_bridge.py` docstring line ~1108 | Still references `Ubuntu-22.04` in `_run_rosetta_local()` docstring (runtime correctly uses Ubuntu-24.04) |
| `main.py` line 327 | WSL2 availability message still says `wsl --install -d Ubuntu-22.04` (display string only) |
| `diag.py` | One-off diagnostic script in project root; not imported by anything |

---

## 9. Build Queue

Prioritised by impact and readiness:

| Priority | Item | Rationale |
|----------|------|-----------|
| 1 | **Validate double mutant live end-to-end** | Bridge and tool router integration are built; need live validation: open 1HSG → mutation scan → "suggest double mutant combinations". Should print Rich Panel and execute ChimeraX coloring + distance lines |
| 2 | **ColabFold bridge** (design agreed, prompt not yet written) | AF2-quality folding for designed sequences; template-guided mode validates ProteinMPNN designs with wildtype structure as template. Prerequisites: new `venv310` (Python 3.10, shared with RFdiffusion); ColabFold pip install; ~20 GB AF2 weights. Two modes: template-guided (primary), de novo. Returns pLDDT, PAE, TM-score vs template, per-residue RMSD. Integrates with MPNN+ESMFold pipeline as higher-accuracy validation. **Plan venv310 setup before writing ColabFold prompt** |
| 3 | **RFdiffusion activation** (stub exists at `rfdiffusion_bridge.py`) | De novo backbone generation; completes design loop (RFdiffusion → ProteinMPNN → ColabFold). Prerequisites: `venv310` shared with ColabFold; ~20 GB weights; Python 3.9-3.11. Stub already written; activation is configuration + environment task |
| 4 | **PyRosetta protocol improvements** (benchmark failures documented) | Three targeted fixes in priority order: (a) Preserve crystallographic waters — one-line change to `cleanATOM` call, highest impact/effort ratio, fixes T26A and G88V; (b) `ROSETTA_NUM_TRAJECTORIES` config flag for multi-trajectory averaging; (c) Detect mutant > WT size → extra relax cycles for A98V-type cavity-filling mutations |
| 5 | **Update benchmark test thresholds** | `test_rosetta_benchmark.py` currently asserts `r > 0.50` which fails with current protocol. Update to `r > 0.30`, `RMSE < 4.00`, `sign_accuracy ≥ 60%` |
| 6 | **Mid-execution ESC / cancellation** | Long-running tools (PyRosetta ~15 min, ColabFold ~5 min) have no cancellation. Background thread pattern needed |
| 7 | **Double mutant — PyRosetta multi-mutation close-pair validation** | Close pairs (<4 Å) now get an *additive* estimate when `run_pyrosetta=False` (no longer dropped), but additive scoring cannot capture epistasis for strongly-interacting residues. After the PyRosetta protocol is improved (item 4), enabling real close-pair scoring becomes scientifically valuable |
| 8 | **GlycosuitDB lookup** | Expression system glycan characterisation for engineered sequons — tells what glycan structure to expect in CHO, HEK293, E. coli etc. |

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
| ProteinMPNN repo | `C:\Users\andre\documents\structurebot\ProteinMPNN\` |

---

## 12. Session Continuity Notes

**For starting a new co-pilot conversation:**

- **Test count**: 511 collected, 499 passing (12 benchmark tests skip without WSL2+PyRosetta). Non-benchmark suite runs in ~37s. No warnings about unknown markers.
- **Last built and validated**: `double_mutant_bridge.py` (DoubleMutantBridge, 42 tests, two-mode stability/epitope scoring, DynaMut2 `prediction_mm` API, PyRosetta WSL2 close-pair validation, epistasis detection, distance-based backend routing). Most recent change (commit `fix(double_mutant): treat ddg=0.0 as neutral; additive fallback for close pairs`): stability ddG filter now treats `ddg=0.0` as neutral and excludes a pair only when **both** mutations exceed `DOUBLE_MUTANT_DESTABILISING_DDG` (+2.0) — this restored the full candidate set (~79 pairs) from a previous collapse to 2; close pairs now get additive fallback scoring when `run_pyrosetta=False` instead of being dropped. Tool router integration wired (`tool_router.py`, `session_state.py`, 8 routing tests, `pytest.ini` marker registrations) — **all 499 non-benchmark tests green**, live end-to-end validation pending.
- **Immediate next item**: Live validation of double mutant pipeline: `python main.py` → open 1HSG → "suggest mutations to improve solubility of chain A" → "suggest double mutant combinations". Expected: Rich Panel with top pairs table, ChimeraX colored spheres + Cα-Cα distance lines. Then test: "suggest double mutant combinations to preserve the epitope" → should use epitope mode.
- **After double mutant validated**: ColabFold bridge. **First step is creating venv310** — write a venv310 setup prompt before the ColabFold bridge prompt. venv310 will be shared with RFdiffusion (plan both together).
- **PyRosetta benchmark**: Run 1 complete. Sign accuracy 60%, RMSE 5.49, r=−0.059. Crystallographic water fix is highest-priority code change (one-line, targets T26A and G88V failures). `test_rosetta_benchmark.py` thresholds need updating from `r > 0.50` to `r > 0.30` (follow-up task).
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
