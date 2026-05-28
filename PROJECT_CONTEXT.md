# StructureBot — Project Context

<!-- Regenerate with: claude "Read PROJECT_CONTEXT.md for regeneration instructions,
then regenerate it in full by reading the entire codebase. Preserve the Changelog
section and append a new entry." -->

## Meta

| Field | Value |
|-------|-------|
| Generated | 2026-05-29 |
| Test count at generation | 452 collected / 440 passing (12 benchmark skipped pending WSL2+PyRosetta) |
| Regenerate with | `claude "Read PROJECT_CONTEXT.md for regeneration instructions, then regenerate it in full by reading the entire codebase. Preserve the Changelog section and append a new entry."` |

This file is the single source of truth for project state. It should be regenerated after every build session. All source of truth lives in the `.py` files, not here.

---

## 1. Project Overview

StructureBot is a Windows-native natural-language interface for UCSF ChimeraX 1.11.1, running on Python 3.14 (main venv) with a GPU/ML delegation layer in Python 3.12 (venv312). Users type free-text biology requests ("suggest mutations to improve solubility of chain A, avoiding interfaces") into a Rich-console REPL; the system translates those requests via the Anthropic Claude API (claude-sonnet-4-6 with prompt caching), routes them through a pipeline of computational bridges (CamSol, ESM-2, ProteinMPNN, PyRosetta, DynaMut2, disulfide/proline/glycan/cavity/salt-bridge analysers), and executes the resulting ChimeraX commands via the ChimeraX REST API on port 60001. The application also supports a `--script` batch mode and session persistence via `session.json`.

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
  │  4. User confirmation / auto-proceed countdown
  │  5. ChimeraXBridge.run_commands()  ← initial viz commands
  │  6. ToolRouter.execute()    ← computational pipeline
  │     ├─ CamsolBridge         ← local algorithm, no network
  │     ├─ EsmBridge            ← delegates to venv312 subprocess
  │     ├─ ESMFoldBridge        ← delegates to venv312 subprocess
  │     ├─ RosettaBridge        ← DynaMut2 API or WSL2 PyRosetta
  │     ├─ MutationScanner      ← orchestrates CamSol+ESM+Rosetta
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
| `venv/` | 3.14 | Main process: Anthropic SDK, Rich, BioPython, requests, rosetta_bridge, all bridges |
| `venv312/` | 3.12 | GPU/ML delegation: torch 2.11.0+cu128 (RTX 5070 Ti, sm_120 Blackwell), ESM-2 inference, ESMFold inference |

The main venv cannot run GPU inference because no PyTorch cu128 build exists for Python 3.14. ESM-related bridges spawn `venv312/Scripts/python.exe` as a subprocess with JSON I/O (`--input`, `--output` temp files). The subprocess has **no imports from the project** — it only imports `torch`, `transformers`, `esm`, and stdlib.

### WSL2 Layer (PyRosetta)

When `ROSETTA_BACKEND=local`, `RosettaBridge._run_rosetta_local()` builds a standalone Python worker script as an f-string, writes it to a Windows temp file, translates the path to `/mnt/c/...` form, and runs it via `WSLBridge.run_python_script()` → `wsl.exe --distribution Ubuntu-24.04 --exec bash -c "{PYROSETTA_PYTHON} '{wsl_path}'"`. Results are returned as a JSON file at `/tmp/rosetta_ddg_{hash}.json`, copied back to Windows via `wsl.exe`.

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
| `tool_router.py` | `ToolRouter`, `ToolStepResult` | ✅ Complete | Dispatches 13 tool types; handles MPNN+ESMFold combined pipeline; FASTA export; sequence display fast-path |
| `session_state.py` | `SessionState`, `parse_pdb_header()`, `fetch_rcsb_metadata()` | ✅ Complete | Persists all tool results, scan results, rosetta jobs, disulfide/glycan/proline/cavity/salt-bridge results; save/load/snapshot/restore |
| `chimerax_bridge.py` | `ChimeraXBridge`, `find_chimerax()` | ✅ Complete | REST API on port 60001; blank-image post-save guard; `run_commands()` |
| `wsl_bridge.py` | `WSLBridge`, `PYROSETTA_PYTHON` | ✅ Complete | `PYROSETTA_PYTHON="/home/andre/pyrosetta_env/bin/python"`; default distro `Ubuntu-24.04`; `check_pyrosetta()` uses `chr(79)+chr(75)` to avoid quote-escaping |
| `rosetta_bridge.py` | `RosettaBridge`, `_select_backend()` | ⚠️ Known issues | 4 backends (dynamut2/empirical/pyrosetta-stub/local-WSL2); see §8 for audit findings; PyRosetta backend is the active WSL2 path |
| `esm_bridge.py` | `EsmBridge` | ✅ Complete | ESM-2 `esm2_t6_8M_UR50D` default; delegates GPU inference to `esm_worker.py` via venv312 subprocess; disk cache `cache/esm_{hash}.json` |
| `esm_worker.py` | standalone subprocess script | ✅ Complete | No project imports; writes JSON result file; run by venv312 python |
| `esmfold_bridge.py` | `ESMFoldBridge` | ✅ Complete | Primary: venv312 GPU via `esmfold_worker.py`; fallback: ESM Atlas API; `compare_to_wildtype()`, `check_disulfide_foldability()` |
| `esmfold_worker.py` | standalone subprocess script | ✅ Complete | HuggingFace `facebook/esmfold_v1`; no project imports; pLDDT normalisation guard (×100 if mean < 2.0) |
| `mutation_scanner.py` | `MutationScanner` | ✅ Complete | CamSol+ESM+Rosetta pipeline; combined score `0.5×(-ddG) + 0.3×camsol_delta + 0.2×esm_tolerance`; Pro/Cys exclusion; interface protection |
| `camsol_bridge.py` | `CamsolBridge` | ✅ Complete | Local CamSol algorithm; window=9, β=3.0; no network required; web-API fallback via `PROTEIN-SOL_URL` |
| `disulfide_bridge.py` | `DisulfideBridge` | ✅ Complete | Cβ-Cβ geometry (4.5 Å cutoff) + ESM tolerance + DynaMut2 stability; combined score 0.4/0.3/0.3 |
| `proline_bridge.py` | `ProlineBridge` | ✅ Complete | φ-angle scoring; DSSP or Ramachandran fallback; functional residue exclusion |
| `glycan_bridge.py` | `GlycanBridge` | ✅ Complete | NXS/T sequon detection + SASA + SS + ESM + projection scoring; engineered sequon suggestion; NetNGlyc integration |
| `netnglyc_bridge.py` | `predict_glycosylation()`, `integrate_with_glycan_candidates()` | ✅ Complete | DTU NetNGlyc 1.0 REST API; OST recognition scoring; harmonic mean integration |
| `assembly_analyser.py` | `AssemblyAnalyser`, `fetch_assembly_info()` | ✅ Complete | RCSB assembly API; ChimeraX zone-select for interface detection (5 Å CA); monomer/multimer mode |
| `salt_bridge_bridge.py` | `SaltBridgeBridge` | ✅ Complete | BioPython + FreeSASA; Asp/Glu↔Arg/Lys/His within 4 Å |
| `cavity_bridge.py` | `CavityBridge` | ✅ Complete | BFS clustering on buried Cα; SASA < threshold; approximate volume (n_residues × 15 Å³); assembly-aware |
| `structural_utils.py` | `extract_backbone_angles()`, `compute_sasa()`, `compute_projection_score()`, `classify_sequon_geometry()` | ✅ Complete | Shared geometry utilities; used by `glycan_bridge.py` and `proline_bridge.py` |
| `rfdiffusion_bridge.py` | `RFdiffusionBridge` | 🔲 Stub | Documented stub; returns helpful error unless `RFDIFFUSION_DIR` set; Python 3.9-3.11 only |
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
Rationale: `wsl.exe` is a Windows console app. Without these flags it calls `SetConsoleMode()` on the inherited stdin handle, which permanently disables `ReadConsole()` for the rest of the process lifetime, breaking the Rich REPL.

### Path handling rule
All file paths passed to ChimeraX commands **must** use `.as_posix()` to produce forward slashes. ChimeraX on Windows rejects backslashes in `save "..."` and `open "..."` commands.
```python
cx_fwd = Path(some_path).as_posix()
result = bridge.run_command(f'save "{cx_fwd}"')
```

### Worker script rules (ESM worker, ESMFold worker, PyRosetta worker)
Worker scripts that run in subprocesses or WSL2 **must**:
1. Have zero imports from the StructureBot project
2. Communicate exclusively via JSON files (never stdin/stdout for structured data)
3. Write a result file even on exception (`try/except` at the outermost level)
4. Use `flush=True` on all print calls (subprocess stdout is not line-buffered by default)
5. PyRosetta worker: use double-braces `{{...}}` throughout because the script is embedded in a Python f-string

### ChimeraX selector ordering rule
In ChimeraX commands, specifiers must appear in this order to avoid syntax errors:
```
#{model}/{chain}:{residue}@{atom}
```
Example: `#1/A:82@CA` — model first, then chain, then residue, then atom.

### Primary model guard
Commands that should only act on the first loaded model must use `#1` explicitly. The ChimeraX default (no model specifier) acts on all open models, which causes unintended effects when multiple structures are loaded. The translator's static system prompt explicitly instructs Claude to always emit `#1` for single-model operations.

### f-string double-brace rule in worker scripts
The PyRosetta worker script is embedded as a Python f-string. Any literal `{...}` that should appear in the worker code (e.g., dict literals, f-string expressions in worker) must be written as `{{...}}`. The only single-brace `{...}` should be actual f-string interpolations from the outer scope (e.g., `{wsl_pdb!r}`, `{mut_list_json!r}`).

### Error-first return convention (all bridges)
All bridge `analyze()` methods return `ToolStepResult` — they **never raise**. On failure, `result.success = False` and `result.error` contains the message. Callers check `result.success` and degrade gracefully (e.g., fall back to empirical scoring).

---

## 6. Test Suite Summary

**Total: 452 collected** | **440 passing** (12 benchmark tests skip unless WSL2+PyRosetta present)

Run commands:
```bash
# Full suite (excludes slow benchmarks)
pytest tests/ --ignore=tests/test_rosetta_benchmark.py -q

# Including benchmark collection check
pytest tests/ -q --collect-only

# PyRosetta benchmarks (slow, 12–20 min each, requires WSL2+PyRosetta)
pytest tests/test_rosetta_benchmark.py -m benchmark -v --timeout=1800 -s

# Single benchmark spot-check
pytest tests/test_rosetta_benchmark.py -m benchmark -v -s -k "t4_l99a"
```

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_glycan_bridge.py` | 56 | N-glycan sequon detection, SASA scoring, engineered sequon suggestion, NetNGlyc integration |
| `test_tool_router.py` | 47 | Route dispatch, MPNN+ESMFold routing, FASTA export, active-site commands, tool icon registry |
| `test_proline_bridge.py` | 35 | φ-angle scoring, functional residue exclusion, DSSP fallback, BioPython parsing |
| `test_disulfide.py` | 35 | Cβ geometry, dihedral scoring, ESM tolerance, DynaMut2 mock, combined score |
| `test_mpnn_esmfold_pipeline.py` | 29 | MPNN+ESMFold combined pipeline, session routing, pLDDT comparison |
| `test_rosetta.py` | 27 | Backend detection, DynaMut2 HTTP mock, MutationScanner, combined scoring, session persistence, router wiring |
| `test_proteinmpnn.py` | 26 | ProteinMPNN subprocess call, JSON output parsing, error paths |
| `test_esmfold.py` | 25 | ESMFold local/atlas paths, pLDDT normalisation, foldability risk thresholds |
| `test_tools.py` | 24 | Integration tests: CamSol, ESM, ChimeraX commands, DynaMut2 stub |
| `test_assembly.py` | 21 | RCSB assembly metadata, monomer/multimer mode, interface detection mock |
| `test_cavity_bridge.py` | 20 | BFS cavity clustering, SASA burial, volume estimation, interface flagging |
| `test_netnglyc_bridge.py` | 19 | OST score parsing, harmonic mean integration, API mock |
| `test_structural_utils.py` | 17 | `extract_backbone_angles()`, `compute_sasa()`, `compute_projection_score()` |
| `test_wsl.py` | 12 | `WSLBridge` availability, path translation, `run_command()` (skip if no WSL2) |
| `test_rosetta_benchmark.py` | 12 | PyRosetta ddG benchmarks vs ProThermDB; correlation analysis (skip if no WSL2+PyRosetta) |
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
| V82A ddG (DynaMut2) | ~+1.5 to +2.0 kcal/mol (Mahalingam et al.); used as benchmark reference in `test_rosetta_benchmark.py` |
| Assembly analysis | 1HSG detected as homodimer (A2 stoichiometry); interface residues between chains A+B stored in session |
| Disulfide candidates | Chain A↔B candidates with Cβ-Cβ < 4.5 Å ranked by geometry+ESM+stability |
| Proline scan | φ-angle candidates on chain A; functional residue exclusion when active-site set |
| Glycan scan | NXS/T sequons on chain A scored for surface exposure, SS content, ESM tolerance |
| ProteinMPNN redesign | Chain A sequences generated; 3 top designs validated with ESMFold |
| Salt bridge analysis | Asp/Glu↔Arg/Lys/His contacts within 4 Å on chain A |
| Cavity detection | Internal voids in chain A ranked by burial depth and size |

Full pipeline validation script: `scripts/validate_full_pipeline.txt`
```
open 1HSG → solubility scan → proline scan → glycan scan →
salt bridges → cavities → MPNN redesign → show sequences → export
```

---

## 8. Known Issues and Technical Debt

### From `scripts/rosetta_validation_notes.md` (PyRosetta protocol audit)

| Issue | Severity | Detail |
|-------|----------|--------|
| Single relax trajectory | ⚠️ Medium | 1 trajectory per mutation; Kellogg 2011 recommends 50. Adds ~1–2 kcal/mol stochastic noise. Expected RMSE ~1.5–2.5 kcal/mol vs ~1.0 for 50-replicate protocol |
| No large backbone flexibility | ⚠️ Medium | FastRelax moves sidechains + minimises backbone but can't model local unfolding |
| Implicit solvation | ⚠️ Low | `ref2015` uses Lazaridis-Karplus; buried charged residue mutations systematically off |
| Homodimer chain context | ⚠️ Low | 1HSG chains A+B both loaded; interface contacts from B on A mutations are present but not explicitly managed |
| Runtime per mutation | ℹ️ | ~12–20 min per mutation (symmetric 3+3 cycle FastRelax); 50 mutations ≈ 10–17 hours |

### From code review (TODO/FIXME/stub findings)

| Location | Issue |
|----------|-------|
| `rfdiffusion_bridge.py` | Entire module is a documented stub; returns error unless `RFDIFFUSION_DIR` configured; requires Python 3.9-3.11 |
| `rosetta_bridge.py` — `_run_pyrosetta()` | Backend A (`pyrosetta` mode, line ~1056) is a documented stub for direct PyRosetta import (no Python 3.14 wheel); the active WSL2 path (`local` backend) works correctly |
| `rosetta_bridge.py` docstring line 1108 | Still references `Ubuntu-22.04` in `_run_rosetta_local()` docstring (the runtime code correctly uses Ubuntu-24.04 via wsl_bridge defaults) |
| `main.py` line 327 | WSL2 availability message still says `wsl --install -d Ubuntu-22.04` (display string, not functional) |
| `esmfold_worker.py` | `compute_tm` may still produce NaN on very short sequences despite existing guard (`fix(esmfold): mixed precision` commit) — `ESMFOLD_FORCE_COLD_TIMEOUT=False` is correct default |
| `rosetta_bridge.py` worker | `cleanATOM` call uses `cleanATOM(pdb_path, cleaned_path)` — fixed from incorrect keyword-arg version; not battle-tested yet on real PDBs with HETATM |
| `diag.py` | One-off diagnostic script in project root; not imported by anything; safe to ignore or delete |

---

## 9. Build Queue

Prioritised by impact and readiness:

| Priority | Item | Rationale |
|----------|------|-----------|
| 1 | **Run PyRosetta benchmark suite** | 12 benchmark tests exist against ProThermDB mutations (1BNI, 1UBQ, 2SNS, 2LZM, 1HSG). Currently untested. Run once to establish Pearson r and RMSE baseline. `pytest tests/test_rosetta_benchmark.py -m benchmark -v -s` |
| 2 | **Fix Ubuntu-22.04 display strings** | `_run_rosetta_local()` docstring (line 1108) and `main.py` startup message (line 327) still say Ubuntu-22.04. Low risk but causes confusion |
| 3 | **Multi-replica PyRosetta averaging** | Run 3 replicates per mutation and average ΔΔG to reduce ~1–2 kcal/mol trajectory noise. Would bring expected RMSE from ~2.0 down to ~1.5. Cost: 3× runtime |
| 4 | **RFdiffusion activation** | Documented stub awaiting Python 3.9-3.11 environment setup and ~20 GB weight download. Would enable de novo binder design. Blocked on separate Python venv |
| 5 | **LigandMPNN integration** | `PROTEINMPNN_DIR` supports LigandMPNN (ligand-aware redesign). Needs testing on 1HSG (MK1 ligand) to verify ligand-context designs differ from vanilla MPNN |
| 6 | **ProteinMPNN scan hotspot mode** | ProteinMPNN can design only a specified region (hotspot residues). Currently whole-chain redesign only. Would combine with assembly interface data to redesign only non-interface surface |
| 7 | **PyRosetta interface ΔΔG** | Current protocol scores single-chain ΔΔG. For interface mutations (e.g. 1HSG V82 which contacts chain B), an `InterfaceAnalyzerMover` protocol would give more accurate results |
| 8 | **CamSol web-API fallback testing** | `PROTEIN-SOL_URL` env var enables Protein-Sol web API fallback. Untested. Low priority — local algorithm is adequate |

---

## 10. External Dependencies

### Main venv (`venv/`, Python 3.14)

| Package | Version constraint | Purpose |
|---------|-------------------|---------|
| `anthropic` | `>=0.40.0` | Claude API (translator, prompt caching) |
| `requests` | `>=2.31.0` | DynaMut2 API, RCSB API, ChimeraX REST, ESM Atlas fallback |
| `rich` | `>=13.7.0` | Console REPL, tables, panels |
| `biopython` | [verify] | PDB parsing in disulfide/proline/cavity/salt-bridge bridges |
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

**License note:** PyRosetta requires a free academic license from [https://www.rosettacommons.org/software/license-and-download](https://www.rosettacommons.org/software/license-and-download). Commercial use is not permitted under this license.

### External Services

| Service | URL | Auth | Usage |
|---------|-----|------|-------|
| Anthropic API | `api.anthropic.com` | `ANTHROPIC_API_KEY` | Translation (every request) |
| DynaMut2 | `biosig.lab.uq.edu.au/dynamut2/api` | None | ddG scoring (default backend) |
| RCSB PDB | `data.rcsb.org/rest/v1/` | None | Assembly metadata, chain info |
| NetNGlyc 1.0 | `services.healthtech.dtu.dk` | None | OST recognition prediction |
| ESM Atlas | `esmatlas.com/api/fold` | None | ESMFold fallback (when local fails) |
| HuggingFace | `huggingface.co` | None | ESMFold model weights (`facebook/esmfold_v1`) |

### Not pip-installable (must clone and configure)

| Tool | Setup | Config |
|------|-------|--------|
| ProteinMPNN | `git clone https://github.com/dauparas/ProteinMPNN` | `PROTEINMPNN_DIR=C:\Users\andre\documents\structurebot\ProteinMPNN` |
| RFdiffusion | `git clone https://github.com/RosettaCommons/RFdiffusion` + `bash scripts/download_models.sh` | `RFDIFFUSION_DIR=<path>` (not yet set) |

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
| HuggingFace model cache | `~/.cache/huggingface/hub/models--facebook--esmfold_v1/` (default HF location) |
| Session JSONL logs | `<root>\logs\session_YYYYMMDD_HHMMSS.jsonl` |
| Named sessions | `<root>\sessions\{name}.cxs` + `{name}.json` |
| Live session | `<root>\session.json` |
| Worker debug dump | `%TEMP%\structurebot_worker_debug.py` (written each time local Rosetta runs) |
| PDB download cache | `<root>\cache\{PDBID}.pdb` (benchmark tests) |
| Benchmark results | `<root>\scripts\benchmark_results.json` |
| Full pipeline script | `<root>\scripts\validate_full_pipeline.txt` |
| Rosetta audit notes | `<root>\scripts\rosetta_validation_notes.md` |
| ProteinMPNN repo | `C:\Users\andre\documents\structurebot\ProteinMPNN\` |

---

## 12. Session Continuity Notes

**For starting a new co-pilot conversation:**

- **Test count**: 452 collected, 440 passing (12 benchmark tests skip without WSL2+PyRosetta)
- **Last built and validated**: PyRosetta WSL2 backend (`_run_rosetta_local()` in `rosetta_bridge.py`) — full protocol with symmetric 3+3 cycle FastRelax, `cleanATOM` PDB prep, one-letter→three-letter AA conversion, `pose.pdb_info().pdb2pose()` modern API, `ref2015` score function; `wsl_bridge.py` updated with `PYROSETTA_PYTHON` constant and Ubuntu-24.04 default. Also created `tests/test_rosetta_benchmark.py` (11 ProThermDB mutations + correlation analysis) and `scripts/rosetta_validation_notes.md` (full protocol audit).
- **Immediate next item**: Run the benchmark suite once to establish baseline Pearson r and RMSE: `pytest tests/test_rosetta_benchmark.py -m benchmark -v -s -k "t4_l99a"` (single test first); then full suite.
- **In-progress / half-done**: Nothing half-done. The last session committed cleanly at `7cf9807`.
- **Active configuration flags**:
  - `ROSETTA_BACKEND=local` in `.env.local` — routes ddG to PyRosetta via WSL2. Change to `dynamut2` to revert to web API if WSL2 is unavailable.
  - `ESM_USE_VENV312=auto` — GPU inference if CUDA smoke-test passes.
  - `ESMFOLD_FORCE_COLD_TIMEOUT=False` — correct, do not change.
- **API key**: Set in `.env.local` as `ANTHROPIC_API_KEY` — do not commit this file.

---

## 13. Changelog

| Date | Tests | What changed |
|------|-------|-------------|
| 2026-05-29 | 452 | PyRosetta WSL2 backend fully implemented: symmetric FastRelax ddG protocol, cleanATOM PDB prep, modern pdb2pose API, ref2015 score function, wsl_bridge PYROSETTA_PYTHON constant + Ubuntu-24.04 default; benchmark test suite (11 ProThermDB mutations) + rosetta_validation_notes.md created; PROJECT_CONTEXT.md generated |
| 2026-05-28 | 440 | feat(netnglyc): NetNGlyc 1.0 OST recognition integration (Task 3) |
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
