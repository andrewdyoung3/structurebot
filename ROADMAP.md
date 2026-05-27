# StructureBot Roadmap

## Session 9 (next)
- ESMFold live validation on real mutation candidates (model weights download)
- WSL2 install + local Rosetta (PyRosetta via WSL2)
- Mid-execution ESC (background thread cancellation)

## Near term
- Salt bridge analysis module
- Cavity detection (pyCAST)
- Duration estimation (pre-scan time estimate)
- Session log analysis improvements (timing data pipeline)
- Glycan bridge: full SASA-based surface exposure scoring (current stub uses ESM only)

## Future
- RFdiffusion activation (~20GB weights)
- AlphaFold2 local
- Mac mini / WSL2 GPU path

## Session history
- S1: ChimeraX bridge
- S2: LLM translator + REPL
- S3: CamSol + ESM-2 + tool router
- S4: DynaMut2 + MutationScanner
- S5: Assembly analyser + multimer
- S6: Disulfide + GPU venv312
- S7: Dihedral fix + ProteinMPNN + RFdiffusion + QoL
- S8A: Parallel DynaMut2 + ESMFold bridge (Atlas) + log analyser + Cys audit
- S8B: ESMFold local GPU inference (venv312 + transformers, 184 tests)
- S9A: proline_bridge (backbone φ/ψ scanner, 27 tests) + glycan_bridge stub + session_state/tool_router hooks (215 tests total)
- S9B: esmfold_bridge cold/warm timeout + Atlas demotion + pLDDT guard (218 tests total)
- S9C: proline routing fix — _PROLINE_KEYWORDS + route() override + _dispatch_tool guard + exclusion counts + φ labels (230 tests total)
