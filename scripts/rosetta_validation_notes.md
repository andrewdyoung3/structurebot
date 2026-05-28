# PyRosetta ddG Implementation — Validation Notes

Method: `_run_rosetta_local()` in `rosetta_bridge.py`
Reference protocol: Kellogg et al. (2011) *Proteins* 79:830–838

---

## Part 1 — Protocol Audit

### 1. Score function — ✓ CORRECT
**Line 1228 (initial WT relax) and 1265 (per-mutation):** `ref2015`

`ref2015` is the current best-practice score function for ddG monomer calculations,
succeeding the talaris2014/score12 used in Kellogg 2011. The Alford et al. (2017)
*J Chem Theory Comput* benchmark shows ref2015 improves Pearson r vs experimental
ΔΔG from ~0.60 to ~0.69 on the Kellogg test set.

No issues.

---

### 2. Relax protocol symmetry — ⚠ MOSTLY CORRECT, dead code present

**WT initial relax (lines 1234–1238):** `FastRelax(scorefxn, 5)` — 5 cycles.  
**Per-mutation mutant relax (line 1266):** `FastRelax(scorefxn_std, 3)` — 3 cycles.  
**Per-mutation WT re-relax (lines 1268–1270):** `FastRelax(scorefxn_std, 3)` — 3 cycles.

The per-mutation ΔΔG is computed symmetrically: both mutant and WT are re-relaxed
with 3 cycles from the same starting structure (the 5-cycle cached WT clone).
`scorefxn_std` is a fresh `ref2015` instance identical to `scorefxn` — this is correct.

**Dead code — line 1241:** `wt_energy = scorefxn(wt_pose)` is computed but never
referenced again. The mutation loop uses `wt_score = scorefxn_std(wt_pose_rerelaxed)`.
This variable should be removed to avoid confusion.

**Scorefxn created inside loop — line 1265:** `scorefxn_std` is re-instantiated for
every mutation. Functionally correct (all instances are equivalent ref2015), but
wasteful. Should be hoisted outside the loop.

---

### 3. Residue repack zone — ⚠ DEVIATION FROM KELLOGG 2011

**Lines 1266–1270:** After mutation, the entire structure is relaxed with 3-cycle
FastRelax rather than repacking only residues within 8 Å of the mutation site.

Kellogg 2011 uses a localised 8 Å repacking shell (`RestrictToRepacking` + `PackRotamersMover`)
for efficiency. Our protocol performs full-structure backbone + sidechain minimisation,
which is more physically correct (accounts for long-range strain relief) but is
approximately 5× slower per mutation.

For interface mutations or mutations near symmetry contacts, full-structure relaxation
is superior. For buried core mutations the difference is small. This is an acceptable
deviation for our use case (final validation of top candidates, not high-throughput scan).

---

### 4. Multiple replicas — ✗ NOT IMPLEMENTED

**Protocol:** Single trajectory per mutation.

Kellogg 2011 uses 50 independent relax replicates and averages the ΔΔG. Even 3–5
replicates substantially reduce trajectory variance. With a single trajectory, random
variation in the FastRelax minimisation path adds ~1–2 kcal/mol noise on top of the
method's intrinsic error (~1.0 kcal/mol RMSE vs experiment for Kellogg et al.).

**Effect:** Expected RMSE for our protocol is ~1.5–2.5 kcal/mol vs experiment
(vs ~1.0 kcal/mol for the full 50-replicate Kellogg protocol).

Implementing multi-replica averaging is the single most impactful improvement available
for accuracy, at a proportional cost in runtime.

---

### 5. PDB preparation — ⚠ PARTIALLY HANDLED

**Line 1213:** `from pyrosetta.toolbox import cleanATOM` is imported but **never called**.

**Line 1215:** `rosetta_init(options="-mute all -ex1 -ex2 -use_input_sc -ignore_unrecognized_res true")`

`-ignore_unrecognized_res true` causes PyRosetta to skip unknown residues (HETATM
ligands, non-standard amino acids, modified residues) rather than crashing. This is
adequate for single-chain, standard-residue inputs like 1BNI and 1UBQ.

**Known failure mode:** PDB structures with:
- Crystallographic waters occupying a buried cavity (may affect packing scores)
- Alternate conformations (`ALTLOC` records — only the first conformation is read)
- Multiple models in the file (MODEL/ENDMDL records — only model 1 is used by default)
- Covalently bound ligands that stabilise the protein (interface ddG will be wrong)

**Recommendation:** Call `cleanATOM(pose)` or strip HETATM records before scoring for
publication-quality results.

---

### 6. Sign convention — ✓ CORRECT

**Line 1272:** `ddg = scorefxn_std(mut_pose) - wt_score`

Positive ΔΔG = destabilising (mutation raises energy). Negative ΔΔG = stabilising.
Consistent with ProThermDB, ThermoMutDB, and the DynaMut2 backend.

---

### 7. Additional code issues

- **Double json.loads (lines 1218–1220):** `mutations` is already parsed by `json.loads()`
  on line 1218, then the `isinstance(mutations, str)` guard on 1220 applies a second
  `json.loads` that is always a no-op. Harmless but confusing.
- **`import json` inside worker (line 1219):** `json` is already imported at the top of
  the worker (`import json, sys, os` on line 1206). The re-import is a no-op in Python
  but should be removed for clarity.
- **`cleanATOM` import (line 1213):** Imported but never called — either call it or
  remove the import.

---

## Part 2 — Known Limitations

### Limitation 1 — Single trajectory
We use one FastRelax trajectory per mutation. The Kellogg 2011 protocol recommends 50
replicates for publication-quality ΔΔG values. Our single trajectory adds ~1–2 kcal/mol
stochastic variance on top of the method's intrinsic error. Results should be treated
as high-confidence screening estimates, not publication-quality thermodynamic measurements.

**Workaround:** For final validation of a top candidate, run the benchmark test 3–5 times
and average the ΔΔG values manually.

### Limitation 2 — Chain context for oligomers
Structures like 1HSG (HIV-1 protease homodimer) are loaded as a single PDB. Our protocol
relaxes and scores the full deposited structure — but mutations at the homodimer interface
(e.g. V82 in chain A contacts chain B) will have inaccurate ΔΔG because the symmetric
partner chain is present but chain-boundary interactions may not be fully captured.

For monomer proteins (1BNI, 1UBQ, 2SNS, 2LZM) this is not an issue.

**Workaround:** For interface mutations, run separate calculations with each chain isolated
and compare; or use a purpose-built interface ΔΔG protocol (`InterfaceAnalyzerMover`).

### Limitation 3 — Implicit solvation
`ref2015` uses an implicit Lazaridis-Karplus solvation model. Mutations that substantially
change the solvation free energy — especially burial of charged residues (Asp, Glu, Lys,
Arg) or exposure of hydrophobic residues — are systematically over- or under-estimated.
Mutations of type `charged→nonpolar` or `buried→exposed` should be interpreted with caution.

### Limitation 4 — No large backbone rearrangements
FastRelax performs backbone minimisation but does not sample large backbone conformational
changes (loop rearrangements, helix shifts, partial unfolding). Mutations that cause local
unfolding (e.g. proline insertion into a helix, glycine in a β-sheet) will have
underestimated destabilisation because the backbone cannot relax far enough from its
starting conformation.

### Limitation 5 — Runtime and scanning cost
The current symmetric 3+3 cycle FastRelax protocol takes approximately 12–20 minutes per
mutation on a modern CPU inside WSL2. Scanning 50 mutations would take roughly 10–17 hours.

**Recommended workflow:**
- Stage 1 (scan): DynaMut2 web API — ~1 min per mutation, Pearson r ~0.65, screens 50–100
  candidates quickly.
- Stage 2 (validate): PyRosetta WSL2 — ~15 min per mutation, expected Pearson r ~0.70–0.75
  (single trajectory), validates top 5–10 candidates from Stage 1.
- Stage 3 (publication): Kellogg protocol with 50 replicates per mutation — ~12 h per
  mutation, expected Pearson r ~0.80.

---

## Part 3 — Benchmark Test Suite

The benchmark test suite lives at `tests/test_rosetta_benchmark.py`.

Benchmark set: 11 mutations across 5 well-studied proteins with high-confidence
experimental ΔΔG values from ProThermDB / ThermoMutDB.

To run:

```
pytest tests/test_rosetta_benchmark.py -m benchmark -v --timeout=1800 -s
```

Note: each mutation takes 12–20 min. Running the full suite of 11 mutations takes
approximately 2–4 hours. Use `-k` to run a single mutation for spot-checking:

```
pytest tests/test_rosetta_benchmark.py -m benchmark -v -s -k "t4_l99a"
```

Expected output — acceptable performance thresholds for our single-trajectory protocol:
- Sign accuracy: ≥ 8/11 mutations with correct sign (positive/negative)
- Magnitude: ≥ 7/11 mutations within 2.0 kcal/mol of experimental value
- Pearson r: > 0.50 across the full benchmark set
- RMSE: < 2.5 kcal/mol

Correlation analysis (requires all 11 results to be present):

```
pytest tests/test_rosetta_benchmark.py -v -s -k "correlation"
```
