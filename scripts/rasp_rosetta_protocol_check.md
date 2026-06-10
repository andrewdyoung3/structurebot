# RaSP vs deployed-Rosetta protocol check (Task 6 — INVESTIGATION + PROPOSAL, NOT IMPLEMENTED)

**STOP FOR REVIEW.** This reports findings and proposes options for the §9 one-physics-axis
handoff. It changes **no** handoff logic, weights, or protocol. The decision is attended.

## Question

The dry-run found RaSP ~ Rosetta ≈ 0.61 — *not* the near-redundancy §9's handoff assumes
("RaSP and Rosetta are ONE physics axis; RaSP is a trained Rosetta surrogate; never both
at once"). Hypothesis: the **deployed** Rosetta (WSL PyRosetta) runs a **different protocol**
than the Rosetta that RaSP was trained to approximate.

## Evidence

### 1. Correlation re-examined (computed from `cache/stability_datagen/rows.jsonl`, n≈917, 1PGA+T4L)

| Comparison | n | Pearson | Spearman |
|---|---|---|---|
| RaSP vs **deployed** Rosetta — ALL | 917 | **+0.617** | +0.670 |
| &nbsp;&nbsp;excl clash artifacts (rosetta ≤ +10 REU) | 847 | +0.683 | +0.608 |
| &nbsp;&nbsp;excl prolines (mut ≠ P) | 863 | +0.709 | +0.634 |
| &nbsp;&nbsp;excl clash **and** prolines | 819 | +0.680 | +0.592 |
| **RaSP vs Rosetta REFERENCE** (`rosetta_ref_ddg`, RaSP's training target) | 907 | **+0.887** | +0.864 |
| RaSP vs experiment (excl clash) | 847 | +0.734 | +0.616 |
| deployed Rosetta vs experiment (excl clash) | 847 | +0.628 | +0.589 |

**The 0.61 is real and stable** (≈0.62 here). Excluding clash artifacts / prolines moves it
only modestly (0.62 → 0.68–0.71 Pearson) — **artifacts are not the explanation.**

**The decisive contrast:** RaSP correlates **0.89** with the Rosetta *reference shipped with
the dataset* (`rosetta_ref_ddg`) but only **0.62** with the *deployed* Rosetta. The shipped
reference IS the cartesian_ddg output RaSP was trained to approximate (matches RaSP's paper
~0.86, and our own prior RaSP-port validation: Pearson 0.88 / Spearman 0.87 vs this reference,
§13). So **RaSP is a faithful surrogate — of cartesian_ddg, not of what we deploy.**

### 2. The two protocols differ (already documented in §7)

- **Deployed Rosetta** (`rosetta_bridge.py:1449-1464`, per §7): a **manual symmetric-relax**
  protocol — clone cache-relaxed WT → MutateResidue → **torsion-space `FastRelax`** of the
  mutant; independently FastRelax a re-cloned WT; subtract **full-pose plain `ref2015`** total
  scores; median over trajectories. Explicitly **NOT `cartesian_ddg`, NOT `ref2015_cart`, NOT
  cartesian minimisation**; raw uncalibrated REU; static-water clash over-destabilisation
  (the clash tail Task 5 flags). (The stub docstrings still *say* "CartesianDDG/ref2015_cart"
  — §7 flags these as aspirational labels the running code never matched.)
- **RaSP's training target**: canonical Rosetta **`cartesian_ddg` / `ref2015_cart`**
  (Park 2016) — cartesian relax + minimisation. This is what produced the `ddG_Rosetta`
  reference RaSP ships and learns from.

So the deployed estimator and RaSP's target are a **different score-function variant
(`ref2015` vs `ref2015_cart`), a different conformational space (torsion FastRelax vs
cartesian), and a different ddG construction** — three stacked differences.

## Conclusion

The hypothesis is **confirmed with high confidence.** The 0.62 is **not** evidence that RaSP
carries physics information independent of Rosetta. It is the gap between two *different physics
protocols*: RaSP ≈ cartesian_ddg (0.89, near-redundant), deployed Rosetta ≈ torsion-FastRelax-
ref2015. §9's "RaSP is a trained Rosetta surrogate" is true — **but the surrogate relationship
holds against cartesian_ddg, which we do not deploy.** As deployed, RaSP and Rosetta are two
~0.62-correlated physics estimators, not one redundant axis.

## Proposal (for attended decision — pick one; nothing changed yet)

The §9 handoff ("physics = RaSP-if-no-Rosetta ELSE Rosetta; never both; RaSP−Rosetta delta =
proxy-QC") rests on a redundancy that **does not hold for the deployed pair**. Options:

1. **Align the deployed protocol to cartesian_ddg / ref2015_cart** (the §7 backlog already
   contemplates this). Then RaSP genuinely surrogates the deployed gold (≈0.89), the handoff
   premise is restored, and the RaSP−Rosetta delta becomes a meaningful proxy-QC. Largest
   effort; also fixes the calibration + clash issues §7 notes. **Recommended IF physics
   absolute accuracy matters.**
2. **Keep the handoff, fix the proxy-QC reference.** Treat physics as one axis (don't double-
   count two physics estimators), but compute the proxy-QC delta as **RaSP vs the cartesian_ddg
   reference**, not RaSP vs the torsion deployment — otherwise the delta measures protocol
   mismatch, not RaSP error. Cheapest; documents the caveat without re-running anything.
3. **Model them as two partially-independent physics estimators** (ρ≈0.62), weighting by
   independence×confidence rather than a hard handoff. Most faithful to the data, but adds a
   correlated-estimator weighting the §9 model doesn't yet have, and risks over-counting
   physics relative to the ML/dynamics axes.

**My recommendation:** decide the deployed-protocol question first (it is a *known* §7
deviation, not new) — it determines everything. If we move the deployment to canonical
cartesian_ddg, take **Option 1** and the handoff stands. If we keep the torsion-space protocol
for speed, take **Option 2** now (handoff + corrected proxy-QC reference) and revisit Option 3
only if the benchmark shows the deployed Rosetta and RaSP disagreeing in a way that *helps* vs
experiment. Do **not** silently keep treating 0.62 as redundancy.

## Caveats

- All correlations are on the **1PGA-dominated dry-run set** (one protein + T4L; ThermoMPNN-
  leakage-contaminated). The *direction* (RaSP≈cartesian_ddg ≫ RaSP≈deployed) is robust and
  protocol-grounded, but exact magnitudes should be re-confirmed on the diverse Task-1 set.
- RaSP's cartesian_ddg training target is established from the RaSP paper + the shipped
  `ddG_Rosetta` reference; the label-generation protocol is not re-derivable from the local
  RaSP repo code (only the trained model + reference labels ship locally).

---

# Task B — protocol-resolution prep (added; no production change)

## B1 — Why the deployed Rosetta uses torsion-space FastRelax, not cartesian_ddg

Inspected §7, `scripts/rosetta_validation_notes.md`, the Rosetta config, and git history.
**Honest finding: the deviation was never recorded as a deliberate "we evaluated
cartesian_ddg and rejected it for speed/memory" decision.** The evidence:

- The local Rosetta path was an **incremental from-scratch PyRosetta implementation**
  (first appears `68b931e feat: add Rosetta/Robetta ddG bridge`, then `7cf9807`/`acbfaf9`):
  cleanATOM → FastRelax (torsion) → MutateResidue + FastRelax → subtract full-pose
  `ref2015` scores. The docstrings call it "CartesianDDG"/"ref2015_cart" but §7 explicitly
  flags those as **aspirational labels the running code never matched** — i.e., the name was
  intended, the cartesian implementation never landed. So the "choice" was largely
  **incidental**, later surfaced by the §7 audit, not a costed trade-off.
- That said, the constraints that **would** bound a cartesian deployment are real and
  documented, and plausibly explain why nobody upgraded it:
  - **Runtime:** the torsion arm already costs ~12–20 min/mutation (validation notes L154);
    cartesian minimisation + `ref2015_cart` is heavier still.
  - **WSL memory envelope / worker cap:** the deployment runs PyRosetta inside WSL2 under a
    bounded RAM budget — `ROSETTA_WSL_MEM_BUDGET_MB=12000`, per-worker footprint
    `500 + 2.2·n_res` MB capped at `ROSETTA_WORKER_FOOTPRINT_MB=1200`, workers =
    `min(8, physical−2, mem_budget/footprint)` (added in the 2HHB pose-size worker-cap fix
    `929df5f`, to avoid oversubscribing WSL into swap). A heavier per-worker protocol shrinks
    the worker pool and/or risks the swap wall the cap exists to prevent.
- §7 also records that adopting **canonical cartesian_ddg wholesale** (cartesian relax +
  `ref2015_cart` + averaging + the 2.94 REU→kcal/mol factor) is already the **stated backlog
  path** if calibrated absolute magnitudes ever become a requirement.

**Conclusion:** the torsion-space protocol is best described as an un-upgraded initial
implementation, not a deliberate speed/memory optimisation — but the memory/worker-cap and
runtime envelope are the real reasons it has not been upgraded, and they bound B2/B3.

## B2 — cartesian_ddg feasibility in the current environment

Probed live (WSL2 Ubuntu-24.04 + `~/pyrosetta_env`):

- `ref2015_cart` score function: **available**.
- `CartesianddGMover` class: **NOT exposed** in this PyRosetta build (ImportError). The
  cartesian protocol must therefore be reproduced **manually via cartesian-mode FastRelax**
  (`FastRelax.cartesian(True)` + `ref2015_cart`) — structurally the same manual pattern the
  deployed code already uses for the torsion protocol.
- Bounded run (1PGA, 56 res, 1 cartesian relax cycle): **runs in ~4 s, peak RSS ≈ 832 MB.**
- Single-mutation cartesian_ddg smoke (1PGA M1A / K50P): **M1A +6.34, K50P +9.45 REU**
  (both destabilising; correct sign). Notably K50P cartesian +9.45 vs the deployed torsion
  **+97.18** — cartesian relieves the proline clash the torsion arm (static waters, no
  cartesian min) blows up on. (Reported as a protocol difference, NOT a verdict.)

**Does it re-hit the §7 constraint?** At calibration-set protein sizes — **no.** 832 MB for
56 res sits under the 1200 MB per-worker footprint cap and far under the 12 GB WSL budget.
S669/Ssym are small-to-medium domains (mostly <200 res), so cartesian is feasible there.
**Caveat:** cartesian is ~1.3× the torsion memory model's prediction at 56 res and scales with
pose size, so for large multimers it would shrink the worker pool — a conservative footprint
estimate should be used if the benchmark ever runs large poses, and per-mutation runtime will
exceed the torsion arm's. For the diverse-but-small calibration set, none of this blocks.

## B3 — benchmark-only cartesian arm: WIRED (production untouched)

Wired `scripts/rosetta_cartesian_bench.py` (standalone; does **not** import or modify
`rosetta_bridge.py`, the deep-tier path, the handoff, or any default) + a `--rosetta-cart`
flag on the data-gen harness. When set, the cartesian arm runs on the **same** mutations as
the deployed torsion arm and writes `rosetta_cart_ddg` ALONGSIDE `rosetta_ddg`, so the
overnight benchmark can score **both protocols vs experiment** and let the data decide.
Default OFF → the row schema is byte-for-byte unchanged. Verified: standalone smoke produces
correct-sign cartesian ddG; harness wiring tested (off → field absent; on → field written),
suite green. **No protocol decision is made here** — only the capability + rationale are
prepared, per the brief.
