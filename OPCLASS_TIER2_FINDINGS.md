# TIER 2 findings — selection op-class & double-mutant residue-scope

Autonomous session on `feat/opclass-color-selection` (2026-06-09). Both TIER 2
items were STEP-0 probed and **STOPPED** (not built) — each crosses the "non-trivial
/ touches internals" threshold the session brief set for unsupervised work. TIER 1
(color op-class, translator cap latch) shipped on this branch.

---

## ITEM 2 — Selection op-class — STOPPED

### What exists
`tool_router.handle_selection_command()` (called from `main._dispatch_input`
BEFORE `_handle_request`) is a **consume-the-live-selection** path: it reads the
user's current ChimeraX selection over REST (`selection.read_selection`) and
feeds it into report / scan(CamSol) / redesign(ProteinMPNN). `_detect_selection_intent`
matches only "...the selection" / "...the selected" phrasings
(`_SELECTION_REPORT/SCAN/REDESIGN_PHRASES`).

### Why a `select.*` op-class does NOT route through it
The proposed op-class (`select.chain`, `select.residue_range`, `select.zone`) is the
**opposite direction** — it *creates* a selection by emitting a `select` command.
It shares none of the read-live-selection machinery. So it would NOT extend
`handle_selection_command`; it would be a **new additive gate** mirroring color
(SELECT_REGISTRY + gate in `_handle_request`/`route` + `_run_selection`). The two
are disjoint and ordering-safe (consume-path runs first; "select chain A" falls
through to the create-path), so the working selection handler is NOT at risk.

### Why it was STOPPED anyway
The **payoff** named for this migration is "kill zone-vs-resnum miscategorization",
which lives in `select.zone`. Deterministically rendering zone selections is the
hard, high-risk part:
- Zone requires parsing a **distance**, a **reference spec** ("the ligand" / "chain B"
  / "residue 50" / "the active site"), the **operator** (`:<` inside / `:>` outside,
  `@<` atom-zone), and connectives ("within / around / of"). That is a parameterized
  parser, not a fixed template like the viewer/color renders.
- This is **already handled maturely** by the translator + the backend-agnostic
  `_sanitize_zone_syntax` guard (verified working in `tests/test_ollama_backend.py`
  test B: `select #1/A & (zone #1/B 4.5)` → `select (#1/A & ~ligand & ~solvent &
  ~ions) & ((#1/B & ~ligand & ~solvent & ~ions) :<4.5)`).
- There is **in-flight work on this exact area**: a stash `WIP on
  fix/translator-zone-operators` ("emit ChimeraX zone operators, never Chimera-1
  `zone`"). A second, deterministic zone renderer would create two competing
  sources of truth for zone syntax and risk colliding with that work.

`select.chain` / `select.residue_range` are trivially simple but **low value** — a
bare "select chain A" already works via free-translation + the chain-scope guard,
and there is no documented bug there. An op-class covering only those (deferring
zone) would deliver little and still leave the named payoff unaddressed.

### Recommendation
Defer until the `fix/translator-zone-operators` work lands. Then, if still desired,
build `select.chain` + `select.residue_range` as a small additive op-class and keep
**zone on the translator path** (don't duplicate it). Revisit "zone-vs-resnum
miscategorization" as a translator-prompt / guard issue, not an op-class render.

---

## ITEM 4 — Double-mutant residue-scope ("residues 1-3") — STOPPED

### Where the range is dropped
**It is never parsed.** Not "parsed but not threaded":
- `_run_mutation_scan` (tool_router.py ~3181) consumes `model_id, chain, focus,
  analysis_mode, sequence, pdb_path` + a fixed `filters` set (thresholds / weights /
  `binding_site_residues`). There is **no** positions/range/scope key read.
- `_run_double_mutant` (tool_router.py ~3304) does **no** range parsing of
  `user_input`; it reads `session.get_scan_result(model_id)` — the previously stored
  **full-chain** single-point scan — and pairs across **all** of it.
- The translator's `mutation_scan` tool_inputs schema (translator.py ~806/839) has
  **no** residue-range field (only `model_id`/`chain`/`focus`/`analysis_mode`).
  `design_positions` exists only for `proteinmpnn`.
- `MutationScanner.scan()` (mutation_scanner.py ~197) has **no include-positions
  parameter** — only `protected_residues`, which is an **exclude** list (and is
  already overloaded for interface protection in multimer mode).

This matches the live-verify symptom (single scan reported full chain; double-mutant
paired residues 6-12) — the "residues 1-3" range never reaches any scan stage.

### Why it was STOPPED (not a clean thread-through)
A correct fix is multi-file and touches scan internals + a semantic decision:
1. **Translator**: add a `scan_positions` / `residue_range` field to the
   `mutation_scan` schema + prompt instruction + an example (parse "residues 1-3",
   "positions 10-20", "1-3 of chain A").
2. **`_run_mutation_scan`**: consume it and thread it down.
3. **`MutationScanner.scan()`**: add an **include-positions** parameter and honor it
   in the candidate-generation loop (scanner internals — distinct from the existing
   exclude semantics; must not clobber `protected_residues`).
4. **`_run_double_mutant`**: decide **reuse-vs-rescan** — it currently reuses a stored
   full-chain scan, so even a range-restricted single scan would not constrain the
   pairs unless the double-mutant funnel either (a) re-scans with the restriction, or
   (b) filters the stored mutations list by the range. This is a design choice, not a
   mechanical change.

Per the brief ("if the fix touches scan internals broadly → STOP"), this qualifies.
Live verify would also need a slow Rosetta/PyRosetta run (WSL2) → would be PENDING
regardless.

### Recommendation (smallest correct shape, for a supervised pass)
- Add `scan_positions: Optional[List[int]]` to `MutationScanner.scan()` as an
  **include** filter (intersect candidate positions with it when present; leave the
  exclude path untouched).
- Parse the range in the translator (new schema field) AND, for the double-mutant
  reuse path, filter `mutations` by the range in `_run_double_mutant` so a reused
  full-chain scan is still constrained (decision: filter-stored, do NOT silently
  re-scan).
- Pin with a test: "scan double mutants on residues 1-3" → exactly 3 positions enter
  the pairing, not the full chain.
