# Confirmation Surfaces — Design Proposal (echo-back + interactive sequence window)

**Status:** proposal for review — **nothing is implemented**. Date: 2026-06-04.
**Goal:** give the user *verifiable, correctable* views of what the translator is about
to do / has done, so the failure modes the 3-dimension eval exposes — **wrong chain,
off-by-one range, resnum ≠ sequence-position, ambiguous interpretation** — are caught
and overridable *at the point of use* rather than propagating silently. Two surfaces in
one "confirmation" family: a **textual echo-back** and an **interactive sequence
window**.

---

## 0. Survey — the real interaction layer (grounded, not assumed)

**Where a translation surfaces today.** There is **no web/desktop GUI** — the product is
a **Rich-console CLI REPL** (`main.py`) talking to ChimeraX over its REST server
(`:60001`), plus ChimeraX's own windows (the 3D view and its native Sequence Viewer).
The per-request path in `main._dispatch_input` is:

1. `result = self.translator.translate(user_input, session)` → the normalized 7-key
   dict (`commands / explanations / warnings / clarification_needed / confidence /
   tools_needed / tool_inputs`, plus the new `refused`).
2. If `clarification_needed` is set → ask the user and re-translate.
3. **Preview** (before any execution): print `warnings`, then `_show_tool_pipeline`
   (when `has_extra_tools`) and `_show_preview(commands, explanations, confidence)`.
4. **Confirmation gate** — `_confirm_execution(confidence)`:
   - `auto_proceed` + confidence ∈ {high, medium} → a **countdown** auto-proceed
     (`AUTO_PROCEED_DELAY`); any keypress pauses → manual confirm.
   - otherwise → `_manual_confirm()`. `--no-auto-proceed` forces an explicit confirm.
   - an **`edit`** option already exists (`_edit_commands`) to hand-edit the command
     list before it runs.
5. `self.router.execute(result, …)` runs the tools / ChimeraX commands.

→ **A confirmation gate, a command preview, and a hand-edit path already exist.** This
proposal *enriches the preview* (echo-back), *tiers the gate by reversibility*, and adds
a *ground-truth read-back* surface — it does not invent the gate from scratch.

**How the ChimeraX selection is read/displayed now.** `selection.read_selection(run_
command)` runs **`info residues sel`** against ChimeraX's ACTUAL state and parses it into
a `Selection` of `(model, chain, resnum, resname)` with `.by_chain()` / `.resnums(chain)`
/ `.chains` — i.e. **real auth residue numbers + chain labels, read back from ground
truth, already exist.** `tool_router.handle_selection_command` already turns the live
selection into report / scan / redesign actions, and `session.selection` is the
"redesign what I've selected" channel.

**The native Sequence Viewer is already integrated** (`sequence_viewer.py`):
`ensure_sequence_viewer_commands()` opens + *associates* a model's chains in ChimeraX's
Sequence Viewer, after which **bidirectional highlighting is native** (3D-select →
column highlights; drag a sequence span → 3D selects). `build_scf_file` +
`build_scf_runscript` colour per-residue sequence regions (the `.scf` + `runscript`
workaround, because `open *.scf` is unsupported over REST in 1.11.1). `proteinmpnn_
bridge.chain_resnum_to_seqpos` already maps PDB resnum → 1-based chain position (the
exact resnum≠position conversion the eval flags).

**Where a summary attaches.** The natural attach point is the **`result` dict between
`translate()` and the confirm gate** — a new field rendered in `_show_preview`, and (for
faithfulness) a **deterministic post-translate guard** alongside the existing
`_sanitize_zone_syntax` in `CommandTranslator.translate`.

---

## 1. Surface A — Echo-back (textual interpretation)

**What.** The translation carries a **human-readable interpretation of the concrete
operation**, surfaced in the preview before the gate — especially when an *inferential*
request was resolved. e.g. for "redesign the dimer interface":
> *Interpreting "the dimer interface" as **chain-A residues within 4 Å of chain B**, then
> running ProteinMPNN on those positions.*

**Schema.** A new field produced by **both backends** (Claude via prompt, Ollama via the
constrained `format` — exactly how `refused` was just added), distinct from
`commands`/`tool_inputs`. To make the faithfulness check *mechanical* (not NL↔NL), the
recommended shape is **two coupled fields**:

- `interpretation` — the one-sentence human string (what the user reads).
- `interpreted_action` — a small **structured mirror** the check compares to the command:
  `{ "operation": "redesign|select|colour|scan|…", "chains": ["A"], "partner": "B"|null,
     "scope": "interface|selected|20-30|whole-chain|null", "distance": 4|null,
     "tool": "<router literal>" }`.

The human string is shown; the structured mirror exists for the check (and is trivial for
the model to fill, since it already chose those exact values for `tool_inputs`/commands).

**Data flow.** `translate()` returns both fields → `CommandTranslator.translate` runs the
**faithfulness guard** (below) → `_dispatch_input` renders `interpretation` in
`_show_preview` (and, if the guard flagged it, a visible warning + a confidence
downgrade) → existing gate.

**Faithfulness check (make-or-break).** A confident-but-wrong summary is *worse than
none* → treat an unfaithful summary as a **defect**, not a nicety. The check is a
deterministic, pre-execution **description ↔ action consistency** guard that compares
`interpreted_action` (and key tokens of `interpretation`) against the **actually emitted**
`commands` + `tool_inputs`:

- **chain**: every chain named in the summary appears in the command spec
  (`#1/A`, `/A`) / `tool_inputs[...].chain(_id)`; no chain is colored/redesigned that the
  summary didn't mention, and vice-versa.
- **range/scope**: a summary range (`20-30`) equals the command/`design_positions` set;
  `interface` ⇒ a partner-scoped zone or BioPython interface path is actually present;
  `selected` ⇒ the live-selection path; `whole-chain` is never claimed when a scope was
  emitted (or vice-versa).
- **distance**: a summary "within N Å" matches the zone operator value (`:<N`).
- **tool**: `interpreted_action.tool` == the routed literal.

On any mismatch: **drop/neutralise the summary**, append a `warnings` entry
("interpretation did not match the emitted command — review carefully"), and **downgrade
`confidence` to low** so the gate stops auto-proceeding. Lands as a new guard next to
`_sanitize_zone_syntax`; testable model-independently (feed a translation with a summary
that disagrees with its command → guard flags it). **The echo-back never overrides the
command; it can only flag/withhold** — the command stays the source of truth and the
sequence window (Surface B) is the post-execution ground-truth check.

---

## 2. Surface B — Interactive sequence window (spatial verification)

**What.** A sequence track that highlights the **CURRENT SELECTION read back from
ChimeraX's ACTUAL state** — not the LLM's parsed intent — showing **real auth residue
numbers + chain labels**, so `resnum ≠ position` and off-by-one errors in the *command*
are instantly visible. Reuse ChimeraX's **native Sequence Viewer** (already wired by
`ensure_sequence_viewer_commands`) rather than rebuilding a track.

**Read-back, not claim.** The set displayed comes from `selection.read_selection(run_
command)` (`info residues sel`) — ChimeraX's truth after the command ran — rendered as
sequence regions via the existing `build_scf_file` + `build_scf_runscript` region API,
labelled with auth IDs. This is what makes it a *real* check: it shows what the command
**actually** selected (catching an off-by-one or resnum/position slip the echo-back, a
mirror of the model's own parse, would happily confirm).

**Bidirectional + editable.** Selecting in 3D highlights the column and dragging a
sequence span selects in 3D — **already native** once associated. The new behaviour is
making the corrected set *flow back*: the user drags to fix the selection, and the
corrected residues become the **current selection** via the existing `session.selection`
mechanism (`read_selection` after the user's drag → store as the active selection → the
"redesign / scan what I've selected" path consumes it). No new selection model — it ties
into the path that already exists.

**Verification beat.** For selection/scope operations the flow becomes: command runs →
read-back selection rendered in the Sequence Viewer with auth IDs → user *glances*
(ambient) → optionally drags to correct → the corrected selection is what downstream
(redesign/scan) uses. The 3D view + the sequence track show the same ground-truth set.

---

## 3. Design principles

- **Ambient over modal.** Prefer always-visible current-state surfaces the user *glances*
  at (the rendered selection in the Sequence Viewer; the interpretation line in the
  preview) over a confirm-dialog per action — per-action modals breed click-through
  fatigue and the auto-proceed countdown already exists for cheap actions.
- **Tier the gate to reversibility** (refining the current confidence-only gate):
  - *cheap / reversible* (select, colour, view) → **passive display + easy undo** (let
    the auto-proceed countdown stand; the read-back surface is the safety net).
  - *destructive / overwriting* (a redesign that **overwrites** a stored design, deletes,
    saves over a file) → an **explicit gate before execution** (force `_manual_confirm`
    regardless of confidence; the router already knows which tools are heavy/overwriting).
  This is an additive policy layer over `_confirm_execution`, keyed on the routed tool's
  reversibility class — not a new dialog system.
- **Read-back from ChimeraX ground truth, never the LLM's claim.** Surface B reads
  `info residues sel`; the echo-back's faithfulness guard checks the *emitted command*,
  not the model's narration. Together they close the **Accuracy ↔ Functionality gap at
  the point of use**: echo-back catches "the tool/args the model *claims*" vs what it
  *emitted* (accuracy), the sequence window catches what the command *did* (functionality).

---

## 4. Guardrail — entirely out of the eval loop

The confirmation/override layer is a **production feature** and must **NEVER feed a scored
benchmark translation** — the eval measures the **LLM unaided**. Concretely:

- The eval path (`eval_runner.make_*_caller` → `backend.translate(...)`) calls the backend
  **directly** and scores the raw translation (+ the deterministic guard). The echo-back
  faithfulness guard is fine to run there (it only *flags*, never edits the command, and
  it's deterministic, model-independent — it would even be a useful extra signal), **but
  the override path (user edits, drag-corrected selection, the gate) must not touch the
  scored translation.** Surfaces A/B live in `main.py`/ChimeraX, not in
  `eval_runner`/`eval_harness`.
- `interpreted_action` is gold-independent and may be *recorded* in the eval CSV as
  diagnostics, but is **not** a scored dimension and must not change `tool_inputs`/
  `commands` the scorer sees.
- A test should assert the eval callers never invoke the confirmation/override layer.

---

## 5. Phasing & dependencies

**Build order (smaller, lower-risk first):**

1. **Echo-back (Surface A).** A schema field + the deterministic faithfulness guard —
   adjacent to work already done (`refused` field + `_sanitize_zone_syntax` pattern), and
   the highest leverage per line (catches *wrong chain / wrong range / wrong tool claimed*
   before execution). Sub-steps: (1a) add `interpretation` + `interpreted_action` to the
   schema + prompt (both backends) + `_parse_response`; (1b) the faithfulness guard +
   model-independent tests; (1c) render in `_show_preview` + the confidence-downgrade on
   mismatch.
2. **Reversibility-tiered gate.** Small additive policy over `_confirm_execution` keyed on
   the routed tool's reversibility class (force manual confirm for overwriting/destructive
   tools). Independent of Surface B; can land alongside 1.
3. **Interactive sequence window (Surface B).** Larger (UI + the drag-correction →
   `session.selection` round-trip on top of the native Sequence Viewer). Builds on the
   existing `read_selection` + `sequence_viewer` layers; the read-back-render half is
   close to the CamSol region path, the editable-write-back half is the new work.

**Dependencies / sequencing:**

- **Surface A touches the shared translator schema/prompt** (`TRANSLATION_JSON_SCHEMA`,
  `_parse_response`, the system prompt) — the **same surface the in-flight Claude-caller
  fix and the clean-non-action / `refused` work touched.** It must **sequence after any
  in-flight translator changes land** (and after the current full benchmark completes, so
  the baseline isn't measured against a moving schema). A schema change also re-touches
  `is_schema_valid` and the Ollama `format` constraint — coordinate as one change.
- Adding a field changes what the model must emit; **re-baseline / re-validate the
  benchmark** after Surface A lands (the eval measures the unaided model, but the prompt
  changed), and confirm no over-instruction regression.
- Surface B depends only on the existing selection + Sequence-Viewer layers; it can
  proceed in parallel with A's review once the schema for A is frozen.

---

## Open questions / risks (for review)

- **Structured vs free-text summary.** Recommend the coupled `interpretation` (text) +
  `interpreted_action` (structured mirror) so the faithfulness check is mechanical; a
  free-text-only field forces NL extraction in the guard (itself error-prone). Decide
  whether the model fills both or the structured mirror is derived from `tool_inputs`.
- **Faithfulness coverage.** The guard is strong for chain/range/distance/tool; fuzzier
  intents ("hydrophilic", "stabilising") have no spatial referent to cross-check —
  scope the guard to the *checkable* axes and don't over-claim faithfulness on the rest.
- **Over-clarification interplay.** The reversibility gate must not turn into a confirm
  wall; keep cheap actions on the countdown. Pairs with the existing "don't over-clarify"
  prompt guidance.
- **Sequence Viewer for non-1 starts / multi-model.** Auth IDs and the resnum→position
  mapping must be shown explicitly (that *is* the point); verify against `chain_resnum_
  to_seqpos` for chains that don't start at 1 (e.g. 1IL8 chain A starts at 2).
