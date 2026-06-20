"""
variant_model.py
----------------
The Variant-Design Workbench data model (Stage 1). Pure Python, no Qt / no
ChimeraX — fully mock-testable. A generalization of `seq_editor.controller.ChainSeq`
(which carries ONE edit overlay) to MULTIPLE named variant rows per unique chain,
on a shared COLUMN axis, with provenance + (empty) result slots.

Two PRE-SHAPES so later stages are population, not rework:
  (a) MULTIPLE designs from one MPNN run import as parallel variant rows, each
      carrying provenance {"mpnn_run": R, "design_k": k} — see import_mpnn_designs.
  (b) EVERY cell (template AND variants) is an AlignedCell on a shared column axis
      and may be a residue OR a GAP (aa/resnum None). Stage 1 (substitution-only,
      ungapped) produces ZERO gaps — column ≡ template residue 1:1 — so the later
      linker/indel case is cell POPULATION on the same axis, not a mapping rewrite.

Per unique chain (homo-oligomer copies collapsed via seq_library.group_chains_by_
sequence): a `ChainDesign` = template T + ordered variants; a `DesignSession` =
those keyed by unique-chain. Persisted via session_state (JSON-friendly dicts).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# Source provenance for a single substitution (Stage 2+ populate manual/suggested).
MutationSource = str  # "manual" | "proteinmpnn" | "mutation_scanner" | "accepted_suggestion"

# Standard-20 codes a substitution may pick (canonical biological constant; mirrors
# seq_editor.controller.VALID_AA but kept local so the pure model has no Qt/spine pull).
_STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class AlignedCell:
    """One cell on the shared column axis. A GAP iff aa is None (resnum then None
    too). Stage 1 never emits a gap; the field shape is the indel seam (pre-shape b)."""
    col:    int
    resnum: Optional[int]   # author residue number (None = gap)
    aa:     Optional[str]   # 1-letter residue (None = gap)

    @property
    def is_gap(self) -> bool:
        return self.aa is None


@dataclass
class Mutation:
    resnum:  int
    from_aa: str
    to_aa:   str
    source:  MutationSource = "manual"


@dataclass
class IndelEvent:
    """An insertion/deletion on a variant — distinct from a substitution Mutation (whose
    (resnum, from_aa, to_aa) shape can't encode a length change). Stage A: kind="deletion"
    (a variant residue removed → its cell becomes a gap). `col` locates it on the shared
    column axis; `resnum`/`from_aa` record the deleted residue for provenance + revert (the
    TEMPLATE cell at that column still holds its WT resnum/aa, so a restore is lossless)."""
    kind:     str                   # "deletion" (Stage A) | "insertion" (Stage B)
    col:      int                   # current first column of the event (kept accurate on axis growth/shrink)
    resnum:   Optional[int] = None  # deletion: the removed resnum; insertion: the "after" template resnum
    from_aa:  Optional[str] = None  # deletion: the residue removed
    residues: Optional[str] = None  # insertion: the inserted aa string (the deleted side uses from_aa)


@dataclass
class ResultSlots:
    """Per-variant computed results — ALL empty in Stage 1 (Stage 4 fills them)."""
    fold:       Optional[Dict[str, Any]] = None
    stability:  Optional[Dict[str, Any]] = None
    solubility: Optional[Dict[str, Any]] = None
    scans:      Dict[str, Any] = field(default_factory=dict)


@dataclass
class Variant:
    id:         str                         # "V1", "V2", …
    parent:     str                         # "T" or another variant id
    source:     MutationSource
    provenance: Dict[str, Any] = field(default_factory=dict)   # e.g. {"mpnn_run":R,"design_k":k}
    cells:      List[AlignedCell] = field(default_factory=list)   # one per column (pre-shape b)
    mutations:  List[Mutation] = field(default_factory=list)      # substitutions vs template
    indels:     List[IndelEvent] = field(default_factory=list)    # deletions (Stage A) / insertions
    results:    ResultSlots = field(default_factory=ResultSlots)

    @property
    def sequence(self) -> str:
        """The ungapped variant sequence (gap cells contribute nothing)."""
        return "".join(c.aa for c in self.cells if c.aa is not None)


@dataclass
class ChainDesign:
    """One UNIQUE chain: the template T + its ordered variants. `members` = every
    (model, chain) copy this represents (homo-oligomer → >1; edits apply to all)."""
    group_key:      str                       # stable str form of seq_library key
    rep_model:      str
    rep_chain:      str
    members:        List[Tuple[str, str]]     # all (model, chain) copies
    template_cells: List[AlignedCell]         # T, one AlignedCell per column
    variants:       List[Variant] = field(default_factory=list)
    # S4c: seed-pinned WT reference fold + per-residue noise floor, keyed by the fold
    # combo "engine:target" (e.g. "esmfold:monomer", "boltz:assembly"). Fold-vs-fold
    # cancellation only holds same-engine/target, so each combo has its OWN reference;
    # established lazily, cached once per design (not per variant). Value:
    # {engine, target, seed, model_id, path, floor:{"resno"|"chain:resno" -> Å}}.
    wt_refs:        Dict[str, Any] = field(default_factory=dict)
    # DE-NOVO constructs: T has no crystal — the "reference structure" is the FOLD of T itself.
    # `template_fold` holds that fold (fold_summary shape: {model_id, plddt, chains, target,
    # engine, …}) once the construct is folded; `members`/`rep_model` are then re-pointed from
    # the synthetic ids to the fold model's chains so selection + property-colour come alive.
    # Empty for a crystal-seeded design (its reference IS the loaded structure).
    template_fold:  Dict[str, Any] = field(default_factory=dict)
    # Stage 3: the latest US-align STRUCTURAL alignment of this construct's fold onto a chosen
    # reference PDB (sequence-independent). Distinct from `wt_refs`/deviation — this captures a
    # superposition + scores (matchmaker's output is fired-and-ignored; US-align's is captured).
    # Shape: {reference, ref_label, tm_ref, tm_query, rmsd, n_aligned, matrix:[12], norm, ...}.
    structural_align: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_columns(self) -> int:
        return len(template_cells) if (template_cells := self.template_cells) else 0

    def resnum_for_col(self, col: int) -> Optional[int]:
        """Author resnum at a column on the TEMPLATE axis (None if gap/out-of-range)."""
        if 0 <= col < len(self.template_cells):
            return self.template_cells[col].resnum
        return None

    # ── Stage 2: variant creation + manual edit (pure; mirrors ChainSeq overlay) ────
    def get_variant(self, variant_id: str) -> Optional["Variant"]:
        for v in self.variants:
            if v.id == variant_id:
                return v
        return None

    def add_variant(self, variant_id: str, *, parent: str = "T",
                    source: MutationSource = "manual") -> "Variant":
        """Create a new variant row as an aligned COPY of the template column axis (no
        mutations yet) and append it. The cells are fresh AlignedCells (never alias the
        template's), so editing one variant can't bleed into T or its siblings."""
        cells = [AlignedCell(col=c.col, resnum=c.resnum, aa=c.aa) for c in self.template_cells]
        v = Variant(id=variant_id, parent=parent, source=source, cells=cells)
        self.variants.append(v)
        return v

    def edit_variant(self, variant_id: str, col: int, new_aa: str,
                     *, source: Optional[str] = None,
                     note: Optional[Dict[str, Any]] = None) -> None:
        """Substitute one residue in a variant at a column, lossless against T: sets the
        variant cell's aa and re-derives the Mutation list vs the template. Editing back
        to the template residue REVERTS (drops the mutation). Raises on unknown variant /
        out-of-range column / non-standard aa / gap column (no substituting a gap in S2).
        The template is NEVER touched (T is immutable — the design baseline).

        *source* tags the resulting Mutation (e.g. "accepted_suggestion" when cherry-
        picked from a mutation_scanner candidate — honest provenance distinct from the
        variant's own source); defaults to the variant's source. *note* (e.g. the scan
        `combined_score` + recommendation) is recorded in `provenance["accepted"]` keyed
        by resnum, and cleared on revert so it never outlives its mutation."""
        v = self.get_variant(variant_id)
        if v is None:
            raise KeyError(f"variant not found: {variant_id!r}")
        if not (0 <= col < len(self.template_cells)) or col >= len(v.cells):
            raise ValueError(f"column out of range: {col}")
        tmpl = self.template_cells[col]
        cell = v.cells[col]
        if tmpl.is_gap or cell.is_gap or tmpl.resnum is None:
            raise ValueError("cannot substitute a gap column")
        aa = (new_aa or "").strip().upper()
        if aa not in _STD_AA:
            raise ValueError(f"not a standard amino acid: {new_aa!r}")
        cell.aa = aa
        v.mutations = [m for m in v.mutations if m.resnum != tmpl.resnum]
        # drop any stale accepted-suggestion note for this position (revert OR re-pick)
        accepted = v.provenance.get("accepted")
        if isinstance(accepted, list):
            v.provenance["accepted"] = [a for a in accepted if a.get("resnum") != tmpl.resnum]
        if tmpl.aa is not None and aa != tmpl.aa:
            v.mutations.append(Mutation(resnum=tmpl.resnum, from_aa=tmpl.aa,
                                        to_aa=aa, source=source or v.source))
            if note:
                v.provenance.setdefault("accepted", []).append(
                    {"resnum": tmpl.resnum, "to_aa": aa, **note})
        v.mutations.sort(key=lambda m: m.resnum)

    def delete_variant_residue(self, variant_id: str, col: int) -> None:
        """DELETE a residue from a VARIANT: set its cell at *col* to a GAP and record a
        deletion `IndelEvent`. Guarded — variant only (T immutable), a NON-gap variant cell
        at a real template column. Drops any substitution Mutation at that template position
        (the residue no longer exists). LOSSLESS/REVERTABLE: the template cell still holds
        (resnum, aa), so `restore_variant_residue` rebuilds it. This is a cell change on the
        EXISTING axis — it never renumbers the template or siblings (deletion = the easy
        indel axis); the resnum-keyed deviation is handled by the panel's column-pairing map,
        NOT by this method."""
        v = self.get_variant(variant_id)
        if v is None:
            raise KeyError(f"variant not found: {variant_id!r}")
        if not (0 <= col < len(self.template_cells)) or col >= len(v.cells):
            raise ValueError(f"column out of range: {col}")
        tmpl = self.template_cells[col]
        cell = v.cells[col]
        if tmpl.is_gap or tmpl.resnum is None:
            raise ValueError("cannot delete at a non-template column")
        if cell.is_gap:
            raise ValueError("cell is already a gap (nothing to delete)")
        removed_aa = cell.aa
        cell.aa = None
        cell.resnum = None
        v.mutations = [m for m in v.mutations if m.resnum != tmpl.resnum]
        acc = v.provenance.get("accepted")
        if isinstance(acc, list):
            v.provenance["accepted"] = [a for a in acc if a.get("resnum") != tmpl.resnum]
        v.indels = [e for e in v.indels if e.col != col]          # idempotent per column
        v.indels.append(IndelEvent(kind="deletion", col=col,
                                   resnum=tmpl.resnum, from_aa=removed_aa))
        v.indels.sort(key=lambda e: e.col)

    def restore_variant_residue(self, variant_id: str, col: int) -> None:
        """Undo a deletion: rebuild the variant's cell at *col* from the TEMPLATE (WT
        resnum + aa) and drop the deletion event. Lossless to the template baseline (a prior
        substitution at this position is NOT restored — restore returns to WT)."""
        v = self.get_variant(variant_id)
        if v is None:
            raise KeyError(f"variant not found: {variant_id!r}")
        if not (0 <= col < len(self.template_cells)) or col >= len(v.cells):
            raise ValueError(f"column out of range: {col}")
        tmpl = self.template_cells[col]
        v.cells[col].resnum = tmpl.resnum
        v.cells[col].aa = tmpl.aa
        v.indels = [e for e in v.indels if e.col != col]

    def _reindex_cols(self) -> None:
        """Re-set every cell's `.col` to its list index after an axis grow/shrink (template
        + all variants stay in lockstep so every row has the same column count)."""
        for i, c in enumerate(self.template_cells):
            c.col = i
        for vv in self.variants:
            for i, c in enumerate(vv.cells):
                c.col = i

    def insert_variant_residues(self, variant_id: str, after_col: int, residues: str) -> None:
        """INSERT residues into a VARIANT after column *after_col* (after_col=-1 → before the
        first column). Grows the SHARED axis by k=len(residues) NEW columns: the template and
        every OTHER variant get k GAP cells there (resnum=None, aa=None), the inserting variant
        gets the k residues (aa set, resnum=None — an inserted residue has no crystal resnum),
        and ALL cells re-index. INDEPENDENT per-variant blocks: a sibling's insertion at the
        same locus is its OWN columns (this never coalesces). Records an insertion IndelEvent
        (residues + the 'after' template resnum); existing events past the insert point shift
        by +k so they stay accurate. Guarded: variant only, standard aa, valid position."""
        v = self.get_variant(variant_id)
        if v is None:
            raise KeyError(f"variant not found: {variant_id!r}")
        seq = (residues or "").strip().upper()
        if not seq or any(a not in _STD_AA for a in seq):
            raise ValueError(f"not standard amino acids: {residues!r}")
        n = len(self.template_cells)
        if not (-1 <= after_col < n):
            raise ValueError(f"insertion position out of range: {after_col}")
        ip = after_col + 1                        # first NEW column index
        k = len(seq)
        after_resnum = (self.template_cells[after_col].resnum if after_col >= 0 else None)
        self.template_cells[ip:ip] = [AlignedCell(col=ip + i, resnum=None, aa=None)
                                      for i in range(k)]
        for vv in self.variants:
            if vv.id == variant_id:
                cells = [AlignedCell(col=ip + i, resnum=None, aa=seq[i]) for i in range(k)]
            else:
                cells = [AlignedCell(col=ip + i, resnum=None, aa=None) for i in range(k)]
            vv.cells[ip:ip] = cells
            for e in vv.indels:                   # shift events past the insert point
                if e.col >= ip:
                    e.col += k
        self._reindex_cols()
        v.indels.append(IndelEvent(kind="insertion", col=ip, resnum=after_resnum, residues=seq))
        v.indels.sort(key=lambda e: e.col)

    def remove_variant_insertion(self, variant_id: str, col: int) -> None:
        """Undo an insertion: remove the contiguous block of INSERTED columns (template gap +
        THIS variant non-gap) containing *col*, shrinking the shared axis back, and drop the
        matching event; later events shift by -k. Guarded: *col* must be one of this variant's
        inserted columns. Restores the axis (no residues left behind)."""
        v = self.get_variant(variant_id)
        if v is None:
            raise KeyError(f"variant not found: {variant_id!r}")
        if not (0 <= col < len(self.template_cells)):
            raise ValueError(f"column out of range: {col}")
        if not self.template_cells[col].is_gap or v.cells[col].is_gap:
            raise ValueError("not an inserted column for this variant")
        lo = hi = col                             # expand to this variant's contiguous insert run
        while lo - 1 >= 0 and self.template_cells[lo - 1].is_gap and not v.cells[lo - 1].is_gap:
            lo -= 1
        while (hi + 1 < len(self.template_cells) and self.template_cells[hi + 1].is_gap
               and not v.cells[hi + 1].is_gap):
            hi += 1
        k = hi - lo + 1
        del self.template_cells[lo:hi + 1]
        for vv in self.variants:
            del vv.cells[lo:hi + 1]
            vv.indels = [e for e in vv.indels
                         if not (vv.id == variant_id and lo <= e.col <= hi)]
            for e in vv.indels:
                if e.col > hi:
                    e.col -= k
        self._reindex_cols()

    def delete_variant(self, variant_id: str) -> bool:
        """Remove a variant ROW (and everything on it — mutations, provenance, and its
        ResultSlots fold/deviation/stability/solubility/scans) from this design. Returns
        True if a row was removed. ROW-LEVEL ONLY: never touches `template_cells` (the
        residue numbering / column axis), sibling variants, or the shared `wt_refs` (the
        WT reference folds, which are per-engine and outlive any one variant). This does
        NOT renumber anything — residue/indel deletion is a SEPARATE deferred increment."""
        before = len(self.variants)
        self.variants = [v for v in self.variants if v.id != variant_id]
        return len(self.variants) < before


@dataclass
class DesignSession:
    """All unique-chain designs for one loaded model. `next_id` is monotonic across
    the whole session so variant ids never collide between chains."""
    model_id: str
    chains:   Dict[str, ChainDesign] = field(default_factory=dict)   # unique_key -> ChainDesign
    next_id:  int = 1
    # "structure" = seeded from a loaded crystal (model_id is a real ChimeraX id);
    # "sequence"  = DE-NOVO construct (model_id is synthetic "denovo-…"; nothing in ChimeraX
    # until the construct is folded). Drives the no-crystal restore + fold-as-N-mer paths.
    source:   str = "structure"

    def new_variant_id(self) -> str:
        vid = f"V{self.next_id}"
        self.next_id += 1
        return vid

    # ── (de)serialization (JSON-friendly; persisted via session_state) ──────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DesignSession":
        chains: Dict[str, ChainDesign] = {}
        for k, cd in (d.get("chains") or {}).items():
            chains[k] = ChainDesign(
                group_key      = cd["group_key"],
                rep_model      = cd["rep_model"],
                rep_chain      = cd["rep_chain"],
                members        = [tuple(m) for m in cd.get("members", [])],
                template_cells = [AlignedCell(**c) for c in cd.get("template_cells", [])],
                variants       = [_variant_from_dict(v) for v in cd.get("variants", [])],
                wt_refs        = dict(cd.get("wt_refs") or {}),
                template_fold  = dict(cd.get("template_fold") or {}),
                structural_align = dict(cd.get("structural_align") or {}),
            )
        return cls(model_id=d["model_id"], chains=chains, next_id=int(d.get("next_id", 1)),
                   source=d.get("source", "structure"))


def _variant_from_dict(v: Dict[str, Any]) -> Variant:
    return Variant(
        id=v["id"], parent=v["parent"], source=v["source"],
        provenance=dict(v.get("provenance", {})),
        cells=[AlignedCell(**c) for c in v.get("cells", [])],
        mutations=[Mutation(**m) for m in v.get("mutations", [])],
        indels=[IndelEvent(**e) for e in v.get("indels", [])],
        results=ResultSlots(**(v.get("results") or {})),
    )


# ── builders ────────────────────────────────────────────────────────────────────

def build_design_session(chain_seqs, model_id: str) -> DesignSession:
    """Build a DesignSession from `seq_editor.controller.ChainSeq` objects: group the
    chains into unique-chain designs (homo-oligomer copies collapsed) and populate
    each template T on the shared column axis (Stage 1: 1:1 with residues, no gaps).
    Reuses `seq_library.group_chains_by_sequence`."""
    from seq_library import group_chains_by_sequence
    by_key = {(cs.model, cs.chain): cs for cs in chain_seqs}
    session = DesignSession(model_id=str(model_id))
    for grp in group_chains_by_sequence(chain_seqs):
        rep_cs = by_key[grp.rep]
        cells = [AlignedCell(col=i, resnum=c.resnum, aa=c.wt_aa)
                 for i, c in enumerate(rep_cs.cells)]
        ukey = f"{grp.key[0]}|{rep_cs.model}/{rep_cs.chain}"   # stable unique-chain key
        session.chains[ukey] = ChainDesign(
            group_key      = grp.key[0],
            rep_model      = rep_cs.model,
            rep_chain      = rep_cs.chain,
            members        = list(grp.members),
            template_cells = cells,
        )
    return session


_DENOVO_AA = set("ACDEFGHIKLMNPQRSTVWY")


def build_design_session_from_sequence(name: str, chains, model_id: Optional[str] = None
                                       ) -> "DesignSession":
    """Seed a DE-NOVO `DesignSession` from typed sequence(s) — NO crystal. `chains` is a list of
    `(sequence, copy_count)`: each DISTINCT sequence → one `ChainDesign` (the unique chain), with
    `copy_count` member chains (the known stoichiometry — a homo-monomer construct is one chain;
    the N-MER is a FOLD-TIME choice, not design-time). Template T is numbered 1..N
    (`AlignedCell(col=i, resnum=i+1, aa=aa)`); `model_id` is a synthetic, session-unique
    `denovo-…` id (the stable persistence key — nothing is in ChimeraX until the construct is
    folded). Modelled as a chain LIST so hetero-complexes drop in later without rework. Pure / no
    ChimeraX — the grid renders entirely off `template_cells`. Raises on empty / non-standard aa."""
    import uuid
    mid = model_id or f"denovo-{uuid.uuid4().hex[:8]}"
    session = DesignSession(model_id=mid, source="sequence")
    chain_ids = (chr(c) for c in range(ord("A"), ord("Z") + 1))   # A, B, C… across all copies
    for ci, (seq, copies) in enumerate(chains):
        s = "".join((seq or "").split()).upper()
        if not s or any(a not in _DENOVO_AA for a in s):
            raise ValueError(f"not a standard amino-acid sequence: {seq!r}")
        n = max(1, int(copies))
        members = [(mid, next(chain_ids)) for _ in range(n)]
        rep_chain = members[0][1]
        cells = [AlignedCell(col=i, resnum=i + 1, aa=a) for i, a in enumerate(s)]
        ukey = f"{name}:{ci}|{mid}/{rep_chain}"        # stable unique-chain key
        session.chains[ukey] = ChainDesign(
            group_key      = f"{name}:{ci}" if len(chains) > 1 else name,
            rep_model      = mid,
            rep_chain      = rep_chain,
            members        = members,
            template_cells = cells,
        )
    return session


def column_tracks(design: "ChainDesign") -> Tuple[List[str], List[float]]:
    """Per-column (consensus residue, conservation) over the template + all variants.
    Conservation = fraction of rows matching the consensus; gaps count as '-'. Pure /
    testable. Stage 1 (template only) → consensus == T, conservation == 1.0 everywhere.
    """
    from collections import Counter
    n = len(design.template_cells)
    rows: List[List[str]] = [[(c.aa or "-") for c in design.template_cells]]
    for v in design.variants:
        rows.append([(c.aa or "-") for c in v.cells] if len(v.cells) == n else ["-"] * n)
    consensus: List[str] = []
    conservation: List[float] = []
    for col in range(n):
        colvals = [r[col] for r in rows]
        top, topn = Counter(colvals).most_common(1)[0]
        consensus.append(top)
        conservation.append(round(topn / len(colvals), 3))
    return consensus, conservation


def build_fold_column_map(variant: "Variant",
                          template_cells: List[AlignedCell]) -> Dict[int, int]:
    """`{variant_fold_resnum: reference_fold_resnum}` — the indel-aware correspondence that
    pairs a folded variant to the WT reference fold by COLUMN, never by resnum. Both folds
    are numbered 1..len by the engine (the variant fold over the variant's non-gap cells, the
    reference fold over the template's non-gap cells). This walks the shared column axis,
    assigning each its running 1-based position, and pairs ONLY columns where BOTH have a
    residue. A deletion (variant gap) is ABSENT (no variant-fold residue); residues AFTER it
    pair to the template position +1 — the shift the resnum==resnum path gets wrong. For a
    substitution-only variant (no gaps either side) this is the IDENTITY map {j: j} — the
    additive guarantee. Pure / mock-testable."""
    m: Dict[int, int] = {}
    var_pos = 0
    ref_pos = 0
    for col, tcell in enumerate(template_cells):
        vcell = variant.cells[col] if col < len(variant.cells) else None
        t_res = (tcell is not None) and (not tcell.is_gap)
        v_res = (vcell is not None) and (not vcell.is_gap)
        if t_res:
            ref_pos += 1
        if v_res:
            var_pos += 1
        if t_res and v_res:
            m[var_pos] = ref_pos
    return m


def build_color_commands(cells: List[AlignedCell],
                         members: List[Tuple[str, str]],
                         color_for: Callable[[Optional[str]], Optional[str]],
                         reset: str = "#ffffff") -> List[str]:
    """Compact ChimeraX `color` commands that paint a sequence-PROPERTY view onto the
    shared backbone (Stage 2 preview — color-by-identity, no rotamers rebuilt).

    *cells* are the ACTIVE row's AlignedCells (template OR a variant — substitution-only,
    so resnums map 1:1 onto the real atoms). For each (model, chain) in *members* (every
    homo-oligomer copy): emit one `color #M/C {reset}` baseline, then run-grouped
    `color #M/C:<resnums> <hex>` for residues whose color differs from the reset —
    mirroring `camsol_bridge._build_viz_commands`'s consecutive-run compaction. Gap cells
    and residues with no color opinion (color_for→None) fall through to the reset, so they
    cost no command. `color_for` is the SAME fn the panel paints with (color_modes), which
    is what pins panel-cell color == 3D-residue color (the sync invariant). Pure/testable.
    """
    # color per non-gap residue (resnum, hex) in axis order
    colored: List[Tuple[int, str]] = []
    for c in cells:
        if c.is_gap or c.resnum is None:
            continue
        hexc = color_for(c.aa) or reset
        colored.append((c.resnum, hexc))

    # group consecutive same-color residues into runs (in axis order)
    runs: List[Tuple[str, List[int]]] = []
    for resnum, hexc in colored:
        if runs and runs[-1][0] == hexc:
            runs[-1][1].append(resnum)
        else:
            runs.append((hexc, [resnum]))

    cmds: List[str] = []
    for (model, chain) in members:
        spec0 = f"#{model}/{chain}"
        cmds.append(f"color {spec0} {reset}")          # baseline (covers all reset-color residues)
        for hexc, resnos in runs:
            if hexc == reset:
                continue                                # already the baseline color
            if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
                rng = f"{resnos[0]}-{resnos[-1]}"
            else:
                rng = ",".join(str(r) for r in resnos)
            cmds.append(f"color {spec0}:{rng} {hexc}")
    return cmds


def build_color_commands_by_resnum(resnums: List[int],
                                   value_for: Callable[[int], Optional[str]],
                                   members: List[Tuple[str, str]],
                                   reset: str = "#ffffff") -> List[str]:
    """Stage 4 result-mode analog of `build_color_commands`, keyed by per-residue VALUE
    instead of residue identity. *resnums* is the author-resnum order to paint (the active
    row's non-gap resnums); *value_for(resnum)* returns a hex (e.g. `color_modes.ddg_color`
    applied to that residue's ddG) or None (no data → reset). Emits one `color #M/C {reset}`
    baseline per (model, chain) copy, then run-grouped `color #M/C:<resnums> <hex>` for
    residues whose value differs from the reset. Pure / testable; same run-compaction +
    all-copies guarantee as `build_color_commands`."""
    colored: List[Tuple[int, str]] = []
    for rn in resnums:
        if rn is None:
            continue
        colored.append((rn, value_for(rn) or reset))

    runs: List[Tuple[str, List[int]]] = []
    for resnum, hexc in colored:
        if runs and runs[-1][0] == hexc:
            runs[-1][1].append(resnum)
        else:
            runs.append((hexc, [resnum]))

    cmds: List[str] = []
    for (model, chain) in members:
        spec0 = f"#{model}/{chain}"
        cmds.append(f"color {spec0} {reset}")
        for hexc, resnos in runs:
            if hexc == reset:
                continue
            if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
                rng = f"{resnos[0]}-{resnos[-1]}"
            else:
                rng = ",".join(str(r) for r in resnos)
            cmds.append(f"color {spec0}:{rng} {hexc}")
    return cmds


def build_model_color_commands(model_id: str,
                               per_chain_resnums: Dict[str, List[int]],
                               value_for: Callable[[str, int], Optional[str]],
                               reset: str = "#ffffff") -> List[str]:
    """S4c 3D push: colour a SINGLE predicted model (`#model_id`) per chain by a
    per-residue VALUE — the deviation lives on the folded variant model's real atoms
    (like pLDDT), NOT on the shared crystal backbone, so this targets `#mid/<chain>`
    rather than the `members` copies. *per_chain_resnums* maps each chain on the
    predicted model to the ordered resnums to paint; *value_for(chain, resno)* returns a
    hex (e.g. `color_modes.deviation_color` of that residue's floor-gated deviation) or
    None (→ reset/neutral). Emits one `color #mid/C {reset}` baseline per chain, then
    run-grouped `color #mid/C:<resnums> <hex>` for residues whose value clears the reset.
    Pure / testable; same run-compaction idiom as `build_color_commands_by_resnum`."""
    cmds: List[str] = []
    for chain in sorted(per_chain_resnums):
        spec0 = f"#{model_id}/{chain}"
        cmds.append(f"color {spec0} {reset}")
        runs: List[Tuple[str, List[int]]] = []
        for rn in per_chain_resnums[chain]:
            if rn is None:
                continue
            hexc = value_for(chain, rn) or reset
            if runs and runs[-1][0] == hexc:
                runs[-1][1].append(rn)
            else:
                runs.append((hexc, [rn]))
        for hexc, resnos in runs:
            if hexc == reset:
                continue
            if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
                rng = f"{resnos[0]}-{resnos[-1]}"
            else:
                rng = ",".join(str(r) for r in resnos)
            cmds.append(f"color {spec0}:{rng} {hexc}")
    return cmds


def import_mpnn_designs(design: ChainDesign, mpnn_result: Dict[str, Any],
                        run_id: int, next_id_fn) -> List[Variant]:
    """PRE-SHAPE (a): turn one ProteinMPNN run's designs into parallel variant rows
    on *design*'s template column axis. `mpnn_result` is `session.get_proteinmpnn_
    result(model_id)` (carries `sequences: [{sequence, ...}, …]`). Each design k →
    one Variant with provenance {"mpnn_run": run_id, "design_k": k}. Substitution-
    only/ungapped: the design sequence is laid 1:1 over the template columns.

    Stage 1 ships this shape (tested) but does not wire it into the UI (Stage 3).
    """
    tmpl = design.template_cells
    fasta_path = mpnn_result.get("fasta_path")     # run identity for cross-import dedupe
    out: List[Variant] = []
    for k, d in enumerate(mpnn_result.get("sequences", []) or []):
        seq = str(d.get("sequence", ""))
        if len(seq) != len(tmpl):
            continue   # length mismatch (indel/SEQRES drift) — Stage-3 alignment handles it
        cells, muts = [], []
        for col, (tc, aa) in enumerate(zip(tmpl, seq)):
            cells.append(AlignedCell(col=col, resnum=tc.resnum, aa=aa))
            if tc.aa is not None and aa != tc.aa and tc.resnum is not None:
                muts.append(Mutation(resnum=tc.resnum, from_aa=tc.aa, to_aa=aa,
                                     source="proteinmpnn"))
        prov: Dict[str, Any] = {"mpnn_run": run_id, "design_k": k}
        if fasta_path:
            prov["fasta_path"] = fasta_path
        out.append(Variant(
            id=next_id_fn(), parent="T", source="proteinmpnn",
            provenance=prov, cells=cells, mutations=muts,
        ))
    return out


# ── Stage 3a: consume cached tool results (batch import dedupe + inline suggestions) ─

def filter_new_mpnn_variants(existing: List[Variant], candidates: List[Variant]) -> List[Variant]:
    """Drop candidate MPNN variants already imported — identity = (fasta_path, design_k)
    from provenance — so re-clicking Import is idempotent (no duplicate rows). Candidates
    with no fasta_path (uncached run) are always kept (can't prove a prior import)."""
    seen = {(v.provenance.get("fasta_path"), v.provenance.get("design_k"))
            for v in existing if v.source == "proteinmpnn"}
    out: List[Variant] = []
    for v in candidates:
        key = (v.provenance.get("fasta_path"), v.provenance.get("design_k"))
        if key[0] is not None and key in seen:
            continue
        out.append(v)
        seen.add(key)
    return out


def group_scan_suggestions(scan_results, chains, template_cells) -> Dict[int, List[Dict[str, Any]]]:
    """Group mutation_scanner candidates into per-COLUMN ranked suggestion lists for the
    inline Suggest track. Filters to *chains* (the unique-chain tab's member chains — so a
    scan run on ONE homo-oligomer copy lands in that copy's collapsed tab), maps each
    candidate's AUTHOR resnum → template column, and sorts each column's list by
    combined_score (desc). SPARSE BY CONSTRUCTION: only columns the scan actually covered
    appear — a scoped scan yields a sparse track, never implying a suggestion where none
    was computed. Pure / testable."""
    chains = {str(c) for c in chains}
    resnum_to_col = {c.resnum: c.col for c in template_cells if c.resnum is not None}
    by_col: Dict[int, List[Dict[str, Any]]] = {}
    for cand in (scan_results or []):
        if str(cand.get("chain", "")) not in chains:
            continue
        resnum = cand.get("resnum", cand.get("position"))
        col = resnum_to_col.get(resnum)
        if col is None:                            # candidate resnum not in the template
            continue
        by_col.setdefault(col, []).append(cand)
    for col in by_col:
        by_col[col].sort(key=lambda c: c.get("combined_score", 0.0), reverse=True)
    return by_col


# ── Stage 4a: reduce a stability scan (scored for a variant's EXACT mutations) ──────

def candidate_ddg(c: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """Best-available per-mutation ddG from a scan candidate + its source axis: deep
    Rosetta first (the calibrated physics axis), else ThermoMPNN (ML), else RaSP (proxy).
    Returns (ddg, source) — (None, "not_computed") when no axis scored it."""
    for key, src in (("ddg", "rosetta"), ("thermompnn_ddg", "thermompnn"),
                     ("rasp_ddg", "rasp")):
        v = c.get(key)
        if isinstance(v, (int, float)):
            return float(v), src
    return None, "not_computed"


def stability_summary(candidates: List[Dict[str, Any]],
                      mutations: List["Mutation"]) -> Dict[str, Any]:
    """Reduce scan candidates (run for a variant's EXACT mutations) to a per-variant
    stability result for ResultSlots.stability: a per-resnum ddG map (for the diverging
    result color mode), one row per scored mutation (the expandable detail), and the
    summed ddG (the badge). Matches each candidate against the variant's own (resnum,
    to_aa) so only the designed substitutions are attributed. Pure / testable."""
    want = {(m.resnum, m.to_aa) for m in mutations}
    per_resnum: Dict[int, Optional[float]] = {}
    rows: List[Dict[str, Any]] = []
    for c in candidates or []:
        rn, to_aa = c.get("resnum"), c.get("to_aa")
        if (rn, to_aa) not in want:
            continue
        ddg, src = candidate_ddg(c)
        per_resnum[rn] = ddg
        rows.append({"resnum": rn, "from_aa": c.get("from_aa"), "to_aa": to_aa,
                     "ddg": ddg, "ddg_source": src,
                     "combined_score": c.get("combined_score"),
                     "recommendation": c.get("recommendation")})
    rows.sort(key=lambda r: r["resnum"])
    vals = [r["ddg"] for r in rows if r["ddg"] is not None]
    return {"per_resnum": per_resnum, "rows": rows,
            "sum_ddg": round(sum(vals), 2) if vals else None,
            "n_scored": len(rows),
            "tier": "deep" if any(r["ddg_source"] == "rosetta" for r in rows) else "fast"}


# ── Stage 4b: reduce a fold result (engine-agnostic) for ResultSlots.fold ───────────

def fold_summary(step_data: Dict[str, Any],
                 author_resnums: List[int],
                 reference_model_id: Optional[str] = None) -> Dict[str, Any]:
    """Reduce a fold tool's step data into the NORMALIZED, engine-agnostic fold contract
    that the workbench viz / pLDDT color mode / per-model toggles all read — so a later
    engine (Boltz) populates the SAME slot without changing any consumer.

    The engine numbers its per-residue pLDDT 1..N over the ungapped folded sequence;
    *author_resnums* (the variant's ordered ungapped author resnums) maps it back to the
    design's numbering for the panel color mode. The predicted model's OWN 3D colouring is
    numbering-agnostic (ChimeraX `palette alphafold` over the B-factor), so it needs no map.
    Pure / testable. Boltz later adds 'iptm' + per-chain pLDDT additively."""
    d = step_data or {}
    raw = d.get("plddt") or {}
    plddt: Dict[int, float] = {}
    for i, rn in enumerate(author_resnums, start=1):
        val = raw.get(i, raw.get(str(i)))
        if val is not None and rn is not None:
            plddt[int(rn)] = float(val)
    mid = d.get("new_model_id", d.get("model_id"))
    ref = reference_model_id if reference_model_id is not None else d.get("reference_model_id")
    out = {
        "engine":             d.get("engine", "esmfold"),
        "target":             d.get("target", "monomer"),   # monomer | assembly (S4c combo key)
        "model_id":           str(mid) if mid is not None else None,
        "mean_plddt":         d.get("mean_plddt"),
        "plddt":              plddt,                       # author-resnum-keyed
        "reference_model_id": str(ref) if ref is not None else None,
        "rmsd":               d.get("rmsd"),               # matchmaker RMSD (None if uncaptured)
        "source":             d.get("source"),             # LOCAL-ONLY provenance
        "length":             d.get("length"),
        "chain":              d.get("chain", "A"),
    }
    # Multimer-engine fields (Boltz) — ADDITIVE: present only when the engine emits them, so
    # ESMFold (monomer) results are byte-identical to before. ipTM = interface confidence.
    if d.get("iptm") is not None:
        out["iptm"] = float(d["iptm"])
    if d.get("chains_ptm") is not None:
        out["chains_ptm"] = d["chains_ptm"]                # per-chain pTM, {chain_idx: ptm}
    if d.get("seed") is not None:
        out["seed"] = d["seed"]                            # seed-pinned provenance (reproducibility)
    # Predicted-structure file path — ADDITIVE: lets a reused fold (e.g. a de-novo construct's
    # T-fold serving as the deviation WT reference) be REOPENED if its model is closed mid-session.
    if d.get("cif_path") is not None:
        out["cif_path"] = d["cif_path"]
    if d.get("pdb_path") is not None:
        out["pdb_path"] = d["pdb_path"]
    return out


# combined_score → hex for the Suggest track cell (mirrors mutation_scanner's gradient
# bands: blue strong / cyan good / yellow marginal / orange-red not-recommended).
def suggestion_color(score: float) -> str:
    if score >= 1.5:
        return "#2a6fdb"      # strong
    if score >= 0.5:
        return "#3ec0c9"      # good (cyan)
    if score >= 0.0:
        return "#e8c33a"      # marginal (yellow)
    return "#e2663b"          # not recommended (orange-red)
