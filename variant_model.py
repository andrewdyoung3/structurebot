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


@dataclass
class DesignSession:
    """All unique-chain designs for one loaded model. `next_id` is monotonic across
    the whole session so variant ids never collide between chains."""
    model_id: str
    chains:   Dict[str, ChainDesign] = field(default_factory=dict)   # unique_key -> ChainDesign
    next_id:  int = 1

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
            )
        return cls(model_id=d["model_id"], chains=chains, next_id=int(d.get("next_id", 1)))


def _variant_from_dict(v: Dict[str, Any]) -> Variant:
    return Variant(
        id=v["id"], parent=v["parent"], source=v["source"],
        provenance=dict(v.get("provenance", {})),
        cells=[AlignedCell(**c) for c in v.get("cells", [])],
        mutations=[Mutation(**m) for m in v.get("mutations", [])],
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
