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
from typing import Any, Dict, List, Optional, Tuple

# Source provenance for a single substitution (Stage 2+ populate manual/suggested).
MutationSource = str  # "manual" | "proteinmpnn" | "mutation_scanner" | "accepted_suggestion"


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
        out.append(Variant(
            id=next_id_fn(), parent="T", source="proteinmpnn",
            provenance={"mpnn_run": run_id, "design_k": k},
            cells=cells, mutations=muts,
        ))
    return out
