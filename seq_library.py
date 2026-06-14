"""
seq_library.py
--------------
Reusable, PURE sequence mapping/grouping helpers — the "library" half of the
sequence_viewer split. No Qt, no ChimeraX; unit-testable. Shared by the in-process
Qt surfaces (seq_editor `_ChainGrid` at supersession, the Variant-Design Workbench)
and re-exporting the existing pure spine so there is ONE copy of each.

Re-exports (single source — NOT re-implemented):
  • proteinmpnn_bridge.chain_resnum_to_seqpos     — resnum → 1-based column (gap-aware)
  • sequence_viewer.build_numbering_header_content — the resnum RULER (already pure;
    the ChimeraX-native `numbering_header_command` calls this SAME function in-process,
    so the ruler is genuinely one copy / both callers).

New: the unique-sequence GROUPING used to collapse homo-oligomer copies into one
row/tab ("edits apply to all copies").

BOUNDARY NOTE — grouping vs the ChimeraX-native path: `sequence_viewer.
consolidated_viewers_command` runs its grouping INSIDE a runscript over ChimeraX
`session` objects, so it CANNOT import this helper across the process boundary. It
uses the IDENTICAL key formula inline; `tests/test_seq_library.py` pins the two
equal so they cannot silently drift. (The ruler had no such boundary — it is a
true shared call.)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

# Re-export the pure spine (one source of truth; do not re-implement).
from proteinmpnn_bridge import chain_resnum_to_seqpos               # noqa: F401
from sequence_viewer import build_numbering_header_content          # noqa: F401


def sequence_group_key(seq: str, resnums: Sequence[int]) -> Tuple[str, Tuple[int, ...]]:
    """The unique-chain grouping key: ``(md5(seq)[:12], sorted-resnums-tuple)``.

    IDENTICAL to the key in `sequence_viewer.consolidated_viewers_command`'s
    runscript (`md5(seq).hexdigest()[:12]` + `sorted(r.number …)`) — pinned equal
    by `tests/test_seq_library.py` so the two (one in-process, one runscript-side)
    cannot drift.
    """
    seq_hash = hashlib.md5((seq or "").encode()).hexdigest()[:12]
    return (seq_hash, tuple(sorted(int(r) for r in resnums)))


@dataclass
class ChainGroup:
    """One unique (sequence, resnums) group and every chain copy that shares it."""
    key:     Tuple[str, Tuple[int, ...]]
    rep:     Tuple[str, str]               # (model, chain) — the representative copy
    members: List[Tuple[str, str]]         # ALL (model, chain) with this exact seq+resnums
    seq:     str
    resnums: Tuple[int, ...]


def group_chains_by_sequence(chains) -> List[ChainGroup]:
    """Collapse chains sharing an identical (sequence, resnums) into one group — the
    homo-oligomer "one tab, edits apply to all copies" grouping.

    *chains* is an iterable of objects exposing ``.model``, ``.chain``, ``.wt_seq``
    and ``.resnums()`` (e.g. `seq_editor.controller.ChainSeq`). Stable ordering:
    groups in first-seen order; members sorted; rep = first member.
    """
    groups: Dict[Tuple[str, Tuple[int, ...]], ChainGroup] = {}
    order:  List[Tuple[str, Tuple[int, ...]]] = []
    for cs in chains:
        key = sequence_group_key(cs.wt_seq, cs.resnums())
        if key not in groups:
            groups[key] = ChainGroup(key=key, rep=(cs.model, cs.chain),
                                     members=[], seq=cs.wt_seq, resnums=key[1])
            order.append(key)
        groups[key].members.append((cs.model, cs.chain))
    for k in order:
        groups[k].members.sort()
        groups[k].rep = groups[k].members[0]
    return [groups[k] for k in order]
