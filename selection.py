"""
selection.py
------------
Make the live ChimeraX selection a first-class input to StructureBot.

A user selects residues in the 3D view (Ctrl-click) or by dragging on the
Sequence Viewer; a selection-consuming command then reads that selection over
REST (`info residues sel`) and acts on it — no paste, no manual residue typing.

Poll-on-command: the selection is read WHEN a selection-consuming command is
issued. No timers, no background polling.

`info residues sel` output lines look like::

    residue id /A:10 name LEU index 9          (single model — no '#')
    residue id #2/B:7 name GLN index 6         (multi model)

We reuse this exact capture approach from the ProteinMPNN selection path, and
reuse `proteinmpnn_bridge.chain_resnum_to_seqpos` wherever a residue-number ↔
sequence-position mapping is needed (e.g. CamSol-on-subset).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

# Captures: optional model number, chain id, residue number, residue name.
_SEL_RE = re.compile(
    r"residue id\s+(?:#(\d+))?/([^:\s]+):(-?\d+)\s+name\s+(\S+)"
)

ResidueRef = Tuple[str, str, int, str]   # (model, chain, resnum, resname)


@dataclass
class Selection:
    """A structured snapshot of the live ChimeraX residue selection."""
    residues: List[ResidueRef] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.residues

    @property
    def count(self) -> int:
        return len(self.residues)

    @property
    def chains(self) -> List[str]:
        return sorted({c for _, c, _, _ in self.residues})

    @property
    def models(self) -> List[str]:
        return sorted({m for m, _, _, _ in self.residues})

    def by_chain(self) -> Dict[str, List[int]]:
        """{chain: sorted unique residue numbers}."""
        out: Dict[str, List[int]] = {}
        for _, c, r, _ in self.residues:
            out.setdefault(c, []).append(r)
        return {c: sorted(set(v)) for c, v in out.items()}

    def resnums(self, chain: str = None) -> List[int]:
        """Sorted unique residue numbers, optionally restricted to *chain*."""
        return sorted({r for _, c, r, _ in self.residues
                       if chain is None or c == chain})


def parse_selection_text(text: str, default_model: str = "1") -> List[ResidueRef]:
    """Parse `info residues …` output into [(model, chain, resnum, resname), …]."""
    out: List[ResidueRef] = []
    for m in _SEL_RE.finditer(text or ""):
        model = m.group(1) or default_model
        out.append((model, m.group(2), int(m.group(3)), m.group(4)))
    return out


def read_selection(run_command: Callable, default_model: str = "1") -> Selection:
    """
    Read the current ChimeraX selection over REST and return a `Selection`.

    Error-first: returns an EMPTY `Selection` (never raises) when nothing is
    selected or on any failure.  *run_command* is a callable ``cmd -> dict|str``
    (e.g. ``ChimeraXBridge.run_command``).
    """
    try:
        res = run_command("info residues sel")
        text = res.get("value") if isinstance(res, dict) else res
        if not isinstance(text, str):
            text = ""
        return Selection(parse_selection_text(text, default_model))
    except Exception:
        return Selection([])
