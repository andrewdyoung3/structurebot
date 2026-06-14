r"""
seq_editor.controller — pure-Python logic for the standalone sequence editor.

NO Qt here. Everything is unit-testable with ChimeraX REST + ColabFold mocked: the
controller takes a `run_command` callable (cmd -> dict|str, e.g. ChimeraXBridge.
run_command) and a `fold_fn` callable (sequence=... -> dict, e.g. ColabFoldBridge.
predict). The Qt view (view.py) owns rendering + threads and calls into this.

Reuses the existing spine — does NOT re-implement any of it:
  • `selection.parse_selection_text` — parse `info residues` output (model,chain,resnum,resname)
  • `selection.read_selection`       — read the live 3D selection (reverse sync)
  • `proteinmpnn_bridge.chain_resnum_to_seqpos` — resnum → 1-based chain position
Model enumeration parses `info models` (regex `model id #(\d+)`), mirroring tool_router.

Lossless: WT is never mutated. An edit is a per-resnum entry in `ChainSeq.edits`; the
variant sequence is derived (WT with those positions substituted). Reverting (editing
back to WT) drops the entry.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Make the repo root importable so the spine modules resolve when this package is
# launched as `python -m seq_editor` from anywhere.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from selection import parse_selection_text, read_selection  # noqa: E402
from proteinmpnn_bridge import chain_resnum_to_seqpos        # noqa: E402

_MODEL_RE = re.compile(r"model id #(\d+)\b")

# 3-letter → 1-letter (standard 20); anything else → 'X' (kept, not dropped — lossless).
_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
# the 20 standard one-letter codes a substitution may pick
VALID_AA = set(_THREE_TO_ONE.values())


def three_to_one(resname: str) -> str:
    return _THREE_TO_ONE.get((resname or "").strip().upper(), "X")


@dataclass
class ResidueCell:
    """One residue in the grid — fully tagged (model, chain, resnum, wt_aa, seqpos)."""
    model: str
    chain: str
    resnum: int
    wt_aa: str
    seqpos: int   # 1-based position within the chain (gap-aware)


@dataclass
class ChainSeq:
    """A chain's ordered residues + its edit overlay (variant)."""
    model: str
    chain: str
    cells: List[ResidueCell]
    edits: Dict[int, str] = field(default_factory=dict)   # resnum -> substituted aa

    @property
    def key(self) -> Tuple[str, str]:
        return (self.model, self.chain)

    @property
    def wt_seq(self) -> str:
        """The lossless wild-type sequence — never altered by edits."""
        return "".join(c.wt_aa for c in self.cells)

    @property
    def variant_seq(self) -> str:
        """WT with the recorded substitutions applied (the foldable variant)."""
        return "".join(self.edits.get(c.resnum, c.wt_aa) for c in self.cells)

    @property
    def is_edited(self) -> bool:
        return bool(self.edits)

    def resnums(self) -> List[int]:
        return [c.resnum for c in self.cells]

    def wt_at(self, resnum: int) -> Optional[str]:
        for c in self.cells:
            if c.resnum == resnum:
                return c.wt_aa
        return None


class SequenceEditorController:
    """Headless logic for the editor. Inject `run_command` + `fold_fn` (real or mock)."""

    def __init__(self, run_command: Callable[[str], object],
                 fold_fn: Callable[..., dict], default_model: str = "1"):
        self._run = run_command
        self._fold = fold_fn
        self._default_model = default_model
        self.chains: Dict[Tuple[str, str], ChainSeq] = {}

    # ── REST read helpers ────────────────────────────────────────────────────────

    def _value(self, command: str) -> str:
        """run_command → its text 'value' (str), '' on anything unexpected. Never raises."""
        try:
            r = self._run(command)
        except Exception:
            return ""
        if isinstance(r, dict):
            v = r.get("value")
            return v if isinstance(v, str) else ""
        return r if isinstance(r, str) else ""

    def list_model_ids(self) -> List[str]:
        text = self._value("info models")
        ids: List[str] = []
        for m in _MODEL_RE.finditer(text):
            if m.group(1) not in ids:
                ids.append(m.group(1))
        return ids

    def load_model(self, model_id: str) -> List[ChainSeq]:
        """Read ONE model's macromolecule residues over REST and build/replace its
        per-chain sequence grids in `self.chains` WITHOUT clearing the other models
        (so a freshly-displayed model is *added* alongside existing tabs). Returns
        that model's ChainSeqs. Reuses `parse_selection_text` + `chain_resnum_to_seqpos`
        exactly like the all-models path."""
        mid = str(model_id).lstrip("#").strip()
        # exclude solvent/ligand/ions exactly like chimerax_bridge._model_chain_resnums
        text = self._value(f"info residues #{mid} & ~solvent & ~ligand & ~ions")
        refs = parse_selection_text(text, default_model=mid)
        by_chain: Dict[str, Dict[int, str]] = {}
        for (_model, chain, resnum, resname) in refs:
            # first occurrence wins (insertion codes collapse to base number — the
            # documented spine limitation; common non-1-start/gap cases preserved)
            by_chain.setdefault(chain, {}).setdefault(resnum, resname)
        out: List[ChainSeq] = []
        for chain, rn_to_name in by_chain.items():
            ordered_resnums = sorted(rn_to_name)
            pos_map = chain_resnum_to_seqpos(ordered_resnums)
            cells = [ResidueCell(mid, chain, rn, three_to_one(rn_to_name[rn]),
                                 pos_map[rn]) for rn in ordered_resnums]
            cs = ChainSeq(mid, chain, cells)
            self.chains[(mid, chain)] = cs
            out.append(cs)
        return out

    def load_models(self) -> List[ChainSeq]:
        """Read every loaded model's macromolecule residues over REST and build the
        per-chain sequence grid (the manual 'load all' path). Clears first, then
        delegates per model to `load_model`. Non-1-start / gapped chains render
        correctly because positions come from the spine, not from assuming 1."""
        self.chains.clear()
        out: List[ChainSeq] = []
        for mid in self.list_model_ids():
            out.extend(self.load_model(mid))
        return out

    def get_chain(self, model: str, chain: str) -> Optional[ChainSeq]:
        return self.chains.get((model, chain))

    # ── viewer → 3D select ───────────────────────────────────────────────────────

    @staticmethod
    def build_select_command(model: str, chain: str, resnums: List[int]) -> str:
        spec = ",".join(str(r) for r in resnums)
        return f"select #{model}/{chain}:{spec}"

    def select_residues_multi(self, specs: List[Tuple[str, str, List[int]]]):
        """Select residues across MULTIPLE (model, chain) copies in ONE `select`
        command, so all homo-oligomer copies highlight together (the Workbench
        column-click → 3D path). *specs* = [(model, chain, [resnums]), …]. No-op
        (None) when nothing resolves. ERROR-FIRST: never raises."""
        parts: List[str] = []
        for model, chain, resnums in specs:
            if resnums:
                parts.append(f"#{model}/{chain}:" + ",".join(str(r) for r in resnums))
        if not parts:
            return None
        cmd = "select " + " ".join(parts)
        try:
            r = self._run(cmd)
        except Exception as exc:
            return {"value": None, "error": f"{type(exc).__name__}: {exc}"}
        return r if isinstance(r, dict) else {"value": r, "error": None}

    def run_commands(self, commands: List[str]) -> dict:
        """Run raw ChimeraX commands over REST in order (the Workbench 3D color push —
        same trusted-tool-viz path as select: it goes direct to the bridge, not through
        the free-translation emission guard). ERROR-FIRST: never raises; returns
        {"value": last, "error": first_error|None}."""
        err: Optional[str] = None
        last = None
        for cmd in commands:
            try:
                r = self._run(cmd)
            except Exception as exc:
                err = err or f"{type(exc).__name__}: {exc}"
                continue
            if isinstance(r, dict):
                last = r.get("value")
                if r.get("error") and err is None:
                    err = r.get("error")
            else:
                last = r
        return {"value": last, "error": err}

    def select_in_3d(self, model: str, chain: str, resnums: List[int]):
        """Push a selection to ChimeraX. No-op (returns None) on empty resnum list.

        ERROR-FIRST: never raises — a failed REST call returns {"value": None,
        "error": str} so the async worker can surface it in the Messages log without
        crashing the UI thread. The select command itself is unchanged (one combined
        `select #M/C:r1,r2,…` — this is also the coalesced form)."""
        if not resnums:
            return None
        cmd = self.build_select_command(model, chain, resnums)
        try:
            r = self._run(cmd)
        except Exception as exc:
            return {"value": None, "error": f"{type(exc).__name__}: {exc}"}
        return r if isinstance(r, dict) else {"value": r, "error": None}

    # ── 3D → viewer reverse sync (ON COMMAND — no polling) ────────────────────────

    def sync_from_chimerax(self) -> List[Tuple[str, str, int]]:
        """Read the live 3D selection and return [(model, chain, resnum), …] limited to
        residues that belong to a currently-loaded chain. Error-first via
        read_selection (returns [] on nothing-selected / failure)."""
        sel = read_selection(self._run, default_model=self._default_model)
        out: List[Tuple[str, str, int]] = []
        for (model, chain, resnum, _resname) in sel.residues:
            if (model, chain) in self.chains:
                out.append((model, chain, resnum))
        return out

    # ── substitution edit → variant (lossless) ───────────────────────────────────

    def apply_substitution(self, model: str, chain: str, resnum: int, new_aa: str) -> None:
        """Record a substitution as a variant edit. WT is never touched. Editing a
        position back to its WT residue reverts (drops the edit)."""
        cs = self.chains.get((model, chain))
        if cs is None:
            raise KeyError(f"chain not loaded: #{model}/{chain}")
        aa = (new_aa or "").strip().upper()
        if aa not in VALID_AA:
            raise ValueError(f"not a standard amino acid: {new_aa!r}")
        wt = cs.wt_at(resnum)
        if wt is None:
            raise ValueError(f"resnum {resnum} not in #{model}/{chain}")
        if aa == wt:
            cs.edits.pop(resnum, None)     # revert to WT
        else:
            cs.edits[resnum] = aa

    def revert_all(self, model: str, chain: str) -> None:
        cs = self.chains.get((model, chain))
        if cs is not None:
            cs.edits.clear()

    def variant_sequence(self, model: str, chain: str) -> str:
        cs = self.chains.get((model, chain))
        if cs is None:
            raise KeyError(f"chain not loaded: #{model}/{chain}")
        return cs.variant_seq

    # ── fold the variant ─────────────────────────────────────────────────────────

    def fold_variant(self, model: str, chain: str, **fold_kwargs) -> dict:
        """Fold the chain's current variant sequence via the injected fold_fn
        (ColabFoldBridge.predict). Synchronous here; the view runs it off-thread."""
        seq = self.variant_sequence(model, chain)
        if not seq:
            return {"success": False, "error": "empty sequence"}
        return self._fold(sequence=seq, **fold_kwargs)

    def open_pdb_in_chimerax(self, pdb_path: str):
        """Optionally load a folded result into ChimeraX as a new model."""
        if not pdb_path:
            return None
        return self._run(f"open {pdb_path}")
