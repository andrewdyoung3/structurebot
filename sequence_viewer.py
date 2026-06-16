"""
sequence_viewer.py
------------------
ChimeraX window layout/presentation helpers + the residue-number ruler content
builder shared with the StructureBot panel.

ChimeraX is STRUCTURE-ONLY in StructureBot: it never opens a Sequence Viewer.
Sequence viewing and editing live in the StructureBot window (Variant Workbench /
seq editor). The former ChimeraX-side Sequence-Viewer machinery — SCF colour mirror,
`ensure_sequence_viewer_commands`, per-chain/consolidated viewers, dock-to-bottom,
the ChimeraX numbering runscript, left-click-select, disentangle — was removed
2026-06-16. What remains here:

  • `build_numbering_header_content` — the PURE per-column residue-number ruler
    string (one char per residue, labels every N). Shared by `seq_library` for the
    StructureBot panel's numbering header (in-process, no ChimeraX).
  • `lean_layout_commands` / `apply_lean_layout` — the structure-only window layout
    (hide Log / Model Panel / CLI / Toolbar), applied at startup + once per session.
  • `default_presentation_commands` / `apply_default_presentation` — the per-open
    deterministic presentation (cartoon, bychain, ligands, bg, …).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple


def build_numbering_header_content(
    ordered_resnums: Sequence[int], interval: int = 10,
) -> str:
    """
    Build the per-column RULER string for a chain's sequence: one character per
    residue column, spaces except where a residue number is printed every *interval*
    residues (and at the last residue).

    The labels are the ACTUAL PDB residue numbers — `ordered_resnums` is the chain's
    residue numbers in SEQUENCE order (handles a non-1 start and numbering gaps), and
    each label is placed at its column via `chain_resnum_to_seqpos` (so it matches the
    MPNN alignment numbering). A non-1-start chain (e.g. starts at 2) therefore reads
    2,12,22… NOT 1,11,21. Each number's UNITS digit sits at its residue column (the
    digits extend leftward, FASTA-ruler style; clamped at the left edge).
    """
    from proteinmpnn_bridge import chain_resnum_to_seqpos
    n = len(ordered_resnums)
    if n == 0 or interval < 1:
        return " " * n
    pos1 = chain_resnum_to_seqpos(ordered_resnums)      # resnum -> 1-based position
    chars = [" "] * n
    label_idxs = list(range(0, n, interval))
    last = n - 1
    if last not in label_idxs:                          # label the last residue too,
        prev = label_idxs[-1]                            # unless it would crowd/merge
        if last - prev > len(str(int(ordered_resnums[last]))):  # into the previous label
            label_idxs.append(last)
    for idx in label_idxs:
        resnum = int(ordered_resnums[idx])
        col = pos1[resnum] - 1                           # 0-based column (== idx here)
        s = str(resnum)
        start = max(0, col - (len(s) - 1))               # units digit at the column
        for i, ch in enumerate(s):
            c = start + i
            if 0 <= c < n:
                chars[c] = ch
    return "".join(chars)


# ── Deterministic layout + presentation ─────────────────────────────────────────
# Config-driven command lists (config.CHIMERAX_*), applied by StructureBot — NOT
# LLM-generated, NOT the built-in `preset`.  apply_* run them error-first so a
# single failing command logs and the rest continue (an open is never aborted).

def lean_layout_commands(enabled: Optional[bool] = None) -> List[str]:
    """The structure-only lean-window-layout commands (or [] when disabled)."""
    import config
    if enabled is None:
        enabled = getattr(config, "CHIMERAX_LEAN_LAYOUT", True)
    return list(config.CHIMERAX_LEAN_LAYOUT_COMMANDS) if enabled else []


def default_presentation_commands(enabled: Optional[bool] = None) -> List[str]:
    """The per-open default-presentation commands (or [] when disabled)."""
    import config
    if enabled is None:
        enabled = getattr(config, "CHIMERAX_DEFAULT_PRESENTATION", True)
    return list(config.CHIMERAX_DEFAULT_PRESENTATION_COMMANDS) if enabled else []


def _run_error_first(run_command, commands: List[str]) -> Tuple[List[str], List[str]]:
    """
    Run *commands* in order via *run_command* (a callable cmd -> dict-or-anything).
    A command that raises OR returns a dict with a truthy ``error`` is recorded as
    failed and execution CONTINUES.  Returns (attempted, failed).
    """
    attempted: List[str] = []
    failed:    List[str] = []
    for c in commands:
        attempted.append(c)
        try:
            r = run_command(c)
            if isinstance(r, dict) and r.get("error"):
                failed.append(c)
        except Exception:
            failed.append(c)
    return attempted, failed


def apply_lean_layout(
    run_command, enabled: Optional[bool] = None,
) -> Tuple[List[str], List[str]]:
    """Apply the lean window layout error-first.  Returns (attempted, failed).

    The once-per-session guard lives in the caller (e.g. the bridge), so this can
    be unit-tested directly; pass *enabled* to override the config switch.
    """
    return _run_error_first(run_command, lean_layout_commands(enabled))


def apply_default_presentation(
    run_command, enabled: Optional[bool] = None,
) -> Tuple[List[str], List[str]]:
    """Apply the per-open default presentation error-first.  Returns
    (attempted, failed)."""
    return _run_error_first(run_command, default_presentation_commands(enabled))
