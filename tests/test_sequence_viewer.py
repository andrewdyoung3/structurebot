"""
tests/test_sequence_viewer.py
-----------------------------
After the 2026-06-16 STRUCTURE-ONLY change, sequence_viewer.py holds only the pure
ruler-content builder + the ChimeraX window layout/presentation helpers (the
ChimeraX-side Sequence-Viewer machinery — SCF mirror, ensure/dock/consolidate/
numbering-runscript/left-click/disentangle — was removed; sequence viewing lives in
the StructureBot window). These tests cover what remains, plus the structure-only
invariants: opening a model emits NO `sequence chain` command, and the lean layout
hides the Models panel.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sequence_viewer import (
    build_numbering_header_content,
    build_numbering_header_with_insertions,
    lean_layout_commands,
    default_presentation_commands,
    apply_lean_layout,
    apply_default_presentation,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
_results = {"pass": 0, "fail": 0}


def _ok(name: str) -> None:
    print(f"  {PASS} {name}")
    _results["pass"] += 1


def _fail(name: str, reason: str) -> None:
    print(f"  {FAIL} {name}: {reason}")
    _results["fail"] += 1


def _assert(cond: bool, name: str, msg: str = "") -> bool:
    if cond:
        _ok(name)
        return True
    _fail(name, msg or "assertion failed")
    return False


def _ruler_labels(content):
    return [int(x) for x in re.findall(r"\d+", content)]


# -- residue-number ruler (shared with the StructureBot panel via seq_library) ---

def test_numbering_labels_are_actual_resnums_via_seqpos() -> None:
    print("\n=== residue-number ruler ===")
    from proteinmpnn_bridge import chain_resnum_to_seqpos
    resnums = list(range(2, 51))                     # non-1-start chain → 2,12,22…
    content = build_numbering_header_content(resnums, 10)
    _assert(_ruler_labels(content)[:3] == [2, 12, 22],
            "non-1-start chain labelled with ACTUAL resnums (2,12,22 not 1,11,21)",
            f"got {_ruler_labels(content)}")
    _assert(len(content) == len(resnums), "one column per residue")
    pos1 = chain_resnum_to_seqpos(resnums)
    for r in (2, 12, 22, 42):
        col = pos1[r] - 1
        _assert(content[col] == str(r)[-1],
                f"resnum {r} units digit at column {col}", f"content={content!r}")


def test_numbering_interval_and_first_last() -> None:
    c5 = build_numbering_header_content(list(range(1, 26)), 5)
    _assert(_ruler_labels(c5) == [1, 6, 11, 16, 21, 25], "interval=5 respected + last",
            f"got {_ruler_labels(c5)}")
    c10 = build_numbering_header_content(list(range(1, 29)), 10)
    labs = _ruler_labels(c10)
    _assert(labs[0] == 1 and labs[-1] == 28, "first and last residue both labelled", f"got {labs}")
    cmerge = build_numbering_header_content(list(range(1, 24)), 10)
    _assert(2123 not in _ruler_labels(cmerge), "last label dropped when it would merge",
            f"got {_ruler_labels(cmerge)}")
    cg = build_numbering_header_content(list(range(1, 51)) + list(range(60, 71)), 10)
    _assert(60 in _ruler_labels(cg) and 51 not in _ruler_labels(cg),
            "gap-aware (real resnum 60, not naive 51)", f"got {_ruler_labels(cg)}")
    _assert(build_numbering_header_content([], 10) == "", "empty chain -> empty ruler")


def test_numbering_with_insertions() -> None:
    print("\n=== insertion-aware ruler ===")
    # MKVLW with 2 inserted columns after resnum 2 (52A/52B-style codes, here 2A/2B):
    # cols → [1, 2, ins, ins, 3, 4, 5]
    cols = [1, 2, None, None, 3, 4, 5]
    r = build_numbering_header_with_insertions(cols, interval=1)
    _assert(len(r) == len(cols), "one column per axis column (length preserved)")
    _assert(r[2] == "A" and r[3] == "B", "contiguous inserted columns get codes A,B",
            f"got {r!r}")
    _assert(r[0] == "1" and r[4] == "3", "real columns keep their resnum digits",
            f"got {r!r}")
    # gap-free axis must be byte-identical to the plain builder (additive guarantee)
    plain = build_numbering_header_content(list(range(1, 26)), 10)
    same = build_numbering_header_with_insertions(list(range(1, 26)), 10)
    _assert(plain == same, "no-gap axis == plain ruler (additive)", f"{plain!r} vs {same!r}")
    # leading insertion (before residue 1) still gets a code, not a blank
    lead = build_numbering_header_with_insertions([None, 1, 2, 3], interval=1)
    _assert(lead[0] == "A", "leading insertion column marked", f"got {lead!r}")
    # a fresh run of codes restarts at A after a real residue
    multi = build_numbering_header_with_insertions([1, None, 2, None, 3], interval=1)
    _assert(multi[1] == "A" and multi[3] == "A", "each insertion run restarts at A",
            f"got {multi!r}")
    assert r[2] == "A" and r[3] == "B" and r[0] == "1" and r[4] == "3"
    assert plain == same and lead[0] == "A" and multi[1] == "A" and multi[3] == "A"


# -- layout + presentation command lists -----------------------------------------

def test_layout_and_presentation_command_lists() -> None:
    print("\n=== layout + presentation ===")
    import config as _cfg
    _assert(lean_layout_commands() == list(_cfg.CHIMERAX_LEAN_LAYOUT_COMMANDS),
            "lean_layout_commands == config list, in order")
    _assert(default_presentation_commands() == list(_cfg.CHIMERAX_DEFAULT_PRESENTATION_COMMANDS),
            "default_presentation_commands == config list, in order")
    _assert(lean_layout_commands()[0] == "tool hide Log", "layout starts with tool hide Log")
    _assert('tool hide "Model Panel"' in lean_layout_commands(),
            "lean layout hides the Models panel (structure-only)")
    _assert(default_presentation_commands()[-1] == "view",
            "presentation ends with 'view'")


class _Rec:
    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on
    def __call__(self, c, timeout=30):
        self.calls.append(c)
        if self.fail_on and self.fail_on in c:
            return {"value": None, "error": "boom"}
        return {"value": "ok", "error": None}


def test_apply_runs_all_in_order() -> None:
    r = _Rec()
    _, failed = apply_default_presentation(r)
    _assert(r.calls == default_presentation_commands() and failed == [],
            "apply_default_presentation runs every command in order, no failures",
            f"calls={r.calls}")
    r2 = _Rec()
    apply_lean_layout(r2)
    _assert(r2.calls == lean_layout_commands(), "apply_lean_layout runs layout in order")


def test_disable_override_paths() -> None:
    _assert(lean_layout_commands(enabled=False) == [], "layout disabled -> []")
    _assert(default_presentation_commands(enabled=False) == [],
            "presentation disabled -> []")
    r = _Rec()
    apply_default_presentation(r, enabled=False)
    apply_lean_layout(r, enabled=False)
    _assert(r.calls == [], "disabled apply_* issue no commands")


def test_failing_command_does_not_abort() -> None:
    r = _Rec(fail_on="set bgColor black")
    attempted, failed = apply_default_presentation(r)
    _assert(attempted == default_presentation_commands(),
            "every command still attempted after a failure (error-first)")
    _assert(failed == ["set bgColor black"], "the failing command is recorded",
            f"got {failed}")


# -- bridge: structure-only open path + layout/presentation hooks -----------------

def test_bridge_lean_layout_reapplied_per_open(monkeypatch) -> None:
    # Determinism fix (2026-06-17): the lean layout is RE-APPLIED on EVERY structure open
    # (idempotent `tool hide`), because ChimeraX re-shows panels (Log/Models) on load — a
    # once-per-session apply left the panel state non-deterministic across opens.
    print("\n=== bridge hooks ===")
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []
    monkeypatch.setattr(b, "run_command",
                        lambda c, timeout=30: (calls.append(c),
                                               {"value": "Opened #1", "error": None})[1])
    b.run_commands(["open 1hsg"])
    b.run_commands(["open 2lyz"])
    n_layout = sum(calls.count(c) for c in lean_layout_commands())
    _assert(n_layout == 2 * len(lean_layout_commands()),
            "lean layout re-applied once per open (twice across two opens)", f"got {n_layout}")


def test_bridge_presentation_per_open(monkeypatch) -> None:
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []
    monkeypatch.setattr(b, "run_command",
                        lambda c, timeout=30: (calls.append(c),
                                               {"value": "Opened #1", "error": None})[1])
    b.run_commands(["open 1hsg"])
    b.run_commands(["open 2lyz"])
    _assert(calls.count("cartoon") == 2, "presentation applied per-open (twice)",
            f"got {calls.count('cartoon')}")


def test_open_emits_no_sequence_viewer(monkeypatch) -> None:
    """STRUCTURE-ONLY invariant: opening a model never issues a `sequence chain`
    command (no ChimeraX Sequence Viewer)."""
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []

    def fake(c, timeout=30):
        calls.append(c)
        if c.startswith("info chains"):
            return {"value": "chain id /A chain_id A\nchain id /B chain_id B", "error": None}
        return {"value": "Opened #1", "error": None}

    monkeypatch.setattr(b, "run_command", fake)
    b.run_commands(["open 1hsg"])
    _assert(not any(str(c).startswith("sequence chain") for c in calls),
            "no `sequence chain` command on open (structure-only)", f"got {calls}")
    _assert(not any("dock_sequences" in str(c) for c in calls),
            "no sequence-viewer dock runscript on open")


def test_model_chains_parses_info_chains(monkeypatch) -> None:
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    monkeypatch.setattr(b, "run_command",
                        lambda c, timeout=30: {"value": "chain id /A chain_id A\n"
                                                        "chain id /B chain_id B", "error": None})
    _assert(b._model_chains("1") == ["A", "B"], "parses chain ids from info chains")


def test_bridge_model_chain_resnums_sorted_excludes_solvent(monkeypatch) -> None:
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []
    val = "residue id /A:5 name LEU\nresidue id /A:2 name GLN\nresidue id /A:9 name VAL"

    def fake(c, timeout=30):
        calls.append(c)
        return {"value": val, "error": None}

    monkeypatch.setattr(b, "run_command", fake)
    _assert(b._model_chain_resnums("1", "A") == [2, 5, 9], "resnums parsed + sorted ascending")
    _assert(any("~solvent" in c and "~ligand" in c for c in calls),
            "info residues scoped to the macromolecule (no solvent/ligand bleed)")


# -- Runner --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("tests/test_sequence_viewer.py")
    print("=" * 60)
    import pytest
    return pytest.main([__file__, "-q"])


if __name__ == "__main__":
    raise SystemExit(main())
