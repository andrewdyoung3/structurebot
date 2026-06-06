"""
tests/test_sequence_viewer.py
-----------------------------
Tests for the Sequence-Viewer integration layer (sequence_viewer.py) and the
CamSol SCF mirror. No live ChimeraX — pure file/command construction.

A. build_scf_file        -- format, RGB clamping, resnum→0-based (incl non-1
                            start + gaps), seq_index, skip-unknown-resnum
B. build_scf_runscript   -- writes a runscript loader, returns `runscript "<py>"`
C. Part A helpers        -- ensure_sequence_viewer_commands / mouse-mode toggle
D. CamSol integration    -- analyze() appends sequence-viewer cmds + writes .scf

Usage:
  python -m pytest tests/test_sequence_viewer.py -v
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sequence_viewer import (
    build_scf_file,
    build_scf_runscript,
    ensure_sequence_viewer_commands,
    dock_sequences_bottom_command,
    build_numbering_header_content,
    numbering_header_command,
    left_click_select_command,
    lean_layout_commands,
    default_presentation_commands,
    apply_lean_layout,
    apply_default_presentation,
    _clamp8,
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


def _parse_scf(path: Path):
    """Return [(pos, seq_index, r, g, b, name), ...] from an .scf file."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        body, _, comment = line.partition("//")
        p = body.split()
        rows.append((int(p[0]), int(p[1]), int(p[2]), int(p[3]), int(p[4]),
                     comment.strip()))
    return rows


# -- A. build_scf_file ---------------------------------------------------------

def test_scf_basic_format_and_positions() -> None:
    print("\n=== A. build_scf_file ===")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "x.scf"
        # chain starts at residue 1, contiguous
        posix = build_scf_file(
            {1: (255, 0, 0), 3: (0, 0, 255)}, [1, 2, 3, 4], out,
            seq_index=0, region_name="agg")
        _assert(posix.endswith("x.scf") and "/" in posix and "\\" not in posix,
                "returns forward-slash posix path", f"got {posix!r}")
        rows = _parse_scf(out)
    # resnum 1 -> pos 0, resnum 3 -> pos 2; sorted by resnum
    _assert(rows[0] == (0, 0, 255, 0, 0, "agg"), "resnum 1 -> col 0, red",
            f"got {rows[0]}")
    _assert(rows[1] == (2, 0, 0, 0, 255, "agg"), "resnum 3 -> col 2, blue",
            f"got {rows[1]}")


def test_scf_non_one_start_and_gaps() -> None:
    """A chain numbered 2..72 (like 1IL8 A): resnum→0-based is the index in the
    ordered residue list, not resnum-1."""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "g.scf"
        # ordered residues 2,3,4,7 (a gap before 7)
        build_scf_file({2: (1, 2, 3), 7: (4, 5, 6)}, [2, 3, 4, 7], out)
        rows = _parse_scf(out)
    _assert(rows[0][0] == 0, "resnum 2 (first) -> col 0", f"got {rows[0][0]}")
    _assert(rows[1][0] == 3, "resnum 7 (4th, after gap) -> col 3",
            f"got {rows[1][0]}")


def test_scf_rgb_clamping() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "c.scf"
        build_scf_file({5: (300, -5, 128.7)}, [5], out)
        rows = _parse_scf(out)
    _assert(rows[0][2:5] == (255, 0, 129), "RGB clamped to 0-255 (rounded)",
            f"got {rows[0][2:5]}")
    _assert(_clamp8(256) == 255 and _clamp8(-1) == 0 and _clamp8(12.5) == 12,
            "_clamp8 clamps + rounds")


def test_scf_seq_index_and_skip_unknown() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "s.scf"
        # resnum 99 is NOT in ordered_resnums → skipped
        build_scf_file({1: (10, 20, 30), 99: (0, 0, 0)}, [1, 2], out, seq_index=2)
        rows = _parse_scf(out)
    _assert(len(rows) == 1, "residue absent from chain order is skipped",
            f"got {len(rows)} rows")
    _assert(rows[0][1] == 2, "seq_index written in column 2", f"got {rows[0][1]}")


def test_scf_empty_when_no_colors() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "e.scf"
        build_scf_file({}, [1, 2, 3], out)
        _assert(out.read_text() == "", "empty residue_colors -> empty file")


# -- B. build_scf_runscript ----------------------------------------------------

def test_scf_runscript_writes_loader_and_command() -> None:
    print("\n=== B. build_scf_runscript ===")
    with tempfile.TemporaryDirectory() as td:
        scf_posix = (Path(td) / "x.scf").as_posix()
        py = Path(td) / "x.scf.py"
        cmd = build_scf_runscript(scf_posix, py)
        _assert(cmd == f'runscript "{py.as_posix()}"',
                "returns runscript command with posix path", f"got {cmd!r}")
        body = py.read_text(encoding="utf-8")
        _assert(scf_posix in body, "loader embeds the .scf path")
        _assert("new_region" in body and "SequenceViewer" in body,
                "loader drives new_region on the SequenceViewer")
        _assert("__SCF_PATH__" not in body, "template placeholder fully substituted")


# -- C. Part A helpers ---------------------------------------------------------

def test_ensure_viewer_commands() -> None:
    print("\n=== C. Part A helpers ===")
    _assert(ensure_sequence_viewer_commands("1", ["A", "B"]) ==
            ["sequence chain #1/A", "sequence chain #1/B"],
            "per-chain sequence chain commands")
    _assert(ensure_sequence_viewer_commands("2", None) == ["sequence chain #2"],
            "no chains -> whole-model sequence chain")


def test_dock_sequences_bottom_command(tmp_path) -> None:
    out = tmp_path / "dock.py"
    cmd = dock_sequences_bottom_command(out)
    _assert(cmd == f'runscript "{out.as_posix()}"', "returns runscript command")
    _assert(out.exists(), "loader written")
    body = out.read_text(encoding="utf-8")
    _assert("BottomDockWidgetArea" in body, "loader docks to the BOTTOM area")
    _assert("splitDockWidget" in body and "Vertical" in body,
            "loader stacks viewers vertically")
    _assert("SequenceViewer" in body, "loader targets Sequence Viewers")


def _ruler_labels(content):
    import re
    return [int(x) for x in re.findall(r"\d+", content)]


def test_numbering_labels_are_actual_resnums_via_seqpos() -> None:
    print("\n=== C2. residue-number ruler ===")
    from proteinmpnn_bridge import chain_resnum_to_seqpos
    # a non-1-start chain (2..50) → 2,12,22… by interval 10, NOT 1,11,21
    resnums = list(range(2, 51))
    content = build_numbering_header_content(resnums, 10)
    _assert(_ruler_labels(content)[:3] == [2, 12, 22],
            "non-1-start chain labelled with ACTUAL resnums (2,12,22 not 1,11,21)",
            f"got {_ruler_labels(content)}")
    _assert(len(content) == len(resnums), "one column per residue")
    # placement is via chain_resnum_to_seqpos: each label's UNITS digit sits at its
    # canonical column → consistent with the MPNN alignment numbering
    pos1 = chain_resnum_to_seqpos(resnums)
    for r in (2, 12, 22, 42):
        col = pos1[r] - 1
        _assert(content[col] == str(r)[-1],
                f"resnum {r} units digit at MPNN column {col}", f"content={content!r}")


def test_numbering_interval_and_first_last() -> None:
    # interval 5 honored; a 1-start chain (1..25) → 1,6,11,16,21 + last (25)
    c5 = build_numbering_header_content(list(range(1, 26)), 5)
    _assert(_ruler_labels(c5) == [1, 6, 11, 16, 21, 25], "interval=5 respected + last",
            f"got {_ruler_labels(c5)}")
    # interval 10 over 1..28 → first (1) AND last (28) both labelled (no merge)
    c10 = build_numbering_header_content(list(range(1, 29)), 10)
    labs = _ruler_labels(c10)
    _assert(labs[0] == 1 and labs[-1] == 28, "first and last residue both labelled", f"got {labs}")
    # a last residue that would MERGE into the previous label is dropped (no "2123")
    cmerge = build_numbering_header_content(list(range(1, 24)), 10)
    _assert(2123 not in _ruler_labels(cmerge), "last label dropped when it would merge",
            f"got {_ruler_labels(cmerge)}")
    # gap-aware: 1..50 then 60..70 → shows 60 (the real resnum), never 51
    cg = build_numbering_header_content(list(range(1, 51)) + list(range(60, 71)), 10)
    _assert(60 in _ruler_labels(cg) and 51 not in _ruler_labels(cg),
            "gap-aware (real resnum 60, not naive 51)", f"got {_ruler_labels(cg)}")
    _assert(build_numbering_header_content([], 10) == "", "empty chain → empty ruler")


def test_numbering_header_command_is_well_formed(tmp_path) -> None:
    out = tmp_path / "num.py"
    cmd = numbering_header_command("1", "A", list(range(2, 51)), 10, out_py_path=out)
    _assert(cmd == f'runscript "{out.as_posix()}"', "returns runscript command")
    body = out.read_text(encoding="utf-8")
    _assert('"1/A"' in body, "loader targets the chain A viewer (ident 1/A)")
    _assert("add_fixed_header" in body, "loader adds a fixed header (Route 2)")
    _assert('"Numbering"' in body, "header is named 'Numbering'")
    _assert(numbering_header_command("1", "A", [], 10, out_py_path=tmp_path / "x.py") == "",
            "no resnums → no command")


def test_left_click_toggle() -> None:
    _assert(left_click_select_command(True) == "mousemode left select",
            "enable -> mousemode left select")
    _assert(left_click_select_command(False) == "mousemode left rotate",
            "disable -> mousemode left rotate (default)")


# -- D. CamSol integration -----------------------------------------------------

def test_camsol_emits_sequence_viewer_viz(monkeypatch) -> None:
    print("\n=== D. CamSol integration ===")
    import config as _cfg
    import camsol_bridge as _cb
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(_cfg, "SEQVIEW_CACHE_DIR", Path(td))
        # A sequence with a clearly aggregation-prone hydrophobic stretch.
        seq = "DDDDDIIIIIWWWWWLLLLLKKKKK"
        res = _cb.CamsolBridge().analyze(seq, model_id="1", chain="A")
        cmds = res.viz_commands
        # structure colours still present
        assert any(c.startswith("color #1") for c in cmds), "structure colours kept"
        # sequence-viewer mirror appended
        _assert(any(c.startswith("sequence chain #1/A") for c in cmds),
                "appends `sequence chain #1/A`", f"got {cmds}")
        runs = [c for c in cmds if c.startswith("runscript ")]
        _assert(len(runs) == 1, "appends one runscript SCF loader", f"got {runs}")
        scfs = list(Path(td).glob("*.scf"))
        _assert(len(scfs) == 1 and scfs[0].read_text().strip() != "",
                "wrote a non-empty .scf", f"got {scfs}")


# -- E. Lean layout + default presentation -------------------------------------

class _RecordingRunner:
    """Records commands; optionally fails (raise or error-dict) on one command."""
    def __init__(self, fail_on=None, mode="raise"):
        self.calls = []
        self.fail_on = fail_on
        self.mode = mode

    def __call__(self, cmd, timeout=30):
        self.calls.append(cmd)
        if cmd == self.fail_on:
            if self.mode == "raise":
                raise RuntimeError("boom")
            return {"value": None, "error": "nope"}
        return {"value": "", "error": None}


def test_layout_and_presentation_command_lists() -> None:
    print("\n=== E. layout + presentation ===")
    import config as _cfg
    _assert(lean_layout_commands() == list(_cfg.CHIMERAX_LEAN_LAYOUT_COMMANDS),
            "lean_layout_commands == config list, in order")
    _assert(default_presentation_commands() == list(_cfg.CHIMERAX_DEFAULT_PRESENTATION_COMMANDS),
            "default_presentation_commands == config list, in order")
    _assert(lean_layout_commands()[0] == "tool hide Log", "layout starts with tool hide Log")
    _assert(default_presentation_commands()[-1] == "view",
            "presentation ends with view")


def test_apply_runs_all_in_order() -> None:
    r = _RecordingRunner()
    attempted, failed = apply_default_presentation(r)
    _assert(r.calls == default_presentation_commands() and failed == [],
            "apply_default_presentation runs every command in order, no failures",
            f"got {r.calls}")
    r2 = _RecordingRunner()
    apply_lean_layout(r2)
    _assert(r2.calls == lean_layout_commands(), "apply_lean_layout runs layout in order")


def test_disable_override_paths() -> None:
    _assert(lean_layout_commands(enabled=False) == [], "layout disabled -> []")
    _assert(default_presentation_commands(enabled=False) == [],
            "presentation disabled -> []")
    r = _RecordingRunner()
    apply_default_presentation(r, enabled=False)
    apply_lean_layout(r, enabled=False)
    _assert(r.calls == [], "apply_* with enabled=False run nothing")


def test_failing_command_does_not_abort() -> None:
    # A raising command in the middle must NOT stop the rest.
    r = _RecordingRunner(fail_on="color bychain", mode="raise")
    attempted, failed = apply_default_presentation(r)
    _assert(attempted == default_presentation_commands(),
            "all presentation commands attempted despite a mid-list failure",
            f"got {attempted}")
    _assert(failed == ["color bychain"], "the failing command is recorded",
            f"got {failed}")
    _assert("view" in r.calls, "commands AFTER the failure still ran")
    # An error-dict (not a raise) is also treated as a failure, not a stop.
    r2 = _RecordingRunner(fail_on="set bgColor black", mode="error")
    _, failed2 = apply_default_presentation(r2)
    _assert(failed2 == ["set bgColor black"] and r2.calls == default_presentation_commands(),
            "error-dict failure recorded, rest still run")


def test_bridge_lean_layout_once_per_session(monkeypatch) -> None:
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []
    monkeypatch.setattr(b, "run_command",
                        lambda c, timeout=30: (calls.append(c),
                                               {"value": "Opened #1", "error": None})[1])
    # Two opens in one session → layout applied exactly once.
    b.run_commands(["open 1hsg"])
    b.run_commands(["open 2lyz"])
    n_layout = sum(calls.count(c) for c in lean_layout_commands())
    _assert(n_layout == len(lean_layout_commands()),
            "lean layout applied exactly once across two opens", f"got {n_layout}")
    _assert(b._lean_layout_applied, "guard flag set after first open")


def test_bridge_presentation_per_open(monkeypatch) -> None:
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []
    monkeypatch.setattr(b, "run_command",
                        lambda c, timeout=30: (calls.append(c),
                                               {"value": "Opened #1", "error": None})[1])
    b.run_commands(["open 1hsg"])
    b.run_commands(["open 2lyz"])
    # presentation (e.g. `cartoon`) runs once per open → twice
    _assert(calls.count("cartoon") == 2, "presentation applied per-open (twice)",
            f"got {calls.count('cartoon')}")


def _seq_bridge(monkeypatch, chains_value):
    """A bridge whose run_command records calls and answers `info chains` with
    *chains_value* and opens with a #1 model."""
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []

    def fake(c, timeout=30):
        calls.append(c)
        if c.startswith("info chains"):
            return {"value": chains_value, "error": None}
        return {"value": "Opened #1", "error": None}

    monkeypatch.setattr(b, "run_command", fake)
    return b, calls


def test_model_chains_parses_info_chains(monkeypatch) -> None:
    b, _ = _seq_bridge(monkeypatch, "chain id /A chain_id A\nchain id /B chain_id B")
    _assert(b._model_chains("1") == ["A", "B"], "parses chain ids from info chains")


def test_bridge_per_chain_sequence_and_dock_on_open(monkeypatch) -> None:
    b, calls = _seq_bridge(monkeypatch, "chain id /A chain_id A\nchain id /B chain_id B")
    b.run_commands(["open 1hsg"])
    _assert("sequence chain #1/A" in calls and "sequence chain #1/B" in calls,
            "opens a viewer PER CHAIN", f"got {calls}")
    _assert("sequence chain #1" not in calls, "no grouped whole-model viewer")
    _assert(any(c.startswith("runscript ") for c in calls),
            "re-docks to the bottom via runscript")


def test_bridge_per_chain_cap_falls_back_to_grouped(monkeypatch) -> None:
    # 10 chains > default cap (8) → single grouped viewer, not 10 panels.
    many = "\n".join(f"chain id /{ch} chain_id {ch}" for ch in "ABCDEFGHIJ")
    b, calls = _seq_bridge(monkeypatch, many)
    b.run_commands(["open big"])
    _assert("sequence chain #1" in calls, "above the cap -> grouped viewer")
    _assert("sequence chain #1/A" not in calls, "no per-chain panels above the cap")


# -- numbering on the per-chain open path (bridge, mocked REST) -----------------

def _info_res(chain, resnums):
    return "\n".join(f"residue id /{chain}:{n} name ALA index {i}"
                     for i, n in enumerate(resnums))


def _seq_bridge_numbering(monkeypatch, residues_by_chain, *,
                          numbering=True, interval=10, fail_substr=None):
    """A bridge whose run_command answers info chains (A,B) + info residues per chain,
    optionally raising for a command containing *fail_substr*. Numbering config is
    set on the config module."""
    from chimerax_bridge import ChimeraXBridge
    import config as _cfg
    monkeypatch.setattr(_cfg, "CHIMERAX_SEQUENCE_NUMBERING", numbering)
    monkeypatch.setattr(_cfg, "CHIMERAX_SEQUENCE_NUMBER_INTERVAL", interval)
    b = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []

    def fake(c, timeout=30):
        calls.append(c)
        if fail_substr and fail_substr in c:
            raise RuntimeError("boom")
        if c.startswith("info chains"):
            return {"value": "chain id /A chain_id A\nchain id /B chain_id B", "error": None}
        if c.startswith("info residues"):
            ch = "A" if "/A" in c else ("B" if "/B" in c else "?")
            return {"value": residues_by_chain.get(ch, ""), "error": None}
        return {"value": "Opened #1", "error": None}

    monkeypatch.setattr(b, "run_command", fake)
    return b, calls


def test_bridge_model_chain_resnums_sorted_excludes_solvent(monkeypatch) -> None:
    # unordered input → sorted ascending (sequence order); solvent excluded by the
    # `& ~solvent & ~ligand & ~ions` scoping in the issued command.
    val = "residue id /A:5 name LEU\nresidue id /A:2 name GLN\nresidue id /A:9 name VAL"
    b, calls = _seq_bridge_numbering(monkeypatch, {"A": val})
    _assert(b._model_chain_resnums("1", "A") == [2, 5, 9], "resnums parsed + sorted ascending")
    _assert(any("~solvent" in c and "~ligand" in c for c in calls),
            "info residues scoped to the macromolecule (no solvent/ligand bleed)")


def test_bridge_numbering_on_open(monkeypatch) -> None:
    resn = {"A": _info_res("A", range(2, 31)), "B": _info_res("B", range(1, 21))}
    b, calls = _seq_bridge_numbering(monkeypatch, resn, interval=10)
    b.run_commands(["open 1hsg"])
    _assert(any("numbering_1_A" in c for c in calls) and any("numbering_1_B" in c for c in calls),
            "a per-chain numbering runscript is emitted for each chain", f"got {calls}")


def test_bridge_numbering_toggle_off(monkeypatch) -> None:
    resn = {"A": _info_res("A", range(2, 31)), "B": _info_res("B", range(1, 21))}
    b, calls = _seq_bridge_numbering(monkeypatch, resn, numbering=False)
    b.run_commands(["open 1hsg"])
    _assert(not any("numbering_" in c for c in calls), "toggle OFF → zero numbering commands")
    _assert("sequence chain #1/A" in calls, "per-chain viewers still open")
    _assert(any("dock_sequences_bottom" in c for c in calls), "dock hook still runs")


def test_bridge_numbering_failure_does_not_abort_open(monkeypatch) -> None:
    # the chain-A numbering runscript raises → recorded + skipped, chain B + dock run on
    resn = {"A": _info_res("A", range(2, 31)), "B": _info_res("B", range(1, 21))}
    b, calls = _seq_bridge_numbering(monkeypatch, resn, fail_substr="numbering_1_A")
    b.run_commands(["open 1hsg"])    # must NOT raise
    _assert(any("numbering_1_B" in c for c in calls),
            "chain B numbering still ran after chain A failed (error-first)")
    _assert(any("dock_sequences_bottom" in c for c in calls),
            "dock hook still runs after a numbering failure (open completes)")


# -- Runner --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("tests/test_sequence_viewer.py")
    print("=" * 60)
    test_scf_basic_format_and_positions()
    test_scf_non_one_start_and_gaps()
    test_scf_rgb_clamping()
    test_scf_seq_index_and_skip_unknown()
    test_scf_empty_when_no_colors()
    test_scf_runscript_writes_loader_and_command()
    test_ensure_viewer_commands()
    test_dock_sequences_bottom_command(Path(tempfile.mkdtemp()))
    test_numbering_labels_are_actual_resnums_via_seqpos()
    test_numbering_interval_and_first_last()
    test_numbering_header_command_is_well_formed(Path(tempfile.mkdtemp()))
    test_left_click_toggle()
    test_layout_and_presentation_command_lists()
    test_apply_runs_all_in_order()
    test_disable_override_paths()
    test_failing_command_does_not_abort()
    print("\n(test_camsol_* and test_bridge_* need pytest's monkeypatch)")
    print("\n" + "=" * 60)
    print(f"Results: {_results['pass']} passed, {_results['fail']} failed")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
