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
