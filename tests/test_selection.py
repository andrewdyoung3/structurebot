"""
tests/test_selection.py
-----------------------
The live ChimeraX selection as a first-class StructureBot input. No live
ChimeraX — the bridge is mocked.

A. parse_selection_text / Selection  -- empty / single / multi-chain / non-contiguous / multi-model
B. read_selection                    -- error-first (empty / raise -> empty Selection)
C. _detect_selection_intent          -- report / scan / redesign phrasings; non-matches -> None
D. handle_selection_command          -- report path, empty-selection graceful path, scan reuses
                                        chain_resnum_to_seqpos, bridge-absent path
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from selection import Selection, parse_selection_text, read_selection
from tool_router import ToolRouter, ToolStepResult

PASS, FAIL = "[PASS]", "[FAIL]"
_results = {"pass": 0, "fail": 0}


def _ok(n): print(f"  {PASS} {n}"); _results["pass"] += 1
def _fail(n, why): print(f"  {FAIL} {n}: {why}"); _results["fail"] += 1
def _assert(c, n, m=""):
    (_ok(n) if c else _fail(n, m or "assertion failed")); return c


def _make_router(run_command=None) -> ToolRouter:
    mb = MagicMock()
    if run_command is not None:
        mb.run_command.side_effect = run_command
    ms = MagicMock()
    ms.structures = {"1": {"name": "1HSG"}}
    return ToolRouter(bridge=mb, session=ms)


# -- A. parse_selection_text / Selection --------------------------------------

_SINGLE = "residue id /A:50 name ILE index 49"
_MULTI = ("residue id /A:10 name LEU index 9\n"
          "residue id /A:25 name ASP index 24\n"
          "residue id /A:13 name ILE index 12\n"   # non-contiguous + out of order
          "residue id /B:7 name GLN index 6")
_MULTIMODEL = ("residue id #1/A:10 name LEU index 9\n"
               "residue id #2/B:7 name GLN index 6")


def test_parse_empty() -> None:
    print("\n=== A. parse / Selection ===")
    _assert(parse_selection_text("") == [], "empty text -> []")
    _assert(Selection(parse_selection_text("")).is_empty, "empty Selection.is_empty")


def test_parse_single() -> None:
    refs = parse_selection_text(_SINGLE)
    _assert(refs == [("1", "A", 50, "ILE")], "single residue parsed", f"got {refs}")
    s = Selection(refs)
    _assert(s.count == 1 and s.chains == ["A"] and s.resnums() == [50],
            "single Selection fields")


def test_parse_multi_chain_noncontiguous() -> None:
    s = Selection(parse_selection_text(_MULTI))
    _assert(s.count == 4, "4 residues", f"got {s.count}")
    _assert(s.chains == ["A", "B"], "two chains", f"got {s.chains}")
    _assert(s.by_chain() == {"A": [10, 13, 25], "B": [7]},
            "by_chain sorted + non-contiguous A", f"got {s.by_chain()}")
    _assert(s.resnums("A") == [10, 13, 25], "resnums(chain) filters + sorts")


def test_parse_multimodel() -> None:
    s = Selection(parse_selection_text(_MULTIMODEL))
    _assert(s.models == ["1", "2"], "two models parsed", f"got {s.models}")


# -- B. read_selection (error-first) ------------------------------------------

def test_read_selection_empty_and_error() -> None:
    print("\n=== B. read_selection error-first ===")
    s = read_selection(lambda c: {"value": "", "error": None})
    _assert(s.is_empty, "empty response -> empty Selection")

    def _boom(cmd): raise RuntimeError("no chimerax")
    _assert(read_selection(_boom).is_empty, "exception -> empty Selection (no raise)")

    s2 = read_selection(lambda c: {"value": _SINGLE, "error": None})
    _assert(s2.resnums() == [50], "populated response parsed")


# -- C. intent detection -------------------------------------------------------

def test_intent_detection() -> None:
    print("\n=== C. intent detection ===")
    r = _make_router()
    cases = {
        "what's selected": "report",
        "what is selected?": "report",
        "show the selection": "report",
        "scan the selection": "scan",
        "analyze the selected residues": "scan",
        "solubility of the selection": "scan",
        "redesign the selection": "redesign",
        "design the selected residues": "redesign",
    }
    for phrase, want in cases.items():
        got = r._detect_selection_intent(phrase)
        _assert(got == want, f"{phrase!r} -> {want}", f"got {got!r}")
    for non in ("open 1HSG", "select the interface residues on chain A",
                "color chain A red", "redesign chain A"):
        _assert(r._detect_selection_intent(non) is None,
                f"non-selection {non!r} -> None")


# -- D. handle_selection_command ----------------------------------------------

def _sel_runner(text):
    def run(cmd, timeout=30):
        if cmd == "info residues sel":
            return {"value": text, "error": None}
        return {"value": "", "error": None}
    return run


def test_handle_report() -> None:
    print("\n=== D. handle_selection_command ===")
    r = _make_router(_sel_runner(_MULTI))
    out = r.handle_selection_command("what's selected")
    _assert(out is not None and "4" in out and "A" in out and "B" in out,
            "report names count + chains", f"got {out!r}")
    _assert("10,13,25" in out or "10" in out, "report lists resnums", f"got {out!r}")


def test_handle_empty_graceful() -> None:
    r = _make_router(_sel_runner(""))   # nothing selected
    out = r.handle_selection_command("scan the selection")
    _assert(out is not None and "nothing is selected" in out.lower(),
            "empty selection -> graceful message, not error", f"got {out!r}")


def test_handle_non_selection_passes_through() -> None:
    r = _make_router(_sel_runner(_MULTI))
    _assert(r.handle_selection_command("open 1HSG") is None,
            "non-selection command -> None (fall through to LLM)")


def test_handle_bridge_absent() -> None:
    r = _make_router(_sel_runner(_MULTI))
    r.bridge = None
    out = r.handle_selection_command("what's selected")
    _assert(out is not None and "not connected" in out.lower(),
            "bridge absent -> friendly message")


def test_scan_reuses_chain_resnum_to_seqpos() -> None:
    """The scan path maps selection resnums -> sequence positions via
    proteinmpnn_bridge.chain_resnum_to_seqpos (chain numbered 2..N here)."""
    # Chain A numbered 2,3,4,5; select 2 and 4 -> seq positions 1 and 3.
    sel_text = ("residue id /A:2 name ALA index 1\n"
                "residue id /A:4 name ALA index 3")
    info_text = ("residue id /A:2 name ALA index 1\n"
                 "residue id /A:3 name ALA index 2\n"
                 "residue id /A:4 name ALA index 3\n"
                 "residue id /A:5 name ALA index 4")

    def run(cmd, timeout=30):
        if cmd == "info residues sel":
            return {"value": sel_text, "error": None}
        if cmd.startswith("info residues #"):
            return {"value": info_text, "error": None}
        return {"value": "", "error": None}

    r = _make_router(run)
    # CamSol returns scores keyed by start_resno+i (start=2): positions 1..4
    fake_cam = MagicMock()
    fake_cam.analyze.return_value = ToolStepResult(
        tool="camsol", success=True,
        data={"scores": {2: -2.0, 3: 0.0, 4: 1.5, 5: 0.1}})
    seen = {}
    with patch.object(r, "_get_camsol_bridge", return_value=fake_cam), \
         patch.object(r, "_fetch_sequence", return_value="AAAA"), \
         patch("proteinmpnn_bridge.chain_resnum_to_seqpos",
               side_effect=lambda o: (seen.update(ordered=list(o)),
                                      {rr: i + 1 for i, rr in enumerate(o)})[1]):
        out = r.handle_selection_command("scan the selection")
    _assert(seen.get("ordered") == [2, 3, 4, 5],
            "chain_resnum_to_seqpos called with the chain's ordered resnums",
            f"got {seen.get('ordered')}")
    # resnum 2 -> pos1 -> score -2.0 (aggregation-prone); resnum 4 -> pos3 -> 1.5
    _assert(out is not None and "A:2" in out and "A:4" in out,
            "scan reports the selected residues", f"got {out!r}")
    _assert("-2.0" in out or "-2.00" in out, "resnum 2 scored from position 1",
            f"got {out!r}")


def main() -> int:
    print("=" * 60); print("tests/test_selection.py"); print("=" * 60)
    test_parse_empty(); test_parse_single()
    test_parse_multi_chain_noncontiguous(); test_parse_multimodel()
    test_read_selection_empty_and_error()
    test_intent_detection()
    test_handle_report(); test_handle_empty_graceful()
    test_handle_non_selection_passes_through(); test_handle_bridge_absent()
    test_scan_reuses_chain_resnum_to_seqpos()
    print(f"\nResults: {_results['pass']} passed, {_results['fail']} failed")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
