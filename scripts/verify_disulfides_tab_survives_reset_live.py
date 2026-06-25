"""
Live-verify — the persistent Disulfides tab SURVIVES a session reset (Clear/Load/reconnect) AND is
CLEARED, in the REAL StructureBotWindow + REAL ChimeraX. The regression: `_reset_view_for_session`
kept only the workbench tab and swept the sibling Disulfides tab out (never re-added) — so any
Clear/Load/reconnect dropped it. GPU-FREE: 1A2W is itself a 2-chain domain-swapped dimer, so the
interface + Mode-D scans run on a real fold without folding.

Confirms (the reset-survival + clear gate):
  1. Build the REAL window → its tabs are [Variant Workbench, Disulfides].
  2. Run a REAL interface scan AND a REAL Mode-D scan on the 2-chain structure → both sections
     populate (the tab carries pairs).
  3. Session → Clear (the real menu path) → the Disulfides tab PERSISTS but is EMPTY (cleared).
  4. Re-run a scan → the tab REPOPULATES (ready for the new session).
  5. A direct `_reset_view_for_session` (the shared Load/reconnect mechanism) → tab PERSISTS + empty.
  6. The original workflow: re-seed + interface scan → the tab STAYS and shows the inter-chain pairs.

Run: venv/Scripts/python.exe scripts/verify_disulfides_tab_survives_reset_live.py   (no GPU; needs ChimeraX :60001)
"""
import os, sys, re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from unittest.mock import patch
from PySide6 import QtWidgets
import config
from gui_app import StructureBotWindow

CIF = str(Path(__file__).resolve().parent.parent / "cache" / "1A2W.cif")

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _tab_widgets(win):
    return [win.tabs.widget(i) for i in range(win.tabs.count())]


def _seed_dimer_construct(win, mid):
    """A de-novo construct whose T-fold IS the opened 2-chain 1A2W (members A,B)."""
    p = win.workbench
    p._add_sequence_construct("dimer", "A" * 100)
    cd = next(iter(p._design.chains.values()))
    cd.members = [(mid, "A"), (mid, "B")]
    cd.rep_model, cd.rep_chain = mid, "A"
    cd.template_fold = {"engine": "boltz", "target": "assembly", "model_id": mid, "cif_path": CIF}
    return p, cd


def _run_scans(win):
    """Run the REAL interface + Mode-D scans on the construct's fold and apply them to the tab."""
    p = win.workbench
    uk = {"_align_ukey": p._cur_cd_ukey()}
    iscan = win.router._run_disulfide_interface_scan({"cif_path": CIF})
    p.apply_disulfide_interface_scan_result(uk, {"tool_step_results": [
        {"tool": "disulfide_interface_scan", "success": True, "data": iscan.data, "summary": iscan.summary}]})
    dscan = win.router._run_disulfide_scan({"cif_path": CIF})
    p.apply_disulfide_scan_result(uk, {"tool_step_results": [
        {"tool": "disulfide_scan", "success": True, "data": dscan.data, "summary": dscan.summary}]})
    return iscan, dscan


def _rows(win, key):
    return win.workbench.disulfides_tab._sec[key]["table"].rowCount()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = StructureBotWindow(port=config.REST_PORT)
    run = win.bridge.run_command
    # the row-preselect glow dispatches through _run_commands_bg (a thread-pool worker); make it
    # SYNCHRONOUS for the verify so re-seeding doesn't GC a worker's signal source mid-flight.
    win.workbench._run_commands_bg = lambda cmds: [run(c) for c in cmds]

    before = set(re.findall(r"model id #(\d+) ", run("info models").get("value") or ""))
    run(f'open "{Path(CIF).as_posix()}"')
    mids = sorted(set(re.findall(r"model id #(\d+) ", run("info models").get("value") or "")) - before, key=int)
    if not mids:
        print("  could not open 1A2W"); return 1
    mid = mids[-1]
    dtab = win.workbench.disulfides_tab

    print("1) The window has the persistent Disulfides tab…")
    check("Disulfides tab is a sibling tab in the window", dtab in _tab_widgets(win))

    print("2) Run a REAL interface scan + Mode-D scan → the tab populates…")
    _seed_dimer_construct(win, mid)
    iscan, dscan = _run_scans(win)
    check("interface scan found inter-chain pairs (section I populated)",
          bool(iscan.data["pairs"]) and _rows(win, "I") > 0, f"{_rows(win,'I')} I-rows")
    check("Mode-D scan found intra-chain pairs (section D populated)",
          bool(dscan.data["pairs"]) and _rows(win, "D") > 0, f"{_rows(win,'D')} D-rows")

    print("3) Session → Clear → the tab PERSISTS but is EMPTY…")
    with patch.object(QtWidgets.QMessageBox, "question",
                      return_value=QtWidgets.QMessageBox.StandardButton.Yes):
        win._on_clear_session()
    check("Disulfides tab SURVIVED Clear (not swept out)", dtab in _tab_widgets(win))
    check("the tab was CLEARED (no stale pairs)", _rows(win, "I") == 0 and _rows(win, "D") == 0,
          f"I={_rows(win,'I')} D={_rows(win,'D')}")
    check("dormant placeholder restored", not dtab._sec["D"]["placeholder"].isHidden())

    print("4) Re-run a scan → the tab REPOPULATES (ready for the new session)…")
    _seed_dimer_construct(win, mid)
    _run_scans(win)
    check("the kept tab repopulates after reset", _rows(win, "I") > 0 and _rows(win, "D") > 0,
          f"I={_rows(win,'I')} D={_rows(win,'D')}")

    print("5) A direct _reset_view_for_session (the shared Load/reconnect path) → persists + empty…")
    win._reset_view_for_session()
    check("Disulfides tab SURVIVED the Load/reconnect reset mechanism", dtab in _tab_widgets(win))
    check("…and is empty", _rows(win, "I") == 0 and _rows(win, "D") == 0)

    print("6) The original workflow: re-seed + interface scan → tab stays + shows the pairs…")
    _seed_dimer_construct(win, mid)
    iscan2, _ = _run_scans(win)
    top = iscan2.data["pairs"][0] if iscan2.data["pairs"] else None
    label = dtab._sec["I"]["table"].item(0, 0).text() if _rows(win, "I") else ""
    check("interface scan keeps the tab + shows chain-labeled inter-chain pairs",
          dtab in _tab_widgets(win) and _rows(win, "I") > 0 and ":" in label and "↔" in label,
          f"top I-row '{label}'")

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
