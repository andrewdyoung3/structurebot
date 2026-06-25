"""
Live-verify — the PERSISTENT "Disulfides" tab in the REAL Workbench + REAL ChimeraX. GPU-FREE
(geometry + Qt + ChimeraX show/hide; the scan reads a fold's backbone, it does NOT fold). The
ACTUAL bug was a modeless dialog that VANISHED — so the gate here is PERSISTENCE ACROSS A
TAB-SWITCH, not "the table renders once".

Confirms (mirroring how gui_app hosts the two top-level tabs):
  1. After a scan the "Disulfides" tab's D section lists the ranked pairs (source of truth).
  2. Switch to "Variant Workbench" and BACK → the pairs are STILL listed (the persistence bug).
  3. Clicking a row HIGHLIGHTS the RIGHT pair on the current construct in REAL ChimeraX.
  4. Unrun modes (A/B/C) show a dormant "Run … to populate" placeholder, not a blank/broken table.
  5. The pipeline strip shows real tool names (no "Unknown tool: disulfide_scan").

Run: venv/Scripts/python.exe scripts/verify_disulfides_tab_live.py   (no GPU; needs ChimeraX :60001)
"""
import os, sys, re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from unittest.mock import MagicMock
from PySide6 import QtWidgets
from chimerax_bridge import ChimeraXBridge
from session_state import SessionState
from tool_router import ToolRouter
from variant_workbench import VariantWorkbenchPanel

CIF = str(Path(__file__).resolve().parent.parent / "cache" / "1A2W.cif")
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState())
    panel._run_commands_bg = lambda cmds: [run(c) for c in cmds]    # real ChimeraX, synchronous
    # host the TWO top-level tabs exactly as gui_app does
    win = QtWidgets.QMainWindow()
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(panel, "Variant Workbench")
    tabs.addTab(panel.disulfides_tab, "Disulfides")
    win.setCentralWidget(tabs); win.resize(1280, 900); win.show()
    app.processEvents()

    panel._add_sequence_construct("probe", "ACDEFGHIKLMNPQRSTVWY" * 7)   # 140 res (spans 1A2W)
    before = set(_model_ids())
    run(f'open "{Path(CIF).as_posix()}"')
    mid = sorted(set(_model_ids()) - before, key=int)
    if not mid:
        print("  could not open 1A2W in ChimeraX"); return 1
    mid = mid[-1]
    cd = next(iter(panel._design.chains.values()))
    cd.rep_chain = "A"
    cd.template_fold = {"engine": "boltz", "target": "monomer", "model_id": mid, "cif_path": CIF}

    print("1) Scan → the Disulfides tab's D section lists the ranked pairs")
    r = ToolRouter(bridge=MagicMock(), session=SessionState())
    scan = r._run_disulfide_scan({"cif_path": CIF})
    panel.apply_disulfide_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    app.processEvents()
    dtbl = panel.disulfides_tab._sec["D"]["table"]
    n0 = dtbl.rowCount()
    check("D section lists the ranked sites", n0 == len(scan.data["pairs"]) and n0 > 0,
          f"{n0} rows; top {dtbl.item(0,0).text()}")

    print("2) Switch to 'Variant Workbench' and BACK → pairs STILL listed (the persistence bug)")
    tabs.setCurrentIndex(0); app.processEvents()         # away (the dialog used to vanish here)
    tabs.setCurrentIndex(1); app.processEvents()         # back
    check("the table survived the tab-switch with content intact",
          panel.disulfides_tab._sec["D"]["table"].rowCount() == n0
          and panel.disulfides_tab._sec["D"]["table"].item(0, 0).text() == dtbl.item(0, 0).text())

    print("3) Row-click → highlight the RIGHT pair on the current construct in REAL ChimeraX")
    top = scan.data["pairs"][0]
    run("~select")
    dtbl.cellClicked.emit(0, 0)                          # click the best site
    app.processEvents()
    sel = run("info atoms sel").get("value") or ""
    check("clicking the top site SELECTS its residues in ChimeraX",
          str(top["resnum_a"]) in sel and str(top["resnum_b"]) in sel,
          f"selected {top['resnum_a']}/{top['resnum_b']}")

    print("4) Unrun modes are dormant (placeholder), not blank/broken")
    t = panel.disulfides_tab
    check("A/B/C show a 'Run … to populate' placeholder, no blank table",
          all(not t._sec[k]["placeholder"].isHidden() and t._sec[k].get("table") is None
              or (t._sec[k].get("table") is not None and t._sec[k]["table"].isHidden())
              for k in ("A", "B", "C")))

    print("5) The pipeline strip shows real tool names (no 'Unknown tool')")
    descs = [r._step_description(t_, {t_: {}}, "") for t_ in
             ("disulfide_scan", "disulfide_discovery", "disulfide_geometry")]
    check("no 'Unknown tool' for the three disulfide tools",
          all("Unknown tool" not in d and "Disulfide" in d for d in descs),
          " / ".join(d.split("—")[0].strip() for d in descs))

    run(f"close #{mid}")
    win.close()
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
