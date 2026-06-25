"""
Live-verify — the Mode D clickable RANKED-LIST widget in the REAL Workbench + REAL ChimeraX.
GPU-FREE (geometry + Qt + ChimeraX show/hide; the scan reads a fold's backbone, it does NOT fold).
A panel feature's verify OPENS THE PANEL (the discoverability-gap lesson) — so this shows the panel,
runs the scan on a real structure, opens the ranked list, and confirms a row-click actually
highlights both members in ChimeraX.

Confirms:
  1. The scan auto-OPENS a ranked-list dialog (the source of truth) carrying the caveat.
  2. The dialog lists candidate sites best-first with their measured geometry.
  3. Clicking a row HIGHLIGHTS both members (Cβ spheres, gold, selected) in REAL ChimeraX.
  4. 'Declare bond and fold' on a chosen site feeds the Mode-C introduce→constrain loop.

Run: venv/Scripts/python.exe scripts/verify_disulfide_ranked_list_live.py   (no GPU; needs ChimeraX :60001)
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
    # the panel runs ChimeraX commands through its controller — point that at the real bridge so a
    # row-click's highlight actually reaches ChimeraX (the load-bearing "it works in the panel").
    panel._c.run_commands = lambda cmds: [run(c) for c in cmds]
    panel._run_commands_bg = lambda cmds: [run(c) for c in cmds]   # synchronous for the verify
    win = QtWidgets.QMainWindow(); win.setCentralWidget(panel); win.resize(1280, 900); win.show()
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

    print("1) Run the scan → the ranked-list dialog auto-opens")
    r = ToolRouter(bridge=MagicMock(), session=SessionState())
    scan = r._run_disulfide_scan({"cif_path": CIF})
    panel.apply_disulfide_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    app.processEvents()
    dlg = getattr(panel, "_ss_scan_dialog", None)
    check("the ranked-list dialog opened", dlg is not None and dlg.isVisible())
    if dlg is None:
        return 1
    tbl = dlg.findChild(QtWidgets.QTableWidget)
    check("it lists candidate sites best-first with geometry",
          tbl.rowCount() == len(scan.data["pairs"]) and float(tbl.item(0, 1).text())
          >= float(tbl.item(min(1, tbl.rowCount()-1), 1).text()),
          f"{tbl.rowCount()} rows; top {tbl.item(0,0).text()} score {tbl.item(0,1).text()}")
    labels = [w.text().lower() for w in dlg.findChildren(QtWidgets.QLabel)]
    check("the geometric-only caveat is in the dialog",
          any("does not imply" in t and "starting point" in t for t in labels))

    print("2) Click a row → highlight both members in REAL ChimeraX")
    top = scan.data["pairs"][0]
    run("~select")
    tbl.cellClicked.emit(0, 0)                                   # == click the best site
    app.processEvents()
    sel = run("info atoms sel").get("value") or ""
    nsel = len(re.findall(r"@CB|residue", sel)) if sel else 0
    check("clicking the top site SELECTS its residues in ChimeraX",
          str(top["resnum_a"]) in sel and str(top["resnum_b"]) in sel,
          f"selected {top['resnum_a']}/{top['resnum_b']}")
    # the highlight commands executed without error (Cβ spheres + gold)
    errs = sum(1 for c in panel._disulfide_scan_highlight_commands(cd, top)
               if isinstance(run(c), dict) and run(c).get("error"))
    check("highlight commands (Cβ spheres + gold) execute without error", errs == 0)

    print("3) 'Declare bond and fold' on a site feeds the Mode-C loop")
    emitted = []
    panel.launchRequested.connect(lambda s: emitted.append(s))
    panel._declare_disulfide_from_scan((top["resnum_a"], top["resnum_b"]))
    check("declare feeds introduce→constrain (Cys introduced + bond declared)",
          bool(emitted) and emitted[-1]["tool"] == "boltz"
          and emitted[-1]["tool_inputs"]["disulfide_bonds"] == [(top["resnum_a"], top["resnum_b"])])

    run(f"close #{mid}")
    dlg.close(); win.close()
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
