"""
Live-verify — Mode D (disulfide engineering scan) in the REAL Workbench panel + REAL ChimeraX.
GPU-FREE (geometry + colour + Qt; the scan reads a fold's backbone, it does NOT fold). The lesson
from the discoverability gap: a panel feature's verify OPENS THE PANEL — so this shows the panel,
runs the scan on a real structure, and confirms the heatmap actually paints the model in ChimeraX.

Confirms:
  1. The renamed Disulfides menu (4 actions, scope language) is top-level + VISIBLE.
  2. "Find engineerable disulfide sites" greys without a fold, enables once folded.
  3. The backbone scan on a REAL structure (1A2W) finds ranked sites (prefilter → fast).
  4. apply → the 'Disulfide sites' heatmap auto-surfaces, the geometric-only CAVEAT rides on it.
  5. The heatmap colour commands reach REAL ChimeraX and paint the top sites gold (it WORKS in
     the panel, not just in tests).
  6. D→introduce→C: a scanned pair feeds the introduce→constrain loop.

Run: venv/Scripts/python.exe scripts/verify_disulfide_mode_d_live.py   (no GPU; needs ChimeraX :60001)
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
from variant_workbench import VariantWorkbenchPanel, _RESULT_DISULFIDE_MODE

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
    win = QtWidgets.QMainWindow(); win.setCentralWidget(panel); win.resize(1280, 800); win.show()
    app.processEvents()

    print("1) Renamed menu, top-level + visible")
    check("'Disulfides' top-level button visible", panel._ss_menu_btn.isVisible())
    labels = [a.text() for a in panel._ss_menu_btn.menu().actions() if a.text()]
    check("4 actions with scope language (assess/measure/find/declare)",
          any("existing" in l.lower() for l in labels) and any("find" in l.lower() for l in labels)
          and any("measure" in l.lower() for l in labels) and any("declare" in l.lower() for l in labels),
          " | ".join(labels))
    check("'Find…' is the only find/discover label (A says 'existing', never 'discover')",
          "discover" not in panel._ss_discover_btn.text().lower()
          and "find" in panel._ss_scan_btn.text().lower())

    print("2) Enable-gating for Find sites")
    check("greyed with no construct", not panel._ss_scan_btn.isEnabled())
    # the construct IS the thing folded → give it enough residues to span 1A2W's numbering, so a
    # scanned pair (real resnums) maps back to construct columns for the D→introduce→C loop.
    panel._add_sequence_construct("probe", "ACDEFGHIKLMNPQRSTVWY" * 7)   # 140 residues (1..140)
    app.processEvents()
    check("still greyed before folding", not panel._ss_scan_btn.isEnabled())

    print("3) Open a real structure + point the construct's fold at it")
    before = set(_model_ids())
    run(f'open "{Path(CIF).as_posix()}"')
    mid = sorted(set(_model_ids()) - before, key=int)
    if not mid:
        print("  could not open 1A2W in ChimeraX"); return 1
    mid = mid[-1]
    cd = next(iter(panel._design.chains.values()))
    cd.rep_chain = "A"
    cd.template_fold = {"engine": "boltz", "target": "monomer", "model_id": mid, "cif_path": CIF}
    panel._sync_disulfide_menu_enabled()
    check("folded → 'Find engineerable sites' ENABLED", panel._ss_scan_btn.isEnabled())

    print("4) Run the backbone scan (real geometry) + surface the heatmap")
    r = ToolRouter(bridge=MagicMock(), session=SessionState())
    scan = r._run_disulfide_scan({"cif_path": CIF})
    check("scan found ranked engineerable sites", scan.success and len(scan.data["pairs"]) > 0,
          f"{len(scan.data['pairs'])} sites; top {scan.data['pairs'][0]['resnum_a']}–"
          f"{scan.data['pairs'][0]['resnum_b']} score {scan.data['pairs'][0]['score']}")
    panel.apply_disulfide_scan_result(
        {"_align_ukey": panel._cur_cd_ukey()},
        {"tool_step_results": [{"tool": "disulfide_scan", "success": True,
                                "data": scan.data, "summary": scan.summary}]})
    check("'Disulfide sites' heatmap auto-surfaced", panel._mode_key == _RESULT_DISULFIDE_MODE)
    # the caveat must ride on the mode — re-select it and read the status
    idx = panel._mode_combo.findData(_RESULT_DISULFIDE_MODE)
    panel._mode_combo.setCurrentIndex(idx)
    st = panel._status.text().lower()
    check("geometric-only CAVEAT rides on the heatmap",
          "does not imply" in st and "starting point" in st)

    print("5) The heatmap colours the REAL model in ChimeraX")
    tab = panel._cur_tab()
    cmds = panel.color_commands_for(tab)
    top = scan.data["pairs"][0]
    check("colour commands generated for the scanned model + a top site",
          any(f"#{mid}" in c for c in cmds) and any(str(top["resnum_a"]) in c for c in cmds),
          f"{len(cmds)} commands")
    errs = 0
    for c in cmds:
        res = run(c)
        if isinstance(res, dict) and res.get("error"):
            errs += 1
    check("all heatmap colour commands executed in ChimeraX without error", errs == 0,
          f"{len(cmds)} commands, {errs} errors")

    print("6) D→introduce→C engineering loop")
    pair = (top["resnum_a"], top["resnum_b"])
    spec = panel.build_disulfide_introduce_spec([pair])
    check("a scanned pair feeds introduce→constrain (Cys introduced + bond declared)",
          spec is not None and spec["tool_inputs"]["disulfide_bonds"] == [pair],
          f"declared {pair}")

    run(f"close #{mid}")
    win.close()
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
