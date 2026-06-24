"""
Live-verify — the Disulfides suite is DISCOVERABLE in the real Workbench panel. GPU-FREE (Qt
offscreen, no Boltz, no ChimeraX): this is the verify the no-ChimeraX v1 lacked — discoverability
is the whole fix, so SEEING it in the panel is the gate, not "built+tested".

Confirms, on the REAL VariantWorkbenchPanel (the first tab the GUI embeds):
  1. "Disulfides" is a TOP-LEVEL toolbar button (sibling of "Fold"), VISIBLE once the panel is
     shown — not buried under Fold ▾ → "Fold construct (de novo)".
  2. Its menu holds the three actions (Discover / Geometry readout / Fold with declared bond).
  3. The actions GREY/ENABLE by precondition as a de-novo construct is added then folded.
  4. Each action's handler FIRES on a real menu click (QAction.trigger()).

Run: venv/Scripts/python.exe scripts/verify_disulfide_menu_visible_live.py
"""
import os, sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from unittest.mock import MagicMock
from PySide6 import QtWidgets
from variant_workbench import VariantWorkbenchPanel
from session_state import SessionState

_checks = []
def check(name, ok, detail=""):
    _checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    # SHOW the panel in a window (offscreen) — exactly what the GUI does (first tab).
    win = QtWidgets.QMainWindow()
    win.setCentralWidget(panel)
    win.resize(1280, 800)
    win.show()
    app.processEvents()

    print("1) Top-level discoverability")
    ss, fold = panel._ss_menu_btn, panel._fold_menu_btn
    check("'Disulfides' is a top-level toolbar button", ss.text() == "Disulfides")
    check("it sits in the SAME toolbar as 'Fold' (sibling, not nested)",
          ss.parent() is fold.parent(), f"ss.parent={type(ss.parent()).__name__}")
    check("the button is VISIBLE on screen", ss.isVisible())
    check("NOT buried under the construct-fold submenu",
          panel._ss_discover_btn not in panel._construct_fold_menu.actions())

    print("2) The three modes are present in its menu")
    acts = set(ss.menu().actions())
    check("Discover / Geometry / Fold-with-bond all present",
          {panel._ss_discover_btn, panel._ss_geometry_btn, panel._ss_constrain_btn} <= acts,
          " / ".join(a.text() for a in ss.menu().actions()))

    print("3) Enable/grey tracks the precondition")
    check("no construct → ALL greyed",
          not panel._ss_discover_btn.isEnabled()
          and not panel._ss_geometry_btn.isEnabled()
          and not panel._ss_constrain_btn.isEnabled())
    panel._add_sequence_construct("binder", "MKVLWAACGTDE")
    app.processEvents()
    check("add a de-novo construct → Discover ENABLED, the fold-needing two still greyed",
          panel._ss_discover_btn.isEnabled()
          and not panel._ss_geometry_btn.isEnabled()
          and not panel._ss_constrain_btn.isEnabled())
    cd = next(iter(panel._design.chains.values()))
    cd.template_fold = {"engine": "boltz", "model_id": "7", "cif_path": "/tmp/x.cif"}
    panel._sync_disulfide_menu_enabled()
    check("construct folded → all three ENABLED",
          panel._ss_discover_btn.isEnabled()
          and panel._ss_geometry_btn.isEnabled()
          and panel._ss_constrain_btn.isEnabled())

    print("4) Each action's handler fires on a real click (QAction.trigger())")
    emitted = []
    panel.launchRequested.connect(lambda spec: emitted.append(spec))
    panel._ss_discover_btn.trigger()
    check("click 'Discover' → disulfide_discovery launch",
          bool(emitted) and emitted[-1]["tool"] == "disulfide_discovery")
    panel._ss_geometry_btn.trigger()
    check("click 'Geometry readout' → disulfide_geometry launch",
          emitted[-1]["tool"] == "disulfide_geometry")
    from PySide6 import QtWidgets as _W
    _W.QInputDialog.getText = staticmethod(lambda *a, **k: ("2 9", True))   # answer the pair prompt
    panel._ss_constrain_btn.trigger()
    check("click 'Fold with declared bond' → constrained fold (Cys2–Cys9)",
          emitted[-1]["tool"] == "boltz"
          and emitted[-1]["tool_inputs"].get("disulfide_bonds") == [(2, 9)])

    win.close()
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
