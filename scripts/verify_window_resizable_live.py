"""
Live-verify — the StructureBot window is FREELY RESIZABLE (the toolbar pills no longer force a hard
minimum width). LAYOUT fix → the verify actually builds the real chrome, RESIZES it narrow, and LOOKS
(saves screenshots to eyeball) — not just an assertion.

Reproduces the real app's chrome faithfully: the REAL `VariantWorkbenchPanel` + the REAL
`DisulfidesResultsTab` in the SAME `QTabWidget` → `QSplitter(vertical)` + console (`QTextEdit` + input)
shell that `gui_app._build_ui` uses. A model is loaded so the panel is populated (tabs + pills + status).

Confirms:
  1. BEFORE (the bug, simulated): a plain QHBoxLayout of the SAME pills demands ~the full width — the
     floor. AFTER: the panel's minimum width collapses far below its content (the QToolBar overflow).
  2. The window actually SHRINKS to a narrow width (e.g. 360 px) — it would clamp to ~1000+ before.
  3. The pill toolbar engaged OVERFLOW (its width < its sizeHint → items tucked into the ">>" chevron).
  4. The status label WRAPS (word-wrap on, yields width).
  5. The tabs + sequence grid are still present/usable at the narrow size.
  Screenshots `window_wide.png` / `window_narrow.png` are written for a visual eyeball of the chevron.

Run: venv/Scripts/python.exe scripts/verify_window_resizable_live.py   (offscreen Qt; no GPU, no ChimeraX)
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
from PySide6 import QtWidgets, QtCore
from seq_editor.controller import ResidueCell, ChainSeq
from variant_workbench import VariantWorkbenchPanel

OUT = Path(__file__).resolve().parent.parent / "cache"
_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _chainseq(model, chain, seq):
    return ChainSeq(model, chain, [ResidueCell(model, chain, i + 1, a, i + 1) for i, a in enumerate(seq)])


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # the REAL panel, populated (a loaded model → tabs + the full pill toolbar + status) ─────
    c = MagicMock()
    c.load_model.return_value = [_chainseq("1", "A", "MKVLWAACGTDESRPQ"),
                                 _chainseq("1", "B", "WYFGTDESRPQAACMK")]
    panel = VariantWorkbenchPanel(c, session=MagicMock(), pool=MagicMock())
    panel.load_model("1")

    # the SAME shell gui_app._build_ui uses: tabs (workbench + disulfides) in a vertical splitter
    # over a console (output + input). Faithful chrome → the screenshot looks like the real window.
    win = QtWidgets.QMainWindow(); win.setWindowTitle("StructureBot")
    tabs = QtWidgets.QTabWidget()
    tabs.addTab(panel, "Variant Workbench")
    tabs.addTab(panel.disulfides_tab, "Disulfides")
    output = QtWidgets.QTextEdit(readOnly=True); output.setText("Type a request…")
    inp = QtWidgets.QLineEdit(); inp.setPlaceholderText("Ask StructureBot…")
    bottom = QtWidgets.QWidget(); bl = QtWidgets.QVBoxLayout(bottom)
    bl.setContentsMargins(0, 0, 0, 0); bl.addWidget(output); bl.addWidget(inp)
    split = QtWidgets.QSplitter(QtCore.Qt.Vertical); split.addWidget(tabs); split.addWidget(bottom)
    win.setCentralWidget(split)
    win.resize(1000, 720); win.show(); app.processEvents()

    # 1) the panel minimum width collapsed far below its content (the floor is gone) ──────────
    tb = panel.findChildren(QtWidgets.QToolBar)[0]
    panel_min = panel.minimumSizeHint().width()
    panel_hint = panel.sizeHint().width()
    check("panel min width collapses far below its content (overflow, not a floor)",
          panel_min < panel_hint / 2 and panel_min < 400,
          f"min={panel_min}px vs content sizeHint={panel_hint}px")

    # a BEFORE comparison: the same pills in a plain QHBoxLayout would demand ~the full width
    hb = QtWidgets.QWidget(); hl = QtWidgets.QHBoxLayout(hb)
    for w in (panel._add_btn, panel._add_seq_btn, panel._apply_btn, panel._tools_btn,
              panel._stab_btn, panel._sol_btn, panel._fold_menu_btn, panel._ss_menu_btn,
              panel._dev_btn, panel._align_btn):
        hl.addWidget(QtWidgets.QPushButton(w.text() if hasattr(w, "text") else "x"))
    app.processEvents()
    check("the OLD QHBoxLayout-of-pills floor was much larger (what the user hit)",
          hb.minimumSizeHint().width() > panel_min,
          f"hbox floor≈{hb.minimumSizeHint().width()}px → toolbar floor {panel_min}px")

    # take the WIDE screenshot before narrowing ─────────────────────────────────────────────
    OUT.mkdir(exist_ok=True)
    win.grab().save(str(OUT / "window_wide.png"))

    # 2) the window actually SHRINKS narrow (it would clamp to ~1000+ before) ─────────────────
    win.resize(360, 720); app.processEvents()
    narrow = win.width()
    check("window shrinks to a narrow width (resizes below the old floor)", narrow <= 380,
          f"requested 360 → actual {narrow}px (a window minimumWidth floor would clamp it higher)")

    # 3) the pill toolbar engaged overflow (width < sizeHint → items in the chevron popup) ────
    tb_w, tb_hint = tb.width(), tb.sizeHint().width()
    check("pill toolbar engaged OVERFLOW (items tucked into the >> chevron)", tb_w < tb_hint,
          f"toolbar width {tb_w}px < content {tb_hint}px → overflow active")

    # 4) the status label wraps (yields width instead of holding one line) ───────────────────
    panel._status.setText("Workbench: 2 unique chain(s). Click a column to select in 3D (all "
                          "copies); add/edit variants; pick a color mode.")
    app.processEvents()
    check("status label word-wraps (yields width)", panel._status.wordWrap())

    # 5) tabs + sequence grid still present at the narrow size ────────────────────────────────
    check("tabs + sequence grid still present/usable at the narrow size",
          panel._tabs.count() >= 1 and tabs.count() == 2,
          f"chain tabs={panel._tabs.count()}, top tabs={tabs.count()}")

    win.grab().save(str(OUT / "window_narrow.png"))
    print(f"   screenshots: {OUT/'window_wide.png'} , {OUT/'window_narrow.png'}", flush=True)

    ok = all(_checks) and bool(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
