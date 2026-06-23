"""
LIVE-VERIFY — Workbench sequence-grid UNIFIED block-scroll, against a real ChimeraX. The scroll
change is small; the REGRESSION SURFACE is the point. On 1HSG (HIV protease HOMODIMER — chains A
and B identical → ONE ChainDesign with TWO members, so a column-click must select BOTH copies):
  (a) long sequence (99 res) → MULTIPLE blocks, each content-sized with NO per-block scrollbar,
      the tab itself the single outer QScrollArea;
  (b) several STACKED variants → blocks taller, still no internal scrollbar;
  (c) an INSERTION (indel / insertion-code ruler) — the column axis + ruler survive;
  (d) an ACTIVE color mode painting cells.
And every coupling from the probe still works: column-click → select ALL copies in 3D, the
geometry→cell mapping the right-click menu uses (itemAt), color lands on the right cell across
blocks, ruler resnums aligned.

Run (needs a ChimeraX REST server on :60001; no GPU):
  QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe scripts/verify_seqgrid_unified_scroll_live.py
"""
import os, re, sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from PySide6 import QtWidgets, QtCore
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from variant_workbench import VariantWorkbenchPanel, _COLS, _RESNUM_ROLE, _ROW_ROLE

PASS, FAIL = [], []
def check(name, ok): (PASS if ok else FAIL).append(name); print(("  OK  " if ok else " FAIL ") + name)

bridge = ChimeraXBridge(port=60001); bridge.start(timeout=60); run = bridge.run_command
def models(): return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))
def sel_chains_at(resnum):
    """chains that have residue *resnum* currently selected (from `info residues sel`)."""
    out = run("info residues sel").get("value") or ""
    return set(re.findall(rf"/([A-Za-z])\s*:?\s*{resnum}\b", out)) | \
           set(re.findall(rf"#\d+/([A-Za-z]):{resnum}\b", out))
print("[chimerax] attached on :60001")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
panel = VariantWorkbenchPanel(ctrl, session=session)

run("close session")
before = set(models()); run("open 1hsg")
mid = (sorted(set(models()) - before, key=int) or ["1"])[-1]
panel.load_model(mid)
tab = panel._cur_tab()
cd = tab.design
print(f"[setup] 1HSG #{mid}: unique-chain '{cd.rep_chain}', members={cd.members}, {len(cd.template_cells)} cols")

# (a) homodimer collapsed to ONE design with TWO members (so column-click hits both copies)
check("1HSG homodimer → one ChainDesign with 2 members (A+B)", len(cd.members) == 2)

# (b) stack several variants
for _ in range(4):
    panel._add_variant()
tab = panel._cur_tab()
n = len(cd.template_cells)
check("long sequence wraps into multiple blocks", len(tab._blocks) == (n + _COLS - 1) // _COLS and len(tab._blocks) > 1)
check("the tab is the single OUTER scroller (QScrollArea, resizable)",
      isinstance(tab, QtWidgets.QScrollArea) and tab.widgetResizable())
all_off = all(b.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff for b in tab._blocks)
fixed = all(b.minimumHeight() == b.maximumHeight() for b in tab._blocks)
full = all(b.height() >= sum(b.rowHeight(r) for r in range(b.rowCount())) for b in tab._blocks)
check("every block: no internal vertical scrollbar (with variants stacked)", all_off)
check("every block: fixed to content height (full-height, all rows visible)", fixed and full)

# (d) ACTIVE color mode (result coloring) paints the active row — color must land on the right cell
col = n - 5                                              # a column in the LAST block
resnum = cd.resnum_for_col(col)
tab.set_active_row("T")
tab.set_result_coloring("T", {resnum: "#ff0000"})
check("color mode paints the right cell across blocks (color_hex_at on the last block)",
      tab.color_hex_at("T", col) in ("#ff0000", "#f00"))

# coupling: the geometry→cell mapping the RIGHT-CLICK menu uses (itemAt) — full-height block, no offset
last = tab._blocks[-1]
lc = col - (len(tab._blocks) - 1) * _COLS
rect = last.visualItemRect(last.item(1, lc))            # T row, that column
hit = last.itemAt(rect.center())
check("itemAt(viewport pos) → the correct cell (substitute-menu path intact)",
      hit is not None and hit.data(_RESNUM_ROLE) == col and hit.data(_ROW_ROLE) == "T")

# coupling: column-click → select_residues_multi selects residue in BOTH copies (A and B)
run("select clear")
ctrl.select_residues_multi(panel.select_specs_for_column(cd, col))
chains = sel_chains_at(resnum)
check("column-click selects the residue in ALL copies (chains A AND B) in 3D",
      {"A", "B"} <= chains)

# (c) INSERTION → indel column axis + insertion-code ruler survive
v0 = cd.variants[0].id
cd.insert_variant_residues(v0, after_col=2, residues="GG")
tab.rebuild()
tab = panel._cur_tab()
n2 = len(cd.template_cells)
check("insertion grew the shared column axis by 2", n2 == n + 2)
# the inserted template columns are gaps (resnum None) → ruler shows insertion-code letters there
ins_cols = [c for c in range(n2) if cd.template_cells[c].resnum is None]
check("inserted columns are template gaps (resnum None) — indel axis intact", len(ins_cols) == 2)
# ruler cell at an inserted column carries that gcol (column→residue identity preserved post-insert)
def ruler_cell(gcol):
    b = tab._blocks[gcol // _COLS]
    return b.item(0, gcol % _COLS)
rc = ruler_cell(ins_cols[0])
check("ruler cell at an inserted column maps back to its gcol (insertion-code ruler aligned)",
      rc is not None and rc.data(_RESNUM_ROLE) == ins_cols[0])
check("blocks still content-sized + scrollbar-off after the insertion rebuild",
      all(b.verticalScrollBarPolicy() == QtCore.Qt.ScrollBarAlwaysOff
          and b.minimumHeight() == b.maximumHeight() for b in tab._blocks))

print(f"\n══ RESULT: {len(PASS)} passed, {len(FAIL)} failed ══")
if FAIL: print("FAILED:", FAIL); sys.exit(1)
print("DONE — unified block-scroll: one outer scroller, no per-block scrollbars, all couplings intact.")
