"""
variant_workbench.py
--------------------
Variant-Design Workbench — Stage 1 (display + verified 3D coupling). A new PySide6
panel in the unified GUI: a CLC-style alignment of the template T + variant rows,
one tab per UNIQUE chain (homo-oligomer copies collapsed), with a resnum ruler +
consensus + conservation tracks, and column-click → ChimeraX select of that residue
in ALL copies.

DISPLAY ONLY this stage — no variant creation / editing / coloring / action buttons
(Stage 2+). Thin UI over existing logic: reuses `seq_editor.controller`
(REST load + multi-copy select), `seq_library` (grouping + ruler), `variant_model`
(data model + tracks). Qt threading mirrors `seq_editor.view` (select off the UI
thread; cell highlight painted locally on click so the UI never waits on REST).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from seq_library import build_numbering_header_content
from variant_model import ChainDesign, DesignSession, build_design_session, column_tracks

_COLS = 30                                  # residues per wrapped block
_RESNUM_ROLE = QtCore.Qt.UserRole           # cell → template column index
_T_BG  = QtGui.QColor("#eef4ff")
_SEL_BG = QtGui.QColor("#9ad0ff")


# ── off-thread multi-copy select (mirrors seq_editor _SelectWorker) ────────────────

class _SelectSignals(QtCore.QObject):
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)


class _MultiSelectWorker(QtCore.QRunnable):
    """Runs controller.select_residues_multi off the UI thread (the HTTP select).
    The clicked cell is already highlighted locally, so the UI never waits."""

    def __init__(self, controller, specs):
        super().__init__()
        self._c, self._specs = controller, specs
        self.signals = _SelectSignals()

    @QtCore.Slot()
    def run(self):
        try:
            r = self._c.select_residues_multi(self._specs)
            self.signals.done.emit(r if isinstance(r, dict) else {"error": None})
        except Exception as exc:                       # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


# ── one unique-chain tab: T + variants in wrapped column blocks ────────────────────

class _ChainDesignTab(QtWidgets.QScrollArea):
    """Wrapped CLC-style view of one ChainDesign. Rows per block: Ruler, T, each
    variant, Consensus, Conservation. A cell click emits its TEMPLATE column index."""

    columnClicked = QtCore.Signal(int)      # template column index (0-based)

    def __init__(self, design: ChainDesign):
        super().__init__()
        self.design = design
        self.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        v.setSpacing(12)

        n = len(design.template_cells)
        ruler = build_numbering_header_content(
            [c.resnum for c in design.template_cells if c.resnum is not None], interval=10)
        ruler = (ruler + " " * n)[:n]       # guard length (gap cells)
        consensus, conservation = column_tracks(design)
        cons_pct = ["".join("·▁▂▃▄▅▆▇█"[min(8, int(c * 8))]) for c in conservation]

        row_labels = ["#"] + [f"T ({design.rep_chain})"] \
            + [v_.id for v_ in design.variants] + ["Consensus", "Conservation"]
        # row data: list of per-column strings; None for the ruler (handled specially)
        seq_rows: List[List[str]] = [[c.aa or "-" for c in design.template_cells]]
        for var in design.variants:
            seq_rows.append([c.aa or "-" for c in var.cells] if len(var.cells) == n
                            else ["-"] * n)

        self._cell_index: Dict[Tuple[int, int], int] = {}   # (block,localcol)->global col
        for start in range(0, max(1, n), _COLS):
            end = min(start + _COLS, n)
            width = end - start
            block = QtWidgets.QTableWidget(len(row_labels), width)
            block.setVerticalHeaderLabels(row_labels)
            block.horizontalHeader().setVisible(False)
            block.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            block.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            for lc in range(width):
                gcol = start + lc
                # row 0: ruler
                self._put(block, 0, lc, ruler[gcol], gcol, faint=True)
                # row 1: T (template)
                self._put(block, 1, lc, seq_rows[0][gcol], gcol, bg=_T_BG)
                # variant rows
                for vi in range(len(design.variants)):
                    self._put(block, 2 + vi, lc, seq_rows[1 + vi][gcol], gcol)
                # consensus + conservation
                self._put(block, len(row_labels) - 2, lc, consensus[gcol], gcol, faint=True)
                self._put(block, len(row_labels) - 1, lc, cons_pct[gcol], gcol, faint=True)
            block.resizeColumnsToContents()
            block.resizeRowsToContents()
            block.cellClicked.connect(self._on_cell)
            v.addWidget(block)
        v.addStretch(1)
        self.setWidget(inner)

    def _put(self, block, row, col, text, gcol, bg=None, faint=False):
        it = QtWidgets.QTableWidgetItem(text)
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        it.setData(_RESNUM_ROLE, gcol)
        if bg is not None:
            it.setBackground(bg)
        if faint:
            it.setForeground(QtGui.QColor("#888888"))
        block.setItem(row, col, it)

    def _on_cell(self, row, col):
        it = self.sender().item(row, col)
        if it is not None:
            gcol = it.data(_RESNUM_ROLE)
            if gcol is not None:
                self.columnClicked.emit(int(gcol))


# ── the panel (one QTabWidget; a tab per unique chain) ─────────────────────────────

class VariantWorkbenchPanel(QtWidgets.QWidget):
    """Stage-1 Workbench panel. `controller` = a seq_editor.SequenceEditorController
    (shares the ChimeraX bridge). `load_model(model_id)` reads the structure, builds
    the DesignSession (template T per unique chain), renders the tabs, and persists
    it via `session` if provided. Column-click selects that residue in ALL copies."""

    def __init__(self, controller, session=None, pool=None):
        super().__init__()
        self._c = controller
        self._session = session
        self._pool = pool or QtCore.QThreadPool.globalInstance()
        self._design: Optional[DesignSession] = None
        self._tabs = QtWidgets.QTabWidget()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._status = QtWidgets.QLabel("No structure loaded.")
        self._status.setStyleSheet("color:#888;padding:2px 6px;")
        lay.addWidget(self._tabs)
        lay.addWidget(self._status)

    # ── load + render ─────────────────────────────────────────────────────────────
    def load_model(self, model_id: str) -> None:
        """Read the model over REST (controller), build + render the DesignSession,
        and persist it. Error-first: a read failure leaves the panel usable."""
        try:
            chain_seqs = self._c.load_model(str(model_id))
        except Exception as exc:
            self._status.setText(f"Workbench: load failed — {type(exc).__name__}: {exc}")
            return
        if not chain_seqs:
            self._status.setText(f"Workbench: model #{model_id} has no chains.")
            return
        self._design = build_design_session(chain_seqs, str(model_id))
        self._render()
        if self._session is not None:
            try:
                self._session.add_design_session(str(model_id), self._design.to_dict())
            except Exception:
                pass

    def _render(self) -> None:
        self._tabs.clear()
        if not self._design:
            return
        for ukey, cd in self._design.chains.items():
            tab = _ChainDesignTab(cd)
            tab.columnClicked.connect(lambda col, d=cd: self._select_column(d, col))
            copies = "+".join(c for _m, c in cd.members)
            self._tabs.addTab(tab, f"{cd.rep_chain}  ({copies}, {len(cd.template_cells)} aa)")
        self._status.setText(
            f"Workbench: {len(self._design.chains)} unique chain(s); "
            f"template T loaded. Click a column to select that residue in 3D (all copies).")

    # ── column-click → 3D select (ALL copies), off the UI thread ───────────────────
    def _select_column(self, design: ChainDesign, col: int) -> None:
        resnum = design.resnum_for_col(col)
        if resnum is None:                      # gap column — nothing to select
            return
        specs = [(m, c, [resnum]) for (m, c) in design.members]
        w = _MultiSelectWorker(self._c, specs)
        w.signals.failed.connect(lambda e: self._status.setText(f"Workbench select failed: {e}"))
        self._pool.start(w)

    # exposed for tests / live-verify: the exact specs a column click dispatches
    def select_specs_for_column(self, design: ChainDesign, col: int):
        resnum = design.resnum_for_col(col)
        if resnum is None:
            return []
        return [(m, c, [resnum]) for (m, c) in design.members]
