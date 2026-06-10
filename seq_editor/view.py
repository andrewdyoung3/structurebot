"""
seq_editor.view — the PySide6 Qt layer for the standalone sequence editor.

Rendering + threads only; ALL logic lives in controller.SequenceEditorController.
Error-first: every controller call is guarded — a ChimeraX/fold failure shows a
message in the window, it never crashes the window (and, being a separate process,
never the REPL or ChimeraX). The minutes-long fold runs off the UI thread via
QThreadPool so the window never freezes. REST reads/selects stay synchronous (fast).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from .controller import SequenceEditorController, ChainSeq, VALID_AA

_COLS = 30                       # residues per grid row
_EDIT_BG = QtGui.QColor("#ffd27f")
_SYNC_BG = QtGui.QColor("#9ad0ff")
_RESNUM_ROLE = QtCore.Qt.UserRole
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"


# ── async fold worker ────────────────────────────────────────────────────────────

class _FoldSignals(QtCore.QObject):
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)


class _FoldWorker(QtCore.QRunnable):
    """Runs controller.fold_variant off the UI thread; result delivered via signals."""

    def __init__(self, controller, model, chain, **kw):
        super().__init__()
        self._c, self._model, self._chain, self._kw = controller, model, chain, kw
        self.signals = _FoldSignals()

    @QtCore.Slot()
    def run(self):
        try:
            res = self._c.fold_variant(self._model, self._chain, **self._kw)
            self.signals.finished.emit(res if isinstance(res, dict) else {"success": False,
                                       "error": "no result"})
        except Exception as exc:                       # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class _SelectSignals(QtCore.QObject):
    done = QtCore.Signal(dict)                          # {"error": str|None}
    failed = QtCore.Signal(str)


class _SelectWorker(QtCore.QRunnable):
    """Runs controller.select_in_3d off the UI thread (the HTTP select). HTTP only —
    the result returns via signal; the cell highlight is already painted locally by Qt
    on click, so the UI never waits on REST. Mirrors _FoldWorker (the proven pattern)."""

    def __init__(self, controller, model, chain, resnums):
        super().__init__()
        self._c, self._model, self._chain, self._resnums = controller, model, chain, resnums
        self.signals = _SelectSignals()

    @QtCore.Slot()
    def run(self):
        try:
            res = self._c.select_in_3d(self._model, self._chain, self._resnums)
            self.signals.done.emit(res if isinstance(res, dict) else {"error": None})
        except Exception as exc:                        # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class _SyncSignals(QtCore.QObject):
    done = QtCore.Signal(list)                          # [(model, chain, resnum), …]
    failed = QtCore.Signal(str)


class _SyncWorker(QtCore.QRunnable):
    """Reads the live 3D selection off the UI thread (the `info residues sel` HTTP read);
    the main thread applies the highlights. Same pattern as the select/fold workers."""

    def __init__(self, controller):
        super().__init__()
        self._c = controller
        self.signals = _SyncSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.done.emit(self._c.sync_from_chimerax())
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


# ── one chain's residue grid ─────────────────────────────────────────────────────

class _ChainGrid(QtWidgets.QTableWidget):
    """A wrapped residue grid for one chain. Cells carry resnum (UserRole); the letter
    shown is the VARIANT residue (WT until edited). Edited cells are highlighted."""

    selectionPushRequested = QtCore.Signal()

    def __init__(self, chain: ChainSeq):
        self.chain = chain
        n = len(chain.cells)
        rows = max(1, (n + _COLS - 1) // _COLS)
        super().__init__(rows, _COLS)
        self.horizontalHeader().setVisible(False)
        self.verticalHeader().setVisible(True)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setShowGrid(True)
        for i, cell in enumerate(chain.cells):
            r, col = divmod(i, _COLS)
            item = QtWidgets.QTableWidgetItem(cell.wt_aa)
            item.setData(_RESNUM_ROLE, cell.resnum)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            item.setToolTip(f"#{cell.model}/{cell.chain}:{cell.resnum}  WT {cell.wt_aa}"
                            f"  (pos {cell.seqpos})")
            self.setItem(r, col, item)
        # row headers = first resnum of each row
        for r in range(rows):
            idx = r * _COLS
            label = str(chain.cells[idx].resnum) if idx < n else ""
            self.setVerticalHeaderItem(r, QtWidgets.QTableWidgetItem(label))
        self.resizeColumnsToContents()
        self.itemSelectionChanged.connect(self.selectionPushRequested)

    def selected_resnums(self) -> List[int]:
        out = []
        for it in self.selectedItems():
            rn = it.data(_RESNUM_ROLE)
            if rn is not None:
                out.append(int(rn))
        return sorted(set(out))

    def refresh_cell(self, resnum: int, aa: str, edited: bool):
        for i, cell in enumerate(self.chain.cells):
            if cell.resnum == resnum:
                r, col = divmod(i, _COLS)
                it = self.item(r, col)
                if it is not None:
                    it.setText(aa)
                    it.setBackground(_EDIT_BG if edited else QtGui.QBrush())
                    it.setToolTip(f"#{cell.model}/{cell.chain}:{cell.resnum}  WT "
                                  f"{cell.wt_aa}" + (f" → {aa}" if edited else ""))
                return

    def highlight_resnums(self, resnums, color):
        wanted = set(resnums)
        self.blockSignals(True)                        # don't echo a push back to 3D
        self.clearSelection()
        for i, cell in enumerate(self.chain.cells):
            if cell.resnum in wanted:
                r, col = divmod(i, _COLS)
                it = self.item(r, col)
                if it is not None:
                    it.setSelected(True)
        self.blockSignals(False)


# ── main window ──────────────────────────────────────────────────────────────────

class SequenceEditorWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: SequenceEditorController, fold_quick: bool = True):
        super().__init__()
        self._c = controller
        self._fold_quick = fold_quick
        self._pool = QtCore.QThreadPool.globalInstance()
        self._grids: dict = {}                         # (model,chain) -> _ChainGrid
        # Debounce/coalesce rapid clicks: each selection change (re)starts a 40 ms
        # single-shot timer; on timeout ONE combined select is dispatched off-thread
        # for the whole current selection (so N quick clicks = 1 round-trip, not N).
        self._pending_grid = None
        self._sel_timer = QtCore.QTimer(self)
        self._sel_timer.setSingleShot(True)
        self._sel_timer.setInterval(40)
        self._sel_timer.timeout.connect(self._flush_selection)
        self.setWindowTitle("StructureBot — Sequence Editor (MVP)")
        self.resize(900, 600)

        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        tb = self.addToolBar("main")
        tb.addAction("Reload models", self.reload_models)
        tb.addAction("Sync from ChimeraX", self.sync_from_chimerax)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Substitute → "))
        self.aa_combo = QtWidgets.QComboBox()
        self.aa_combo.addItems(list(_AA_ORDER))
        tb.addWidget(self.aa_combo)
        tb.addAction("Apply substitution", self.apply_substitution)
        tb.addAction("Revert chain", self.revert_chain)
        tb.addSeparator()
        self.open_chk = QtWidgets.QCheckBox("Open fold in ChimeraX")
        tb.addWidget(self.open_chk)
        self.fold_action = tb.addAction("Fold variant", self.fold_variant)

        self.log = QtWidgets.QPlainTextEdit(readOnly=True, maximumBlockCount=500)
        dock = QtWidgets.QDockWidget("Messages", self)
        dock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)
        self.statusBar().showMessage("Ready")
        self.reload_models()

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _msg(self, text: str):
        self.log.appendPlainText(text)
        self.statusBar().showMessage(text, 8000)

    def _active_grid(self) -> Optional[_ChainGrid]:
        w = self.tabs.currentWidget()
        return w if isinstance(w, _ChainGrid) else None

    # ── actions (all error-first) ─────────────────────────────────────────────────

    def reload_models(self):
        try:
            self.tabs.clear()
            self._grids.clear()
            chains = self._c.load_models()
            if not chains:
                self._msg("No models found (is ChimeraX running with a model loaded "
                          "and the REST server on?).")
                return
            for ch in chains:
                grid = _ChainGrid(ch)
                grid.selectionPushRequested.connect(
                    lambda g=grid: self._on_selection_changed(g))
                self._grids[ch.key] = grid
                self.tabs.addTab(grid, f"#{ch.model}/{ch.chain}  ({len(ch.cells)} aa)")
            self._msg(f"Loaded {len(chains)} chain(s).")
        except Exception as exc:
            self._msg(f"reload failed: {type(exc).__name__}: {exc}")

    def _on_selection_changed(self, grid: _ChainGrid):
        # Returns immediately — Qt has already painted the cell highlight locally; we
        # only (re)arm the debounce timer, so the event loop never blocks on REST.
        self._pending_grid = grid
        self._sel_timer.start()                         # restart → coalesces rapid clicks

    def _flush_selection(self):
        grid = self._pending_grid
        if grid is None:
            return
        resnums = grid.selected_resnums()               # the FULL current selection → 1 command
        if not resnums:
            return
        worker = _SelectWorker(self._c, grid.chain.model, grid.chain.chain, resnums)
        worker.signals.done.connect(self._on_select_done)
        worker.signals.failed.connect(
            lambda e: self._msg(f"select failed: {e}"))
        self._pool.start(worker)                         # HTTP off the UI thread

    @QtCore.Slot(dict)
    def _on_select_done(self, res: dict):
        err = res.get("error") if isinstance(res, dict) else None
        if err:                                          # error-first: surface, never swallow
            self._msg(f"select failed: {err}")

    def sync_from_chimerax(self):
        # Off-thread the `info residues sel` read so the button never freezes the UI.
        worker = _SyncWorker(self._c)
        worker.signals.done.connect(self._apply_sync)
        worker.signals.failed.connect(lambda e: self._msg(f"sync failed: {e}"))
        self._pool.start(worker)

    @QtCore.Slot(list)
    def _apply_sync(self, synced):
        if not synced:
            self._msg("Nothing selected in ChimeraX (or none in a loaded chain).")
            return
        by_key: dict = {}
        for (m, c, rn) in synced:
            by_key.setdefault((m, c), []).append(rn)
        for key, grid in self._grids.items():           # widget updates on the main thread
            grid.highlight_resnums(by_key.get(key, []), _SYNC_BG)
        self._msg(f"Synced {len(synced)} selected residue(s) from ChimeraX.")

    def apply_substitution(self):
        grid = self._active_grid()
        if grid is None:
            return
        aa = self.aa_combo.currentText()
        resnums = grid.selected_resnums()
        if not resnums:
            self._msg("Select one or more residues to substitute.")
            return
        try:
            for rn in resnums:
                self._c.apply_substitution(grid.chain.model, grid.chain.chain, rn, aa)
                edited = rn in grid.chain.edits
                grid.refresh_cell(rn, grid.chain.edits.get(rn, grid.chain.wt_at(rn)),
                                  edited)
            self._msg(f"Applied {aa} to {len(resnums)} residue(s). "
                      f"Variant: {grid.chain.variant_seq[:60]}"
                      f"{'…' if len(grid.chain.variant_seq) > 60 else ''}")
        except Exception as exc:
            self._msg(f"substitution failed: {type(exc).__name__}: {exc}")

    def revert_chain(self):
        grid = self._active_grid()
        if grid is None:
            return
        try:
            self._c.revert_all(grid.chain.model, grid.chain.chain)
            for cell in grid.chain.cells:
                grid.refresh_cell(cell.resnum, cell.wt_aa, False)
            self._msg("Reverted chain to WT.")
        except Exception as exc:
            self._msg(f"revert failed: {type(exc).__name__}: {exc}")

    def fold_variant(self):
        grid = self._active_grid()
        if grid is None:
            return
        seq = grid.chain.variant_seq
        if not seq:
            self._msg("Nothing to fold.")
            return
        self.fold_action.setEnabled(False)
        self._msg(f"Folding #{grid.chain.model}/{grid.chain.chain} "
                  f"({len(seq)} aa, {'edited' if grid.chain.is_edited else 'WT'}) — "
                  f"running off-thread, the window stays responsive…")
        worker = _FoldWorker(self._c, grid.chain.model, grid.chain.chain,
                             quick=self._fold_quick)
        worker.signals.finished.connect(self._on_fold_done)
        worker.signals.failed.connect(self._on_fold_failed)
        self._pool.start(worker)

    @QtCore.Slot(dict)
    def _on_fold_done(self, res: dict):
        self.fold_action.setEnabled(True)
        if not res.get("success"):
            self._msg(f"Fold failed: {res.get('error')}")
            return
        self._msg(f"Fold done: mean pLDDT {res.get('mean_plddt')}  pTM {res.get('ptm')}  "
                  f"→ {res.get('ranked_pdb')}")
        if self.open_chk.isChecked() and res.get("ranked_pdb"):
            try:
                self._c.open_pdb_in_chimerax(res["ranked_pdb"])
                self._msg("Opened folded model in ChimeraX.")
            except Exception as exc:
                self._msg(f"open in ChimeraX failed: {type(exc).__name__}: {exc}")

    @QtCore.Slot(str)
    def _on_fold_failed(self, err: str):
        self.fold_action.setEnabled(True)
        self._msg(f"Fold error: {err}")
