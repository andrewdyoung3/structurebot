"""
variant_workbench.py
--------------------
Variant-Design Workbench — Stage 2 (variant creation + manual edit + property
coloring, BOTH views). A PySide6 panel in the unified GUI: a CLC-style alignment of
the template T + designed variant rows, one tab per UNIQUE chain (homo-oligomer
copies collapsed), a resnum ruler + consensus + conservation tracks, and:

  • column-click → ChimeraX select of that residue in ALL copies (Stage 1);
  • "+ Add variant" → a new row that is an aligned copy of T;
  • manual residue edit of a VARIANT row (combo + Apply; T is the immutable baseline);
  • color MODES (hydrophobicity / charge / cysteine / aromatic) painted on the panel
    cells AND pushed to the 3D over REST — the 3D follows the ACTIVE row (T by default;
    a selected variant takes over and recolors on edit). The 3D coloring is a sequence-
    PROPERTY PREVIEW on the shared backbone (color-by-identity), NOT a remodeled
    structure (rotamers are S4). The sync invariant: active-row panel color == 3D color.

Thin UI over existing logic: reuses `seq_editor.controller` (REST load + multi-copy
select + `run_commands` color push), `seq_library` (grouping + ruler), `variant_model`
(model + tracks + `build_color_commands`), `color_modes` (the single-source registry).
Qt threading mirrors `seq_editor.view`: HTTP (select / color push) runs off the UI
thread; cells repaint locally so the UI never waits on REST.

Later polish (not S2): inline-typing edit (click a cell, type the AA) instead of the
combo+Apply mechanism; per-block column headers; selection debounce if rapid multi-
select gets chatty.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from color_modes import all_modes, get_mode
from seq_library import build_numbering_header_content
from variant_model import (AlignedCell, ChainDesign, DesignSession,
                           build_color_commands, build_design_session, column_tracks)

_COLS = 30                                  # residues per wrapped block
_RESNUM_ROLE = QtCore.Qt.UserRole           # cell → template column index
_ROW_ROLE = QtCore.Qt.UserRole + 1          # cell → row id ("T"/"V1"/… or None)
_AA_ROLE = QtCore.Qt.UserRole + 2           # cell → residue 1-letter ("-"/None for non-seq)
_EDITED_ROLE = QtCore.Qt.UserRole + 3       # cell → bool (variant cell differs from T)
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_T_BG  = QtGui.QColor("#eef4ff")
_EDIT_BG = QtGui.QColor("#ffd27f")          # mirrors seq_editor edited-cell highlight
_RESET_BG = QtGui.QColor("#ffffff")         # neutral / no-opinion under a color mode


# ── off-thread HTTP workers (mirror seq_editor _SelectWorker) ──────────────────────

class _Signals(QtCore.QObject):
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)


class _MultiSelectWorker(QtCore.QRunnable):
    """Runs controller.select_residues_multi off the UI thread (the column→3D select)."""

    def __init__(self, controller, specs):
        super().__init__()
        self._c, self._specs = controller, specs
        self.signals = _Signals()

    @QtCore.Slot()
    def run(self):
        try:
            r = self._c.select_residues_multi(self._specs)
            self.signals.done.emit(r if isinstance(r, dict) else {"error": None})
        except Exception as exc:                       # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


class _ColorWorker(QtCore.QRunnable):
    """Runs controller.run_commands(color_cmds) off the UI thread (the 3D color push).
    The panel cells are already repainted locally, so the UI never waits on REST."""

    def __init__(self, controller, commands):
        super().__init__()
        self._c, self._cmds = controller, commands
        self.signals = _Signals()

    @QtCore.Slot()
    def run(self):
        try:
            r = self._c.run_commands(self._cmds)
            self.signals.done.emit(r if isinstance(r, dict) else {"error": None})
        except Exception as exc:                       # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


# ── one unique-chain tab: T + variants in wrapped column blocks ────────────────────

class _ChainDesignTab(QtWidgets.QScrollArea):
    """Wrapped CLC-style view of one ChainDesign. Rows per block: Ruler, T, each
    variant, Consensus, Conservation. Tracks the ACTIVE row (drives 3D coloring) and
    the current color mode. A cell click emits (row_id, template-column)."""

    cellClicked2 = QtCore.Signal(object, int)   # (row_id: str|None, template col 0-based)

    def __init__(self, design: ChainDesign):
        super().__init__()
        self.design = design
        self.active_row_id: str = "T"               # T drives 3D coloring by default
        self._mode = get_mode("none")               # current color mode (OFF by default)
        self._blocks: List[QtWidgets.QTableWidget] = []
        self._row_ids: List[Optional[str]] = []     # by table row index
        self.setWidgetResizable(True)
        self._build()

    # ── construction (re-run by rebuild() after add/edit) ──────────────────────────
    def _build(self) -> None:
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        v.setSpacing(12)
        self._blocks = []
        design = self.design
        n = len(design.template_cells)

        ruler = build_numbering_header_content(
            [c.resnum for c in design.template_cells if c.resnum is not None], interval=10)
        ruler = (ruler + " " * n)[:n]               # guard length (gap cells)
        consensus, conservation = column_tracks(design)
        cons_pct = ["·▁▂▃▄▅▆▇█"[min(8, int(c * 8))] for c in conservation]

        # row identity, in table-row order: ruler, T, variants…, consensus, conservation
        self._row_ids = [None, "T"] + [vv.id for vv in design.variants] + [None, None]
        labels = ["#", f"T ({design.rep_chain})"] \
            + [vv.id for vv in design.variants] + ["Consensus", "Conservation"]
        tmpl_aa = [c.aa or "-" for c in design.template_cells]
        var_aa = {vv.id: ([c.aa or "-" for c in vv.cells] if len(vv.cells) == n
                          else ["-"] * n) for vv in design.variants}

        for start in range(0, max(1, n), _COLS):
            end = min(start + _COLS, n)
            width = end - start
            block = QtWidgets.QTableWidget(len(labels), width)
            block.setVerticalHeaderLabels(labels)
            block.horizontalHeader().setVisible(False)
            block.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            block.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            for lc in range(width):
                gcol = start + lc
                self._put(block, 0, lc, ruler[gcol], gcol, None, None, faint=True)
                self._put(block, 1, lc, tmpl_aa[gcol], gcol, "T", tmpl_aa[gcol])
                for vi, vv in enumerate(design.variants):
                    aa = var_aa[vv.id][gcol]
                    edited = (aa != "-" and aa != tmpl_aa[gcol])
                    self._put(block, 2 + vi, lc, aa, gcol, vv.id, aa, edited=edited)
                self._put(block, len(labels) - 2, lc, consensus[gcol], gcol, None, None, faint=True)
                self._put(block, len(labels) - 1, lc, cons_pct[gcol], gcol, None, None, faint=True)
            block.resizeColumnsToContents()
            block.resizeRowsToContents()
            block.cellClicked.connect(self._on_cell)
            self._blocks.append(block)
            v.addWidget(block)
        v.addStretch(1)
        self.setWidget(inner)
        self._mark_active_header()
        self.set_color_mode(self._mode)            # re-apply the active mode after a rebuild

    def _put(self, block, row, col, text, gcol, row_id, aa, *, edited=False, faint=False):
        it = QtWidgets.QTableWidgetItem(text)
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        it.setData(_RESNUM_ROLE, gcol)
        it.setData(_ROW_ROLE, row_id)
        it.setData(_AA_ROLE, aa)
        it.setData(_EDITED_ROLE, bool(edited))
        if faint:
            it.setForeground(QtGui.QColor("#888888"))
        if edited:
            f = it.font(); f.setBold(True); it.setFont(f)      # edit visible under any mode
        it.setBackground(self._default_bg(row_id, edited))
        block.setItem(row, col, it)

    @staticmethod
    def _default_bg(row_id, edited) -> QtGui.QBrush:
        if row_id == "T":
            return QtGui.QBrush(_T_BG)
        if edited:
            return QtGui.QBrush(_EDIT_BG)
        return QtGui.QBrush()                                    # clear

    # ── color mode: repaint sequence rows (T + variants) by each cell's aa ──────────
    def set_color_mode(self, mode) -> None:
        """Paint sequence-row cell backgrounds via *mode* (a color_modes.ColorMode — the
        SAME registry that drives the 3D). Under an ACTIVE mode every sequence cell shows
        the mode color, or WHITE for a no-opinion/gap residue, exactly mirroring the 3D
        (reset-to-white + colored runs) so the panel↔3D sync invariant holds for EVERY
        residue. Under the OFF mode ('none') the row defaults return (T tint / edit
        highlight). Ruler/consensus/conservation keep their faint styling regardless."""
        self._mode = mode
        active = mode.fn is not None
        for block in self._blocks:
            for r in range(block.rowCount()):
                for c in range(block.columnCount()):
                    it = block.item(r, c)
                    if it is None:
                        continue
                    row_id = it.data(_ROW_ROLE)
                    if row_id is None:                          # non-sequence row
                        continue
                    if not active:
                        it.setBackground(self._default_bg(row_id, bool(it.data(_EDITED_ROLE))))
                        continue
                    aa = it.data(_AA_ROLE)
                    hexc = mode.color_for(aa) if aa not in (None, "-") else None
                    it.setBackground(QtGui.QBrush(QtGui.QColor(hexc) if hexc else _RESET_BG))

    def color_hex_at(self, row_id: str, col: int) -> Optional[str]:
        """The painted background hex of (row_id, template col) — for the sync-invariant
        test (panel cell color == 3D command color). Scans every wrapped block."""
        for block in self._blocks:
            for r in range(block.rowCount()):
                it = block.item(r, 0)
                if it is None or it.data(_ROW_ROLE) != row_id:
                    continue
                for c in range(block.columnCount()):
                    cell = block.item(r, c)
                    if cell is not None and cell.data(_RESNUM_ROLE) == col:
                        return cell.background().color().name()
        return None

    # ── active row (drives 3D coloring) ────────────────────────────────────────────
    def set_active_row(self, row_id: str) -> None:
        self.active_row_id = row_id
        self._mark_active_header()

    def active_row_cells(self) -> List[AlignedCell]:
        if self.active_row_id == "T":
            return self.design.template_cells
        v = self.design.get_variant(self.active_row_id)
        return v.cells if v is not None else self.design.template_cells

    def _mark_active_header(self) -> None:
        labels = ["#", f"T ({self.design.rep_chain})"] \
            + [vv.id for vv in self.design.variants] + ["Consensus", "Conservation"]
        for block in self._blocks:
            for r, base in enumerate(labels):
                rid = self._row_ids[r] if r < len(self._row_ids) else None
                txt = ("► " + base) if (rid is not None and rid == self.active_row_id) else base
                hdr = QtWidgets.QTableWidgetItem(txt)
                if rid is not None and rid == self.active_row_id:
                    f = hdr.font(); f.setBold(True); hdr.setFont(f)
                block.setVerticalHeaderItem(r, hdr)

    def rebuild(self) -> None:
        """Re-lay the blocks from the (mutated) design — recomputes consensus/
        conservation and re-applies the active color mode + active-row marker."""
        old = self.takeWidget()
        if old is not None:
            old.deleteLater()
        self._build()

    def _on_cell(self, row, col):
        it = self.sender().item(row, col)
        if it is None:
            return
        gcol = it.data(_RESNUM_ROLE)
        if gcol is not None:
            self.cellClicked2.emit(it.data(_ROW_ROLE), int(gcol))


# ── the panel (toolbar + one QTabWidget; a tab per unique chain) ───────────────────

class VariantWorkbenchPanel(QtWidgets.QWidget):
    """Stage-2 Workbench panel. `controller` = a seq_editor.SequenceEditorController
    (shares the ChimeraX bridge). `load_model(model_id)` reads the structure, builds the
    DesignSession, renders the tabs, persists it. Toolbar: add variant, substitute
    (combo+Apply), color mode. Column-click selects in 3D (all copies); a color mode
    paints the panel AND recolors the 3D by the active row (T or a selected variant)."""

    def __init__(self, controller, session=None, pool=None):
        super().__init__()
        self._c = controller
        self._session = session
        self._pool = pool or QtCore.QThreadPool.globalInstance()
        self._design: Optional[DesignSession] = None
        self._edit_target: Optional[Tuple[str, int]] = None   # (variant_id, col)
        self._mode_key = "none"

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)
        self._add_btn = QtWidgets.QPushButton("+ Add variant")
        self._add_btn.clicked.connect(self._add_variant)
        bar.addWidget(self._add_btn)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Substitute →"))
        self._aa_combo = QtWidgets.QComboBox()
        self._aa_combo.addItems(list(_AA_ORDER))
        bar.addWidget(self._aa_combo)
        self._apply_btn = QtWidgets.QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply_substitution)
        bar.addWidget(self._apply_btn)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Color:"))
        self._mode_combo = QtWidgets.QComboBox()
        for m in all_modes():
            self._mode_combo.addItem(m.label, m.key)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(self._mode_combo)
        bar.addStretch(1)
        lay.addLayout(bar)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self._tabs)
        self._status = QtWidgets.QLabel("No structure loaded.")
        self._status.setStyleSheet("color:#888;padding:2px 6px;")
        lay.addWidget(self._status)

    # ── load + render ─────────────────────────────────────────────────────────────
    def load_model(self, model_id: str) -> None:
        """Read the model over REST, build + render the DesignSession, persist it.
        Error-first: a read failure leaves the panel usable."""
        try:
            chain_seqs = self._c.load_model(str(model_id))
        except Exception as exc:
            self._status.setText(f"Workbench: load failed — {type(exc).__name__}: {exc}")
            return
        if not chain_seqs:
            self._status.setText(f"Workbench: model #{model_id} has no chains.")
            return
        self._design = build_design_session(chain_seqs, str(model_id))
        self._edit_target = None
        self._render()
        self._persist()

    def _render(self) -> None:
        self._tabs.clear()
        if not self._design:
            return
        for _ukey, cd in self._design.chains.items():
            tab = _ChainDesignTab(cd)
            tab.cellClicked2.connect(lambda rid, col, t=tab: self._on_cell(t, rid, col))
            copies = "+".join(c for _m, c in cd.members)
            self._tabs.addTab(tab, f"{cd.rep_chain}  ({copies}, {len(cd.template_cells)} aa)")
        self._status.setText(
            f"Workbench: {len(self._design.chains)} unique chain(s). Click a column to "
            f"select in 3D (all copies); add/edit variants; pick a color mode.")

    # ── toolbar actions ────────────────────────────────────────────────────────────
    def _cur_tab(self) -> Optional[_ChainDesignTab]:
        w = self._tabs.currentWidget()
        return w if isinstance(w, _ChainDesignTab) else None

    def _add_variant(self) -> None:
        tab = self._cur_tab()
        if tab is None or self._design is None:
            return
        vid = self._design.new_variant_id()
        tab.design.add_variant(vid)
        tab.rebuild()
        tab.set_active_row(vid)                 # the new row becomes active (ready to edit)
        self._edit_target = None
        self._apply_color_to(tab)
        self._push_3d_color(tab)               # active row changed → 3D follows (= T until edited)
        self._persist()
        self._status.setText(f"Added variant {vid} (aligned copy of T) — now the active row.")

    def _apply_substitution(self) -> None:
        tab = self._cur_tab()
        if tab is None:
            return
        if self._edit_target is None:
            self._status.setText("Select a residue in a VARIANT row first (T is the immutable template).")
            return
        vid, col = self._edit_target
        aa = self._aa_combo.currentText()
        try:
            tab.design.edit_variant(vid, col, aa)
        except Exception as exc:
            self._status.setText(f"Substitution failed: {type(exc).__name__}: {exc}")
            return
        tab.rebuild()
        tab.set_active_row(vid)
        self._apply_color_to(tab)
        if tab.active_row_id == vid:
            self._push_3d_color(tab)           # active variant edited → recolor 3D (all copies)
        self._persist()
        resnum = tab.design.resnum_for_col(col)
        self._status.setText(f"{vid}: residue {resnum} → {aa}. "
                             f"3D recolored by {vid} (preview on the shared backbone).")

    def _on_mode_changed(self) -> None:
        self._mode_key = self._mode_combo.currentData() or "none"
        tab = self._cur_tab()
        if tab is None:
            return
        self._apply_color_to(tab)
        self._push_3d_color(tab)
        if self._mode_key == "none":
            self._status.setText("Color: None — panel cleared; 3D left as-is (non-destructive).")
        else:
            self._status.setText(f"Color: {get_mode(self._mode_key).label} — "
                                 f"3D follows the active row ({tab.active_row_id}).")

    def _on_tab_changed(self, _idx) -> None:
        tab = self._cur_tab()
        if tab is None:
            return
        self._edit_target = None
        self._apply_color_to(tab)
        self._push_3d_color(tab)

    # ── cell click: 3D-select the column; set active row / edit target ──────────────
    def _on_cell(self, tab: _ChainDesignTab, row_id, col: int) -> None:
        self._select_column(tab.design, col)               # column→3D select (all copies)
        if row_id == "T":
            tab.set_active_row("T")
            self._edit_target = None
            self._push_3d_color(tab)
        elif row_id is not None:                            # a variant row
            tab.set_active_row(row_id)
            self._edit_target = (row_id, col)
            resnum = tab.design.resnum_for_col(col)
            wt = tab.design.template_cells[col].aa if 0 <= col < len(tab.design.template_cells) else "?"
            self._status.setText(f"Edit target: {row_id} col {col} (residue {resnum}, T={wt}). "
                                 f"Pick an aa and Apply.")
            self._push_3d_color(tab)

    # ── coloring helpers ───────────────────────────────────────────────────────────
    def _apply_color_to(self, tab: _ChainDesignTab) -> None:
        tab.set_color_mode(get_mode(self._mode_key))

    def _push_3d_color(self, tab: _ChainDesignTab) -> None:
        """Recolor the 3D by the tab's ACTIVE row across all copies. No-op for the OFF
        mode (non-destructive: we do not know the pre-overlay coloring to restore)."""
        mode = get_mode(self._mode_key)
        if mode.fn is None:                                 # "none" → leave 3D untouched
            return
        cmds = build_color_commands(tab.active_row_cells(), tab.design.members, mode.color_for)
        if not cmds:
            return
        w = _ColorWorker(self._c, cmds)
        w.signals.failed.connect(lambda e: self._status.setText(f"Workbench 3D color failed: {e}"))
        self._pool.start(w)

    # ── column-click → 3D select (ALL copies), off the UI thread ───────────────────
    def _select_column(self, design: ChainDesign, col: int) -> None:
        specs = self.select_specs_for_column(design, col)
        if not specs:
            return
        w = _MultiSelectWorker(self._c, specs)
        w.signals.failed.connect(lambda e: self._status.setText(f"Workbench select failed: {e}"))
        self._pool.start(w)

    # exposed for tests / live-verify: the exact specs a column click dispatches
    def select_specs_for_column(self, design: ChainDesign, col: int):
        resnum = design.resnum_for_col(col)
        if resnum is None:
            return []
        return [(m, c, [resnum]) for (m, c) in design.members]

    # exposed for tests / live-verify: the exact color commands the active row pushes
    def color_commands_for(self, tab: _ChainDesignTab):
        mode = get_mode(self._mode_key)
        if mode.fn is None:
            return []
        return build_color_commands(tab.active_row_cells(), tab.design.members, mode.color_for)

    def _persist(self) -> None:
        if self._session is None or self._design is None:
            return
        try:
            self._session.add_design_session(self._design.model_id, self._design.to_dict())
        except Exception:
            pass
