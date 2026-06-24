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

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets

from color_modes import (all_modes, combined_disruption_color, ddg_color,
                         get_mode, plddt_color)
from seq_library import (build_numbering_header_content,
                         build_numbering_header_with_insertions)
from variant_model import (AlignedCell, ChainDesign, DesignSession,
                           build_color_commands, build_color_commands_by_resnum,
                           build_model_color_commands, build_fold_column_map,
                           build_design_session, build_design_session_from_sequence,
                           column_tracks, DesignSession,
                           filter_new_mpnn_variants, group_scan_suggestions,
                           fold_summary, import_mpnn_designs, stability_summary,
                           suggestion_color)

_COLS = 30                                  # residues per wrapped block
_SUGGEST_ROW = "__suggest__"                # sentinel row id for the inline Suggest track
_RESULT_DDG_MODE = "result:ddg"             # S4a result-backed color mode (per-residue ddG)
_RESULT_PLDDT_MODE = "result:plddt"         # S4b result-backed color mode (per-residue pLDDT)
_RESULT_DEVIATION_MODE = "result:deviation" # S4c floor-gated variant-vs-WT Cα deviation
_DEVIATION_FLOOR_MIN_A = 0.25               # mirrors ToolRouter._DEVIATION_FLOOR_MIN_A (gate floor)
_LDDT_NEUTRAL_CAP = 0.9                      # mirrors ToolRouter._LDDT_NEUTRAL_CAP (lDDT gate cap)
_DDM_FLOOR_MIN_A = 0.5                       # mirrors ToolRouter._DDM_FLOOR_MIN_A (dRMSD gate floor)
_FOLD_ENGINES = ("esmfold", "boltz", "colabfold")   # engines whose step data the fold seam consumes
_LOCAL_FOLD_ENGINES = ("esmfold", "boltz")  # LOCAL-ONLY engines; colabfold leaves the boundary (remote MSA)
_GUIDED_TEMPLATE_THRESHOLD_A = 10.0         # default Boltz template force-threshold (Å) for hard steering
_RESNUM_ROLE = QtCore.Qt.UserRole           # cell → template column index
_ROW_ROLE = QtCore.Qt.UserRole + 1          # cell → row id ("T"/"V1"/… / _SUGGEST_ROW / None)
_AA_ROLE = QtCore.Qt.UserRole + 2           # cell → residue 1-letter ("-"/None for non-seq)
_EDITED_ROLE = QtCore.Qt.UserRole + 3       # cell → bool (variant cell differs from T)
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
_T_BG  = QtGui.QColor("#eef4ff")
_EDIT_BG = QtGui.QColor("#ffd27f")          # mirrors seq_editor edited-cell highlight
_RESET_BG = QtGui.QColor("#ffffff")         # neutral / no-opinion under a color mode
_GLYPH_DARK = QtGui.QColor("#1a1a1a")       # default readable glyph on light/default cells
# Out-of-focus (non-active) rows under a RESULT mode: an EXPLICIT light bg + a dim-but-clearly
# -readable glyph. Explicit (not a cleared/inherited brush) so it never goes dark-on-dark under
# a dark widget palette — de-emphasised vs the active row, yet legible.
_DIM_BG = QtGui.QColor("#eceef2")
_DIM_FG = QtGui.QColor("#3c4250")
# Default (no colour-mode) VARIANT sequence cell: an EXPLICIT medium-light grey so the black
# glyph is high-contrast, instead of a cleared brush that inherited the dark widget base
# (the old black-on-dark-grey). Distinct from the T tint (#eef4ff) + the edit highlight (#ffd27f).
_VARIANT_BG = QtGui.QColor("#d3d7dd")


def _contrast_fg(qcolor: QtGui.QColor) -> QtGui.QColor:
    """A glyph colour that stays legible on *qcolor*: dark glyph on a light cell, white glyph
    on a dark one (e.g. deep-blue pLDDT >90 / saturated ddG). Keeps every coloured cell's
    letter readable rather than black-on-dark."""
    lum = (0.299 * qcolor.red() + 0.587 * qcolor.green() + 0.114 * qcolor.blue()) / 255.0
    return _GLYPH_DARK if lum > 0.55 else QtGui.QColor("#ffffff")


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


class _FsSignals(QtCore.QObject):
    done = QtCore.Signal(list, list)                    # (primary [(pid,ch,TM)…], low-bucket [(…)…])
    failed = QtCore.Signal(str)


class _FoldseekWorker(QtCore.QRunnable):
    """Runs the LOCAL-ONLY foldseek structural-neighbour search off the UI thread (Stage 2
    template auto-discovery). The search is seconds-fast but WSL-bound, so it never blocks Qt.
    Requests the two-bucket return: the primary list (TM≥min_tm) + the low-confidence bucket
    ([low_bound, min_tm)) for the "show lower-confidence hits" expander — both from ONE search."""

    def __init__(self, bridge, query_path, max_results=30, min_tm=0.3, low_bound=0.2):
        super().__init__()
        self._b, self._q = bridge, query_path
        self._n, self._t, self._low = max_results, min_tm, low_bound
        self.signals = _FsSignals()

    @QtCore.Slot()
    def run(self):
        try:
            primary, low = self._b.search_neighbors(
                self._q, self._n, self._t, with_low_bucket=True, low_bound=self._low)
            self.signals.done.emit(list(primary), list(low))
        except Exception as exc:                       # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


# ── one unique-chain tab: T + variants in wrapped column blocks ────────────────────

class _ChainDesignTab(QtWidgets.QScrollArea):
    """Wrapped CLC-style view of one ChainDesign. Rows per block: Ruler, T, each
    variant, [Suggest], Consensus, Conservation. Tracks the ACTIVE row (drives 3D
    coloring) and the current color mode. A cell click emits (row_id, template-column);
    a click on the sparse inline Suggest track emits (_SUGGEST_ROW, col)."""

    cellClicked2 = QtCore.Signal(object, int, bool)   # (row_id, template col, ctrl-held)
    rowHeaderSelected = QtCore.Signal(object)         # row_id (header click → SELECT active row)
    rowMenuRequested  = QtCore.Signal(object, object)  # (row_id, global QPoint) — header right-click
    cellMenuRequested = QtCore.Signal(object, int, object)  # (row_id, col, global QPoint)

    def __init__(self, design: ChainDesign, suggestions: Optional[Dict[int, List[dict]]] = None):
        super().__init__()
        self.design = design
        self.suggestions: Dict[int, List[dict]] = dict(suggestions or {})  # col -> ranked cands
        self.active_row_id: str = "T"               # T drives 3D coloring by default
        self._mode = get_mode("none")               # current color mode (OFF by default)
        self.badges: Dict[str, str] = {}            # vid -> inline result badge (S4a)
        self._result_coloring: Optional[Tuple[str, Dict[int, str]]] = None  # (row_id, resnum->hex)
        self._blocks: List[QtWidgets.QTableWidget] = []
        self._row_ids: List[Optional[str]] = []     # by table row index
        self._vp_to_block: Dict[Any, QtWidgets.QTableWidget] = {}  # header viewport → its block
        self.setWidgetResizable(True)
        self._build()

    # ── construction (re-run by rebuild() after add/edit) ──────────────────────────
    def _build(self) -> None:
        inner = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(inner)
        v.setSpacing(12)
        self._blocks = []
        self._vp_to_block = {}
        design = self.design
        n = len(design.template_cells)

        # Per-column ruler: real columns numbered, INSERTED (template-gap) columns get PDB
        # insertion-code letters (52A, 52B…). Length == n by construction (one entry/column).
        ruler = build_numbering_header_with_insertions(
            [c.resnum for c in design.template_cells], interval=10)
        consensus, conservation = column_tracks(design)
        cons_pct = ["·▁▂▃▄▅▆▇█"[min(8, int(c * 8))] for c in conservation]

        # The inline Suggest track appears ONLY when a scan produced candidates for this
        # chain (sparse by construction — never implies a suggestion where none was run).
        has_sugg = bool(self.suggestions)
        sugg_label = ["Suggest"] if has_sugg else []
        sugg_rid   = [_SUGGEST_ROW] if has_sugg else []

        # row identity, in table-row order: ruler, T, variants…, [Suggest], consensus, conservation
        self._row_ids = [None, "T"] + [vv.id for vv in design.variants] + sugg_rid + [None, None]
        labels = ["#", f"T ({design.rep_chain})"] \
            + [self._variant_label(vv.id) for vv in design.variants] \
            + sugg_label + ["Consensus", "Conservation"]
        tmpl_aa = [c.aa or "-" for c in design.template_cells]
        var_aa = {vv.id: ([c.aa or "-" for c in vv.cells] if len(vv.cells) == n
                          else ["-"] * n) for vv in design.variants}
        n_var = len(design.variants)
        sugg_row = (2 + n_var) if has_sugg else -1     # table-row index of the Suggest track

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
                if has_sugg:
                    self._put_suggest(block, sugg_row, lc, gcol)
                self._put(block, len(labels) - 2, lc, consensus[gcol], gcol, None, None, faint=True)
                self._put(block, len(labels) - 1, lc, cons_pct[gcol], gcol, None, None, faint=True)
            block.resizeColumnsToContents()
            block.resizeRowsToContents()
            self._size_block_to_content(block)         # full-height, no per-block scrollbar (unified scroll)
            block.cellClicked.connect(self._on_cell)
            block.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            block.customContextMenuRequested.connect(
                lambda pos, b=block: self._on_context_menu(b, pos))
            # Row-header clicks: NAME region → SELECT (active row); BADGE region → result
            # detail. sectionClicked gives only the row index (not the click-x), so an event
            # filter on the header viewport reads the x to split name-vs-badge.
            vh = block.verticalHeader()
            vh.setDefaultAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            vp = vh.viewport()
            vp.installEventFilter(self)
            self._vp_to_block[vp] = block
            self._blocks.append(block)
            v.addWidget(block)
        v.addStretch(1)
        self.setWidget(inner)
        self._mark_active_header()
        self.set_color_mode(self._mode)            # re-apply the active mode after a rebuild

    @staticmethod
    def _size_block_to_content(block: QtWidgets.QTableWidget) -> None:
        """Make a block show ALL its rows with NO internal scrollbar, so the OUTER QScrollArea is
        the SINGLE scroller (the unified block-scroll: T + variants + consensus/conservation scroll
        together). Fixed to content height + width; with fixed-count blocks the outer area provides
        ONE horizontal scrollbar if a block is wider than the window. Cause-targeting: the cause of
        the old per-block scrollbars was each QTableWidget self-capping its height — this removes it
        without touching any cell data / coupling (everything stays keyed on item data roles)."""
        block.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        block.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        block.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        fr = 2 * block.frameWidth()
        # horizontalHeader is hidden → no column-header height term; the vertical (row-label)
        # header sits to the LEFT, so it adds to width, not height.
        h = sum(block.rowHeight(r) for r in range(block.rowCount())) + fr
        w = (sum(block.columnWidth(c) for c in range(block.columnCount()))
             + block.verticalHeader().width() + fr)
        block.setFixedHeight(h)
        block.setFixedWidth(w)

    def _variant_label(self, vid: str) -> str:
        """Variant row header: the id plus its inline result badge (S4a), if any."""
        badge = self.badges.get(vid)
        return f"{vid}  {badge}" if badge else vid

    def eventFilter(self, obj, event) -> bool:
        """Row-header click router. LEFT-click ANYWHERE on a row header → SELECT that variant
        as the active row. RIGHT-click on a VARIANT header → emit rowMenuRequested (the panel
        shows a 'Delete variant' menu). T (template) has no menu. The per-mutation result-
        DETAIL display is PARKED. Returns False (never consumes — normal header painting
        proceeds)."""
        block = self._vp_to_block.get(obj)
        if block is None:
            return False
        if event.type() != QtCore.QEvent.Type.MouseButtonPress:
            return False
        if event.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            return False
        vh = block.verticalHeader()
        try:
            y = int(event.position().y())                # Qt6 QMouseEvent
        except AttributeError:
            y = event.y()
        section = vh.logicalIndexAt(y)
        rid = self._row_ids[section] if 0 <= section < len(self._row_ids) else None
        if rid in (None, _SUGGEST_ROW):
            return False
        if event.button() == QtCore.Qt.RightButton:
            if rid != "T":                               # T is the immutable template — no delete
                self.rowMenuRequested.emit(rid, event.globalPosition().toPoint()
                                           if hasattr(event, "globalPosition")
                                           else event.globalPos())
            return False
        self.rowHeaderSelected.emit(rid)                 # left-click → SELECT (always)
        return False

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

    def _put_suggest(self, block, row, col, gcol):
        """One inline Suggest-track cell: the top-ranked candidate's residue colored by
        combined_score (sparse — blank where the scan produced nothing for this column)."""
        cands = self.suggestions.get(gcol)
        if cands:
            top = cands[0]
            it = QtWidgets.QTableWidgetItem(str(top.get("to_aa", "?")))
            it.setData(_ROW_ROLE, _SUGGEST_ROW)
            it.setBackground(QtGui.QBrush(QtGui.QColor(
                suggestion_color(top.get("combined_score", 0.0)))))
            f = it.font(); f.setBold(True); it.setFont(f)
            it.setToolTip("\n".join(
                f"{c.get('from_aa','?')}{c.get('resnum','?')}{c.get('to_aa','?')}  "
                f"score {c.get('combined_score', 0.0):+.2f}"
                + (f"  ·  {c.get('recommendation','')}" if c.get("recommendation") else "")
                for c in cands))
        else:
            it = QtWidgets.QTableWidgetItem("")              # sparse: nothing here
            it.setData(_ROW_ROLE, None)                      # not clickable
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        it.setData(_RESNUM_ROLE, gcol)
        block.setItem(row, col, it)

    @staticmethod
    def _default_bg(row_id, edited) -> QtGui.QBrush:
        if row_id == "T":
            return QtGui.QBrush(_T_BG)
        if edited:
            return QtGui.QBrush(_EDIT_BG)
        return QtGui.QBrush(_VARIANT_BG)         # EXPLICIT medium-light grey (was clear → dark-on-dark)

    # ── color mode: repaint sequence rows (T + variants) by each cell's aa ──────────
    def set_color_mode(self, mode) -> None:
        """Paint sequence-row cell backgrounds via *mode* (a color_modes.ColorMode — the
        SAME registry that drives the 3D). Under an ACTIVE mode every sequence cell shows
        the mode color, or WHITE for a no-opinion/gap residue, exactly mirroring the 3D
        (reset-to-white + colored runs) so the panel↔3D sync invariant holds for EVERY
        residue. Under the OFF mode ('none') the row defaults return (T tint / edit
        highlight). Ruler/consensus/conservation keep their faint styling regardless."""
        self._mode = mode
        self._result_coloring = None                # leaving any result-mode coloring
        active = mode.fn is not None
        for block in self._blocks:
            for r in range(block.rowCount()):
                for c in range(block.columnCount()):
                    it = block.item(r, c)
                    if it is None:
                        continue
                    row_id = it.data(_ROW_ROLE)
                    if row_id is None or row_id == _SUGGEST_ROW:   # non-sequence / score-colored
                        continue
                    if not active:
                        it.setBackground(self._default_bg(row_id, bool(it.data(_EDITED_ROLE))))
                        it.setForeground(QtGui.QBrush(_GLYPH_DARK))
                        continue
                    aa = it.data(_AA_ROLE)
                    hexc = mode.color_for(aa) if aa not in (None, "-") else None
                    if hexc:
                        qc = QtGui.QColor(hexc)
                        it.setBackground(QtGui.QBrush(qc))
                        it.setForeground(QtGui.QBrush(_contrast_fg(qc)))
                    else:
                        it.setBackground(QtGui.QBrush(_RESET_BG))
                        it.setForeground(QtGui.QBrush(_GLYPH_DARK))

    # ── result color mode (S4a): paint ONE row by per-residue computed value ─────────
    def set_result_coloring(self, active_row_id: str, resnum_to_hex: Dict[int, str]) -> None:
        """Paint the ACTIVE row's cells by a per-RESNUM result value (e.g. ddG via
        color_modes.ddg_color), mirroring the 3D (which recolors only the active row's
        residues — its no-data cells reset to white). NON-active rows keep their PLAIN
        readable default (T tint / edit highlight / clear) — NOT a white blank — so every
        sequence stays legible (the white-reset is active-row↔3D sync, not a global blank).
        Coloured cells get a contrasting glyph. Records the coloring so rebuild() re-applies."""
        self._mode = get_mode("none")
        self._result_coloring = (active_row_id, dict(resnum_to_hex))
        for block in self._blocks:
            for r in range(block.rowCount()):
                for c in range(block.columnCount()):
                    it = block.item(r, c)
                    if it is None:
                        continue
                    row_id = it.data(_ROW_ROLE)
                    if row_id is None or row_id == _SUGGEST_ROW:
                        continue
                    if row_id != active_row_id:                  # non-active → dim BUT readable
                        edited = bool(it.data(_EDITED_ROLE))
                        if row_id == "T":
                            bg = _T_BG                            # keep T's identity tint
                        elif edited:
                            bg = _EDIT_BG                         # keep edits visible
                        else:
                            bg = _DIM_BG                          # EXPLICIT light (never clear→dark)
                        it.setBackground(QtGui.QBrush(bg))
                        it.setForeground(QtGui.QBrush(_GLYPH_DARK if edited else _DIM_FG))
                        continue
                    rn = self.design.resnum_for_col(it.data(_RESNUM_ROLE))
                    hexc = resnum_to_hex.get(rn)
                    if hexc:
                        qc = QtGui.QColor(hexc)
                        it.setBackground(QtGui.QBrush(qc))
                        it.setForeground(QtGui.QBrush(_contrast_fg(qc)))
                    else:
                        it.setBackground(QtGui.QBrush(_RESET_BG))   # active row, no data → reset
                        it.setForeground(QtGui.QBrush(_GLYPH_DARK))

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
        # base labels by row id (mirrors _build's layout, incl. the optional Suggest row)
        base_by_rid = {None: "", "T": f"T ({self.design.rep_chain})",
                       _SUGGEST_ROW: "Suggest"}
        for vv in self.design.variants:
            base_by_rid[vv.id] = self._variant_label(vv.id)
        tails = ["#", "Consensus", "Conservation"]   # the three None-id rows, in order
        for block in self._blocks:
            ti = iter(tails)
            for r, rid in enumerate(self._row_ids):
                base = base_by_rid.get(rid, "") if rid is not None else next(ti)
                active = rid is not None and rid == self.active_row_id
                hdr = QtWidgets.QTableWidgetItem(("► " + base) if active else base)
                if active:
                    f = hdr.font(); f.setBold(True); hdr.setFont(f)
                block.setVerticalHeaderItem(r, hdr)

    def rebuild(self) -> None:
        """Re-lay the blocks from the (mutated) design — recomputes consensus/
        conservation and re-applies the active color mode + active-row marker."""
        old = self.takeWidget()
        if old is not None:
            old.deleteLater()
        self._build()

    def set_suggestions(self, suggestions: Dict[int, List[dict]]) -> None:
        """Replace the inline Suggest-track data (per-column ranked candidates) and
        re-lay. Empty → the track disappears (sparse / honest by absence)."""
        self.suggestions = dict(suggestions or {})
        self.rebuild()

    def _on_cell(self, row, col):
        it = self.sender().item(row, col)
        if it is None:
            return
        gcol = it.data(_RESNUM_ROLE)
        if gcol is not None:
            # Ctrl(+Cmd)-click is the DISTINCT scan-set gesture; a plain click keeps its
            # S2 meaning (edit target / active row). Disambiguated here at the source.
            mods = QtWidgets.QApplication.keyboardModifiers()
            ctrl = bool(mods & (QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier))
            self.cellClicked2.emit(it.data(_ROW_ROLE), int(gcol), ctrl)

    def _on_context_menu(self, block, pos) -> None:
        """Right-click on a cell → ask the panel for the substitute/delete menu. Only the
        cell identity + a global screen position travel up; the panel owns the actions."""
        it = block.itemAt(pos)
        if it is None:
            return
        gcol = it.data(_RESNUM_ROLE)
        row_id = it.data(_ROW_ROLE)
        if gcol is None or row_id in (None, "T", _SUGGEST_ROW):
            return                                  # only editable VARIANT residues get a menu
        self.cellMenuRequested.emit(row_id, int(gcol), block.viewport().mapToGlobal(pos))


class _AddSequenceDialog(QtWidgets.QDialog):
    """De-novo construct input: a name + ONE-OR-MORE distinct chain sequences, each with a
    copy-count (the known stoichiometry, e.g. PCNA×3 + p21×3). Each row is a sequence box +
    a copies spinner; rows add/remove dynamically. Every sequence is standard-AA validated on
    accept. `result_chains()` returns [(sequence, copies), …] — the chain LIST the model seeds
    one ChainDesign per distinct sequence from. Pure Qt — no ChimeraX."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add sequence (de-novo construct)")
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel("Name:"))
        self._name = QtWidgets.QLineEdit()
        self._name.setPlaceholderText("e.g. PCNA_p21_complex")
        lay.addWidget(self._name)
        lay.addWidget(QtWidgets.QLabel(
            "Chains (1-letter; spaces/newlines ignored) — one row per DISTINCT sequence, "
            "with its copy count:"))
        # Dynamic chain rows live in their own VBox so add/remove just edits this layout.
        self._rows: List[Tuple[QtWidgets.QPlainTextEdit, QtWidgets.QSpinBox, QtWidgets.QWidget]] = []
        self._rows_box = QtWidgets.QVBoxLayout()
        lay.addLayout(self._rows_box)
        self._add_row_btn = QtWidgets.QPushButton("+ Add chain")
        self._add_row_btn.clicked.connect(lambda: self._add_row())
        lay.addWidget(self._add_row_btn)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._add_row()                          # start with one chain row

    def _add_row(self) -> None:
        row = QtWidgets.QWidget()
        rl = QtWidgets.QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        seq = QtWidgets.QPlainTextEdit()
        seq.setPlaceholderText("MKVLW…")
        seq.setMaximumHeight(64)
        rl.addWidget(seq, 1)
        spin = QtWidgets.QSpinBox()
        spin.setRange(1, 26)                      # A..Z is the chain-id ceiling
        spin.setValue(1)
        spin.setPrefix("×")
        spin.setToolTip("Number of identical copies of this chain (stoichiometry).")
        rl.addWidget(spin)
        rm = QtWidgets.QPushButton("–")
        rm.setFixedWidth(28)
        rm.setToolTip("Remove this chain")
        rm.clicked.connect(lambda _checked=False, w=row: self._remove_row(w))
        rl.addWidget(rm)
        self._rows.append((seq, spin, row))
        self._rows_box.addWidget(row)

    def _remove_row(self, row: QtWidgets.QWidget) -> None:
        if len(self._rows) <= 1:                  # always keep at least one chain row
            return
        self._rows = [r for r in self._rows if r[2] is not row]
        self._rows_box.removeWidget(row)
        row.deleteLater()

    def _on_ok(self) -> None:
        total = 0
        for seq_w, spin, _row in self._rows:
            seq = "".join(seq_w.toPlainText().split()).upper()
            if not seq or any(a not in _AA_ORDER for a in seq):
                QtWidgets.QMessageBox.warning(self, "Invalid sequence",
                                              "Each chain must be a non-empty standard amino-acid "
                                              "sequence (the 20 one-letter codes).")
                return
            total += spin.value()
        if total > 26:
            QtWidgets.QMessageBox.warning(self, "Too many chains",
                                          "Total copies across all chains exceed 26 (the A–Z "
                                          "chain-id ceiling). Reduce the copy counts.")
            return
        self.accept()

    def result_name(self) -> str:
        return self._name.text().strip() or "construct"

    def result_chains(self) -> List[Tuple[str, int]]:
        """[(sequence, copies), …] in row order — distinct chains for the construct."""
        return [("".join(seq_w.toPlainText().split()).upper(), int(spin.value()))
                for seq_w, spin, _row in self._rows]


# ── the panel (toolbar + one QTabWidget; a tab per unique chain) ───────────────────

class VariantWorkbenchPanel(QtWidgets.QWidget):
    """Stage-3b Workbench panel. `controller` = a seq_editor.SequenceEditorController
    (shares the ChimeraX bridge). `load_model(model_id)` reads the structure, builds the
    DesignSession, renders the tabs, persists it. Toolbar: add variant, substitute
    (combo+Apply), color mode, AND the Stage-3b tool LAUNCH buttons ("Run ProteinMPNN…",
    "Scan…"). Column-click toggles the position into the SCAN SET (the deterministic scan
    scope) and selects the whole set in 3D (all copies); a color mode paints the panel AND
    recolors the 3D by the active row.

    Stage-3b launches go through the SAME engine spine as the NL path: a click builds a
    deterministic launch spec and emits `launchRequested(spec)`; the window runs it on a
    worker thread via `engine.handle_tool_request` (so the mutation-scan confirm-gate/
    tiering fires and the real subprocess runs), then calls the S3a consume path
    (`_import_mpnn` / `_load_suggestions`) so results auto-render. The panel never launches
    a tool itself — it has no engine; it only describes the request."""

    # Stage 3b: panel → window. Payload = {tool, tool_inputs, user_input, confidence,
    # refresh}. The window turns it into engine.handle_tool_request on the worker thread.
    launchRequested = QtCore.Signal(dict)

    def __init__(self, controller, session=None, pool=None):
        super().__init__()
        self._c = controller
        self._session = session
        self._pool = pool or QtCore.QThreadPool.globalInstance()
        self._design: Optional[DesignSession] = None
        self._edit_target: Optional[Tuple[str, int]] = None   # (variant_id, col)
        self._scan_cols: set = set()        # template columns chosen as the scan scope
        self._mode_key = "none"
        self._scan_cache_snapshot = None    # (model_id, prior scan cache) for stability runs
        self._tiled = False                 # True after "Tile folds"; next row-select un-tiles

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        # The toolbar groups the low-frequency tool/fold controls behind two QToolButton
        # menus (Tools ▾ / Fold ▾) so the row sheds width and the window narrows to ~content
        # width; the high-frequency authoring/test/colour controls stay as direct widgets.
        # The menu entries are QActions wired to the SAME handlers the old buttons used —
        # no parallel UI path. (State read elsewhere — _scan_set_lbl.text(), _fold_vis_btn/
        # _show_fold_cb/_show_ref_cb .isChecked()/.setChecked() — works identically on QAction.)
        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(6, 4, 6, 0)
        self._add_btn = QtWidgets.QPushButton("+ Add variant")
        self._add_btn.clicked.connect(self._add_variant)
        bar.addWidget(self._add_btn)
        # DE-NOVO: seed a workbench design from a typed sequence — no crystal. Nothing renders
        # until the construct is folded (Fold ▾ → Fold construct).
        self._add_seq_btn = QtWidgets.QPushButton("Add sequence")
        self._add_seq_btn.setToolTip("Start a de-novo construct from a typed/pasted sequence "
                                     "(no loaded structure). Fold it as a mono/di/tri/tetramer.")
        self._add_seq_btn.clicked.connect(self._on_add_sequence)
        bar.addWidget(self._add_seq_btn)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Substitute →"))
        self._aa_combo = QtWidgets.QComboBox()
        self._aa_combo.addItems(list(_AA_ORDER))
        bar.addWidget(self._aa_combo)
        self._apply_btn = QtWidgets.QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply_substitution)
        bar.addWidget(self._apply_btn)
        bar.addSpacing(12)

        # Tools ▾ — Stage-3b LAUNCH (Scan / Run ProteinMPNN, through the engine spine) +
        # Stage-3a cached-result IMPORT + the scan-set status/clear.
        self._tools_btn = QtWidgets.QToolButton()
        self._tools_btn.setText("Tools")
        self._tools_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._tools_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        tools_menu = QtWidgets.QMenu(self._tools_btn)
        tools_menu.setToolTipsVisible(True)
        self._scan_btn = tools_menu.addAction("Scan…")
        self._scan_btn.setToolTip("Mutation-scan the scan set (Ctrl+click residues to build "
                                  "it; whole chain if empty) through the tool spine.")
        self._scan_btn.triggered.connect(self._on_scan_clicked)
        self._mpnn_run_btn = tools_menu.addAction("Run ProteinMPNN…")
        self._mpnn_run_btn.setToolTip("Redesign the chain (the scan set via Ctrl+click, or "
                                      "whole chain) with ProteinMPNN through the tool spine.")
        self._mpnn_run_btn.triggered.connect(self._on_mpnn_clicked)
        tools_menu.addSeparator()
        # Stage 3a: pull the latest cached tool results into the panel (import = capture).
        self._import_btn = tools_menu.addAction("Import MPNN designs")
        self._import_btn.triggered.connect(self._import_mpnn)
        self._sugg_btn = tools_menu.addAction("Load scan suggestions")
        self._sugg_btn.triggered.connect(self._load_suggestions)
        tools_menu.addSeparator()
        self._scan_set_lbl = tools_menu.addAction("scan set: 0")
        self._scan_set_lbl.setEnabled(False)          # display-only live count
        self._clear_scan_btn = tools_menu.addAction("Clear scan set")
        self._clear_scan_btn.setToolTip("Clear the scan set.")
        self._clear_scan_btn.triggered.connect(self._clear_scan_set)
        self._tools_btn.setMenu(tools_menu)
        bar.addWidget(self._tools_btn)
        bar.addSpacing(12)

        # Stage 4a: per-variant action buttons (act on the ACTIVE variant row).
        # Stability runs the 4-axis voter on the variant's EXACT mutations through the
        # engine spine (deep → confirm-gate); solubility is the pure CamSol scalar (instant).
        self._stab_btn = QtWidgets.QPushButton("Test stability")
        self._stab_btn.setToolTip("Score the ACTIVE variant's exact mutations (4-axis ddG "
                                  "voter) through the tool spine. Deep adds Rosetta (gated).")
        self._stab_btn.clicked.connect(self._on_test_stability)
        bar.addWidget(self._stab_btn)
        self._sol_btn = QtWidgets.QPushButton("Test solubility")
        self._sol_btn.setToolTip("CamSol intrinsic-solubility of the ACTIVE variant vs the "
                                 "template (instant, local).")
        self._sol_btn.clicked.connect(self._on_test_solubility)
        bar.addWidget(self._sol_btn)
        bar.addSpacing(12)

        # Fold ▾ — Stage-4b fold the ACTIVE variant (engine picker; ESMFold local) +
        # fold/reference visibility + the tile escape-hatch.
        self._fold_menu_btn = QtWidgets.QToolButton()
        self._fold_menu_btn.setText("Fold")
        self._fold_menu_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self._fold_menu_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        fold_menu = QtWidgets.QMenu(self._fold_menu_btn)
        fold_menu.setToolTipsVisible(True)
        # Fold the ACTIVE variant, opening a real pLDDT-coloured model matchmaker-overlaid
        # on the template. Gated.
        self._fold_btn = fold_menu.addAction("Fold…")
        self._fold_btn.setToolTip("Fold the ACTIVE variant (you pick the engine; ESMFold runs "
                                  "LOCAL-ONLY) → a pLDDT-coloured model overlaid on the template.")
        self._fold_btn.triggered.connect(self._on_fold_clicked)
        # DE-NOVO construct fold: fold T (the typed sequence) as a mono/di/tri/tetramer. No
        # reference (nothing loaded) → matchmaker is EXPLICITLY skipped. ESMFold is monomer-only
        # (N>1 disabled); Boltz takes N. Active only for a sequence-seeded design.
        self._construct_fold_menu = fold_menu.addMenu("Fold construct (de novo)")
        self._construct_fold_acts = []
        _NMER = [(1, "monomer"), (2, "dimer"), (3, "trimer"), (4, "tetramer")]
        for _eng in ("esmfold", "boltz", "colabfold"):
            # ColabFold is the one REMOTE engine — labelled + tool-tipped so the boundary is
            # visible AT selection (the spine's consent gate then surfaces it again before any call).
            _menu_label = ("colabfold (remote MSA — leaves local-only)"
                           if _eng == "colabfold" else _eng)
            _sub = self._construct_fold_menu.addMenu(_menu_label)
            if _eng == "colabfold":
                _sub.setToolTip("ColabFold (AF2) — uses the REMOTE MSA server, so this LEAVES "
                                "LOCAL-ONLY. A comparative accuracy reference vs the local Boltz "
                                "fold: local single-sequence Boltz vs MSA-informed ColabFold "
                                "(largely measures the MSA's value, not a fair model-vs-model test).")
            for _n, _lbl in _NMER:
                _a = _sub.addAction(f"{_lbl} (N={_n})")
                if _eng == "esmfold" and _n > 1:
                    _a.setEnabled(False)                  # ESMFold is monomer-only
                _a.triggered.connect(
                    lambda _checked=False, e=_eng, k=_n: self._on_construct_fold(e, k))
                self._construct_fold_acts.append(_a)
        # TEMPLATE-GUIDED construct fold (Boltz primary): fold the de-novo construct STEERED by a
        # chosen structural template (a homolog PDB). Sequence-first → the template biases the
        # fold; the assist readout then measures whether it helped (ΔpLDDT + Δflexibility vs the
        # unguided T-fold). Monomer first; the per-template list is multimer-ready. Reuses the
        # Align dialog's PDB-id / loaded-model picker; pre-fills a shared-fold align reference.
        self._guided_fold_btn = self._construct_fold_menu.addMenu("guided by template (Boltz)")
        self._guided_fold_btn.setToolTip("Boltz template-guided: per-chain MULTI-template "
                                         "soft-consensus (force/threshold) — LOCAL-ONLY.")
        self._guided_fold_acts = []
        for _n, _lbl in _NMER:
            _g = self._guided_fold_btn.addAction(f"{_lbl} (N={_n})")
            _g.triggered.connect(
                lambda _checked=False, k=_n: self._on_construct_fold_guided("boltz", k))
            self._guided_fold_acts.append(_g)
        # ColabFold template-guided — COARSER: a SINGLE custom PDB template (not Boltz's per-chain
        # multi-template scheme) AND remote-MSA. Both differences are surfaced in the menu label +
        # tooltip so "ColabFold template-guided" can't be mistaken for Boltz family behaviour.
        self._guided_fold_cf_btn = self._construct_fold_menu.addMenu(
            "guided by template (ColabFold — 1 template, remote MSA)")
        self._guided_fold_cf_btn.setToolTip("ColabFold template-guided: a SINGLE custom PDB "
                                            "template (coarser than Boltz's per-chain multi-template "
                                            "force/threshold) AND uses the REMOTE MSA server "
                                            "(leaves LOCAL-ONLY). Extra templates are ignored.")
        self._guided_fold_cf_acts = []
        for _n, _lbl in _NMER:
            _gc = self._guided_fold_cf_btn.addAction(f"{_lbl} (N={_n})")
            _gc.triggered.connect(
                lambda _checked=False, k=_n: self._on_construct_fold_guided("colabfold", k))
            self._guided_fold_cf_acts.append(_gc)
        # STAGE 2 — AUTO template discovery: foldseek the construct's UNGUIDED MONOMER fold against
        # the LOCAL PDB DB → ranked structural neighbours → the SAME picker as manual selection →
        # guided re-fold. Closes the orphan case (a designed sequence with no known family to type
        # in). LOCAL-ONLY (foldseek easy-search vs a pre-downloaded DB; no network at query time).
        self._find_tmpl_btn = self._construct_fold_menu.addMenu("Find templates (foldseek)")
        self._find_tmpl_acts = []
        for _n, _lbl in _NMER:
            _f = self._find_tmpl_btn.addAction(f"{_lbl} (N={_n})")
            _f.setToolTip("Search the LOCAL PDB DB for structural neighbours of the construct's "
                          "unguided fold, then guide a re-fold by the ones you pick. Needs an "
                          "unguided construct fold first.")
            _f.triggered.connect(
                lambda _checked=False, k=_n: self._on_find_templates("boltz", k))
            self._find_tmpl_acts.append(_f)
        # Template assist: did the template actually help? ΔpLDDT + per-residue Δflexibility of
        # the guided fold vs the unguided baseline (honest — surfaces both + the delta).
        self._assist_btn = self._construct_fold_menu.addAction("Template assist (guided vs unguided)…")
        self._assist_btn.setToolTip("Measure whether the template helped: ΔpLDDT + per-residue "
                                    "Δflexibility of the guided fold vs the unguided baseline. "
                                    "Needs BOTH a guided fold and an unguided construct fold.")
        self._assist_btn.triggered.connect(self._on_template_assist_clicked)
        # Validate the guided fold ADOPTED the template (US-align guided fold vs the guiding
        # template → TM; reuses Stage 3, zero new alignment code).
        self._validate_guided_btn = self._construct_fold_menu.addAction(
            "Validate guided fold (US-align vs template)…")
        self._validate_guided_btn.setToolTip("Structurally align the GUIDED fold to the guiding "
                                             "template (US-align, sequence-independent) — TM>0.5 "
                                             "means the fold adopted the template.")
        self._validate_guided_btn.triggered.connect(self._on_validate_guided_clicked)
        # COMPARE two folds (any engines) — the comparative-fold readout: US-align TM/RMSD +
        # per-residue agreement between e.g. the LOCAL Boltz fold and the REMOTE-MSA ColabFold fold.
        self._compare_folds_btn = self._construct_fold_menu.addAction(
            "Compare two folds (US-align + per-residue)…")
        self._compare_folds_btn.setToolTip("Structurally compare two open predicted models (any "
                                           "engines) — e.g. local Boltz vs MSA-informed ColabFold. "
                                           "US-align TM/RMSD + superposition-free per-residue "
                                           "deviation. The asymmetry is stated in the readout.")
        self._compare_folds_btn.triggered.connect(self._on_compare_folds_clicked)
        self._fold_vis_btn = fold_menu.addAction("Hide folds")
        self._fold_vis_btn.setToolTip("Show/hide ALL predicted fold models in 3D.")
        self._fold_vis_btn.setCheckable(True)
        self._fold_vis_btn.toggled.connect(self._on_fold_visibility_toggled)
        # GLOBAL alignment-reference visibility — the exact parallel to "Hide folds", under the SAME
        # single-source authority (fold_visibility_commands). Force-hides EVERY US-align reference
        # (the PDBs the construct fold was overlaid onto), so they don't accumulate stuck-visible.
        self._align_ref_vis_btn = fold_menu.addAction("Hide alignment references")
        self._align_ref_vis_btn.setToolTip("Show/hide ALL US-align reference structures (the PDBs "
                                           "you aligned the construct's fold onto).")
        self._align_ref_vis_btn.setCheckable(True)
        self._align_ref_vis_btn.toggled.connect(self._on_align_ref_visibility_toggled)
        # Escape hatch: lay the variant folds + the WT reference out SIDE-BY-SIDE (not
        # overlaid) in the one 3D scene via ChimeraX `tile`. Targets the specific fold models
        # (not bare `tile`, which would drag in any hidden models). Shared camera; the models
        # leave superposition (accepted — this is "lay them out", not "overlay").
        self._tile_btn = fold_menu.addAction("Tile folds")
        self._tile_btn.setToolTip("Lay the variant folds + reference out side-by-side (not "
                                  "overlaid). Select a variant afterwards to return to the "
                                  "overlay. Needs ≥2 models.")
        self._tile_btn.triggered.connect(self._on_tile_clicked)
        fold_menu.addSeparator()
        # Independent overlay toggles (distinct from the global "Hide folds"): show just the
        # active variant's FOLD, just the WT REFERENCE, or both (default).
        self._show_fold_cb = fold_menu.addAction("Variant fold")
        self._show_fold_cb.setCheckable(True)
        self._show_fold_cb.setChecked(True)
        self._show_fold_cb.setToolTip("Show the ACTIVE variant's predicted fold model in the overlay.")
        self._show_fold_cb.toggled.connect(self._on_overlay_toggle)
        self._show_ref_cb = fold_menu.addAction("Template")
        self._show_ref_cb.setCheckable(True)
        self._show_ref_cb.setChecked(True)
        self._show_ref_cb.setToolTip("Show the WT template (reference) structure in the overlay.")
        self._show_ref_cb.toggled.connect(self._on_overlay_toggle)
        # PER-CD alignment reference toggle (mirrors Template/Variant-fold): show/hide the ACTIVE
        # construct's US-align reference. State persists on cd.structural_align["hidden"] (so each
        # cd's reference is independently remembered); effective visibility = global AND per.
        self._show_align_ref_cb = fold_menu.addAction("Aligned reference")
        self._show_align_ref_cb.setCheckable(True)
        self._show_align_ref_cb.setChecked(True)
        self._show_align_ref_cb.setToolTip("Show the ACTIVE construct's US-align reference (the PDB "
                                           "its fold was aligned onto) in the overlay.")
        self._show_align_ref_cb.toggled.connect(self._on_align_ref_overlay_toggle)
        self._fold_menu_btn.setMenu(fold_menu)
        bar.addWidget(self._fold_menu_btn)
        bar.addSpacing(12)

        # Stage 4c: per-residue Cα deviation of the ACTIVE folded variant vs a seed-pinned
        # WT reference fold (same engine+target). Establishes the WT reference + noise floor
        # on first use for that combo (folds T; Boltz also folds the cross-seed floor set).
        self._dev_btn = QtWidgets.QPushButton("Deviation vs WT")
        self._dev_btn.setToolTip("Per-residue Cα deviation of the ACTIVE folded variant vs a "
                                 "seed-pinned WT reference fold (same engine). Floor-gated: "
                                 "residues within the noise floor stay neutral. First use for "
                                 "an engine folds the WT reference + its noise floor (cached).")
        self._dev_btn.clicked.connect(self._on_deviation_clicked)
        bar.addWidget(self._dev_btn)
        bar.addSpacing(12)

        # Stage 3: structurally align the DE-NOVO construct's fold onto a chosen PDB,
        # SEQUENCE-INDEPENDENTLY (US-align, LOCAL-ONLY) — the case ChimeraX matchmaker can't
        # reach. Captures TM-score/RMSD and overlays the pLDDT-coloured fold on the reference.
        self._align_btn = QtWidgets.QPushButton("Align to PDB")
        self._align_btn.setToolTip("Structurally align the construct's FOLD onto a chosen PDB "
                                   "(sequence-independent, US-align LOCAL-ONLY). Captures TM-score "
                                   "+ RMSD and overlays the fold on the reference. De-novo only; "
                                   "fold the construct first.")
        self._align_btn.clicked.connect(self._on_align_clicked)
        bar.addWidget(self._align_btn)
        bar.addSpacing(12)
        bar.addWidget(QtWidgets.QLabel("Color:"))
        self._mode_combo = QtWidgets.QComboBox()
        for m in all_modes():
            self._mode_combo.addItem(m.label, m.key)
        self._mode_combo.addItem("ddG (result)", _RESULT_DDG_MODE)   # S4a result-backed mode
        self._mode_combo.addItem("pLDDT (result)", _RESULT_PLDDT_MODE)  # S4b fold confidence
        self._mode_combo.addItem("Deviation vs WT", _RESULT_DEVIATION_MODE)  # S4c floor-gated dev
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

    def attach_session(self, session) -> None:
        """Re-point the panel at a different SessionState — used on session RESTORE, where the
        app swaps `self.session` for the loaded one. Without this the panel keeps writing to /
        reading from the ORIGINAL (empty) session, so a restored design never rehydrates and new
        edits never persist into the restored file."""
        self._session = session

    def reset(self) -> None:
        """Clear the panel to the no-design state — used on session Clear and before displaying a
        DIFFERENT session (load = replace). Does NOT persist (the caller already swapped the
        SessionState); a subsequent rehydrate/load_model repopulates it."""
        self._design = None
        self._edit_target = None
        self._scan_cols.clear()
        self._update_scan_label()
        self._render()
        self._status.setText("No structure loaded.")

    def rehydrate_denovo(self, design_dict: Dict[str, Any]) -> None:
        """DE-NOVO restore (no crystal): rehydrate the construct DIRECTLY from persisted data —
        NOT via controller.load_model (the synthetic id isn't in ChimeraX, and the crystal
        rehydrate's chain-set guard would fail against an empty fresh build). Re-displays the
        construct fold if one was persisted: members already point at the fold model, which
        survives the app restart in the still-open ChimeraX, so pLDDT recolours it with NO
        re-fold. Reuses the session-restore re-display contract."""
        self._design = DesignSession.from_dict(design_dict)
        self._edit_target = None
        self._scan_cols.clear()
        self._update_scan_label()
        self._render()
        self._persist()
        cd = next(iter(self._design.chains.values()), None)
        if cd is not None and (cd.template_fold or {}).get("model_id"):
            self._select_result_mode(_RESULT_PLDDT_MODE)
            tab = self._cur_tab()
            if tab is not None:
                self._apply_color_to(tab)
                self._push_3d_color(tab)      # recolour the still-open fold (no re-fold)

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
        # REHYDRATE a persisted design for this model (variants + indels + fold/deviation
        # results) instead of discarding it — the fresh build above is the template/validation
        # baseline. Rehydrate only when the persisted design's unique-chain set matches the live
        # model (same structure), else keep the fresh build. ChimeraX persists across an app
        # restart, so the referenced fold model ids stay valid → 3D + the cached deviation come
        # back with no re-fold. A corrupt/mismatched persisted blob falls back to fresh.
        persisted = (self._session.get_design_session(str(model_id))
                     if self._session is not None else None)
        if persisted:
            try:
                restored = DesignSession.from_dict(persisted)
                if set(restored.chains) == set(self._design.chains):
                    self._design = restored
            except Exception:
                pass
        self._edit_target = None
        self._scan_cols.clear()
        self._update_scan_label()
        self._render()
        self._persist()

    def _render(self) -> None:
        self._tabs.clear()
        if not self._design:
            return
        for _ukey, cd in self._design.chains.items():
            tab = _ChainDesignTab(cd, self._suggestions_for(cd))
            tab.badges = {v.id: self._badge_for(v) for v in cd.variants if self._badge_for(v)}
            tab.cellClicked2.connect(
                lambda rid, col, ctrl, t=tab: self._on_cell(t, rid, col, to_scan=ctrl))
            tab.cellMenuRequested.connect(
                lambda rid, col, gp, t=tab: self._show_cell_menu(t, rid, col, gp))
            tab.rowHeaderSelected.connect(lambda rid, t=tab: self._select_variant_row(t, rid))
            tab.rowMenuRequested.connect(lambda rid, gp, t=tab: self._on_row_menu(t, rid, gp))
            copies = "+".join(c for _m, c in cd.members)
            self._tabs.addTab(tab, f"{cd.rep_chain}  ({copies}, {len(cd.template_cells)} aa)")
        self._sync_align_ref_toggle()         # reflect the (rehydrated) active cd's reference state
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

    # ── DE-NOVO "Add sequence" construct ─────────────────────────────────────────────
    def _on_add_sequence(self) -> None:
        """Dialog → name + one-or-more chain sequences (each with a copy count) → seed a de-novo
        construct (no crystal). Hetero complexes (distinct sequences) become one ChainDesign each,
        folded together as one assembly."""
        dlg = _AddSequenceDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        name, chains = dlg.result_name(), dlg.result_chains()
        try:
            self._add_sequence_construct(name, chains)
        except Exception as exc:
            self._status.setText(f"Add sequence failed: {type(exc).__name__}: {exc}")

    def _add_sequence_construct(self, name: str, chains) -> None:
        """Core (testable, no dialog): build the de-novo DesignSession from a chain LIST, make it
        the active design, render the grid, persist. ChimeraX is untouched — nothing renders until
        the construct is folded. *chains* is a list of (sequence, copies) OR a bare sequence string
        (single-chain, copies=1 — back-compat)."""
        if isinstance(chains, str):
            chains = [(chains, 1)]
        design = build_design_session_from_sequence(name, list(chains))
        self._design = design
        self._edit_target = None
        self._scan_cols.clear()
        self._update_scan_label()
        self._render()
        self._persist()
        n_chains = len(design.chains)
        n_copies = sum(len(cd.members) for cd in design.chains.values())
        shape = (f"{n_chains} distinct chains, {n_copies} copies"
                 if n_chains > 1 else
                 f"{len(next(iter(design.chains.values())).template_cells)} aa")
        self._status.setText(
            f"Construct '{name}' added ({shape}, de novo — no structure). "
            f"Fold ▾ → Fold construct (Boltz for a multi-chain assembly); nothing renders until then.")

    def construct_fold_launch_spec(self, engine: str, n_copies: int = 1) -> Optional[dict]:
        """Deterministic fold spec for the DE-NOVO construct: fold the WHOLE declared assembly —
        EVERY ChainDesign contributes its own copies as a GROUPED, contiguous block of chain ids
        (cd0 → A,B,C; cd1 → D,E,F; …), one Boltz assembly. LOCAL-ONLY with the EXPLICIT no-reference
        flag (no crystal → matchmaker skipped, never falling back to a loaded primary). None unless
        the active design is sequence-seeded.

        Copy counts are the DESIGN-TIME stoichiometry baked into each cd's `members` (the dialog's
        per-chain spinner). The mono/di/tri/tetramer menu (*n_copies*) only applies to the simple
        SINGLE-chain / single-member Stage-1 case (one typed sequence → fold as an N-mer).

        Hetero (>1 fold chain) is BOLTZ-ONLY — ESMFold is monomer-only; returns None (with the
        caller reporting) for esmfold + a multi-chain assembly."""
        tab = self._cur_tab()
        if (tab is None or self._design is None or self._design.source != "sequence"):
            return None
        cds = list(self._design.chains.items())            # [(ukey, cd), …] (insertion = grouped order)
        # Single typed sequence (one cd, one member) → the menu's N-mer multiplier still drives it;
        # any declared stoichiometry (multi-copy or multi-chain) folds exactly as declared.
        single = len(cds) == 1 and len(cds[0][1].members) <= 1
        letters = (chr(c) for c in range(ord("A"), ord("Z") + 1))   # grouped, contiguous across cds
        blocks: Dict[str, List[str]] = {}                  # ukey -> [chain ids] (this cd's block)
        fold_chains: List[Dict[str, str]] = []             # ordered [{id, sequence}] for the YAML
        for ukey, cd in cds:
            seq = "".join(c.aa for c in cd.template_cells if c.aa is not None)
            n = max(1, int(n_copies)) if single else max(1, len(cd.members))
            block = [next(letters) for _ in range(n)]
            blocks[ukey] = block
            fold_chains.extend({"id": ch, "sequence": seq} for ch in block)
        n_total = len(fold_chains)
        if n_total > 1 and engine not in ("boltz", "colabfold"):
            return None                                    # ESMFold can't fold an assembly (Boltz/ColabFold can)
        ti: Dict[str, object] = {
            "model_id":    self._design.model_id,          # synthetic (stable persistence key)
            "engine":      engine,
            "open_model":  True,
            # ColabFold leaves LOCAL-ONLY (remote MSA) — flag it so the spine's consent gate
            # fires; the local engines stay local_only. allow_remote is NOT set here (the gate
            # sets it only after the user consents → no silent network call).
            "local_only":  engine in _LOCAL_FOLD_ENGINES,
            "no_reference": True,                          # EXPLICIT skip-matchmaker (no crystal)
        }
        if n_total > 1:
            ti["chains"] = fold_chains
            target = f"{n_total}-chain assembly"
        else:
            ti["sequence"] = fold_chains[0]["sequence"]
            target = "monomer"
        label = (self._design.model_id if len(cds) > 1 else cds[0][1].group_key)
        return {
            "tool":         engine,
            "tool_inputs":  ti,
            "user_input":   f"[Workbench] fold construct {label} — {engine} {target}, "
                            f"LOCAL-ONLY, no reference",
            "confidence":   "low",
            "refresh":      "construct_fold",
            "_denovo_chain_blocks": blocks,                # ukey -> its grouped fold-chain block
        }

    def _on_construct_fold(self, engine: str, n_copies: int) -> None:
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Fold construct is for de-novo sequences — use Add sequence first.")
            return
        spec = self.construct_fold_launch_spec(engine, n_copies)
        if spec is None:
            if len(self._design.chains) > 1 and engine != "boltz":
                self._status.setText("A multi-chain construct must be folded with Boltz "
                                     "(ESMFold is monomer-only).")
            return
        self.launchRequested.emit(spec)

    # ── TEMPLATE-GUIDED construct fold (Boltz primary) ───────────────────────────────
    def _build_template_entry(self, ref: Dict[str, Any], spec: dict) -> Optional[Dict[str, Any]]:
        """One Boltz `templates:` entry from a template_ref `{path|pdb_id, force, threshold,
        chain_id?, template_id?}`. path/pdb_id resolved by the router; chain_id defaults to this
        cd's fold-chain block (monomer → "A"). None if no structure given."""
        entry: Dict[str, Any] = {}
        path = ref.get("path")
        if path:
            key = "cif" if str(path).lower().endswith((".cif", ".mmcif")) else "pdb"
            entry[key] = str(path)
        elif ref.get("pdb_id"):
            entry["pdb_id"] = str(ref["pdb_id"]).upper()
        else:
            return None
        chain_id = ref.get("chain_id")
        if chain_id is None:
            blocks = spec.get("_denovo_chain_blocks") or {}
            block = blocks.get(self._cur_cd_ukey()) or []
            chain_id = (block[0] if len(block) == 1 else block) or None
        if chain_id is not None:
            entry["chain_id"] = chain_id
        if ref.get("template_id") is not None:
            entry["template_id"] = ref["template_id"]
        if ref.get("force"):                               # soft is the default; family is all-soft
            entry["force"] = True
            entry["threshold"] = float(ref.get("threshold", _GUIDED_TEMPLATE_THRESHOLD_A))
        return entry

    def construct_fold_guided_spec(self, engine: str, n_copies: int,
                                   template_refs: Any) -> Optional[dict]:
        """Spec to fold the de-novo construct STEERED by ONE OR MORE chosen structural templates
        (the multi-template/FAMILY path). Reuses `construct_fold_launch_spec` (same chains/blocks →
        apples-to-apples with the unguided T-fold) then attaches `ti["templates"]` — the full
        PER-TEMPLATE list (the bridge/`_build_yaml` + `_resolve_boltz_templates` already carry N).
        *template_refs* is a single dict OR a list of dicts, each `{path | pdb_id, label, force,
        threshold, chain_id?, template_id?}`; the router resolves each to a LOCAL file. Boltz folds
        with ALL templates at once (soft-consensus). None unless the active design is sequence-seeded.

        STAGE 1 = MANUAL selection (the caller picks the templates). SOFT-DEFAULT: a FAMILY (N>1)
        is all-soft (no hard exposure — calibration: hard is catastrophic below threshold); a single
        template may still be hard via its ref. Boltz default-searches the template chain when
        `template_id` is omitted; chain_id defaults to the construct's fold-chain block (monomer)."""
        refs = [template_refs] if isinstance(template_refs, dict) else list(template_refs or [])
        refs = [r for r in refs if r and (r.get("path") or r.get("pdb_id"))]
        if not refs:
            return None
        spec = self.construct_fold_launch_spec(engine, n_copies)
        if spec is None or engine not in ("boltz", "colabfold"):
            return None                                    # template-guided is Boltz / ColabFold only
        ti = spec["tool_inputs"]
        labels = [r.get("label") or r.get("pdb_id") or "template" for r in refs]

        if engine == "colabfold":
            # ColabFold custom-template mode takes a SINGLE PDB template (--custom-template-path)
            # — COARSER than Boltz's per-chain multi-template force/threshold scheme. If the user
            # picked several, only the FIRST is used; surface that so "ColabFold template-guided"
            # is never silently mistaken for Boltz's family / multi-template behaviour.
            first = refs[0]
            ti["template"] = first.get("path") or first.get("pdb_id")
            dropped = (f" (+{len(refs) - 1} more IGNORED — ColabFold uses ONE template)"
                       if len(refs) > 1 else "")
            spec["user_input"] = (f"[Workbench] fold construct guided by {labels[0]}{dropped} — "
                                  f"ColabFold, REMOTE MSA (leaves LOCAL-ONLY), single custom template")
            spec["refresh"] = "construct_fold_guided"
            spec["_guided_template"] = {
                "label": labels[0], "labels": labels[:1], "n_templates": 1,
                "force": False, "threshold": None, "coarse_single_template": True,
                "pdb_id": first.get("pdb_id"), "path": first.get("path"),
            }
            return spec

        # Boltz: the full per-template list (force/threshold, multi-template soft-consensus).
        entries = [e for e in (self._build_template_entry(r, spec) for r in refs) if e]
        if not entries:
            return None
        ti["templates"] = entries
        n = len(entries)
        any_hard = any(e.get("force") for e in entries)
        label = labels[0] if n == 1 else f"{n} templates ({', '.join(labels)})"
        mode = (f"hard force≤{_GUIDED_TEMPLATE_THRESHOLD_A:.0f}Å" if any_hard else "soft")
        spec["user_input"] = (f"[Workbench] fold construct guided by {label} ({mode}) — "
                              f"Boltz, LOCAL-ONLY template-guided")
        spec["refresh"] = "construct_fold_guided"
        spec["_guided_template"] = {
            "label": label, "labels": labels, "n_templates": n,
            "force": any_hard, "threshold": (_GUIDED_TEMPLATE_THRESHOLD_A if any_hard else None),
            # singular fields kept for the N==1 adoption-validation back-compat (first ref).
            "pdb_id": refs[0].get("pdb_id"), "path": refs[0].get("path"),
        }
        return spec

    @staticmethod
    def _suggested_template_ref(cd) -> str:
        """The headline structural-align→template feed: when this construct was aligned to a PDB
        that SHARES its fold (TM>0.5), return that reference as the suggested guide-fold template;
        else "". Pure (reads persisted `cd.structural_align`) — drives the dialog pre-fill."""
        sa = (cd.structural_align if cd else None) or {}
        if sa.get("shared_fold") and (sa.get("reference") or sa.get("ref_label")):
            return str(sa.get("reference") or sa.get("ref_label"))
        return ""

    def _resolve_template_token(self, tok: str) -> Optional[Dict[str, Any]]:
        """Turn one user token into a template_ref. A loaded model id (`#3`/`3`) → saved to a temp
        PDB (same as Align); a 4-char PDB id → `{pdb_id}` (router downloads the mmCIF). None +
        status on a bad token. force/threshold are added by the caller (soft-default)."""
        tok = tok.strip()
        m = re.match(r"^#?(\d+(?:\.\d+)*)$", tok)
        if m:
            mid = m.group(1)
            tmp = os.path.join(tempfile.gettempdir(), f"guided_tmpl_{mid.replace('.', '_')}.pdb")
            try:
                self._c._run(f'save "{Path(tmp).as_posix()}" models #{mid}')
            except Exception as exc:
                self._status.setText(f"Could not save model #{mid} as a template: {exc}")
                return None
            if not os.path.isfile(tmp):
                self._status.setText(f"Saving model #{mid} produced no file — is #{mid} open?")
                return None
            return {"path": tmp, "label": f"#{mid}"}
        if re.match(r"^[A-Za-z0-9]{4}$", tok):
            return {"pdb_id": tok.upper(), "label": tok.upper()}
        self._status.setText(f"'{tok}' is not a 4-char PDB id or a loaded model id (e.g. 1MBN, #3).")
        return None

    def _on_construct_fold_guided(self, engine: str, n_copies: int) -> None:
        """Pick ONE OR MORE structural templates (comma/space-separated PDB ids / loaded model ids,
        reusing the Align picker), pre-filling a shared-fold structural-align reference, then launch
        the guided fold. A FAMILY (N>1) is all-SOFT (no hard prompt — calibration: hard is
        catastrophic below threshold); a SINGLE template still offers soft (default) vs hard."""
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Guided fold is for de-novo constructs — use Add sequence first.")
            return
        tab = self._cur_tab()
        cd = tab.design if tab else None
        if cd is None:
            return
        suggested = self._suggested_template_ref(cd)        # headline structural-align→template feed
        prompt = ("Template(s) — one or more 4-char PDB ids and/or loaded model ids (e.g. "
                  "'1MBN, 4HHB, #3'), comma- or space-separated. Multiple = soft-consensus family:")
        if suggested:
            prompt = (f"You aligned to {suggested} and it shares the construct's fold — include it?\n\n"
                      + prompt)
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Fold construct guided by template(s)", prompt, text=suggested)
        if not ok or not (text or "").strip():
            return
        toks = [t for t in re.split(r"[,\s]+", text.strip()) if t]
        refs: List[Dict[str, Any]] = []
        for tok in toks:
            r = self._resolve_template_token(tok)
            if r is None:
                return                                      # bad token → status already set, abort
            refs.append(r)
        if not refs:
            return
        # SOFT-DEFAULT. A family (N>1) is all-soft. A single template may opt into hard steering.
        hard = False
        if len(refs) == 1:
            hard = (QtWidgets.QMessageBox.question(
                self, "Steering strength",
                "Hard steering (force the fold toward the template, threshold "
                f"{_GUIDED_TEMPLATE_THRESHOLD_A:.0f} Å)?\n\nNo = soft (template biases the fold).",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No) == QtWidgets.QMessageBox.Yes)
        for r in refs:
            r["force"] = hard
            if hard:
                r["threshold"] = _GUIDED_TEMPLATE_THRESHOLD_A
        spec = self.construct_fold_guided_spec(engine, n_copies, refs)
        if spec is None:
            self._status.setText("Could not build the guided-fold request (de-novo Boltz only).")
            return
        fam = f"{len(refs)} templates (soft-consensus)" if len(refs) > 1 else \
              f"{refs[0]['label']} ({'hard' if hard else 'soft'})"
        _where = ("ColabFold (REMOTE MSA — leaves local-only; single template)"
                  if engine == "colabfold" else "Boltz (LOCAL-ONLY)")
        self._status.setText(f"Folding the construct guided by {fam} via {_where}…")
        self.launchRequested.emit(spec)

    # ── STAGE 2: foldseek auto template discovery → the same picker → guided re-fold ──
    @staticmethod
    def _foldseek_refs(picked: List[str]) -> List[Dict[str, Any]]:
        """Selected PDB ids → guided-fold template refs (all-SOFT consensus). The SAME `{pdb_id,
        label, force}` shape manual selection produces, so auto- and manual-discovery converge on
        `construct_fold_guided_spec` (one path). Pure / testable."""
        return [{"pdb_id": str(p).upper(), "label": str(p).upper(), "force": False}
                for p in picked if p]

    def _foldseek_query_path(self, src: str) -> str:
        """Monomer query for foldseek: the FIRST chain of *src* (the construct's unguided fold).
        A monomer fold passes through unchanged; an N-mer fold is reduced to one chain so the search
        finds FOLD homologs (quaternary geometry comes from each hit PDB's OWN assembly at re-fold).
        Falls back to *src* unchanged if it is not a parseable mmCIF."""
        if not src.lower().endswith((".cif", ".mmcif")):
            return src                                  # PDB etc. — foldseek reads it directly
        try:
            lines = open(src, encoding="utf-8", errors="replace").read().splitlines()
        except OSError:
            return src
        hdr: List[str] = []; kept: List[str] = []; first = None; i, n = 0, len(lines)
        while i < n:
            if lines[i].startswith("_atom_site."):
                while i < n and lines[i].startswith("_atom_site."):
                    hdr.append(lines[i].strip()); i += 1
                ci = next((k for k, c in enumerate(hdr) if c.endswith("auth_asym_id")), None)
                if ci is None:
                    ci = next((k for k, c in enumerate(hdr) if c.endswith("label_asym_id")), None)
                while i < n and lines[i].startswith(("ATOM", "HETATM")):
                    p = lines[i].split()
                    if ci is not None and ci < len(p):
                        if first is None:
                            first = p[ci]
                        if p[ci] == first:
                            kept.append(lines[i])
                    i += 1
                break
            i += 1
        if not kept:
            return src
        dst = os.path.join(tempfile.gettempdir(),
                           f"fs_query_{os.getpid()}_{abs(hash(src)) % 100000}.cif")
        open(dst, "w", encoding="utf-8").write(
            "data_query\nloop_\n" + "\n".join(hdr) + "\n" + "\n".join(kept) + "\n")
        return dst

    def _on_find_templates(self, engine: str, n_copies: int) -> None:
        """Run the LOCAL-ONLY foldseek search on the construct's unguided fold off-thread; on
        results, open the picker. Fail-loud: de-novo only; needs an unguided fold; feature disabled
        (with reason) if foldseek/DB absent — never silently empty."""
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Find templates is for de-novo constructs — use Add sequence first.")
            return
        tab = self._cur_tab()
        cd = tab.design if tab else None
        if cd is None:
            return
        tf = cd.template_fold or {}
        query_src = tf.get("cif_path") or tf.get("pdb_path")
        if not (query_src and os.path.isfile(query_src)):
            self._status.setText("Find templates needs an unguided construct fold first "
                                 "(Fold ▾ → Fold construct (de novo) → boltz → monomer).")
            return
        try:
            from foldseek_bridge import FoldseekBridge
            fb = FoldseekBridge()
        except Exception as exc:
            self._status.setText(f"Template discovery unavailable: {exc}")
            return
        if not fb.is_available():
            self._status.setText("Template discovery unavailable — foldseek binary or local PDB DB "
                                 "not found in WSL (set FOLDSEEK_EXE / FOLDSEEK_DB).")
            return
        qpath = self._foldseek_query_path(query_src)
        self._status.setText("Searching the local PDB DB for structural neighbours…")
        w = _FoldseekWorker(fb, qpath)
        qplddt = tf.get("mean_plddt")
        dbl = fb.db_label()
        w.signals.done.connect(
            lambda hits, low, c=cd, e=engine, k=n_copies, qp=qplddt, dl=dbl:
                self._on_foldseek_hits(hits, low, c, e, k, qp, dl))
        w.signals.failed.connect(lambda msg: self._status.setText(f"Template search failed: {msg}"))
        self._pool.start(w)

    def _on_foldseek_hits(self, hits, low_hits, cd, engine: str, n_copies: int,
                          query_plddt, db_label: str) -> None:
        """Pick from ranked neighbours → guided re-fold via the SAME `construct_fold_guided_spec`
        path. 0 hits = a real answer (searched, nothing ≥ TM 0.3), stated as such — never silent.
        `low_hits` ([0.20, 0.30)) populate the collapsed 'lower-confidence' expander (often empty)."""
        if not hits and not low_hits:
            self._status.setText(f"No structural neighbours found ≥ TM 0.3 in the {db_label}. "
                                 "(A miss is a false negative — a good template may not be in this "
                                 "snapshot.)")
            return
        picked = self._foldseek_pick_dialog(hits, low_hits, query_plddt, db_label)
        if not picked:
            self._status.setText("Template discovery cancelled — no templates selected.")
            return
        refs = self._foldseek_refs(picked)
        spec = self.construct_fold_guided_spec(engine, n_copies, refs)
        if spec is None:
            self._status.setText("Could not build the guided-fold request (de-novo Boltz only).")
            return
        self._status.setText(f"Folding the construct guided by {len(refs)} foldseek template(s) "
                             f"(soft-consensus) via Boltz (LOCAL-ONLY)…")
        self.launchRequested.emit(spec)

    @staticmethod
    def _fill_hit_list(lst, hits) -> None:
        """Populate a checkable list widget with foldseek hits (PDB id carried on UserRole)."""
        for pid, ch, tm in hits:
            it = QtWidgets.QListWidgetItem(f"{pid}_{ch}    TM={tm:.3f}")
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            it.setCheckState(QtCore.Qt.Unchecked)
            it.setData(QtCore.Qt.UserRole, pid)
            lst.addItem(it)

    @staticmethod
    def _checked_ids(lst) -> List[str]:
        return [lst.item(i).data(QtCore.Qt.UserRole) for i in range(lst.count())
                if lst.item(i).checkState() == QtCore.Qt.Checked]

    def _foldseek_pick_dialog(self, hits, low_hits, query_plddt, db_label: str) -> Optional[List[str]]:
        """Rich ranked selection: a checkable list of (PDB id, foldseek-TM), a query-pLDDT caution
        when the unguided fold was low-confidence (already-computed, non-circular use-time signal),
        the weak-prior framing, an ALWAYS-shown assembly-variant caveat (foldseek ranks monomer fold
        only — quaternary assembly comes from the picked hit), a collapsed 'lower-confidence' expander
        (TM [0.20, 0.30); shown only when that bucket is non-empty), and the DB-scope label. Returns
        the checked PDB ids from BOTH lists (or None)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Find templates — structural neighbours (foldseek, LOCAL-ONLY)")
        lay = QtWidgets.QVBoxLayout(dlg)
        if isinstance(query_plddt, (int, float)) and query_plddt < 70:
            warn = QtWidgets.QLabel(
                f"⚠ The unguided query fold is low-confidence (mean pLDDT {query_plddt:.0f}). "
                "The query structure — and therefore these neighbours — may be unreliable; treat "
                "with extra caution.")
            warn.setWordWrap(True); warn.setStyleSheet("color: #b36b00;")
            lay.addWidget(warn)
        lay.addWidget(QtWidgets.QLabel(
            "Structural neighbours ranked by foldseek TM (structTM to the query fold — a WEAK "
            "pre-hoc prior, NOT proof the template is correct). Check the ones to guide a re-fold:"))
        # Assembly-variant caveat — ALWAYS shown (foldseek ranks monomer fold, not quaternary state).
        asm = QtWidgets.QLabel(
            "These are single-chain fold homologs: foldseek ranks by monomer structural similarity "
            "and does NOT distinguish quaternary-assembly variants (domain-swap direction, oligomer "
            "size, ring vs sheet). You are choosing a fold FAMILY — the quaternary geometry that gets "
            "imposed comes from the hit you pick (its own biological assembly), so choose the assembly "
            "variant deliberately.")
        asm.setWordWrap(True); asm.setStyleSheet("color: #666;")
        lay.addWidget(asm)
        lst = QtWidgets.QListWidget()
        self._fill_hit_list(lst, hits)
        lay.addWidget(lst)
        # "Show lower-confidence hits" expander — collapsed by default; ONLY when the bucket exists.
        low_lst = None
        if low_hits:
            toggle = QtWidgets.QToolButton()
            toggle.setStyleSheet("QToolButton { border: none; }")
            toggle.setCheckable(True); toggle.setChecked(False)
            toggle.setText("▸ Show lower-confidence hits (TM 0.20–0.30)")
            box = QtWidgets.QWidget()
            box_lay = QtWidgets.QVBoxLayout(box); box_lay.setContentsMargins(12, 0, 0, 0)
            note = QtWidgets.QLabel(
                "Lower-similarity neighbours (TM 0.20–0.30) — usually NOT useful (low structural "
                "similarity). Rarely, a sequence-unrelated low-similarity neighbour has unlocked a "
                "de-novo fold; WHICH ones will is uncharacterised. Look here only if you're hunting "
                "an orphan template for a de-novo design — this is not a recommendation.")
            note.setWordWrap(True); note.setStyleSheet("color: #b36b00;")
            box_lay.addWidget(note)
            low_lst = QtWidgets.QListWidget()
            self._fill_hit_list(low_lst, low_hits)
            box_lay.addWidget(low_lst)
            box.setVisible(False)
            def _toggle(checked, b=box, t=toggle):
                b.setVisible(checked)
                t.setText(("▾ Hide" if checked else "▸ Show") + " lower-confidence hits (TM 0.20–0.30)")
            toggle.toggled.connect(_toggle)
            lay.addWidget(toggle); lay.addWidget(box)
        scope = QtWidgets.QLabel(
            f"Searched: {db_label}. A miss is a false negative — not an exhaustive search; a good "
            "template may simply not be in this snapshot.")
        scope.setWordWrap(True); scope.setStyleSheet("color: #666;")
        lay.addWidget(scope)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return None
        picked = self._checked_ids(lst) + (self._checked_ids(low_lst) if low_lst is not None else [])
        return picked or None

    def apply_construct_fold_guided_result(self, spec: dict, result: dict) -> None:
        """Consume the GUIDED construct fold: store it on `cd.guided_fold` (SEPARATE from the
        unguided `cd.template_fold` baseline, so the assist can compare the two) without
        re-pointing members. The guided model is already opened + pLDDT-coloured live by the
        router; here we persist its summary + the steering provenance and auto-run the assist
        when an unguided baseline exists."""
        data = self._fold_from_result(result)
        if not data or not (data.get("new_model_id") or data.get("model_id")):
            self._status.setText("Guided construct fold produced no model.")
            return
        if self._design is None:
            return
        blocks: Dict[str, List[str]] = dict(spec.get("_denovo_chain_blocks") or {})
        gtmpl = dict(spec.get("_guided_template") or {})
        fold_mid = str(data.get("new_model_id", data.get("model_id")))
        plddt_by_chain = data.get("plddt_by_chain") or {}
        # REPLACE-ON-REFOLD: a de-novo construct holds ONE guided fold; capture the PRIOR guided
        # model id(s) so re-folding (e.g. soft → hard) doesn't accumulate overlaid, untoggleable
        # models in ChimeraX (mirrors the variant refold-replace). Closed after the slot is set.
        prior_guided = {str((self._design.chains[u].guided_fold or {}).get("model_id"))
                        for u in blocks if (self._design.chains.get(u)
                        and (self._design.chains[u].guided_fold or {}).get("model_id"))}
        for ukey, block in (blocks or {}).items():
            cd = self._design.chains.get(ukey)
            if cd is None:
                continue
            rep = block[0]
            author_resnums = [c.resnum for c in cd.template_cells
                              if not c.is_gap and c.resnum is not None]
            chain_plddt = plddt_by_chain.get(rep) or (data.get("plddt") if not plddt_by_chain else {})
            vals = list((chain_plddt or {}).values())
            per_chain = {**data,
                         "plddt":      chain_plddt or {},
                         "mean_plddt": round(sum(vals) / len(vals), 2) if vals else data.get("mean_plddt"),
                         "chain":      rep}
            gf = fold_summary(per_chain, author_resnums, reference_model_id=None)
            gf.update(templated=True, template_label=gtmpl.get("label"),
                      force=bool(gtmpl.get("force")), threshold=gtmpl.get("threshold"),
                      template_pdb_id=gtmpl.get("pdb_id"), template_path=gtmpl.get("path"),
                      adoption=data.get("adoption"),                 # immediate use-time readout
                      per_template_adoption=data.get("per_template_adoption"),
                      # the EXACT per-template list used → the assist re-folds the guided floor
                      # seeds with the same steering (the router re-resolves pdb_id/path).
                      templates=list((spec.get("tool_inputs") or {}).get("templates") or []))
            cd.guided_fold = gf
        self._persist()
        # close the prior guided model(s) now that the new one is stored (replace-on-refold).
        stale = [m for m in prior_guided if m and m not in ("None", fold_mid)]
        if stale:
            self._run_commands_bg([f"close #{m}" for m in stale])
        cur = self._cur_tab()
        cd = cur.design if cur is not None else next(iter(self._design.chains.values()), None)
        mp = (cd.guided_fold.get("mean_plddt") if cd is not None else None)
        mp_txt = f", chain pLDDT {mp:.1f}" if isinstance(mp, (int, float)) else ""
        adopt = (cd.guided_fold.get("adoption") if cd is not None else None)
        # IMMEDIATE "did it reflect the template" readout — structTM(guided fold, template). High
        # adoption = the fold FOLLOWS the template (not proof of correctness — use-time signal).
        adopt_txt = (f" Adopted the template at {adopt:.0%} (structTM; use 'Align to PDB' to overlay "
                     f"the fold on the template)." if isinstance(adopt, (int, float))
                     else " (adoption n/a — 'Align to PDB' to check the fold vs the template).")
        base = bool(cd and cd.template_fold.get("model_id"))
        self._status.setText(
            f"Guided fold ({gtmpl.get('label')}, {'hard' if gtmpl.get('force') else 'soft'}): "
            f"model #{fold_mid}{mp_txt}.{adopt_txt} "
            + ("Run 'Template assist' for ΔpLDDT + Δflexibility vs the unguided fold."
               if base else
               "No unguided baseline yet — fold the construct unguided (Fold construct) to "
               "enable the assist comparison."))

    # ── TEMPLATE ASSIST readout (guided vs unguided) ─────────────────────────────────
    def template_assist_launch_spec(self) -> Optional[dict]:
        """Spec to measure whether the template helped: compares the construct's GUIDED fold
        against its UNGUIDED baseline T-fold (both on disk — neither re-folded). The router reuses
        each as the seed-0 of its own cross-seed flexibility floor → ΔpLDDT + per-residue
        Δflexibility. None unless BOTH folds exist. wt_chains = the WHOLE construct (every cd's
        sequence × its members), mirroring the de-novo deviation reference."""
        tab = self._cur_tab()
        if tab is None or self._design is None or self._design.source != "sequence":
            return None
        cd = tab.design
        gf, uf = cd.guided_fold or {}, cd.template_fold or {}
        if not gf.get("model_id") or not uf.get("model_id"):
            return None
        target = gf.get("target", uf.get("target", "monomer"))
        multichain = (target == "assembly")
        # Full-construct chain set (same as the de-novo deviation reference) → apples-to-apples.
        wt_chains = []
        for _uk, c in self._design.chains.items():
            cseq = "".join(cc.aa for cc in c.template_cells if cc.aa is not None)
            wt_chains.extend({"id": ch, "sequence": cseq} for (_m, ch) in c.members)
        variant_chain = cd.rep_chain
        def _ref(d):
            return {"engine": d.get("engine"), "target": d.get("target"), "seed": d.get("seed"),
                    "model_id": d.get("model_id"), "path": d.get("cif_path") or d.get("pdb_path")}
        ti: Dict[str, object] = {
            "engine":              "boltz",
            "target":              target,
            "multichain":          multichain,
            "variant_chain":       variant_chain,
            "wt_chains":           wt_chains,
            "model_id":            self._design.model_id,
            "unguided_ref":        _ref(uf),
            "guided_ref":          _ref(gf),
            "templates":           list(gf.get("templates") or []),
            "guided_mean_plddt":   gf.get("mean_plddt"),
            "unguided_mean_plddt": uf.get("mean_plddt"),
            "guided_plddt":        gf.get("plddt") or {},
            "unguided_plddt":      uf.get("plddt") or {},
            "template_label":      gf.get("template_label"),
            "force":               bool(gf.get("force")),
            "threshold":           gf.get("threshold"),
        }
        return {
            "tool":        "template_assist",
            "tool_inputs": ti,
            "user_input":  f"[Workbench] template assist — guided ({gf.get('template_label')}) vs "
                           f"unguided, ΔpLDDT + Δflexibility, LOCAL-ONLY",
            "confidence":  "low",                        # folds ~2×(N−1) floor seeds → the gate
            "refresh":     "template_assist",
            "_assist_ukey": self._cur_cd_ukey(),
        }

    def _on_template_assist_clicked(self) -> None:
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Template assist is for de-novo constructs.")
            return
        tab = self._cur_tab()
        cd = tab.design if tab else None
        if cd is None:
            return
        if not (cd.guided_fold or {}).get("model_id"):
            self._status.setText("Fold the construct guided by a template first "
                                 "(Fold ▾ → Fold construct → guided by template).")
            return
        if not (cd.template_fold or {}).get("model_id"):
            self._status.setText("Fold the construct UNGUIDED first (Fold ▾ → Fold construct) — "
                                 "the assist compares guided vs the unguided baseline.")
            return
        spec = self.template_assist_launch_spec()
        if spec is None:
            self._status.setText("Could not build the template-assist request.")
            return
        self._status.setText("Measuring template assist (folds cross-seed floors for both — this "
                             "takes several Boltz folds)…")
        self.launchRequested.emit(spec)

    def _on_validate_guided_clicked(self) -> None:
        """US-align the GUIDED fold against the guiding template → TM (did it adopt the template
        fold?). Reuses the structural-align spec with use_guided=True and the stored template as
        the reference. Zero new alignment code."""
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Validation is for de-novo constructs.")
            return
        tab = self._cur_tab()
        cd = tab.design if tab else None
        gf = (cd.guided_fold if cd else None) or {}
        if not gf.get("model_id"):
            self._status.setText("Fold the construct guided by a template first.")
            return
        pdb_id, path = gf.get("template_pdb_id"), gf.get("template_path")
        label = gf.get("template_label") or pdb_id or "template"
        if pdb_id:
            spec = self.structural_align_launch_spec(reference_pdb_id=str(pdb_id),
                                                     ref_label=str(label), use_guided=True)
        elif path and os.path.isfile(str(path)):
            spec = self.structural_align_launch_spec(reference_path=str(path),
                                                     ref_label=str(label), use_guided=True)
        else:
            self._status.setText("The guiding template is no longer on disk — re-fold guided.")
            return
        if spec is None:
            self._status.setText("Could not build the validation request.")
            return
        self._status.setText(f"Validating the guided fold adopted {label} (US-align, "
                             f"sequence-independent)…")
        self.launchRequested.emit(spec)

    @staticmethod
    def _template_assist_from_result(result: dict) -> dict:
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") == "template_assist":
                return step.get("data") or {}
        return {}

    def apply_template_assist_result(self, spec: dict, result: dict) -> None:
        """Consume the assist: store it on `cd.template_assist`, persist, and report only the
        USE-TIME-knowable effects (ΔpLDDT, cross-seed Δ, adoption) — NEVER "rescue confirmed"
        (correctness is truth-dependent and there is no experimental structure for a de-novo
        construct). High adoption is flagged as a possible-copying caveat."""
        data = self._template_assist_from_result(result)
        if not data:
            self._status.setText("Template assist produced no result.")
            return
        ukey = (spec or {}).get("_assist_ukey")
        cd = self._design.chains.get(ukey) if (self._design and ukey) else None
        if cd is None:
            tab = self._cur_tab()
            cd = tab.design if tab else None
        if cd is None:
            return
        cd.template_assist = data
        self._persist()
        dp = data.get("d_plddt")
        up, gp = data.get("unguided_mean_plddt"), data.get("guided_mean_plddt")
        nstab, ntot = data.get("n_stabilized"), data.get("n_residues")
        mdf = data.get("mean_d_flex")
        madopt = data.get("max_adoption")
        plddt_txt = (f"confidence {up:.1f}→{gp:.1f} (Δ{dp:+.1f})" if dp is not None else "pLDDT n/a")
        adopt_txt = (f"adopted at {madopt:.0%}" if isinstance(madopt, (int, float)) else "adoption n/a")
        # Wording mirrors the router's three-state routing: the REFINED "did not already resemble"
        # claim only when the pre-hoc proxy MEASURED the template as distant ("distant"); the GENERIC
        # wording when it fired conservatively on a missing proxy ("unmeasured"); else no caveat.
        # Fall back to "distant" if the reason field is absent but the bool is set (old result dicts).
        reason = data.get("high_adoption_caveat_reason")
        if reason is None and data.get("high_adoption_caveat"):
            reason = "distant"
        if reason == "distant":
            caveat = ("  ⚠ HIGH adoption of a template the unguided fold did NOT already resemble — "
                      "guidance may be IMPOSING the template fold, not independently converging; "
                      "can't be ruled out without an experimental structure.")
        elif reason == "unmeasured":
            caveat = ("  ⚠ HIGH adoption — the fold may be FOLLOWING the template, not independently "
                      "converging; can't be ruled out without an experimental structure.")
        else:
            caveat = ""
        self._status.setText(
            f"Template assist ({data.get('template_label')}): {plddt_txt}; cross-seed variation "
            f"{nstab}/{ntot} residues tightened (mean Δ {mdf:+.2f} Å); {adopt_txt}. "
            f"USE-TIME-knowable effects — NOT a correctness claim (no experimental structure). "
            f"Guided confidence is template-biased; shown vs the unguided baseline.{caveat}")

    def apply_construct_fold_result(self, spec: dict, result: dict) -> None:
        """Consume the construct fold: for EACH ChainDesign, store its OWN chain's pLDDT on
        `cd.template_fold` and RE-POINT its members/model ids from the synthetic id to that cd's
        block of FOLD chains (so column-click selection + sequence-property colour come alive on
        the right chains — a PCNA click selects the three PCNA chains, a p21 click the three p21).
        The synthetic `model_id` (persistence key) is unchanged.

        READ-BACK GUARD: the bridge returns the OBSERVED chain ids (CIF order); we fail loud if
        sent != observed (a relabel/missing chain → no silent mis-point). The index-keyed
        `chains_ptm` is paired to chain ids THROUGH the observed CIF order (reorder-robust), then
        mapped to each cd by id."""
        data = self._fold_from_result(result)
        if not data or not (data.get("new_model_id") or data.get("model_id")):
            self._status.setText("Construct fold produced no model.")
            return
        if self._design is None:
            return
        # Per-cd chain blocks (new shape); fall back to the flat single-cd shape for old specs.
        blocks: Dict[str, List[str]] = dict(spec.get("_denovo_chain_blocks") or {})
        if not blocks:
            ukey = spec.get("_denovo_chain_key") or next(iter(self._design.chains), None)
            if ukey is not None:
                blocks = {ukey: list(spec.get("_denovo_fold_chains") or ["A"])}
        if not blocks:
            return
        fold_mid = str(data.get("new_model_id", data.get("model_id")))
        sent_ids = [ch for blk in blocks.values() for ch in blk]
        observed = data.get("chain_ids")
        # READ-BACK GUARD — sent must equal observed (as a set), else refuse the re-point.
        if observed is not None and set(observed) != set(sent_ids):
            self._status.setText(
                f"Construct fold chain mismatch — sent {sorted(set(sent_ids))}, got "
                f"{sorted(set(map(str, observed)))}. Refusing to re-point (fold not trusted).")
            return
        plddt_by_chain = data.get("plddt_by_chain") or {}
        ptm_by_chain = self._chains_ptm_by_id(data.get("chains_ptm"), observed or sent_ids)
        for ukey, block in blocks.items():
            cd = self._design.chains.get(ukey)
            if cd is None:
                continue
            rep = block[0]
            author_resnums = [c.resnum for c in cd.template_cells
                              if not c.is_gap and c.resnum is not None]
            chain_plddt = plddt_by_chain.get(rep) or (data.get("plddt") if not plddt_by_chain else {})
            vals = list((chain_plddt or {}).values())
            per_chain = {**data,
                         "plddt":      chain_plddt or {},
                         "mean_plddt": round(sum(vals) / len(vals), 2) if vals else data.get("mean_plddt"),
                         "chain":      rep}
            if rep in ptm_by_chain:
                per_chain["chains_ptm"] = {rep: ptm_by_chain[rep]}   # THIS cd's own chain pTM
            cd.template_fold = fold_summary(per_chain, author_resnums, reference_model_id=None)
            # THE re-point: the construct's "structure" is now its fold — this cd targets its OWN
            # block of fold chains (N copies for an N-mer; one block per distinct chain for hetero).
            cd.members   = [(fold_mid, ch) for ch in block]
            cd.rep_model = fold_mid
            cd.rep_chain = rep
        self._persist()
        tab = self._cur_tab()
        if tab is not None:
            self._select_result_mode(_RESULT_PLDDT_MODE)   # auto-surface pLDDT on the fold
            tab.rebuild()
            self._apply_color_to(tab)
            self._push_3d_color(tab)
        cur = (tab.design if tab is not None else next(iter(self._design.chains.values()), None))
        mp = (cur.template_fold.get("mean_plddt") if cur is not None else None)
        mp_txt = f", chain pLDDT {mp:.1f}" if isinstance(mp, (int, float)) else ""
        self._status.setText(
            f"Construct folded ({data.get('engine')}, {len(sent_ids)}-chain): model "
            f"#{fold_mid}{mp_txt}, pLDDT-coloured. Selection + colour now act on the fold; "
            f"no matchmaker (de novo, no reference).")

    # ── Compare two folds (US-align + per-residue) — comparative-fold readout ─────────
    def _predicted_fold_descriptors(self) -> List[Dict[str, Any]]:
        """Open PREDICTED models as align-folds descriptors {label, engine, model_id, path,
        remote_msa}, newest first — gathered from the live session's structures (the bridges
        register each fold with predicted/engine/remote_msa metadata + its on-disk path)."""
        sess = getattr(self, "_session", None)
        structs = getattr(sess, "structures", {}) if sess is not None else {}
        out: List[Dict[str, Any]] = []
        for mid, info in (structs or {}).items():
            meta = (info or {}).get("metadata") or {}
            if not meta.get("predicted") or not (info or {}).get("path"):
                continue
            eng = meta.get("engine") or "fold"
            out.append({
                "model_id":   str(mid),
                "engine":     eng,
                "remote_msa": bool(meta.get("remote_msa")),
                "path":       info.get("path"),
                "label":      f"#{mid} {eng}" + (" (remote MSA)" if meta.get("remote_msa") else ""),
            })
        out.sort(key=lambda d: int(d["model_id"]) if d["model_id"].isdigit() else 0, reverse=True)
        return out

    @staticmethod
    def build_align_folds_spec(fold_a: Dict[str, Any], fold_b: Dict[str, Any],
                               multichain: bool = False, chain: str = "A") -> Optional[dict]:
        """Pure: align_folds launch spec from two fold descriptors (each {label,engine,model_id,
        path,remote_msa}). None unless BOTH carry an on-disk path (US-align reads files)."""
        if not (fold_a.get("path") and fold_b.get("path")):
            return None
        return {
            "tool": "align_folds",
            "tool_inputs": {"fold_a": fold_a, "fold_b": fold_b,
                            "multichain": multichain, "chain": chain},
            "user_input": (f"[Workbench] compare folds {fold_a.get('label')} vs "
                           f"{fold_b.get('label')} — US-align + per-residue deviation"),
            "confidence": "high",
            "refresh": "align_folds",
        }

    def _on_compare_folds_clicked(self) -> None:
        """Pick two open predicted models → compare via US-align + per-residue deviation."""
        cands = self._predicted_fold_descriptors()
        if len(cands) < 2:
            self._status.setText("Compare folds needs TWO open predicted models — fold the "
                                 "construct with two engines (e.g. Boltz, then ColabFold) first.")
            return
        labels = [c["label"] for c in cands]
        a_lbl, ok = QtWidgets.QInputDialog.getItem(
            self, "Compare folds — first model", "Fold A:", labels, 0, False)
        if not ok:
            return
        rest = [l for l in labels if l != a_lbl] or labels
        b_lbl, ok = QtWidgets.QInputDialog.getItem(
            self, "Compare folds — second model", "Fold B:", rest, 0, False)
        if not ok:
            return
        fold_a = next(c for c in cands if c["label"] == a_lbl)
        fold_b = next(c for c in cands if c["label"] == b_lbl)
        spec = self.build_align_folds_spec(fold_a, fold_b)
        if spec is None:
            self._status.setText("Could not build the compare request (a fold file is missing).")
            return
        self._status.setText(f"Comparing {a_lbl} vs {b_lbl} (US-align + per-residue)…")
        self.launchRequested.emit(spec)

    def apply_align_folds_result(self, spec: dict, result: dict) -> None:
        """Surface the comparative-fold readout (US-align TM/RMSD + per-residue agreement) — the
        honest 'local single-sequence Boltz vs MSA-informed ColabFold' framing is in the summary."""
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") == "align_folds":
                if step.get("success"):
                    self._status.setText(step.get("summary") or "Folds compared.")
                else:
                    self._status.setText(f"Compare folds failed: {step.get('error')}")
                return
        self._status.setText("Compare folds produced no result.")

    @staticmethod
    def _chains_ptm_by_id(chains_ptm, observed) -> Dict[str, Any]:
        """Map Boltz's per-chain pTM to chain ids THROUGH the observed CIF order (reorder-robust).
        `chains_ptm` is index-keyed ({"0": .., "1": ..} or a list); `observed[i]` is the chain id
        at fold position i, so ptm index i → that id. An already-id-keyed dict passes through."""
        if not chains_ptm or not observed:
            return {}
        if isinstance(chains_ptm, dict):
            # id-keyed already? (keys match observed) → pass through.
            if set(map(str, chains_ptm)) & set(map(str, observed)):
                return {str(k): v for k, v in chains_ptm.items()}
            out: Dict[str, Any] = {}
            for k, v in chains_ptm.items():
                try:
                    i = int(k)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(observed):
                    out[str(observed[i])] = v
            return out
        if isinstance(chains_ptm, (list, tuple)):
            return {str(observed[i]): v for i, v in enumerate(chains_ptm) if i < len(observed)}
        return {}

    def _apply_substitution(self) -> None:
        tab = self._cur_tab()
        if tab is None:
            return
        if self._edit_target is None:
            self._status.setText("Select a residue in a VARIANT row first (T is the immutable template).")
            return
        vid, col = self._edit_target
        self._do_substitute(tab, vid, col, self._aa_combo.currentText())

    def _after_variant_edit(self, tab: _ChainDesignTab, vid: str, msg: str) -> None:
        """Shared post-edit refresh for substitute/delete/restore: re-lay the grid, keep the
        edited variant active, re-apply panel + 3D colouring, persist, and report. ONE path
        so the toolbar Apply, the cell context menu, and accept-suggestion stay in sync."""
        tab.rebuild()
        tab.set_active_row(vid)
        self._apply_color_to(tab)
        if tab.active_row_id == vid:
            self._push_3d_color(tab)           # active variant edited → recolor 3D (all copies)
        self._persist()
        self._status.setText(msg)

    def _do_substitute(self, tab: _ChainDesignTab, vid: str, col: int, aa: str) -> None:
        try:
            tab.design.edit_variant(vid, col, aa)
        except Exception as exc:
            self._status.setText(f"Substitution failed: {type(exc).__name__}: {exc}")
            return
        resnum = tab.design.resnum_for_col(col)
        self._after_variant_edit(
            tab, vid, f"{vid}: residue {resnum} → {aa}. "
                      f"3D recolored by {vid} (preview on the shared backbone).")

    def _show_cell_menu(self, tab: _ChainDesignTab, vid: str, col: int, global_pos) -> None:
        """Right-click cell menu for a VARIANT residue: substitute (any of the 20 aa), revert
        to the WT residue, delete the residue, or insert residues after it. The substitution is
        the SAME `edit_variant` as the toolbar Apply, just reachable on the residue. An INSERTED
        column (template gap + this-variant residue) offers only Remove insertion; a DELETED cell
        (template residue + this-variant gap) offers only Restore."""
        v = tab.design.get_variant(vid)
        if v is None or not (0 <= col < len(tab.design.template_cells)):
            return
        tab.set_active_row(vid)
        self._edit_target = (vid, col)
        resnum = tab.design.resnum_for_col(col)
        tmpl_gap = tab.design.template_cells[col].is_gap
        wt = tab.design.template_cells[col].aa
        cell = v.cells[col] if col < len(v.cells) else None
        cur = cell.aa if cell is not None else None
        is_gap = cell is not None and cell.is_gap

        menu = QtWidgets.QMenu(self)
        if tmpl_gap:                                     # an INSERTED column (no WT counterpart)
            if is_gap or cell is None:                   # this variant doesn't share the insertion
                return
            header = menu.addAction(f"{vid} · inserted {cur} (no WT — column added by this variant)")
            header.setEnabled(False)
            menu.addSeparator()
            menu.addAction("Remove insertion",
                           lambda: self._do_remove_insertion(tab, vid, col))
            menu.exec(global_pos)
            return
        header = menu.addAction(f"{vid} · residue {resnum} "
                                + ("(DELETED)" if is_gap else f"(WT {wt}, now {cur})"))
        header.setEnabled(False)
        menu.addSeparator()
        if is_gap:                                       # a deleted cell → offer Restore
            menu.addAction(f"Restore residue (WT {wt})",
                           lambda: self._do_restore_residue(tab, vid, col))
            menu.exec(global_pos)
            return
        sub = menu.addMenu("Substitute →")
        for aa in _AA_ORDER:
            act = sub.addAction(f"{aa}  (WT)" if aa == wt else aa)
            if aa == cur:
                act.setCheckable(True); act.setChecked(True)
            act.triggered.connect(lambda _checked=False, a=aa: self._do_substitute(tab, vid, col, a))
        if wt is not None and cur != wt:
            menu.addAction(f"Revert to WT ({wt})",
                           lambda: self._do_substitute(tab, vid, col, wt))
        menu.addSeparator()
        menu.addAction("Delete residue",
                       lambda: self._do_delete_residue(tab, vid, col))
        menu.addAction("Insert residues after…",
                       lambda: self._do_insert_residues(tab, vid, col))
        menu.exec(global_pos)

    def _do_delete_residue(self, tab: _ChainDesignTab, vid: str, col: int) -> None:
        try:
            tab.design.delete_variant_residue(vid, col)
        except Exception as exc:
            self._status.setText(f"Delete failed: {type(exc).__name__}: {exc}")
            return
        resnum = tab.design.resnum_for_col(col)
        self._after_variant_edit(
            tab, vid, f"{vid}: residue {resnum} DELETED (cell → gap; a fold/deviation now "
                      f"treats it as removed). Right-click the gap → Restore to undo.")

    def _do_restore_residue(self, tab: _ChainDesignTab, vid: str, col: int) -> None:
        try:
            tab.design.restore_variant_residue(vid, col)
        except Exception as exc:
            self._status.setText(f"Restore failed: {type(exc).__name__}: {exc}")
            return
        resnum = tab.design.resnum_for_col(col)
        self._after_variant_edit(tab, vid, f"{vid}: residue {resnum} restored to WT.")

    def _do_insert_residues(self, tab: _ChainDesignTab, vid: str, col: int) -> None:
        """Insert one or more residues into THIS variant after the clicked column. Adds new
        columns owned by this variant; the template and every sibling row gap there (PDB
        insertion-code numbering, e.g. 52A/52B). Inserted residues have no WT counterpart →
        they are excluded from the deviation, not on the crystal until the variant is folded."""
        resnum = tab.design.resnum_for_col(col)
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Insert residues",
            f"Residues to insert after residue {resnum} (1-letter, e.g. GGSGG):")
        if not ok:
            return
        seq = "".join(text.split()).upper()
        if not seq:
            self._status.setText("Insert cancelled (no residues entered).")
            return
        try:
            tab.design.insert_variant_residues(vid, col, seq)
        except Exception as exc:
            self._status.setText(f"Insert failed: {type(exc).__name__}: {exc}")
            return
        self._after_variant_edit(
            tab, vid, f"{vid}: inserted {seq} after residue {resnum} ({len(seq)} new column(s); "
                      f"template + siblings gap there). Re-fold the variant to build them.")

    def _do_remove_insertion(self, tab: _ChainDesignTab, vid: str, col: int) -> None:
        try:
            tab.design.remove_variant_insertion(vid, col)
        except Exception as exc:
            self._status.setText(f"Remove insertion failed: {type(exc).__name__}: {exc}")
            return
        self._after_variant_edit(tab, vid, f"{vid}: insertion removed (axis restored).")

    def _on_row_menu(self, tab: _ChainDesignTab, vid: str, global_pos) -> None:
        """Right-click on a VARIANT row header → a 'Delete variant' menu. Row-level removal
        only (no confirmation — this is a workbench). T never reaches here (filtered in the
        tab's event handler)."""
        v = tab.design.get_variant(vid)
        if v is None:
            return
        menu = QtWidgets.QMenu(self)
        header = menu.addAction(f"{vid} ({len(v.mutations)} mutation(s))")
        header.setEnabled(False)
        menu.addSeparator()
        menu.addAction("Delete variant", lambda: self._delete_variant(tab, vid))
        menu.exec(global_pos)

    def _delete_variant(self, tab: _ChainDesignTab, vid: str) -> None:
        """Remove a variant ROW: hide its predicted fold model (HIDE, never close — per the
        fold-output decision), drop the row + its results from the design, re-render, move
        the active row off it if needed, persist. ROW-LEVEL ONLY — never touches residue
        numbering or the deviation reference data (that is the deferred indel increment)."""
        v = tab.design.get_variant(vid)
        if v is None:
            return
        # Hide the variant's predicted fold model if one exists (don't close it).
        fold = v.results.fold or {}
        mid = fold.get("model_id")
        if mid:
            self._run_commands_bg([f"hide #{mid} models"])
        if not tab.design.delete_variant(vid):
            return
        if tab.active_row_id == vid:                  # active row gone → fall back to T
            tab.set_active_row("T")
            self._edit_target = None
        tab.badges.pop(vid, None)
        tab.rebuild()
        self._apply_color_to(tab)
        self._push_3d_color(tab)                      # refresh fold visibility for the new active row
        self._persist()
        self._status.setText(f"Deleted variant {vid} (row removed; its fold model hidden, "
                             f"not closed). Residue numbering untouched.")

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
        self._scan_cols.clear()                # cols index this tab's template; reset
        self._update_scan_label()
        self._sync_align_ref_toggle()          # reflect THIS cd's aligned-reference state
        self._apply_color_to(tab)
        self._push_3d_color(tab)

    # ── Stage 3a: consume cached tool results (batch import + inline cherry-pick) ────
    def _suggestions_for(self, cd: ChainDesign) -> Dict[int, List[dict]]:
        """Per-column ranked scan suggestions for *cd* from the cached scan result,
        filtered to cd's member chains (a scan on ONE homo-oligomer copy lands in the
        collapsed unique-chain tab). {} when no scan is cached → no Suggest track."""
        if self._session is None or self._design is None:
            return {}
        try:
            scan = self._session.get_scan_result(self._design.model_id)
        except Exception:
            scan = None
        if not scan:
            return {}
        chains = {c for _m, c in cd.members} | {cd.rep_chain}
        return group_scan_suggestions(scan, chains, cd.template_cells)

    def _chaindesign_for_chain(self, chain: Optional[str]) -> Optional[ChainDesign]:
        """The unique-chain ChainDesign that owns *chain* (rep or any member copy);
        the first design when chain is None (uncified result). None if no match."""
        if self._design is None:
            return None
        for cd in self._design.chains.values():
            chains = {c for _m, c in cd.members} | {cd.rep_chain}
            if chain is None or chain in chains:
                return cd
        return None

    def _focus_tab_for_design(self, cd: ChainDesign) -> Optional[_ChainDesignTab]:
        for i in range(self._tabs.count()):
            w = self._tabs.widget(i)
            if isinstance(w, _ChainDesignTab) and w.design is cd:
                self._tabs.setCurrentIndex(i)
                return w
        return None

    def _import_mpnn(self) -> None:
        if self._session is None or self._design is None:
            self._status.setText("No session/structure — nothing to import.")
            return
        try:
            mpnn = self._session.get_proteinmpnn_result(self._design.model_id)
        except Exception:
            mpnn = None
        if not mpnn or not mpnn.get("sequences"):
            self._status.setText("No cached ProteinMPNN designs. Run e.g. 'redesign chain A "
                                 "with ProteinMPNN' first, then Import.")
            return
        chain = str(mpnn.get("chain", "")) or None
        cd = self._chaindesign_for_chain(chain)
        if cd is None:
            self._status.setText(f"ProteinMPNN designs are for chain {chain}, which isn't a "
                                 f"loaded unique chain.")
            return
        run_id = len({v.provenance.get("fasta_path") for v in cd.variants
                      if v.source == "proteinmpnn" and v.provenance.get("fasta_path")})
        # build with throwaway ids, dedupe, then assign real monotonic ids to survivors
        # only (so re-importing the same cache wastes no V-numbers)
        _tmp = iter(range(10 ** 9))
        candidates = import_mpnn_designs(cd, mpnn, run_id, lambda: f"__tmp{next(_tmp)}")
        new = filter_new_mpnn_variants(cd.variants, candidates)
        if not new:
            n_seq = len(mpnn.get("sequences", []))
            if not candidates:
                self._status.setText(f"None of the {n_seq} MPNN design(s) align to the "
                                     f"template length for chain {cd.rep_chain} — skipped.")
            else:
                self._status.setText("These MPNN designs are already imported "
                                     "(no duplicate rows added).")
            return
        for vrow in new:
            vrow.id = self._design.new_variant_id()
        cd.variants.extend(new)
        tab = self._focus_tab_for_design(cd)
        if tab is not None:
            tab.rebuild()
            self._apply_color_to(tab)
        self._persist()
        self._status.setText(f"Imported {len(new)} ProteinMPNN design(s) for chain "
                             f"{cd.rep_chain} as variant rows (run {run_id}).")

    def _load_suggestions(self) -> None:
        if self._design is None:
            return
        total = 0
        for i in range(self._tabs.count()):
            tab = self._tabs.widget(i)
            if isinstance(tab, _ChainDesignTab):
                sugg = self._suggestions_for(tab.design)
                tab.set_suggestions(sugg)
                total += len(sugg)
        cur = self._cur_tab()
        if cur is not None:
            self._apply_color_to(cur)
        if total == 0:
            self._status.setText("No scan suggestions cached. Run a mutation scan (e.g. "
                                 "'scan chain A for stabilizing mutations') first.")
        else:
            self._status.setText(f"Loaded scan suggestions at {total} position(s). Add/select "
                                 f"a variant row, then click a Suggest cell to cherry-pick.")

    def _show_suggestion_menu(self, tab: _ChainDesignTab, col: int) -> None:
        cands = tab.suggestions.get(col) or []
        if not cands:
            return
        menu = QtWidgets.QMenu(self)
        head = menu.addAction(f"Suggestions @ residue {tab.design.resnum_for_col(col)} "
                              f"(into {tab.active_row_id}):")
        head.setEnabled(False)
        menu.addSeparator()
        acts: Dict[QtGui.QAction, dict] = {}
        for c in cands:
            label = (f"{c.get('from_aa','?')}→{c.get('to_aa','?')}   "
                     f"score {c.get('combined_score', 0.0):+.2f}"
                     + (f"   · {c.get('recommendation','')}" if c.get("recommendation") else ""))
            acts[menu.addAction(label)] = c
        chosen = menu.exec(QtGui.QCursor.pos())
        if chosen in acts:
            self._accept_suggestion(tab, col, acts[chosen])

    def _accept_suggestion(self, tab: _ChainDesignTab, col: int, cand: dict) -> None:
        """Cherry-pick a scanner candidate into the ACTIVE variant (provenance
        accepted_suggestion + the scan score). Applies to all copies; recolors panel+3D."""
        if tab.active_row_id == "T" or tab.design.get_variant(tab.active_row_id) is None:
            self._status.setText("Add or select a VARIANT row first, then accept a suggestion "
                                 "(T is the immutable template).")
            return
        vid = tab.active_row_id
        to_aa = str(cand.get("to_aa", ""))
        note = {"combined_score": cand.get("combined_score"),
                "recommendation": cand.get("recommendation"),
                "from_tool": "mutation_scanner"}
        try:
            tab.design.edit_variant(vid, col, to_aa, source="accepted_suggestion", note=note)
        except Exception as exc:
            self._status.setText(f"Accept failed: {type(exc).__name__}: {exc}")
            return
        tab.rebuild()
        tab.set_active_row(vid)
        self._edit_target = (vid, col)
        self._apply_color_to(tab)
        self._push_3d_color(tab)
        self._persist()
        resnum = tab.design.resnum_for_col(col)
        self._status.setText(
            f"Accepted {cand.get('from_aa','?')}{resnum}{to_aa} into {vid} "
            f"(score {cand.get('combined_score', 0.0):+.2f}, accepted_suggestion). "
            f"Applies to all copies; 3D recolored.")

    # ── cell click: plain = S2 edit-target; Ctrl+click = scan-set (distinct gesture) ──
    def _on_cell(self, tab: _ChainDesignTab, row_id, col: int, to_scan: bool = False) -> None:
        if row_id == _SUGGEST_ROW:                         # inline cherry-pick affordance
            self._select_column(tab.design, col)           # show where it is in 3D
            self._show_suggestion_menu(tab, col)
            return
        # Stage 3b — DISTINCT gesture: Ctrl(+Cmd)-click TOGGLES the position in the scan
        # set (the deterministic scan scope) and 3D-selects the whole set across copies.
        # It does NOT touch the edit target / active row, so a plain click keeps its full
        # S2 meaning below.
        if to_scan:
            if tab.design.resnum_for_col(col) is None:     # gap column → not scannable
                return
            self._scan_cols.symmetric_difference_update({col})
            self._update_scan_label()
            self._select_scan_set(tab.design)              # scan set → 3D (all copies)
            n = len(self._scan_set_resnums(tab.design))
            self._status.setText(f"Scan set: {n} site(s). Ctrl+click to toggle; "
                                 f"Scan…/Run ProteinMPNN… cover these (or the whole chain if empty).")
            return
        # Plain click — UNCHANGED S2 behavior: select just this column in 3D and set the
        # active row / (variant) edit target.
        self._select_column(tab.design, col)               # column→3D select (all copies)
        if row_id == "T":
            tab.set_active_row("T")
            self._edit_target = None
            self._apply_color_to(tab)                       # panel result-coloring FOLLOWS active
            self._push_3d_color(tab)
        elif row_id is not None:                            # a variant row
            tab.set_active_row(row_id)
            self._edit_target = (row_id, col)
            resnum = tab.design.resnum_for_col(col)
            in_range = 0 <= col < len(tab.design.template_cells)
            wt = tab.design.template_cells[col].aa if in_range else "?"
            tmpl_gap = in_range and tab.design.template_cells[col].is_gap
            vv = tab.design.get_variant(row_id)
            deleted = vv is not None and col < len(vv.cells) and vv.cells[col].is_gap
            if tmpl_gap and not deleted:                    # decision #3: an INSERTED residue is
                cur = vv.cells[col].aa if vv and col < len(vv.cells) else "?"
                self._status.setText(                       # not on the crystal — nothing painted
                    f"Inserted {cur} in {row_id} — not on the crystal backbone until the variant "
                    f"is folded (no WT counterpart; excluded from deviation). Right-click → Remove "
                    f"insertion to undo.")
            elif deleted:                                   # decision #3: keep selecting the
                self._status.setText(                       # crystal residue + surface the cue
                    f"Residue {resnum} — DELETED in {row_id} (the crystal residue is selected; "
                    f"right-click the gap → Restore to undo).")
            else:
                readout = self._residue_deviation_readout(tab, col)   # DIAGNOSTIC probe
                base = (f"Edit target: {row_id} col {col} (residue {resnum}, T={wt}). "
                        f"Pick an aa and Apply.  (Ctrl+click builds the scan set.)")
                self._status.setText(f"{base}  [deviation] {readout}" if readout else base)
            self._apply_color_to(tab)                       # panel result-coloring FOLLOWS active
            self._push_3d_color(tab)

    def _residue_deviation_readout(self, tab: _ChainDesignTab, col: int) -> str:
        """DIAGNOSTIC probe: the clicked variant residue's per-residue dRMSD vs its floor (and
        lDDT), so a 'white but visibly displaced' residue can be checked directly — is its raw
        dRMSD genuinely low (the variant didn't change it vs the WT FOLD — note the metric
        references the WT *fold*, while the 3D overlays on the *crystal*), or high-but-GATED by
        the floor? Empty when not in deviation mode / no value for the residue."""
        if self._mode_key != _RESULT_DEVIATION_MODE:
            return ""
        block = self._active_deviation(tab)
        if not block:
            return ""
        v = tab.design.get_variant(tab.active_row_id)
        if v is None or not (0 <= col < len(tab.design.template_cells)):
            return ""
        if tab.design.template_cells[col].is_gap:
            return "inserted (no WT reference)"
        var_pos = sum(1 for c in range(col + 1)
                      if c < len(v.cells) and v.cells[c] is not None and not v.cells[c].is_gap)
        if var_pos == 0:
            return ""
        ref = build_fold_column_map(v, tab.design.template_cells).get(var_pos)
        ddm = block.get("ddm") or {}
        if ref is None or str(ref) not in ddm:
            return ""
        dv = ddm[str(ref)]
        df = (block.get("floor_ddm") or {}).get(str(ref), _DDM_FLOOR_MIN_A)
        lv = (block.get("lddt") or {}).get(str(ref))
        lf = (block.get("floor_lddt") or {}).get(str(ref), _LDDT_NEUTRAL_CAP)
        confident = (dv > df) or (lv is not None and lv < lf)
        verdict = ("CONFIDENT disruption" if confident
                   else "distinct but within WT noise (grey)" if dv > _DDM_FLOOR_MIN_A
                   else "aligned")
        s = (f"ref{ref}: dRMSD {dv:.1f}/floor {df:.1f} Å"
             + (f", lDDT {lv:.2f}/floor {lf:.2f}" if lv is not None else "")
             + f" → {verdict}")
        return s

    def _select_variant_row(self, tab: _ChainDesignTab, row_id) -> None:
        """Row-header NAME click → SELECT *row_id* as the active row and drive the 3D via the
        active-row coupling (HIDE switching: `_push_3d_color` → `fold_visibility_commands`
        shows this variant's fold + the reference, hides the other folds). A ROW-level select
        (no per-column edit target) and SILENT — never the results modal (that is the badge
        gesture). Mirrors the cell-click select. Prerequisite for the active-row fold switch."""
        if tab is None or row_id is None:
            return
        tab.set_active_row(row_id)
        self._edit_target = None
        self._apply_color_to(tab)                           # panel result-coloring FOLLOWS active
        self._push_3d_color(tab)
        if row_id == "T":
            self._status.setText("Selected T (template baseline) — the active row.")
        else:
            self._status.setText(
                f"Selected {row_id} — the active row (fold + reference shown, others hidden). "
                f"Click its result badge for the per-mutation detail.")

    # ── coloring helpers ───────────────────────────────────────────────────────────
    def _active_ddg_map(self, tab: _ChainDesignTab) -> Dict[int, float]:
        """{resnum: ddg} for the active variant's stability result (empty if none/T)."""
        v = tab.design.get_variant(tab.active_row_id)
        if v is None or not v.results.stability:
            return {}
        return {int(rn): d for rn, d in (v.results.stability.get("per_resnum") or {}).items()
                if d is not None}

    def _active_plddt_map(self, tab: _ChainDesignTab) -> Dict[int, float]:
        """{author_resnum: pLDDT} for the active row's fold. A variant → its `results.fold`; the
        TEMPLATE row of a DE-NOVO construct → the construct's own fold on `cd.template_fold`
        (T has no Variant object). Empty when nothing folded."""
        v = tab.design.get_variant(tab.active_row_id)
        if v is not None and v.results.fold:
            return {int(rn): float(p) for rn, p in (v.results.fold.get("plddt") or {}).items()
                    if p is not None}
        if v is None and tab.design.template_fold:          # T of a folded de-novo construct
            return {int(rn): float(p) for rn, p in (tab.design.template_fold.get("plddt") or {}).items()
                    if p is not None}
        return {}

    def _active_deviation(self, tab: _ChainDesignTab) -> Optional[Dict[str, Any]]:
        """The active variant's stored deviation block (None if not folded/no deviation)."""
        v = tab.design.get_variant(tab.active_row_id)
        if v is None or not v.results.fold:
            return None
        return v.results.fold.get("deviation")

    @staticmethod
    def _dev_chain_keys(dev: Dict[str, float], multichain: bool, rep_chain: str) -> List[str]:
        """Ordered deviation keys for the panel's UNIQUE chain: the rep chain's entries for
        an assembly (multichain `"chain:resno"` keys), else all (single-chain `"resno"`)."""
        if multichain:
            keys = [k for k in dev if k.split(":", 1)[0] == rep_chain]
            return sorted(keys, key=lambda k: int(k.split(":", 1)[1]))
        return sorted(dev, key=lambda k: int(k))

    def _deviation_panel_hex(self, tab: _ChainDesignTab) -> Dict[int, str]:
        """{author_resnum: hex} for the active variant's floor-gated dRMSD disruption, panel
        side. The fold numbers residues 1..N over the ungapped sequence, so the rep chain's
        dRMSD keys map POSITIONALLY onto the active row's ordered author resnums (the same
        positional contract `fold_summary` uses for pLDDT). 3-tier via `combined_disruption_color`
        (the ONE source shared with the 3D push). Inserted residues are absent from the dRMSD keys
        (no WT counterpart) → they never land on an author resnum → neutral."""
        block = self._active_deviation(tab)
        if not block:
            return {}
        ddm = block.get("ddm") or {}
        floor_ddm = block.get("floor_ddm") or {}
        lddt = block.get("lddt") or {}
        floor_lddt = block.get("floor_lddt") or {}
        keys = self._dev_chain_keys(ddm, bool(block.get("multichain")), tab.design.rep_chain)
        author = [c.resnum for c in tab.active_row_cells()
                  if not c.is_gap and c.resnum is not None]
        out: Dict[int, str] = {}
        for i, k in enumerate(keys):
            if i >= len(author):
                break
            hexc = combined_disruption_color(ddm.get(k), floor_ddm.get(k, _DDM_FLOOR_MIN_A),
                                             lddt.get(k), floor_lddt.get(k, _LDDT_NEUTRAL_CAP))
            if hexc:
                out[author[i]] = hexc
        return out

    def _refresh_color_mode_availability(self, tab: _ChainDesignTab) -> None:
        """Grey out a RESULT colour mode (ddG / pLDDT / Deviation vs WT) when the ACTIVE
        variant has no such result yet — so e.g. selecting 'Color: Deviation vs WT' can't be
        a silent no-op before the 'Deviation vs WT' button has computed it. If the CURRENT
        mode just became unavailable (e.g. switching to a row that hasn't been folded/scanned),
        revert the combo to None so the displayed mode always matches what's painted."""
        avail = {
            _RESULT_DDG_MODE:       bool(self._active_ddg_map(tab)),
            _RESULT_PLDDT_MODE:     bool(self._active_plddt_map(tab)),
            _RESULT_DEVIATION_MODE: bool(self._active_deviation(tab)),
        }
        model = self._mode_combo.model()
        for i in range(self._mode_combo.count()):
            key = self._mode_combo.itemData(i)
            if key in avail and (item := model.item(i)) is not None:
                item.setEnabled(avail[key])
        if self._mode_key in avail and not avail[self._mode_key]:
            self._select_result_mode("none")          # current result has no data → revert
            self._mode_key = "none"

    def _apply_color_to(self, tab: _ChainDesignTab) -> None:
        self._refresh_color_mode_availability(tab)    # grey result modes lacking data
        if self._mode_key == _RESULT_DDG_MODE:
            hexmap = {rn: ddg_color(d) for rn, d in self._active_ddg_map(tab).items()}
            tab.set_result_coloring(tab.active_row_id, {k: v for k, v in hexmap.items() if v})
        elif self._mode_key == _RESULT_PLDDT_MODE:
            hexmap = {rn: plddt_color(p) for rn, p in self._active_plddt_map(tab).items()}
            tab.set_result_coloring(tab.active_row_id, {k: v for k, v in hexmap.items() if v})
        elif self._mode_key == _RESULT_DEVIATION_MODE:
            tab.set_result_coloring(tab.active_row_id, self._deviation_panel_hex(tab))
        else:
            tab.set_color_mode(get_mode(self._mode_key))

    def _push_3d_color(self, tab: _ChainDesignTab) -> None:
        """Recolor the 3D by the tab's ACTIVE row across all copies AND make predicted fold
        models follow the active row (per-model visibility). No-op for the OFF mode with no
        folds (non-destructive: we do not know the pre-overlay coloring to restore).
        If a `tile` is in effect, this RESTORES superposition first (item 4: tile is a
        transient comparison — the next active-row push returns to the overlay)."""
        restore = self._tiled
        cmds = self.untile_commands() if restore else []
        if restore:
            self._tiled = False
        cmds += self.fold_visibility_commands(tab) + self.color_commands_for(tab)
        if restore:
            cmds.append("view")          # frame the restored overlay (after show/hide)
        self._run_commands_bg(cmds)

    def untile_commands(self) -> List[str]:
        """Re-superpose every variant fold onto the WT reference — the inverse of
        `tile_commands` (which laid them side-by-side and broke superposition). Re-running
        matchmaker restores the overlay; the caller appends a `view` AFTER visibility is set
        so the final overlaid set is framed. [] if no design / no folds. Pure → single
        source for the push + tests."""
        if self._design is None:
            return []
        ref = str(self._design.model_id)
        folds = [m for m in self._fold_models(self._design).values() if m and m != ref]
        return [f"matchmaker #{m} to #{ref}" for m in folds]

    def _run_commands_bg(self, cmds: List[str]) -> None:
        """Fire-and-forget a ChimeraX command list off the UI thread (the shared 3D push)."""
        if not cmds:
            return
        w = _ColorWorker(self._c, cmds)
        w.signals.failed.connect(lambda e: self._status.setText(f"Workbench 3D command failed: {e}"))
        self._pool.start(w)

    def _select_result_mode(self, mode_key: str) -> None:
        """Set the active color mode + sync the combo display WITHOUT re-firing
        _on_mode_changed (the caller re-renders). Auto-surfaces a fresh result's mode."""
        self._mode_key = mode_key
        cb = self._mode_combo
        for i in range(cb.count()):
            if cb.itemData(i) == mode_key:
                cb.blockSignals(True)
                cb.setCurrentIndex(i)
                cb.blockSignals(False)
                return

    # ── column-click → 3D select (ALL copies), off the UI thread ───────────────────
    def _select_column(self, design: ChainDesign, col: int) -> None:
        specs = self.select_specs_for_column(design, col)
        if not specs:
            return
        w = _MultiSelectWorker(self._c, specs)
        w.signals.failed.connect(lambda e: self._status.setText(f"Workbench select failed: {e}"))
        self._pool.start(w)

    def _select_scan_set(self, design: ChainDesign) -> None:
        """3D-select every column in the scan set across all copies (off the UI thread).
        Empty set → a bare `select` clears the ChimeraX selection."""
        specs: List[Tuple[str, str, List[int]]] = []
        for col in sorted(self._scan_cols):
            specs.extend(self.select_specs_for_column(design, col))
        w = _MultiSelectWorker(self._c, specs)
        w.signals.failed.connect(lambda e: self._status.setText(f"Workbench select failed: {e}"))
        self._pool.start(w)

    def _update_scan_label(self) -> None:
        n = len(self._scan_cols)
        self._scan_set_lbl.setText(f"scan set: {n}")        # in-menu count (tests read .text())
        self._tools_btn.setText(f"Tools ({n})" if n else "Tools")  # at-a-glance badge on the button

    def _clear_scan_set(self) -> None:
        self._scan_cols.clear()
        self._update_scan_label()
        tab = self._cur_tab()
        if tab is not None:
            self._select_scan_set(tab.design)              # clears the 3D selection
        self._status.setText("Scan set cleared — Scan…/Run ProteinMPNN… now cover the whole chain.")

    # exposed for tests / live-verify: the exact specs a column click dispatches
    def select_specs_for_column(self, design: ChainDesign, col: int):
        resnum = design.resnum_for_col(col)
        if resnum is None:
            return []
        return [(m, c, [resnum]) for (m, c) in design.members]

    # ── Stage 3b: build a deterministic launch spec; emit it for the window to run ───
    def _scan_set_resnums(self, design: ChainDesign) -> List[int]:
        """The scan set as sorted author resnums (gap columns dropped)."""
        nums = {design.resnum_for_col(c) for c in self._scan_cols}
        return sorted(n for n in nums if n is not None)

    def scan_launch_spec(self, deep: bool) -> Optional[dict]:
        """Deterministic mutation_scan launch spec for the current tab + scan set.
        deep=True → opt-in Rosetta tier (run_rosetta pre-set; the spine surfaces the
        runtime estimate + confirm-gate, confidence='low' so it never auto-proceeds).
        Empty scan set → whole-chain scan. None when no structure is loaded."""
        tab = self._cur_tab()
        if tab is None or self._design is None:
            return None
        cd      = tab.design
        resnums = self._scan_set_resnums(cd)
        ti: Dict[str, object] = {"model_id": self._design.model_id, "chain": cd.rep_chain}
        if resnums:
            ti["scan_positions"] = resnums
        if deep:
            ti["run_rosetta"] = True
        scope = f"{len(resnums)} site(s)" if resnums else "the whole chain"
        tier  = "deep" if deep else "fast"
        return {
            "tool":        "mutation_scan",
            "tool_inputs": ti,
            # The label is cosmetic — tier/scope come from `ti`, which route()'s tiering
            # honors. It MUST stay free of any token the spine would parse: no
            # "selected"/"selection"/"highlighted" (live-selection scope), no
            # "rosetta"/"rosie" (deep), no thoroughness/shortlist words, and no
            # "residue(s)/position(s) <digits>" (explicit-scope parse). "site(s)" is safe.
            "user_input":  f"[Workbench] mutation scan on chain {cd.rep_chain} — "
                           f"{scope}, {tier} tier",
            "confidence":  "low" if deep else "high",
            "refresh":     "scan",
        }

    def mpnn_launch_spec(self, soluble: bool) -> Optional[dict]:
        """Deterministic ProteinMPNN launch spec for the current tab + scan set.
        soluble=True → soft hydrophilic bias (the solubility design profile);
        empty scan set → whole-chain redesign. None when no structure is loaded."""
        tab = self._cur_tab()
        if tab is None or self._design is None:
            return None
        cd      = tab.design
        resnums = self._scan_set_resnums(cd)
        ti: Dict[str, object] = {"model_id": self._design.model_id, "chain_id": cd.rep_chain}
        if resnums:
            ti["design_positions"] = resnums
        if soluble:
            ti["bias_toward"] = "soluble"
        scope   = f"{len(resnums)} site(s)" if resnums else "the whole chain"
        profile = "solubility-biased" if soluble else "default"
        return {
            "tool":        "proteinmpnn",
            "tool_inputs": ti,
            # trigger-free label (see scan_launch_spec) — scope comes from `ti`.
            "user_input":  f"[Workbench] ProteinMPNN redesign of chain {cd.rep_chain} — "
                           f"{scope}, {profile}",
            "confidence":  "high",
            "refresh":     "mpnn",
        }

    def _on_scan_clicked(self) -> None:
        if self._cur_tab() is None or self._design is None:
            self._status.setText("Load a structure first.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Mutation scan")
        n = len(self._scan_set_resnums(self._cur_tab().design))
        box.setText(f"Scan {n} selected position(s)" if n else "Scan the whole chain")
        box.setInformativeText("Fast = CamSol + ESM (seconds). Deep adds Rosetta ddG — "
                               "the spine shows a runtime estimate and asks before launching.")
        fast = box.addButton("Fast", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        deep = box.addButton("Deep (+Rosetta)", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (fast, deep):
            return
        spec = self.scan_launch_spec(deep=clicked is deep)
        if spec is not None:
            self.launchRequested.emit(spec)

    def _on_mpnn_clicked(self) -> None:
        if self._cur_tab() is None or self._design is None:
            self._status.setText("Load a structure first.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Run ProteinMPNN")
        n = len(self._scan_set_resnums(self._cur_tab().design))
        box.setText(f"Redesign {n} selected position(s)" if n else "Redesign the whole chain")
        box.setInformativeText("Default = unconstrained ProteinMPNN. Solubility biases the "
                               "design toward polar/charged residues.")
        default = box.addButton("Default", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        sol     = box.addButton("Solubility", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked not in (default, sol):
            return
        spec = self.mpnn_launch_spec(soluble=clicked is sol)
        if spec is not None:
            self.launchRequested.emit(spec)

    # ── Stage 4a: per-variant action buttons (act on the ACTIVE variant) ────────────
    def _active_variant(self, tab: _ChainDesignTab):
        """The active row as a Variant, or None when the active row is T / not a variant."""
        return None if tab is None else tab.design.get_variant(tab.active_row_id)

    def stability_launch_spec(self, deep: bool) -> Optional[dict]:
        """Deterministic mutation_scan spec that scores the ACTIVE variant's EXACT
        mutations (score_mutations={resnum: to_aa}) through the 4-axis voter. deep=True →
        Rosetta (confidence='low' → confirm-gate). None when no active variant / no
        mutations to score."""
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None or self._design is None or not v.mutations:
            return None
        score_mutations = {m.resnum: m.to_aa for m in v.mutations}
        ti: Dict[str, object] = {"model_id": self._design.model_id, "chain": tab.design.rep_chain,
                                 "score_mutations": score_mutations,
                                 # scope the deep-tier estimate to the scored positions
                                 "scan_positions": sorted(score_mutations)}
        if deep:
            ti["run_rosetta"] = True
        tier = "deep" if deep else "fast"
        return {
            "tool":        "mutation_scan",
            "tool_inputs": ti,
            # trigger-free label (scope/tier come from ti); names the variant for the log.
            "user_input":  f"[Workbench] stability of {v.id} on chain {tab.design.rep_chain} "
                           f"— {len(score_mutations)} mutation(s), {tier} tier",
            "confidence":  "low" if deep else "high",
            "refresh":     "stability",
            "_variant_id": v.id,
        }

    def _on_test_stability(self) -> None:
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None:
            self._status.setText("Select a VARIANT row first (T is the template — nothing to test).")
            return
        if not v.mutations:
            self._status.setText(f"{tab.active_row_id} has no mutations vs T — nothing to score.")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Test stability")
        box.setText(f"Score {v.id}: {len(v.mutations)} mutation(s) "
                    f"({', '.join(f'{m.from_aa}{m.resnum}{m.to_aa}' for m in v.mutations[:6])}"
                    f"{'…' if len(v.mutations) > 6 else ''})")
        box.setInformativeText("Fast = CamSol+ESM+ThermoMPNN+RaSP (seconds). Deep adds Rosetta "
                               "ddG — the spine shows a runtime estimate and asks before launching.")
        fast = box.addButton("Fast", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        deep = box.addButton("Deep (+Rosetta)", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() not in (fast, deep):
            return
        spec = self.stability_launch_spec(deep=box.clickedButton() is deep)
        if spec is not None:
            # snapshot the shared scan cache so a stability run (which the scanner caches
            # model-keyed) does not clobber the S3a Suggest-track cache; restored on apply.
            self._scan_cache_snapshot = (self._design.model_id,
                                         self._read_scan_cache(self._design.model_id))
            self.launchRequested.emit(spec)

    def _read_scan_cache(self, model_id):
        if self._session is None:
            return None
        try:
            return self._session.get_scan_result(model_id)
        except Exception:
            return None

    def apply_stability_result(self, variant_id: str, result: dict) -> None:
        """Consume the executed mutation_scan result (from the engine on_result seam) into
        the named variant's ResultSlots.stability, restore the S3a scan cache, then
        re-render badges + (if the ddG mode is active) recolor. Called on the UI thread."""
        cd_v = self._find_variant(variant_id)
        if cd_v is None:
            return
        cd, v = cd_v
        candidates = self._candidates_from_result(result)
        v.results.stability = stability_summary(candidates, v.mutations)
        # restore the suggestion-scan cache the stability run overwrote
        snap = getattr(self, "_scan_cache_snapshot", None)
        if snap is not None and self._session is not None:
            mid, prior = snap
            try:
                if prior is None:
                    self._session.scan_results.pop(str(mid), None)
                else:
                    self._session.add_scan_result(str(mid), prior)
            except Exception:
                pass
            self._scan_cache_snapshot = None
        self._persist()
        self._rerender_results(cd, v)
        s = v.results.stability
        self._status.setText(
            f"{v.id} stability ({s.get('tier')}): ΣddG "
            f"{('%+.2f' % s['sum_ddg']) if s.get('sum_ddg') is not None else 'n/a'} "
            f"over {s.get('n_scored', 0)} mutation(s). Click the {v.id} row header for detail; "
            f"pick the 'ddG (result)' color mode to map it.")

    @staticmethod
    def _candidates_from_result(result: dict) -> List[dict]:
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") == "mutation_scan":
                return (step.get("data") or {}).get("candidates", []) or []
        return []

    def _on_test_solubility(self) -> None:
        """Pure CamSol intrinsic-solubility scalar for the active variant vs the template —
        instant/local, no spine launch, no gate."""
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None:
            self._status.setText("Select a VARIANT row first (T is the template baseline).")
            return
        from camsol_bridge import camsol_solubility_score
        wt_seq  = "".join(c.aa for c in tab.design.template_cells if c.aa)
        var_seq = v.sequence
        wt  = camsol_solubility_score(wt_seq)
        var = camsol_solubility_score(var_seq)
        v.results.solubility = {"variant": round(var, 3), "wt": round(wt, 3),
                                "delta": round(var - wt, 3)}
        self._persist()
        self._rerender_results(tab.design, v)
        self._status.setText(f"{v.id} solubility {var:+.2f} (Δ {var - wt:+.2f} vs T) — "
                             f"{'more' if var > wt else 'less'} soluble than the template.")

    # ── Stage 4b: fold the active variant through an engine (user picks) ─────────────
    def _fold_engine_availability(self) -> Dict[str, bool]:
        """Capability flag (B2 3-state): which fold engines can run NOW. ESMFold = the local
        venv312 worker; Boltz = the dedicated ~/boltz_env import chain (WSL). Engines are SHOWN
        enabled-or-disabled in the picker, never silently dropped."""
        esm = boltz = False
        try:
            from esmfold_bridge import ESMFoldBridge
            esm = ESMFoldBridge().local_available()
        except Exception:
            esm = False
        try:
            from boltz_bridge import boltz_available
            boltz = boltz_available()
        except Exception:
            boltz = False
        return {"esmfold": esm, "boltz": boltz}

    def fold_launch_spec(self, engine: str, assembly: bool = False) -> Optional[dict]:
        """Deterministic fold spec for the ACTIVE variant through *engine*: fold LOCAL-ONLY,
        open the model, pLDDT-colour it, matchmaker onto the WT reference. *assembly* (Boltz
        only) folds the full homo-oligomer — one chain per `cd.members` copy, each the variant's
        sequence — overlaid on the WT oligomer. confidence='low' → the spine's confirm-gate.
        None when there is no active variant. The SAME spec shape every engine reuses."""
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None or self._design is None:
            return None
        cd = tab.design
        # DE-NOVO (sequence-seeded): the variant folds at the CONSTRUCT's engine + oligomer (GAP C)
        # and superposes onto the construct's T-FOLD, not the synthetic design id (GAP A). The
        # T-fold IS the de-novo analog of the crystal reference; engine/target are pinned from it,
        # not the picker, so the variant, the WT reference, and the floor seeds all fold the same.
        denovo = self._design.source == "sequence"
        tf = cd.template_fold if denovo else {}
        if denovo:
            if not tf.get("model_id"):
                return None                       # construct not folded yet → no reference to fold against
            engine     = tf.get("engine", engine)             # PIN engine from the construct fold
            assembly   = (tf.get("target") == "assembly")     # PIN oligomer from the construct fold
            compare_to = tf.get("model_id")                   # superpose onto the T-fold (GAP A)
        else:
            compare_to = self._design.model_id    # crystal design: matchmaker onto the loaded WT
        ti: Dict[str, object] = {
            "model_id":   self._design.model_id,
            "chain":      cd.rep_chain,
            "engine":     engine,
            "open_model": True,
            "local_only": True,                   # LOCAL-ONLY: no remote Atlas/MSA server
            "compare_to": compare_to,
        }
        if engine == "boltz" and assembly:
            if denovo:
                # FULL-COMPLEX composition (Stage 2b): a hetero construct's variant folds the
                # WHOLE declared assembly, not the active chain alone. The ACTIVE cd contributes
                # the variant's sequence × its members; EVERY OTHER cd contributes its own
                # template T sequence × its members. Chain ids come from each cd's members (already
                # re-pointed to the construct T-fold's chains), so the variant complex lines up
                # 1:1 with the T-fold (the deviation reference) — same ids, same order.
                chains: List[Dict[str, str]] = []
                for _uk, c in self._design.chains.items():
                    seq = v.sequence if c is cd else "".join(
                        cc.aa for cc in c.template_cells if cc.aa is not None)
                    chains.extend({"id": ch, "sequence": seq} for (_m, ch) in c.members)
                ti["chains"] = chains
                target = f"{len(chains)}-chain assembly"
            else:
                # CRYSTAL homo-oligomer: every copy chain folded with the variant's sequence
                # (variant sequence × the construct's copy count).
                ti["chains"] = [{"id": c, "sequence": v.sequence} for (_m, c) in cd.members]
                target = f"{len(cd.members)}-chain assembly"
        else:
            ti["sequence"] = v.sequence           # the VARIANT sequence (its mutations)
            target = "monomer"
        return {
            "tool":        engine,                # esmfold|boltz → route() dispatch
            "tool_inputs": ti,
            "user_input":  f"[Workbench] fold {v.id} on chain {cd.rep_chain} "
                           f"— {engine} {target}, LOCAL-ONLY",
            "confidence":  "low",                 # fold = non-trivial compute → confirm-gate
            "refresh":     "fold",
            "_variant_id": v.id,
        }

    def _on_fold_clicked(self) -> None:
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None:
            self._status.setText("Select a VARIANT row first (T is the template baseline).")
            return
        cd = self._cur_tab().design
        # DE-NOVO: no engine picker — the variant folds at the CONSTRUCT's engine + oligomer
        # (pinned in fold_launch_spec from cd.template_fold). Requires the construct be folded first.
        if self._design is not None and self._design.source == "sequence":
            tf = cd.template_fold
            if not tf.get("model_id"):
                self._status.setText("Fold the construct first (Fold ▾ → Fold construct) — a "
                                     "variant folds at the construct's engine + oligomer.")
                return
            spec = self.fold_launch_spec(tf.get("engine", "boltz"))   # engine/target pinned inside
            if spec is not None:
                self.launchRequested.emit(spec)
            return
        n_copies = len(cd.members)
        avail = self._fold_engine_availability()
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Fold variant")
        box.setText(f"Fold {v.id} ({len(v.sequence)} aa) — pick engine + target:")
        box.setInformativeText(
            "ESMFold = local monomer (fast). Boltz-2 = higher-quality, LOCAL-ONLY, "
            "seed-pinned; can fold the full assembly. The spine shows a runtime estimate "
            "and asks before launching.")
        # Free user choice (no auto-constrain by target): each combo enabled per capability.
        combos = [("ESMFold (monomer)", "esmfold", False),
                  ("Boltz-2 (monomer)", "boltz", False)]
        if n_copies > 1:
            combos.append((f"Boltz-2 (assembly, {n_copies} chains)", "boltz", True))
        btns: Dict[object, Tuple[str, bool]] = {}
        for label, eng, asm in combos:
            b = box.addButton(label, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            if not avail.get(eng, False):
                b.setEnabled(False)
                b.setToolTip("Boltz env (~/boltz_env) not available" if eng == "boltz"
                             else "Local ESMFold worker (venv312) not installed")
            btns[b] = (eng, asm)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.exec()
        choice = btns.get(box.clickedButton())
        if choice is None:
            return
        eng, asm = choice
        spec = self.fold_launch_spec(eng, assembly=asm)
        if spec is not None:
            self.launchRequested.emit(spec)

    @staticmethod
    def _fold_from_result(result: dict) -> dict:
        """The fold engine's step data from the executed pipeline result (engine-agnostic)."""
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") in _FOLD_ENGINES:
                return step.get("data") or {}
        return {}

    def apply_fold_result(self, variant_id: str, result: dict) -> None:
        """Consume the executed fold result (engine on_result seam) into the variant's
        ResultSlots.fold via the normalized contract, then re-render the badge, AUTO-SURFACE
        the pLDDT result mode (no manual step), and couple the new model to the active row.
        Re-folding the SAME variant REPLACES its prior model (close it — never stack). UI thread."""
        cd_v = self._find_variant(variant_id)
        if cd_v is None:
            return
        cd, v = cd_v
        data = self._fold_from_result(result)
        if not data or not (data.get("new_model_id") or data.get("model_id")):
            self._status.setText(f"{variant_id}: fold produced no model.")
            return
        # DE-NOVO COMPLEX PARITY GUARD (Stage 2b): a variant ASSEMBLY fold must return the SAME
        # chains as the construct T-fold (cd.members across all cds — the deviation reference's
        # chains). Fail loud on id drift so it surfaces here rather than silently dropping
        # residues from the deviation's `common` set later. Mirrors apply_construct_fold_result's
        # read-back guard. (Crystal designs / monomer folds skip — gated on source + target.)
        if self._design is not None and self._design.source == "sequence" \
                and data.get("target") == "assembly":
            observed = data.get("chain_ids")
            expected = [ch for c in self._design.chains.values() for (_m, ch) in c.members]
            if observed is not None and set(map(str, observed)) != set(map(str, expected)):
                self._status.setText(
                    f"{variant_id}: variant fold chain mismatch — expected "
                    f"{sorted(set(map(str, expected)))}, got "
                    f"{sorted(set(map(str, observed)))}. Refusing (fold not trusted).")
                return
        prior_mid = (v.results.fold or {}).get("model_id")     # re-fold → replace this
        author_resnums = [c.resnum for c in v.cells if not c.is_gap and c.resnum is not None]
        v.results.fold = fold_summary(data, author_resnums,
                                      reference_model_id=data.get("reference_model_id"))
        new_mid = v.results.fold.get("model_id")
        self._persist()
        # REPLACE-ON-REFOLD: close the variant's prior predicted model so re-folding doesn't
        # accumulate (one model per variant). New + old are distinct models → no race with
        # the recolor push below.
        if prior_mid and str(prior_mid) != str(new_mid):
            self._run_commands_bg([f"close #{prior_mid}"])
        # AUTO-SURFACE: switch to the pLDDT result mode so the fresh fold maps itself (the
        # _rerender_results below applies it); reuses the single color_modes seam.
        self._select_result_mode(_RESULT_PLDDT_MODE)
        self._rerender_results(cd, v)
        f = v.results.fold
        mp = f.get("mean_plddt")
        if mp is not None:
            iptm = f.get("iptm")
            iptm_txt = f", ipTM {iptm:.3f}" if isinstance(iptm, (int, float)) else ""
            self._status.setText(
                f"{v.id} folded ({f.get('engine')}, {f.get('source')}): model "
                f"#{f.get('model_id')}, mean pLDDT {mp:.1f}{iptm_txt} — overlaid on "
                f"#{f.get('reference_model_id')}, pLDDT-coloured (auto).")
        else:
            self._status.setText(f"{v.id} folded — model #{f.get('model_id')} (pLDDT-coloured, auto).")

    # ── Stage 4c: per-residue variant-vs-WT deviation + noise floor ──────────────────
    def deviation_launch_spec(self) -> Optional[dict]:
        """Deterministic spec to compute the ACTIVE folded variant's per-residue Cα
        deviation vs a seed-pinned WT reference fold of the SAME engine+target. Reuses the
        cached `cd.wt_refs[combo]` when present (cheap, confirm-gate skipped); otherwise the
        router folds the WT reference + cross-seed floor (confidence='low' → the gate, since
        that is N folds). None when the active row isn't a folded variant."""
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None or self._design is None:
            return None
        fold = v.results.fold
        if not fold or not fold.get("model_id"):
            return None
        cd = tab.design
        engine = fold.get("engine", "esmfold")
        target = fold.get("target", "monomer")
        multichain = (target == "assembly")
        combo = f"{engine}:{target}"
        t_seq = "".join(c.aa for c in cd.template_cells if c.aa is not None)
        if multichain:
            if self._design.source == "sequence":
                # FULL-COMPLEX WT reference (Stage 2b): the floor + reference are the WHOLE WT
                # complex (= the construct T-fold), so EVERY cd contributes its own T sequence ×
                # its members — the active cd included (the reference is all-WT; the variant's
                # change lives only in the variant fold). 1:1 with the variant complex above.
                wt_chains = []
                for _uk, c in self._design.chains.items():
                    cseq = "".join(cc.aa for cc in c.template_cells if cc.aa is not None)
                    wt_chains.extend({"id": ch, "sequence": cseq} for (_m, ch) in c.members)
            else:
                wt_chains = [{"id": c, "sequence": t_seq} for (_m, c) in cd.members]
            variant_chain = cd.rep_chain
        else:
            variant_chain = "A" if engine == "esmfold" else cd.rep_chain
            wt_chains = [{"id": variant_chain, "sequence": t_seq}]
        # WT reference: a cached one wins; else for a DE-NOVO construct REUSE the displayed T-fold
        # (no fresh fold of T) — its model_id is seed-0 of the floor and the deviation reference.
        # The router reads ref Cα from this open model and, finding no floor yet, folds only the
        # N-1 extra seeds. A crystal design passes None → the router folds the reference + floor.
        wt_ref = cd.wt_refs.get(combo)
        if not wt_ref and self._design.source == "sequence" and cd.template_fold.get("model_id"):
            tf = cd.template_fold
            wt_ref = {"engine":   tf.get("engine"),  "target": tf.get("target"),
                      "seed":     tf.get("seed"),    "model_id": tf.get("model_id"),
                      "path":     tf.get("cif_path") or tf.get("pdb_path")}
        ti: Dict[str, object] = {
            "variant_model_id": fold["model_id"],
            "engine":           engine,
            "target":           target,
            "multichain":       multichain,
            "variant_chain":    variant_chain,
            "wt_chains":        wt_chains,
            "compare_to":       self._design.model_id,   # reference matchmaker onto crystal WT
            "model_id":         self._design.model_id,
            "wt_ref":           wt_ref,                   # cached/reused reference (skip T fold) or None
            "local_only":       True,
        }
        # INDEL-AWARE pairing (Stage A, monomer): carry the variant-fold→reference-fold
        # column map so a deletion's downstream residues pair correctly. Identity for a
        # substitution-only variant → byte-identical deviation (the additive guarantee).
        if not multichain:
            ti["fold_column_map"] = build_fold_column_map(v, cd.template_cells)
        have_ref = bool(cd.wt_refs.get(combo))
        return {
            "tool":        "variant_deviation",
            "tool_inputs": ti,
            "user_input":  f"[Workbench] deviation {v.id} vs WT — {combo}, LOCAL-ONLY",
            "confidence":  "high" if have_ref else "low",   # folding the WT reference → gate
            "refresh":     "deviation",
            "_variant_id": v.id,
        }

    def _on_deviation_clicked(self) -> None:
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None:
            self._status.setText("Select a VARIANT row first (T is the template baseline).")
            return
        if not (v.results.fold and v.results.fold.get("model_id")):
            self._status.setText(f"Fold {v.id} first — deviation compares FOLDED models.")
            return
        # Stage A: indel-aware column pairing is MONOMER-only. Refuse an indel variant folded
        # as an ASSEMBLY rather than silently mis-pair at the deletion (the §0 silent-wrong
        # guard — multichain per-chain mapping is a later phase).
        if v.indels and v.results.fold.get("target") == "assembly":
            self._status.setText(
                f"{v.id} has {len(v.indels)} indel(s) AND was folded as an ASSEMBLY — "
                f"indel-aware deviation is monomer-only for now. Fold {v.id} as a MONOMER "
                f"to compare (refusing to mis-pair the multimer deletion).")
            return
        spec = self.deviation_launch_spec()
        if spec is not None:
            self.launchRequested.emit(spec)

    @staticmethod
    def _deviation_from_result(result: dict) -> dict:
        """The variant_deviation step data from the executed pipeline result."""
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") == "variant_deviation":
                return step.get("data") or {}
        return {}

    def apply_deviation_result(self, variant_id: str, result: dict) -> None:
        """Consume the executed deviation result: store the deviation block on the variant's
        fold slot, cache the WT reference on the design (keyed by engine:target so each combo
        reuses its own same-engine reference), persist, and re-render (the deviation color
        mode maps it). UI thread."""
        cd_v = self._find_variant(variant_id)
        if cd_v is None:
            return
        cd, v = cd_v
        data = self._deviation_from_result(result)
        if not data or v.results.fold is None:
            self._status.setText(f"{variant_id}: deviation produced no result.")
            return
        v.results.fold["deviation"] = data
        wt_ref = data.get("wt_ref")
        if wt_ref:
            combo = f"{data.get('engine')}:{data.get('target')}"
            if self._design is not None and self._design.source == "sequence" \
                    and data.get("target") == "assembly":
                # FLOOR-ONCE (Stage 2b): the multichain floor folds the WHOLE complex once and its
                # floor_ddm/floor_lddt span ALL chains, so the wt_ref is shared by every cd.
                # Distribute it to EVERY cd's wt_refs[combo] (not just the active one) → a sibling
                # cd's next deviation sees have_ref=True and does NO re-fold (floor folded once).
                for c in self._design.chains.values():
                    c.wt_refs[combo] = wt_ref
            else:
                cd.wt_refs[combo] = wt_ref
        self._persist()
        self._select_result_mode(_RESULT_DEVIATION_MODE)   # AUTO-SURFACE the deviation mode
        self._rerender_results(cd, v)
        self._status.setText(
            f"{v.id} disruption vs WT ({data.get('engine')}:{data.get('target')}) — dRMSD: "
            f"{data.get('n_disrupted','?')}/{data.get('n_residues','?')} residues disrupted "
            f"(above the cross-seed floor); max {data.get('max_ddm','?')} Å · local integrity "
            f"(lDDT) min {data.get('min_lddt','?')}, mean {data.get('mean_lddt','?')}. "
            f"Click a residue for its per-residue dRMSD/lDDT vs floor.")

    # ── Stage 3: structural alignment of the construct fold onto a chosen PDB ─────────
    def _cur_cd_ukey(self) -> Optional[str]:
        """The unique-chain key of the active tab's ChainDesign (for routing the result back)."""
        tab = self._cur_tab()
        if tab is None or self._design is None:
            return None
        return next((k for k, c in self._design.chains.items() if c is tab.design), None)

    def structural_align_launch_spec(self, *, reference_pdb_id: Optional[str] = None,
                                     reference_path: Optional[str] = None,
                                     reference_model_id: Optional[str] = None,
                                     ref_label: Optional[str] = None,
                                     use_guided: bool = False) -> Optional[dict]:
        """Spec to structurally align the ACTIVE de-novo construct's FOLD onto an EXPLICIT chosen
        reference via US-align (sequence-independent). Unlike the construct FOLD path (which forces
        `no_reference` so matchmaker never silently superposes onto a loaded primary), this carries
        the user's EXPLICIT reference and aligns to it. The query is the construct's T-fold on disk
        (`template_fold.cif_path`/`pdb_path`); the open fold model is moved onto the reference
        (option B). None unless the active construct is folded. Pure — no I/O (caller resolves a
        loaded-model reference to a file first).

        *use_guided*: align the GUIDED fold (`cd.guided_fold`) instead of the unguided T-fold —
        the template-adoption VALIDATION (did the guided fold adopt the guiding template? TM is
        the metric). Zero new alignment code — same US-align path, different query."""
        tab = self._cur_tab()
        if tab is None or self._design is None or self._design.source != "sequence":
            return None
        cd = tab.design
        tf = cd.guided_fold if use_guided else cd.template_fold
        query_path = tf.get("cif_path") or tf.get("pdb_path")
        query_mid = tf.get("model_id")
        if not query_path or not query_mid:
            return None
        ti: Dict[str, object] = {"query_path": query_path, "query_model_id": query_mid}
        if reference_pdb_id:
            ti["reference_pdb_id"] = reference_pdb_id.upper()
            ref_label = ref_label or reference_pdb_id.upper()
        if reference_path:
            ti["reference_path"] = reference_path
        if reference_model_id:
            ti["reference_model_id"] = str(reference_model_id)
            ref_label = ref_label or f"#{reference_model_id}"
        if "reference_pdb_id" not in ti and "reference_path" not in ti:
            return None                                  # need a resolvable reference
        ti["ref_label"] = ref_label or "reference"
        which = "guided fold" if use_guided else "construct fold"
        return {
            "tool":         "structural_align",
            "tool_inputs":  ti,
            "user_input":   f"[Workbench] structural align {which} → {ti['ref_label']} "
                            f"(US-align, sequence-independent, LOCAL-ONLY)",
            "confidence":   "high",                      # fast + deterministic → no confirm gate
            "refresh":      "structural_align",
            "_align_ukey":  self._cur_cd_ukey(),
            "_validate_guided": bool(use_guided),        # adoption-validation framing in apply
        }

    def _on_align_clicked(self) -> None:
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Align to PDB is for de-novo constructs — Add sequence → Fold "
                                 "construct first.")
            return
        tab = self._cur_tab()
        cd = tab.design if tab else None
        tf = cd.template_fold if cd else {}
        if not tf.get("model_id") or not (tf.get("cif_path") or tf.get("pdb_path")):
            self._status.setText("Fold the construct first (Fold ▾ → Fold construct) — Align needs "
                                 "the construct's fold on disk.")
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Align to PDB",
            "Reference — a 4-char PDB id (downloaded) or a loaded model id (e.g. #3):")
        ref = (text or "").strip()
        if not ok or not ref:
            return
        m = re.match(r"^#?(\d+(?:\.\d+)*)$", ref)
        if m:
            # LOADED model → save it to a temp PDB so US-align can read it; reuse it for the overlay
            mid = m.group(1)
            tmp = os.path.join(tempfile.gettempdir(), f"usalign_ref_{mid.replace('.', '_')}.pdb")
            try:
                self._c._run(f'save "{Path(tmp).as_posix()}" models #{mid}')
            except Exception as exc:
                self._status.setText(f"Could not save model #{mid} for alignment: {exc}")
                return
            if not os.path.isfile(tmp):
                self._status.setText(f"Saving model #{mid} produced no file — is #{mid} open?")
                return
            spec = self.structural_align_launch_spec(reference_path=tmp, reference_model_id=mid,
                                                     ref_label=f"#{mid}")
        else:
            if not re.match(r"^[A-Za-z0-9]{4}$", ref):
                self._status.setText("Enter a 4-character PDB id (e.g. 1MBN) or a loaded model "
                                     "(e.g. #3).")
                return
            spec = self.structural_align_launch_spec(reference_pdb_id=ref)
        if spec is not None:
            self._status.setText(f"Aligning the construct fold to {spec['tool_inputs']['ref_label']} "
                                 f"via US-align (sequence-independent)…")
            self.launchRequested.emit(spec)

    @staticmethod
    def _structural_align_from_result(result: dict) -> dict:
        """The structural_align step data from the executed pipeline result."""
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") == "structural_align":
                return step.get("data") or {}
        return {}

    def apply_structural_align_result(self, spec: dict, result: dict) -> None:
        """Consume the executed structural-alignment result: store it on the construct's
        `structural_align` slot, persist, and report an HONEST readout (TM>0.5 = shared fold;
        lower = not structurally similar — no overclaiming). The 3D overlay (view matrix on the
        fold model + the reference) already ran live in the router. UI thread."""
        data = self._structural_align_from_result(result)
        if not data:
            self._status.setText("Structural alignment produced no result.")
            return
        ukey = (spec or {}).get("_align_ukey")
        cd = self._design.chains.get(ukey) if (self._design and ukey) else None
        if cd is None:
            tab = self._cur_tab()
            cd = tab.design if tab else None
        if cd is None:
            return
        cd.structural_align = data            # fresh align REPLACES the slot → reference shown (no `hidden`)
        self._sync_align_ref_toggle()         # reflect the fresh reference in the per-cd toggle
        validating = bool((spec or {}).get("_validate_guided"))
        if validating:
            # Mirror the adoption TM into the assist slot so the assist readout can cite it.
            cd.template_assist = {**(cd.template_assist or {}),
                                  "tm_adopt": data.get("tm_ref"),
                                  "tm_adopt_query": data.get("tm_query"),
                                  "adopted": bool(data.get("shared_fold")),
                                  "adopt_ref_label": data.get("ref_label")}
        self._persist()
        tier = ("shared fold ✓" if data.get("shared_fold")
                else "NOT structurally similar (low TM)")
        if validating:
            adopt = ("ADOPTED the template ✓" if data.get("shared_fold")
                     else "did NOT adopt the template (low TM)")
            self._status.setText(
                f"Guided-fold validation vs {data.get('ref_label')}: TM-score {data.get('tm_ref')} "
                f"(ref-norm) / {data.get('tm_query')} (query-norm), RMSD {data.get('rmsd')} Å over "
                f"{data.get('n_aligned')} residues — the guided fold {adopt}. TM>0.5 = adopted.")
        else:
            self._status.setText(
                f"Aligned construct fold → {data.get('ref_label')}: TM-score {data.get('tm_ref')} "
                f"(reference-normalized) / {data.get('tm_query')} (query-normalized), RMSD "
                f"{data.get('rmsd')} Å over {data.get('n_aligned')} residues — {tier}. "
                f"TM>0.5 = shared fold; lower = not structurally similar.")

    # ── Stage 4b: per-model fold visibility (active-row coupling + global toggle) ─────
    def _fold_models(self, design) -> Dict[str, str]:
        """{variant_id: predicted model_id} for variants in *design* that have a fold."""
        out: Dict[str, str] = {}
        for cd in design.chains.values():
            for v in cd.variants:
                fld = v.results.fold
                if fld and fld.get("model_id"):
                    out[v.id] = str(fld["model_id"])
        return out

    def _wt_ref_model_ids(self) -> List[str]:
        """Model ids of the seed-pinned WT REFERENCE folds (the deviation comparison basis),
        across all chains/combos — distinct from the loaded crystal and the variant folds.
        Hidden by fold_visibility_commands (computation artifact, not the displayed result)."""
        ids: List[str] = []
        if self._design is None:
            return ids
        for cd in self._design.chains.values():
            for ref in (cd.wt_refs or {}).values():
                mid = (ref or {}).get("model_id")
                if mid and str(mid) not in ids:
                    ids.append(str(mid))
        return ids

    def fold_visibility_commands(self, tab: _ChainDesignTab) -> List[str]:
        """Per-model 3D visibility coupled to the active row — the clean ≤2 overlay. Two
        INDEPENDENT toggles: show the WT REFERENCE (the loaded model) and/or the ACTIVE
        variant's FOLD (others always hidden). The global 'Hide folds' button force-hides
        ALL folds (overrides the Fold toggle). Re-selecting an already-folded variant
        RE-SHOWS its fold here (`show #mid`) — the whole point of the HIDE design (switch-back
        = show, not re-fold). Pure → the single source for the push, live-verify, and tests."""
        if self._design is None:
            return []
        cmds: List[str] = []
        # The "Template/Reference" toggle shows/hides the loaded crystal. A DE-NOVO construct has
        # no crystal (model_id is synthetic) — emitting `show #denovo-…` would error in ChimeraX,
        # so skip it; the construct's fold is its own displayed structure (members point at it).
        if self._design.source != "sequence":
            ref = str(self._design.model_id)
            cmds.append(f"show #{ref} models" if self._show_ref_cb.isChecked()
                        else f"hide #{ref} models")
        # WT REFERENCE FOLDS (the deviation's seed-pinned comparison basis) are a computation
        # artifact, NOT the displayed result — the deviation is read off the variant fold's
        # floor-gated colouring. _fold_wt_reference opens + overlays them; hide them here so
        # they don't clutter/occlude the variant fold (they previously persisted untoggleable).
        for wt_mid in self._wt_ref_model_ids():
            cmds.append(f"hide #{wt_mid} models")
        hide_all = self._fold_vis_btn.isChecked() or not self._show_fold_cb.isChecked()
        # DE-NOVO construct GUIDED fold(s): an analysis OVERLAY on the construct's base (unguided)
        # structure — toggle it with Hide-folds / the Fold toggle so it is hide-able and replaced
        # on re-fold, not a stuck untoggleable accumulation. (Computed before the no-variants early
        # return below, since a de-novo construct typically has a guided fold but no variants.)
        for gmid in sorted({str((cd.guided_fold or {}).get("model_id"))
                            for cd in self._design.chains.values()
                            if (cd.guided_fold or {}).get("model_id")}):
            cmds.append(f"hide #{gmid} models" if hide_all else f"show #{gmid} models")
        # ALIGNMENT REFERENCES (US-align "Align to PDB"): the chosen PDB the construct's fold was
        # overlaid onto, opened into the scene. Brought under THIS single authority so a color/
        # overlay push can't re-show a toggle-hidden reference (the color-only invariant extended
        # to the alignment case). Effective visibility = (global "Hide alignment references" OFF)
        # AND (this cd's per-reference toggle ON, persisted as `structural_align["hidden"]`).
        cmds += self._align_ref_visibility_commands()
        models = self._fold_models(self._design)
        if not models:
            return cmds
        active = tab.active_row_id
        for vid, mid in models.items():
            show = (not hide_all) and (vid == active)    # only the ACTIVE variant's fold shows
            cmds.append(f"show #{mid} models" if show else f"hide #{mid} models")
        return cmds

    def _on_fold_visibility_toggled(self, checked: bool) -> None:
        self._fold_vis_btn.setText("Show folds" if checked else "Hide folds")
        tab = self._cur_tab()
        if tab is not None:
            self._push_3d_color(tab)

    def _on_overlay_toggle(self, _checked: bool = False) -> None:
        """Fold / Reference overlay toggles changed → re-push the visibility coupling."""
        tab = self._cur_tab()
        if tab is not None:
            self._push_3d_color(tab)

    def _align_ref_visibility_commands(self) -> List[str]:
        """show/hide for each cd's US-align reference under the SAME authority as folds. Pure →
        single source for the push, live-verify, and tests. Effective visibility per reference =
        (global "Hide alignment references" OFF) AND (this cd's persisted toggle ON). De-novo only
        (a crystal design has no construct alignment reference)."""
        if self._design is None:
            return []
        global_hide = self._align_ref_vis_btn.isChecked()
        cmds: List[str] = []
        for cd in self._design.chains.values():
            sa = cd.structural_align or {}
            ref = sa.get("reference_model_id")
            if not ref:
                continue
            show = (not global_hide) and (not sa.get("hidden"))
            cmds.append(f"show #{ref} models" if show else f"hide #{ref} models")
        return cmds

    def _on_align_ref_visibility_toggled(self, checked: bool) -> None:
        """GLOBAL 'Hide alignment references' toggled → re-push (force-hides ALL refs when checked)."""
        self._align_ref_vis_btn.setText("Show alignment references" if checked
                                        else "Hide alignment references")
        tab = self._cur_tab()
        if tab is not None:
            self._push_3d_color(tab)

    def _on_align_ref_overlay_toggle(self, checked: bool = False) -> None:
        """PER-CD 'Aligned reference' toggled → persist the ACTIVE cd's hidden flag, re-push."""
        tab = self._cur_tab()
        cd = tab.design if tab else None
        if cd is not None and cd.structural_align:
            cd.structural_align["hidden"] = (not checked)
            self._persist()
            self._push_3d_color(tab)

    def _sync_align_ref_toggle(self) -> None:
        """Reflect the ACTIVE cd's persisted aligned-reference state in the per-cd toggle, and
        enable it ONLY when the active cd actually has an alignment reference. Signals blocked so
        syncing the checkmark never re-triggers a push."""
        cb = getattr(self, "_show_align_ref_cb", None)
        if cb is None:
            return
        tab = self._cur_tab()
        cd = tab.design if tab else None
        sa = (cd.structural_align if cd else None) or {}
        has_ref = bool(sa.get("reference_model_id"))
        cb.blockSignals(True)
        cb.setEnabled(has_ref)
        cb.setChecked(has_ref and not sa.get("hidden"))
        cb.blockSignals(False)

    def tile_commands(self) -> List[str]:
        """The commands to lay the WT reference + every variant fold out SIDE-BY-SIDE:
        `show` each target then `tile` THOSE SPECIFIC ids — never bare `tile`, which would
        drag in any hidden/unrelated models. [] when fewer than 2 targets exist. Pure →
        the single source for the push, live-verify, and tests."""
        if self._design is None:
            return []
        fold_ids = list(self._fold_models(self._design).values())
        specs: List[str] = []
        for m in [str(self._design.model_id)] + fold_ids:   # reference first, then folds
            if m and m not in specs:
                specs.append(m)
        if len(specs) < 2:
            return []
        return ([f"show #{m} models" for m in specs]
                + ["tile " + " ".join(f"#{m}" for m in specs)])

    def _on_tile_clicked(self) -> None:
        cmds = self.tile_commands()
        if not cmds:
            self._status.setText("Tile needs ≥2 models (the reference + at least one folded variant).")
            return
        self._run_commands_bg(cmds)
        self._tiled = True               # next row-select restores superposition (item 4)
        n = cmds[-1].count("#")
        self._status.setText(f"Tiled {n} models side-by-side (superposition broken — select a "
                             f"variant to snap back to the overlay).")

    # ── result badges + expandable detail ───────────────────────────────────────────
    @staticmethod
    def _badge_for(v) -> str:
        """Compact inline badge for a variant's results (empty when none)."""
        parts: List[str] = []
        stab = v.results.stability
        if stab and stab.get("sum_ddg") is not None:
            d = stab["sum_ddg"]
            parts.append(f"ddG {d:+.1f}{'▲' if d > 0 else '▼'}")
        sol = v.results.solubility
        if sol and sol.get("delta") is not None:
            d = sol["delta"]
            parts.append(f"sol {d:+.2f}{'▲' if d > 0 else '▼'}")
        fold = v.results.fold
        if fold:
            if fold.get("mean_plddt") is not None:
                parts.append(f"pLDDT {fold['mean_plddt']:.0f}")
            if fold.get("iptm") is not None:        # multimer interface confidence (Boltz)
                parts.append(f"ipTM {fold['iptm']:.2f}")
            if fold.get("remote_msa"):              # PROVENANCE: this fold LEFT local-only (ColabFold)
                parts.append("⚠ remote-MSA")
        return " · ".join(parts)

    def _find_variant(self, variant_id: str):
        if self._design is None:
            return None
        for cd in self._design.chains.values():
            v = cd.get_variant(variant_id)
            if v is not None:
                return cd, v
        return None

    def _rerender_results(self, cd: ChainDesign, v) -> None:
        """Refresh a variant's badge in its tab and re-apply coloring (ddG mode reflects
        a fresh stability result on all copies)."""
        tab = self._focus_tab_for_design(cd)
        if tab is None:
            return
        badge = self._badge_for(v)
        if badge:
            tab.badges[v.id] = badge
        else:
            tab.badges.pop(v.id, None)
        tab._mark_active_header()                # repaint header labels (badge text)
        self._apply_color_to(tab)
        # Push under result colour modes (re-map ddG/pLDDT) AND always couple a fresh fold
        # model to the active row (show/hide) — _push_3d_color no-ops when nothing to do.
        self._push_3d_color(tab)

    # (The per-mutation result-detail popup was REMOVED 2026-06-17 — the modal had no purpose
    # with the detail-display PARKED, and it mis-fired on selection. A future increment will
    # build a proper detail surface; until then a header click only ever SELECTS.)

    # exposed for tests / live-verify: the exact color commands the active row pushes
    def color_commands_for(self, tab: _ChainDesignTab):
        """The exact 3D color commands the active row pushes under the current mode —
        single source for the 3D push, the live-verify, and tests. Empty for OFF/no-data."""
        if self._mode_key == _RESULT_DDG_MODE:
            ddg_map = self._active_ddg_map(tab)
            if not ddg_map:
                return []                                # no stability result → nothing to paint
            resnums = [c.resnum for c in tab.active_row_cells()
                       if not c.is_gap and c.resnum is not None]
            return build_color_commands_by_resnum(
                resnums, lambda rn: ddg_color(ddg_map.get(rn)), tab.design.members)
        if self._mode_key == _RESULT_PLDDT_MODE:
            # pLDDT lives on the PREDICTED model, not the shared backbone — colour that
            # model by its B-factor (where the engine stored pLDDT) with the native palette.
            # Numbering-agnostic, and the banding matches the panel's plddt_color (sync).
            v = tab.design.get_variant(tab.active_row_id)
            fold = v.results.fold if v is not None else None
            if fold is None and v is None:               # T of a de-novo construct → its own fold
                fold = tab.design.template_fold or None
            if not fold or not fold.get("model_id"):
                return []                                # active row has no fold → nothing
            mid = fold["model_id"]
            # Visibility is owned SOLELY by fold_visibility_commands (which _push_3d_color
            # runs first); colouring a hidden model is harmless and it shows when re-shown.
            # Emitting `show #mid` here would re-show a model the "Variant fold"/"Hide folds"
            # toggle just hid (the toggles run before this) → the toggle would silently no-op.
            # target acs → the pLDDT colour survives a later NL representation change (spheres
            # reveal coloured atoms, not default-coloured ones).
            return [f"color byattribute bfactor #{mid} palette alphafold target acs"]
        if self._mode_key == _RESULT_DEVIATION_MODE:
            # The disruption (dRMSD) lives on the PREDICTED variant model's real atoms (like
            # pLDDT), NOT the shared crystal backbone — colour #mid per chain in its OWN
            # numbering, 3-tier via the SAME `combined_disruption_color` the panel uses (sync).
            block = self._active_deviation(tab)
            if not block or not block.get("variant_model_id"):
                return []
            ddm = block.get("ddm") or {}
            floor_ddm = block.get("floor_ddm") or {}
            lddt = block.get("lddt") or {}
            floor_lddt = block.get("floor_lddt") or {}
            mid = block["variant_model_id"]
            multichain = bool(block.get("multichain"))
            # the per-residue maps are keyed by REFERENCE-fold resnum (the router re-keyed the
            # variant onto the WT numbering for an indel). The model #mid is numbered in the
            # VARIANT fold's OWN order — so remap ref→variant here before painting, or an
            # insertion's downstream residues get the wrong colour. Inserted residues have no
            # reference counterpart → absent from the remap → neutral. None map → identity.
            fmap = block.get("fold_column_map")
            if fmap and not multichain:
                ref_to_var = {int(r): int(j) for j, r in fmap.items()}
                remap = lambda d: {str(ref_to_var[int(k)]): val for k, val in d.items()
                                   if int(k) in ref_to_var}
                ddm, floor_ddm = remap(ddm), remap(floor_ddm)
                lddt, floor_lddt = remap(lddt), remap(floor_lddt)
            per_chain: Dict[str, List[int]] = {}
            if multichain:
                for k in ddm:
                    ch, rn = k.split(":", 1)
                    per_chain.setdefault(ch, []).append(int(rn))
            else:
                ch = block.get("variant_chain", "A")
                per_chain[ch] = [int(k) for k in ddm]
            for c in per_chain:
                per_chain[c].sort()

            def _val(chain: str, rn: int) -> Optional[str]:
                k = f"{chain}:{rn}" if multichain else str(rn)
                return combined_disruption_color(ddm.get(k), floor_ddm.get(k, _DDM_FLOOR_MIN_A),
                                                 lddt.get(k), floor_lddt.get(k, _LDDT_NEUTRAL_CAP))

            # Visibility owned by fold_visibility_commands (see the pLDDT branch note) — do
            # NOT re-show #mid here or the "Variant fold"/"Hide folds" toggle would no-op.
            return build_model_color_commands(mid, per_chain, _val)
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
