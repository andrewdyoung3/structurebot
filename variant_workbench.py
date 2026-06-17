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

from color_modes import (all_modes, ddg_color, deviation_color, get_mode,
                         plddt_color)
from seq_library import build_numbering_header_content
from variant_model import (AlignedCell, ChainDesign, DesignSession,
                           build_color_commands, build_color_commands_by_resnum,
                           build_model_color_commands,
                           build_design_session, column_tracks,
                           filter_new_mpnn_variants, group_scan_suggestions,
                           fold_summary, import_mpnn_designs, stability_summary,
                           suggestion_color)

_COLS = 30                                  # residues per wrapped block
_SUGGEST_ROW = "__suggest__"                # sentinel row id for the inline Suggest track
_RESULT_DDG_MODE = "result:ddg"             # S4a result-backed color mode (per-residue ddG)
_RESULT_PLDDT_MODE = "result:plddt"         # S4b result-backed color mode (per-residue pLDDT)
_RESULT_DEVIATION_MODE = "result:deviation" # S4c floor-gated variant-vs-WT Cα deviation
_DEVIATION_FLOOR_MIN_A = 0.25               # mirrors ToolRouter._DEVIATION_FLOOR_MIN_A (gate floor)
_FOLD_ENGINES = ("esmfold", "boltz")        # S4b engine picker order (Boltz lands its own stage)
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


# ── one unique-chain tab: T + variants in wrapped column blocks ────────────────────

class _ChainDesignTab(QtWidgets.QScrollArea):
    """Wrapped CLC-style view of one ChainDesign. Rows per block: Ruler, T, each
    variant, [Suggest], Consensus, Conservation. Tracks the ACTIVE row (drives 3D
    coloring) and the current color mode. A cell click emits (row_id, template-column);
    a click on the sparse inline Suggest track emits (_SUGGEST_ROW, col)."""

    cellClicked2 = QtCore.Signal(object, int, bool)   # (row_id, template col, ctrl-held)
    rowHeaderSelected = QtCore.Signal(object)         # row_id (header click → SELECT active row)
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

        ruler = build_numbering_header_content(
            [c.resnum for c in design.template_cells if c.resnum is not None], interval=10)
        ruler = (ruler + " " * n)[:n]               # guard length (gap cells)
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

    def _variant_label(self, vid: str) -> str:
        """Variant row header: the id plus its inline result badge (S4a), if any."""
        badge = self.badges.get(vid)
        return f"{vid}  {badge}" if badge else vid

    def eventFilter(self, obj, event) -> bool:
        """Row-header click router. A left-click ANYWHERE on a row header → SELECT that variant
        as the active row — the only header action (always, never a modal). The per-mutation
        result-DETAIL display is PARKED (the badge-region approach was dead), so there is no
        detail gesture here for now. Returns False (never consumes — normal header painting
        proceeds)."""
        block = self._vp_to_block.get(obj)
        if block is None:
            return False
        if event.type() != QtCore.QEvent.Type.MouseButtonPress:
            return False
        if event.button() != QtCore.Qt.LeftButton:
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
        self.rowHeaderSelected.emit(rid)                 # click → SELECT (always)
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
        # Stage 3a: pull the latest cached tool results into the panel (import = capture).
        self._import_btn = QtWidgets.QPushButton("Import MPNN designs")
        self._import_btn.clicked.connect(self._import_mpnn)
        bar.addWidget(self._import_btn)
        self._sugg_btn = QtWidgets.QPushButton("Load scan suggestions")
        self._sugg_btn.clicked.connect(self._load_suggestions)
        bar.addWidget(self._sugg_btn)
        bar.addSpacing(12)
        # Stage 3b: launch tools FROM the panel through the engine spine (confirm-gate +
        # real subprocess). Scope = the scan set (clicked columns); empty → whole chain.
        self._scan_btn = QtWidgets.QPushButton("Scan…")
        self._scan_btn.setToolTip("Mutation-scan the scan set (Ctrl+click residues to build "
                                  "it; whole chain if empty) through the tool spine.")
        self._scan_btn.clicked.connect(self._on_scan_clicked)
        bar.addWidget(self._scan_btn)
        self._mpnn_run_btn = QtWidgets.QPushButton("Run ProteinMPNN…")
        self._mpnn_run_btn.setToolTip("Redesign the chain (the scan set via Ctrl+click, or "
                                      "whole chain) with ProteinMPNN through the tool spine.")
        self._mpnn_run_btn.clicked.connect(self._on_mpnn_clicked)
        bar.addWidget(self._mpnn_run_btn)
        self._scan_set_lbl = QtWidgets.QLabel("scan set: 0")
        self._scan_set_lbl.setStyleSheet("color:#888;")
        bar.addWidget(self._scan_set_lbl)
        self._clear_scan_btn = QtWidgets.QPushButton("Clear")
        self._clear_scan_btn.setToolTip("Clear the scan set.")
        self._clear_scan_btn.clicked.connect(self._clear_scan_set)
        bar.addWidget(self._clear_scan_btn)
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
        # Stage 4b: fold the ACTIVE variant through an engine (user picks; ESMFold local),
        # opening a real pLDDT-coloured model matchmaker-overlaid on the template. Gated.
        self._fold_btn = QtWidgets.QPushButton("Fold…")
        self._fold_btn.setToolTip("Fold the ACTIVE variant (you pick the engine; ESMFold runs "
                                  "LOCAL-ONLY) → a pLDDT-coloured model overlaid on the template.")
        self._fold_btn.clicked.connect(self._on_fold_clicked)
        bar.addWidget(self._fold_btn)
        self._fold_vis_btn = QtWidgets.QPushButton("Hide folds")
        self._fold_vis_btn.setToolTip("Show/hide ALL predicted fold models in 3D.")
        self._fold_vis_btn.setCheckable(True)
        self._fold_vis_btn.toggled.connect(self._on_fold_visibility_toggled)
        bar.addWidget(self._fold_vis_btn)
        # Escape hatch: lay the variant folds + the WT reference out SIDE-BY-SIDE (not
        # overlaid) in the one 3D scene via ChimeraX `tile`. Targets the specific fold models
        # (not bare `tile`, which would drag in any hidden models). Shared camera; the models
        # leave superposition (accepted — this is "lay them out", not "overlay").
        self._tile_btn = QtWidgets.QPushButton("Tile folds")
        self._tile_btn.setToolTip("Lay the variant folds + reference out side-by-side (not "
                                  "overlaid). Select a variant afterwards to return to the "
                                  "overlay. Needs ≥2 models.")
        self._tile_btn.clicked.connect(self._on_tile_clicked)
        bar.addWidget(self._tile_btn)
        # Independent overlay toggles (distinct from the global "Hide folds"): show just the
        # active variant's FOLD, just the WT REFERENCE, or both (default).
        self._show_fold_cb = QtWidgets.QCheckBox("Fold")
        self._show_fold_cb.setChecked(True)
        self._show_fold_cb.setToolTip("Show the ACTIVE variant's predicted fold in the overlay.")
        self._show_fold_cb.toggled.connect(self._on_overlay_toggle)
        bar.addWidget(self._show_fold_cb)
        self._show_ref_cb = QtWidgets.QCheckBox("Reference")
        self._show_ref_cb.setChecked(True)
        self._show_ref_cb.setToolTip("Show the WT reference structure in the overlay.")
        self._show_ref_cb.toggled.connect(self._on_overlay_toggle)
        bar.addWidget(self._show_ref_cb)
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
        """Right-click cell menu for a VARIANT residue: substitute (any of the 20 aa) or
        revert to the WT residue. The SAME `edit_variant` substitution as the toolbar Apply —
        just reachable directly on the residue. (Deletion/indels are a later increment: they
        shift residue numbering, which the resnum-keyed S4c deviation can't absorb yet.)"""
        v = tab.design.get_variant(vid)
        if v is None or not (0 <= col < len(tab.design.template_cells)):
            return
        tab.set_active_row(vid)
        self._edit_target = (vid, col)
        resnum = tab.design.resnum_for_col(col)
        wt = tab.design.template_cells[col].aa
        cur = v.cells[col].aa if col < len(v.cells) else None

        menu = QtWidgets.QMenu(self)
        header = menu.addAction(f"{vid} · residue {resnum} (WT {wt}, now {cur})")
        header.setEnabled(False)
        menu.addSeparator()
        sub = menu.addMenu("Substitute →")
        for aa in _AA_ORDER:
            act = sub.addAction(f"{aa}  (WT)" if aa == wt else aa)
            if aa == cur:
                act.setCheckable(True); act.setChecked(True)
            act.triggered.connect(lambda _checked=False, a=aa: self._do_substitute(tab, vid, col, a))
        if wt is not None and cur != wt:
            menu.addAction(f"Revert to WT ({wt})",
                           lambda: self._do_substitute(tab, vid, col, wt))
        menu.exec(global_pos)

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
            wt = tab.design.template_cells[col].aa if 0 <= col < len(tab.design.template_cells) else "?"
            self._status.setText(f"Edit target: {row_id} col {col} (residue {resnum}, T={wt}). "
                                 f"Pick an aa and Apply.  (Ctrl+click builds the scan set.)")
            self._apply_color_to(tab)                       # panel result-coloring FOLLOWS active
            self._push_3d_color(tab)

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
        """{author_resnum: pLDDT} for the active variant's fold result (empty if none/T)."""
        v = tab.design.get_variant(tab.active_row_id)
        if v is None or not v.results.fold:
            return {}
        return {int(rn): float(p) for rn, p in (v.results.fold.get("plddt") or {}).items()
                if p is not None}

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
        """{author_resnum: hex} for the active variant's floor-gated deviation, panel side.
        The fold numbers residues 1..N over the ungapped sequence, so the rep chain's
        deviation keys map POSITIONALLY onto the active row's ordered author resnums (the
        same positional contract `fold_summary` uses for pLDDT). Floor-gated via
        `deviation_color` (the ONE source shared with the 3D push)."""
        block = self._active_deviation(tab)
        if not block:
            return {}
        dev, floor = block.get("deviation") or {}, block.get("floor") or {}
        keys = self._dev_chain_keys(dev, bool(block.get("multichain")), tab.design.rep_chain)
        author = [c.resnum for c in tab.active_row_cells()
                  if not c.is_gap and c.resnum is not None]
        out: Dict[int, str] = {}
        for i, k in enumerate(keys):
            if i >= len(author):
                break
            hexc = deviation_color(dev[k], floor.get(k, _DEVIATION_FLOOR_MIN_A))
            if hexc:
                out[author[i]] = hexc
        return out

    def _apply_color_to(self, tab: _ChainDesignTab) -> None:
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
        folds (non-destructive: we do not know the pre-overlay coloring to restore)."""
        cmds = self.fold_visibility_commands(tab) + self.color_commands_for(tab)
        self._run_commands_bg(cmds)

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
        self._scan_set_lbl.setText(f"scan set: {len(self._scan_cols)}")

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
        ti: Dict[str, object] = {
            "model_id":   self._design.model_id,
            "chain":      cd.rep_chain,
            "engine":     engine,
            "open_model": True,
            "local_only": True,                   # LOCAL-ONLY: no remote Atlas/MSA server
            "compare_to": self._design.model_id,  # matchmaker onto the loaded WT (oligomer)
        }
        if engine == "boltz" and assembly:
            # the full homo-oligomer: every copy chain folded with the variant's sequence.
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
            wt_chains = [{"id": c, "sequence": t_seq} for (_m, c) in cd.members]
            variant_chain = cd.rep_chain
        else:
            variant_chain = "A" if engine == "esmfold" else cd.rep_chain
            wt_chains = [{"id": variant_chain, "sequence": t_seq}]
        ti: Dict[str, object] = {
            "variant_model_id": fold["model_id"],
            "engine":           engine,
            "target":           target,
            "multichain":       multichain,
            "variant_chain":    variant_chain,
            "wt_chains":        wt_chains,
            "compare_to":       self._design.model_id,   # reference matchmaker onto crystal WT
            "model_id":         self._design.model_id,
            "wt_ref":           cd.wt_refs.get(combo),    # cached reference (skip folding) or None
            "local_only":       True,
        }
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
            cd.wt_refs[f"{data.get('engine')}:{data.get('target')}"] = wt_ref
        self._persist()
        self._select_result_mode(_RESULT_DEVIATION_MODE)   # AUTO-SURFACE the deviation mode
        self._rerender_results(cd, v)
        ar = data.get("anchor_residual_rmsd")
        self._status.setText(
            f"{v.id} deviation vs WT ({data.get('engine')}:{data.get('target')}): "
            f"{data.get('n_cleared_floor')}/{data.get('n_residues')} residues clear the "
            f"noise floor; max {data.get('max_deviation')} Å, anchor residual "
            f"{ar if ar is not None else '?'} Å (floor-gated colour, auto).")

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
        ref = str(self._design.model_id)                 # the WT reference (loaded crystal)
        cmds.append(f"show #{ref} models" if self._show_ref_cb.isChecked()
                    else f"hide #{ref} models")
        models = self._fold_models(self._design)
        if not models:
            return cmds
        hide_all = self._fold_vis_btn.isChecked() or not self._show_fold_cb.isChecked()
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
        n = cmds[-1].count("#")
        self._status.setText(f"Tiled {n} models side-by-side (superposition broken — select a "
                             f"variant to return to the overlay).")

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
            if not fold or not fold.get("model_id"):
                return []                                # active row has no fold → nothing
            mid = fold["model_id"]
            return [f"show #{mid} models",
                    f"color byattribute bfactor #{mid} palette alphafold"]
        if self._mode_key == _RESULT_DEVIATION_MODE:
            # Deviation lives on the PREDICTED variant model's real atoms (like pLDDT),
            # NOT the shared crystal backbone — colour #mid per chain in its OWN numbering,
            # floor-gated via the SAME `deviation_color` the panel uses (panel↔3D sync).
            block = self._active_deviation(tab)
            if not block or not block.get("variant_model_id"):
                return []
            dev, floor = block.get("deviation") or {}, block.get("floor") or {}
            mid = block["variant_model_id"]
            multichain = bool(block.get("multichain"))
            per_chain: Dict[str, List[int]] = {}
            if multichain:
                for k in dev:
                    ch, rn = k.split(":", 1)
                    per_chain.setdefault(ch, []).append(int(rn))
            else:
                ch = block.get("variant_chain", "A")
                per_chain[ch] = [int(k) for k in dev]
            for c in per_chain:
                per_chain[c].sort()

            def _val(chain: str, rn: int) -> Optional[str]:
                k = f"{chain}:{rn}" if multichain else str(rn)
                return deviation_color(dev.get(k), floor.get(k, _DEVIATION_FLOOR_MIN_A))

            return [f"show #{mid} models"] + build_model_color_commands(mid, per_chain, _val)
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
