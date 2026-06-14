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
                           build_color_commands, build_design_session, column_tracks,
                           filter_new_mpnn_variants, group_scan_suggestions,
                           import_mpnn_designs, suggestion_color)

_COLS = 30                                  # residues per wrapped block
_SUGGEST_ROW = "__suggest__"                # sentinel row id for the inline Suggest track
_RESNUM_ROLE = QtCore.Qt.UserRole           # cell → template column index
_ROW_ROLE = QtCore.Qt.UserRole + 1          # cell → row id ("T"/"V1"/… / _SUGGEST_ROW / None)
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
    variant, [Suggest], Consensus, Conservation. Tracks the ACTIVE row (drives 3D
    coloring) and the current color mode. A cell click emits (row_id, template-column);
    a click on the sparse inline Suggest track emits (_SUGGEST_ROW, col)."""

    cellClicked2 = QtCore.Signal(object, int, bool)   # (row_id, template col, ctrl-held)

    def __init__(self, design: ChainDesign, suggestions: Optional[Dict[int, List[dict]]] = None):
        super().__init__()
        self.design = design
        self.suggestions: Dict[int, List[dict]] = dict(suggestions or {})  # col -> ranked cands
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

        # The inline Suggest track appears ONLY when a scan produced candidates for this
        # chain (sparse by construction — never implies a suggestion where none was run).
        has_sugg = bool(self.suggestions)
        sugg_label = ["Suggest"] if has_sugg else []
        sugg_rid   = [_SUGGEST_ROW] if has_sugg else []

        # row identity, in table-row order: ruler, T, variants…, [Suggest], consensus, conservation
        self._row_ids = [None, "T"] + [vv.id for vv in design.variants] + sugg_rid + [None, None]
        labels = ["#", f"T ({design.rep_chain})"] \
            + [vv.id for vv in design.variants] + sugg_label + ["Consensus", "Conservation"]
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
        # base labels by row id (mirrors _build's layout, incl. the optional Suggest row)
        base_by_rid = {None: "", "T": f"T ({self.design.rep_chain})",
                       _SUGGEST_ROW: "Suggest"}
        for vv in self.design.variants:
            base_by_rid[vv.id] = vv.id
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
            tab.cellClicked2.connect(
                lambda rid, col, ctrl, t=tab: self._on_cell(t, rid, col, to_scan=ctrl))
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
            self._push_3d_color(tab)
        elif row_id is not None:                            # a variant row
            tab.set_active_row(row_id)
            self._edit_target = (row_id, col)
            resnum = tab.design.resnum_for_col(col)
            wt = tab.design.template_cells[col].aa if 0 <= col < len(tab.design.template_cells) else "?"
            self._status.setText(f"Edit target: {row_id} col {col} (residue {resnum}, T={wt}). "
                                 f"Pick an aa and Apply.  (Ctrl+click builds the scan set.)")
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
