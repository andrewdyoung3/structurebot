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

from color_modes import (all_modes, cavity_compat_color, combined_disruption_color, ddg_color,
                         disulfide_compat_color, get_mode, plddt_color, proline_compat_color,
                         saltbridge_compat_color)
from disulfide_geometry import pair_chains, pair_label
from seq_library import (build_numbering_header_content,
                         build_numbering_header_with_insertions)
from variant_model import (AlignedCell, ChainDesign, DesignSession,
                           build_color_commands, build_color_commands_by_resnum,
                           build_model_color_commands, build_fold_column_map,
                           build_design_session, build_design_session_from_sequence,
                           column_tracks, DesignSession,
                           filter_new_mpnn_variants, group_scan_suggestions,
                           fold_summary, import_mpnn_designs, stability_summary,
                           merge_stability, suggestion_color, _STD_AA)

_COLS = 30                                  # residues per wrapped block
_SUGGEST_ROW = "__suggest__"                # sentinel row id for the inline Suggest track
_RESULT_DDG_MODE = "result:ddg"             # S4a result-backed color mode (per-residue ddG)
_RESULT_PLDDT_MODE = "result:plddt"         # S4b result-backed color mode (per-residue pLDDT)
_RESULT_DEVIATION_MODE = "result:deviation" # S4c floor-gated variant-vs-WT Cα deviation
_RESULT_DISULFIDE_MODE = "result:disulfide_scan"  # Mode D best-partner engineerability heatmap
_RESULT_PROLINE_MODE = "result:proline_scan"      # proline-stabilization favourability heatmap
_RESULT_CAVITY_MODE = "result:cavity_scan"        # cavity-filling best-fill heatmap (lining→gold)
_RESULT_SALTBRIDGE_MODE = "result:saltbridge_scan"  # salt-bridge best charge-pair heatmap (pale→blue)
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


# ── colored metric-badge chips in the variant row headers ─────────────────────────

class _BadgeHeaderView(QtWidgets.QHeaderView):
    """Vertical header that renders a variant's result badge as colored metric CHIPS
    after the row name (the Qt equivalent of a reusable MetricBadge).

    The badge STRING (`tab.badges[rid]`, a ` · `-joined summary) stays the single data
    source-of-truth — tests and the session export read it verbatim; this view only
    PARSES and PAINTS it. The row name, active-row marker (► + bold), selection styling
    and section geometry all still come from the base painter / header items; the chips
    are drawn on top, and `sizeHint` widens the header so they fit."""

    _GAP = 8                    # px between the row name and the first chip
    _CHIP_GAP = 4               # px between adjacent chips
    _CHIP_HPAD = 5              # horizontal padding inside a chip

    # kind → (background, foreground). Green = beneficial, red = detrimental, gray =
    # neutral metric (pLDDT / ipTM / provenance), amber = a provenance WARNING.
    _CHIP_COLORS = {
        "good":    (QtGui.QColor("#2e7d32"), QtGui.QColor("white")),
        "bad":     (QtGui.QColor("#c62828"), QtGui.QColor("white")),
        "neutral": (QtGui.QColor("#9e9e9e"), QtGui.QColor("white")),
        "warn":    (QtGui.QColor("#f9a825"), QtGui.QColor("black")),
    }

    def __init__(self, tab: "_ChainDesignTab"):
        super().__init__(QtCore.Qt.Vertical, tab)
        self._tab = tab

    @staticmethod
    def _chip_kind(seg: str) -> str:
        """Classify one badge segment. ddG stabilizes when NEGATIVE (green); solubility
        helps when its delta is POSITIVE (green). The signed number is always present
        ({:+} formatting), so the sign char disambiguates direction reliably."""
        s = seg.strip()
        if s.startswith("ddG"):
            return "good" if "-" in s else "bad"
        if s.startswith("sol"):
            return "good" if "+" in s else "bad"
        if "remote-MSA" in s or s.startswith("⚠"):
            return "warn"
        return "neutral"        # pLDDT / ipTM / SS-bond provenance

    def _chip_font(self) -> QtGui.QFont:
        f = QtGui.QFont(self.font())
        f.setPointSizeF(max(6.0, self.font().pointSizeF() - 1.0))
        return f

    def _badge_for_section(self, logicalIndex: int) -> Optional[str]:
        rids = getattr(self._tab, "_row_ids", [])
        rid = rids[logicalIndex] if 0 <= logicalIndex < len(rids) else None
        if rid in (None, "T", _SUGGEST_ROW):
            return None
        return self._tab.badges.get(rid)

    def _margin(self) -> int:
        return self.style().pixelMetric(QtWidgets.QStyle.PM_HeaderMargin, None, self)

    def _name_end_x(self, rect: QtCore.QRect, logicalIndex: int) -> float:
        """X where the base-painted row name ends (left margin + name text width)."""
        model = self.model()
        text = model.headerData(logicalIndex, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole)
        text = "" if text is None else str(text)
        f = model.headerData(logicalIndex, QtCore.Qt.Vertical, QtCore.Qt.FontRole)
        fm = QtGui.QFontMetrics(f if isinstance(f, QtGui.QFont) else self.font())
        return rect.left() + self._margin() + fm.horizontalAdvance(text)

    def paintSection(self, painter: QtGui.QPainter, rect: QtCore.QRect, logicalIndex: int) -> None:
        super().paintSection(painter, rect, logicalIndex)   # name + marker + selection
        badge = self._badge_for_section(logicalIndex)
        if not badge:
            return
        chip_font = self._chip_font()
        cfm = QtGui.QFontMetrics(chip_font)
        ch = cfm.height()
        cy = rect.top() + (rect.height() - ch) / 2.0
        x = self._name_end_x(rect, logicalIndex) + self._GAP
        painter.save()
        painter.setFont(chip_font)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        for seg in (s.strip() for s in badge.split("·")):
            if not seg:
                continue
            bg, fg = self._CHIP_COLORS[self._chip_kind(seg)]
            w = cfm.horizontalAdvance(seg) + 2 * self._CHIP_HPAD
            if x + w > rect.right():
                break                       # out of header width — stop cleanly
            chip = QtCore.QRectF(x, cy, w, ch)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(chip, ch / 2.0, ch / 2.0)
            painter.setPen(fg)
            painter.drawText(chip, QtCore.Qt.AlignCenter, seg)
            x += w + self._CHIP_GAP
        painter.restore()

    def sizeHint(self) -> QtCore.QSize:
        """Widen the header so the longest (name + its chips) fits — otherwise the chips
        would be clipped by the name-only header width."""
        base = super().sizeHint()
        model = self.model()
        margin = self._margin()
        cfm = QtGui.QFontMetrics(self._chip_font())
        want = base.width()
        for i in range(self.count()):
            text = model.headerData(i, QtCore.Qt.Vertical, QtCore.Qt.DisplayRole)
            text = "" if text is None else str(text)
            f = model.headerData(i, QtCore.Qt.Vertical, QtCore.Qt.FontRole)
            nfm = QtGui.QFontMetrics(f if isinstance(f, QtGui.QFont) else self.font())
            total = 2 * margin + nfm.horizontalAdvance(text)
            badge = self._badge_for_section(i)
            if badge:
                total += self._GAP + sum(
                    cfm.horizontalAdvance(seg) + 2 * self._CHIP_HPAD + self._CHIP_GAP
                    for seg in (s.strip() for s in badge.split("·")) if seg)
            want = max(want, total)
        base.setWidth(int(want))
        return base


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
            block.setVerticalHeader(_BadgeHeaderView(self))   # paints result badges as chips
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
        vh = block.verticalHeader()
        hw = max(vh.width(), vh.sizeHint().width())   # widen for badge chips (post-fold/stability)
        h = sum(block.rowHeight(r) for r in range(block.rowCount())) + fr
        w = (sum(block.columnWidth(c) for c in range(block.columnCount())) + hw + fr)
        block.setFixedHeight(h)
        block.setFixedWidth(w)

    def _variant_label(self, vid: str) -> str:
        """Variant row-header NAME (just the id). The result badge (S4a) is no longer
        appended as text here — it is painted as colored chips by `_BadgeHeaderView`
        from `self.badges[vid]`, which stays the string source-of-truth."""
        return vid

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
            # a fresh badge (post-fold/stability) can widen the header → re-fit the block
            # so the chips aren't clipped, then repaint the header.
            self._size_block_to_content(block)
            block.verticalHeader().viewport().update()

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


# ── shared base for the persistent stabilization-strategy results tabs ─────────────────────
class _StabilizationResultsTab(QtWidgets.QWidget):
    """Shared base for the persistent stabilization-strategy results tabs — Disulfides / Proline /
    Cavity, the framework's first-class peer strategies. Factors the plumbing every strategy tab
    repeats so the third one (Cavity) doesn't clone it a third time (the consolidation debt, paid at
    the natural moment before it grows to four):
      • the intro header + the 'Clear <strategy> view (un-ghost)' control and its `set_glow_active`
        enable-state — the glow seam is SHARED (ONE `_glow_state` across all tabs; each Clear control
        tracks the one spotlight, lit only while something glows);
      • `_make_ranked_table` — the standard NoEdit / SelectRows / Single-selection / hidden-until-
        populated / cellClicked→row table every tab builds;
      • `_make_caveat` — the measured-not-promised caveat label (amber, hidden until there are rows).
    Subclasses build their OWN body on top (the Disulfides multi-section A/B/D/I/C layout vs the
    Proline/Cavity single ranked table) — the base owns the COMMON mechanics, NOT the table shape, so
    each tab's behaviour is unchanged. Persists across tab-switch + session reset (keep-and-clear: the
    subclass `reset()` empties content, the tab lives on as a sibling in `gui_app.tabs`)."""

    # Emitted when this tab gains NEW results (a populate produced rows). The gui tab-bar consumes
    # it to paint a "new / unviewed results" dot on the tab; cleared when the tab is activated.
    # Behaviour-neutral (a notification only — no analysis/data change).
    resultsArrived = QtCore.Signal()

    def _announce(self) -> None:
        """Signal that fresh results landed in this tab (drives the tab-bar new-results dot)."""
        self.resultsArrived.emit()

    def __init__(self, *, on_highlight, on_clear_glow=None, on_add_to_basket=None):
        super().__init__()
        self._on_highlight = on_highlight          # (cd, item) -> glow it in 3D (the shared seam)
        self._on_clear_glow = on_clear_glow        # ()         -> restore the normal 3D representation
        self._on_add_to_basket = on_add_to_basket  # (cd, item) -> stage the pick into the design basket
        self._outer = QtWidgets.QVBoxLayout(self)

    def _add_header(self, intro_text: str, clear_label: str, clear_tooltip: str) -> None:
        """Build the intro label + the Clear-view (un-ghost) control into the tab's outer layout.
        Subclasses call this first, then add their own body widgets to ``self._outer``."""
        intro = QtWidgets.QLabel(intro_text)
        intro.setWordWrap(True); intro.setStyleSheet("color:#666;")
        self._outer.addWidget(intro)
        self._clear_glow_btn = QtWidgets.QPushButton(clear_label)
        self._clear_glow_btn.setToolTip(clear_tooltip)
        self._clear_glow_btn.setEnabled(False)
        self._clear_glow_btn.clicked.connect(lambda: self._on_clear_glow() if self._on_clear_glow else None)
        self._outer.addWidget(self._clear_glow_btn)

    def set_glow_active(self, active: bool) -> None:
        """Enable the Clear-view control only while a glow is active (lit = a spotlight to restore).
        The panel calls this on every stabilization tab when it applies / clears the shared glow."""
        if getattr(self, "_clear_glow_btn", None) is not None:
            self._clear_glow_btn.setEnabled(bool(active))

    @staticmethod
    def _make_ranked_table(cols: List[str], on_cell_clicked) -> "QtWidgets.QTableWidget":
        """The standard ranked-results table: read-only, whole-row single-selection, hidden until
        populated, row-click → *on_cell_clicked(row, col)*. The one factory all three tabs use."""
        tbl = QtWidgets.QTableWidget(0, len(cols))
        tbl.setHorizontalHeaderLabels(cols)
        tbl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tbl.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        tbl.setVisible(False)
        tbl.cellClicked.connect(on_cell_clicked)
        return tbl

    @staticmethod
    def _make_caveat() -> "QtWidgets.QLabel":
        """The measured-not-promised caveat label (amber, word-wrapped, hidden until there are rows)."""
        cav = QtWidgets.QLabel(); cav.setWordWrap(True)
        cav.setStyleSheet("color:#b36b00;"); cav.setVisible(False)
        return cav


# ── persistent Disulfides results tab (whole-suite home; replaces the transient dialogs) ──

class DisulfidesResultsTab(_StabilizationResultsTab):
    """Persistent home for the disulfide suite's results — A (assess existing pairs) / B (measure
    geometry) / D (engineerable-site scan) / C (declared-bond fold) — replacing the four transient
    dialogs/status-lines (a modeless dialog VANISHED on focus loss; a tab PERSISTS across top-level
    tab-switches + panel rebuilds). Each mode has its OWN section: an unrun mode shows a clear
    'Run … to populate' PLACEHOLDER (dormant, not broken-looking — never a blank table); a run mode
    shows its table. A row-click on a pair table HIGHLIGHTS both members in 3D through the panel's
    EXISTING selection seam (`on_highlight`) — this tab is the §9 unified-highlighting convergence
    surface, built reusing the seam not shadowing it. The tables are the SOURCE OF TRUTH for which
    residue pairs with which; the 'Disulfide sites' colour mode is a complementary navigational index."""

    def __init__(self, *, on_highlight, on_declare, on_estimate_ddg=None, on_clear_glow=None,
                 on_add_to_basket=None):
        super().__init__(on_highlight=on_highlight, on_clear_glow=on_clear_glow,
                         on_add_to_basket=on_add_to_basket)
        self._on_declare = on_declare            # (cd, (a,b))     -> Mode C introduce→constrain
        self._on_estimate_ddg = on_estimate_ddg  # (cd, pair_dict) -> ΔΔG escalation (legacy bridge)
        # Explicit escape from the pair-click spotlight: restore the normal representation (un-ghost
        # the structure + drop the highlight). LIT only while a pair is glowing (something to clear).
        # The common case — switching to another colour mode — clears the glow automatically; this is
        # the manual control for "turn it off without picking a different mode".
        self._add_header(
            "Disulfide suite results. These tables are the SOURCE OF TRUTH for which residue pairs "
            "with which; the 'Disulfide sites' colour mode in the Workbench is a complementary "
            "navigational index (the glow points the eye, the table carries the pairing).",
            "Clear disulfide view (un-ghost)",
            "Restore the normal representation: un-ghost the structure and remove the pair spotlight. "
            "Enabled while a pair glow is active.")
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget(); self._vl = QtWidgets.QVBoxLayout(body)
        scroll.setWidget(body); self._outer.addWidget(scroll, 1)
        self._sec: Dict[str, dict] = {}
        self._make_section("A", "Existing Cys pairs — bonding frequency (Mode A: assess existing)",
                           "Run “Assess existing Cys pairs” to populate.",
                           ["Pair", "Frequency", "N folds", "median SG–SG (Å)"])
        self._make_section("B", "Measured pair geometry (Mode B: measure)",
                           "Run “Measure pair geometry” to populate.",
                           ["Pair", "SG–SG (Å)", "Cβ–Cβ (Å)", "Cα–Cα (Å)", "χSS (°)", "Compatible"])
        self._make_section("D", "Engineerable disulfide sites — backbone scan (Mode D: find novel)",
                           "Run “Find engineerable disulfide sites” to populate.",
                           ["Pair", "Score", "Cα–Cα (Å)", "Cβ–Cβ (Å)",
                            "Sγ–Sγ (Å)", "χSS (°)", "Clash", "Orientation (°)"],
                           declare=True, caveat=True, basket=True)
        self._make_section("I", "Interface disulfide sites — inter-chain scan (find inter-subunit)",
                           "Run “Find interface disulfide sites” on a folded MULTIMER to populate.",
                           ["Pair", "Score", "Cα–Cα (Å)", "Cβ–Cβ (Å)",
                            "Sγ–Sγ (Å)", "χSS (°)", "Clash", "Orientation (°)", "ΔΔG (kcal/mol)"],
                           declare=True, caveat=True, escalate=True, basket=True)
        self._make_section("C", "Declared-bond fold (Mode C: intervene)",
                           "Use “Declare cysteine bonds and fold” (or Declare below) to record the "
                           "constrained fold here.", None)
        self._vl.addStretch(1)

    # ── section scaffolding ────────────────────────────────────────────────────────
    def _make_section(self, key: str, title: str, placeholder: str,
                      cols: Optional[List[str]], *, declare: bool = False, caveat: bool = False,
                      escalate: bool = False, basket: bool = False) -> None:
        box = QtWidgets.QGroupBox(title)
        gl = QtWidgets.QVBoxLayout(box)
        ph = QtWidgets.QLabel(placeholder); ph.setWordWrap(True); ph.setStyleSheet("color:#888;")
        gl.addWidget(ph)
        sec = {"box": box, "placeholder": ph, "cols": cols, "cd": None, "pairs": [],
               "table": None, "caveat": None, "detail": None}
        if caveat:
            cav = self._make_caveat(); gl.addWidget(cav); sec["caveat"] = cav
        if cols is not None:
            tbl = self._make_ranked_table(cols, lambda r, _c, k=key: self._on_row(k, r))
            gl.addWidget(tbl); sec["table"] = tbl
            detail = QtWidgets.QLabel(); detail.setWordWrap(True); detail.setStyleSheet("color:#444;")
            detail.setVisible(False); gl.addWidget(detail); sec["detail"] = detail
        else:                                       # readout-only section (Mode C)
            ro = QtWidgets.QLabel(); ro.setWordWrap(True); ro.setVisible(False)
            gl.addWidget(ro); sec["readout"] = ro
        if declare:
            btn = QtWidgets.QPushButton("Declare bond and fold")
            btn.setEnabled(False)
            btn.clicked.connect(lambda _=False, k=key: self._declare(k))
            gl.addWidget(btn); sec["declare_btn"] = btn
        if escalate:
            # ΔΔG-escalation: an energetic read for the SELECTED geometric hit via the legacy bridge —
            # explicit, per-row, long-running (PyRosetta minutes × 2 mutations); NEVER auto-run.
            eb = QtWidgets.QPushButton("Estimate ΔΔG (legacy)")
            eb.setEnabled(False)
            eb.setToolTip("Energetic estimate (uncalibrated — ranking/sign only) for the selected pair. "
                          "Runs the legacy ΔΔG bridge on the X→C mutations; minutes per pair.")
            eb.clicked.connect(lambda _=False, k=key: self._estimate(k))
            gl.addWidget(eb); sec["escalate_btn"] = eb
        if basket:
            bb = QtWidgets.QPushButton("Add to design")
            bb.setToolTip("Stage the selected disulfide (both residues → Cys) into the Design basket "
                          "(collect picks across strategies, then enact one variant).")
            bb.setEnabled(False)
            bb.clicked.connect(lambda _=False, k=key: self._add_to_basket(k))
            gl.addWidget(bb); sec["basket_btn"] = bb
        self._sec[key] = sec
        self._vl.addWidget(box)

    def _add_to_basket(self, key: str) -> None:
        sec = self._sec.get(key) or {}
        pairs, cd = sec.get("pairs") or [], sec.get("cd")
        tbl = sec.get("table")
        row = tbl.currentRow() if tbl is not None else -1
        if 0 <= row < len(pairs) and cd is not None and self._on_add_to_basket is not None:
            self._on_add_to_basket(cd, pairs[row])

    # ── row-click → highlight the RIGHT pair on the RIGHT construct (the §9 seam) ─────
    def _on_row(self, key: str, row: int) -> None:
        sec = self._sec.get(key) or {}
        pairs, cd = sec.get("pairs") or [], sec.get("cd")
        if not (0 <= row < len(pairs)) or cd is None:
            return
        p = pairs[row]
        self._on_highlight(cd, p)                   # routes through the panel's existing seam
        d = sec.get("detail")
        if d is not None:
            d.setText(self._pair_detail(key, p)); d.setVisible(True)

    def _declare(self, key: str) -> None:
        sec = self._sec.get(key) or {}
        pairs, cd = sec.get("pairs") or [], sec.get("cd")
        tbl = sec.get("table")
        row = tbl.currentRow() if tbl is not None else -1
        if not (0 <= row < len(pairs)) or cd is None:
            return
        # pass the FULL pair dict (carries chain_a/chain_b) so an INTERFACE (cross-chain) pair reaches
        # the cross-chain Mode-C declare; an intra Mode-D pair (chain_a == chain_b) takes the intra path.
        self._on_declare(cd, pairs[row])

    def _estimate(self, key: str) -> None:
        """ΔΔG-escalate the SELECTED interface row → the panel's legacy-bridge route. Per-row, explicit."""
        sec = self._sec.get(key) or {}
        pairs, cd = sec.get("pairs") or [], sec.get("cd")
        tbl = sec.get("table")
        row = tbl.currentRow() if tbl is not None else -1
        if not (0 <= row < len(pairs)) or cd is None or self._on_estimate_ddg is None:
            return
        self._on_estimate_ddg(cd, pairs[row])

    @staticmethod
    def _reach_detail(p: dict) -> str:
        """The reachability clause shared by the D/I detail lines — the rotamer Sγ readout (best
        achievable Sγ–Sγ + χSS) + the rigid-backbone clash, with the orientation now a measured aside."""
        sg, chi, clash = p.get("best_sg_sg"), p.get("best_chi_ss"), p.get("clash")
        ornt = p.get("orientation")
        reach = ("Sγ-reachability n/a (no backbone N)" if sg is None
                 else f"best Sγ–Sγ {sg:.2f} Å at χSS {chi:.0f}° (rotamer-placed)")
        clash_s = "" if clash is None else (" — ⚠ rigid-backbone CLASH" if clash else " — clash-free")
        ornt_s = "n/a (Gly)" if ornt is None else f"{ornt:.0f}°"
        return f"{reach}{clash_s}; backbone orientation {ornt_s} (measured, no longer ranked)"

    @classmethod
    def _pair_detail(cls, key: str, p: dict) -> str:
        lbl = pair_label(p, cys=True)
        if key == "D":
            return (f"{lbl} — score {p.get('score', 0):.2f} (MEASURED, this fold): "
                    f"Cα–Cα {p.get('ca_ca')} Å, Cβ–Cβ {p.get('cb_cb')} Å; "
                    f"{cls._reach_detail(p)}. Geometry permits a disulfide — installing needs "
                    f"introducing the Cys pair + re-folding (Declare below).")
        if key == "I":
            base = (f"{lbl} — score {p.get('score', 0):.2f} (geometric, this structure): "
                    f"Cα–Cα {p.get('ca_ca')} Å, Cβ–Cβ {p.get('cb_cb')} Å; {cls._reach_detail(p)}.")
            da, db = p.get("ddg_a"), p.get("ddg_b")
            if da is None or db is None:
                return base + " ΔΔG not yet estimated — “Estimate ΔΔG (legacy)” for an energetic read."
            be = p.get("ddg_backend")
            ddg = (f" ΔΔG (legacy{f', {be}' if be else ''}): {p.get('from_aa_a','')}{p.get('resnum_a')}C "
                   f"{da:+.2f}, {p.get('from_aa_b','')}{p.get('resnum_b')}C {db:+.2f} kcal/mol. " + cls._DDG_CAVEAT)
            if p.get("ddg_source") == "denovo":
                ddg += " " + cls._DDG_DENOVO_LAYER
            return base + ddg
        return f"{lbl} highlighted in 3D."

    # ── populate (each consumer calls its own) ──────────────────────────────────────
    def _fill_table(self, key: str, cd, pairs: List[dict], rowvals) -> None:
        sec = self._sec[key]
        sec["cd"], sec["pairs"] = cd, list(pairs)
        tbl = sec["table"]
        tbl.setRowCount(len(pairs))
        for i, p in enumerate(pairs):
            for j, v in enumerate(rowvals(p)):
                tbl.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        tbl.resizeColumnsToContents()
        has = len(pairs) > 0
        sec["placeholder"].setVisible(not has)
        tbl.setVisible(has)
        if sec.get("declare_btn") is not None:
            sec["declare_btn"].setEnabled(has)
        if sec.get("escalate_btn") is not None:
            sec["escalate_btn"].setEnabled(has)
        if sec.get("basket_btn") is not None:
            sec["basket_btn"].setEnabled(has and self._on_add_to_basket is not None)
        if has:
            tbl.selectRow(0); self._on_row(key, 0)         # preselect + highlight the best/first
            self._announce()                               # new results → tab-bar dot

    def populate_assess(self, cd, pairs: List[dict]) -> None:
        self._fill_table("A", cd, pairs, lambda p: [
            pair_label(p), f"{p.get('n_compatible', 0)}/{p.get('n_folds', 0)}",
            str(p.get("n_folds", "")), str(p.get("median_sg_sg", "") if p.get("median_sg_sg") is not None else "—")])

    def populate_geometry(self, cd, pairs: List[dict]) -> None:
        def _row(p):
            chi = p.get("chi_ss")
            return [pair_label(p), str(p.get("sg_sg", "—")),
                    str(p.get("cb_cb", "—")), str(p.get("ca_ca", "—")),
                    ("—" if chi is None else f"{chi:.0f}"),
                    "yes" if p.get("bonding_compatible") else "no"]
        self._fill_table("B", cd, pairs, _row)

    @staticmethod
    def _reach_cells(p: dict) -> List[str]:
        """The shared reachability columns (Sγ–Sγ, χSS, Clash) for a Mode-D / interface scan row —
        the rotamer-placement readout that REPLACED the backbone-orientation proxy in the ranking."""
        sg, chi, clash = p.get("best_sg_sg"), p.get("best_chi_ss"), p.get("clash")
        return [
            ("—" if sg is None else f"{sg:.2f}"),
            ("—" if chi is None else f"{chi:.0f}"),
            ("—" if clash is None else ("clash" if clash else "ok")),
        ]

    def populate_scan(self, cd, scan: dict) -> None:
        pairs = (scan or {}).get("pairs") or []
        sec = self._sec["D"]
        if sec.get("caveat") is not None:
            sec["caveat"].setText("⚠ " + ((scan or {}).get("caveat")
                                  or "Geometric compatibility only — a starting point, not a recommendation."))
            sec["caveat"].setVisible(bool(pairs))
        self._fill_table("D", cd, pairs, lambda p: [
            pair_label(p), f"{p.get('score', 0):.2f}",
            f"{p.get('ca_ca', '')}", f"{p.get('cb_cb', '')}",
            *self._reach_cells(p),
            ("—" if p.get("orientation") is None else f"{p['orientation']:.0f}")])

    def populate_interface(self, cd, scan: dict) -> None:
        """Interface (inter-chain) scan results — the cross-chain analogue of `populate_scan`. Pairs
        carry chain_a != chain_b, so `pair_label` renders chain:resnum on BOTH members (A:140 ↔ B:88)."""
        pairs = (scan or {}).get("pairs") or []
        sec = self._sec["I"]
        if sec.get("caveat") is not None:
            sec["caveat"].setText("⚠ " + ((scan or {}).get("caveat")
                                  or "Geometric compatibility only — a starting point, not a recommendation."))
            sec["caveat"].setVisible(bool(pairs))
        self._fill_table("I", cd, pairs, lambda p: [
            pair_label(p), f"{p.get('score', 0):.2f}",
            f"{p.get('ca_ca', '')}", f"{p.get('cb_cb', '')}",
            *self._reach_cells(p),
            ("—" if p.get("orientation") is None else f"{p['orientation']:.0f}"),
            self._ddg_cell(p)])

    # ── ΔΔG-escalation display (the energetic read attached to a geometric interface hit) ─────
    @staticmethod
    def _ddg_cell(p: dict) -> str:
        """The ΔΔG table cell for an interface row — '—' until escalated, else the two per-position
        X→C values (the geometric score stays the PRIMARY column; ΔΔG is additional, never a verdict)."""
        da, db = p.get("ddg_a"), p.get("ddg_b")
        if da is None or db is None:
            return "—"
        return f"{da:+.1f} / {db:+.1f}"

    # The base ΔΔG caveat (display) + the de-novo extra layer. Mirrors the router's text so the
    # readout is honest whether the user reads the cell, the detail line, or the status.
    _DDG_CAVEAT = ("ΔΔG estimate (legacy, uncalibrated — ranking/sign only, ~±2.7 kcal/mol). A second "
                   "soft signal on the geometric suggestion — not confirmation the bond will form. "
                   "Validate by declaring → re-folding → measuring.")
    _DDG_DENOVO_LAYER = ("DE-NOVO fold: this ΔΔG sits on a PREDICTED structure (Boltz coordinate error) "
                         "UNDER an already-uncalibrated ΔΔG — an estimate on an estimate. Treat as a "
                         "weak directional hint only.")

    def populate_constraint(self, cd, info: dict) -> None:
        """Mode C is ADDITIVE — the constrained fold still lands as a model + provenance badge
        elsewhere; this only RECORDS the readout (declared bonds + the landed model)."""
        sec = self._sec["C"]
        bonds = info.get("disulfide_bonds") or []
        bstr = ", ".join(f"Cys{a}–Cys{b}" for a, b in bonds) or "—"
        mid = info.get("model_id")
        sec["cd"] = cd
        sec["readout"].setText(
            f"Declared bond(s) {bstr} — folded with the constraint (BIASES toward the bond; does "
            f"NOT enforce geometry). Result: model #{mid}{(' (' + info['variant_id'] + ')') if info.get('variant_id') else ''}. "
            f"Measure its geometry with “Measure pair geometry” to read the as-produced bond.")
        sec["readout"].setVisible(True)
        sec["placeholder"].setVisible(False)
        self._announce()                                   # new results → tab-bar dot

    def reset(self) -> None:
        """Clear EVERY section back to its dormant 'Run … to populate' placeholder. The tab itself
        PERSISTS across a session reset (it is a sibling of the workbench in `gui_app.tabs`), but its
        content belongs to the PRIOR session — after a Load it would reference a construct no longer
        loaded (stale-data-wrong). So `_reset_view_for_session` KEEPS the tab and calls this to empty
        it; the new session's scans repopulate. Parallel to the panel's own `reset()`."""
        for sec in self._sec.values():
            sec["cd"], sec["pairs"] = None, []
            tbl = sec.get("table")
            if tbl is not None:
                tbl.setRowCount(0)
                tbl.setVisible(False)
            for k in ("caveat", "detail", "readout"):
                w = sec.get(k)
                if w is not None:
                    w.setVisible(False)
            if sec.get("declare_btn") is not None:
                sec["declare_btn"].setEnabled(False)
            if sec.get("escalate_btn") is not None:
                sec["escalate_btn"].setEnabled(False)
            if sec.get("basket_btn") is not None:
                sec["basket_btn"].setEnabled(False)
            sec["placeholder"].setVisible(True)
        self.set_glow_active(False)              # a reset 3D scene has no active glow to clear


class ProlineResultsTab(_StabilizationResultsTab):
    """Persistent home for the PROLINE-stabilization scan — a peer of the Disulfides tab (stabilization
    is the program's core; the strategies are first-class peers). ONE ranked table (X→Pro candidates by
    backbone φ/ψ proline-compatibility × a backbone-H-bond-donor penalty), an existing-prolines line
    (the 'see what's already there' design-context half), the measured-not-promised caveat, and the
    Clear-disulfide-style 'Clear view' control. A row-click HIGHLIGHTS the residue in 3D through the
    panel's EXISTING glow seam; per-row Declare-and-validate (substitute→fold→compare) + Estimate-ΔΔG.
    Persists across tab-switch + session reset (keep-and-clear, the disulfide-tab lesson)."""

    def __init__(self, *, on_highlight, on_declare=None, on_estimate_ddg=None, on_clear_glow=None,
                 on_show_existing=None, on_add_to_basket=None):
        super().__init__(on_highlight=on_highlight, on_clear_glow=on_clear_glow,
                         on_add_to_basket=on_add_to_basket)
        self._on_declare = on_declare              # (cd, cand) -> substitute→Pro + validate (fold/compare)
        self._on_estimate_ddg = on_estimate_ddg    # (cd, cand) -> X→Pro ΔΔG escalation (legacy bridge)
        self._on_show_existing = on_show_existing  # (cd)       -> highlight the existing prolines
        self._cd = None
        self._cands: List[dict] = []
        self._add_header(
            "Proline-stabilization candidates. X→Pro sites ranked by backbone φ/ψ proline-compatibility "
            "with a backbone-H-bond-donor penalty (proline can't donate the N–H···O bond helices/sheets "
            "rely on, so stabilizing prolines live in loops/turns). The table is the SOURCE OF TRUTH; "
            "the 'Proline sites' colour mode is a complementary navigational index.",
            "Clear proline view (un-ghost)",
            "Restore the normal representation: un-ghost the structure and remove the residue spotlight. "
            "Enabled while a residue glow is active.")

        box = QtWidgets.QGroupBox("Proline-substitution sites (ranked)")
        gl = QtWidgets.QVBoxLayout(box)
        self._caveat = self._make_caveat()
        gl.addWidget(self._caveat)
        self._existing_lbl = QtWidgets.QLabel(); self._existing_lbl.setWordWrap(True)
        self._existing_lbl.setStyleSheet("color:#777;"); self._existing_lbl.setVisible(False)
        gl.addWidget(self._existing_lbl)
        self._placeholder = QtWidgets.QLabel("Run “Find proline-stabilization sites” to populate.")
        self._placeholder.setWordWrap(True); self._placeholder.setStyleSheet("color:#888;")
        gl.addWidget(self._placeholder)
        cols = ["Residue", "φ (°)", "ψ (°)", "Score", "H-bond donor", "ΔΔG (kcal/mol)"]
        self._tbl = self._make_ranked_table(cols, lambda r, _c: self._on_row(r))
        gl.addWidget(self._tbl)
        self._detail = QtWidgets.QLabel(); self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color:#444;"); self._detail.setVisible(False)
        gl.addWidget(self._detail)
        row = QtWidgets.QHBoxLayout()
        self._existing_btn = QtWidgets.QPushButton("Show existing prolines")
        self._existing_btn.setEnabled(False)
        self._existing_btn.clicked.connect(lambda: self._on_show_existing(self._cd)
                                           if (self._on_show_existing and self._cd) else None)
        row.addWidget(self._existing_btn)
        self._declare_btn = QtWidgets.QPushButton("Substitute → Pro and validate")
        self._declare_btn.setToolTip("Substitute the selected residue to proline and validate by "
                                     "re-folding (no constraint) + comparing — the real test.")
        self._declare_btn.setEnabled(False)
        self._declare_btn.clicked.connect(self._declare)
        row.addWidget(self._declare_btn)
        self._escalate_btn = QtWidgets.QPushButton("Estimate ΔΔG (legacy)")
        self._escalate_btn.setToolTip("Energetic estimate (uncalibrated — ranking/sign only) for the "
                                      "selected X→Pro mutation; minutes.")
        self._escalate_btn.setEnabled(False)
        self._escalate_btn.clicked.connect(self._estimate)
        row.addWidget(self._escalate_btn)
        self._basket_btn = QtWidgets.QPushButton("Add to design")
        self._basket_btn.setToolTip("Stage the selected X→Pro substitution into the Design basket "
                                    "(collect picks across strategies, then enact one variant).")
        self._basket_btn.setEnabled(False)
        self._basket_btn.clicked.connect(self._add_to_basket)
        row.addWidget(self._basket_btn)
        gl.addLayout(row)
        self._outer.addWidget(box)
        self._outer.addStretch(1)

    def _add_to_basket(self) -> None:
        r = self._tbl.currentRow()
        if 0 <= r < len(self._cands) and self._cd is not None and self._on_add_to_basket is not None:
            self._on_add_to_basket(self._cd, self._cands[r])

    def _on_row(self, r: int) -> None:
        if not (0 <= r < len(self._cands)) or self._cd is None:
            return
        c = self._cands[r]
        self._on_highlight(self._cd, c)
        self._detail.setText(self._cand_detail(c)); self._detail.setVisible(True)

    @staticmethod
    def _cand_detail(c: dict) -> str:
        psi = c.get("psi")
        don = (" — ⚠ its backbone amide N–H DONATES a backbone H-bond (helix/sheet); a proline here "
               "would break it (soft-penalized, validate)") if c.get("hbond_donates") else \
              " — does not donate a backbone H-bond (loop/turn-like — a favourable proline site)"
        ddg = c.get("ddg")
        ddg_s = (f" ΔΔG (legacy): {c.get('from_aa','')}{c.get('position')}P {ddg:+.2f} kcal/mol."
                 if ddg is not None else " ΔΔG not yet estimated — “Estimate ΔΔG (legacy)”.")
        return (f"{c.get('from_aa','')}{c.get('position')}→P (chain {c.get('chain')}) — score "
                f"{c.get('score', 0):.2f}: φ {c.get('phi')}°, ψ {'n/a' if psi is None else f'{psi}°'}"
                f"{don}.{ddg_s} Validate by substituting → re-folding → comparing.")

    def _declare(self) -> None:
        r = self._tbl.currentRow()
        if 0 <= r < len(self._cands) and self._cd is not None and self._on_declare is not None:
            self._on_declare(self._cd, self._cands[r])

    def _estimate(self) -> None:
        r = self._tbl.currentRow()
        if 0 <= r < len(self._cands) and self._cd is not None and self._on_estimate_ddg is not None:
            self._on_estimate_ddg(self._cd, self._cands[r])

    def populate(self, cd, scan: dict) -> None:
        """Fill the ranked table from a proline scan ({candidates, existing, caveat})."""
        cands = (scan or {}).get("candidates") or []
        self._cd, self._cands = cd, list(cands)
        self._caveat.setText("⚠ " + ((scan or {}).get("caveat") or "Geometric suggestion only."))
        self._caveat.setVisible(bool(cands))
        existing = (scan or {}).get("existing") or []
        self._existing_lbl.setText(
            f"Existing prolines in this structure: {len(existing)}"
            + (" — " + ", ".join(f"{ch}:{rn}" for ch, rn in existing[:12]) + ("…" if len(existing) > 12 else "")
               if existing else "."))
        self._existing_lbl.setVisible(True)
        self._existing_btn.setEnabled(bool(existing))
        self._tbl.setRowCount(len(cands))
        for i, c in enumerate(cands):
            psi = c.get("psi")
            cells = [f"{c.get('from_aa','')}{c.get('position')}P", f"{c.get('phi')}",
                     ("—" if psi is None else f"{psi}"), f"{c.get('score', 0):.2f}",
                     ("donor" if c.get("hbond_donates") else "—"), self._ddg_cell(c)]
            for j, v in enumerate(cells):
                self._tbl.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self._tbl.resizeColumnsToContents()
        has = len(cands) > 0
        self._placeholder.setVisible(not has)
        self._tbl.setVisible(has)
        self._declare_btn.setEnabled(has)
        self._escalate_btn.setEnabled(has)
        self._basket_btn.setEnabled(has and self._on_add_to_basket is not None)
        if has:
            self._tbl.selectRow(0); self._on_row(0)
            self._announce()                               # new results → tab-bar dot

    @staticmethod
    def _ddg_cell(c: dict) -> str:
        ddg = c.get("ddg")
        return "—" if ddg is None else f"{ddg:+.2f}"

    def reset(self) -> None:
        """Clear back to the dormant placeholder (keep-and-clear on session reset — the disulfide-tab
        lesson: the tab PERSISTS as a sibling, but its content belongs to the prior session)."""
        self._cd, self._cands = None, []
        self._tbl.setRowCount(0); self._tbl.setVisible(False)
        self._caveat.setVisible(False); self._existing_lbl.setVisible(False)
        self._detail.setVisible(False); self._placeholder.setVisible(True)
        self._declare_btn.setEnabled(False); self._escalate_btn.setEnabled(False)
        self._existing_btn.setEnabled(False)
        if getattr(self, "_basket_btn", None) is not None:
            self._basket_btn.setEnabled(False)
        self.set_glow_active(False)


class CavityResultsTab(_StabilizationResultsTab):
    """Persistent home for the CAVITY-FILLING scan — the third peer of Disulfides / Proline (built on
    the shared `_StabilizationResultsTab` base, the consolidation paid at the third instance). ONE
    ranked table (small→larger hydrophobic fills of detected internal voids, by void-fill fraction ×
    rotamer reach-into-void, clash-demoted), a detected-cavities summary line, the context-dependent
    caveat (cavity-filling is modest for generic thermostability but powerful for CONFORMATIONAL locks
    — the RSV prefusion-F lesson — and the geometric scan can't tell which a void is), and the shared
    Clear-view control. Row-click HIGHLIGHTS the residue through the SAME glow seam; per-row Substitute-
    and-validate (substitute→fold→compare, no constraint) + Estimate-ΔΔG + Add-to-design. Persists
    across tab-switch + session reset (keep-and-clear)."""

    def __init__(self, *, on_highlight, on_declare=None, on_estimate_ddg=None, on_clear_glow=None,
                 on_add_to_basket=None):
        super().__init__(on_highlight=on_highlight, on_clear_glow=on_clear_glow,
                         on_add_to_basket=on_add_to_basket)
        self._on_declare = on_declare              # (cd, cand) -> substitute→fill + validate (fold/compare)
        self._on_estimate_ddg = on_estimate_ddg    # (cd, cand) -> fill ΔΔG escalation (legacy bridge)
        self._cd = None
        self._cands: List[dict] = []
        self._add_header(
            "Cavity-filling candidates. Small→larger hydrophobic fills of detected INTERNAL voids, "
            "ranked by void-fill fraction × how well a rotamer reaches into the void clash-free. The "
            "table is the SOURCE OF TRUTH; the 'Cavity sites' colour mode is a complementary "
            "navigational index. Cavity-filling is modest for generic thermostability but a proven "
            "CONFORMATIONAL-locking tool (the RSV prefusion-F lesson) — the scan finds geometrically-"
            "viable fills; YOU judge whether a cavity is strategically important.",
            "Clear cavity view (un-ghost)",
            "Restore the normal representation: un-ghost the structure and remove the residue spotlight. "
            "Enabled while a residue glow is active.")

        box = QtWidgets.QGroupBox("Cavity-filling sites (ranked)")
        gl = QtWidgets.QVBoxLayout(box)
        self._caveat = self._make_caveat()
        gl.addWidget(self._caveat)
        self._cavities_lbl = QtWidgets.QLabel(); self._cavities_lbl.setWordWrap(True)
        self._cavities_lbl.setStyleSheet("color:#777;"); self._cavities_lbl.setVisible(False)
        gl.addWidget(self._cavities_lbl)
        self._placeholder = QtWidgets.QLabel("Run “Find cavity-filling sites” to populate.")
        self._placeholder.setWordWrap(True); self._placeholder.setStyleSheet("color:#888;")
        gl.addWidget(self._placeholder)
        cols = ["Substitution", "Cavity", "Void (Å³)", "Fill", "Clash", "Score", "ΔΔG (kcal/mol)"]
        self._tbl = self._make_ranked_table(cols, lambda r, _c: self._on_row(r))
        gl.addWidget(self._tbl)
        self._detail = QtWidgets.QLabel(); self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color:#444;"); self._detail.setVisible(False)
        gl.addWidget(self._detail)
        row = QtWidgets.QHBoxLayout()
        self._declare_btn = QtWidgets.QPushButton("Substitute → fill and validate")
        self._declare_btn.setToolTip("Substitute the selected residue to the larger fill residue and "
                                     "validate by re-folding (no constraint) + comparing — the real test.")
        self._declare_btn.setEnabled(False)
        self._declare_btn.clicked.connect(self._declare)
        row.addWidget(self._declare_btn)
        self._escalate_btn = QtWidgets.QPushButton("Estimate ΔΔG (legacy)")
        self._escalate_btn.setToolTip("Energetic estimate (uncalibrated — ranking/sign only) for the "
                                      "selected fill mutation; minutes.")
        self._escalate_btn.setEnabled(False)
        self._escalate_btn.clicked.connect(self._estimate)
        row.addWidget(self._escalate_btn)
        self._basket_btn = QtWidgets.QPushButton("Add to design")
        self._basket_btn.setToolTip("Stage the selected fill substitution into the Design basket "
                                    "(collect picks across strategies, then enact one variant).")
        self._basket_btn.setEnabled(False)
        self._basket_btn.clicked.connect(self._add_to_basket)
        row.addWidget(self._basket_btn)
        gl.addLayout(row)
        self._outer.addWidget(box)
        self._outer.addStretch(1)

    def _add_to_basket(self) -> None:
        r = self._tbl.currentRow()
        if 0 <= r < len(self._cands) and self._cd is not None and self._on_add_to_basket is not None:
            self._on_add_to_basket(self._cd, self._cands[r])

    def _on_row(self, r: int) -> None:
        if not (0 <= r < len(self._cands)) or self._cd is None:
            return
        c = self._cands[r]
        self._on_highlight(self._cd, c)
        self._detail.setText(self._cand_detail(c)); self._detail.setVisible(True)

    @staticmethod
    def _cand_detail(c: dict) -> str:
        clash = (" — ⚠ every reaching rotamer CLASHES the walls (over-pack/strain risk on the rigid "
                 "backbone; soft-demoted, validate)") if c.get("clash") else \
                " — a clash-free rotamer fills the void"
        ddg = c.get("ddg")
        ddg_s = (f" ΔΔG (legacy): {c.get('from_aa','')}{c.get('position')}{c.get('to_aa','')} {ddg:+.2f} "
                 f"kcal/mol." if ddg is not None else " ΔΔG not yet estimated — “Estimate ΔΔG (legacy)”.")
        return (f"{c.get('from_aa','')}{c.get('position')}→{c.get('to_aa','')} (chain {c.get('chain')}) — "
                f"score {c.get('score', 0):.2f}: cavity {c.get('cavity_id')}, void {c.get('void_volume', 0):.0f} Å³, "
                f"fills {c.get('fill_fraction', 0):.0%}, reach {c.get('reach_score', 0):.0%}{clash}."
                f"{ddg_s} Validate by substituting → re-folding → comparing.")

    def _declare(self) -> None:
        r = self._tbl.currentRow()
        if 0 <= r < len(self._cands) and self._cd is not None and self._on_declare is not None:
            self._on_declare(self._cd, self._cands[r])

    def _estimate(self) -> None:
        r = self._tbl.currentRow()
        if 0 <= r < len(self._cands) and self._cd is not None and self._on_estimate_ddg is not None:
            self._on_estimate_ddg(self._cd, self._cands[r])

    def populate(self, cd, scan: dict) -> None:
        """Fill the ranked table from a cavity scan ({candidates, cavities, caveat})."""
        cands = (scan or {}).get("candidates") or []
        self._cd, self._cands = cd, list(cands)
        self._caveat.setText("⚠ " + ((scan or {}).get("caveat") or "Geometric suggestion only."))
        self._caveat.setVisible(bool(cands))
        cavities = (scan or {}).get("cavities") or []
        if cavities:
            head = ", ".join(f"cav{c.get('cavity_id')} {c.get('volume', 0):.0f} Å³ ({c.get('n_lining', 0)} lining"
                             + ("/interface" if c.get("is_interface") else "") + ")" for c in cavities[:8])
            self._cavities_lbl.setText(f"Internal cavities detected: {len(cavities)} — {head}"
                                       + ("…" if len(cavities) > 8 else "."))
        else:
            self._cavities_lbl.setText("No internal cavities ≥ the volume floor (well-packed at this probe).")
        self._cavities_lbl.setVisible(True)
        self._tbl.setRowCount(len(cands))
        for i, c in enumerate(cands):
            cells = [f"{c.get('from_aa','')}{c.get('position')}{c.get('to_aa','')}",
                     str(c.get("cavity_id", "")), f"{c.get('void_volume', 0):.0f}",
                     f"{c.get('fill_fraction', 0):.2f}", ("⚠" if c.get("clash") else "ok"),
                     f"{c.get('score', 0):.2f}", self._ddg_cell(c)]
            for j, v in enumerate(cells):
                self._tbl.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self._tbl.resizeColumnsToContents()
        has = len(cands) > 0
        self._placeholder.setVisible(not has)
        self._tbl.setVisible(has)
        self._declare_btn.setEnabled(has)
        self._escalate_btn.setEnabled(has)
        self._basket_btn.setEnabled(has and self._on_add_to_basket is not None)
        if has:
            self._tbl.selectRow(0); self._on_row(0)
            self._announce()                               # new results → tab-bar dot

    @staticmethod
    def _ddg_cell(c: dict) -> str:
        ddg = c.get("ddg")
        return "—" if ddg is None else f"{ddg:+.2f}"

    def reset(self) -> None:
        """Clear back to the dormant placeholder (keep-and-clear on session reset — the disulfide-tab
        lesson: the tab PERSISTS as a sibling, but its content belongs to the prior session)."""
        self._cd, self._cands = None, []
        self._tbl.setRowCount(0); self._tbl.setVisible(False)
        self._caveat.setVisible(False); self._cavities_lbl.setVisible(False)
        self._detail.setVisible(False); self._placeholder.setVisible(True)
        self._declare_btn.setEnabled(False); self._escalate_btn.setEnabled(False)
        if getattr(self, "_basket_btn", None) is not None:
            self._basket_btn.setEnabled(False)
        self.set_glow_active(False)


class SaltBridgeResultsTab(_StabilizationResultsTab):
    """Persistent home for the SALT-BRIDGE scan — the fourth peer of Disulfides / Proline / Cavity
    (built on the shared `_StabilizationResultsTab` base). Salt-bridge is PAIRWISE (like disulfide), so
    it carries TWO ranked PAIR tables: (1) EXISTING Asp/Glu↔Arg/Lys pairs (assessment — pure measurement;
    row-click highlights both members; 4–5 Å near-misses flagged optimizable), and (2) NOVEL complementary
    charge-pair sites (engineering — row-click highlights both, + Substitute-both-and-validate, Estimate-
    ΔΔG, Add-to-design). Each row is a two-chain pair (chain_a/resnum_a + chain_b/resnum_b — intra OR
    inter-chain via `pair_label`). The context-dependent desolvation caveat rides in `_make_caveat`.
    Persists across tab-switch + session reset (keep-and-clear)."""

    def __init__(self, *, on_highlight, on_declare=None, on_estimate_ddg=None, on_clear_glow=None,
                 on_add_to_basket=None):
        super().__init__(on_highlight=on_highlight, on_clear_glow=on_clear_glow,
                         on_add_to_basket=on_add_to_basket)
        self._on_declare = on_declare              # (cd, cand) -> substitute BOTH positions + validate
        self._on_estimate_ddg = on_estimate_ddg    # (cd, cand) -> 2-residue ΔΔG escalation (legacy bridge)
        self._cd = None
        self._existing: List[dict] = []
        self._novel: List[dict] = []
        self._add_header(
            "Salt-bridge candidates. Existing Asp/Glu↔Arg/Lys pairs (assessed by closest carboxyl-O↔"
            "basic-N distance + burial) and NOVEL complementary charge-pair sites (geometry × burial). "
            "The tables are the SOURCE OF TRUTH; the 'Salt-bridge sites' colour mode is a complementary "
            "navigational index. Salt bridges are context-dependent: favourable when geometry + burial "
            "align, marginal/destabilizing at the surface (desolvation) — YOU judge the context, the "
            "re-fold validates.",
            "Clear salt-bridge view (un-ghost)",
            "Restore the normal representation: un-ghost the structure and remove the pair spotlight. "
            "Enabled while a glow is active.")

        self._caveat = self._make_caveat()
        self._outer.addWidget(self._caveat)

        # ── EXISTING pairs (assessment) ──
        ex_box = QtWidgets.QGroupBox("Existing salt bridges (assessed)")
        exl = QtWidgets.QVBoxLayout(ex_box)
        self._ex_placeholder = QtWidgets.QLabel("Run “Find salt-bridge sites” to populate.")
        self._ex_placeholder.setWordWrap(True); self._ex_placeholder.setStyleSheet("color:#888;")
        exl.addWidget(self._ex_placeholder)
        self._ex_tbl = self._make_ranked_table(
            ["Pair", "Type", "O–N (Å)", "H-bond", "Burial", "Score"],
            lambda r, _c: self._on_existing_row(r))
        exl.addWidget(self._ex_tbl)
        self._outer.addWidget(ex_box)

        # ── NOVEL sites (engineering) ──
        nv_box = QtWidgets.QGroupBox("Novel salt-bridge sites (ranked)")
        nvl = QtWidgets.QVBoxLayout(nv_box)
        self._nv_placeholder = QtWidgets.QLabel("Run “Find salt-bridge sites” to populate.")
        self._nv_placeholder.setWordWrap(True); self._nv_placeholder.setStyleSheet("color:#888;")
        nvl.addWidget(self._nv_placeholder)
        self._nv_tbl = self._make_ranked_table(
            ["Substitutions", "Pair", "O–N (Å)", "Cβ–Cβ", "H-bond", "Burial", "Clash", "Score",
             "ΔΔG (kcal/mol)"],
            lambda r, _c: self._on_novel_row(r))
        nvl.addWidget(self._nv_tbl)
        self._detail = QtWidgets.QLabel(); self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color:#444;"); self._detail.setVisible(False)
        nvl.addWidget(self._detail)
        row = QtWidgets.QHBoxLayout()
        self._declare_btn = QtWidgets.QPushButton("Substitute → charge pair and validate")
        self._declare_btn.setToolTip("Substitute BOTH positions to the complementary charged residues "
                                     "on a new variant and validate by re-folding (no constraint — "
                                     "charged residues fold natively) + comparing — the real test.")
        self._declare_btn.setEnabled(False)
        self._declare_btn.clicked.connect(self._declare)
        row.addWidget(self._declare_btn)
        self._escalate_btn = QtWidgets.QPushButton("Estimate ΔΔG (legacy)")
        self._escalate_btn.setToolTip("Energetic estimate (uncalibrated — ranking/sign only) for the "
                                      "selected charge-pair (both positions); minutes.")
        self._escalate_btn.setEnabled(False)
        self._escalate_btn.clicked.connect(self._estimate)
        row.addWidget(self._escalate_btn)
        self._basket_btn = QtWidgets.QPushButton("Add to design")
        self._basket_btn.setToolTip("Stage the selected charge-pair substitution (both positions) into "
                                    "the Design basket (collect picks across strategies, then enact).")
        self._basket_btn.setEnabled(False)
        self._basket_btn.clicked.connect(self._add_to_basket)
        row.addWidget(self._basket_btn)
        nvl.addLayout(row)
        self._outer.addWidget(nv_box)
        self._outer.addStretch(1)

    # ── row handlers ──
    def _on_existing_row(self, r: int) -> None:
        if not (0 <= r < len(self._existing)) or self._cd is None:
            return
        self._nv_tbl.clearSelection()
        self._on_highlight(self._cd, self._existing[r])
        self._detail.setText(self._existing_detail(self._existing[r])); self._detail.setVisible(True)

    def _on_novel_row(self, r: int) -> None:
        if not (0 <= r < len(self._novel)) or self._cd is None:
            return
        self._ex_tbl.clearSelection()
        c = self._novel[r]
        self._on_highlight(self._cd, c)
        self._detail.setText(self._novel_detail(c)); self._detail.setVisible(True)
        has = self._cd is not None
        self._declare_btn.setEnabled(has)
        self._escalate_btn.setEnabled(has)
        self._basket_btn.setEnabled(has and self._on_add_to_basket is not None)

    @staticmethod
    def _existing_detail(p: dict) -> str:
        opt = (" — ⚠ a 4–5 Å OPTIMIZABLE near-miss (tightening the rotamers/substitution could form a "
               "full bridge)") if p.get("optimizable") else (" — within the ≤4 Å salt-bridge cutoff")
        hb = " It is also an H-bond (≤3.5 Å — the strongest sub-class)." if p.get("hbond_like") else ""
        bur = ("buried" if p.get("buried") else "surface") if p.get("buried") is not None else "burial unknown"
        return (f"{pair_label(p)} ({p.get('type','')}) — closest O–N {p.get('on_dist')} Å{opt}.{hb} "
                f"Context: {bur} (score {p.get('score', 0):.2f}). Salt bridges are context-dependent — "
                f"buried bridges are typically stabilizing; surface ones are marginal (desolvation).")

    @staticmethod
    def _novel_detail(c: dict) -> str:
        clash = (" — ⚠ no reaching rotamer pair dodges a clash (over-pack risk on the rigid backbone; "
                 "soft-demoted, validate)") if c.get("clash") else " — a clash-free placement reaches the bridge"
        bur = ("buried" if c.get("buried") else "surface") if c.get("buried") is not None else "burial unknown"
        ddg = c.get("ddg_mean")
        ddg_s = (f" ΔΔG (legacy): mean {ddg:+.2f} kcal/mol." if ddg is not None
                 else " ΔΔG not yet estimated — “Estimate ΔΔG (legacy)”.")
        return (f"{c.get('from_aa_a','')}{c.get('resnum_a')}{c.get('to_aa_a','')} + "
                f"{c.get('from_aa_b','')}{c.get('resnum_b')}{c.get('to_aa_b','')} "
                f"({pair_label(c)}) — score {c.get('score', 0):.2f}: closest O–N {c.get('best_on')} Å, "
                f"Cβ–Cβ {c.get('cb_cb')} Å, {bur}{clash}.{ddg_s} Validate by substituting both → "
                f"re-folding (no constraint) → comparing.")

    def _add_to_basket(self) -> None:
        r = self._nv_tbl.currentRow()
        if 0 <= r < len(self._novel) and self._cd is not None and self._on_add_to_basket is not None:
            self._on_add_to_basket(self._cd, self._novel[r])

    def _declare(self) -> None:
        r = self._nv_tbl.currentRow()
        if 0 <= r < len(self._novel) and self._cd is not None and self._on_declare is not None:
            self._on_declare(self._cd, self._novel[r])

    def _estimate(self) -> None:
        r = self._nv_tbl.currentRow()
        if 0 <= r < len(self._novel) and self._cd is not None and self._on_estimate_ddg is not None:
            self._on_estimate_ddg(self._cd, self._novel[r])

    @staticmethod
    def _burial_cell(p: dict) -> str:
        b = p.get("buried")
        return "—" if b is None else ("buried" if b else "surface")

    @staticmethod
    def _ddg_cell(c: dict) -> str:
        da, db = c.get("ddg_a"), c.get("ddg_b")
        if da is None or db is None:
            return "—"
        return f"{da:+.1f} / {db:+.1f}"

    def populate(self, cd, scan: dict) -> None:
        """Fill both tables from a salt-bridge scan ({existing, novel, caveat})."""
        existing = (scan or {}).get("existing") or []
        novel = (scan or {}).get("novel") or []
        self._cd, self._existing, self._novel = cd, list(existing), list(novel)
        self._caveat.setText("⚠ " + ((scan or {}).get("caveat") or "Geometric suggestion only."))
        self._caveat.setVisible(bool(existing or novel))
        # existing table
        self._ex_tbl.setRowCount(len(existing))
        for i, p in enumerate(existing):
            on = f"{p.get('on_dist')}" + (" (opt)" if p.get("optimizable") else "")
            cells = [pair_label(p), p.get("type", ""), on,
                     ("H-bond" if p.get("hbond_like") else "—"), self._burial_cell(p),
                     f"{p.get('score', 0):.2f}"]
            for j, v in enumerate(cells):
                self._ex_tbl.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self._ex_tbl.resizeColumnsToContents()
        self._ex_tbl.setVisible(bool(existing))
        self._ex_placeholder.setVisible(not existing)
        # novel table
        self._nv_tbl.setRowCount(len(novel))
        for i, c in enumerate(novel):
            sub = (f"{c.get('from_aa_a','')}{c.get('resnum_a')}{c.get('to_aa_a','')}+"
                   f"{c.get('from_aa_b','')}{c.get('resnum_b')}{c.get('to_aa_b','')}")
            cells = [sub, pair_label(c), f"{c.get('best_on')}", f"{c.get('cb_cb')}",
                     ("H-bond" if c.get("hbond_like") else "—"), self._burial_cell(c),
                     ("⚠" if c.get("clash") else "ok"), f"{c.get('score', 0):.2f}", self._ddg_cell(c)]
            for j, v in enumerate(cells):
                self._nv_tbl.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self._nv_tbl.resizeColumnsToContents()
        has_nv = bool(novel)
        self._nv_tbl.setVisible(has_nv)
        self._nv_placeholder.setVisible(not has_nv)
        self._declare_btn.setEnabled(has_nv)
        self._escalate_btn.setEnabled(has_nv)
        self._basket_btn.setEnabled(has_nv and self._on_add_to_basket is not None)
        if has_nv:
            self._nv_tbl.selectRow(0); self._on_novel_row(0)
        elif existing:
            self._ex_tbl.selectRow(0); self._on_existing_row(0)
        if has_nv or existing:
            self._announce()                               # new results → tab-bar dot

    def reset(self) -> None:
        """Clear back to the dormant placeholders (keep-and-clear on session reset — the tab PERSISTS as
        a sibling, its content belongs to the prior session)."""
        self._cd, self._existing, self._novel = None, [], []
        for tbl, ph in ((self._ex_tbl, self._ex_placeholder), (self._nv_tbl, self._nv_placeholder)):
            tbl.setRowCount(0); tbl.setVisible(False); ph.setVisible(True)
        self._caveat.setVisible(False); self._detail.setVisible(False)
        self._declare_btn.setEnabled(False); self._escalate_btn.setEnabled(False)
        if getattr(self, "_basket_btn", None) is not None:
            self._basket_btn.setEnabled(False)
        self.set_glow_active(False)


class DesignBasketPanel(QtWidgets.QWidget):
    """The CROSS-STRATEGY substitution-staging panel — the framework centerpiece. The designer
    collects suggested substitutions from the strategy tabs (Disulfides, Proline, … any future
    geometric-scan strategy) into ONE basket, then ENACTS them into a single new variant that flows
    into the EXISTING variant machinery (fold / compare / deviation / template-overlay — all opt-in,
    unchanged). The basket COMPOSES the variant; the variant system VALIDATES it — the fold reveals
    combination effects, so the basket does NOT proxy spatial interference (only certain same-residue
    conflicts are flagged). Docks as a QDockWidget; persists across tab-switch + session reset
    (keep-and-clear). Each entry = ``{cls, label, score, subs:[{chain,position,from_aa,to_aa}],
    metrics_text}`` — a substitution targets 1+ positions (proline → one→Pro; disulfide → two→Cys)."""

    def __init__(self, *, on_enact=None, on_chain_equiv=None):
        super().__init__()
        self._on_enact = on_enact                  # (entries) -> compose a variant via the existing path
        # (chain id) -> [all fold-chain ids sharing that chain's ChainDesign]. Homo-oligomer copies
        # collapse onto ONE cd (an edit applies to ALL copies — identical sequence), so conflict /
        # dedupe MUST key by cd-equivalence, not raw chain: two picks at the same resnum on equivalent
        # copies are the SAME residue. None → no resolver wired (independent testability) → raw-chain
        # behaviour. Drives both the conflict key and the "applies to A, B" display.
        self._chain_equiv = on_chain_equiv
        self.entries: List[dict] = []
        outer = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Design basket — collect substitutions across strategy tabs, then Enact one variant. The "
            "basket composes the variant; fold/compare it as you would any variant (the FOLD reveals "
            "combination effects — the basket does not predict them).")
        intro.setWordWrap(True); intro.setStyleSheet("color:#666;")
        outer.addWidget(intro)
        self._conflict = QtWidgets.QLabel(); self._conflict.setWordWrap(True)
        self._conflict.setStyleSheet("color:#c0392b;"); self._conflict.setVisible(False)
        outer.addWidget(self._conflict)
        self._placeholder = QtWidgets.QLabel('Empty — use "Add to design" on a Disulfides or Proline row.')
        self._placeholder.setWordWrap(True); self._placeholder.setStyleSheet("color:#888;")
        outer.addWidget(self._placeholder)
        self._list = QtWidgets.QTableWidget(0, 4)
        self._list.setHorizontalHeaderLabels(["Class", "Substitution", "Score", "Detail"])
        self._list.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._list.setVisible(False)
        outer.addWidget(self._list, 1)
        row = QtWidgets.QHBoxLayout()
        self._remove_btn = QtWidgets.QPushButton("Remove selected")
        self._remove_btn.setEnabled(False); self._remove_btn.clicked.connect(self._remove_selected)
        row.addWidget(self._remove_btn)
        self._clear_btn = QtWidgets.QPushButton("Clear basket")
        self._clear_btn.setEnabled(False); self._clear_btn.clicked.connect(self.reset)
        row.addWidget(self._clear_btn)
        self._enact_btn = QtWidgets.QPushButton("Enact → variant")
        self._enact_btn.setToolTip("Compose ALL basket substitutions into one new variant (per chain), "
                                   "ready to fold / compare in the Variant Workbench.")
        self._enact_btn.setEnabled(False); self._enact_btn.clicked.connect(self._enact)
        row.addWidget(self._enact_btn)
        outer.addLayout(row)

    def _group_key(self, chain):
        """Cd-EQUIVALENCE key for a chain — homo-oligomer copies share one ChainDesign, so a pick on
        any copy applies to all → they key TOGETHER for conflict + dedupe (a pick on A and a pick on
        its equivalent copy B at the same resnum are the SAME residue, not two). Falls back to the raw
        chain id when no resolver is wired or the chain is unresolved (each its own group)."""
        if self._chain_equiv is None:
            return chain
        return frozenset(self._chain_equiv(chain))

    def add_entry(self, entry: dict) -> None:
        """Stage one strategy pick. Idempotent-ish: an entry whose subs match an existing entry's by
        (cd-equivalence, position, to_aa) is not duplicated — so the SAME substitution staged on two
        equivalent homo-oligomer copies (A:35→W and B:35→W) collapses to ONE entry."""
        sig = tuple((self._group_key(s["chain"]), s["position"], s["to_aa"]) for s in entry.get("subs", []))
        if any(tuple((self._group_key(s["chain"]), s["position"], s["to_aa"]) for s in e.get("subs", [])) == sig
               for e in self.entries):
            return
        self.entries.append(entry)
        self._refresh()

    def _remove_selected(self) -> None:
        r = self._list.currentRow()
        if 0 <= r < len(self.entries):
            del self.entries[r]
            self._refresh()

    def _enact(self) -> None:
        if self._on_enact is not None and self.entries and not self._conflicts():
            self._on_enact(list(self.entries))

    def _conflicts(self) -> List[str]:
        """A (cd-equivalence, position) targeted by 2+ entries with DIFFERENT substitutions — you can't
        make one residue two things. Keys by cd-EQUIVALENCE not raw chain (via `_group_key`), so a pick
        at the same resnum on equivalent homo-oligomer copies (A:35→W vs B:35→Y) is caught — they map
        to ONE shared template column at enact, where the second would silently overwrite the first.
        CERTAIN conflicts only (no spatial-proximity / interference prediction — that's the designer's
        call + the re-fold; the fold reveals combination effects honestly)."""
        seen: Dict[tuple, Tuple[str, object]] = {}     # (group_key, pos) -> (to_aa, originating chain)
        clashes: List[str] = []
        for e in self.entries:
            for s in e.get("subs", []):
                key = (self._group_key(s["chain"]), s["position"])
                cur = seen.get(key)
                if cur is not None and cur[0] != s["to_aa"]:
                    if str(cur[1]) == str(s["chain"]):
                        clashes.append(f"{s['chain']}:{s['position']} (→{cur[0]} and →{s['to_aa']})")
                    else:                              # equivalent copies (same cd) — name BOTH chains
                        clashes.append(f"{cur[1]}/{s['chain']} are equivalent copies — "
                                       f":{s['position']} (→{cur[0]} and →{s['to_aa']})")
                seen.setdefault(key, (s["to_aa"], s["chain"]))
        return sorted(set(clashes))

    def _refresh(self) -> None:
        n = len(self.entries)
        self._list.setRowCount(n)
        for i, e in enumerate(self.entries):
            sub_s = ", ".join(f"{s['from_aa']}{s['position']}{s['to_aa']}" for s in e["subs"])
            # Fix 2: when a pick targets a chain whose cd is SHARED by other copies, surface every
            # chain the substitution will apply to ("applies to A, B") — a direct consequence of the
            # cd-equivalence rule (homo-oligomer copies are one design), so the designer isn't misled
            # into thinking a homo pick is chain-specific. Display-only (entry data unchanged).
            applies = set()
            for s in e["subs"]:
                eq = self._chain_equiv(s["chain"]) if self._chain_equiv else [s["chain"]]
                if len(eq) > 1:
                    applies.update(str(c) for c in eq)
            detail = e.get("metrics_text", "")
            if applies:
                detail = f"applies to {', '.join(sorted(applies))} · " + detail
            for j, v in enumerate([e["cls"], sub_s, f"{e.get('score', 0):.2f}", detail]):
                self._list.setItem(i, j, QtWidgets.QTableWidgetItem(v))
        self._list.resizeColumnsToContents()
        has = n > 0
        self._placeholder.setVisible(not has)
        self._list.setVisible(has)
        self._remove_btn.setEnabled(has)
        self._clear_btn.setEnabled(has)
        conflicts = self._conflicts()
        if conflicts:
            self._conflict.setText("⚠ Same-residue conflict — two picks target the same residue "
                                   "(equivalent homo-oligomer copies count as one): "
                                   + "; ".join(conflicts) + ". Remove one before enacting "
                                   "(one residue can't be two substitutions).")
        self._conflict.setVisible(bool(conflicts))
        # block enact on a HARD conflict (the apply seam would silently let the last edit win → a
        # wrong variant). Spatial interference is NOT blocked — that's the fold's job.
        self._enact_btn.setEnabled(has and not conflicts)

    def reset(self) -> None:
        """Empty the basket (Clear button + session reset — keep-and-clear: the dock PERSISTS as a
        sibling, its contents belong to the prior curation session)."""
        self.entries = []
        self._refresh()


class _FlowLayout(QtWidgets.QLayout):
    """A left-to-right layout that WRAPS to the next row when the current row fills — so a toolbar
    never clips/hides items behind an overflow chevron at narrow widths (every action stays visible,
    just on a second row). Reports `heightForWidth` so its container grows taller as the row wraps.
    Adapted from the canonical Qt FlowLayout example."""

    def __init__(self, parent=None, margin=0, hspacing=6, vspacing=4):
        super().__init__(parent)
        self._items: List[QtWidgets.QLayoutItem] = []
        self._hspace = hspacing
        self._vspace = vspacing
        self.setContentsMargins(margin, margin, margin, margin)

    # QLayout plumbing ----------------------------------------------------------------
    def addItem(self, item):               # noqa: N802 (Qt override)
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):               # noqa: N802
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):               # noqa: N802
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):         # noqa: N802
        return QtCore.Qt.Orientations(QtCore.Qt.Orientation(0))

    def hasHeightForWidth(self):           # noqa: N802
        return True

    def heightForWidth(self, width):       # noqa: N802
        return self._do_layout(QtCore.QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):           # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):                    # noqa: N802
        return self.minimumSize()

    def minimumSize(self):                 # noqa: N802
        size = QtCore.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        x = rect.x() + m.left()
        y = rect.y() + m.top()
        right = rect.right() - m.right()
        line_height = 0
        row = 0
        placed: List[Tuple[object, int, int, "QtCore.QSize", int]] = []   # (item, x, row_top, hint, row)
        row_h: Dict[int, int] = {}                                        # row -> tallest item
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._hspace
            if next_x - self._hspace > right and line_height > 0:    # wrap to the next row
                x = rect.x() + m.left()
                y = y + line_height + self._vspace
                row += 1
                next_x = x + hint.width() + self._hspace
                line_height = 0
            placed.append((item, x, y, hint, row))
            x = next_x
            line_height = max(line_height, hint.height())
            row_h[row] = max(row_h.get(row, 0), hint.height())
        if not test_only:
            # Vertically CENTER each item within its row's height so labels / menu buttons / combos
            # of differing heights line up on a common centre line (not top-aligned, which left the
            # shorter 'Substitute →' / 'Tools' items sitting high).
            for item, ix, row_top, hint, r in placed:
                iy = row_top + (row_h[r] - hint.height()) // 2
                item.setGeometry(QtCore.QRect(QtCore.QPoint(ix, iy), hint))
        return y + line_height - rect.y() + m.bottom()


# ── the panel (toolbar + one QTabWidget; a tab per unique chain) ───────────────────

class VariantWorkbenchPanel(QtWidgets.QWidget):
    """Stage-3b Workbench panel. `controller` = a seq_editor.SequenceEditorController
    (shares the ChimeraX bridge). `load_model(model_id)` reads the structure, builds the
    DesignSession, renders the tabs, persists it. Toolbar groups: Edit (add variant /
    add sequence), Analysis (Tools ▾ — scan / MPNN / per-variant tests / Disulfides +
    Stabilize submenus), View (Fold ▾ / Align / Color / Deviation pill). Substitution is
    residue-targeted via the cell right-click menu ("Substitute →"), not a toolbar control. Column-click toggles the position into the SCAN SET (the deterministic scan
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

        # A visible ACTIVE-MODEL indicator (panel header) — the single source of truth for which
        # model the tools/scans address (e.g. "Active: Assembly #3 (Homotrimer) · 3 chains"). It
        # surfaces the bio-assembly ingest: when a `bio_assembly` build swaps the active design from
        # the 1-chain AU to the flat multi-chain assembly, the user can SEE that the interface tools
        # and scans now target the assembly, not the monomer.
        self._active_lbl = QtWidgets.QLabel("Active: (no model loaded)")
        self._active_lbl.setStyleSheet("color:#7fd1ff;padding:2px 6px;font-weight:bold;")
        lay.addWidget(self._active_lbl)

        # The toolbar groups the low-frequency tool/fold controls behind two QToolButton
        # menus (Tools ▾ / Fold ▾) so the row sheds width and the window narrows to ~content
        # width; the high-frequency authoring/test/colour controls stay as direct widgets.
        # The menu entries are QActions wired to the SAME handlers the old buttons used —
        # no parallel UI path. (State read elsewhere — _scan_set_lbl.text(), _fold_vis_btn/
        # _show_fold_cb/_show_ref_cb .isChecked()/.setChecked() — works identically on QAction.)
        # A WRAPPING flow layout (not a QToolBar, not a bare QHBoxLayout): at a narrow width the
        # pill row WRAPS onto a second row instead of clipping items behind a ">>" overflow chevron
        # (the chevron hid actions — they must ALL stay visible at any width). The container reports
        # heightForWidth so the panel grows taller as the row wraps; its minimum width collapses to
        # the widest single pill, so the window still narrows freely.
        bar_container = QtWidgets.QWidget()
        bar = _FlowLayout(bar_container, margin=4, hspacing=6, vspacing=4)
        self._toolbar_container = bar_container       # exposed for the resize test
        self._toolbar_layout = bar
        _sp = bar_container.sizePolicy()
        _sp.setHeightForWidth(True)
        bar_container.setSizePolicy(_sp)
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
        # (Substitute combo + Apply were REMOVED from the toolbar — the same edit is reachable by
        #  right-clicking a residue in a variant row: the cell menu's "Substitute →" runs the exact
        #  same `edit_variant`, residue-targeted, so no capability is lost.)
        bar.addWidget(self._toolbar_vsep())

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
        # ── Analysis group: per-variant tests (formerly top-level "Test stability/solubility"
        #    buttons) collapsed under Tools ▾ as menu items — SAME handlers, just regrouped.
        tools_menu.addSeparator()
        self._stab_btn = tools_menu.addAction("Test stability")
        self._stab_btn.setToolTip("Score the ACTIVE variant's exact mutations (4-axis ddG "
                                  "voter) through the tool spine. Deep adds Rosetta (gated).")
        self._stab_btn.triggered.connect(self._on_test_stability)
        self._sol_btn = tools_menu.addAction("Test solubility")
        self._sol_btn.setToolTip("CamSol intrinsic-solubility of the ACTIVE variant vs the "
                                 "template (instant, local).")
        self._sol_btn.triggered.connect(self._on_test_solubility)
        # (the Disulfides + Stabilize SUBMENUS are appended to this same Tools menu below, at
        #  their build sites — menus are live, so order-of-construction doesn't matter.)
        self._tools_btn.setMenu(tools_menu)
        bar.addWidget(self._tools_btn)
        bar.addWidget(self._toolbar_vsep())          # divider: Analysis | View

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
        self._fold_menu_btn.setMenu(fold_menu)        # added to the View group (pill) below

        # Disulfides — now a SUBMENU under Tools ▾ (Analysis group), no longer a top-level button.
        # Three DISTINCT modes (A/B observe the unconstrained fold; C intervenes):
        # Labels surface the load-bearing distinction — OBSERVE existing cysteines (A/B) vs FIND
        # NOVEL installable sites (D) vs INTERVENE (C). "Find/discover/suggest" belongs ONLY to D
        # (the mode that finds sites not already present); A "assesses existing", never "discovers".
        #   A Assess existing Cys pairs — multi-fold bonding FREQUENCY (the model's pairing prior)
        #   B Measure pair geometry — measured Cα/Cβ/SG/χSS vs windows for THIS fold (cheap, no fold)
        #   D Find engineerable sites — residue-agnostic BACKBONE scan for NOVEL disulfide sites
        #   C Declare cysteine bonds and fold — introduce Cys (substitution) + fold WITH the bond(s)
        # Each action is ENABLED by its real precondition (greyed = visibly unavailable, not a silent
        # no-op): A needs a de-novo construct; B/C/D need it FOLDED. Synced on tab-change/render/fold.
        tools_menu.addSeparator()
        # Explicit parent (the long-lived Tools button) + addMenu(object): the addMenu(str) overload
        # returns a QMenu shiboken can let be GC'd; constructing with a parent keeps it alive.
        self._ss_menu = QtWidgets.QMenu("Disulfides", self._tools_btn)
        self._ss_menu.setToolTipsVisible(True)
        tools_menu.addMenu(self._ss_menu)
        ss_menu = self._ss_menu                        # actions below populate this nested submenu
        self._ss_discover_btn = ss_menu.addAction("Assess existing Cys pairs (multi-fold)…")
        self._ss_discover_btn.setToolTip("ASSESS the construct's EXISTING cysteine pairs: fold "
                                         "UNCONSTRAINED across N seeds and report how often each pair "
                                         "sits in bonding geometry — the model's empirical pairing "
                                         "prior, MEASURED with N. Folds N seeds (compute). Needs a "
                                         "de-novo construct with cysteines (Add sequence).")
        self._ss_discover_btn.triggered.connect(self._on_disulfide_discover)
        self._ss_geometry_btn = ss_menu.addAction("Measure pair geometry (this fold)…")
        self._ss_geometry_btn.setToolTip("MEASURE Cα–Cα / Cβ–Cβ / SG–SG / χSS vs canonical windows "
                                         "for the existing cysteine pairs in the CURRENT fold. Cheap "
                                         "— reads coordinates, does NOT fold. Needs a folded construct.")
        self._ss_geometry_btn.triggered.connect(self._on_disulfide_geometry)
        self._ss_scan_btn = ss_menu.addAction("Find engineerable disulfide sites (backbone scan)…")
        self._ss_scan_btn.setToolTip("FIND NOVEL installable disulfide sites: an all-pairs BACKBONE "
                                     "scan (residue-agnostic — where COULD a disulfide go if both "
                                     "residues were mutated to Cys). Geometric compatibility ONLY in "
                                     "this predicted fold — a starting point, NOT a recommendation to "
                                     "mutate. Cheap (geometry, no fold). Needs a folded construct.")
        self._ss_scan_btn.triggered.connect(self._on_disulfide_scan)
        self._ss_interface_btn = ss_menu.addAction("Find interface disulfide sites (inter-chain scan)…")
        self._ss_interface_btn.setToolTip("FIND NOVEL INTER-SUBUNIT sites: a CROSS-chain backbone scan "
                                          "for disulfides that would LOCK the interface between chains "
                                          "(interface-bounded — only residues close enough across "
                                          "chains to bond). Geometric compatibility only. Needs a "
                                          "folded MULTIMER (≥2 chains).")
        self._ss_interface_btn.triggered.connect(self._on_disulfide_interface_scan)
        ss_menu.addSeparator()
        self._ss_constrain_btn = ss_menu.addAction("Declare cysteine bonds and fold…")
        self._ss_constrain_btn.setToolTip("DECLARE one or more disulfide bonds (introduce Cys at each "
                                          "position via the substitution path) and fold WITH the "
                                          "bond constraint(s). The constraint BIASES toward the bond; "
                                          "it does NOT enforce geometry — the result is measured, "
                                          "never assumed. Needs a folded construct.")
        self._ss_constrain_btn.triggered.connect(self._on_disulfide_constrain)

        # Stabilize — now a SUBMENU under Tools ▾ (Analysis group), peer of Disulfides. Proline-
        # stabilization scan: per-residue X→Pro sites by
        # backbone φ/ψ proline-compatibility + a backbone-H-bond-donor penalty. Cheap (reads the
        # structure, no fold); works on a de-novo fold OR a loaded crystal/model.
        self._pro_menu = QtWidgets.QMenu("Stabilize", self._tools_btn)
        self._pro_menu.setToolTipsVisible(True)
        tools_menu.addMenu(self._pro_menu)
        pro_menu = self._pro_menu                       # actions below populate this nested submenu
        self._pro_scan_btn = pro_menu.addAction("Find proline-stabilization sites…")
        self._pro_scan_btn.setToolTip("FIND X→Pro stabilizing sites: rank residues by backbone φ/ψ "
                                      "proline-compatibility with a backbone-H-bond-donor penalty "
                                      "(proline can't donate the N–H···O bond helices/sheets rely on, so "
                                      "stabilizing prolines live in loops/turns). Geometric suggestion "
                                      "only — a starting point, NOT a recommendation to mutate. Cheap "
                                      "(no fold). Needs a folded construct or a loaded structure.")
        self._pro_scan_btn.triggered.connect(self._on_proline_scan)
        # Cavity-filling scan — a peer strategy in the SAME Stabilize ▾ menu. Detect internal voids +
        # rank small→larger hydrophobic fills that pack them clash-free. Cheap (reads coordinates, no
        # fold); works on a de-novo fold OR a loaded crystal/model, like the proline scan.
        self._cav_scan_btn = pro_menu.addAction("Find cavity-filling sites…")
        self._cav_scan_btn.setToolTip("FIND cavity-filling stabilization sites: detect internal voids "
                                      "and rank small→larger hydrophobic substitutions that pack them "
                                      "clash-free (rotamer-aware). Cavity-filling is modest for generic "
                                      "thermostability but powerful for CONFORMATIONAL locking (the RSV "
                                      "prefusion-F lesson) — geometric suggestion only, you judge which "
                                      "cavity matters. Cheap (no fold). Needs a folded construct or a "
                                      "loaded structure.")
        self._cav_scan_btn.triggered.connect(self._on_cavity_scan)
        # Salt-bridge scan — the fourth peer strategy in the SAME Stabilize ▾ menu. Assess existing
        # Asp/Glu↔Arg/Lys pairs + suggest novel complementary charge-pair sites. Cheap (reads
        # coordinates, no fold); works on a de-novo fold OR a loaded crystal/model, intra + inter-chain.
        self._sb_scan_btn = pro_menu.addAction("Find salt-bridge sites…")
        self._sb_scan_btn.setToolTip("FIND salt-bridge stabilization sites: assess existing Asp/Glu↔"
                                     "Arg/Lys pairs (closest carboxyl-O↔basic-N + burial) and rank NOVEL "
                                     "complementary charge-pair substitutions (geometry × burial, rotamer-"
                                     "aware, intra + inter-chain). Salt bridges are CONTEXT-DEPENDENT "
                                     "(favourable buried, marginal at the surface via desolvation) — "
                                     "geometric suggestion only, you judge the context. Cheap (no fold). "
                                     "Needs a folded construct or a loaded structure.")
        self._sb_scan_btn.triggered.connect(self._on_saltbridge_scan)
        self._sync_disulfide_menu_enabled()           # initial precondition state (greyed until ready)

        # Stage 4c: per-residue Cα deviation of the ACTIVE folded variant vs a seed-pinned
        # WT reference fold (same engine+target). Establishes the WT reference + noise floor
        # on first use for that combo (folds T; Boltz also folds the cross-seed floor set).
        self._dev_btn = QtWidgets.QPushButton("Deviation vs WT")
        self._dev_btn.setToolTip("Per-residue Cα deviation of the ACTIVE folded variant vs a "
                                 "seed-pinned WT reference fold (same engine). Floor-gated: "
                                 "residues within the noise floor stay neutral. First use for "
                                 "an engine folds the WT reference + its noise floor (cached).")
        self._dev_btn.clicked.connect(self._on_deviation_clicked)   # added to the View group below

        # Stage 3: structurally align the DE-NOVO construct's fold onto a chosen PDB,
        # SEQUENCE-INDEPENDENTLY (US-align, LOCAL-ONLY) — the case ChimeraX matchmaker can't
        # reach. Captures TM-score/RMSD and overlays the pLDDT-coloured fold on the reference.
        self._align_btn = QtWidgets.QPushButton("Align to PDB")
        self._align_btn.setToolTip("Structurally align the construct's FOLD onto a chosen PDB "
                                   "(sequence-independent, US-align LOCAL-ONLY). Captures TM-score "
                                   "+ RMSD and overlays the fold on the reference. De-novo only; "
                                   "fold the construct first.")
        self._align_btn.clicked.connect(self._on_align_clicked)   # added to the View group below
        self._mode_combo = QtWidgets.QComboBox()
        for m in all_modes():
            self._mode_combo.addItem(m.label, m.key)
        self._mode_combo.addItem("ddG (result)", _RESULT_DDG_MODE)   # S4a result-backed mode
        self._mode_combo.addItem("pLDDT (result)", _RESULT_PLDDT_MODE)  # S4b fold confidence
        self._mode_combo.addItem("Deviation vs WT", _RESULT_DEVIATION_MODE)  # S4c floor-gated dev
        self._mode_combo.addItem("Disulfide sites", _RESULT_DISULFIDE_MODE)  # Mode D engineerability
        self._mode_combo.addItem("Proline sites", _RESULT_PROLINE_MODE)      # proline stabilization
        self._mode_combo.addItem("Cavity sites", _RESULT_CAVITY_MODE)        # cavity-filling best-fill
        self._mode_combo.addItem("Salt-bridge sites", _RESULT_SALTBRIDGE_MODE)  # salt-bridge charge-pair
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        # ── View group: a segmented "pill" cluster (Fold ▾ · Align to PDB · Color · Deviation vs WT)
        #    — lighter-weight controls on a grouped translucent background, visually distinct from the
        #    Edit/Analysis buttons. PURE grouping + styling: every widget + handler is unchanged; only
        #    the parent (this pill instead of the flow row) and a flat/auto-raise look differ.
        self._view_group = QtWidgets.QFrame()
        self._view_group.setObjectName("workbenchViewGroup")
        self._view_group.setStyleSheet(
            "#workbenchViewGroup{background:rgba(127,127,127,0.14);border-radius:7px;}")
        _vg = QtWidgets.QHBoxLayout(self._view_group)
        _vg.setContentsMargins(7, 2, 7, 2)
        _vg.setSpacing(6)
        self._fold_menu_btn.setAutoRaise(True)        # lighter-weight, segmented look
        self._align_btn.setFlat(True)
        self._dev_btn.setFlat(True)
        _vg.addWidget(self._fold_menu_btn)
        _vg.addWidget(self._align_btn)
        _vg.addWidget(QtWidgets.QLabel("Color:"))
        _vg.addWidget(self._mode_combo)
        _vg.addWidget(self._dev_btn)
        bar.addWidget(self._view_group)
        # The flow layout left-aligns + wraps; no trailing stretch (a stretch item would consume the
        # row and stop wrapping). The container carries every pill — nothing is ever hidden.
        lay.addWidget(self._toolbar_container)

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        lay.addWidget(self._tabs)
        # Word-wrap + a tiny minimum so a long status string WRAPS (and the label yields) instead of
        # holding one line and forcing the panel/window minimum width (the other half of the floor).
        self._status = QtWidgets.QLabel("No structure loaded.")
        self._status.setWordWrap(True)
        self._status.setMinimumWidth(0)
        self._status.setStyleSheet("color:#888;padding:2px 6px;")
        lay.addWidget(self._status)

        # The PERSISTENT Disulfides results tab (whole-suite home) — a SEPARATE top-level widget the
        # window adds as a sibling tab (`gui_app` reads `self.disulfides_tab`). It survives top-level
        # tab-switches + panel rebuilds (the old modeless dialog vanished on focus loss; this does not).
        self._glow_state: Optional[dict] = None       # the prior disulfide GLOW (for non-destructive restore)
        self.disulfides_tab = DisulfidesResultsTab(
            on_highlight=self._highlight_disulfide_pair,
            on_declare=self._declare_disulfide_pair,
            on_estimate_ddg=self._estimate_ddg_pair,
            on_clear_glow=self._clear_disulfide_glow,
            on_add_to_basket=self._add_disulfide_to_basket)
        self.proline_tab = ProlineResultsTab(
            on_highlight=self._highlight_proline_residue,
            on_declare=self._declare_proline,
            on_estimate_ddg=self._estimate_proline_ddg,
            on_clear_glow=self._clear_disulfide_glow,        # the glow seam is shared (one _glow_state)
            on_show_existing=self._show_existing_prolines,
            on_add_to_basket=self._add_proline_to_basket)
        self.cavity_tab = CavityResultsTab(
            on_highlight=self._highlight_cavity_residue,
            on_declare=self._declare_cavity,
            on_estimate_ddg=self._estimate_cavity_ddg,
            on_clear_glow=self._clear_disulfide_glow,        # the glow seam is shared (one _glow_state)
            on_add_to_basket=self._add_cavity_to_basket)
        self.saltbridge_tab = SaltBridgeResultsTab(
            on_highlight=self._highlight_saltbridge_pair,    # pairwise glow (reuses the disulfide seam)
            on_declare=self._declare_saltbridge,
            on_estimate_ddg=self._estimate_saltbridge_ddg,
            on_clear_glow=self._clear_disulfide_glow,        # the glow seam is shared (one _glow_state)
            on_add_to_basket=self._add_saltbridge_to_basket)
        # The CROSS-STRATEGY design basket — the framework centerpiece (gui_app docks it on the right).
        self.design_basket = DesignBasketPanel(on_enact=self._enact_basket,
                                               on_chain_equiv=self._chain_equiv)

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
        self._glow_state = None                  # the 3D scene is being cleared/replaced → no glow to track
        self._sync_glow_clear_button()
        if getattr(self, "design_basket", None) is not None:
            self.design_basket.reset()           # the prior session's picks reference a gone design
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

    @staticmethod
    def _toolbar_vsep() -> "QtWidgets.QFrame":
        """A thin vertical separator for the wrapping toolbar (the flow-layout analog of
        `QToolBar.addSeparator`). A FRESH widget per call — a layout item can't be shared."""
        f = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.VLine)
        f.setFrameShadow(QtWidgets.QFrame.Sunken)
        return f

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
        self._update_active_indicator()       # repaint the panel-header active-model label
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
        self._sync_disulfide_menu_enabled()   # grey/enable the Disulfides actions by precondition
        self._status.setText(
            f"Workbench: {len(self._design.chains)} unique chain(s). Click a column to "
            f"select in 3D (all copies); add/edit variants; pick a color mode.")

    # ── toolbar actions ────────────────────────────────────────────────────────────
    def _cur_tab(self) -> Optional[_ChainDesignTab]:
        if getattr(self, "_tabs", None) is None:     # pre-`_tabs` __init__ calls (e.g. the gate sync)
            return None
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
        self._sync_disulfide_menu_enabled()   # the construct is now folded → enable Geometry/Fold-with-bond
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

    # ── Disulfide-suite handlers ─────────────────────────────────────────────────────
    # ── structure-source abstraction (§9 universal-disulfide convergence, input side) ──────────
    # B (geometry) / D (scan) / interface READ coordinates — they don't care whether the structure
    # is a DE-NOVO construct's fold OR a LOADED crystal/model. One helper yields the active design's
    # {cif_path, model_id} from EITHER source so the cheap read-modes work on a loaded PDB, not just
    # a de-novo fold (the reported symptom: these were greyed on a loaded PDB). The DEEP cross-chain
    # ESM+ΔΔG engineering tool (`disulfide_bridge`) stays the parallel escalation path, not merged.

    def _generated_assembly_for_model(self, mid) -> Optional[Dict[str, Any]]:
        """The generated bio-assembly record whose flat/normalized model is *mid*, or None — so the
        active-model indicator can label an INGESTED biological assembly distinctly from a plain AU."""
        if self._session is None:
            return None
        gen = getattr(self._session, "generated_assemblies", {}) or {}
        for rec in gen.values():
            if rec and str(rec.get("assembly_model_id")) == str(mid):
                return rec
        return None

    def _update_active_indicator(self) -> None:
        """Repaint the active-model header from the LIVE design (model id, member-chain count, and
        whether it is an ingested biological assembly). Best-effort; pre-`_active_lbl` calls no-op."""
        if getattr(self, "_active_lbl", None) is None:
            return
        if self._design is None:
            self._active_lbl.setText("Active: (no model loaded)")
            return
        mid = str(self._design.model_id)
        n_chains = sum(len(c.members) for c in self._design.chains.values())
        chains_word = "chain" if n_chains == 1 else "chains"
        asm = self._generated_assembly_for_model(mid)
        if asm:
            kind = asm.get("assembly_type") or f"assembly {asm.get('assembly_id')}"
            label = f"Assembly #{mid} ({kind})"
        else:
            label = f"#{mid}"
        self._active_lbl.setText(f"Active: {label} · {n_chains} {chains_word}")

    def _has_structure(self) -> bool:
        """True if the ACTIVE design has a structure to read disulfide geometry from — EITHER a
        de-novo construct that's been FOLDED (`cd.template_fold`) OR a LOADED crystal/model (source
        'structure', a live ChimeraX model id). The CHEAP gate predicate — no ChimeraX I/O (the
        actual save + fail-closed happens at action time in `_active_structure`)."""
        if self._design is None:
            return False
        cd = self._cur_tab().design if self._cur_tab() else None
        if cd is None:
            return False
        tf = cd.template_fold or {}
        if tf.get("cif_path") or tf.get("model_id"):
            return True                                        # de-novo folded (a fold on disk/in 3D)
        return self._design.source == "structure" and bool(cd.rep_model)   # loaded crystal/model

    def _active_structure(self) -> Optional[Dict[str, Any]]:
        """The active design's structure as ``{cif_path, model_id}`` from EITHER source, or None when
        there is no readable structure (fail-CLOSED — a closed/unsaveable loaded model returns None so
        the caller greys/no-ops instead of scanning absent or STALE coordinates).

        de-novo → the FOLD on `cd.template_fold` (its cif_path + the fold model id), unchanged.
        loaded  → the live ChimeraX model SAVED FRESH to a temp mmCIF on EVERY call (never cached on
                  the design). Re-saving here — rather than reading a stored path — is the staleness
                  guard: an edit / replace-at-same-id / close-reopen is always captured because the
                  file is re-written from the model's CURRENT state at read time. `model_id` is the
                  LIVE rendered id (what the user sees), NOT the temp copy — so 3D highlight lands on
                  the structure on screen."""
        cd = self._cur_tab().design if self._cur_tab() else None
        if cd is None or self._design is None:
            return None
        tf = cd.template_fold or {}
        if tf.get("cif_path") or tf.get("model_id"):
            cif = tf.get("cif_path")
            return {"cif_path": cif, "model_id": tf.get("model_id")} if cif else None
        if self._design.source != "structure":
            return None
        mid = cd.rep_model or self._design.model_id
        cif = self._save_loaded_model_cif(mid)
        return {"cif_path": cif, "model_id": str(mid)} if cif else None

    def _save_loaded_model_cif(self, mid) -> Optional[str]:
        """Save a LIVE ChimeraX model to a FRESH temp mmCIF for coordinate reading; None if the save
        fails or yields no file (model closed / not open → fail-CLOSED). ALWAYS deletes any prior file
        at the path and re-saves, so the file can NEVER serve a stale model's coordinates. mmCIF (not
        PDB) — the disulfide parser reads the `_atom_site` loop."""
        mid = str(mid).lstrip("#").strip()
        tmp = os.path.join(tempfile.gettempdir(), f"ss_struct_{mid.replace('.', '_')}.cif")
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)                                 # never serve a prior save's coordinates
        except OSError:
            pass
        try:
            self._c._run(f'save "{Path(tmp).as_posix()}" models #{mid}')
        except Exception as exc:
            self._status.setText(f"Could not read model #{mid} for disulfide analysis: {exc}")
            return None
        if not os.path.isfile(tmp):
            self._status.setText(f"Model #{mid} produced no structure file — is it still open?")
            return None
        return tmp

    def _save_structure_pdb(self, mid) -> Optional[str]:
        """Save a LIVE ChimeraX model to a FRESH temp **PDB** (for the legacy ΔΔG path: PyRosetta's
        `cleanATOM` is PDB-only, so the mmCIF from `_active_structure` won't do here). Same fresh-save
        discipline as `_save_loaded_model_cif` (delete-prior-then-save → never stale), fail-CLOSED to
        None when the model is closed/unsaveable so the caller never scores absent coordinates."""
        mid = str(mid).lstrip("#").strip()
        tmp = os.path.join(tempfile.gettempdir(), f"ss_ddg_{mid.replace('.', '_')}.pdb")
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)                                 # never serve a prior save's coordinates
        except OSError:
            pass
        try:
            self._c._run(f'save "{Path(tmp).as_posix()}" models #{mid}')
        except Exception as exc:
            self._status.setText(f"Could not read model #{mid} for ΔΔG estimate: {exc}")
            return None
        if not os.path.isfile(tmp):
            self._status.setText(f"Model #{mid} produced no PDB — is it still open?")
            return None
        return tmp

    def _from_aa_for(self, chain, resnum) -> Optional[str]:
        """The WT residue letter at (chain, resnum), recovered from the DESIGN (template_cells) of the
        cd that OWNS that chain — the cross-chain own-chain discipline (same as cross-chain Mode C):
        `_chain_to_cd` resolves each member to ITS OWN chain's cd, so an A-side and a B-side position
        read their from_aa from the RIGHT chain. The router VERIFIES this against the scored structure
        (fail-closed mismatch) — recovering here is design-intent, verified there is ground-truth."""
        cd = self._chain_to_cd().get(str(chain))
        if cd is None:
            return None
        for c in cd.template_cells:
            if (not c.is_gap) and c.resnum is not None and int(c.resnum) == int(resnum):
                return c.aa
        return None

    def build_disulfide_ddg_spec(self, pair: dict) -> Optional[dict]:
        """ΔΔG-escalation spec for ONE interface pair → the legacy ΔΔG bridge (refresh='disulfide_ddg').
        Saves the LIVE structure to PDB (PyRosetta needs PDB), recovers each member's WT residue with
        the own-chain discipline, and tags the source so the router gate can block a de-novo web upload
        and the readout can carry the de-novo two-layer caveat. None (caller explains) when there is no
        readable structure, a from_aa can't be recovered, or the PDB save fails — never a silent
        score of the wrong/absent mutation."""
        if self._design is None:
            return None
        src = self._active_structure()
        if src is None:
            return None
        cha, chb = pair_chains(pair)
        ra, rb = pair.get("resnum_a"), pair.get("resnum_b")
        if cha is None or chb is None or ra is None or rb is None:
            return None
        aa_a, aa_b = self._from_aa_for(cha, ra), self._from_aa_for(chb, rb)
        if aa_a is None or aa_b is None:
            return None
        pdb = self._save_structure_pdb(src["model_id"])
        if not pdb:
            return None
        source = "denovo" if self._design.source == "sequence" else "loaded"
        return {
            "tool": "disulfide_ddg_estimate",
            "tool_inputs": {"pdb_path": pdb,
                            "chain_a": str(cha), "resnum_a": int(ra), "from_aa_a": aa_a,
                            "chain_b": str(chb), "resnum_b": int(rb), "from_aa_b": aa_b,
                            "source": source},
            "user_input": f"[Workbench] estimate ΔΔG (legacy) for interface pair {pair_label(pair)}",
            "confidence": "low", "refresh": "disulfide_ddg",
            "_align_ukey": self._cur_cd_ukey(),
            "_ss_pair": {"chain_a": str(cha), "resnum_a": int(ra),
                         "chain_b": str(chb), "resnum_b": int(rb)},
        }

    def _estimate_ddg_pair(self, cd, pair: dict) -> None:
        """Escalate a geometric interface hit to an energetic ΔΔG read (legacy bridge). FOCUS the
        pair's cd tab first (so the structure/from_aa resolve against the RIGHT design), build the
        spec, and launch on the worker seam (long-running — minutes per pair — so it awaits off the
        UI thread via the standard launch path; NEVER auto-run across a scan)."""
        self._focus_tab_for_design(cd)
        spec = self.build_disulfide_ddg_spec(pair)
        if spec is None:
            self._status.setText("Can't estimate ΔΔG — needs a readable structure and recoverable WT "
                                 "residues for both positions (re-scan if the structure changed).")
            return
        self._status.setText(f"Estimating ΔΔG (legacy, uncalibrated) for {pair_label(pair)} — "
                             "X→C at both positions; minutes per pair…")
        self.launchRequested.emit(spec)

    def apply_disulfide_ddg_result(self, spec: dict, result: dict) -> None:
        """Attach the escalated ΔΔG onto the MATCHING interface pair (by chain+resnum from the spec)
        and re-render the I section (ΔΔG column + per-row caveat). Fail-LOUD: a failed/aborted estimate
        (e.g. from_aa-mismatch guard, de-novo web-upload block, backend unavailable) shows its reason;
        it never writes a fabricated number."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "disulfide_ddg_estimate"), None)
        if step is None:
            self._status.setText("ΔΔG estimate produced no result.")
            return
        if not step.get("success"):
            self._status.setText(f"ΔΔG estimate: {step.get('error') or 'failed'}")
            return
        cd = self._scan_target_cd(spec)
        want = (spec or {}).get("_ss_pair") or {}
        data = step.get("data") or {}
        pairs = ((cd.disulfide_interface_scan or {}).get("pairs") if cd else None) or []
        target = next((p for p in pairs
                       if str(pair_chains(p)[0]) == str(want.get("chain_a"))
                       and str(pair_chains(p)[1]) == str(want.get("chain_b"))
                       and p.get("resnum_a") == want.get("resnum_a")
                       and p.get("resnum_b") == want.get("resnum_b")), None)
        if target is None:
            self._status.setText("ΔΔG came back but its interface pair is no longer listed (re-scan).")
            return
        target.update({"ddg_a": data.get("ddg_a"), "ddg_b": data.get("ddg_b"),
                       "ddg_mean": data.get("ddg_mean"), "ddg_backend": data.get("backend"),
                       "ddg_source": data.get("source")})
        self._persist()
        self.disulfides_tab.populate_interface(cd, cd.disulfide_interface_scan)
        self._status.setText(step.get("summary") or "ΔΔG estimate complete.")

    def _sync_disulfide_menu_enabled(self) -> None:
        """Enable each Disulfides action by its REAL precondition so an unavailable action is VISIBLY
        GREYED (far clearer than a silent no-op). Assess-existing (A) + Declare-bonds (C) fold a
        SEQUENCE → DE-NOVO constructs only (C also needs the construct folded). Measure-geometry (B) +
        Find-sites (D) + Interface READ a structure → enabled on EITHER a folded de-novo construct OR a
        LOADED crystal/model (the structure-source abstraction); interface additionally needs ≥2 chains.
        Synced on tab-change / render / fold / load."""
        if getattr(self, "_ss_discover_btn", None) is None:
            return
        denovo = self._design is not None and self._design.source == "sequence"
        cd = self._cur_tab().design if self._cur_tab() else None
        denovo_folded = bool(denovo and cd and (cd.template_fold or {}).get("model_id"))
        self._ss_discover_btn.setEnabled(denovo)               # A: folds N seeds → needs a construct
        self._ss_constrain_btn.setEnabled(denovo_folded)       # C: folds a variant against the T-fold
        has_struct = self._has_structure()                     # de-novo folded OR loaded crystal/model
        self._ss_geometry_btn.setEnabled(has_struct)           # B: reads the existing structure coords
        self._ss_scan_btn.setEnabled(has_struct)               # D: reads the backbone (no fold)
        # Interface scan needs ≥2 chains (an inter-chain bond is meaningless on a monomer) — greyed
        # (not a silent no-op) on a single-chain structure, de-novo OR loaded alike.
        if getattr(self, "_ss_interface_btn", None) is not None:
            n_chains = sum(len(c.members) for c in self._design.chains.values()) if self._design else 0
            self._ss_interface_btn.setEnabled(has_struct and n_chains >= 2)
        if getattr(self, "_pro_scan_btn", None) is not None:
            self._pro_scan_btn.setEnabled(has_struct)          # reads the backbone (no fold), like D
        if getattr(self, "_cav_scan_btn", None) is not None:
            self._cav_scan_btn.setEnabled(has_struct)          # reads coordinates (no fold), like proline
        if getattr(self, "_sb_scan_btn", None) is not None:
            self._sb_scan_btn.setEnabled(has_struct)           # reads coordinates (no fold), like cavity

    def _on_disulfide_discover(self) -> None:
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Existing-cysteine assessment needs a de-novo construct — "
                                 "Add sequence first.")
            return
        spec = self.disulfide_discovery_launch_spec()
        if spec is None:                                       # only None when not a de-novo construct
            self._status.setText("Existing-cysteine assessment needs a de-novo construct with "
                                 "cysteines — Add sequence first.")
            return
        self._status.setText("Assessing existing Cys pairs — folding N unconstrained seeds (compute)…")
        self.launchRequested.emit(spec)

    def _on_disulfide_geometry(self) -> None:
        spec = self.disulfide_geometry_launch_spec()
        if spec is None:
            self._status.setText("Measure pair geometry needs a readable structure — fold the "
                                 "construct, or check the loaded model is still open.")
            return
        self.launchRequested.emit(spec)

    def _on_disulfide_scan(self) -> None:
        spec = self.disulfide_scan_launch_spec()
        if spec is None:
            self._status.setText("Find engineerable sites needs a readable structure — fold the "
                                 "construct, or check the loaded model is still open.")
            return
        self._status.setText("Scanning the backbone for engineerable disulfide sites "
                             "(geometric compatibility only — a starting point)…")
        self.launchRequested.emit(spec)

    def _on_proline_scan(self) -> None:
        spec = self.proline_scan_launch_spec()
        if spec is None:
            self._status.setText("Proline scan needs a readable structure — fold the construct, or "
                                 "check the loaded model is still open.")
            return
        self._status.setText("Scanning for proline-stabilization sites (backbone φ/ψ + H-bond, "
                             "geometric suggestion only — a starting point)…")
        self.launchRequested.emit(spec)

    def _on_cavity_scan(self) -> None:
        spec = self.cavity_scan_launch_spec()
        if spec is None:
            self._status.setText("Cavity scan needs a readable structure — fold the construct, or "
                                 "check the loaded model is still open.")
            return
        self._status.setText("Scanning for cavity-filling sites (internal voids + rotamer-aware fills, "
                             "geometric suggestion only — you judge which cavity matters)…")
        self.launchRequested.emit(spec)

    def _on_saltbridge_scan(self) -> None:
        spec = self.saltbridge_scan_launch_spec()
        if spec is None:
            self._status.setText("Salt-bridge scan needs a readable structure — fold the construct, or "
                                 "check the loaded model is still open.")
            return
        self._status.setText("Scanning for salt-bridge sites (existing Asp/Glu↔Arg/Lys pairs + novel "
                             "charge-pair geometry × burial; context-dependent — you judge the context)…")
        self.launchRequested.emit(spec)

    def _on_disulfide_interface_scan(self) -> None:
        spec = self.disulfide_interface_scan_launch_spec()
        if spec is None:
            self._status.setText("Find interface sites needs a readable MULTIMER (≥2 chains) — fold "
                                 "the construct as an assembly, or load a multi-chain structure.")
            return
        self._status.setText("Scanning the chain–chain interface for inter-subunit disulfide sites "
                             "(geometric compatibility only — a starting point)…")
        self.launchRequested.emit(spec)

    def _on_disulfide_constrain(self) -> None:
        if self._design is None or self._design.source != "sequence":
            self._status.setText("Declared cysteine bonds are for de-novo constructs.")
            return
        cd = self._cur_tab().design if self._cur_tab() else None
        if cd is None or not (cd.template_fold or {}).get("model_id"):
            self._status.setText("Fold the construct first (the bond folds a variant against the T-fold).")
            return
        text, ok = QtWidgets.QInputDialog.getText(
            self, "Declare cysteine bonds and fold",
            "Residue pairs to bond (introduces Cys at each), e.g. '12-45, 7-33':")
        if not ok or not (text or "").strip():
            return
        pairs = self._parse_bond_pairs(text)
        if not pairs:
            self._status.setText("Enter one or more residue pairs, e.g. '12-45' or '12-45, 7-33'.")
            return
        spec = self.build_disulfide_introduce_spec(pairs, engine="boltz")
        if spec is None:
            self._status.setText("Could not declare those bonds — check each is a valid, distinct "
                                 "residue pair in the construct.")
            return
        self._persist()
        _pp = ", ".join(f"{a}–{b}" for a, b in pairs)
        self._status.setText(f"Introduced Cys for {_pp}; folding WITH the declared bond(s) "
                             f"(biases toward them, geometry measured on the result)…")
        self.launchRequested.emit(spec)

    @staticmethod
    def _parse_bond_pairs(text: str) -> List[tuple]:
        """Parse '12-45, 7-33' (or '12 45' for a single pair) → [(12,45),(7,33)]. A pair is two
        residue numbers joined by '-'/'/'/':' or whitespace; comma/semicolon separate pairs.
        Returns [] on any malformed token (fail-closed — the caller reports)."""
        out: List[tuple] = []
        chunks = [c for c in re.split(r"[,;]", text.strip()) if c.strip()]
        if len(chunks) == 1 and re.fullmatch(r"\s*\d+\s+\d+\s*", chunks[0]):
            chunks = [chunks[0]]                              # a single "12 45" pair
        for ch in chunks:
            nums = re.findall(r"\d+", ch)
            if len(nums) != 2:
                return []
            out.append((int(nums[0]), int(nums[1])))
        return out

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
        elif self._mode_key == _RESULT_DISULFIDE_MODE:
            # the load-bearing CAVEAT rides ON the heatmap (the most over-read-prone surface) — a
            # glowing residue is a geometrically-viable ENGINEERING site, NOT a promise the mutation
            # is tolerated / the protein still folds / the bond forms.
            cav = (tab.design.disulfide_scan or {}).get("caveat") or ""
            self._status.setText("Disulfide sites heatmap (each residue = its BEST-partner score; a "
                                 "navigational index into the ranked list). " + cav)
        elif self._mode_key == _RESULT_SALTBRIDGE_MODE:
            cav = (tab.design.saltbridge_scan or {}).get("caveat") or ""
            self._status.setText("Salt-bridge sites heatmap (each residue = its best charge-pair score; "
                                 "a navigational index into the ranked tables). " + cav)
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
        self._sync_disulfide_menu_enabled()    # grey/enable the Disulfides actions by precondition
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

    def _active_disulfide_scan(self, tab: _ChainDesignTab) -> Dict[int, float]:
        """{author_resnum: best-partner score} for the active cd's engineering scan (Mode D), rep
        chain only. Empty when no scan has run. The heatmap colours each residue by its BEST
        available partner — a NAVIGATIONAL INDEX into the ranked pair-list (the source of truth)."""
        cd = tab.design
        scan = (cd.disulfide_scan or {})
        best = (scan.get("best_partner") or {})
        # best_partner is {chain: {resnum: score}}; the fold numbers 1..N, but the scan keyed by the
        # CIF's auth_seq_id which == the construct's author resnums for the rep chain.
        chain_best = best.get(cd.rep_chain)
        if chain_best is None and len(best) == 1:
            chain_best = next(iter(best.values()))       # monomer: the only chain
        return {int(rn): float(sc) for rn, sc in (chain_best or {}).items()}

    def _disulfide_scan_panel_hex(self, tab: _ChainDesignTab) -> Dict[int, str]:
        """{author_resnum: hex} for the engineering-scan heatmap (best-partner → pale→gold)."""
        out: Dict[int, str] = {}
        for rn, sc in self._active_disulfide_scan(tab).items():
            hexc = disulfide_compat_color(sc)
            if hexc:
                out[rn] = hexc
        return out

    def _active_proline_scan(self, tab: _ChainDesignTab) -> Dict[int, float]:
        """{author_resnum: proline-favourability score} for the active cd's proline scan, rep chain
        only. Empty when no scan has run. The heatmap colours each residue by its X→Pro score — the
        NAVIGATIONAL INDEX into the ranked candidate list (the source of truth)."""
        cd = tab.design
        best = ((cd.proline_scan or {}).get("best_partner") or {})
        chain_best = best.get(cd.rep_chain)
        if chain_best is None and len(best) == 1:
            chain_best = next(iter(best.values()))
        return {int(rn): float(sc) for rn, sc in (chain_best or {}).items()}

    def _proline_scan_panel_hex(self, tab: _ChainDesignTab) -> Dict[int, str]:
        """{author_resnum: hex} for the proline heatmap (favourability → pale→magenta)."""
        out: Dict[int, str] = {}
        for rn, sc in self._active_proline_scan(tab).items():
            hexc = proline_compat_color(sc)
            if hexc:
                out[rn] = hexc
        return out

    # ── Proline-stabilization scan: launch / apply / highlight / declare / ΔΔG ─────────────
    def proline_scan_launch_spec(self) -> Optional[dict]:
        """Proline-scan spec — per-residue X→Pro scan of the active design's EXISTING structure (de-novo
        fold OR a LOADED crystal/model, via `_active_structure`). CHEAP (reads coordinates; NEVER
        folds). None until there is a structure to read."""
        src = self._active_structure()
        if src is None:
            return None
        return {
            "tool": "proline_scan", "tool_inputs": {"cif_path": src["cif_path"]},
            "user_input": "[Workbench] find proline-stabilization sites (φ/ψ + H-bond scan, this structure)",
            "confidence": "high", "refresh": "proline_scan",           # cheap + deterministic → no gate
            "_align_ukey": self._cur_cd_ukey(),
        }

    def apply_proline_scan_result(self, spec: dict, result: dict) -> None:
        """Store the proline scan on the active cd (ranked list = source of truth + best-score heatmap
        map + existing prolines) and AUTO-SURFACE the 'Proline sites' colour mode. The geometric-only
        CAVEAT rides on the tab + status."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "proline_scan"), None)
        if step is None or not step.get("success"):
            self._status.setText(f"Proline scan failed: {step.get('error') if step else 'no result'}")
            return
        data = step.get("data") or {}
        cd = self._scan_target_cd(spec)
        if cd is None:
            return
        cd.proline_scan = {"candidates": data.get("candidates") or [],
                           "best_partner": data.get("best_partner") or {},
                           "existing": data.get("existing") or [],
                           "caveat": data.get("caveat")}
        self._persist()
        if cd.proline_scan["candidates"]:
            self._select_result_mode(_RESULT_PROLINE_MODE)             # AUTO-SURFACE the heatmap
            tab = self._cur_tab()
            if tab is not None:
                self._apply_color_to(tab)
                self._push_3d_color(tab)
        self.proline_tab.populate(cd, cd.proline_scan)
        self._status.setText(step.get("summary") or "Proline scan complete.")

    def _proline_residue_spec(self, cd, cand: dict):
        """(mid, spec) for a candidate residue on the design's LIVE structure, or (None, None) when
        there is no real rendered id (an unfolded de-novo construct's synthetic id is not a target)."""
        mid = (cd.template_fold or {}).get("model_id") or cd.rep_model
        if mid and str(mid).startswith("denovo-"):
            mid = None
        ch, pos = (cand.get("chain") or cd.rep_chain), cand.get("position")
        if not mid or pos is None:
            return None, None
        return str(mid), f"#{mid}/{ch}:{pos}"

    def _highlight_proline_residue(self, cd, cand: dict) -> None:
        """Glow ONE proline candidate residue — the single-residue analog of the disulfide pair glow,
        through the SAME `_glow_state`/`_consume_glow_restore` seam (restore prior glow first so it
        never stacks; the Clear-view control + any colour-mode switch clear it)."""
        mid, spec = self._proline_residue_spec(cd, cand)
        if mid is None:
            self._glow_state = None
            self._sync_glow_clear_button()
            return
        hue = self._GLOW_HUE
        apply = [
            f"show {spec} atoms", f"style {spec} sphere", f"color {spec} {hue} target a",
            f"transparency #{mid} 70 target c", f"transparency {spec} 0 target c",
            f"graphics selection color {hue}", "graphics selection width 5", f"select {spec}",
        ]
        self._apply_or_toggle_glow({"mid": mid, "both": spec}, apply)   # re-click toggles it off

    def _show_existing_prolines(self, cd) -> None:
        """Highlight the EXISTING prolines in 3D (the 'see what's already there' design-context half) —
        a simple distinct colouring, NOT the glow spotlight (this is context, not a candidate)."""
        existing = ((cd.proline_scan or {}).get("existing")) or []
        mid = (cd.template_fold or {}).get("model_id") or cd.rep_model
        if not mid or str(mid).startswith("denovo-") or not existing:
            self._status.setText("No existing prolines to show (or no live model).")
            return
        specs = " ".join(f"#{mid}/{ch}:{rn}" for ch, rn in existing)
        self._run_commands_bg([f"color {specs} medium purple", f"show {specs} atoms",
                               f"style {specs} stick"])
        self._status.setText(f"Existing prolines ({len(existing)}) shown in purple.")

    def _declare_proline(self, cd, cand: dict) -> None:
        """Substitute the selected residue to proline on a NEW variant — the validation path is the
        EXISTING fold/deviation flow (proline folds NATIVELY, no constraints): one click does the
        substitution; the user Folds the variant to validate (re-fold + compare). No new fold-
        orchestration machinery (the locked decision)."""
        tab = self._focus_tab_for_design(cd)
        if tab is None or self._design is None:
            self._status.setText("Can't substitute — the candidate's chain tab isn't available.")
            return
        col = self._col_for_resnum(cd, cand.get("position"))
        if col is None:
            self._status.setText(f"Can't substitute — residue {cand.get('position')} not on the axis.")
            return
        vid = self._design.new_variant_id()
        cd.add_variant(vid)
        try:
            cd.edit_variant(vid, col, "P")
        except Exception as exc:
            self._status.setText(f"Substitution failed: {type(exc).__name__}: {exc}")
            return
        self._after_variant_edit(
            tab, vid, f"{vid}: {cand.get('from_aa','')}{cand.get('position')}→P substituted. "
                      f"Fold this variant to VALIDATE (re-fold + deviation — proline folds natively, "
                      f"no constraint).")

    def _estimate_proline_ddg(self, cd, cand: dict) -> None:
        """X→Pro ΔΔG escalation for ONE candidate (the disulfide-escalation pattern, to_aa='P'). FOCUS
        the cd's tab so from_aa resolves against the RIGHT design, build the spec, launch on the worker
        seam (long-running; never auto-run across a scan)."""
        self._focus_tab_for_design(cd)
        spec = self.build_proline_ddg_spec(cand)
        if spec is None:
            self._status.setText("Can't estimate ΔΔG — needs a readable structure and a recoverable WT "
                                 "residue (re-scan if the structure changed).")
            return
        self._status.setText(f"Estimating ΔΔG (legacy, uncalibrated) for "
                             f"{cand.get('from_aa','')}{cand.get('position')}P — minutes…")
        self.launchRequested.emit(spec)

    def build_proline_ddg_spec(self, cand: dict) -> Optional[dict]:
        """ΔΔG-escalation spec for ONE proline candidate → the legacy ΔΔG bridge (refresh='proline_ddg').
        Saves the LIVE structure to PDB (PyRosetta needs PDB), recovers the WT residue own-chain, tags
        the source so the router gate blocks a de-novo web upload. None when there's no readable
        structure / from_aa can't be recovered / PDB save fails — never a silent wrong-mutation score."""
        if self._design is None:
            return None
        src = self._active_structure()
        if src is None:
            return None
        ch, pos = (cand.get("chain") or self._cur_tab().design.rep_chain), cand.get("position")
        if ch is None or pos is None:
            return None
        aa = self._from_aa_for(ch, pos)
        if aa is None:
            return None
        pdb = self._save_structure_pdb(src["model_id"])
        if not pdb:
            return None
        source = "denovo" if self._design.source == "sequence" else "loaded"
        return {
            "tool": "proline_ddg_estimate",
            "tool_inputs": {"pdb_path": pdb, "chain": str(ch), "resnum": int(pos),
                            "from_aa": aa, "source": source},
            "user_input": f"[Workbench] estimate ΔΔG (legacy) for {aa}{pos}P",
            "confidence": "low", "refresh": "proline_ddg",
            "_align_ukey": self._cur_cd_ukey(),
            "_pro_cand": {"chain": str(ch), "position": int(pos)},
        }

    def apply_proline_ddg_result(self, spec: dict, result: dict) -> None:
        """Attach the escalated X→Pro ΔΔG onto the MATCHING candidate (by chain+position) and re-render
        the proline table. Fail-LOUD: a failed/aborted estimate shows its reason, never a fabricated
        number."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "proline_ddg_estimate"), None)
        if step is None:
            self._status.setText("ΔΔG estimate produced no result.")
            return
        if not step.get("success"):
            self._status.setText(f"ΔΔG estimate: {step.get('error') or 'failed'}")
            return
        cd = self._scan_target_cd(spec)
        want = (spec or {}).get("_pro_cand") or {}
        data = step.get("data") or {}
        cands = ((cd.proline_scan or {}).get("candidates") if cd else None) or []
        target = next((c for c in cands if str(c.get("chain")) == str(want.get("chain"))
                       and c.get("position") == want.get("position")), None)
        if target is None:
            self._status.setText("ΔΔG came back but its candidate is no longer listed (re-scan).")
            return
        target.update({"ddg": data.get("ddg"), "ddg_backend": data.get("backend"),
                       "ddg_source": data.get("source")})
        self._persist()
        self.proline_tab.populate(cd, cd.proline_scan)
        self._status.setText(step.get("summary") or "ΔΔG estimate complete.")

    # ── Cavity-filling scan: launch / apply / highlight / declare / ΔΔG (the proline-mode shape) ──
    def _active_cavity_scan(self, tab: _ChainDesignTab) -> Dict[int, float]:
        """{author_resnum: best cavity-fill score} for the active cd's cavity scan, rep chain only.
        Empty when no scan has run. The heatmap colours each lining residue by its best fill score —
        the NAVIGATIONAL INDEX into the ranked candidate list (the source of truth)."""
        cd = tab.design
        best = ((cd.cavity_scan or {}).get("best_partner") or {})
        chain_best = best.get(cd.rep_chain)
        if chain_best is None and len(best) == 1:
            chain_best = next(iter(best.values()))
        return {int(rn): float(sc) for rn, sc in (chain_best or {}).items()}

    def _cavity_scan_panel_hex(self, tab: _ChainDesignTab) -> Dict[int, str]:
        """{author_resnum: hex} for the cavity heatmap (best-fill → teal→gold)."""
        out: Dict[int, str] = {}
        for rn, sc in self._active_cavity_scan(tab).items():
            hexc = cavity_compat_color(sc)
            if hexc:
                out[rn] = hexc
        return out

    def cavity_scan_launch_spec(self) -> Optional[dict]:
        """Cavity-scan spec — detect internal voids + rank fills on the active design's EXISTING
        structure (de-novo fold OR a LOADED crystal/model, via `_active_structure`). CHEAP (reads
        coordinates; NEVER folds). None until there is a structure to read."""
        src = self._active_structure()
        if src is None:
            return None
        return {
            "tool": "cavity_scan", "tool_inputs": {"cif_path": src["cif_path"]},
            "user_input": "[Workbench] find cavity-filling sites (internal-void detection + rotamer fills)",
            "confidence": "high", "refresh": "cavity_scan",           # cheap + deterministic → no gate
            "_align_ukey": self._cur_cd_ukey(),
        }

    def apply_cavity_scan_result(self, spec: dict, result: dict) -> None:
        """Store the cavity scan on the active cd (ranked fills = source of truth + best-score heatmap
        map + the per-void summary) and AUTO-SURFACE the 'Cavity sites' colour mode. The geometric-only
        context-dependent CAVEAT rides on the tab + status."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "cavity_scan"), None)
        if step is None or not step.get("success"):
            self._status.setText(f"Cavity scan failed: {step.get('error') if step else 'no result'}")
            return
        data = step.get("data") or {}
        cd = self._scan_target_cd(spec)
        if cd is None:
            return
        cd.cavity_scan = {"candidates": data.get("candidates") or [],
                          "best_partner": data.get("best_partner") or {},
                          "cavities": data.get("cavities") or [],
                          "caveat": data.get("caveat")}
        self._persist()
        if cd.cavity_scan["candidates"]:
            self._select_result_mode(_RESULT_CAVITY_MODE)             # AUTO-SURFACE the heatmap
            tab = self._cur_tab()
            if tab is not None:
                self._apply_color_to(tab)
                self._push_3d_color(tab)
        self.cavity_tab.populate(cd, cd.cavity_scan)
        self._status.setText(step.get("summary") or "Cavity scan complete.")

    def _highlight_cavity_residue(self, cd, cand: dict) -> None:
        """Glow ONE cavity-fill candidate residue — through the SAME `_glow_state` seam as the proline
        single-residue glow (restore the prior glow first so it never stacks)."""
        mid, spec = self._proline_residue_spec(cd, cand)         # same (mid, spec) resolver — residue glow
        if mid is None:
            self._glow_state = None
            self._sync_glow_clear_button()
            return
        hue = self._GLOW_HUE
        apply = [
            f"show {spec} atoms", f"style {spec} sphere", f"color {spec} {hue} target a",
            f"transparency #{mid} 70 target c", f"transparency {spec} 0 target c",
            f"graphics selection color {hue}", "graphics selection width 5", f"select {spec}",
        ]
        self._apply_or_toggle_glow({"mid": mid, "both": spec}, apply)   # re-click toggles it off

    def _declare_cavity(self, cd, cand: dict) -> None:
        """Substitute the selected residue to its FILL residue on a NEW variant — the validation path is
        the EXISTING fold/deviation flow (the fill folds NATIVELY, no constraints): one click does the
        substitution; the user Folds the variant to validate (re-fold + compare). Same seam as the
        proline declare, but the target is the candidate's `to_aa` (variable), not a fixed 'P'."""
        tab = self._focus_tab_for_design(cd)
        if tab is None or self._design is None:
            self._status.setText("Can't substitute — the candidate's chain tab isn't available.")
            return
        col = self._col_for_resnum(cd, cand.get("position"))
        if col is None:
            self._status.setText(f"Can't substitute — residue {cand.get('position')} not on the axis.")
            return
        to_aa = cand.get("to_aa")
        if not to_aa:
            self._status.setText("Can't substitute — the fill has no target residue.")
            return
        vid = self._design.new_variant_id()
        cd.add_variant(vid)
        try:
            cd.edit_variant(vid, col, to_aa)
        except Exception as exc:
            self._status.setText(f"Substitution failed: {type(exc).__name__}: {exc}")
            return
        self._after_variant_edit(
            tab, vid, f"{vid}: {cand.get('from_aa','')}{cand.get('position')}→{to_aa} substituted "
                      f"(cavity fill). Fold this variant to VALIDATE (re-fold + deviation — the fill "
                      f"folds natively, no constraint).")

    def _estimate_cavity_ddg(self, cd, cand: dict) -> None:
        """Cavity-fill ΔΔG escalation for ONE candidate (the disulfide/proline-escalation pattern, with
        the candidate's `to_aa`). FOCUS the cd's tab so from_aa resolves against the RIGHT design,
        build the spec, launch on the worker seam (long-running; never auto-run across a scan)."""
        self._focus_tab_for_design(cd)
        spec = self.build_cavity_ddg_spec(cand)
        if spec is None:
            self._status.setText("Can't estimate ΔΔG — needs a readable structure and a recoverable WT "
                                 "residue (re-scan if the structure changed).")
            return
        self._status.setText(f"Estimating ΔΔG (legacy, uncalibrated) for "
                             f"{cand.get('from_aa','')}{cand.get('position')}{cand.get('to_aa','')} — minutes…")
        self.launchRequested.emit(spec)

    def build_cavity_ddg_spec(self, cand: dict) -> Optional[dict]:
        """ΔΔG-escalation spec for ONE cavity-fill candidate → the legacy ΔΔG bridge
        (refresh='cavity_ddg'). Saves the LIVE structure to PDB (PyRosetta needs PDB), recovers the WT
        residue own-chain, carries the candidate's `to_aa` (variable target), tags the source so the
        router gate blocks a de-novo web upload. None when there's no readable structure / from_aa can't
        be recovered / no target / PDB save fails — never a silent wrong-mutation score."""
        if self._design is None:
            return None
        src = self._active_structure()
        if src is None:
            return None
        ch, pos = (cand.get("chain") or self._cur_tab().design.rep_chain), cand.get("position")
        to_aa = cand.get("to_aa")
        if ch is None or pos is None or not to_aa:
            return None
        aa = self._from_aa_for(ch, pos)
        if aa is None:
            return None
        pdb = self._save_structure_pdb(src["model_id"])
        if not pdb:
            return None
        source = "denovo" if self._design.source == "sequence" else "loaded"
        return {
            "tool": "cavity_ddg_estimate",
            "tool_inputs": {"pdb_path": pdb, "chain": str(ch), "resnum": int(pos),
                            "from_aa": aa, "to_aa": str(to_aa), "source": source},
            "user_input": f"[Workbench] estimate ΔΔG (legacy) for {aa}{pos}{to_aa}",
            "confidence": "low", "refresh": "cavity_ddg",
            "_align_ukey": self._cur_cd_ukey(),
            "_cav_cand": {"chain": str(ch), "position": int(pos)},
        }

    def apply_cavity_ddg_result(self, spec: dict, result: dict) -> None:
        """Attach the escalated fill ΔΔG onto the MATCHING candidate (by chain+position) and re-render
        the cavity table. Fail-LOUD: a failed/aborted estimate shows its reason, never a fabricated
        number."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "cavity_ddg_estimate"), None)
        if step is None:
            self._status.setText("ΔΔG estimate produced no result.")
            return
        if not step.get("success"):
            self._status.setText(f"ΔΔG estimate: {step.get('error') or 'failed'}")
            return
        cd = self._scan_target_cd(spec)
        want = (spec or {}).get("_cav_cand") or {}
        data = step.get("data") or {}
        cands = ((cd.cavity_scan or {}).get("candidates") if cd else None) or []
        target = next((c for c in cands if str(c.get("chain")) == str(want.get("chain"))
                       and c.get("position") == want.get("position")), None)
        if target is None:
            self._status.setText("ΔΔG came back but its candidate is no longer listed (re-scan).")
            return
        target.update({"ddg": data.get("ddg"), "ddg_backend": data.get("backend"),
                       "ddg_source": data.get("source")})
        self._persist()
        self.cavity_tab.populate(cd, cd.cavity_scan)
        self._status.setText(step.get("summary") or "ΔΔG estimate complete.")

    # ── Salt-bridge scan: launch / apply / heatmap / pair-glow / basket / declare / ΔΔG ──────────
    def saltbridge_scan_launch_spec(self) -> Optional[dict]:
        """Salt-bridge-scan spec — assess existing + suggest novel charge pairs on the active design's
        EXISTING structure (de-novo fold OR a LOADED crystal/model, via `_active_structure`). CHEAP
        (reads coordinates; NEVER folds). None until there is a structure to read."""
        src = self._active_structure()
        if src is None:
            return None
        return {
            "tool": "saltbridge_scan", "tool_inputs": {"cif_path": src["cif_path"]},
            "user_input": "[Workbench] find salt-bridge sites (existing pairs + novel charge-pair scan, this structure)",
            "confidence": "high", "refresh": "saltbridge_scan",        # cheap + deterministic → no gate
            "_align_ukey": self._cur_cd_ukey(),
        }

    def apply_saltbridge_scan_result(self, spec: dict, result: dict) -> None:
        """Store the salt-bridge scan on the active cd (existing + novel ranked lists = source of truth +
        best-score heatmap map) and AUTO-SURFACE the 'Salt-bridge sites' colour mode. The context-
        dependent desolvation CAVEAT rides on the tab + status."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "saltbridge_scan"), None)
        if step is None or not step.get("success"):
            self._status.setText(f"Salt-bridge scan failed: {step.get('error') if step else 'no result'}")
            return
        data = step.get("data") or {}
        cd = self._scan_target_cd(spec)
        if cd is None:
            return
        cd.saltbridge_scan = {"existing": data.get("existing") or [],
                              "novel": data.get("novel") or [],
                              "best_partner": data.get("best_partner") or {},
                              "caveat": data.get("caveat")}
        self._persist()
        if cd.saltbridge_scan["best_partner"]:
            self._select_result_mode(_RESULT_SALTBRIDGE_MODE)          # AUTO-SURFACE the heatmap
            tab = self._cur_tab()
            if tab is not None:
                self._apply_color_to(tab)
                self._push_3d_color(tab)
        self.saltbridge_tab.populate(cd, cd.saltbridge_scan)
        self._status.setText(step.get("summary") or "Salt-bridge scan complete.")

    def _active_saltbridge_scan(self, tab: _ChainDesignTab) -> Dict[int, float]:
        """{author_resnum: best salt-bridge score} for the active cd's scan, rep chain only. Empty when
        no scan has run. The heatmap colours each residue by its best charge-pair score — the
        NAVIGATIONAL INDEX into the ranked tables (the source of truth)."""
        cd = tab.design
        best = ((cd.saltbridge_scan or {}).get("best_partner") or {})
        chain_best = best.get(cd.rep_chain)
        if chain_best is None and len(best) == 1:
            chain_best = next(iter(best.values()))
        return {int(rn): float(sc) for rn, sc in (chain_best or {}).items()}

    def _saltbridge_scan_panel_hex(self, tab: _ChainDesignTab) -> Dict[int, str]:
        """{author_resnum: hex} for the salt-bridge heatmap (best score → pale→electric-blue)."""
        out: Dict[int, str] = {}
        for rn, sc in self._active_saltbridge_scan(tab).items():
            hexc = saltbridge_compat_color(sc)
            if hexc:
                out[rn] = hexc
        return out

    def _highlight_saltbridge_pair(self, cd, pair: dict) -> None:
        """Glow a salt-bridge pair's two members — REUSES the disulfide pair-glow seam (same two-chain
        pair shape), through the shared `_glow_state` (restore-then-apply, never stacking). Works for an
        intra-chain pair AND a cross-chain one (`_disulfide_pair_specs` reads chain_a/chain_b)."""
        apply = self._disulfide_scan_highlight_commands(cd, pair)
        mid, _a, _b, both = self._disulfide_pair_specs(cd, pair)
        new_state = {"mid": mid, "both": both} if (mid is not None and apply) else None
        self._apply_or_toggle_glow(new_state, apply)   # re-clicking the same pair toggles it off

    def _add_saltbridge_to_basket(self, cd, cand: dict) -> None:
        """Normalize a NOVEL salt-bridge candidate → a basket entry (TWO positions → the complementary
        charged residues). The candidate carries from_aa + to_aa for BOTH members (basket-aware-shaped,
        unlike disulfide which recovers from_aa) — so the 2-substitution entry drops straight in."""
        ca, cb = pair_chains(cand)
        ca, cb = (ca or cd.rep_chain), (cb or cd.rep_chain)
        ra, rb = cand.get("resnum_a"), cand.get("resnum_b")
        ta, tb = cand.get("to_aa_a"), cand.get("to_aa_b")
        aa_a, aa_b = cand.get("from_aa_a"), cand.get("from_aa_b")
        if None in (ra, rb) or not ta or not tb or aa_a is None or aa_b is None:
            self._status.setText("Can't add — the candidate is missing residue/target info (re-scan).")
            return
        bur = ("buried" if cand.get("buried") else "surface" if cand.get("buried") is False else "burial n/a")
        entry = {
            "cls": "Salt bridge", "score": float(cand.get("score", 0.0)),
            "subs": [{"chain": str(ca), "position": int(ra), "from_aa": aa_a, "to_aa": str(ta)},
                     {"chain": str(cb), "position": int(rb), "from_aa": aa_b, "to_aa": str(tb)}],
            "metrics_text": (f"O–N {cand.get('best_on')}Å Cβ–Cβ {cand.get('cb_cb')}Å; {bur}; "
                             + ("⚠ clash" if cand.get("clash") else "clash-free")),
        }
        reason = self._unstageable_reason(entry["subs"])
        if reason:                                    # block a pick that can't land (no silent vanish at enact)
            self._status.setText(f"Can't add — {reason}. Re-scan if the structure changed.")
            return
        self.design_basket.add_entry(entry)
        self._status.setText(f"Added {pair_label(cand)} ({aa_a}{ra}{ta}+{aa_b}{rb}{tb}) to the design basket.")

    def _declare_saltbridge(self, cd, cand: dict) -> None:
        """Substitute BOTH positions to the complementary charged residues on a NEW variant — validation
        is the EXISTING fold/deviation flow (charged residues fold NATIVELY, no constraint — SIMPLER than
        disulfide's bond-constrained fold). One click does both substitutions; the user Folds to validate.
        A CROSS-chain pair applies each position on ITS OWN chain-design (the `_enact_basket` seam)."""
        if self._design is None:
            self._status.setText("Can't substitute — no active design.")
            return
        cd_map = self._chain_to_cd()
        ca, cb = pair_chains(cand)
        ca, cb = (ca or cd.rep_chain), (cb or cd.rep_chain)
        subs = [{"chain": str(ca), "position": cand.get("resnum_a"), "to_aa": cand.get("to_aa_a")},
                {"chain": str(cb), "position": cand.get("resnum_b"), "to_aa": cand.get("to_aa_b")}]
        by_cd: Dict[int, tuple] = {}
        skipped: List[str] = []                       # same loud-not-silent discipline as _enact_basket
        for s in subs:
            tcd, col, reason = self._map_basket_sub(cd_map, s)
            if reason:
                skipped.append(reason)
                continue
            by_cd.setdefault(id(tcd), (tcd, []))[1].append(
                (col, str(s["to_aa"]).strip().upper(), int(s["position"])))
        made: List[tuple] = []
        for tcd, items in by_cd.values():
            vid = self._design.new_variant_id()
            tcd.add_variant(vid)
            for col, to_aa, pos in items:
                try:
                    tcd.edit_variant(vid, col, to_aa)
                except Exception as exc:
                    skipped.append(f"{tcd.rep_chain}:{pos}→{to_aa} ({type(exc).__name__})")
            v = tcd.get_variant(vid)
            n = len(v.mutations) if v is not None else 0
            if n == 0:
                tcd.variants = [x for x in tcd.variants if x.id != vid]   # never leave an empty variant
            else:
                made.append((tcd, vid, n))
        if not made:
            msg = "Substitution failed — no positions mapped to the design"
            if skipped:
                msg += " (" + "; ".join(sorted(set(skipped))) + ")"
            self._status.setText(msg + ".")
            return
        made.sort(key=lambda m: -m[2])
        primary_cd, primary_vid, _ = made[0]
        tab = self._focus_tab_for_design(primary_cd)
        msg = (f"{primary_vid}: {pair_label(cand)} charge pair substituted "
               f"({'across 2 chains' if len(made) > 1 else 'both positions'}).")
        if skipped:
            msg += " Skipped " + "; ".join(sorted(set(skipped))) + "."
        msg += " Fold this variant to VALIDATE (re-fold + deviation — charged residues fold natively, no constraint)."
        if tab is not None:
            self._after_variant_edit(tab, primary_vid, msg)
        else:
            self._persist()
            self._status.setText(msg)

    def _estimate_saltbridge_ddg(self, cd, cand: dict) -> None:
        """2-residue ΔΔG escalation for ONE novel salt-bridge candidate (the disulfide-escalation pattern,
        two variable targets). FOCUS the cd's tab so from_aa resolves against the RIGHT design, build the
        spec, launch on the worker seam (long-running; never auto-run across a scan)."""
        self._focus_tab_for_design(cd)
        spec = self.build_saltbridge_ddg_spec(cand)
        if spec is None:
            self._status.setText("Can't estimate ΔΔG — needs a readable structure and recoverable WT "
                                 "residues for both positions (re-scan if the structure changed).")
            return
        self._status.setText(f"Estimating ΔΔG (legacy, uncalibrated) for {pair_label(cand)} charge pair "
                             "— both positions; minutes…")
        self.launchRequested.emit(spec)

    def build_saltbridge_ddg_spec(self, cand: dict) -> Optional[dict]:
        """ΔΔG-escalation spec for ONE novel salt-bridge candidate → the legacy ΔΔG bridge
        (refresh='saltbridge_ddg'). Saves the LIVE structure to PDB (PyRosetta needs PDB), recovers EACH
        member's WT residue own-chain (cross-chain discipline), carries each position's target charged
        residue, tags the source so the router gate blocks a de-novo web upload. None when there's no
        readable structure / a from_aa can't be recovered / PDB save fails — never a silent wrong score."""
        if self._design is None:
            return None
        src = self._active_structure()
        if src is None:
            return None
        ca, cb = pair_chains(cand)
        ra, rb = cand.get("resnum_a"), cand.get("resnum_b")
        ta, tb = cand.get("to_aa_a"), cand.get("to_aa_b")
        if ca is None or cb is None or ra is None or rb is None or not ta or not tb:
            return None
        aa_a, aa_b = self._from_aa_for(ca, ra), self._from_aa_for(cb, rb)
        if aa_a is None or aa_b is None:
            return None
        pdb = self._save_structure_pdb(src["model_id"])
        if not pdb:
            return None
        source = "denovo" if self._design.source == "sequence" else "loaded"
        return {
            "tool": "saltbridge_ddg_estimate",
            "tool_inputs": {"pdb_path": pdb,
                            "chain_a": str(ca), "resnum_a": int(ra), "from_aa_a": aa_a, "to_aa_a": str(ta),
                            "chain_b": str(cb), "resnum_b": int(rb), "from_aa_b": aa_b, "to_aa_b": str(tb),
                            "source": source},
            "user_input": f"[Workbench] estimate ΔΔG (legacy) for salt-bridge pair {pair_label(cand)}",
            "confidence": "low", "refresh": "saltbridge_ddg",
            "_align_ukey": self._cur_cd_ukey(),
            "_sb_cand": {"chain_a": str(ca), "resnum_a": int(ra),
                         "chain_b": str(cb), "resnum_b": int(rb)},
        }

    def apply_saltbridge_ddg_result(self, spec: dict, result: dict) -> None:
        """Attach the escalated 2-residue ΔΔG onto the MATCHING novel candidate (by chain+resnum) and
        re-render the tab. Fail-LOUD: a failed/aborted estimate shows its reason, never a fabricated
        number."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "saltbridge_ddg_estimate"), None)
        if step is None:
            self._status.setText("ΔΔG estimate produced no result.")
            return
        if not step.get("success"):
            self._status.setText(f"ΔΔG estimate: {step.get('error') or 'failed'}")
            return
        cd = self._scan_target_cd(spec)
        want = (spec or {}).get("_sb_cand") or {}
        data = step.get("data") or {}
        novel = ((cd.saltbridge_scan or {}).get("novel") if cd else None) or []
        target = next((c for c in novel
                       if str(pair_chains(c)[0]) == str(want.get("chain_a"))
                       and str(pair_chains(c)[1]) == str(want.get("chain_b"))
                       and c.get("resnum_a") == want.get("resnum_a")
                       and c.get("resnum_b") == want.get("resnum_b")), None)
        if target is None:
            self._status.setText("ΔΔG came back but its candidate is no longer listed (re-scan).")
            return
        target.update({"ddg_a": data.get("ddg_a"), "ddg_b": data.get("ddg_b"),
                       "ddg_mean": data.get("ddg_mean"), "ddg_backend": data.get("backend"),
                       "ddg_source": data.get("source")})
        self._persist()
        self.saltbridge_tab.populate(cd, cd.saltbridge_scan)
        self._status.setText(step.get("summary") or "ΔΔG estimate complete.")

    def _add_cavity_to_basket(self, cd, cand: dict) -> None:
        """Normalize a cavity-fill candidate → a basket entry (one position → the larger fill residue).
        Cavity rows already carry `from_aa` + `to_aa` (basket-aware-shaped)."""
        entry = {
            "cls": "Cavity", "score": float(cand.get("score", 0.0)),
            "subs": [{"chain": str(cand.get("chain") or cd.rep_chain), "position": int(cand["position"]),
                      "from_aa": cand.get("from_aa", "X"), "to_aa": str(cand.get("to_aa", ""))}],
            "metrics_text": (f"void {cand.get('void_volume', 0):.0f} Å³ cav {cand.get('cavity_id')}; "
                             f"fills {cand.get('fill_fraction', 0):.0%}; "
                             + ("⚠ clash" if cand.get("clash") else "clash-free")),
        }
        reason = self._unstageable_reason(entry["subs"])
        if reason:                                    # block a pick that can't land (no silent vanish at enact)
            self._status.setText(f"Can't add — {reason}. Re-scan if the structure changed.")
            return
        self.design_basket.add_entry(entry)
        self._status.setText(f"Added {cand.get('from_aa','')}{cand.get('position')}→{cand.get('to_aa','')} "
                             f"(cavity fill) to the design basket.")

    # ── Design basket: stage strategy picks → compose ONE variant via the existing path ───────
    def _add_proline_to_basket(self, cd, cand: dict) -> None:
        """Normalize a proline candidate → a basket entry (one position → Pro). Proline rows already
        carry `from_aa` (basket-aware-shaped)."""
        psi = cand.get("psi")
        entry = {
            "cls": "Proline", "score": float(cand.get("score", 0.0)),
            "subs": [{"chain": str(cand.get("chain") or cd.rep_chain), "position": int(cand["position"]),
                      "from_aa": cand.get("from_aa", "X"), "to_aa": "P"}],
            "metrics_text": (f"φ {cand.get('phi')}° ψ {'n/a' if psi is None else f'{psi}°'}; "
                             + ("H-bond donor (penalized)" if cand.get("hbond_donates")
                                else "no backbone H-bond donor")),
        }
        reason = self._unstageable_reason(entry["subs"])
        if reason:                                    # block a pick that can't land (no silent vanish at enact)
            self._status.setText(f"Can't add — {reason}. Re-scan if the structure changed.")
            return
        self.design_basket.add_entry(entry)
        self._status.setText(f"Added {cand.get('from_aa','')}{cand.get('position')}→P to the design basket.")

    def _add_disulfide_to_basket(self, cd, pair: dict) -> None:
        """Normalize a disulfide pair → a basket entry (TWO positions → Cys). Disulfide rows are
        residue-AGNOSTIC (no `from_aa`), so recover each WT residue own-chain via `_from_aa_for`
        (the retrofit — same recovery the ΔΔG escalation uses)."""
        ca, cb = pair_chains(pair)
        ca, cb = (ca or cd.rep_chain), (cb or cd.rep_chain)
        ra, rb = pair.get("resnum_a"), pair.get("resnum_b")
        aa_a, aa_b = self._from_aa_for(ca, ra), self._from_aa_for(cb, rb)
        if aa_a is None or aa_b is None or ra is None or rb is None:
            self._status.setText("Can't add — couldn't recover the WT residue for the pair "
                                 "(re-scan if the structure changed).")
            return
        sg, chi = pair.get("best_sg_sg"), pair.get("best_chi_ss")
        clash = pair.get("clash")
        entry = {
            "cls": "Disulfide", "score": float(pair.get("score", 0.0)),
            "subs": [{"chain": str(ca), "position": int(ra), "from_aa": aa_a, "to_aa": "C"},
                     {"chain": str(cb), "position": int(rb), "from_aa": aa_b, "to_aa": "C"}],
            "metrics_text": (f"Sγ–Sγ {sg if sg is not None else '—'}Å χSS {chi if chi is not None else '—'}°; "
                             + ("⚠ clash" if clash else "clash-free" if clash is False else "geometric")),
        }
        reason = self._unstageable_reason(entry["subs"])
        if reason:                                    # block a pick that can't land (no silent vanish at enact)
            self._status.setText(f"Can't add — {reason}. Re-scan if the structure changed.")
            return
        self.design_basket.add_entry(entry)
        self._status.setText(f"Added {pair_label(pair)} (both→Cys) to the design basket.")

    def _enact_basket(self, entries: List[dict]) -> None:
        """Compose ALL basket substitutions into one new variant PER chain-design (the proven seam:
        `add_variant` + N×`edit_variant`), then hand off to the EXISTING variant machinery — the
        designer folds / compares / overlays-template the variant as any variant (those stay opt-in).
        The basket COMPOSES; the variant system VALIDATES (the fold reveals the combination's effect —
        not proxied here). Same-residue conflicts are blocked upstream by the basket panel."""
        if self._design is None or not entries:
            return
        cd_map = self._chain_to_cd()
        by_cd: Dict[int, tuple] = {}                  # id(cd) -> (cd, [(col, to_aa, pos)]) by owner cd
        skipped: List[str] = []                       # human reasons a sub couldn't land — NEVER silent
        for e in entries:
            for s in e.get("subs", []):
                tcd, col, reason = self._map_basket_sub(cd_map, s)
                if reason:                            # off-template resnum / unmapped chain / bad target
                    skipped.append(reason)
                    continue
                by_cd.setdefault(id(tcd), (tcd, []))[1].append((col, s["to_aa"], int(s["position"])))
        # BACKSTOP (Fix 1b): two subs grouped onto the SAME cd at the SAME resnum but with DIFFERENT
        # target residues would silently overwrite (edit_variant dedups v.mutations by resnum) — the
        # homo-collapse loss. The basket conflict-check blocks this upstream; this is the defensive net
        # for any caller that reaches enact without it (programmatic basket construction, future seams).
        # Refuse the WHOLE enact rather than compose a half-built, last-write-wins variant.
        for tcd, items in by_cd.values():
            want: Dict[int, str] = {}
            for _col, aa, pos in items:
                if want.get(pos, aa) != aa:
                    self._status.setText(
                        f"Enact blocked — conflicting substitutions at {tcd.rep_chain}:{pos} "
                        f"(→{want[pos]} and →{aa}) on one chain-design (equivalent copies are one "
                        f"residue). Resolve the basket conflict first.")
                    return
                want[pos] = aa
        made: List[tuple] = []                        # (cd, vid, n_applied)
        for tcd, items in by_cd.values():
            vid = self._design.new_variant_id()
            tcd.add_variant(vid)
            for col, to_aa, pos in items:
                try:
                    tcd.edit_variant(vid, col, to_aa)
                except Exception as exc:              # defensive — _map_basket_sub already validated
                    skipped.append(f"{tcd.rep_chain}:{pos}→{to_aa} ({type(exc).__name__})")
            v = tcd.get_variant(vid)
            n = len(v.mutations) if v is not None else 0   # REAL changes (a to-WT sub reverts → 0)
            if n == 0:
                tcd.variants = [x for x in tcd.variants if x.id != vid]   # never leave an empty variant
            else:
                made.append((tcd, vid, n))
        # Nothing landed → surface WHY (not a silent empty variant), and KEEP the basket so the picks
        # are recoverable / re-targetable rather than consumed into nothing.
        if not made:
            msg = "Enact failed — no substitutions could be applied"
            if skipped:
                msg += " (" + "; ".join(sorted(set(skipped))) + ")"
            self._status.setText(msg + ". The basket is unchanged — re-scan if the structure changed.")
            return
        made.sort(key=lambda m: -m[2])                # refresh on the most-substituted cd's new variant
        primary_cd, primary_vid, _ = made[0]
        total = sum(m[2] for m in made)
        msg = (f"Enacted variant {primary_vid} — {total} substitution(s)"
               + (f" across {len(made)} chains" if len(made) > 1 else "") + ".")
        if skipped:                                   # partial success is reported, never hidden
            uniq = sorted(set(skipped))
            msg += f" Skipped {len(uniq)}: " + "; ".join(uniq) + "."
        msg += " Fold it to VALIDATE (the fold reveals the combination's effect)."
        tab = self._focus_tab_for_design(primary_cd)
        if tab is not None:
            self._after_variant_edit(tab, primary_vid, msg)
        else:
            self._persist()
            self._status.setText(msg)
        self.design_basket.reset()                    # the picks have flowed into the variant — consume

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
            _RESULT_DISULFIDE_MODE: bool(self._active_disulfide_scan(tab)),
            _RESULT_PROLINE_MODE:   bool(self._active_proline_scan(tab)),
            _RESULT_CAVITY_MODE:    bool(self._active_cavity_scan(tab)),
            _RESULT_SALTBRIDGE_MODE: bool(self._active_saltbridge_scan(tab)),
        }
        model = self._mode_combo.model()
        for i in range(self._mode_combo.count()):
            key = self._mode_combo.itemData(i)
            if key in avail and (item := model.item(i)) is not None:
                item.setEnabled(avail[key])
        if self._mode_key in avail and not avail[self._mode_key]:
            self._select_result_mode("none")          # current result has no data → revert
            self._mode_key = "none"

    def _sync_deviation_enabled(self, tab: "_ChainDesignTab") -> None:
        """Grey the Deviation button when the active variant's fold can't be compared to a WT
        reference of the SAME engine+oligomer (an un-pinned fold at a different shape than the
        construct baseline) — clearer than a click that then refuses. Enabled in every other case
        (incl. no fold yet — the handler then tells you to fold first). Synced on render/active-row."""
        btn = getattr(self, "_dev_btn", None)
        if btn is None:
            return
        ok = True
        v = self._active_variant(tab) if tab is not None else None
        cd = tab.design if tab is not None else None
        if (v is not None and v.results.fold and cd is not None
                and self._design is not None and self._design.source == "sequence"):
            combo = f"{v.results.fold.get('engine')}:{v.results.fold.get('target')}"
            tf = cd.template_fold or {}
            ok = combo in cd.wt_refs or f"{tf.get('engine')}:{tf.get('target')}" == combo
        btn.setEnabled(ok)

    def _apply_color_to(self, tab: _ChainDesignTab) -> None:
        self._refresh_color_mode_availability(tab)    # grey result modes lacking data
        self._sync_deviation_enabled(tab)             # grey Deviation on an un-comparable fold combo
        if self._mode_key == _RESULT_DDG_MODE:
            hexmap = {rn: ddg_color(d) for rn, d in self._active_ddg_map(tab).items()}
            tab.set_result_coloring(tab.active_row_id, {k: v for k, v in hexmap.items() if v})
        elif self._mode_key == _RESULT_PLDDT_MODE:
            hexmap = {rn: plddt_color(p) for rn, p in self._active_plddt_map(tab).items()}
            tab.set_result_coloring(tab.active_row_id, {k: v for k, v in hexmap.items() if v})
        elif self._mode_key == _RESULT_DEVIATION_MODE:
            tab.set_result_coloring(tab.active_row_id, self._deviation_panel_hex(tab))
        elif self._mode_key == _RESULT_DISULFIDE_MODE:
            tab.set_result_coloring(tab.active_row_id, self._disulfide_scan_panel_hex(tab))
        elif self._mode_key == _RESULT_PROLINE_MODE:
            tab.set_result_coloring(tab.active_row_id, self._proline_scan_panel_hex(tab))
        elif self._mode_key == _RESULT_CAVITY_MODE:
            tab.set_result_coloring(tab.active_row_id, self._cavity_scan_panel_hex(tab))
        elif self._mode_key == _RESULT_SALTBRIDGE_MODE:
            tab.set_result_coloring(tab.active_row_id, self._saltbridge_scan_panel_hex(tab))
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
        # A different visual mode is taking over → clear any ACTIVE disulfide glow FIRST (un-ghost +
        # un-highlight) so we never get the hybrid (new colours + leftover disulfide transparency).
        # Same restore-before-apply discipline the glow uses within the disulfide flow, now triggered
        # when a colour/visibility push runs. No-op (empty) when nothing glows.
        cmds = self._consume_glow_restore() + cmds
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
        # MERGE, don't replace: a deep Rosetta run augments an earlier fast ThermoMPNN/RaSP run so
        # ALL method axes survive (the export keeps every set, even after a higher-quality analysis).
        v.results.stability = merge_stability(v.results.stability,
                                              stability_summary(candidates, v.mutations))
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

    def fold_launch_spec(self, engine: str, assembly: bool = False,
                         n_copies: Optional[int] = None) -> Optional[dict]:
        """Deterministic fold spec for the ACTIVE variant through *engine*: fold LOCAL-ONLY,
        open the model, pLDDT-colour it, matchmaker onto the WT reference. *assembly* (Boltz
        only) folds the full homo-oligomer. confidence='low' → the spine's confirm-gate.
        None when there is no active variant. The SAME spec shape every engine reuses.

        DE-NOVO oligomer:
          • *n_copies* OMITTED (the disulfide-constrain caller + the loaded picker) → PINNED: the
            variant folds at the CONSTRUCT's engine + oligomer and superposes onto the T-fold, so
            the variant, the WT reference, and the floor seeds all fold the same (deviation-valid).
          • *n_copies* GIVEN (the de-novo variant picker) → UN-PINNED: the user's engine + N-mer are
            honoured — the variant is its OWN fold (the construct baseline is untouched). Superpose
            onto the T-fold ONLY when engine + oligomer MATCH the baseline (you can't overlay a
            monomer on a trimer reference); a mismatched fold is standalone (no_reference) and
            deviation-vs-WT guards/greys on the combo (a per-oligomer reference is the follow-up)."""
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None or self._design is None:
            return None
        cd = tab.design
        denovo = self._design.source == "sequence"
        tf = cd.template_fold if denovo else {}
        unpinned = denovo and n_copies is not None
        n = matches = 0
        if denovo:
            if not tf.get("model_id"):
                return None                       # construct not folded yet → no baseline/reference
            if unpinned:
                n = max(1, int(n_copies))
                if n > 1 and engine not in ("boltz", "colabfold"):
                    return None                   # ESMFold is monomer-only
                assembly   = n > 1
                matches    = (engine == tf.get("engine") and n == len(cd.members))   # baseline shape?
                compare_to = tf.get("model_id") if matches else None                 # superpose iff matched
            else:
                engine     = tf.get("engine", engine)             # PIN engine from the construct fold
                assembly   = (tf.get("target") == "assembly")     # PIN oligomer from the construct fold
                compare_to = tf.get("model_id")                   # superpose onto the T-fold
        else:
            compare_to = self._design.model_id    # crystal design: matchmaker onto the loaded WT
        ti: Dict[str, object] = {
            "model_id":   self._design.model_id,
            "chain":      cd.rep_chain,
            "engine":     engine,
            "open_model": True,
            "local_only": (engine in _LOCAL_FOLD_ENGINES) if unpinned else True,
        }
        if compare_to is not None:
            ti["compare_to"] = compare_to         # superpose onto the WT reference
        else:
            ti["no_reference"] = True             # mismatched-shape variant fold → standalone (no overlay)
        if engine == "boltz" and assembly:
            if unpinned:
                ids = ([ch for (_m, ch) in cd.members] if matches
                       else [chr(ord("A") + i) for i in range(n)])    # reuse baseline ids iff matched
                ti["chains"] = [{"id": ch, "sequence": v.sequence} for ch in ids]
                target = f"{len(ids)}-chain assembly"
            elif denovo:
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

    # ── Disulfide suite (Modes A discovery / B geometry-readout / C declared-constraint) ──────
    def disulfide_discovery_launch_spec(self, n_copies: int = 1) -> Optional[dict]:
        """MODE A spec — fold the construct UNCONSTRAINED across N seeds → per-cys-pair bonding
        FREQUENCY. Reuses `construct_fold_launch_spec` for the construct's chains/sequence (same
        seq source as the fold), then routes to the `disulfide_discovery` tool. De-novo only."""
        base = self.construct_fold_launch_spec("boltz", n_copies)
        if base is None:
            return None
        bti = base["tool_inputs"]
        ti: Dict[str, object] = {"model_id": bti.get("model_id")}
        if bti.get("chains"):
            ti["chains"] = bti["chains"]
        else:
            ti["sequence"] = bti.get("sequence")
            ti["chain"] = base.get("_denovo_fold_chains", ["A"])[0] if base.get("_denovo_fold_chains") else "A"
        return {
            "tool": "disulfide_discovery", "tool_inputs": ti,
            "user_input": "[Workbench] disulfide discovery — multi-seed bonding frequency (unconstrained)",
            "confidence": "low", "refresh": "disulfide_discovery",
        }

    def disulfide_geometry_launch_spec(self) -> Optional[dict]:
        """MODE B spec — read the active design's EXISTING structure (the de-novo construct's fold OR
        a LOADED crystal/model, via `_active_structure`) and report the MEASURED cysteine-pair
        geometry. CHEAP (reads coordinates; NEVER folds). None until there is a structure to read."""
        src = self._active_structure()
        if src is None:
            return None
        return {
            "tool": "disulfide_geometry", "tool_inputs": {"cif_path": src["cif_path"]},
            "user_input": "[Workbench] disulfide geometry readout (measured, this structure)",
            "confidence": "high", "refresh": "disulfide_geometry",      # cheap + deterministic → no gate
        }

    def disulfide_scan_launch_spec(self) -> Optional[dict]:
        """MODE D spec — backbone scan of the active design's EXISTING structure (de-novo fold OR a
        LOADED crystal/model, via `_active_structure`) for NOVEL engineerable disulfide sites. CHEAP
        (reads coordinates; NEVER folds). None until there is a structure to read."""
        src = self._active_structure()
        if src is None:
            return None
        return {
            "tool": "disulfide_scan", "tool_inputs": {"cif_path": src["cif_path"]},
            "user_input": "[Workbench] find engineerable disulfide sites (backbone scan, this structure)",
            "confidence": "high", "refresh": "disulfide_scan",          # cheap + deterministic → no gate
            "_align_ukey": self._cur_cd_ukey(),
        }

    def disulfide_interface_scan_launch_spec(self) -> Optional[dict]:
        """INTERFACE-SCAN spec — cross-chain backbone scan of the active design's EXISTING structure
        (de-novo fold OR a LOADED crystal/model, via `_active_structure`) for NOVEL inter-subunit
        disulfide sites. CHEAP (reads coordinates; NEVER folds). None unless the structure has a
        MULTIMER (≥2 chains — an inter-chain bond needs ≥2 chains)."""
        src = self._active_structure()
        if src is None or self._design is None:
            return None
        if sum(len(c.members) for c in self._design.chains.values()) < 2:
            return None                                                 # monomer → no interface
        return {
            "tool": "disulfide_interface_scan", "tool_inputs": {"cif_path": src["cif_path"]},
            "user_input": "[Workbench] find interface disulfide sites (inter-chain scan, this structure)",
            "confidence": "high", "refresh": "disulfide_interface_scan",   # cheap + deterministic → no gate
            "_align_ukey": self._cur_cd_ukey(),
        }

    @staticmethod
    def _col_for_resnum(cd, resnum: int) -> Optional[int]:
        for i, c in enumerate(cd.template_cells):
            if (not c.is_gap) and c.resnum == int(resnum):
                return i
        return None

    def _map_basket_sub(self, cd_map: Dict[str, "ChainDesign"], sub: dict):
        """Resolve ONE basket sub → ``(cd, col, None)`` when it can land, or ``(None, None, reason)``
        with a human reason when it can't: the target isn't a standard residue, its chain isn't on
        the active design, or its resnum isn't a column on that design's template. The SINGLE source
        of truth for "can this pick land?" — used to GUARD staging AND to apply-and-REPORT at enact,
        so a pick that can't land is surfaced loud, NEVER silently dropped into an empty variant.
        The standard-residue check uses the SAME `_STD_AA` set `edit_variant` enforces, so the guard
        can't disagree with the editor."""
        chain = str(sub.get("chain"))
        pos = sub.get("position")
        raw = sub.get("to_aa")
        to_aa = (str(raw) if raw is not None else "").strip().upper()
        if to_aa not in _STD_AA:
            return None, None, f"{chain}:{pos}→{raw!r} (not a standard residue)"
        tcd = cd_map.get(chain)
        if tcd is None:
            return None, None, f"{chain}:{pos} (chain not on the active design)"
        try:
            col = self._col_for_resnum(tcd, pos)
        except (TypeError, ValueError):
            col = None
        if col is None:
            return None, None, f"{chain}:{pos} (residue not in the design template)"
        return tcd, col, None

    def _unstageable_reason(self, subs) -> Optional[str]:
        """First reason any of *subs* can't land on the active design, or None if all can — the
        curate-time guard so a pick that would silently vanish at enact is blocked AT STAGING
        (the deferred "zero substitutions landed" gap: stale/mismatched chain or off-template resnum)."""
        if self._design is None:
            return "no active design"
        cd_map = self._chain_to_cd()
        for s in (subs or []):
            _cd, _col, reason = self._map_basket_sub(cd_map, s)
            if reason:
                return reason
        return None

    def _normalize_ss_pairs(self, pairs):
        """Mode-C pair input → ``[(chain_a, resnum_a, chain_b, resnum_b)]``. Accepts a single entry
        or a list of: ``(a, b)`` resnum tuples (SAME-chain on the active cd) or reshaped pair dicts
        carrying ``chain_a``/``chain_b`` (or a legacy single ``chain`` → both, via `pair_chains`).
        None on malformed input. PURE-ish (reads the active cd's rep chain for bare tuples)."""
        if isinstance(pairs, (tuple, dict)):
            pairs = [pairs]
        tab = self._cur_tab()
        active_rep = (tab.design.rep_chain if (tab and tab.design) else None)
        out = []
        for p in (pairs or []):
            if isinstance(p, dict):
                cha, chb = pair_chains(p)
                ra, rb = p.get("resnum_a"), p.get("resnum_b")
            else:
                try:
                    ra, rb = p
                except (TypeError, ValueError):
                    return None
                cha = chb = active_rep
            if ra is None or rb is None or cha is None or chb is None:
                return None
            out.append((str(cha), int(ra), str(chb), int(rb)))
        return out

    def _chain_to_cd(self) -> Dict[str, "ChainDesign"]:
        """Map each fold chain id → the ChainDesign that owns it (every cd's `members` chain).
        Post-construct-fold a homo-oligomer's copies share ONE cd (members A,B,… on the same cd);
        a hetero complex has one cd per distinct chain — so this resolves a declared chain to the
        cd whose `ordered` resnum list a member must be mapped against (the cross-chain hinge)."""
        m: Dict[str, "ChainDesign"] = {}
        for cd in (self._design.chains.values() if self._design else []):
            for (_mdl, ch) in cd.members:
                m[str(ch)] = cd
        return m

    def _chain_equiv(self, chain) -> List[str]:
        """All fold-chain ids that resolve to the SAME ChainDesign as *chain* — the homo-oligomer
        copies that share one cd (an edit applies to ALL: identical sequence). `[chain]` when the
        chain is unresolved / there's no design (each its own equivalence class). The design basket
        consumes this to (a) key conflict + dedupe by cd-equivalence rather than raw chain, and (b)
        surface 'applies to: A, B' on a homo pick."""
        cd = self._chain_to_cd().get(str(chain))
        if cd is None:
            return [str(chain)]
        return [str(ch) for (_m, ch) in cd.members] or [str(chain)]

    def build_disulfide_introduce_spec(self, pairs, engine: str = "boltz") -> Optional[dict]:
        """MODE C (introduce) — declare ONE OR MORE disulfide bonds between positions that are NOT
        yet cysteine: GENUINELY COMPOSE with the substitution subsystem (`add_variant` +
        `edit_variant(col,"C")` at EVERY declared position — the existing path, not a parallel
        cysteine-introduction), then fold THAT variant WITH all declared `bond` constraints. Each
        author-resnum→1-based chain-index conversion (the correctness hinge) runs through the tested
        `disulfide_geometry.resnum_to_chain_index`. *pairs* accepts `(a,b)` resnum tuples (SAME-chain
        on the active cd) AND reshaped cross-chain pair dicts (`chain_a`/`chain_b`). Returns the
        constrained fold spec (refresh='fold'), or None (any pair invalid)."""
        import disulfide_geometry as _dg
        norm = self._normalize_ss_pairs(pairs)
        if not norm or self._design is None or self._design.source != "sequence":
            return None
        tab = self._cur_tab()
        active_rep = (tab.design.rep_chain if (tab and tab.design) else None)
        # CROSS-chain (or same-chain on a non-active cd) → the inter-chain path; only when EVERY pair
        # is same-chain on the ACTIVE cd do we take the unchanged intra-chain path (byte-identical).
        if not all(cha == chb == active_rep for (cha, _ra, chb, _rb) in norm):
            return self._build_cross_chain_ss_spec(norm, engine, _dg)
        cd = tab.design
        pairs = [(ra, rb) for (_ca, ra, _cb, rb) in norm]
        # validate columns up front (all must resolve, distinct within a pair)
        cols = {}
        for a, b in pairs:
            ca, cb = self._col_for_resnum(cd, a), self._col_for_resnum(cd, b)
            if ca is None or cb is None or ca == cb:
                return None
            cols[a], cols[b] = ca, cb
        # COMPOSE with substitution: ONE variant carrying Cys at EVERY declared position.
        vid = "SS_" + "_".join(f"{a}-{b}" for a, b in pairs)
        if cd.get_variant(vid) is None:
            cd.add_variant(vid, source="manual")
        for pos, col in cols.items():
            cd.edit_variant(vid, col, "C", source="disulfide_introduce")
        tab.set_active_row(vid)
        spec = self.fold_launch_spec(engine)            # folds THIS (now-Cys) variant
        if spec is None:
            return None
        v = cd.get_variant(vid)
        ordered = [c.resnum for c in v.cells if (not c.is_gap) and c.resnum is not None]
        chain_id = cd.rep_chain
        constraints = []
        for a, b in pairs:
            ia, ib = _dg.resnum_to_chain_index(ordered, a), _dg.resnum_to_chain_index(ordered, b)
            if ia is None or ib is None:
                return None
            # SAME-chain (intra) declare: both atoms on cd.rep_chain → identical constraint to before.
            constraints.append(_dg.bond_constraint(chain_id, ia, chain_id, ib))
        spec["tool_inputs"]["disulfide_constraints"] = constraints
        spec["tool_inputs"]["disulfide_bonds"] = list(pairs)
        _pp = ", ".join(f"Cys{a}–Cys{b}" for a, b in pairs)
        spec["user_input"] = (f"[Workbench] fold {vid} with DECLARED bond(s) {_pp} "
                              f"(biases toward them; does NOT enforce geometry) — {engine}, LOCAL-ONLY")
        spec["_variant_id"] = vid
        return spec

    def _build_cross_chain_ss_spec(self, norm, engine: str, _dg) -> Optional[dict]:
        """MODE C, INTER-chain — declare a disulfide whose two residues live on DIFFERENT fold
        chains (inter-subunit). The correctness gate (silent-wrong-residue, not a crash): each
        member is resolved against ITS OWN chain's ordered list and the Cys substitution is composed
        on BOTH owning cds, then the WHOLE assembly is folded with `atom1:[CHAIN_A,…], atom2:
        [CHAIN_B,…]`. None on any invalid endpoint. Requires the construct already folded (the
        deviation/compare reference) + a multi-chain (Boltz) assembly."""
        chain_cd = self._chain_to_cd()
        cols_by_cd: Dict[int, Tuple["ChainDesign", set]] = {}   # id(cd) -> (cd, {col})
        endpoints = []   # (cha, cda, ra, chb, cdb, rb)
        for (cha, ra, chb, rb) in norm:
            cda, cdb = chain_cd.get(cha), chain_cd.get(chb)
            if cda is None or cdb is None:
                return None
            cola, colb = self._col_for_resnum(cda, ra), self._col_for_resnum(cdb, rb)
            if cola is None or colb is None:
                return None
            if cha == chb and cola == colb:          # a residue can't bond to itself
                return None
            cols_by_cd.setdefault(id(cda), (cda, set()))[1].add(cola)
            cols_by_cd.setdefault(id(cdb), (cdb, set()))[1].add(colb)
            endpoints.append((cha, cda, ra, chb, cdb, rb))
        # COMPOSE on BOTH cds: one SS variant per involved cd, Cys at all its declared cols.
        vid = "SS_" + "_".join(f"{cha}{ra}-{chb}{rb}" for (cha, ra, chb, rb) in norm)
        for (cd, cols) in cols_by_cd.values():
            if cd.get_variant(vid) is None:
                cd.add_variant(vid, source="manual")
            for col in cols:
                cd.edit_variant(vid, col, "C", source="disulfide_introduce")
        # primary cd: the active one if involved, else the first involved (the fold/readout anchor).
        tab = self._cur_tab()
        active_cd = tab.design if tab else None
        primary = active_cd if (active_cd is not None and id(active_cd) in cols_by_cd) \
            else next(iter(cols_by_cd.values()))[0]
        ptab = self._focus_tab_for_design(primary)
        if ptab is not None:
            ptab.set_active_row(vid)
        tf = primary.template_fold or {}
        if not tf.get("model_id"):
            return None                              # construct not folded → no reference to fold against
        eng = tf.get("engine", engine)
        # FULL-COMPLEX composition: each involved cd contributes its SS-variant sequence (Cys
        # introduced), every other cd its template T — across that cd's member chains (same ids/order
        # as the construct T-fold, so it lines up 1:1 with the deviation reference).
        chains: List[Dict[str, str]] = []
        for c in self._design.chains.values():
            cv = c.get_variant(vid)
            seq = cv.sequence if cv is not None else "".join(
                cc.aa for cc in c.template_cells if cc.aa is not None)
            chains.extend({"id": ch, "sequence": seq} for (_m, ch) in c.members)
        if len(chains) < 2 or eng not in ("boltz", "colabfold"):
            return None                              # an inter-chain bond needs a multi-chain fold
        # CONSTRAINTS: each endpoint resolved against ITS OWN cd's ordered list (the mis-map guard).
        constraints, bonds = [], []
        for (cha, cda, ra, chb, cdb, rb) in endpoints:
            oa = [c.resnum for c in cda.get_variant(vid).cells if (not c.is_gap) and c.resnum is not None]
            ob = [c.resnum for c in cdb.get_variant(vid).cells if (not c.is_gap) and c.resnum is not None]
            ia, ib = _dg.resnum_to_chain_index(oa, ra), _dg.resnum_to_chain_index(ob, rb)
            if ia is None or ib is None:
                return None
            constraints.append(_dg.bond_constraint(cha, ia, chb, ib))   # atom1:[A,ia], atom2:[B,ib]
            bonds.append((ra, rb))
        _pp = ", ".join(f"Cys{cha}:{ra}–Cys{chb}:{rb}" for (cha, ra, chb, rb) in norm)
        return {
            "tool": eng,
            "tool_inputs": {
                "model_id":    self._design.model_id,
                "chain":       primary.rep_chain,
                "engine":      eng,
                "open_model":  True,
                "local_only":  True,
                "compare_to":  tf.get("model_id"),
                "chains":      chains,
                "disulfide_constraints": constraints,
                "disulfide_bonds":       bonds,
            },
            "user_input": (f"[Workbench] fold {vid} with DECLARED inter-chain bond(s) {_pp} "
                           f"(biases toward them; does NOT enforce geometry) — {eng}, LOCAL-ONLY"),
            "confidence": "low",
            "refresh": "fold",
            "_variant_id": vid,
        }

    def _scan_target_cd(self, spec: dict):
        """The construct a disulfide result belongs to: the spec's `_align_ukey` cd, else the active."""
        ukey = (spec or {}).get("_align_ukey")
        return (self._design.chains.get(ukey) if (self._design and ukey) else None) \
            or (self._cur_tab().design if self._cur_tab() else None)

    def apply_disulfide_discovery_result(self, spec: dict, result: dict) -> None:
        """MODE A readout — the model's EMPIRICAL pairing prior, measured with N (never asserted).
        Populates the persistent Disulfides tab's A section (the source of truth) + status."""
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") == "disulfide_discovery":
                if step.get("success"):
                    cd = self._scan_target_cd(spec)
                    if cd is not None:
                        self.disulfides_tab.populate_assess(cd, (step.get("data") or {}).get("pairs") or [])
                self._status.setText(step.get("summary") if step.get("success")
                                     else f"Disulfide discovery failed: {step.get('error')}")
                return
        self._status.setText("Disulfide discovery produced no result.")

    def apply_disulfide_geometry_result(self, spec: dict, result: dict) -> None:
        """MODE B readout — measured geometry of THIS fold (factual, not a bond declaration).
        Populates the persistent Disulfides tab's B section + status."""
        for step in (result or {}).get("tool_step_results", []) or []:
            if step.get("tool") == "disulfide_geometry":
                if step.get("success"):
                    cd = self._scan_target_cd(spec)
                    if cd is not None:
                        self.disulfides_tab.populate_geometry(cd, (step.get("data") or {}).get("pairs") or [])
                self._status.setText(step.get("summary") if step.get("success")
                                     else f"Disulfide geometry failed: {step.get('error')}")
                return
        self._status.setText("Disulfide geometry produced no result.")

    def apply_disulfide_scan_result(self, spec: dict, result: dict) -> None:
        """MODE D — store the engineering scan on the active cd (ranked list = source of truth +
        best-partner map for the heatmap) and AUTO-SURFACE the 'Disulfide sites' colour mode. The
        geometric-only CAVEAT rides in the status (the heatmap is the most over-read-prone surface)."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "disulfide_scan"), None)
        if step is None or not step.get("success"):
            self._status.setText(f"Engineering scan failed: {step.get('error') if step else 'no result'}")
            return
        data = step.get("data") or {}
        cd = self._scan_target_cd(spec)
        if cd is None:
            return
        cd.disulfide_scan = {"pairs": data.get("pairs") or [],
                             "best_partner": data.get("best_partner") or {},
                             "caveat": data.get("caveat")}
        self._persist()
        if cd.disulfide_scan["pairs"]:
            self._select_result_mode(_RESULT_DISULFIDE_MODE)   # AUTO-SURFACE the heatmap (the index)
            tab = self._cur_tab()
            if tab is not None:
                self._apply_color_to(tab)
                self._push_3d_color(tab)
        self.disulfides_tab.populate_scan(cd, cd.disulfide_scan)  # the PERSISTENT ranked list = truth
        self._status.setText(step.get("summary") or "Engineering scan complete.")

    def apply_disulfide_interface_scan_result(self, spec: dict, result: dict) -> None:
        """INTERFACE SCAN readout — store the inter-chain ranked list on the active cd + populate the
        persistent Disulfides tab's I section (the source of truth). The found cross-chain pairs feed
        the PROVEN cross-chain Mode-C declare (a row's Declare → `build_disulfide_introduce_spec`).
        No heatmap (cross-chain spans chains); the table + 3D highlight + caveat are the surface."""
        step = next((s for s in (result or {}).get("tool_step_results", []) or []
                     if s.get("tool") == "disulfide_interface_scan"), None)
        if step is None or not step.get("success"):
            self._status.setText(f"Interface scan failed: {step.get('error') if step else 'no result'}")
            return
        data = step.get("data") or {}
        cd = self._scan_target_cd(spec)
        if cd is None:
            return
        cd.disulfide_interface_scan = {"pairs": data.get("pairs") or [],
                                       "best_partner": data.get("best_partner") or {},
                                       "caveat": data.get("caveat")}
        self._persist()
        self.disulfides_tab.populate_interface(cd, cd.disulfide_interface_scan)
        self._status.setText(step.get("summary") or "Interface scan complete.")

    _GLOW_HUE = "gold"           # the lit hue (sphere + halo); one constant so glow + halo match

    @staticmethod
    def _disulfide_pair_specs(cd, pair: dict):
        """(mid, a, b, both) ChimeraX specs for a pair on the design's structure. The model id is the
        de-novo construct's FOLD model (`cd.template_fold.model_id`) OR — for a LOADED crystal/model —
        the LIVE rendered id `cd.rep_model` (so a pair-click highlights the structure ON SCREEN, not
        the temp save). Chain PER MEMBER (`pair_chains` tolerates the legacy single-`chain` shape);
        rep chain when absent. SAME-chain collapses to one combined spec, CROSS-chain unions the two
        members. (None,…) if there is no REAL structure id — an UNFOLDED de-novo construct's
        `rep_model` is the synthetic ``denovo-…`` id (nothing in ChimeraX), so it is NOT a highlight
        target; only a fold model id or a loaded model's live id is."""
        mid = (cd.template_fold or {}).get("model_id") or cd.rep_model
        if mid and str(mid).startswith("denovo-"):         # synthetic de-novo id → nothing rendered yet
            mid = None
        if not mid or not pair:
            return None, None, None, None
        ca, cb = pair_chains(pair)
        ca, cb = (ca or cd.rep_chain), (cb or cd.rep_chain)
        ra, rb = pair.get("resnum_a"), pair.get("resnum_b")
        a, b = f"#{mid}/{ca}:{ra}", f"#{mid}/{cb}:{rb}"
        both = f"#{mid}/{ca}:{ra},{rb}" if ca == cb else f"{a} {b}"
        return mid, a, b, both

    @classmethod
    def _disulfide_scan_highlight_commands(cls, cd, pair: dict) -> List[str]:
        """GLOW-style highlight of a pair on the construct's fold: bright OPAQUE spheres on the
        residue(s) (the lit element) + TRANSPARENCY-the-rest of the model (the spotlight cue — the
        glowed residue is the one solid object in a ghosted scene) + a matching-hue selection HALO,
        plus the Cβ–Cβ distance. Pure → single source for the click, the live-verify, and tests; []
        if not folded. The PRIOR glow's RESTORE is `_glow_restore_commands` — the panel orchestrates
        restore-then-apply (`_highlight_disulfide_pair`) so glows/transparency don't STACK. Static
        only — no pulse (frame-loop lifecycle deferred to the §9 unified-highlighting item)."""
        mid, a, b, both = cls._disulfide_pair_specs(cd, pair)
        if mid is None:
            return []
        hue = cls._GLOW_HUE
        return [
            f"show {both} atoms",
            f"style {both} sphere",                 # whole-residue spheres → a brighter lit blob
            f"color {both} {hue} target a",         # colour ATOMS only — the cartoon keeps its mode colour
            f"transparency #{mid} 70 target c",     # ghost the rest of the model (cartoon) → spotlight
            f"transparency {both} 0 target c",       # keep the glowed residue's own cartoon opaque
            f"graphics selection color {hue}",      # halo outline — GLOBAL (a nicer permanent default)
            "graphics selection width 5",
            f"select {both}",
            f"distance {a}@CB {b}@CB",               # the Cβ–Cβ measured span
        ]

    @staticmethod
    def _glow_restore_commands(prev: Optional[dict]) -> List[str]:
        """RESTORE a prior glow (stored ``{mid, both}``) NON-DESTRUCTIVELY: un-ghost the whole model
        (transparency → opaque) + return the glowed residue(s) to cartoon (hide their atoms; the
        cartoon's mode colour shows through, never recoloured) + clear the pair distance + DESELECT
        (so the gold selection outline disappears — the full visible reset). [] if there is nothing to
        restore. Reused before every new glow so A's spotlight is fully reset before B AND when a
        different visual mode takes over (so the glow never bleeds into the new view)."""
        if not prev or not prev.get("mid") or not prev.get("both"):
            return []
        mid, both = prev["mid"], prev["both"]
        return [
            f"transparency #{mid} 0 target c",       # un-ghost the whole model (was 70%)
            f"hide {both} atoms",                    # glowed residues back to cartoon
            "~distance",                              # clear the prior pair's distance label
            "~select",                                # drop the pair selection → its gold outline goes
        ]

    def _consume_glow_restore(self) -> List[str]:
        """Return the restore commands for any ACTIVE disulfide glow AND clear its state + the tab's
        Clear button — the single 'turn the glow off' primitive. Reused by `_push_3d_color` (a
        different visual mode is taking over) and the explicit Clear action. [] when nothing glows."""
        cmds = self._glow_restore_commands(getattr(self, "_glow_state", None))
        if cmds:
            self._glow_state = None
            self._sync_glow_clear_button()
        return cmds

    def _clear_disulfide_glow(self) -> None:
        """Explicit 'Clear disulfide view' — restore the normal representation (un-ghost + un-highlight)
        WITHOUT applying a new colour mode. The manual escape from the pair-click spotlight."""
        self._run_commands_bg(self._consume_glow_restore())

    def _apply_or_toggle_glow(self, new_state: Optional[dict], apply_cmds: List[str]) -> None:
        """Apply a new 3D glow — OR, when the requested glow is IDENTICAL to the one already active,
        TOGGLE it OFF (restore the standard view). Re-clicking the highlighted row is thus a visible
        escape hatch on the item itself, complementing the 'Clear … view' button. Always restores any
        PRIOR glow first so spotlights never stack. *new_state* is the ``{mid, both}`` the glow records
        (None when the spec is degenerate); *apply_cmds* are the highlight commands."""
        prior = getattr(self, "_glow_state", None)
        if new_state is not None and new_state == prior:    # same item re-clicked → un-glow
            self._run_commands_bg(self._consume_glow_restore())
            return
        self._run_commands_bg(self._glow_restore_commands(prior) + apply_cmds)
        self._glow_state = new_state
        self._sync_glow_clear_button()

    def _sync_glow_clear_button(self) -> None:
        """Light the Clear-view control on EVERY stabilization tab iff a glow is currently active. The
        glow seam (`_glow_state`) is SHARED across the Disulfides + Proline tabs (one spotlight at a
        time), so both Clear controls track the one state."""
        active = bool(getattr(self, "_glow_state", None))
        for tab_name in ("disulfides_tab", "proline_tab", "cavity_tab", "saltbridge_tab"):
            tab = getattr(self, tab_name, None)
            if tab is not None:
                tab.set_glow_active(active)

    def _highlight_disulfide_pair(self, cd, pair: dict) -> None:
        """Glow a pair's two members in 3D — the §9 unified-highlighting GOOD CITIZEN: routes through
        the EXISTING `_run_commands_bg` selection seam (no new scattered highlight path; this tab is
        the convergence surface). RESTORE the prior glow FIRST (so A un-glows + the scene un-ghosts
        before B glows — non-destructive, never stacking), apply the new glow, then record its state."""
        apply = self._disulfide_scan_highlight_commands(cd, pair)
        mid, _a, _b, both = self._disulfide_pair_specs(cd, pair)
        new_state = {"mid": mid, "both": both} if (mid is not None and apply) else None
        self._apply_or_toggle_glow(new_state, apply)   # re-clicking the same pair toggles it off

    def _declare_disulfide_pair(self, cd, pair) -> None:
        """Declare a bond from a Disulfides-tab table → Mode C. FOCUS the pair's construct tab first
        so the introduce path targets the RIGHT cd (not whatever chain tab is active), then run the
        SAME `_declare_disulfide_from_scan` loop the menu uses. *pair* is the full pair dict (intra
        Mode D OR cross-chain interface) or a bare (a,b) tuple — forwarded as-is so the chains survive."""
        self._focus_tab_for_design(cd)
        self._declare_disulfide_from_scan(pair)

    @staticmethod
    def _ss_pair_label(pair) -> str:
        """Display label for a declared pair — `pair_label` for a dict (cross-chain shows both
        chains), bare `a–b` for a (a,b) tuple."""
        return pair_label(pair) if isinstance(pair, dict) else f"{pair[0]}–{pair[1]}"

    def _declare_disulfide_from_scan(self, pair) -> None:
        """Feed a scanned pair into the Mode-C introduce→constrain loop (the engineering loop's
        D→C / interface→C step) — the SAME `build_disulfide_introduce_spec` the menu uses, which
        routes an intra pair to the intra path and a cross-chain pair to the proven cross-chain path."""
        spec = self.build_disulfide_introduce_spec([pair], engine="boltz")
        lbl = self._ss_pair_label(pair)
        if spec is None:
            self._status.setText(f"Could not declare a bond at {lbl} "
                                 f"(check both are valid construct positions).")
            return
        self._persist()
        self._status.setText(f"Introduced Cys for {lbl}; folding WITH the declared "
                             f"bond (biases toward it, geometry measured on the result)…")
        self.launchRequested.emit(spec)

    def _on_fold_clicked(self) -> None:
        tab = self._cur_tab()
        v = self._active_variant(tab)
        if v is None:
            self._status.setText("Select a VARIANT row first (T is the template baseline).")
            return
        cd = self._cur_tab().design
        # DE-NOVO: a variant folds against the construct's baseline (T-fold), which must exist first.
        # Engine + oligomer are now USER-CHOSEN (un-pinned) — the result lands on the variant's OWN
        # fold (the baseline is untouched); deviation-vs-WT guards/greys when the chosen combo differs
        # from the baseline. Same picker idea as a loaded variant, with mono/di/tri/tetramer.
        if self._design is not None and self._design.source == "sequence":
            if not (cd.template_fold or {}).get("model_id"):
                self._status.setText("Fold the construct first (Fold ▾ → Fold construct) — a "
                                     "variant folds against the construct's baseline.")
                return
            avail = self._fold_engine_availability()
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Fold variant")
            box.setText(f"Fold {v.id} ({len(v.sequence)} aa) — pick engine + oligomer:")
            box.setInformativeText(
                "ESMFold = local monomer (fast). Boltz-2 = higher-quality, LOCAL-ONLY, seed-pinned; "
                "folds mono/di/tri/tetramer. Deviation-vs-WT needs the SAME oligomer as the construct "
                "baseline (it greys otherwise — re-fold to match).")
            combos = [("ESMFold (monomer)", "esmfold", 1), ("Boltz-2 (monomer)", "boltz", 1),
                      ("Boltz-2 (dimer)", "boltz", 2), ("Boltz-2 (trimer)", "boltz", 3),
                      ("Boltz-2 (tetramer)", "boltz", 4)]
            dn_btns: Dict[object, Tuple[str, int]] = {}
            for label, eng, nc in combos:
                b = box.addButton(label, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
                if not avail.get(eng, False):
                    b.setEnabled(False)
                    b.setToolTip("Boltz env (~/boltz_env) not available" if eng == "boltz"
                                 else "Local ESMFold worker (venv312) not installed")
                dn_btns[b] = (eng, nc)
            box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            box.exec()
            dn_choice = dn_btns.get(box.clickedButton())
            if dn_choice is None:
                return
            dn_eng, dn_nc = dn_choice
            spec = self.fold_launch_spec(dn_eng, n_copies=dn_nc)       # UN-PINNED: user's engine + N-mer
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
        # MODE C (additive): if this fold was BIASED by a declared disulfide bond, ALSO record the
        # readout in the persistent Disulfides tab's C section. The model + provenance badge + the
        # `constrained`/`disulfide_bonds` tags already landed above (unchanged) — this only adds the
        # tab readout, it does NOT replace the existing surfacing.
        if f.get("constrained"):
            self.disulfides_tab.populate_constraint(cd, {
                "disulfide_bonds": f.get("disulfide_bonds") or [],
                "model_id": f.get("model_id"), "variant_id": v.id})

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
            # Reuse the construct T-fold as the reference ONLY when it was folded the SAME way as the
            # variant (engine:oligomer). An un-pinned variant folded at a different shape has no valid
            # baseline reference → leave None (the _on_deviation_clicked guard refuses; a per-oligomer
            # reference fold is the deferred "adapt" follow-up).
            if f"{tf.get('engine')}:{tf.get('target')}" == combo:
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
        # GREY-OUT + WARN (un-pinned-fold guard): deviation compares per-residue vs a WT reference
        # folded the SAME way (engine:oligomer). A variant folded at a different shape than the
        # construct baseline (now possible) has no valid reference — refuse rather than mis-compare a
        # monomer to a trimer. (The per-oligomer WT reference fold is the deferred "adapt" follow-up.)
        if self._design.source == "sequence":
            combo = f"{v.results.fold.get('engine')}:{v.results.fold.get('target')}"
            tf = tab.design.template_fold or {}
            if combo not in tab.design.wt_refs and f"{tf.get('engine')}:{tf.get('target')}" != combo:
                self._status.setText(
                    f"Deviation-vs-WT needs a WT reference folded the same way as {v.id} ({combo}); "
                    f"the construct baseline is {tf.get('engine')}:{tf.get('target')}. Re-fold {v.id} "
                    f"to match the baseline (a per-oligomer reference is coming) — refusing to "
                    f"compare different fold shapes.")
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
            if fold.get("constrained"):             # PROVENANCE: biased by a declared disulfide bond
                nb = len(fold.get("disulfide_bonds") or [])
                parts.append(f"SS-bond×{nb}" if nb else "SS-bond")
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
        if self._mode_key == _RESULT_DISULFIDE_MODE:
            # The engineerability heatmap colours the construct's T-FOLD model (where the scanned
            # backbone lives) by each residue's BEST-partner score. A navigational index — the
            # ranked pair-list is the data. The rep chain's CIF numbering == the author resnums.
            best = self._active_disulfide_scan(tab)
            tf = tab.design.template_fold or {}
            mid = tf.get("model_id")
            if not best or not mid:
                return []
            rep = tab.design.rep_chain
            return build_model_color_commands(
                mid, {rep: sorted(best)},
                lambda ch, rn: disulfide_compat_color(best.get(int(rn))))
        if self._mode_key == _RESULT_SALTBRIDGE_MODE:
            # The salt-bridge heatmap colours the construct's T-FOLD model by each residue's best
            # charge-pair score (pale→electric-blue) — a navigational index into the ranked tables.
            best = self._active_saltbridge_scan(tab)
            tf = tab.design.template_fold or {}
            mid = tf.get("model_id")
            if not best or not mid:
                return []
            rep = tab.design.rep_chain
            return build_model_color_commands(
                mid, {rep: sorted(best)},
                lambda ch, rn: saltbridge_compat_color(best.get(int(rn))))
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
