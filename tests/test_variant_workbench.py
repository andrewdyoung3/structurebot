"""
tests/test_variant_workbench.py
-------------------------------
Workbench panel logic (Qt offscreen, controller mocked): one tab per unique chain,
the column→3D select specs (ALL copies — the coupling the live-verify confirms on
real ChimeraX), gap columns select nothing, and load persists the DesignSession.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("PySide6")
from PySide6 import QtWidgets, QtCore, QtGui

from seq_editor.controller import ResidueCell, ChainSeq
from variant_workbench import (VariantWorkbenchPanel, _ChainDesignTab, _RESULT_DDG_MODE,
                               _RESULT_PLDDT_MODE, _RESULT_DEVIATION_MODE, _ROW_ROLE)
from variant_model import ChainDesign, AlignedCell, build_design_session
from color_modes import get_mode, ddg_color, plddt_color


@pytest.fixture(scope="module")
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _chainseq(model, chain, seq, start=1):
    return ChainSeq(model, chain,
                    [ResidueCell(model, chain, start + i, aa, i + 1)
                     for i, aa in enumerate(seq)])


def _panel(chainseqs, session=None):
    # a no-op pool so off-thread workers never dispatch in unit tests (we assert the
    # deterministic command surface via color_commands_for / select_specs_for_column).
    c = MagicMock()
    c.load_model.return_value = chainseqs
    return VariantWorkbenchPanel(c, session=session, pool=MagicMock()), c


def _set_mode(panel, key):
    for i in range(panel._mode_combo.count()):
        if panel._mode_combo.itemData(i) == key:
            panel._mode_combo.setCurrentIndex(i)
            return


class TestPanel:
    def test_homo_oligomer_one_tab(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")])
        p.load_model("1")
        assert p._tabs.count() == 1

    def test_hetero_two_tabs(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "WYF")])
        p.load_model("1")
        assert p._tabs.count() == 2

    def test_column_select_specs_all_copies_nonstart(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV", start=50),
                       _chainseq("1", "B", "MKV", start=50)])
        p.load_model("1")
        cd = next(iter(p._design.chains.values()))
        assert p.select_specs_for_column(cd, 2) == [("1", "A", [52]), ("1", "B", [52])]

    def test_gap_column_selects_nothing(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MK")])
        p.load_model("1")
        cd = ChainDesign("k", "1", "A", [("1", "A")], [AlignedCell(0, None, None)])
        assert p.select_specs_for_column(cd, 0) == []

    def test_load_persists_design_session(self, _app):
        sess = MagicMock()
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=sess)
        p.load_model("1")
        sess.add_design_session.assert_called_once()
        mid, payload = sess.add_design_session.call_args.args
        assert mid == "1" and "chains" in payload


class TestStage2:
    def test_add_variant_grows_rows_and_persists(self, _app):
        sess = MagicMock()
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=sess)
        p.load_model("1")
        tab = p._cur_tab()
        p._add_variant()
        assert len(tab.design.variants) == 1
        assert tab.active_row_id == tab.design.variants[0].id   # new row becomes active
        assert sess.add_design_session.call_count == 2          # load + add

    def test_edit_via_target_updates_cell_and_persists(self, _app):
        sess = MagicMock()
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=sess)
        p.load_model("1")
        tab = p._cur_tab()
        p._add_variant()
        vid = tab.design.variants[0].id
        p._on_cell(tab, vid, 1)                  # click V1 col1 → edit target (K)
        p._aa_combo.setCurrentText("A")
        p._apply_substitution()
        assert tab.design.get_variant(vid).cells[1].aa == "A"
        assert sess.add_design_session.call_count == 3          # load + add + edit

    def test_color_mode_paints_panel_matching_3d(self, _app):
        # the sync invariant: the T-row panel cell color == the hex the 3D command uses.
        p, _ = _panel([_chainseq("1", "A", "MKRDE")])           # K,R blue; D,E red
        p.load_model("1")
        tab = p._cur_tab()
        _set_mode(p, "charge")
        charge = get_mode("charge")
        # panel: col1 = K → strong blue; col3 = D → red
        assert tab.color_hex_at("T", 1) == charge.color_for("K")
        assert tab.color_hex_at("T", 3) == charge.color_for("D")
        # 3D commands carry the same hexes (active row defaults to T)
        cmds = " ".join(p.color_commands_for(tab))
        assert charge.color_for("K")[1:] in cmds.replace("#", "")   # blue present
        assert charge.color_for("D")[1:] in cmds.replace("#", "")   # red present

    def test_active_variant_edit_drives_3d_color(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")])
        p.load_model("1")
        tab = p._cur_tab()
        p._add_variant()
        vid = tab.design.variants[0].id
        _set_mode(p, "charge")
        p._on_cell(tab, vid, 2)                  # V col2 (V) active + edit target
        p._aa_combo.setCurrentText("D")
        p._apply_substitution()                  # V→D : col2 now negative
        cmds = p.color_commands_for(tab)
        red = get_mode("charge").color_for("D")
        # active row is the edited variant → resnum 3 colored red on BOTH copies
        assert f"color #1/A:3 {red}" in cmds and f"color #1/B:3 {red}" in cmds

    def test_neutral_cell_is_white_under_active_mode(self, _app):
        # the sync invariant must hold for NO-OPINION residues too: under an active mode
        # a neutral cell is white (#ffffff) — exactly the 3D reset — not the T row tint.
        p, _ = _panel([_chainseq("1", "A", "MKV")])     # M is neutral under charge
        p.load_model("1")
        tab = p._cur_tab()
        _set_mode(p, "charge")
        assert tab.color_hex_at("T", 0) == "#ffffff"
        _set_mode(p, "none")                            # OFF → row default (T tint) returns
        assert tab.color_hex_at("T", 0) == "#eef4ff"

    def test_none_mode_pushes_no_3d(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV")])
        p.load_model("1")
        tab = p._cur_tab()
        assert p.color_commands_for(tab) == []   # OFF mode is non-destructive


class TestCellEditMenu:
    """Direct residue substitution from the sequence view: the cell context menu exposes the
    EXISTING edit_variant substitution path (right-click a variant residue). Deletion/indels
    are deferred (they'd shift the resnum-keyed S4c deviation numbering)."""

    def _variant(self, seq="MKV"):
        sess = MagicMock()
        p, _ = _panel([_chainseq("1", "A", seq), _chainseq("1", "B", seq)], session=sess)
        p.load_model("1")
        tab = p._cur_tab()
        p._add_variant()
        return p, tab, tab.design.variants[0].id, sess

    def test_do_substitute_edits_cell_and_persists(self, _app):
        p, tab, vid, sess = self._variant()
        p._do_substitute(tab, vid, 1, "A")          # K2A
        assert tab.design.get_variant(vid).cells[1].aa == "A"
        assert sess.add_design_session.call_count == 3   # load + add + substitute

    def test_revert_to_wt_via_substitute(self, _app):
        p, tab, vid, _ = self._variant()
        p._do_substitute(tab, vid, 1, "A")          # K2A
        p._do_substitute(tab, vid, 1, "K")          # revert to WT
        v = tab.design.get_variant(vid)
        assert v.cells[1].aa == "K" and v.mutations == []

    def test_context_menu_only_for_variant_cells(self, _app):
        # the tab emits cellMenuRequested for a VARIANT residue but NOT for T / ruler / gaps.
        # Use a STANDALONE tab so the only slot on cellMenuRequested is this probe — a
        # panel-attached tab also wires _show_cell_menu, whose modal menu.exec() would block.
        cd = next(iter(build_design_session([_chainseq("1", "A", "MKV")], "1").chains.values()))
        vid = cd.add_variant("V1").id
        tab = _ChainDesignTab(cd)
        seen = []
        tab.cellMenuRequested.connect(lambda rid, col, gp: seen.append((rid, col)))
        block = tab._blocks[0]
        v_item = next(block.item(r, 0) for r in range(block.rowCount())
                      if block.item(r, 0) and block.item(r, 0).data(_ROW_ROLE) == vid)
        t_item = next(block.item(r, 0) for r in range(block.rowCount())
                      if block.item(r, 0) and block.item(r, 0).data(_ROW_ROLE) == "T")
        tab._on_context_menu(block, block.visualItemRect(v_item).center())
        tab._on_context_menu(block, block.visualItemRect(t_item).center())
        assert seen == [(vid, 0)]                   # only the variant cell raised a menu


class TestFoldOutputUsability:
    """Fold-output UX: auto-surface the result mode on a fresh fold/deviation, REPLACE the
    prior model on re-fold (no stacking), and TILE the specific fold models side-by-side."""

    def _folded_panel(self):
        sess = MagicMock()
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")], session=sess)
        p.load_model("1")
        tab = p._cur_tab()
        p._add_variant()
        vid = tab.design.variants[0].id
        tab.design.edit_variant(vid, 0, "W")
        return p, tab, vid

    @staticmethod
    def _fold_result(model_id="2", engine="esmfold"):
        return {"tool_step_results": [{"tool": engine, "data": {
            "engine": engine, "target": "monomer", "new_model_id": model_id,
            "reference_model_id": "1", "mean_plddt": 80.0, "length": 3,
            "source": "local_venv312", "plddt": {1: 90.0, 2: 80.0, 3: 70.0}}}]}

    def test_auto_surface_plddt_on_fold(self, _app):
        p, tab, vid = self._folded_panel()
        p._mode_key = "none"
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        assert p._mode_key == _RESULT_PLDDT_MODE                  # auto-surfaced, no manual step
        assert p._mode_combo.currentData() == _RESULT_PLDDT_MODE  # combo display synced

    def test_no_close_on_first_fold(self, _app):
        p, tab, vid = self._folded_panel()
        captured = []
        p._run_commands_bg = lambda cmds: captured.extend(cmds)
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        assert not any(str(c).startswith("close ") for c in captured)

    def test_replace_on_refold_closes_prior(self, _app):
        p, tab, vid = self._folded_panel()
        captured = []
        p._run_commands_bg = lambda cmds: captured.extend(cmds)
        p.apply_fold_result(vid, self._fold_result(model_id="2"))   # first fold
        captured.clear()
        p.apply_fold_result(vid, self._fold_result(model_id="5"))   # re-fold → replace #2
        assert "close #2" in captured

    def test_auto_surface_deviation(self, _app):
        p, tab, vid = self._folded_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        dev = {"tool_step_results": [{"tool": "variant_deviation", "data": {
            "engine": "esmfold", "target": "monomer", "multichain": False,
            "variant_chain": "A", "variant_model_id": "2", "reference_model_id": "7",
            "deviation": {"1": 0.1}, "floor": {}, "anchor_residual_rmsd": 0.01,
            "all_pairs_rmsd": 0.5, "n_residues": 1, "n_cleared_floor": 0,
            "max_deviation": 0.1, "floor_kind": "deterministic",
            "wt_ref": {"engine": "esmfold", "target": "monomer", "model_id": "7", "floor": {}}}}]}
        p._mode_key = "none"
        p.apply_deviation_result(vid, dev)
        assert p._mode_key == _RESULT_DEVIATION_MODE

    def test_tile_commands_targets_specific_models(self, _app):
        p, tab, v1 = self._folded_panel()
        p._add_variant()
        v2 = tab.design.variants[1].id
        tab.design.edit_variant(v2, 0, "Y")
        p.apply_fold_result(v1, self._fold_result(model_id="2"))
        p.apply_fold_result(v2, self._fold_result(model_id="3"))
        cmds = p.tile_commands()
        assert cmds[-1] == "tile #1 #2 #3"            # reference + folds, SPECIFIC ids
        assert {"show #1 models", "show #2 models", "show #3 models"} <= set(cmds)

    def test_tile_commands_empty_under_two(self, _app):
        p, tab, vid = self._folded_panel()            # no folds → only the reference (#1)
        assert p.tile_commands() == []

    def test_result_mode_nonactive_row_stays_readable(self, _app):
        # legibility: under a result mode the active row shows the result colour; NON-active
        # rows get an EXPLICIT dim-but-readable bg (never white-blanked or clear→dark).
        from variant_workbench import _DIM_BG
        cd = next(iter(build_design_session([_chainseq("1", "A", "MKV")], "1").chains.values()))
        cd.add_variant("V1")
        cd.add_variant("V2")
        tab = _ChainDesignTab(cd)
        tab.set_result_coloring("V1", {1: "#0053d6"})    # active V1, dark-blue result at resnum 1
        assert tab.color_hex_at("V1", 0) == "#0053d6"    # active row coloured
        assert tab.color_hex_at("T", 0) != "#ffffff"     # non-active T keeps its tint, not white
        assert tab.color_hex_at("V2", 0) == _DIM_BG.name()  # non-active variant → explicit dim bg

    def test_contrast_fg_dark_vs_light(self, _app):
        from variant_workbench import _contrast_fg
        assert _contrast_fg(QtGui.QColor("#0053d6")).name() == "#ffffff"   # dark bg → white glyph
        assert _contrast_fg(QtGui.QColor("#ffffff")).name() == "#1a1a1a"   # light bg → dark glyph


class TestRowHeaderSelect:
    """A SINGLE row-header click anywhere SELECTS the variant (active row, silent — never a
    modal, even for a FOLDED variant that carries a badge). DOUBLE-click → result detail.
    Prerequisite for the active-row HIDE switching (the old name/badge x-split misfired)."""

    def _tab_with_variant(self, badge=None):
        cd = next(iter(build_design_session([_chainseq("1", "A", "MKV")], "1").chains.values()))
        cd.add_variant("V1")
        tab = _ChainDesignTab(cd)
        if badge:
            tab.badges["V1"] = badge
            tab.rebuild()
        return tab

    @staticmethod
    def _evt(kind):
        p = QtCore.QPointF(5.0, 5.0)
        return QtGui.QMouseEvent(kind, p, p, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton,
                                 QtCore.Qt.NoModifier)

    def _fire(self, tab, monkeypatch):
        block = tab._blocks[0]
        vh = block.verticalHeader()
        section = tab._row_ids.index("V1")
        monkeypatch.setattr(vh, "logicalIndexAt", lambda _y: section)
        sel = []
        tab.rowHeaderSelected.connect(sel.append)
        tab.eventFilter(vh.viewport(), self._evt(QtCore.QEvent.Type.MouseButtonPress))
        return sel

    def test_single_click_selects_no_badge(self, _app, monkeypatch):
        tab = self._tab_with_variant(badge=None)
        assert self._fire(tab, monkeypatch) == ["V1"]

    def test_single_click_selects_folded_variant_with_badge(self, _app, monkeypatch):
        # THE BUG FIX: a folded variant carries a (pLDDT) badge; a click must still SELECT it
        # (the old x-split swallowed it into the now-removed detail modal).
        tab = self._tab_with_variant(badge="pLDDT 80 · ipTM 0.96")
        assert self._fire(tab, monkeypatch) == ["V1"]

    def test_no_rowheaderclicked_signal(self, _app):
        # the detail modal is removed/parked — the header has no detail signal anymore
        tab = self._tab_with_variant(badge="pLDDT 80")
        assert not hasattr(tab, "rowHeaderClicked")

    def test_select_variant_row_sets_active_silently(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")],
                      session=MagicMock())
        p.load_model("1")
        tab = p._cur_tab()
        p._add_variant()
        vid = tab.design.variants[0].id
        tab.set_active_row("T")
        p._select_variant_row(tab, vid)              # header-name select (no results on V1)
        assert tab.active_row_id == vid and p._edit_target is None


class TestStage3aImport:
    def _sess(self, mpnn=None, scan=None):
        s = MagicMock()
        s.get_proteinmpnn_result.return_value = mpnn
        s.get_scan_result.return_value = scan
        return s

    def test_import_mpnn_makes_rows_with_provenance(self, _app):
        mpnn = {"chain": "A", "wildtype_sequence": "MKV", "fasta_path": "r.fa",
                "sequences": [{"sequence": "MAV"}, {"sequence": "MKL"}]}
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=self._sess(mpnn=mpnn))
        p.load_model("1")
        p._import_mpnn()
        cd = p._cur_tab().design
        assert len(cd.variants) == 2
        assert all(v.source == "proteinmpnn" for v in cd.variants)
        assert cd.variants[0].provenance["fasta_path"] == "r.fa"

    def test_reimport_is_idempotent(self, _app):
        mpnn = {"chain": "A", "fasta_path": "r.fa",
                "sequences": [{"sequence": "MAV"}]}
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=self._sess(mpnn=mpnn))
        p.load_model("1")
        p._import_mpnn(); p._import_mpnn()                 # twice
        assert len(p._cur_tab().design.variants) == 1     # no duplicate row

    def test_import_targets_the_mpnn_chain(self, _app):
        mpnn = {"chain": "B", "fasta_path": "r.fa", "sequences": [{"sequence": "WYF"}]}
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "WYF")],
                      session=self._sess(mpnn=mpnn))
        p.load_model("1")
        p._import_mpnn()
        # the design owning chain B got the row; chain A's did not
        by_chain = {cd.rep_chain: cd for cd in p._design.chains.values()}
        assert len(by_chain["B"].variants) == 1 and len(by_chain["A"].variants) == 0

    def test_length_mismatch_skipped_with_status(self, _app):
        mpnn = {"chain": "A", "fasta_path": "r.fa", "sequences": [{"sequence": "MKVQQ"}]}
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=self._sess(mpnn=mpnn))
        p.load_model("1")
        p._import_mpnn()
        assert len(p._cur_tab().design.variants) == 0
        assert "align" in p._status.text().lower()


class TestStage3aSuggestions:
    def _scan(self, chain="A"):
        # candidates at resnums 2 and 3 (chain A); resnum 1 has NONE → sparse track
        return [
            {"chain": chain, "resnum": 2, "position": 2, "from_aa": "K", "to_aa": "A",
             "combined_score": 0.6, "recommendation": "good"},
            {"chain": chain, "resnum": 2, "position": 2, "from_aa": "K", "to_aa": "D",
             "combined_score": 1.7, "recommendation": "strong"},
            {"chain": chain, "resnum": 3, "position": 3, "from_aa": "V", "to_aa": "L",
             "combined_score": -0.3, "recommendation": "marginal"},
        ]

    def test_load_suggestions_is_sparse(self, _app):
        s = MagicMock(); s.get_scan_result.return_value = self._scan()
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=s)
        p.load_model("1")
        p._load_suggestions()
        tab = p._cur_tab()
        assert set(tab.suggestions) == {1, 2}      # cols for resnums 2,3; resnum 1 (col0) absent
        assert [c["to_aa"] for c in tab.suggestions[1]] == ["D", "A"]   # sorted desc

    def test_accept_into_active_variant_with_provenance(self, _app):
        s = MagicMock(); s.get_scan_result.return_value = self._scan()
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=s)
        p.load_model("1")
        p._load_suggestions()
        tab = p._cur_tab()
        p._add_variant()                            # V1 becomes the active row
        vid = tab.design.variants[0].id
        top = tab.suggestions[1][0]                 # K2D, score 1.7
        p._accept_suggestion(tab, 1, top)
        v = tab.design.get_variant(vid)
        assert v.cells[1].aa == "D"
        assert v.mutations[0].source == "accepted_suggestion"
        assert v.provenance["accepted"][0]["combined_score"] == 1.7

    def test_accept_without_variant_is_guarded(self, _app):
        s = MagicMock(); s.get_scan_result.return_value = self._scan()
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=s)
        p.load_model("1")
        p._load_suggestions()
        tab = p._cur_tab()                          # active row defaults to T
        p._accept_suggestion(tab, 1, tab.suggestions[1][0])
        assert tab.design.variants == []           # nothing edited
        assert "variant" in p._status.text().lower()

    def test_no_scan_no_suggest_track(self, _app):
        s = MagicMock(); s.get_scan_result.return_value = None
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=s)
        p.load_model("1")
        assert p._cur_tab().suggestions == {}


class TestStage3bLaunch:
    """The deterministic launch-spec surface the panel emits to the window (which turns
    it into engine.handle_tool_request — the SAME spine). Asserted directly, like the
    select/color command surfaces; the dialog/emit handlers are thin wrappers over these."""

    def test_ctrlclick_toggles_scan_set(self, _app):
        # Ctrl+click (to_scan=True) is the DISTINCT scan-set gesture.
        p, _ = _panel([_chainseq("1", "A", "MKVLA")])
        p.load_model("1")
        tab = p._cur_tab()
        p._on_cell(tab, "T", 0, to_scan=True)
        p._on_cell(tab, "T", 2, to_scan=True)
        assert p._scan_cols == {0, 2}
        p._on_cell(tab, "T", 0, to_scan=True)       # second ctrl+click toggles it back off
        assert p._scan_cols == {2}
        assert p._scan_set_lbl.text() == "scan set: 1"

    def test_plain_click_is_s2_edit_target_not_scan_set(self, _app):
        # The disambiguation: a PLAIN click keeps its full S2 meaning (active row + edit
        # target) and must NOT touch the scan set — editing stays as easy as in S2.
        p, _ = _panel([_chainseq("1", "A", "MKV")])
        p.load_model("1")
        tab = p._cur_tab()
        p._add_variant()                            # V1 active
        vid = tab.design.variants[0].id
        p._on_cell(tab, vid, 1)                     # plain click on a variant cell
        assert p._edit_target == (vid, 1)          # S2 edit target set
        assert p._scan_cols == set()               # scan set untouched
        # and a plain click never grows the scan set even after several clicks
        p._on_cell(tab, vid, 0); p._on_cell(tab, "T", 2)
        assert p._scan_cols == set()

    def test_scan_spec_empty_set_is_whole_chain(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV")])
        p.load_model("1")
        spec = p.scan_launch_spec(deep=False)
        assert spec["tool"] == "mutation_scan"
        assert "scan_positions" not in spec["tool_inputs"]
        assert "run_rosetta" not in spec["tool_inputs"]
        assert spec["confidence"] == "high" and spec["refresh"] == "scan"

    def test_scan_spec_deep_presets_rosetta_and_low_confidence(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV", start=10)])
        p.load_model("1")
        tab = p._cur_tab()
        p._on_cell(tab, "T", 0, to_scan=True); p._on_cell(tab, "T", 2, to_scan=True)   # resnums 10, 12
        spec = p.scan_launch_spec(deep=True)
        assert spec["tool_inputs"]["scan_positions"] == [10, 12]
        assert spec["tool_inputs"]["run_rosetta"] is True
        assert spec["confidence"] == "low"          # deep → explicit confirm, no auto-proceed

    def test_scan_spec_user_input_carries_no_text_triggers(self, _app):
        # The tier/scope come from tool_inputs; the label must NOT smuggle a token the
        # spine's tiering would parse — incl. "selected"/"selection"/"highlighted", which
        # would hijack the scope to the (empty) live ChimeraX selection. (This is the bug
        # the Stage-3b live-verify caught: "N selected position(s)" zeroed the scope.)
        p, _ = _panel([_chainseq("1", "A", "MKV", start=10)])
        p.load_model("1")
        tab = p._cur_tab(); p._on_cell(tab, "T", 0, to_scan=True)
        for deep in (True, False):
            ui = p.scan_launch_spec(deep=deep)["user_input"].lower()
            for tok in ("rosetta", "rosie", "proline", "glyco", "exhaustive",
                        "comprehensive", "deep-dive", "deep dive", "gold-standard",
                        "shortlist", "asymmetric", "selected", "selection", "highlighted"):
                assert tok not in ui, f"label leaked trigger {tok!r}: {ui!r}"
            # no "residue(s)/position(s) <digits>" explicit-scope pattern either
            assert not re.search(r"(residues?|positions?)\s+\d", ui)
        m_ui = p.mpnn_launch_spec(soluble=True)["user_input"].lower()
        for tok in ("selected", "selection", "highlighted"):
            assert tok not in m_ui

    def test_mpnn_spec_default_and_soluble(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV", start=5)])
        p.load_model("1")
        tab = p._cur_tab()
        p._on_cell(tab, "T", 1, to_scan=True)       # resnum 6
        d = p.mpnn_launch_spec(soluble=False)
        assert d["tool"] == "proteinmpnn" and d["refresh"] == "mpnn"
        assert d["tool_inputs"]["chain_id"] == "A"
        assert d["tool_inputs"]["design_positions"] == [6]
        assert "bias_toward" not in d["tool_inputs"]
        s = p.mpnn_launch_spec(soluble=True)
        assert s["tool_inputs"]["bias_toward"] == "soluble"

    def test_mpnn_spec_empty_set_whole_chain(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV")])
        p.load_model("1")
        d = p.mpnn_launch_spec(soluble=False)
        assert "design_positions" not in d["tool_inputs"]

    def test_clear_scan_set(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV")])
        p.load_model("1")
        p._on_cell(p._cur_tab(), "T", 0, to_scan=True)
        assert p._scan_cols
        p._clear_scan_set()
        assert p._scan_cols == set() and p._scan_set_lbl.text() == "scan set: 0"

    def test_tab_change_resets_scan_set(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "WYF")])
        p.load_model("1")
        p._on_cell(p._cur_tab(), "T", 0, to_scan=True)
        assert p._scan_cols
        p._tabs.setCurrentIndex(1)                  # fires _on_tab_changed
        assert p._scan_cols == set()

    def test_launch_spec_none_without_structure(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV")])  # load_model NOT called
        assert p.scan_launch_spec(deep=False) is None
        assert p.mpnn_launch_spec(soluble=False) is None

    def test_gap_column_not_added_to_scan_set(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MK")])
        p.load_model("1")
        tab = p._cur_tab()
        # swap in a single-cell gap design (resnum_for_col(0) is None) — the toggle path
        # reads tab.design.resnum_for_col, so a gap column must not enter the scan set.
        tab.design = ChainDesign("k", "1", "A", [("1", "A")], [AlignedCell(0, None, None)])
        p._on_cell(tab, None, 0, to_scan=True)
        assert p._scan_cols == set()


class TestStage4a:
    """Per-variant action buttons → ResultSlots → badges + the per-residue ddG result
    color mode. Pure surfaces asserted directly (the launch spec, the result-apply, the
    color commands); QMessageBox dialogs are thin wrappers not exercised here."""

    def _variant_panel(self):
        # one variant (V1) with a single mutation M1W on chain A
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")],
                      session=MagicMock())
        p.load_model("1")
        p._add_variant()                            # V1 is active
        tab = p._cur_tab()
        vid = tab.design.variants[0].id
        tab.design.edit_variant(vid, 0, "W")        # resnum 1: M→W
        return p, tab, vid

    def test_stability_spec_scores_exact_mutations(self, _app):
        p, tab, vid = self._variant_panel()
        spec = p.stability_launch_spec(deep=False)
        assert spec["tool"] == "mutation_scan" and spec["refresh"] == "stability"
        assert spec["_variant_id"] == vid
        assert spec["tool_inputs"]["score_mutations"] == {1: "W"}
        assert "run_rosetta" not in spec["tool_inputs"] and spec["confidence"] == "high"

    def test_stability_spec_deep_gates(self, _app):
        p, tab, vid = self._variant_panel()
        spec = p.stability_launch_spec(deep=True)
        assert spec["tool_inputs"]["run_rosetta"] is True
        assert spec["confidence"] == "low"          # deep → confirm-gate, no auto-proceed

    def test_stability_spec_none_for_template_or_no_mutations(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=MagicMock())
        p.load_model("1")
        assert p.stability_launch_spec(deep=False) is None      # active row is T
        p._add_variant()                            # V1 active but no mutations yet
        assert p.stability_launch_spec(deep=False) is None

    def test_stability_label_has_no_trigger_tokens(self, _app):
        p, tab, vid = self._variant_panel()
        ui = p.stability_launch_spec(deep=True)["user_input"].lower()
        for tok in ("rosetta", "rosie", "selected", "selection", "highlighted",
                    "exhaustive", "comprehensive"):
            assert tok not in ui

    def test_apply_stability_result_fills_slots_and_badge(self, _app):
        p, tab, vid = self._variant_panel()
        result = {"tool_step_results": [{"tool": "mutation_scan", "data": {"candidates": [
            {"resnum": 1, "from_aa": "M", "to_aa": "W", "ddg": 2.0, "combined_score": -0.1}]}}]}
        p._scan_cache_snapshot = ("1", None)        # as set by _on_test_stability
        p.apply_stability_result(vid, result)
        v = tab.design.get_variant(vid)
        assert v.results.stability["per_resnum"] == {1: 2.0}
        assert v.results.stability["sum_ddg"] == 2.0
        assert "ddG +2.0" in tab.badges[vid]        # inline badge rendered

    def test_solubility_pure_compute_fills_slot(self, _app):
        p, tab, vid = self._variant_panel()
        p._on_test_solubility()
        v = tab.design.get_variant(vid)
        assert set(v.results.solubility) == {"variant", "wt", "delta"}
        assert v.results.solubility["delta"] == round(
            v.results.solubility["variant"] - v.results.solubility["wt"], 3)
        assert "sol" in tab.badges[vid]

    def test_result_ddg_color_mode_paints_active_variant_all_copies(self, _app):
        p, tab, vid = self._variant_panel()
        result = {"tool_step_results": [{"tool": "mutation_scan", "data": {"candidates": [
            {"resnum": 1, "from_aa": "M", "to_aa": "W", "ddg": 3.0}]}}]}
        p._scan_cache_snapshot = ("1", None)
        p.apply_stability_result(vid, result)
        p._mode_key = _RESULT_DDG_MODE
        cmds = p.color_commands_for(tab)            # active row is V1 (has the result)
        red = ddg_color(3.0)
        assert f"color #1/A:1 {red}" in cmds and f"color #1/B:1 {red}" in cmds  # all copies

    def test_result_ddg_mode_no_result_is_empty(self, _app):
        p, tab, vid = self._variant_panel()         # V1 has no stability result yet
        p._mode_key = _RESULT_DDG_MODE
        assert p.color_commands_for(tab) == []


class TestStage4b:
    """Engine-agnostic monomer fold seam: the launch spec, the result-apply into
    ResultSlots.fold, the pLDDT result colour mode (panel + predicted model), per-model
    visibility (active-row coupling + global toggle), and the engine-picker capability flag."""

    def _variant_panel(self, n_variants=1):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")],
                      session=MagicMock())
        p.load_model("1")
        ids = []
        for _ in range(n_variants):
            p._add_variant()
            tab = p._cur_tab()
            vid = tab.design.variants[-1].id
            tab.design.edit_variant(vid, 0, "W")    # give it a mutation (M1W)
            ids.append(vid)
        return p, p._cur_tab(), ids

    @staticmethod
    def _fold_result(model_id="2", ref="1", plddt=None, source="local_venv312"):
        plddt = plddt or {1: 95.0, 2: 80.0, 3: 40.0}
        return {"tool_step_results": [{"tool": "esmfold", "data": {
            "engine": "esmfold", "new_model_id": model_id, "reference_model_id": ref,
            "mean_plddt": round(sum(plddt.values()) / len(plddt), 1), "length": len(plddt),
            "source": source, "plddt": plddt}}]}

    def test_fold_launch_spec_shape(self, _app):
        p, tab, (vid,) = self._variant_panel()
        spec = p.fold_launch_spec("esmfold")
        assert spec["tool"] == "esmfold" and spec["refresh"] == "fold"
        assert spec["_variant_id"] == vid and spec["confidence"] == "low"   # → confirm-gate
        ti = spec["tool_inputs"]
        assert ti["open_model"] is True and ti["local_only"] is True
        assert ti["compare_to"] == "1" and ti["engine"] == "esmfold"
        assert ti["sequence"] == tab.design.get_variant(vid).sequence

    def test_fold_launch_spec_none_for_template(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV")], session=MagicMock())
        p.load_model("1")                            # active row is T
        assert p.fold_launch_spec("esmfold") is None

    def test_apply_fold_result_fills_slot_and_badge(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result())
        v = tab.design.get_variant(vid)
        f = v.results.fold
        assert f["model_id"] == "2" and f["source"] == "local_venv312"
        assert f["plddt"] == {1: 95.0, 2: 80.0, 3: 40.0}   # author-resnum-keyed
        assert "pLDDT 72" in tab.badges[vid]               # mean 71.7 → 72

    def test_local_only_breach_marker_in_source(self, _app):
        # the contract surfaces source; a non-local source is visible (not silently trusted)
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(source="atlas_api"))
        assert tab.design.get_variant(vid).results.fold["source"] == "atlas_api"

    def test_plddt_color_mode_targets_predicted_model(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        p._mode_key = _RESULT_PLDDT_MODE
        cmds = p.color_commands_for(tab)             # active row is V1 (has the fold)
        # colour ONLY — visibility is owned by fold_visibility_commands (no `show` here, or
        # it would re-show a model the Variant-fold toggle just hid).
        assert cmds == ["color byattribute bfactor #2 palette alphafold"]

    def test_plddt_mode_does_not_defeat_variant_fold_toggle(self, _app):
        # Regression: folding AUTO-surfaces pLDDT mode; in that mode color_commands_for used
        # to emit `show #mid`, which runs AFTER fold_visibility_commands in _push_3d_color and
        # silently re-showed the fold the "Variant fold" toggle had just hidden → the toggle
        # appeared to do nothing (the user-reported Boltz-multimer symptom). The COMBINED push
        # must hide the active fold and never re-show it.
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))   # → active row V1, pLDDT mode
        tab.set_active_row(vid)
        assert p._mode_key == _RESULT_PLDDT_MODE                    # auto-surfaced by the fold
        p._show_fold_cb.setChecked(False)                          # turn "Variant fold" OFF
        combined = p.fold_visibility_commands(tab) + p.color_commands_for(tab)
        assert "hide #2 models" in combined
        assert "show #2 models" not in combined                    # NOT re-shown by the colour push

    def test_plddt_color_mode_no_fold_is_empty(self, _app):
        p, tab, (vid,) = self._variant_panel()       # V1 not folded yet
        p._mode_key = _RESULT_PLDDT_MODE
        assert p.color_commands_for(tab) == []

    def test_fold_visibility_couples_to_active_row(self, _app):
        p, tab, (v1, v2) = self._variant_panel(n_variants=2)
        p.apply_fold_result(v1, self._fold_result(model_id="2"))
        p.apply_fold_result(v2, self._fold_result(model_id="3"))
        tab.set_active_row(v2)
        cmds = p.fold_visibility_commands(tab)
        assert "show #3 models" in cmds and "hide #2 models" in cmds

    def test_fold_visibility_global_hide_toggle(self, _app):
        p, tab, (v1, v2) = self._variant_panel(n_variants=2)
        p.apply_fold_result(v1, self._fold_result(model_id="2"))
        p.apply_fold_result(v2, self._fold_result(model_id="3"))
        p._fold_vis_btn.setChecked(True)             # "Hide folds"
        cmds = p.fold_visibility_commands(tab)
        # global Hide-folds hides every fold; the WT reference (#1) stays shown by default.
        assert {"hide #2 models", "hide #3 models"} <= set(cmds)
        assert not any(c.startswith("show #2") or c.startswith("show #3") for c in cmds)
        assert "show #1 models" in cmds              # reference toggle (default on) independent

    def test_switch_back_reshows_active_fold(self, _app):
        # the HIDE design's whole point: re-selecting a folded variant RE-SHOWS its fold.
        p, tab, (v1, v2) = self._variant_panel(n_variants=2)
        p.apply_fold_result(v1, self._fold_result(model_id="2"))
        p.apply_fold_result(v2, self._fold_result(model_id="3"))
        tab.set_active_row(v2)
        cmds = p.fold_visibility_commands(tab)
        assert "show #3 models" in cmds and "hide #2 models" in cmds
        tab.set_active_row(v1)                       # switch BACK to v1
        cmds = p.fold_visibility_commands(tab)
        assert "show #2 models" in cmds and "hide #3 models" in cmds

    def test_overlay_toggles_independent(self, _app):
        p, tab, (v1,) = self._variant_panel()
        p.apply_fold_result(v1, self._fold_result(model_id="2"))
        tab.set_active_row(v1)
        assert {"show #1 models", "show #2 models"} <= set(p.fold_visibility_commands(tab))
        p._show_ref_cb.setChecked(False)             # Reference OFF → hide the reference only
        cmds = p.fold_visibility_commands(tab)
        assert "hide #1 models" in cmds and "show #2 models" in cmds
        p._show_ref_cb.setChecked(True)
        p._show_fold_cb.setChecked(False)            # Fold OFF → hide the fold, keep reference
        cmds = p.fold_visibility_commands(tab)
        assert "show #1 models" in cmds and "hide #2 models" in cmds

    def test_fold_menu_actions_are_wired(self, _app):
        # Regression guard for the toolbar→menu refactor: the Fold ▾ visibility items are
        # CHECKABLE QActions whose `toggled`/`triggered` must still reach the handlers.
        # The other fold-visibility tests assert the PURE method + setChecked() only, so a
        # DROPPED signal connection would slip past them — this drives the actual user
        # gesture (`QAction.trigger()` == a menu click) and asserts the handler ran.
        p, tab, (v1,) = self._variant_panel()
        p.apply_fold_result(v1, self._fold_result(model_id="2"))
        tab.set_active_row(v1)
        calls = {"n": 0}
        orig = p._push_3d_color
        p._push_3d_color = lambda t: (calls.__setitem__("n", calls["n"] + 1), orig(t))[1]
        for act in (p._fold_vis_btn, p._show_fold_cb, p._show_ref_cb):
            assert act.isCheckable()                 # the toggle survived the QAction conversion
            before_checked, before_n = act.isChecked(), calls["n"]
            act.trigger()                            # == a user click on the menu item
            assert act.isChecked() != before_checked         # state toggled
            assert calls["n"] == before_n + 1                # AND the handler fired (signal live)
        # the global Hide-folds action also updates its own label via its handler
        assert p._fold_vis_btn.text() in ("Hide folds", "Show folds")

    def test_engine_picker_capability_flag(self, _app, monkeypatch):
        # Both engines are SHOWN with a real capability verdict (B2 3-state, never dropped).
        # Mock the probes so the unit test doesn't depend on the live WSL/venv312 envs.
        import esmfold_bridge, boltz_bridge
        monkeypatch.setattr(esmfold_bridge.ESMFoldBridge, "local_available", lambda self: True)
        monkeypatch.setattr(boltz_bridge, "boltz_available", lambda: False)
        p, _, _ = self._variant_panel()
        avail = p._fold_engine_availability()
        assert avail == {"esmfold": True, "boltz": False}
        # and when the boltz env is present, the picker enables it (no longer hard-False)
        monkeypatch.setattr(boltz_bridge, "boltz_available", lambda: True)
        assert p._fold_engine_availability()["boltz"] is True

    # ── Stage 4c: variant-vs-WT deviation launch + apply + floor-gated colour ────────
    @staticmethod
    def _dev_result(variant_mid="2", engine="esmfold", target="monomer", multichain=False,
                    deviation=None, floor=None):
        deviation = deviation or {"1": 0.1, "2": 1.5, "3": 0.2}
        floor = floor or {}
        wt_ref = {"engine": engine, "target": target, "model_id": "7",
                  "floor": floor, "path": "/tmp/wtref"}
        return {"tool_step_results": [{"tool": "variant_deviation", "data": {
            "engine": engine, "target": target, "multichain": multichain,
            "variant_chain": "A", "variant_model_id": variant_mid, "reference_model_id": "7",
            "deviation": deviation, "floor": floor, "anchor_residual_rmsd": 0.01,
            "all_pairs_rmsd": 0.5, "n_residues": len(deviation), "n_cleared_floor": 1,
            "max_deviation": max(deviation.values()), "floor_kind": "deterministic",
            "wt_ref": wt_ref}}]}

    def test_deviation_launch_spec_uncached_low_confidence(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))   # esmfold monomer
        spec = p.deviation_launch_spec()
        assert spec["tool"] == "variant_deviation" and spec["refresh"] == "deviation"
        assert spec["_variant_id"] == vid
        assert spec["confidence"] == "low"          # no cached ref → folds WT → confirm-gate
        ti = spec["tool_inputs"]
        assert ti["variant_model_id"] == "2" and ti["engine"] == "esmfold"
        assert ti["target"] == "monomer" and ti["multichain"] is False
        assert ti["wt_chains"] == [{"id": "A", "sequence": "MKV"}]   # the TEMPLATE T seq, NOT the variant
        assert ti["wt_ref"] is None and ti["compare_to"] == "1"

    def test_deviation_launch_spec_cached_high_confidence(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        tab.design.wt_refs["esmfold:monomer"] = {"model_id": "7", "floor": {}}
        spec = p.deviation_launch_spec()
        assert spec["confidence"] == "high"         # cached ref → cheap → no fold gate
        assert spec["tool_inputs"]["wt_ref"] == {"model_id": "7", "floor": {}}

    def test_deviation_launch_spec_none_when_unfolded(self, _app):
        p, tab, (vid,) = self._variant_panel()      # V1 not folded
        assert p.deviation_launch_spec() is None

    def test_apply_deviation_result_stores_block_and_caches_ref(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        p.apply_deviation_result(vid, self._dev_result(variant_mid="2"))
        v = tab.design.get_variant(vid)
        assert v.results.fold["deviation"]["variant_model_id"] == "2"
        # the WT reference is cached on the design per combo (so the next variant reuses it)
        assert tab.design.wt_refs["esmfold:monomer"]["model_id"] == "7"

    def test_deviation_panel_hex_is_floor_gated(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        p.apply_deviation_result(vid, self._dev_result(
            variant_mid="2", deviation={"1": 0.1, "2": 1.5, "3": 0.2}))
        hexmap = p._deviation_panel_hex(tab)
        # res 2 (1.5 Å > 0.25 global-min floor) coloured; res 1 & 3 (sub-floor) NEUTRAL
        assert hexmap == {2: "#ffd166"}

    def test_deviation_3d_targets_predicted_model_per_chain(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))
        p.apply_deviation_result(vid, self._dev_result(
            variant_mid="2", deviation={"1": 0.1, "2": 1.5, "3": 0.2}))
        p._mode_key = _RESULT_DEVIATION_MODE
        cmds = p.color_commands_for(tab)
        # colours the PREDICTED variant model #2 (its own numbering), not the crystal backbone.
        # No `show` — visibility is owned by fold_visibility_commands (else the Variant-fold
        # toggle would be re-defeated, the same bug as the pLDDT mode).
        assert cmds == ["color #2/A #ffffff", "color #2/A:2 #ffd166"]

    def test_deviation_color_mode_no_deviation_is_empty(self, _app):
        p, tab, (vid,) = self._variant_panel()
        p.apply_fold_result(vid, self._fold_result(model_id="2"))   # folded but no deviation yet
        p._mode_key = _RESULT_DEVIATION_MODE
        assert p.color_commands_for(tab) == []


class TestBoltzStage:
    """Boltz as the assembly + selectable-monomer engine on the SAME seam: the assembly
    launch spec (multi-chain from cd.members), the monomer spec, and the ipTM badge."""

    def _dimer_variant(self):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")],
                      session=MagicMock())
        p.load_model("1")
        p._add_variant()
        tab = p._cur_tab()
        vid = tab.design.variants[-1].id
        tab.design.edit_variant(vid, 0, "W")        # M1W
        return p, tab, vid

    def test_boltz_assembly_spec_is_multichain(self, _app):
        p, tab, vid = self._dimer_variant()
        spec = p.fold_launch_spec("boltz", assembly=True)
        ti = spec["tool_inputs"]
        assert spec["tool"] == "boltz" and spec["refresh"] == "fold"
        assert "chains" in ti and "sequence" not in ti
        seq = tab.design.get_variant(vid).sequence
        assert ti["chains"] == [{"id": "A", "sequence": seq}, {"id": "B", "sequence": seq}]
        assert ti["local_only"] is True and ti["compare_to"] == "1"

    def test_boltz_monomer_spec_single_sequence(self, _app):
        p, tab, vid = self._dimer_variant()
        spec = p.fold_launch_spec("boltz", assembly=False)
        ti = spec["tool_inputs"]
        assert "sequence" in ti and "chains" not in ti
        assert ti["sequence"] == tab.design.get_variant(vid).sequence

    def test_iptm_badge_rendered(self, _app):
        p, tab, vid = self._dimer_variant()
        result = {"tool_step_results": [{"tool": "boltz", "data": {
            "engine": "boltz", "new_model_id": "3", "reference_model_id": "1",
            "mean_plddt": 96.3, "iptm": 0.959, "plddt": {1: 96.0},
            "source": "local_boltz_env", "seed": 0}}]}
        p.apply_fold_result(vid, result)
        f = tab.design.get_variant(vid).results.fold
        assert f["iptm"] == 0.959 and f["source"] == "local_boltz_env"
        assert "ipTM 0.96" in tab.badges[vid] and "pLDDT 96" in tab.badges[vid]
