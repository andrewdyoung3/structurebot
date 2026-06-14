"""
tests/test_variant_workbench.py
-------------------------------
Workbench panel logic (Qt offscreen, controller mocked): one tab per unique chain,
the column→3D select specs (ALL copies — the coupling the live-verify confirms on
real ChimeraX), gap columns select nothing, and load persists the DesignSession.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("PySide6")
from PySide6 import QtWidgets

from seq_editor.controller import ResidueCell, ChainSeq
from variant_workbench import VariantWorkbenchPanel
from variant_model import ChainDesign, AlignedCell
from color_modes import get_mode


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
