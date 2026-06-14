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


@pytest.fixture(scope="module")
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _chainseq(model, chain, seq, start=1):
    return ChainSeq(model, chain,
                    [ResidueCell(model, chain, start + i, aa, i + 1)
                     for i, aa in enumerate(seq)])


def _panel(chainseqs, session=None):
    c = MagicMock()
    c.load_model.return_value = chainseqs
    return VariantWorkbenchPanel(c, session=session), c


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
