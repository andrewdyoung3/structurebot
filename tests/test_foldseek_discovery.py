"""
tests/test_foldseek_discovery.py
--------------------------------
Stage 2 template auto-discovery WIRING in the Variant Workbench (Qt offscreen, controller + pool
mocked): the menu actions, de-novo / unguided-fold gating, fail-loud availability, the monomer
query extraction, the refs convergence shape, and the hits→guided-spec seam.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent))

pytest.importorskip("PySide6")
from PySide6 import QtWidgets, QtCore

import foldseek_bridge
from variant_workbench import VariantWorkbenchPanel


@pytest.fixture(scope="module")
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _panel():
    c = MagicMock()
    return VariantWorkbenchPanel(c, session=None, pool=MagicMock())


# A small real-ish RNase-A-like sequence is unnecessary — any sequence seeds a de-novo construct.
SEQ = "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV"


class TestMenuAndGating:
    def test_menu_actions_present(self, _app):
        p = _panel()
        assert hasattr(p, "_find_tmpl_acts") and len(p._find_tmpl_acts) == 4

    def test_gating_requires_denovo(self, _app):
        p = _panel()                                   # no design at all
        p._on_find_templates("boltz", 1)
        assert "de-novo" in p._status.text().lower() or "add sequence" in p._status.text().lower()

    def test_gating_requires_unguided_fold(self, _app):
        p = _panel()
        p._add_sequence_construct("x", SEQ)            # de-novo, but never folded
        p._on_find_templates("boltz", 1)
        assert "unguided construct fold" in p._status.text().lower()

    def test_fail_loud_when_unavailable(self, _app, monkeypatch):
        p = _panel()
        p._add_sequence_construct("x", SEQ)
        cd = next(iter(p._design.chains.values()))
        qf = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        qf.write(b"data_query\n"); qf.close()
        cd.template_fold = {"cif_path": qf.name, "mean_plddt": 90.0}
        monkeypatch.setattr(foldseek_bridge.FoldseekBridge, "is_available", lambda self: False)
        try:
            p._on_find_templates("boltz", 1)
            assert "unavailable" in p._status.text().lower()
        finally:
            os.unlink(qf.name)

    def test_available_path_dispatches_search(self, _app, monkeypatch):
        p = _panel()
        p._add_sequence_construct("x", SEQ)
        cd = next(iter(p._design.chains.values()))
        qf = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        qf.write(b"data_query\n"); qf.close()
        cd.template_fold = {"cif_path": qf.name, "mean_plddt": 90.0}
        monkeypatch.setattr(foldseek_bridge.FoldseekBridge, "is_available", lambda self: True)
        monkeypatch.setattr(foldseek_bridge.FoldseekBridge, "db_label", lambda self: "PDB snapshot 2025-01 (local foldseek DB)")
        try:
            p._on_find_templates("boltz", 1)
            assert "searching" in p._status.text().lower()
            p._pool.start.assert_called_once()         # off-thread search dispatched
        finally:
            os.unlink(qf.name)


class TestRefsAndQuery:
    def test_refs_shape_soft_consensus(self):
        refs = VariantWorkbenchPanel._foldseek_refs(["1abc", "2XyZ"])
        assert refs == [
            {"pdb_id": "1ABC", "label": "1ABC", "force": False},
            {"pdb_id": "2XYZ", "label": "2XYZ", "force": False},
        ]

    def test_query_path_reduces_to_first_chain(self, _app):
        p = _panel()
        cif = ("data_x\nloop_\n"
               "_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
               "_atom_site.auth_asym_id\n_atom_site.Cartn_x\n"
               "ATOM 1 CA A 1.0\nATOM 2 CA A 2.0\nATOM 3 CA B 3.0\nATOM 4 CA B 4.0\n")
        src = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        src.write(cif.encode()); src.close()
        try:
            out = p._foldseek_query_path(src.name)
            assert out != src.name                       # a reduced temp file was written
            data = [ln for ln in open(out).read().splitlines() if ln.startswith("ATOM")]
            assert len(data) == 2                         # only chain A kept
            assert all(ln.split()[3] == "A" for ln in data)
        finally:
            os.unlink(src.name)

    def test_query_path_passthrough_for_pdb(self, _app):
        p = _panel()
        assert p._foldseek_query_path("C:/x/fold.pdb") == "C:/x/fold.pdb"


class TestHitsSeam:
    def test_no_hits_states_it(self, _app):
        p = _panel()
        p._add_sequence_construct("x", SEQ)
        cd = next(iter(p._design.chains.values()))
        # BOTH buckets empty = a real "searched, nothing found" answer.
        p._on_foldseek_hits([], [], cd, "boltz", 1, 90.0, "PDB snapshot 2025-01 (local foldseek DB)")
        assert "no structural neighbours" in p._status.text().lower()

    def test_hits_build_guided_spec_and_launch(self, _app, monkeypatch):
        p = _panel()
        p._add_sequence_construct("x", SEQ)
        cd = next(iter(p._design.chains.values()))
        # user picks one neighbour (skip the real Qt dialog)
        monkeypatch.setattr(p, "_foldseek_pick_dialog", lambda hits, low, qp, dl: ["1ABC"])
        emitted = []
        p.launchRequested.connect(lambda spec: emitted.append(spec))
        p._on_foldseek_hits([("1ABC", "A", 0.91)], [], cd, "boltz", 1, 90.0, "local DB")
        assert len(emitted) == 1
        templates = emitted[0]["tool_inputs"].get("templates")
        assert templates and any(t.get("pdb_id") == "1ABC" for t in templates)

    def test_low_bucket_only_still_opens_picker(self, _app, monkeypatch):
        # primary empty but the low bucket has hits → still a real answer; the dialog opens with the
        # low-bucket hits available (NOT the "no neighbours" message).
        p = _panel()
        p._add_sequence_construct("x", SEQ)
        cd = next(iter(p._design.chains.values()))
        seen = {}
        def fake_dialog(hits, low, qp, dl):
            seen["hits"], seen["low"] = hits, low
            return ["9LOW"]
        monkeypatch.setattr(p, "_foldseek_pick_dialog", fake_dialog)
        emitted = []
        p.launchRequested.connect(lambda spec: emitted.append(spec))
        p._on_foldseek_hits([], [("9LOW", "A", 0.22)], cd, "boltz", 1, 90.0, "local DB")
        assert seen["hits"] == [] and seen["low"] == [("9LOW", "A", 0.22)]
        assert len(emitted) == 1
        assert any(t.get("pdb_id") == "9LOW" for t in emitted[0]["tool_inputs"]["templates"])

    def test_hits_cancel_no_launch(self, _app, monkeypatch):
        p = _panel()
        p._add_sequence_construct("x", SEQ)
        cd = next(iter(p._design.chains.values()))
        monkeypatch.setattr(p, "_foldseek_pick_dialog", lambda hits, low, qp, dl: None)
        emitted = []
        p.launchRequested.connect(lambda spec: emitted.append(spec))
        p._on_foldseek_hits([("1ABC", "A", 0.91)], [], cd, "boltz", 1, 90.0, "local DB")
        assert not emitted
        assert "cancelled" in p._status.text().lower()


class TestPickDialog:
    """The real Qt dialog (offscreen) — caveat always shown, expander only when the low bucket is
    non-empty, and low-bucket picks harvested. Auto-accept by stubbing exec() + checking items."""

    def _open(self, p, monkeypatch, hits, low, check_pids):
        captured = {}
        def fake_exec(self_dlg):
            # find the checkable list widgets, tick the requested pids, then accept
            lists = self_dlg.findChildren(QtWidgets.QListWidget)
            captured["lists"] = lists
            captured["labels"] = [w.text() for w in self_dlg.findChildren(QtWidgets.QLabel)]
            captured["toggles"] = [w.text() for w in self_dlg.findChildren(QtWidgets.QToolButton)]
            for lw in lists:
                for i in range(lw.count()):
                    if lw.item(i).data(QtCore.Qt.UserRole) in check_pids:
                        lw.item(i).setCheckState(QtCore.Qt.Checked)
            return QtWidgets.QDialog.Accepted
        monkeypatch.setattr(QtWidgets.QDialog, "exec", fake_exec, raising=True)
        picked = p._foldseek_pick_dialog(hits, low, 90.0, "PDB snapshot 2025-01 (local foldseek DB)")
        return picked, captured

    def test_caveat_always_shown_and_no_expander_when_low_empty(self, _app, monkeypatch):
        p = _panel()
        picked, cap = self._open(p, monkeypatch, [("1ABC", "A", 0.91)], [], check_pids=set())
        joined = " ".join(cap["labels"]).lower()
        assert "single-chain fold homologs" in joined and "fold family" in joined  # assembly caveat
        assert cap["toggles"] == []                                                # NO expander
        assert "lower-similarity neighbours" not in joined                         # no low-bucket note

    def test_expander_present_when_low_nonempty(self, _app, monkeypatch):
        p = _panel()
        picked, cap = self._open(p, monkeypatch, [("1ABC", "A", 0.91)],
                                 [("9LOW", "A", 0.22)], check_pids=set())
        assert any("lower-confidence hits" in t.lower() for t in cap["toggles"])   # expander toggle
        joined = " ".join(cap["labels"]).lower()
        assert "lower-similarity neighbours" in joined and "not a recommendation" in joined

    def test_low_bucket_pick_is_harvested(self, _app, monkeypatch):
        p = _panel()
        picked, cap = self._open(p, monkeypatch, [("1ABC", "A", 0.91)],
                                 [("9LOW", "A", 0.22)], check_pids={"9LOW"})
        assert picked == ["9LOW"]                                                  # low-bucket pick flows out
