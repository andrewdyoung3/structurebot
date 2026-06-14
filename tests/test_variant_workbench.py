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

    def test_scan_set_toggle_via_click(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKVLA")])
        p.load_model("1")
        tab = p._cur_tab()
        p._on_cell(tab, "T", 0)
        p._on_cell(tab, "T", 2)
        assert p._scan_cols == {0, 2}
        p._on_cell(tab, "T", 0)                     # second click toggles it back off
        assert p._scan_cols == {2}
        assert p._scan_set_lbl.text() == "scan set: 1"

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
        p._on_cell(tab, "T", 0); p._on_cell(tab, "T", 2)   # resnums 10, 12
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
        tab = p._cur_tab(); p._on_cell(tab, "T", 0)
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
        p._on_cell(tab, "T", 1)                     # resnum 6
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
        p._on_cell(p._cur_tab(), "T", 0)
        assert p._scan_cols
        p._clear_scan_set()
        assert p._scan_cols == set() and p._scan_set_lbl.text() == "scan set: 0"

    def test_tab_change_resets_scan_set(self, _app):
        p, _ = _panel([_chainseq("1", "A", "MKV"), _chainseq("1", "B", "WYF")])
        p.load_model("1")
        p._on_cell(p._cur_tab(), "T", 0)
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
        p._on_cell(tab, None, 0)
        assert p._scan_cols == set()
