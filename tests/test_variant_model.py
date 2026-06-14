"""
tests/test_variant_model.py
---------------------------
The Workbench data model: build from ChainSeq (homo-oligomer collapse), the column
axis (pre-shape b: a gap cell round-trips even though Stage 1 emits none), MPNN
import (pre-shape a: provenance), consensus/conservation tracks, and persistence
round-trip (DesignSession.to_dict ↔ from_dict).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from variant_model import (
    AlignedCell, Mutation, Variant, ChainDesign, DesignSession,
    build_design_session, import_mpnn_designs, column_tracks,
)
from seq_editor.controller import ResidueCell, ChainSeq


def _chainseq(model, chain, seq, start=1):
    cells = [ResidueCell(model, chain, start + i, aa, i + 1) for i, aa in enumerate(seq)]
    return ChainSeq(model, chain, cells)


class TestBuild:
    def test_homo_oligomer_one_chaindesign_all_members(self):
        cs = [_chainseq("1", "A", "MKV"), _chainseq("1", "B", "MKV")]
        ds = build_design_session(cs, "1")
        assert len(ds.chains) == 1
        cd = next(iter(ds.chains.values()))
        assert sorted(cd.members) == [("1", "A"), ("1", "B")]
        assert [c.aa for c in cd.template_cells] == list("MKV")
        assert [c.resnum for c in cd.template_cells] == [1, 2, 3]

    def test_non_1_start_resnums_preserved(self):
        ds = build_design_session([_chainseq("1", "A", "MKV", start=50)], "1")
        cd = next(iter(ds.chains.values()))
        assert [c.resnum for c in cd.template_cells] == [50, 51, 52]
        assert cd.resnum_for_col(2) == 52


class TestColumnAxisGap:
    def test_gap_cell_roundtrips(self):
        # pre-shape (b): a gap (aa=None) survives serialization even though Stage 1
        # never produces one — the indel seam is cell population, not a remap.
        v = Variant(id="V1", parent="T", source="manual",
                    cells=[AlignedCell(0, 1, "M"), AlignedCell(1, None, None),
                           AlignedCell(2, 2, "K")])
        assert v.cells[1].is_gap and v.sequence == "MK"   # gap omitted from sequence
        cd = ChainDesign("k", "1", "A", [("1", "A")],
                         [AlignedCell(0, 1, "M"), AlignedCell(1, 2, "K")], [v])
        ds = DesignSession("1", {"u": cd}, next_id=2)
        ds2 = DesignSession.from_dict(ds.to_dict())
        rv = ds2.chains["u"].variants[0]
        assert rv.cells[1].is_gap and rv.cells[1].resnum is None


class TestMpnnImport:
    def test_designs_become_variants_with_provenance(self):
        cd = ChainDesign("k", "1", "A", [("1", "A")],
                         [AlignedCell(0, 1, "M"), AlignedCell(1, 2, "K"),
                          AlignedCell(2, 3, "V")])
        mpnn = {"sequences": [{"sequence": "MAV"}, {"sequence": "MKL"}]}
        nid = iter(["V1", "V2"]).__next__
        variants = import_mpnn_designs(cd, mpnn, run_id=7, next_id_fn=nid)
        assert [v.id for v in variants] == ["V1", "V2"]
        assert variants[0].provenance == {"mpnn_run": 7, "design_k": 0}
        assert variants[0].source == "proteinmpnn"
        # design 0 = MAV vs MKV → one mutation K2A, source proteinmpnn
        assert [(m.resnum, m.from_aa, m.to_aa) for m in variants[0].mutations] == [(2, "K", "A")]
        assert variants[1].sequence == "MKL"

    def test_length_mismatch_skipped(self):
        cd = ChainDesign("k", "1", "A", [("1", "A")], [AlignedCell(0, 1, "M")])
        out = import_mpnn_designs(cd, {"sequences": [{"sequence": "MKV"}]}, 1, iter(["V1"]).__next__)
        assert out == []   # indel/SEQRES drift deferred to Stage-3 alignment


class TestTracks:
    def test_template_only_consensus_is_T_conservation_1(self):
        cd = next(iter(build_design_session([_chainseq("1", "A", "MKV")], "1").chains.values()))
        cons, csv = column_tracks(cd)
        assert cons == list("MKV") and csv == [1.0, 1.0, 1.0]

    def test_variant_lowers_conservation_at_diff_column(self):
        cd = ChainDesign("k", "1", "A", [("1", "A")],
                         [AlignedCell(0, 1, "M"), AlignedCell(1, 2, "K")],
                         [Variant("V1", "T", "manual",
                                  cells=[AlignedCell(0, 1, "M"), AlignedCell(1, 2, "A")])])
        cons, csv = column_tracks(cd)
        assert cons[0] == "M" and csv[0] == 1.0          # col0 all M
        assert csv[1] == 0.5                              # col1 K vs A → 50%


class TestPersistenceRoundtrip:
    def test_full_roundtrip(self):
        ds = build_design_session([_chainseq("1", "A", "MKV"),
                                   _chainseq("1", "B", "MKV")], "1")
        ds.chains[next(iter(ds.chains))].variants.append(
            Variant("V1", "T", "proteinmpnn", provenance={"mpnn_run": 1, "design_k": 0},
                    cells=[AlignedCell(0, 1, "M"), AlignedCell(1, 2, "A"), AlignedCell(2, 3, "V")],
                    mutations=[Mutation(2, "K", "A", "proteinmpnn")]))
        ds2 = DesignSession.from_dict(ds.to_dict())
        assert ds2.model_id == "1" and ds2.next_id == ds.next_id
        cd = next(iter(ds2.chains.values()))
        assert cd.members == [("1", "A"), ("1", "B")]
        assert cd.variants[0].provenance == {"mpnn_run": 1, "design_k": 0}
        assert cd.variants[0].mutations[0].to_aa == "A"
