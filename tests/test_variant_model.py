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
    build_design_session, import_mpnn_designs, column_tracks, build_color_commands,
    build_color_commands_by_resnum, candidate_ddg, stability_summary,
    filter_new_mpnn_variants, group_scan_suggestions, suggestion_color,
)
from color_modes import ddg_color
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


class TestVariantCreateEdit:
    def _cd(self, seq="MKV", start=1):
        return next(iter(build_design_session(
            [_chainseq("1", "A", seq, start=start)], "1").chains.values()))

    def test_add_variant_is_aligned_copy_of_template(self):
        cd = self._cd()
        v = cd.add_variant("V1")
        assert [c.aa for c in v.cells] == list("MKV")
        assert [c.resnum for c in v.cells] == [1, 2, 3]
        assert v.mutations == [] and cd.variants == [v]
        # fresh cells — editing the variant must not bleed into the template
        v.cells[0].aa = "A"
        assert cd.template_cells[0].aa == "M"

    def test_edit_variant_sets_aa_and_tracks_mutation(self):
        cd = self._cd()
        cd.add_variant("V1")
        cd.edit_variant("V1", 1, "A")          # K2A
        v = cd.get_variant("V1")
        assert v.cells[1].aa == "A"
        assert [(m.resnum, m.from_aa, m.to_aa) for m in v.mutations] == [(2, "K", "A")]

    def test_edit_back_to_template_reverts(self):
        cd = self._cd()
        cd.add_variant("V1")
        cd.edit_variant("V1", 1, "A")
        cd.edit_variant("V1", 1, "K")          # back to T
        v = cd.get_variant("V1")
        assert v.cells[1].aa == "K" and v.mutations == []

    def test_edit_uses_template_resnum_offset(self):
        cd = self._cd(start=50)
        cd.add_variant("V1")
        cd.edit_variant("V1", 2, "L")          # col 2 → resnum 52
        assert cd.get_variant("V1").mutations[0].resnum == 52

    def test_edit_rejects_bad_inputs(self):
        cd = self._cd()
        cd.add_variant("V1")
        with pytest.raises(ValueError):
            cd.edit_variant("V1", 1, "Z")      # non-standard aa
        with pytest.raises(ValueError):
            cd.edit_variant("V1", 99, "A")     # out-of-range column
        with pytest.raises(KeyError):
            cd.edit_variant("Vx", 1, "A")      # unknown variant


class TestColorCommands:
    def _green(self, aa):                       # a trivial mode: K→green, else None
        return "#00ff00" if aa == "K" else None

    def test_runs_grouped_and_reset_first_all_copies(self):
        cells = [AlignedCell(0, 1, "M"), AlignedCell(1, 2, "K"),
                 AlignedCell(2, 3, "K"), AlignedCell(3, 4, "V")]
        cmds = build_color_commands(cells, [("1", "A"), ("1", "B")], self._green)
        assert cmds == [
            "color #1/A #ffffff", "color #1/A:2-3 #00ff00",
            "color #1/B #ffffff", "color #1/B:2-3 #00ff00",
        ]

    def test_noncontiguous_resnums_use_comma_list(self):
        cells = [AlignedCell(0, 2, "K"), AlignedCell(1, 3, "V"), AlignedCell(2, 5, "K")]
        cmds = build_color_commands(cells, [("1", "A")], self._green)
        assert cmds == ["color #1/A #ffffff", "color #1/A:2 #00ff00", "color #1/A:5 #00ff00"]

    def test_gap_cells_skipped(self):
        # the gap cell contributes no command; the two K's (same color) merge into one
        # comma-list run (1 and 3 are non-contiguous once the gap is dropped).
        cells = [AlignedCell(0, 1, "K"), AlignedCell(1, None, None), AlignedCell(2, 3, "K")]
        cmds = build_color_commands(cells, [("1", "A")], self._green)
        assert cmds == ["color #1/A #ffffff", "color #1/A:1,3 #00ff00"]


class TestStage4aStability:
    def test_candidate_ddg_prefers_rosetta_then_thermompnn_then_rasp(self):
        assert candidate_ddg({"ddg": 1.0, "thermompnn_ddg": 2.0}) == (1.0, "rosetta")
        assert candidate_ddg({"ddg": None, "thermompnn_ddg": 2.0}) == (2.0, "thermompnn")
        assert candidate_ddg({"rasp_ddg": -0.5}) == (-0.5, "rasp")
        assert candidate_ddg({}) == (None, "not_computed")

    def test_stability_summary_matches_variant_mutations_only(self):
        muts = [Mutation(10, "I", "R"), Mutation(25, "K", "A")]
        candidates = [
            {"resnum": 10, "from_aa": "I", "to_aa": "R", "ddg": 1.8, "combined_score": -0.2},
            {"resnum": 25, "from_aa": "K", "to_aa": "A", "thermompnn_ddg": -0.4},
            {"resnum": 10, "from_aa": "I", "to_aa": "K", "ddg": 0.1},   # NOT this variant's pick
        ]
        s = stability_summary(candidates, muts)
        assert s["per_resnum"] == {10: 1.8, 25: -0.4}    # only the variant's exact subs
        assert s["sum_ddg"] == 1.4
        assert s["n_scored"] == 2
        assert s["tier"] == "deep"                       # a Rosetta ddg was present
        assert [r["resnum"] for r in s["rows"]] == [10, 25]
        assert s["rows"][1]["ddg_source"] == "thermompnn"

    def test_stability_summary_fast_tier_when_no_rosetta(self):
        s = stability_summary([{"resnum": 5, "to_aa": "D", "thermompnn_ddg": 0.3}],
                              [Mutation(5, "A", "D")])
        assert s["tier"] == "fast" and s["per_resnum"] == {5: 0.3}


class TestStage4aColor:
    def test_ddg_color_diverging_neutral_band(self):
        assert ddg_color(None) is None
        assert ddg_color(0.0) == "#ffffff"
        assert ddg_color(0.8) == "#ffffff"               # within ±1 neutral band
        red, blue = ddg_color(3.0), ddg_color(-3.0)
        r_red = int(red[1:3], 16); b_red = int(red[5:7], 16)
        r_blue = int(blue[1:3], 16); b_blue = int(blue[5:7], 16)
        assert r_red > b_red and b_blue > r_blue         # destabilizing red, stabilizing blue

    def test_build_color_commands_by_resnum_runs_all_copies(self):
        ddg = {10: 3.0, 11: 3.0, 25: -3.0}
        value_for = lambda rn: ddg_color(ddg.get(rn))
        cmds = build_color_commands_by_resnum([10, 11, 25], value_for,
                                              [("1", "A"), ("1", "B")])
        # baseline per copy, a merged run for 10-11 (same hex), a separate 25
        assert cmds[0] == "color #1/A #ffffff"
        assert "color #1/A:10-11 " + ddg_color(3.0) in cmds
        assert "color #1/A:25 " + ddg_color(-3.0) in cmds
        assert "color #1/B #ffffff" in cmds and "color #1/B:25 " + ddg_color(-3.0) in cmds


class TestAcceptSuggestionProvenance:
    def _cd(self):
        cd = next(iter(build_design_session(
            [_chainseq("1", "A", "MKV")], "1").chains.values()))
        cd.add_variant("V1")
        return cd

    def test_source_override_and_note_recorded(self):
        cd = self._cd()
        cd.edit_variant("V1", 1, "A", source="accepted_suggestion",
                        note={"combined_score": 1.23, "recommendation": "good"})
        v = cd.get_variant("V1")
        assert v.mutations[0].source == "accepted_suggestion"     # not the variant's "manual"
        acc = v.provenance["accepted"]
        assert acc == [{"resnum": 2, "to_aa": "A", "combined_score": 1.23,
                        "recommendation": "good"}]

    def test_revert_clears_accepted_note(self):
        cd = self._cd()
        cd.edit_variant("V1", 1, "A", source="accepted_suggestion", note={"combined_score": 1.0})
        cd.edit_variant("V1", 1, "K")                              # back to T → revert
        v = cd.get_variant("V1")
        assert v.mutations == [] and v.provenance.get("accepted", []) == []

    def test_default_source_is_variant_source(self):
        cd = self._cd()
        cd.edit_variant("V1", 1, "A")                             # no override
        assert cd.get_variant("V1").mutations[0].source == "manual"


class TestMpnnDedup:
    def test_fasta_path_in_provenance(self):
        cd = ChainDesign("k", "1", "A", [("1", "A")],
                         [AlignedCell(0, 1, "M"), AlignedCell(1, 2, "K")])
        mpnn = {"sequences": [{"sequence": "MA"}], "fasta_path": "cache/run.fa"}
        v = import_mpnn_designs(cd, mpnn, 0, iter(["V1"]).__next__)[0]
        assert v.provenance["fasta_path"] == "cache/run.fa"

    def test_filter_drops_already_imported(self):
        existing = [Variant("V1", "T", "proteinmpnn",
                            provenance={"fasta_path": "r.fa", "design_k": 0})]
        cands = [Variant("Vx", "T", "proteinmpnn", provenance={"fasta_path": "r.fa", "design_k": 0}),
                 Variant("Vy", "T", "proteinmpnn", provenance={"fasta_path": "r.fa", "design_k": 1})]
        new = filter_new_mpnn_variants(existing, cands)
        assert [v.provenance["design_k"] for v in new] == [1]      # k=0 deduped, k=1 kept

    def test_no_fasta_path_always_kept(self):
        cands = [Variant("Vx", "T", "proteinmpnn", provenance={"design_k": 0})]
        assert len(filter_new_mpnn_variants([], cands)) == 1


class TestScanSuggestions:
    def _cells(self):
        return [AlignedCell(0, 10, "M"), AlignedCell(1, 11, "K"), AlignedCell(2, 12, "V")]

    def _cand(self, chain, resnum, to_aa, score):
        return {"chain": chain, "resnum": resnum, "position": resnum,
                "from_aa": "K", "to_aa": to_aa, "combined_score": score}

    def test_groups_by_col_sorted_and_sparse(self):
        scan = [self._cand("A", 11, "A", 0.5), self._cand("A", 11, "D", 1.8),
                self._cand("A", 12, "L", -0.2)]
        sugg = group_scan_suggestions(scan, {"A"}, self._cells())
        # only scored columns appear (col 0 / resnum 10 absent → sparse)
        assert set(sugg) == {1, 2}
        assert [c["to_aa"] for c in sugg[1]] == ["D", "A"]        # sorted by score desc
        assert 0 not in sugg

    def test_filters_to_member_chains(self):
        scan = [self._cand("A", 11, "A", 1.0), self._cand("B", 11, "W", 2.0)]
        sugg = group_scan_suggestions(scan, {"A"}, self._cells())   # only chain A
        assert [c["to_aa"] for c in sugg[1]] == ["A"]

    def test_candidate_resnum_not_in_template_skipped(self):
        sugg = group_scan_suggestions([self._cand("A", 999, "A", 1.0)], {"A"}, self._cells())
        assert sugg == {}

    def test_suggestion_color_bands(self):
        assert suggestion_color(2.0) == "#2a6fdb"     # strong
        assert suggestion_color(0.8) == "#3ec0c9"     # good
        assert suggestion_color(0.2) == "#e8c33a"     # marginal
        assert suggestion_color(-1.0) == "#e2663b"    # not recommended


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
