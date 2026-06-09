"""
tests/test_thermompnn.py
------------------------
ThermoMPNN fast-tier voter + the residue-identity fix (author resnum vs sequence
index).  Tests drive the REAL pipeline — they do NOT feed the bridge author
resnums with a hand-crafted offset CSV (the prior false-confidence pattern).  The
GPU inference subprocess and Rosetta WSL call are mocked; everything else (the
WT-anchored alignment, the seqindex→resnum spine, scope filtering, candidate
keys) is the real code, exercised on UNFRIENDLY structures (non-1 start, internal
gap, insertion code, multichain).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from mutation_scanner import (
    MutationScanner, present_voters_score, combined_score, effective_weights,
)
from thermompnn_bridge import (
    ThermoMPNNBridge, candidate_key, ordered_chain_residues,
)
from session_state import SessionState

_AA3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
        "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
        "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}


# ── PDB / CSV synthesis ────────────────────────────────────────────────────────

def _atom(serial: int, aa: str, chain: str, resseq: int, icode: str = " ") -> str:
    return (f"ATOM  {serial:>5}  CA  {_AA3[aa]} {chain}{resseq:>4}{icode}"
            f"   {11.0:8.3f}{11.0:8.3f}{11.0:8.3f}  1.00  0.00           C")


def _write_pdb(residues: List[Tuple[str, int, str, str]]) -> str:
    """residues = [(aa, resseq, icode, chain), ...] → temp PDB path."""
    lines = [_atom(i + 1, aa, ch, rs, ic) for i, (aa, rs, ic, ch) in enumerate(residues)]
    f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    f.write("\n".join(lines) + "\nEND\n"); f.close()
    return f.name


def _faithful_csv(pdb_path: str, chain: str, ddg_of=None) -> str:
    """Build the CSV THE WAY REAL ThermoMPNN would for *chain*: one block of 19
    substitution rows per PRESENT residue, in author order, positions = ThermoMPNN's
    author-offset numbering (gaps consume an index as '-', insertion adds one).
    The WT-anchored alignment only relies on the present rows being in author order."""
    ordered = ordered_chain_residues(pdb_path, chain)        # the SAME spine the bridge uses
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    f.write(",Model,Dataset,ddG_pred,position,wildtype,mutation,pdb,chain\n")
    i = 0
    # The WT-anchored alignment uses the ORDER of present positions (sorted), so a
    # strictly-monotonic position per residue (its author-order rank) is faithful —
    # real ThermoMPNN's author-offset numbers sort to the same order.
    for pos, (rn, ic, aa) in enumerate(ordered):
        for mut in "ACDEFGHIKLMNPQRSTVWY":
            if mut == aa:
                continue
            d = ddg_of(rn, ic, aa, mut) if ddg_of else 0.1
            f.write(f"{i},T,X,{d},{pos},{aa},{mut},X,{chain}\n"); i += 1
    f.close()
    return f.name


def _bridge(csv_path: str) -> ThermoMPNNBridge:
    b = ThermoMPNNBridge()
    b.is_available = lambda: True
    b._run_inference = lambda pdb, chain, log: csv_path
    return b


# unfriendly chain A: starts at 50, gap (53,54 missing), insertion 52A; chain B reuses 50
_UNFRIENDLY = [
    ("M", 50, "", "A"), ("K", 51, "", "A"), ("A", 52, "", "A"),
    ("G", 52, "A", "A"),                       # insertion 52A
    ("V", 55, "", "A"), ("L", 56, "", "A"),    # gap before 55
    ("W", 50, "", "B"),                        # chain B, resnum 50 (collision bait)
]


# ── 1. present_voters_score (unchanged renormalise seam) ───────────────────────

class TestPresentVotersScore:
    def test_graceful_exact_fast(self):
        assert present_voters_score(None, None, 1.2, 0.7) == \
               combined_score(0.0, 1.2, 0.7, *effective_weights(False))
    def test_graceful_exact_deep(self):
        assert present_voters_score(-1.5, None, 1.2, 0.7) == \
               combined_score(-1.5, 1.2, 0.7, 0.5, 0.3, 0.2)
    def test_thermompnn_stabiliser_raises(self):
        assert present_voters_score(None, -2.0, 1.0, 0.5) > present_voters_score(None, None, 1.0, 0.5)
    def test_weights_sum_to_one(self):
        assert present_voters_score(-1.0, -1.0, 1.0, 1.0) == pytest.approx(1.0)
        assert present_voters_score(None, None, 1.0, 1.0) == pytest.approx(1.0)


# ── 2. Bridge — WT-anchored alignment (gaps + insertion + non-1 start) ─────────

class TestWTAnchoredAlignment:

    def test_unavailable_returns_empty(self):
        b = ThermoMPNNBridge(); b.is_available = lambda: False
        ddg, src = b.score_mutations("x.pdb", "A", [{"position": 50, "from_aa": "M", "to_aa": "A"}])
        assert ddg == {} and src == {}

    def test_maps_gap_and_insertion_to_TRUE_resnums(self):
        pdb = _write_pdb(_UNFRIENDLY)
        csv = _faithful_csv(pdb, "A")
        b = _bridge(csv)
        # candidates addressed by AUTHOR resnum (what the scanner now produces)
        cands = [{"position": r, "from_aa": w, "to_aa": "W"} for r, w in
                 [(50, "M"), (51, "K"), (52, "A"), (55, "V"), (56, "L")]]   # →W (never self)
        ddg, src = b.score_mutations(pdb, "A", cands)
        # every named residue mapped to its TRUE author resnum — V55 NOT shifted to 56
        for r, w in [(50, "M"), (51, "K"), (52, "A"), (55, "V"), (56, "L")]:
            assert candidate_key("A", r, w, "W") in ddg, f"{w}{r}W missing"
        assert all(v == "thermompnn" for v in src.values())

    def test_insertion_residue_not_misattributed(self):
        # the insertion (G at 52A) gets its own slot; V55/L56 keep their true resnums
        pdb = _write_pdb(_UNFRIENDLY)
        b = _bridge(_faithful_csv(pdb, "A"))
        ddg, _ = b.score_mutations(pdb, "A", [{"position": 55, "from_aa": "V", "to_aa": "A"}])
        assert candidate_key("A", 55, "V", "A") in ddg     # not 56

    def test_sign_stabiliser_stays_negative(self):
        pdb = _write_pdb(_UNFRIENDLY)
        csv = _faithful_csv(pdb, "A", ddg_of=lambda rn, ic, aa, mut: -1.8 if rn == 50 else 0.1)
        ddg, _ = _bridge(csv).score_mutations(pdb, "A", [{"position": 50, "from_aa": "M", "to_aa": "A"}])
        assert ddg[candidate_key("A", 50, "M", "A")] == -1.8   # negative = stabilising, unflipped

    def test_alignment_hard_error_on_AA_mismatch(self):
        # a CSV whose residue sequence diverges from the structure → HARD ERROR ({}),
        # never a probabilistic pass / mis-attribution
        pdb = _write_pdb([("M", 50, "", "A"), ("K", 51, "", "A"), ("A", 52, "", "A")])
        f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
        f.write(",Model,Dataset,ddG_pred,position,wildtype,mutation,pdb,chain\n")
        # wrong WT at the 2nd residue (Q instead of K)
        for i, (p, w) in enumerate([(0, "M"), (1, "Q"), (2, "A")]):
            f.write(f"{i},T,X,9.9,{p},{w},A,X,A\n")
        f.close()
        ddg, src = _bridge(f.name).score_mutations(
            pdb, "A", [{"position": 51, "from_aa": "K", "to_aa": "A"}])
        assert ddg == {} and src == {}        # hard error, not a silent slip

    def test_alignment_hard_error_on_length_divergence(self):
        pdb = _write_pdb([("M", 50, "", "A"), ("K", 51, "", "A"), ("A", 52, "", "A")])
        f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
        f.write(",Model,Dataset,ddG_pred,position,wildtype,mutation,pdb,chain\n")
        f.write("0,T,X,1.0,0,M,A,X,A\n")        # only 1 residue vs 3 in structure
        f.close()
        ddg, _ = _bridge(f.name).score_mutations(
            pdb, "A", [{"position": 50, "from_aa": "M", "to_aa": "A"}])
        assert ddg == {}

    def test_chain_key_disambiguated(self):
        assert candidate_key("A", 50, "M", "V") != candidate_key("B", 50, "M", "V")


# ── 3. End-to-end through scan() on an UNFRIENDLY structure ────────────────────

def _scanner_for(pdb_path: str, chain: str) -> Tuple[MutationScanner, str]:
    """A scanner whose CamSol/ESM are pre-seeded (keyed by 1-based SEQINDEX into the
    coordinate sequence the scan will build) so we exercise the real mapping."""
    ordered = ordered_chain_residues(pdb_path, chain)
    n = len(ordered)
    s = SessionState(); s.add_structure("1", "TEST", metadata={})
    s.add_tool_result("camsol", "1", {"scores": {str(i): -1.0 for i in range(1, n + 1)},
                                      "aggregation_hot_spots": list(range(1, n + 1))})
    s.add_tool_result("esm", "1", {"conservation": {str(i): 0.1 for i in range(1, n + 1)},
                                   "mean_conservation": 0.1})
    return MutationScanner(session=s, model_id="1", progress_callback=lambda *_: None), \
           "".join(aa for _, _, aa in ordered)


class TestEndToEndUnfriendly:

    def test_scope_is_author_resnum_not_seqindex(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])  # 50..60
        sc, seq = _scanner_for(pdb, "A")
        # user scope "residues 55-57" = AUTHOR resnums; seqindex would be 6-8 (wrong)
        res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                      include_positions=[55, 56, 57], run_rosetta=False, run_thermompnn=False)
        sel = sorted({r["resnum"] for r in res})
        assert sel == [55, 56, 57], f"scope hit {sel}, expected author 55-57"
        assert all(r["position"] == r["resnum"] for r in res)   # record identity = author

    def test_thermompnn_maps_through_scan_on_unfriendly(self):
        pdb = _write_pdb(_UNFRIENDLY)
        sc, seq = _scanner_for(pdb, "A")
        csv = _faithful_csv(pdb, "A", ddg_of=lambda rn, ic, aa, mut: round(-0.5 + 0.01 * rn, 3))
        with patch.object(ThermoMPNNBridge, "is_available", lambda self: True), \
             patch.object(ThermoMPNNBridge, "_run_inference", lambda self, p, c, log: csv):
            res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                          include_positions=[55, 56], run_rosetta=False)
        got = {r["resnum"]: r for r in res}
        assert 55 in got and 56 in got
        # CROSS-SOURCE IDENTITY: the ThermoMPNN value for resnum 55 must be 55's value
        for rn in (55, 56):
            assert got[rn]["thermompnn_source"] == "thermompnn"
            assert got[rn]["thermompnn_ddg"] == round(-0.5 + 0.01 * rn, 3)
            assert got[rn]["key"] == candidate_key("A", rn, got[rn]["from_aa"], got[rn]["to_aa"])

    def test_rosetta_addresses_author_resnum_on_nonstart_chain(self):
        # AUDIT bug: deep ddG must land on the AUTHOR resnum, not the seqindex.
        # Patch the LAYER BELOW _run_rosetta_batch (RosettaBridge.analyze, which the
        # real _run_rosetta_batch — carrying the fix — feeds), so we observe exactly
        # what residue numbers Rosetta is asked to mutate.
        import types
        import rosetta_bridge
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])  # 50..60
        sc, seq = _scanner_for(pdb, "A")
        captured = {}
        def fake_analyze(self, pdb_path, mutations, **kw):
            captured["positions"] = sorted({m["position"] for m in mutations})
            keys = {f"{m['from_aa']}{m['position']}{m['to_aa']}": -1.0 for m in mutations}
            return types.SimpleNamespace(
                success=True,
                data={"ddg_scores": keys, "ddg_source": {k: "pyrosetta" for k in keys},
                      "backend": "pyrosetta_wsl2"})
        with patch.object(rosetta_bridge.RosettaBridge, "analyze", fake_analyze):
            sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                    include_positions=[55, 56], run_rosetta=True, run_thermompnn=False)
        # Rosetta received AUTHOR resnums (55,56) — NOT seqindex (6,7)
        assert captured["positions"] == [55, 56], captured["positions"]

    def test_graceful_disabled_byte_for_byte(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("ACDEFGHIKL")])
        sc, seq = _scanner_for(pdb, "A")
        res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                      include_positions=[52, 53], run_rosetta=False, run_thermompnn=False)
        for r in res:
            assert r["thermompnn_ddg"] is None and r["thermompnn_source"] == "not_computed"
            assert r["combined_score"] == combined_score(
                0.0, r["solubility_delta"], r["esm_tolerance"], *effective_weights(False))

    def test_regression_contiguous_1based_unchanged(self):
        # a clean 1-based chain: resnum == seqindex, behaviour identical to before
        pdb = _write_pdb([(a, 1 + i, "", "A") for i, a in enumerate("ACDEFGHIKL")])
        sc, seq = _scanner_for(pdb, "A")
        res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                      include_positions=[3, 4, 5], run_rosetta=False, run_thermompnn=False)
        assert sorted({r["resnum"] for r in res}) == [3, 4, 5]
        assert all(r["position"] == r["seqindex"] for r in res)   # coincide when 1-based
