"""
tests/test_thermompnn.py
------------------------
ThermoMPNN fast-tier stability voter — bridge (sign, position→resnum mapping,
graceful), present-voters renormalise, and scanner integration (lossless record).
All offline: the GPU inference subprocess is mocked; mapping is tested on a HARD
PDB (non-1 start + internal gap + insertion code + multi-chain), not a clean 1..N.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from mutation_scanner import MutationScanner, present_voters_score, combined_score, effective_weights
from thermompnn_bridge import ThermoMPNNBridge, candidate_key
from session_state import SessionState


# ── PDB / CSV synthesis helpers ────────────────────────────────────────────────

def _atom(serial: int, res3: str, chain: str, resseq: int, icode: str = " ") -> str:
    # column-accurate CA record (cols: 13-16 atom, 18-20 res, 22 chain, 23-26 seq, 27 icode)
    return (f"ATOM  {serial:>5}  CA  {res3} {chain}{resseq:>4}{icode}"
            f"   {11.0:8.3f}{11.0:8.3f}{11.0:8.3f}  1.00  0.00           C")


def _hard_pdb() -> str:
    """Chain A: starts at resnum 50, internal gap (53,54 missing), an insertion
    code (52A), and a SECOND chain B re-using resnum 50 (collision bait)."""
    lines = [
        _atom(1, "MET", "A", 50),          # pos0 → 50 M
        _atom(2, "LYS", "A", 51),          # pos1 → 51 K
        _atom(3, "ALA", "A", 52),          # pos2 → 52 A
        _atom(4, "GLY", "A", 52, "A"),     # 52A insertion (bridge skips it)
        # gap: 53, 54 missing
        _atom(5, "VAL", "A", 55),          # pos5 → 55 V
        _atom(6, "TRP", "B", 50),          # chain B resnum 50 (key-collision bait)
        "END",
    ]
    f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    f.write("\n".join(lines) + "\n"); f.close()
    return f.name


def _csv(rows: List[tuple], chain: str = "A") -> str:
    """rows = [(position0based, wt, mut, ddg_pred), ...] → ThermoMPNN-style CSV path."""
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    f.write(",Model,Dataset,ddG_pred,position,wildtype,mutation,pdb,chain\n")
    for i, (pos, wt, mut, ddg) in enumerate(rows):
        f.write(f"{i},ThermoMPNN,X,{ddg},{pos},{wt},{mut},X,{chain}\n")
    f.close(); return f.name


def _bridge_with_csv(csv_path: str) -> ThermoMPNNBridge:
    b = ThermoMPNNBridge()
    b.is_available = lambda: True              # pretend installed
    b._run_inference = lambda pdb, chain, log: csv_path   # mock the GPU subprocess
    return b


# ── 1. present_voters_score (renormalise seam) ─────────────────────────────────

class TestPresentVotersScore:

    def test_graceful_exact_fast(self):
        # both ddG voters absent → 0.6·camsol + 0.4·esm (pre-ThermoMPNN fast tier)
        old = combined_score(0.0, 1.2, 0.7, *effective_weights(False))
        assert present_voters_score(None, None, 1.2, 0.7) == old

    def test_graceful_exact_deep(self):
        old = combined_score(-1.5, 1.2, 0.7, 0.5, 0.3, 0.2)   # rosetta, no thermompnn
        assert present_voters_score(-1.5, None, 1.2, 0.7) == old

    def test_thermompnn_present_raises_for_stabiliser(self):
        base = present_voters_score(None, None, 1.0, 0.5)
        assert present_voters_score(None, -2.0, 1.0, 0.5) > base   # stabilising helps

    def test_thermompnn_present_lowers_for_destabiliser(self):
        base = present_voters_score(None, None, 1.0, 0.5)
        assert present_voters_score(None, +2.0, 1.0, 0.5) < base

    def test_weights_sum_to_one_both_ways(self):
        # identical inputs across all voters → score == that input (weights sum to 1)
        assert present_voters_score(-1.0, -1.0, 1.0, 1.0) == pytest.approx(1.0)
        assert present_voters_score(None, None, 1.0, 1.0) == pytest.approx(1.0)
        assert present_voters_score(None, -1.0, 1.0, 1.0) == pytest.approx(1.0)


# ── 2. Bridge — sign, mapping, graceful ────────────────────────────────────────

class TestThermoMPNNBridge:

    def test_unavailable_returns_empty_never_fake(self):
        b = ThermoMPNNBridge(); b.is_available = lambda: False
        ddg, src = b.score_mutations("x.pdb", "A",
                                     [{"position": 50, "from_aa": "M", "to_aa": "V"}])
        assert ddg == {} and src == {}

    def test_sign_stabiliser_stays_stabilising(self):
        # ThermoMPNN negative = stabilising == system convention (sign +1, no flip)
        pdb = _hard_pdb()
        csv = _csv([(0, "M", "V", -1.8)])           # strongly stabilising
        b = _bridge_with_csv(csv)
        ddg, src = b.score_mutations(pdb, "A", [{"position": 50, "from_aa": "M", "to_aa": "V"}])
        k = candidate_key("A", 50, "M", "V")
        assert ddg[k] == -1.8 and ddg[k] < 0          # negative = stabilising
        assert src[k] == "thermompnn"

    def test_mapping_hard_case_nonstart_and_gap(self):
        # position = author_resnum − min(50): pos0→50M, pos1→51K, pos2→52A, pos5→55V
        pdb = _hard_pdb()
        csv = _csv([(0, "M", "A", -0.1), (1, "K", "A", 0.2),
                    (2, "A", "G", 0.3), (5, "V", "A", -0.4)])
        b = _bridge_with_csv(csv)
        cands = [{"position": r, "from_aa": w, "to_aa": "A"} for r, w in
                 [(50, "M"), (51, "K"), (52, "A"), (55, "V")]]
        ddg, _ = b.score_mutations(pdb, "A", cands)
        assert ddg[candidate_key("A", 50, "M", "A")] == -0.1
        assert ddg[candidate_key("A", 51, "K", "A")] == 0.2
        assert ddg[candidate_key("A", 55, "V", "A")] == -0.4   # across the gap, correct

    def test_wildtype_mismatch_is_dropped_not_misattributed(self):
        # one row has the WRONG wildtype for its mapped resnum → must be dropped,
        # not attached to the wrong residue (others still map; >80% verify)
        pdb = _hard_pdb()
        csv = _csv([(0, "M", "A", -0.1), (1, "K", "A", 0.2),
                    (2, "A", "G", 0.3), (5, "Q", "A", 9.9)])  # pos5 wt should be V, not Q
        b = _bridge_with_csv(csv)
        ddg, _ = b.score_mutations(pdb, "A",
                                   [{"position": 55, "from_aa": "Q", "to_aa": "A"},
                                    {"position": 55, "from_aa": "V", "to_aa": "A"}])
        assert candidate_key("A", 55, "Q", "A") not in ddg   # bogus row not attributed
        assert candidate_key("A", 55, "V", "A") not in ddg   # real V55A wasn't in CSV

    def test_map_trust_guard_bails_when_widely_wrong(self):
        # every wildtype wrong → <80% verify → whole batch dropped (not_computed)
        pdb = _hard_pdb()
        csv = _csv([(0, "Z", "A", 1.0), (1, "Z", "A", 1.0), (2, "Z", "A", 1.0)])
        b = _bridge_with_csv(csv)
        ddg, src = b.score_mutations(pdb, "A", [{"position": 50, "from_aa": "M", "to_aa": "A"}])
        assert ddg == {} and src == {}

    def test_chain_key_disambiguated_multimer(self):
        # A:M50V and B:M50V must be distinct keys (no cross-chain merge)
        assert candidate_key("A", 50, "M", "V") != candidate_key("B", 50, "M", "V")


# ── 3. Scanner integration (lossless record + graceful) ───────────────────────

def _scanner(seq: str) -> MutationScanner:
    s = SessionState(); s.add_structure("1", "TEST", metadata={"sequences": {"A": seq}})
    n = len(seq)
    s.add_tool_result("camsol", "1", {"scores": {str(i): -1.0 for i in range(1, n + 1)},
                                      "aggregation_hot_spots": list(range(1, n + 1))})
    s.add_tool_result("esm", "1", {"conservation": {str(i): 0.1 for i in range(1, n + 1)},
                                   "mean_conservation": 0.1})
    return MutationScanner(session=s, model_id="1")


def _fake_pdb() -> str:
    f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    f.write("HEADER    TEST\nEND\n"); f.close(); return f.name


class TestScannerIntegration:

    def test_records_carry_thermompnn_ddg_keyed(self):
        sc = _scanner("ACDEFGHIKL")
        # mock the fast-tier ThermoMPNN call → ddG for two candidates
        def fake_tm(pdb, chain, raw, enabled=True):
            d, s = {}, {}
            for c in raw[:2]:
                k = candidate_key("A", c["position"], c["from_aa"], c["to_aa"])
                d[k] = -0.5; s[k] = "thermompnn"
            return d, s
        sc._run_thermompnn = fake_tm
        res = sc.scan(pdb_path=_fake_pdb(), chain_id="A", sequence="ACDEFGHIKL",
                      include_positions=[3, 4, 5], run_rosetta=False)
        have = [r for r in res if r["thermompnn_ddg"] is not None]
        assert have, "some records should carry a ThermoMPNN ddG"
        for r in have:
            assert r["thermompnn_source"] == "thermompnn"
            assert r["thermompnn_ddg"] == -0.5
            assert r["key"] == candidate_key("A", r["position"], r["from_aa"], r["to_aa"])
        # candidates without a ThermoMPNN value are retained as not_computed (lossless)
        absent = [r for r in res if r["thermompnn_ddg"] is None]
        assert all(r["thermompnn_source"] == "not_computed" for r in absent)

    def test_thermompnn_present_changes_combined_score(self):
        sc = _scanner("ACDEFGHIKL")
        sc._run_thermompnn = lambda pdb, chain, raw, enabled=True: (
            {candidate_key("A", c["position"], c["from_aa"], c["to_aa"]): -3.0 for c in raw},
            {candidate_key("A", c["position"], c["from_aa"], c["to_aa"]): "thermompnn" for c in raw},
        )
        res = sc.scan(pdb_path=_fake_pdb(), chain_id="A", sequence="ACDEFGHIKL",
                      include_positions=[3], run_rosetta=False)
        r = res[0]
        expected = present_voters_score(None, -3.0, r["solubility_delta"], r["esm_tolerance"],
                                        w_thermo=config.THERMOMPNN_WEIGHT)
        assert r["combined_score"] == expected

    def test_graceful_disabled_equals_pre_thermompnn(self):
        sc = _scanner("ACDEFGHIKL")
        # run_thermompnn=False → no ThermoMPNN; scores must equal CamSol+ESM only
        res = sc.scan(pdb_path=_fake_pdb(), chain_id="A", sequence="ACDEFGHIKL",
                      include_positions=[3, 4, 5], run_rosetta=False, run_thermompnn=False)
        for r in res:
            assert r["thermompnn_ddg"] is None
            assert r["thermompnn_source"] == "not_computed"
            old = combined_score(0.0, r["solubility_delta"], r["esm_tolerance"],
                                 *effective_weights(False))
            assert r["combined_score"] == old      # byte-identical to pre-ThermoMPNN
