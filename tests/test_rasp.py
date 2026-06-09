"""
tests/test_rasp.py
------------------
RaSP fast-tier PHYSICS-PROXY voter + the per-candidate physics handoff.

Drives the REAL pipeline: the only thing mocked is the WSL worker subprocess
(`RaSPBridge._run_worker` → a crafted CSV) and availability — the SHARED spine +
WT-anchored alignment, the candidate keying, the handoff, and the scanner wiring
are all real code, exercised on UNFRIENDLY structures (non-1 start, gap, insertion,
multichain).  RaSP must reuse the shared mapping — never its own.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from mutation_scanner import (
    MutationScanner, present_voters_score, combined_score, effective_weights,
)
from rasp_bridge import RaSPBridge
from residue_mapping import candidate_key, ordered_chain_residues
from thermompnn_bridge import ThermoMPNNBridge
from session_state import SessionState

_AA3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
        "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
        "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}


def _atom(serial, aa, chain, resseq, icode=" "):
    return (f"ATOM  {serial:>5}  CA  {_AA3[aa]} {chain}{resseq:>4}{icode}"
            f"   {11.0:8.3f}{11.0:8.3f}{11.0:8.3f}  1.00  0.00           C")


def _write_pdb(residues: List[Tuple[str, int, str, str]]) -> str:
    import uuid
    # a unique REMARK so identical structures across tests get distinct content
    # hashes (the RaSP cache keys on pdb content — otherwise tests collide).
    lines = [f"REMARK 999 {uuid.uuid4().hex}"]
    lines += [_atom(i + 1, aa, ch, rs, ic) for i, (aa, rs, ic, ch) in enumerate(residues)]
    f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    f.write("\n".join(lines) + "\nEND\n"); f.close()
    return f.name


def _worker_csv(pdb_path, chain, ddg_of=None, drop_insertion=True) -> str:
    """A realistic rasp_worker CSV: residues RENUMBERED 1..N (as pdbfixer does) and
    insertion-coded residues DROPPED (pdbfixer strips them) — so the bridge's
    WT-anchored alignment must re-anchor to the ORIGINAL author resnums."""
    ordered = ordered_chain_residues(pdb_path, chain)
    keep = [(rn, ic, aa) for (rn, ic, aa) in ordered if not (drop_insertion and ic)]
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
    f.write("chain,resnum,wt,mt,ddg\n")
    for new_rn, (rn, ic, aa) in enumerate(keep, 1):           # renumbered 1..N
        for mt in "ACDEFGHIKLMNPQRSTVWY":
            if mt == aa:
                continue
            d = ddg_of(rn, aa, mt) if ddg_of else 0.3
            f.write(f"{chain},{new_rn},{aa},{mt},{d}\n")
    f.close()
    return f.name


def _bridge(csv_path) -> RaSPBridge:
    b = RaSPBridge()
    b.is_available = lambda: True
    b._run_worker = lambda pdb, chain, log: csv_path
    return b


# unfriendly chain A: starts at 50, gap (53,54 missing), insertion 52A; chain B reuses 50
_UNFRIENDLY = [
    ("M", 50, "", "A"), ("K", 51, "", "A"), ("A", 52, "", "A"),
    ("V", 55, "", "A"), ("L", 56, "", "A"),
    ("W", 50, "", "B"),
]
_WITH_INSERTION = [
    ("M", 50, "", "A"), ("K", 51, "", "A"), ("A", 52, "", "A"),
    ("G", 52, "A", "A"),                       # insertion pdbfixer would strip
    ("V", 55, "", "A"),
]


# ── 1. Bridge — shared spine + WT-anchored alignment, sign, graceful ───────────

class TestRaSPBridge:

    def test_unavailable_returns_empty_never_fake(self):
        b = RaSPBridge(); b.is_available = lambda: False
        ddg, src = b.score_mutations("x.pdb", "A", [{"position": 50, "from_aa": "M", "to_aa": "A"}])
        assert ddg == {} and src == {}

    def test_maps_renumbered_worker_to_TRUE_author_resnums(self):
        pdb = _write_pdb(_UNFRIENDLY)
        b = _bridge(_worker_csv(pdb, "A"))
        cands = [{"position": r, "from_aa": w, "to_aa": "W"} for r, w in
                 [(50, "M"), (51, "K"), (52, "A"), (55, "V"), (56, "L")]]
        ddg, src = b.score_mutations(pdb, "A", cands)
        for r, w in [(50, "M"), (51, "K"), (52, "A"), (55, "V"), (56, "L")]:
            assert candidate_key("A", r, w, "W") in ddg, f"{w}{r}W missing"
        assert all(v == "rasp" for v in src.values())

    def test_sign_stabiliser_stays_negative(self):
        pdb = _write_pdb(_UNFRIENDLY)
        csv = _worker_csv(pdb, "A", ddg_of=lambda rn, aa, mt: -1.7 if rn == 50 else 0.3)
        ddg, _ = _bridge(csv).score_mutations(pdb, "A", [{"position": 50, "from_aa": "M", "to_aa": "W"}])
        assert ddg[candidate_key("A", 50, "M", "W")] == -1.7    # positive=destabilising, unflipped

    def test_insertion_hard_errors_to_not_computed(self):
        # pdbfixer strips the insertion → worker has N-1 residues vs structure N →
        # length divergence → HARD ERROR → {} (safe-but-lossy, never mis-attributed)
        pdb = _write_pdb(_WITH_INSERTION)
        b = _bridge(_worker_csv(pdb, "A", drop_insertion=True))
        ddg, src = b.score_mutations(pdb, "A", [{"position": 55, "from_aa": "V", "to_aa": "W"}])
        assert ddg == {} and src == {}

    def test_worker_failure_returns_empty(self):
        pdb = _write_pdb(_UNFRIENDLY)
        b = RaSPBridge(); b.is_available = lambda: True
        b._run_worker = lambda pdb, chain, log: None          # worker crashed
        ddg, src = b.score_mutations(pdb, "A", [{"position": 50, "from_aa": "M", "to_aa": "W"}])
        assert ddg == {} and src == {}


# ── 2. Physics handoff (present_voters_score) ──────────────────────────────────

class TestPhysicsHandoff:

    def test_rosetta_supersedes_rasp_no_double_count(self):
        # both present → score == Rosetta-only (RaSP contributes 0)
        with_both = present_voters_score(-1.5, None, 1.0, 0.5, rasp_ddg=+3.0)
        rosetta_only = present_voters_score(-1.5, None, 1.0, 0.5, rasp_ddg=None)
        assert with_both == rosetta_only

    def test_only_rasp_fills_physics(self):
        # no Rosetta → RaSP fills the physics slot (score moves vs no-physics)
        no_physics = present_voters_score(None, None, 1.0, 0.5, rasp_ddg=None)
        rasp_phys  = present_voters_score(None, None, 1.0, 0.5, rasp_ddg=-2.0)
        assert rasp_phys != no_physics
        # and equals putting that value in the physics slot directly
        assert rasp_phys == present_voters_score(None, None, 1.0, 0.5, rasp_ddg=-2.0)

    def test_rasp_stabiliser_raises_score(self):
        base = present_voters_score(None, None, 1.0, 0.5, rasp_ddg=None)
        assert present_voters_score(None, None, 1.0, 0.5, rasp_ddg=-2.0) > base

    def test_graceful_no_physics_byte_for_byte(self):
        # no physics, no ThermoMPNN → exactly the pre-voter CamSol:ESM fallback
        assert present_voters_score(None, None, 1.2, 0.7, rasp_ddg=None) == \
               combined_score(0.0, 1.2, 0.7, *effective_weights(False))


# ── 3. Scanner integration: handoff in scan(), cross-source identity ───────────

def _scanner_for(pdb_path, chain):
    ordered = ordered_chain_residues(pdb_path, chain)
    n = len(ordered)
    s = SessionState(); s.add_structure("1", "TEST", metadata={})
    s.add_tool_result("camsol", "1", {"scores": {str(i): -1.0 for i in range(1, n + 1)},
                                      "aggregation_hot_spots": list(range(1, n + 1))})
    s.add_tool_result("esm", "1", {"conservation": {str(i): 0.1 for i in range(1, n + 1)},
                                   "mean_conservation": 0.1})
    return (MutationScanner(session=s, model_id="1", progress_callback=lambda *_: None),
            "".join(aa for _, _, aa in ordered))


def _patch_rasp(pdb, chain, ddg_of=None):
    csv = _worker_csv(pdb, chain, ddg_of=ddg_of)
    return (patch.object(RaSPBridge, "is_available", lambda self: True),
            patch.object(RaSPBridge, "_run_worker", lambda self, p, c, log: csv))


class TestScannerRaSP:

    def test_fast_tier_rasp_populates_and_keys_author_resnum(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])  # 50..60
        sc, seq = _scanner_for(pdb, "A")
        av, wk = _patch_rasp(pdb, "A", ddg_of=lambda rn, aa, mt: round(-0.2 + 0.01 * rn, 3))
        with av, wk:
            res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                          include_positions=[55, 56], run_rosetta=False, run_thermompnn=False)
        got = {r["resnum"]: r for r in res}
        for rn in (55, 56):
            assert got[rn]["rasp_source"] == "rasp"
            assert got[rn]["rasp_ddg"] == round(-0.2 + 0.01 * rn, 3)
            assert got[rn]["physics_source"] == "rasp"        # no Rosetta in fast tier
            assert got[rn]["key"] == candidate_key("A", rn, got[rn]["from_aa"], got[rn]["to_aa"])

    def test_handoff_rosetta_supersedes_rasp_in_scan(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])
        sc, seq = _scanner_for(pdb, "A")
        def fake_batch(self, pdb_path, candidates, chain_id, scan_deadline=None, ddg_basis="symmetric"):
            sc_ = {f"{c['from_aa']}{c.get('resnum',c['position'])}{c['to_aa']}": -1.5 for c in candidates}
            return (sc_, {k: "pyrosetta" for k in sc_}, {}, {})
        av, wk = _patch_rasp(pdb, "A", ddg_of=lambda rn, aa, mt: 4.0)   # RaSP very different
        with av, wk, patch.object(MutationScanner, "_run_rosetta_batch", fake_batch):
            res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                          include_positions=[55], run_rosetta=True, run_thermompnn=False)
        r = res[0]
        assert r["physics_source"] == "rosetta"               # real supersedes proxy
        assert r["rasp_ddg"] == 4.0                            # RaSP value RETAINED
        assert r["rasp_minus_rosetta"] == round(4.0 - (-1.5), 4)   # proxy-QC delta stored
        # combined score == handoff (Rosetta in physics, RaSP NOT double-counted)
        expected = present_voters_score(-1.5, None, r["solubility_delta"], r["esm_tolerance"],
                                        rasp_ddg=None)
        assert r["combined_score"] == expected

    def test_cross_source_identity_nonstart(self):
        # RaSP, ThermoMPNN, Rosetta for ONE candidate all refer to the SAME residue
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])
        sc, seq = _scanner_for(pdb, "A")
        # ThermoMPNN faithful CSV (author-offset positions in order)
        ordered = ordered_chain_residues(pdb, "A")
        tf = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
        tf.write(",Model,Dataset,ddG_pred,position,wildtype,mutation,pdb,chain\n")
        i = 0
        for pos, (rn, ic, aa) in enumerate(ordered):
            for mt in "ACDEFGHIKLMNPQRSTVWY":
                if mt != aa:
                    tf.write(f"{i},T,X,{0.5},{pos},{aa},{mt},X,A\n"); i += 1
        tf.close()
        def fake_batch(self, pdb_path, candidates, chain_id, scan_deadline=None, ddg_basis="symmetric"):
            sc_ = {f"{c['from_aa']}{c.get('resnum',c['position'])}{c['to_aa']}": -1.0 for c in candidates}
            return (sc_, {k: "pyrosetta" for k in sc_}, {}, {})
        av, wk = _patch_rasp(pdb, "A", ddg_of=lambda rn, aa, mt: -0.9)
        with av, wk, \
             patch.object(ThermoMPNNBridge, "is_available", lambda self: True), \
             patch.object(ThermoMPNNBridge, "_run_inference", lambda self, p, c, log: tf.name), \
             patch.object(MutationScanner, "_run_rosetta_batch", fake_batch):
            res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                          include_positions=[55], run_rosetta=True, run_thermompnn=True)
        r = res[0]
        assert r["resnum"] == 55 and r["from_aa"] == "F"   # author 55 in MKAEDFGHILV = F
        # all three sources present for THIS physical residue (author 55)
        assert r["rasp_ddg"] == -0.9
        assert r["thermompnn_ddg"] == 0.5
        assert r["ddg"] == -1.0
        assert r["key"] == candidate_key("A", 55, "F", r["to_aa"])

    def test_graceful_disabled_equals_pre_rasp(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("ACDEFGHIKL")])
        sc, seq = _scanner_for(pdb, "A")
        res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq,
                      include_positions=[52, 53], run_rosetta=False,
                      run_thermompnn=False, run_rasp=False)
        for r in res:
            assert r["rasp_ddg"] is None and r["rasp_source"] == "not_computed"
            assert r["physics_source"] == "not_computed"
            assert r["combined_score"] == combined_score(
                0.0, r["solubility_delta"], r["esm_tolerance"], *effective_weights(False))
