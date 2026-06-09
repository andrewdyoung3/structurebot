"""
tests/test_dynamut2.py
----------------------
DynaMut2 as the DYNAMICS-axis shortlist voter — wiring, sign normalisation, the
INDEPENDENT axis (counted on its own, NEVER zeroed/handed off — the inverse of the
RaSP physics handoff), cross-source residue identity, and graceful degradation.

The remote API is never hit: the bridge's RosettaBridge.analyze is mocked, or
MutationScanner._run_dynamut2 is mocked at the scanner level.  (conftest gates
DYNAMUT2_ENABLE off for the whole suite; these tests re-enable it explicitly.)
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from mutation_scanner import (
    MutationScanner, present_voters_score, combined_score, effective_weights,
)
from rosetta_bridge import normalize_dynamut2_ddg
import rosetta_bridge
from residue_mapping import candidate_key
from session_state import SessionState

_AA3 = {"A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
        "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
        "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL"}


def _atom(serial, aa, chain, resseq, icode=" "):
    return (f"ATOM  {serial:>5}  CA  {_AA3[aa]} {chain}{resseq:>4}{icode}"
            f"   {11.0:8.3f}{11.0:8.3f}{11.0:8.3f}  1.00  0.00           C")


def _write_pdb(residues: List[Tuple[str, int, str, str]]) -> str:
    import uuid
    lines = [f"REMARK 999 {uuid.uuid4().hex}"]
    lines += [_atom(i + 1, aa, ch, rs, ic) for i, (aa, rs, ic, ch) in enumerate(residues)]
    f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    f.write("\n".join(lines) + "\nEND\n"); f.close()
    return f.name


@pytest.fixture
def enable_dynamut2(monkeypatch):
    monkeypatch.setattr(config, "DYNAMUT2_ENABLE", "auto", raising=False)


# ── 1. SIGN — physical anchor (locked by physics, not a recorded raw number) ───

class TestDynaMut2Sign:
    """Sign locked by PHYSICS (known mutations), not a recorded raw number.

    Live single-endpoint sign battery (2026-06-10, flipped DynaMut2 vs experiment):
      • POPULATION CORRELATION on the T4L panel: Pearson +0.75 / Spearman +0.81
        (n=8) — decisively positive, at/above DynaMut2's published range (a wrong
        sign would be strongly NEGATIVE).
      • DESTABILISER anchor: L99A (exp +5.0) → DynaMut2 raw −3.32 → system +3.32.
      • STABILISER-TAIL anchor: A149V → raw +0.38 → system −0.38 (stabilising).
      • ANTI-SYMMETRY: fwd destabilising (+) / rev stabilising-or-neutral (−/≈0),
        directionally consistent; magnitude compression = the documented benign
        destabilising bias (NOT a sign inversion).
    => −1 (config.DYNAMUT2_DDG_SIGN) justified; sign locked by the anchors below."""

    def test_destabiliser_anchor_L99A_comes_out_positive(self):
        # L99A (canonical destabiliser, exp +5.0): DynaMut2 raw −3.32 → system POSITIVE
        assert normalize_dynamut2_ddg(-3.32) > 0

    def test_stabiliser_tail_anchor_A149V_comes_out_negative(self):
        # A149V (live stabiliser): DynaMut2 raw +0.38 → system NEGATIVE (stabilising)
        assert normalize_dynamut2_ddg(+0.38) < 0
        # and a stronger stabiliser stays stabilising
        assert normalize_dynamut2_ddg(+2.0) < 0

    def test_convention_is_pure_sign_flip_antisymmetric(self):
        # The convention is a PURE sign flip (no offset/scale) — regression guard so
        # normalisation can never drift to an asymmetric transform.
        for x in (-3.32, -0.38, 0.0, 0.5, 2.0, 5.1):
            assert normalize_dynamut2_ddg(x) == pytest.approx(-normalize_dynamut2_ddg(-x))

    def test_parser_normalises_to_system(self):
        from rosetta_bridge import _parse_dynamut2_result
        # raw DONE prediction −3.32 (DynaMut2 stabilising) → system +3.32 (destabilising)
        assert _parse_dynamut2_result({"status": "DONE", "prediction": -3.32}, "L99A") == 3.32


# ── 2. INDEPENDENT axis (the inverse of the RaSP handoff) ──────────────────────

class TestIndependentAxis:

    def test_counted_as_own_slot_when_present(self):
        base = present_voters_score(None, None, 1.0, 0.5, dynamut2_ddg=None)
        with_dyna = present_voters_score(None, None, 1.0, 0.5, dynamut2_ddg=-2.0)
        assert with_dyna != base                      # dynamics axis changes the score

    def test_NOT_zeroed_when_rosetta_present(self):
        # THE guard: unlike RaSP (which hands off → zeroed when Rosetta present),
        # DynaMut2 is ALWAYS counted.  Both must differ from rosetta-only.
        rosetta_only = present_voters_score(-1.5, None, 1.0, 0.5, dynamut2_ddg=None)
        with_dyna    = present_voters_score(-1.5, None, 1.0, 0.5, dynamut2_ddg=-2.0)
        assert with_dyna != rosetta_only              # DynaMut2 still votes alongside Rosetta
        # contrast: RaSP IS zeroed when Rosetta present
        rasp_both = present_voters_score(-1.5, None, 1.0, 0.5, rasp_ddg=-2.0)
        assert rasp_both == rosetta_only

    def test_renormalises_when_absent_byte_for_byte(self):
        assert present_voters_score(None, None, 1.2, 0.7, dynamut2_ddg=None) == \
               combined_score(0.0, 1.2, 0.7, *effective_weights(False))

    def test_all_four_axes_sum_to_one(self):
        # identical value across physics+ML+dynamics+both properties → score == value
        v = present_voters_score(-1.0, -1.0, 1.0, 1.0, dynamut2_ddg=-1.0)
        assert v == pytest.approx(1.0)


# ── 3. Bridge orchestration (cap + author-resnum keying), API mocked ───────────

def _mock_analyze(ddg_by_rosettakey):
    def analyze(self, pdb_path, mutations, **kw):
        scores = {f"{m['from_aa']}{m['position']}{m['to_aa']}": ddg_by_rosettakey.get(
            f"{m['from_aa']}{m['position']}{m['to_aa']}", 0.0) for m in mutations}
        return types.SimpleNamespace(success=True,
                                     data={"ddg_scores": scores, "backend": "dynamut2"})
    return analyze


class TestBridgeOrchestration:

    def _scanner(self):
        s = SessionState(); s.add_structure("1", "T", metadata={})
        return MutationScanner(session=s, model_id="1", progress_callback=lambda *_: None)

    def test_keys_by_author_resnum(self, enable_dynamut2):
        sc = self._scanner()
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDF")])  # 50..55
        cands = [{"chain": "A", "resnum": 52, "position": 3, "from_aa": "A", "to_aa": "W",
                  "_fast_score": 1.0}]
        with patch.object(rosetta_bridge.RosettaBridge, "analyze", _mock_analyze({"A52W": -1.5})):
            ddg, src = sc._run_dynamut2(pdb, "A", cands, True)
        k = candidate_key("A", 52, "A", "W")
        assert ddg.get(k) == -1.5 and src.get(k) == "dynamut2"   # author-resnum keyed

    def test_caps_to_top_n_by_fast_score(self, enable_dynamut2, monkeypatch):
        monkeypatch.setattr(config, "DYNAMUT2_MAX_CANDIDATES", 2, raising=False)
        sc = self._scanner()
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDF")])
        cands = [{"chain": "A", "resnum": 50 + i, "position": i + 1, "from_aa": a,
                  "to_aa": "W", "_fast_score": float(i)} for i, a in enumerate("MKAEDF")]
        captured = {}
        def analyze(self, pdb_path, mutations, **kw):
            captured["n"] = len(mutations)
            return types.SimpleNamespace(success=True, data={"ddg_scores": {}, "backend": "dynamut2"})
        with patch.object(rosetta_bridge.RosettaBridge, "analyze", analyze):
            sc._run_dynamut2(pdb, "A", cands, True)
        assert captured["n"] == 2                       # only the top-2 by fast score covered

    def test_empirical_fallback_excluded_not_dynamics(self, enable_dynamut2):
        # a candidate that fell back to empirical BLOSUM/B-factor is NOT a dynamics
        # signal → excluded → dynamics not_computed (the deep run's other axes stand).
        sc = self._scanner()
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKA")])
        cands = [{"chain": "A", "resnum": 50, "position": 1, "from_aa": "M", "to_aa": "W",
                  "_fast_score": 1.0},
                 {"chain": "A", "resnum": 51, "position": 2, "from_aa": "K", "to_aa": "W",
                  "_fast_score": 1.0}]
        def analyze(self, pdb_path, mutations, **kw):
            scores = {"M50W": -1.0, "K51W": 0.5}     # K51W fell back to empirical
            return types.SimpleNamespace(success=True, data={
                "ddg_scores": scores, "backend": "dynamut2+empirical",
                "empirical_fallbacks": ["K51W"]})
        with patch.object(rosetta_bridge.RosettaBridge, "analyze", analyze):
            ddg, src = sc._run_dynamut2(pdb, "A", cands, True)
        assert candidate_key("A", 50, "M", "W") in ddg          # real dynamut2 kept
        assert candidate_key("A", 51, "K", "W") not in ddg      # empirical excluded

    def test_api_failure_returns_empty(self, enable_dynamut2):
        sc = self._scanner()
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKA")])
        def analyze(self, pdb_path, mutations, **kw):
            return types.SimpleNamespace(success=False, error="DynaMut2 down", data={})
        cands = [{"chain": "A", "resnum": 50, "position": 1, "from_aa": "M", "to_aa": "W",
                  "_fast_score": 1.0}]
        with patch.object(rosetta_bridge.RosettaBridge, "analyze", analyze):
            ddg, src = sc._run_dynamut2(pdb, "A", cands, True)
        assert ddg == {} and src == {}

    def test_disabled_returns_empty(self):  # conftest leaves it disabled
        sc = self._scanner()
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKA")])
        cands = [{"chain": "A", "resnum": 50, "position": 1, "from_aa": "M", "to_aa": "W",
                  "_fast_score": 1.0}]
        ddg, src = sc._run_dynamut2(pdb, "A", cands, True)
        assert ddg == {} and src == {}


# ── 4. Scanner integration (assembly, cross-source identity, graceful) ─────────

def _scanner_for(pdb_path, chain):
    from residue_mapping import ordered_chain_residues
    ordered = ordered_chain_residues(pdb_path, chain); n = len(ordered)
    s = SessionState(); s.add_structure("1", "T", metadata={})
    s.add_tool_result("camsol", "1", {"scores": {str(i): -1.0 for i in range(1, n + 1)},
                                      "aggregation_hot_spots": list(range(1, n + 1))})
    s.add_tool_result("esm", "1", {"conservation": {str(i): 0.1 for i in range(1, n + 1)},
                                   "mean_conservation": 0.1})
    return (MutationScanner(session=s, model_id="1", progress_callback=lambda *_: None),
            "".join(aa for _, _, aa in ordered))


class TestScannerIntegration:

    def test_record_carries_dynamut2_and_independent_score(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])  # 50..60
        sc, seq = _scanner_for(pdb, "A")
        def fake_rosetta(self, pdb_path, candidates, chain_id, scan_deadline=None, ddg_basis="symmetric"):
            sc_ = {f"{c['from_aa']}{c.get('resnum',c['position'])}{c['to_aa']}": -1.0 for c in candidates}
            return (sc_, {k: "pyrosetta" for k in sc_}, {}, {})
        def fake_dyna(self, pdb_path, chain_id, deep, enabled=True):
            d = {candidate_key("A", c["resnum"], c["from_aa"], c["to_aa"]): -2.0 for c in deep}
            return d, {k: "dynamut2" for k in d}
        with patch.object(MutationScanner, "_run_rosetta_batch", fake_rosetta), \
             patch.object(MutationScanner, "_run_dynamut2", fake_dyna):
            res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq, include_positions=[55],
                          run_rosetta=True, run_thermompnn=False, run_rasp=False)
        r = res[0]
        assert r["dynamut2_ddg"] == -2.0 and r["dynamut2_source"] == "dynamut2"
        # independent: score includes BOTH Rosetta (physics) and DynaMut2 (dynamics)
        expected = present_voters_score(-1.0, None, r["solubility_delta"], r["esm_tolerance"],
                                        dynamut2_ddg=-2.0, w_dyna=config.DYNAMUT2_WEIGHT)
        assert r["combined_score"] == expected

    def test_cross_source_identity_all_axes_nonstart(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])
        sc, seq = _scanner_for(pdb, "A")
        def fake_rosetta(self, pdb_path, candidates, chain_id, scan_deadline=None, ddg_basis="symmetric"):
            sc_ = {f"{c['from_aa']}{c.get('resnum',c['position'])}{c['to_aa']}": -1.0 for c in candidates}
            return (sc_, {k: "pyrosetta" for k in sc_}, {}, {})
        def fake_thermo(self, pdb_path, chain_id, raw, enabled=True):
            d = {candidate_key("A", c["resnum"], c["from_aa"], c["to_aa"]): 0.5 for c in raw}
            return d, {k: "thermompnn" for k in d}
        def fake_rasp(self, pdb_path, chain_id, raw, enabled=True):
            d = {candidate_key("A", c["resnum"], c["from_aa"], c["to_aa"]): -0.9 for c in raw}
            return d, {k: "rasp" for k in d}
        def fake_dyna(self, pdb_path, chain_id, deep, enabled=True):
            d = {candidate_key("A", c["resnum"], c["from_aa"], c["to_aa"]): -2.0 for c in deep}
            return d, {k: "dynamut2" for k in d}
        with patch.object(MutationScanner, "_run_rosetta_batch", fake_rosetta), \
             patch.object(MutationScanner, "_run_thermompnn", fake_thermo), \
             patch.object(MutationScanner, "_run_rasp", fake_rasp), \
             patch.object(MutationScanner, "_run_dynamut2", fake_dyna):
            res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq, include_positions=[55],
                          run_rosetta=True, run_thermompnn=True, run_rasp=True)
        r = res[0]
        # ALL FOUR axes present + independently contributing, SAME physical residue (55=F)
        assert r["resnum"] == 55 and r["from_aa"] == "F"
        assert r["ddg"] == -1.0            # physics (Rosetta)
        assert r["thermompnn_ddg"] == 0.5  # ML
        assert r["rasp_ddg"] == -0.9       # physics proxy (retained; handed off)
        assert r["dynamut2_ddg"] == -2.0   # dynamics (independent)
        assert r["physics_source"] == "rosetta"
        assert r["key"] == candidate_key("A", 55, "F", r["to_aa"])

    def test_graceful_disabled_equals_pre_dynamut2(self):
        pdb = _write_pdb([(a, 50 + i, "", "A") for i, a in enumerate("MKAEDFGHILV")])
        sc, seq = _scanner_for(pdb, "A")
        def fake_rosetta(self, pdb_path, candidates, chain_id, scan_deadline=None, ddg_basis="symmetric"):
            sc_ = {f"{c['from_aa']}{c.get('resnum',c['position'])}{c['to_aa']}": -1.0 for c in candidates}
            return (sc_, {k: "pyrosetta" for k in sc_}, {}, {})
        with patch.object(MutationScanner, "_run_rosetta_batch", fake_rosetta):
            res = sc.scan(pdb_path=pdb, chain_id="A", sequence=seq, include_positions=[55],
                          run_rosetta=True, run_thermompnn=False, run_rasp=False, run_dynamut2=False)
        r = res[0]
        assert r["dynamut2_ddg"] is None and r["dynamut2_source"] == "not_computed"
        # score == Rosetta + properties only (no dynamics axis) — pre-DynaMut2
        assert r["combined_score"] == present_voters_score(
            -1.0, None, r["solubility_delta"], r["esm_tolerance"])
