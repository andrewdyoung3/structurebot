"""
Identity-correctness tests for the Ssym fwd/rev anti-symmetry mapping
(scripts/ssym_mapping.py).

The mapping is identity-critical: it must pair forward (X→A on WT) with reverse
(A→X on the mutant structure) and HARD-ERROR rather than ever silently mis-pair.
These tests exercise the real local Ssym data plus synthetic violations that must
raise SsymPairError.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import ssym_mapping as sm  # noqa: E402


def _row(pdbid, variant, score):
    return {"pdbid": pdbid, "chainid": "A", "variant": variant, "score": str(score)}


# The real-data tests below need the external RaSP_repo Ssym CSVs (an upstream clone
# + download, NOT committed to this repo). On a fresh clone they are absent → SKIP,
# mirroring the live PyRosetta/ColabFold benchmark skips — a machine-dependent test
# must skip, never hard-fail. The synthetic-data gate tests further down build their
# own rows and always run.
_SSYM_CSV = (Path(__file__).resolve().parent.parent
             / "RaSP_repo" / "data" / "test" / "Ssym_dir" / "ddG_experimental" / "ddg.csv")
_needs_ssym_data = pytest.mark.skipif(
    not _SSYM_CSV.is_file(),
    reason=f"external Ssym data absent ({_SSYM_CSV}); clone+download RaSP_repo to run",
)


# ── real data ────────────────────────────────────────────────────────────────────

@_needs_ssym_data
def test_real_ssym_builds_and_flags_renumbering():
    pairs = sm.build_ssym_pairs()
    assert len(pairs) == 352
    # the 3 known WT-vs-mutant renumbered pairs (all 4BVM ↔ 5N4*) are flagged, not fatal
    renum = [p for p in pairs if p["position_renumbered"]]
    assert len(renum) == 3
    assert {p["fwd"]["pdbid"] for p in renum} == {"4BVM"}
    for p in renum:
        assert p["position_offset"] == 1
    # every pair passed the identity gates → inverse substitution + anti-symmetry
    for p in pairs:
        f, r = p["fwd"], p["rev"]
        assert f["wt"] == r["mut"] and f["mut"] == r["wt"]
        assert abs(f["exp_ddg"] + r["exp_ddg"]) < 1e-6


@_needs_ssym_data
def test_pair_schema_is_link_ready():
    pairs = sm.build_ssym_pairs()
    p = pairs[0]
    assert p["fwd"]["pair_id"] == p["rev"]["pair_id"] == p["pair_id"]
    assert p["fwd"]["antisym_dir"] == "fwd" and p["rev"]["antisym_dir"] == "rev"
    # fwd from the WT set, rev from the mutant set — distinct structures
    assert p["fwd"]["set"] == "Ssym_dir" and p["rev"]["set"] == "Ssym_inv"
    assert "Ssym_dir" in p["fwd"]["pdb_path"] and "Ssym_inv" in p["rev"]["pdb_path"]
    # Ssym provenance: S2648 family → DynaMut2 training
    assert p["fwd"]["provenance"]["dynamut2"] == "training"


@_needs_ssym_data
def test_flatten_two_mutations_per_pair():
    pairs = sm.build_ssym_pairs()
    muts = sm.ssym_pair_mutations(pairs)
    assert len(muts) == 2 * len(pairs)
    assert sum(1 for m in muts if m["antisym_dir"] == "fwd") == len(pairs)
    assert sum(1 for m in muts if m["antisym_dir"] == "rev") == len(pairs)


# ── hard-error gates (never mis-pair) ────────────────────────────────────────────

def test_inverse_substitution_violation_raises():
    d = [_row("1AAA", "C10F", 1.0)]
    bad = [_row("2BBB", "F10A", -1.0)]   # rev.wt F ok, but rev.mut A != fwd.wt C
    with pytest.raises(sm.SsymPairError, match="inverse substitution"):
        sm.build_ssym_pairs(d, bad)


def test_non_antisymmetric_score_raises():
    d = [_row("1AAA", "C10F", 1.0)]
    bad = [_row("2BBB", "F10C", -2.0)]   # inverse subst OK but 1.0 + (-2.0) != 0
    with pytest.raises(sm.SsymPairError, match="anti-symmetric"):
        sm.build_ssym_pairs(d, bad)


def test_row_count_mismatch_raises():
    d = [_row("1AAA", "C10F", 1.0), _row("1AAA", "C10S", 1.5)]
    inv = [_row("2BBB", "F10C", -1.0)]
    with pytest.raises(sm.SsymPairError, match="row counts differ"):
        sm.build_ssym_pairs(d, inv)


def test_renumbered_pair_still_pairs_when_identity_holds():
    # position differs (10 vs 11) but inverse-subst + anti-symmetry hold → verified, flagged
    d = [_row("4BVM", "I10N", 7.63)]
    inv = [_row("5N4M", "N11I", -7.63)]
    pairs = sm.build_ssym_pairs(d, inv)
    assert len(pairs) == 1
    assert pairs[0]["position_renumbered"] is True
    assert pairs[0]["position_offset"] == 1
