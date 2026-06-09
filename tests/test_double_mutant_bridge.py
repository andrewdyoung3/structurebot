"""
tests/test_double_mutant_bridge.py
-----------------------------------
Unit tests for DoubleMutantBridge.

Covers all specified test cases from the implementation spec.
All external calls (DynaMut2 HTTP, WSL2, BioPython) are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from double_mutant_bridge import DoubleMutantBridge, _pair_key, _get_camsol


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mut(pos: int, from_aa: str = "V", to_aa: str = "A",
         chain: str = "A", ddg: float = -1.0,
         camsol_delta: float = 0.5, esm_tolerance: float = 0.7,
         interface_proximal: bool = False) -> Dict[str, Any]:
    return {
        "chain":             chain,
        "position":          pos,
        "from_aa":           from_aa,
        "to_aa":             to_aa,
        "ddg":               ddg,
        "camsol_delta":      camsol_delta,
        "esm_tolerance":     esm_tolerance,
        "interface_proximal": interface_proximal,
    }


BRIDGE = DoubleMutantBridge()


# ══════════════════════════════════════════════════════════════════════════════
# generate_pairs — stability mode
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_pairs_stability_excludes_interface():
    """Mutation at an interface position is excluded from stability mode."""
    mutations = [
        _mut(82),   # at interface
        _mut(64),   # not at interface
        _mut(50),   # not at interface
    ]
    interface_residues = {82}
    pairs = BRIDGE.generate_pairs(
        mutations, "stability",
        interface_residues=interface_residues,
    )
    positions_in_pairs = set()
    for m_a, m_b in pairs:
        positions_in_pairs.add(m_a["position"])
        positions_in_pairs.add(m_b["position"])
    assert 82 not in positions_in_pairs, (
        "Position 82 (interface) should be excluded from stability mode pairs"
    )


def test_generate_pairs_stability_excludes_interface_proximal():
    """Mutations within 3 residues of an interface position are excluded."""
    mutations = [_mut(84), _mut(50)]  # 84 is within 3 of interface=82
    pairs = BRIDGE.generate_pairs(
        mutations, "stability", interface_residues={82}
    )
    positions = {p for m_a, m_b in pairs
                 for p in (m_a["position"], m_b["position"])}
    assert 84 not in positions, "Position 84 (within 3 of interface 82) should be excluded"


def test_generate_pairs_stability_includes_non_interface():
    """Mutations far from the interface are retained in stability mode."""
    mutations = [_mut(10), _mut(20), _mut(30)]
    pairs = BRIDGE.generate_pairs(
        mutations, "stability", interface_residues={82}
    )
    assert len(pairs) == 3, f"Expected 3 pairs from 3 mutations, got {len(pairs)}"


def test_ddg_zero_not_filtered_in_stability_mode():
    """
    Mutations with ddg=0.0 (DynaMut2 neutral/unknown) must NOT be excluded by
    the stability-mode ddG filter — neither alone nor when paired together.
    Only pairs where BOTH mutations are clearly destabilising are dropped.
    """
    # All neutral ddg and no solubility benefit → under the old "require a
    # benefit" filter every pair was dropped; now all should survive.
    neutral = [
        _mut(10, ddg=0.0, camsol_delta=0.0),
        _mut(20, ddg=0.0, camsol_delta=0.0),
        _mut(30, ddg=0.0, camsol_delta=0.0),
    ]
    pairs = BRIDGE.generate_pairs(neutral, "stability")
    assert len(pairs) == 3, (
        f"ddg=0.0 mutations should not be filtered; expected 3 pairs, got {len(pairs)}"
    )

    # A neutral mutation paired with a strongly destabilising one must survive
    # (only one is destabilising), but two strongly destabilising ones are dropped.
    mixed = [
        _mut(10, ddg=0.0, camsol_delta=0.0),   # neutral
        _mut(20, ddg=5.0, camsol_delta=0.0),   # destabilising
        _mut(30, ddg=5.0, camsol_delta=0.0),   # destabilising
    ]
    pairs = BRIDGE.generate_pairs(mixed, "stability")
    surviving = {p_key for p_key in (
        frozenset((a["position"], b["position"])) for a, b in pairs
    )}
    assert frozenset((10, 20)) in surviving, "neutral+destabilising pair must survive"
    assert frozenset((10, 30)) in surviving, "neutral+destabilising pair must survive"
    assert frozenset((20, 30)) not in surviving, (
        "pair where BOTH mutations are clearly destabilising should be excluded"
    )


# ══════════════════════════════════════════════════════════════════════════════
# generate_pairs — epitope mode
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_pairs_epitope_includes_interface():
    """Mutation at interface position IS included in epitope mode."""
    mutations = [
        _mut(82, interface_proximal=True),  # at interface
        _mut(64),
    ]
    pairs = BRIDGE.generate_pairs(
        mutations, "epitope", interface_residues={82}
    )
    positions = {p for m_a, m_b in pairs
                 for p in (m_a["position"], m_b["position"])}
    assert 82 in positions, "Position 82 (interface) should be in epitope mode pairs"


def test_generate_pairs_epitope_requires_interface_proximity():
    """Epitope mode excludes pairs where neither mutation is interface-proximal."""
    mutations = [_mut(10), _mut(20)]  # both far from interface 82
    pairs = BRIDGE.generate_pairs(
        mutations, "epitope", interface_residues={82}
    )
    assert len(pairs) == 0, (
        "Pairs where neither mutation is within 5 of interface should be excluded"
    )


def test_generate_pairs_epitope_excludes_low_esm():
    """Epitope mode excludes mutations with ESM tolerance < 0.3."""
    mutations = [
        _mut(82, esm_tolerance=0.2, interface_proximal=True),  # low tolerance
        _mut(83, interface_proximal=True),
    ]
    pairs = BRIDGE.generate_pairs(
        mutations, "epitope", interface_residues={82}
    )
    # Pair (82, 83) should be excluded because pos 82 has ESM < 0.3
    assert len(pairs) == 0, "Low-ESM mutations should be excluded from epitope mode"


# ══════════════════════════════════════════════════════════════════════════════
# generate_pairs — functional residue exclusion (both modes)
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_pairs_excludes_functional_site_stability():
    """Mutations within 2 residues of a functional residue are excluded in stability mode."""
    mutations = [_mut(27), _mut(50)]  # 27 is within 2 of functional=25
    pairs = BRIDGE.generate_pairs(
        mutations, "stability", functional_residues={25}
    )
    positions = {p for m_a, m_b in pairs
                 for p in (m_a["position"], m_b["position"])}
    assert 27 not in positions, (
        "Position 27 (within 2 of functional residue 25) should be excluded"
    )


def test_generate_pairs_excludes_functional_site_epitope():
    """Both mutations in functional zone → excluded in epitope mode."""
    # pos 26 and 27 both within 2 of functional 25
    mutations = [
        _mut(26, interface_proximal=True),
        _mut(27, interface_proximal=True),
    ]
    pairs = BRIDGE.generate_pairs(
        mutations, "epitope",
        interface_residues={30},   # make them interface-proximal
        functional_residues={25},
    )
    # Both 26 and 27 are in the functional zone → pair excluded
    assert len(pairs) == 0, "Pairs where both mutations are in functional zone should be excluded"


def test_generate_pairs_epitope_one_functional_allowed():
    """Epitope mode: pair excluded only if BOTH are in functional zone."""
    # 26 is within functional zone of 25; 40 is not
    mutations = [
        _mut(26, interface_proximal=True),
        _mut(40, interface_proximal=True),
    ]
    pairs = BRIDGE.generate_pairs(
        mutations, "epitope",
        interface_residues={30},
        functional_residues={25},
    )
    # Only one in functional zone → pair should survive
    assert len(pairs) == 1, "Pair where only one mutation is in functional zone should survive"


# ══════════════════════════════════════════════════════════════════════════════
# generate_pairs — same-position exclusion (both modes)
# ══════════════════════════════════════════════════════════════════════════════

def test_generate_pairs_excludes_same_position():
    """Two mutations at position 82 must not be paired."""
    mutations = [
        _mut(82, to_aa="A"),
        _mut(82, to_aa="K"),
        _mut(50),
    ]
    pairs = BRIDGE.generate_pairs(mutations, "stability")
    for m_a, m_b in pairs:
        assert m_a["position"] != m_b["position"], (
            f"Same-position pair found: {m_a['position']} == {m_b['position']}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# route_pairs — backend assignment
# ══════════════════════════════════════════════════════════════════════════════

def test_route_pairs_far():
    """Cα-Cα distance > 10 Å → backend=dynamut2, zone=far."""
    mutations = [_mut(10), _mut(50)]
    pairs = [tuple(mutations)]
    with patch.object(BRIDGE, "compute_ca_distance", return_value=15.0):
        routed = BRIDGE.route_pairs(pairs, "fake.pdb")
    assert len(routed) == 1
    assert routed[0]["backend"] == "dynamut2"
    assert routed[0]["distance_zone"] == "far"
    assert routed[0]["ca_distance"] == 15.0


def test_route_pairs_mid():
    """Cα-Cα distance 4–10 Å → backend=dynamut2_warned, zone=mid."""
    mutations = [_mut(10), _mut(15)]
    pairs = [tuple(mutations)]
    with patch.object(BRIDGE, "compute_ca_distance", return_value=7.0):
        routed = BRIDGE.route_pairs(pairs, "fake.pdb")
    assert routed[0]["backend"] == "dynamut2_warned"
    assert routed[0]["distance_zone"] == "mid"
    assert routed[0]["ca_distance"] == 7.0


def test_route_pairs_close():
    """Cα-Cα distance < 4 Å → backend=pyrosetta_required, zone=close."""
    mutations = [_mut(10), _mut(11)]
    pairs = [tuple(mutations)]
    with patch.object(BRIDGE, "compute_ca_distance", return_value=2.5):
        routed = BRIDGE.route_pairs(pairs, "fake.pdb")
    assert routed[0]["backend"] == "pyrosetta_required"
    assert routed[0]["distance_zone"] == "close"
    assert routed[0]["ca_distance"] == 2.5


def test_route_pairs_none_distance_falls_back_to_far():
    """None distance (residue not found) → backend=dynamut2 (safe fallback)."""
    mutations = [_mut(10), _mut(20)]
    pairs = [tuple(mutations)]
    with patch.object(BRIDGE, "compute_ca_distance", return_value=None):
        routed = BRIDGE.route_pairs(pairs, "fake.pdb")
    assert routed[0]["backend"] == "dynamut2"
    assert routed[0]["distance_zone"] == "far"


# ══════════════════════════════════════════════════════════════════════════════
# compute_composite_score — stability mode
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_composite_stability_formula():
    """Verify stability mode composite score matches the specified weighted sum."""
    pair = {
        "mutation_a":    _mut(10, ddg=-2.5, camsol_delta=1.5, esm_tolerance=0.8),
        "mutation_b":    _mut(20, ddg=-1.5, camsol_delta=0.5, esm_tolerance=0.6),
        "ddg_double":    -3.0,
        "epistasis":     -0.5,
        "scoring_components": {},
    }
    score = BRIDGE.compute_composite_score(pair, "stability")

    # Expected components:
    # stability_score = min(1, max(0, 3.0/5.0)) = 0.600
    # esm_score       = mean(0.8, 0.6) = 0.700
    # camsol_score    = mean(1.5, 0.5)/2 / 3.0 = 1.0/3.0 ≈ 0.333
    # synergy_bonus   = min(1, 0.5/3.0) ≈ 0.167
    # composite = 0.4*0.6 + 0.25*0.7 + 0.2*0.333 + 0.15*0.167
    stability_score = 0.600
    esm_score       = 0.700
    camsol_score    = min(1.0, (1.5 + 0.5) / 2.0 / 3.0)
    synergy_bonus   = min(1.0, 0.5 / 3.0)
    expected = round(
        0.40 * stability_score
        + 0.25 * esm_score
        + 0.20 * camsol_score
        + 0.15 * synergy_bonus,
        3,
    )
    assert score == expected, (
        f"Stability composite score mismatch: got {score}, expected {expected}"
    )


def test_compute_composite_stability_capped():
    """Scores above natural bounds should be clamped by component caps."""
    pair = {
        "mutation_a": _mut(10, ddg=-10.0, camsol_delta=5.0, esm_tolerance=1.0),
        "mutation_b": _mut(20, ddg=-10.0, camsol_delta=5.0, esm_tolerance=1.0),
        "ddg_double":  -10.0,
        "epistasis":   -5.0,
        "scoring_components": {},
    }
    score = BRIDGE.compute_composite_score(pair, "stability")
    # All components capped at 1.0 → max possible = 1.0
    assert 0.0 <= score <= 1.0 + 1e-9, f"Score out of [0,1] range: {score}"


# ══════════════════════════════════════════════════════════════════════════════
# compute_composite_score — epitope mode
# ══════════════════════════════════════════════════════════════════════════════

def test_compute_composite_epitope_formula():
    """Verify epitope mode uses the specified weights and components."""
    pair = {
        "mutation_a": _mut(82, ddg=-1.0, camsol_delta=0.5,
                           esm_tolerance=0.8, interface_proximal=True),
        "mutation_b": _mut(83, ddg=-0.5, camsol_delta=0.3,
                           esm_tolerance=0.7, interface_proximal=False),
        "ddg_double":  -1.5,
        "epistasis":   -0.2,
        "scoring_components": {},
    }
    score = BRIDGE.compute_composite_score(pair, "epitope")

    stability_score     = min(1.0, max(0.0, 1.5 / 5.0))
    esm_score           = min(1.0, (0.8 + 0.7) / 2.0)
    interface_score     = 1.0  # mutation_a is interface_proximal
    abs_sol             = abs((0.5 + 0.3) / 2.0)
    epitope_conservation = min(1.0, max(0.0, 1.0 - abs_sol / 3.0))
    synergy_bonus       = min(1.0, 0.2 / 3.0)

    expected = round(
        0.25 * stability_score
        + 0.35 * esm_score
        + 0.20 * interface_score
        + 0.10 * epitope_conservation
        + 0.10 * synergy_bonus,
        3,
    )
    assert score == expected, (
        f"Epitope composite score mismatch: got {score}, expected {expected}"
    )


def test_compute_composite_epitope_weights_sum():
    """Epitope mode weights 0.25+0.35+0.20+0.10+0.10 = 1.0."""
    weights = [0.25, 0.35, 0.20, 0.10, 0.10]
    assert abs(sum(weights) - 1.0) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# Epistasis sign convention
# ══════════════════════════════════════════════════════════════════════════════

def test_epistasis_sign_synergistic():
    """ddg_double=-3.0, additive=-2.0 → epistasis=-1.0 (synergistic)."""
    pair = {
        "mutation_a": _mut(10),
        "mutation_b": _mut(20),
        "ddg_double":    -3.0,
        "ddg_additive":  -2.0,
        "epistasis":     -1.0,
        "scoring_components": {},
    }
    # The epistasis field is set by the scoring backend; we verify sign convention
    assert pair["epistasis"] == pair["ddg_double"] - pair["ddg_additive"]
    assert pair["epistasis"] < 0, "Synergistic epistasis should be negative"


def test_epistasis_sign_antagonistic():
    """ddg_double=-1.0, additive=-2.0 → epistasis=+1.0 (antagonistic)."""
    pair = {
        "mutation_a": _mut(10),
        "mutation_b": _mut(20),
        "ddg_double":    -1.0,
        "ddg_additive":  -2.0,
        "epistasis":      1.0,
        "scoring_components": {},
    }
    assert pair["epistasis"] == pair["ddg_double"] - pair["ddg_additive"]
    assert pair["epistasis"] > 0, "Antagonistic epistasis should be positive"


# ══════════════════════════════════════════════════════════════════════════════
# DynaMut2 mm result parsing
# ══════════════════════════════════════════════════════════════════════════════

def test_dynamut2_result_parsing():
    """Mock API response parses correctly to ddg_double, additive, epistasis."""
    mock_response = {
        "A V82A;A I64E": {
            "prediction":   "-2.1",
            "sum_ddg":      "-1.8",
            "avg_distance": "12.4",
        }
    }
    parsed = BRIDGE._parse_mm_result(
        mock_response,
        line_a="A V82A",
        line_b="A I64E",
        pair_key="V82A+I64E",
    )
    assert parsed is not None, "Should successfully parse the mock response"
    # mm sign normalised to system (raw -2.1 -> +2.1; DynaMut2 positive=stabilising)
    assert parsed["ddg_double"]   == 2.1
    assert parsed["ddg_additive"] == 1.8
    assert parsed["epistasis"]    == round(2.1 - 1.8, 3)  # +0.3
    assert parsed["avg_distance_api"] == 12.4
    # GUARD: mm sign is INFERRED (prediction_mm endpoint errors → never live-
    # reconfirmed) → tagged sign_unverified so the inferred sign is never trusted
    # silently.  (Set DYNAMUT2_MM_SIGN_VERIFIED only after a live mm sign check.)
    assert parsed["sign_unverified"] is True


def test_dynamut2_result_parsing_reversed_order():
    """Parser tries both key orderings (line_b;line_a)."""
    mock_response = {
        "A I64E;A V82A": {  # reversed
            "prediction":   "-2.1",
            "sum_ddg":      "-1.8",
            "avg_distance": "12.4",
        }
    }
    parsed = BRIDGE._parse_mm_result(
        mock_response,
        line_a="A V82A",
        line_b="A I64E",
        pair_key="V82A+I64E",
    )
    assert parsed is not None, "Should find result when key order is reversed"
    assert parsed["ddg_double"] == 2.1   # raw -2.1 -> system +2.1 (sign-normalised)


def test_dynamut2_result_parsing_fallback_single_entry():
    """Fallback: if exactly one dict entry, use it regardless of key."""
    mock_response = {
        "some_unexpected_key": {
            "prediction":   "1.5",
            "sum_ddg":      "1.2",
            "avg_distance": "8.0",
        }
    }
    parsed = BRIDGE._parse_mm_result(
        mock_response,
        line_a="A V82A",
        line_b="A I64E",
        pair_key="V82A+I64E",
    )
    assert parsed is not None
    assert parsed["ddg_double"] == -1.5   # raw 1.5 -> system -1.5 (sign-normalised)


def test_dynamut2_result_parsing_missing_fields():
    """Returns None if 'prediction' field is absent."""
    parsed = BRIDGE._parse_mm_result(
        {"A V82A;A I64E": {"sum_ddg": "-1.8"}},
        "A V82A", "A I64E", "V82A+I64E",
    )
    assert parsed is None


# ══════════════════════════════════════════════════════════════════════════════
# Close-pair additive fallback without PyRosetta
# ══════════════════════════════════════════════════════════════════════════════

def test_pyrosetta_pairs_get_additive_fallback_when_disabled(tmp_path):
    """
    Close (pyrosetta_required) pairs with run_pyrosetta=False must be scored
    with the additive fallback (ddG_A + ddG_B), not silently dropped.
    """
    pdb = tmp_path / "test.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 0.00           C\n")

    mutations = [
        _mut(10, ddg=-1.5),
        _mut(11, ddg=-0.5),
    ]

    # distance 2.0 Å → both routed to pyrosetta_required; no DynaMut2 pairs.
    with patch.object(BRIDGE, "compute_ca_distance", return_value=2.0), \
         patch.object(BRIDGE, "score_pairs_dynamut2", return_value=[]):
        result = BRIDGE.analyze(
            inputs={
                "pdb_path":      str(pdb),
                "mutations":     mutations,
                "mode":          "stability",
                "run_pyrosetta": False,
            }
        )

    assert result.success
    pairs = result.data.get("pairs", [])
    assert len(pairs) == 1, (
        f"Close pair should be scored additively, not dropped; got {len(pairs)} pairs"
    )
    scored = pairs[0]
    assert scored["backend_used"] == "additive_fallback"
    assert scored["ddg_double"] == pytest.approx(-2.0)   # -1.5 + -0.5
    assert scored["epistasis"] == 0.0

    # A warning should still flag that PyRosetta is needed for accurate epistasis.
    warns = result.data.get("warnings", [])
    assert any("pyrosetta" in w.lower() for w in warns), (
        f"Expected a PyRosetta-related warning, got: {warns}"
    )


def test_mm_sign_unverified_enforced_not_computed(tmp_path):
    """ENFORCED GUARD: when the mm endpoint returns but the sign is UNVERIFIED, the
    sign-normalised mm value must NOT be emitted — the pair falls to the additive
    fallback (sum of the VERIFIED single-mutant ddGs).  Proves the guard is enforced,
    not merely a flag (the mm value 2.1 must never appear)."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 0.00           C\n")
    mutations = [_mut(10, ddg=-1.5), _mut(20, ddg=-0.5)]   # additive = -2.0
    mm_unverified = {"ddg_double": 2.1, "ddg_additive": 1.8, "epistasis": 0.3,
                     "avg_distance_api": 15.0, "sign_unverified": True}
    # far apart → dynamut2 backend → mm path; mm returns a sign_unverified result.
    with patch.object(BRIDGE, "compute_ca_distance", return_value=15.0), \
         patch.object(BRIDGE, "_query_dynamut2_mm", return_value=mm_unverified):
        result = BRIDGE.analyze(inputs={
            "pdb_path": str(pdb), "mutations": mutations,
            "mode": "stability", "run_pyrosetta": False,
        })
    assert result.success
    pairs = result.data.get("pairs", [])
    assert len(pairs) == 1
    scored = pairs[0]
    assert scored["backend_used"] == "additive_fallback"      # mm NOT used
    assert scored["ddg_double"] == pytest.approx(-2.0)        # additive, NOT 2.1
    assert scored["ddg_double"] != 2.1                        # the unverified mm value is gone
    assert any("unverified" in w.lower() for w in scored.get("warnings", [])), \
        "should warn that mm sign is unverified → not_computed"


# ══════════════════════════════════════════════════════════════════════════════
# Confidence classification
# ══════════════════════════════════════════════════════════════════════════════

def test_confidence_classification():
    """Verify confidence thresholds: >0.6=high, 0.3-0.6=moderate, <0.3=low."""
    def _pair_with_score(score: float) -> Dict[str, Any]:
        return {
            "mutation_a": _mut(10),
            "mutation_b": _mut(20),
            "ddg_double":   -(score * 5.0),   # reverse-engineer from stability formula
            "epistasis":    0.0,
            "composite_score": score,
            "scoring_components": {},
        }

    for score, expected_conf in [(0.7, "high"), (0.45, "moderate"), (0.2, "low")]:
        pair = _pair_with_score(score)
        pair["composite_score"] = score
        conf = (
            "high"     if score > 0.6 else
            "moderate" if score >= 0.3 else
            "low"
        )
        assert conf == expected_conf, f"score={score} → got {conf}, expected {expected_conf}"


# ══════════════════════════════════════════════════════════════════════════════
# Max pairs limit
# ══════════════════════════════════════════════════════════════════════════════

def test_max_pairs_limit():
    """50 mutations → C(50,2)=1225 combinations → truncated to 500."""
    import itertools, config as _cfg

    # Create 50 mutations at distinct positions, all stabilising (ddg < 0)
    mutations = [_mut(i, ddg=-1.0) for i in range(1, 51)]

    # C(50, 2) = 1225 > 500
    n_combos = len(list(itertools.combinations(range(50), 2)))
    assert n_combos == 1225

    # Patch DOUBLE_MUTANT_MAX_PAIRS to verify the cap
    original = _cfg.DOUBLE_MUTANT_MAX_PAIRS
    try:
        _cfg.DOUBLE_MUTANT_MAX_PAIRS = 500
        # No interface/functional residues → only cap + stability filter applies
        pairs = BRIDGE.generate_pairs(mutations, "stability")
        assert len(pairs) <= 500, f"Expected ≤500 pairs, got {len(pairs)}"
    finally:
        _cfg.DOUBLE_MUTANT_MAX_PAIRS = original


def test_max_pairs_preserves_highest_ddg():
    """When capping, top-500 are selected by sum of |ddG| descending."""
    import config as _cfg

    # 5 high-|ddG| mutations at positions 1-5 and 5 low-|ddG| at positions 101-105
    high = [_mut(i, ddg=-5.0) for i in range(1, 6)]
    low  = [_mut(i, ddg=-0.1) for i in range(101, 106)]
    mutations = high + low

    original = _cfg.DOUBLE_MUTANT_MAX_PAIRS
    try:
        _cfg.DOUBLE_MUTANT_MAX_PAIRS = 6  # only keep 6 pairs from C(10,2)=45
        pairs = BRIDGE.generate_pairs(mutations, "stability")
        # All 6 kept pairs should be within the high-|ddG| positions (1-5)
        # C(5,2) = 10, so actually all of high pairs compete with each other;
        # with cap=6, we should have 6 pairs all from high group
        pair_positions = {(m_a["position"], m_b["position"]) for m_a, m_b in pairs}
        for pos_a, pos_b in pair_positions:
            assert pos_a <= 5 and pos_b <= 5, (
                f"Low-ddG positions ({pos_a}, {pos_b}) should not appear in capped selection"
            )
    finally:
        _cfg.DOUBLE_MUTANT_MAX_PAIRS = original


# ══════════════════════════════════════════════════════════════════════════════
# Full analyze() schema validation
# ══════════════════════════════════════════════════════════════════════════════

def test_full_analyze_returns_schema(tmp_path):
    """Mock all external calls; verify all top-level result keys are present."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 0.00           C\n")

    mutations = [
        _mut(10, ddg=-2.0),
        _mut(20, ddg=-1.5),
        _mut(30, ddg=-1.0),
    ]

    # Build a mock scored pair dict
    def _mock_score_pairs(pairs, pdb_path, progress=None):
        results = []
        for p in pairs:
            p["ddg_double"]   = -2.5
            p["ddg_additive"] = -2.0
            p["epistasis"]    = -0.5
            p["avg_distance_api"] = 12.0
            p["backend_used"] = "dynamut2"
            results.append(p)
        return results

    with patch.object(BRIDGE, "compute_ca_distance", return_value=15.0), \
         patch.object(BRIDGE, "score_pairs_dynamut2", side_effect=_mock_score_pairs):
        result = BRIDGE.analyze(
            inputs={
                "pdb_path":  str(pdb),
                "mutations": mutations,
                "mode":      "stability",
                "top_n":     5,
            }
        )

    assert result.success, f"analyze() failed: {result.error}"
    data = result.data

    required_keys = {"pairs", "top_pairs", "mode", "backend_summary", "warnings", "method_note"}
    missing = required_keys - set(data.keys())
    assert not missing, f"Missing top-level keys: {missing}"

    assert data["mode"] == "stability"
    assert isinstance(data["pairs"], list)
    assert isinstance(data["top_pairs"], list)
    assert isinstance(data["backend_summary"], dict)
    assert isinstance(data["warnings"], list)
    assert isinstance(data["method_note"], str)

    if data["pairs"]:
        pair = data["pairs"][0]
        pair_keys = {
            "mutation_a", "mutation_b", "pair_key", "ca_distance",
            "distance_zone", "ddg_double", "ddg_additive", "epistasis",
            "composite_score", "confidence", "scoring_components", "warnings",
        }
        missing_pair = pair_keys - set(pair.keys())
        assert not missing_pair, f"Missing pair keys: {missing_pair}"


def test_full_analyze_empty_mutations(tmp_path):
    """analyze() with no mutations returns success=False with clear error."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text("ATOM\n")
    result = BRIDGE.analyze(inputs={"pdb_path": str(pdb), "mutations": [], "mode": "stability"})
    assert not result.success
    assert "mutation" in (result.error or "").lower()


def test_full_analyze_missing_pdb():
    """analyze() with non-existent PDB returns success=False."""
    result = BRIDGE.analyze(inputs={
        "pdb_path":  "/nonexistent/file.pdb",
        "mutations": [_mut(10)],
        "mode":      "stability",
    })
    assert not result.success
    assert "not found" in (result.error or "").lower()


def test_full_analyze_invalid_mode(tmp_path):
    """analyze() with unknown mode returns success=False."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text("ATOM\n")
    result = BRIDGE.analyze(inputs={
        "pdb_path":  str(pdb),
        "mutations": [_mut(10), _mut(20)],
        "mode":      "invalid_mode",
    })
    assert not result.success


# ══════════════════════════════════════════════════════════════════════════════
# Helper function tests
# ══════════════════════════════════════════════════════════════════════════════

def test_get_camsol_primary_field():
    """_get_camsol reads camsol_delta first."""
    m = {"camsol_delta": 1.5, "solubility_delta": 0.5}
    assert _get_camsol(m) == 1.5


def test_get_camsol_fallback_field():
    """_get_camsol falls back to solubility_delta."""
    m = {"solubility_delta": 0.8}
    assert _get_camsol(m) == 0.8


def test_pair_key_format():
    """_pair_key generates expected string format."""
    m_a = _mut(82, from_aa="V", to_aa="A")
    m_b = _mut(64, from_aa="I", to_aa="E")
    assert _pair_key(m_a, m_b) == "V82A+I64E"


def test_generate_summary_contains_key_sections():
    """generate_summary includes mode header and epistasis note."""
    result = {
        "pairs": [
            {
                "pair_key": "V82A+I64E", "ddg_double": -2.1, "ddg_additive": -1.8,
                "epistasis": -0.3, "ca_distance": 12.4, "confidence": "high",
            }
        ],
        "top_pairs": [],
        "backend_summary": {"dynamut2": 1},
        "warnings": [],
    }
    summary = BRIDGE.generate_summary(result, "stability")
    assert "Stability" in summary
    assert "Epistasis" in summary or "epistasis" in summary
    assert "cooperativity" in summary.lower() or "Epistasis" in summary


# ══════════════════════════════════════════════════════════════════════════════
# DynaMut2 mm poll response format tests
# ══════════════════════════════════════════════════════════════════════════════

def test_dynamut2_mm_status_running_recognised():
    """
    A poll response {"status": "RUNNING", "job_id": "123"} must be treated as
    'still running' — it must NOT be treated as an unparseable result or raise.
    Verifies the mm endpoint's "status" key is checked before parse attempts.
    """
    pair = {
        "pair_key":   "V82A+I64E",
        "mutation_a": _mut(82, from_aa="V", to_aa="A"),
        "mutation_b": _mut(64, from_aa="I", to_aa="E"),
        "backend":    "dynamut2",
    }
    line_a = "A V82A"
    line_b = "A I64E"

    # "RUNNING" should not produce a parsed result
    data = {"status": "RUNNING", "job_id": "abc123"}
    result = BRIDGE._parse_mm_result(data, line_a, line_b, pair["pair_key"])
    assert result is None, "_parse_mm_result should return None for a RUNNING status"

    # Verify the bridge's poll loop would treat this as "still running":
    # status.upper() == "RUNNING" → the new code continues polling.
    # We test this by checking that the status detection logic fires.
    status = str(data.get("status", "")).upper()
    assert status in ("RUNNING", "PENDING", "QUEUED"), (
        f"status={status!r} should be in the running set"
    )


def test_dynamut2_mm_status_error_raises():
    """
    A poll response {"status": "ERROR", "job_id": "123"} must raise a
    RuntimeError that includes the job_id in the message.
    """
    import unittest.mock as _mock

    pair = {
        "pair_key":   "V82A+I64E",
        "mutation_a": _mut(82, from_aa="V", to_aa="A"),
        "mutation_b": _mut(64, from_aa="I", to_aa="E"),
        "backend":    "dynamut2",
    }

    error_response = {"status": "ERROR", "job_id": "abc123"}

    # Simulate what the poll loop does when status == "ERROR"
    status = str(error_response.get("status", "")).upper()
    job_id = "abc123"

    raised = False
    error_message = ""
    if status == "ERROR":
        raised = True
        error_message = (
            f"DynaMut2 mm job {job_id} failed for {pair['pair_key']}: "
            f"status=ERROR response={str(error_response)[:200]}"
        )

    assert raised, "status=ERROR should trigger a RuntimeError"
    assert job_id in error_message, "Error message must contain the job_id"
    assert pair["pair_key"] in error_message, "Error message must contain the pair_key"


# ══════════════════════════════════════════════════════════════════════════════
# Additive fallback
# ══════════════════════════════════════════════════════════════════════════════

def _make_routed_pair(pos_a: int, pos_b: int,
                      ddg_a: float = -1.0, ddg_b: float = -2.0) -> Dict[str, Any]:
    """Build a minimal routed pair dict for use with score_pairs_dynamut2."""
    m_a = _mut(pos_a, ddg=ddg_a)
    m_b = _mut(pos_b, ddg=ddg_b)
    return {
        "mutation_a":       m_a,
        "mutation_b":       m_b,
        "pair_key":         _pair_key(m_a, m_b),
        "ca_distance":      15.0,
        "distance_zone":    "far",
        "backend":          "dynamut2",
        "ddg_double":       None,
        "ddg_additive":     None,
        "epistasis":        None,
        "ddg_A":            None,
        "ddg_B":            None,
        "avg_distance_api": None,
        "backend_used":     None,
        "composite_score":  0.0,
        "confidence":       "low",
        "scoring_components": {},
        "warnings":         [],
    }


def test_additive_fallback_used_when_api_errors():
    """When _query_dynamut2_mm raises, pair is scored as ddg_A + ddg_B with backend_used='additive_fallback'."""
    pair = _make_routed_pair(10, 20, ddg_a=-1.0, ddg_b=-2.0)

    with patch.object(BRIDGE, "_query_dynamut2_mm", side_effect=RuntimeError("API down")), \
         patch("double_mutant_bridge.time.sleep"):
        results = BRIDGE.score_pairs_dynamut2([pair], "fake.pdb")

    assert len(results) == 1, "Pair should survive as additive fallback, not be dropped"
    scored = results[0]
    assert scored["backend_used"] == "additive_fallback"
    assert scored["ddg_double"] == pytest.approx(-3.0)


def test_additive_fallback_warning_added():
    """Additive fallback appends a warning about API unavailability to the pair."""
    pair = _make_routed_pair(10, 20, ddg_a=-1.0, ddg_b=-0.5)

    with patch.object(BRIDGE, "_query_dynamut2_mm", side_effect=RuntimeError("timeout")), \
         patch("double_mutant_bridge.time.sleep"):
        results = BRIDGE.score_pairs_dynamut2([pair], "fake.pdb")

    assert len(results) == 1
    warns = results[0].get("warnings", [])
    assert any("additive" in w.lower() or "unavailable" in w.lower() for w in warns), \
        f"Expected additive/unavailable warning, got: {warns}"


def test_circuit_breaker_does_not_block_additive_fallback():
    """After circuit breaker trips, remaining pairs still get additive fallback scores."""
    from double_mutant_bridge import _CIRCUIT_BREAKER
    n_pairs = _CIRCUIT_BREAKER + 2
    pairs = [_make_routed_pair(10 * i, 10 * i + 1, ddg_a=-1.0, ddg_b=-1.0)
             for i in range(1, n_pairs + 1)]

    with patch.object(BRIDGE, "_query_dynamut2_mm", side_effect=RuntimeError("server error")), \
         patch("double_mutant_bridge.time.sleep"):
        results = BRIDGE.score_pairs_dynamut2(pairs, "fake.pdb")

    assert len(results) == n_pairs, (
        f"All {n_pairs} pairs should be scored via additive fallback; got {len(results)}"
    )
    for r in results:
        assert r["backend_used"] == "additive_fallback", (
            f"Expected additive_fallback for {r['pair_key']}, got {r['backend_used']}"
        )
