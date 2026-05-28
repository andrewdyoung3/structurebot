"""
tests/test_netnglyc_bridge.py
------------------------------
Unit tests for netnglyc_bridge.py — NetNGlyc 1.0 OST recognition prediction.

All tests mock the requests.post call so no network access is required.
Tests are split into sections:

  Section A — module-level helpers
  Section B — predict_glycosylation()
  Section C — score_engineered_sequon()
  Section D — integrate_with_glycan_candidates()
"""

import sys
import os

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from unittest.mock import MagicMock, patch

import netnglyc_bridge as nb
from netnglyc_bridge import (
    NetNGlycBridge,
    _harmonic_mean,
    _classify_ost_score,
    _strip_html,
    _parse_netnglyc_output,
    _find_prediction_at,
)


# ── helpers / fixtures ────────────────────────────────────────────────────────

# Sample NetNGlyc-style text table returned by the server.
# Columns: name  pos  aa  sequon  score  prediction
_SAMPLE_RESPONSE = """\
<html><body><pre>
query   5  N  NIT  0.5766  +
query  12  N  NAS  0.3100  -
query  25  N  NGT  0.7800  +
</pre></body></html>
"""

_SEQUENCE = "ACDENITSACDNASACDEFGHIKLNGTACE"
#             123456789012345678901234567890
#                 ^5      ^12             ^25 (N at positions 5, 12, 25)


def _make_response(text: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.raise_for_status = MagicMock()
    return resp


# ── Section A — module-level helpers ─────────────────────────────────────────

class TestHelpers:

    def test_harmonic_mean_basic(self):
        result = _harmonic_mean(0.9, 0.6)
        assert abs(result - 2 * 0.9 * 0.6 / (0.9 + 0.6)) < 1e-6

    def test_harmonic_mean_zero(self):
        assert _harmonic_mean(0.0, 0.8) == 0.0
        assert _harmonic_mean(0.5, 0.0) == 0.0

    def test_classify_ost_score(self):
        assert _classify_ost_score(0.75) == "high"
        assert _classify_ost_score(0.60) == "medium"
        assert _classify_ost_score(0.40) == "low"
        assert _classify_ost_score(0.70) == "high"   # boundary: ≥ 0.7
        assert _classify_ost_score(0.50) == "medium"  # boundary: ≥ 0.5

    def test_strip_html(self):
        assert _strip_html("<b>hello</b>") == "hello"
        assert _strip_html("no tags") == "no tags"
        assert _strip_html("<p><em>nested</em></p>") == "nested"

    def test_parse_netnglyc_output_primary_regex(self):
        rows = _parse_netnglyc_output(_SAMPLE_RESPONSE)
        assert len(rows) == 3
        pos5 = next(r for r in rows if r["position"] == 5)
        assert abs(pos5["score"] - 0.5766) < 1e-4
        assert pos5["prediction"] == "Glycosylated"

        pos12 = next(r for r in rows if r["position"] == 12)
        assert pos12["prediction"] == "Not glycosylated"

    def test_find_prediction_at(self):
        preds = [
            {"position": 5,  "ost_score": 0.5766},
            {"position": 25, "ost_score": 0.7800},
        ]
        assert _find_prediction_at(preds, 5)["ost_score"] == 0.5766
        assert _find_prediction_at(preds, 25)["ost_score"] == 0.7800
        assert _find_prediction_at(preds, 99) is None
        assert _find_prediction_at(preds, None) is None


# ── Section B — predict_glycosylation() ──────────────────────────────────────

class TestPredictGlycosylation:

    @patch("requests.post")
    def test_returns_success_dict(self, mock_post):
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        result = bridge.predict_glycosylation(_SEQUENCE, name="testseq")

        assert result["success"] is True
        assert result["sequence"] == _SEQUENCE.upper()
        assert result["n_residues"] == len(_SEQUENCE)
        assert result["api_version"] == "NetNGlyc-1.0"
        assert isinstance(result["predictions"], list)
        assert result["error"] is None

    @patch("requests.post")
    def test_high_score_count(self, mock_post):
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        result = bridge.predict_glycosylation(_SEQUENCE)
        # Only position 25 (score 0.78) should be above 0.7
        assert result["n_sequons_high_score"] == 1

    @patch("requests.post")
    def test_enriched_predictions_have_ost_category(self, mock_post):
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        result = bridge.predict_glycosylation(_SEQUENCE)
        for pred in result["predictions"]:
            assert "ost_category" in pred
            assert pred["ost_category"] in ("high", "medium", "low")

    @patch("requests.post")
    def test_empty_sequence_returns_failure(self, mock_post):
        bridge = NetNGlycBridge()
        result = bridge.predict_glycosylation("")
        assert result["success"] is False
        assert "error" in result
        mock_post.assert_not_called()

    @patch("requests.post", side_effect=ConnectionError("unreachable"))
    def test_connection_error_returns_failure(self, mock_post):
        bridge = NetNGlycBridge()
        result = bridge.predict_glycosylation(_SEQUENCE)
        assert result["success"] is False
        assert "unreachable" in result["error"].lower() or "NetNGlyc" in result["error"]


# ── Section C — score_engineered_sequon() ────────────────────────────────────

class TestScoreEngineeredSequon:

    @patch("requests.post")
    def test_returns_position_and_ost_score(self, mock_post):
        # Engineered sequence where pos 5 is changed to N-I-T (existing sequon)
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        result = bridge.score_engineered_sequon(
            sequon_position      = 5,
            engineered_sequence  = _SEQUENCE,
            wildtype_sequence    = _SEQUENCE,
        )
        assert "position" in result
        assert result["position"] == 5
        assert "ost_score" in result
        assert "ost_category" in result

    @patch("requests.post")
    def test_delta_computed_when_wt_is_sequon(self, mock_post):
        # Both WT and engineered sequences are submitted — two POST calls expected
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        # Position 5 is a sequon in the wildtype too
        result = bridge.score_engineered_sequon(5, _SEQUENCE, _SEQUENCE)
        vw = result.get("vs_wildtype_sequon", {})
        assert "wt_ost_score" in vw

    @patch("requests.post")
    def test_notes_non_empty(self, mock_post):
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        result = bridge.score_engineered_sequon(5, _SEQUENCE, _SEQUENCE)
        assert isinstance(result.get("notes"), str)
        assert len(result["notes"]) > 0


# ── Section D — integrate_with_glycan_candidates() ───────────────────────────

class TestIntegrateWithGlycanCandidates:

    @patch("requests.post")
    def test_annotates_matching_candidates(self, mock_post):
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        candidates = [
            {"position": 5,  "confidence": "moderate", "composite_score": 0.7},
            {"position": 25, "confidence": "high",      "composite_score": 0.9},
        ]
        annotated = bridge.integrate_with_glycan_candidates(
            candidates, _SEQUENCE
        )
        assert len(annotated) == 2
        c5 = next(c for c in annotated if c["position"] == 5)
        c25 = next(c for c in annotated if c["position"] == 25)
        assert "ost_score" in c5
        assert "combined_confidence" in c25
        assert 0.0 <= c25["combined_confidence"] <= 1.0

    @patch("requests.post")
    def test_combined_confidence_is_harmonic_mean(self, mock_post):
        mock_post.return_value = _make_response(_SAMPLE_RESPONSE)
        bridge = NetNGlycBridge()
        candidates = [{"position": 25, "confidence": "high", "composite_score": 0.9}]
        annotated = bridge.integrate_with_glycan_candidates(candidates, _SEQUENCE)
        c = annotated[0]
        # struct_conf for "high" is 0.9 (from _CONF_MAP); ost_conf from score
        ost_conf = c.get("ost_confidence", 0.0)
        struct_conf = 0.9
        expected = 2 * struct_conf * max(ost_conf, 1e-6) / (struct_conf + max(ost_conf, 1e-6))
        assert abs(c["combined_confidence"] - round(expected, 4)) < 1e-3

    @patch("requests.post", side_effect=ConnectionError("down"))
    def test_graceful_degradation_on_api_failure(self, mock_post):
        bridge = NetNGlycBridge()
        candidates = [{"position": 5, "confidence": "moderate", "composite_score": 0.7}]
        result = bridge.integrate_with_glycan_candidates(candidates, _SEQUENCE)
        # Should return candidates unchanged (with ost_score=None and ost_error set)
        assert len(result) == 1
        assert result[0].get("ost_score") is None
        assert result[0].get("ost_error") is not None

    def test_empty_candidates_returns_empty(self):
        bridge = NetNGlycBridge()
        assert bridge.integrate_with_glycan_candidates([], _SEQUENCE) == []

    def test_empty_sequence_returns_candidates_unchanged(self):
        bridge = NetNGlycBridge()
        cands = [{"position": 5}]
        result = bridge.integrate_with_glycan_candidates(cands, "")
        assert result == cands
