"""
tests/test_mpnn_esmfold_pipeline.py
------------------------------------
Unit tests for MPNNESMFoldPipeline (mpnn_esmfold_pipeline.py).

All ESMFold calls are mocked — no subprocess or GPU required.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from mpnn_esmfold_pipeline import (
    MPNNESMFoldPipeline,
    _diff_sequences,
    _plddt_confidence,
)
from session_state import SessionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WT_SEQ  = "ACDEFGHIKLM"   # 11 residues
_SEQ_MUT = "ACDEFGHIKAM"   # mutation: L10A  (1-indexed)


def _make_fold_result(seq: str, mean_plddt: float = 75.0) -> dict:
    """Build a fake successful ESMFold predict() return value."""
    n = len(seq)
    return {
        "success":     True,
        "label":       "test",
        "pdb_str":     f"MOCK PDB DATA plddt={mean_plddt}",
        "plddt":       {i + 1: mean_plddt for i in range(n)},
        "mean_plddt":  mean_plddt,
        "length":      n,
        "error":       None,
        "source":      "local",
    }


def _make_session_with_mpnn(
    model_id: str = "1",
    sequences=None,
    wildtype_seq: str = _WT_SEQ,
) -> SessionState:
    """Return a SessionState pre-loaded with ProteinMPNN results."""
    if sequences is None:
        sequences = [
            {
                "sequence":  _SEQ_MUT,
                "score":     -1.234,
                "recovery":  0.91,
                "mutations": ["L10A"],
            },
        ]
    session = SessionState()
    session.add_proteinmpnn_result(
        model_id,
        {
            "sequences":         sequences,
            "wildtype_sequence": wildtype_seq,
            "fixed_positions":   [],
            "backend":           "local",
        },
    )
    return session


# ===========================================================================
# 1. _diff_sequences
# ===========================================================================

class TestDiffSequences:
    def test_basic_mutation(self):
        muts = _diff_sequences("ACDE", "ACAE")
        assert muts == ["D3A"]

    def test_identical_sequences(self):
        assert _diff_sequences("ACDE", "ACDE") == []

    def test_multiple_mutations(self):
        muts = _diff_sequences("AAAA", "ABCA")
        assert muts == ["A2B", "A3C"]

    def test_stops_at_shorter_sequence(self):
        # designed is shorter — should not raise, just compare up to len(designed)
        muts = _diff_sequences("ACDE", "ACD")
        assert muts == []   # no difference in overlapping region


# ===========================================================================
# 2. _plddt_confidence
# ===========================================================================

class TestPlddtConfidence:
    def test_high_boundary(self):
        assert _plddt_confidence(70.0) == "high"

    def test_above_high(self):
        assert _plddt_confidence(90.0) == "high"

    def test_medium_boundary(self):
        assert _plddt_confidence(50.0) == "medium"

    def test_medium_range(self):
        assert _plddt_confidence(65.0) == "medium"

    def test_low(self):
        assert _plddt_confidence(40.0) == "low"

    def test_zero(self):
        assert _plddt_confidence(0.0) == "low"


# ===========================================================================
# 3. MPNNESMFoldPipeline.run() — failure paths (no ESMFold needed)
# ===========================================================================

class TestRunFailurePaths:
    def test_no_mpnn_results_in_session(self):
        pipeline = MPNNESMFoldPipeline()
        session  = SessionState()   # no MPNN results stored
        result   = pipeline.run(
            model_id="1",
            pdb_path=None,
            chain_id="A",
            session=session,
        )
        assert result["success"] is False
        assert "ProteinMPNN" in result["error"]
        assert result["validated_designs"] == []

    def test_empty_sequences_list(self):
        pipeline = MPNNESMFoldPipeline()
        session  = SessionState()
        session.add_proteinmpnn_result("1", {
            "sequences":         [],
            "wildtype_sequence": _WT_SEQ,
            "fixed_positions":   [],
            "backend":           "local",
        })
        result = pipeline.run(
            model_id="1",
            pdb_path=None,
            chain_id="A",
            session=session,
        )
        assert result["success"] is False
        assert "no designed sequences" in result["error"].lower()


# ===========================================================================
# 4. MPNNESMFoldPipeline.run() — success path (mocked ESMFold)
# ===========================================================================

class TestRunSuccessPath:
    """All tests patch ESMFoldBridge so no subprocess is spawned."""

    def test_basic_run_returns_success(self):
        pipeline = MPNNESMFoldPipeline()
        session  = _make_session_with_mpnn()

        with patch("mpnn_esmfold_pipeline.ESMFoldBridge") as MockBridge:
            inst = MagicMock()
            MockBridge.return_value = inst
            inst.predict.return_value = _make_fold_result(_SEQ_MUT, mean_plddt=80.0)

            result = pipeline.run(
                model_id="1",
                pdb_path=None,
                chain_id="A",
                session=session,
                top_n=1,
                plddt_threshold=70.0,
            )

        assert result["success"] is True
        assert result["passed_count"] == 1
        assert result["failed_count"] == 0
        assert len(result["validated_designs"]) >= 1   # ≥1 because WT included

    def test_include_wildtype_adds_wt_entry(self):
        pipeline = MPNNESMFoldPipeline()
        session  = _make_session_with_mpnn()

        with patch("mpnn_esmfold_pipeline.ESMFoldBridge") as MockBridge:
            inst = MagicMock()
            MockBridge.return_value = inst
            inst.predict.return_value = _make_fold_result(_WT_SEQ, mean_plddt=78.0)

            result = pipeline.run(
                model_id="1",
                pdb_path=None,
                chain_id="A",
                session=session,
                top_n=1,
                include_wildtype=True,
            )

        wt_entries = [d for d in result["validated_designs"] if d.get("is_wildtype")]
        assert len(wt_entries) == 1

    def test_exclude_wildtype_omits_wt_entry(self):
        pipeline = MPNNESMFoldPipeline()
        session  = _make_session_with_mpnn()

        with patch("mpnn_esmfold_pipeline.ESMFoldBridge") as MockBridge:
            inst = MagicMock()
            MockBridge.return_value = inst
            inst.predict.return_value = _make_fold_result(_SEQ_MUT, mean_plddt=80.0)

            result = pipeline.run(
                model_id="1",
                pdb_path=None,
                chain_id="A",
                session=session,
                top_n=1,
                include_wildtype=False,
            )

        wt_entries = [d for d in result["validated_designs"] if d.get("is_wildtype")]
        assert len(wt_entries) == 0

    def test_design_below_threshold_fails(self):
        pipeline = MPNNESMFoldPipeline()
        session  = _make_session_with_mpnn()

        with patch("mpnn_esmfold_pipeline.ESMFoldBridge") as MockBridge:
            inst = MagicMock()
            MockBridge.return_value = inst
            # Return low pLDDT for all calls
            inst.predict.return_value = _make_fold_result(_SEQ_MUT, mean_plddt=40.0)

            result = pipeline.run(
                model_id="1",
                pdb_path=None,
                chain_id="A",
                session=session,
                top_n=1,
                plddt_threshold=70.0,
                include_wildtype=False,
            )

        assert result["passed_count"] == 0
        assert result["failed_count"] == 1
        design = [d for d in result["validated_designs"] if not d.get("is_wildtype")][0]
        assert design["pass_threshold"] is False
        assert design["confidence"] == "low"

    def test_esmfold_error_sets_error_field(self):
        """If ESMFold returns success=False, the design's error field is populated."""
        pipeline = MPNNESMFoldPipeline()
        session  = _make_session_with_mpnn()

        with patch("mpnn_esmfold_pipeline.ESMFoldBridge") as MockBridge:
            inst = MagicMock()
            MockBridge.return_value = inst
            inst.predict.return_value = {
                "success":    False,
                "error":      "GPU OOM",
                "pdb_str":    "",
                "plddt":      {},
                "mean_plddt": 0.0,
                "length":     0,
            }

            result = pipeline.run(
                model_id="1",
                pdb_path=None,
                chain_id="A",
                session=session,
                top_n=1,
                include_wildtype=False,
            )

        assert result["success"] is True    # pipeline ran OK
        design = [d for d in result["validated_designs"] if not d.get("is_wildtype")][0]
        assert design["error"] == "GPU OOM"
        assert design["pass_threshold"] is False


# ===========================================================================
# 5. generate_chimerax_commands
# ===========================================================================

class TestGenerateChimeraXCommands:
    def test_empty_designs_returns_empty(self):
        pipeline = MPNNESMFoldPipeline()
        cmds, exps = pipeline.generate_chimerax_commands([], model_id="1")
        assert cmds == []
        assert exps == []

    def test_wildtype_only_returns_empty(self):
        """If only the wildtype entry is present, no commands generated."""
        pipeline = MPNNESMFoldPipeline()
        designs = [{"rank": 0, "is_wildtype": True, "mutations": ["A1B"],
                    "plddt": {1: 80.0}, "pass_threshold": True}]
        cmds, exps = pipeline.generate_chimerax_commands(designs, model_id="1")
        assert cmds == []
        assert exps == []

    def test_generates_three_commands_per_mutation(self):
        """Each mutated residue should produce show + color + label commands."""
        pipeline = MPNNESMFoldPipeline()
        designs = [
            {
                "rank":           1,
                "is_wildtype":    False,
                "mutations":      ["L10A", "V20G"],
                "plddt":          {10: 80.0, 20: 45.0},
                "pass_threshold": True,
            }
        ]
        cmds, exps = pipeline.generate_chimerax_commands(
            designs, model_id="1", chain_id="A"
        )
        assert len(cmds) == 6      # 3 per mutation × 2 mutations
        assert len(exps) == 6
        # Check first mutation (L10A, high pLDDT)
        assert "show #1/A:10 atoms" in cmds
        assert "color #1/A:10 #00cc00" in cmds
        # Check second mutation (V20G, low pLDDT)
        assert "color #1/A:20 #cc0000" in cmds

    def test_no_mutations_returns_empty(self):
        """A design with no mutations yields no visualization commands."""
        pipeline = MPNNESMFoldPipeline()
        designs = [
            {"rank": 1, "is_wildtype": False, "mutations": [],
             "plddt": {}, "pass_threshold": True}
        ]
        cmds, exps = pipeline.generate_chimerax_commands(designs, model_id="1")
        assert cmds == []

    def test_prefers_passing_design_for_coloring(self):
        """When one design passes and one fails, the passing design is used."""
        pipeline = MPNNESMFoldPipeline()
        designs = [
            # Rank 1 — fails
            {
                "rank":           1,
                "is_wildtype":    False,
                "mutations":      ["A5B"],
                "plddt":          {5: 30.0},
                "pass_threshold": False,
            },
            # Rank 2 — passes
            {
                "rank":           2,
                "is_wildtype":    False,
                "mutations":      ["C8D"],
                "plddt":          {8: 85.0},
                "pass_threshold": True,
            },
        ]
        cmds, _ = pipeline.generate_chimerax_commands(designs, model_id="1", chain_id="A")
        # The passing design (rank 2, mutation C8D at pos 8) should be chosen
        assert any(":8" in c for c in cmds)
        # The failing design (rank 1, mutation A5B at pos 5) should NOT be used
        assert not any(":5" in c for c in cmds)


# ===========================================================================
# 6. generate_summary
# ===========================================================================

class TestGenerateSummary:
    def test_summary_is_multiline(self):
        """Summary must contain '\\n' so main.py renders a Rich Panel."""
        pipeline = MPNNESMFoldPipeline()
        designs  = [
            {
                "rank":           1,
                "is_wildtype":    False,
                "mutations":      ["L10A"],
                "mpnn_score":     -1.2,
                "recovery":       0.9,
                "mean_plddt":     78.0,
                "confidence":     "high",
                "pass_threshold": True,
                "error":          None,
            }
        ]
        summary = pipeline.generate_summary(designs, model_id="1")
        assert "\n" in summary

    def test_summary_contains_model_id(self):
        pipeline = MPNNESMFoldPipeline()
        summary  = pipeline.generate_summary([], model_id="42")
        assert "42" in summary

    def test_summary_shows_pass_count(self):
        pipeline = MPNNESMFoldPipeline()
        designs  = [
            {"rank": 1, "is_wildtype": False, "mutations": [],
             "mpnn_score": 0.0, "recovery": 1.0,
             "mean_plddt": 80.0, "confidence": "high",
             "pass_threshold": True, "error": None},
            {"rank": 2, "is_wildtype": False, "mutations": [],
             "mpnn_score": 0.0, "recovery": 0.8,
             "mean_plddt": 40.0, "confidence": "low",
             "pass_threshold": False, "error": None},
        ]
        summary = pipeline.generate_summary(designs, model_id="1")
        assert "1/2" in summary


# ===========================================================================
# 7. Session state — pdb_str stripped on save
# ===========================================================================

class TestSessionPdbStrStripping:
    def test_pdb_str_stripped_on_save(self, tmp_path):
        """pdb_str must be absent from validated_designs after save+load."""
        session = SessionState()
        result  = {
            "success":     True,
            "model_id":    "1",
            "chain_id":    "A",
            "validated_designs": [
                {
                    "rank":           1,
                    "sequence":       "ACDE",
                    "pdb_str":        "LARGE PDB CONTENT",
                    "mean_plddt":     75.0,
                    "pass_threshold": True,
                    "is_wildtype":    False,
                }
            ],
            "chimerax_commands":     [],
            "chimerax_explanations": [],
            "summary":               "test",
        }
        session.set_mpnn_esmfold_results("1", result)

        save_path = str(tmp_path / "test_session.json")
        session.save(save_path)

        loaded = SessionState.load(save_path)
        lr = loaded.get_mpnn_esmfold_results("1")
        assert lr is not None
        designs = lr.get("validated_designs", [])
        assert len(designs) == 1
        assert "pdb_str" not in designs[0], (
            "pdb_str should be stripped when saving to disk"
        )

    def test_pdb_str_present_in_memory(self):
        """pdb_str is NOT stripped from the in-memory copy."""
        session = SessionState()
        result  = {
            "success":    True,
            "model_id":   "1",
            "validated_designs": [
                {"rank": 1, "pdb_str": "IN MEMORY PDB", "is_wildtype": False}
            ],
        }
        session.set_mpnn_esmfold_results("1", result)
        lr = session.get_mpnn_esmfold_results("1")
        assert lr["validated_designs"][0]["pdb_str"] == "IN MEMORY PDB"
