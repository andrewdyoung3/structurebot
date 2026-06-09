"""
tests/test_mutation_scan_tiering.py
-----------------------------------
Mutation-scan triage/validate refactor (Priority 1): assignable scope + opt-in
Rosetta (fast default tier) + tier-choice surface.  All offline — the Rosetta /
WSL2 call is mocked or simply asserted NOT called.

Groups
------
1. Scope (route-level)      — explicit range / list / single / live selection
2. Scope (scanner-level)    — include_positions restricts scanned positions
3. Default tier             — bare scan = CamSol+ESM, Rosetta NEVER called, ddG None
4. Tier triggers            — rosetta/rosie → deep; bare "rose" → not deep
5. Tier-choice surface      — thoroughness phrase raises a choice, no auto-deep
6. Estimate                 — deep tier emits an n-scaled pre-launch estimate
7. Weights                  — effective_weights renormalises when ddG dropped
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter, ToolStepResult
from mutation_scanner import MutationScanner, effective_weights, combined_score
from rosetta_bridge import resolve_rosetta_workers, mutation_seed
from session_state import SessionState


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_router() -> ToolRouter:
    bridge  = MagicMock()
    session = MagicMock()
    session.structures = {"1": {"name": "2hhb", "path": None}}
    session.get_structure.return_value = {"name": "2hhb", "path": None}
    return ToolRouter(bridge=bridge, session=session)


def _ms_stub(chain: str = "A") -> Dict[str, Any]:
    """A translator result already routed to mutation_scan."""
    return {
        "commands": [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "high",
        "tools_needed": ["mutation_scan"],
        "tool_inputs": {"mutation_scan": {"model_id": "1", "chain": chain,
                                          "focus": "solubility"}},
    }


def _route(router: ToolRouter, prompt: str, *, seq_len: int = 141,
           pose_res: int = 141) -> Dict[str, Any]:
    # _fetch_sequence → chain length (n_pos); _pose_residue_count → full pose size
    # (drives per-mutation cost). Both mocked so route tests need no network/WSL.
    with patch.object(router, "_fetch_sequence", return_value="A" * seq_len), \
         patch.object(router, "_pose_residue_count", return_value=pose_res):
        return router.route(_ms_stub(), user_input=prompt)


def _scanner_with_scores(seq: str) -> MutationScanner:
    session = SessionState()
    session.add_structure("1", "TEST", metadata={"sequences": {"A": seq}})
    n = len(seq)
    session.add_tool_result("camsol", "1", {
        "scores": {str(i): -1.0 for i in range(1, n + 1)},
        "aggregation_hot_spots": list(range(1, n + 1)),
    })
    session.add_tool_result("esm", "1", {
        "conservation": {str(i): 0.1 for i in range(1, n + 1)},
        "mean_conservation": 0.1,
    })
    return MutationScanner(session=session, model_id="1")


def _fake_pdb() -> str:
    import tempfile
    f = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    f.write("HEADER    TEST\nEND\n")
    f.close()
    return f.name


# ── 1. Scope — route-level ─────────────────────────────────────────────────────

class TestScopeRouting:

    def test_explicit_range(self):
        r = _route(_make_router(), "scan mutations on residues 30-45 of chain A")
        pos = r["tool_inputs"]["mutation_scan"]["scan_positions"]
        assert pos == list(range(30, 46)), pos
        assert len(pos) == 16

    def test_range_with_to(self):
        r = _route(_make_router(), "scan residues 30 to 45 of chain A")
        assert r["tool_inputs"]["mutation_scan"]["scan_positions"] == list(range(30, 46))

    def test_explicit_list(self):
        r = _route(_make_router(), "scan residues 30, 32 and 40")
        assert r["tool_inputs"]["mutation_scan"]["scan_positions"] == [30, 32, 40]

    def test_single_residue(self):
        r = _route(_make_router(), "scan residue 88 for solubility")
        assert r["tool_inputs"]["mutation_scan"]["scan_positions"] == [88]

    def test_no_scope_is_whole_chain(self):
        r = _route(_make_router(), "scan chain A for solubility")
        assert "scan_positions" not in r["tool_inputs"]["mutation_scan"]

    def test_live_selection_scope(self):
        router = _make_router()
        with patch.object(router, "_read_selected_residues", return_value=[5, 6, 7]), \
             patch.object(router, "_fetch_sequence", return_value="A" * 141):
            r = router.route(_ms_stub(), user_input="scan the selected residues")
        assert r["tool_inputs"]["mutation_scan"]["scan_positions"] == [5, 6, 7]

    def test_empty_selection_scope_resolves_empty(self):
        router = _make_router()
        with patch.object(router, "_read_selected_residues", return_value=[]), \
             patch.object(router, "_fetch_sequence", return_value="A" * 141):
            r = router.route(_ms_stub(), user_input="scan the selected residues")
        # scope requested but empty → [] (the handler errors, no full-chain fallback)
        assert r["tool_inputs"]["mutation_scan"]["scan_positions"] == []


# ── 2. Scope — scanner-level ───────────────────────────────────────────────────

class TestScopeScanner:

    def test_include_positions_restricts(self):
        scanner = _scanner_with_scores("ACDEFGHIKLMNPQR")
        pdb = _fake_pdb()
        with patch.object(scanner, "_run_rosetta_batch") as mock_ros:
            results = scanner.scan(
                pdb_path=pdb, chain_id="A", sequence="ACDEFGHIKLMNPQR",
                include_positions=[3, 4, 5], run_rosetta=False,
            )
        positions = {r["position"] for r in results}
        assert positions <= {3, 4, 5}, positions
        assert positions, "scoped scan should produce candidates within scope"
        mock_ros.assert_not_called()

    def test_scope_bypasses_threshold_prefilter(self):
        # Strict thresholds would exclude everything in a whole-chain scan, but an
        # explicit scope scans the named positions regardless.
        scanner = _scanner_with_scores("ACDEFGHIKLMNPQR")
        pdb = _fake_pdb()
        with patch.object(scanner, "_run_rosetta_batch"):
            results = scanner.scan(
                pdb_path=pdb, chain_id="A", sequence="ACDEFGHIKLMNPQR",
                filters={"camsol_threshold": -99.0, "esm_threshold": -99.0},
                include_positions=[4, 5], run_rosetta=False,
            )
        assert {r["position"] for r in results} <= {4, 5}
        assert results, "scope must bypass the CamSol/ESM pre-filter"


# ── 3. Default (fast) tier ─────────────────────────────────────────────────────

class TestDefaultTier:

    def test_route_default_is_no_rosetta(self):
        r = _route(_make_router(), "scan chain A for solubility")
        assert r["tool_inputs"]["mutation_scan"]["run_rosetta"] is False

    def test_scanner_default_tier_skips_rosetta(self):
        scanner = _scanner_with_scores("ACDEFGHIKLM")
        pdb = _fake_pdb()
        with patch.object(scanner, "_run_rosetta_batch") as mock_ros:
            results = scanner.scan(
                pdb_path=pdb, chain_id="A", sequence="ACDEFGHIKLM",
                run_rosetta=False,
            )
        mock_ros.assert_not_called()
        assert results
        for r in results:
            assert r["ddg"] is None, "fast tier must report ddG as None, never 0.0"
            assert r["ddg_source"] == "not_computed"

    def test_default_tier_score_is_renormalised(self):
        scanner = _scanner_with_scores("ACDEFGHIKLM")
        pdb = _fake_pdb()
        with patch.object(scanner, "_run_rosetta_batch"):
            results = scanner.scan(
                pdb_path=pdb, chain_id="A", sequence="ACDEFGHIKLM",
                run_rosetta=False,
            )
        r = results[0]
        expected = combined_score(0.0, r["solubility_delta"], r["esm_tolerance"],
                                  0.0, 0.6, 0.4)
        assert r["combined_score"] == expected


# ── 4. Tier triggers ───────────────────────────────────────────────────────────

class TestTierTriggers:

    def test_rosetta_word_triggers_deep(self):
        r = _route(_make_router(), "rosetta scan chain A")
        assert r["tool_inputs"]["mutation_scan"]["run_rosetta"] is True

    def test_rosie_triggers_deep(self):
        r = _route(_make_router(), "scan chain A with rosie")
        assert r["tool_inputs"]["mutation_scan"]["run_rosetta"] is True

    def test_bare_rose_does_not_trigger_deep(self):
        r = _route(_make_router(), "the b-factor rose, scan chain A")
        assert r["tool_inputs"]["mutation_scan"]["run_rosetta"] is False
        assert not r.get("_tier_choice")


# ── 5. Tier-choice surface ─────────────────────────────────────────────────────

class TestTierChoiceSurface:

    def test_comprehensive_raises_choice_no_autodeep(self):
        r = _route(_make_router(), "comprehensive scan of chain A")
        assert r.get("_tier_choice") is True
        assert r["tool_inputs"]["mutation_scan"]["run_rosetta"] is False
        q = r.get("clarification_needed") or ""
        assert "Base" in q and "Deep" in q

    def test_exhaustive_raises_choice(self):
        r = _route(_make_router(), "exhaustive scan of chain A")
        assert r.get("_tier_choice") is True
        assert r["tool_inputs"]["mutation_scan"]["run_rosetta"] is False

    def test_explicit_rosetta_overrides_thoroughness(self):
        # "comprehensive ... with rosetta" → deep directly, no tier choice
        r = _route(_make_router(), "comprehensive scan of chain A with rosetta")
        assert r["tool_inputs"]["mutation_scan"]["run_rosetta"] is True
        assert not r.get("_tier_choice")


# ── 6. Estimate ────────────────────────────────────────────────────────────────

class TestEstimate:

    def test_deep_scoped_emits_estimate(self):
        # 16 positions × 3 subs = 48 mutations, pose 574 res
        r = _route(_make_router(), "rosetta scan on residues 30-45 of chain A",
                   pose_res=574)
        warns = " ".join(r.get("warnings", []))
        assert "approximate runtime" in warns.lower()
        assert "48 mutation" in warns          # count, not 16 positions
        assert "16 position" in warns
        assert "574-residue" in warns

    def test_counts_candidates_per_pos_not_positions(self):
        r = _route(_make_router(), "rosetta scan on residues 30-45 of chain A",
                   pose_res=574)
        warns = " ".join(r.get("warnings", []))
        assert "48 mutation" in warns and "16 mutation" not in warns

    def test_REGRESSION_2hhb_scale_is_tens_of_minutes_not_50s(self):
        # The merge-blocker: ~48 mutations on the 574-res tetramer at the RESOLVED
        # (footprint-capped) worker count must NOT estimate ~50 s.  Lands ~2 h.
        router = _make_router()
        workers = router._resolve_deep_workers(574)          # = 6 (mem-capped)
        secs = self._secs(router, n_mut=48, n_res=574, workers=workers)
        assert secs > 1800, f"estimate {secs:.0f}s still undershoots (was ~50s)"
        assert secs < 6 * 3600, f"estimate {secs:.0f}s implausibly high"

    def test_size_scaling_present_large_vs_small(self):
        router = _make_router()
        small = self._secs(router, n_mut=20, n_res=46)
        large = self._secs(router, n_mut=20, n_res=574)
        assert large > 20 * small, "per-mutation cost must scale steeply with pose size"

    def test_no_undershoot_at_anchors(self):
        # estimate must not undershoot either measured anchor (per mutation, solo)
        from rosetta_bridge import per_mutation_sec
        assert per_mutation_sec(46) >= 10           # 1CRN anchor
        assert per_mutation_sec(574) >= 733          # 2HHB lower bound (biased high)

    @staticmethod
    def _secs(router, *, n_mut, n_res, workers=1):
        """Parse the human duration string back to seconds (workers=1 to isolate)."""
        s = router._estimate_rosetta_runtime(n_mut, n_res, workers)
        assert s is not None
        v = float(s.strip("~ ").split()[0]); unit = s.split()[-1]
        return v * {"s": 1, "min": 60, "h": 3600}[unit]


# ── 7. effective_weights ───────────────────────────────────────────────────────

class TestEffectiveWeights:

    def test_with_ddg_unchanged(self):
        assert effective_weights(True) == (0.50, 0.30, 0.20)

    def test_without_ddg_renormalised(self):
        assert effective_weights(False) == (0.0, 0.6, 0.4)

    def test_renorm_sums_to_one(self):
        _, w_sol, w_tol = effective_weights(False)
        assert round(w_sol + w_tol, 6) == 1.0


# ── 8. FIX D — deep-tier parallelization ───────────────────────────────────────

class TestWorkerCap:

    def test_configured_is_binding_when_smallest(self):
        assert resolve_rosetta_workers(8, 24, 12000, 1200) == 8

    def test_cpu_headroom_cap(self):
        # configured huge, mem ample → capped at physical_cores - 2
        assert resolve_rosetta_workers(64, 24, 999999, 1200) == 22

    def test_mem_budget_cap(self):
        # 4000 MB / 1200 MB ≈ 3 workers
        assert resolve_rosetta_workers(32, 24, 4000, 1200) == 3

    def test_never_below_one(self):
        assert resolve_rosetta_workers(0, 1, 0, 0) == 1

    def test_never_uses_all_logical_threads(self):
        # the i9-14900HX hazard: 32 logical threads must NOT be the answer
        w = resolve_rosetta_workers(32, 24, 12000, 1200)
        assert w < 32 and w <= 22


class TestDeterministicSeeding:
    """The identical-results contract: seeds depend ONLY on (base, key, traj),
    never on worker id or scheduling order → parallel output == serial output."""

    def test_seed_is_deterministic(self):
        assert mutation_seed(1, "V82A") == mutation_seed(1, "V82A")

    def test_distinct_keys_differ(self):
        assert mutation_seed(1, "V82A") != mutation_seed(1, "I72R")

    def test_distinct_trajectories_differ(self):
        assert mutation_seed(1, "V82A", 0) != mutation_seed(1, "V82A", 1)

    def test_base_seed_changes_result(self):
        assert mutation_seed(1, "V82A") != mutation_seed(2, "V82A")

    def test_order_independent_mapping(self):
        muts = ["V82A", "I72R", "G88V", "T26A"]
        forward  = {k: mutation_seed(1, k) for k in muts}
        backward = {k: mutation_seed(1, k) for k in reversed(muts)}
        assert forward == backward, "per-mutation seed must not depend on order"

    def test_in_int32_range(self):
        for k in ("V82A", "I72R", "G88V"):
            s = mutation_seed(1, k)
            assert 0 <= s < 2147483647


class TestWorkerParallelStructure:
    """The real pool runs inside the WSL f-string worker (not unit-runnable in CI);
    guard that the parallel + deterministic-seed constructs are present + threaded."""

    def _src(self) -> str:
        import rosetta_bridge
        return Path(rosetta_bridge.__file__).read_text(encoding="utf-8")

    def test_worker_uses_multiprocessing_fork_pool(self):
        src = self._src()
        assert "import multiprocessing" in src
        assert 'get_context("fork")' in src
        assert "_ctx.Pool(" in src

    def test_worker_threads_num_workers_and_base_seed(self):
        src = self._src()
        assert "_n_workers = {num_workers}" in src
        assert "_base_seed = {base_seed}" in src

    def test_worker_has_deterministic_seed_fn(self):
        src = self._src()
        assert "def _seed_for(" in src
        assert "sha256" in src

    def test_worker_has_serial_fallback(self):
        src = self._src()
        assert "serial fallback" in src.lower()


class TestEstimateFoldsWorkers:

    def test_estimate_divides_by_workers(self):
        router = _make_router()
        from rosetta_bridge import per_mutation_sec
        per = per_mutation_sec(574)
        e8 = router._estimate_rosetta_runtime(48, 574, 8)
        e1 = router._estimate_rosetta_runtime(48, 574, 1)
        assert e8 != e1
        assert e1 == router._format_duration(48 * per / 1)
        assert e8 == router._format_duration(48 * per / 8)

    def test_estimate_none_when_no_mutations(self):
        assert _make_router()._estimate_rosetta_runtime(None, 574, 8) is None


class TestWorkerCapPoseSize:
    """FIX 2 — the worker cap now scales with the ACTUAL pose (swap prevention)."""

    def test_footprint_scales_with_residues(self):
        from rosetta_bridge import worker_footprint_mb
        assert worker_footprint_mb(574) > worker_footprint_mb(46)

    def test_large_pose_drives_workers_below_8(self):
        from rosetta_bridge import resolve_rosetta_workers, worker_footprint_mb
        # 574-res tetramer: ~1763 MB/worker → 12000//1763 = 6 (not the lucky 8)
        w = resolve_rosetta_workers(8, 24, 12000, worker_footprint_mb(574))
        assert w == 6, w

    def test_huge_pose_shrinks_further(self):
        from rosetta_bridge import resolve_rosetta_workers, worker_footprint_mb
        w = resolve_rosetta_workers(8, 24, 12000, worker_footprint_mb(1500))
        assert w < 6

    def test_small_pose_keeps_full_pool(self):
        from rosetta_bridge import resolve_rosetta_workers, worker_footprint_mb
        assert resolve_rosetta_workers(8, 24, 12000, worker_footprint_mb(46)) == 8

    def test_footprint_fallback_when_size_unknown(self):
        from rosetta_bridge import worker_footprint_mb
        import config
        assert worker_footprint_mb(None) == config.ROSETTA_WORKER_FOOTPRINT_MB
