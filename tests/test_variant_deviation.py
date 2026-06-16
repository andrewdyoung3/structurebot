"""
tests/test_variant_deviation.py
-------------------------------
S4c: the variant-vs-WT per-residue Cα deviation tool + the seed-pinned WT reference
fold / per-residue noise-floor establishment. All mocked — no live ChimeraX, no real
folds in CI.

A. _run_variant_deviation  — reuses the EXISTING Kabsch (auto-anchor + _anchor_kabsch):
   clean anchor residual ≈ 0, per-residue deviation, floor-gated clear counting.
B. _fold_wt_reference      — deterministic engine → global-min floor; stochastic engine →
   cross-seed MAX displacement + global minimum (conservative at small N).
C. Error-first             — missing variant model, too few common Cα.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter


def _router() -> ToolRouter:
    return ToolRouter(bridge=MagicMock(), session=MagicMock())


def _rigid(n: int = 30) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(7)
    return {i + 1: rng.standard_normal(3) * 10 for i in range(n)}


def _displace(coords, resnums, delta) -> Dict[int, np.ndarray]:
    d = np.array(delta, dtype=float)
    return {rn: (c + d if rn in resnums else c.copy()) for rn, c in coords.items()}


class TestRunVariantDeviation:
    def test_clean_fit_and_floor_gated_clears(self):
        ref = _rigid(30)
        var = _displace(ref, {10, 11, 12}, [2.0, 0.0, 0.0])   # only 3 residues move ~2 Å
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor": {}, "path": "/tmp/r.pdb"},
        })
        assert out.success
        d = out.data
        # anchor pruned the 3 movers → the rigid core fits with ≈0 residual (clean readback)
        assert d["anchor_residual_rmsd"] < 0.05
        # only the 3 displaced residues clear the (global-min 0.25 Å) floor
        assert d["n_cleared_floor"] == 3
        assert d["deviation"]["10"] > 1.5 and d["deviation"]["1"] < 0.1
        assert abs(d["max_deviation"] - 2.0) < 0.1
        assert d["floor_kind"] == "deterministic"

    def test_per_residue_floor_suppresses_subfloor_noise(self):
        ref = _rigid(30)
        var = _displace(ref, {5}, [0.4, 0.0, 0.0])            # res 5 moves 0.4 Å
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        # measured floor at res 5 is 0.6 Å → its 0.4 Å shift is BELOW floor → not cleared
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "boltz", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor": {"5": 0.6}, "path": "/tmp/r.cif"},
        })
        assert out.success
        assert out.data["n_cleared_floor"] == 0          # 0.4 < 0.6 floor → suppressed
        assert out.data["floor_kind"] == "measured"

    def test_missing_variant_model_errors(self):
        out = _router()._run_variant_deviation({"engine": "esmfold"})
        assert not out.success and "model id" in out.error

    def test_too_few_common_ca_errors(self):
        r = _router()
        ref = {1: np.zeros(3), 2: np.ones(3)}
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else {1: np.zeros(3)})
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "wt_ref": {"model_id": "7", "floor": {}}})
        assert not out.success and "common" in out.error


class TestFoldWtReference:
    def test_esmfold_deterministic_floor_is_global_min(self):
        ref = _rigid(20)
        r = _router()
        r._get_esmfold_bridge = MagicMock(return_value=MagicMock(
            predict=MagicMock(return_value={"success": True, "pdb_str": "PDB"})))
        r._fold_viz_commands = MagicMock(return_value=(["open x"], ["e"], "5"))
        r._read_fold_ca = MagicMock(return_value=ref)
        wt = r._fold_wt_reference({
            "engine": "esmfold", "target": "monomer", "multichain": False,
            "variant_chain": "A", "wt_chains": [{"id": "A", "sequence": "MKV"}],
            "model_id": "1"})
        assert wt is not None
        assert wt["n_floor_seeds"] == 1
        # deterministic → no cross-seed motion → every residue floored at the global minimum
        assert set(wt["floor"].values()) == {r._DEVIATION_FLOOR_MIN_A}
        assert len(wt["floor"]) == 20

    def test_boltz_floor_is_cross_seed_max_plus_global_min(self):
        ref = _rigid(30)                          # ≥~25 res → the anchor prune isolates movers
        f1 = _displace(ref, {5}, [1.0, 0, 0])     # seed 1: res5 moves 1.0 Å
        f2 = _displace(ref, {5, 8}, [1.5, 0, 0])  # seed 2: res5 1.5 Å, res8 1.5 Å
        f3 = dict(ref)                            # seed 3: identical
        reads = [ref, f1, f2, f3]                 # ref read, then 3 floor-fold reads
        r = _router()
        boltz = MagicMock(predict=MagicMock(return_value={"success": True,
                                                          "cif_path": "/tmp/x.cif", "seed": 0}))
        r._get_boltz_bridge = MagicMock(return_value=boltz)
        r._fold_viz_commands = MagicMock(return_value=(["open x"], ["e"], "5"))
        r.bridge.run_command = MagicMock(return_value={"value": "#6"})   # floor folds open as #6
        r._read_fold_ca = MagicMock(side_effect=reads)
        wt = r._fold_wt_reference({
            "engine": "boltz", "target": "monomer", "multichain": False,
            "variant_chain": "A", "wt_chains": [{"id": "A", "sequence": "MKV"}],
            "model_id": "1"})   # seeds default to [0,1,2,3] → 3 floor folds
        assert wt is not None and wt["n_floor_seeds"] == 4
        fl = wt["floor"]
        assert abs(fl["5"] - 1.5) < 0.1          # cross-seed MAX (1.5 > 1.0) at res 5
        assert abs(fl["8"] - 1.5) < 0.1          # res 8 moved 1.5 Å in seed 2
        assert abs(fl["1"] - r._DEVIATION_FLOOR_MIN_A) < 1e-9   # unmoved → global min
