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

    def test_fold_column_map_identity_is_byte_identical(self):
        # additive guarantee: an identity map (substitution-only) == no map.
        ref = _rigid(30)
        var = _displace(ref, {10, 11, 12}, [2.0, 0.0, 0.0])
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        base = {"variant_model_id": "9", "engine": "esmfold", "target": "monomer",
                "variant_chain": "A", "multichain": False,
                "wt_ref": {"model_id": "7", "floor": {}, "path": "/tmp/r.pdb"}}
        out_nomap = r._run_variant_deviation(dict(base))
        out_id = r._run_variant_deviation({**base, "fold_column_map": {i: i for i in range(1, 31)}})
        assert out_nomap.data["deviation"] == out_id.data["deviation"]
        assert out_nomap.data["max_deviation"] == out_id.data["max_deviation"]

    def test_fold_column_map_pairs_post_deletion_residues(self):
        # the WHOLE point of the map: a deletion at template pos 15 → variant fold has 29
        # residues numbered 1..29; residues AFTER the deletion must pair to template pos+1,
        # NOT pos (the resnum==resnum mis-pair). Displace variant-fold residue 20 (→ ref 21).
        ref = _rigid(30)
        fold_map = {j: j for j in range(1, 15)}
        fold_map.update({j: j + 1 for j in range(15, 30)})       # 15->16 … 29->30
        var = {j: ref[fold_map[j]].copy() for j in range(1, 30)}  # overlay each on its ref pair
        var[20] = var[20] + np.array([2.0, 0.0, 0.0])            # displace fold res 20 (→ ref 21)
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor": {}, "path": "/tmp/r.pdb"},
            "fold_column_map": fold_map})
        assert out.success
        d = out.data
        assert d["anchor_residual_rmsd"] < 0.05                  # clean rigid-core fit
        assert d["deviation"]["21"] > 1.5                        # pairs at REFERENCE resnum 21
        assert d["deviation"].get("20", 0.0) < 0.1              # not the one-off-shifted bug
        assert "15" not in d["deviation"]                        # the deleted position is absent

    def test_fold_column_map_drops_inserted_residues(self):
        # Stage B: a variant-fold residue ABSENT from the map is an INSERTED residue (no WT
        # counterpart — build_fold_column_map omits it by design). It is DROPPED from the
        # deviation (excluded, rendered neutral), NOT failed-loud; the shared residues still
        # pair correctly. Here fold res 16 is the insertion: cols 1..15 are identity, then the
        # insertion at 16, then 17..30 map to ref 16..29.
        ref = _rigid(30)                                        # WT reference: 29 residues used
        fold_map = {j: j for j in range(1, 16)}                 # 1..15 identity
        fold_map.update({j: j - 1 for j in range(17, 31)})      # 17->16 … 30->29  (16 = insertion)
        var = {j: ref[fold_map[j]].copy() for j in fold_map}    # overlay shared on their ref pair
        var[16] = ref[1] + np.array([50.0, 0.0, 0.0])          # the inserted residue, way off-axis
        var[20] = var[20] + np.array([2.0, 0.0, 0.0])          # displace a SHARED residue (→ ref 19)
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor": {}, "path": "/tmp/r.pdb"},
            "fold_column_map": fold_map})                        # omits fold resnum 16 (the insert)
        assert out.success
        d = out.data
        assert d["anchor_residual_rmsd"] < 0.05                  # inserted res dropped → clean fit
        assert d["deviation"]["19"] > 1.5                        # shared residue pairs at ref 19
        # the inserted residue (way off-axis) was DROPPED — never appears at any ref resnum
        assert all(dv < 1.0 for k, dv in d["deviation"].items() if k != "19")
        # the APPLIED map is echoed (str-keyed) so the 3D push can invert ref→variant numbering
        assert d["fold_column_map"] == {str(j): r for j, r in fold_map.items()}

    def test_fold_column_map_echo_none_when_substitution_only(self):
        # no map / identity → no remap needed downstream → echo None (the 3D push uses identity).
        ref = _rigid(30)
        var = _displace(ref, {10}, [2.0, 0.0, 0.0])
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor": {}, "path": "/tmp/r.pdb"}})
        assert out.success and out.data["fold_column_map"] is None

    def test_returns_lddt_primary_signal(self):
        # the deviation tool now also computes the SUPERPOSITION-FREE Cα-lDDT (the paint signal).
        ref = _rigid(30)
        var = _displace(ref, {10, 11, 12}, [6.0, 0.0, 0.0])   # big local shove → lDDT drops there
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor": {}, "floor_lddt": {}, "path": "/tmp/r.pdb"}})
        assert out.success
        d = out.data
        assert "lddt" in d and "floor_lddt" in d
        assert d["min_lddt"] < 1.0 and d["mean_lddt"] <= 1.0
        assert d["n_disrupted"] >= 1                          # shoved residues fall below the cap

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
        # lDDT floor (no extra seeds) → the neutral cap everywhere
        assert set(wt["floor_lddt"].values()) == {r._LDDT_NEUTRAL_CAP}
        assert len(wt["floor_lddt"]) == 20

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
        # lDDT floor = min cross-seed lDDT, capped at the neutral cap; an unmoved residue stays
        # locally consistent across seeds → floored at the cap (0.9), never above.
        fll = wt["floor_lddt"]
        assert fll["1"] == r._LDDT_NEUTRAL_CAP
        assert all(v <= r._LDDT_NEUTRAL_CAP for v in fll.values())


class TestPerResidueLddt:
    """The superposition-FREE per-residue Cα-lDDT kernel — invariant to rigid-body motion (the
    whole reason for the metric), drops only where local geometry genuinely changes."""

    def test_identical_is_one(self):
        coords = _rigid(20)
        m = ToolRouter._per_residue_lddt(coords, coords, sorted(coords))
        assert m and all(abs(v - 1.0) < 1e-9 for v in m.values())

    def test_rigid_body_transform_is_invariant(self):
        # THE point: a global rotation + large translation does NOT lower lDDT (no superposition
        # is performed), unlike the Kabsch deviation which a small anchor blows up to tens of Å.
        ref = _rigid(30)
        th = 0.7
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th),  np.cos(th), 0.0],
                      [0.0, 0.0, 1.0]])
        t = np.array([100.0, -50.0, 25.0])
        moved = {k: R @ v + t for k, v in ref.items()}
        m = ToolRouter._per_residue_lddt(ref, moved, sorted(ref))
        assert all(abs(v - 1.0) < 1e-9 for v in m.values())

    def test_local_distortion_drops_only_locally(self):
        ref = _rigid(40)
        var = {k: v.copy() for k, v in ref.items()}
        var[21] = var[21] + np.array([8.0, 0.0, 0.0])    # displace ONE residue
        m = ToolRouter._per_residue_lddt(ref, var, sorted(ref))
        assert m[21] == min(m.values()) and m[21] < 0.9  # the moved residue is the most disrupted
        assert sum(1 for v in m.values() if v > 0.9) >= 30   # the rest stay locally conserved
