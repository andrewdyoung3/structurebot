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
    def test_floor_gated_disruption(self):
        # FLOOR-GATING (migrated to dRMSD): a block of residues moved 2 Å clears the global-min
        # floor and is flagged disrupted; the rest stay below it (not everything paints).
        ref = _rigid(30)
        var = _displace(ref, {10, 11, 12}, [2.0, 0.0, 0.0])   # 3 residues move ~2 Å
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor_ddm": {}, "floor_lddt": {}, "path": "/tmp/r.pdb"},
        })
        assert out.success
        d = out.data
        # the movers clear the global-min dRMSD floor (0.5 Å); gating leaves the rest below it
        assert d["ddm"]["10"] > 0.5 and d["ddm"]["11"] > 0.5 and d["ddm"]["12"] > 0.5
        assert 3 <= d["n_disrupted"] < d["n_residues"]    # gated → not everything painted
        assert d["floor_kind"] == "deterministic"
        assert "deviation" not in d and "anchor_residual_rmsd" not in d   # old path stripped

    def test_per_residue_floor_suppresses_real_motion(self):
        # FLOOR-GATING: a residue whose dRMSD beats the global min (0.5) but is ≤ its MEASURED
        # cross-seed floor is suppressed — real motion declared "within WT noise."
        ref = _rigid(30)
        var = _displace(ref, {5}, [2.0, 0.0, 0.0])            # res 5 moves 2 Å (dRMSD ~1 Å)
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "boltz", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor_ddm": {"5": 3.0}, "floor_lddt": {},
                       "path": "/tmp/r.cif"}})
        assert out.success
        d = out.data
        assert d["ddm"]["5"] > 0.5            # WOULD clear the global-min floor …
        assert d["n_disrupted"] == 0          # … but ≤ its 3.0 measured floor → suppressed
        assert d["floor_kind"] == "measured"

    def test_fold_column_map_identity_is_byte_identical(self):
        # additive guarantee: an identity map (substitution-only) == no map (same ddm + lddt).
        ref = _rigid(30)
        var = _displace(ref, {10, 11, 12}, [2.0, 0.0, 0.0])
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        base = {"variant_model_id": "9", "engine": "esmfold", "target": "monomer",
                "variant_chain": "A", "multichain": False,
                "wt_ref": {"model_id": "7", "floor_ddm": {}, "floor_lddt": {}, "path": "/tmp/r.pdb"}}
        out_nomap = r._run_variant_deviation(dict(base))
        out_id = r._run_variant_deviation({**base, "fold_column_map": {i: i for i in range(1, 31)}})
        assert out_nomap.data["ddm"] == out_id.data["ddm"]
        assert out_nomap.data["lddt"] == out_id.data["lddt"]

    def test_fold_column_map_pairs_post_deletion_residues(self):
        # LOAD-BEARING (column pairing): a deletion at template pos 15 → variant fold has 29
        # residues numbered 1..29; residues AFTER the deletion must pair to template pos+1, NOT
        # pos (the resnum==resnum mis-pair). Displace variant-fold residue 20 (→ ref 21): its
        # dRMSD must land at REFERENCE resnum 21, the deleted pos 15 is absent, and the mis-pair
        # resnum 20 must NOT carry the signal.
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
            "wt_ref": {"model_id": "7", "floor_ddm": {}, "floor_lddt": {}, "path": "/tmp/r.pdb"},
            "fold_column_map": fold_map})
        assert out.success
        d = out.data
        assert d["ddm"]["21"] == max(d["ddm"].values())         # the signal lands at REFERENCE 21
        assert d["ddm"]["21"] > d["ddm"].get("20", 0.0)         # NOT the one-off mis-pair resnum
        assert "15" not in d["ddm"]                              # the deleted position is absent

    def test_fold_column_map_drops_inserted_residues(self):
        # LOAD-BEARING (column pairing + insertion): a variant-fold residue ABSENT from the map is
        # an INSERTED residue (no WT counterpart — omitted by build_fold_column_map). It is
        # DROPPED (never appears at any ref resnum), while the shared residues still pair
        # correctly. fold res 16 is the insertion (way off-axis); shared res 20 maps to ref 19.
        ref = _rigid(30)
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
            "wt_ref": {"model_id": "7", "floor_ddm": {}, "floor_lddt": {}, "path": "/tmp/r.pdb"},
            "fold_column_map": fold_map})                        # omits fold resnum 16 (the insert)
        assert out.success
        d = out.data
        assert d["ddm"]["19"] == max(d["ddm"].values())         # shared residue pairs at ref 19
        # the inserted residue (50 Å off-axis) was DROPPED → no entry carries that huge signal
        assert max(d["ddm"].values()) < 5.0
        assert len(d["ddm"]) == 29                              # 30 ref − the unmapped insert pos
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

    def test_returns_ddm_painted_signal_and_lddt_secondary(self):
        # the tool paints superposition-free dRMSD and ALSO reports lDDT (local integrity).
        ref = _rigid(30)
        var = _displace(ref, {10, 11, 12}, [6.0, 0.0, 0.0])   # big local shove
        r = _router()
        r._read_fold_ca = MagicMock(side_effect=lambda mid, mc, ch: ref if mid == "7" else var)
        out = r._run_variant_deviation({
            "variant_model_id": "9", "engine": "esmfold", "target": "monomer",
            "variant_chain": "A", "multichain": False,
            "wt_ref": {"model_id": "7", "floor": {}, "floor_lddt": {}, "floor_ddm": {},
                       "path": "/tmp/r.pdb"}})
        assert out.success
        d = out.data
        assert "ddm" in d and "floor_ddm" in d and "lddt" in d
        assert d["max_ddm"] > 0.0 and d["min_lddt"] < 1.0
        assert d["n_disrupted"] >= 1                          # shoved residues above the dRMSD floor

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
        # deterministic → no cross-seed motion → dRMSD floor at the global min everywhere
        assert set(wt["floor_ddm"].values()) == {r._DDM_FLOOR_MIN_A}
        assert len(wt["floor_ddm"]) == 20
        # lDDT floor (no extra seeds) → the neutral cap everywhere
        assert set(wt["floor_lddt"].values()) == {r._LDDT_NEUTRAL_CAP}
        assert len(wt["floor_lddt"]) == 20
        assert "floor" not in wt                  # the legacy displacement floor is stripped

    def test_boltz_floor_is_cross_seed_variation(self):
        # the MEASURED floors come from the extra WT seed folds (superposition-free): dRMSD floor
        # = cross-seed MAX dRMSD (≥ global min); lDDT floor = cross-seed MIN lDDT (≤ cap). Residues
        # that move across seeds are elevated above the global min; an unmoved one sits at it.
        ref = _rigid(30)
        f1 = _displace(ref, {5}, [3.0, 0, 0])     # seed 1: res5 moves 3 Å
        f2 = _displace(ref, {5, 8}, [4.0, 0, 0])  # seed 2: res5 4 Å, res8 4 Å
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
        fd = wt["floor_ddm"]
        assert fd["5"] > r._DDM_FLOOR_MIN_A       # res 5 moved across seeds → elevated dRMSD floor
        assert fd["8"] > r._DDM_FLOOR_MIN_A       # res 8 moved in seed 2 → elevated
        assert fd["5"] >= fd["8"]                 # cross-seed MAX: res5 (up to 4 Å) ≥ res8 (4 Å once)
        # lDDT floor = min cross-seed lDDT, capped; an unmoved residue stays locally consistent
        fll = wt["floor_lddt"]
        assert fll["1"] == r._LDDT_NEUTRAL_CAP    # unmoved → the cap (no local variation)
        assert all(v <= r._LDDT_NEUTRAL_CAP for v in fll.values())
        assert "floor" not in wt                  # legacy displacement floor stripped


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


class TestPerResidueDdm:
    """The PAINTED per-residue dRMSD (all-pairs distance-RMSD) — captures rigid DISPLACEMENT of
    intact structure (what lDDT misses), zero for a whole-body rigid move (no false signal)."""

    def test_identical_is_zero(self):
        coords = _rigid(20)
        m = ToolRouter._per_residue_ddm(coords, coords, sorted(coords))
        assert m and all(abs(v) < 1e-9 for v in m.values())

    def test_whole_body_rigid_move_is_zero(self):
        # the key advantage over Kabsch-with-bad-anchor: a global rotation+translation preserves
        # EVERY pairwise distance → dRMSD 0 everywhere (no lever-arm artifact).
        ref = _rigid(30)
        th = 0.9
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th),  np.cos(th), 0.0],
                      [0.0, 0.0, 1.0]])
        t = np.array([40.0, 10.0, -20.0])
        moved = {k: R @ v + t for k, v in ref.items()}
        m = ToolRouter._per_residue_ddm(ref, moved, sorted(ref))
        assert all(abs(v) < 1e-6 for v in m.values())

    def test_rigid_displacement_of_a_block_lights_up(self):
        # THE user's case: a contiguous block moved rigidly keeps its INTERNAL distances (lDDT
        # stays ~1, reads white) but its distances to the REST change → dRMSD rises. Deterministic
        # extended-chain geometry (3.8 Å spacing along x) so the contrast is exact.
        ref = {i: np.array([i * 3.8, 0.0, 0.0]) for i in range(1, 41)}
        block = set(range(1, 11))                          # residues 1..10 move together (+12 x)
        var = {k: (v + np.array([12.0, 0.0, 0.0]) if k in block else v.copy())
               for k, v in ref.items()}
        ddm = ToolRouter._per_residue_ddm(ref, var, sorted(ref))
        lddt = ToolRouter._per_residue_lddt(ref, var, sorted(ref))
        # block-interior residue 5: all its in-radius neighbours are in the block (moved with it)
        # → lDDT ≈ 1.0 (white under lDDT), yet its distances to residues 11..40 changed → dRMSD
        # is large. The metric the user wants lights up exactly where lDDT could not.
        assert lddt[5] > 0.99
        assert ddm[5] > 5.0
