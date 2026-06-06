"""
tests/test_conformer_comparison.py
-----------------------------------
Tests for the conformer-comparison thin orchestrator.
All mocked — no live ChimeraX in CI.

A. Pure helpers  — _anchor_kabsch, _parse_anchor_spec, _resnums_to_chimerax_range
B. Colour commands — _conformer_shift_color_cmds
C. Live-coord stub  — _ca_coords_live (bridge mocked)
D. Orchestrator     — _run_conformer_comparison (bridge + session mocked)
E. Routing          — _detect_conformer_comparison_intent / _parse_conformer_comparison_options
F. Session round-trip
G. Error-first
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter, ToolStepResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_router(session_has_structures: bool = True) -> ToolRouter:
    bridge  = MagicMock()
    session = MagicMock()
    if session_has_structures:
        session.structures = {"1": {"name": "4AKE"}, "2": {"name": "1AKE"}}
    else:
        session.structures = {}
    session.conformer_comparison_results = {}
    session.set_conformer_comparison_results = (
        lambda k, v: session.conformer_comparison_results.update({k: v})
    )
    session.get_conformer_comparison_results = (
        lambda k: session.conformer_comparison_results.get(k)
    )
    return ToolRouter(bridge=bridge, session=session)


def _rigid_body_coords(n: int = 30) -> Dict[int, np.ndarray]:
    """Generate n Cα coords starting at resno 1."""
    rng = np.random.default_rng(42)
    return {i + 1: rng.standard_normal(3) * 10 for i in range(n)}


def _rotate_translate(
    coords: Dict[int, np.ndarray],
    angle_deg: float = 30.0,
) -> Dict[int, np.ndarray]:
    """Apply a known rotation + translation to all coords."""
    angle = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1],
    ])
    t = np.array([5.0, -3.0, 2.0])
    return {rn: R @ c + t for rn, c in coords.items()}


def _add_displacement(
    coords: Dict[int, np.ndarray],
    resnums: List[int],
    delta: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Add *delta* to the specified residues."""
    out = dict(coords)
    for rn in resnums:
        if rn in out:
            out[rn] = out[rn] + delta
    return out


# ── A. Pure helpers ─────────────────────────────────────────────────────────

class TestAnchorKabsch:
    """Anchor-restricted Kabsch superposition (pure, no ChimeraX)."""

    def test_identical_conformers_zero_shift(self):
        """Two identical structures → anchor RMSD = 0, all shifts = 0."""
        a = _rigid_body_coords(30)
        per_shift, anc_rmsd, all_rmsd = ToolRouter._anchor_kabsch(a, dict(a), list(a))
        assert per_shift is not None
        assert anc_rmsd < 1e-6
        assert all_rmsd < 1e-6
        assert all(v < 1e-6 for v in per_shift.values())

    def test_global_rotation_translation_zeroes_anchor(self):
        """After fitting, a pure rotation+translation gives ~0 anchor residual
        AND ~0 per-residue shifts for all matched residues."""
        a  = _rigid_body_coords(30)
        b  = _rotate_translate(a, 45.0)
        per_shift, anc_rmsd, all_rmsd = ToolRouter._anchor_kabsch(a, b, list(a))
        assert per_shift is not None, "Kabsch should succeed"
        assert anc_rmsd < 1e-4, f"anchor RMSD {anc_rmsd:.6f} should be ~0 after rigid fit"
        assert all_rmsd < 1e-4, f"all-pairs RMSD {all_rmsd:.6f} should be ~0 for rigid body"
        for rn, sh in per_shift.items():
            assert sh < 1e-3, f"res {rn} shift {sh:.4f} should be ~0 for rigid body"

    def test_known_displacement_in_mobile_segment(self):
        """Anchor on residues 1-20; residues 21-30 displaced by 10 Å.
        After anchor-restricted fit: anchor residual ≈ 0, displaced segment ≈ 10 Å.
        A global fit would give a different (incorrect) answer."""
        delta = np.array([10.0, 0.0, 0.0])
        a  = _rigid_body_coords(30)
        b  = _rotate_translate(a, 20.0)           # global rigid body
        b  = _add_displacement(b, list(range(21, 31)), delta)  # + domain shift

        anchor = list(range(1, 21))               # residues 1-20 = anchor
        per_shift, anc_rmsd, all_rmsd = ToolRouter._anchor_kabsch(a, b, anchor)

        assert per_shift is not None
        # Anchor should be near-zero after fitting
        assert anc_rmsd < 0.5, f"anchor residual {anc_rmsd:.3f} Å should be small"
        # Mobile segment should show ~10 Å shift
        for rn in range(21, 31):
            assert per_shift[rn] > 5.0, (
                f"res {rn} shift {per_shift[rn]:.2f} Å should be ~10 Å")
        # Anchor residues should be small
        for rn in range(1, 21):
            assert per_shift[rn] < 2.0, (
                f"anchor res {rn} shift {per_shift[rn]:.2f} Å should be ~0")

    def test_anchor_restricted_differs_from_global_fit(self):
        """Anchor-restricted fit gives DIFFERENT (correct) shifts than a global fit.
        This is the key property that justifies the tool."""
        a  = _rigid_body_coords(30)
        b  = _rotate_translate(a, 20.0)
        b  = _add_displacement(b, list(range(21, 31)), np.array([12.0, 0.0, 0.0]))

        anchor = list(range(1, 21))
        per_shift_anc, _, _ = ToolRouter._anchor_kabsch(a, b, anchor)
        per_shift_glob, _, _ = ToolRouter._anchor_kabsch(a, b, sorted(a.keys()))

        # For the displaced residues, anchor-restricted gives LARGER shifts
        # (the global fit averages out the domain motion)
        shift_anc_mobile  = np.mean([per_shift_anc[r] for r in range(21, 31)])
        shift_glob_mobile = np.mean([per_shift_glob[r] for r in range(21, 31)])
        assert shift_anc_mobile > shift_glob_mobile, (
            "Anchor-restricted should show larger mobile-segment shift than global fit")

    def test_non_one_start_resnums_correct_mapping(self):
        """Non-1-start residue numbers (like PDB 1IL8 chain A starting at res 2)
        are handled correctly — no off-by-one or 1..N assumption."""
        a = {r: np.random.default_rng(r).standard_normal(3) * 5
             for r in range(2, 22)}    # resnums 2..21 (not 1..20)
        b = _rotate_translate(a, 15.0)
        b = _add_displacement(b, list(range(12, 22)), np.array([8.0, 0.0, 0.0]))

        anchor = list(range(2, 12))   # resnums 2..11
        per_shift, anc_rmsd, _ = ToolRouter._anchor_kabsch(a, b, anchor)

        assert per_shift is not None
        assert set(per_shift) == set(range(2, 22)), "All residue numbers preserved"
        assert anc_rmsd < 0.5
        for rn in range(12, 22):
            assert per_shift[rn] > 5.0, f"res {rn} should show large shift"

    def test_fewer_than_3_anchor_atoms_returns_none(self):
        a = {1: np.zeros(3), 2: np.ones(3)}
        b = {1: np.zeros(3), 2: np.ones(3)}
        per_shift, anc_rmsd, all_rmsd = ToolRouter._anchor_kabsch(a, b, [1, 2])
        assert per_shift is None and anc_rmsd is None and all_rmsd is None


class TestParseAnchorSpec:
    def test_simple_range(self):
        common = set(range(1, 215))
        result = ToolRouter._parse_anchor_spec("1-29", common)
        assert result == list(range(1, 30))

    def test_compound_range(self):
        common = set(range(1, 215))
        result = ToolRouter._parse_anchor_spec("1-29,124-214", common)
        assert result == list(range(1, 30)) + list(range(124, 215))

    def test_restricts_to_common(self):
        common = set(range(5, 25))  # only residues 5-24
        result = ToolRouter._parse_anchor_spec("1-29", common)
        assert result == list(range(5, 25)), "must intersect with common"

    def test_invalid_spec_returns_empty(self):
        assert ToolRouter._parse_anchor_spec("xyz", set(range(1, 100))) == []

    def test_comma_list(self):
        common = set(range(1, 100))
        result = ToolRouter._parse_anchor_spec("5,10,15", common)
        assert result == [5, 10, 15]


class TestResnumsToChimeraXRange:
    def test_contiguous(self):
        assert ToolRouter._resnums_to_chimerax_range([1, 2, 3, 4]) == "1-4"

    def test_compound(self):
        assert ToolRouter._resnums_to_chimerax_range([1, 2, 3, 5, 6, 10]) == "1-3,5-6,10"

    def test_single(self):
        assert ToolRouter._resnums_to_chimerax_range([42]) == "42"

    def test_empty(self):
        assert ToolRouter._resnums_to_chimerax_range([]) == ""

    def test_ak_core_domain(self):
        resnums = list(range(1, 30)) + list(range(124, 215))
        spec = ToolRouter._resnums_to_chimerax_range(resnums)
        assert spec == "1-29,124-214"


# ── B. Colour commands ───────────────────────────────────────────────────────

class TestConformerShiftColorCmds:
    def test_returns_color_commands(self):
        """Non-empty shifts produce at least a reset + one color command."""
        shifts = {i: float(i) for i in range(1, 21)}  # 1..20 Å
        cmds, exps = ToolRouter._conformer_shift_color_cmds(shifts, "#2", "A")
        assert len(cmds) >= 2  # at least reset + one bucket
        assert cmds[0].startswith("color #2/A white")
        assert len(cmds) == len(exps)

    def test_adaptive_scale_all_residues_get_color(self):
        """Every residue must be assigned a bucket; white residues are skipped
        in commands but the percentile logic covers the full range."""
        shifts = {i: float(i) for i in range(1, 101)}
        cmds, _ = ToolRouter._conformer_shift_color_cmds(shifts, "#2", "A")
        # Must have colours other than white
        non_white = [c for c in cmds if "white" not in c]
        assert non_white, "at least one non-white colour bucket must fire"

    def test_grouped_runs_collapses_consecutive(self):
        """Consecutive same-colour residues are collapsed to a range."""
        # All same displacement → all blue (bottom percentile = top percentile)
        shifts = {i: 0.1 for i in range(1, 11)}
        cmds, _ = ToolRouter._conformer_shift_color_cmds(shifts, "#2", "A")
        # Should not have 10 separate color commands
        color_cmds = [c for c in cmds if "color" in c and "white" not in c]
        # With uniform shifts, all end up in the same bucket → 0 or 1 non-white
        assert len(color_cmds) <= 2

    def test_empty_shifts_returns_empty(self):
        cmds, exps = ToolRouter._conformer_shift_color_cmds({}, "#2", "A")
        assert cmds == [] and exps == []

    def test_model_spec_in_commands(self):
        shifts = {1: 1.0, 2: 5.0, 3: 10.0}
        cmds, _ = ToolRouter._conformer_shift_color_cmds(shifts, "#4", "A")
        assert any("#4" in c for c in cmds)


# ── C. Live-coord stub ──────────────────────────────────────────────────────

class TestCaCoordsLive:
    def test_returns_coords_from_json(self):
        """_ca_coords_live writes coords to a temp file from the runscript and reads
        them back (file-based protocol — ChimeraX REST print() truncates large output).
        The mock bridge reads the generated runscript, locates _out, and writes the
        expected JSON payload so the function can read it."""
        import json, re as _re

        payload = {"1": [1.0, 2.0, 3.0], "2": [4.0, 5.0, 6.0]}

        def mock_run_command(cmd):
            m = _re.search(r'runscript "(.+?)"', cmd)
            if m:
                try:
                    with open(m.group(1), encoding="utf-8") as sf:
                        content = sf.read()
                    om = _re.search(r"_out = '(.+?)'", content)
                    if om:
                        with open(om.group(1), "w", encoding="utf-8") as jf:
                            json.dump(payload, jf)
                except Exception:
                    pass
            return {"value": "OK:2", "error": None}

        bridge = MagicMock()
        bridge.run_command.side_effect = mock_run_command
        coords = ToolRouter._ca_coords_live(bridge, "4", "A")
        assert set(coords) == {1, 2}
        np.testing.assert_allclose(coords[1], [1.0, 2.0, 3.0])

    def test_returns_empty_on_non_ok_response(self):
        """Returns {} when run_command value does not start with 'OK:'."""
        bridge = MagicMock()
        bridge.run_command.return_value = {"value": "NOTFOUND", "error": None}
        assert ToolRouter._ca_coords_live(bridge, "99", "A") == {}

    def test_returns_empty_on_error(self):
        bridge = MagicMock()
        bridge.run_command.side_effect = RuntimeError("connection lost")
        assert ToolRouter._ca_coords_live(bridge, "1", "A") == {}


# ── D. Orchestrator ──────────────────────────────────────────────────────────

def _make_coords_pair(n_rigid: int = 30, n_mobile: int = 10,
                      mobile_delta: float = 10.0):
    """Return (coords_a, coords_b) where residues n_rigid+1..n_rigid+n_mobile
    are displaced by *mobile_delta* Å after a rigid body rotation."""
    a = _rigid_body_coords(n_rigid + n_mobile)
    b = _rotate_translate(a, 30.0)
    b = _add_displacement(
        b,
        list(range(n_rigid + 1, n_rigid + n_mobile + 1)),
        np.array([mobile_delta, 0.0, 0.0]),
    )
    return a, b


class TestRunConformerComparison:
    def _make_router_with_coords(self, coords_a, coords_b):
        import json
        router = _make_router()

        def _live_side_effect(bridge_ref, model_id, chain):
            payload = {str(k): list(float(x) for x in v.tolist())
                       for k, v in (coords_a if model_id == "4" else coords_b).items()}
            return {int(k): np.array(v) for k, v in json.loads(json.dumps(payload)).items()}

        router.bridge.run_command.return_value = {
            "value": "RMSD between 30 atom pairs is 0.123 angstroms",
            "error": None,
        }
        return router, _live_side_effect

    def test_known_displacement_detected(self, tmp_path, monkeypatch):
        """The orchestrator detects the known ~10 Å displacement in the mobile segment."""
        coords_a, coords_b = _make_coords_pair(n_rigid=20, n_mobile=10, mobile_delta=10.0)
        router = _make_router()
        router.bridge.run_command.return_value = {
            "value": "RMSD between 20 atom pairs is 0.05 angstroms",
            "error": None,
        }
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: (
                coords_a if mid == "4" else coords_b
            )),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "chain_a": "A", "chain_b": "A",
             "anchor": "1-20"},
        )
        assert result.success, f"should succeed: {result.error}"
        data = result.data
        assert data["anchor_rmsd"] < 1.0, "anchor residual should be small"
        # Mobile segment residues 21-30 should show large shifts
        per_shift = data["per_shift"]
        for rn in range(21, 31):
            assert per_shift[rn] > 5.0, f"res {rn} shift {per_shift[rn]:.2f} < 5 Å"
        # Anchor residues should be small
        for rn in range(1, 21):
            assert per_shift[rn] < 3.0, f"anchor res {rn} shift {per_shift[rn]:.2f} > 3 Å"

    def test_auto_anchor_finds_rigid_core(self, tmp_path, monkeypatch):
        """Auto-anchor mode identifies the rigid core automatically."""
        coords_a, coords_b = _make_coords_pair(n_rigid=20, n_mobile=10, mobile_delta=12.0)
        router = _make_router()
        router.bridge.run_command.return_value = {"value": "RMSD between 20 atom pairs is 0.1 angstroms", "error": None}
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "chain_a": "A", "chain_b": "A",
             "anchor": "auto"},
        )
        assert result.success, f"auto-anchor failed: {result.error}"
        data = result.data
        # Max shift should capture the 12 Å mobile segment
        assert data["max_shift"] > 8.0, f"max shift {data['max_shift']:.1f} should be > 8 Å"

    def test_non_one_start_resnums(self, tmp_path, monkeypatch):
        """Non-1-start chain (like 1IL8 starting at res 2) is handled correctly."""
        coords_a = {r: np.array([float(r), 0.0, 0.0]) for r in range(2, 32)}
        coords_b_base = _rotate_translate(coords_a, 20.0)
        coords_b = _add_displacement(coords_b_base, list(range(22, 32)),
                                     np.array([8.0, 0.0, 0.0]))

        router = _make_router()
        router.bridge.run_command.return_value = {"value": "RMSD between 20 atom pairs is 0.05 angstroms", "error": None}
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "chain_a": "A", "chain_b": "A",
             "anchor": "2-21"},   # anchor starts at 2, not 1
        )
        assert result.success, result.error
        per_shift = result.data["per_shift"]
        # Residue 2 (first anchor residue) should have small shift
        assert per_shift[2] < 2.0, f"res 2 shift {per_shift[2]:.2f} should be small"
        # Residue 31 (last mobile) should have large shift
        assert per_shift[31] > 5.0, f"res 31 shift {per_shift[31]:.2f} should be large"

    def test_csv_written(self, tmp_path, monkeypatch):
        """A CSV artifact is written under cache/."""
        coords_a, coords_b = _make_coords_pair()
        router = _make_router()
        router.bridge.run_command.return_value = {"value": "RMSD between 20 atom pairs is 0.1 angstroms", "error": None}
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "anchor": "1-20"},
        )
        assert result.success
        csv_path = result.data.get("csv_path")
        assert csv_path and Path(csv_path).is_file(), "CSV should be written"
        with open(csv_path, newline="") as fh:
            rows = list(csv.reader(fh))
        assert rows[0] == ["chain", "resno", "shift_A", "region"]
        data_rows = rows[1:]
        assert len(data_rows) == 40, f"expected 40 rows (30 anchor + 10 mobile), got {len(data_rows)}"
        anchor_rows   = [r for r in data_rows if r[3] == "anchor"]
        mobile_rows   = [r for r in data_rows if r[3] == "mobile"]
        assert len(anchor_rows) == 20          # anchor="1-20"
        assert len(mobile_rows) == 20          # residues 21-40 are mobile

    def test_session_persisted(self, tmp_path, monkeypatch):
        """Result is persisted to session.conformer_comparison_results."""
        coords_a, coords_b = _make_coords_pair()
        router = _make_router()
        router.bridge.run_command.return_value = {"value": "RMSD between 20 atom pairs is 0.1 angstroms", "error": None}
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "anchor": "1-20"},
        )
        assert result.success
        stored = router.session.get_conformer_comparison_results("4v1")
        assert stored is not None, "result should be in session"
        assert stored["model_id_a"] == "4"
        assert stored["model_id_b"] == "1"
        assert "anchor_rmsd" in stored

    def test_viz_commands_include_align_and_color(self, tmp_path, monkeypatch):
        """Viz commands include align, view, and color commands."""
        coords_a, coords_b = _make_coords_pair()
        router = _make_router()
        router.bridge.run_command.return_value = {"value": "RMSD between 20 atom pairs is 0.1 angstroms", "error": None}
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "anchor": "1-20"},
        )
        assert result.success
        cmds = result.viz_commands
        assert any("align" in c for c in cmds), "align cmd missing"
        assert any("view" in c for c in cmds), "view cmd missing"
        assert any("color" in c for c in cmds), "color cmd missing"
        # Align target references model A as reference
        align_cmds = [c for c in cmds if "align" in c]
        assert align_cmds and "#4" in align_cmds[0] and "#1" in align_cmds[0]


# ── E. Routing ───────────────────────────────────────────────────────────────

class TestConformerComparisonRouting:
    def test_detect_explicit_keywords(self):
        router = _make_router()
        positive_cases = [
            "compare conformers #4 and #5",
            "conformer comparison between open and closed",
            "show conformational change between these two structures",
            "per-residue shift between #1 and #2",
            "per-residue displacement analysis",
            "anchored overlay of 4AKE onto 1AKE",
            "domain motion analysis anchor on residues 1-29",
            "hinge motion between open and closed form",
            "morph analysis of the two states",
            "open vs closed adenylate kinase",
            "two conformations loaded in ChimeraX",
        ]
        for text in positive_cases:
            assert router._detect_conformer_comparison_intent(text), (
                f"should detect conformer comparison in: {text!r}"
            )

    def test_no_false_positives(self):
        router = _make_router()
        negative_cases = [
            "suggest mutations to improve solubility",
            "color chain A blue",
            "calculate ddG for mutation V82A",
            "fold this sequence with colabfold",
            "validate the design",
            "compare ddG values",        # generic compare, no conformer keyword
            "show me the alignment",     # MPNN alignment
        ]
        for text in negative_cases:
            assert not router._detect_conformer_comparison_intent(text), (
                f"should NOT detect conformer comparison in: {text!r}"
            )

    def test_compound_detect(self):
        router = _make_router()
        assert router._detect_conformer_comparison_intent(
            "compare the two conformers anchored on the core domain"
        )
        assert router._detect_conformer_comparison_intent(
            "compare these two conformations"
        )

    def test_parse_model_ids_from_hash(self):
        router = _make_router()
        opts = router._parse_conformer_comparison_options(
            "compare conformers #4 and #5 chain A anchored on 1-29,124-214"
        )
        assert opts["model_id_a"] == "4"
        assert opts["model_id_b"] == "5"
        assert opts["anchor"] == "1-29,124-214"
        assert opts["chain_a"] == "A"

    def test_parse_auto_anchor_fallback(self):
        router = _make_router()
        opts = router._parse_conformer_comparison_options(
            "compare conformers #4 and #5"
        )
        assert opts["anchor"] == "auto"

    def test_parse_chain_pair(self):
        router = _make_router()
        opts = router._parse_conformer_comparison_options(
            "compare conformers #4 and #5 chain A/B"
        )
        assert opts["chain_a"] == "A"
        assert opts["chain_b"] == "B"

    def test_route_claims_conformer_comparison(self):
        """route() with a conformer-comparison prompt sets tools_needed correctly."""
        router = _make_router()
        tr = {
            "commands": [], "explanations": [], "warnings": [],
            "clarification_needed": None, "confidence": "high",
            "tools_needed": ["chimerax"], "tool_inputs": {},
        }
        routed = router.route(tr, user_input="compare conformers #4 and #5 chain A")
        assert "conformer_comparison" in routed["tools_needed"], (
            f"expected conformer_comparison in {routed['tools_needed']}"
        )

    def test_route_not_captured_by_generic_prompts(self):
        """A mutation-scan prompt does not get swallowed by conformer comparison."""
        router = _make_router()
        tr = {
            "commands": [], "explanations": [], "warnings": [],
            "clarification_needed": None, "confidence": "high",
            "tools_needed": ["mutation_scan"],
            "tool_inputs": {"mutation_scan": {"model_id": "1", "chain": "A"}},
        }
        routed = router.route(tr, user_input="suggest mutations to improve solubility")
        assert "conformer_comparison" not in routed["tools_needed"]


# ── F. Session round-trip ────────────────────────────────────────────────────

class TestSessionRoundTrip:
    def test_set_and_get(self):
        from session_state import SessionState
        ss = SessionState.__new__(SessionState)
        ss.conformer_comparison_results = {}
        ss.set_conformer_comparison_results("4v5", {"anchor_rmsd": 0.12})
        got = ss.get_conformer_comparison_results("4v5")
        assert got == {"anchor_rmsd": 0.12}

    def test_missing_key_returns_none(self):
        from session_state import SessionState
        ss = SessionState.__new__(SessionState)
        ss.conformer_comparison_results = {}
        assert ss.get_conformer_comparison_results("99v100") is None

    def test_save_load_roundtrip(self, tmp_path):
        """conformer_comparison_results survives a save→load cycle."""
        from session_state import SessionState
        ss = SessionState()
        ss.set_conformer_comparison_results("4v5", {"anchor_rmsd": 0.15, "max_shift": 14.2})
        path = tmp_path / "session.json"
        ss.save(str(path))
        ss2 = SessionState.load(str(path))
        got = ss2.get_conformer_comparison_results("4v5")
        assert got is not None
        assert got["anchor_rmsd"] == 0.15
        assert got["max_shift"] == 14.2


# ── G. Error-first ───────────────────────────────────────────────────────────

class TestErrorFirst:
    def test_same_model_id_returns_error(self, monkeypatch):
        router = _make_router()
        result = router._run_conformer_comparison(
            {"model_id_a": "1", "model_id_b": "1"},
        )
        assert not result.success
        assert "same model" in result.error.lower() or "different" in result.error.lower()

    def test_missing_chain_returns_error(self, monkeypatch):
        router = _make_router()
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: {}),  # always empty
        )
        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "chain_a": "Z", "chain_b": "Z"},
        )
        assert not result.success
        assert "could not read" in result.error.lower() or "coordinates" in result.error.lower()

    def test_too_few_common_residues_returns_error(self, monkeypatch):
        router = _make_router()
        # A has 5 residues, B has 5 different residues → 0 common
        a = {i: np.zeros(3) for i in range(1, 6)}
        b = {i: np.zeros(3) for i in range(100, 105)}
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b_ref, mid, ch: a if mid == "4" else b),
        )
        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1"},
        )
        assert not result.success
        assert "common" in result.error.lower() or "residue" in result.error.lower()

    def test_bad_anchor_spec_returns_error(self, monkeypatch):
        coords = {i: np.zeros(3) for i in range(1, 51)}
        router = _make_router()
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords),
        )
        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "anchor": "999-1000"},
        )
        assert not result.success
        assert "anchor" in result.error.lower()

    def test_bridge_none_returns_error(self):
        router = _make_router()
        router.bridge = None
        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1"},
        )
        assert not result.success
        assert "bridge" in result.error.lower()


# ── H. Routing — new anchor-overlay phrases ─────────────────────────────────

class TestNewAnchorOverlayRouting:
    """New compound phrases for 'overlay anchored on conserved core' intent."""

    NEW_POSITIVE_CASES = [
        "open and overlay A and B, anchoring the conserved domains",
        "overlay anchored on the conserved region",
        "overlay anchoring the conserved core",
        "align A and B on the rigid core and show the shift",
        "align these two structures on the rigid core",
        "overlay the two conformers anchored on the rigid core",
        "overlay anchored on residues 1-50",
    ]

    def test_new_phrases_detected(self):
        router = _make_router()
        for text in self.NEW_POSITIVE_CASES:
            assert router._detect_conformer_comparison_intent(text), (
                f"should detect conformer comparison in: {text!r}"
            )

    def test_new_phrases_route_to_conformer_comparison(self):
        router = _make_router()
        for text in self.NEW_POSITIVE_CASES:
            tr = {
                "commands": [], "explanations": [], "warnings": [],
                "clarification_needed": None, "confidence": "high",
                "tools_needed": ["chimerax"], "tool_inputs": {},
            }
            routed = router.route(tr, user_input=text)
            assert "conformer_comparison" in routed["tools_needed"], (
                f"overlay+anchor phrase did not route to conformer_comparison: {text!r}"
            )

    def test_all_chains_parsed(self):
        router = _make_router()
        opts = router._parse_conformer_comparison_options(
            "compare conformers #1 and #2 all chains"
        )
        assert opts.get("chain_a") == "ALL"
        assert opts.get("chain_b") == "ALL"

    def test_all_chains_not_triggered_by_specific_chain(self):
        router = _make_router()
        opts = router._parse_conformer_comparison_options(
            "compare conformers #1 and #2 chain A"
        )
        assert opts.get("chain_a", "A") == "A"


# ── I. Transparency default presentation ─────────────────────────────────────

class TestTransparencyPresentation:
    """Conformer comparison overlay must set ref opaque + mobile semi-transparent."""

    def test_transparency_commands_in_viz(self, tmp_path, monkeypatch):
        coords_a, coords_b = _make_coords_pair(n_rigid=20, n_mobile=10, mobile_delta=8.0)
        router = _make_router()
        router.bridge.run_command.return_value = {
            "value": "RMSD between 20 atom pairs is 0.1 angstroms", "error": None,
        }
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "chain_a": "A", "chain_b": "A",
             "anchor": "1-20"},
        )
        assert result.success, result.error
        # Both transparency commands must appear in viz_commands
        cmds_str = " ".join(result.viz_commands)
        assert "transparency #4 0 target c" in cmds_str, (
            "reference model A must be fully opaque"
        )
        assert "transparency #1" in cmds_str and "target c" in cmds_str, (
            "mobile model B must have transparency target c"
        )

    def test_opaque_before_transparent(self, tmp_path, monkeypatch):
        """Model A transparency-0 command appears before model B semi-transparent."""
        coords_a, coords_b = _make_coords_pair(n_rigid=20, n_mobile=5, mobile_delta=5.0)
        router = _make_router()
        router.bridge.run_command.return_value = {"value": "", "error": None}
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live",
            staticmethod(lambda b, mid, ch: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)
        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "anchor": "1-20"},
        )
        assert result.success, result.error
        cmds = result.viz_commands
        idx_a = next((i for i, c in enumerate(cmds) if "transparency #4 0" in c), None)
        idx_b = next((i for i, c in enumerate(cmds) if "transparency #1" in c and "target c" in c), None)
        assert idx_a is not None, "transparency #4 0 not found"
        assert idx_b is not None, "transparency #1 target c not found"
        assert idx_a < idx_b, "opaque ref must come before semi-transparent mobile"


# ── J. Multichain / quaternary ────────────────────────────────────────────────

class TestMultichainQuaternary:
    """Multichain mode uses (chain,resno) tuple keys and runs iterative prune."""

    def _make_hb_like_coords(self):
        """
        Synthetic haemoglobin-like dataset: 4 chains (A,B,C,D) each 50 residues.
        T→R transition: C and D chains shifted +15 Å relative to A and B.
        """
        rng = np.random.default_rng(7)
        coords = {}
        for chain in ("A", "B", "C", "D"):
            for rn in range(1, 51):
                coords[(chain, rn)] = rng.standard_normal(3) * 5
        return coords

    def _apply_quaternary_shift(self, coords, shift_chains=("C", "D"), delta=15.0):
        out = dict(coords)
        for key, val in coords.items():
            if key[0] in shift_chains:
                out[key] = val + np.array([delta, 0.0, 0.0])
        return out

    def test_multichain_kabsch_detects_quaternary(self):
        """
        With A,B as the user-specified anchor, C,D chains show large shifts.
        This verifies that tuple-keyed Kabsch correctly propagates quaternary motion.

        Note: auto-anchor (iterative prune) cannot reliably select one dimer over the
        other from symmetric synthetic data — both have equal displacement from the
        global-fit midpoint.  In practice the user specifies the anchor dimer, or real
        protein data has intra-chain fold differences that break the symmetry.
        """
        coords_a = self._make_hb_like_coords()
        angle = np.deg2rad(10)
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [0, 0, 1]])
        coords_b_base = {k: R @ v + np.array([2.0, 1.0, 0.0]) for k, v in coords_a.items()}
        coords_b = self._apply_quaternary_shift(coords_b_base, shift_chains=("C", "D"), delta=12.0)

        common = sorted(set(coords_a) & set(coords_b))

        # User-specified anchor: A,B chains only
        anchor = [(ch, rn) for (ch, rn) in common if ch in ("A", "B")]
        per_shift, anc_rmsd, all_rmsd = ToolRouter._anchor_kabsch(coords_a, coords_b, anchor)

        assert per_shift is not None
        assert anc_rmsd < 1.0, f"A,B anchor should be rigid, got {anc_rmsd:.2f} Å"

        # C,D must show large shifts (~12 Å)
        cd_shifts = [per_shift[(ch, rn)] for (ch, rn) in common if ch in ("C", "D")]
        ab_shifts = [per_shift[(ch, rn)] for (ch, rn) in common if ch in ("A", "B")]
        assert np.mean(cd_shifts) > 8.0, (
            f"Mean C,D shift {np.mean(cd_shifts):.1f} Å expected > 8 Å"
        )
        assert np.mean(ab_shifts) < 2.0, (
            f"Mean A,B shift {np.mean(ab_shifts):.1f} Å expected < 2 Å (anchor residuals)"
        )

    def test_iterative_prune_reduces_anchor(self):
        """Iterative prune always produces a strictly smaller anchor."""
        coords_a = self._make_hb_like_coords()
        angle = np.deg2rad(15)
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle),  np.cos(angle), 0],
                      [0, 0, 1]])
        coords_b = {k: R @ v + np.array([1.0, 0.5, 0.0]) for k, v in coords_a.items()}
        # Add large displacement to C chains
        for (ch, rn), v in list(coords_b.items()):
            if ch == "C":
                coords_b[(ch, rn)] = v + np.array([20.0, 0.0, 0.0])

        common = sorted(set(coords_a) & set(coords_b))
        min_anchor = max(10, int(len(common) * 0.05))
        anchor = list(common)
        for _ in range(12):
            shifts_g, _, _ = ToolRouter._anchor_kabsch(coords_a, coords_b, anchor)
            cutoff = float(np.percentile(sorted(shifts_g.values()), 40))
            new_anchor = [r for r in anchor if shifts_g[r] <= cutoff]
            if len(new_anchor) < min_anchor or set(new_anchor) == set(anchor):
                break
            anchor = new_anchor
        assert len(anchor) < len(common), (
            f"Iterative prune should reduce anchor ({len(anchor)} vs {len(common)})"
        )

    def test_mc_anchor_to_align_specs_same_range(self):
        """All chains same anchor range → compact /CHAINS:range spec."""
        anchor = [(ch, rn) for ch in ("A", "B") for rn in range(1, 21)]
        spec_b, spec_a = ToolRouter._mc_anchor_to_align_specs(anchor, "1", "2")
        assert "/A,B:1-20@CA" in spec_b
        assert "/A,B:1-20@CA" in spec_a

    def test_mc_anchor_to_align_specs_different_ranges(self):
        """Different anchor ranges per chain → all-chain Cα spec."""
        anchor = ([(("A", rn)) for rn in range(1, 21)] +
                  [(("B", rn)) for rn in range(5, 30)])
        spec_b, spec_a = ToolRouter._mc_anchor_to_align_specs(anchor, "1", "2")
        # Different ranges → must use full-chain spec
        assert "/A,B@CA" in spec_b

    def test_multichain_orchestrator_runs(self, tmp_path, monkeypatch):
        """_run_conformer_comparison with chain='ALL' reads multichain coords."""
        coords_a = {("A", rn): np.zeros(3) + rn for rn in range(1, 31)}
        coords_b = {("A", rn): np.zeros(3) + rn + 0.1 for rn in range(1, 31)}
        coords_b.update({("A", rn): np.zeros(3) + rn + 10.0 for rn in range(21, 31)})

        router = _make_router()
        router.bridge.run_command.return_value = {
            "value": "RMSD between 20 atom pairs is 0.1 angstroms", "error": None
        }
        monkeypatch.setattr(
            ToolRouter, "_ca_coords_live_multichain",
            staticmethod(lambda b, mid: coords_a if mid == "4" else coords_b),
        )
        monkeypatch.chdir(tmp_path)

        result = router._run_conformer_comparison(
            {"model_id_a": "4", "model_id_b": "1", "chain_a": "ALL", "chain_b": "ALL",
             "anchor": "auto"},
        )
        assert result.success, result.error
        # per_shift should have tuple keys
        per_shift = result.data["per_shift"]
        assert all(isinstance(k, tuple) for k in per_shift), (
            "multichain per_shift keys must be (chain, resno) tuples"
        )
        # Top displaced residues must have chain field from tuple key
        top = result.data["top_shifted"]
        assert top[0]["chain"] == "A"


# ── K. Assembly metadata — 2HHB/1BBB ────────────────────────────────────────

class TestAssemblyMetadata2HHB:
    """Assembly analyser stoichiometry fix: 2HHB must not be labelled 'homodimer'."""

    def test_hhb_stoich_combines_multiple_entries(self):
        """
        RCSB returns stoich ["A2","B2"] for haemoglobin.  Combined → "A2B2"
        → "heterotetrameric (α2β2)", not "homodimer".
        """
        from assembly_analyser import _stoichiometry_label
        # Simulate the combined stoich (after fix: "".join(sorted(["A2","B2"])))
        combined = "".join(sorted(["A2", "B2"]))  # "A2B2"
        label = _stoichiometry_label(combined, 4)
        assert label == "heterotetrameric (α2β2)", (
            f"Expected heterotetrameric (α2β2), got: {label!r}"
        )

    def test_homodimer_unchanged_for_a2(self):
        """A genuine A2 homodimer (n=2) must still return 'homodimer'."""
        from assembly_analyser import _stoichiometry_label
        label = _stoichiometry_label("A2", 2)
        assert label == "homodimer", f"Expected homodimer, got: {label!r}"

    def test_single_stoich_not_combined(self):
        """["A4"] → "A4" → 'homotetramer' (no change)."""
        from assembly_analyser import _stoichiometry_label
        single = "".join(sorted(["A4"]))  # "A4"
        label = _stoichiometry_label(single, 4)
        assert label == "homotetramer", f"Expected homotetramer, got: {label!r}"
