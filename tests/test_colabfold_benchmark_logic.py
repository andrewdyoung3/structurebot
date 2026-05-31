"""
tests/test_colabfold_benchmark_logic.py
---------------------------------------
CI unit tests for the ColabFold-benchmark HARNESS LOGIC (no folds, no env). The
benchmark module itself skips its live tests by default; these import its pure
helpers and verify them with synthetic data so the harness is guarded in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from _pytest.outcomes import Skipped

import tests.test_colabfold_benchmark as bm


# ── BioPython all-pairs Cα RMSD helper ──────────────────────────────────────────

def test_rmsd_identical_is_zero():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 1]], dtype=float)
    assert bm._all_pairs_ca_rmsd(pts, pts.copy()) == 0.0


def test_rmsd_invariant_to_rigid_translation():
    pts = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 1]], dtype=float)
    moved = pts + np.array([10.0, -5.0, 3.0])     # pure translation
    assert bm._all_pairs_ca_rmsd(pts, moved) == 0.0   # superposition removes it


def test_rmsd_known_nonzero_displacement():
    # Two residues exactly +d on z; superposition can't remove a non-rigid offset.
    native = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    pred   = native.copy()
    pred[1, 2] += 2.0
    pred[2, 2] += 2.0
    r = bm._all_pairs_ca_rmsd(native, pred)
    assert r is not None and r > 0.0
    # Not larger than the largest single displacement.
    assert r <= 2.0


def test_rmsd_too_few_points_returns_none():
    pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    assert bm._all_pairs_ca_rmsd(pts, pts) is None


def test_rmsd_matches_min_length():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    b = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)   # shorter
    assert bm._all_pairs_ca_rmsd(a, b) == 0.0     # compares over the 3 matched


# ── Native chain extraction (BioPython) ─────────────────────────────────────────

def _write_chain_pdb(path, chain="A", n=5):
    lines = []
    for i in range(1, n + 1):
        ln = list(" " * 66)
        ln[0:6]   = "ATOM  "
        ln[6:11]  = f"{i:>5}"
        ln[12:16] = " CA "
        ln[17:20] = "ALA"
        ln[21]    = chain
        ln[22:26] = f"{i:>4}"
        ln[30:38] = f"{float(i):>8.3f}"
        ln[38:46] = f"{0.0:>8.3f}"
        ln[46:54] = f"{0.0:>8.3f}"
        lines.append("".join(ln))
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")
    return str(path)


def test_chain_ca_extracts_sequence_and_coords(tmp_path):
    p = _write_chain_pdb(tmp_path / "m.pdb", "A", 5)
    seq, ca, resnos = bm._chain_ca(p, "A")
    assert seq == "AAAAA"
    assert ca.shape == (5, 3)
    assert resnos == [1, 2, 3, 4, 5]
    assert tuple(ca[2]) == (3.0, 0.0, 0.0)


def test_predicted_ca_reads_chain(tmp_path):
    p = _write_chain_pdb(tmp_path / "p.pdb", "A", 4)
    ca = bm._predicted_ca(p, "A")
    assert ca.shape == (4, 3)


# ── Recording + aggregate medians ───────────────────────────────────────────────

def test_make_record_schema():
    rec = bm._make_record("1CRN", "A", 46, 92.1, 0.85, 0.9, 120.0, True)
    assert rec == {
        "pdb": "1CRN", "chain": "A", "length": 46, "mean_plddt": 92.1,
        "ptm": 0.85, "all_pairs_ca_rmsd": 0.9, "wall_s": 120.0, "gpu_used": True,
    }


def test_panel_medians_and_gpu_counts():
    entries = [
        {"all_pairs_ca_rmsd": 1.0, "mean_plddt": 90.0, "gpu_used": True},
        {"all_pairs_ca_rmsd": 2.0, "mean_plddt": 80.0, "gpu_used": True},
        {"all_pairs_ca_rmsd": 3.0, "mean_plddt": 70.0, "gpu_used": False},
    ]
    s = bm._panel_medians(entries)
    assert s["n"] == 3
    assert s["median_rmsd"] == 2.0
    assert s["median_plddt"] == 80.0
    assert s["n_gpu"] == 2 and s["n_cpu"] == 1


def _write_results(path, rows):
    data = {r["pdb"]: r for r in rows}
    Path(path).write_text(json.dumps(data), encoding="utf-8")
    return path


def test_compute_panel_stats_passes_above_min(tmp_path):
    rows = [
        {"pdb": "1CRN", "all_pairs_ca_rmsd": 1.0, "mean_plddt": 90.0, "ptm": 0.8,
         "length": 46, "gpu_used": True},
        {"pdb": "1UBQ", "all_pairs_ca_rmsd": 1.5, "mean_plddt": 85.0, "ptm": 0.8,
         "length": 76, "gpu_used": True},
        {"pdb": "2LZM", "all_pairs_ca_rmsd": 2.0, "mean_plddt": 80.0, "ptm": 0.8,
         "length": 164, "gpu_used": True},
    ]
    p = _write_results(tmp_path / "r.json", rows)
    stats = bm.compute_panel_stats(results_path=p, min_entries=3)
    assert stats["n"] == 3 and stats["median_rmsd"] == 1.5 and stats["n_cpu"] == 0


def test_compute_panel_stats_skips_below_min(tmp_path):
    rows = [
        {"pdb": "1CRN", "all_pairs_ca_rmsd": 1.0, "mean_plddt": 90.0, "ptm": 0.8,
         "length": 46, "gpu_used": True},
        {"pdb": "1UBQ", "all_pairs_ca_rmsd": 1.5, "mean_plddt": 85.0, "ptm": 0.8,
         "length": 76, "gpu_used": True},
    ]
    p = _write_results(tmp_path / "r.json", rows)
    with pytest.raises(Skipped):                       # SKIP, not fail, below min-N
        bm.compute_panel_stats(results_path=p, min_entries=3)


def test_compute_panel_stats_skips_when_file_missing(tmp_path):
    with pytest.raises(Skipped):
        bm.compute_panel_stats(results_path=tmp_path / "nope.json", min_entries=3)


def test_panel_and_thresholds_are_sane():
    # Guard the panel/threshold constants themselves.
    assert len(bm.PANEL) >= bm.MIN_BENCHMARK_ENTRIES
    assert all(len(pid) == 4 and ch for pid, ch in bm.PANEL)
    assert bm.MEDIAN_RMSD_MAX_A == 3.0 and bm.MEDIAN_PLDDT_MIN == 70.0
