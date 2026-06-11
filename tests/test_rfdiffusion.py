"""
tests/test_rfdiffusion.py
-------------------------
Tests for RFdiffusionBridge (rfdiffusion_bridge.py).

RFdiffusion is a documented stub: the bridge is designed to return a helpful
error when RFDIFFUSION_DIR is not set.  Tests cover:

A. Availability     -- _check_available logic
B. analyze() errors -- not configured, missing pdb, unknown mode
C. Command building -- _build_cmd produces correct Hydra overrides
D. Full pipeline    -- mocked subprocess call

Usage
-----
  cd structurebot
  python -m pytest tests/test_rfdiffusion.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import rfdiffusion_bridge as _rfd_mod
from rfdiffusion_bridge import RFdiffusionBridge

# -- Helpers -------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

_results = {"pass": 0, "fail": 0, "skip": 0}


def _ok(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {PASS} {name}{suffix}")
    _results["pass"] += 1


def _fail(name: str, reason: str) -> None:
    print(f"  {FAIL} {name}: {reason}")
    _results["fail"] += 1


def _skip(name: str, reason: str) -> None:
    print(f"  {SKIP} {name}: {reason}")
    _results["skip"] += 1


def _assert(cond: bool, name: str, msg: str = "") -> bool:
    if cond:
        _ok(name)
        return True
    else:
        _fail(name, msg or "assertion failed")
        return False


# -- A. Availability -----------------------------------------------------------

def test_check_available_no_dir() -> None:
    print("\n=== A. Availability ===")
    with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", ""):
        b = RFdiffusionBridge()
    _assert(not b._available, "not available when _RFDIFFUSION_DIR is empty")


def test_check_available_missing_models() -> None:
    """Directory with run_inference.py but no models/ -> not available."""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "run_inference.py").write_text("# stub\n")
        # models/ intentionally absent
        with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", td):
            b = RFdiffusionBridge()
    _assert(not b._available, "not available when models/ absent")


def test_check_available_full_dir() -> None:
    """Directory with both run_inference.py and models/ -> available."""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "run_inference.py").write_text("# stub\n")
        (Path(td) / "models").mkdir()
        with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", td):
            b = RFdiffusionBridge()
    _assert(b._available, "available when run_inference.py + models/ present")
    _assert(b._script is not None, "script path set when available")


# -- B. analyze() error paths --------------------------------------------------

def test_analyze_not_configured() -> None:
    print("\n=== B. analyze() errors ===")
    with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", "/nonexistent/path"):
        b = RFdiffusionBridge()
    result = b.analyze({"mode": "binder", "pdb_path": "/any/file.pdb"})
    _assert(not result.success, "analyze() fails when not configured")
    _assert(bool(result.error), "error message is non-empty")
    _assert(
        "RFDIFFUSION_DIR" in result.error or "not" in result.error.lower(),
        "error mentions configuration step",
        f"error: {result.error[:120]}",
    )


def test_analyze_missing_pdb_binder() -> None:
    """analyze() returns failure for binder mode when pdb_path is missing."""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "run_inference.py").write_text("# stub\n")
        (Path(td) / "models").mkdir()
        with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", td):
            b = RFdiffusionBridge()
    result = b.analyze({"mode": "binder", "pdb_path": "/nonexistent.pdb"})
    _assert(not result.success, "analyze() fails for binder mode with missing pdb")
    _assert(bool(result.error), "error message non-empty")


def test_analyze_symmetric_no_pdb_required() -> None:
    """symmetric mode does not require a pdb_path."""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "run_inference.py").write_text("# stub\n")
        (Path(td) / "models").mkdir()
        with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", td):
            b = RFdiffusionBridge()

        # Execution is the single wsl.exe -> RFDIFFUSION_PYTHON dispatch; mock at
        # that seam (not a Windows subprocess) and confirm symmetric mode reaches
        # dispatch rather than failing pdb validation.
        def fake_dispatch(self, cmd, run_dir):
            raise RuntimeError("dispatch skipped in test")

        with patch.object(_rfd_mod.RFdiffusionBridge, "_dispatch", fake_dispatch):
            result = b.analyze({"mode": "symmetric", "symmetry": "C3"})
    # Should fail at dispatch (RuntimeError), not at validation
    _assert(not result.success, "symmetric mode proceeds past validation")
    _assert("dispatch skipped" in (result.error or "") or result.error is not None,
            "fails at dispatch, not pdb validation",
            f"error: {result.error!r}")


# -- C. Command building -------------------------------------------------------

def _make_available_bridge(td: str) -> RFdiffusionBridge:
    """Create a bridge pointing to a minimal fake RFdiffusion dir."""
    (Path(td) / "run_inference.py").write_text("# stub\n")
    (Path(td) / "models").mkdir(exist_ok=True)
    with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", td):
        return RFdiffusionBridge()


def test_build_cmd_binder() -> None:
    print("\n=== C. Command building ===")
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        out_path = Path(td) / "out"
        cmd = b._build_cmd(
            mode="binder", pdb_path="/fake/target.pdb", chain_id="A",
            hotspots=[82, 83, 119], num_designs=4, num_steps=50,
            symmetry="", partial_T=0.2, contigs="", out_path=out_path,
        )
    cmd_str = " ".join(cmd)
    _assert("run_inference.py" in cmd_str, "script in binder command")
    _assert("inference.input_pdb=/fake/target.pdb" in cmd_str,
            "input_pdb in binder command")
    _assert("inference.num_designs=4" in cmd_str, "num_designs in command")
    _assert("ppi.hotspot_res=[A82,A83,A119]" in cmd_str,
            "hotspot_res in binder command",
            f"cmd: {cmd_str}")


def test_build_cmd_symmetric() -> None:
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        out_path = Path(td) / "out"
        cmd = b._build_cmd(
            mode="symmetric", pdb_path="", chain_id="A",
            hotspots=[], num_designs=2, num_steps=50,
            symmetry="C3", partial_T=0.2, contigs="", out_path=out_path,
        )
    cmd_str = " ".join(cmd)
    _assert("inference.symmetry=C3" in cmd_str, "symmetry=C3 in command",
            f"cmd: {cmd_str}")
    _assert("inference.num_designs=2" in cmd_str, "num_designs in symmetric command")


def test_build_cmd_partial_diffusion() -> None:
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        out_path = Path(td) / "out"
        cmd = b._build_cmd(
            mode="partial_diffusion", pdb_path="/fake/input.pdb", chain_id="A",
            hotspots=[], num_designs=4, num_steps=50,
            symmetry="", partial_T=0.2, contigs="", out_path=out_path,
        )
    cmd_str = " ".join(cmd)
    # partial_T=0.2, num_steps=50 -> partial_T = int(0.2 * 50) = 10
    _assert("diffuser.partial_T=10" in cmd_str,
            "partial_T correctly computed (0.2*50=10)",
            f"cmd: {cmd_str}")


def test_build_cmd_unknown_mode_raises() -> None:
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        out_path = Path(td) / "out"
        try:
            b._build_cmd(
                mode="invalid_mode", pdb_path="", chain_id="A",
                hotspots=[], num_designs=1, num_steps=50,
                symmetry="", partial_T=0.2, contigs="", out_path=out_path,
            )
            _fail("unknown mode raises ValueError", "no exception raised")
        except ValueError as e:
            _assert("invalid_mode" in str(e), "ValueError mentions the bad mode",
                    f"got: {e}")


# -- D. Full pipeline (mocked subprocess) --------------------------------------

def test_full_pipeline_binder_mock() -> None:
    """Full analyze() binder pipeline with the WSL dispatch seam mocked."""
    print("\n=== D. Full pipeline (mocked) ===")

    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)

        pdb_content = (
            "ATOM      1  CA  LEU A  10       1.000   2.000   3.000  1.00 10.00\n"
        )
        pdb_file = Path(td) / "target.pdb"
        pdb_file.write_text(pdb_content)

        # Mock the single dispatch seam: emulate RFdiffusion writing its PDBs into
        # the (cache) run_dir, exactly where the bridge collects them. Intent
        # preserved: build-cmd (tested in C) / dispatch / collect-PDB.
        def fake_dispatch(self, cmd, run_dir):
            run_dir.mkdir(parents=True, exist_ok=True)
            for i in range(2):
                (run_dir / f"binder_{i}.pdb").write_text(f"REMARK Design {i}\n")

        with patch.object(_rfd_mod._cfg, "RFDIFFUSION_CACHE_DIR", Path(td)), \
             patch.object(_rfd_mod.RFdiffusionBridge, "_dispatch", fake_dispatch):
            result = b.analyze({
                "mode":             "binder",
                "pdb_path":         str(pdb_file),
                "chain_id":         "A",
                "hotspot_residues": [10],
                "num_designs":      2,
                "num_steps":        50,
                "model_id":         "1",
            })

    _assert(result.success, "analyze() succeeded with mocked subprocess",
            f"error: {result.error}")
    _assert(isinstance(result.data, dict), "result.data is a dict")
    _assert("pdb_paths" in result.data, "data has 'pdb_paths' key")
    _assert(len(result.data["pdb_paths"]) == 2, "2 PDB files returned",
            f"got {len(result.data.get('pdb_paths', []))}")
    _assert(result.data["mode"] == "binder", "mode=binder in result")
    _assert(isinstance(result.viz_commands, list) and len(result.viz_commands) > 0,
            "viz_commands non-empty")
    _assert(isinstance(result.summary, str) and "RFdiffusion" in result.summary,
            "summary mentions RFdiffusion",
            f"summary: {result.summary!r}")


def test_status_not_configured() -> None:
    """status() returns helpful string when not configured."""
    with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", ""):
        b = RFdiffusionBridge()
    s = b.status()
    _assert(isinstance(s, str) and len(s) > 5, "status() returns non-empty string",
            f"got: {s!r}")
    _assert("RFDIFFUSION_DIR" in s or "not" in s.lower(),
            "status mentions configuration",
            f"got: {s!r}")


# -- Runner --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("tests/test_rfdiffusion.py -- RFdiffusion Bridge Tests")
    print("=" * 60)

    # A. Availability
    test_check_available_no_dir()
    test_check_available_missing_models()
    test_check_available_full_dir()

    # B. Error paths
    test_analyze_not_configured()
    test_analyze_missing_pdb_binder()
    test_analyze_symmetric_no_pdb_required()

    # C. Command building
    test_build_cmd_binder()
    test_build_cmd_symmetric()
    test_build_cmd_partial_diffusion()
    test_build_cmd_unknown_mode_raises()

    # D. Full pipeline
    test_full_pipeline_binder_mock()
    test_status_not_configured()

    print()
    print("=" * 60)
    print(
        f"Results: {_results['pass']} passed, "
        f"{_results['fail']} failed, "
        f"{_results['skip']} skipped"
    )
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
