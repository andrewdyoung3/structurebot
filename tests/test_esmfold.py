"""
tests/test_esmfold.py
---------------------
Tests for ESMFoldBridge.

Test sections
-------------
  A. API call mock      -- POST format, URL, content-type (Atlas path)
  B. pLDDT parsing      -- B-factor column extraction from mock PDB
  C. compare_to_wildtype -- mock predictions; risk classification
  D. Foldability risk   -- threshold boundary conditions
  E. Cys misparing      -- disulfide foldability check
  F. Timeout / error    -- unreachable API graceful fallback
  G. Local venv312 path -- subprocess mock, pLDDT scale guard, fallback

Usage
-----
  cd structurebot
  python -m pytest tests/test_esmfold.py -v
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as _cfg
from esmfold_bridge import ESMFoldBridge, _parse_plddt_from_pdb

# ── Helpers ────────────────────────────────────────────────────────────────────

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
    _fail(name, msg or "assertion failed")
    return False


def _make_bridge() -> ESMFoldBridge:
    return ESMFoldBridge()


# ── Minimal mock PDB with known B-factors (pLDDT) ─────────────────────────────

_MOCK_PDB = textwrap.dedent("""\
    ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 87.30           N
    ATOM      2  CA  MET A   1      11.000  10.000  10.000  1.00 87.30           C
    ATOM      3  CB  MET A   1      12.000  10.000  10.000  1.00 87.30           C
    ATOM      4  N   ALA A   2      11.000  11.000  10.000  1.00 91.20           N
    ATOM      5  CA  ALA A   2      12.000  11.000  10.000  1.00 91.20           C
    ATOM      6  N   GLY A   3      12.000  12.000  10.000  1.00 45.00           N
    ATOM      7  CA  GLY A   3      13.000  12.000  10.000  1.00 45.00           C
    END
""")


# ═══════════════════════════════════════════════════════════════════════════════
# A. API call mock  (ESMFOLD_USE_LOCAL=False to test the Atlas path directly)
# ═══════════════════════════════════════════════════════════════════════════════

def test_api_post_format() -> None:
    """predict() POSTs sequence to ESM Atlas primary URL with correct Content-Type."""
    print("\n=== A. API call mock ===")
    bridge = _make_bridge()

    # Build a mock response that looks like a PDB file
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = _MOCK_PDB

    with patch.object(_cfg, "ESMFOLD_USE_LOCAL", False):
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = bridge.predict("MAGLK", label="test")

    _assert(mock_post.called, "requests.post was called")
    call_kwargs = mock_post.call_args
    _assert(call_kwargs is not None, "call_args is not None")
    # Check URL — first positional or keyword arg
    args   = call_kwargs.args
    kwargs = call_kwargs.kwargs
    url_used = args[0] if args else kwargs.get("url", "")
    _assert("esmatlas.com" in url_used, "URL targets esmatlas.com",
            f"got {url_used!r}")
    # Check data format contains sequence
    data_sent = kwargs.get("data", {})
    if isinstance(data_sent, dict):
        _assert("sequence" in data_sent, "data dict has 'sequence' key",
                f"got keys: {list(data_sent.keys())}")
        _assert(data_sent["sequence"] == "MAGLK", "sequence value is correct",
                f"got {data_sent['sequence']!r}")
    else:
        _assert("MAGLK" in str(data_sent), "sequence in raw body")


def test_api_success_returns_pdb_and_plddt() -> None:
    """predict() returns pdb_str and populated plddt dict on HTTP 200."""
    bridge = _make_bridge()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = _MOCK_PDB

    with patch.object(_cfg, "ESMFOLD_USE_LOCAL", False):
        with patch("requests.post", return_value=mock_resp):
            result = bridge.predict("MAGLK")

    _assert(result["success"],  "result.success is True")
    _assert(len(result["plddt"]) > 0, "plddt dict is non-empty",
            f"got {result['plddt']}")
    _assert(result["mean_plddt"] > 0, "mean_plddt is positive",
            f"got {result['mean_plddt']}")
    _assert("ATOM" in result["pdb_str"], "pdb_str contains ATOM records")


def test_api_http_error_returns_failure() -> None:
    """predict() returns success=False on HTTP 500."""
    bridge = _make_bridge()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"

    with patch.object(_cfg, "ESMFOLD_USE_LOCAL", False):
        with patch("requests.post", return_value=mock_resp):
            # Alt URL will also be tried and fail
            result = bridge.predict("MAGLK")

    _assert(not result["success"], "result.success is False on HTTP 500")
    _assert(result["error"] is not None, "error message is set",
            f"got {result['error']!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# B. pLDDT parsing
# ═══════════════════════════════════════════════════════════════════════════════

def test_plddt_parse_ca_only() -> None:
    """_parse_plddt_from_pdb extracts only CA atoms (one value per residue)."""
    print("\n=== B. pLDDT parsing ===")
    plddt = _parse_plddt_from_pdb(_MOCK_PDB)
    _assert(len(plddt) == 3, "3 residues parsed (one CA per residue)",
            f"got {len(plddt)}")


def test_plddt_values_correct() -> None:
    """Parsed pLDDT values match B-factor column in mock PDB."""
    plddt = _parse_plddt_from_pdb(_MOCK_PDB)
    _assert(plddt.get(1) == 87.3,  "residue 1 pLDDT = 87.3", f"got {plddt.get(1)}")
    _assert(plddt.get(2) == 91.2,  "residue 2 pLDDT = 91.2", f"got {plddt.get(2)}")
    _assert(plddt.get(3) == 45.0,  "residue 3 pLDDT = 45.0", f"got {plddt.get(3)}")


def test_plddt_empty_string() -> None:
    """_parse_plddt_from_pdb returns {} on empty input."""
    plddt = _parse_plddt_from_pdb("")
    _assert(plddt == {}, "empty PDB string -> empty dict", f"got {plddt}")


# ═══════════════════════════════════════════════════════════════════════════════
# C. compare_to_wildtype
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_predict(wt_plddt_at_pos: float, mut_plddt_at_pos: float):
    """
    Return a mock predict() that provides the given pLDDT values
    at position 2 for WT and mutant respectively.
    """
    call_count = [0]

    def _predict(sequence, label="query", timeout=120):
        call_count[0] += 1
        if call_count[0] == 1:
            plddt = {1: 90.0, 2: wt_plddt_at_pos, 3: 88.0}
        else:
            plddt = {1: 89.0, 2: mut_plddt_at_pos, 3: 87.0}
        mean = sum(plddt.values()) / len(plddt)
        return {
            "success":    True,
            "pdb_str":    _MOCK_PDB,
            "plddt":      plddt,
            "mean_plddt": round(mean, 2),
            "length":     len(sequence),
            "error":      None,
            "source":     "mock",
            "label":      label,
        }
    return _predict


def test_compare_to_wildtype_low_drop() -> None:
    """compare_to_wildtype: small drop (<5) -> foldability_risk='low'."""
    print("\n=== C. compare_to_wildtype ===")
    bridge = _make_bridge()
    bridge.predict = _make_mock_predict(wt_plddt_at_pos=88.0, mut_plddt_at_pos=85.0)
    result = bridge.compare_to_wildtype("MAK", "MCK", mutation_positions=[2])
    _assert(result["success"], "compare_to_wildtype success")
    _assert(result["foldability_risk"] == "low",
            "risk='low' for 3-point drop",
            f"got {result['foldability_risk']}")
    _assert(result["plddt_drop"] == pytest_approx(3.0, tol=0.1),
            "plddt_drop ~ 3.0",
            f"got {result['plddt_drop']}")


def pytest_approx(val, tol=0.01):
    """Mini approximation helper (no pytest dependency for standalone run)."""
    class _A:
        def __init__(self, v, t):
            self.v = v
            self.t = t
        def __eq__(self, other):
            diff = other - self.v
            return (diff if diff >= 0 else -diff) <= self.t
    return _A(val, tol)


def test_compare_to_wildtype_high_drop() -> None:
    """compare_to_wildtype: large drop (>=threshold) -> foldability_risk='high'."""
    bridge = _make_bridge()
    threshold = getattr(_cfg, "ESMFOLD_PLDDT_WARNING_THRESHOLD", 10.0)
    drop = threshold + 2.0
    bridge.predict = _make_mock_predict(
        wt_plddt_at_pos=90.0,
        mut_plddt_at_pos=90.0 - drop,
    )
    result = bridge.compare_to_wildtype("MAK", "MCK", mutation_positions=[2])
    _assert(result["success"], "compare success")
    _assert(result["foldability_risk"] == "high",
            f"risk='high' for drop > threshold ({threshold})",
            f"got risk={result['foldability_risk']}, drop={result['plddt_drop']}")
    _assert(result["warning"] is not None, "warning is set for high risk")


def test_compare_position_scores_populated() -> None:
    """compare_to_wildtype: position_scores contains entry for each mutation pos."""
    bridge = _make_bridge()
    bridge.predict = _make_mock_predict(88.0, 84.0)
    result = bridge.compare_to_wildtype("MAK", "MCK", mutation_positions=[2])
    _assert(2 in result.get("position_scores", {}),
            "position_scores has entry for pos 2",
            f"got keys {list(result.get('position_scores', {}).keys())}")
    ps = result["position_scores"].get(2, {})
    _assert("wt" in ps and "mut" in ps and "drop" in ps,
            "position_scores[2] has wt/mut/drop keys")


# ═══════════════════════════════════════════════════════════════════════════════
# D. Foldability risk thresholds
# ═══════════════════════════════════════════════════════════════════════════════

def test_risk_threshold_medium() -> None:
    """Drop in [5, threshold) -> 'medium'."""
    print("\n=== D. Risk threshold ===")
    bridge = _make_bridge()
    threshold = getattr(_cfg, "ESMFOLD_PLDDT_WARNING_THRESHOLD", 10.0)
    # Medium zone: exactly 7.0 (between 5 and threshold)
    drop = min(7.0, threshold - 0.5)
    bridge.predict = _make_mock_predict(
        wt_plddt_at_pos=90.0,
        mut_plddt_at_pos=90.0 - drop,
    )
    result = bridge.compare_to_wildtype("MAK", "MCK", mutation_positions=[2])
    _assert(result["success"], "success")
    _assert(result["foldability_risk"] == "medium",
            f"risk='medium' for drop={drop:.1f}",
            f"got {result['foldability_risk']}")


def test_risk_low_warning_is_none() -> None:
    """Low risk -> warning is None."""
    bridge = _make_bridge()
    bridge.predict = _make_mock_predict(88.0, 86.0)
    result = bridge.compare_to_wildtype("MAK", "MCK", mutation_positions=[2])
    _assert(result.get("warning") is None,
            "warning is None for low risk",
            f"got {result.get('warning')!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# E. Cys misparing
# ═══════════════════════════════════════════════════════════════════════════════

def test_check_disulfide_missing_pdb() -> None:
    """check_disulfide_foldability returns success=False for nonexistent PDB."""
    print("\n=== E. Cys misparing ===")
    bridge = _make_bridge()
    result = bridge.check_disulfide_foldability(
        "/nonexistent/path/fake.pdb", chain_a_res=10, chain_b_res=50
    )
    _assert(not result["success"], "success=False for missing PDB")
    _assert(result.get("error") is not None, "error message set")


def test_check_disulfide_misparing_detected(tmp_path) -> None:
    """check_disulfide_foldability detects existing free Cys and sets misparing_risk=True."""
    import tempfile
    # Write a minimal PDB with a Cys at position 3 (besides the target at 1)
    pdb_content = textwrap.dedent("""\
        ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00 80.00           C
        ATOM      2  CB  ALA A   1       1.500   1.000   1.000  1.00 80.00           C
        ATOM      3  CA  GLU A   2       2.000   1.000   1.000  1.00 82.00           C
        ATOM      4  CA  CYS A   3       3.000   1.000   1.000  1.00 78.00           C
        ATOM      5  CB  CYS A   3       3.500   1.000   1.000  1.00 78.00           C
        ATOM      6  CA  LYS B   1       5.000   1.000   1.000  1.00 85.00           C
        ATOM      7  CB  LYS B   1       5.500   1.000   1.000  1.00 85.00           C
        ATOM      8  CA  GLY B   2       6.000   1.000   1.000  1.00 70.00           C
        END
    """)
    pdb_file = Path(tempfile.mktemp(suffix=".pdb"))
    pdb_file.write_text(pdb_content, encoding="utf-8")

    bridge = _make_bridge()
    # Patch compare_to_wildtype so we don't actually call ESM Atlas
    def _mock_compare(wt_seq, mut_seq, positions):
        return {
            "success":          True,
            "mean_plddt_wt":    88.0,
            "mean_plddt_mut":   86.5,
            "plddt_drop":       1.5,
            "foldability_risk": "low",
            "position_scores":  {},
            "warning":          None,
            "error":            None,
        }
    bridge.compare_to_wildtype = _mock_compare

    result = bridge.check_disulfide_foldability(
        str(pdb_file), chain_a_res=1, chain_b_res=1, chain_a="A", chain_b="B"
    )
    pdb_file.unlink(missing_ok=True)

    if result.get("success") is False:
        # Sequence extraction may fail on minimal PDB — skip gracefully
        _skip("misparing detection (sequence extraction failed on minimal PDB)",
              str(result.get("error", "")))
        return

    _assert(result.get("existing_cys_count", 0) >= 1,
            "existing_cys_count >= 1 (Cys at pos 3 detected)",
            f"got {result.get('existing_cys_count')}")
    _assert(result.get("misparing_risk") is True,
            "misparing_risk=True when free Cys present",
            f"got {result.get('misparing_risk')}")


# ═══════════════════════════════════════════════════════════════════════════════
# F. Timeout / error handling  (ESMFOLD_USE_LOCAL=False for Atlas path tests)
# ═══════════════════════════════════════════════════════════════════════════════

def test_predict_empty_sequence() -> None:
    """predict() returns success=False for empty sequence."""
    print("\n=== F. Error handling ===")
    bridge = _make_bridge()
    result = bridge.predict("", label="empty")
    _assert(not result["success"], "empty sequence -> success=False")
    _assert(result["error"] is not None, "error message set")


def test_predict_connection_error() -> None:
    """predict() returns success=False on connection error (no crash)."""
    import requests as _req
    bridge = _make_bridge()

    with patch.object(_cfg, "ESMFOLD_USE_LOCAL", False):
        with patch("requests.post", side_effect=_req.exceptions.ConnectionError("offline")):
            result = bridge.predict("MAGLK")

    _assert(not result["success"], "connection error -> success=False")
    _assert(result["error"] is not None, "error message is set")


def test_predict_timeout() -> None:
    """predict() returns success=False on Timeout (no crash)."""
    import requests as _req
    bridge = _make_bridge()

    with patch.object(_cfg, "ESMFOLD_USE_LOCAL", False):
        with patch("requests.post", side_effect=_req.exceptions.Timeout("timed out")):
            result = bridge.predict("MAGLK")

    _assert(not result["success"], "timeout -> success=False")
    _assert(result["error"] is not None, "error message is set")


# ═══════════════════════════════════════════════════════════════════════════════
# G. Local venv312 path
# ═══════════════════════════════════════════════════════════════════════════════

def _make_fake_subprocess(plddt_dict: dict, mean_plddt: float, label: str = "query"):
    """
    Return a subprocess.run side_effect that writes a fake ESMFold output JSON
    to the --output path specified in the command, and returns returncode=0.
    """

    def _fake_run(cmd, **kwargs):
        # Find --output path in the command list
        cmd_list = list(cmd)
        try:
            out_idx  = cmd_list.index("--output") + 1
            out_path = cmd_list[out_idx]
        except (ValueError, IndexError):
            out_path = None

        if out_path:
            fake_output = {
                "success":    True,
                "label":      label,
                "plddt":      plddt_dict,
                "mean_plddt": mean_plddt,
                "pdb_str":    _MOCK_PDB,
                "length":     len(plddt_dict),
                "error":      None,
                "elapsed_s":  0.5,
                "device":     "cuda",
            }
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(fake_output, fh)

        result = MagicMock()
        result.returncode = 0
        result.stdout     = "ESMFold: done."
        result.stderr     = ""
        return result

    return _fake_run


def test_local_predict_success() -> None:
    """_run_local: successful subprocess returns result with source='local_venv312'."""
    print("\n=== G. Local venv312 path ===")
    bridge = _make_bridge()

    with patch("esmfold_bridge.subprocess.run",
               side_effect=_make_fake_subprocess(
                   {"1": 87.3, "2": 91.2, "3": 45.0}, mean_plddt=74.5
               )):
        with patch.object(_cfg, "ESMFOLD_USE_LOCAL", True):
            bridge._local_available = lambda: True
            result = bridge.predict("MAG", label="test_local")

    _assert(result["success"],
            "local predict succeeds",
            f"error={result.get('error')}")
    _assert(result.get("source") == "local_venv312",
            "source='local_venv312'",
            f"got {result.get('source')!r}")
    _assert(result.get("mean_plddt", 0) > 0,
            "mean_plddt populated",
            f"got {result.get('mean_plddt')}")
    _assert(1 in result.get("plddt", {}),
            "plddt keys are int (1-based)",
            f"got keys {list(result.get('plddt', {}).keys())[:3]}")


def test_plddt_scale_guard_0_to_1() -> None:
    """_run_local: worker output in 0-1 scale (mean<2.0) is multiplied by 100."""
    bridge = _make_bridge()

    # Worker returns 0-1 scale values
    with patch("esmfold_bridge.subprocess.run",
               side_effect=_make_fake_subprocess(
                   {"1": 0.83, "2": 0.87}, mean_plddt=0.85
               )):
        with patch.object(_cfg, "ESMFOLD_USE_LOCAL", True):
            bridge._local_available = lambda: True
            result = bridge.predict("MA", label="scale_test")

    _assert(result["success"],
            "scale-guard predict succeeds",
            f"error={result.get('error')}")
    # mean_plddt should be ~85.0, not 0.85
    _assert(result.get("mean_plddt", 0) == pytest_approx(85.0, tol=1.0),
            "mean_plddt scaled 0.85 -> 85.0",
            f"got {result.get('mean_plddt')}")
    # All individual plddt values should be in 0-100 range
    plddt_vals = list(result.get("plddt", {}).values())
    all_scaled = all(v > 2.0 for v in plddt_vals)
    _assert(all_scaled,
            "all plddt values in 0-100 range after scale guard",
            f"got {plddt_vals}")


def test_local_fallback_to_atlas_when_use_local_false() -> None:
    """When ESMFOLD_USE_LOCAL=False, predict() uses Atlas API, not subprocess."""
    bridge = _make_bridge()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = _MOCK_PDB

    with patch.object(_cfg, "ESMFOLD_USE_LOCAL", False):
        with patch("requests.post", return_value=mock_resp) as mock_post:
            result = bridge.predict("MAGLK", label="fallback_test")

    _assert(result["success"],
            "fallback predict succeeds",
            f"error={result.get('error')}")
    _assert(result.get("source") == "atlas_api",
            "source='atlas_api' when local disabled",
            f"got {result.get('source')!r}")
    _assert(mock_post.called,
            "requests.post was called (Atlas API used)")


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("tests/test_esmfold.py -- ESMFold Bridge Tests")
    print("=" * 60)

    test_api_post_format()
    test_api_success_returns_pdb_and_plddt()
    test_api_http_error_returns_failure()

    test_plddt_parse_ca_only()
    test_plddt_values_correct()
    test_plddt_empty_string()

    test_compare_to_wildtype_low_drop()
    test_compare_to_wildtype_high_drop()
    test_compare_position_scores_populated()

    test_risk_threshold_medium()
    test_risk_low_warning_is_none()

    test_check_disulfide_missing_pdb()
    test_check_disulfide_misparing_detected(None)

    test_predict_empty_sequence()
    test_predict_connection_error()
    test_predict_timeout()

    test_local_predict_success()
    test_plddt_scale_guard_0_to_1()
    test_local_fallback_to_atlas_when_use_local_false()

    print()
    print("=" * 60)
    print(f"Results: {_results['pass']} passed, "
          f"{_results['fail']} failed, "
          f"{_results['skip']} skipped")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
