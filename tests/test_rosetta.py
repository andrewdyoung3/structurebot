"""
tests/test_rosetta.py
---------------------
Tests for RosettaBridge, MutationScanner, and the related session/router wiring.

Test categories
---------------
  A. Backend detection
       Verify ROSETTA_BACKEND / PYROSETTA_AVAILABLE env vars select the
       correct backend.  Verify PyRosetta stub returns a well-formed error.
       Verify empirical backend produces estimates with confidence="low".

  B. DynaMut2 HTTP mock
       Mock requests.post to verify the correct URL, multipart body format
       (no auth header), and mutation string format are used without
       making live network calls.

  C. MutationScanner candidate selection
       With hand-crafted CamSol + ESM scores, verify the threshold
       logic, substitution generation (Pro/Cys exclusion, solubility
       preference), and max-candidates cap.

  D. Combined score calculation
       Verify the weighting formula and edge cases.

  E. ChimeraX command generation from scan results
       Verify colour assignment, sphere/label commands for top-N, and
       general ChimeraX syntax.

  F. SessionState: stability analysis persistence
       add_rosetta_job / get_rosetta_job / update_rosetta_job /
       list_rosetta_jobs survive save() -> load() round-trip.

  G. ToolRouter: dispatch wiring
       Verify route() augments the result correctly for "rosetta" and
       "mutation_scan" tools; verify error path for missing PDB / mutations.

  H. Full pipeline mock (DynaMut2 mocked, no live network)
       End-to-end through RosettaBridge._run_dynamut2() with requests.post
       replaced by a fake; checks ddg_scores, viz_commands, session storage.

Usage
-----
  python tests/test_rosetta.py                # run all groups
  python tests/test_rosetta.py --detection    # A only
  python tests/test_rosetta.py --mock         # B + H
  python tests/test_rosetta.py --scanner      # C + D + E
  python tests/test_rosetta.py --session      # F
  python tests/test_rosetta.py --router       # G
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import config
config.load_env_file()

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
_results: Dict[str, int] = {"pass": 0, "fail": 0, "skip": 0}


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
        _ok(name, msg)
        return True
    _fail(name, msg or "assertion failed")
    return False


# ── A minimal PDB string for tests that need a real file ─────────────────────
# HIV protease monomer, chain A, residues 1-4 (synthetic).
# Real Rosetta/DynaMut2 calls are mocked in all tests.
_MINIMAL_PDB = textwrap.dedent("""\
    HEADER    HIV PROTEASE (SYNTHETIC FRAGMENT)
    ATOM      1  N   PRO A   1       5.000   5.000   5.000  1.00 10.00           N
    ATOM      2  CA  PRO A   1       5.500   5.500   5.500  1.00 10.00           C
    ATOM      3  C   PRO A   1       6.000   6.000   6.000  1.00 10.00           C
    ATOM      4  O   PRO A   1       6.500   6.500   6.500  1.00 10.00           O
    ATOM      5  N   GLN A   2       7.000   7.000   7.000  1.00 10.00           N
    ATOM      6  CA  GLN A   2       7.500   7.500   7.500  1.00 10.00           C
    ATOM      7  N   ILE A   3       8.000   8.000   8.000  1.00 10.00           N
    ATOM      8  CA  ILE A   3       8.500   8.500   8.500  1.00 10.00           C
    ATOM      9  N   THR A   4       9.000   9.000   9.000  1.00 10.00           N
    ATOM     10  CA  THR A   4       9.500   9.500   9.500  1.00 10.00           C
    END
""")


def _write_temp_pdb() -> str:
    """Write a minimal PDB to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    )
    f.write(_MINIMAL_PDB)
    f.close()
    return f.name


# ════════════════════════════════════════════════════════════════════════════════
# A. Backend detection
# ════════════════════════════════════════════════════════════════════════════════

def test_backend_detection() -> None:
    print("\n--- A. Backend detection ---")

    from rosetta_bridge import _select_backend

    # Force dynamut2 via env var
    os.environ["ROSETTA_BACKEND"] = "dynamut2"
    _assert(_select_backend() == "dynamut2", "ROSETTA_BACKEND=dynamut2 forces dynamut2")
    del os.environ["ROSETTA_BACKEND"]

    # Force pyrosetta via env var (even if not importable)
    os.environ["ROSETTA_BACKEND"] = "pyrosetta"
    _assert(_select_backend() == "pyrosetta", "ROSETTA_BACKEND=pyrosetta forces pyrosetta")
    del os.environ["ROSETTA_BACKEND"]

    # Force empirical
    os.environ["ROSETTA_BACKEND"] = "empirical"
    _assert(_select_backend() == "empirical", "ROSETTA_BACKEND=empirical forces empirical")
    del os.environ["ROSETTA_BACKEND"]

    # Auto mode: PYROSETTA_AVAILABLE not set -> dynamut2
    os.environ.pop("PYROSETTA_AVAILABLE", None)
    os.environ.pop("ROSETTA_BACKEND",    None)
    _assert(
        _select_backend() == "dynamut2",
        "auto mode without PYROSETTA_AVAILABLE -> dynamut2",
    )


def test_pyrosetta_stub_error() -> None:
    """The PyRosetta stub must return success=False with an instructive message."""
    from rosetta_bridge import RosettaBridge

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "pyrosetta"
        os.environ["PYROSETTA_AVAILABLE"] = "true"

        with patch.dict("sys.modules", {"pyrosetta": MagicMock()}):
            bridge = RosettaBridge()
            result = bridge.analyze(
                pdb_path  = pdb_path,
                mutations = [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}],
            )

        _assert(not result.success,       "PyRosetta stub returns success=False")
        _assert(result.error is not None, "PyRosetta stub sets error message")
        _assert(
            "pyrosetta" in (result.error or "").lower() or
            "python" in (result.error or "").lower(),
            "PyRosetta error mentions PyRosetta/Python version",
            repr((result.error or "")[:80]),
        )
    finally:
        os.environ.pop("ROSETTA_BACKEND",    None)
        os.environ.pop("PYROSETTA_AVAILABLE", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_empirical_backend_forced() -> None:
    """ROSETTA_BACKEND=empirical must return estimates with confidence='low'."""
    from rosetta_bridge import RosettaBridge

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "empirical"

        bridge = RosettaBridge()
        result = bridge.analyze(
            pdb_path  = pdb_path,
            mutations = [{"chain": "A", "position": 1, "from_aa": "P", "to_aa": "K"}],
        )

        if not _assert(result.success,
                       "empirical returns success=True",
                       getattr(result, "error", "")):
            return

        _assert(result.data["confidence"] == "low",   "empirical confidence='low'")
        _assert(result.data["backend"] == "empirical","empirical backend label")
        _assert(len(result.data["warnings"]) > 0,     "empirical includes accuracy warning")
        acc_warn = any("accuracy" in w.lower() or "blosum" in w.lower()
                       for w in result.data["warnings"])
        _assert(acc_warn, "warning mentions accuracy/BLOSUM62")
        _assert("P1K" in result.data["ddg_scores"],   "P1K scored")
    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_local_backend_wsl_not_installed() -> None:
    """
    ROSETTA_BACKEND=local when WSL2 is not installed must return
    a helpful error explaining how to install WSL2.
    """
    from rosetta_bridge import RosettaBridge

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "local"

        with patch("wsl_bridge.WSLBridge.is_available", return_value=False):
            bridge = RosettaBridge()
            result = bridge.analyze(
                pdb_path  = pdb_path,
                mutations = [{"chain": "A", "position": 1, "from_aa": "P", "to_aa": "K"}],
            )

        _assert(not result.success, "local backend without WSL2 -> success=False")
        err = (result.error or "").lower()
        _assert(
            "wsl" in err or "ubuntu" in err or "not installed" in err,
            "error mentions WSL2 installation",
            repr((result.error or "")[:120]),
        )
    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_local_backend_wsl_no_pyrosetta() -> None:
    """
    ROSETTA_BACKEND=local with WSL2 available but PyRosetta absent
    must return an error explaining how to install PyRosetta.
    """
    from rosetta_bridge import RosettaBridge

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "local"

        with patch("wsl_bridge.WSLBridge.is_available",   return_value=True), \
             patch("wsl_bridge.WSLBridge.check_pyrosetta", return_value=False):
            bridge = RosettaBridge()
            result = bridge.analyze(
                pdb_path  = pdb_path,
                mutations = [{"chain": "A", "position": 1, "from_aa": "P", "to_aa": "K"}],
            )

        _assert(not result.success, "local backend without PyRosetta -> success=False")
        err = (result.error or "").lower()
        _assert(
            "pyrosetta" in err,
            "error mentions PyRosetta installation",
            repr((result.error or "")[:120]),
        )
    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_backend_status_local_no_wsl() -> None:
    """backend_status() for local backend without WSL2 mentions WSL2 install."""
    from rosetta_bridge import RosettaBridge

    os.environ["ROSETTA_BACKEND"] = "local"
    try:
        with patch("wsl_bridge.WSLBridge.is_available", return_value=False):
            bridge  = RosettaBridge()
            status  = bridge.backend_status().lower()
        _assert(
            "wsl" in status or "not installed" in status,
            "backend_status mentions WSL2 when unavailable",
            f"got: {bridge.backend_status()!r}",
        )
    finally:
        os.environ.pop("ROSETTA_BACKEND", None)


def test_missing_pdb_error() -> None:
    """Non-existent PDB path must produce a clear error before any API call."""
    from rosetta_bridge import RosettaBridge

    os.environ["ROSETTA_BACKEND"] = "dynamut2"
    try:
        bridge = RosettaBridge()
        result = bridge.analyze(
            pdb_path  = "/tmp/does_not_exist_at_all.pdb",
            mutations = [{"chain": "A", "position": 1, "from_aa": "P", "to_aa": "K"}],
        )
        _assert(not result.success,           "missing PDB -> success=False")
        _assert("not found" in (result.error or "").lower() or
                "pdb" in (result.error or "").lower(),
                "error mentions PDB file",
                repr((result.error or "")[:80]))
    finally:
        os.environ.pop("ROSETTA_BACKEND", None)


# ════════════════════════════════════════════════════════════════════════════════
# B. DynaMut2 HTTP mock -- correct request format
# ════════════════════════════════════════════════════════════════════════════════

def test_dynamut2_request_format() -> None:
    """
    Verify _query_dynamut2_single:
      - POSTs to the correct submit URL
      - Body has chain + mutation fields (no Authorization header)
      - GETs the result URL with job_id param
      - Returns the 'prediction' float
    """
    print("\n--- B. DynaMut2 HTTP mock ---")

    from rosetta_bridge import RosettaBridge, _DYNAMUT2_SUBMIT_URL, _DYNAMUT2_RESULT_URL

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "dynamut2"

        mut = {"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}

        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"job_id": "test-job-abc"}

        result_resp = MagicMock(status_code=200)
        result_resp.raise_for_status = MagicMock()
        result_resp.json.return_value = {
            "prediction": 1.47, "chain": "A", "res_number": 82,
            "wild-type": "V", "mutant": "A",
        }

        with patch("requests.post", return_value=submit_resp) as mock_post, \
             patch("requests.get",  return_value=result_resp) as mock_get, \
             patch("time.sleep"):
            bridge = RosettaBridge()
            ddg = bridge._query_dynamut2_single(pdb_path, mut, progress=lambda s: None)

        # --- Submit call checks ---
        _assert(mock_post.called, "requests.post was called (submit)")
        post_kwargs = mock_post.call_args
        post_url = (
            post_kwargs.args[0]
            if post_kwargs.args
            else post_kwargs.kwargs.get("url", "")
        )
        _assert(str(post_url) == _DYNAMUT2_SUBMIT_URL,
                "POST to correct submit URL", str(post_url))

        # No auth header
        headers = post_kwargs.kwargs.get("headers", {}) or {}
        _assert("Authorization" not in headers,
                "no Authorization header (DynaMut2 is public)")

        post_data = post_kwargs.kwargs.get("data", {}) or {}
        _assert(post_data.get("chain") == "A",       "chain in body", str(post_data))
        _assert(post_data.get("mutation") == "V82A", "mutation=V82A in body", str(post_data))
        files = post_kwargs.kwargs.get("files", {}) or {}
        _assert("pdb_file" in files, "pdb_file in multipart files")

        # --- Poll call checks ---
        _assert(mock_get.called, "requests.get was called (poll)")
        get_kwargs = mock_get.call_args
        get_url = (
            get_kwargs.args[0]
            if get_kwargs.args
            else get_kwargs.kwargs.get("url", "")
        )
        _assert(str(get_url) == _DYNAMUT2_RESULT_URL,
                "GET to correct result URL", str(get_url))
        get_params = get_kwargs.kwargs.get("params", {}) or {}
        _assert(get_params.get("job_id") == "test-job-abc",
                "job_id param passed to GET", str(get_params))

        _assert(ddg == 1.47, "prediction parsed from result response", str(ddg))

    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_dynamut2_response_parsing() -> None:
    """_parse_dynamut2_result should extract 'prediction' field."""
    from rosetta_bridge import _parse_dynamut2_result

    # Normal completed result
    _assert(
        _parse_dynamut2_result(
            {"prediction": 1.47, "chain": "A", "res_number": 82}, "V82A"
        ) == 1.47,
        "parses 'prediction' from complete result"
    )
    # Negative value
    _assert(
        _parse_dynamut2_result({"prediction": -0.82, "chain": "A"}, "L10K") == -0.82,
        "parses negative prediction"
    )


def test_dynamut2_rate_limit_retry() -> None:
    """429 on submit triggers retry with backoff; succeeds on second attempt."""
    from rosetta_bridge import RosettaBridge

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "dynamut2"

        mut = {"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}

        rate_limited = MagicMock(status_code=429)
        rate_limited.headers = {"Retry-After": "0"}

        ok_submit = MagicMock(status_code=200)
        ok_submit.json.return_value = {"job_id": "retry-job-1"}

        ok_result = MagicMock(status_code=200)
        ok_result.raise_for_status = MagicMock()
        ok_result.json.return_value = {"prediction": 1.47}

        with patch("requests.post", side_effect=[rate_limited, ok_submit]), \
             patch("requests.get",  return_value=ok_result), \
             patch("time.sleep"):
            bridge = RosettaBridge()
            ddg = bridge._query_dynamut2_single(pdb_path, mut, progress=lambda s: None)

        _assert(ddg == 1.47, "succeeds after one 429 retry", str(ddg))

    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


# ────────────────────────────────────────────────────────────────────────────────
# B2. DynaMut2 single-mutation poll response format (current "status" API)
# ────────────────────────────────────────────────────────────────────────────────

def test_dynamut2_single_status_done_parsed() -> None:
    """Current API: {"status": "DONE", "prediction": 0.789, ...} -> ddg == 0.789."""
    from rosetta_bridge import _parse_dynamut2_result

    data = {
        "status": "DONE", "prediction": 0.789, "chain": "A",
        "wild-type": "ILE", "mutant": "ARG", "position": "72",
        "results_page": "http://example/x",
    }
    ddg = _parse_dynamut2_result(data, "I72R")
    _assert(ddg == 0.789, "status=DONE prediction parsed", str(ddg))
    assert ddg == 0.789


def test_dynamut2_single_status_running_continues() -> None:
    """
    {"status": "RUNNING"} must be treated as still-running, NOT a result:
    the parser raises, and the poll loop keeps polling until DONE.
    """
    from rosetta_bridge import RosettaBridge, _parse_dynamut2_result

    # Parser: RUNNING is not a completed result (raises, does not return a value).
    running_raises = False
    try:
        _parse_dynamut2_result({"status": "RUNNING", "job_id": "j1"}, "I72R")
    except ValueError:
        running_raises = True
    _assert(running_raises, "status=RUNNING raises (not parsed as result)")
    assert running_raises

    # Poll loop: RUNNING then DONE -> two GETs, returns the final prediction.
    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "dynamut2"
        mut = {"chain": "A", "position": 72, "from_aa": "I", "to_aa": "R"}

        submit_resp = MagicMock(status_code=200)
        submit_resp.json.return_value = {"job_id": "job-running-1"}

        running_resp = MagicMock(status_code=200)
        running_resp.raise_for_status = MagicMock()
        running_resp.json.return_value = {"status": "RUNNING", "job_id": "job-running-1"}

        done_resp = MagicMock(status_code=200)
        done_resp.raise_for_status = MagicMock()
        done_resp.json.return_value = {"status": "DONE", "prediction": 0.789}

        with patch("requests.post", return_value=submit_resp), \
             patch("requests.get",  side_effect=[running_resp, done_resp]) as mock_get, \
             patch("time.sleep"):
            bridge = RosettaBridge()
            ddg = bridge._query_dynamut2_single(pdb_path, mut, progress=lambda s: None)

        _assert(mock_get.call_count == 2,
                "polled twice (RUNNING then DONE)", str(mock_get.call_count))
        _assert(ddg == 0.789, "returns prediction after RUNNING poll", str(ddg))
        assert mock_get.call_count == 2 and ddg == 0.789
    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_dynamut2_single_prediction_string_cast() -> None:
    """String prediction form {"status":"DONE","prediction":"1.4"} -> 1.4 (float)."""
    from rosetta_bridge import _parse_dynamut2_result

    ddg = _parse_dynamut2_result({"status": "DONE", "prediction": "1.4"}, "I72R")
    _assert(ddg == 1.4, "string prediction cast to float", str(ddg))
    _assert(isinstance(ddg, float), "extracted ddg is a float")
    assert ddg == 1.4 and isinstance(ddg, float)


def test_dynamut2_single_error_not_zero() -> None:
    """
    {"status": "ERROR"} must be flagged as a failure (raises), never silently
    parsed as a real ddG of 0.0 — a failed lookup must stay distinguishable.
    """
    from rosetta_bridge import _parse_dynamut2_result

    extracted: Any = "sentinel"
    raised = False
    try:
        extracted = _parse_dynamut2_result(
            {"status": "ERROR", "job_id": "j-err", "message": "internal failure"},
            "I72R",
        )
    except (RuntimeError, ValueError):
        raised = True
    _assert(raised, "status=ERROR raises instead of returning a value")
    _assert(extracted != 0.0, "ERROR is NOT parsed as ddg 0.0", repr(extracted))
    assert raised and extracted != 0.0


# ════════════════════════════════════════════════════════════════════════════════
# C. MutationScanner -- candidate selection
# ════════════════════════════════════════════════════════════════════════════════

def _make_mock_session(
    camsol_scores: Dict[int, float],
    esm_scores:    Dict[int, float],
    sequence:      str = "",
) -> Any:
    """Return a minimal mock session with pre-loaded tool results."""
    class _MockSession:
        def __init__(self) -> None:
            self._camsol = {"scores": {str(k): v for k, v in camsol_scores.items()}}
            self._esm    = {"conservation": {str(k): v for k, v in esm_scores.items()}}
            self._seq    = sequence

        def get_tool_result(self, tool: str, model_id: str) -> Any:
            if tool == "camsol": return self._camsol
            if tool == "esm":    return self._esm
            return None

        def get_structure(self, model_id: str) -> Any:
            if self._seq:
                return {"name": "TEST", "path": None,
                        "metadata": {"sequence": self._seq}, "loaded_at": ""}
            return None

        def add_tool_result(self, *a, **kw) -> None: pass
        def add_scan_result(self, *a, **kw) -> None: pass

    return _MockSession()


def test_scanner_candidate_selection() -> None:
    print("\n--- C. MutationScanner candidate selection ---")

    from mutation_scanner import MutationScanner

    seq = "ACIFGHIKLM"
    camsol_scores = {
        1: 0.5, 2: 0.3, 3: -1.2,   # position 3 = aggregation-prone
        4: 0.1, 5: 0.2, 6: 0.0,
        7: 0.4, 8: 0.3, 9: -0.1, 10: 0.5,
    }
    esm_scores = {
        1: 0.9, 2: 0.8, 3: 0.1,    # position 3 = evolutionarily tolerant
        4: 0.7, 5: 0.6, 6: 0.5,
        7: 0.4, 8: 0.3, 9: 0.2, 10: 0.1,
    }

    session = _make_mock_session(camsol_scores, esm_scores, sequence=seq)
    scanner = MutationScanner(session=session, model_id="1")

    pdb_path = _write_temp_pdb()
    try:
        with patch("rosetta_bridge.RosettaBridge") as MockRosetta:
            mock_bridge = MagicMock()
            mock_bridge.analyze.return_value = MagicMock(
                success=True,
                data={"ddg_scores": {}},
            )
            MockRosetta.return_value = mock_bridge

            results = scanner.scan(pdb_path=pdb_path, chain_id="A", sequence=seq)

        positions = {r["position"] for r in results}
        _assert(3 in positions, "position 3 (I, aggregation-prone + tolerant) is a candidate")
        _assert(1 not in positions, "position 1 (CamSol=+0.5) is not a candidate")
        _assert(2 not in positions, "position 2 (CamSol=+0.3) is not a candidate")

        bad = [r for r in results if r["to_aa"] in ("P", "C")]
        _assert(len(bad) == 0, "no Pro or Cys substitution candidates")

        for r in results:
            expected_from = seq[r["position"] - 1]
            _assert(
                r["from_aa"] == expected_from,
                f"from_aa matches sequence at position {r['position']}",
                f"{r['from_aa']} == {expected_from}",
            )
    finally:
        Path(pdb_path).unlink(missing_ok=True)


def test_scanner_protected_residues() -> None:
    """Protected (binding-site) residues must be excluded from candidates."""
    from mutation_scanner import MutationScanner

    seq = "IIIIII"
    scores = {i: -1.0 for i in range(1, 7)}
    esm    = {i: 0.1  for i in range(1, 7)}

    session = _make_mock_session(scores, esm, sequence=seq)
    scanner = MutationScanner(session=session, model_id="1")

    pdb_path = _write_temp_pdb()
    try:
        with patch("rosetta_bridge.RosettaBridge") as MockRosetta:
            mock_bridge = MagicMock()
            mock_bridge.analyze.return_value = MagicMock(
                success=True, data={"ddg_scores": {}}
            )
            MockRosetta.return_value = mock_bridge

            results = scanner.scan(
                pdb_path = pdb_path,
                chain_id = "A",
                sequence = seq,
                filters  = {"binding_site_residues": [1, 2, 3]},
            )

        positions = {r["position"] for r in results}
        for p in (1, 2, 3):
            _assert(p not in positions, f"protected position {p} excluded")
        for p in (4, 5, 6):
            _assert(p in positions, f"unprotected position {p} is a candidate")
    finally:
        Path(pdb_path).unlink(missing_ok=True)


def test_scanner_max_candidates_cap() -> None:
    """max_candidates filter must limit total output."""
    from mutation_scanner import MutationScanner

    seq = "I" * 20
    scores = {i: -1.0 for i in range(1, 21)}
    esm    = {i: 0.1  for i in range(1, 21)}

    session = _make_mock_session(scores, esm, sequence=seq)
    scanner = MutationScanner(session=session, model_id="1")

    pdb_path = _write_temp_pdb()
    try:
        with patch("rosetta_bridge.RosettaBridge") as MockRosetta:
            mock_bridge = MagicMock()
            mock_bridge.analyze.return_value = MagicMock(
                success=True, data={"ddg_scores": {}}
            )
            MockRosetta.return_value = mock_bridge

            results = scanner.scan(
                pdb_path = pdb_path,
                chain_id = "A",
                sequence = seq,
                filters  = {"max_candidates": 5},
            )

        _assert(len(results) <= 5, "max_candidates=5 caps output", str(len(results)))
    finally:
        Path(pdb_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# D. Combined score calculation
# ════════════════════════════════════════════════════════════════════════════════

def test_combined_score() -> None:
    print("\n--- D. Combined score ---")

    from mutation_scanner import combined_score

    high = combined_score(ddg=-2.0, camsol_delta=2.0, esm_tolerance=0.9)
    _assert(high > 1.0, "stabilising+soluble+tolerant -> score > 1.0", f"{high:.4f}")

    low = combined_score(ddg=2.0, camsol_delta=0.0, esm_tolerance=0.1)
    _assert(low < 0.0, "destabilising+no-improvement+conserved -> score < 0", f"{low:.4f}")

    neutral = combined_score(ddg=0.0, camsol_delta=0.0, esm_tolerance=0.0)
    _assert(abs(neutral) < 0.01, "all-zero inputs -> score ~0", f"{neutral:.4f}")

    score_ddg_only = combined_score(
        ddg=-1.0, camsol_delta=999.0, esm_tolerance=999.0,
        w_ddg=1.0, w_sol=0.0, w_tol=0.0,
    )
    _assert(
        abs(score_ddg_only - 1.0) < 0.01,
        "custom weights: only ddg counts",
        f"{score_ddg_only:.4f}",
    )


def test_hydrophobicity_delta() -> None:
    """Hydrophobic->charged substitution should yield positive camsol_delta."""
    from mutation_scanner import _estimate_camsol_delta

    delta_IK = _estimate_camsol_delta("I", "K")
    delta_KI = _estimate_camsol_delta("K", "I")

    _assert(delta_IK > 0, "I->K gives positive camsol_delta (improves solubility)",
            f"delta={delta_IK:.3f}")
    _assert(delta_KI < 0, "K->I gives negative camsol_delta (worsens solubility)",
            f"delta={delta_KI:.3f}")


# ════════════════════════════════════════════════════════════════════════════════
# E. ChimeraX command generation
# ════════════════════════════════════════════════════════════════════════════════

def test_chimerax_commands_from_scan() -> None:
    print("\n--- E. ChimeraX command generation ---")

    from mutation_scanner import MutationScanner
    from session_state import SessionState

    session = SessionState()
    scanner = MutationScanner(session=session, model_id="1")

    scan_results = [
        {
            "position": 75, "chain": "A", "from_aa": "L", "to_aa": "K",
            "ddg": -0.8, "solubility_delta": 1.2, "esm_tolerance": 0.85,
            "combined_score": 2.3,
            "camsol_score": -0.7,
            "recommendation": "Strong candidate",
        },
        {
            "position": 40, "chain": "A", "from_aa": "I", "to_aa": "E",
            "ddg": -0.2, "solubility_delta": 0.9, "esm_tolerance": 0.70,
            "combined_score": 0.9,
            "camsol_score": -0.6,
            "recommendation": "Good candidate",
        },
    ]

    cmds, exps = scanner.generate_chimerax_commands(scan_results, top_n=1)

    _assert(len(cmds) > 0,         "commands list is non-empty")
    _assert(len(cmds) == len(exps), "commands and explanations same length")
    _assert(any("cartoon" in c for c in cmds), "includes cartoon command")
    _assert(any("color #1" in c for c in cmds), "includes model-specific color reset")
    _assert(any("sphere" in c and ":75" in c for c in cmds),
            "top candidate gets sphere command")
    _assert(any("label" in c and ":75" in c for c in cmds),
            "top candidate gets label command")
    _assert(any(":40" in c for c in cmds),  "position 40 coloured")

    bad = [c for c in cmds if c.strip().startswith("background")]
    _assert(len(bad) == 0, "no deprecated 'background' commands", str(bad))


def test_empty_scan_no_commands() -> None:
    """generate_chimerax_commands on an empty list returns empty lists."""
    from mutation_scanner import MutationScanner
    from session_state import SessionState

    scanner = MutationScanner(session=SessionState(), model_id="1")
    cmds, exps = scanner.generate_chimerax_commands([], top_n=5)
    _assert(cmds == [],  "empty scan -> empty commands list")
    _assert(exps == [],  "empty scan -> empty explanations list")


# ════════════════════════════════════════════════════════════════════════════════
# F. SessionState: stability analysis persistence
# ════════════════════════════════════════════════════════════════════════════════

def test_session_rosetta_jobs() -> None:
    print("\n--- F. SessionState stability analysis persistence ---")

    from session_state import SessionState

    session = SessionState()

    job_data = {
        "mutations":    [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}],
        "pdb_path":     "/tmp/test.pdb",
        "backend":      "dynamut2",
        "submitted_at": "2026-05-26T12:00:00",
        "status":       "completed",
        "results":      {"V82A": 1.47},
    }

    session.add_rosetta_job("42", job_data)
    _assert(session.get_rosetta_job("42") is not None, "job stored and retrieved")

    session.update_rosetta_job("42", {"status": "completed"})
    _assert(session.get_rosetta_job("42")["status"] == "completed",
            "update_rosetta_job works")

    _assert(session.get_rosetta_job("999") is None,
            "get_rosetta_job returns None for unknown job")

    all_jobs = session.list_rosetta_jobs()
    _assert("42" in all_jobs, "list_rosetta_jobs includes our job")

    session.clear_rosetta_job("42")
    _assert(session.get_rosetta_job("42") is None, "clear_rosetta_job removes job")


def test_session_rosetta_jobs_persistence() -> None:
    """Stability analysis results survive save() -> load()."""
    from session_state import SessionState

    session = SessionState()
    session.add_rosetta_job("99", {
        "mutations": [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}],
        "pdb_path":  "/tmp/test.pdb",
        "backend":   "dynamut2",
        "submitted_at": "2026-05-26T12:00:00",
        "status":    "completed",
        "results":   {"V82A": 1.47},
    })
    session.add_scan_result("1", [{"position": 75, "combined_score": 2.3}])

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as fh:
        tmp = fh.name
    try:
        session.save(tmp)
        loaded = SessionState.load(tmp)

        job = loaded.get_rosetta_job("99")
        _assert(job is not None,                    "job persists after save/load")
        _assert(job["status"] == "completed",       "job status preserved")
        _assert(job["results"].get("V82A") == 1.47, "job results preserved")
        _assert(job["backend"] == "dynamut2",       "backend label preserved")

        scan = loaded.get_scan_result("1")
        _assert(scan is not None,                   "scan result persists")
        _assert(len(scan) == 1,                     "scan result length preserved")
    finally:
        Path(tmp).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# G. ToolRouter dispatch wiring
# ════════════════════════════════════════════════════════════════════════════════

def test_router_route_rosetta() -> None:
    print("\n--- G. ToolRouter dispatch ---")

    from session_state import SessionState
    from tool_router import ToolRouter

    session = SessionState()
    session.add_structure("1", "1HSG")

    class _MockBridge:
        def is_running(self): return False

    router = ToolRouter(_MockBridge(), session)

    result = {
        "commands": [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "high",
        "tools_needed": ["rosetta"],
        "tool_inputs":  {"rosetta": {"model_id": "1", "mutations": [
            {"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}
        ]}},
    }
    routed = router.route(result)
    _assert(routed.get("has_extra_tools") is True, "rosetta: has_extra_tools=True")
    steps = routed.get("tool_steps_info", [])
    _assert(any(s["tool"] == "rosetta" for s in steps),
            "rosetta step present in tool_steps_info")


def test_router_route_mutation_scan() -> None:
    from session_state import SessionState
    from tool_router import ToolRouter

    session = SessionState()
    session.add_structure("1", "1HSG")

    class _MockBridge:
        def is_running(self): return False

    router = ToolRouter(_MockBridge(), session)

    result = {
        "commands": [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "high",
        "tools_needed": ["mutation_scan"],
        "tool_inputs":  {"mutation_scan": {"model_id": "1", "chain": "A",
                                           "focus": "solubility"}},
    }
    routed = router.route(result)
    _assert(routed.get("has_extra_tools") is True, "mutation_scan: has_extra_tools=True")
    steps = routed.get("tool_steps_info", [])
    _assert(any(s["tool"] == "mutation_scan" for s in steps),
            "mutation_scan step present in tool_steps_info")


def test_router_rosetta_no_mutations_error() -> None:
    """rosetta tool without mutations in tool_inputs returns a clear error."""
    from session_state import SessionState
    from tool_router import ToolRouter

    session = SessionState()
    session.add_structure("1", "1HSG")

    class _MockBridge:
        def is_running(self): return False
        def run_command(self, cmd): return {"value": "", "error": None}

    router = ToolRouter(_MockBridge(), session)
    routed = {
        "commands": [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "high",
        "tools_needed": ["rosetta"],
        "tool_inputs":  {"rosetta": {"model_id": "1"}},  # no mutations!
        "has_extra_tools": True,
        "tool_steps_info": [],
    }
    executed = router.execute(routed)
    _assert(executed.get("pipeline_success") is False,
            "rosetta without mutations -> pipeline_success=False")
    err = executed.get("pipeline_error", "")
    _assert("mutation" in err.lower(),
            "error message mentions 'mutations'", repr(err[:80]))


# ════════════════════════════════════════════════════════════════════════════════
# H. Full pipeline mock (DynaMut2 mocked, no live network)
# ════════════════════════════════════════════════════════════════════════════════

def test_dynamut2_full_pipeline_mock() -> None:
    """
    End-to-end through RosettaBridge._run_dynamut2() with requests.post mocked.

    Verifies:
      - Two mutations send two POST requests (one each)
      - ddg_scores populated from mocked responses
      - confidence="high" (all DynaMut2, no fallback)
      - backend="dynamut2"
      - viz_commands generated
      - session.rosetta_jobs has a completed entry
    """
    print("\n--- H. Full pipeline mock (DynaMut2) ---")

    from rosetta_bridge import RosettaBridge
    from session_state import SessionState

    pdb_path = _write_temp_pdb()
    session  = SessionState()

    mutations = [
        {"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"},
        {"chain": "A", "position": 10, "from_aa": "L", "to_aa": "K"},
    ]

    # Two-step async per mutation:
    # POST /prediction_single  -> {"job_id": "..."}
    # GET  /prediction_single?job_id=... -> {"prediction": ...}
    submit_v82a = MagicMock(status_code=200)
    submit_v82a.json.return_value = {"job_id": "job-v82a"}

    submit_l10k = MagicMock(status_code=200)
    submit_l10k.json.return_value = {"job_id": "job-l10k"}

    result_v82a = MagicMock(status_code=200)
    result_v82a.raise_for_status = MagicMock()
    result_v82a.json.return_value = {"prediction": 1.47}

    result_l10k = MagicMock(status_code=200)
    result_l10k.raise_for_status = MagicMock()
    result_l10k.json.return_value = {"prediction": -0.82}

    try:
        os.environ["ROSETTA_BACKEND"] = "dynamut2"

        with patch("requests.post", side_effect=[submit_v82a, submit_l10k]), \
             patch("requests.get",  side_effect=[result_v82a, result_l10k]), \
             patch("time.sleep"):
            bridge = RosettaBridge()
            result = bridge.analyze(
                pdb_path  = pdb_path,
                mutations = mutations,
                session   = session,
            )

        _assert(result.success,                          "mocked pipeline returns success=True")
        ddg = result.data.get("ddg_scores", {})
        _assert("V82A" in ddg,                           "V82A in ddg_scores")
        _assert("L10K" in ddg,                           "L10K in ddg_scores")
        _assert(ddg["V82A"] == 1.47,                     "V82A ddg = 1.47", str(ddg["V82A"]))
        _assert(ddg["L10K"] == -0.82,                    "L10K ddg = -0.82", str(ddg["L10K"]))
        _assert(result.data["confidence"] == "high",     "confidence=high (all DynaMut2)")
        _assert(result.data["backend"] == "dynamut2",    "backend=dynamut2")
        _assert(result.data["job_id"] is None,           "job_id=None (synchronous)")

        # Session: a completed entry must exist
        jobs = session.list_rosetta_jobs()
        _assert(len(jobs) >= 1,                          "session has at least one job entry")
        latest = list(jobs.values())[-1]
        _assert(latest["status"] == "completed",         "job status=completed")
        _assert(latest["backend"] in ("dynamut2", "dynamut2+empirical"),
                "job backend=dynamut2*")
        _assert("V82A" in latest.get("results", {}),     "V82A in session job results")

        # Viz commands
        _assert(len(result.viz_commands) > 0,            "viz_commands non-empty")
        _assert(any("color" in c for c in result.viz_commands),
                "viz includes color commands")

        # Summary should mention the most-stabilising mutation (L10K, ddg=-0.82)
        _assert("L10K" in result.summary,                "summary mentions most stabilising mutation")

    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_dynamut2_empirical_fallback_per_mutation() -> None:
    """
    When DynaMut2 fails for one mutation but not another, the successful one
    uses DynaMut2 and the failed one falls back to empirical.
    Result should have confidence='medium' and backend='dynamut2+empirical'.
    """
    from rosetta_bridge import RosettaBridge
    from session_state import SessionState
    import requests

    pdb_path = _write_temp_pdb()
    session  = SessionState()

    mutations = [
        {"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"},
        {"chain": "A", "position": 10, "from_aa": "L", "to_aa": "K"},
    ]

    ok_submit = MagicMock(status_code=200)
    ok_submit.json.return_value = {"job_id": "job-v82a-ok"}

    ok_result = MagicMock(status_code=200)
    ok_result.raise_for_status = MagicMock()
    ok_result.json.return_value = {"prediction": 1.47}

    try:
        os.environ["ROSETTA_BACKEND"] = "dynamut2"

        # First mutation (V82A): submit OK + result OK
        # Second mutation (L10K): submit raises ConnectionError on all retries
        call_count = {"n": 0}

        def post_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:   # first mutation submit
                return ok_submit
            raise requests.exceptions.ConnectionError("simulated network failure")

        with patch("requests.post", side_effect=post_side_effect), \
             patch("requests.get",  return_value=ok_result), \
             patch("time.sleep"):
            bridge = RosettaBridge()
            result = bridge.analyze(
                pdb_path  = pdb_path,
                mutations = mutations,
                session   = session,
            )

        _assert(result.success,                                "mixed pipeline success=True")
        _assert(result.data["confidence"] == "medium",         "confidence=medium (mixed)")
        _assert("dynamut2" in result.data["backend"],          "backend contains dynamut2")
        _assert("empirical" in result.data["backend"],         "backend contains empirical")
        _assert(len(result.data["warnings"]) > 0,              "warnings present")
        _assert("V82A" in result.data["ddg_scores"],           "V82A scored (DynaMut2)")
        _assert("L10K" in result.data["ddg_scores"],           "L10K scored (empirical)")

    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# I. Parallel DynaMut2 scoring
# ════════════════════════════════════════════════════════════════════════════════

def _make_parallel_pdb() -> str:
    """Write a minimal PDB to a temp file; return path string."""
    import tempfile
    pdb = textwrap.dedent("""\
        ATOM      1  CA  ILE A  64       1.000   1.000   1.000  1.00 20.00           C
        ATOM      2  CA  GLY A  73       2.000   2.000   2.000  1.00 30.00           C
        ATOM      3  CA  LEU A  63       3.000   3.000   3.000  1.00 25.00           C
        ATOM      4  CA  THR A  74       4.000   4.000   4.000  1.00 35.00           C
        END
    """)
    tf = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    tf.write(pdb)
    tf.close()
    return tf.name


def _mock_post_factory(ddg_map: Dict[str, float]):
    """
    Build a requests.post mock that returns DynaMut2 results from ddg_map.
    Submit step: returns {"job_id": "job-<mut>"}
    Poll step:   returns {"prediction": ddg_map[mut]}
    """
    call_log: List[str] = []

    def _mock_post(url, files=None, data=None, params=None, timeout=None, **kw):
        resp = MagicMock()
        resp.status_code = 200
        if files is not None:
            # Submit step
            mut_str = data.get("mutation", "X1Y") if data else "X1Y"
            resp.json.return_value = {"job_id": f"job-{mut_str}"}
            call_log.append(("submit", mut_str))
        else:
            # Poll step (GET via requests.post shouldn't happen — handle via get)
            resp.json.return_value = {"prediction": 0.0}
        return resp

    def _mock_get(url, params=None, timeout=None, **kw):
        resp = MagicMock()
        resp.status_code = 200
        job_id = (params or {}).get("job_id", "job-X1Y")
        # Extract mutation from job_id: "job-V82A" -> "V82A"
        mut = job_id.replace("job-", "")
        ddg = ddg_map.get(mut, 0.0)
        resp.json.return_value = {"prediction": ddg}
        call_log.append(("poll", mut))
        return resp

    return _mock_post, _mock_get, call_log


def test_parallel_all_mutations_scored() -> None:
    """_score_mutations_parallel returns a score for every mutation submitted."""
    print("\n=== I. Parallel DynaMut2 ===")
    pdb = _make_parallel_pdb()
    mutations = [
        {"chain": "A", "position": 64, "from_aa": "I", "to_aa": "E"},
        {"chain": "A", "position": 73, "from_aa": "G", "to_aa": "K"},
        {"chain": "A", "position": 63, "from_aa": "L", "to_aa": "K"},
        {"chain": "A", "position": 74, "from_aa": "T", "to_aa": "R"},
    ]
    ddg_map = {"I64E": -3.53, "G73K": -0.82, "L63K": -1.12, "T74R": 0.23}
    mock_post, mock_get, call_log = _mock_post_factory(ddg_map)

    from rosetta_bridge import RosettaBridge
    bridge = RosettaBridge.__new__(RosettaBridge)
    bridge._backend = "dynamut2"

    with patch("requests.post", mock_post), patch("requests.get", mock_get):
        ddg_scores, fallbacks, warnings = bridge._score_mutations_parallel(
            pdb, mutations, max_workers=4,
            scan_deadline=None, progress=lambda _: None
        )

    import os as _os
    _os.unlink(pdb)

    _assert(len(ddg_scores) == 4, "4 mutations all scored",
            f"got {len(ddg_scores)}: {ddg_scores}")
    for mut in mutations:
        key = f"{mut['from_aa']}{mut['position']}{mut['to_aa']}"
        _assert(key in ddg_scores,
                f"score present for {key}",
                f"missing from {list(ddg_scores.keys())}")


def test_parallel_circuit_breaker_uses_empirical() -> None:
    """Parallel scoring: circuit breaker trips -> remaining mutations use empirical."""
    pdb = _make_parallel_pdb()
    mutations = [
        {"chain": "A", "position": 64, "from_aa": "I", "to_aa": "E"},
        {"chain": "A", "position": 73, "from_aa": "G", "to_aa": "K"},
        {"chain": "A", "position": 63, "from_aa": "L", "to_aa": "K"},
    ]

    from rosetta_bridge import RosettaBridge, _DYNAMUT2_CIRCUIT_BREAKER
    bridge = RosettaBridge.__new__(RosettaBridge)
    bridge._backend = "dynamut2"

    # _query_dynamut2_single always raises -> circuit breaker trips
    with patch.object(bridge, "_query_dynamut2_single",
                      side_effect=ConnectionError("server down")):
        ddg_scores, fallbacks, warnings = bridge._score_mutations_parallel(
            pdb, mutations, max_workers=4,
            scan_deadline=None, progress=lambda _: None
        )

    import os as _os
    _os.unlink(pdb)

    _assert(len(ddg_scores) == 3, "all 3 mutations have a score (empirical)",
            f"got {len(ddg_scores)}")
    _assert(len(fallbacks) == 3, "all 3 mutations in empirical_fallbacks",
            f"got {len(fallbacks)}")
    _assert(any("circuit" in w.lower() for w in warnings),
            "circuit breaker warning present",
            f"got warnings: {warnings}")


def test_parallel_wall_time_faster_than_sequential() -> None:
    """
    Parallel scoring (4 workers) completes faster than sequential for 4 mutations
    when each mutation takes ~0.1s.

    Uses a mock that sleeps briefly to simulate real latency.
    """
    import time

    pdb = _make_parallel_pdb()
    mutations = [
        {"chain": "A", "position": 64, "from_aa": "I", "to_aa": "E"},
        {"chain": "A", "position": 73, "from_aa": "G", "to_aa": "K"},
        {"chain": "A", "position": 63, "from_aa": "L", "to_aa": "K"},
        {"chain": "A", "position": 74, "from_aa": "T", "to_aa": "R"},
    ]

    def _slow_mock(pdb_path, mut, progress_fn, deadline):
        time.sleep(0.10)   # simulate 100ms DynaMut2 latency
        return -1.0

    from rosetta_bridge import RosettaBridge
    bridge = RosettaBridge.__new__(RosettaBridge)
    bridge._backend = "dynamut2"

    with patch.object(bridge, "_query_dynamut2_single", side_effect=_slow_mock):
        t0 = time.perf_counter()
        scores_par, _, _ = bridge._score_mutations_parallel(
            pdb, mutations, max_workers=4,
            scan_deadline=None, progress=lambda _: None
        )
        wall_par = time.perf_counter() - t0

    with patch.object(bridge, "_query_dynamut2_single", side_effect=_slow_mock):
        t0 = time.perf_counter()
        scores_seq, _, _ = bridge._score_mutations_sequential(
            pdb, mutations,
            scan_deadline=None, progress=lambda _: None
        )
        wall_seq = time.perf_counter() - t0

    import os as _os
    _os.unlink(pdb)

    _assert(len(scores_par) == 4, "parallel: 4 mutations scored",
            f"got {len(scores_par)}")
    _assert(len(scores_seq) == 4, "sequential: 4 mutations scored",
            f"got {len(scores_seq)}")
    # Parallel should be noticeably faster (>1.5x) — 4x ideal, allow generous margin
    speedup = wall_seq / wall_par if wall_par > 0 else 1.0
    _assert(speedup >= 1.5, f"parallel speedup >= 1.5x (got {speedup:.1f}x)",
            f"parallel={wall_par:.2f}s  sequential={wall_seq:.2f}s")


# ════════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════════

def run_all(groups: List[str]) -> None:
    run_detection = "detection" in groups or "all" in groups
    run_mock      = "mock"      in groups or "all" in groups
    run_scanner   = "scanner"   in groups or "all" in groups
    run_session   = "session"   in groups or "all" in groups
    run_router    = "router"    in groups or "all" in groups
    run_parallel  = "parallel"  in groups or "all" in groups

    if run_detection:
        test_backend_detection()
        test_pyrosetta_stub_error()
        test_empirical_backend_forced()
        test_local_backend_wsl_not_installed()
        test_local_backend_wsl_no_pyrosetta()
        test_backend_status_local_no_wsl()
        test_missing_pdb_error()

    if run_mock:
        test_dynamut2_request_format()
        test_dynamut2_response_parsing()
        test_dynamut2_rate_limit_retry()
        test_dynamut2_full_pipeline_mock()
        test_dynamut2_empirical_fallback_per_mutation()

    if run_scanner:
        test_scanner_candidate_selection()
        test_scanner_protected_residues()
        test_scanner_max_candidates_cap()
        test_combined_score()
        test_hydrophobicity_delta()
        test_chimerax_commands_from_scan()
        test_empty_scan_no_commands()

    if run_session:
        test_session_rosetta_jobs()
        test_session_rosetta_jobs_persistence()

    if run_router:
        test_router_route_rosetta()
        test_router_route_mutation_scan()
        test_router_rosetta_no_mutations_error()

    if run_parallel:
        test_parallel_all_mutations_scored()
        test_parallel_circuit_breaker_uses_empirical()
        test_parallel_wall_time_faster_than_sequential()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StructureBot DynaMut2 / mutation scanner tests"
    )
    parser.add_argument("--detection", action="store_true", help="A: backend detection")
    parser.add_argument("--mock",      action="store_true", help="B+H: HTTP mock tests")
    parser.add_argument("--scanner",   action="store_true", help="C+D+E: scanner tests")
    parser.add_argument("--session",   action="store_true", help="F: session persistence")
    parser.add_argument("--router",    action="store_true", help="G: router wiring")
    parser.add_argument("--parallel",  action="store_true", help="I: parallel DynaMut2")
    args = parser.parse_args()

    groups: List[str] = []
    if args.detection: groups.append("detection")
    if args.mock:      groups.append("mock")
    if args.scanner:   groups.append("scanner")
    if args.session:   groups.append("session")
    if args.router:    groups.append("router")
    if not groups:
        groups = ["all"]

    print("=" * 60)
    print("StructureBot -- DynaMut2 / Mutation Scanner Tests")
    print("=" * 60)

    run_all(groups)

    print()
    print("=" * 60)
    total = sum(_results.values())
    print(
        f"Results: {_results['pass']}/{total} passed  "
        f"({_results['fail']} failed, {_results['skip']} skipped)"
    )
    print("=" * 60)

    sys.exit(1 if _results["fail"] > 0 else 0)


if __name__ == "__main__":
    main()
