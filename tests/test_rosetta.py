"""
tests/test_rosetta.py
---------------------
Tests for RosettaBridge, MutationScanner, and the related session/router wiring.

Test categories
---------------
  A. Backend detection
       Verify ROSETTA_BACKEND / PYROSETTA_AVAILABLE env vars select the
       correct backend and that the stub returns a well-formed error.

  B. Robetta job submission mock
       Mock the requests library to verify that the correct API format
       (endpoint URL, multipart body, auth header) is used without
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

  F. SessionState: Rosetta job persistence
       add_rosetta_job / get_rosetta_job / update_rosetta_job /
       list_rosetta_jobs survive save() → load() round-trip.

  G. ToolRouter: dispatch wiring
       Verify route() augments the result correctly for "rosetta" and
       "mutation_scan" tools; verify error path for missing PDB / mutations.

  H. Full pipeline mock (Robetta mocked, no live network)
       End-to-end through RosettaBridge._run_robetta() with the HTTP
       layer replaced by fakes; check ddg_scores, viz_commands, job_id
       persisted in session.

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
# HIV protease monomer, chain A, residues 1–4 (synthetic — just needs to exist
# as a valid file; real Rosetta calls are mocked in these tests).
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

    # Force robetta via env var
    os.environ["ROSETTA_BACKEND"] = "robetta"
    _assert(_select_backend() == "robetta", "ROSETTA_BACKEND=robetta forces robetta")
    del os.environ["ROSETTA_BACKEND"]

    # Force pyrosetta via env var (even if not importable)
    os.environ["ROSETTA_BACKEND"] = "pyrosetta"
    _assert(_select_backend() == "pyrosetta", "ROSETTA_BACKEND=pyrosetta forces pyrosetta")
    del os.environ["ROSETTA_BACKEND"]

    # Auto mode: PYROSETTA_AVAILABLE not set → robetta
    os.environ.pop("PYROSETTA_AVAILABLE", None)
    os.environ.pop("ROSETTA_BACKEND",    None)
    _assert(
        _select_backend() == "robetta",
        "auto mode without PYROSETTA_AVAILABLE -> robetta",
    )


def test_pyrosetta_stub_error() -> None:
    """The PyRosetta stub must return success=False with an instructive message."""
    from rosetta_bridge import RosettaBridge

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "pyrosetta"
        os.environ["PYROSETTA_AVAILABLE"] = "true"

        # Patch the import so pyrosetta appears available
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
            "PyRosetta error message mentions PyRosetta/Python version",
            repr((result.error or "")[:80]),
        )
    finally:
        os.environ.pop("ROSETTA_BACKEND",    None)
        os.environ.pop("PYROSETTA_AVAILABLE", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_missing_api_key_error() -> None:
    """Robetta backend with no API key must return a helpful error."""
    from rosetta_bridge import RosettaBridge

    pdb_path = _write_temp_pdb()
    try:
        saved = os.environ.pop("ROBETTA_API_KEY", None)
        os.environ["ROSETTA_BACKEND"] = "robetta"

        bridge = RosettaBridge()
        result = bridge.analyze(
            pdb_path  = pdb_path,
            mutations = [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}],
        )

        _assert(not result.success,        "no API key → success=False")
        _assert("ROBETTA_API_KEY" in (result.error or ""),
                "error mentions ROBETTA_API_KEY env var")
    finally:
        if saved:
            os.environ["ROBETTA_API_KEY"] = saved
        os.environ.pop("ROSETTA_BACKEND", None)
        Path(pdb_path).unlink(missing_ok=True)


def test_missing_pdb_error() -> None:
    """Non-existent PDB path must produce a clear error before any API call."""
    from rosetta_bridge import RosettaBridge

    os.environ["ROSETTA_BACKEND"] = "robetta"
    os.environ["ROBETTA_API_KEY"] = "test-key-for-path-check"
    try:
        bridge = RosettaBridge()
        result = bridge.analyze(
            pdb_path  = "/tmp/does_not_exist_at_all.pdb",
            mutations = [{"chain": "A", "position": 1, "from_aa": "P", "to_aa": "K"}],
        )
        _assert(not result.success,           "missing PDB → success=False")
        _assert("not found" in (result.error or "").lower() or
                "pdb" in (result.error or "").lower(),
                "error mentions PDB file",
                repr((result.error or "")[:80]))
    finally:
        os.environ.pop("ROSETTA_BACKEND",  None)
        os.environ.pop("ROBETTA_API_KEY",  None)


# ════════════════════════════════════════════════════════════════════════════════
# B. Robetta HTTP mock — correct API format
# ════════════════════════════════════════════════════════════════════════════════

def test_robetta_submit_format() -> None:
    """Verify submit_ddg_job sends the correct URL, auth header, and body."""
    print("\n--- B. Robetta HTTP mock ---")

    from rosetta_bridge import RosettaBridge, _ROBETTA_SUBMIT_URL

    pdb_path = _write_temp_pdb()
    try:
        os.environ["ROSETTA_BACKEND"] = "robetta"
        os.environ["ROBETTA_API_KEY"] = "test-api-key-123"

        mutations = [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}]

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "job_id": "42",
            "warnings": [],
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            bridge = RosettaBridge()
            job_id, warnings = bridge._submit_ddg_job(pdb_path, mutations)

        _assert(mock_post.called, "requests.post was called")

        call_kwargs = mock_post.call_args
        url = call_kwargs.args[0] if call_kwargs.args else call_kwargs.kwargs.get("url", "")
        _assert(
            _ROBETTA_SUBMIT_URL in str(url),
            "POST sent to correct Robetta URL",
            str(url),
        )

        headers = call_kwargs.kwargs.get("headers", {})
        _assert(
            "Token test-api-key-123" in headers.get("Authorization", ""),
            "Authorization header uses Token format",
            headers.get("Authorization", "")[:40],
        )

        payload = call_kwargs.kwargs.get("data", {})
        parsed_muts = json.loads(payload.get("mutations", "[]"))
        _assert(
            parsed_muts == mutations,
            "mutations serialised as JSON in body",
            str(parsed_muts),
        )

        _assert(job_id == "42", "job_id extracted from response", job_id)
        _assert(warnings == [],  "empty warnings list returned")

    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        os.environ.pop("ROBETTA_API_KEY",  None)
        Path(pdb_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# C. MutationScanner — candidate selection
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

    # Sequence: 10 residues — position 3 (I) is aggregation-prone + tolerant
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
        # Patch Rosetta to return zeros (avoids real API call)
        with patch("rosetta_bridge.RosettaBridge") as MockRosetta:
            mock_bridge = MagicMock()
            mock_bridge.analyze.return_value = MagicMock(
                success=True,
                data={"ddg_scores": {}},
            )
            MockRosetta.return_value = mock_bridge

            results = scanner.scan(pdb_path=pdb_path, chain_id="A", sequence=seq)

        # At least position 3 (I) should be in the results
        positions = {r["position"] for r in results}
        _assert(3 in positions, "position 3 (I, aggregation-prone + tolerant) is a candidate")

        # Positions that are not aggregation-prone should be absent
        _assert(1 not in positions, "position 1 (CamSol=+0.5) is not a candidate")
        _assert(2 not in positions, "position 2 (CamSol=+0.3) is not a candidate")

        # No Pro or Cys substitutions
        bad = [r for r in results if r["to_aa"] in ("P", "C")]
        _assert(len(bad) == 0, "no Pro or Cys substitution candidates")

        # All from_aa match the sequence
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
    # All positions: aggregation-prone + tolerant
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

    # Strongly stabilising + solubility improvement + high tolerance → high score
    high = combined_score(ddg=-2.0, camsol_delta=2.0, esm_tolerance=0.9)
    _assert(high > 1.0, "stabilising+soluble+tolerant → score > 1.0", f"{high:.4f}")

    # Destabilising + no improvement + conserved → low score
    low = combined_score(ddg=2.0, camsol_delta=0.0, esm_tolerance=0.1)
    _assert(low < 0.0, "destabilising+no-improvement+conserved → score < 0", f"{low:.4f}")

    # Neutral inputs → score ≈ 0
    neutral = combined_score(ddg=0.0, camsol_delta=0.0, esm_tolerance=0.0)
    _assert(abs(neutral) < 0.01, "all-zero inputs → score ≈ 0", f"{neutral:.4f}")

    # Weight override: only DDG matters (w_sol=0, w_tol=0)
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
    """Hydrophobic→charged substitution should yield positive camsol_delta."""
    from mutation_scanner import _estimate_camsol_delta

    delta_IK = _estimate_camsol_delta("I", "K")   # Ile→Lys: hydro to charged
    delta_KI = _estimate_camsol_delta("K", "I")   # reverse

    _assert(delta_IK > 0, "I→K gives positive camsol_delta (improves solubility)",
            f"delta={delta_IK:.3f}")
    _assert(delta_KI < 0, "K→I gives negative camsol_delta (worsens solubility)",
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
    _assert(len(cmds) == len(exps), "commands and explanations are same length")

    _assert(any("cartoon" in c for c in cmds), "includes cartoon command")
    _assert(any("color #1" in c for c in cmds), "includes model-specific color reset")

    # Top-1 (position 75) should get sphere + label
    _assert(any("sphere" in c and ":75" in c for c in cmds),
            "top candidate gets sphere command")
    _assert(any("label" in c and ":75" in c for c in cmds),
            "top candidate gets label command")

    # Second candidate (position 40) should get a colour but not a sphere
    _assert(any(":40" in c for c in cmds),  "position 40 coloured")

    # No deprecated 'background' commands
    bad = [c for c in cmds if c.strip().startswith("background")]
    _assert(len(bad) == 0, "no deprecated 'background' commands", str(bad))


def test_empty_scan_no_commands() -> None:
    """generate_chimerax_commands on an empty result list returns empty lists."""
    from mutation_scanner import MutationScanner
    from session_state import SessionState

    scanner = MutationScanner(session=SessionState(), model_id="1")
    cmds, exps = scanner.generate_chimerax_commands([], top_n=5)
    _assert(cmds == [],  "empty scan → empty commands list")
    _assert(exps == [],  "empty scan → empty explanations list")


# ════════════════════════════════════════════════════════════════════════════════
# F. SessionState: Rosetta job persistence
# ════════════════════════════════════════════════════════════════════════════════

def test_session_rosetta_jobs() -> None:
    print("\n--- F. SessionState Rosetta job persistence ---")

    from session_state import SessionState

    session = SessionState()

    job_data = {
        "mutations":    [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}],
        "pdb_path":     "/tmp/test.pdb",
        "backend":      "robetta",
        "submitted_at": "2026-05-26T12:00:00",
        "status":       "submitted",
    }

    session.add_rosetta_job("42", job_data)
    _assert(session.get_rosetta_job("42") is not None, "job stored and retrieved")

    session.update_rosetta_job("42", {"status": "running"})
    _assert(session.get_rosetta_job("42")["status"] == "running",
            "update_rosetta_job changes status")

    # Not-found job
    _assert(session.get_rosetta_job("999") is None,
            "get_rosetta_job returns None for unknown job")

    # list_rosetta_jobs
    all_jobs = session.list_rosetta_jobs()
    _assert("42" in all_jobs, "list_rosetta_jobs includes our job")

    # clear_rosetta_job
    session.clear_rosetta_job("42")
    _assert(session.get_rosetta_job("42") is None, "clear_rosetta_job removes job")


def test_session_rosetta_jobs_persistence() -> None:
    """Rosetta jobs survive save() → load()."""
    from session_state import SessionState

    session = SessionState()
    session.add_rosetta_job("99", {
        "mutations": [{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}],
        "pdb_path":  "/tmp/test.pdb",
        "backend":   "robetta",
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
        _assert(job is not None,                            "job persists after save/load")
        _assert(job["status"] == "completed",               "job status preserved")
        _assert(job["results"].get("V82A") == 1.47,         "job results preserved")

        scan = loaded.get_scan_result("1")
        _assert(scan is not None,                           "scan result persists")
        _assert(len(scan) == 1,                             "scan result length preserved")
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
        "tool_inputs":  {"mutation_scan": {"model_id": "1", "chain": "A", "focus": "solubility"}},
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
            "rosetta without mutations → pipeline_success=False")
    err = executed.get("pipeline_error", "")
    _assert("mutation" in err.lower(),
            "error message mentions 'mutations'", repr(err[:80]))


# ════════════════════════════════════════════════════════════════════════════════
# H. Full pipeline mock
# ════════════════════════════════════════════════════════════════════════════════

def test_robetta_full_pipeline_mock() -> None:
    """
    End-to-end through RosettaBridge._run_robetta() with all HTTP calls mocked.
    Verifies: job submitted → polled → results fetched → ToolStepResult populated.
    """
    print("\n--- H. Full pipeline mock ---")

    from rosetta_bridge import RosettaBridge
    from session_state import SessionState

    pdb_path = _write_temp_pdb()
    session  = SessionState()

    mutations = [
        {"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"},
        {"chain": "A", "position": 10, "from_aa": "L", "to_aa": "K"},
    ]

    fake_submit  = MagicMock(status_code=201,
                             json=lambda: {"job_id": "1337", "warnings": []})
    fake_status  = MagicMock(status_code=200,
                             json=lambda: {"status": "completed"})
    fake_results = MagicMock(status_code=200,
                             json=lambda: {
                                 "results": [
                                     {"mutation": "V82A", "ddg": 1.47},
                                     {"mutation": "L10K", "ddg": -0.82},
                                 ]
                             })
    fake_status.raise_for_status = lambda: None
    fake_results.raise_for_status = lambda: None

    try:
        os.environ["ROSETTA_BACKEND"] = "robetta"
        os.environ["ROBETTA_API_KEY"] = "mock-key"

        with patch("requests.post", return_value=fake_submit), \
             patch("requests.get",  side_effect=[fake_status, fake_results]):
            bridge = RosettaBridge()
            result = bridge.analyze(
                pdb_path  = pdb_path,
                mutations = mutations,
                session   = session,
            )

        _assert(result.success,           "mocked pipeline returns success=True")
        ddg = result.data.get("ddg_scores", {})
        _assert("V82A" in ddg,            "V82A in ddg_scores")
        _assert("L10K" in ddg,            "L10K in ddg_scores")
        _assert(ddg["V82A"] == 1.47,      "V82A ddg = 1.47", str(ddg["V82A"]))
        _assert(ddg["L10K"] == -0.82,     "L10K ddg = -0.82", str(ddg["L10K"]))
        _assert(result.data["job_id"] == "1337", "job_id stored in data")
        _assert(result.data["backend"] == "robetta", "backend = robetta")

        # Job persisted in session
        job = session.get_rosetta_job("1337")
        _assert(job is not None,          "job stored in session")
        _assert(job["status"] == "completed", "job status updated to completed")
        _assert("V82A" in job.get("results", {}), "results stored in session job")

        # Viz commands generated
        _assert(len(result.viz_commands) > 0,          "viz_commands non-empty")
        _assert(any("color" in c for c in result.viz_commands),
                "viz includes color commands")

        # Best mutation should be L10K (most negative ddg)
        _assert("L10K" in result.summary, "summary mentions most stabilising mutation")

    finally:
        os.environ.pop("ROSETTA_BACKEND", None)
        os.environ.pop("ROBETTA_API_KEY",  None)
        Path(pdb_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════════

def run_all(groups: List[str]) -> None:
    run_detection = "detection" in groups or "all" in groups
    run_mock      = "mock"      in groups or "all" in groups
    run_scanner   = "scanner"   in groups or "all" in groups
    run_session   = "session"   in groups or "all" in groups
    run_router    = "router"    in groups or "all" in groups

    if run_detection:
        test_backend_detection()
        test_pyrosetta_stub_error()
        test_missing_api_key_error()
        test_missing_pdb_error()

    if run_mock:
        test_robetta_submit_format()
        test_robetta_full_pipeline_mock()

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


def main() -> None:
    parser = argparse.ArgumentParser(description="StructureBot Rosetta / mutation tests")
    parser.add_argument("--detection", action="store_true", help="A: backend detection")
    parser.add_argument("--mock",      action="store_true", help="B+H: HTTP mock tests")
    parser.add_argument("--scanner",   action="store_true", help="C+D+E: scanner tests")
    parser.add_argument("--session",   action="store_true", help="F: session persistence")
    parser.add_argument("--router",    action="store_true", help="G: router wiring")
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
    print("StructureBot — Rosetta / Mutation Scanner Tests")
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
