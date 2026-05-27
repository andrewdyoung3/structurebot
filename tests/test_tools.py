"""
tests/test_tools.py
-------------------
Unit and integration tests for the StructureBot tool expansion layer.

Test categories
---------------
  A. CamSol bridge — local algorithm, colour binning, viz commands
  B. ESM bridge    — cache layer, score shape, no-model fallback
  C. ToolRouter    — dispatch, session integration, route() augmentation
  D. ProteinMPNN   — stub returns correct error structure
  E. SessionState  — tool_results persistence (add, get, save, load)

Usage
-----
  python tests/test_tools.py              # run all tests
  python tests/test_tools.py --camsol     # CamSol tests only
  python tests/test_tools.py --router     # router tests only

All tests here run WITHOUT a live ChimeraX instance and WITHOUT the ESM
model (ESM tests are skipped unless fair-esm or transformers is installed).
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from camsol_bridge  import CamsolBridge, camsol_score
from proteinmpnn_bridge import ProteinMPNNBridge
from session_state  import SessionState
from tool_router    import ToolRouter, ToolStepResult

# ── Helpers ────────────────────────────────────────────────────────────────────

PASS  = "[PASS]"
FAIL  = "[FAIL]"
SKIP  = "[SKIP]"

_results: dict = {"pass": 0, "fail": 0, "skip": 0}

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

def _assert(cond: bool, test_name: str, msg: str = "") -> bool:
    if cond:
        _ok(test_name, msg)
        return True
    else:
        _fail(test_name, msg or "assertion failed")
        return False

# Test sequences
_SHORT_SEQ = "MTEYKLVVVGAGGVGKS"            # 17 AA — KRAS partial
_MED_SEQ   = (
    "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSY"
    "RKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRT"
)  # 74 AA — KRAS p21 partial
# 1HSG chain A first 50 residues (HIV protease)
_HIV_SEQ   = (
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHK"
)


# ════════════════════════════════════════════════════════════════════════════════
# A. CamSol Bridge Tests
# ════════════════════════════════════════════════════════════════════════════════

def test_camsol_score_basic() -> None:
    print("\n--- A. CamSol bridge ---")

    # Score length matches sequence
    scores = camsol_score(_MED_SEQ)
    _assert(len(scores) == len(_MED_SEQ),
            "score length == sequence length",
            f"{len(scores)} == {len(_MED_SEQ)}")

    # Scores are floats
    _assert(all(isinstance(s, float) for s in scores),
            "all scores are floats")

    # Mean ≈ 0 (z-normalised)
    mean = sum(scores) / len(scores)
    _assert(abs(mean) < 0.05,
            "scores are mean-centred",
            f"mean={mean:.4f}")

    # Std ≈ 1 (z-normalised)
    std = (sum(s*s for s in scores) / len(scores)) ** 0.5
    _assert(0.8 < std < 1.2,
            "scores have unit std (approx)",
            f"std={std:.4f}")


def test_camsol_hydrophobic_vs_charged() -> None:
    """Hydrophobic-rich sequence should score lower than charge-rich one."""
    # All Ile — maximally hydrophobic
    hydro_seq = "IIIIIIIIIIIIIII"
    # Mix of Lys (charge) vs Ile (hydrophobic)
    mixed_seq = "KRKRKRKRKRKRKRK"

    h_scores = camsol_score(hydro_seq)
    m_scores = camsol_score(mixed_seq)

    # Mean of hydrophobic should be lower (more aggregation-prone)
    h_mean = sum(h_scores) / len(h_scores)
    m_mean = sum(m_scores) / len(m_scores)
    _assert(h_mean <= m_mean,
            "hydrophobic sequence scores lower than charged sequence",
            f"h_mean={h_mean:.2f} <= m_mean={m_mean:.2f}")


def test_camsol_bridge_analyze() -> None:
    bridge = CamsolBridge()
    result = bridge.analyze(_HIV_SEQ, model_id="1", chain="A")

    _assert(result.success,
            "analyze() succeeds on valid sequence")
    _assert("scores" in result.data,
            "data contains 'scores' key")
    _assert(len(result.data["scores"]) == len(_HIV_SEQ),
            "scores dict has one entry per residue",
            f"{len(result.data['scores'])} == {len(_HIV_SEQ)}")
    _assert("aggregation_hot_spots" in result.data,
            "data contains 'aggregation_hot_spots'")
    _assert("highly_soluble" in result.data,
            "data contains 'highly_soluble'")
    _assert(len(result.viz_commands) > 2,
            "viz_commands generated",
            f"{len(result.viz_commands)} commands")
    _assert(len(result.summary) > 0,
            "summary is non-empty",
            repr(result.summary[:60]))


def test_camsol_too_short() -> None:
    bridge = CamsolBridge()
    result = bridge.analyze("MAAA", model_id="1")
    _assert(not result.success,
            "analyze() fails on sequence < 5 AA")
    _assert(result.error is not None,
            "error message is set for short sequence")


def test_camsol_viz_commands_format() -> None:
    """Verify generated commands are valid ChimeraX syntax."""
    bridge = CamsolBridge()
    result = bridge.analyze(_MED_SEQ, model_id="1", chain="A")
    if not result.success:
        _fail("viz_commands_format", "analyze() failed")
        return

    cmds = result.viz_commands
    # First two commands should be cartoon and color reset
    _assert(any("cartoon" in c for c in cmds),
            "viz includes cartoon command")
    _assert(any("color #1" in c for c in cmds),
            "viz includes model-specific color command")
    # No command should start with 'background' (old wrong syntax)
    bad = [c for c in cmds if c.startswith("background")]
    _assert(len(bad) == 0,
            "no deprecated 'background' commands",
            f"{len(bad)} bad commands found")


def test_camsol_start_resno() -> None:
    """start_resno parameter shifts residue numbers."""
    bridge = CamsolBridge()
    r1 = bridge.analyze("ACDEFGHIK", model_id="1", start_resno=1)
    r50 = bridge.analyze("ACDEFGHIK", model_id="1", start_resno=50)
    first_key_1  = min(r1.data["scores"].keys())
    first_key_50 = min(r50.data["scores"].keys())
    _assert(first_key_1 == 1,   "start_resno=1 gives key 1")
    _assert(first_key_50 == 50, "start_resno=50 gives key 50")


# ════════════════════════════════════════════════════════════════════════════════
# B. ESM Bridge Tests
# ════════════════════════════════════════════════════════════════════════════════

def test_esm_import_availability() -> None:
    print("\n--- B. ESM bridge ---")
    has_esm = False
    has_hf  = False
    try:
        import esm as _esm_lib  # noqa: F401
        has_esm = True
    except ImportError:
        pass
    try:
        from transformers import EsmForMaskedLM  # noqa: F401
        has_hf = True
    except ImportError:
        pass

    if has_esm:
        _ok("fair-esm available")
    elif has_hf:
        _ok("transformers (HuggingFace) available as fallback")
    else:
        _skip("ESM-2 inference", "neither fair-esm nor transformers is installed")
        return

    # Only run inference tests if a library is available
    _test_esm_cache()
    _test_esm_basic_scores()


def _test_esm_cache() -> None:
    """Disk cache is written and read back correctly."""
    from esm_bridge import EsmBridge, _CACHE_DIR
    bridge = EsmBridge()

    seq = "ACDEFGHIKLM"
    key = bridge._cache_key(seq)
    path = bridge._cache_path(key)

    # Remove any existing cache
    if path.is_file():
        path.unlink()

    # Write a fake cache entry
    fake_probs = [[0.05] * 20 for _ in seq]
    bridge._save_cache(key, fake_probs)
    _assert(path.is_file(), "cache file written")

    loaded = bridge._load_cache(key)
    _assert(loaded == fake_probs, "cache round-trips correctly")

    # Clean up
    path.unlink(missing_ok=True)
    _ok("ESM cache write/read")


def _test_esm_basic_scores() -> None:
    """With a tiny sequence, scores are in [0, 1]."""
    from esm_bridge import EsmBridge
    bridge = EsmBridge()
    seq    = "ACDEFGHIK"   # 9 residues — fast

    result = bridge.analyze(seq, model_id="1")
    if not result.success:
        if "download" in (result.error or "").lower() or "import" in (result.error or "").lower():
            _skip("ESM basic scores", "model not yet downloaded")
            return
        _fail("ESM basic scores", result.error or "unknown error")
        return

    cons = result.data.get("conservation", {})
    _assert(len(cons) == len(seq), "conservation dict length == sequence length")
    for pos, s in cons.items():
        if not (0.0 <= s <= 1.0):
            _fail("ESM conservation in [0,1]", f"pos {pos} = {s}")
            return
    _ok("ESM conservation scores in [0, 1]")


# ════════════════════════════════════════════════════════════════════════════════
# C. ToolRouter Tests
# ════════════════════════════════════════════════════════════════════════════════

def test_router_route_chimerax_only() -> None:
    print("\n--- C. ToolRouter ---")

    session = SessionState()
    session.add_structure("1", "1HSG")

    # Minimal mock bridge
    class _MockBridge:
        def is_running(self): return False

    router = ToolRouter(_MockBridge(), session)

    translator_result = {
        "commands":            ["open 1HSG", "cartoon #1"],
        "explanations":        ["Open 1HSG", "Cartoon"],
        "warnings":            [],
        "clarification_needed": None,
        "confidence":          "high",
        "tools_needed":        ["chimerax"],
        "tool_inputs":         {},
    }

    routed = router.route(translator_result)
    _assert(routed.get("has_extra_tools") is False,
            "chimerax-only: has_extra_tools=False")
    _assert(routed.get("tools_needed") == ["chimerax"],
            "tools_needed preserved as ['chimerax']")
    _assert(len(routed.get("tool_steps_info", [])) == 1,
            "one step in tool_steps_info")


def test_router_route_with_camsol() -> None:
    session = SessionState()
    session.add_structure("1", "1HSG")

    class _MockBridge:
        def is_running(self): return False

    router = ToolRouter(_MockBridge(), session)

    translator_result = {
        "commands":            [],
        "explanations":        [],
        "warnings":            [],
        "clarification_needed": None,
        "confidence":          "high",
        "tools_needed":        ["camsol"],
        "tool_inputs":         {"camsol": {"model_id": "1"}},
    }

    routed = router.route(translator_result)
    _assert(routed.get("has_extra_tools") is True,
            "camsol: has_extra_tools=True")
    _assert("camsol" in routed.get("tools_needed", []),
            "camsol in tools_needed")
    steps = routed.get("tool_steps_info", [])
    _assert(any(s["tool"] == "camsol" for s in steps),
            "camsol step in tool_steps_info")


def test_router_execute_camsol() -> None:
    """Router.execute() with a real CamSol bridge (no ChimeraX needed)."""
    session = SessionState()
    session.add_structure("1", "1HSG")

    class _MockBridge:
        def is_running(self): return False
        def run_command(self, cmd): return {"value": "", "error": None}

    router = ToolRouter(_MockBridge(), session)

    # Provide sequence directly in tool_inputs to avoid network calls
    routed = {
        "commands":            [],
        "explanations":        [],
        "warnings":            [],
        "clarification_needed": None,
        "confidence":          "high",
        "tools_needed":        ["camsol"],
        "tool_inputs":         {"camsol": {"model_id": "1", "sequence": _HIV_SEQ}},
        "has_extra_tools":     True,
        "tool_steps_info":     [{"tool": "camsol", "icon": "💧", "description": "test"}],
    }

    executed = router.execute(routed)
    _assert(executed.get("pipeline_success") is True,
            "CamSol pipeline succeeds")
    _assert(len(executed.get("all_viz_commands", [])) > 0,
            "viz_commands populated after CamSol")
    summaries = executed.get("tool_summaries", {})
    _assert("camsol" in summaries,
            "camsol summary in tool_summaries",
            summaries.get("camsol", "")[:60])

    # Check session state was updated
    cached = session.get_tool_result("camsol", "1")
    _assert(cached is not None,
            "CamSol result stored in session state")
    _assert("scores" in (cached or {}),
            "stored data contains 'scores'")


def test_router_unknown_tool() -> None:
    session = SessionState()

    class _MockBridge:
        def is_running(self): return False

    router = ToolRouter(_MockBridge(), session)
    routed = {
        "commands": [], "explanations": [], "warnings": [],
        "clarification_needed": None, "confidence": "medium",
        "tools_needed": ["unknown_tool_xyz"], "tool_inputs": {},
        "has_extra_tools": True, "tool_steps_info": [],
    }
    executed = router.execute(routed)
    _assert(executed.get("pipeline_success") is False,
            "unknown tool causes pipeline failure")
    _assert("unknown_tool_xyz" in (executed.get("pipeline_error") or ""),
            "error message mentions the unknown tool name")


def test_router_defaults_for_missing_fields() -> None:
    """route() should handle translator results that lack tools_needed."""
    session = SessionState()

    class _MockBridge:
        def is_running(self): return False

    router = ToolRouter(_MockBridge(), session)
    minimal_result = {
        "commands":            ["cartoon #1"],
        "explanations":        ["cartoon"],
        "warnings":            [],
        "clarification_needed": None,
        "confidence":          "high",
        # No tools_needed / tool_inputs
    }
    routed = router.route(minimal_result)
    _assert(routed["tools_needed"] == ["chimerax"],
            "missing tools_needed defaults to ['chimerax']")
    _assert(routed["tool_inputs"] == {},
            "missing tool_inputs defaults to {}")


# ════════════════════════════════════════════════════════════════════════════════
# D. ProteinMPNN Stub Tests
# ════════════════════════════════════════════════════════════════════════════════

def test_proteinmpnn_stub() -> None:
    print("\n--- D. ProteinMPNN stub ---")

    # Ensure PROTEINMPNN_DIR is not accidentally set
    os.environ.pop("PROTEINMPNN_DIR", None)

    bridge = ProteinMPNNBridge()
    result = bridge.analyze({}, session=None)

    _assert(not result.success,
            "stub returns success=False")
    _assert(result.error is not None and len(result.error) > 10,
            "stub provides a descriptive error message",
            repr(result.error[:60]))
    _assert("PROTEINMPNN_DIR" in (result.error or ""),
            "error mentions PROTEINMPNN_DIR env var")
    _assert(result.tool == "proteinmpnn",
            "tool field is 'proteinmpnn'")
    _ok("ProteinMPNN stub returns correct structure")


# ════════════════════════════════════════════════════════════════════════════════
# E. SessionState Tool Results Tests
# ════════════════════════════════════════════════════════════════════════════════

def test_session_tool_results() -> None:
    print("\n--- E. SessionState tool results ---")

    session = SessionState()

    # add_tool_result
    session.add_tool_result("camsol", "1", {"scores": {1: 0.5, 2: -0.3}})
    _assert(session.get_tool_result("camsol", "1") is not None,
            "add_tool_result / get_tool_result round-trip")

    data = session.get_tool_result("camsol", "1")
    _assert(data == {"scores": {1: 0.5, 2: -0.3}},
            "retrieved data matches stored data")

    # None for missing
    _assert(session.get_tool_result("esm", "1") is None,
            "get_tool_result returns None for missing tool")
    _assert(session.get_tool_result("camsol", "2") is None,
            "get_tool_result returns None for missing model_id")


def test_session_tool_results_persistence() -> None:
    """tool_results survive save() → load()."""
    session = SessionState()
    session.add_structure("1", "1HSG")
    session.add_tool_result("camsol", "1", {"scores": {1: 0.8, 2: -1.2}})
    session.add_tool_result("esm",    "1", {"conservation": {1: 0.95, 2: 0.3}})

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        tmp_path = f.name

    try:
        session.save(tmp_path)
        loaded = SessionState.load(tmp_path)

        camsol_data = loaded.get_tool_result("camsol", "1")
        _assert(camsol_data is not None,
                "camsol result survives save/load")
        # JSON serialises integer dict keys as strings; check values match
        scores = camsol_data.get("scores", {})
        scores_values = sorted(scores.values())
        _assert(scores_values == sorted([-1.2, 0.8]),
                "camsol scores values preserved through JSON",
                str(scores_values))

        esm_data = loaded.get_tool_result("esm", "1")
        _assert(esm_data is not None,
                "esm result survives save/load")
        cons = esm_data.get("conservation", {})
        cons_values = sorted(cons.values())
        _assert(cons_values == sorted([0.3, 0.95]),
                "esm conservation values preserved through JSON",
                str(cons_values))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_session_clear_tool_results() -> None:
    session = SessionState()
    session.add_tool_result("camsol", "1", {"x": 1})
    session.add_tool_result("esm",    "1", {"y": 2})

    session.clear_tool_results(tool="camsol")
    _assert(session.get_tool_result("camsol", "1") is None,
            "clear_tool_results(tool=) removes that tool")
    _assert(session.get_tool_result("esm", "1") is not None,
            "clear_tool_results(tool=) leaves other tools intact")

    session.clear_tool_results()
    _assert(session.get_tool_result("esm", "1") is None,
            "clear_tool_results() clears everything")


def test_session_context_summary_shows_tools() -> None:
    """get_context_summary() mentions cached tool results."""
    session = SessionState()
    session.add_structure("1", "1HSG")
    session.add_tool_result("camsol", "1", {
        "scores":               {1: 0.5},
        "aggregation_hot_spots": [1],
    })
    summary = session.get_context_summary()
    _assert("camsol" in summary.lower(),
            "context summary mentions camsol results")


def test_session_snapshot_restore_roundtrip() -> None:
    """snapshot/restore: adding a tool result then restoring removes it."""
    session = SessionState()
    session.add_structure("1", "1HSG")

    snap = session.snapshot()

    # Modify state after snapshot
    session.add_tool_result("esm", "1", {"conservation": {1: 0.9}})
    session.add_structure("2", "2HHB")
    _assert(session.get_tool_result("esm", "1") is not None,
            "esm result present before restore")
    _assert("2" in session.structures,
            "structure #2 present before restore")

    # Restore
    session.restore(snap)

    _assert("1" in session.structures,
            "structure #1 still present after restore")
    _assert("2" not in session.structures,
            "structure #2 gone after restore")
    _assert(session.get_tool_result("esm", "1") is None,
            "esm result gone after restore")


def test_session_snapshot_independence() -> None:
    """Modifying session after snapshot does not affect the snapshot."""
    session = SessionState()
    session.add_structure("1", "1HSG")

    snap = session.snapshot()

    # Mutate the session after taking snapshot
    session.structures["1"]["name"] = "MUTATED"
    session.add_tool_result("camsol", "1", {"scores": {1: 9.99}})

    # Snapshot should be unaffected
    snap_structs = snap["structures"]
    _assert(snap_structs["1"]["name"] == "1HSG",
            "snapshot structure name unchanged after session mutation")
    _assert("camsol" not in snap.get("tool_results", {}),
            "snapshot has no camsol result after post-snapshot mutation")


# ════════════════════════════════════════════════════════════════════════════════
# F. Log analyser
# ════════════════════════════════════════════════════════════════════════════════

def test_parse_logs_empty_dir() -> None:
    """parse_logs on an empty directory returns zero sessions and requests."""
    print("\n=== F. Log analyser ===")
    import tempfile
    from log_analyser import parse_logs

    with tempfile.TemporaryDirectory() as tmpdir:
        data = parse_logs(Path(tmpdir))

    _assert(data["n_sessions"] == 0, "n_sessions=0 for empty dir",
            f"got {data['n_sessions']}")
    _assert(data["n_requests"] == 0, "n_requests=0 for empty dir",
            f"got {data['n_requests']}")


def test_parse_logs_basic_entries() -> None:
    """parse_logs counts requests, successes, and PDB IDs from mock log files."""
    import json as _json, tempfile
    from log_analyser import parse_logs

    entries = [
        {"timestamp": "2026-05-27T10:00:00", "user_input": "open 1HSG",
         "commands": ["open 1HSG"], "success": True, "error": None},
        {"timestamp": "2026-05-27T10:01:00", "user_input": "suggest mutations to improve solubility",
         "commands": [], "success": True, "error": None},
        {"timestamp": "2026-05-27T10:02:00", "user_input": "open 2HHB",
         "commands": ["open 2HHB"], "success": False, "error": "timeout"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session_20260527_100000.jsonl"
        with open(log_path, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(_json.dumps(e) + "\n")
        data = parse_logs(Path(tmpdir))

    _assert(data["n_sessions"] == 1, "n_sessions=1",  f"got {data['n_sessions']}")
    _assert(data["n_requests"] == 3, "n_requests=3",  f"got {data['n_requests']}")
    _assert(data["n_success"]  == 2, "n_success=2",   f"got {data['n_success']}")
    _assert("1HSG" in data["pdb_id_counts"],  "1HSG in pdb_id_counts")
    _assert("2HHB" in data["pdb_id_counts"],  "2HHB in pdb_id_counts")


def test_parse_logs_tool_counts() -> None:
    """parse_logs counts scan requests from user_input keywords."""
    import json as _json, tempfile
    from log_analyser import parse_logs

    entries = [
        {"timestamp": "2026-05-27T10:00:00",
         "user_input": "suggest mutations to improve solubility of chain A",
         "commands": [], "success": True, "error": None},
        {"timestamp": "2026-05-27T10:05:00",
         "user_input": "suggest disulfide bonds to stabilise the dimer",
         "commands": [], "success": True, "error": None},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session_20260527_100000.jsonl"
        with open(log_path, "w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(_json.dumps(e) + "\n")
        data = parse_logs(Path(tmpdir))

    _assert("mutation_scan" in data["tool_counts"],
            "mutation_scan detected from 'suggest mutations'",
            f"tool_counts={data['tool_counts']}")
    _assert("disulfide" in data["tool_counts"],
            "disulfide detected from 'disulfide bonds'",
            f"tool_counts={data['tool_counts']}")


def test_parse_logs_enhanced_tool_steps() -> None:
    """parse_logs extracts timing and top_candidate from enhanced log entries."""
    import json as _json, tempfile
    from log_analyser import parse_logs

    entry = {
        "timestamp": "2026-05-27T10:00:00",
        "user_input": "suggest mutations",
        "commands": [],
        "success": True,
        "error": None,
        "tool_steps": [
            {
                "tool": "mutation_scan",
                "elapsed_ms": 92340.0,
                "success": True,
                "n_candidates": 5,
                "top_candidate": "I64E",
                "top_ddg": -3.53,
                "backend": "dynamut2",
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "session_20260527_100000.jsonl"
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(_json.dumps(entry) + "\n")
        data = parse_logs(Path(tmpdir))

    _assert("mutation_scan" in data["tool_timings"],
            "mutation_scan timing recorded",
            f"tool_timings keys: {list(data['tool_timings'].keys())}")
    timings = data["tool_timings"].get("mutation_scan", [])
    _assert(len(timings) == 1 and abs(timings[0] - 92.34) < 0.1,
            "elapsed_ms converted to seconds",
            f"got {timings}")
    _assert("I64E" in data["top_candidates"],
            "top_candidate I64E recorded",
            f"top_candidates={data['top_candidates']}")
    _assert("dynamut2" in data["backends_used"],
            "backend dynamut2 recorded",
            f"backends={data['backends_used']}")


def test_generate_stats_report_non_empty() -> None:
    """generate_stats_report returns a non-empty renderable for non-zero data."""
    from log_analyser import generate_stats_report

    data = {
        "n_sessions": 5,
        "n_requests": 42,
        "n_success":  38,
        "pdb_id_counts":  {"1HSG": 10, "2HHB": 5},
        "tool_counts":    {"mutation_scan": 20, "disulfide": 8},
        "tool_timings":   {"mutation_scan": [92.3, 88.5, 95.1]},
        "top_candidates": {"I64E": 8, "L63K": 3},
        "backends_used":  {"dynamut2": 25, "empirical": 5},
        "log_files":      [],
    }

    report = generate_stats_report(data)
    # Rich Panel stores its body in .renderable; plain-string fallback otherwise
    report_content = getattr(report, "renderable", str(report))
    _assert("42" in report_content or "Sessions" in report_content,
            "report contains request count or 'Sessions'",
            f"report: {str(report_content)[:120]}")


# ════════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════════

def run_all(groups: list) -> None:
    run_camsol      = "camsol"     in groups or "all" in groups
    run_esm         = "esm"        in groups or "all" in groups
    run_router      = "router"     in groups or "all" in groups
    run_proteinmpnn = "proteinmpnn" in groups or "all" in groups
    run_session     = "session"    in groups or "all" in groups
    run_log         = "log"        in groups or "all" in groups

    if run_camsol:
        test_camsol_score_basic()
        test_camsol_hydrophobic_vs_charged()
        test_camsol_bridge_analyze()
        test_camsol_too_short()
        test_camsol_viz_commands_format()
        test_camsol_start_resno()

    if run_esm:
        test_esm_import_availability()

    if run_router:
        test_router_route_chimerax_only()
        test_router_route_with_camsol()
        test_router_execute_camsol()
        test_router_unknown_tool()
        test_router_defaults_for_missing_fields()

    if run_proteinmpnn:
        test_proteinmpnn_stub()

    if run_session:
        test_session_tool_results()
        test_session_tool_results_persistence()
        test_session_clear_tool_results()
        test_session_context_summary_shows_tools()
        test_session_snapshot_restore_roundtrip()
        test_session_snapshot_independence()

    if run_log:
        test_parse_logs_empty_dir()
        test_parse_logs_basic_entries()
        test_parse_logs_tool_counts()
        test_parse_logs_enhanced_tool_steps()
        test_generate_stats_report_non_empty()


def main() -> None:
    parser = argparse.ArgumentParser(description="StructureBot tool expansion tests")
    parser.add_argument("--camsol",      action="store_true")
    parser.add_argument("--esm",         action="store_true")
    parser.add_argument("--router",      action="store_true")
    parser.add_argument("--proteinmpnn", action="store_true")
    parser.add_argument("--session",     action="store_true")
    parser.add_argument("--log",         action="store_true")
    args = parser.parse_args()

    groups = []
    if args.camsol:      groups.append("camsol")
    if args.esm:         groups.append("esm")
    if args.router:      groups.append("router")
    if args.proteinmpnn: groups.append("proteinmpnn")
    if args.session:     groups.append("session")
    if args.log:         groups.append("log")
    if not groups:
        groups = ["all"]

    print("=" * 60)
    print("StructureBot Tool Expansion Tests")
    print("=" * 60)

    run_all(groups)

    print()
    print("=" * 60)
    total = _results["pass"] + _results["fail"] + _results["skip"]
    print(
        f"Results: {_results['pass']}/{total} passed  "
        f"({_results['fail']} failed, {_results['skip']} skipped)"
    )
    print("=" * 60)

    sys.exit(1 if _results["fail"] > 0 else 0)


if __name__ == "__main__":
    main()
