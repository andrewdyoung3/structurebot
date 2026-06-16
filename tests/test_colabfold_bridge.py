"""
tests/test_colabfold_bridge.py
------------------------------
Unit tests for the ColabFold v1 standalone bridge + router/viz wiring.

NO LIVE FOLD: the WSL2 subprocess and the ChimeraX REST calls are mocked. A
single optional end-to-end live fold is gated behind STRUCTUREBOT_RUN_LIVE_
COLABFOLD=1 (mirrors the PyRosetta-benchmark opt-in) AND env availability, so it
never runs in CI.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as _cfg
from colabfold_bridge import ColabFoldBridge, _format_worker_failure
from tool_router import ToolRouter
from session_state import SessionState
from chimerax_bridge import ChimeraXBridge


# ── Fake WSL bridge (captures the worker script, fabricates result files) ───────

class _FakeWSL:
    """Stand-in for WSLBridge: records the worker script and writes fake outputs."""

    def __init__(self, payload, *, available=True, cf=True, run_ok=True,
                 stdout="Running on GPU"):
        self._payload   = payload
        self._available = available
        self._cf        = cf
        self._run_ok    = run_ok
        self._stdout    = stdout
        self.scripts    = []
        self.python_bin = None
        self.copied_back = []
        self.run_called = 0

    def is_available(self):   return self._available
    def check_colabfold(self): return self._cf
    def translate_path(self, p): return p
    def copy_to_wsl(self, p, dest_dir="/tmp"):
        return f"{dest_dir.rstrip('/')}/{Path(p).name}"

    def run_python_script(self, script, timeout=600, python_bin=""):
        self.scripts.append(script)
        self.python_bin = python_bin
        self.run_called += 1
        return {
            "ok": self._run_ok, "returncode": 0 if self._run_ok else 1,
            "stdout": self._stdout, "stderr": "" if self._run_ok else "boom",
            "error": None if self._run_ok else "worker exited 1",
        }

    def copy_from_wsl(self, wsl_path, windows_dest):
        Path(windows_dest).parent.mkdir(parents=True, exist_ok=True)
        if wsl_path.endswith("_result.json"):
            Path(windows_dest).write_text(json.dumps(self._payload), encoding="utf-8")
        elif windows_dest.endswith(".pdb") or wsl_path.endswith(".pdb"):
            Path(windows_dest).write_text("ATOM      1  N   ALA A   1\nEND\n",
                                          encoding="utf-8")
        else:
            Path(windows_dest).write_bytes(b"\x89PNG\r\n fake image bytes")
        self.copied_back.append((wsl_path, windows_dest))
        return True


def _ok_payload():
    return {
        "success":        True,
        "ranked_pdb_wsl": "/tmp/colabfold_x/m_unrelaxed_rank_001_model_1_seed_000.pdb",
        "plddt":          [80.0, 90.0, 70.0, 95.0],
        "pae":            [[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]],
        "ptm":            0.44,
        "iptm":           0.61,
        "pngs": {
            "pae":      "/tmp/colabfold_x/m_pae.png",
            "plddt":    "/tmp/colabfold_x/m_plddt.png",
            "coverage": "/tmp/colabfold_x/m_coverage.png",
        },
    }


@pytest.fixture
def cache_tmp(tmp_path, monkeypatch):
    """Point the ColabFold cache dir at a temp dir for hermetic cache tests."""
    monkeypatch.setattr(_cfg, "COLABFOLD_CACHE_DIR", tmp_path / "cf_cache")
    return tmp_path


# ══════════════════════════════════════════════════════════════════════════════
# Worker script construction
# ══════════════════════════════════════════════════════════════════════════════

def test_worker_compiles_and_has_remote_msa_flags():
    s = ColabFoldBridge._build_worker(
        seq_line="ACDE", jobname="q", out_dir="/tmp/o", fasta_path="/tmp/o/in.fasta",
        result_path="/tmp/r.json", n_models=5, n_recycle=3,
        msa_mode="mmseqs2_uniref_env", tmpl_dir="",
    )
    compile(s, "<worker>", "exec")            # double-brace correctness
    assert "colabfold_batch" in s
    assert "--msa-mode" in s and "mmseqs2_uniref_env" in s
    assert "--num-models" in s and "--num-recycle" in s
    # de novo: the embedded template dir is empty, so the runtime branch is skipped
    import re as _re
    assert _re.search(r"tmpl_dir\s*=\s*''", s)


def test_worker_includes_template_flags_when_given():
    s = ColabFoldBridge._build_worker(
        seq_line="ACDE", jobname="q", out_dir="/tmp/o", fasta_path="/tmp/o/in.fasta",
        result_path="/tmp/r.json", n_models=1, n_recycle=1,
        msa_mode="mmseqs2_uniref_env", tmpl_dir="/tmp/colabfold_templates",
    )
    compile(s, "<worker>", "exec")
    assert "--templates" in s
    # the template dir is actually embedded → the runtime branch will fire
    import re as _re
    assert _re.search(r"tmpl_dir\s*=\s*'/tmp/colabfold_templates'", s)


def test_worker_sets_jax_compilation_cache_dir():
    s = ColabFoldBridge._build_worker(
        seq_line="ACDE", jobname="q", out_dir="/tmp/o", fasta_path="/tmp/o/in.fasta",
        result_path="/tmp/r.json", n_models=1, n_recycle=1,
        msa_mode="mmseqs2_uniref_env", tmpl_dir="",
        jax_compile_cache_dir="~/.cache/colabfold_jax_compile",
    )
    compile(s, "<worker>", "exec")
    assert 'os.environ["JAX_COMPILATION_CACHE_DIR"]' in s
    assert "~/.cache/colabfold_jax_compile" in s
    # set BEFORE the colabfold_batch subprocess so the child inherits it
    assert s.index("JAX_COMPILATION_CACHE_DIR") < s.index("subprocess.run")
    # explicit min-compile-time so caching is version-robust
    assert "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" in s


def test_worker_omits_jax_cache_when_empty():
    s = ColabFoldBridge._build_worker(
        seq_line="ACDE", jobname="q", out_dir="/tmp/o", fasta_path="/tmp/o/in.fasta",
        result_path="/tmp/r.json", n_models=1, n_recycle=1,
        msa_mode="mmseqs2_uniref_env", tmpl_dir="", jax_compile_cache_dir="",
    )
    compile(s, "<worker>", "exec")
    import re as _re
    assert _re.search(r"jax_cache\s*=\s*''", s)   # disabled → guarded branch skips


def test_predict_passes_jax_cache_from_config(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(_ok_payload())
    bridge.predict("ACDEFGHIKLMNPQRSTVWY", quick=True)
    assert _cfg.COLABFOLD_JAX_COMPILE_CACHE_DIR in bridge._wsl.scripts[0]


# ══════════════════════════════════════════════════════════════════════════════
# Oligomer input building (colon-join + copy count → multimer)
# ══════════════════════════════════════════════════════════════════════════════

def test_oligomer_colon_join_in_worker(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(_ok_payload())
    bridge.predict("ACDEFGHIKL", copies=3, quick=True)
    # The FASTA line embedded in the worker must be the colon-joined homo-trimer.
    assert any("ACDEFGHIKL:ACDEFGHIKL:ACDEFGHIKL" in s for s in bridge._wsl.scripts)


def test_monomer_has_no_colon(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(_ok_payload())
    bridge.predict("ACDEFGHIKL", copies=1, quick=True)
    script = bridge._wsl.scripts[0]
    # Find the embedded seq_line literal; it must not contain a chain separator.
    assert "ACDEFGHIKL" in script
    assert "ACDEFGHIKL:" not in script


# ══════════════════════════════════════════════════════════════════════════════
# Total-residue guard (blocks over budget, never launches)
# ══════════════════════════════════════════════════════════════════════════════

def test_residue_guard_blocks_over_budget(monkeypatch):
    monkeypatch.setattr(_cfg, "COLABFOLD_MAX_TOTAL_RESIDUES", 100)
    bridge = ColabFoldBridge()
    fake = _FakeWSL(_ok_payload())
    bridge._wsl = fake
    res = bridge.predict("A" * 60, copies=2)   # 120 > 100
    assert res["success"] is False
    assert res["oom_risk"] is True
    assert "budget" in res["error"].lower()
    assert fake.run_called == 0                # never launched the worker


def test_oligomer_guard_counts_copies(monkeypatch):
    monkeypatch.setattr(_cfg, "COLABFOLD_MAX_TOTAL_RESIDUES", 1500)
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(_ok_payload())
    # 500 aa x 4 copies = 2000 > 1500
    res = bridge.predict("A" * 500, copies=4)
    assert res["success"] is False and res["oom_risk"] is True
    assert "2000" in res["error"]


def test_invalid_residue_rejected():
    res = ColabFoldBridge().predict("ACDEXZ123")
    assert res["success"] is False
    assert "non-standard" in res["error"].lower()


def test_empty_sequence_rejected():
    res = ColabFoldBridge().predict("   ")
    assert res["success"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Result parsing (pLDDT / PAE / pTM / ipTM)
# ══════════════════════════════════════════════════════════════════════════════

def test_result_parsing_success(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(_ok_payload())
    res = bridge.predict("ACDEFGHIKLMNPQRSTVWY", copies=2, quick=True)
    assert res["success"] is True
    assert res["ptm"] == 0.44
    assert res["iptm"] == 0.61
    assert res["mean_plddt"] == pytest.approx((80 + 90 + 70 + 95) / 4, abs=0.01)
    assert res["plddt"][1] == 80.0 and res["plddt"][4] == 95.0   # 1-based keys
    assert res["copies"] == 2 and res["total_residues"] == 40
    assert Path(res["ranked_pdb"]).is_file()
    assert set(res["png_paths"]) == {"pae", "plddt", "coverage"}
    assert res["source"] == "colabfold_wsl2"
    # The worker must have been launched via the ColabFold env interpreter.
    assert bridge._wsl.python_bin.endswith("colabfold_env/bin/python")


def test_runtime_oom_surfaced(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(
        _ok_payload(), run_ok=False,
        stdout="2026 RESOURCE_EXHAUSTED: Out of memory while allocating",
    )
    res = bridge.predict("ACDEFGHIKLMNPQRSTVWY", quick=True)
    assert res["success"] is False
    assert res["oom_risk"] is True


def test_worker_error_payload_surfaced(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL({"success": False, "error": "no rank_001 outputs", "oom": False})
    res = bridge.predict("ACDEFGHIKLMNPQRSTVWY", quick=True)
    assert res["success"] is False
    assert "no rank_001" in res["error"]


# ── Failure-path error reporting (the REAL cause, not the benign tail) ──────────

# Realistic colabfold_batch output: the REAL Python traceback is in stdout, while
# stderr ENDS with a benign TF/oneDNN line (the noise the old single-tail surfaced).
_REAL_TRACEBACK_STDOUT = (
    "2026 Query 1/1: bench_1CRN (length 46)\n"
    "Traceback (most recent call last):\n"
    '  File ".../colabfold/batch.py", line 1, in run\n'
    "RuntimeError: THE REAL CAUSE — model params failed to load\n"
)
_BENIGN_STDERR = (
    "some earlier warning\n"
    "I0000 00:00:1780262789.707871 386 port.cc:153] oneDNN custom operations are on. "
    "You may see slightly different numerical results ... set TF_ENABLE_ONEDNN_OPTS=0.\n"
)


def test_format_worker_failure_surfaces_real_cause_not_benign_tail():
    err = _format_worker_failure(
        "colabfold_batch exited non-zero", 1,
        _REAL_TRACEBACK_STDOUT, _BENIGN_STDERR,
    )
    # The REAL cause (from stdout) must be present...
    assert "THE REAL CAUSE — model params failed to load" in err
    assert "RuntimeError" in err
    # ...with BOTH streams labelled separately (so neither buries the other).
    assert "--- STDOUT (last 60 lines) ---" in err
    assert "--- STDERR (last 60 lines) ---" in err
    assert "exit 1" in err
    # The benign oneDNN line is still shown (under its STDERR label), but it is no
    # longer the ONLY thing surfaced.
    assert "oneDNN" in err and err.strip() != _BENIGN_STDERR.strip()


def test_worker_sets_xla_platform_allocator():
    """The worker must set the on-demand XLA allocator env vars so a default
    5-model fold doesn't preallocate the giant pinned-host pool that SIGSEGVs on
    the memory-capped WSL2 env (PROJECT_CONTEXT §8)."""
    s = ColabFoldBridge._build_worker(
        seq_line="ACDE", jobname="q", out_dir="/tmp/o", fasta_path="/tmp/o/in.fasta",
        result_path="/tmp/r.json", n_models=5, n_recycle=3,
        msa_mode="mmseqs2_uniref_env", tmpl_dir="",
    )
    assert 'os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")' in s
    assert 'os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")' in s
    # set BEFORE colabfold_batch is launched so the child inherits them
    assert s.index("XLA_PYTHON_CLIENT_ALLOCATOR") < s.index("subprocess.run")


def test_worker_uses_both_stream_labels_not_single_tail():
    """The standalone worker must mirror the both-stream-labelled failure format
    and NOT the old single concatenated `log[-3000:]` tail."""
    s = ColabFoldBridge._build_worker(
        seq_line="ACDE", jobname="q", out_dir="/tmp/o", fasta_path="/tmp/o/in.fasta",
        result_path="/tmp/r.json", n_models=5, n_recycle=3,
        msa_mode="mmseqs2_uniref_env", tmpl_dir="",
    )
    assert "STDOUT (last 60 lines)" in s and "STDERR (last 60 lines)" in s
    assert "log[-3000:]" not in s            # old single-tail truncation gone
    assert "stdout_tail" in s and "stderr_tail" in s


def test_bridge_propagates_full_worker_error_not_300_clip(cache_tmp):
    """The bridge must surface the full labelled worker error (real cause), not a
    300-char clip of the benign tail."""
    big_error = _format_worker_failure(
        "colabfold_batch exited non-zero", 1,
        _REAL_TRACEBACK_STDOUT + ("x" * 500), _BENIGN_STDERR,
    )
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL({"success": False, "error": big_error, "oom": False,
                            "returncode": 1, "stdout_tail": _REAL_TRACEBACK_STDOUT,
                            "stderr_tail": _BENIGN_STDERR})
    res = bridge.predict("ACDEFGHIKLMNPQRSTVWY", quick=True)
    assert res["success"] is False
    assert "THE REAL CAUSE" in res["error"]            # real cause propagated
    assert len(res["error"]) > 300                     # not the old 300-char clip
    assert res.get("stderr_tail") and res.get("returncode") == 1


# ══════════════════════════════════════════════════════════════════════════════
# Cache hit / miss
# ══════════════════════════════════════════════════════════════════════════════

def test_cache_hit_returns_without_folding(cache_tmp):
    bridge = ColabFoldBridge()
    fake = _FakeWSL(_ok_payload())
    bridge._wsl = fake
    seq = "ACDEFGHIKLMNPQRSTVWY"

    first = bridge.predict(seq, copies=1, quick=True)
    assert first["success"] and first["cached"] is False
    assert fake.run_called == 1

    # Second identical call → cache hit, no second worker launch.
    second = bridge.predict(seq, copies=1, quick=True)
    assert second["success"] and second["cached"] is True
    assert second["source"] == "cache"
    assert fake.run_called == 1                       # unchanged
    assert second["plddt"][1] == 80.0                 # int keys restored from JSON


def test_cache_key_changes_with_inputs():
    k = ColabFoldBridge._cache_key
    base = k("ACDE", 1, None, 5, 3)
    assert base != k("ACDE", 2, None, 5, 3)           # copies
    assert base != k("ACDE", 1, None, 1, 3)           # models
    assert base != k("ACDF", 1, None, 5, 3)           # sequence


# ══════════════════════════════════════════════════════════════════════════════
# ETA estimate
# ══════════════════════════════════════════════════════════════════════════════

def test_eta_scales_and_warm_is_cheaper():
    b = ColabFoldBridge()
    cold = b.estimate_runtime_s(100, 5, 3)
    b._compiled_this_session = True
    warm = b.estimate_runtime_s(100, 5, 3)
    assert cold > warm
    assert b.estimate_runtime_s(400, 5, 3) > b.estimate_runtime_s(100, 5, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Intent detection + routing
# ══════════════════════════════════════════════════════════════════════════════

def _router():
    return ToolRouter(bridge=None, session=SessionState())


@pytest.mark.parametrize("text", [
    "fold MKTAYIAK as a dimer with colabfold",
    "use alphafold to predict this structure",
    "fold ACDEFGHIKLMNPQRSTVWY as a tetramer",
    "use PDB 1HSG as template to fold my sequence",
])
def test_colabfold_intent_positive(text):
    assert _router()._detect_colabfold_intent(text) is True


@pytest.mark.parametrize("text", [
    "validate design with esmfold",
    "fold sequence",                       # generic ESMFold phrasing
    "check fold of mutation I64E",
    "suggest mutations to improve solubility",
])
def test_colabfold_intent_negative(text):
    assert _router()._detect_colabfold_intent(text) is False


def test_route_selects_colabfold():
    r = _router()
    routed = r.route(
        {"tools_needed": ["chimerax"], "tool_inputs": {}},
        user_input="fold ACDEFGHIKLMNPQRSTVWYACDE as a dimer with colabfold",
    )
    assert routed["tools_needed"] == ["colabfold"]
    cf = routed["tool_inputs"]["colabfold"]
    assert cf["copies"] == 2
    assert cf["sequence"] == "ACDEFGHIKLMNPQRSTVWYACDE"


def test_route_does_not_hijack_esmfold():
    r = _router()
    routed = r.route(
        {"tools_needed": ["esmfold"], "tool_inputs": {"esmfold": {}}},
        user_input="check the fold of this design",
    )
    assert "colabfold" not in routed["tools_needed"]


def test_parse_template_and_quick():
    opts = _router()._parse_colabfold_options(
        "use PDB 2lzm as template to fold ACDEFGHIKLMNPQRSTVWYAC quick"
    )
    assert opts["template"] == "2LZM"
    assert opts["quick"] is True
    assert opts["sequence"] == "ACDEFGHIKLMNPQRSTVWYAC"


# ══════════════════════════════════════════════════════════════════════════════
# Viz command construction
# ══════════════════════════════════════════════════════════════════════════════

def test_viz_commands_basic():
    r = ToolRouter(bridge=None, session=SessionState())
    result = {"ranked_pdb": "C:/cache/colabfold_x/ranked.pdb", "copies": 1}
    cmds, exps = r._build_colabfold_viz(result, {"chain": "A"}, model_id="1")
    joined = "\n".join(cmds)
    assert any(c.startswith("open ") for c in cmds)
    assert "color byattribute bfactor" in joined and "palette alphafold" in joined
    # Structure-only: the predicted-model viz no longer opens a ChimeraX Sequence Viewer.
    assert not any(c.startswith("sequence chain") for c in cmds)
    assert len(cmds) == len(exps)
    # No structure loaded → no matchmaker
    assert "matchmaker" not in joined


def test_viz_matchmaker_when_compare_to():
    r = ToolRouter(bridge=None, session=SessionState())
    result = {"ranked_pdb": "C:/cache/colabfold_x/ranked.pdb", "copies": 1}
    cmds, _ = r._build_colabfold_viz(result, {"compare_to": "#1/A"}, model_id="2")
    assert any(c.startswith("matchmaker ") and "#1/A" in c for c in cmds)


def test_viz_empty_when_no_pdb():
    r = ToolRouter(bridge=None, session=SessionState())
    cmds, exps = r._build_colabfold_viz({"ranked_pdb": ""}, {}, model_id="1")
    assert cmds == [] and exps == []


# ══════════════════════════════════════════════════════════════════════════════
# Structure-only: ChimeraX never opens a Sequence Viewer (sequence lives in the
# StructureBot window). The former sequence-on-open hook was removed 2026-06-16.
# ══════════════════════════════════════════════════════════════════════════════

def test_open_emits_no_sequence_viewer(monkeypatch):
    bridge = ChimeraXBridge(chimerax_path="X", port=60001)
    calls = []

    def fake_run(command, timeout=30):
        calls.append(command)
        if command.startswith("open "):
            return {"value": "Opened test.pdb as #3, 1 model(s)", "error": None}
        return {"value": "", "error": None}

    monkeypatch.setattr(bridge, "run_command", fake_run)
    bridge.run_commands(["open 1hsg"])
    assert not any(c.startswith("sequence chain") for c in calls)


# ══════════════════════════════════════════════════════════════════════════════
# Availability / skip gating
# ══════════════════════════════════════════════════════════════════════════════

def test_predict_errors_when_env_unavailable(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(_ok_payload(), available=False)
    res = bridge.predict("ACDEFGHIKLMNPQRSTVWY", quick=True)
    assert res["success"] is False
    assert "wsl2" in res["error"].lower()


def test_predict_errors_when_colabfold_missing(cache_tmp):
    bridge = ColabFoldBridge()
    bridge._wsl = _FakeWSL(_ok_payload(), cf=False)
    res = bridge.predict("ACDEFGHIKLMNPQRSTVWY", quick=True)
    assert res["success"] is False
    assert "colabfold env" in res["error"].lower()


# ── Optional live fold (opt-in; never runs in CI) ───────────────────────────────

_RUN_LIVE = os.environ.get("STRUCTUREBOT_RUN_LIVE_COLABFOLD") == "1"


@pytest.mark.gpu
@pytest.mark.skipif(
    not _RUN_LIVE,
    reason="live ColabFold fold (minutes, GPU); set STRUCTUREBOT_RUN_LIVE_COLABFOLD=1 to run",
)
def test_live_tiny_monomer_fold():
    bridge = ColabFoldBridge()
    if not bridge.is_available():
        pytest.skip("ColabFold env / WSL2 not available")
    seq = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"   # villin HP36
    res = bridge.predict(seq, copies=1, quick=True)
    assert res["success"], res.get("error")
    assert res["mean_plddt"] > 0
    assert Path(res["ranked_pdb"]).is_file()
