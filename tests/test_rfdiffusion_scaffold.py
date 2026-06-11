"""
tests/test_rfdiffusion_scaffold.py
----------------------------------
Tests for the RFdiffusion bridge SCAFFOLD additions (Part A), complementary to
tests/test_rfdiffusion.py (the 12 stub tests, which stay green untouched).

Covers the NEW logic only — all inference MOCKED, no GPU, no real WSL run:
  E. NL -> contig / hotspot / symmetry parsing (the substantive new logic)
  F. _build_cmd additive tokens (binder contig, --config-name symmetry, interpreter)
  G. clone-location availability (win_clone vs wsl_clone probe vs honest error)
  H. content-hash caching (re-run returns cached, no recompute)
  I. error-first (no PDBs -> failure, never fabricated)
  J. handoff: backbone -> EXISTING ProteinMPNN -> EXISTING ColabFold (inputs reused)

Usage
-----
  cd structurebot
  python tests/test_rfdiffusion_scaffold.py
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as _cfg
import rfdiffusion_bridge as _rfd_mod
from rfdiffusion_bridge import (
    RFdiffusionBridge,
    build_binder_contig,
    build_motif_contig,
    normalize_symmetry,
)
from wsl_bridge import WSLBridge, RFDIFFUSION_PYTHON
from tool_router import ToolStepResult

PASS, FAIL = "[PASS]", "[FAIL]"
_results = {"pass": 0, "fail": 0, "skip": 0}


def _assert(cond: bool, name: str, msg: str = "") -> bool:
    if cond:
        print(f"  {PASS} {name}")
        _results["pass"] += 1
        return True
    print(f"  {FAIL} {name}: {msg or 'assertion failed'}")
    _results["fail"] += 1
    return False


# -- Helpers -------------------------------------------------------------------

def _make_available_bridge(td: str) -> RFdiffusionBridge:
    """Windows-visible fake clone -> win_clone backend (dispatch-mockable)."""
    (Path(td) / "run_inference.py").write_text("# stub\n")
    (Path(td) / "models").mkdir(exist_ok=True)
    with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", td):
        return RFdiffusionBridge()


def _write_pdb(path: Path, resnums) -> None:
    lines = []
    for i, r in enumerate(resnums, 1):
        lines.append(
            f"ATOM  {i:>5}  CA  ALA A {r:>3}      "
            f"{1.0*i:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00 10.00\n"
        )
    path.write_text("".join(lines))


# -- E. Parsing ----------------------------------------------------------------

def test_build_binder_contig() -> None:
    print("\n=== E. NL -> contig / hotspot / symmetry parsing ===")
    _assert(build_binder_contig("A", (1, 100), 80) == "A1-100/0 80-80",
            "binder contig: target range /0 + binder length")
    _assert(build_binder_contig("B", (5, 72), 60) == "B5-72/0 60-60",
            "binder contig honours chain id + extent")
    _assert(build_binder_contig("A", None, 80) == "",
            "binder contig empty when target extent unknown (never guess)")


def test_build_motif_contig() -> None:
    got = build_motif_contig("A", [10, 11, 12, 20, 21], flank=5)
    _assert(got == "5-5/A10-12/5-5/A20-21/5-5",
            "motif contig collapses contiguous + spacers across gaps", got)
    _assert(build_motif_contig("A", [], flank=5) == "",
            "motif contig empty when no motif residues")


def test_normalize_symmetry() -> None:
    cases = {
        "C3": "c3", "c3": "c3", "D2": "d2", "tetrahedral": "tetrahedral",
        "OCTAHEDRAL": "octahedral", "3-fold": "c3", "": "", "nonsense": "",
    }
    ok = all(normalize_symmetry(k) == v for k, v in cases.items())
    _assert(ok, "normalize_symmetry canonicalises cyclic/dihedral/named",
            {k: normalize_symmetry(k) for k in cases})


def test_resolve_spec_binder_derives_contig() -> None:
    # Patch the residue-extent spine so the test exercises the _resolve_spec
    # wiring deterministically (BioPython's column parsing is covered elsewhere).
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        pdb = Path(td) / "target.pdb"
        _write_pdb(pdb, [10, 11, 12])
        with patch.object(_rfd_mod, "_chain_extent", lambda p, c: (10, 12)):
            spec = b._resolve_spec({
                "mode": "binder", "pdb_path": str(pdb), "chain_id": "A",
                "hotspot_residues": [11], "binder_length": 90,
            })
    _assert(spec["contigs"] == "A10-12/0 90-90",
            "binder _resolve_spec derives contig from PDB extent", spec["contigs"])
    _assert(spec["hotspots"] == [11], "hotspots carried through")


def test_resolve_spec_symmetry_normalized() -> None:
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        spec = b._resolve_spec({"mode": "symmetric", "symmetry": "C3"})
    _assert(spec["symmetry"] == "c3",
            "symmetric _resolve_spec normalises C3 -> c3", spec["symmetry"])


# -- F. _build_cmd additive tokens ---------------------------------------------

def test_build_cmd_binder_adds_contig() -> None:
    print("\n=== F. _build_cmd additive tokens ===")
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        cmd = b._build_cmd(
            mode="binder", pdb_path="/fake/t.pdb", chain_id="A",
            hotspots=[82, 119], num_designs=4, num_steps=50, symmetry="",
            partial_T=0.2, contigs="A1-100/0 80-80", out_path=Path(td) / "o",
        )
    cmd_str = " ".join(cmd)
    _assert("contigmap.contigs=[A1-100/0 80-80]" in cmd_str,
            "binder command includes contigmap.contigs", cmd_str)
    _assert("ppi.hotspot_res=[A82,A119]" in cmd_str, "binder still has hotspots")


def test_build_cmd_symmetric_config_name() -> None:
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        cmd = b._build_cmd(
            mode="symmetric", pdb_path="", chain_id="A", hotspots=[],
            num_designs=2, num_steps=50, symmetry="c3", partial_T=0.2,
            contigs="", out_path=Path(td) / "o",
        )
    _assert("--config-name" in cmd and "symmetry" in cmd,
            "symmetric command prepends --config-name symmetry", cmd)


def test_build_cmd_interpreter_is_rfdiffusion_python() -> None:
    with tempfile.TemporaryDirectory() as td:
        b = _make_available_bridge(td)
        cmd = b._build_cmd(
            mode="partial_diffusion", pdb_path="/fake/i.pdb", chain_id="A",
            hotspots=[], num_designs=1, num_steps=50, symmetry="",
            partial_T=0.2, contigs="", out_path=Path(td) / "o",
        )
    _assert(cmd[0] == RFDIFFUSION_PYTHON,
            "interpreter is RFDIFFUSION_PYTHON (the WSL env)", cmd[0])
    _assert("venv312" not in " ".join(cmd).lower(),
            "VENV312 never appears in the RFdiffusion command")


# -- G. dual-backend availability ----------------------------------------------

def test_wsl_backend_available_when_probe_succeeds() -> None:
    print("\n=== G. dual-backend availability ===")
    with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", "/nonexistent/win/path"), \
         patch.object(WSLBridge, "is_available", lambda self: True), \
         patch.object(WSLBridge, "check_rfdiffusion", lambda self, d="/x": True):
        b = RFdiffusionBridge()
    _assert(b.is_available(), "WSL probe success -> available")
    _assert(b._backend == "wsl_clone", "backend is 'wsl_clone' when WSL clone present",
            b._backend)


def test_wsl_backend_unavailable_honest_error() -> None:
    with patch.object(_rfd_mod, "_RFDIFFUSION_DIR", "/nonexistent/win/path"), \
         patch.object(WSLBridge, "is_available", lambda self: True), \
         patch.object(WSLBridge, "check_rfdiffusion", lambda self, d="/x": False):
        b = RFdiffusionBridge()
        res = b.analyze({"mode": "binder", "pdb_path": "/any.pdb"})
    _assert(not b.is_available(), "WSL probe fail -> not available")
    _assert(not res.success and "not yet configured" in (res.error or "").lower(),
            "honest 'not configured' error, never a fake run", res.error)


# -- H. caching ----------------------------------------------------------------

def test_cache_hit_skips_recompute() -> None:
    print("\n=== H. content-hash caching ===")
    with tempfile.TemporaryDirectory() as td, \
         tempfile.TemporaryDirectory() as cache:
        with patch.object(_cfg, "RFDIFFUSION_CACHE_DIR", Path(cache)):
            b = _make_available_bridge(td)
            pdb = Path(td) / "t.pdb"
            _write_pdb(pdb, [10, 11, 12])
            inputs = {"mode": "binder", "pdb_path": str(pdb), "chain_id": "A",
                      "hotspot_residues": [11], "num_designs": 2}

            def fake_dispatch(self, cmd, run_dir):
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "binder_0.pdb").write_text("REMARK 0\n")

            def boom_dispatch(self, cmd, run_dir):
                raise RuntimeError("should not run")

            with patch.object(_rfd_mod.RFdiffusionBridge, "_dispatch", fake_dispatch):
                r1 = b.analyze(inputs)
            # Second call: dispatch RAISES — a cache hit must avoid it entirely.
            with patch.object(_rfd_mod.RFdiffusionBridge, "_dispatch", boom_dispatch):
                r2 = b.analyze(inputs)

    _assert(r1.success and not r1.data.get("cached"), "first run computes (cached=False)",
            f"success={r1.success} err={r1.error}")
    _assert(r2.success and r2.data.get("cached") is True,
            "second identical run returns cache (cached=True), no recompute", r2.error)


# -- I. error-first ------------------------------------------------------------

def test_no_pdb_output_is_honest_failure() -> None:
    print("\n=== I. error-first (no fabrication) ===")
    with tempfile.TemporaryDirectory() as td, \
         tempfile.TemporaryDirectory() as cache:
        with patch.object(_cfg, "RFDIFFUSION_CACHE_DIR", Path(cache)):
            b = _make_available_bridge(td)
            pdb = Path(td) / "t.pdb"
            _write_pdb(pdb, [10, 11])

            def noout_dispatch(self, cmd, run_dir):
                return  # "succeeds" but writes NO pdb

            with patch.object(_rfd_mod.RFdiffusionBridge, "_dispatch", noout_dispatch):
                res = b.analyze({"mode": "binder", "pdb_path": str(pdb),
                                 "chain_id": "A", "hotspot_residues": [10]})
    _assert(not res.success, "no PDB output -> failure (never a fabricated backbone)")
    _assert("no .pdb output" in (res.error or "") or "no backbone" in (res.error or ""),
            "error explains nothing was generated", res.error)


# -- J. handoff ----------------------------------------------------------------

def test_handoff_reuses_mpnn_and_colabfold() -> None:
    print("\n=== J. handoff -> EXISTING ProteinMPNN -> EXISTING ColabFold ===")
    with tempfile.TemporaryDirectory() as td:
        backbone = Path(td) / "design_0.pdb"
        _write_pdb(backbone, [1, 2, 3])

        captured = {}

        class FakeMPNN:
            def analyze(self, inputs, session=None):
                captured["mpnn_inputs"] = inputs
                return ToolStepResult(
                    tool="proteinmpnn", success=True,
                    data={"sequences": [{"sequence": "ACDEFGHIK"}]},
                )

        class FakeCF:
            def predict(self, seq, label="x", **kw):
                captured["fold_seq"] = seq
                return {"success": True, "mean_plddt": 88.0, "ranked_pdb": "r.pdb"}

        with patch("proteinmpnn_bridge.ProteinMPNNBridge", FakeMPNN), \
             patch("colabfold_bridge.ColabFoldBridge", FakeCF):
            b = RFdiffusionBridge()
            out = b.run_handoff(str(backbone), chain_id="A", num_sequences=4)

    _assert(captured.get("mpnn_inputs", {}).get("pdb_path") == str(backbone),
            "handoff passes the generated backbone PDB to ProteinMPNN")
    _assert(captured["mpnn_inputs"].get("chain_id") == "A", "handoff passes chain_id")
    _assert(captured.get("fold_seq") == "ACDEFGHIK",
            "handoff forwards the top MPNN sequence to ColabFold.predict")
    _assert(out["success"] and out["stage"] == "colabfold", "handoff reports success")


def test_handoff_error_first_on_mpnn_failure() -> None:
    with tempfile.TemporaryDirectory() as td:
        backbone = Path(td) / "design_0.pdb"
        _write_pdb(backbone, [1, 2, 3])

        class FailMPNN:
            def analyze(self, inputs, session=None):
                return ToolStepResult(tool="proteinmpnn", success=False,
                                      error="not configured")

        cf_called = {"v": False}

        class FakeCF:
            def predict(self, seq, label="x", **kw):
                cf_called["v"] = True
                return {"success": True}

        with patch("proteinmpnn_bridge.ProteinMPNNBridge", FailMPNN), \
             patch("colabfold_bridge.ColabFoldBridge", FakeCF):
            out = RFdiffusionBridge().run_handoff(str(backbone))

    _assert(not out["success"] and out["stage"] == "proteinmpnn",
            "handoff is error-first: MPNN failure stops before folding")
    _assert(cf_called["v"] is False, "ColabFold not called when MPNN fails")


# -- Runner --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("tests/test_rfdiffusion_scaffold.py -- RFdiffusion scaffold (Part A)")
    print("=" * 60)
    test_build_binder_contig()
    test_build_motif_contig()
    test_normalize_symmetry()
    test_resolve_spec_binder_derives_contig()
    test_resolve_spec_symmetry_normalized()
    test_build_cmd_binder_adds_contig()
    test_build_cmd_symmetric_config_name()
    test_build_cmd_interpreter_is_rfdiffusion_python()
    test_wsl_backend_available_when_probe_succeeds()
    test_wsl_backend_unavailable_honest_error()
    test_cache_hit_skips_recompute()
    test_no_pdb_output_is_honest_failure()
    test_handoff_reuses_mpnn_and_colabfold()
    test_handoff_error_first_on_mpnn_failure()
    print()
    print("=" * 60)
    print(f"Results: {_results['pass']} passed, {_results['fail']} failed, "
          f"{_results['skip']} skipped")
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
