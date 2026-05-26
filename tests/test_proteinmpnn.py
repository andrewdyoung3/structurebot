"""
tests/test_proteinmpnn.py
-------------------------
Tests for ProteinMPNNBridge (proteinmpnn_bridge.py).

Test categories
---------------
A. Utility helpers    -- _diff_sequences, _sequence_recovery
B. FASTA parsing      -- _parse_proteinmpnn_fasta
C. Visualization      -- _build_recovery_viz
D. Availability       -- _check_available logic
E. analyze() errors   -- missing dir, missing pdb
F. Full pipeline      -- mocked subprocess call

Usage
-----
  cd structurebot
  python -m pytest tests/test_proteinmpnn.py -v
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import proteinmpnn_bridge as _pmpnn_mod
from proteinmpnn_bridge import (
    ProteinMPNNBridge,
    _diff_sequences,
    _sequence_recovery,
    _build_recovery_viz,
)

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


def _approx_eq(a: float, b: float, tol: float = 0.001) -> bool:
    return abs(a - b) < tol


# -- A. Utility helpers --------------------------------------------------------

def test_diff_sequences_identical() -> None:
    print("\n=== A. Utility helpers ===")
    mutations = _diff_sequences("ACDEF", "ACDEF")
    _assert(mutations == [], "identical sequences -> no mutations",
            f"got {mutations}")


def test_diff_sequences_one_change() -> None:
    # Position 3: G -> K
    mutations = _diff_sequences("ACGEF", "ACKEF")
    _assert(len(mutations) == 1, "one mutation found",
            f"got {mutations}")
    _assert(mutations == ["G3K"], "mutation label correct",
            f"got {mutations[0]!r}")


def test_diff_sequences_multiple() -> None:
    # A->V at 1, G->K at 3
    mutations = _diff_sequences("ACGEF", "VCKEF")
    _assert(len(mutations) == 2, "two mutations found",
            f"got {mutations}")
    _assert("A1V" in mutations and "G3K" in mutations,
            "both mutations identified",
            f"got {mutations}")


def test_sequence_recovery_identical() -> None:
    rec = _sequence_recovery("ACDEF", "ACDEF")
    _assert(_approx_eq(rec, 1.0), "identical sequences -> recovery 1.0",
            f"got {rec:.4f}")


def test_sequence_recovery_zero() -> None:
    rec = _sequence_recovery("AAAAA", "CCCCC")
    _assert(_approx_eq(rec, 0.0), "fully different sequences -> recovery 0.0",
            f"got {rec:.4f}")


def test_sequence_recovery_partial() -> None:
    # 2 of 4 match
    rec = _sequence_recovery("ACDE", "ACFF")
    _assert(_approx_eq(rec, 0.5, tol=0.01), "50% recovery",
            f"got {rec:.4f}")


def test_sequence_recovery_empty() -> None:
    rec = _sequence_recovery("", "")
    _assert(_approx_eq(rec, 0.0), "empty sequences -> 0.0",
            f"got {rec:.4f}")


# -- B. FASTA parsing ----------------------------------------------------------

def _write_fasta(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_parse_fasta_basic() -> None:
    print("\n=== B. FASTA parsing ===")
    fasta_content = (
        ">score=-1.5000, global_score=-1.4, seq_recovery=0.87, T=0.1, sample=1\n"
        "ACDEFGHIK\n"
        ">score=-1.2000, global_score=-1.3, seq_recovery=0.90, T=0.1, sample=2\n"
        "ACDEFGHIL\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "test.fa"
        _write_fasta(fa, fasta_content)
        b = ProteinMPNNBridge()   # availability doesn't matter for _parse
        result = b._parse_proteinmpnn_fasta(fa)

    # First entry is WT; remaining is designed
    _assert(result["wildtype_sequence"] == "ACDEFGHIK",
            "wildtype_sequence parsed correctly",
            f"got {result['wildtype_sequence']!r}")
    seqs = result["sequences"]
    _assert(len(seqs) == 1, "1 designed sequence parsed",
            f"got {len(seqs)}")
    _assert(_approx_eq(seqs[0]["score"], -1.2, tol=0.01),
            "score parsed from header",
            f"got {seqs[0]['score']}")


def test_parse_fasta_mutations() -> None:
    """_parse_proteinmpnn_fasta computes correct mutation list."""
    fasta_content = (
        ">WT\n"
        "ACDEF\n"
        ">score=-1.0, sample=1\n"
        "ACKEF\n"   # D->K at position 3
    )
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "test.fa"
        _write_fasta(fa, fasta_content)
        b = ProteinMPNNBridge()
        result = b._parse_proteinmpnn_fasta(fa)

    seqs = result["sequences"]
    _assert(len(seqs) == 1, "one designed sequence")
    _assert(seqs[0]["mutations"] == ["D3K"],
            "mutation D3K detected",
            f"got {seqs[0]['mutations']}")
    _assert(_approx_eq(seqs[0]["recovery"], 0.8, tol=0.01),
            "recovery 4/5 = 0.8",
            f"got {seqs[0]['recovery']}")


def test_parse_fasta_empty() -> None:
    """Empty FASTA returns empty result dict."""
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "empty.fa"
        _write_fasta(fa, "")
        b = ProteinMPNNBridge()
        result = b._parse_proteinmpnn_fasta(fa)

    _assert(result["sequences"] == [], "empty FASTA -> empty sequences")
    _assert(result["wildtype_sequence"] == "", "empty FASTA -> empty WT")


def test_parse_fasta_sorted_by_score() -> None:
    """Designed sequences are sorted ascending by score (lower = better)."""
    fasta_content = (
        ">WT\n"
        "ACDEF\n"
        ">score=-0.5, sample=1\n"
        "ACDEF\n"
        ">score=-1.5, sample=2\n"
        "ACDEF\n"
        ">score=-1.0, sample=3\n"
        "ACDEF\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fa = Path(td) / "test.fa"
        _write_fasta(fa, fasta_content)
        b = ProteinMPNNBridge()
        result = b._parse_proteinmpnn_fasta(fa)

    scores = [s["score"] for s in result["sequences"]]
    _assert(scores == sorted(scores), "sequences sorted ascending by score",
            f"got {scores}")
    _assert(_approx_eq(scores[0], -1.5, tol=0.01), "lowest score is first",
            f"got {scores[0]}")


# -- C. Visualization ----------------------------------------------------------

def test_build_recovery_viz_mixed() -> None:
    print("\n=== C. Visualization ===")
    # WT: ACDE, designed: ACFE -> position 3 D->F is mutated
    cmds, exps = _build_recovery_viz("ACDE", "ACFE", model_id="1", chain_id="A")
    cmd_str = " ".join(cmds)
    _assert("cornflower blue" in cmd_str, "conserved residues colored blue")
    _assert("tomato" in cmd_str, "mutated residues colored red/tomato")
    _assert("#1" in cmd_str, "model_id=1 in commands")
    _assert("/A" in cmd_str, "chain A in commands")


def test_build_recovery_viz_all_conserved() -> None:
    """When all residues match WT, no red coloring."""
    cmds, exps = _build_recovery_viz("ACDE", "ACDE", model_id="2", chain_id="B")
    cmd_str = " ".join(cmds)
    _assert("cornflower blue" in cmd_str, "conserved residues colored blue")
    _assert("tomato" not in cmd_str, "no red when all conserved")


def test_build_recovery_viz_no_chain() -> None:
    """chain_id=None produces commands without /None chain specifier."""
    cmds, _ = _build_recovery_viz("AC", "AK", model_id="1", chain_id=None)
    cmd_str = " ".join(cmds)
    _assert("/None" not in cmd_str, "no /None in commands when chain_id=None")


# -- D. Availability -----------------------------------------------------------

def test_check_available_no_dir() -> None:
    print("\n=== D. Availability ===")
    # Patch the module-level _PROTEINMPNN_DIR to "" so the bridge sees no dir
    with patch.object(_pmpnn_mod, "_PROTEINMPNN_DIR", ""):
        b = ProteinMPNNBridge()
    _assert(not b._available, "not available when _PROTEINMPNN_DIR is empty")


def test_check_available_missing_weights() -> None:
    """Directory exists but vanilla_model_weights/ is absent -> not available."""
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "protein_mpnn_run.py").write_text("# stub\n")
        # vanilla_model_weights/ is intentionally absent
        with patch.object(_pmpnn_mod, "_PROTEINMPNN_DIR", td):
            b = ProteinMPNNBridge()
    _assert(not b._available, "not available when vanilla_model_weights absent")


def test_check_available_real_dir() -> None:
    """Cloned ProteinMPNN directory is detected as available."""
    repo = Path(__file__).parent.parent / "ProteinMPNN"
    if not repo.is_dir():
        _skip("check_available_real_dir", "ProteinMPNN not cloned")
        return
    with patch.object(_pmpnn_mod, "_PROTEINMPNN_DIR", str(repo)):
        b = ProteinMPNNBridge()
    _assert(b._available, "real cloned ProteinMPNN dir is available")
    _assert(b._backend == "proteinmpnn",
            f"backend=proteinmpnn (got {b._backend!r})")


# -- E. analyze() error paths --------------------------------------------------

def test_analyze_not_configured() -> None:
    print("\n=== E. analyze() errors ===")
    with patch.object(_pmpnn_mod, "_PROTEINMPNN_DIR", "/nonexistent/path"):
        b = ProteinMPNNBridge()
    result = b.analyze({"pdb_path": "/any/file.pdb"})
    _assert(not result.success, "analyze() fails when not configured")
    _assert(bool(result.error), "error message is non-empty")
    _assert(
        "not" in result.error.lower() or "PROTEINMPNN_DIR" in result.error,
        "error mentions configuration step",
        f"error: {result.error[:120]}",
    )


def test_analyze_missing_pdb() -> None:
    """analyze() returns failure when pdb_path is missing or nonexistent."""
    repo = Path(__file__).parent.parent / "ProteinMPNN"
    if not repo.is_dir():
        _skip("analyze_missing_pdb", "ProteinMPNN not cloned")
        return
    with patch.object(_pmpnn_mod, "_PROTEINMPNN_DIR", str(repo)):
        b = ProteinMPNNBridge()
    result = b.analyze({"pdb_path": "/nonexistent/protein.pdb"})
    _assert(not result.success, "analyze() fails for missing PDB")
    _assert(bool(result.error), "error message is non-empty")


def test_analyze_no_pdb_key() -> None:
    """analyze() returns failure when pdb_path key is absent."""
    repo = Path(__file__).parent.parent / "ProteinMPNN"
    if not repo.is_dir():
        _skip("analyze_no_pdb_key", "ProteinMPNN not cloned")
        return
    with patch.object(_pmpnn_mod, "_PROTEINMPNN_DIR", str(repo)):
        b = ProteinMPNNBridge()
    result = b.analyze({})   # no pdb_path key
    _assert(not result.success, "analyze() fails with no pdb_path key")


# -- F. Full pipeline (subprocess mocked) -------------------------------------

def test_full_pipeline_mock() -> None:
    """Full analyze() pipeline with subprocess.run mocked to produce a FASTA."""
    print("\n=== F. Full pipeline (mocked) ===")

    repo = Path(__file__).parent.parent / "ProteinMPNN"
    if not repo.is_dir():
        _skip("full_pipeline_mock", "ProteinMPNN not cloned")
        return

    # Minimal PDB content (2 residues)
    pdb_content = (
        "ATOM      1  CA  LEU A  10       1.000   2.000   3.000  1.00 10.00\n"
        "ATOM      2  CB  LEU A  10       1.500   2.500   3.500  1.00 10.00\n"
        "ATOM      3  CA  VAL A  11       4.000   5.000   6.000  1.00 10.00\n"
        "ATOM      4  CB  VAL A  11       4.500   5.500   6.500  1.00 10.00\n"
    )

    fasta_output = (
        ">WT, score=-1.0000\n"
        "LV\n"
        ">score=-1.3000, sample=1\n"
        "IV\n"
        ">score=-1.1000, sample=2\n"
        "LI\n"
    )

    with tempfile.TemporaryDirectory() as td:
        pdb_path = Path(td) / "test.pdb"
        pdb_path.write_text(pdb_content)

        def fake_run(cmd, **kwargs):
            """Simulate ProteinMPNN writing a FASTA to the expected output path."""
            try:
                idx = cmd.index("--out_folder")
                out_folder = Path(cmd[idx + 1])
            except (ValueError, IndexError):
                raise RuntimeError("--out_folder not found in cmd")
            seqs_dir = out_folder / "seqs"
            seqs_dir.mkdir(parents=True, exist_ok=True)
            stem = pdb_path.stem
            (seqs_dir / f"{stem}.fa").write_text(fasta_output, encoding="utf-8")

        with patch.object(_pmpnn_mod, "_PROTEINMPNN_DIR", str(repo)):
            b = ProteinMPNNBridge()

        with patch("subprocess.run", side_effect=fake_run):
            result = b.analyze({
                "pdb_path":      str(pdb_path),
                "chain_id":      "A",
                "num_sequences": 2,
                "temperature":   0.1,
                "model_id":      "1",
            })

    _assert(result.success, "analyze() succeeded with mocked subprocess",
            f"error: {result.error}")
    _assert(isinstance(result.data, dict), "result.data is a dict")
    _assert("sequences" in result.data, "data has 'sequences' key")
    _assert(len(result.data["sequences"]) == 2, "2 designed sequences",
            f"got {len(result.data.get('sequences', []))}")
    _assert(result.data["wildtype_sequence"] == "LV",
            "wildtype_sequence correct",
            f"got {result.data.get('wildtype_sequence')!r}")

    seqs = result.data["sequences"]
    _assert(seqs[0]["score"] <= seqs[1]["score"],
            "sequences sorted by score ascending",
            f"{seqs[0]['score']} vs {seqs[1]['score']}")
    _assert(isinstance(result.viz_commands, list) and len(result.viz_commands) > 0,
            "viz_commands non-empty")
    _assert(isinstance(result.summary, str) and "MPNN" in result.summary,
            "summary mentions MPNN",
            f"summary: {result.summary!r}")


# -- G. _generate_summary tests -----------------------------------------------

def _make_seq(score: float = -1.2, recovery: float = 0.85,
              mutations: Optional[List[str]] = None) -> Dict[str, Any]:
    return {
        "sequence":  "ACDEFGHIK",
        "score":     score,
        "recovery":  recovery,
        "mutations": mutations or ["A1V", "C2G"],
    }


def test_generate_summary_returns_string() -> None:
    print("\n=== G. _generate_summary ===")
    seqs = [_make_seq()]
    result = ProteinMPNNBridge._generate_summary(
        sequences=seqs, wt_seq="VCDEFGHIK",
        fixed_positions=[], backend="proteinmpnn",
    )
    _assert(isinstance(result, str) and len(result) > 0,
            "generate_summary returns non-empty string")
    _assert("\n" in result,
            "generate_summary returns multi-line string")


def test_generate_summary_mentions_mpnn() -> None:
    """Summary mentions ProteinMPNN or MPNN."""
    seqs = [_make_seq()]
    result = ProteinMPNNBridge._generate_summary(
        sequences=seqs, wt_seq="VCDEFGHIK",
        fixed_positions=[], backend="proteinmpnn",
    )
    _assert("MPNN" in result or "mpnn" in result.lower(),
            "summary mentions MPNN")


def test_generate_summary_mentions_next_steps() -> None:
    """Summary includes recommended next steps."""
    seqs = [_make_seq()]
    result = ProteinMPNNBridge._generate_summary(
        sequences=seqs, wt_seq="VCDEFGHIK",
        fixed_positions=[5, 10], backend="proteinmpnn",
    ).lower()
    has_steps = (
        "next step" in result
        or "esmfold" in result
        or "screen" in result
        or "expression" in result
    )
    _assert(has_steps,
            "summary mentions next steps or ESMFold/screening")


def test_generate_summary_empty_sequences() -> None:
    """Empty sequences list returns a short fallback string."""
    result = ProteinMPNNBridge._generate_summary(
        sequences=[], wt_seq="", fixed_positions=[], backend="proteinmpnn",
    )
    _assert(isinstance(result, str) and len(result) > 0,
            "empty sequences: summary is non-empty fallback string")
    _assert("No sequences" in result or "no sequences" in result.lower(),
            "fallback mentions no sequences")


def test_generate_summary_fixed_positions() -> None:
    """Summary mentions fixed positions when they are set."""
    seqs = [_make_seq()]
    result = ProteinMPNNBridge._generate_summary(
        sequences=seqs, wt_seq="VCDEFGHIK",
        fixed_positions=[3, 7, 12], backend="proteinmpnn",
    )
    _assert("Fixed" in result or "fixed" in result,
            "summary mentions fixed positions")


# -- Runner --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("tests/test_proteinmpnn.py -- ProteinMPNN Bridge Tests")
    print("=" * 60)

    # A. Utility helpers
    test_diff_sequences_identical()
    test_diff_sequences_one_change()
    test_diff_sequences_multiple()
    test_sequence_recovery_identical()
    test_sequence_recovery_zero()
    test_sequence_recovery_partial()
    test_sequence_recovery_empty()

    # B. FASTA parsing
    test_parse_fasta_basic()
    test_parse_fasta_mutations()
    test_parse_fasta_empty()
    test_parse_fasta_sorted_by_score()

    # C. Visualization
    test_build_recovery_viz_mixed()
    test_build_recovery_viz_all_conserved()
    test_build_recovery_viz_no_chain()

    # D. Availability
    test_check_available_no_dir()
    test_check_available_missing_weights()
    test_check_available_real_dir()

    # E. Error paths
    test_analyze_not_configured()
    test_analyze_missing_pdb()
    test_analyze_no_pdb_key()

    # F. Full pipeline
    test_full_pipeline_mock()

    # G. _generate_summary
    test_generate_summary_returns_string()
    test_generate_summary_mentions_mpnn()
    test_generate_summary_mentions_next_steps()
    test_generate_summary_empty_sequences()
    test_generate_summary_fixed_positions()

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
