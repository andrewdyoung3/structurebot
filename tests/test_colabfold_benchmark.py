"""
tests/test_colabfold_benchmark.py
---------------------------------
OPT-IN ColabFold structure-prediction accuracy benchmark — a REGRESSION GUARD,
NOT a novel accuracy claim.

Folds a small panel of well-characterized monomers with known deposited
structures and measures predicted-vs-native ALL-PAIRS Cα RMSD + mean pLDDT,
records every result, and gates on PANEL MEDIANS (conservative thresholds AF2 is
expected to clear easily). The gate catches PIPELINE breakage — wrong chain,
silent CPU fallback, an MSA/parse bug — not accuracy frontiers.

Honesty pattern mirrors tests/test_rosetta_benchmark.py exactly:
  * SKIP BY DEFAULT — runs live ONLY when STRUCTUREBOT_RUN_LIVE_COLABFOLD=1 AND
    the ColabFold env is available. CI / no-env → collects + skips, 0 folds.
  * AGGREGATE gate with a MIN_BENCHMARK_ENTRIES guard that SKIPS (not fails)
    below the floor; per-protein results are RECORDED, not hard-gated on RMSD.
  * RMSD is computed HEADLESS with BioPython (no ChimeraX). Folding reuses the
    ColabFold bridge (result cache + JAX compile cache); folding is not
    reimplemented here.

Run live:
    STRUCTUREBOT_RUN_LIVE_COLABFOLD=1 pytest tests/test_colabfold_benchmark.py -m benchmark -v -s
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from colabfold_bridge import ColabFoldBridge  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT  = Path(__file__).parent.parent
CACHE_DIR = _PROJECT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_RESULTS_PATH = _PROJECT / "scripts" / "colabfold_benchmark_results.json"

# Minimum number of valid live results before the aggregate gate is meaningful.
# Below this the gate SKIPS (never fails) — a partial/empty live run must never
# turn it red. The panel has 4 proteins; 3 is a majority.
MIN_BENCHMARK_ENTRIES = 3

# ── Conservative regression-guard thresholds (NOT accuracy frontiers) ───────────
# AF2 nails these well-characterized monomers (sub-Å to ~1.5 Å on a good day), so
# a generous bound flags pipeline breakage, not modelling error.
MEDIAN_RMSD_MAX_A    = 3.0    # Å — median predicted-vs-native all-pairs Cα RMSD
MEDIAN_PLDDT_MIN     = 70.0   # median mean-pLDDT floor

# ── Panel: well-characterized single-chain monomers (pdb_id, chain) ─────────────
# crambin, ubiquitin, GB1 (B1 domain of protein G), T4 lysozyme.
PANEL: List[Tuple[str, str]] = [
    ("1CRN", "A"),   # crambin,        ~46 aa
    ("1UBQ", "A"),   # ubiquitin,      ~76 aa
    ("1PGB", "A"),   # GB1,            ~56 aa
    ("2LZM", "A"),   # T4 lysozyme,   ~164 aa (the larger case; within GPU budget)
]

# ── Skip condition (opt-in + env) — mirrors the rosetta benchmark gate ──────────
# Live folds are minutes each on the GPU; this must NEVER run by accident just
# because the ColabFold env is installed. Run live ONLY on explicit opt-in. The
# `and` short-circuits so a bare (non-opted-in) collection never probes the env.
RUN_LIVE_BENCHMARK = os.environ.get("STRUCTUREBOT_RUN_LIVE_COLABFOLD") == "1"
_ENV_AVAILABLE = bool(RUN_LIVE_BENCHMARK and ColabFoldBridge().is_available())

pytestmark = [
    pytest.mark.skipif(
        not RUN_LIVE_BENCHMARK,
        reason="live ColabFold accuracy benchmark (minutes/fold, GPU); "
               "set STRUCTUREBOT_RUN_LIVE_COLABFOLD=1 to run",
    ),
    pytest.mark.skipif(
        RUN_LIVE_BENCHMARK and not _ENV_AVAILABLE,
        reason="ColabFold env / WSL2 not available",
    ),
]

# ── Module-level results accumulator ──────────────────────────────────────────

_RESULTS: Dict[str, Any] = {}


@pytest.fixture(scope="module", autouse=True)
def _persist_results():
    """Write accumulated benchmark results to JSON at module teardown."""
    yield
    if not _RESULTS:
        return
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(_RESULTS, fh, indent=2)
    print(f"\n[cf-benchmark] Results written to {BENCHMARK_RESULTS_PATH}", flush=True)


# ── PDB + native helpers (BioPython, headless) ──────────────────────────────────

def _fetch_pdb(pdb_id: str) -> str:
    """Return path to a deposited PDB, downloading from RCSB if not cached."""
    pdb_id = pdb_id.upper()
    dest = CACHE_DIR / f"{pdb_id}.pdb"
    if not dest.is_file():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"[cf-benchmark] Downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, str(dest))
    return str(dest)


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",  # selenomethionine → M
}


def _chain_ca(pdb_path: str, chain: str) -> Tuple[str, "Any", List[int]]:
    """
    Extract a chain's RESOLVED single-chain sequence + ordered Cα coordinates.

    Only standard-AA residues that have a Cα atom are kept (missing residues are
    skipped → we compare over resolved residues only). Returns
    (sequence_str, ndarray[N,3] of Cα coords in residue order, [resseq,...]).
    """
    import numpy as np
    from Bio.PDB import PDBParser  # type: ignore
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", pdb_path)
    model = next(iter(structure))
    if chain not in [c.id for c in model]:
        return "", np.zeros((0, 3)), []
    ch = model[chain]
    seq, coords, resnos = [], [], []
    for res in ch:
        if res.id[0] != " ":            # skip hetero/water
            continue
        aa = _THREE_TO_ONE.get(res.resname.strip().upper())
        if aa is None or "CA" not in res:
            continue
        seq.append(aa)
        coords.append(res["CA"].get_coord())
        resnos.append(res.id[1])
    return "".join(seq), np.array(coords, dtype=float), resnos


def _predicted_ca(pdb_path: str, chain: str = "A") -> "Any":
    """Ordered Cα coordinates of *chain* in a predicted PDB (ndarray[N,3])."""
    import numpy as np
    from Bio.PDB import PDBParser  # type: ignore
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("p", pdb_path)
    model = next(iter(structure))
    ids = [c.id for c in model]
    ch = model[chain] if chain in ids else model[ids[0]]
    coords = [res["CA"].get_coord() for res in ch
              if res.id[0] == " " and "CA" in res]
    return np.array(coords, dtype=float)


def _all_pairs_ca_rmsd(native_ca: "Any", pred_ca: "Any") -> Optional[float]:
    """
    All-pairs Cα RMSD (Å) of predicted vs native, over matched residues (by
    order; both are the folded resolved sequence). Superimposes with BioPython's
    SVDSuperimposer (the same all-pairs quantity matchmaker reports, headless).
    Returns None if fewer than 3 matched Cα.
    """
    import numpy as np
    n = min(len(native_ca), len(pred_ca))
    if n < 3:
        return None
    ref = np.asarray(native_ca[:n], dtype=float)
    mov = np.asarray(pred_ca[:n], dtype=float)
    from Bio.SVDSuperimposer import SVDSuperimposer  # type: ignore
    sup = SVDSuperimposer()
    sup.set(ref, mov)
    sup.run()
    return round(float(sup.get_rms()), 3)


# ── Recording + aggregate stats (pure — unit-tested in CI) ──────────────────────

def _make_record(
    pdb_id: str, chain: str, length: int,
    mean_plddt: Optional[float], ptm: Optional[float],
    rmsd: Optional[float], wall_s: Optional[float], gpu_used: Optional[bool],
) -> Dict[str, Any]:
    """One panel result row (pure)."""
    return {
        "pdb": pdb_id, "chain": chain, "length": length,
        "mean_plddt": mean_plddt, "ptm": ptm,
        "all_pairs_ca_rmsd": rmsd, "wall_s": wall_s, "gpu_used": gpu_used,
    }


def _record(key: str, **kw) -> None:
    _RESULTS[key] = _make_record(**kw)


def _panel_medians(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate medians + GPU/CPU counts over recorded entries (pure)."""
    rmsds  = [e["all_pairs_ca_rmsd"] for e in entries
              if isinstance(e.get("all_pairs_ca_rmsd"), (int, float))]
    plddts = [e["mean_plddt"] for e in entries
              if isinstance(e.get("mean_plddt"), (int, float))]
    n_cpu = sum(1 for e in entries if e.get("gpu_used") is False)
    n_gpu = sum(1 for e in entries if e.get("gpu_used") is True)
    return {
        "n":             len(entries),
        "n_rmsd":        len(rmsds),
        "median_rmsd":   round(statistics.median(rmsds), 3) if rmsds else None,
        "median_plddt":  round(statistics.median(plddts), 2) if plddts else None,
        "n_cpu":         n_cpu,
        "n_gpu":         n_gpu,
    }


def compute_panel_stats(
    results_path: Optional[Path] = None,
    min_entries:  int = MIN_BENCHMARK_ENTRIES,
) -> Dict[str, Any]:
    """
    Read the results JSON, compute panel medians, print a summary table. SKIPS
    (never fails) below *min_entries* valid results — a partial live run must not
    gate red.
    """
    path = results_path or BENCHMARK_RESULTS_PATH
    if not Path(path).is_file():
        pytest.skip(f"ColabFold benchmark results file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    entries = [v for v in data.values()
               if isinstance(v.get("all_pairs_ca_rmsd"), (int, float))
               and isinstance(v.get("mean_plddt"), (int, float))]
    if len(entries) < min_entries:
        pytest.skip(f"Only {len(entries)} valid ColabFold benchmark entries "
                    f"(need {min_entries}). Fold more of the panel first.")
    stats = _panel_medians(entries)
    print("\n" + "=" * 64)
    print(f"{'ColabFold accuracy benchmark (regression guard)':^64}")
    print("=" * 64)
    print(f"  {'PDB':<6} {'len':>4} {'pLDDT':>7} {'pTM':>6} {'RMSD(A)':>8} {'GPU':>5}")
    print("  " + "-" * 50)
    for e in sorted(entries, key=lambda x: x["pdb"]):
        gpu = "[PASS]" if e.get("gpu_used") else ("[CPU!]" if e.get("gpu_used") is False else "?")
        print(f"  {e['pdb']:<6} {e['length']:>4} {e['mean_plddt']:>7.1f} "
              f"{(e.get('ptm') or 0):>6.2f} {e['all_pairs_ca_rmsd']:>8.2f} {gpu:>5}")
    print("  " + "-" * 50)
    print(f"  n={stats['n']}  median RMSD={stats['median_rmsd']} A  "
          f"median pLDDT={stats['median_plddt']}  GPU={stats['n_gpu']} CPU={stats['n_cpu']}")
    print("=" * 64 + "\n")
    return stats


# ── Live panel: fold + measure + record (one test per protein) ──────────────────

def _fold_measure_record(pdb_id: str, chain: str) -> Dict[str, Any]:
    """Fold the native chain sequence, measure RMSD-vs-native, record. Returns
    the fold result dict (for per-protein sanity assertions)."""
    native = _fetch_pdb(pdb_id)
    seq, native_ca, _resnos = _chain_ca(native, chain)
    assert seq, f"could not extract a sequence for {pdb_id} chain {chain}"

    bridge = ColabFoldBridge()
    result = bridge.predict(sequence=seq, copies=1, label=f"bench_{pdb_id}")
    assert result.get("success"), f"{pdb_id} fold failed: {result.get('error')}"

    pred_ca = _predicted_ca(result["ranked_pdb"], "A")
    rmsd = _all_pairs_ca_rmsd(native_ca, pred_ca)
    _record(
        pdb_id, pdb_id=pdb_id, chain=chain, length=len(seq),
        mean_plddt=result.get("mean_plddt"), ptm=result.get("ptm"),
        rmsd=rmsd, wall_s=result.get("elapsed_s"), gpu_used=result.get("gpu_used"),
    )
    print(f"[cf-benchmark] {pdb_id}/{chain}: len={len(seq)} "
          f"pLDDT={result.get('mean_plddt')} RMSD={rmsd} A "
          f"gpu={result.get('gpu_used')} cached={result.get('cached')}", flush=True)
    # Per-protein = pipeline sanity only (NOT an accuracy gate): fold produced a
    # structure and RMSD was computable over matched Cα.
    assert rmsd is not None, f"{pdb_id}: RMSD-vs-native could not be computed"
    return result


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
@pytest.mark.parametrize("pdb_id, chain", PANEL)
def test_panel_fold_and_record(pdb_id, chain):
    _fold_measure_record(pdb_id, chain)


# ── Aggregate gate (the real gate) ──────────────────────────────────────────────

@pytest.mark.benchmark
def test_colabfold_accuracy_acceptable():
    """
    THE benchmark gate. Per-protein tests only fold + record; this aggregate test
    is the pass/fail, on conservative regression-guard thresholds:
      median predicted-vs-native all-pairs Cα RMSD < 3.0 Å
      median mean-pLDDT                            > 70
      no fold explicitly ran on CPU (silent CPU fallback = pipeline failure)
    SKIPS cleanly below MIN_BENCHMARK_ENTRIES live results.
    """
    stats = compute_panel_stats()
    assert stats["median_rmsd"] is not None and stats["median_rmsd"] < MEDIAN_RMSD_MAX_A, (
        f"median predicted-vs-native Cα RMSD {stats['median_rmsd']} Å exceeds "
        f"{MEDIAN_RMSD_MAX_A} Å — likely a pipeline regression (wrong chain, "
        "parse bug), not modelling error on these easy monomers."
    )
    assert stats["median_plddt"] is not None and stats["median_plddt"] > MEDIAN_PLDDT_MIN, (
        f"median mean-pLDDT {stats['median_plddt']} below {MEDIAN_PLDDT_MIN} — "
        "AF2 should be confident on these; check the MSA / model path."
    )
    assert stats["n_cpu"] == 0, (
        f"{stats['n_cpu']} fold(s) ran on CPU — silent GPU→CPU fallback is a real "
        "failure mode (folds would be far too slow). Check the CUDA/JAX env."
    )
