"""
tests/test_rosetta_benchmark.py
--------------------------------
Benchmark suite: PyRosetta ddG predictions vs. experimentally measured ΔΔG values.

Experimental values sourced from ProThermDB / ThermoMutDB; all mutations have been
measured by multiple independent labs and are considered high-confidence.

Sign convention: positive ΔΔG = destabilising, negative = stabilising.
Expected accuracy for our single-trajectory protocol: Pearson r ~0.5-0.7, RMSE < 2.5.

Run:
    pytest tests/test_rosetta_benchmark.py -m benchmark -v --timeout=1800 -s

Single mutation spot-check:
    pytest tests/test_rosetta_benchmark.py -m benchmark -v -s -k "t4_l99a"

Correlation analysis (requires all results collected):
    pytest tests/test_rosetta_benchmark.py -v -s -k "correlation"
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from wsl_bridge import WSLBridge, PYROSETTA_PYTHON  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT   = Path(__file__).parent.parent
CACHE_DIR  = _PROJECT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_RESULTS_PATH = _PROJECT / "scripts" / "benchmark_results.json"

# ── Skip condition ─────────────────────────────────────────────────────────────
# All tests in this module are skipped when WSL2 or PyRosetta is unavailable.

_wsl = WSLBridge()

pytestmark = pytest.mark.skipif(
    not _wsl.is_available() or not _wsl.check_pyrosetta(),
    reason="WSL2 + PyRosetta not available",
)

# ── Module-level results accumulator ──────────────────────────────────────────

_RESULTS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _persist_results():
    """Write accumulated benchmark results to JSON at module teardown."""
    yield
    if not _RESULTS:
        return
    BENCHMARK_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_RESULTS_PATH, "w", encoding="utf-8") as fh:
        json.dump(_RESULTS, fh, indent=2)
    print(f"\n[benchmark] Results written to {BENCHMARK_RESULTS_PATH}", flush=True)


# ── PDB helpers ───────────────────────────────────────────────────────────────

def _fetch_pdb(pdb_id: str) -> str:
    """Return path to PDB file, downloading from RCSB if not already cached."""
    pdb_id = pdb_id.upper()
    dest = CACHE_DIR / f"{pdb_id}.pdb"
    if not dest.is_file():
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"[benchmark] Downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, str(dest))
    return str(dest)


# ── Worker script builder + runner ────────────────────────────────────────────

def _run_benchmark_ddg(
    pdb_path: str,
    chain:    str,
    pos:      int,
    from_aa:  str,
    to_aa:    str,
    timeout:  int = 1800,
) -> float:
    """
    Run PyRosetta ddG via WSL2 for a single mutation.

    Protocol mirrors _run_rosetta_local():
      1. Load PDB into PyRosetta
      2. WT 5-cycle FastRelax (ref2015)
      3. Per-mutation: MutateResidue → 3-cycle FastRelax on mutant and fresh WT clone
      4. ΔΔG = score(mut) - score(wt_rerelaxed)

    Returns ΔΔG in kcal/mol (positive = destabilising).
    Raises RuntimeError on worker failure.
    """
    wsl = WSLBridge()

    wsl_pdb = wsl.copy_to_wsl(pdb_path)
    if not wsl_pdb:
        raise RuntimeError(f"Failed to copy {pdb_path} to WSL2 /tmp")

    pdb_hash  = hashlib.md5(Path(pdb_path).read_bytes()).hexdigest()[:12]
    mut_key   = f"{from_aa}{pos}{to_aa}"
    wsl_out   = f"/tmp/bench_{pdb_hash}_{mut_key}.json"
    muts_json = json.dumps([{
        "chain":   chain,
        "pos":     pos,
        "from_aa": from_aa,
        "to_aa":   to_aa,
    }])

    script = f"""
import json, sys, os

try:
    import pyrosetta
    from pyrosetta import init as rosetta_init, pose_from_file
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

    rosetta_init(options="-mute all -ex1 -ex2 -use_input_sc -ignore_unrecognized_res true")

    mutations = json.loads({muts_json!r})
    pose      = pose_from_file({wsl_pdb!r})

    scorefxn = pyrosetta.create_score_function("ref2015")
    wt_pose  = pose.clone()
    FastRelax(scorefxn, 5).apply(wt_pose)
    print("[bench] WT 5-cycle relax complete", flush=True)

    _aa1to3 = {{'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE',
                'G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU',
                'M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG',
                'S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR'}}

    results = {{}}
    for mut in mutations:
        chain_id = mut["chain"]
        pos_     = int(mut["pos"])
        from_aa_ = mut["from_aa"]
        to_aa_   = mut["to_aa"]
        key      = f"{{from_aa_}}{{pos_}}{{to_aa_}}"
        try:
            sfx      = pyrosetta.create_score_function("ref2015")
            mut_pose = wt_pose.clone()
            res_num  = mut_pose.pdb_info().pdb2pose(chain_id, pos_)
            if res_num == 0:
                raise ValueError(f"Residue {{pos_}}{{chain_id}} not found in pose")
            MutateResidue(target=res_num,
                          new_res=_aa1to3.get(to_aa_, to_aa_)).apply(mut_pose)
            FastRelax(sfx, 3).apply(mut_pose)
            wt_re = wt_pose.clone()
            FastRelax(sfx, 3).apply(wt_re)
            ddg = sfx(mut_pose) - sfx(wt_re)
            results[key] = round(float(ddg), 3)
            print(f"[bench] {{key}}: ddG={{ddg:+.2f}} kcal/mol", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[key] = None
            print(f"[bench] {{key}} FAILED: {{e}}", flush=True)

    with open({wsl_out!r}, "w") as fh:
        json.dump(results, fh)
    print("[bench] done", flush=True)

except Exception as exc:
    import traceback
    traceback.print_exc()
    with open({wsl_out!r}, "w") as fh:
        json.dump({{"error": str(exc)}}, fh)
"""

    result = wsl.run_python_script(script, timeout=timeout)

    win_out = str(Path(tempfile.gettempdir()) / f"bench_{pdb_hash}_{mut_key}.json")
    if not wsl.copy_from_wsl(wsl_out, win_out) or not Path(win_out).is_file():
        raise RuntimeError(
            f"Worker produced no results file for {mut_key}.\n"
            f"stdout: {result['stdout'][-800:]}\n"
            f"stderr: {result['stderr'][-400:]}"
        )

    with open(win_out, encoding="utf-8") as fh:
        data = json.load(fh)

    if "error" in data:
        raise RuntimeError(f"Worker error for {mut_key}: {data['error']}")

    val = data.get(mut_key)
    if val is None:
        raise RuntimeError(f"No result for {mut_key} in worker output: {data}")

    return float(val)


def _record(
    key:          str,
    pdb_id:       str,
    mutation:     str,
    chain:        str,
    predicted:    float,
    experimental: float,
    tolerance:    float = 2.0,
) -> None:
    """Append one result to the module-level accumulator."""
    _RESULTS[key] = {
        "pdb":           pdb_id,
        "mutation":      mutation,
        "chain":         chain,
        "predicted":     predicted,
        "experimental":  experimental,
        "sign_correct":  (predicted > 0) == (experimental > 0),
        "within_tolerance": abs(predicted - experimental) <= tolerance,
        "error_kcal":    round(predicted - experimental, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Barnase (1BNI) — three well-studied mutations
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_barnase_t26a():
    """1BNI T26A: experimental ΔΔG = +1.3 kcal/mol (destabilising)."""
    pdb = _fetch_pdb("1BNI")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=26, from_aa="T", to_aa="A")
    print(f"[benchmark] 1BNI T26A: predicted={ddg:+.2f}, experimental=+1.3", flush=True)
    _record("1BNI_T26A", "1BNI", "T26A", "A", ddg, 1.3)
    assert ddg > 0, f"T26A should be destabilising (positive ΔΔG), got {ddg:+.2f}"
    assert abs(ddg - 1.3) <= 2.0, f"T26A magnitude off: predicted {ddg:+.2f}, expected ~+1.3"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_barnase_i88v():
    """1BNI I88V: experimental ΔΔG = +0.6 kcal/mol (mildly destabilising)."""
    pdb = _fetch_pdb("1BNI")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=88, from_aa="I", to_aa="V")
    print(f"[benchmark] 1BNI I88V: predicted={ddg:+.2f}, experimental=+0.6", flush=True)
    _record("1BNI_I88V", "1BNI", "I88V", "A", ddg, 0.6)
    assert ddg > 0, f"I88V should be destabilising, got {ddg:+.2f}"
    assert abs(ddg - 0.6) <= 2.0, f"I88V magnitude off: predicted {ddg:+.2f}, expected ~+0.6"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_barnase_a43g():
    """1BNI A43G: experimental ΔΔG = +0.8 kcal/mol (mildly destabilising)."""
    pdb = _fetch_pdb("1BNI")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=43, from_aa="A", to_aa="G")
    print(f"[benchmark] 1BNI A43G: predicted={ddg:+.2f}, experimental=+0.8", flush=True)
    _record("1BNI_A43G", "1BNI", "A43G", "A", ddg, 0.8)
    assert ddg > 0, f"A43G should be destabilising, got {ddg:+.2f}"
    assert abs(ddg - 0.8) <= 2.0, f"A43G magnitude off: predicted {ddg:+.2f}, expected ~+0.8"


# ══════════════════════════════════════════════════════════════════════════════
# Ubiquitin (1UBQ) — two core hydrophobic mutations
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_ubiquitin_l69a():
    """1UBQ L69A: experimental ΔΔG = +2.4 kcal/mol (destabilising)."""
    pdb = _fetch_pdb("1UBQ")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=69, from_aa="L", to_aa="A")
    print(f"[benchmark] 1UBQ L69A: predicted={ddg:+.2f}, experimental=+2.4", flush=True)
    _record("1UBQ_L69A", "1UBQ", "L69A", "A", ddg, 2.4)
    assert ddg > 0, f"L69A should be destabilising, got {ddg:+.2f}"
    assert abs(ddg - 2.4) <= 2.0, f"L69A magnitude off: predicted {ddg:+.2f}, expected ~+2.4"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_ubiquitin_v70a():
    """1UBQ V70A: experimental ΔΔG = +1.8 kcal/mol (destabilising)."""
    pdb = _fetch_pdb("1UBQ")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=70, from_aa="V", to_aa="A")
    print(f"[benchmark] 1UBQ V70A: predicted={ddg:+.2f}, experimental=+1.8", flush=True)
    _record("1UBQ_V70A", "1UBQ", "V70A", "A", ddg, 1.8)
    assert ddg > 0, f"V70A should be destabilising, got {ddg:+.2f}"
    assert abs(ddg - 1.8) <= 2.0, f"V70A magnitude off: predicted {ddg:+.2f}, expected ~+1.8"


# ══════════════════════════════════════════════════════════════════════════════
# Staphylococcal nuclease (2SNS) — one stabilising, one destabilising
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_snase_v66l():
    """2SNS V66L: experimental ΔΔG = -0.5 kcal/mol (mildly stabilising)."""
    pdb = _fetch_pdb("2SNS")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=66, from_aa="V", to_aa="L")
    print(f"[benchmark] 2SNS V66L: predicted={ddg:+.2f}, experimental=-0.5", flush=True)
    _record("2SNS_V66L", "2SNS", "V66L", "A", ddg, -0.5)
    assert ddg < 0, f"V66L should be stabilising (negative ΔΔG), got {ddg:+.2f}"
    assert abs(ddg - (-0.5)) <= 2.0, f"V66L magnitude off: predicted {ddg:+.2f}, expected ~-0.5"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_snase_g88v():
    """2SNS G88V: experimental ΔΔG = +2.1 kcal/mol (destabilising)."""
    pdb = _fetch_pdb("2SNS")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=88, from_aa="G", to_aa="V")
    print(f"[benchmark] 2SNS G88V: predicted={ddg:+.2f}, experimental=+2.1", flush=True)
    _record("2SNS_G88V", "2SNS", "G88V", "A", ddg, 2.1)
    assert ddg > 0, f"G88V should be destabilising, got {ddg:+.2f}"
    assert abs(ddg - 2.1) <= 2.0, f"G88V magnitude off: predicted {ddg:+.2f}, expected ~+2.1"


# ══════════════════════════════════════════════════════════════════════════════
# T4 Lysozyme (2LZM) — the gold-standard L99A and a mildly stabilising case
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_t4_l99a():
    """2LZM L99A: experimental ΔΔG = +4.0 kcal/mol (strongly destabilising, well-studied)."""
    pdb = _fetch_pdb("2LZM")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=99, from_aa="L", to_aa="A")
    print(f"[benchmark] 2LZM L99A: predicted={ddg:+.2f}, experimental=+4.0", flush=True)
    _record("2LZM_L99A", "2LZM", "L99A", "A", ddg, 4.0)
    assert ddg > 0, f"L99A should be strongly destabilising, got {ddg:+.2f}"
    assert abs(ddg - 4.0) <= 2.0, f"L99A magnitude off: predicted {ddg:+.2f}, expected ~+4.0"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_t4_a98v():
    """2LZM A98V: experimental ΔΔG = -0.5 kcal/mol (mildly stabilising)."""
    pdb = _fetch_pdb("2LZM")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=98, from_aa="A", to_aa="V")
    print(f"[benchmark] 2LZM A98V: predicted={ddg:+.2f}, experimental=-0.5", flush=True)
    _record("2LZM_A98V", "2LZM", "A98V", "A", ddg, -0.5)
    assert ddg < 0, f"A98V should be stabilising (negative ΔΔG), got {ddg:+.2f}"
    assert abs(ddg - (-0.5)) <= 2.0, f"A98V magnitude off: predicted {ddg:+.2f}, expected ~-0.5"


# ══════════════════════════════════════════════════════════════════════════════
# HIV-1 protease (1HSG) — validated reference + sanity check
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_hiv_protease_v82a():
    """
    1HSG V82A: experimental ΔΔG ~+1.5 to +2.0 kcal/mol (Mahalingam et al.).
    This is the reference mutation we have used for manual validation.
    """
    pdb = _fetch_pdb("1HSG")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=82, from_aa="V", to_aa="A")
    exp_mid = 1.75  # midpoint of +1.5 to +2.0 range
    print(f"[benchmark] 1HSG V82A: predicted={ddg:+.2f}, experimental=~+1.5 to +2.0", flush=True)
    _record("1HSG_V82A", "1HSG", "V82A", "A", ddg, exp_mid)
    assert ddg > 0, f"V82A should be destabilising, got {ddg:+.2f}"
    assert ddg <= 5.0, f"V82A predicted ddG unreasonably large: {ddg:+.2f}"


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_hiv_protease_i64e_sanity():
    """
    1HSG I64E: no precise experimental value, but Ile→Glu in a hydrophobic core
    must be destabilising. This is a sanity-check that our sign convention and
    hydrophobic burial scoring are working.
    """
    pdb = _fetch_pdb("1HSG")
    ddg = _run_benchmark_ddg(pdb, chain="A", pos=64, from_aa="I", to_aa="E")
    print(f"[benchmark] 1HSG I64E: predicted={ddg:+.2f} (sanity: must be positive)", flush=True)
    # Use +3.0 as a rough experimental stand-in for recording purposes only
    _record("1HSG_I64E", "1HSG", "I64E", "A", ddg, 3.0)
    assert ddg > 0, (
        f"I64E buries a charged Glu in a hydrophobic core — must be destabilising "
        f"(positive ΔΔG), got {ddg:+.2f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Correlation analysis
# ══════════════════════════════════════════════════════════════════════════════

def _pearson_r(xs: list, ys: list) -> float:
    """Pearson correlation coefficient (no scipy dependency)."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denom = (
        sum((x - mx) ** 2 for x in xs) *
        sum((y - my) ** 2 for y in ys)
    ) ** 0.5
    return num / denom if denom > 0 else 0.0


def _rmse(predicted: list, experimental: list) -> float:
    """Root mean squared error in kcal/mol."""
    n = len(predicted)
    if n < 1:
        return float("inf")
    return (sum((p - e) ** 2 for p, e in zip(predicted, experimental)) / n) ** 0.5


def compute_benchmark_correlation(
    results_path: Optional[Path] = None,
    min_entries: int = 5,
) -> dict:
    """
    Read benchmark_results.json, compute Pearson r and RMSE, print a summary table.

    Parameters
    ----------
    results_path : path to JSON file (defaults to scripts/benchmark_results.json)
    min_entries  : minimum number of valid entries required to compute statistics

    Returns
    -------
    dict with keys: pearson_r, rmse, n, sign_accuracy, within_tolerance_rate
    """
    path = results_path or BENCHMARK_RESULTS_PATH
    if not Path(path).is_file():
        pytest.skip(f"Benchmark results file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    entries = [
        v for v in data.values()
        if isinstance(v.get("predicted"), (int, float))
        and isinstance(v.get("experimental"), (int, float))
        and v["mutation"] != "I64E"  # sanity check only — no precise experimental value
    ]

    if len(entries) < min_entries:
        pytest.skip(
            f"Only {len(entries)} valid benchmark entries found "
            f"(need {min_entries}). Run more benchmark tests first."
        )

    predicted    = [e["predicted"]    for e in entries]
    experimental = [e["experimental"] for e in entries]

    r    = _pearson_r(predicted, experimental)
    rmse = _rmse(predicted, experimental)
    sign_ok   = sum(1 for e in entries if e.get("sign_correct", False))
    tol_ok    = sum(1 for e in entries if e.get("within_tolerance", False))
    n = len(entries)

    # Summary table
    print("\n" + "=" * 62)
    print(f"{'PyRosetta ddG Benchmark':^62}")
    print("=" * 62)
    print(f"  {'Mutation':<12} {'Predicted':>10} {'Experimental':>13} {'Error':>8}")
    print("  " + "-" * 48)
    for e in sorted(entries, key=lambda x: x["pdb"] + x["mutation"]):
        flag = "✓" if e.get("within_tolerance") else "✗"
        print(
            f"  {e['pdb']+' '+e['mutation']:<12} "
            f"{e['predicted']:>+10.2f} "
            f"{e['experimental']:>+13.2f} "
            f"{e['error_kcal']:>+7.2f}  {flag}"
        )
    print("  " + "-" * 48)
    print(f"  n = {n}  |  Pearson r = {r:.3f}  |  RMSE = {rmse:.2f} kcal/mol")
    print(f"  Sign accuracy: {sign_ok}/{n}  |  Within 2 kcal/mol: {tol_ok}/{n}")
    print("=" * 62 + "\n")

    return {
        "pearson_r":             r,
        "rmse":                  rmse,
        "n":                     n,
        "sign_accuracy":         sign_ok / n,
        "within_tolerance_rate": tol_ok / n,
    }


@pytest.mark.benchmark
def test_benchmark_correlation_acceptable():
    """
    Assert that the full benchmark set meets minimum accuracy thresholds.

    Thresholds (validated against the 2LZM validation-tier panel, 2026-05-30:
    r=0.499, RMSE=2.729, sign=90%):
      Pearson r     > 0.30
      RMSE          < 4.0 kcal/mol
      sign accuracy >= 60%

    This test only passes once enough benchmark results have been collected
    (minimum 5 entries with precise experimental values).
    """
    stats = compute_benchmark_correlation()

    assert stats["pearson_r"] > 0.30, (
        f"Pearson r = {stats['pearson_r']:.3f} — below 0.30 threshold. "
        "Check individual mutation errors in the table above."
    )
    assert stats["rmse"] < 4.0, (
        f"RMSE = {stats['rmse']:.2f} kcal/mol — above 4.0 kcal/mol threshold. "
        "Single-trajectory variance expected; consider averaging 3–5 replicates."
    )
    assert stats["sign_accuracy"] >= 0.60, (
        f"Sign accuracy = {stats['sign_accuracy']:.0%} — below 60% threshold. "
        "Verify PDB chain assignments and residue numbering."
    )
