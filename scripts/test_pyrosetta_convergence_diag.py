#!/usr/bin/env python3
"""
scripts/test_pyrosetta_convergence_diag.py
-------------------------------------------
Bounded diagnostic: is the large-cavity ddG BIAS (L99A predicted ~+12 median
vs experimental ~+5) fixable by better relaxation convergence, or is it baked
into the single-structure FastRelax protocol?

Context: the aggregation diagnostic already showed per-trajectory NOISE is
large (~8 kcal/mol spread) and that MEDIAN handles the noise. This test
isolates the separate BIAS question: as relax convergence increases, does the
*median* over-prediction shrink toward experiment, or stay put?

Method (T4 lysozyme 2LZM), 3 mutations spanning the bias question:
  L99A  — large cavity,  exp +5.0  (badly over-predicted, ~+12 median)
  V87M  — cavity-fill,   exp +1.5  (moderately over-predicted)
  N116D — surface neutral, exp +0.1 (was pure noise; should converge to ~0)

For each mutation, score at THREE convergence levels, taking the MEDIAN of 5
trajectories at each level (noise controlled; we watch how the median moves):
  Level A: relax_cycles=3  (3+3 symmetric — current production protocol)
  Level B: relax_cycles=5  (5+5)
  Level C: relax_cycles=8  (8+8 — more thorough convergence)

The relax cycle count is passed through RosettaBridge._run_rosetta_local's
`relax_cycles` parameter (default 3 = unchanged production behaviour); only
the per-mutation mutant relax + symmetric WT re-relax are affected, not the
cached WT baseline or the production scan path.

Usage:
    python scripts/test_pyrosetta_convergence_diag.py

Runtime: 3 mutations x 3 levels x 5 trajectories = 45 scores, slower at higher
cycle counts. Expect ~2-2.5 hours. Run as a deliberate background job.
Requires ROSETTA_BACKEND=local (PyRosetta in WSL2).
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

# ── Project path setup ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402

config.load_env_file()
os.environ["ROSETTA_BACKEND"] = "local"

from rosetta_bridge import RosettaBridge  # noqa: E402

_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

N_TRAJECTORIES = 5

# (level_label, relax_cycles)
LEVELS = [
    ("A: 3+3 (current)", 3),
    ("B: 5+5",           5),
    ("C: 8+8",           8),
]

# (position, from_aa, to_aa, exp_ddg, label)
CONTROLS = [
    (99,  "L", "A", 5.0, "large cavity"),
    (87,  "V", "M", 1.5, "cavity-fill"),
    (116, "N", "D", 0.1, "surface neutral"),
]


def _p(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _rule(title: str = "") -> None:
    _p("=" * 92)
    if title:
        _p(title)
        _p("=" * 92)


def _ensure_2lzm() -> Path:
    cache_dir = _ROOT / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdb = cache_dir / "2LZM.pdb"
    if pdb.is_file() and pdb.stat().st_size > 1000:
        return pdb
    _p("cache/2LZM.pdb missing — downloading from RCSB...")
    import requests
    resp = requests.get("https://files.rcsb.org/download/2LZM.pdb", timeout=30)
    resp.raise_for_status()
    pdb.write_bytes(resp.content)
    return pdb


def _parse_chain_a(pdb: Path) -> dict[int, str]:
    seq: dict[int, str] = {}
    for ln in pdb.read_text(errors="replace").splitlines():
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA" and ln[21:22] == "A":
            try:
                seq[int(ln[22:26])] = _3TO1.get(ln[17:20].strip(), "X")
            except ValueError:
                pass
    return seq


def _est_seconds(cycles: int) -> float:
    """Rough per-trajectory wall-clock model: ~35 s fixed + ~30 s per cycle."""
    return 35.0 + 30.0 * cycles


def main() -> int:
    _rule("PyRosetta convergence-vs-bias diagnostic — 2LZM (3 levels x 3 mutations)")

    # Runtime estimate up front.
    total_s = sum(_est_seconds(c) for _, c in LEVELS) * N_TRAJECTORIES * len(CONTROLS)
    _p(f"Plan: {len(CONTROLS)} mutations x {len(LEVELS)} levels x "
       f"{N_TRAJECTORIES} trajectories = "
       f"{len(CONTROLS) * len(LEVELS) * N_TRAJECTORIES} scores.")
    _p(f"Estimated runtime: ~{total_s / 3600:.1f} hours "
       f"(rough; higher cycle counts are slower). Background job recommended.")

    pdb_path = _ensure_2lzm()
    seq = _parse_chain_a(pdb_path)
    _p(f"PDB: {pdb_path}; chain A {min(seq)}..{max(seq)}")

    bridge = RosettaBridge()
    _p(f"Backend: {bridge._backend!r} — {bridge.backend_status()}")
    if bridge._backend != "local":
        _p("WARNING: backend is not 'local'; this is not a valid PyRosetta diagnostic.")

    # Verify wild-type identities.
    panel = []
    for pos, claimed, to_aa, exp, label in CONTROLS:
        actual = seq.get(pos, "?")
        from_aa = actual if (actual != claimed and actual in _3TO1.values()) else claimed
        key = f"{from_aa}{pos}{to_aa}"
        flag = "OK" if actual == claimed else f"corrected->{actual}"
        _p(f"  {key:<7} pos {pos:<4} WT={actual}  exp={exp:+.1f}  {label}  {flag}")
        panel.append({"key": key, "chain": "A", "position": pos,
                      "from_aa": from_aa, "to_aa": to_aa, "exp": exp, "label": label})

    def _quiet(msg: str) -> None:
        if any(t in msg for t in ("ddG =", "FATAL", "failed", "falling back")):
            _p(f"          {msg.strip()}")

    # results[key][level_label] = {"median":, "spread":, "vals":, "srcs":}
    results: dict[str, dict[str, dict]] = {}
    t_start = time.perf_counter()

    _rule("SCORING")
    for m in panel:
        results[m["key"]] = {}
        mut = {"chain": m["chain"], "position": m["position"],
               "from_aa": m["from_aa"], "to_aa": m["to_aa"]}
        _p(f"\n  === {m['key']} ({m['label']}, exp={m['exp']:+.2f}) ===")
        for level_label, cycles in LEVELS:
            vals: list[float] = []
            srcs: list[str] = []
            _p(f"    Level {level_label}  (relax_cycles={cycles}):")
            for t in range(1, N_TRAJECTORIES + 1):
                t0 = time.perf_counter()
                result = bridge._run_rosetta_local(
                    pdb_path          = str(pdb_path),
                    mutations         = [mut],
                    model_id          = "1",
                    chain             = "A",
                    progress_callback = _quiet,
                    relax_cycles      = cycles,
                )
                dt = time.perf_counter() - t0
                data = result.data or {}
                ddg = (data.get("ddg_scores", {}) or {}).get(m["key"])
                src = (data.get("ddg_source", {}) or {}).get(m["key"], "none")
                srcs.append(src)
                if ddg is not None:
                    vals.append(ddg)
                ddg_s = f"{ddg:+.3f}" if ddg is not None else "None"
                _p(f"      traj {t}/{N_TRAJECTORIES}: ddG={ddg_s} [{src}] ({dt:.0f}s)")
            entry = {"vals": vals, "srcs": srcs}
            if vals:
                entry["median"] = statistics.median(vals)
                entry["spread"] = max(vals) - min(vals)
            results[m["key"]][level_label] = entry

    _p(f"\n  Total elapsed: {(time.perf_counter() - t_start) / 60:.1f} min")

    # ── Per-mutation tables ───────────────────────────────────────────────────
    _rule("PER-MUTATION CONVERGENCE TABLES")
    for m in panel:
        exp = m["exp"]
        _p(f"\n  {m['key']} ({m['label']}, exp={exp:+.2f}):")
        _p(f"    {'Level':<18} {'cycles':>6} {'median ddG':>11} {'spread':>8} "
           f"{'exp':>6} {'err vs exp':>11}")
        _p("    " + "-" * 64)
        for (level_label, cycles) in LEVELS:
            e = results[m["key"]][level_label]
            if "median" not in e:
                _p(f"    {level_label:<18} {cycles:>6}   (no successful trajectories)")
                continue
            err = e["median"] - exp
            _p(f"    {level_label:<18} {cycles:>6} {e['median']:>+11.3f} "
               f"{e['spread']:>8.3f} {exp:>+6.2f} {err:>+11.3f}")

    # ── Analysis ──────────────────────────────────────────────────────────────
    _rule("ANALYSIS")

    def _medians(key):
        return [results[key][ll].get("median") for (ll, _) in LEVELS]

    def _spreads(key):
        return [results[key][ll].get("spread") for (ll, _) in LEVELS]

    # 1. L99A bias trend
    l99 = next(m for m in panel if m["key"].endswith("99A") or m["position"] == 99)
    med = _medians(l99["key"])
    fixable = None
    if all(x is not None for x in med):
        a, b, c = med
        exp = l99["exp"]
        moved_down = a - c                       # +ve if median dropped A->C
        closer = abs(c - exp) < abs(a - exp)
        _p(f"1) L99A (large cavity, exp {exp:+.2f}) median by level: "
           f"A={a:+.2f} -> B={b:+.2f} -> C={c:+.2f}")
        _p(f"   movement A->C = {moved_down:+.2f} kcal/mol "
           f"({'toward' if closer else 'not toward'} experiment)")
        if moved_down >= 1.5 and closer and (c < a):
            fixable = True
            _p("   => BIAS IS CONVERGENCE-RELATED (median moves toward experiment "
               "as cycles increase).")
        elif abs(c - exp) > 3.0:
            fixable = False
            _p("   => BIAS IS BAKED IN (median stays well above experiment "
               "regardless of convergence).")
        else:
            fixable = "partial"
            _p("   => PARTIAL / INCONCLUSIVE (some movement, but small or noisy).")
    else:
        _p("1) L99A produced incomplete data — cannot assess bias trend.")

    # 2. N116D convergence toward zero
    n116 = next((m for m in panel if m["position"] == 116), None)
    if n116:
        med_n = _medians(n116["key"])
        if all(x is not None for x in med_n):
            a, b, c = med_n
            converged = abs(c) < abs(a) and abs(c) <= 1.0
            _p(f"\n2) N116D (surface, exp +0.10) median by level: "
               f"A={a:+.2f} -> B={b:+.2f} -> C={c:+.2f}")
            _p(f"   => {'CONVERGES toward ~0' if converged else 'does NOT cleanly converge'} "
               "as cycles increase "
               f"(|C median|={abs(c):.2f}).")
        else:
            _p("\n2) N116D produced incomplete data.")

    # 3. Spread shrink with cycles
    _p("\n3) Trajectory spread vs cycles (averaged over mutations):")
    for (ll, cyc) in LEVELS:
        sp = [results[m["key"]][ll].get("spread") for m in panel
              if results[m["key"]][ll].get("spread") is not None]
        if sp:
            _p(f"   {ll:<18} mean spread = {statistics.mean(sp):.3f} kcal/mol")
    spreads_by_level = []
    for (ll, _) in LEVELS:
        sp = [results[m["key"]][ll].get("spread") for m in panel
              if results[m["key"]][ll].get("spread") is not None]
        spreads_by_level.append(statistics.mean(sp) if sp else None)
    spread_shrinks = (
        all(x is not None for x in spreads_by_level)
        and spreads_by_level[-1] < spreads_by_level[0]
    )
    if all(x is not None for x in spreads_by_level):
        _p(f"   => spread {'SHRINKS' if spread_shrinks else 'does NOT shrink'} "
           f"A->C ({spreads_by_level[0]:.2f} -> {spreads_by_level[-1]:.2f}).")

    # ── Recommendation ────────────────────────────────────────────────────────
    _rule("RECOMMENDATION")
    if fixable is True:
        _p("Large-cavity magnitude bias is CONVERGENCE-FIXABLE.")
        _p("  More FastRelax cycles pull the median toward experiment.")
        _p("  => For the validation tier, use more relax cycles (e.g. 8+8) WITH")
        _p("     median-of-N aggregation. Expect calibrated absolute ddG to improve;")
        _p("     re-run the 2LZM panel at the chosen cycle count to confirm RMSE.")
    elif fixable is False:
        _p("Large-cavity magnitude bias is BAKED INTO THE PROTOCOL / SCORE FUNCTION.")
        _p("  More cycles do not bring the median down to experiment.")
        _p("  => Do NOT claim calibrated absolute ddG for buried/large-cavity")
        _p("     mutations. Use the protocol for RANKING (sign + relative order,")
        _p("     where it scored r~+0.49), and DISCLOSE absolute magnitudes as")
        _p("     approximate / upper-bounds. Spend effort on median aggregation +")
        _p("     a documented caveat, not on chasing convergence.")
    else:
        _p("INCONCLUSIVE for large-cavity bias from this bounded sweep.")
        _p("  Movement was small or noisy. Safer default: treat absolute magnitudes")
        _p("  as approximate (ranking-grade) and rely on median aggregation;")
        _p("  consider a wider cycle sweep or cartesian_ddg before claiming")
        _p("  calibrated ddG.")

    _p("")
    _p("Note: experimental values are approximate literature figures; read the")
    _p("MEDIAN TREND across levels as the signal, not the absolute error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
