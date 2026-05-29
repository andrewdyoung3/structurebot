#!/usr/bin/env python3
"""
scripts/test_pyrosetta_aggregation_diag.py
-------------------------------------------
Diagnostic: which aggregation strategy fixes the systematic ddG OVER-estimation
seen in the T4 lysozyme validation?

The 2LZM run showed predicted ddG systematically too high (surface controls
T152S +3.2 vs exp +0.5, N116D +5.0 vs +0.1; outlier L133A +15.1 vs +2.7).
Before building a multi-trajectory ddG, we need to know whether that bias is
(a) bad-minimum trajectory noise — fixed by aggregating with MIN / mean-of-
lowest-N, or (b) a systematic baseline/protocol issue — in which case multi-
trajectory won't help and the relax/scorefunction/WT baseline needs work.

Method: take 3 mutations spanning the problem and run 5 INDEPENDENT
relax+score trajectories each via the existing RosettaBridge._run_rosetta_local
entry point. Each trajectory is a fresh WSL PyRosetta process, so Rosetta's
default (non-constant) seeding produces independent FastRelax trajectories —
no seed injection into the bridge is needed (this script is read-only).

For each mutation it reports the 5 trajectory ddG values, their MEAN / MEDIAN
/ MIN, and the experimental value; then recommends the best aggregator.

  L99A  — large cavity     (exp +5.0)
  V87M  — cavity-fill outlier (exp +1.5)
  N116D — surface control   (exp +0.1)

Usage:
    python scripts/test_pyrosetta_aggregation_diag.py

3 mutations x 5 trajectories = 15 scores (~25-35 min; WT relax is cached).
Requires ROSETTA_BACKEND=local (PyRosetta in WSL2). Read-only.
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
os.environ["ROSETTA_BACKEND"] = "local"   # force the PyRosetta/WSL2 path

from rosetta_bridge import RosettaBridge  # noqa: E402

_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

N_TRAJECTORIES = 5

# (position, from_aa, to_aa, exp_ddg, label)
CONTROLS = [
    (99,  "L", "A", 5.0, "large cavity"),
    (87,  "V", "M", 1.5, "cavity-fill outlier"),
    (116, "N", "D", 0.1, "surface control"),
]


def _p(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)


def _rule(title: str = "") -> None:
    _p("=" * 88)
    if title:
        _p(title)
        _p("=" * 88)


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


def main() -> int:
    _rule("PyRosetta ddG aggregation diagnostic — 2LZM (5 trajectories x 3 mutations)")

    pdb_path = _ensure_2lzm()
    seq = _parse_chain_a(pdb_path)
    _p(f"PDB: {pdb_path} ({pdb_path.stat().st_size} bytes); chain A {min(seq)}..{max(seq)}")

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
            _p(f"        {msg.strip()}")

    # ── Run N independent trajectories per mutation ───────────────────────────
    rows = []
    _rule(f"SCORING — {N_TRAJECTORIES} independent trajectories per mutation")
    for m in panel:
        mut = {"chain": m["chain"], "position": m["position"],
               "from_aa": m["from_aa"], "to_aa": m["to_aa"]}
        vals: list[float] = []
        srcs: list[str] = []
        _p(f"\n  {m['key']} ({m['label']}, exp={m['exp']:+.2f}):")
        for t in range(1, N_TRAJECTORIES + 1):
            t0 = time.perf_counter()
            result = bridge._run_rosetta_local(
                pdb_path          = str(pdb_path),
                mutations         = [mut],
                model_id          = "1",
                chain             = "A",
                progress_callback = _quiet,
            )
            dt = time.perf_counter() - t0
            data = result.data or {}
            ddg = (data.get("ddg_scores", {}) or {}).get(m["key"])
            src = (data.get("ddg_source", {}) or {}).get(m["key"], "none")
            srcs.append(src)
            if ddg is not None:
                vals.append(ddg)
            ddg_s = f"{ddg:+.3f}" if ddg is not None else "None"
            _p(f"    traj {t}/{N_TRAJECTORIES}: ddG={ddg_s} [{src}] ({dt:.0f}s)")
        rows.append({**m, "vals": vals, "srcs": srcs})

    # ── Per-mutation table ────────────────────────────────────────────────────
    _rule("PER-MUTATION TRAJECTORY SUMMARY")
    _p(f"  {'Mutation':<7} {'exp':>6} {'mean':>8} {'median':>8} {'min':>8} "
       f"{'spread':>7}   trajectories")
    _p("  " + "-" * 84)
    for r in rows:
        v = r["vals"]
        if not v:
            _p(f"  {r['key']:<7} {r['exp']:>+6.2f}   (no successful trajectories)")
            continue
        mean_v = statistics.mean(v)
        med_v = statistics.median(v)
        min_v = min(v)
        spread = max(v) - min(v)
        traj_str = ", ".join(f"{x:+.2f}" for x in v)
        _p(f"  {r['key']:<7} {r['exp']:>+6.2f} {mean_v:>+8.3f} {med_v:>+8.3f} "
           f"{min_v:>+8.3f} {spread:>7.3f}   [{traj_str}]")
        r["mean"], r["median"], r["min"], r["spread"] = mean_v, med_v, min_v, spread

    scored = [r for r in rows if r.get("vals")]
    if len(scored) < len(CONTROLS):
        _p("\nWARNING: some mutations produced no PyRosetta value; partial diagnostic.")
    if any(s != "pyrosetta" for r in scored for s in r["srcs"]):
        _p("\nWARNING: some trajectories used the EMPIRICAL fallback — not pure PyRosetta.")
    if not scored:
        _rule("RECOMMENDATION")
        _p("No trajectories scored — cannot diagnose. Check the WSL2/PyRosetta path.")
        return 0

    # ── Aggregator comparison vs experiment ──────────────────────────────────
    _rule("AGGREGATOR ACCURACY (|aggregate - experimental|, averaged over mutations)")
    aggs = {
        "mean":   lambda v: statistics.mean(v),
        "median": lambda v: statistics.median(v),
        "min":    lambda v: min(v),
    }
    mae = {}
    for name, fn in aggs.items():
        errs = [abs(fn(r["vals"]) - r["exp"]) for r in scored]
        mae[name] = statistics.mean(errs)
    for name in ("mean", "median", "min"):
        _p(f"  {name:<7} MAE = {mae[name]:.3f} kcal/mol   "
           + "   ".join(f"{r['key']}:{aggs[name](r['vals']):+.2f}" for r in scored))

    best = min(mae, key=mae.get)
    mean_spread = statistics.mean(r["spread"] for r in scored)
    _p(f"\n  best aggregator by MAE : {best}  (MAE {mae[best]:.3f})")
    _p(f"  mean trajectory spread : {mean_spread:.3f} kcal/mol "
       "(max-min within a mutation, averaged)")

    # ── Recommendation (decision tree from the task) ──────────────────────────
    _rule("RECOMMENDATION")
    TIGHT = 1.0          # spread below this = trajectories cluster tightly
    MARGIN = 0.5         # MAE improvement needed to call a winner meaningful

    min_much_better = (mae["min"] < mae["median"] - MARGIN
                       and mae["min"] < mae["mean"] - MARGIN)
    median_best = (mae["median"] <= mae["min"] and mae["median"] <= mae["mean"])

    if mean_spread < TIGHT and mae["min"] > 1.5:
        _p("SYSTEMATIC BIAS (not trajectory noise).")
        _p(f"  Trajectories cluster tightly (mean spread {mean_spread:.2f} kcal/mol) "
           "yet all sit well above experiment.")
        _p("  => Multi-trajectory aggregation will NOT fix this. The over-estimation "
           "is a baseline/protocol issue:")
        _p("     - WT relax baseline mismatch (cached WT vs per-mutant re-relax),")
        _p("     - too few FastRelax cycles / not converged,")
        _p("     - scorefunction or cartesian vs torsion-space relax.")
        _p("  Investigate the protocol before investing in multi-trajectory.")
    elif min_much_better:
        _p("BAD-MINIMUM TRAJECTORY NOISE — aggregate by MIN (or mean-of-lowest-N).")
        _p(f"  MIN MAE {mae['min']:.2f} is clearly better than median "
           f"{mae['median']:.2f} / mean {mae['mean']:.2f}.")
        _p("  The over-estimation comes from individual trajectories landing in "
           "strained minima; the lowest-energy trajectory is closest to truth.")
        _p("  => Build multi-trajectory ddG aggregating by MIN, or mean of the "
           "lowest 2-3 of N, rather than median.")
    elif median_best:
        _p("MEDIAN is the best aggregator — the originally-planned approach is right.")
        _p(f"  MEDIAN MAE {mae['median']:.2f} <= mean {mae['mean']:.2f}, "
           f"min {mae['min']:.2f}.")
        _p("  => Build multi-trajectory ddG aggregating by MEDIAN over N trajectories.")
    else:
        _p(f"Best aggregator is '{best}' (MAE {mae[best]:.3f}), but the margin over "
           "the others is small.")
        _p("  => Prefer the simplest robust choice; if min and median are close, "
           "use median for stability, or mean-of-lowest-N to hedge against "
           "occasional bad minima.")

    _p("")
    _p("Note: experimental values are approximate literature figures; treat the "
       "aggregator ranking as the signal, not the absolute MAE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
