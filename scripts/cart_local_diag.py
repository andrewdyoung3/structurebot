"""
scripts/cart_local_diag.py — CANONICAL-LOCAL cartesian_ddg DIAGNOSTIC (the archaeology settle).

SCRATCH/diagnostic only (NOT production, NOT wired). Answers the two crisp questions the 2026-06-28
archaeology left open — on the SAME clean 24-mut subset (2LZM/2WQG/2HBB) the whole-pose multi-cycle
diagnostic used (cart_cycle_diag.py), so it is apples-to-apples vs the recorded whole-pose cyc3/cyc5
+ torsion + exp numbers:
  (1) CONVERGENCE — does canonical-LOCAL converge by ~3 iterations (Δ running-mean < ~1 REU)?
      (the whole-pose arm did not, even at 5.)
  (2) CALIBRATION — does ÷2.94 calibrate canonical-local REU → kcal/mol with a REPRODUCIBLE
      cross-protein offset (regression slope ≈ 2.94, small per-protein offset sd)? (whole-pose over-corrected.)
Plus ranking/sign vs torsion and per-mutation cost.

Uses scripts/rosetta_cartesian_bench.score_cartesian_local (the canonical-LOCAL arm: ~6 Å MoveMap +
ref2015_cart + N paired local opts + coord-constrained one-time baseline). LOSSLESS: writes only the
gitignored cache/stability_datagen/cart_local_diag.jsonl (crash-safe append + resume); never touches
benchmark_rows.jsonl.

Usage:
  python scripts/cart_local_diag.py --run       # compute (long; resumable) → jsonl
  python scripts/cart_local_diag.py --analyze    # read jsonl + benchmark_rows → the report
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(_ROOT / "scripts"))
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_BENCH = _ROOT / "cache" / "stability_datagen" / "benchmark_rows.jsonl"
_MANIFEST = _ROOT / "scripts" / "calibration_manifest.draft.json"
_OUT = _ROOT / "cache" / "stability_datagen" / "cart_local_diag.jsonl"
_PROTEINS = ["2LZM", "2WQG", "2HBB"]
_ITERS = 5
_RADIUS = 6.0
_CAL = 2.94   # Park 2016 canonical cartesian_ddg REU→kcal/mol factor (the calibration under test)

_THREE = {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','E':'GLU','Q':'GLN','G':'GLY','H':'HIS',
          'I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER','T':'THR','W':'TRP',
          'Y':'TYR','V':'VAL'}


def _resolve(p: str) -> Path:
    q = Path(p); return q if q.is_absolute() else (_ROOT / p)


def _residues(pdb: Path) -> Dict[Any, str]:
    out: Dict[Any, str] = {}
    for ln in open(pdb):
        if ln.startswith("ATOM"):
            out[(ln[21], ln[22:27].strip())] = ln[17:20].strip()
    return out


def _benchmark_pdb(set_name: str, pdbid: str) -> Optional[str]:
    mf = json.load(open(_MANIFEST))
    sd = next((e.get("struct_dir") for e in mf["entries"]
               if e.get("set") == set_name and e.get("pdbid") == pdbid), None)
    struct_dir = _resolve(sd) if sd is not None else (
        _ROOT / "RaSP_repo" / "data" / "test" / set_name.split("/")[0] / "structure" / "raw")
    cand = sorted(struct_dir.glob(f"{pdbid}*.pdb"))
    return str(cand[0]) if cand else None


def curated() -> List[Dict[str, Any]]:
    """The 24-mut subset: triple-present (torsion+cart1+exp) rows for the 3 proteins, each resolved to
    the EXACT structure the benchmark scored, wt-verified (never mis-attribute)."""
    rows = [json.loads(l) for l in open(_BENCH)]
    base = [{"key": r["key"], "set": r["set"], "pdbid": r["pdbid"], "chain": r["chain"],
             "resnum": r["resnum"], "wt": r["wt"], "mut": r["mut"], "variant": r["variant"],
             "exp_ddg": r["exp_ddg"], "rosetta_ddg": r["rosetta_ddg"],
             "rosetta_cart1_ddg": r["rosetta_cart_ddg"]}
            for r in rows if r["pdbid"] in _PROTEINS
            and r.get("rosetta_ddg") is not None and r.get("rosetta_cart_ddg") is not None
            and r.get("exp_ddg") is not None]
    pdb_cache: Dict[Any, Optional[str]] = {}
    for m in base:
        sk = (m["set"], m["pdbid"])
        if sk not in pdb_cache:
            pdb_cache[sk] = _benchmark_pdb(*sk)
        m["pdb_path"] = pdb_cache[sk]
    for m in base:
        if not m["pdb_path"] or not Path(m["pdb_path"]).is_file():
            raise SystemExit(f"structure unresolved for {m['set']}/{m['pdbid']}")
        got = _residues(Path(m["pdb_path"])).get((m["chain"], str(m["resnum"])))
        if got != _THREE[m["wt"]]:
            raise SystemExit(f"WT MISMATCH {m['variant']} ({m['pdbid']}): {got} != {_THREE[m['wt']]}")
    return base


def _groups(muts):
    g: Dict[Any, List[Dict[str, Any]]] = {}
    for m in muts:
        g.setdefault((m["pdbid"], m["chain"], m["pdb_path"]), []).append(m)
    return g


def run(log=print) -> None:
    from rosetta_cartesian_bench import score_cartesian_local
    muts = curated()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if _OUT.is_file():
        for ln in open(_OUT):
            try: done.add(json.loads(ln)["key"])
            except Exception: pass
        log(f"resume: {len(done)} rows already in {_OUT.name}")
    fh = open(_OUT, "a", buffering=1)
    for (pid, chain, pdb), group in _groups(muts).items():
        pending = [m for m in group if m["key"] not in done]
        if not pending:
            continue
        log(f"[{pid}/{chain}] {len(pending)} pending of {len(group)} (canonical-local, R={_RADIUS}, N={_ITERS})")
        t0 = time.time()
        out = score_cartesian_local(
            pdb, chain,
            [{"resnum": m["resnum"], "wt": m["wt"], "mut": m["mut"], "variant": m["variant"]} for m in pending],
            radius=_RADIUS, iters=_ITERS, timeout=7200, log=log)
        base_sec, timing = out.get("_base_sec"), out.get("_timing", {})
        for m in pending:
            r = out.get(m["variant"])
            if not isinstance(r, dict) or "ddg_iters" not in r:
                log(f"  {m['variant']}: not_computed (omitted)"); continue
            fh.write(json.dumps({**{k: m[k] for k in ("key","pdbid","chain","resnum","wt","mut",
                     "variant","exp_ddg","rosetta_ddg","rosetta_cart1_ddg")},
                     "ddg_iters": r["ddg_iters"], "n_sel": r.get("n_sel"),
                     "sec_per_iter": timing.get(m["variant"]), "base_sec": base_sec}) + "\n")
        log(f"[{pid}] done in {time.time()-t0:.0f}s (baseline {base_sec}s)")
    fh.close()
    log("RUN COMPLETE")


def _pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return None if vx == 0 or vy == 0 else cov/((vx*vy)**0.5)


def _slope(xs, ys):
    """OLS slope of y on x (y=exp, x=pred-REU): exp ≈ slope·pred. The calibration factor is 1/slope;
    canonical wants pred/2.94≈exp i.e. slope≈1/2.94≈0.34 — equivalently regress pred on exp → ≈2.94."""
    n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
    vx = sum((x-mx)**2 for x in xs)
    return None if vx == 0 else sum((x-mx)*(y-my) for x, y in zip(xs, ys))/vx


def analyze(log=print) -> None:
    if not _OUT.is_file():
        raise SystemExit("no cart_local_diag.jsonl — run --run first")
    rows = [json.loads(l) for l in open(_OUT)]
    log(f"\n=== CANONICAL-LOCAL cartesian_ddg — {len(rows)} muts (2LZM/2WQG/2HBB), R={_RADIUS} N={_ITERS} ===\n")

    # (1) CONVERGENCE — running mean after each iteration; Δ(k→k+1) averaged across muts
    log("[1] CONVERGENCE (running mean of per-iteration ddG; Δ between consecutive running means):")
    niter = min(len(r["ddg_iters"]) for r in rows)
    run_means = []  # per mut: [mean of first k iters for k=1..niter]
    for r in rows:
        it = r["ddg_iters"][:niter]
        run_means.append([sum(it[:k])/k for k in range(1, niter+1)])
    for k in range(niter-1):
        deltas = [abs(rm[k+1]-rm[k]) for rm in run_means]
        log(f"    iter {k+1}→{k+2}: mean |Δ running-mean| = {statistics.mean(deltas):.3f} REU "
            f"(max {max(deltas):.2f}); converged(<1 REU): {sum(d<1.0 for d in deltas)}/{len(deltas)}")
    conv3 = [abs(rm[2]-rm[1]) for rm in run_means] if niter >= 3 else []
    if conv3:
        log(f"    → BY ITERATION 3: {sum(d<1.0 for d in conv3)}/{len(conv3)} muts have Δ(2→3)<1 REU "
            f"(mean {statistics.mean(conv3):.3f})")

    # final canonical-local ddG = mean of all iterations
    for r in rows:
        r["local_reu"] = sum(r["ddg_iters"])/len(r["ddg_iters"])

    # (2) CALIBRATION — slope(pred-REU on exp) should ≈2.94; ÷2.94 offset reproducible cross-protein?
    log("\n[2] CALIBRATION (÷2.94 REU→kcal/mol; slope of local-REU regressed on exp should ≈2.94):")
    exp = [r["exp_ddg"] for r in rows]; loc = [r["local_reu"] for r in rows]
    s = _slope(exp, loc)   # loc ≈ s·exp → s is the empirical REU-per-kcal/mol factor (canonical≈2.94)
    log(f"    empirical REU/(kcal·mol⁻¹) slope = {s:.2f}  (canonical target ≈ 2.94; "
        f"{'consistent' if s and 2.0<=s<=4.0 else 'OFF'})")
    log("    per-protein offset of (local_REU/2.94 − exp) — reproducible (small sd) ⇒ calibratable:")
    offs = {}
    for p in _PROTEINS:
        pr = [r for r in rows if r["pdbid"] == p]
        if not pr: continue
        resid = [r["local_reu"]/_CAL - r["exp_ddg"] for r in pr]
        offs[p] = statistics.mean(resid)
        log(f"      {p}: offset {offs[p]:+.2f} kcal/mol (n={len(pr)}, sd {statistics.pstdev(resid):.2f})")
    if len(offs) >= 2:
        sd = statistics.pstdev(list(offs.values()))
        log(f"    → CROSS-PROTEIN offset sd = {sd:.2f} kcal/mol "
            f"(prior torsion 1.1–1.6 = NOT calibratable; <~0.7 ⇒ a global offset works)")

    # (3) RANKING/SIGN vs torsion + cart-1, on the SAME 24 (apples-to-apples)
    log("\n[3] RANKING / SIGN vs exp (same 24 muts):")
    big = [r for r in rows if abs(r["exp_ddg"]) > 1.0]
    def _signacc(pred_key, conv=lambda v: v):
        ok = [( (conv(r[pred_key])>0) == (r["exp_ddg"]>0) ) for r in big if r.get(pred_key) is not None]
        return (sum(ok)/len(ok), len(ok)) if ok else (None, 0)
    for label, key in (("canonical-local", "local_reu"), ("torsion ref2015", "rosetta_ddg"),
                       ("whole-pose cart-1", "rosetta_cart1_ddg")):
        xs = [r[key] for r in rows if r.get(key) is not None]
        ys = [r["exp_ddg"] for r in rows if r.get(key) is not None]
        r_ = _pearson(xs, ys); sa, nsa = _signacc(key)
        log(f"    {label:18s}: r={r_ if r_ is None else round(r_,3)}  "
            f"sign-acc(|exp|>1)={None if sa is None else round(sa,2)} (n={nsa})  n={len(xs)}")
    # per-protein r (the 2LZM anchor the prior diag highlighted: cart-5 r 0.781 > torsion 0.611)
    log("    per-protein Pearson r (canonical-local | torsion):")
    for p in _PROTEINS:
        pr = [r for r in rows if r["pdbid"] == p]
        rl = _pearson([r["local_reu"] for r in pr], [r["exp_ddg"] for r in pr])
        rt = _pearson([r["rosetta_ddg"] for r in pr], [r["exp_ddg"] for r in pr])
        log(f"      {p} (n={len(pr)}): local {rl if rl is None else round(rl,3)} | "
            f"torsion {rt if rt is None else round(rt,3)}")

    # (4) COST
    log("\n[4] COST:")
    spi = [r["sec_per_iter"] for r in rows if r.get("sec_per_iter")]
    bsec = sorted({r.get("base_sec") for r in rows if r.get("base_sec")})
    if spi:
        # effective per-mutation cost at N iters (paired wt+mut per iter)
        log(f"    sec per iteration (1 WT + 1 MUT local relax): mean {statistics.mean(spi):.1f}s "
            f"(range {min(spi):.1f}–{max(spi):.1f})")
        log(f"    effective per-mutation (N={_ITERS} iters): ~{statistics.mean(spi)*_ITERS:.0f}s "
            f"+ amortized one-time baseline {bsec} s/protein")
        log(f"    vs WHOLE-POSE recorded 73–115 s/mut → canonical-local is "
            f"{'CHEAPER' if statistics.mean(spi)*_ITERS < 73 else 'comparable/higher'}")
    log("")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for m in curated():
            print(f"{m['pdbid']} {m['chain']}{m['resnum']} {m['wt']}->{m['mut']} exp={m['exp_ddg']}")
    if a.run:
        run()
    if a.analyze:
        analyze()
