"""
scripts/rosetta_cartesian_bench.py — BENCHMARK-ONLY cartesian_ddg Rosetta arm.

Purpose: give the overnight calibration benchmark a SECOND Rosetta protocol —
canonical-style cartesian_ddg (ref2015_cart + cartesian-space FastRelax) — to run
ALONGSIDE the deployed torsion-space FastRelax arm, so each can be scored against
experiment. This exists ONLY for the benchmark.

NOT WIRED INTO PRODUCTION. It does not touch rosetta_bridge.py, the deep-tier path,
the §9 handoff, or any default. The production Rosetta protocol is unchanged. The
data-gen harness invokes this only behind the explicit `--rosetta-cart` flag.

Protocol (manual cartesian_ddg — this PyRosetta build does not expose
CartesianddGMover, so the cartesian protocol is reproduced the same way the deployed
code reproduces the torsion one, just in cartesian space):
  ref2015_cart score fn → cartesian FastRelax the WT once → per mutation: clone the
  relaxed WT, MutateResidue, cartesian FastRelax the mutant, ddG = score(mut) −
  score(WT). Asymmetric (single cached WT) to match the harness's deployed arm.

Sign convention matches the system + the deployed arm: positive = destabilising
(ref2015 REU, uncalibrated). Error-first: any per-mutation failure → that variant is
omitted (→ not_computed upstream), never faked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# canonical cartesian_ddg uses a few iterations; keep modest for benchmark throughput
_DEFAULT_CART_CYCLES = 1


def _worker_script(pdb_wsl: str, chain: str, muts: List[Dict[str, Any]],
                   cart_cycles: int) -> str:
    """Build the self-contained PyRosetta cartesian_ddg worker (runs in WSL)."""
    payload = json.dumps(muts)
    return f'''
import json, sys
import pyrosetta
pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false")
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.core.scoring import cart_bonded

_three = {{
 "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN","G":"GLY",
 "H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER",
 "T":"THR","W":"TRP","Y":"TYR","V":"VAL"}}

sf = pyrosetta.create_score_function("ref2015_cart")
muts = json.loads({payload!r})
chain = {chain!r}

def cart_relax(pose):
    fr = FastRelax(sf, {cart_cycles})
    fr.cartesian(True)
    fr.max_iter(200)
    fr.apply(pose)

try:
    wt = pose_from_pdb({pdb_wsl!r})
except Exception as e:
    print("WORKER_FATAL load:"+str(e)[:120]); sys.exit(1)
cart_relax(wt)
wt_score = sf(wt)
pi = wt.pdb_info()

results = {{}}; errors = {{}}
for m in muts:
    var = m["variant"]
    try:
        pr = pi.pdb2pose(chain, int(m["resnum"]))
        if pr == 0:
            errors[var] = "resnum not in pose"; continue
        # verify wildtype identity before mutating (never mis-attribute)
        if wt.residue(pr).name1() != m["wt"]:
            errors[var] = f"wt mismatch pose={{wt.residue(pr).name1()}} exp={{m['wt']}}"; continue
        mp = wt.clone()
        MutateResidue(pr, _three[m["mut"]]).apply(mp)
        cart_relax(mp)
        results[var] = round(sf(mp) - wt_score, 4)   # +=destabilising (ref2015 REU)
    except Exception as e:
        errors[var] = type(e).__name__+":"+str(e)[:80]

print("WORKER_RESULT "+json.dumps({{"results": results, "errors": errors,
      "n_res": wt.total_residue(), "wt_score": round(wt_score,2)}}))
'''


def score_cartesian(pdb_path: str, chain: str, mutations: List[Dict[str, Any]],
                    cart_cycles: int = _DEFAULT_CART_CYCLES,
                    timeout: int = 1800, log=print) -> Dict[str, Optional[float]]:
    """Run the benchmark cartesian_ddg arm for one (pdb, chain) batch.

    mutations: [{"resnum", "wt", "mut", "variant"}].  Returns {variant: ddg} for the
    variants that scored; failures are omitted (→ not_computed upstream, never faked).
    """
    from wsl_bridge import WSLBridge
    wsl = WSLBridge()
    if not wsl.is_available():
        log("    [rosetta_cart] WSL unavailable → all not_computed")
        return {}
    pdb_wsl = wsl.copy_to_wsl(pdb_path)
    script = _worker_script(pdb_wsl, chain, mutations, cart_cycles)
    res = wsl.run_python_script(script, timeout=timeout)
    if not res.get("ok"):
        log(f"    [rosetta_cart] worker failed → not_computed: "
            f"{(res.get('stderr') or res.get('error') or '')[:120]}")
        return {}
    for line in (res.get("stdout") or "").splitlines():
        if line.startswith("WORKER_RESULT "):
            data = json.loads(line[len("WORKER_RESULT "):])
            if data.get("errors"):
                log(f"    [rosetta_cart] {len(data['errors'])} per-mut errors "
                    f"(omitted, not faked): {list(data['errors'])[:5]}")
            return {k: v for k, v in data.get("results", {}).items()}
    log("    [rosetta_cart] no WORKER_RESULT line → not_computed")
    return {}


# ─────────────────────────────────────────────────────────────────────────────────────────────────
# CANONICAL-LOCAL cartesian_ddg arm (2026-06-28) — the archaeology fix.
# The score_cartesian above is a WHOLE-POSE manual cartesian FastRelax (no MoveMap) — the archaeology
# (PROJECT_CONTEXT §7/§13 2026-06-28) showed its non-convergence/bias/cost are whole-pose artifacts,
# NOT canonical cartesian_ddg. This arm implements the CANONICAL-LOCAL protocol the build can express
# (it lacks CartesianddGMover), changing exactly the flagged deviations:
#   (b) LOCAL relax — a MoveMap restricting bb+chi to a ~6 Å neighbourhood of the mutation (the
#       constrained local problem canonical cartesian_ddg solves), NOT whole-pose.
#   (d) canonical ITERATIONS / mean-of-converged — N paired local opts, report the running mean +
#       per-iteration ddG so convergence (Δ<~1 REU) is measurable.
#   (e) PAIRED WT — the WT is re-locally-relaxed each iteration (not the asymmetric single cached WT).
#   (c) ref2015_cart — kept (already canonical).
# Plus the canonical "relax the input once" baseline (coordinate-constrained, so the one-time relax
# doesn't drift — the per-mutation cost is then purely the cheap local opt). BENCHMARK-ONLY, additive,
# NOT wired into production (does not touch _run_rosetta_local / the deep tier).

_DEFAULT_LOCAL_RADIUS = 6.0
_DEFAULT_LOCAL_ITERS = 5


def _local_worker_script(pdb_wsl: str, chain: str, muts: List[Dict[str, Any]],
                         radius: float, iters: int) -> str:
    """Self-contained PyRosetta CANONICAL-LOCAL cartesian_ddg worker (runs in WSL)."""
    payload = json.dumps(muts)
    return f'''
import json, sys, time
import pyrosetta
pyrosetta.init("-mute all -ignore_unrecognized_res -ignore_zero_occupancy false")
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.core.kinematics import MoveMap

_three = {{
 "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","E":"GLU","Q":"GLN","G":"GLY",
 "H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE","P":"PRO","S":"SER",
 "T":"THR","W":"TRP","Y":"TYR","V":"VAL"}}

sf = pyrosetta.create_score_function("ref2015_cart")
muts = json.loads({payload!r})
chain = {chain!r}
R = {radius}
N = {iters}

def neighbors(pose, center, radius):
    sel = set([center])
    cres = pose.residue(center)
    chx = [cres.xyz(a) for a in range(1, cres.natoms()+1) if not cres.atom_is_hydrogen(a)]
    for j in range(1, pose.total_residue()+1):
        if j == center: continue
        rj = pose.residue(j)
        # cheap nbr-atom prefilter, then refine on heavy-atom min distance
        if cres.nbr_atom_xyz().distance(rj.nbr_atom_xyz()) > radius + 8.0: continue
        close = False
        for a in range(1, rj.natoms()+1):
            if rj.atom_is_hydrogen(a): continue
            xa = rj.xyz(a)
            for xi in chx:
                if (xi - xa).norm() <= radius: close = True; break
            if close: break
        if close: sel.add(j)
    return sel

def movemap_for(sel):
    mm = MoveMap(); mm.set_bb(False); mm.set_chi(False); mm.set_jump(False)
    for j in sel: mm.set_bb(j, True); mm.set_chi(j, True)
    return mm

def local_relax(pose, mm, repeats=1):
    fr = FastRelax(sf, repeats); fr.cartesian(True); fr.set_movemap(mm); fr.max_iter(200)
    fr.apply(pose)

# canonical "relax the input ONCE" baseline, coordinate-constrained so it doesn't drift
try:
    base = pose_from_pdb({pdb_wsl!r})
except Exception as e:
    print("WORKER_FATAL load:"+str(e)[:120]); sys.exit(1)
fr0 = FastRelax(sf, 1); fr0.cartesian(True); fr0.constrain_relax_to_start_coords(True); fr0.max_iter(300)
t_base = time.time(); fr0.apply(base); base_sec = round(time.time()-t_base, 1)
pi = base.pdb_info()

results = {{}}; errors = {{}}; timing = {{}}
for m in muts:
    var = m["variant"]
    try:
        pr = pi.pdb2pose(chain, int(m["resnum"]))
        if pr == 0: errors[var] = "resnum not in pose"; continue
        if base.residue(pr).name1() != m["wt"]:
            errors[var] = f"wt mismatch pose={{base.residue(pr).name1()}} exp={{m['wt']}}"; continue
        sel = neighbors(base, pr, R); mm = movemap_for(sel)
        ddgs = []; t0 = time.time()
        for i in range(N):
            wt = base.clone(); local_relax(wt, mm); ws = sf(wt)
            mp = base.clone(); MutateResidue(pr, _three[m["mut"]]).apply(mp); local_relax(mp, mm); ms = sf(mp)
            ddgs.append(round(ms - ws, 4))
        timing[var] = round((time.time()-t0)/max(1, N), 2)   # sec per iteration (one WT + one MUT local relax)
        results[var] = {{"ddg_iters": ddgs, "n_sel": len(sel)}}
    except Exception as e:
        errors[var] = type(e).__name__+":"+str(e)[:80]

print("WORKER_RESULT "+json.dumps({{"results": results, "errors": errors, "timing": timing,
      "n_res": base.total_residue(), "base_sec": base_sec}}))
'''


def score_cartesian_local(pdb_path: str, chain: str, mutations: List[Dict[str, Any]],
                          radius: float = _DEFAULT_LOCAL_RADIUS, iters: int = _DEFAULT_LOCAL_ITERS,
                          timeout: int = 3600, log=print) -> Dict[str, Any]:
    """CANONICAL-LOCAL cartesian_ddg for one (pdb, chain) batch → {variant: {ddg_iters:[...], n_sel}}
    plus {"_timing": {variant: sec/iter}, "_base_sec": one-time baseline relax, "_n_res"}. Failures
    omitted (→ not_computed upstream, never faked)."""
    from wsl_bridge import WSLBridge
    wsl = WSLBridge()
    if not wsl.is_available():
        log("    [cart_local] WSL unavailable → all not_computed"); return {}
    pdb_wsl = wsl.copy_to_wsl(pdb_path)
    res = wsl.run_python_script(_local_worker_script(pdb_wsl, chain, mutations, radius, iters),
                                timeout=timeout)
    if not res.get("ok"):
        log(f"    [cart_local] worker failed: {(res.get('stderr') or res.get('error') or '')[:160]}")
        return {}
    for line in (res.get("stdout") or "").splitlines():
        if line.startswith("WORKER_RESULT "):
            data = json.loads(line[len("WORKER_RESULT "):])
            if data.get("errors"):
                log(f"    [cart_local] {len(data['errors'])} per-mut errors (omitted): {list(data['errors'])[:5]}")
            out = dict(data.get("results", {}))
            out["_timing"] = data.get("timing", {}); out["_base_sec"] = data.get("base_sec")
            out["_n_res"] = data.get("n_res")
            return out
    log("    [cart_local] no WORKER_RESULT line → not_computed"); return {}


if __name__ == "__main__":
    # tiny self-smoke (1PGA, 2 muts) — bounded feasibility check, NOT a benchmark run
    pdb = str(_ROOT / "RaSP_repo" / "data" / "test" / "Protein_G" / "structure" / "raw" / "1PGA.pdb")
    muts = [{"resnum": 1, "wt": "M", "mut": "A", "variant": "M1A"},
            {"resnum": 50, "wt": "K", "mut": "P", "variant": "K50P"}]
    out = score_cartesian(pdb, "A", muts)
    print("cartesian_ddg smoke:", out)
