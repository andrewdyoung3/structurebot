"""
scripts/stability_datagen.py — cross-voter-vs-experiment DATA-GENERATION harness.

PURPOSE (data-gen ONLY — NO interpretation): per mutation, run every LIVE voter
(physics: Rosetta + RaSP; ML: ThermoMPNN; dynamics: DynaMut2; properties: CamSol,
ESM) and append one row to a crash-safe, RESUMABLE JSONL.  Stores all voter
outputs (with not_computed markers), the experimental + Rosetta-reference labels,
and fwd/rev pairing fields.  FORBIDDEN here: correlations-as-confidence, weights,
accuracy claims — this only COLLECTS.

Whole-chain voters (ThermoMPNN/RaSP/CamSol/ESM) are computed ONCE per (pdb,chain)
and looked up per mutation.  Rosetta + DynaMut2 are per-mutation (the long poles):
Rosetta runs a CAPPED subset; DynaMut2 a SMALL fixed subset.

Error-first: any voter failing on any mutation/protein → not_computed, logged, the
run continues.  Resumable: re-running skips rows already present in the JSONL.

Usage:
  python scripts/stability_datagen.py --assemble-report      # inventory only
  python scripts/stability_datagen.py --smoke                # 3 muts, all voters
  python scripts/stability_datagen.py --out <jsonl> --rosetta-cap N --dynamut2-cap M
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import config  # noqa: E402
from residue_mapping import candidate_key, ordered_chain_residues  # noqa: E402

_RASP_DATA = _ROOT / "RaSP_repo" / "data" / "test"
_CACHE = _ROOT / "cache"


# ── Assembly (LOCAL labeled sets + matching structures) ────────────────────────

def _parse_variant(v: str) -> Optional[Tuple[str, int, str]]:
    v = v.strip()
    try:
        return v[0], int(v[1:-1]), v[-1]
    except (ValueError, IndexError):
        return None


def _load_csv_set(set_name: str, struct_subdir: str, label_kind: str) -> List[Dict[str, Any]]:
    """Load one RaSP-format labeled set (pdbid,chainid,variant,score) + structures."""
    csv_path = _RASP_DATA / set_name / f"ddG_{label_kind}" / "ddg.csv"
    struct_dir = _RASP_DATA / set_name.split("/")[0] / "structure" / "raw"
    out: List[Dict[str, Any]] = []
    if not csv_path.is_file():
        return out
    for row in csv.DictReader(open(csv_path)):
        pv = _parse_variant(row.get("variant", ""))
        if not pv:
            continue
        wt, resnum, mut = pv
        pdbid = row["pdbid"].strip()
        pdb = struct_dir / f"{pdbid}.pdb"
        # some sets suffix the raw file (homology models) — take the first match
        if not pdb.is_file():
            cand = sorted(struct_dir.glob(f"{pdbid}*.pdb"))
            pdb = cand[0] if cand else pdb
        out.append({
            "set": set_name, "pdbid": pdbid, "chain": row.get("chainid", "A").strip() or "A",
            "wt": wt, "resnum": resnum, "mut": mut, "variant": f"{wt}{resnum}{mut}",
            "exp_ddg": _f(row.get("score")), "pdb_path": str(pdb),
        })
    return out


def _f(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _t4l_controls() -> List[Dict[str, Any]]:
    """T4 lysozyme (2LZM) experimental anchors — continuity with the sign battery."""
    pdb = str(_CACHE / "2LZM.pdb")
    controls = [(99,"L","A",5.0),(133,"L","A",2.7),(121,"L","A",2.7),(153,"F","A",3.0),
                (149,"V","A",2.0),(118,"L","A",2.5),(87,"V","M",1.5),(117,"S","V",0.9),
                (152,"T","S",0.5),(116,"N","D",0.1)]
    return [{"set":"T4L_2LZM","pdbid":"2LZM","chain":"A","wt":f,"resnum":p,"mut":t,
             "variant":f"{f}{p}{t}","exp_ddg":float(e),"pdb_path":pdb} for (p,f,t,e) in controls]


def assemble() -> List[Dict[str, Any]]:
    """Assemble the FULL chosen set from LOCAL data only."""
    muts: List[Dict[str, Any]] = []
    # PRIMARY: Protein G (1PGA) — one structure, BOTH tails, exp + Rosetta ref.
    pg = _load_csv_set("Protein_G", "structure", "experimental")
    ref = {(r["pdbid"], r["chain"], r["variant"]): r["exp_ddg"]
           for r in _load_csv_set("Protein_G", "structure", "Rosetta")}
    for m in pg:
        m["rosetta_ref_ddg"] = ref.get((m["pdbid"], m["chain"], m["variant"]))
        m["pair_id"] = None
    muts += pg
    # T4L anchors
    for m in _t4l_controls():
        m["rosetta_ref_ddg"] = None; m["pair_id"] = None
        muts += [m]
    # ANTI-SYMMETRY (Ssym fwd/rev) is DEFERRED: Ssym puts ~1 mutant structure per
    # mutation (~372 structures) → the whole-chain voters would run hundreds of
    # times (hours), and its fwd (Ssym_dir) / rev (Ssym_inv) entries carry DIFFERENT
    # pdbids (WT vs mutant) with no shared key, so a clean fwd↔rev pairing needs the
    # Ssym mapping file.  Out of scope for this windowed pass; anti-symmetry was
    # already empirically validated in Part-1's sign battery (§13).  The row schema
    # (pair_id / antisym_dir) is anti-symmetry-ready for a focused follow-up.
    return muts


# ── Whole-chain voters (computed once per protein) ─────────────────────────────

def _safe(fn, label, log):
    try:
        return fn()
    except Exception as exc:
        log(f"    [{label}] FAILED → not_computed: {type(exc).__name__}: {str(exc)[:120]}")
        return None


def _whole_chain_voters(pdb_path, chain, group, log):
    """Run ThermoMPNN/RaSP/CamSol/ESM ONCE for this protein; return lookup dicts."""
    out = {"thermo": {}, "rasp": {}, "camsol_pos": {}, "esm_pos": {}, "resnum2seq": {}}
    ordered = ordered_chain_residues(pdb_path, chain)
    if ordered:
        out["resnum2seq"] = {rn: i for i, (rn, _ic, _aa) in enumerate(ordered, 1)}
        seq = "".join(aa for _, _, aa in ordered)
    else:
        seq = None
    cands = [{"position": m["resnum"], "from_aa": m["wt"], "to_aa": m["mut"]} for m in group]

    def _thermo():
        from thermompnn_bridge import ThermoMPNNBridge
        b = ThermoMPNNBridge()
        if not b.is_available():
            log("    [thermompnn] unavailable → not_computed"); return {}
        d, _ = b.score_mutations(pdb_path, chain, cands, progress=lambda *_: None)
        return d
    out["thermo"] = _safe(_thermo, "thermompnn", log) or {}

    def _rasp():
        from rasp_bridge import RaSPBridge
        b = RaSPBridge()
        if not b.is_available():
            log("    [rasp] unavailable → not_computed"); return {}
        d, _ = b.score_mutations(pdb_path, chain, cands, progress=lambda *_: None)
        return d
    out["rasp"] = _safe(_rasp, "rasp", log) or {}

    if seq:
        def _camsol():
            from camsol_bridge import CamsolBridge
            r = CamsolBridge().analyze(seq, model_id="1", chain=chain)
            return {int(k): float(v) for k, v in r.data["scores"].items()} if r.success else {}
        out["camsol_pos"] = _safe(_camsol, "camsol", log) or {}

        def _esm():
            from esm_bridge import EsmBridge
            r = EsmBridge().analyze(seq, model_id="1", inference_timeout=120)
            return {int(k): float(v) for k, v in r.data["conservation"].items()} if r.success else {}
        out["esm_pos"] = _safe(_esm, "esm", log) or {}
    return out


def _rosetta_batch(pdb_path, chain, muts, backend, log) -> Dict[str, float]:
    """Per-mutation Rosetta or DynaMut2 batch → {f'{wt}{resnum}{mut}': ddg}."""
    if not muts:
        return {}
    try:
        from rosetta_bridge import RosettaBridge
        b = RosettaBridge()
        if backend:
            b._backend = backend
        mutations = [{"chain": chain, "position": m["resnum"], "from_aa": m["wt"],
                      "to_aa": m["mut"]} for m in muts]
        r = b.analyze(pdb_path=pdb_path, mutations=mutations, chain=chain,
                      progress_callback=lambda *_: None, ddg_basis="asymmetric")
        if not r.success:
            log(f"    [{backend or 'rosetta'}] batch failed → not_computed: {(r.error or '')[:100]}")
            return {}
        scores = r.data.get("ddg_scores", {})
        emp = set(r.data.get("empirical_fallbacks", []))  # exclude empirical (not real)
        return {k: v for k, v in scores.items() if k not in emp}
    except Exception as exc:
        log(f"    [{backend or 'rosetta'}] batch error → not_computed: {str(exc)[:120]}")
        return {}


# ── Harness ────────────────────────────────────────────────────────────────────

def _key(m) -> str:
    return f"{m['pdbid']}:{m['chain']}:{m['wt']}{m['resnum']}{m['mut']}"


def run(manifest, out_path, rosetta_keys, dynamut2_keys, log=print):
    done = set()
    if Path(out_path).is_file():
        for ln in open(out_path):
            try:
                done.add(json.loads(ln)["key"])
            except Exception:
                pass
        log(f"resume: {len(done)} rows already present in {out_path}")

    # group by protein
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for m in manifest:
        groups.setdefault((m["pdbid"], m["chain"], m["pdb_path"]), []).append(m)

    fh = open(out_path, "a", buffering=1)  # line-buffered → crash-safe append
    n_written = 0
    for (pdbid, chain, pdb_path), group in groups.items():
        pending = [m for m in group if _key(m) not in done]
        if not pending:
            continue
        if not Path(pdb_path).is_file():
            log(f"[{pdbid}/{chain}] structure missing ({pdb_path}) → all not_computed")
        log(f"[{pdbid}/{chain}] {len(pending)} pending of {len(group)} …")
        wc = _whole_chain_voters(pdb_path, chain, pending, log) if Path(pdb_path).is_file() else \
            {"thermo": {}, "rasp": {}, "camsol_pos": {}, "esm_pos": {}, "resnum2seq": {}}

        ros_muts = [m for m in pending if _key(m) in rosetta_keys]
        dyn_muts = [m for m in pending if _key(m) in dynamut2_keys]
        # PHYSICS Rosetta = the REAL WSL PyRosetta ("local" backend) — NOT the
        # default backend (which resolves to dynamut2 here → would collide with the
        # dynamics voter).  DynaMut2 = the "dynamut2" backend.  Distinct voters.
        ros = _rosetta_batch(pdb_path, chain, ros_muts, "local", log) if ros_muts else {}
        dyn = _rosetta_batch(pdb_path, chain, dyn_muts, "dynamut2", log) if dyn_muts else {}

        for m in pending:
            ck = candidate_key(chain, m["resnum"], m["wt"], m["mut"])
            rk = f"{m['wt']}{m['resnum']}{m['mut']}"
            seqi = wc["resnum2seq"].get(m["resnum"])
            row = {
                "key": _key(m), "set": m["set"], "pdbid": pdbid, "chain": chain,
                "resnum": m["resnum"], "wt": m["wt"], "mut": m["mut"], "variant": m["variant"],
                "exp_ddg": m.get("exp_ddg"), "rosetta_ref_ddg": m.get("rosetta_ref_ddg"),
                "pair_id": m.get("pair_id"), "antisym_dir": m.get("antisym_dir"),
                # voters — None == not_computed (NEVER fabricated/0.0)
                "rosetta_ddg":   ros.get(rk),
                "rasp_ddg":      wc["rasp"].get(ck),
                "thermompnn_ddg": wc["thermo"].get(ck),
                "dynamut2_ddg":  dyn.get(rk),
                "camsol_score":  wc["camsol_pos"].get(seqi) if seqi else None,
                "esm_tolerance": (round(1.0 - wc["esm_pos"][seqi], 4)
                                  if (seqi and seqi in wc["esm_pos"]) else None),
                "seqindex": seqi,
                "rosetta_in_subset": _key(m) in rosetta_keys,
                "dynamut2_in_subset": _key(m) in dynamut2_keys,
            }
            fh.write(json.dumps(row) + "\n"); fh.flush()
            n_written += 1
    fh.close()
    log(f"wrote {n_written} new rows → {out_path}")
    return n_written


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble-report", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(_ROOT / "cache" / "stability_datagen" / "rows.jsonl"))
    ap.add_argument("--rosetta-cap", type=int, default=0)
    ap.add_argument("--dynamut2-cap", type=int, default=0)
    a = ap.parse_args()

    manifest = assemble()
    by_set: Dict[str, int] = {}
    for m in manifest:
        by_set[m["set"]] = by_set.get(m["set"], 0) + 1
    print(f"ASSEMBLED {len(manifest)} mutations across {len({m['pdbid'] for m in manifest})} structures")
    for s, n in sorted(by_set.items()):
        print(f"  {s}: {n}")
    if a.assemble_report:
        return

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if a.smoke:
        # 3 1PGA mutations spanning both tails, ALL voters
        pg = [m for m in manifest if m["pdbid"] == "1PGA"]
        pg.sort(key=lambda m: (m["exp_ddg"] if m["exp_ddg"] is not None else 0))
        sm = [pg[0], pg[len(pg)//2], pg[-1]] if len(pg) >= 3 else pg
        keys = {_key(m) for m in sm}
        out = str(Path(a.out).with_name("smoke.jsonl"))
        if Path(out).is_file():
            Path(out).unlink()
        print(f"SMOKE: {len(sm)} mutations, all voters → {out}")
        t0 = time.time()
        run(sm, out, rosetta_keys=keys, dynamut2_keys=keys)
        print(f"smoke elapsed {time.time()-t0:.0f}s")
        for ln in open(out):
            r = json.loads(ln)
            print("  ", {k: r[k] for k in ("variant","exp_ddg","rosetta_ddg","rasp_ddg",
                  "thermompnn_ddg","dynamut2_ddg","camsol_score","esm_tolerance")})
        return

    # full run — caller supplies caps; build the rosetta/dynamut2 subsets
    ros_keys, dyn_keys = _select_subsets(manifest, a.rosetta_cap, a.dynamut2_cap)
    print(f"rosetta subset: {len(ros_keys)} | dynamut2 subset: {len(dyn_keys)}")
    run(manifest, a.out, ros_keys, dyn_keys)


def _select_subsets(manifest, rosetta_cap, dynamut2_cap):
    """Rosetta subset: prefer 1PGA both-tails + anti-symmetry pairs, capped.
    DynaMut2 subset: small fixed sample incl. anti-symmetry pairs + both tails."""
    def both_tail_sample(rows, cap):
        labeled = [m for m in rows if m.get("exp_ddg") is not None]
        labeled.sort(key=lambda m: m["exp_ddg"])
        if cap <= 0 or cap >= len(labeled):
            return labeled
        # even stride across the sorted-by-exp range → spans both tails
        step = len(labeled) / cap
        return [labeled[int(i*step)] for i in range(cap)]
    ros = both_tail_sample([m for m in manifest if m["pdbid"] == "1PGA"], rosetta_cap)
    # always include the anti-symmetry (Ssym) pairs + T4L anchors in rosetta if present
    extra = [m for m in manifest if m.get("pair_id") or m["set"] == "T4L_2LZM"]
    ros_keys = {_key(m) for m in ros} | {_key(m) for m in extra}
    dyn = both_tail_sample([m for m in manifest if m["pdbid"] == "1PGA"], dynamut2_cap)
    dyn_keys = {_key(m) for m in dyn} | {_key(m) for m in manifest if m["set"] == "T4L_2LZM"}
    return ros_keys, dyn_keys


if __name__ == "__main__":
    main()
