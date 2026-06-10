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

# Windows UTF-8 stdout convention (§5): log lines use →/Greek; cp1252 would raise.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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


# ── Manifest-driven assembly (Task 2 — arbitrary multi-protein, provenance-tagged) ─

def _resolve(path: str) -> Path:
    """Resolve a manifest path: absolute as-is, else relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else (_ROOT / path)


def _index_csv_by_pdbid(path: str) -> Dict[str, List[Dict[str, str]]]:
    idx: Dict[str, List[Dict[str, str]]] = {}
    p = _resolve(path)
    if not p.is_file():
        return idx
    for row in csv.DictReader(open(p)):
        pid = (row.get("pdbid") or "").strip()
        if pid:
            idx.setdefault(pid, []).append(row)
    return idx


def assemble_from_manifest(manifest_path: str, proposed_only: bool = True,
                           log=print) -> List[Dict[str, Any]]:
    """Assemble an arbitrary multi-protein set from a Task-1 manifest.

    Consumes scripts/calibration_manifest.draft.json's `entries` (per-(set,pdbid,chain)
    + provenance + struct/exp/rosetta-ref pointers).  Loads each protein's mutations
    from its experimental CSV, attaches the Rosetta reference where present, resolves
    the structure via the struct_dir glob, and CARRIES the per-voter provenance tags +
    role onto every mutation (lossless — collection stays complete; the analysis layer
    filters by provenance).  Pure assembly; runs no voters."""
    data = json.loads(_resolve(manifest_path).read_text(encoding="utf-8"))
    entries = data.get("entries", [])
    if proposed_only:
        entries = [e for e in entries if e.get("proposed_include")]
    exp_cache: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    ref_cache: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    muts: List[Dict[str, Any]] = []
    n_missing_struct = 0
    for e in entries:
        exp_idx = exp_cache.setdefault(e["exp_csv"], _index_csv_by_pdbid(e["exp_csv"]))
        ref_path = e.get("rosetta_ref_csv")
        ref_idx = (ref_cache.setdefault(ref_path, _index_csv_by_pdbid(ref_path))
                   if ref_path else {})
        struct_dir = _resolve(e["struct_dir"])
        cand = sorted(struct_dir.glob(f"{e['pdbid']}*.pdb"))
        pdb_path = str(cand[0]) if cand else str(struct_dir / f"{e['pdbid']}.pdb")
        if not cand:
            n_missing_struct += 1
        ref_by_var = {(r.get("variant") or "").strip(): _f(r.get("score"))
                      for r in ref_idx.get(e["pdbid"], [])}
        for r in exp_idx.get(e["pdbid"], []):
            pv = _parse_variant(r.get("variant", ""))
            if not pv:
                continue
            wt, resnum, mut = pv
            var = f"{wt}{resnum}{mut}"
            muts.append({
                "set": e["set"], "pdbid": e["pdbid"],
                "chain": (r.get("chainid") or e.get("chain") or "A").strip() or "A",
                "wt": wt, "resnum": resnum, "mut": mut, "variant": var,
                "exp_ddg": _f(r.get("score")),
                "rosetta_ref_ddg": ref_by_var.get(var),
                "pdb_path": pdb_path,
                "pair_id": None, "antisym_dir": None,
                "provenance": e.get("provenance") or {}, "role": e.get("role"),
            })
    log(f"manifest: {len(entries)} protein entries → {len(muts)} mutations"
        f"{f' ({n_missing_struct} missing structures)' if n_missing_struct else ''}")
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

        # CHUNK the per-mutation long poles (Rosetta/DynaMut2) so rows are written
        # incrementally (~every CHUNK muts) — crash-safe + resumable MID-protein, not
        # only at protein end.  PHYSICS Rosetta = REAL WSL PyRosetta ("local"); the
        # default backend resolves to dynamut2 here (smoke caught that collision).
        CHUNK = 20
        for ci in range(0, len(pending), CHUNK):
            chunk = pending[ci:ci + CHUNK]
            cros = [m for m in chunk if _key(m) in rosetta_keys]
            cdyn = [m for m in chunk if _key(m) in dynamut2_keys]
            ros = _rosetta_batch(pdb_path, chain, cros, "local", log) if cros else {}
            dyn = _rosetta_batch(pdb_path, chain, cdyn, "dynamut2", log) if cdyn else {}
            for m in chunk:
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
                    # provenance tags carried from the manifest (lossless; the
                    # analysis layer filters per-voter by training-disjointness).
                    # Legacy (non-manifest) rows leave these None.
                    "role": m.get("role"),
                    "prov_thermompnn": (m.get("provenance") or {}).get("thermompnn"),
                    "prov_dynamut2": (m.get("provenance") or {}).get("dynamut2"),
                    "prov_rasp": (m.get("provenance") or {}).get("rasp"),
                    "prov_rosetta": (m.get("provenance") or {}).get("rosetta"),
                }
                fh.write(json.dumps(row) + "\n"); fh.flush()
                n_written += 1
            log(f"  [{pdbid}/{chain}] chunk {ci//CHUNK + 1}: +{len(chunk)} rows "
                f"(total {n_written}); rosetta {len(ros)}/{len(cros)} dynamut2 {len(dyn)}/{len(cdyn)}")
    fh.close()
    log(f"wrote {n_written} new rows → {out_path}")
    return n_written


# ── CLI ─────────────────────────────────────────────────────────────────────────

_VOTERS = ["rosetta_ddg", "rasp_ddg", "thermompnn_ddg", "dynamut2_ddg",
           "camsol_score", "esm_tolerance"]


def summarize(jsonl_path: str) -> None:
    """DESCRIPTIVE-ONLY summary (brief 2.6): counts, coverage, ranges, not_computed
    tallies, anti-symmetry fwd+rev sums.  NO correlation-as-confidence, NO weights,
    NO verdicts — this only describes the collected data."""
    rows = [json.loads(ln) for ln in open(jsonl_path)] if Path(jsonl_path).is_file() else []
    print(f"=== DESCRIPTIVE SUMMARY (data only — no interpretation) — {jsonl_path}")
    print(f"rows: {len(rows)}")
    if not rows:
        return
    by_set: Dict[str, int] = {}
    for r in rows:
        by_set[r.get("set", "?")] = by_set.get(r.get("set", "?"), 0) + 1
    print("per-set:", dict(sorted(by_set.items())))
    # label coverage
    for lab in ("exp_ddg", "rosetta_ref_ddg"):
        vals = [r[lab] for r in rows if r.get(lab) is not None]
        if vals:
            neg = sum(1 for v in vals if v < 0); pos = sum(1 for v in vals if v > 0)
            print(f"label {lab}: {len(vals)}/{len(rows)} present | range [{min(vals):.2f},{max(vals):.2f}] | neg {neg} pos {pos}")
        else:
            print(f"label {lab}: 0/{len(rows)} present")
    # per-voter coverage + ranges + not_computed
    print("per-voter (computed / not_computed | range | median):")
    for v in _VOTERS:
        vals = [r[v] for r in rows if r.get(v) is not None]
        nc = len(rows) - len(vals)
        if vals:
            sv = sorted(vals); med = sv[len(sv)//2]
            print(f"  {v:15s}: {len(vals):4d} computed / {nc:4d} not_computed | "
                  f"[{min(vals):+.2f},{max(vals):+.2f}] | median {med:+.2f}")
        else:
            print(f"  {v:15s}:    0 computed / {nc:4d} not_computed")
    # anti-symmetry (data only — store fwd+rev sums, NO verdict)
    pairs: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        if r.get("pair_id"):
            pairs.setdefault(r["pair_id"], {})[r.get("antisym_dir", "?")] = r
    complete = [p for p in pairs.values() if "fwd" in p and "rev" in p]
    print(f"anti-symmetry pairs (fwd+rev both present): {len(complete)}")
    if not complete:
        print("  (none — Ssym anti-symmetry deferred this pass; schema is pair-ready)")
    print("NOTE: correlations / weights / accuracy verdicts are OUT OF SCOPE here "
          "(the attended calibration step).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble-report", action="store_true")
    ap.add_argument("--summary", metavar="JSONL")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--manifest", metavar="JSON",
                    help="drive assembly from a Task-1 calibration manifest (multi-protein)")
    ap.add_argument("--all-entries", action="store_true",
                    help="with --manifest: include ALL entries, not just proposed_include")
    ap.add_argument("--list", action="store_true",
                    help="list assembled proteins/mutation counts and exit (no voters)")
    ap.add_argument("--out", default=str(_ROOT / "cache" / "stability_datagen" / "rows.jsonl"))
    ap.add_argument("--rosetta-cap", type=int, default=0)
    ap.add_argument("--dynamut2-cap", type=int, default=0)
    a = ap.parse_args()

    if a.summary:
        summarize(a.summary)
        return

    manifest = (assemble_from_manifest(a.manifest, proposed_only=not a.all_entries)
                if a.manifest else assemble())
    by_set: Dict[str, int] = {}
    for m in manifest:
        by_set[m["set"]] = by_set.get(m["set"], 0) + 1
    print(f"ASSEMBLED {len(manifest)} mutations across {len({m['pdbid'] for m in manifest})} structures")
    for s, n in sorted(by_set.items()):
        print(f"  {s}: {n}")
    if a.list:
        # per-protein listing + provenance (no voters) — proves manifest iteration
        per: Dict[Tuple[str, str], int] = {}
        for m in manifest:
            per[(m["set"], m["pdbid"])] = per.get((m["set"], m["pdbid"]), 0) + 1
        for (s, pid), n in sorted(per.items()):
            prov = next((m.get("provenance") for m in manifest
                         if m["pdbid"] == pid and m["set"] == s), {}) or {}
            print(f"  {s}/{pid}: {n} muts | prov={prov}")
        return
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


def _both_tail_sample(rows, cap):
    """Even stride across the exp-sorted range → spans both tails.  cap<=0 → all."""
    labeled = [m for m in rows if m.get("exp_ddg") is not None]
    labeled.sort(key=lambda m: m["exp_ddg"])
    if cap <= 0 or cap >= len(labeled):
        return labeled
    step = len(labeled) / cap
    return [labeled[int(i * step)] for i in range(cap)]


def _select_subsets(manifest, rosetta_cap, dynamut2_cap):
    """Multi-protein subset selection for the two PER-MUTATION long poles.

    ROSETTA (leakage-free anchor — wanted broadly): both-tail sample across ALL
    proteins, capped; anti-symmetry pairs + T4L anchors always included.
    DYNAMUT2 (remote API, capped, S2648-overlapping): small fixed both-tail sample,
    PREFERRING dynamut2-clean proteins (provenance != 'training') so the scarce
    DynaMut2 budget is spent where it is training-disjoint; falls back to the whole
    set when no provenance tags are present (legacy)."""
    ros = _both_tail_sample(manifest, rosetta_cap)
    extra = [m for m in manifest if m.get("pair_id") or m.get("set") == "T4L_2LZM"]
    ros_keys = {_key(m) for m in ros} | {_key(m) for m in extra}

    dyn_pool = [m for m in manifest
                if (m.get("provenance") or {}).get("dynamut2") != "training"]
    if not dyn_pool:                       # legacy / untagged → whole set
        dyn_pool = manifest
    dyn = _both_tail_sample(dyn_pool, dynamut2_cap)
    dyn_keys = {_key(m) for m in dyn} | {_key(m) for m in manifest
                                         if m.get("set") == "T4L_2LZM"}
    return ros_keys, dyn_keys


if __name__ == "__main__":
    main()
