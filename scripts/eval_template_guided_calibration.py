"""
eval_template_guided_calibration.py — TEMPLATE-GUIDANCE useful-envelope TITRATION (eval, NOT a
feature). Titrates ΔAccuracy vs template divergence on SOLVED structures (ground truth S_P).

ChimeraX-FREE by design: folds via the LOCAL Boltz bridge, aligns via the LOCAL US-align binary,
measures sequence identity via Biopython — so the study runs headless with no viewer/session.

Per (target, template):
  • U  = UNGUIDED fold  (Boltz, MSA-free)                         — the no-template baseline
  • G  = GUIDED-SOFT fold (Boltz + templates:[template])         [+ HARD on a subset]
  • US-align to GROUND TRUTH:  TM(U, S_P)  and  TM(G, S_P)        — accuracy toward the truth
  • US-align to TEMPLATE:      TM(G, template)                    — adoption
  • divergence of template from the target: NWalign seq-id  AND  US-align structural-TM(template, S_P)
  • ΔAccuracy = TM(G,S_P) − TM(U,S_P)        ( >0 = guidance helped,  <0 = guidance hurt )

TM convention: USalign(<query>, <reference>) — we always pass the GROUND-TRUTH (or template) as the
REFERENCE, so the headline number is `tm2` = TM normalized by the reference length (coverage of the
true/template fold). `tm1` (query-normalized) is also recorded.

Modes:
  --prescreen   fold the HARD-candidate sequences UNGUIDED only and report mean pLDDT + cross-seed
                flex → EVIDENCE for flagging which targets are genuinely hard (no guidance yet).
  --titrate     the full per-(target,template) titration.
  --target NAME restrict to one registry target.   --hard  also run guided-HARD (force+threshold).
  --flex-seeds N  cross-seed flex over N seeds (default 0 = off; 2 = one extra seed, cheap).

Run (minimal first pass): venv/Scripts/python.exe scripts/eval_template_guided_calibration.py --titrate --target myoglobin
"""
from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config as _cfg
from boltz_bridge import BoltzBridge
from tool_router import ToolRouter
from wsl_bridge import WSLBridge


# ── Test-set registry ─────────────────────────────────────────────────────────────────
# Each target: a solved structure (ground truth) + a homologue TEMPLATE LADDER spanning a
# divergence range. The identical-template entry (template == target's own solved structure) is
# the POSITIVE CONTROL / ceiling (the steer-sanity deferred from the build). divergence (seq-id +
# structural-TM-to-truth) is MEASURED at run time, not assumed — `tag` is only a human label.
TARGETS: Dict[str, dict] = {
    # Easy anchor: all-alpha globin — folds well unguided, so guidance has little to add. The
    # ladder still gives a clean DIVERGENCE axis (identical → 27% → 15% id, all same fold).
    "myoglobin": {
        "pdb": "1MBN", "chain": "A",
        "templates": [
            {"pdb": "1MBN", "chain": "A", "tag": "identical (positive control / ceiling)"},
            {"pdb": "1MBA", "chain": "A", "tag": "homolog (Aplysia myoglobin)"},
            {"pdb": "4HHB", "chain": "A", "tag": "homolog ~27% id (hemoglobin alpha)"},
            {"pdb": "1LH1", "chain": "A", "tag": "distant ~15% id (leghemoglobin)"},
        ],
    },
}

# HARD-candidate targets — FLAGGED for the user before committing the full titration. Criteria:
# solved structure + a divergent homologue ladder + LIKELY poor unguided single-sequence Boltz
# fold (β-rich / large divergent families tend to be MSA-dependent). Whether each is ACTUALLY hard
# for THIS Boltz is verified by --prescreen (measure, don't guess — §0). NOT run until approved.
HARD_CANDIDATES: Dict[str, dict] = {
    "cspb":  {"pdb": "1MJC", "chain": "A",   # cold-shock protein B, ~69 aa OB-fold β-barrel; CSD superfamily
              "templates": [{"pdb": "1MJC", "chain": "A", "tag": "identical"},
                            {"pdb": "1C9O", "chain": "A", "tag": "homolog (Bc-Csp)"}]},
    "sh3":   {"pdb": "1SHG", "chain": "A",   # α-spectrin SH3, ~62 aa β-barrel; huge divergent SH3 family
              "templates": [{"pdb": "1SHG", "chain": "A", "tag": "identical"},
                            {"pdb": "1SEM", "chain": "A", "tag": "homolog SH3"}]},
    "fn3":   {"pdb": "1TEN", "chain": "A",   # tenascin FN3, ~90 aa β-sandwich; Ig-like superfamily
              "templates": [{"pdb": "1TEN", "chain": "A", "tag": "identical"},
                            {"pdb": "1FNF", "chain": "A", "tag": "homolog (fibronectin FN3)"}]},
}

_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M",
}

_router = ToolRouter(bridge=MagicMock(), session=MagicMock())
_wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
_USALIGN = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")


# ── structure / sequence helpers ──────────────────────────────────────────────────────
def download(pdb_id: str) -> Optional[str]:
    return _router._download_pdb_by_id(pdb_id)


def chain_seq_from_pdb(pdb_path: str, chain: str) -> str:
    """ATOM-derived one-letter sequence of *chain* (first model), in residue order. Modeled
    residues only — exactly what we fold + compare against the same structure."""
    from Bio.PDB import PDBParser
    p = PDBParser(QUIET=True).get_structure("s", pdb_path)
    model = next(iter(p))
    if chain not in model:
        # fall back to the first protein chain present
        for ch in model:
            if any(r.get_resname() in _3TO1 for r in ch):
                chain = ch.id
                break
    seq = []
    seen = set()
    for res in model[chain]:
        het, resseq, icode = res.id
        if het != " ":
            continue                         # skip HETATM/waters
        aa = _3TO1.get(res.get_resname())
        if aa and (resseq, icode) not in seen:
            seen.add((resseq, icode))
            seq.append(aa)
    return "".join(seq)


def nw_seq_id(a: str, b: str) -> float:
    """Global Needleman-Wunsch sequence identity (BLOSUM62), normalized by the SHORTER sequence
    (the NWalign -seqID convention). Biopython PairwiseAligner — deterministic, local, no binary."""
    if not a or not b:
        return 0.0
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.mode = "global"
    aln = aligner.align(a, b)[0]
    # count identical aligned columns
    idents = 0
    for (s1, e1), (s2, e2) in zip(aln.aligned[0], aln.aligned[1]):
        for i, j in zip(range(s1, e1), range(s2, e2)):
            if a[i] == b[j]:
                idents += 1
    return round(idents / min(len(a), len(b)), 4)


def usalign_tm(query_path: str, ref_path: str) -> Optional[dict]:
    """US-align *query* onto *ref* (ref = the normalization reference). Returns parsed
    {tm1(query-norm), tm2(ref-norm), rmsd, lali, idali, l1, l2} or None."""
    if not (query_path and ref_path and os.path.isfile(query_path) and os.path.isfile(ref_path)):
        return None
    q = _wsl.translate_path(os.path.abspath(query_path))
    r = _wsl.translate_path(os.path.abspath(ref_path))
    cmd = f"{shlex.quote(_USALIGN)} {shlex.quote(q)} {shlex.quote(r)} -outfmt 2 -m -"
    res = _wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 120))
    if not res.get("ok"):
        return None
    return ToolRouter._parse_usalign_output(res.get("stdout", ""))


def ca_xyz_from_cif(cif_path: str) -> Dict[int, Tuple[float, float, float]]:
    """{1-based residue index → CA xyz} from a Boltz output CIF (first chain), for cross-seed flex."""
    cols: List[str] = []
    out: Dict[int, Tuple[float, float, float]] = {}
    idx = 0
    seen = set()
    in_loop = False
    try:
        with open(cif_path) as fh:
            for line in fh:
                s = line.strip()
                if s.startswith("_atom_site."):
                    cols.append(s.split(".")[1]); in_loop = True; continue
                if in_loop and s.startswith("ATOM"):
                    parts = s.split()
                    if len(parts) < len(cols):
                        continue
                    col = {c: parts[i] for i, c in enumerate(cols)}
                    if col.get("label_atom_id") != "CA":
                        continue
                    rid = col.get("auth_seq_id") or col.get("label_seq_id")
                    if rid in seen:
                        continue
                    seen.add(rid); idx += 1
                    try:
                        out[idx] = (float(col["Cartn_x"]), float(col["Cartn_y"]), float(col["Cartn_z"]))
                    except (KeyError, ValueError):
                        pass
                elif in_loop and s and not s.startswith(("ATOM", "HETATM", "_atom_site.")):
                    if out:
                        break
    except OSError:
        return {}
    return out


def cross_seed_flex(cifs: List[str]) -> Optional[float]:
    """Mean over residues of the cross-seed MAX per-residue dRMSD (Å) — superposition-free, the
    same signal the deviation floor uses. Higher = wigglier ensemble. None if <2 folds."""
    if len(cifs) < 2:
        return None
    import numpy as np
    cas = [ca_xyz_from_cif(c) for c in cifs]
    common = sorted(set.intersection(*[set(c) for c in cas])) if cas else []
    if not common:
        return None
    arrs = [np.array([c[k] for k in common]) for c in cas]
    n = len(common)
    per_res_max = np.zeros(n)
    for a in range(len(arrs)):
        for b in range(a + 1, len(arrs)):
            A, B = arrs[a], arrs[b]
            DA = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
            DB = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=-1)
            sq = (DA - DB) ** 2
            np.fill_diagonal(sq, 0.0)
            drmsd = np.sqrt(sq.sum(axis=1) / max(1, n - 1))
            per_res_max = np.maximum(per_res_max, drmsd)
    return round(float(per_res_max.mean()), 3)


# ── folding ───────────────────────────────────────────────────────────────────────────
_bridge = BoltzBridge()


def fold(seq: str, *, templates: Optional[list] = None, seed: int = 0,
         label: str = "fold") -> dict:
    t0 = time.perf_counter()
    res = _bridge.predict([{"id": "A", "sequence": seq}], seed=seed, templates=templates,
                          label=label)
    res["_elapsed_s"] = round(time.perf_counter() - t0, 1)
    return res


def make_template(tmpl: dict, force: bool = False, threshold: float = 10.0) -> Optional[dict]:
    """Build a per-template entry for the monomer construct chain. chain_id ONLY (the SEARCH
    path) — do NOT set template_id: Boltz checks template_id against the template's chains as IT
    parses/renames them (NOT the PDB author chain), so a scalar author id ('A') raises
    'Template chain A is not one of the protein chains'. Boltz SWALLOWS that per-input ValueError
    and still exits 0 → the bridge only sees 'no predicted CIF'. The search path lets Boltz pick
    the template chain itself (works); explicit template_id is deferred to the multimer extension
    (which must map to Boltz's internal chain naming)."""
    path = download(tmpl["pdb"])
    if not path:
        return None
    entry = {"pdb": path, "chain_id": "A"}
    if force:
        entry["force"] = True
        entry["threshold"] = threshold
    return entry


# ── prescreen (flag hard targets) ──────────────────────────────────────────────────────
def prescreen(flex_seeds: int, names: Optional[List[str]] = None) -> None:
    print("=== PRESCREEN — unguided fold of HARD candidates (flag which are genuinely hard) ===")
    rows = []
    items = [(n, HARD_CANDIDATES[n]) for n in (names or HARD_CANDIDATES) if n in HARD_CANDIDATES]
    for name, tgt in items:
        pdb_path = download(tgt["pdb"])
        if not pdb_path:
            print(f"[{name}] could not download {tgt['pdb']} — skipping"); continue
        seq = chain_seq_from_pdb(pdb_path, tgt["chain"])
        print(f"[{name}] {tgt['pdb']}/{tgt['chain']} {len(seq)} aa — folding unguided…")
        seeds = list(range(max(1, flex_seeds)))
        cifs = []
        plddts = []
        for s in seeds:
            r = fold(seq, seed=s, label=f"{name}_u{s}")
            if not r.get("success"):
                print(f"  seed {s} FAILED: {r.get('error')}"); continue
            cifs.append(r["cif_path"]); plddts.append(r["mean_plddt"])
        mp = round(sum(plddts) / len(plddts), 2) if plddts else None
        flex = cross_seed_flex(cifs) if flex_seeds >= 2 else None
        hard = (mp is not None and mp < 80)
        rows.append((name, tgt["pdb"], len(seq), mp, flex, hard))
        print(f"  → mean pLDDT {mp}  cross-seed flex {flex}  {'HARD ✓' if hard else 'folds OK'}")
    print("\n--- PRESCREEN SUMMARY (HARD = unguided mean pLDDT < 80) ---")
    print(f"  {'target':10} {'pdb':6} {'len':>4} {'pLDDT':>7} {'flex':>6}  hard?")
    for name, pdb, n, mp, flex, hard in rows:
        print(f"  {name:10} {pdb:6} {n:>4} {str(mp):>7} {str(flex):>6}  {'YES' if hard else 'no'}")
    print("\nFlag the HARD ones to include in the titration.")


# ── titration ──────────────────────────────────────────────────────────────────────────
def titrate(target_names: List[str], do_hard: bool, flex_seeds: int, seed: int,
            out_tsv: str) -> None:
    rows = []
    for name in target_names:
        tgt = TARGETS.get(name) or HARD_CANDIDATES.get(name)
        if not tgt:
            print(f"[skip] unknown target {name}"); continue
        sp_path = download(tgt["pdb"])
        if not sp_path:
            print(f"[skip] {name}: ground truth {tgt['pdb']} unavailable"); continue
        tseq = chain_seq_from_pdb(sp_path, tgt["chain"])
        print(f"\n=== TARGET {name} ({tgt['pdb']}/{tgt['chain']}, {len(tseq)} aa) ===")

        # UNGUIDED baseline — folded ONCE per target, reused across all its templates.
        print(f"[{name}] UNGUIDED baseline fold (seed {seed})…")
        u_cifs = []
        u = fold(tseq, seed=seed, label=f"{name}_unguided")
        if not u.get("success"):
            print(f"[skip] {name}: unguided fold failed: {u.get('error')}"); continue
        u_cifs.append(u["cif_path"])
        for es in range(seed + 1, seed + flex_seeds):
            r = fold(tseq, seed=es, label=f"{name}_u{es}")
            if r.get("success"):
                u_cifs.append(r["cif_path"])
        u_tm = usalign_tm(u["cif_path"], sp_path)            # accuracy of U toward the truth
        u_acc = u_tm["tm2"] if u_tm else None
        u_flex = cross_seed_flex(u_cifs)
        print(f"[{name}] unguided: mean pLDDT {u['mean_plddt']}  TM(U,S_P)={u_acc}  flex={u_flex}")

        for tmpl in tgt["templates"]:
            tpath = download(tmpl["pdb"])
            if not tpath:
                print(f"  [skip template {tmpl['pdb']}] unavailable"); continue
            tmpl_seq = chain_seq_from_pdb(tpath, tmpl.get("chain", "A"))
            seqid = nw_seq_id(tseq, tmpl_seq)                # divergence axis 1 (sequence)
            t_struct = usalign_tm(tpath, sp_path)            # divergence axis 2 (structural-TM-to-truth)
            t_tm_to_sp = t_struct["tm2"] if t_struct else None
            print(f"  -- template {tmpl['pdb']} [{tmpl['tag']}]  seq-id={seqid}  "
                  f"structTM(tmpl,S_P)={t_tm_to_sp}")

            for mode, force in ([("soft", False), ("hard", True)] if do_hard else [("soft", False)]):
                tentry = make_template(tmpl, force=force)
                g_cifs = []
                g = fold(tseq, templates=[tentry], seed=seed, label=f"{name}_{tmpl['pdb']}_{mode}")
                if not g.get("success"):
                    print(f"     [{mode}] guided fold FAILED: {g.get('error')}"); continue
                g_cifs.append(g["cif_path"])
                for es in range(seed + 1, seed + flex_seeds):
                    r = fold(tseq, templates=[tentry], seed=es, label=f"{name}_{mode}_{es}")
                    if r.get("success"):
                        g_cifs.append(r["cif_path"])
                g_tm = usalign_tm(g["cif_path"], sp_path)      # accuracy of G toward the truth
                g_acc = g_tm["tm2"] if g_tm else None
                adopt = usalign_tm(g["cif_path"], tpath)       # adoption: G vs the template
                g_adopt = adopt["tm2"] if adopt else None
                g_flex = cross_seed_flex(g_cifs)
                d_acc = (round(g_acc - u_acc, 4) if (g_acc is not None and u_acc is not None) else None)
                d_plddt = round(g["mean_plddt"] - u["mean_plddt"], 2)
                d_flex = (round(u_flex - g_flex, 3) if (u_flex is not None and g_flex is not None) else None)
                print(f"     [{mode}] pLDDT {g['mean_plddt']} (Δ{d_plddt:+}) | TM(G,S_P)={g_acc} "
                      f"ΔAcc={d_acc} | adopt TM(G,tmpl)={g_adopt} | Δflex={d_flex}")
                rows.append({
                    "target": name, "target_pdb": tgt["pdb"], "len": len(tseq),
                    "template": tmpl["pdb"], "tag": tmpl["tag"], "mode": mode,
                    "seq_id": seqid, "structTM_tmpl_to_SP": t_tm_to_sp,
                    "U_plddt": u["mean_plddt"], "G_plddt": g["mean_plddt"], "d_plddt": d_plddt,
                    "TM_U_SP": u_acc, "TM_G_SP": g_acc, "dAccuracy": d_acc,
                    "adopt_TM_G_tmpl": g_adopt, "U_flex": u_flex, "G_flex": g_flex, "d_flex": d_flex,
                })

    # write TSV + a compact calibration table
    if rows:
        with open(out_tsv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(rows)
        print(f"\n[out] wrote {len(rows)} rows → {out_tsv}")
    print("\n=== CALIBRATION TABLE (ΔAccuracy vs divergence) ===")
    print(f"  {'target':9} {'tmpl':6} {'mode':4} {'seqid':>6} {'sTM→SP':>7} "
          f"{'TM_U':>6} {'TM_G':>6} {'ΔAcc':>7} {'adopt':>6} {'Δflex':>6}")
    for r in rows:
        print(f"  {r['target']:9} {r['template']:6} {r['mode']:4} {str(r['seq_id']):>6} "
              f"{str(r['structTM_tmpl_to_SP']):>7} {str(r['TM_U_SP']):>6} {str(r['TM_G_SP']):>6} "
              f"{str(r['dAccuracy']):>7} {str(r['adopt_TM_G_tmpl']):>6} {str(r['d_flex']):>6}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prescreen", action="store_true")
    ap.add_argument("--titrate", action="store_true")
    ap.add_argument("--target", action="append", help="restrict to these registry targets")
    ap.add_argument("--hard", action="store_true", help="also run guided-HARD (force+threshold)")
    ap.add_argument("--flex-seeds", type=int, default=0, help="cross-seed flex over N seeds (0=off)")
    ap.add_argument("--seed", type=int, default=int(getattr(_cfg, "BOLTZ_SEED", 0)))
    ap.add_argument("--out", default="cache/template_calibration.tsv")
    args = ap.parse_args()

    if not _bridge.is_available():
        print("[abort] Boltz env unavailable — the calibration folds need it."); sys.exit(2)

    if args.prescreen:
        prescreen(args.flex_seeds, args.target)
        return
    if args.titrate:
        names = args.target or list(TARGETS.keys())
        titrate(names, args.hard, args.flex_seeds, args.seed, args.out)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
