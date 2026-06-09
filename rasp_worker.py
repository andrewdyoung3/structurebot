"""
rasp_worker.py — RaSP per-mutation ddG worker (runs in the WSL2 rasp_env venv).

Invoked by rasp_bridge.py via:
    wsl -d <distro> -- <rasp_env python> <this, over /mnt/c> \
        --repo <RaSP_repo> --pdb <pdb> --chain <C> --out_csv <csv>

Pipeline (RaSP's own, with the modern-stack port applied to the repo copy):
    clean_pdb.py (reduce + pdbfixer + openmm)  →  extract_environments.py
    →  cavity (latent) + downstream ensemble (median) → inverse-Fermi → ddG.

Output CSV columns: chain,resnum,wt,mt,ddg   (ddg in kcal/mol, RaSP/Rosetta sign:
positive = destabilising).  resnum/wt are RaSP's view of the CLEANED structure;
the bridge re-anchors them to the ORIGINAL author resnums via the shared
WT-anchored alignment (so pdbfixer renumbering / insertion-stripping can never
mis-attribute — divergence becomes a hard error there).

ERROR-FIRST: any failure exits non-zero with a short stderr; the bridge treats a
non-zero/absent CSV as not_computed (the fast tier renormalises without RaSP).
"""
import argparse, glob, os, subprocess, sys, tempfile, shutil
import numpy as np

_ALPHA, _BETA, _EPS = 3.0, 0.4, 1e-12
def _inv_fermi(x):
    if x == 1.0: return 40.0
    if x == 0.0: return -40.0
    if 0.0 < x < 1.0: return (_ALPHA * _BETA - np.log(-1.0 + 1.0 / x + _EPS)) / _BETA
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--chain", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--reduce", default=None)
    a = ap.parse_args()

    src = os.path.join(a.repo, "src")
    parser_dir = os.path.join(src, "pdb_parser_scripts")
    reduce_exe = a.reduce or os.path.join(parser_dir, "reduce", "reduce_src", "reduce")
    sys.path.insert(0, src)

    work = tempfile.mkdtemp(prefix="rasp_")
    try:
        cleaned, parsed = os.path.join(work, "cleaned"), os.path.join(work, "parsed")
        os.makedirs(cleaned); os.makedirs(parsed)

        # 1. clean (reduce + pdbfixer + openmm) — single-model, single-chain input
        r = subprocess.run(
            [sys.executable, "clean_pdb.py", "--pdb_file_in", a.pdb,
             "--out_dir", cleaned + "/", "--reduce_exe", reduce_exe],
            cwd=parser_dir, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write("RaSP clean failed: " + r.stderr[-500:]); return 2
        clean_pdbs = glob.glob(os.path.join(cleaned, "*_clean.pdb"))
        if not clean_pdbs:
            sys.stderr.write("RaSP clean produced no output"); return 2

        # 2. extract residue environments (CNN grid input)
        r = subprocess.run(
            [sys.executable, "extract_environments.py", "--pdb_in", clean_pdbs[0],
             "--out_dir", parsed], cwd=parser_dir, capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write("RaSP extract failed: " + r.stderr[-500:]); return 2

        # 3. inference (reuse rasp_model; helpers/run_pipeline are unimportable here)
        import torch, pandas as pd
        from torch.utils.data import DataLoader
        from Bio.PDB.Polypeptide import index_to_one, one_to_index
        import rasp_model as rm

        npz = sorted(glob.glob(os.path.join(parsed, "*.npz")))
        if not npz:
            sys.stderr.write("RaSP: no parsed environments"); return 2
        dataset = rm.ResidueEnvironmentsDataset(npz, transformer=None)
        nlf = -np.log(np.load(os.path.join(
            a.repo, "data/train/cavity/pdb_frequencies.npz"))["frequencies"])

        resenv_map, rows = {}, []
        for re_ in dataset:
            if re_.chain_id != a.chain:
                continue
            wt = index_to_one(re_.restype_index)
            resenv_map[f"{re_.pdb_id}{re_.chain_id}_{re_.pdb_residue_number}{wt}"] = re_
            for mt in "ACDEFGHIKLMNPQRSTVWY":
                if mt != wt:
                    rows.append({"pdbid": re_.pdb_id, "chainid": re_.chain_id,
                                 "variant": f"{wt}{re_.pdb_residue_number}{mt}"})
        if not rows:
            sys.stderr.write(f"RaSP: no residues for chain {a.chain}"); return 2
        df = pd.DataFrame(rows)
        df["wt_idx"] = df.variant.apply(lambda v: one_to_index(v[0]))
        df["mt_idx"] = df.variant.apply(lambda v: one_to_index(v[-1]))
        df["wt_nlf"] = df.wt_idx.apply(lambda i: nlf[i])
        df["mt_nlf"] = df.mt_idx.apply(lambda i: nlf[i])
        df["resenv"] = df.apply(
            lambda r: resenv_map.get(f"{r.pdbid}{r.chainid}_{r.variant[1:-1]}{r.variant[0]}"),
            axis=1)
        df = df[df.resenv.notnull()].reset_index(drop=True)

        cav = rm.CavityModel(get_latent=True)
        cav.load_state_dict(torch.load(
            os.path.join(a.repo, "pretrained_models/cavity/model.pt"),
            map_location="cpu", weights_only=False)); cav.eval()
        ds_nets = []
        for p in sorted(glob.glob(os.path.join(a.repo, "pretrained_models/ds/ds_model_*/model.pt"))):
            d = rm.DownstreamModel()
            d.load_state_dict(torch.load(p, map_location="cpu", weights_only=False))
            d.eval(); ds_nets.append(d)

        t = rm.DDGToTensor("pred", "cpu")
        loader = DataLoader(rm.DDGDataset(df, transformer=t), batch_size=100,
                            shuffle=False, drop_last=False, collate_fn=t.collate_multi)
        out = []
        with torch.no_grad():
            for _pid, ch_b, var_b, xcav, xds in loader:
                cp = cav(xcav)
                ens = torch.cat([d(torch.cat((cp, xds), 1)) for d in ds_nets], 1)
                fermi = torch.median(ens, 1, keepdim=True)[0].cpu().numpy().ravel()
                for ch, v, fm in zip(ch_b, var_b, fermi):
                    v = v[0] if isinstance(v, (list, tuple)) else v
                    ch = ch[0] if isinstance(ch, (list, tuple)) else ch
                    out.append((ch, int(v[1:-1]), v[0], v[-1], round(_inv_fermi(float(fm)), 4)))

        res = pd.DataFrame(out, columns=["chain", "resnum", "wt", "mt", "ddg"])
        res.to_csv(a.out_csv, index=False)
        sys.stderr.write(f"RaSP: {len(res)} variant ddGs over {res.resnum.nunique()} residues\n")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
