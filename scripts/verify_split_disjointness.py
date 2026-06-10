"""
scripts/verify_split_disjointness.py — Task A1 reproducible verification.

Checks the calibration sets (S669, Ssym) for IDENTITY overlap with the AUTHORITATIVE
ThermoMPNN training split — the actual training CSVs from the ThermoMPNN GitHub repo,
NOT the README prose. Fetches once into cache/thermompnn_splits/ (gitignored), then
reports, per calibration set, which pdbids appear in ThermoMPNN's training data.

CRITICAL NUANCE: the DEPLOYED model is `thermoMPNN_default.pt` = the MEGASCALE-trained
model (config.THERMOMPNN_MODEL + the README + the negated-Megascale sign note in
thermompnn_bridge.py). So the relevant training set is MEGASCALE. FireProt overlaps
are reported separately for completeness but pertain to a DIFFERENT (un-deployed) model.

Run:  python scripts/verify_split_disjointness.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path
from urllib.request import urlopen

_ROOT = Path(__file__).resolve().parent.parent
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_CACHE = _ROOT / "cache" / "thermompnn_splits"
_RASP = _ROOT / "RaSP_repo" / "data" / "test"
_BASE = "https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/main"
_TRAIN_FILES = {
    "mega_train": "data_all/training/mega_train.csv",
    "mega_val": "data_all/training/mega_val.csv",
    "fireprot_train": "data_all/training/fireprot_train.csv",
    "fireprot_val": "data_all/training/fireprot_val.csv",
}


def _fetch(rel: str, dest: Path) -> bool:
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(f"{_BASE}/{rel}", timeout=90) as r:
            dest.write_bytes(r.read())
        return True
    except Exception as exc:
        print(f"  FETCH FAILED {rel}: {exc}")
        return False


def _pdb_tokens(s: str) -> set:
    out = set()
    for tok in re.split(r"[|,;\s]+", str(s).strip()):
        tok = tok.strip().upper()
        if re.fullmatch(r"[0-9][A-Z0-9]{3}", tok):
            out.add(tok)
    return out


def _training_ids():
    """Return (megascale_pdbids, fireprot_pdbids) from the authoritative split."""
    mega, fire = set(), set()
    for key, rel in _TRAIN_FILES.items():
        dest = _CACHE / Path(rel).name
        if not _fetch(rel, dest):
            continue
        for row in csv.DictReader(open(dest, encoding="utf-8")):
            if key.startswith("mega"):
                mega |= _pdb_tokens((row.get("WT_name") or "").replace(".pdb", ""))
            else:
                fire |= _pdb_tokens(row.get("pdb_id", ""))
                fire |= _pdb_tokens(row.get("pdb_id_corrected", ""))
    return mega, fire


def _set_pdbids(name: str) -> set:
    p = _RASP / name / "ddG_experimental" / "ddg.csv"
    return {row["pdbid"].strip().upper()
            for row in csv.DictReader(open(p, encoding="utf-8"))}


def main() -> int:
    mega, fire = _training_ids()
    print(f"ThermoMPNN training ids: megascale={len(mega)}  fireprot={len(fire)}")
    print("DEPLOYED model = megascale (thermoMPNN_default.pt) → megascale is authoritative\n")
    contaminated = {}
    for name in ("S669", "Ssym_dir", "Ssym_inv"):
        s = _set_pdbids(name)
        ovm = sorted(s & mega)
        ovf = sorted(s & fire)
        contaminated[name] = ovm
        print(f"{name} ({len(s)} proteins):")
        print(f"   vs DEPLOYED megascale training : {ovm if ovm else 'CLEAN'}")
        print(f"   vs fireprot (un-deployed model): {ovf if ovf else 'clean'}")
    bad = sum(len(v) for v in contaminated.values())
    print(f"\nVERDICT: {'CONTAMINATION' if bad else 'CLEAN'} vs deployed megascale model "
          f"({bad} protein(s)).")
    if bad:
        print("→ exclude these from ThermoMPNN vs-experiment scoring "
              "(they remain valid for Rosetta/RaSP/DynaMut2).")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
