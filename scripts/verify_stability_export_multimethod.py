"""
Verify — the session export keeps EVERY ddG method, even after a higher-quality run (GPU-free, no
ChimeraX). Drives the user's exact flow through the REAL functions:

  fast ThermoMPNN/RaSP/DynaMut2 scan  ->  stability_summary
  THEN a deep Rosetta scan            ->  merge_stability  (augment, not overwrite)
  THEN                                ->  export_session    (real xlsx + csv)

and reads back the real `stability_ddg.csv`, asserting Rosetta AND ThermoMPNN AND RaSP AND DynaMut2
all survive as separate columns for the same mutation. Candidate dicts are shaped EXACTLY like
`mutation_scanner` emits (thermompnn_ddg/rasp_ddg/dynamut2_ddg always present; `ddg` = the Rosetta
physics axis, None on the fast tier).

Run: venv/Scripts/python.exe scripts/verify_stability_export_multimethod.py
"""
import os, sys, csv, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from variant_model import Mutation, stability_summary, merge_stability
from session_export import export_session

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    muts = [Mutation(40, "S", "W"), Mutation(72, "G", "C")]

    # FAST tier candidates (mutation_scanner shape): ML/proxy/dynamics axes, no Rosetta `ddg`.
    fast = [
        {"resnum": 40, "from_aa": "S", "to_aa": "W", "ddg": None, "thermompnn_ddg": -0.62,
         "rasp_ddg": 0.18, "dynamut2_ddg": None, "combined_score": -0.31, "recommendation": "favorable"},
        {"resnum": 72, "from_aa": "G", "to_aa": "C", "ddg": None, "thermompnn_ddg": 0.44,
         "rasp_ddg": 0.30, "dynamut2_ddg": None, "combined_score": 0.12, "recommendation": "neutral"},
    ]
    # DEEP tier candidates: Rosetta physics axis filled; this run did NOT recompute ThermoMPNN.
    deep = [
        {"resnum": 40, "from_aa": "S", "to_aa": "W", "ddg": -1.10, "thermompnn_ddg": None,
         "rasp_ddg": None, "dynamut2_ddg": 0.7, "combined_score": -0.55, "recommendation": "favorable"},
        {"resnum": 72, "from_aa": "G", "to_aa": "C", "ddg": 0.95, "thermompnn_ddg": None,
         "rasp_ddg": None, "dynamut2_ddg": 0.2, "combined_score": 0.20, "recommendation": "neutral"},
    ]

    prev = stability_summary(fast, muts)
    check("fast run reports ThermoMPNN (pre-Rosetta)",
          prev["rows"][0]["thermompnn_ddg"] == -0.62 and prev["rows"][0]["rosetta_ddg"] is None,
          f"tier={prev['tier']}")

    merged = merge_stability(prev, stability_summary(deep, muts))
    r0 = merged["rows"][0]
    check("after the deep Rosetta run, BOTH axes survive on the same mutation",
          r0["rosetta_ddg"] == -1.10 and r0["thermompnn_ddg"] == -0.62,
          f"rosetta={r0['rosetta_ddg']} thermompnn={r0['thermompnn_ddg']} "
          f"rasp={r0['rasp_ddg']} dynamut2={r0['dynamut2_ddg']} best={r0['ddg']}({r0['ddg_source']})")

    # REAL export of a session whose variant carries the merged (multi-method) stability slot.
    sessions = {"dn-1": {"model_id": "1", "source": "sequence", "chains": {"c1": {
        "rep_chain": "A", "members": [["1", "A"]],
        "variants": [{"id": "V1",
                      "mutations": [{"resnum": 40, "from_aa": "S", "to_aa": "W", "source": "manual"},
                                    {"resnum": 72, "from_aa": "G", "to_aa": "C", "source": "manual"}],
                      "results": {"fold": None, "solubility": None, "stability": merged}}]}}}}

    with tempfile.TemporaryDirectory() as td:
        export_session(sessions, Path(td))
        csv_path = Path(td) / "csv" / "stability_ddg.csv"
        rows = list(csv.reader(open(csv_path, newline="", encoding="utf-8")))
        hdr, data = rows[0], rows[1:]
        for col in ("rosetta_ddg", "thermompnn_ddg", "rasp_ddg", "dynamut2_ddg"):
            check(f"exported stability_ddg.csv has a '{col}' column", col in hdr)
        by_resnum = {d[hdr.index("resnum")]: dict(zip(hdr, d)) for d in data}
        row40 = by_resnum.get("40", {})
        check("exported row 40 carries BOTH Rosetta and ThermoMPNN (not one overwriting the other)",
              row40.get("rosetta_ddg") == "-1.1" and row40.get("thermompnn_ddg") == "-0.62",
              f"rosetta={row40.get('rosetta_ddg')} thermompnn={row40.get('thermompnn_ddg')} "
              f"rasp={row40.get('rasp_ddg')} dynamut2={row40.get('dynamut2_ddg')} "
              f"ddg(best)={row40.get('ddg')} src={row40.get('ddg_source')}")
        print("\n  exported Stability ddG rows:")
        cols = ["variant", "resnum", "from_aa", "to_aa", "ddg", "ddg_source",
                "rosetta_ddg", "thermompnn_ddg", "rasp_ddg", "dynamut2_ddg"]
        print("    " + " | ".join(f"{c}" for c in cols))
        for d in data:
            row = dict(zip(hdr, d))
            print("    " + " | ".join(f"{row.get(c, '')}" for c in cols))

    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
