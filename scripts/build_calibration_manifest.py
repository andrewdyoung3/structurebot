"""
scripts/build_calibration_manifest.py — Task 1 artifact generator (DRAFT).

Builds the per-protein PROVENANCE AUDIT + a machine-readable calibration MANIFEST
from the LOCAL RaSP benchmark sets (RaSP_repo/data/test/*), applying the per-voter
training-disjointness rules below.  PURE INVENTORY + TAGGING — it locks NOTHING.
The final selection is an ATTENDED decision (Task 1 STOP-FOR-REVIEW).

Outputs (committed under scripts/, reviewable):
  scripts/calibration_manifest.draft.json   — per-(set,pdbid,chain) entries + provenance
  scripts/calibration_provenance_audit.md   — the protein × voter table + summary

PROVENANCE RULES (DRAFT — documented, not authoritative; confirm at review):
  ThermoMPNN  trained on Megascale (Tsuboyama 2023).  Per the ThermoMPNN README,
              S669 + SSYM are its HELD-OUT TEST sets (→ clean); Protein G (1PGA) and
              Rocklin's designed mini-proteins are the Megascale lineage (→ training).
              AUTHORITATIVE confirmation = ThermoMPNN dataset_splits/*.pkl (not local).
  DynaMut2    trained on the S2648 / VariBench family.  Ssym is S2648-derived (→ training);
              S669 was constructed (Pancotti 2022) to be disjoint from S2648 (→ clean).
  RaSP        Rosetta-supervised surrogate → RaSP-vs-Rosetta is ALWAYS circular; RaSP-vs-
              EXPERIMENT is valid.  Tagged 'circular_vs_rosetta' on every protein.
  Rosetta     physics, no training → leakage-free anchor → 'clean' everywhere.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

_ROOT = Path(__file__).resolve().parent.parent
_TEST = _ROOT / "RaSP_repo" / "data" / "test"

# Candidate sets that ship a LOCAL experimental-ddG CSV + structures.
# (ProTherm/Human ship no usable experimental CSV locally; VAMP/Xstal/Rosetta_10/
#  Speedtest carry Rosetta labels only — all excluded from the experimental audit.)
_CANDIDATE_SETS = ["S669", "Ssym_dir", "Ssym_inv", "Rocklin", "Protein_G"]

# DRAFT provenance + proposal per set.  role/include are the PROPOSED draft only.
_SET_META = {
    "S669": dict(
        thermompnn="clean", dynamut2="clean", rasp="circular_vs_rosetta", rosetta="clean",
        include=True, role="diversity_core",
        note="94 proteins; ThermoMPNN+DynaMut2 held-out; the diversity backbone."),
    "Ssym_dir": dict(
        thermompnn="clean", dynamut2="training", rasp="circular_vs_rosetta", rosetta="clean",
        include=True, role="antisymmetry_fwd",
        note="S2648-family → DynaMut2 TRAINING (score DynaMut2 with caution/flag); "
             "fwd half of the anti-symmetry pair (Task 4)."),
    "Ssym_inv": dict(
        thermompnn="clean", dynamut2="training", rasp="circular_vs_rosetta", rosetta="clean",
        include=True, role="antisymmetry_rev",
        note="Reverse mutant structures; pairs with Ssym_dir for anti-symmetry (Task 4)."),
    "Rocklin": dict(
        thermompnn="training", dynamut2="unknown", rasp="circular_vs_rosetta", rosetta="clean",
        include=False, role="thermompnn_contaminated_designed",
        note="164 designed mini-proteins = Megascale lineage → ThermoMPNN TRAINING; "
             "164k rows, low protein-diversity value → EXCLUDE from proposal (available)."),
    "Protein_G": dict(
        thermompnn="training", dynamut2="unknown", rasp="circular_vs_rosetta", rosetta="clean",
        include=False, role="leakage_demo_already_used",
        note="1PGA is in Megascale → ThermoMPNN TRAINING; already run (the dry-run set). "
             "Keep only as the documented leakage demonstrator."),
}


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _set_pdbids(set_name: str) -> Dict[str, List[str]]:
    """Return {pdbid: [variants]} from a set's experimental ddg.csv."""
    csv_path = _TEST / set_name / "ddG_experimental" / "ddg.csv"
    out: Dict[str, List[str]] = {}
    if not csv_path.is_file():
        return out
    for row in csv.DictReader(open(csv_path)):
        pid = (row.get("pdbid") or "").strip()
        var = (row.get("variant") or "").strip()
        if pid:
            out.setdefault(pid, []).append(var)
    return out


def _struct_present(set_name: str, pdbid: str) -> bool:
    sd = _TEST / set_name / "structure" / "raw"
    return bool(list(sd.glob(f"{pdbid}*.pdb")))


def build() -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for set_name in _CANDIDATE_SETS:
        meta = _SET_META[set_name]
        pdbids = _set_pdbids(set_name)
        for pid in sorted(pdbids):
            entries.append({
                "set": set_name,
                "pdbid": pid,
                "chain": "A",  # RaSP sets are chain A; harness re-anchors to author resnum
                "n_mutations": len(pdbids[pid]),
                "exp_csv": _rel(_TEST / set_name / "ddG_experimental" / "ddg.csv"),
                "rosetta_ref_csv": _rel(_TEST / set_name / "ddG_Rosetta" / "ddg.csv"),
                "struct_dir": _rel(_TEST / set_name / "structure" / "raw"),
                "structure_present": _struct_present(set_name, pid),
                "provenance": {
                    "thermompnn": meta["thermompnn"],
                    "dynamut2": meta["dynamut2"],
                    "rasp": meta["rasp"],
                    "rosetta": meta["rosetta"],
                },
                "proposed_include": meta["include"],
                "role": meta["role"],
            })
    return {
        "manifest_version": 1,
        "status": "DRAFT — NOT LOCKED; final selection is an attended decision (Task 1 STOP).",
        "provenance_rules": "see scripts/build_calibration_manifest.py docstring + audit md",
        "set_notes": {k: v["note"] for k, v in _SET_META.items()},
        "entries": entries,
    }


def write_audit_md(manifest: Dict[str, Any], path: Path) -> None:
    e = manifest["entries"]
    by_set: Dict[str, List[Dict[str, Any]]] = {}
    for row in e:
        by_set.setdefault(row["set"], []).append(row)
    L: List[str] = []
    L.append("# Calibration-set provenance audit (Task 1 — DRAFT, NOT LOCKED)\n")
    L.append("Generated by `scripts/build_calibration_manifest.py` from LOCAL "
             "`RaSP_repo/data/test/*`. Final selection is an **attended decision** "
             "(Task 1 STOP-FOR-REVIEW). Tags: `clean` = training-disjoint, "
             "`training` = in that voter's training set (leakage), "
             "`circular_vs_rosetta` = RaSP-vs-Rosetta circular (RaSP-vs-experiment OK), "
             "`unknown` = not confirmed locally.\n")
    # set-level table
    L.append("## Set-level audit (protein groups × voter)\n")
    L.append("| Set | Proteins | Muts | Struct OK | ThermoMPNN | DynaMut2 | RaSP | Rosetta | Proposed | Role |")
    L.append("|-----|----------|------|-----------|------------|----------|------|---------|----------|------|")
    for s, rows in by_set.items():
        nmut = sum(r["n_mutations"] for r in rows)
        sok = sum(1 for r in rows if r["structure_present"])
        p = rows[0]["provenance"]
        L.append(f"| {s} | {len(rows)} | {nmut} | {sok}/{len(rows)} | "
                 f"{p['thermompnn']} | {p['dynamut2']} | {p['rasp']} | {p['rosetta']} | "
                 f"{'INCLUDE' if rows[0]['proposed_include'] else 'exclude'} | {rows[0]['role']} |")
    L.append("")
    # per-voter held-out scoring subset (draft)
    L.append("## Draft per-voter held-out scoring subsets\n")
    L.append("- **ThermoMPNN** → score on **S669 + Ssym** (its held-out test sets); "
             "EXCLUDE Rocklin + Protein_G (Megascale training).")
    L.append("- **DynaMut2** → score on **S669 only** (Ssym = S2648 training family); "
             "small fixed subset (remote API).")
    L.append("- **RaSP** → score vs **EXPERIMENT** on all; NEVER report RaSP-vs-Rosetta "
             "agreement as corroboration (circular).")
    L.append("- **Rosetta** → score on **all** (physics, leakage-free anchor).")
    L.append("")
    L.append("## Proposed draft set (NOT locked)\n")
    inc = sorted({r["set"] for r in e if r["proposed_include"]})
    exc = sorted({r["set"] for r in e if not r["proposed_include"]})
    L.append(f"- **INCLUDE:** {', '.join(inc)}")
    L.append(f"- **EXCLUDE (available, flagged):** {', '.join(exc)}")
    L.append("")
    for s in by_set:
        L.append(f"  - `{s}`: {manifest['set_notes'][s]}")
    L.append("")
    L.append("## Awaiting attended decision\n")
    L.append("Confirm/adjust: (1) lock the protein set; (2) confirm ThermoMPNN disjointness "
             "against `dataset_splits/*.pkl` (authoritative, not local); (3) decide whether to "
             "score DynaMut2 on Ssym at all (training overlap); (4) decide Rocklin subset (if any).")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    m = build()
    out_json = _ROOT / "scripts" / "calibration_manifest.draft.json"
    out_md = _ROOT / "scripts" / "calibration_provenance_audit.md"
    out_json.write_text(json.dumps(m, indent=2), encoding="utf-8")
    write_audit_md(m, out_md)
    n_inc = sum(1 for e in m["entries"] if e["proposed_include"])
    print(f"wrote {out_json.name}: {len(m['entries'])} protein entries "
          f"({n_inc} proposed-include)")
    print(f"wrote {out_md.name}")
