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

# AUTHORITATIVE per-protein provenance overrides from Task A1/A2 verification
# (scripts/verify_split_disjointness.py against ThermoMPNN's training CSVs).
# These S669 proteins ALSO appear in the DEPLOYED megascale ThermoMPNN training set
# (1O6X/2HBB/2WQG verified as real megascale entries) → ThermoMPNN-CONTAMINATED,
# despite S669 being ThermoMPNN's nominal held-out test.  Excluded from ThermoMPNN
# vs-experiment scoring; still valid for Rosetta/RaSP/DynaMut2.
_THERMOMPNN_TRAIN_OVERRIDE = {"1O6X", "2HBB", "2WQG"}

# DRAFT provenance + proposal per set.  role/include are the PROPOSED draft only.
# Per-protein overrides above take precedence over these set-level defaults.
_SET_META = {
    "S669": dict(
        thermompnn="clean", dynamut2="unknown", rasp="circular_vs_rosetta", rosetta="clean",
        include=True, role="diversity_core",
        note="94 proteins; ThermoMPNN held-out EXCEPT the 3 megascale-overlap proteins "
             "(A1: 1O6X/2HBB/2WQG → training). DynaMut2 overlap UNKNOWN (A2: authoritative "
             "DynaMut2 training list not obtainable) → DynaMut2 S669 scores PROVISIONAL. "
             "The diversity backbone."),
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
                    "thermompnn": ("training" if pid.upper() in _THERMOMPNN_TRAIN_OVERRIDE
                                   else meta["thermompnn"]),
                    "dynamut2": meta["dynamut2"],
                    "rasp": meta["rasp"],
                    "rosetta": meta["rosetta"],
                },
                "proposed_include": meta["include"],
                "role": meta["role"],
            })
    thermompnn_contam = sorted({e["pdbid"] for e in entries
                                if e["provenance"]["thermompnn"] == "training"
                                and e["set"] == "S669"})
    contam_muts = sum(e["n_mutations"] for e in entries
                      if e["set"] == "S669" and e["pdbid"] in thermompnn_contam)
    s669_muts = sum(e["n_mutations"] for e in entries if e["set"] == "S669")
    # authoritative megascale training-row counts for the 3 contaminated proteins
    # (from scripts/verify_split_disjointness.py against ThermoMPNN's mega_train/val CSVs)
    contam_train_rows = {"1O6X": 1361, "2HBB": 786, "2WQG": 658}
    return {
        "manifest_version": 1,
        "status": "LOCKED — Task A resolution approved (2026-06-10). S669 + Ssym; "
                  "Rocklin + Protein_G excluded. Per-voter scoring scopes below are "
                  "authoritative.",
        "verification": {
            "A1_thermompnn": {
                "method": "identity overlap vs the AUTHORITATIVE ThermoMPNN training CSVs "
                          "(data_all/training/mega_*.csv + fireprot_*.csv) via "
                          "scripts/verify_split_disjointness.py; deployed model = megascale "
                          "(thermoMPNN_default.pt)",
                "S669_contaminated_proteins": thermompnn_contam,
                "S669_contaminated_mutations": contam_muts,
                "S669_thermompnn_clean_mutations": s669_muts - contam_muts,
                "megascale_training_rows_per_protein": contam_train_rows,
                "Ssym_contaminated": [],
                "outcome": "CONTAMINATION CONFIRMED — 3 S669 proteins also in the deployed "
                           "megascale training set (verified real entries: 1O6X 1361 rows, "
                           "2HBB 786, 2WQG 658); Ssym clean. RESOLUTION (approved): these 3 "
                           "(95 muts) are EXCLUDED FROM ThermoMPNN vs-experiment scoring ONLY "
                           "— kept in the dataset, valid for Rosetta/RaSP/DynaMut2. Not deleted.",
            },
            "A2_dynamut2": {
                "method": "authoritative DynaMut2 training list not obtainable "
                          "(reachable repos ship other tools' datasets, not DynaMut2's)",
                "S669_overlap": "unknown",
                "outcome": "UNKNOWN-FLAGGED — DynaMut2 S669 vs-experiment scores are "
                           "PROVISIONAL (flagged, NOT excluded). Ssym = DynaMut2 training → "
                           "Ssym EXCLUDED from DynaMut2 vs-experiment (still used for "
                           "reference-free anti-symmetry).",
            },
        },
        "scoring_scopes": {
            "_note": "Per-voter VS-EXPERIMENT confidence scopes (what each voter is scored "
                     "on). Anti-symmetry (Ssym fwd vs rev) is reference-free and separate.",
            "thermompnn": {
                "vs_experiment": "S669 MINUS {1O6X,2HBB,2WQG} (574 muts) + Ssym",
                "excluded": "1O6X/2HBB/2WQG (megascale training); Rocklin + Protein_G",
                "status": "clean",
            },
            "rosetta": {
                "vs_experiment": "FULL set (S669 + Ssym)",
                "note": "leakage-free physics anchor (no training)",
                "status": "clean",
            },
            "rasp": {
                "vs_experiment": "FULL set (S669 + Ssym) — generalization (never saw "
                                 "experiment)",
                "note": "RaSP-vs-ROSETTA stays flagged CIRCULAR (RaSP is a Rosetta "
                        "surrogate); never report RaSP-vs-Rosetta agreement as corroboration",
                "status": "clean_vs_experiment / circular_vs_rosetta",
            },
            "dynamut2": {
                "vs_experiment": "S669 only, PROVISIONAL (disjointness unknown); Ssym "
                                 "EXCLUDED (training)",
                "anti_symmetry": "Ssym fwd vs rev (reference-free) — INCLUDED for all voters",
                "status": "provisional",
            },
            "anti_symmetry": {
                "scope": "Ssym fwd (Ssym_dir) vs rev (Ssym_inv), reference-free — ALL voters "
                         "incl. DynaMut2 (no experiment leakage in an anti-symmetry check)",
            },
        },
        "provenance_rules": "see scripts/build_calibration_manifest.py docstring + audit md",
        "set_notes": {k: v["note"] for k, v in _SET_META.items()},
        "entries": entries,
    }


def write_audit_md(manifest: Dict[str, Any], path: Path) -> None:
    e = manifest["entries"]
    by_set: Dict[str, List[Dict[str, Any]]] = {}
    for row in e:
        by_set.setdefault(row["set"], []).append(row)
    v = manifest.get("verification", {})
    L: List[str] = []
    L.append("# Calibration-set provenance audit (Task A — VERIFIED; **LOCKED** 2026-06-10)\n")
    L.append("Generated by `scripts/build_calibration_manifest.py` from LOCAL "
             "`RaSP_repo/data/test/*`, with provenance from the **authoritative** "
             "Task A1/A2 verification (`scripts/verify_split_disjointness.py`), not the "
             "README. Tags: `clean` = training-disjoint, `training` = in that voter's "
             "training set (leakage), `circular_vs_rosetta` = RaSP-vs-Rosetta circular "
             "(RaSP-vs-experiment OK), `unknown` = not confirmable.\n")
    L.append(f"**STATUS: {manifest['status']}**\n")
    a1 = v.get("A1_thermompnn", {}); a2 = v.get("A2_dynamut2", {})
    L.append("## Authoritative-split verification result (traceable provenance)\n")
    rows_pp = a1.get("megascale_training_rows_per_protein", {})
    rows_str = ", ".join(f"{k} {rows_pp[k]} rows" for k in sorted(rows_pp))
    L.append(f"- **A1 (ThermoMPNN, deployed = megascale `thermoMPNN_default.pt`):** "
             f"identity overlap vs ThermoMPNN's training CSVs (`data_all/training/"
             f"mega_*.csv`). **CONTAMINATION CONFIRMED** — S669 proteins also present in the "
             f"deployed megascale training set: `{a1.get('S669_contaminated_proteins')}` "
             f"(**{a1.get('S669_contaminated_mutations')} mutations**), verified as real "
             f"megascale entries ({rows_str}). **Ssym ∩ megascale = CLEAN.** (FireProt "
             f"overlaps exist but pertain to the un-deployed FireProt model — not "
             f"contamination for the deployed megascale model.)")
    L.append(f"- **A2 (DynaMut2):** authoritative DynaMut2 training list not obtainable → "
             f"S669∩DynaMut2 **UNKNOWN** → DynaMut2 S669 scores **PROVISIONAL** (flagged, "
             f"not excluded). Ssym = DynaMut2 training family.")
    L.append(f"- **Resolution (APPROVED, applied):** keep S669 + Ssym; the 3 contaminated "
             f"proteins ({a1.get('S669_contaminated_mutations')} muts) are EXCLUDED FROM "
             f"ThermoMPNN vs-experiment scoring ONLY — **retained in the dataset and valid "
             f"for Rosetta / RaSP / DynaMut2. Not deleted.**\n")
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
    # per-voter vs-experiment scoring scopes (LOCKED)
    L.append("## Per-voter vs-experiment scoring scopes (LOCKED — authoritative)\n")
    L.append("| Voter | vs-experiment scope | Excluded | Status |")
    L.append("|-------|---------------------|----------|--------|")
    L.append("| **ThermoMPNN** | S669 minus {1O6X,2HBB,2WQG} (574 muts) + Ssym | "
             "1O6X/2HBB/2WQG (megascale training); Rocklin + Protein_G | clean |")
    L.append("| **Rosetta** | FULL set (S669 + Ssym) | — (leakage-free anchor, no training) | "
             "clean |")
    L.append("| **RaSP** | FULL set (S669 + Ssym) — generalization | — | "
             "clean vs-exp / **circular vs-Rosetta** |")
    L.append("| **DynaMut2** | S669 only (PROVISIONAL); Ssym excluded | Ssym (training family) | "
             "provisional |")
    L.append("")
    L.append("- **Anti-symmetry (Ssym fwd vs rev)** is **reference-free** (fwd vs rev, not vs "
             "experiment) and runs across **ALL voters incl. DynaMut2** (no experiment leakage "
             "in an anti-symmetry check). This is separate from the vs-experiment scopes above.")
    L.append("- **RaSP-vs-Rosetta** stays flagged **circular** (RaSP is a Rosetta surrogate); "
             "never reported as corroboration.")
    L.append("- The 3 ThermoMPNN-contaminated proteins remain fully in the dataset and are "
             "scored by Rosetta/RaSP/DynaMut2; they are dropped from ThermoMPNN vs-experiment "
             "confidence ONLY. **Nothing is deleted.**")
    L.append("")
    L.append("## Locked set\n")
    inc = sorted({r["set"] for r in e if r["proposed_include"]})
    exc = sorted({r["set"] for r in e if not r["proposed_include"]})
    L.append(f"- **INCLUDE (locked):** {', '.join(inc)}")
    L.append(f"- **EXCLUDE (available, not in set):** {', '.join(exc)}")
    L.append("")
    for s in by_set:
        L.append(f"  - `{s}`: {manifest['set_notes'][s]}")
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
