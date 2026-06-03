#!/usr/bin/env python3
"""
freeze_zone_gold.py — freeze the deterministic selection_resnums gold for zone cases.

The corpus marks every selection_resnums 'expected' field as PENDING_FREEZE because
the resnum set is a deterministic consequence of the *structure*, not of any model
(so it is legitimate model-independent gold — see §0 — but must be computed once on
the real reference PDB rather than guessed).

This script, run on the StructureBot host WITH live ChimeraX on REST port 60001:
  1. for each PENDING zone case, opens the case's reference PDB (1HSG/2LZM/1IL8),
  2. runs the canonical gold command (command_contains_any[0]),
  3. reads `info residues sel`, parses the resnum set (reusing selection.py),
  4. writes the frozen set back into the manifest's assertion.expected.

It NEVER calls a model. Review the canonical command for each case before trusting
the frozen set (the script prints each command + the set it produced).

Usage:
    python scripts/freeze_zone_gold.py eval_corpus_manifest.json
    # or a dry run that only prints what it would do:
    python scripts/freeze_zone_gold.py eval_corpus_manifest.json --dry-run
"""
import json, sys, argparse

# Reuse the project's REST bridge + selection parser so the gold matches production.
try:
    from chimerax_bridge import ChimeraXBridge
    from selection import read_selection            # parses `info residues sel`
except Exception as e:                              # pragma: no cover
    print("Run this from the StructureBot repo root (needs chimerax_bridge + selection).")
    print("Import error:", e)
    sys.exit(2)

PENDING = "PENDING_FREEZE"

# Map the bare PDB id used in the corpus to how it should be opened.
def open_cmd(pdb):
    return f"open {pdb}"

def canonical_select(case):
    """The canonical gold selection command for a zone case = command_contains_any[0],
    wrapped in `select ...` if it isn't already a select/command."""
    pats = case["gold_accuracy"]["required_args"]["command_contains_any"]
    expr = pats[0].strip()
    if expr.lower().startswith(("select", "~select")):
        return expr
    return f"select {expr}"

def freeze(manifest_path, dry_run=False):
    with open(manifest_path) as fh:
        man = json.load(fh)

    bridge = ChimeraXBridge()
    bridge.ensure_connected()
    frozen, skipped = 0, 0
    cur_pdb = None

    for case in man["cases"]:
        f = case.get("gold_functionality", {})
        a = f.get("assertion", {})
        if a.get("probe") != "selection_resnums" or a.get("expected") != PENDING:
            continue
        pdb = a.get("structure")
        chain = a.get("chain")
        cmd = canonical_select(case)

        if cur_pdb != pdb:
            bridge.run_command("close session")
            bridge.run_command(open_cmd(pdb))
            cur_pdb = pdb

        print(f"\n[{case['id']}]  {pdb}")
        print(f"    criterion : {a.get('criterion','')}")
        print(f"    command   : {cmd}")
        if dry_run:
            continue

        bridge.run_command("select clear")
        bridge.run_command(cmd)
        sel = read_selection(bridge.run_command)         # -> Selection
        resnums = sorted(sel.resnums(chain)) if chain else sorted(
            {rn for ch in sel.chains for rn in sel.resnums(ch)})
        print(f"    -> {len(resnums)} residues: {resnums}")
        a["expected"] = resnums
        frozen += 1

    if not dry_run:
        with open(manifest_path, "w") as fh:
            json.dump(man, fh, indent=2)
        print(f"\nFroze {frozen} selection_resnums gold sets into {manifest_path}.")
    else:
        print("\n(dry run — nothing written)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    freeze(args.manifest, dry_run=args.dry_run)
