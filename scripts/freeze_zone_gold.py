#!/usr/bin/env python3
"""
freeze_zone_gold.py — freeze the deterministic selection_resnums gold for zone cases.

The corpus marks every selection_resnums 'expected' field as PENDING_FREEZE because
the resnum set is a deterministic consequence of the *structure*, not of any model
(legitimate model-independent gold — see §0 — but must be computed once on the real
reference PDB rather than guessed).

CRITICAL: the frozen gold and the scoring-time measurement MUST come from ONE parser
path, or the comparison is invalid. So this script reuses the EXACT reader the effect
scorer uses — `eval_harness.session_open_commands()` (the loaded-state precondition,
including the case's `session.selection`) + `eval_harness._parse_info_residues()` (the
`info residues sel` parser). Transport is the project's `ChimeraXBridge` REST client,
adapted to the scorer's `command -> str` probe interface.

It NEVER calls a model. Review the canonical command (command_contains_any[0]) for
each case before trusting the frozen set — the script prints each command + the set.

Usage (run on the StructureBot host WITH live ChimeraX on REST port 60001):
    python scripts/freeze_zone_gold.py                 # freezes scripts/eval_corpus_manifest.json
    python scripts/freeze_zone_gold.py --dry-run       # resolve + PRINT against live ChimeraX, write nothing
    python scripts/freeze_zone_gold.py <path> [--dry-run]
"""
import os
import sys
import json
import argparse

# A bare `python scripts/freeze_zone_gold.py` run does NOT put the repo root on
# sys.path — add it so the harness + bridge import.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import eval_harness as eh                       # the SAME reader the scorer uses
from chimerax_bridge import ChimeraXBridge

PENDING = eh.PENDING_FREEZE
DEFAULT_MANIFEST = os.path.join(_REPO, "scripts", "eval_corpus_manifest.json")
_MIN_PLAUSIBLE = 3                              # flag empty / implausibly tiny sets


def make_probe(bridge):
    """Adapt ChimeraXBridge.run_command (→ {'value': <plain text>}) to the scorer's
    `command -> str` probe — exactly what score_functionality(effect) drives."""
    def probe(command: str) -> str:
        r = bridge.run_command(command)
        if isinstance(r, dict):
            return str(r.get("value") or "")
        return str(r or "")
    return probe


def canonical_select(case):
    """Canonical gold selection command = command_contains_any[0], wrapped in
    `select …` unless it is already a select command."""
    expr = case["gold_accuracy"]["required_args"]["command_contains_any"][0].strip()
    if expr.lower().startswith(("select", "~select")):
        return expr
    return f"select {expr}"


def freeze(manifest_path, dry_run=False):
    with open(manifest_path, encoding="utf-8") as fh:
        man = json.load(fh)

    bridge = ChimeraXBridge()
    bridge.ensure_connected()
    probe = make_probe(bridge)

    frozen, suspects = 0, []
    for case in man["cases"]:
        a = (case.get("gold_functionality") or {}).get("assertion", {})
        if a.get("probe") != "selection_resnums" or a.get("expected") != PENDING:
            continue
        chain = a.get("chain")
        cmd = canonical_select(case)

        # Reconstruct the EXACT loaded-state precondition the effect scorer applies:
        # a fresh scene, open the declared model(s), and apply the case's
        # session.selection — so `sel`-relative zone commands resolve correctly.
        probe("close session")
        pre = eh.session_open_commands(case.get("session"))
        for oc in pre:
            probe(oc)

        # Run the canonical gold selection, then read it back through the scorer's
        # OWN parser (chain filter included) — gold == measurement, one path.
        probe(cmd)
        out = probe("info residues sel")
        resnums = sorted(eh._parse_info_residues(out, chain=chain))

        tiny = len(resnums) < _MIN_PLAUSIBLE
        if tiny:
            suspects.append((case["id"], len(resnums)))
        print(f"\n[{case['id']}]  {a.get('structure')}  chain={chain}")
        print(f"    criterion : {a.get('criterion', '')}")
        print(f"    precond   : {pre}")
        print(f"    command   : {cmd}")
        print(f"    -> {len(resnums)} residues: {resnums}" + ("   <-- EMPTY/TINY" if tiny else ""))

        if not dry_run:
            a["expected"] = resnums
            frozen += 1

    if dry_run:
        print("\n(dry run — nothing written)")
    else:
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(man, fh, indent=2)
        print(f"\nFroze {frozen} selection_resnums gold sets into {manifest_path}.")

    if suspects:
        print("\nWARNING — empty/implausibly-tiny sets (review the canonical command / "
              "chain filter / session.selection before trusting these):")
        for cid, n in suspects:
            print(f"    {cid}: {n} residue(s)")
    else:
        print("\nAll resolved sets are non-trivial (>= %d residues)." % _MIN_PLAUSIBLE)
    return suspects


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", nargs="?", default=DEFAULT_MANIFEST,
                    help="manifest path (default: scripts/eval_corpus_manifest.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve + print each command and its live resnum set; write nothing")
    args = ap.parse_args()
    mp = args.manifest
    if not os.path.isfile(mp):                  # accept a bare filename → resolve under scripts/
        alt = os.path.join(_REPO, "scripts", os.path.basename(mp))
        mp = alt if os.path.isfile(alt) else mp
    freeze(mp, dry_run=args.dry_run)
