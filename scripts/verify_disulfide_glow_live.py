"""
Live-verify — static GLOW highlight for the disulfide pair-click, in the REAL Workbench + REAL
ChimeraX. A VISUAL feature's verify must LOOK at the result: this glows pair A, screenshots it,
glows pair B, screenshots it, and the gate is the eyeball check — A pops (bright spheres + ghosted
surroundings + halo), and on clicking B the scene RESETS (A un-glows, transparency clears) and B
pops. No GPU (opens a cached structure; the glow is display-state only).

Confirms:
  1. The glow commands RUN against a real fold with NO error (the recipe is valid).
  2. Glow A → A's residues SELECTED + shown (bright spheres), the model GHOSTED (transparency).
  3. Glow B → A's residues RESTORED (no longer displayed as atoms), B SELECTED + shown — the prior
     glow did NOT stack (non-destructive restore).
  4. Two screenshots saved for the eyeball gate (glow_A.png / glow_B.png).

Run: venv/Scripts/python.exe scripts/verify_disulfide_glow_live.py   (no GPU; needs ChimeraX :60001)
"""
import os, sys, re
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from unittest.mock import MagicMock
from PySide6 import QtWidgets
from chimerax_bridge import ChimeraXBridge
from session_state import SessionState
from variant_workbench import VariantWorkbenchPanel

CIF = str(Path(__file__).resolve().parent.parent / "cache" / "1A2W.cif")
OUT = Path(__file__).resolve().parent.parent / "cache"
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
_errors = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _run_traced(cmds):
    """Run a glow command list through ChimeraX, collecting any errors (the recipe-valid check)."""
    for c in cmds:
        r = run(c)
        if isinstance(r, dict) and r.get("error"):
            _errors.append((c, r["error"]))


def _displayed_atoms(mid, chain, resnums):
    """Count of DISPLAYED atoms for *resnums* on *chain* of model *mid* (runscript probe — the
    reliable display-aware count; `info atoms` lists atoms regardless of display). 0 = restored to
    cartoon (atoms hidden)."""
    import tempfile as _tmp, os as _os
    want = ",".join(str(r) for r in resnums)
    script = (
        "from chimerax.atomic import all_atomic_structures\n"
        f"want = {{{want}}}\n"
        "n = 0\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{mid}':\n"
        "        for a in m.atoms:\n"
        f"            if a.residue.chain_id == '{chain}' and a.residue.number in want and a.display:\n"
        "                n += 1\n"
        "        break\n"
        "print(n)\n"
    )
    fd, path = _tmp.mkstemp(suffix=".py")
    try:
        with _os.fdopen(fd, "w") as f:
            f.write(script)
        r = run(f"runscript {path}")
    finally:
        try:
            _os.unlink(path)
        except OSError:
            pass
    m = re.search(r"\b(\d+)\b", (r.get("value") or "").strip())
    return int(m.group(1)) if m else -1


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    panel._run_commands_bg = lambda cmds: _run_traced(cmds)        # synchronous → reaches ChimeraX + traced

    before = set(_model_ids())
    run(f'open "{Path(CIF).as_posix()}"')
    mids = sorted(set(_model_ids()) - before, key=int)
    if not mids:
        print("  could not open 1A2W in ChimeraX"); return 1
    mid = mids[-1]
    run(f"hide #{mid} atoms"); run(f"show #{mid} cartoons")        # a clean cartoon scene to spotlight

    panel._add_sequence_construct("probe", "A" * 130)
    cd = next(iter(panel._design.chains.values()))
    cd.rep_chain = "A"
    cd.template_fold = {"engine": "boltz", "target": "assembly", "model_id": mid, "cif_path": CIF}

    pairA = {"chain_a": "A", "resnum_a": 50, "chain_b": "A", "resnum_b": 60}
    pairB = {"chain_a": "A", "resnum_a": 100, "chain_b": "A", "resnum_b": 110}

    print("1) Glow pair A (residues 50, 60)…")
    panel._highlight_disulfide_pair(cd, pairA)
    run(f"view #{mid}")
    selA = run("info atoms sel").get("value") or ""
    check("glow A — its residues are SELECTED (50 & 60)", "50" in selA and "60" in selA)
    check("glow A — A's residues are shown as atoms (the lit spheres)",
          _displayed_atoms(mid, "A", [50, 60]) > 0)
    run(f'save "{(OUT / "glow_A.png").as_posix()}" width 900 height 700')
    print(f"   screenshot → {OUT / 'glow_A.png'}")

    print("2) Glow pair B (residues 100, 110) — A must RESET…")
    panel._highlight_disulfide_pair(cd, pairB)
    run(f"view #{mid}")
    selB = run("info atoms sel").get("value") or ""
    check("glow B — its residues are SELECTED (100 & 110)", "100" in selB and "110" in selB)
    a_now = _displayed_atoms(mid, "A", [50, 60])
    check("glow B — pair A RESTORED (its atoms no longer displayed — not stacked)",
          a_now == 0, f"A displayed-atoms now {a_now}")
    check("glow B — B's residues are shown as atoms", _displayed_atoms(mid, "A", [100, 110]) > 0)
    run(f'save "{(OUT / "glow_B.png").as_posix()}" width 900 height 700')
    print(f"   screenshot → {OUT / 'glow_B.png'}")

    check("the glow recipe ran against a REAL fold with NO ChimeraX errors",
          not _errors, "; ".join(f"{c} → {e}" for c, e in _errors[:3]))

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
