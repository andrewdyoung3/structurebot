"""
Live-verify — turning OFF the disulfide pair-click glow (GPU-FREE). The reported bug: the glow
(transparency-the-rest + the pair spotlight) persisted when switching to another colour mode → a
hybrid (new colours + leftover disulfide transparency) with no way to exit. This verifies BOTH
escape paths against REAL ChimeraX:

  1. Switching to another visual mode (any `_push_3d_color`) clears the glow FIRST — the un-ghost
     (`transparency #mid 0`) + deselect are emitted BEFORE the new colours, and `_glow_state` clears.
  2. The explicit "Clear disulfide view" control restores the normal representation on demand.

GPU-FREE: a real cached crystal CIF supplies the structure (no fold), opened in REAL ChimeraX (the
loaded-PDB source path); the glow + its restore run as real ChimeraX commands. The observable is the
live selection (the glow selects the pair; clearing deselects) + that the un-ghost command reaches
ChimeraX without error.

Run: venv/Scripts/python.exe scripts/verify_disulfide_glow_clear_live.py   (needs ChimeraX :60001; NO GPU)
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

CIF = Path(__file__).resolve().parent.parent / "cache" / "1A2W.cif"
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _model_ids():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _sel_count():
    v = run("info atoms sel").get("value") or ""
    return len(re.findall(r"atom id", v))


def main():
    if not CIF.is_file():
        print(f"cached CIF not found: {CIF}"); return 1
    if not bridge.is_running():
        print("ChimeraX REST not reachable on :60001 — open ChimeraX (or the app) first.")
        return 1
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = VariantWorkbenchPanel(MagicMock(), session=SessionState(), pool=MagicMock())
    sent = []
    panel._run_commands_bg = lambda cmds: (sent.extend(cmds), [run(c) for c in cmds])  # record + REACH ChimeraX

    # open the dimer in REAL ChimeraX + seed a loaded-structure construct
    before = set(_model_ids())
    run(f'open "{CIF.as_posix()}"')
    mids = sorted(set(_model_ids()) - before, key=int)
    if not mids:
        print("could not open the dimer in ChimeraX"); return 1
    mid = mids[-1]
    chains = sorted(set(re.findall(r"/([A-Za-z0-9]+)", run(f"info chains #{mid}").get("value") or "")))[:2] or ["A", "B"]
    panel._add_sequence_construct("dimer", "A" * 20)
    cd = next(iter(panel._design.chains.values()))
    cd.members = [(mid, chains[0]), (mid, chains[1])]
    cd.rep_model, cd.rep_chain = mid, chains[0]
    cd.template_fold = {"engine": "loaded", "target": "assembly", "model_id": mid, "cif_path": str(CIF)}
    tab = panel._cur_tab()
    pair = {"chain_a": chains[0], "resnum_a": 20, "chain_b": chains[1], "resnum_b": 20}

    # 1) GLOW a pair → ghost-the-rest applied, pair selected, Clear button lit ──────────────
    print("1) Glow a disulfide pair (spotlight ON)…")
    run("~select")
    sent.clear()
    panel._highlight_disulfide_pair(cd, pair)
    app.processEvents()
    check("the glow ghosts the rest of the model (transparency 70 emitted to ChimeraX)",
          f"transparency #{mid} 70 target c" in sent)
    check("the pair is SELECTED in real ChimeraX (the highlight)", _sel_count() >= 1,
          f"{_sel_count()} atoms selected")
    check("glow state recorded + the Clear control is lit",
          panel._glow_state is not None and panel.disulfides_tab._clear_glow_btn.isEnabled())

    # 2) SWITCH to a colour mode → glow clears FIRST, no hybrid ─────────────────────────────
    print("2) Switch to a colour mode → the glow must clear (no leftover transparency)…")
    sent.clear()
    panel._push_3d_color(tab)
    app.processEvents()
    check("un-ghost (transparency 0) emitted BEFORE the new colour/visibility commands",
          f"transparency #{mid} 0 target c" in sent
          and sent.index(f"transparency #{mid} 0 target c") == 0)
    check("the highlight is dropped (~select emitted, ChimeraX selection now empty)",
          "~select" in sent and _sel_count() == 0, f"{_sel_count()} atoms selected after switch")
    check("glow state cleared + Clear control dimmed (no hybrid, clean view)",
          panel._glow_state is None and not panel.disulfides_tab._clear_glow_btn.isEnabled())

    # 3) EXPLICIT 'Clear disulfide view' → back to normal on demand ──────────────────────────
    print("3) Glow again → explicit 'Clear disulfide view'…")
    run("~select")
    panel._highlight_disulfide_pair(cd, pair)
    app.processEvents()
    glowing = panel._glow_state is not None and _sel_count() >= 1
    sent.clear()
    panel._clear_disulfide_glow()                       # the explicit control
    app.processEvents()
    check("explicit clear restores normal representation (un-ghost + deselect) and dims the control",
          glowing and f"transparency #{mid} 0 target c" in sent and "~select" in sent
          and _sel_count() == 0 and panel._glow_state is None
          and not panel.disulfides_tab._clear_glow_btn.isEnabled())

    run(f"close #{mid}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
