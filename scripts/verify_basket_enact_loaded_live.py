"""
Live-verify — DESIGN-BASKET enact on a LOADED PDB, REAL scan → REAL stage → REAL enact, against a
DEDICATED ChimeraX :60002 (the user's :60001 is left untouched). GPU-free.

The reported bug: enacting a basket on a loaded structure produced a variant with only a few (or no)
substitutions, silently. This drives the REAL loaded-structure path — `panel.load_model` (author
resnums into the design template) → `_active_structure` (fresh CIF save) → a REAL disulfide geometry
scan (candidate positions in the structure's author numbering) → the REAL `_add_disulfide_to_basket`
(curate-time guard) → the REAL `_enact_basket` — on a structure RENUMBERED to a non-1-based author
range (start 101), the exact "loaded PDB not numbered from 1" case the symptom pointed at.

Checks:
  1. real scan candidates carry the OFFSET author numbering (>=101) — the test actually stresses it.
  2. staging two disjoint disulfide pairs via the real picker is accepted (curate guard passes).
  3. ENACT lands ALL four cysteines at the real author positions (no silent drop) — ONE variant.
  4. a pick at an off-template resnum is BLOCKED AT STAGING (loud), never silently vanishing at enact.
  5. a programmatic enact with one off-template sub APPLIES the mappable one + REPORTS the skip (no
     empty variant, no abort).

Run: venv/Scripts/python.exe scripts/verify_basket_enact_loaded_live.py
"""
import os, sys, time, subprocess, urllib.request, urllib.parse
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from unittest.mock import MagicMock
from PySide6 import QtWidgets
from chimerax_bridge import find_chimerax, ChimeraXBridge
from session_state import SessionState
from seq_editor.controller import SequenceEditorController
from variant_workbench import VariantWorkbenchPanel
import disulfide_geometry as dg

PORT = 60002
_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    bridge = ChimeraXBridge(port=PORT)
    proc = None
    if not bridge.is_running():
        print(f"[setup] launching dedicated ChimeraX on :{PORT} …")
        proc = subprocess.Popen(
            [find_chimerax(), "--cmd", f"remotecontrol rest start port {PORT}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0)
        deadline = time.time() + 90
        while time.time() < deadline and not bridge.is_running():
            time.sleep(1.0)
        if not bridge.is_running():
            print(f"[setup] FAILED — :{PORT} never came up"); return 1
        time.sleep(1.0)
        print("[setup] reachable.")
    else:
        print(f"[setup] reusing ChimeraX already on :{PORT}")

    run = bridge.run_command
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def model_ids():
        import re
        return re.findall(r"model id #(\d+)\b", run("info models").get("value") or "")

    try:
        run("close session")
        before = set(model_ids())
        run("open 1ubq")
        mid = sorted(set(model_ids()) - before, key=int)[-1]
        run(f"renumber #{mid} start 101")          # OFFSET author numbering: protein now 101..176

        ctrl = SequenceEditorController(run_command=run, fold_fn=lambda *a, **k: {})
        panel = VariantWorkbenchPanel(ctrl, session=SessionState(), pool=MagicMock())
        # 3D pushes are not what we verify here (the variant DATA model is) — no-op them so the
        # data-model checks don't depend on the live render channel staying up mid-enact.
        panel._run_commands_bg = lambda cmds: None
        panel.load_model(str(mid))                 # build a "structure" design: template resnums 101..
        cd = panel._cur_tab().design
        tmpl_resnums = [c.resnum for c in cd.template_cells if not c.is_gap]
        print(f"  loaded design: rep_chain={cd.rep_chain} template resnums {min(tmpl_resnums)}..{max(tmpl_resnums)}")

        # 1) REAL disulfide scan on the saved CIF — candidate positions in author (offset) numbering
        src = panel._active_structure()
        atoms = dg.parse_backbone_atoms(src["cif_path"])
        ranked, _best = dg.scan_engineerable_sites(atoms)
        # pick TWO pairs with 4 DISTINCT positions (so no same-residue conflict blocks enact)
        chosen, used = [], set()
        for pr in ranked:
            ps = {pr["resnum_a"], pr["resnum_b"]}
            if len(ps) == 2 and not (ps & used):
                chosen.append(pr); used |= ps
            if len(chosen) == 2:
                break
        check("1) real scan yields candidates in the OFFSET author numbering (>=101)",
              len(chosen) == 2 and min(used) >= 101, f"chosen positions={sorted(used)}")
        if len(chosen) < 2:
            print("  (not enough disjoint pairs to proceed)"); return 1

        # 2) stage both via the REAL picker (curate guard must accept in-template author positions)
        n_before = len(panel.design_basket.entries)
        for pr in chosen:
            panel._add_disulfide_to_basket(cd, pr)
        check("2) two disulfide pairs stage cleanly (curate guard passes valid author positions)",
              len(panel.design_basket.entries) == n_before + 2,
              f"entries={len(panel.design_basket.entries)} status={panel._status.text()!r}")

        # 3) ENACT — all four cysteines must land at the real author positions
        v_before = len(cd.variants)
        panel.design_basket._enact()
        landed = []
        if len(cd.variants) > v_before:
            landed = sorted((m.resnum, m.to_aa) for m in cd.variants[-1].mutations)
        expect = sorted((p, "C") for p in used)
        check("3) enact lands ALL four cysteines at the real author positions (no silent drop)",
              landed == expect, f"landed={landed} expected={expect}")

        # 4) curate guard BLOCKS an off-template pick at staging (loud)
        n_now = len(panel.design_basket.entries)
        bogus = {"chain_a": cd.rep_chain, "resnum_a": 99999, "chain_b": cd.rep_chain,
                 "resnum_b": 99998, "best_sg_sg": 2.05, "best_chi_ss": -87.0, "clash": False, "score": 0.9}
        panel._add_disulfide_to_basket(cd, bogus)
        check("4) an off-template pick is BLOCKED at staging (never silently vanishes at enact)",
              len(panel.design_basket.entries) == n_now and "Can't add" in panel._status.text(),
              f"status={panel._status.text()!r}")

        # 5) programmatic partial enact: one good + one off-template → good lands, skip reported
        good_pos = sorted(used)[0]
        v_before = len(cd.variants)
        panel._enact_basket([{"cls": "Mix", "subs": [
            {"chain": cd.rep_chain, "position": good_pos, "from_aa": "X", "to_aa": "W"},
            {"chain": cd.rep_chain, "position": 99999,   "from_aa": "X", "to_aa": "W"}]}])
        v = cd.variants[-1] if len(cd.variants) > v_before else None
        muts = [(m.resnum, m.to_aa) for m in v.mutations] if v else []
        check("5) partial enact applies the mappable sub AND reports the skip (no empty variant/abort)",
              muts == [(good_pos, "W")] and "Skipped" in panel._status.text() and "99999" in panel._status.text(),
              f"muts={muts} status={panel._status.text()!r}")

        try:
            run("close session")
        except Exception:
            pass                                   # all checks already ran; teardown only
    finally:
        if proc is not None:
            print("\n[teardown] closing dedicated ChimeraX …")
            try: run("exit")
            except Exception: pass
            try: proc.terminate()
            except Exception: pass

    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
