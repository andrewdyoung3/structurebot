"""
Live-verify — alignment-overlay visibility toggle (US-align reference under the single-source
authority) against REAL ChimeraX (:60001). GPU-FREE: no fold, no remote — two LOCAL cache
structures stand in for the construct fold + the aligned reference, and a real `view matrix`
stands in for the option-B overlay transform. Confirms the toggles actually show/hide the RIGHT
models in ChimeraX, that fold and reference visibility are independent + composable, and that an
option-B-overlaid fold STAYS PUT (its coordinates are unchanged) when its reference is toggled.

Run: venv/Scripts/python.exe scripts/verify_alignment_overlay_visibility_live.py
Requires: ChimeraX REST on :60001 (no GPU, no network).
"""
import os, sys, tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PySide6 import QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from variant_workbench import VariantWorkbenchPanel

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command
FOLD_CIF = str(Path(__file__).resolve().parent.parent / "cache" / "1A2W.cif")
REF_CIF  = str(Path(__file__).resolve().parent.parent / "cache" / "2ACY.cif")

_checks = []
def check(name, ok, detail=""):
    _checks.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def _runscript(script: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    try:
        return run(f'runscript "{path}"').get("value") or ""
    finally:
        try: os.unlink(path)
        except OSError: pass


def open_model(path):
    before = set(_model_ids())
    run(f'open "{Path(path).as_posix()}"')
    after = set(_model_ids())
    new = sorted(after - before, key=int)
    return new[-1] if new else None


def _model_ids():
    import re
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def displayed(mid):
    """True/False — is model #mid currently displayed (read from ChimeraX ground truth)."""
    out = _runscript(
        "from chimerax.core.commands import run as r\n"
        f"m=[x for x in session.models.list() if str(x.id_string)=='{mid}']\n"
        "print('DISP', bool(m and m[0].display))\n")
    return "DISP True" in out


def com(mid):
    """Centroid of model #mid's atom coords (option-B stays-put probe)."""
    out = _runscript(
        f"m=[x for x in session.models.list() if str(x.id_string)=='{mid}']\n"
        "c=m[0].atoms.coords.mean(axis=0) if m else None\n"
        "print('COM', None if c is None else (round(float(c[0]),3),round(float(c[1]),3),round(float(c[2]),3)))\n")
    return out.strip().split("COM", 1)[-1].strip()


def push(cmds):
    for c in cmds:
        run(c)


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    print("Opening two LOCAL structures (stand-ins for the construct fold + the aligned reference)…")
    fold_id = open_model(FOLD_CIF)
    ref_id  = open_model(REF_CIF)
    check("two stand-in models opened", bool(fold_id and ref_id and fold_id != ref_id),
          f"fold #{fold_id}, reference #{ref_id}")
    if not (fold_id and ref_id):
        print("Could not open stand-in models — is ChimeraX on :60001?"); return 1

    # Build a de-novo construct panel whose 'fold' IS the real model, with an alignment reference.
    ctrl = SequenceEditorController(run, lambda **k: {})
    sess = SessionState()
    panel = VariantWorkbenchPanel(ctrl, session=sess)
    panel._add_sequence_construct("probe", "MKVLWAACGTDEFHIKLMNP")
    cd = next(iter(panel._design.chains.values()))
    cd.template_fold = {"engine": "boltz", "target": "monomer", "model_id": fold_id,
                        "cif_path": FOLD_CIF, "mean_plddt": 80.0}
    cd.members  = [(fold_id, "A")]
    cd.rep_model = fold_id
    cd.structural_align = {"ref_label": "2ACY", "tm_ref": 0.8, "reference_model_id": ref_id,
                           "shared_fold": True}
    panel._sync_align_ref_toggle()
    tab = panel._cur_tab()

    # Option-B overlay stand-in: transform the FOLD model in place + record its coordinates.
    run(f"view matrix models #{fold_id},1,0,0,25,0,1,0,0,0,0,1,0")   # translate +25 Å in x
    com_before = com(fold_id)

    print("1) Default state — reference SHOWN, under fold_visibility_commands authority")
    push(panel.fold_visibility_commands(tab))
    check("reference shown by default", displayed(ref_id))
    check("per-cd toggle reflects shown + enabled",
          panel._show_align_ref_cb.isEnabled() and panel._show_align_ref_cb.isChecked())

    print("2) Per-cd 'Aligned reference' OFF → reference hidden, fold stays, flag persisted")
    panel._show_align_ref_cb.setChecked(False)
    push(panel.fold_visibility_commands(tab))
    check("reference hidden", not displayed(ref_id))
    check("fold still shown (independent)", displayed(fold_id))
    check("hidden flag persisted on cd.structural_align", cd.structural_align.get("hidden") is True)
    check("option-B fold STAYS PUT when reference toggled", com(fold_id) == com_before,
          f"{com_before} -> {com(fold_id)}")

    print("3) Color-only invariant — a color push must NOT re-show the hidden reference")
    push(panel.fold_visibility_commands(tab) + panel.color_commands_for(tab))
    check("reference still hidden after a color push", not displayed(ref_id))

    print("4) Per-cd ON again → reference re-shown")
    panel._show_align_ref_cb.setChecked(True)
    push(panel.fold_visibility_commands(tab))
    check("reference re-shown", displayed(ref_id))

    print("5) Global 'Hide alignment references' → force-hide; folds independent")
    panel._align_ref_vis_btn.setChecked(True)
    push(panel.fold_visibility_commands(tab))
    check("global hide hides the reference", not displayed(ref_id))
    check("global hide leaves the fold shown", displayed(fold_id))
    panel._align_ref_vis_btn.setChecked(False)
    push(panel.fold_visibility_commands(tab))
    check("global show restores the reference", displayed(ref_id))

    # cleanup
    run(f"close #{fold_id}"); run(f"close #{ref_id}")
    ok = all(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
