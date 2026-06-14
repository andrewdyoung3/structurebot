"""
Live-verify the Variant-Design Workbench Stage 2 against a REAL running ChimeraX
(REST :60001) - the gate: active-row color mode actually recolors the 3D to match the
panel, including AFTER a variant edit, across all homo-oligomer copies. Read-only proof
by reading residue ribbon_color BACK from ChimeraX (no mocks on the 3D side).

Run: venv/Scripts/python.exe scripts/verify_workbench_stage2_live.py [model_id]
"""
import os, sys, tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6 import QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from selection import read_selection
from variant_workbench import VariantWorkbenchPanel
from color_modes import get_mode

MID = sys.argv[1] if len(sys.argv) > 1 else "1"
bridge = ChimeraXBridge(port=60001)
run = bridge.run_command


def hexrgb(h):
    h = h.lstrip("#"); return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def read_colors(model, chain, resnums):
    """Read residue.ribbon_color (RGBA8) back from ChimeraX for the given resnums."""
    rl = ",".join(str(r) for r in resnums)
    script = (
        "from chimerax.atomic import all_atomic_structures\n"
        f"want = set({list(resnums)!r})\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        "        for r in m.residues:\n"
        f"            if r.chain_id == '{chain}' and r.number in want:\n"
        "                c = r.ribbon_color\n"
        "                print(f'{r.number}:{c[0]},{c[1]},{c[2]}')\n"
        "        break\n"
    )
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    try:
        r = run(f'runscript "{path}"')
    finally:
        try: os.unlink(path)
        except OSError: pass
    out = {}
    for line in (r.get("value") or "").splitlines():
        line = line.strip()
        if ":" in line and "," in line:
            rn, rgb = line.split(":", 1)
            try:
                out[int(rn)] = tuple(int(x) for x in rgb.split(",")[:3])
            except ValueError:
                pass
    return out


def near(a, b, tol=6):
    return all(abs(a[i] - b[i]) <= tol for i in range(3))


app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ctrl = SequenceEditorController(run, lambda **k: {})
panel = VariantWorkbenchPanel(ctrl)            # REAL controller, no mocks on the 3D side
panel.load_model(MID)
tab = panel._cur_tab()
assert tab is not None, "no tab - model load failed"
cd = tab.design
print(f"[design] model #{MID}: {len(panel._design.chains)} unique chain(s); "
      f"rep #{cd.rep_model}/{cd.rep_chain}; members={cd.members}; "
      f"{len(cd.template_cells)} cols; "
      f"resnum {cd.template_cells[0].resnum}..{cd.template_cells[-1].resnum}")

# pick template columns by property
def find(pred):
    for c in cd.template_cells:
        if c.aa and pred(c.aa):
            return c
    return None
pos = find(lambda a: a in "KR")        # -> strong blue
neg = find(lambda a: a in "DE")        # -> red
neu = find(lambda a: a in "AGILVMF")   # -> neutral (white)
charge = get_mode("charge")
print(f"[picks] pos(K/R)={pos and (pos.resnum, pos.aa)}  "
      f"neg(D/E)={neg and (neg.resnum, neg.aa)}  neu={neu and (neu.resnum, neu.aa)}")

# -- 1) Stage-1 regression: column-click select reaches the 3D (all copies) ---------
col = cd.template_cells[len(cd.template_cells)//2].col
specs = panel.select_specs_for_column(cd, col)
ctrl.select_residues_multi(specs)
sel = read_selection(run, default_model=MID)
selres = {(s[1], s[2]) for s in sel.residues}
want = {(c, r) for (_m, c, rs) in specs for r in rs}
print(f"[select] col {col} -> specs {specs} -> 3D selection has {want}: "
      f"{'PASS' if want <= selres else 'FAIL got ' + str(sorted(selres))}")

# -- 2) Color mode on the TEMPLATE (active row = T): panel == 3D ---------------------
panel_mode_idx = next(i for i in range(panel._mode_combo.count())
                      if panel._mode_combo.itemData(i) == "charge")
panel._mode_combo.setCurrentIndex(panel_mode_idx)     # triggers panel repaint + (async) push
cmds = panel.color_commands_for(tab)                  # the EXACT commands the panel emits
ctrl.run_commands(cmds)                               # what _ColorWorker runs - real push
print(f"[3D push] {len(cmds)} commands, e.g. {cmds[1] if len(cmds) > 1 else cmds}")

checks = []
for cell, expect_hex, label in [(pos, charge.color_for(pos.aa) if pos else None, "K/R->blue"),
                                (neg, charge.color_for(neg.aa) if neg else None, "D/E->red"),
                                (neu, "#ffffff", "neutral->white")]:
    if cell is None:
        print(f"[T color] {label}: (no such residue in template - skipped)")
        continue
    exp = hexrgb(expect_hex)
    panel_hex = tab.color_hex_at("T", cell.col)
    for (m, c) in cd.members:                          # EVERY homo-oligomer copy
        got = read_colors(m, c, [cell.resnum]).get(cell.resnum)
        ok = got is not None and near(got, exp)
        checks.append(ok)
        print(f"[T color] {label} #{m}/{c}:{cell.resnum} expect {exp} got {got} "
              f"panel={panel_hex} : {'PASS' if ok else 'FAIL'}")

# -- 3) THE Stage-2 invariant: active VARIANT edit drives the 3D ---------------------
vid = panel._design.new_variant_id() if False else None
panel._add_variant()
vid = tab.design.variants[-1].id
# edit a NEUTRAL template column -> D (negative): 3D must turn that residue RED because
# the active row is the edited variant, even though the TEMPLATE there is neutral.
target = neu or cd.template_cells[0]
panel._on_cell(tab, vid, target.col)                  # set active row = variant + edit target
panel._aa_combo.setCurrentText("D")
panel._apply_substitution()
cmds2 = panel.color_commands_for(tab)
ctrl.run_commands(cmds2)
red = hexrgb(charge.color_for("D"))
print(f"[active-row] edited {vid} col {target.col} (resnum {target.resnum}, T={target.aa}) -> D")
for (m, c) in cd.members:
    got = read_colors(m, c, [target.resnum]).get(target.resnum)
    ok = got is not None and near(got, red)
    checks.append(ok)
    print(f"[active-row] #{m}/{c}:{target.resnum} expect RED {red} got {got} : "
          f"{'PASS' if ok else 'FAIL'}")

print("\nRESULT:", "ALL PASS" if checks and all(checks) else "FAILURES PRESENT")
