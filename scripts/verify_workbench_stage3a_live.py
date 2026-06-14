"""
Live-verify the Variant-Design Workbench Stage 3a against a REAL running ChimeraX
(REST :60001) with REAL tool outputs (no mocks on the tool side):
  - REAL ProteinMPNN (venv312 subprocess) -> designs import as correct variant rows
    (right mutations + provenance), on the right chain.
  - REAL mutation_scanner (CamSol+ESM fast tier) -> inline suggestions land at the
    scanned resnums with the real scores; the Suggest track is SPARSE (only where
    the scan ran).
  - accept a suggestion -> the active variant gets the mutation (provenance
    accepted_suggestion + score) and the 3D recolors that residue in ALL homo-oligomer
    copies (1HSG A+B), proven by reading residue ribbon_color BACK from ChimeraX.

Freshly opens 1HSG, then closes it (non-destructive to the user's session).
Run: venv/Scripts/python.exe scripts/verify_workbench_stage3a_live.py
"""
import os, sys, tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # tool text may carry emoji
except Exception:
    pass

from PySide6 import QtCore, QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from mutation_scanner import MutationScanner
from proteinmpnn_bridge import ProteinMPNNBridge
from variant_workbench import VariantWorkbenchPanel
from color_modes import get_mode

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command


def models():
    txt = (run("info models").get("value") or "")
    import re
    return re.findall(r"model id #(\d+) ", txt)


def read_colors(model, chain, resnums):
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
            try: out[int(rn)] = tuple(int(x) for x in rgb.split(",")[:3])
            except ValueError: pass
    return out


def near(a, b, tol=6):
    return a is not None and all(abs(a[i] - b[i]) <= tol for i in range(3))


# ── open a fresh 1HSG ───────────────────────────────────────────────────────────────
before = set(models())
run("open 1hsg")
after = set(models())
new = sorted(after - before, key=int)
MID = new[-1] if new else "1"
print(f"[setup] opened 1HSG as model #{MID}")

pdb_path = (Path(tempfile.gettempdir()) / f"s3a_1hsg_{MID}.pdb").as_posix()
run(f'save "{pdb_path}" format pdb models #{MID}')

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
ctrl = SequenceEditorController(run, lambda **k: {})
session = SessionState()

# read chain A residues from the live model to pick valid scan/design positions
tmp_chains = ctrl.load_model(MID)
chainA = next((cs for cs in tmp_chains if cs.chain == "A"), tmp_chains[0])
resnums_A = chainA.resnums()
scan_pos = [resnums_A[i] for i in (9, 24, 49) if i < len(resnums_A)][:3]
print(f"[setup] chain A: {len(resnums_A)} residues ({resnums_A[0]}..{resnums_A[-1]}); "
      f"scan/design positions = {scan_pos}")

checks = []

# ── REAL mutation_scanner (fast tier: CamSol + ESM) ──────────────────────────────────
print("[scan] running REAL mutation_scanner (CamSol+ESM) on chain A, scoped...")
scanner = MutationScanner(session, model_id=MID)
scan_res = scanner.scan(pdb_path, chain_id="A", include_positions=scan_pos,
                        run_rosetta=False, run_thermompnn=False,
                        run_rasp=False, run_dynamut2=False)
scanned_resnums = sorted({c["resnum"] for c in scan_res})
print(f"[scan] {len(scan_res)} candidates at resnums {scanned_resnums}")
checks.append(("scanner produced candidates", bool(scan_res)))

# ── REAL ProteinMPNN (scoped, few sequences) ─────────────────────────────────────────
print("[mpnn] running REAL ProteinMPNN on chain A, scoped...")
mpnn_step = ProteinMPNNBridge().analyze(
    {"pdb_path": pdb_path, "chain_id": "A", "num_sequences": 4,
     "design_positions": scan_pos, "model_id": MID}, session)
mpnn = session.get_proteinmpnn_result(MID)
n_designs = len(mpnn.get("sequences", [])) if mpnn else 0
print(f"[mpnn] success={getattr(mpnn_step,'success',None)}  cached designs={n_designs}")

# ── drive the REAL panel ─────────────────────────────────────────────────────────────
panel = VariantWorkbenchPanel(ctrl, session=session)
panel.load_model(MID)
tab = panel._cur_tab()
cd = tab.design
print(f"[panel] unique chain tab rep=#{cd.rep_model}/{cd.rep_chain} members={cd.members} "
      f"({len(cd.template_cells)} cols)")
checks.append(("homo-oligomer collapsed (A+B)", len(cd.members) >= 2))

# 1) batch MPNN import -> rows
panel._import_mpnn()
mpnn_rows = [v for v in cd.variants if v.source == "proteinmpnn"]
print(f"[import] {len(mpnn_rows)} MPNN variant row(s); "
      f"provenance0={mpnn_rows[0].provenance if mpnn_rows else None}")
print(f"[import] row0 mutations={[(m.resnum,m.from_aa,m.to_aa) for m in mpnn_rows[0].mutations] if mpnn_rows else None}")
checks.append(("MPNN designs imported as rows", n_designs > 0 and len(mpnn_rows) == n_designs))
checks.append(("import provenance has fasta_path",
               bool(mpnn_rows) and "fasta_path" in mpnn_rows[0].provenance))

# 2) inline suggestions -> sparse track at the scanned columns only
panel._load_suggestions()
sugg_cols = set(tab.suggestions)
sugg_resnums = sorted({cd.resnum_for_col(c) for c in sugg_cols})
print(f"[suggest] track at resnums {sugg_resnums} (scanned {scanned_resnums})")
checks.append(("suggestions at the scanned resnums", sugg_resnums == scanned_resnums))
# sparsity: a residue we did NOT scan has no suggestion
unscanned = next((r for r in resnums_A if r not in scanned_resnums), None)
unscanned_col = next((c.col for c in cd.template_cells if c.resnum == unscanned), None)
checks.append(("Suggest track is SPARSE (no suggestion where unscanned)",
               unscanned_col not in sugg_cols))

# 3) accept a suggestion into an active variant -> mutation + provenance + 3D all copies
panel._add_variant()
vid = cd.variants[-1].id
pick_col = sorted(sugg_cols)[0]
top = tab.suggestions[pick_col][0]
panel._mode_combo.setCurrentIndex(                       # charge, so accept recolors 3D
    next(i for i in range(panel._mode_combo.count())
         if panel._mode_combo.itemData(i) == "charge"))
panel._accept_suggestion(tab, pick_col, top)
v = cd.get_variant(vid)
acc_resnum = cd.resnum_for_col(pick_col)
mut = next((m for m in v.mutations if m.resnum == acc_resnum), None)
print(f"[accept] {vid}: resnum {acc_resnum} -> {top.get('to_aa')} "
      f"(score {top.get('combined_score'):+.2f}); mutation source={mut.source if mut else None}; "
      f"provenance.accepted={v.provenance.get('accepted')}")
checks.append(("accepted mutation recorded", mut is not None and mut.to_aa == top.get("to_aa")))
checks.append(("provenance source = accepted_suggestion",
               mut is not None and mut.source == "accepted_suggestion"))

# 3D readback: the accepted residue colored by its NEW aa, in BOTH copies
charge = get_mode("charge")
exp_hex = charge.color_for(top.get("to_aa")) or "#ffffff"
exp = tuple(int(exp_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
cmds = panel.color_commands_for(tab)
ctrl.run_commands(cmds)
for (m, c) in cd.members:
    got = read_colors(m, c, [acc_resnum]).get(acc_resnum)
    ok = near(got, exp)
    checks.append((f"3D recolored #{m}/{c}:{acc_resnum} = {exp}", ok))
    print(f"[3D] #{m}/{c}:{acc_resnum} expect {exp} got {got} : {'PASS' if ok else 'FAIL'}")

# ── clean up: drain async color/select workers, then close the 1HSG we opened ─────────
QtCore.QThreadPool.globalInstance().waitForDone(4000)
run(f"close #{MID}")
print(f"[cleanup] closed model #{MID}")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
