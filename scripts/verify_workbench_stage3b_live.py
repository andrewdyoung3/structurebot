"""
Live-verify the Variant-Design Workbench Stage 3b against a REAL running ChimeraX
(REST :60001): tools LAUNCHED FROM THE PANEL go through the SAME engine spine as the
NL path (engine.handle_tool_request -> router.route -> router.execute), honoring the
real safeguards — NOT a parallel invocation.

Proves, on a freshly-opened 1HSG homodimer:
  1. DEEP-GATE: a panel-built deep (Rosetta) scan spec enters the spine; the tiering
     surfaces the runtime ESTIMATE and the confirm-gate is asked with confidence='low'
     (no auto-proceed). We answer CANCEL → the Rosetta subprocess NEVER launches
     (proves "an hours-long job never silently launches").
  2. FAST scan: a panel-built fast scan spec runs the REAL mutation_scanner through the
     spine to completion; the session cache it wrote is what the panel reads back, and
     _load_suggestions() renders the Suggest track at exactly the scanned resnums.
  3. MPNN: a panel-built ProteinMPNN spec runs the REAL subprocess through the spine;
     _import_mpnn() lands the designs as variant rows.
  4. ACCEPT: cherry-pick a suggestion into the active variant → that residue recolors in
     BOTH homo-oligomer copies (A+B), proven by reading ribbon_color BACK.

It proves the REAL spine (not a parallel launch): the engine's router IS the production
ToolRouter, and the results the panel renders come from the session cache that
router.execute() populated.

Freshly opens 1HSG, then closes it (non-destructive to the user's session).
Run: venv/Scripts/python.exe scripts/verify_workbench_stage3b_live.py
"""
import os, sys, tempfile, types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from PySide6 import QtCore, QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from tool_router import ToolRouter
from request_engine import RequestEngine
from presenter import Presenter
from variant_workbench import VariantWorkbenchPanel
from color_modes import get_mode

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command


def models():
    import re
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


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


# ── a scripted presenter: records output + answers the confirm-gate on cue ────────────
class ScriptedPresenter(Presenter):
    def __init__(self):
        self.warnings = []
        self.confirm_calls = []     # (confidence) each time the gate is hit
        self.confirm_answer = "proceed"
    # output
    def info(self, t): pass
    def warn(self, t): self.warnings.append(t)
    def error(self, t): self.warnings.append("ERR:" + t)
    def success(self, t): pass
    def dim(self, t): pass
    def blank(self): pass
    def markup(self, t): pass
    def active_site_ok(self, m): pass
    def show_commands(self, c, e, conf): pass
    def show_tool_pipeline(self, r): pass
    def show_interface_summary(self, r): pass
    def tool_summary(self, icon, ok, summary="", tool="", error=""): pass
    def analysis_panel(self, s): pass
    def command_result(self, cmd, ok, value=None, error=None, warning=None): pass
    def blocked(self, cmd, error): pass
    def completed(self, n): pass
    def translation_declined(self, exc): pass
    def translation_error(self, exc): pass
    def status(self, label):
        class _CM:
            def __enter__(s): return s
            def __exit__(s, *a): return False
        return _CM()
    def running_tools(self, label, eta_s=0.0, needs_timer=False):
        return self.status(label)
    def tool_status(self, m): pass
    # input
    def ask_clarification(self, q): return ""
    def confirm(self, confidence):
        self.confirm_calls.append(confidence)
        return self.confirm_answer
    def ask_edit(self, original): return list(original)
    def ask_yes_no(self, q, default="y"): return False


# ── open a fresh 1HSG ─────────────────────────────────────────────────────────────────
before = set(models())
run("open 1hsg")
new = sorted(set(models()) - before, key=int)
MID = new[-1] if new else "1"
print(f"[setup] opened 1HSG as model #{MID}")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})

# Register the structure the way the production open flow does (_maybe_update_structure_state)
# so the spine resolves the PDB + sequence via session.get_structure — name "1hsg" is a
# 4-char PDB id (RCSB fallback) and we also hand it a saved local path.
pdb_path = (Path(tempfile.gettempdir()) / f"s3b_1hsg_{MID}.pdb").as_posix()
run(f'save "{pdb_path}" format pdb models #{MID}')
session.add_structure(MID, "1hsg", path=pdb_path)
print(f"[setup] registered model #{MID} in session (path + RCSB id '1hsg')")

# the engine host: REAL router/session/bridge (the production spine); translator is a
# stub (handle_tool_request never translates), hooks are no-ops.
router = ToolRouter(bridge, session)
host = types.SimpleNamespace(
    bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None,
    _log_exchange=lambda *a, **k: None,
)
engine = RequestEngine(host)
pres = ScriptedPresenter()

# the panel shares the SAME session, so it reads back exactly what execute() cached
panel = VariantWorkbenchPanel(ctrl, session=session)
panel.load_model(MID)
tab = panel._cur_tab()
cd = tab.design
print(f"[panel] unique chain tab rep=#{cd.rep_model}/{cd.rep_chain} members={cd.members} "
      f"({len(cd.template_cells)} cols)")

resnums_A = [c.resnum for c in cd.template_cells if c.resnum is not None]
scan_cols = [i for i, c in enumerate(cd.template_cells) if c.resnum in
             {resnums_A[k] for k in (9, 24, 49) if k < len(resnums_A)}][:3]
panel._scan_cols = set(scan_cols)
scan_pos = panel._scan_set_resnums(cd)
print(f"[setup] chain A: {len(resnums_A)} residues; scan/design positions = {scan_pos}")

checks = []
checks.append(("homo-oligomer collapsed (A+B)", len(cd.members) >= 2))
checks.append(("engine.router IS the production ToolRouter", isinstance(engine.router, ToolRouter)))


def launch(spec):
    """Drive the panel-built spec through the REAL engine spine."""
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"))


# ── 1) DEEP-GATE: estimate + confirm('low'), answer CANCEL → no Rosetta launch ────────
print("[gate] launching a DEEP (Rosetta) scan spec; will CANCEL at the gate...")
pres.warnings.clear(); pres.confirm_calls.clear(); pres.confirm_answer = None
deep_spec = panel.scan_launch_spec(deep=True)
print(f"[gate] spec confidence={deep_spec['confidence']} "
      f"run_rosetta={deep_spec['tool_inputs'].get('run_rosetta')}")
launch(deep_spec)
estimate_shown = any("approximate runtime" in w.lower() for w in pres.warnings)
gate_asked_low = pres.confirm_calls == ["low"]
no_scan_yet = session.get_scan_result(MID) is None
print(f"[gate] estimate_shown={estimate_shown} confirm_calls={pres.confirm_calls} "
      f"scan_ran={not no_scan_yet}")
checks.append(("deep tier surfaced a runtime estimate", estimate_shown))
checks.append(("confirm-gate asked with confidence='low' (no auto-proceed)", gate_asked_low))
checks.append(("CANCEL at gate → Rosetta NEVER launched", no_scan_yet))

# ── 2) FAST scan through the spine → auto-render Suggest track ─────────────────────────
print("[scan] launching a FAST scan spec; will PROCEED...")
pres.warnings.clear(); pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
launch(panel.scan_launch_spec(deep=False))
scan_cached = session.get_scan_result(MID)
scanned_resnums = sorted({c["resnum"] for c in (scan_cached or [])})
print(f"[scan] session cache has {len(scan_cached or [])} candidates at {scanned_resnums}")
checks.append(("FAST scan ran through the spine (session cached)", bool(scan_cached)))
panel._load_suggestions()
sugg_resnums = sorted({cd.resnum_for_col(c) for c in tab.suggestions})
print(f"[scan] Suggest track at resnums {sugg_resnums} (scanned {scanned_resnums})")
checks.append(("Suggest track auto-rendered at the scanned resnums",
               bool(sugg_resnums) and sugg_resnums == scanned_resnums))

# ── 3) MPNN through the spine → auto-import rows ───────────────────────────────────────
print("[mpnn] launching a ProteinMPNN spec; will PROCEED...")
pres.confirm_answer = "proceed"
launch(panel.mpnn_launch_spec(soluble=False))
mpnn = session.get_proteinmpnn_result(MID)
n_designs = len(mpnn.get("sequences", [])) if mpnn else 0
print(f"[mpnn] session cached designs={n_designs}")
checks.append(("ProteinMPNN ran through the spine (session cached)", n_designs > 0))
panel._import_mpnn()
mpnn_rows = [v for v in cd.variants if v.source == "proteinmpnn"]
print(f"[mpnn] imported {len(mpnn_rows)} variant row(s)")
checks.append(("MPNN designs auto-imported as rows", len(mpnn_rows) == n_designs and n_designs > 0))

# ── 4) ACCEPT a suggestion → recolor BOTH copies ──────────────────────────────────────
panel._add_variant()
vid = cd.variants[-1].id
pick_col = sorted(tab.suggestions)[0]
top = tab.suggestions[pick_col][0]
panel._mode_combo.setCurrentIndex(
    next(i for i in range(panel._mode_combo.count())
         if panel._mode_combo.itemData(i) == "charge"))
panel._accept_suggestion(tab, pick_col, top)
acc_resnum = cd.resnum_for_col(pick_col)
v = cd.get_variant(vid)
mut = next((m for m in v.mutations if m.resnum == acc_resnum), None)
print(f"[accept] {vid}: resnum {acc_resnum} -> {top.get('to_aa')} "
      f"(score {top.get('combined_score'):+.2f}); source={mut.source if mut else None}")
checks.append(("accepted suggestion recorded with provenance",
               mut is not None and mut.source == "accepted_suggestion"))

charge = get_mode("charge")
exp_hex = charge.color_for(top.get("to_aa")) or "#ffffff"
exp = tuple(int(exp_hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
ctrl.run_commands(panel.color_commands_for(tab))
for (m, c) in cd.members:
    got = read_colors(m, c, [acc_resnum]).get(acc_resnum)
    ok = near(got, exp)
    checks.append((f"3D recolored #{m}/{c}:{acc_resnum} = {exp}", ok))
    print(f"[3D] #{m}/{c}:{acc_resnum} expect {exp} got {got} : {'PASS' if ok else 'FAIL'}")

# ── cleanup ───────────────────────────────────────────────────────────────────────────
QtCore.QThreadPool.globalInstance().waitForDone(4000)
run(f"close #{MID}")
print(f"[cleanup] closed model #{MID}")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
