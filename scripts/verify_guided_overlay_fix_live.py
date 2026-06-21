"""
Live-verify — the guided-fold OVERLAY/TOGGLE fix + the immediate ADOPTION readout, against the
running ChimeraX + local Boltz. Targets the reported bug: guided re-folds accumulate overlaid +
untoggleable, and no "did it reflect the template" feedback.

  0. de-novo PCNA construct (1AXC chain A) → fold UNGUIDED (baseline).
  1. fold GUIDED by 1AXC (soft) → IMMEDIATE adoption readout (structTM guided-vs-1AXC = "does it
     reflect the template?") + record the guided model id.
  2. RE-FOLD guided HARD → the prior SOFT guided model must be CLOSED (replace-on-refold; no
     accumulation), and a NEW guided model present.
  3. fold_visibility_commands with "Hide folds" → emits `hide #<guided>` (the toggle now reaches it).

Reuses the running ChimeraX (does NOT close session). Run: venv/Scripts/python.exe scripts/verify_guided_overlay_fix_live.py
"""
import os, sys, types, re
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

bridge = ChimeraXBridge(port=60001)
if not bridge.is_running():
    print("[abort] ChimeraX not reachable on :60001 — launch the app first."); sys.exit(2)
run = bridge.run_command


def live_model_ids():
    return set(re.findall(r"model id #(\d+(?:\.\d+)*) ", (run("info models").get("value") or "")))


def _pcna_seq():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", str(Path(__file__).resolve().parent /
                                                  "eval_template_guided_calibration.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.chain_seq_from_pdb(m.download("1AXC"), "A")     # PCNA monomer


class ScriptedPresenter(Presenter):
    def __init__(self): self.warnings = []; self.confirm_answer = "proceed"
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
    def tool_summary(self, *a, **k): pass
    def analysis_panel(self, s): pass
    def command_result(self, *a, **k): pass
    def blocked(self, cmd, error): pass
    def completed(self, n): pass
    def translation_declined(self, exc): pass
    def translation_error(self, exc): pass
    def status(self, label):
        class _CM:
            def __enter__(s): return s
            def __exit__(s, *a): return False
        return _CM()
    def running_tools(self, label, eta_s=0.0, needs_timer=False): return self.status(label)
    def tool_status(self, m): pass
    def ask_clarification(self, q): return ""
    def confirm(self, confidence): return self.confirm_answer
    def ask_edit(self, original): return list(original)
    def ask_yes_no(self, q, default="y"): return False


pcna = _pcna_seq()
print(f"[harvest] PCNA (1AXC/A) {len(pcna)} aa")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
router = ToolRouter(bridge, session)
host = types.SimpleNamespace(
    bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None, _log_exchange=lambda *a, **k: None)
engine = RequestEngine(host)
pres = ScriptedPresenter()
panel = VariantWorkbenchPanel(ctrl, session=session)
if not panel._fold_engine_availability().get("boltz"):
    print("[abort] Boltz env unavailable."); sys.exit(2)


def drive(spec, on_apply, timeout=3_600_000):
    captured = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if captured:
        on_apply(captured[0])
    return captured


checks = []
panel._add_sequence_construct("pcna", pcna)
cd = next(iter(panel._design.chains.values()))

print("[fold] UNGUIDED PCNA baseline…")
uspec = panel.construct_fold_launch_spec("boltz", 1)
drive(uspec, lambda r: panel.apply_construct_fold_result(uspec, r))

print("[fold] GUIDED by 1AXC (soft)…")
gspec = panel.construct_fold_guided_spec("boltz", 1, {"pdb_id": "1AXC", "label": "1AXC", "force": False})
drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
soft_mid = str(cd.guided_fold.get("model_id"))
soft_adopt = cd.guided_fold.get("adoption")
print(f"   soft guided #{soft_mid}  pLDDT {cd.guided_fold.get('mean_plddt')}  ADOPTION(structTM vs 1AXC)={soft_adopt}")
print(f"   readout: {panel._status.text()}")
checks.append(("(1) immediate ADOPTION computed at fold time (does it reflect 1AXC?)",
               isinstance(soft_adopt, (int, float))))
checks.append(("(1b) readout reports the adoption %", "adopted the template at" in panel._status.text().lower()))

print("[fold] RE-FOLD guided HARD → must CLOSE the prior soft model (replace-on-refold)…")
hspec = panel.construct_fold_guided_spec("boltz", 1, {"pdb_id": "1AXC", "label": "1AXC", "force": True})
drive(hspec, lambda r: panel.apply_construct_fold_guided_result(hspec, r))
QtCore.QThreadPool.globalInstance().waitForDone(8000)   # let the bg close land
hard_mid = str(cd.guided_fold.get("model_id"))
live = live_model_ids()
print(f"   hard guided #{hard_mid}; soft #{soft_mid} still open? {soft_mid in live}  (live models: {sorted(live)})")
checks.append(("(2) replace-on-refold CLOSED the prior soft guided model (no accumulation)",
               hard_mid != soft_mid and soft_mid not in live and hard_mid in live))

print("[viz] Hide folds → guided model must get a hide command…")
panel._fold_vis_btn.setChecked(True)
cmds = panel.fold_visibility_commands(panel._cur_tab())
print(f"   visibility cmds: {cmds}")
checks.append(("(3) 'Hide folds' reaches the guided fold (toggle-able, not stuck)",
               any(f"hide #{hard_mid} models" == c for c in cmds)))

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print(f"\nADOPTION (soft 1AXC) = {soft_adopt}  — high = the guided PCNA fold DID reflect the 1AXC template")
print("RESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
