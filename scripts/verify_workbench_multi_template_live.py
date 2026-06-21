"""
Live-verify — STAGE 1 manual MULTI-TEMPLATE / family guided fold (the build's end-to-end gate),
through the SHIPPED panel→engine→assist path on real ChimeraX + the local Boltz env.

The bridge-level Boltz N>1 behavior is confirmed separately (the first live-verify); this drives
the FULL shipped flow: a de-novo construct (CspB/1MJC, a hard-ish natural sequence) folded
UNGUIDED, then GUIDED by a manually-picked 2-template family (soft-consensus), then the template
assist — confirming the honesty layer ships only USE-TIME-knowable signals (ΔpLDDT, cross-seed Δ,
adoption) and NEVER claims rescue-confirmed.

Run: venv/Scripts/python.exe scripts/verify_workbench_multi_template_live.py   (several Boltz folds — minutes)
"""
import os, sys, types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("DEVIATION_FLOOR_N", "2")     # 1 reference + 1 floor seed → tractable assist
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
try:
    bridge.start(timeout=120)
    print("[chimerax] REST server up on :60001")
except Exception as exc:
    print(f"[abort] could not start ChimeraX: {exc}"); sys.exit(2)
run = bridge.run_command


def _seq_from_pdb(pid):
    """CspB sequence via the eval harness's Biopython reader (ChimeraX-free)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", str(Path(__file__).resolve().parent /
                                                  "eval_template_guided_calibration.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.chain_seq_from_pdb(m.download(pid), "A")


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


cspb = _seq_from_pdb("1MJC")
print(f"[harvest] CspB {len(cspb)} aa")

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
panel._add_sequence_construct("cspb", cspb)
cd = next(iter(panel._design.chains.values()))

# 0) UNGUIDED baseline
print("[fold] UNGUIDED baseline (Boltz monomer)…")
uspec = panel.construct_fold_launch_spec("boltz", 1)
drive(uspec, lambda r: panel.apply_construct_fold_result(uspec, r))
checks.append(("(0) unguided baseline folded", bool(cd.template_fold.get("model_id"))))

# 1) GUIDED by a 2-template FAMILY (soft-consensus), manual selection
print("[fold] GUIDED by a 2-template family (1C9O + 1G6P, soft)…")
refs = [{"pdb_id": "1C9O", "label": "1C9O", "force": False},
        {"pdb_id": "1G6P", "label": "1G6P", "force": False}]
gspec = panel.construct_fold_guided_spec("boltz", 1, refs)
print(f"   spec templates N = {len(gspec['tool_inputs']['templates'])}  labels={gspec['_guided_template']['labels']}")
checks.append(("(1a) spec carries N=2 soft entries",
               len(gspec["tool_inputs"]["templates"]) == 2
               and all("force" not in e for e in gspec["tool_inputs"]["templates"])))
pres.warnings.clear()
drive(gspec, lambda r: panel.apply_construct_fold_guided_result(gspec, r))
gf = dict(cd.guided_fold)
errs = [w for w in pres.warnings if "ERR" in w or "fail" in w.lower()]
print(f"   guided family fold #{gf.get('model_id')} pLDDT {gf.get('mean_plddt')} "
      f"templated={gf.get('templated')} n_templates_stored={len(gf.get('templates') or [])} | warnings={errs[:2]}")
checks.append(("(1b) guided FAMILY fold ran (Boltz honored N>1, real model)",
               bool(gf.get("model_id")) and gf.get("templated") is True
               and len(gf.get("templates") or []) == 2 and not errs))

# 2) Template assist — USE-TIME signals + adoption, NO rescue-confirmed
print("[assist] template assist (use-time signals + adoption) — minutes…")
aspec = panel.template_assist_launch_spec()
drive(aspec, lambda r: panel.apply_template_assist_result(aspec, r))
ta = dict(cd.template_assist)
print(f"   assist: ΔpLDDT={ta.get('d_plddt')} meanΔflex={ta.get('mean_d_flex')} "
      f"max_adoption={ta.get('max_adoption')} per_template={[(p.get('label'), p.get('adoption')) for p in (ta.get('per_template') or [])]}")
status = panel._status.text().lower()
print(f"   readout: {panel._status.text()}")
checks.append(("(2a) assist computed use-time signals (ΔpLDDT + Δflex + adoption)",
               ta.get("d_plddt") is not None and ta.get("mean_d_flex") is not None
               and ta.get("max_adoption") is not None))
checks.append(("(2b) readout ships use-time framing, NEVER 'rescue confirmed'",
               "not a correctness claim" in status and "rescue confirmed" not in status))
checks.append(("(2c) per-template adoption reported for the family (N=2)",
               len(ta.get("per_template") or []) == 2))

# 3) persist / restore
import json as _json
from variant_model import DesignSession
try:
    dd = session.get_design_session(panel._design.model_id)
    restored = DesignSession.from_dict(_json.loads(_json.dumps(dd)))
    rcd = next(iter(restored.chains.values()))
    rt_ok = bool(rcd.guided_fold.get("templates")) and bool(rcd.template_assist)
except Exception as exc:
    print(f"[persist] error: {exc}"); rt_ok = False
checks.append(("(3) guided family + assist persist/restore", rt_ok))

QtCore.QThreadPool.globalInstance().waitForDone(5000)
print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
