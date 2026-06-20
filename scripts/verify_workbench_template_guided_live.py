"""
Live-verify — TEMPLATE-GUIDED fold (first build) — the EXPERIMENT + the gate. Against the REAL
local Boltz env (~/boltz_env, LOCAL-ONLY), the REAL US-align binary, and REAL ChimeraX (:60001).

The probe confirmed Boltz 2.2.1 ACCEPTS a `templates:` block; this confirms it STEERS the fold and
that the assist measures it honestly. Construct = the myoglobin (1MBN) sequence; template = 4HHB
(hemoglobin alpha — a real structural HOMOLOG, ~27% id, same globin fold) — the realistic
"guide a sequence-first construct by a homolog" use.

  0. Fold the construct UNGUIDED (Boltz monomer, MSA-free) → baseline pLDDT.
  1. Fold GUIDED-SOFT by 4HHB (force:false) → (a) runs, Boltz honored the block (no error).
  2. Template assist (soft): ΔpLDDT + per-residue Δflexibility vs the unguided baseline →
     surfaces guided AND unguided AND the delta (d).
  3. US-align(guided-soft, 4HHB) → TM adoption (c): did the guided fold adopt the template?
  4. Fold GUIDED-HARD by 4HHB (force:true, threshold 10 A) → assist + US-align again.
  5. SOFT vs HARD head-to-head: does hard steering move the fold MORE than soft?
  6. Persist/restore the assist result.

(b) "measurable Δ" is REPORTED honestly — improvement OR negligible (a globin folds well MSA-free,
so a small Δ is an expected, honest null that reshapes the family stage). The gate PASSes on the
MECHANISM (guided runs, adoption TM computed, readout integrity, soft/hard both run, persist).

Run: venv/Scripts/python.exe scripts/verify_workbench_template_guided_live.py    (several Boltz folds — many minutes)
"""
import os, sys, types, tempfile
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
# Launch (or reuse) a stable ChimeraX with the REST server — the app isn't hosting it here.
try:
    bridge.start(timeout=120)
    print("[chimerax] REST server up on :60001")
except Exception as exc:
    print(f"[abort] could not start ChimeraX: {exc}"); sys.exit(2)
run = bridge.run_command


def models():
    import re
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _runscript(script: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    try:
        return run(f'runscript "{path}"').get("value") or ""
    finally:
        try: os.unlink(path)
        except OSError: pass


def chain_sequences(model):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        "        for ch in sorted(set(r.chain_id for r in m.residues)):\n"
        "            rs = [r for r in m.residues if r.chain_id==ch and r.polymer_type==r.PT_AMINO]\n"
        "            rs.sort(key=lambda r: r.number)\n"
        "            seq = ''.join((r.one_letter_code or 'X') for r in rs)\n"
        "            if seq: print('SEQ', ch, seq)\n"
        "        break\n")
    seqs = {}
    for line in out.splitlines():
        p = line.strip().split()
        if len(p) == 3 and p[0] == "SEQ":
            seqs[p[1]] = p[2]
    return seqs


class ScriptedPresenter(Presenter):
    def __init__(self):
        self.warnings = []; self.confirm_calls = []; self.confirm_answer = "proceed"
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
    def confirm(self, confidence):
        self.confirm_calls.append(confidence); return self.confirm_answer
    def ask_edit(self, original): return list(original)
    def ask_yes_no(self, q, default="y"): return False


# ── harvest the myoglobin sequence ────────────────────────────────────────────────────
run("close session")
before = set(models())
run("open 1mbn")
MBN = (sorted(set(models()) - before, key=int) or ["1"])[-1]
myo = chain_sequences(MBN).get("A", "")
run(f"close #{MBN}")
if not myo:
    print("[abort] could not harvest the myoglobin sequence from 1MBN"); sys.exit(2)
print(f"[harvest] myoglobin {len(myo)} aa")

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
    print("[abort] Boltz env unavailable — the guided fold needs it."); sys.exit(2)


def drive(spec, on_apply, timeout=3_600_000):
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
    captured = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if captured:
        on_apply(captured[0])
    return captured


TEMPLATE = "4HHB"        # hemoglobin alpha — real structural homolog of myoglobin (same globin fold)
checks = []

# ── 0) UNGUIDED baseline fold (Boltz monomer, MSA-free) ───────────────────────────────
panel._add_sequence_construct("myoglobin", myo)
cd = next(iter(panel._design.chains.values()))
print("[fold] UNGUIDED baseline fold (Boltz monomer, MSA-free) — minutes…")
uspec = panel.construct_fold_launch_spec("boltz", 1)
drive(uspec, lambda r: panel.apply_construct_fold_result(uspec, r))
u_mid = cd.template_fold.get("model_id")
u_plddt = cd.template_fold.get("mean_plddt")
print(f"[fold] unguided #{u_mid} mean pLDDT {u_plddt}")
checks.append(("(0) unguided baseline folded (model + cif on disk)",
               bool(u_mid) and bool(cd.template_fold.get("cif_path") or cd.template_fold.get("pdb_path"))))


def guided_fold(force, label):
    ref = {"pdb_id": TEMPLATE, "label": TEMPLATE, "force": force}
    if force:
        ref["threshold"] = 10.0
    spec = panel.construct_fold_guided_spec("boltz", 1, ref)
    if spec is None:
        print(f"[abort] guided spec ({label}) was None"); return None
    pres.warnings.clear()
    drive(spec, lambda r: panel.apply_construct_fold_guided_result(spec, r))
    gf = dict(cd.guided_fold)
    errs = [w for w in pres.warnings if "ERR" in w or "fail" in w.lower()]
    print(f"[guided/{label}] #{gf.get('model_id')} mean pLDDT {gf.get('mean_plddt')} "
          f"templated={gf.get('templated')} force={gf.get('force')} thr={gf.get('threshold')} "
          f"| warnings={errs[:2]}")
    return gf


def assist():
    spec = panel.template_assist_launch_spec()
    if spec is None:
        print("[abort] assist spec was None"); return None
    drive(spec, lambda r: panel.apply_template_assist_result(spec, r))
    return dict(cd.template_assist)


def validate_adoption():
    spec = panel.structural_align_launch_spec(reference_pdb_id=TEMPLATE, use_guided=True)
    drive(spec, lambda r: panel.apply_structural_align_result(spec, r))
    return dict(cd.structural_align)


# ── 1) GUIDED SOFT (force:false) ───────────────────────────────────────────────────────
print(f"\n[soft] folding GUIDED-SOFT by {TEMPLATE} (force:false) — minutes…")
soft_gf = guided_fold(False, "soft")
checks.append(("(a) guided-SOFT fold RAN (Boltz honored the templates block, real model)",
               bool(soft_gf and soft_gf.get("model_id")) and soft_gf.get("templated") is True))

# ── 2) assist (soft) ───────────────────────────────────────────────────────────────────
print("[soft] template assist (cross-seed floors for both folds) — minutes…")
soft_assist = assist()
if soft_assist:
    print(f"[soft] assist: ΔpLDDT={soft_assist.get('d_plddt')} "
          f"(unguided {soft_assist.get('unguided_mean_plddt')} → guided {soft_assist.get('guided_mean_plddt')}); "
          f"meanΔflex={soft_assist.get('mean_d_flex')} Å; "
          f"{soft_assist.get('n_stabilized')}/{soft_assist.get('n_residues')} stabilized")
checks.append(("(d) assist surfaces guided AND unguided AND the delta",
               bool(soft_assist) and soft_assist.get("guided_mean_plddt") is not None
               and soft_assist.get("unguided_mean_plddt") is not None
               and soft_assist.get("d_plddt") is not None))

# ── 3) US-align adoption (soft) ────────────────────────────────────────────────────────
print(f"[soft] US-align(guided, {TEMPLATE}) adoption…")
soft_adopt = validate_adoption()
soft_tm = soft_adopt.get("tm_ref")
print(f"[soft] adoption TM_ref={soft_tm} TM_query={soft_adopt.get('tm_query')} "
      f"shared={soft_adopt.get('shared_fold')}")
checks.append(("(c) US-align(guided-soft, template) TM computed (adoption metric captured)",
               soft_tm is not None))
soft_d_plddt = soft_assist.get("d_plddt") if soft_assist else None
soft_flex = soft_assist.get("mean_d_flex") if soft_assist else None

# ── 4) GUIDED HARD (force:true, threshold 10 Å) ───────────────────────────────────────
print(f"\n[hard] folding GUIDED-HARD by {TEMPLATE} (force:true, threshold 10 Å) — minutes…")
hard_gf = guided_fold(True, "hard")
checks.append(("(a) guided-HARD fold RAN (force:true honored, real model)",
               bool(hard_gf and hard_gf.get("model_id")) and hard_gf.get("force") is True))
print("[hard] template assist — minutes…")
hard_assist = assist()
if hard_assist:
    print(f"[hard] assist: ΔpLDDT={hard_assist.get('d_plddt')}; "
          f"meanΔflex={hard_assist.get('mean_d_flex')} Å; "
          f"{hard_assist.get('n_stabilized')}/{hard_assist.get('n_residues')} stabilized")
print(f"[hard] US-align(guided, {TEMPLATE}) adoption…")
hard_adopt = validate_adoption()
hard_tm = hard_adopt.get("tm_ref")
print(f"[hard] adoption TM_ref={hard_tm} shared={hard_adopt.get('shared_fold')}")
checks.append(("(c) US-align(guided-hard, template) TM computed", hard_tm is not None))

# ── 5) SOFT vs HARD head-to-head (the steering-strength finding) ──────────────────────
print("\n── SOFT vs HARD ──")
print(f"  soft: ΔpLDDT={soft_d_plddt}  meanΔflex={soft_flex}  adoptionTM={soft_tm}")
print(f"  hard: ΔpLDDT={hard_assist.get('d_plddt') if hard_assist else None}  "
      f"meanΔflex={hard_assist.get('mean_d_flex') if hard_assist else None}  adoptionTM={hard_tm}")
# Both ran end-to-end with distinct steering provenance → the soft/hard lever is exercised.
checks.append(("(5) soft AND hard both ran end-to-end with distinct provenance",
               bool(soft_gf) and bool(hard_gf)
               and soft_gf.get("force") is False and hard_gf.get("force") is True
               and soft_tm is not None and hard_tm is not None))

# ── 6) persist / restore the assist ────────────────────────────────────────────────────
dd = session.get_design_session(panel._design.model_id)
import json as _json
roundtrips = False
try:
    blob = _json.loads(_json.dumps(dd))      # JSON-roundtrip the persisted design
    from variant_model import DesignSession
    restored = DesignSession.from_dict(blob)
    rcd = next(iter(restored.chains.values()))
    roundtrips = bool(rcd.template_assist) and rcd.template_assist.get("d_plddt") == (
        hard_assist.get("d_plddt") if hard_assist else "X")
except Exception as exc:
    print(f"[persist] restore error: {exc}")
print(f"[persist] assist persisted+restored: {roundtrips}")
checks.append(("(6) assist result persists + restores (JSON roundtrip)", roundtrips))

QtCore.QThreadPool.globalInstance().waitForDone(5000)
print("[cleanup] leaving models open for inspection")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\n--- HONEST READOUT (b) — did the template measurably steer the fold? ---")
print(f"  unguided baseline mean pLDDT : {u_plddt}")
print(f"  soft  ΔpLDDT / meanΔflex / adoptionTM : {soft_d_plddt} / {soft_flex} / {soft_tm}")
print(f"  hard  ΔpLDDT / meanΔflex / adoptionTM : "
      f"{hard_assist.get('d_plddt') if hard_assist else None} / "
      f"{hard_assist.get('mean_d_flex') if hard_assist else None} / {hard_tm}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
