"""
Live-verify Stage 2a — de-novo construct VARIANTS + DEVIATION — against REAL ChimeraX (:60001) +
the REAL local fold engines (~/boltz_env Boltz; venv312 ESMFold). Proves a de-novo construct's
T-fold is REUSED as the deviation WT reference (no fresh fold of T) and as seed-0 of the cross-seed
floor (only N-1 extra seeds folded), that a variant folds at the construct's engine + oligomer and
superposes on the T-fold, and that the indel column-pairing holds — the case the offline probe
could not prove.

PART 1 — Boltz HOMO-DIMER construct (the load-bearing reuse + floor gate):
  1. Add a sequence → fold the construct as a Boltz dimer (T-fold = the displayed reference).
  2. Substitution variant → fold it; it PINS the construct's engine + oligomer (Boltz dimer) and
     superposes (matchmaker) onto the T-fold, NOT the synthetic id.
  3. Deviation vs WT:
     (a) REUSES the displayed T-fold as the reference — reference_model_id == the T-fold id and
         NO net-new persistent model spawns (no fresh fold of T).
     (b) the floor is MEASURED from ONLY the N-1 extra seeds (instrumented: boltz.predict is called
         exactly DEVIATION_FLOOR_N-1 times during the deviation; the T-fold is seed-0, reused).
     (c) the dRMSD/lDDT tiers read variant-vs-T-fold and the 3D push paints the variant model.
  4. Persist → restore: members + per-chain T-fold + the cached WT reference + deviation survive.

PART 2 — ESMFold MONOMER construct (the indel column-pairing, fast/deterministic):
  5. Fold a monomer construct; a variant with a DELETION + an INSERTION → fold monomer → deviation
     reuses the monomer T-fold (caps floor), the column-pairing pairs post-indel residues, the
     deleted position is absent, and the inserted residue is dropped (neutral).

Run: venv/Scripts/python.exe scripts/verify_workbench_denovo_variants_deviation_live.py  (folds — minutes)
"""
import os, sys, types
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
from variant_workbench import VariantWorkbenchPanel, _RESULT_DEVIATION_MODE, _DDM_FLOOR_MIN_A
from config import DEVIATION_FLOOR_N

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command

# Two real, well-folding small domains (de-novo here — sequences are just realistic input).
HP36 = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"                       # 36 aa → Boltz dimer (72 res)
GB1  = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"   # 56 aa → ESMFold monomer


def models():
    import re
    return set(re.findall(r"model id #(\d+) ", (run("info models").get("value") or "")))


def _runscript(script: str) -> str:
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    try:
        return run(f'runscript "{path}"').get("value") or ""
    finally:
        try: os.unlink(path)
        except OSError: pass


def _attr(model, expr, tag):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        f"        print('{tag}:', {expr}); break\n")
    for line in out.splitlines():
        if line.strip().startswith(tag + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def chains_of(model):
    v = _attr(model, "','.join(sorted(set(r.chain_id for r in m.residues)))", "CH")
    return [c for c in v.split(",") if c]


def ca_xyz(model, chain, resnum):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        "        for r in m.residues:\n"
        f"            if r.chain_id == '{chain}' and r.number == {resnum}:\n"
        "                a = r.find_atom('CA')\n"
        "                if a is not None:\n"
        "                    print('XYZ:', a.scene_coord[0], a.scene_coord[1], a.scene_coord[2])\n"
        "        break\n")
    for line in out.splitlines():
        if line.strip().startswith("XYZ:"):
            try: return tuple(float(x) for x in line.split(":")[1].split())
            except ValueError: return None
    return None


def dist(a, b):
    return None if a is None or b is None else sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


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


def drive(spec, on_apply, timeout=1_800_000):
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
    captured = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if captured:
        on_apply(captured[0])
    return captured


checks = []
avail = None

# ══ PART 1 — Boltz homo-dimer construct ═══════════════════════════════════════════════
panel = VariantWorkbenchPanel(ctrl, session=session)
avail = panel._fold_engine_availability()
if not avail.get("boltz"):
    print("[abort] Boltz env (~/boltz_env) unavailable — Stage 2a reuse/floor gate needs it.")
    sys.exit(2)

panel._add_sequence_construct("denovo_dimer", HP36)
cd = next(iter(panel._design.chains.values()))
tab = panel._cur_tab()
print("[part1] folding the de-novo construct as a Boltz DIMER — minutes…")
cspec = panel.construct_fold_launch_spec("boltz", 2)
before = models()
drive(cspec, lambda r: panel.apply_construct_fold_result(cspec, r))
t_mid = cd.template_fold.get("model_id")
print(f"[part1] T-fold #{t_mid} chains={chains_of(t_mid) if t_mid else None} "
      f"members={cd.members} target={cd.template_fold.get('target')}")
checks.append(("construct folded as a 2-chain assembly",
               bool(t_mid) and len(chains_of(t_mid)) == 2
               and cd.template_fold.get("target") == "assembly"))

# substitution variant → fold (pins the construct's engine + oligomer; superpose on the T-fold)
panel._add_variant()
v = cd.variants[-1]
tab.set_active_row(v.id)
cd.edit_variant(v.id, 0, "A" if v.cells[0].aa != "A" else "S")
fspec = panel.fold_launch_spec("esmfold")            # ask esmfold/monomer → must PIN boltz/dimer
print(f"[part1] variant fold spec: tool={fspec['tool']} chains={len(fspec['tool_inputs'].get('chains') or [])} "
      f"compare_to={fspec['tool_inputs'].get('compare_to')} (T-fold #{t_mid}, synthetic {panel._design.model_id})")
checks.append(("variant fold PINS engine=boltz + the dimer (GAP C)",
               fspec["tool"] == "boltz" and len(fspec["tool_inputs"].get("chains") or []) == 2))
checks.append(("variant fold compare_to = the T-fold, not the synthetic id (GAP A)",
               fspec["tool_inputs"].get("compare_to") == t_mid
               and fspec["tool_inputs"].get("compare_to") != panel._design.model_id))
print("[part1] folding the variant as a Boltz dimer (superpose on the T-fold) — minutes…")
drive(fspec, lambda r: panel.apply_fold_result(v.id, r))
vfold = cd.get_variant(v.id).results.fold or {}
v_mid = vfold.get("model_id")
checks.append(("variant folded at the construct's engine + oligomer",
               vfold.get("engine") == "boltz" and vfold.get("target") == "assembly"))
# superposition: rep-chain core CA of the variant fold near the T-fold (matchmaker ran)
core = [c.resnum for c in cd.template_cells if c.resnum is not None][6:-6]
sup = sorted(d for rn in core[::4]
             if (d := dist(ca_xyz(v_mid, cd.rep_chain, rn), ca_xyz(t_mid, cd.rep_chain, rn))) is not None)
print(f"[part1] variant↔T-fold core CA dists {[round(x,1) for x in sup[:6]]}")
checks.append(("variant fold superposed onto the T-fold (matchmaker; core CA < 8 Å)",
               bool(sup) and sup[0] < 8.0))

# ── Deviation vs WT — INSTRUMENTED to count the floor folds ───────────────────────────
dspec = panel.deviation_launch_spec()
print(f"[part1] deviation spec: confidence={dspec['confidence']} wt_ref.model_id={dspec['tool_inputs']['wt_ref'].get('model_id')} "
      f"floor_present={bool(dspec['tool_inputs']['wt_ref'].get('floor_ddm'))}")
checks.append(("deviation pre-seeds the T-fold as the WT reference (reuse, not None)",
               dspec["tool_inputs"]["wt_ref"].get("model_id") == t_mid
               and not dspec["tool_inputs"]["wt_ref"].get("floor_ddm")))
checks.append(("first deviation → confirm-gate (folds the N-1 floor seeds)",
               dspec["confidence"] == "low"))

# instrument the boltz bridge to count predict() calls DURING the deviation (the floor folds)
boltz_bridge = router._get_boltz_bridge()
_orig_predict = boltz_bridge.predict
fold_seeds = []
def _counting_predict(*a, **k):
    fold_seeds.append(k.get("seed"))
    return _orig_predict(*a, **k)
boltz_bridge.predict = _counting_predict
pre_dev_models = models()
print(f"[part1] Deviation vs WT — reuse the T-fold (seed-0) + fold ONLY the {DEVIATION_FLOOR_N-1} extra seeds — minutes…")
try:
    drive(dspec, lambda r: panel.apply_deviation_result(v.id, r))
finally:
    boltz_bridge.predict = _orig_predict
post_dev_models = models()
block = (cd.get_variant(v.id).results.fold or {}).get("deviation") or {}

new_persistent = post_dev_models - pre_dev_models
print(f"[part1] deviation folded seeds={fold_seeds}; ref={block.get('reference_model_id')} (T-fold #{t_mid}); "
      f"net-new persistent models={new_persistent}; floor_kind={block.get('floor_kind')}; "
      f"n_disrupted={block.get('n_disrupted')}/{block.get('n_residues')}")
# (a) NO re-fold of T — the reference is the reused T-fold, and nothing new persists
checks.append(("(a) reference IS the reused T-fold (reference_model_id == T-fold id)",
               str(block.get("reference_model_id")) == str(t_mid)))
checks.append(("(a) NO net-new persistent model from the deviation (no fresh fold of T)",
               len(new_persistent) == 0))
# (b) the floor folded ONLY the N-1 extra seeds (T reused as seed-0)
checks.append((f"(b) deviation folded exactly N-1={DEVIATION_FLOOR_N-1} floor seeds (T reused as seed-0)",
               len(fold_seeds) == DEVIATION_FLOOR_N - 1))
measured = [x for x in (block.get("floor_ddm") or {}).values() if x > _DDM_FLOOR_MIN_A]
checks.append(("(b) the floor is MEASURED cross-seed spread (some residues above the global min)",
               block.get("floor_kind") == "measured" and len(measured) >= 1))
# (c) tiers + 3D paint
panel._mode_key = _RESULT_DEVIATION_MODE
cmds = panel.color_commands_for(tab)
checks.append(("(c) dRMSD + lDDT tiers computed variant-vs-T-fold",
               bool(block.get("ddm")) and bool(block.get("lddt"))))
checks.append(("(c) 3D push paints the predicted variant model",
               bool(v_mid) and any(f"#{v_mid}/" in c for c in cmds)))

# ── persist → restore (dimer construct) ───────────────────────────────────────────────
dd = session.get_design_session(panel._design.model_id)
panel_r = VariantWorkbenchPanel(ctrl, session=session)
panel_r.rehydrate_denovo(dd)
cd_r = next(iter(panel_r._design.chains.values()))
vr = cd_r.get_variant(v.id)
print(f"[part1] restore: members={cd_r.members} wt_refs={list(cd_r.wt_refs)} "
      f"deviation={bool((vr.results.fold or {}).get('deviation')) if vr else None}")
checks.append(("restore: members still point at the T-fold chains",
               cd_r.rep_model == t_mid and len(cd_r.members) == 2))
checks.append(("restore: the cached WT reference + the deviation block survive",
               bool(cd_r.wt_refs.get(f"boltz:assembly"))
               and bool((vr.results.fold or {}).get("deviation"))))

# ══ PART 2 — ESMFold monomer construct: indel column-pairing ══════════════════════════
if avail.get("esmfold"):
    panel_m = VariantWorkbenchPanel(ctrl, session=session)
    panel_m._add_sequence_construct("denovo_mono", GB1)
    cdm = next(iter(panel_m._design.chains.values()))
    tabm = panel_m._cur_tab()
    print("[part2] folding a de-novo MONOMER construct (ESMFold)…")
    cms = panel_m.construct_fold_launch_spec("esmfold", 1)
    drive(cms, lambda r: panel_m.apply_construct_fold_result(cms, r))
    tm_mid = cdm.template_fold.get("model_id")
    # variant with a DELETION (col 10) + an INSERTION (after col 20)
    panel_m._add_variant()
    vm = cdm.variants[-1]
    tabm.set_active_row(vm.id)
    cdm.edit_variant(vm.id, 0, "A" if vm.cells[0].aa != "A" else "S")
    cdm.delete_variant_residue(vm.id, 10)               # a deletion (downstream pairs to pos+1)
    cdm.insert_variant_residues(vm.id, 20, "GG")        # an insertion (no WT counterpart → dropped)
    print("[part2] folding the indel variant (ESMFold monomer)…")
    drive(panel_m.fold_launch_spec("esmfold"), lambda r: panel_m.apply_fold_result(vm.id, r))
    dms = panel_m.deviation_launch_spec()
    print(f"[part2] deviation wt_ref.model_id={dms['tool_inputs']['wt_ref'].get('model_id')} (T-fold #{tm_mid}); "
          f"fold_column_map size={len(dms['tool_inputs'].get('fold_column_map') or {})}")
    checks.append(("indel deviation reuses the monomer T-fold reference",
                   dms["tool_inputs"]["wt_ref"].get("model_id") == tm_mid))
    drive(dms, lambda r: panel_m.apply_deviation_result(vm.id, r))
    mblock = (cdm.get_variant(vm.id).results.fold or {}).get("deviation") or {}
    applied = mblock.get("fold_column_map") or {}
    ddm_keys = set(mblock.get("ddm") or {})
    print(f"[part2] floor_kind={mblock.get('floor_kind')}; applied_map size={len(applied)}; "
          f"n_residues={mblock.get('n_residues')}")
    checks.append(("indel: column-pairing applied (non-identity map echoed)",
                   bool(applied) and any(int(k) != int(val) for k, val in applied.items())))
    checks.append(("indel: deleted/inserted positions excluded — paired residue count < template",
                   bool(ddm_keys) and (mblock.get("n_residues") or 0) < len(GB1)))
    checks.append(("indel: ESMFold reference deterministic → floor caps",
                   mblock.get("floor_kind") == "deterministic"))
else:
    print("[part2] SKIPPED (ESMFold/venv312 unavailable)")

# ── cleanup ───────────────────────────────────────────────────────────────────────────
QtCore.QThreadPool.globalInstance().waitForDone(5000)
print("[cleanup] leaving models open for inspection")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
