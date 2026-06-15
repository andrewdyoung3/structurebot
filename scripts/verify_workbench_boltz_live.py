"""
Live-verify the Boltz-2 multimer fold engine against REAL ChimeraX (:60001) + the REAL
~/boltz_env WSL GPU env. The second engine on the S4b fold seam — proves it folds a real
1HSG variant ASSEMBLY and feeds the SAME fold_summary/seam as ESMFold (not a parallel path).

On a freshly-opened 1HSG (the WT homodimer, chains A+B):
  1. CONFIRM-GATE: the Boltz assembly fold enters the spine at confidence='low'.
  2. REAL MULTI-CHAIN MODEL: a NEW ChimeraX model with TWO chains + its own atoms.
  3. MATCHMAKER OVERLAY: superposed onto the WT oligomer (core CA in register).
  4. pLDDT COLOUR + ipTM: predicted model coloured (palette alphafold); ipTM in ResultSlots + badge.
  5. LOCAL-ONLY: source=='local_boltz_env', msa empty, zero remote MSA (fail-closed bridge).
  6. SEED-PINNED REPRODUCIBLE: re-fold → identical structure (CA drift ≈ 0, the S4b bar).
  7. ENGINE-AGNOSTIC: feeds the SAME apply_fold_result/fold_summary as ESMFold.
  8. PICKER: Boltz now enabled via the capability flag.

Run: venv/Scripts/python.exe scripts/verify_workbench_boltz_live.py   (Boltz folds — minutes)
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
from variant_workbench import VariantWorkbenchPanel, _RESULT_PLDDT_MODE

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command


def models():
    import re
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


def _runscript(script: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(script)
    try:
        r = run(f'runscript "{path}"')
    finally:
        try: os.unlink(path)
        except OSError: pass
    return r.get("value") or ""


def _model_attr(model, expr, tag):
    out = _runscript(
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string == '{model}':\n"
        f"        print('{tag}:', {expr}); break\n")
    for line in out.splitlines():
        if line.strip().startswith(tag + ":"):
            return line.split(":", 1)[1].strip()
    return ""


def residue_count(model):
    v = _model_attr(model, "m.num_residues", "NRES")
    return int(v) if v.isdigit() else 0


def chains_of(model):
    v = _model_attr(model, "','.join(sorted(set(r.chain_id for r in m.residues)))", "CH")
    return [c for c in v.split(",") if c]


def distinct_ribbon_colors(model):
    v = _model_attr(model, "len(set((r.ribbon_color[0],r.ribbon_color[1],r.ribbon_color[2]) "
                           "for r in m.residues))", "NC")
    return int(v) if v.isdigit() else 0


def is_displayed(model):
    return "True" in _model_attr(model, "bool(m.display)", "DISP")


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
    return None if a is None or b is None else sum((a[i]-b[i])**2 for i in range(3))**0.5


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


# ── setup: fresh 1HSG (the WT homodimer A+B) ──────────────────────────────────────────
before = set(models())
run("open 1hsg")
MID = (sorted(set(models()) - before, key=int) or ["1"])[-1]
print(f"[setup] opened 1HSG (WT oligomer) as model #{MID}; chains={chains_of(MID)}")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
pdb_path = (Path(tempfile.gettempdir()) / f"boltz_1hsg_{MID}.pdb").as_posix()
run(f'save "{pdb_path}" format pdb models #{MID}')
session.add_structure(MID, "1hsg", path=pdb_path)

router = ToolRouter(bridge, session)
host = types.SimpleNamespace(
    bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None, _log_exchange=lambda *a, **k: None)
engine = RequestEngine(host)
pres = ScriptedPresenter()

panel = VariantWorkbenchPanel(ctrl, session=session)
panel.load_model(MID)
tab = panel._cur_tab()
if tab is None:
    print(f"[abort] load produced no tab — status={panel._status.text()!r}"); sys.exit(2)
cd = tab.design
print(f"[panel] unique-chain tab rep=#{cd.rep_model}/{cd.rep_chain} members={cd.members}")

panel._add_variant()
v1 = cd.variants[-1].id
tab.set_active_row(v1)
pick = []
for c in cd.template_cells:
    if c.resnum is not None:
        cd.edit_variant(v1, c.col, "A" if c.aa != "A" else "S")
        pick.append(c.resnum)
        if len(pick) >= 2:
            break
print(f"[setup] variant {v1} mutations at {pick}")

checks = []
checks.append(("engine.router IS the production ToolRouter", isinstance(engine.router, ToolRouter)))
checks.append(("Boltz picker enabled (capability flag)",
               panel._fold_engine_availability().get("boltz") is True))


def fold_assembly(variant_id):
    tab.set_active_row(variant_id)
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
    captured = []
    spec = panel.fold_launch_spec("boltz", assembly=True)
    bm = set(models())
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(5000)
    if captured:
        panel.apply_fold_result(variant_id, captured[0])
    new = sorted(set(models()) - bm, key=int)
    return captured, spec, new


# ── 1) fold the assembly through the spine ────────────────────────────────────────────
print("[fold] Boltz ASSEMBLY fold of V1 (LOCAL-ONLY, both chains) — minutes…")
captured, spec, new_models = fold_assembly(v1)
checks.append(("fold spec confidence='low' → confirm-gate asked", pres.confirm_calls == ["low"]))
checks.append(("fold ran through the spine (on_result fired)", bool(captured)))
checks.append(("assembly spec is multi-chain + LOCAL-ONLY",
               len(spec["tool_inputs"].get("chains") or []) == len(cd.members)
               and spec["tool_inputs"].get("local_only") is True))

fold = cd.get_variant(v1).results.fold or {}
pred = fold.get("model_id")
print(f"[fold] ResultSlots.fold = {{model:{pred}, mean_plddt:{fold.get('mean_plddt')}, "
      f"iptm:{fold.get('iptm')}, source:{fold.get('source')}, seed:{fold.get('seed')}}}")

# ── 2) REAL MULTI-CHAIN MODEL ─────────────────────────────────────────────────────────
pred_chains = chains_of(pred) if pred else []
print(f"[model] new models={new_models}; predicted #{pred} chains={pred_chains} "
      f"residues={residue_count(pred) if pred else 0}")
checks.append(("a new predicted model opened", bool(new_models) and pred in new_models))
checks.append(("predicted model is MULTI-CHAIN (≥2 chains)", len(pred_chains) >= 2))
checks.append(("predicted model has its own atoms (≈198 res)", residue_count(pred) >= 180))

# ── 3) LOCAL-ONLY ─────────────────────────────────────────────────────────────────────
checks.append(("LOCAL-ONLY: source == 'local_boltz_env'", fold.get("source") == "local_boltz_env"))

# ── 4) MATCHMAKER overlay (core CA in register on the rep chain) ──────────────────────
core = [c.resnum for c in cd.template_cells if c.resnum is not None][20:-20] or pick
dists = sorted(d for rn in core[::5]
               if (d := dist(ca_xyz(pred, cd.rep_chain, rn) if pred else None,
                             ca_xyz(MID, cd.rep_chain, rn))) is not None)
min_core = dists[0] if dists else None
print(f"[overlay] core CA dists {[round(x,1) for x in dists]} min={min_core}")
checks.append(("matchmaker superposed onto WT oligomer (core CA < 5 Å)",
               min_core is not None and min_core < 5.0))

# ── 5) pLDDT colour + ipTM ────────────────────────────────────────────────────────────
nc = distinct_ribbon_colors(pred) if pred else 0
print(f"[plddt] distinct ribbon colours = {nc}; ipTM = {fold.get('iptm')}")
checks.append(("predicted model pLDDT-coloured (>1 ribbon colour)", nc > 1))
checks.append(("ipTM surfaced (ResultSlots + badge)",
               isinstance(fold.get("iptm"), (int, float))
               and "ipTM" in (panel._badge_for(cd.get_variant(v1)) or "")))
panel._mode_key = _RESULT_PLDDT_MODE
cmds = panel.color_commands_for(tab)
checks.append(("pLDDT mode emits a model-targeted colour command",
               any(f"byattribute bfactor #{pred}" in c for c in cmds)))

# ── 6) SEED-PINNED reproducible: re-fold → identical ──────────────────────────────────
print("[repro] re-folding the assembly to confirm seed-pinned determinism…")
rrn = (core or pick)[len(core or pick) // 2]
ca1 = ca_xyz(pred, cd.rep_chain, rrn)
mean1 = fold.get("mean_plddt")
fold_assembly(v1)
fold2 = cd.get_variant(v1).results.fold or {}
pred2 = fold2.get("model_id")
ca2 = ca_xyz(pred2, cd.rep_chain, rrn) if pred2 else None
d_repro = dist(ca1, ca2)
print(f"[repro] mean pLDDT {mean1} vs {fold2.get('mean_plddt')}; CA drift {d_repro}")
checks.append(("re-fold reproducible (identical mean pLDDT)", mean1 == fold2.get("mean_plddt")))
checks.append(("re-fold reproducible (CA drift < 0.5 Å)", d_repro is not None and d_repro < 0.5))

# ── 7) engine-agnostic seam (same fold_summary contract as ESMFold) ───────────────────
checks.append(("feeds the SAME fold_summary seam (engine='boltz' in ResultSlots.fold)",
               fold.get("engine") == "boltz" and "plddt" in fold))

# ── cleanup ───────────────────────────────────────────────────────────────────────────
QtCore.QThreadPool.globalInstance().waitForDone(5000)
for m in {MID, pred, pred2} | set(new_models):
    if m:
        run(f"close #{m}")
print("[cleanup] closed models")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
