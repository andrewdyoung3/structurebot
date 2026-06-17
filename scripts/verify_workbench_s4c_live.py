"""
Live-verify Stage 4c — per-residue variant-vs-WT Cα deviation + per-residue noise floor —
against REAL ChimeraX (:60001) + the REAL local fold engines (venv312 ESMFold; ~/boltz_env
Boltz). Proves the deviation is computed on FOLDED models via the EXISTING Kabsch machinery,
the noise floor is honest (ESMFold deterministic ≈ global-min; Boltz measured cross-seed),
and the floor-GATED colouring lands on BOTH the panel cells AND the 3D folded model through
the SAME color_modes seam as ddG/pLDDT (engine-agnostic, not a parallel path).

On a freshly-opened 1HSG (WT homodimer A+B):
  ESMFold path (deterministic, fast):
    1. Fold a variant (monomer); then "Deviation vs WT" → folds the WT reference (template T)
       at the pinned engine, computes per-residue Cα deviation via auto-anchor + _anchor_kabsch.
    2. ANCHOR RESIDUAL ≈ 0 (clean rigid-core fit — the readback quality check).
    3. FLOOR = the global minimum everywhere (deterministic engine → no cross-seed motion).
    4. FLOOR-GATED COLOUR: panel cells + 3D #pred coloured; sub-floor residues NEUTRAL,
       residues clearing the floor magnitude-coloured — SAME color_modes.deviation_color.
    5. WT reference cached per combo (cd.wt_refs["esmfold:monomer"]); a 2nd variant reuses it.
  Boltz path (assembly, measured floor — MINUTES; skipped if ~/boltz_env absent):
    6. Assembly deviation: WT reference folded across N=4 seeds → a MEASURED per-residue floor
       (some residues exceed the global minimum); deviation + colouring PER-CHAIN.

Run: venv/Scripts/python.exe scripts/verify_workbench_s4c_live.py   (folds — minutes)
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
from variant_workbench import (VariantWorkbenchPanel, _RESULT_DEVIATION_MODE,
                               _DEVIATION_FLOOR_MIN_A)

bridge = ChimeraXBridge(port=60001)
run = bridge.run_command


def models():
    import re
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))


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


# ── setup: fresh 1HSG ──────────────────────────────────────────────────────────────────
before = set(models())
run("open 1hsg")
MID = (sorted(set(models()) - before, key=int) or ["1"])[-1]
print(f"[setup] opened 1HSG as model #{MID}")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
pdb_path = (Path(tempfile.gettempdir()) / f"s4c_1hsg_{MID}.pdb").as_posix()
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


def add_variant():
    panel._add_variant()
    vid = cd.variants[-1].id
    tab.set_active_row(vid)
    picked = []
    for c in cd.template_cells:
        if c.resnum is not None:
            cd.edit_variant(vid, c.col, "A" if c.aa != "A" else "S")
            picked.append(c.resnum)
            if len(picked) >= 2:
                break
    print(f"[setup] variant {vid} mutations at {picked}")
    return vid


def drive(spec, apply_fn, variant_id, timeout=600_000):
    pres.confirm_calls.clear(); pres.confirm_answer = "proceed"
    captured = []
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=captured.append)
    QtCore.QThreadPool.globalInstance().waitForDone(timeout)
    if captured:
        apply_fn(variant_id, captured[0])
    return captured


checks = []

# ══ ESMFold path (deterministic) ════════════════════════════════════════════════════
v1 = add_variant()
print("[esmfold] folding V1 (monomer, LOCAL-ONLY)…")
drive(panel.fold_launch_spec("esmfold"), panel.apply_fold_result, v1)
fold = cd.get_variant(v1).results.fold or {}
checks.append(("ESMFold variant folded", bool(fold.get("model_id"))))

print("[esmfold] Deviation vs WT — folds the WT reference (template T) then Kabsch…")
dev_spec = panel.deviation_launch_spec()
checks.append(("deviation spec confidence='low' (no cached ref → folds WT → gate)",
               dev_spec is not None and dev_spec["confidence"] == "low"))
drive(dev_spec, panel.apply_deviation_result, v1)
block = (cd.get_variant(v1).results.fold or {}).get("deviation") or {}
ar = block.get("anchor_residual_rmsd")
floor_vals = set((block.get("floor") or {}).values())
print(f"[esmfold] anchor residual={ar} Å; floor_kind={block.get('floor_kind')}; "
      f"n_cleared={block.get('n_cleared_floor')}/{block.get('n_residues')}; floor_vals={floor_vals}")
checks.append(("deviation computed on the folded model", bool(block.get("deviation"))))
checks.append(("anchor residual ≈ 0 (clean rigid-core fit readback)",
               ar is not None and ar < 0.5))
checks.append(("ESMFold floor is the global minimum (deterministic → no cross-seed motion)",
               floor_vals == {_DEVIATION_FLOOR_MIN_A}))
checks.append(("WT reference cached per combo (esmfold:monomer)",
               bool(cd.wt_refs.get("esmfold:monomer"))))

# floor-gated colouring — panel cells + 3D model, SAME color_modes.deviation_color
panel._mode_key = _RESULT_DEVIATION_MODE
panel_hex = panel._deviation_panel_hex(tab)
cmds = panel.color_commands_for(tab)
pred = block.get("variant_model_id")
print(f"[esmfold] panel-coloured residues={len(panel_hex)}; 3D cmds[:3]={cmds[:3]}")
checks.append(("floor-gated panel colour (some residues clear floor → coloured)",
               len(panel_hex) >= 1))
checks.append(("within-floor residues stay NEUTRAL (gated, not all painted)",
               len(panel_hex) < (block.get("n_residues") or 0)))
checks.append(("3D push targets the PREDICTED variant model (#pred), not the crystal",
               bool(pred) and any(f"#{pred}/" in c for c in cmds)))
# visibility of #pred is owned by fold_visibility_commands (not the colour push, which would
# otherwise re-show a toggle-hidden model); confirm it shows the active variant's model there.
checks.append(("fold_visibility_commands shows the active variant model (#pred)",
               f"show #{pred} models" in panel.fold_visibility_commands(tab)))

# second variant reuses the cached WT reference (no re-fold → cheap → confidence='high')
v2 = add_variant()
drive(panel.fold_launch_spec("esmfold"), panel.apply_fold_result, v2)
spec2 = panel.deviation_launch_spec()
checks.append(("2nd variant REUSES the cached WT reference (confidence='high', no re-fold)",
               spec2 is not None and spec2["confidence"] == "high"
               and spec2["tool_inputs"]["wt_ref"] is not None))

# ══ Boltz path (assembly, MEASURED floor) — skipped if env absent ════════════════════
if panel._fold_engine_availability().get("boltz") and len(cd.members) > 1:
    vb = add_variant()
    print("[boltz] folding the ASSEMBLY (both chains, LOCAL-ONLY) — minutes…")
    drive(panel.fold_launch_spec("boltz", assembly=True), panel.apply_fold_result, vb)
    bfold = cd.get_variant(vb).results.fold or {}
    if bfold.get("target") == "assembly":
        print("[boltz] Deviation vs WT — folds WT reference across N=4 seeds (floor)…")
        drive(panel.deviation_launch_spec(), panel.apply_deviation_result, vb)
        bblock = (cd.get_variant(vb).results.fold or {}).get("deviation") or {}
        bfloor = bblock.get("floor") or {}
        measured = [v for v in bfloor.values() if v > _DEVIATION_FLOOR_MIN_A]
        chains = {k.split(":", 1)[0] for k in (bblock.get("deviation") or {}) if ":" in k}
        print(f"[boltz] floor_kind={bblock.get('floor_kind')}; measured>{_DEVIATION_FLOOR_MIN_A}Å "
              f"residues={len(measured)}; chains={chains}; anchor residual="
              f"{bblock.get('anchor_residual_rmsd')}")
        checks.append(("Boltz floor is MEASURED (some residues exceed the global minimum)",
                       bblock.get("floor_kind") == "measured" and len(measured) >= 1))
        checks.append(("assembly deviation is PER-CHAIN (≥2 chains keyed)", len(chains) >= 2))
        panel._mode_key = _RESULT_DEVIATION_MODE
        bcmds = panel.color_commands_for(tab)
        bpred = bblock.get("variant_model_id")
        checks.append(("3D per-chain colour targets the predicted assembly model",
                       bool(bpred) and any(f"#{bpred}/" in c for c in bcmds)))
else:
    print("[boltz] SKIPPED (boltz env unavailable or monomeric design)")

# ── cleanup ───────────────────────────────────────────────────────────────────────────
QtCore.QThreadPool.globalInstance().waitForDone(5000)
print("[cleanup] leaving models open for inspection")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
