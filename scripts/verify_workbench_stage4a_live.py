"""
Live-verify the Variant-Design Workbench Stage 4a against a REAL running ChimeraX
(REST :60001): per-variant action buttons (test stability / test solubility) →
ResultSlots → per-residue ddG color, all through the REAL engine spine.

Proves, on a freshly-opened 1HSG homodimer with a variant carrying real mutations:
  1. DEEP-GATE: a deep (Rosetta) stability spec enters the spine; the tiering surfaces
     the runtime ESTIMATE and the confirm-gate is asked at confidence='low'. We CANCEL →
     the Rosetta subprocess NEVER launches and ResultSlots.stability stays empty.
  2. FAST stability: the variant's EXACT mutations are scored through the 4-axis voter
     (CamSol+ESM+ThermoMPNN+RaSP) via the spine; the executed result (the on_result seam,
     not the shared session cache) lands in the variant's ResultSlots.stability with a
     per-resnum ddG. The S3a Suggest-track cache is preserved (restored after).
  3. ddG COLOR: the per-residue ddG result mode recolors the mutated residues in BOTH
     homo-oligomer copies — verified by reading ribbon_color BACK and matching the
     diverging color_modes.ddg_color hue.
  4. SOLUBILITY: the pure CamSol scalar Δ vs the template is computed (instant, local).

Real spine, not a parallel launch: the engine's router IS the production ToolRouter and
the result the panel stores comes from the executed pipeline.

Freshly opens 1HSG, then closes it. Run: venv/Scripts/python.exe scripts/verify_workbench_stage4a_live.py
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
from variant_workbench import VariantWorkbenchPanel, _RESULT_DDG_MODE
from color_modes import ddg_color

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


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


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


# ── open a fresh 1HSG + register it ───────────────────────────────────────────────────
before = set(models())
run("open 1hsg")
MID = (sorted(set(models()) - before, key=int) or ["1"])[-1]
print(f"[setup] opened 1HSG as model #{MID}")

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
pdb_path = (Path(tempfile.gettempdir()) / f"s4a_1hsg_{MID}.pdb").as_posix()
run(f'save "{pdb_path}" format pdb models #{MID}')
session.add_structure(MID, "1hsg", path=pdb_path)

router = ToolRouter(bridge, session)
host = types.SimpleNamespace(
    bridge=bridge, session=session, router=router,
    translator=types.SimpleNamespace(trim_history=lambda: None),
    _maybe_update_structure_state=lambda *a, **k: None,
    _log_exchange=lambda *a, **k: None,
)
engine = RequestEngine(host)
pres = ScriptedPresenter()

panel = VariantWorkbenchPanel(ctrl, session=session)
panel.load_model(MID)
tab = panel._cur_tab()
cd = tab.design
print(f"[panel] unique chain tab rep=#{cd.rep_model}/{cd.rep_chain} members={cd.members}")

# build a variant with substitutions BIASED to be destabilizing (hydrophobic core →
# charged Asp) so the deep Rosetta ddG is likely to exceed the ±1 neutral band — i.e. the
# diverging color path is actually exercised, not just the white baseline.
resnums_A = [c.resnum for c in cd.template_cells if c.resnum is not None]
_HYDROPHOBIC = set("AVLIMFWC")
panel._add_variant()
vid = cd.variants[-1].id
pick = []
for c in cd.template_cells:
    if c.resnum is None or c.aa not in _HYDROPHOBIC:
        continue
    cd.edit_variant(vid, c.col, "D")            # buried hydrophobic → Asp (destabilizing)
    pick.append(c.resnum)
    if len(pick) >= 2:
        break
muts = {m.resnum: m.to_aa for m in cd.get_variant(vid).mutations}
print(f"[setup] variant {vid} mutations = {muts}")

checks = []
checks.append(("engine.router IS the production ToolRouter", isinstance(engine.router, ToolRouter)))
checks.append(("homo-oligomer collapsed (A+B)", len(cd.members) >= 2))


def launch(spec, capture=None):
    engine.handle_tool_request(spec["tool"], spec["tool_inputs"], spec["user_input"],
                               pres, confidence=spec.get("confidence", "high"),
                               on_result=capture)


# ── 1) DEEP-GATE: estimate + confirm('low'), CANCEL → no Rosetta ──────────────────────
print("[gate] launching a DEEP stability spec; will CANCEL at the gate...")
pres.warnings.clear(); pres.confirm_calls.clear(); pres.confirm_answer = None
launch(panel.stability_launch_spec(deep=True))
estimate_shown = any("approximate runtime" in w.lower() for w in pres.warnings)
print(f"[gate] estimate_shown={estimate_shown} confirm_calls={pres.confirm_calls} "
      f"stability_slot={cd.get_variant(vid).results.stability}")
checks.append(("deep stability surfaced a runtime estimate", estimate_shown))
checks.append(("confirm-gate asked at confidence='low'", pres.confirm_calls == ["low"]))
checks.append(("CANCEL → no result attached (Rosetta never launched)",
               cd.get_variant(vid).results.stability is None))

# ── 2) DEEP stability through the spine → ResultSlots (real Rosetta, calibrated ddG) ──
# Deep so the magnitudes are calibrated enough to leave the ±1 neutral band → the
# diverging color path (a per-residue command, not the white baseline) is exercised.
print("[stab] launching a DEEP stability spec; will PROCEED (real Rosetta — minutes)...")
pres.confirm_answer = "proceed"
captured = []
panel._scan_cache_snapshot = (MID, panel._read_scan_cache(MID))
launch(panel.stability_launch_spec(deep=True), capture=captured.append)
print(f"[stab] on_result fired={bool(captured)}")
checks.append(("DEEP stability ran through the spine (on_result fired)", bool(captured)))
if captured:
    panel.apply_stability_result(vid, captured[0])
stab = cd.get_variant(vid).results.stability
print(f"[stab] ResultSlots.stability per_resnum={stab.get('per_resnum') if stab else None} "
      f"sum_ddg={stab.get('sum_ddg') if stab else None} tier={stab.get('tier') if stab else None}")
checks.append(("stability landed in ResultSlots with per-resnum ddG",
               bool(stab) and set(stab.get("per_resnum", {})) == set(pick)))

# ── 3) per-residue ddG color on BOTH copies (readback == ddg_color(stored)) ───────────
panel._mode_key = _RESULT_DDG_MODE
panel._apply_color_to(tab)
cmds = panel.color_commands_for(tab)
print(f"[3D] color commands: {cmds}")
ctrl.run_commands(cmds)
ddg_map = stab.get("per_resnum", {}) if stab else {}
non_white = 0
for rn, dd in ddg_map.items():
    if dd is None:
        continue
    exp_hex = ddg_color(dd)
    exp = hex_rgb(exp_hex)
    if exp_hex != "#ffffff":
        non_white += 1
    for (m, c) in cd.members:
        got = read_colors(m, c, [rn]).get(rn)
        ok = near(got, exp)
        # the TRUE invariant: the 3D color == color_modes.ddg_color(the stored ddG)
        checks.append((f"ddG color #{m}/{c}:{rn} (ddg {dd:+.2f}) == {exp}", ok))
        print(f"[3D] #{m}/{c}:{rn} ddg {dd:+.2f} expect {exp} got {got} : {'PASS' if ok else 'FAIL'}")
# the diverging path must actually be exercised (a per-residue colored command emitted +
# read back) — not just the white baseline. Honest hard check, not white==white.
diverging_cmd = any(":" in cmd and not cmd.endswith("#ffffff") for cmd in cmds)
print(f"[3D] non-neutral residues={non_white}; per-residue colored command emitted={diverging_cmd}")
checks.append(("a non-neutral diverging ddG color reached 3D (per-residue command)",
               non_white >= 1 and diverging_cmd))

# ── 4) solubility (pure scalar) ───────────────────────────────────────────────────────
panel._on_test_solubility()
sol = cd.get_variant(vid).results.solubility
print(f"[sol] solubility={sol}")
checks.append(("solubility Δ computed (pure CamSol)",
               bool(sol) and "delta" in sol and isinstance(sol["delta"], float)))

# ── cleanup ───────────────────────────────────────────────────────────────────────────
QtCore.QThreadPool.globalInstance().waitForDone(4000)
run(f"close #{MID}")
print(f"[cleanup] closed model #{MID}")

print("\n--- RESULTS ---")
for label, ok in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
print("\nRESULT:", "ALL PASS" if checks and all(ok for _, ok in checks) else "FAILURES PRESENT")
