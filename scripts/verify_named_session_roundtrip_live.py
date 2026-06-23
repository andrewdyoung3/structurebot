"""
LIVE-VERIFY — named session save -> clear -> load RE-POINTS a de-novo construct's guided fold
onto the REAL reopened ChimeraX model (the failure mode = a loaded session that LOOKS restored
but mis-points). No GPU: a real PDB stands in for the de-novo fold model (the re-point path is
fold-engine-agnostic — it only cares that members / template_fold.model_id reference a live model
id and that a column click resolves to the right #id/chain). Drives the ACTUAL panel methods
(attach_session / rehydrate_denovo / select_specs_for_column) + the real session_io + a real
ChimeraX REST server, exercising the same code the GUI's Session ▸ Load runs.

What it asserts after save -> close session (clear) -> load (reopens scene.cxs):
  • opening scene.cxs restores the fold model at its SAVED id (no remap needed);
  • the rehydrated design's column-select specs point at that LIVE id, and the select is NON-EMPTY
    (a column click selects the right chain in the reopened model) — and was EMPTY right after clear;
  • guided_fold / template_assist come back attached to the correct model id;
  • the fold CIF was copied into the session's folds/ and survives deletion of the temp original.

Run (needs a ChimeraX REST server on :60001; no GPU):
  QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe scripts/verify_named_session_roundtrip_live.py
"""
import os, re, sys, tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

import config
from PySide6 import QtWidgets
from chimerax_bridge import ChimeraXBridge
from seq_editor.controller import SequenceEditorController
from session_state import SessionState
from variant_workbench import VariantWorkbenchPanel
from variant_model import DesignSession, build_design_session_from_sequence
import session_io

UBQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
PASS, FAIL = [], []
def check(name, ok):
    (PASS if ok else FAIL).append(name)
    print(("  ✓ " if ok else "  ✗ ") + name)

bridge = ChimeraXBridge(port=60001)
bridge.start(timeout=60)
run = bridge.run_command
print("[chimerax] attached on :60001")

def models():
    return re.findall(r"model id #(\d+) ", (run("info models").get("value") or ""))
def sel_residue_count():
    """# selected residues, read from `info residues sel` (plain-text REST response)."""
    out = run("info residues sel").get("value") or ""
    return len(re.findall(r"chain id|residue", out)) if out.strip() else 0

# Point SESSION_DIR at a throwaway dir so we never touch the user's real sessions.
config.SESSION_DIR = Path(tempfile.mkdtemp(prefix="verify_sessions_"))

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
session = SessionState()
ctrl = SequenceEditorController(run, lambda **k: {})
panel = VariantWorkbenchPanel(ctrl, session=session)

# 1) Open a REAL model to stand in for the de-novo construct's fold; note its live id.
run("close session")
before = set(models()); run("open 1ubq")
fold_id = (sorted(set(models()) - before, key=int) or ["1"])[-1]
print(f"[setup] fold model live at #{fold_id}")

# 2) Build the de-novo design and RE-POINT it onto the live fold (as apply_construct_fold_result does).
design = build_design_session_from_sequence("ubq_construct", [(UBQ, 1)])
ukey, cd = next(iter(design.chains.items()))
cd.members   = [(fold_id, "A")]            # the fold model's (id, chain)
cd.rep_model = fold_id
tmp_cif = os.path.join(tempfile.gettempdir(), "verify_fold_ubq.cif")
run(f'save "{Path(tmp_cif).as_posix()}" #{fold_id}')
_plddt = {i: 92.0 for i in range(1, len(UBQ) + 1)}     # real shape: per-residue {resno: pLDDT}
cd.template_fold   = {"model_id": fold_id, "engine": "boltz", "target": "monomer",
                      "plddt": dict(_plddt), "chains": ["A"], "cif_path": tmp_cif}
cd.guided_fold     = {"model_id": fold_id, "engine": "boltz", "target": "monomer",
                      "templated": True, "template_label": "8UB2", "force": False,
                      "threshold": None, "plddt": dict(_plddt), "cif_path": tmp_cif,
                      "adoption": 0.95}
cd.template_assist = {"template_label": "8UB2", "d_plddt": 3.0, "n_stabilized": 2,
                      "d_flex": {1: 0.4, 40: 0.2}}
session.add_design_session(design.model_id, design.to_dict())
panel.attach_session(session)

# pre-save sanity: a column click selects the live fold model (non-empty)
run("select clear")
ctrl.select_residues_multi(panel.select_specs_for_column(cd, 0))
check("pre-save: column-0 select hits the live fold model", sel_residue_count() > 0)

# 3) SAVE the named session (scene.cxs + session.json + copied folds/).
info = session_io.save_named_session(bridge, session, "verify_dn")
check("save wrote session.json", Path(info["json_path"]).is_file())
check("save copied the fold CIF into folds/",
      bool(list((config.SESSION_DIR / "verify_dn" / "folds").glob("*.cif"))))

# 4) CLEAR — close the scene + reset to a fresh session (the model goes away).
run("close session")
run("select clear")
ctrl.select_residues_multi(panel.select_specs_for_column(cd, 0))
check("after clear: the fold model is gone (select empty)", sel_residue_count() == 0)
session = SessionState(); panel.attach_session(session); panel.reset()

# delete the ORIGINAL temp fold CIF — the saved session must not depend on it
try: os.unlink(tmp_cif)
except OSError: pass

# 5) LOAD — reopens scene.cxs (fold model back at its saved id) + restores state (fail-loud).
linfo = session_io.load_named_session(bridge, "verify_dn")
check("load returned a state (no fail-loud error)", linfo["state"] is not None and not linfo["error"])
check("load reopened scene.cxs", bool(linfo["cxs_ok"]))
check("reopened fold model is live at its SAVED id (no remap needed)", fold_id in models())

state = linfo["state"]; panel.attach_session(state)
dd = state.get_design_session(design.model_id)
panel.rehydrate_denovo(dd)
rcd = next(iter(panel._design.chains.values()))

# 6) RE-POINT assertions on the rehydrated design.
specs = panel.select_specs_for_column(rcd, 0)
check("rehydrated column-select targets the saved/live fold id",
      bool(specs) and all(m == fold_id for (m, _c, _r) in specs))
run("select clear")
ctrl.select_residues_multi(specs)
check("post-load: column-0 select is NON-EMPTY on the reopened model (re-point correct)",
      sel_residue_count() > 0)
check("guided_fold re-attached to the correct model id", rcd.guided_fold.get("model_id") == fold_id)
check("template_assist survived (int resno keys -> str)",
      set(rcd.template_assist.get("d_flex", {})) == {"1", "40"})

saved_cif = rcd.guided_fold.get("cif_path", "")
check("restored guided_fold.cif_path points into the durable session folds/",
      Path(saved_cif).is_file() and (config.SESSION_DIR / "verify_dn" / "folds") == Path(saved_cif).parent)

print(f"\n══ RESULT: {len(PASS)} passed, {len(FAIL)} failed ══")
if FAIL:
    print("FAILED:", FAIL); sys.exit(1)
print("DONE — named-session save/clear/load re-points the de-novo guided fold correctly.")
