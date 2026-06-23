"""
LIVE-VERIFY — the Reconnect-ChimeraX reopen + model-id REMAP + re-link, against a real ChimeraX
REST server (no GPU). Simulates "ChimeraX was closed" by closing the session, opens a DECOY so the
re-opened structure lands on a DIFFERENT id than before (the failure mode: re-link to the stale id),
then runs the window's `_do_reconnect` (ensure_visible_gui stubbed — that GUI-relaunch step is the
existing startup path) + `remap_session_model_ids` and asserts the session + a crystal design
re-key to the FRESH id and a column-select hits the reopened model.

Run (needs a ChimeraX REST server on :60001; no GPU):
  QT_QPA_PLATFORM=offscreen venv/Scripts/python.exe scripts/verify_reconnect_chimerax_live.py
"""
import os, re, sys, types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

from chimerax_bridge import ChimeraXBridge
from session_state import SessionState
from seq_editor.controller import SequenceEditorController
import gui_app

PASS, FAIL = [], []
def check(name, ok): (PASS if ok else FAIL).append(name); print(("  ✓ " if ok else "  ✗ ") + name)

bridge = ChimeraXBridge(port=60001); bridge.start(timeout=60)
run = bridge.run_command
def ids():
    return set(re.findall(r"#(\d+)", (run("info models").get("value") or "")))
def sel_count():
    out = run("info residues sel").get("value") or ""
    return len(re.findall(r"chain id|residue", out)) if out.strip() else 0
print("[chimerax] attached on :60001")

# 1) Open a throwaway (so 2hhb lands at #2), then 2hhb; register only 2hhb + a crystal design.
run("close session")
run("open 1crn")                       # throwaway → #1, so 2hhb gets a higher id
b = ids(); run("open 2hhb"); mid = (sorted(ids() - b, key=int) or ["2"])[-1]
print(f"[setup] 2hhb at #{mid} (>1, to force a fresh-id change on reopen)")
session = SessionState()
session.add_structure(mid, "2hhb")
session.add_design_session(mid, {"model_id": mid, "source": "structure",
    "chains": {f"k|{mid}/A": {"group_key": "k", "rep_model": mid, "rep_chain": "A",
                              "members": [[mid, "A"]], "template_cells": [], "variants": []}}})

# 2) Simulate ChimeraX fully closed / relaunched EMPTY (no models) — the real reconnect premise.
run("close session")
check("after 'close': ChimeraX is empty (relaunch premise)", ids() == set())

# 3) Build a minimal window stand-in + run the real reconnect logic (stub the GUI relaunch only).
win = types.SimpleNamespace(bridge=bridge, session=session)
bridge.ensure_visible_gui = lambda timeout=60: "connected"     # the GUI-relaunch step (existing path)
win._do_reconnect = types.MethodType(gui_app.StructureBotWindow._do_reconnect, win)
payload = win._do_reconnect()
print(f"[reconnect] remap={payload['remap']} reopened={payload['reopened']} errors={payload['errors']}")
new_id = payload["remap"].get(mid)
check("2hhb re-opened with a FRESH id (different from before)", bool(new_id) and new_id != mid)

# 4) Apply the remap and confirm the session + design re-key + the design re-links to the new id.
gui_app.remap_session_model_ids(session, {k: v for k, v in payload["remap"].items() if k != v})
check("session.structures re-keyed to the fresh id", new_id in session.structures and mid not in session.structures)
check("crystal design re-keyed to the fresh id",
      new_id in session.design_sessions and mid not in session.design_sessions)
cd = next(iter(session.design_sessions[new_id]["chains"].values()))
check("design members point at the fresh id", cd["members"] == [[new_id, "A"]])

# 5) A column-select built from the re-linked design hits the reopened model (non-empty).
ctrl = SequenceEditorController(run, lambda **k: {})
run("select clear")
ctrl.select_residues_multi([(new_id, "A", [10])])
check("column-select on the re-linked design selects atoms in the reopened model", sel_count() > 0)

print(f"\n══ RESULT: {len(PASS)} passed, {len(FAIL)} failed ══")
if FAIL: print("FAILED:", FAIL); sys.exit(1)
print("DONE — reconnect re-opens, remaps fresh ids, and re-links the sequence/workbench.")
