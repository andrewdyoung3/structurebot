"""
repro_disulfide_discovery_completion.py
---------------------------------------
DIAGNOSTIC reproduction for the Mode-A discovery "running forever" bug (§9, 2026-06-25).

Goal: distinguish the three hypotheses by REPRODUCTION, not plausibility —
  (1) lost-done-signal   (work done, completion never propagated)
  (2) blocking-viz/teardown step
  (3) stale-spinner      (work done, indicator never cleared)

Method (GPU-independent): mock the fold so it returns INSTANTLY (a fast fake CIF), then run
the EXACT engine completion path the GUI worker runs — handle_tool_request → _run_pipeline →
router.execute → _run_disulfide_discovery — with a recording presenter that captures the
busy (running_tools) enter/exit and the on_result callback. If the bug were in the
signal/teardown path, a fast fold would STILL hang or leave busy stuck. If it completes
cleanly, the post-fold path is sound and the hang lives in the long real fold's WSL
subprocess wait (a 4th, blocking-WAIT cause — upstream of any viz).
"""
from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter
from session_state import SessionState
from request_engine import RequestEngine


def _cif(sg_sg: float) -> str:
    return (
        "data_model\nloop_\n"
        "_atom_site.group_PDB\n_atom_site.label_atom_id\n_atom_site.label_comp_id\n"
        "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
        "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
        "ATOM CA CYS A 12 0.0 0.0 0.0\nATOM CB CYS A 12 0.0 1.5 0.0\nATOM SG CYS A 12 0.0 2.5 0.0\n"
        "ATOM CA CYS A 45 5.5 0.0 0.0\nATOM CB CYS A 45 3.8 1.5 0.0\n"
        f"ATOM SG CYS A 45 {sg_sg:.3f} 2.5 0.0\n#\n"
    )


def _write(text: str) -> str:
    f = tempfile.NamedTemporaryFile(suffix=".cif", delete=False, mode="w")
    f.write(text); f.close()
    return f.name


class RecordingPresenter:
    """Captures the busy lifecycle + every call, no real I/O. running_tools is the
    GUI's busy indicator (status CM) — we record enter/exit to see the spinner clear."""

    def __init__(self):
        self.cancelled = False
        self.busy_events = []          # ("enter", label) / ("exit", label)
        self.calls = []

    def running_tools(self, label, eta_s=0.0, needs_timer=False):
        outer = self

        class _CM:
            def __enter__(_s):
                outer.busy_events.append(("enter", label)); return _s

            def __exit__(_s, *a):
                outer.busy_events.append(("exit", label)); return False
        return _CM()

    def confirm(self, confidence):
        self.calls.append(("confirm", confidence)); return "proceed"

    def __getattr__(self, name):
        # every other presenter method → a no-op recorder
        def _rec(*a, **k):
            self.calls.append((name,) + a)
        return _rec


class FakeHost:
    def __init__(self, router, session):
        self.bridge = None             # → probe_chimerax_verbs skipped (no ChimeraX)
        self.translator = MagicMock()
        self.router = router
        self.session = session

    def _maybe_update_structure_state(self, cmds):
        pass

    def _log_exchange(self, *a, **k):
        pass


def main():
    session = SessionState()
    router = ToolRouter(bridge=MagicMock(), session=session)

    # FAST FAKE FOLD: _fold_n_seeds returns instantly (3 compatible + 1 not). This is the
    # GPU-independent stand-in for the N-seed Boltz run — the completion path runs identically.
    compat, incompat = _write(_cif(2.0)), _write(_cif(4.0))
    router._fold_n_seeds = MagicMock(return_value=[compat, compat, compat, incompat])

    engine = RequestEngine(FakeHost(router, session))
    presenter = RecordingPresenter()

    captured = {"result": None, "done": False}

    def on_result(result):
        captured["result"] = result

    # Drive on a worker thread EXACTLY like _ToolRequestWorker.run, and watch for a hang.
    def worker():
        engine.handle_tool_request(
            "disulfide_discovery",
            {"sequence": "MKVC" + "A" * 40 + "C" + "A" * 4, "n_seeds": 4},
            "[Workbench] disulfide discovery", presenter,
            confidence="low", on_result=on_result,
        )
        captured["done"] = True        # the analog of signals.done.emit()

    t0 = time.perf_counter()
    th = threading.Thread(target=worker, daemon=True)
    th.start()
    th.join(timeout=30)                # if the path hangs, join times out → bug reproduced
    elapsed = time.perf_counter() - t0

    print("=" * 70)
    print("MODE-A DISCOVERY COMPLETION REPRODUCTION (fast fake fold)")
    print("=" * 70)
    print(f"worker thread still alive (hung)? : {th.is_alive()}")
    print(f"elapsed                           : {elapsed:.3f}s")
    print(f"handle_tool_request returned      : {captured['done']}")
    print(f"on_result fired (done propagated) : {captured['result'] is not None}")
    print(f"busy events                       : {presenter.busy_events}")
    enter = sum(1 for e in presenter.busy_events if e[0] == 'enter')
    exit_ = sum(1 for e in presenter.busy_events if e[0] == 'exit')
    print(f"busy enter/exit balanced (cleared): {enter == exit_ and enter >= 1}")
    if captured["result"] is not None:
        steps = captured["result"].get("tool_step_results", [])
        disc = next((s for s in steps if s.get("tool") == "disulfide_discovery"), None)
        print(f"discovery step success            : {disc and disc.get('success')}")
        # did the tool emit any viz commands (the 'blocking-viz' surface)?
        print(f"all_viz_commands (viz surface)    : {captured['result'].get('all_viz_commands')}")
    print("=" * 70)
    verdict = (not th.is_alive()) and captured["done"] and captured["result"] is not None \
        and enter == exit_ and enter >= 1
    print("VERDICT:", "CLEAN COMPLETION — signal/teardown sound (hypotheses 1+3 refuted)"
          if verdict else "HANG/INCOMPLETE — a signal/teardown bug IS present")


if __name__ == "__main__":
    main()
