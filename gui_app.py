"""
gui_app.py
----------
The unified StructureBot GUI: a SECOND front-end driving the SAME RequestEngine the
console REPL drives. The window IS the engine host (collaborators + the three side-effect
hooks the engine calls via late binding). The engine runs on a worker thread
(_RequestWorker in the global QThreadPool — the _FoldWorker template); the QtPresenter
renders output to a pane and hands blocking prompts back to the worker via a queue, so
the straight-line engine works unchanged and the window never freezes.

Additive: the console REPL (main.py) keeps working on the same engine. ChimeraX + this
window are the two windows; no IPC (the post-open "focus the new model's tab" is now the
in-process structure-state hook).

Error-first: a presenter/worker/dependency failure reports in the pane and the window
stays usable — it never freezes or crashes the window or the engine.
"""
from __future__ import annotations

import ctypes
import json
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests

from PySide6 import QtCore, QtGui, QtWidgets
from rich.markup import escape

import config
from chimerax_bridge import ChimeraXBridge
from colabfold_bridge import ColabFoldBridge
from translator import CommandTranslator
from session_state import SessionState
from tool_router import ToolRouter
from request_engine import RequestEngine
from qt_presenter import QtPresenter, PresenterSignals, render_html, CANCEL
from seq_editor.controller import SequenceEditorController
from seq_editor.view import _ChainGrid
from variant_workbench import VariantWorkbenchPanel


def _async_raise(tid: int, exctype=KeyboardInterrupt) -> None:
    """Inject *exctype* into thread *tid* (the worker) — the GUI analog of console Ctrl-C.
    Fires at the next Python bytecode boundary (a running C call finishes first, exactly
    as Ctrl-C does in the console). Best-effort; never raises here."""
    try:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(tid), ctypes.py_object(exctype))
        if res > 1:   # oops — undo
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
    except Exception:
        pass


# ── workers ─────────────────────────────────────────────────────────────────────

class _RequestSignals(QtCore.QObject):
    done = QtCore.Signal()
    failed = QtCore.Signal(str)


class _RequestWorker(QtCore.QRunnable):
    """Runs one engine.handle_request off the UI thread. The blocking presenter prompts
    block THIS thread (not the UI). Cancel injects KeyboardInterrupt here → the engine's
    snapshot-restore guard fires."""

    def __init__(self, engine, text, presenter):
        super().__init__()
        self._engine, self._text, self._presenter = engine, text, presenter
        self.signals = _RequestSignals()
        self.tid: Optional[int] = None

    @QtCore.Slot()
    def run(self):
        self.tid = threading.get_ident()
        try:
            # dispatch = semicolon chaining + bypass-LLM fast-paths + handle_request
            # (the verb-guard probe runs inside engine.handle_request).
            self._engine.dispatch(self._text, self._presenter)
        except KeyboardInterrupt:
            self.signals.failed.emit("cancelled")
            return
        except Exception as exc:                       # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.done.emit()


class _ToolRequestWorker(QtCore.QRunnable):
    """Stage 3b: runs one engine.handle_tool_request off the UI thread — a tool LAUNCH
    from the Variant Workbench (build a result dict → the SAME spine as the NL path). The
    confirm-gate/tiering prompts block THIS thread via the QtPresenter; Cancel injects
    KeyboardInterrupt here exactly like the NL worker."""

    def __init__(self, engine, spec, presenter, on_result=None):
        super().__init__()
        self._engine, self._spec, self._presenter = engine, spec, presenter
        self._on_result = on_result
        self.signals = _RequestSignals()
        self.tid: Optional[int] = None

    @QtCore.Slot()
    def run(self):
        self.tid = threading.get_ident()
        try:
            s = self._spec
            self._engine.handle_tool_request(
                s["tool"], s.get("tool_inputs") or {}, s.get("user_input", ""),
                self._presenter, confidence=s.get("confidence", "high"),
                on_result=self._on_result,
            )
        except KeyboardInterrupt:
            self.signals.failed.emit("cancelled")
            return
        except Exception as exc:                       # error-first: never crash the pool
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.signals.done.emit()


class _LoadModelSignals(QtCore.QObject):
    done = QtCore.Signal(str, list)
    failed = QtCore.Signal(str)


class _LoadModelWorker(QtCore.QRunnable):
    """Loads ONE model's chains off the UI thread (ported from the seq-editor view)."""

    def __init__(self, controller, model_id):
        super().__init__()
        self._c, self._model_id = controller, model_id
        self.signals = _LoadModelSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.signals.done.emit(self._model_id, self._c.load_model(self._model_id))
        except Exception as exc:
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")


# ── managed background service (detached / windowless / logged / tracked) ──────────

class ManagedService:
    """A background process the app started (e.g. `ollama serve`): detached, windowless,
    its output logged. Tracked so teardown stops EXACTLY what the app started."""

    def __init__(self, name: str, args: list, log_path, env: Optional[dict] = None):
        self.name = name
        self.args = args
        self.log_path = Path(log_path)
        self.env = env                    # extra env vars merged over os.environ (or None)
        self.proc: Optional[subprocess.Popen] = None
        self._log = None

    def start(self) -> "ManagedService":
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "a", encoding="utf-8", buffering=1)
        except Exception:
            self._log = None
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        proc_env = None
        if self.env:
            import os as _os
            proc_env = {**_os.environ, **self.env}
        self.proc = subprocess.Popen(
            self.args,
            stdout=self._log, stderr=(subprocess.STDOUT if self._log is not None else None),
            creationflags=flags, env=proc_env,
        )
        return self

    def stop(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=4)
                except Exception:
                    proc.kill()
            except Exception:
                pass
        self.proc = None
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None


class _PreflightSignals(QtCore.QObject):
    done = QtCore.Signal()


class _PreflightWorker(QtCore.QRunnable):
    """Runs the startup preflight (ChimeraX + Ollama bring-up, checklist, restore) off
    the UI thread so the window paints immediately and never freezes."""

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.signals = _PreflightSignals()

    @QtCore.Slot()
    def run(self):
        try:
            self.win._run_preflight()
        except Exception as exc:                       # error-first: never crash the pool
            self.win.presenter.error(f"Preflight error: {type(exc).__name__}: {exc}")
        self.signals.done.emit()


class _ReconnectSignals(QtCore.QObject):
    done = QtCore.Signal(dict)


class _ReconnectWorker(QtCore.QRunnable):
    """Runs the ChimeraX relaunch + structure re-open off the UI thread (ensure_visible_gui can
    take 20-40 s to start ChimeraX). Mutates nothing — the UI thread applies the remap + redraw."""

    def __init__(self, window):
        super().__init__()
        self.win = window
        self.signals = _ReconnectSignals()

    @QtCore.Slot()
    def run(self):
        try:
            payload = self.win._do_reconnect()
        except Exception as exc:                       # error-first: never crash the pool
            self.win.presenter.error(f"Reconnect error: {type(exc).__name__}: {exc}")
            payload = {"remap": {}, "reopened": [], "errors": [str(exc)], "outcome": "error"}
        self.signals.done.emit(payload)


def remap_session_model_ids(session, old_to_new: dict) -> None:
    """Rewrite ChimeraX model ids across a SessionState after structures were re-opened with FRESH
    ids (a closed+relaunched ChimeraX reassigns ids). Renames `structures` keys, crystal-seeded
    `design_sessions` keys, and the model-id refs INSIDE every design (`rep_model`, `members`,
    `template_fold`/`guided_fold.model_id`, `wt_refs[*].model_id`, `model_id`). De-novo design keys
    (synthetic `denovo-…`) are NOT ChimeraX ids and are left as-is — only their internal fold-model
    refs are remapped. Pure dict surgery on the persisted blobs (testable, no ChimeraX)."""
    m = {str(o): str(n) for o, n in (old_to_new or {}).items() if str(o) != str(n)}
    if not m:
        return
    structs = getattr(session, "structures", None) or {}
    for old, new in m.items():
        if old in structs:
            structs[new] = structs.pop(old)
    ds = getattr(session, "design_sessions", None) or {}
    for key in list(ds.keys()):
        dd = ds[key] or {}
        for cd in (dd.get("chains") or {}).values():
            if str(cd.get("rep_model")) in m:
                cd["rep_model"] = m[str(cd["rep_model"])]
            cd["members"] = [[m.get(str(mm), mm), ch] for mm, ch in (cd.get("members") or [])]
            for slot in ("template_fold", "guided_fold"):
                blk = cd.get(slot)
                if isinstance(blk, dict) and str(blk.get("model_id")) in m:
                    blk["model_id"] = m[str(blk["model_id"])]
            for ref in (cd.get("wt_refs") or {}).values():
                if isinstance(ref, dict) and str(ref.get("model_id")) in m:
                    ref["model_id"] = m[str(ref["model_id"])]
            # Alignment overlay: the US-align REFERENCE (and the in-place-transformed query fold)
            # ids must follow a reconnect too, or the alignment-visibility toggle would target a
            # stale id after a closed+relaunched ChimeraX reassigns ids.
            sa = cd.get("structural_align")
            if isinstance(sa, dict):
                for fld in ("reference_model_id", "query_model_id"):
                    if str(sa.get(fld)) in m:
                        sa[fld] = m[str(sa[fld])]
        if str(dd.get("model_id")) in m:
            dd["model_id"] = m[str(dd["model_id"])]
        if key in m:                                   # crystal-seeded design key == its model id
            ds[m[key]] = ds.pop(key)


class _HistoryLineEdit(QtWidgets.QLineEdit):
    """Command input with ↑/↓ history recall (the console affordance a user relies on)."""

    def __init__(self, *a):
        super().__init__(*a)
        self._hist: List[str] = []
        self._idx = 0

    def push(self, text: str) -> None:
        if text and (not self._hist or self._hist[-1] != text):
            self._hist.append(text)
        self._idx = len(self._hist)

    def keyPressEvent(self, e) -> None:
        if e.key() == QtCore.Qt.Key_Up and self._hist:
            self._idx = max(0, self._idx - 1)
            self.setText(self._hist[self._idx])
            return
        if e.key() == QtCore.Qt.Key_Down and self._hist:
            self._idx = min(len(self._hist), self._idx + 1)
            self.setText(self._hist[self._idx] if self._idx < len(self._hist) else "")
            return
        super().keyPressEvent(e)


# ── the window (== the engine host) ───────────────────────────────────────────────

class StructureBotWindow(QtWidgets.QMainWindow):
    def __init__(self, port: int = 0, auto_proceed: bool = True, auto_proceed_delay: int = 2):
        super().__init__()
        # collaborators — the four main.py holds (other bridges build lazily in ToolRouter)
        self.bridge = ChimeraXBridge(port=port or config.REST_PORT)
        self.bridge.base_url = f"http://{config.REST_HOST}:{self.bridge.port}"
        self.bridge.run_url = f"{self.bridge.base_url}/run"
        self.translator = CommandTranslator()
        self.session = SessionState()
        self.router = ToolRouter(self.bridge, self.session)
        self.auto_proceed = auto_proceed
        self.auto_proceed_delay = auto_proceed_delay
        self.log_file = config.LOG_DIR / datetime.now().strftime("gui_%Y%m%d_%H%M%S.jsonl")

        # sequence-tab controller shares the same ChimeraX bridge
        self.controller = SequenceEditorController(self.bridge.run_command, ColabFoldBridge().predict)
        # Variant-Design Workbench (Stage 1) — a new panel over the SAME controller +
        # session; populated on structure open (coexists with the chain-grid tabs).
        self.workbench = VariantWorkbenchPanel(self.controller, session=self.session)

        # presenter + engine (the SAME engine the console drives)
        self._sig = PresenterSignals()
        self.presenter = QtPresenter(self._sig)
        self.engine = RequestEngine(self)

        # threading + state
        self._pool = QtCore.QThreadPool.globalInstance()
        self._grids: dict = {}            # (model, chain) -> _ChainGrid
        self._in_flight = False
        self._pending_q = None            # clarification answer queue (input-box driven)
        self._worker: Optional[_RequestWorker] = None
        self._pending_focus: List[str] = []   # next_model_id() guesses — focus FALLBACK
        self._opened_mids: List[str] = []     # REAL opened ids from the bridge — focus TRUTH
        self._assembly_snapshot: dict = {}    # {au_id: assembly_model_id} before a request → detect a NEW bio-assembly
        self._services: List[ManagedService] = []   # things WE started → teardown on close
        self._started_chimerax = False
        self._current_session_name = None       # the named session the live state belongs to (Save/Load/Save As)
        # Models ALREADY open when this window connected — i.e. opened by ANOTHER StructureBot
        # window sharing this ChimeraX (REST :60001). Hidden when this window opens its own model
        # so the current working model isn't overlaid by a different window's structure.
        self._foreign_mids: set = set()

        # Ground-truth tab focus: the bridge hands us the REAL opened model id.
        self.bridge.on_structure_opened = self._note_opened

        self._build_ui()
        self._connect()
        # Self-launching preflight (ChimeraX + Ollama) runs once the event loop starts,
        # off the UI thread; input is disabled until it finishes.
        self.input.setEnabled(False)
        QtCore.QTimer.singleShot(0, self._start_preflight)

    # ── UI ──────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.setWindowTitle("StructureBot")
        self.resize(1000, 720)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.workbench, "Variant Workbench")   # Stage-1 panel (first tab)
        # PERSISTENT Disulfides results tab (whole-suite home) — a sibling top-level tab. It survives
        # switching back to "Variant Workbench" + panel rebuilds (the old modeless dialog vanished).
        self.tabs.addTab(self.workbench.disulfides_tab, "Disulfides")
        # PERSISTENT Proline results tab — a peer of Disulfides (stabilization strategies are first-class
        # peers). Same survive-tab-switch + keep-and-clear-on-reset contract.
        self.tabs.addTab(self.workbench.proline_tab, "Proline")
        # PERSISTENT Cavity-filling results tab — the third peer strategy. Same contract.
        self.tabs.addTab(self.workbench.cavity_tab, "Cavity")
        # PERSISTENT Salt-bridge results tab — the fourth peer strategy. Same contract.
        self.tabs.addTab(self.workbench.saltbridge_tab, "Salt bridges")
        self.output = QtWidgets.QTextEdit(readOnly=True)
        self.output.setStyleSheet("QTextEdit{background:#1e1e1e;color:#dddddd;}")
        self.output.append(render_html(
            "[dim]Type a request, e.g. \"open 2HHB and show it as a cartoon\".  "
            "↑/↓ recalls history.[/dim]"))
        self.input = _HistoryLineEdit()
        self.input.setPlaceholderText("Ask StructureBot…  (e.g. \"open 1hsg and show it as a cartoon\")")

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.output)
        bl.addWidget(self.input)

        # Sequence-favored vertical split: the workbench grid gets the majority (3:2 vs the
        # console/log) so startup isn't a thin sequence strip over a tall log. The divider is
        # a draggable QSplitter; the user's adjustment is persisted via QSettings and restored
        # on the next launch (falling back to the 3:2 default on first run / a stale state).
        self.split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.split.addWidget(self.tabs)
        self.split.addWidget(bottom)
        self.split.setStretchFactor(0, 3)        # workbench grows faster than the console (2)
        self.split.setStretchFactor(1, 2)
        self.split.setSizes([600, 400])          # initial 3:2 (Qt scales to the real height)
        self._settings = QtCore.QSettings("StructureBot", "StructureBot")
        _saved_split = self._settings.value("ui/splitState")
        if _saved_split is not None:
            self.split.restoreState(_saved_split)
        self.setCentralWidget(self.split)

        # Design basket — a QDockWidget on the RIGHT (native Qt docking: resizable / float / close,
        # doesn't fight the central splitter or the resizable-window work). The CROSS-STRATEGY
        # substitution-staging panel: collect Disulfides/Proline picks → enact one variant.
        self.basket_dock = QtWidgets.QDockWidget("Design basket", self)
        self.basket_dock.setObjectName("designBasketDock")
        self.basket_dock.setWidget(self.workbench.design_basket)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.basket_dock)

        tb = self.addToolBar("main")
        self.cancel_action = tb.addAction("Cancel", self._on_cancel)
        self.cancel_action.setEnabled(False)
        # Re-open / refresh the ChimeraX view: relaunch a visible ChimeraX if it was closed and
        # re-open the session's structures so the sequence grids + workbench re-link to live 3D.
        self.reconnect_action = tb.addAction("Reconnect ChimeraX", self._on_reconnect_clicked)
        self.reconnect_action.setToolTip(
            "Re-open the ChimeraX window (if closed) and re-open this session's structures so the "
            "sequence view + Variant Workbench are linked to 3D again.")
        self.statusBar().showMessage("Ready")

        # Session menu — named save / load / clear (workbench sessions). Shares session_io with
        # the CLI (one path); load = REPLACE the current session (multi-design open is deferred).
        sm = self.menuBar().addMenu("&Session")
        sm.addAction("Save…", self._on_save_session)
        sm.addAction("Save As…", self._on_save_as_session)
        sm.addAction("Load…", self._on_load_session)
        sm.addSeparator()
        sm.addAction("Export results…", self._on_export_results)
        sm.addSeparator()
        sm.addAction("Clear / New", self._on_clear_session)

    def _connect(self) -> None:
        self._sig.append_html.connect(self._on_append)
        self._sig.set_busy.connect(self._on_busy)
        self._sig.ask.connect(self._on_ask)
        self.input.returnPressed.connect(self._on_submit)
        # Stage 3b: the Workbench requests a tool launch → run it on the engine spine.
        self.workbench.launchRequested.connect(self._on_tool_launch)

    # ── presenter signal slots (UI thread) ───────────────────────────────────────
    @QtCore.Slot(str)
    def _on_append(self, html: str) -> None:
        self.output.append(html if html else "")
        self.output.moveCursor(QtGui.QTextCursor.End)

    @QtCore.Slot(bool, str)
    def _on_busy(self, busy: bool, label: str) -> None:
        self.statusBar().showMessage(label if busy else "Ready")

    @QtCore.Slot(str, object, object)
    def _on_ask(self, kind: str, payload, q) -> None:
        if kind == "clarification":
            self.output.append(render_html(f"[warn]❓ {escape(str(payload))}[/warn]"))
            self.statusBar().showMessage("Answer the question in the input box…")
            self._pending_q = q
            self.input.setEnabled(True)
            self.input.setFocus()
        elif kind == "confirm":
            self._confirm_dialog(str(payload), q)
        elif kind == "edit":
            text, ok = QtWidgets.QInputDialog.getMultiLineText(
                self, "Edit commands", "One command per line:", "\n".join(payload))
            q.put([ln.strip() for ln in text.splitlines() if ln.strip()] if ok else CANCEL)
        elif kind == "yesno":
            r = QtWidgets.QMessageBox.question(self, "StructureBot", str(payload))
            q.put(r == QtWidgets.QMessageBox.StandardButton.Yes)
        elif kind == "restore":
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Previous session found")
            box.setText(str(payload) + "\n\nRestore it?")
            r_btn = box.addButton("Restore", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            box.addButton("Start fresh", QtWidgets.QMessageBox.ButtonRole.RejectRole)
            c_btn = box.addButton("Clear it", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
            box.exec()
            clicked = box.clickedButton()
            q.put("restore" if clicked is r_btn else "clear" if clicked is c_btn else "fresh")
        else:
            q.put(CANCEL)

    def _confirm_dialog(self, confidence: str, q) -> None:
        # auto-proceed with delay 0 → proceed immediately (matches console countdown(0))
        if self.auto_proceed and confidence in ("high", "medium") and int(self.auto_proceed_delay) <= 0:
            q.put("proceed")
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Execute?")
        box.setText("Execute the proposed commands?")
        exec_btn = box.addButton("Execute", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        edit_btn = box.addButton("Edit", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(exec_btn)
        timer = None
        if self.auto_proceed and confidence in ("high", "medium"):
            # QTimer countdown → auto-clicks Execute on timeout (the msvcrt-countdown analog)
            state = {"n": int(self.auto_proceed_delay)}
            timer = QtCore.QTimer(box)
            timer.setInterval(1000)
            exec_btn.setText(f"Execute ({state['n']}s)")

            def _tick():
                state["n"] -= 1
                if state["n"] <= 0:
                    timer.stop()
                    exec_btn.click()
                else:
                    exec_btn.setText(f"Execute ({state['n']}s)")
            timer.timeout.connect(_tick)
            timer.start()
        box.exec()
        if timer is not None:
            timer.stop()
        clicked = box.clickedButton()
        if clicked is exec_btn:
            q.put("proceed")
        elif clicked is edit_btn:
            q.put("edit")
        else:
            q.put(None)

    # ── input handling ────────────────────────────────────────────────────────────
    def _on_submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        if self._pending_q is not None:        # answering a clarification
            self.input.clear()
            self.input.setEnabled(False)
            q, self._pending_q = self._pending_q, None
            q.put(text)
            return
        if self._in_flight:
            return
        self.input.push(text)                  # ↑/↓ history recall
        self.input.clear()
        self.output.append(f"<pre style='margin:0;color:#7fd1ff'><b>&gt; {escape(text)}</b></pre>")
        self._start_request(text)

    def _start_request(self, text: str) -> None:
        self._in_flight = True
        self._pending_focus = []
        self._opened_mids = []
        # Snapshot the generated-assembly state so the post-request ingest can detect an assembly
        # BUILT (or rebuilt) by THIS request — a `bio_assembly` build never fires the bridge's
        # post-open hook (it composes the model with `combine`, not `open`), so this is the only seam.
        self._assembly_snapshot = self._assembly_model_ids()
        self.presenter.cancelled = False
        self.input.setEnabled(False)
        self.cancel_action.setEnabled(True)
        w = _RequestWorker(self.engine, text, self.presenter)
        w.signals.done.connect(self._on_request_done)
        w.signals.failed.connect(self._on_request_failed)
        self._worker = w
        self._pool.start(w)

    # ── Stage 3b: tool launch from the Variant Workbench ──────────────────────────
    @QtCore.Slot(dict)
    def _on_tool_launch(self, spec: dict) -> None:
        """Run a Workbench-built tool spec through the engine spine on a worker thread.
        On completion, fire the S3a consume path so the result auto-renders in the panel
        (the bridges cache into the session; the panel reads it back)."""
        if self._in_flight:
            self.presenter.warn("A request is already running — wait for it to finish.")
            return
        # Append the pre-built HTML DIRECTLY to the QTextEdit (same as the manual-input
        # echo above). Do NOT pass it through render_html() — that's the Rich→HTML exporter,
        # which would ESCAPE this already-HTML string into literal `<pre…>` characters.
        self.output.append(
            f"<pre style='margin:0;color:#7fd1ff'><b>&gt; {escape(spec.get('user_input',''))}</b></pre>")
        self._in_flight = True
        self._pending_focus = []
        self._opened_mids = []
        self.presenter.cancelled = False
        self.input.setEnabled(False)
        self.cancel_action.setEnabled(True)
        refresh = spec.get("refresh")
        variant_id = spec.get("_variant_id")
        self._launch_spec = spec                  # available on the UI thread in _on_tool_done
        on_result = None
        if refresh in ("stability", "fold", "deviation", "construct_fold", "structural_align",
                       "construct_fold_guided", "template_assist", "align_folds",
                       "disulfide_discovery", "disulfide_geometry", "disulfide_scan",
                       "disulfide_interface_scan", "disulfide_ddg",
                       "proline_scan", "proline_ddg",
                       "cavity_scan", "cavity_ddg",
                       "saltbridge_scan", "saltbridge_ddg"):
            # S4a/S4b: capture the EXECUTED result off the engine seam (not the shared
            # session cache) so it lands in the variant's ResultSlots. Runs on the worker
            # thread; consumed on the UI thread in _on_tool_done.
            self._captured_result = None
            def on_result(result):
                self._captured_result = result
        w = _ToolRequestWorker(self.engine, spec, self.presenter, on_result=on_result)
        w.signals.done.connect(lambda r=refresh, vid=variant_id: self._on_tool_done(r, vid))
        w.signals.failed.connect(self._on_request_failed)
        self._worker = w
        self._pool.start(w)

    @QtCore.Slot()
    def _on_tool_done(self, refresh, variant_id=None) -> None:
        # The tool's results are cached in the session by the bridge; render them via the
        # SAME S3a consume path the manual buttons use (no parallel rendering code).
        try:
            if refresh == "mpnn":
                self.workbench._import_mpnn()
            elif refresh == "scan":
                self.workbench._load_suggestions()
            elif refresh == "stability":
                result = getattr(self, "_captured_result", None)
                if result is not None and variant_id:
                    self.workbench.apply_stability_result(variant_id, result)
                else:
                    self.presenter.dim("Stability run cancelled — no result to attach.")
            elif refresh == "fold":
                result = getattr(self, "_captured_result", None)
                if result is not None and variant_id:
                    self.workbench.apply_fold_result(variant_id, result)
                else:
                    self.presenter.dim("Fold run cancelled — no model to attach.")
            elif refresh == "deviation":
                result = getattr(self, "_captured_result", None)
                if result is not None and variant_id:
                    self.workbench.apply_deviation_result(variant_id, result)
                else:
                    self.presenter.dim("Deviation run cancelled — no result to attach.")
            elif refresh == "construct_fold":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_construct_fold_result(spec, result)
                else:
                    self.presenter.dim("Construct fold cancelled — no model to attach.")
            elif refresh == "structural_align":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_structural_align_result(spec, result)
                else:
                    self.presenter.dim("Structural alignment cancelled — no result to attach.")
            elif refresh == "construct_fold_guided":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_construct_fold_guided_result(spec, result)
                else:
                    self.presenter.dim("Guided construct fold cancelled — no model to attach.")
            elif refresh == "template_assist":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_template_assist_result(spec, result)
                else:
                    self.presenter.dim("Template-assist comparison cancelled — no result to attach.")
            elif refresh == "align_folds":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_align_folds_result(spec, result)
                else:
                    self.presenter.dim("Compare folds cancelled — no result to attach.")
            elif refresh == "disulfide_discovery":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_disulfide_discovery_result(spec, result)
                else:
                    self.presenter.dim("Disulfide discovery cancelled — no result to attach.")
            elif refresh == "disulfide_geometry":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_disulfide_geometry_result(spec, result)
                else:
                    self.presenter.dim("Disulfide geometry cancelled — no result to attach.")
            elif refresh == "disulfide_scan":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_disulfide_scan_result(spec, result)
                else:
                    self.presenter.dim("Engineering scan cancelled — no result to attach.")
            elif refresh == "disulfide_interface_scan":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_disulfide_interface_scan_result(spec, result)
                else:
                    self.presenter.dim("Interface scan cancelled — no result to attach.")
            elif refresh == "disulfide_ddg":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_disulfide_ddg_result(spec, result)
                else:
                    self.presenter.dim("ΔΔG estimate cancelled — no result to attach.")
            elif refresh == "proline_scan":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_proline_scan_result(spec, result)
                else:
                    self.presenter.dim("Proline scan cancelled — no result to attach.")
            elif refresh == "proline_ddg":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_proline_ddg_result(spec, result)
                else:
                    self.presenter.dim("Proline ΔΔG estimate cancelled — no result to attach.")
            elif refresh == "cavity_scan":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_cavity_scan_result(spec, result)
                else:
                    self.presenter.dim("Cavity scan cancelled — no result to attach.")
            elif refresh == "cavity_ddg":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_cavity_ddg_result(spec, result)
                else:
                    self.presenter.dim("Cavity ΔΔG estimate cancelled — no result to attach.")
            elif refresh == "saltbridge_scan":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_saltbridge_scan_result(spec, result)
                else:
                    self.presenter.dim("Salt-bridge scan cancelled — no result to attach.")
            elif refresh == "saltbridge_ddg":
                result = getattr(self, "_captured_result", None)
                spec = getattr(self, "_launch_spec", None)
                if result is not None and spec is not None:
                    self.workbench.apply_saltbridge_ddg_result(spec, result)
                else:
                    self.presenter.dim("Salt-bridge ΔΔG estimate cancelled — no result to attach.")
        except Exception as exc:
            self.presenter.warn(f"Workbench refresh failed: {exc}")
        self._captured_result = None
        opened = bool(self._opened_mids)
        self._finish_request()
        if opened:                              # a fold opened a model → isolate from other windows
            self._isolate_foreign_models()

    @QtCore.Slot()
    def _on_request_done(self) -> None:
        # Ground truth first: the REAL opened ids the bridge captured; the next_model_id()
        # guesses are only the fallback when the bridge saw no open result.
        opened = bool(self._opened_mids)
        focus = list(self._opened_mids) if self._opened_mids else list(self._pending_focus)
        self._finish_request()
        for mid in focus:                       # after an open, focus the new model's tab
            self.show_model(mid)
        if opened:                              # claimed a working model → hide other windows' models
            self._isolate_foreign_models()
        self._ingest_new_assembly()             # a bio_assembly build → make the assembly the active design

    def _assembly_model_ids(self) -> dict:
        """{au_model_id: assembly_model_id} for every generated bio-assembly — the snapshot key the
        ingest diffs across a request to spot a newly built (or rebuilt) assembly."""
        gen = getattr(self.session, "generated_assemblies", {}) or {}
        return {au: rec.get("assembly_model_id") for au, rec in gen.items()}

    def _ingest_new_assembly(self) -> None:
        """PART A: a `bio_assembly` build composes a FLAT, unique-chain assembly model via
        sym→changechains→combine, but `combine` never opens a FILE, so the bridge's post-open hook
        never fires and the Variant Workbench never ingests the assembly — it keeps the 1-chain AU
        as the active design (greying the interchain-disulfide menu, running the cavity scan on the
        monomer). Detect an assembly BUILT or CHANGED during the just-finished request (vs the
        start-of-request snapshot) and make it the active design via the SAME show_model seam an
        open uses, so the interface tools + scans target the assembly. Bio_assembly-specific for now
        (a general post-`combine` ingest path is the follow-up)."""
        before = getattr(self, "_assembly_snapshot", None) or {}
        now = self._assembly_model_ids()
        new_mid = None
        for au, amid in now.items():
            if amid and str(before.get(au)) != str(amid):
                new_mid = str(amid)
        self._assembly_snapshot = now            # new baseline so a later request doesn't re-ingest
        if new_mid:
            self.show_model(new_mid)             # → workbench.load_model: the assembly becomes active

    def _isolate_foreign_models(self) -> None:
        """In a SHARED ChimeraX, HIDE (not close) any still-visible model that was already open
        when this window connected — i.e. opened by another StructureBot window — so the current
        working model isn't overlaid. Best-effort; never touches this window's own models (they
        have ids that weren't in the startup baseline)."""
        if not getattr(self, "_foreign_mids", None):
            return
        try:
            still = self._foreign_mids & set(self.bridge.visible_model_ids())
            if still:
                self.bridge.run_command("hide #" + ",".join(sorted(still, key=int)) + " models")
        except Exception:
            pass

    def _note_opened(self, model_id) -> None:
        """Bridge post-open hook (worker thread): record the REAL opened model id."""
        mid = str(model_id).lstrip("#").strip()
        if mid:
            self._opened_mids.append(mid)

    @QtCore.Slot(str)
    def _on_request_failed(self, err: str) -> None:
        if err == "cancelled":
            self.output.append(render_html("[warn]Cancelled.[/warn]"))
        else:
            self.output.append(render_html(f"[err]Request failed: {escape(err)}[/err]"))
        self._finish_request()

    def _finish_request(self) -> None:
        self._in_flight = False
        self._worker = None
        self._pending_q = None
        self.input.setEnabled(True)
        self.cancel_action.setEnabled(False)
        self.input.setFocus()
        self.statusBar().showMessage("Ready")

    def _on_cancel(self) -> None:
        if not self._in_flight:
            return
        self.presenter.cancelled = True
        if self._pending_q is not None:         # cancel at a prompt → clean abort
            q, self._pending_q = self._pending_q, None
            q.put(CANCEL)
            self.input.setEnabled(False)
        elif self._worker is not None and self._worker.tid:
            _async_raise(self._worker.tid)       # mid-run cancel → snapshot-restore
        self.statusBar().showMessage("Cancelling…")

    # ── sequence tabs (ported view pieces) ────────────────────────────────────────
    def show_model(self, model_id) -> None:
        mid = str(model_id).lstrip("#").strip()
        if not mid:
            return
        # Populate the Variant Workbench (template T per unique chain) for this model.
        # Error-first inside load_model — never blocks the chain-grid path below.
        try:
            self.workbench.load_model(mid)
        except Exception as exc:
            self.presenter.warn(f"Workbench load #{mid} failed: {exc}")
        # The chain-grid tab strip mirrors the SINGLE active design — drop any tab belonging to a
        # previously-active model so a stale strip can't leak alongside the new one (e.g. the AU's
        # #1/A persisting next to a generated assembly's #3/A,#3/B,#3/C). Generation is filtered to
        # the active design's chains, not every model the session has ever shown.
        self._prune_chain_tabs_except(mid)
        if any(k[0] == mid for k in self._grids):
            self._focus_model(mid)
            return
        w = _LoadModelWorker(self.controller, mid)
        w.signals.done.connect(self._on_model_loaded)
        w.signals.failed.connect(lambda e: self.presenter.warn(f"show #{mid} failed: {e}"))
        self._pool.start(w)

    def _prune_chain_tabs_except(self, model_id: str) -> None:
        """Remove chain-grid tabs that do NOT belong to *model_id*. The strip reflects the one active
        design (the panel holds a single design), so a prior model's chain tabs must not persist when
        a different model becomes active. No-op when nothing is stale."""
        stale = [key for key in self._grids if key[0] != str(model_id)]
        for key in stale:
            grid = self._grids.pop(key)
            idx = self.tabs.indexOf(grid)
            if idx != -1:
                self.tabs.removeTab(idx)
            grid.deleteLater()

    @QtCore.Slot(str, list)
    def _on_model_loaded(self, model_id: str, chains: list) -> None:
        if not chains:
            self.presenter.dim(f"Model #{model_id}: no macromolecule chains found.")
            return
        for ch in chains:
            if ch.key not in self._grids:
                self._add_chain_tab(ch)
        self._focus_model(model_id)

    def _add_chain_tab(self, ch):
        grid = _ChainGrid(ch)
        self._grids[ch.key] = grid
        self.tabs.addTab(grid, f"#{ch.model}/{ch.chain}  ({len(ch.cells)} aa)")
        return grid

    def _focus_model(self, model_id: str) -> None:
        for key, grid in self._grids.items():
            if key[0] == model_id:
                self.tabs.setCurrentWidget(grid)
                break

    # ── engine host hooks (called by the engine via late binding) ─────────────────
    def _log_exchange(self, user_input, commands, success, error, tool_steps=None) -> None:
        entry: dict = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "user_input": user_input,
            "commands":   commands,
            "success":    success,
            "error":      error,
        }
        if tool_steps:
            entry["tool_steps"] = tool_steps
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _maybe_update_structure_state(self, commands) -> None:
        """Session-state sync (ported from main) PLUS recording opened models so the
        post-request hook can focus the new model's tab (replaces the retired IPC
        post-open callback — in-process now)."""
        for cmd in commands:
            s = cmd.strip().lower()
            if s.startswith("open ") and "session" not in s:
                parts = cmd.split()
                if len(parts) >= 2:
                    name = parts[1].strip("'\"")
                    mid  = self.session.next_model_id()
                    path = None
                    if name.lower().endswith((".pdb", ".cif", ".mol2")):
                        path = str(Path(self.session.working_dir) / name)
                    self.session.add_structure(mid, name, path=path)
                    self._display_assembly_type_on_open(name, mid)
                    self._pending_focus.append(mid)
            elif s.startswith("close"):
                if "all" in s:
                    self.session.clear_all_structures()
                else:
                    m = re.search(r"#(\d+)", cmd)
                    if m:
                        self.session.remove_structure(m.group(1))
            for kw in ("cartoon", "surface", "style", "color", "show", "hide",
                       "transparency", "rainbow", "mlp", "coulombic", "preset"):
                if s.startswith(kw):
                    self.session.record_style(cmd)
                    break

    def _display_assembly_type_on_open(self, name: str, model_id: str) -> None:
        """Assembly note routed through the presenter to the pane (the Stage-2 follow-up:
        no longer main.console)."""
        if not re.match(r"^[A-Za-z0-9]{4}$", name.strip()):
            return
        try:
            from assembly_analyser import fetch_assembly_info, AssemblyAnalyser
            cached = self.session.get_assembly_info(name.upper())
            asm_info = cached if cached else fetch_assembly_info(name)
            if not cached and asm_info and not asm_info.get("error"):
                self.session.set_assembly_info(name.upper(), asm_info)
            if asm_info and not asm_info.get("error"):
                analyser = AssemblyAnalyser(self.bridge, self.session)
                display  = analyser.get_assembly_display(name, asm_info)
                self.presenter.dim(f"  ✓ {name.upper()} → {display}")
                n_bio = asm_info.get("n_subunits")
                if n_bio and int(n_bio) > 1:
                    try:
                        n_loaded = len(self.bridge._model_chains(model_id))
                        if n_loaded > 0 and int(n_bio) > n_loaded:
                            asm_type = asm_info.get("assembly_type") or "biological assembly"
                            stoich   = asm_info.get("stoichiometry") or ""
                            stoich_s = f" ({stoich})" if stoich else ""
                            self.presenter.dim(
                                f"  ⚠ Loaded the asymmetric unit ({n_loaded} chain"
                                f"{'s' if n_loaded != 1 else ''}); biological assembly is a "
                                f"{asm_type}{stoich_s} — generate it with \"work as "
                                f"{asm_type.split()[-1]}\" / \"generate biological assembly\"?")
                    except Exception:
                        pass
        except Exception:
            pass  # assembly info is non-critical; never interrupt the flow

    # ── startup preflight (ported from main.startup, rendered to the pane) ─────────
    def _start_preflight(self) -> None:
        self.statusBar().showMessage("Starting up…")
        w = _PreflightWorker(self)
        w.signals.done.connect(self._on_preflight_done)
        self._pool.start(w)

    @QtCore.Slot()
    def _on_preflight_done(self) -> None:
        self.input.setEnabled(True)
        self.input.setFocus()
        self.statusBar().showMessage("Ready")
        # Re-display any restored workbench designs now that we're on the UI thread.
        self._redisplay_designs(getattr(self, "_restore_mids", []) or [])
        self._restore_mids = []

    def _redisplay_designs(self, mids) -> None:
        """Re-display restored workbench designs (the SINGLE re-display contract, shared by the
        startup restore and a named Load). For each design id: a DE-NOVO construct rehydrates
        directly (synthetic id not in ChimeraX; its fold model is live in the reopened scene); a
        crystal-seeded design re-shows its model. Must run on the UI thread. Per-design error-first
        so one mis-pointing design can't abort the rest. Relies on the model ids being the SAVED
        ones — true for both the app-restart restore (ChimeraX persists) and a named Load (opening
        scene.cxs restores models at their saved ids)."""
        designs = getattr(self.session, "design_sessions", {}) or {}
        for mid in mids:
            try:
                dd = designs.get(mid) or {}
                if dd.get("source") == "sequence":
                    self.workbench.rehydrate_denovo(dd)
                else:
                    self.show_model(mid)
            except Exception as exc:
                self.presenter.warn(f"Restore display #{mid} failed: {exc}")

    def _reset_view_for_session(self) -> None:
        """Reset the view before displaying a different session (load = REPLACE) / on Clear.
        Removes the CHAIN-GRID tabs but KEEPS the Variant Workbench tab (it is a tab inside
        `self.tabs`, so a blanket `clear()` would destroy the panel + its toolbar — leaving no way
        to start a new session). The workbench is cleared to its no-design state via `reset()`.
        Does NOT close ChimeraX models — a named Load reopens scene.cxs (replacing the scene
        itself); Clear leaves the user's models untouched (parity with the CLI)."""
        # KEEP the persistent suite tabs (the workbench + its toolbar AND the sibling Disulfides tab —
        # a sibling, not the workbench, so it was being swept out with the chain grids and never
        # re-added). Drop only the chain-grid tabs.
        disulfides = getattr(self.workbench, "disulfides_tab", None)
        proline = getattr(self.workbench, "proline_tab", None)
        cavity = getattr(self.workbench, "cavity_tab", None)
        saltbridge = getattr(self.workbench, "saltbridge_tab", None)
        for i in range(self.tabs.count() - 1, -1, -1):
            w = self.tabs.widget(i)
            if w not in (self.workbench, disulfides, proline, cavity, saltbridge):
                self.tabs.removeTab(i)
        self._grids.clear()
        try:
            self.workbench.reset()
        except Exception:
            pass
        # CLEAR-on-reset: the kept Disulfides tab's content is the PRIOR session's (stale after a
        # Load — references a construct no longer loaded). Keep the tab, empty it; new scans refill.
        try:
            if disulfides is not None:
                disulfides.reset()
        except Exception:
            pass
        try:
            if proline is not None:
                proline.reset()
        except Exception:
            pass
        try:
            if cavity is not None:
                cavity.reset()
        except Exception:
            pass
        try:
            if saltbridge is not None:
                saltbridge.reset()
        except Exception:
            pass
        try:
            basket = getattr(self.workbench, "design_basket", None)
            if basket is not None:
                basket.reset()                       # stale picks reference the gone session's design
        except Exception:
            pass

    # ── Session menu: named save / load / clear (shares session_io with the CLI) ──────
    def _on_save_session(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save session", "Session name:")
        if not ok or not (name or "").strip():
            return
        import session_io
        info = session_io.save_named_session(self.bridge, self.session, name)
        if not info["cxs_ok"]:
            self.presenter.warn(f"Scene not saved ({info['cxs_error']}); state was still written.")
        if info["json_error"]:
            self.presenter.error(f"Save failed: {info['json_error']}")
        else:
            self._current_session_name = info["name"]
            self.presenter.success(f"✓ Saved session '{info['name']}' → {info['dir']}")

    def _on_load_session(self) -> None:
        import session_io
        names = session_io.list_saved_sessions()
        if not names:
            self.presenter.dim("No saved sessions yet (Session ▸ Save… first).")
            return
        name, ok = QtWidgets.QInputDialog.getItem(
            self, "Load session", "Session (replaces the current one):", names, 0, False)
        if not ok or not name:
            return
        self.statusBar().showMessage(f"Loading session '{name}'…")
        info = session_io.load_named_session(self.bridge, name)
        if info["error"]:                                  # FAIL-LOUD — never swap to a fresh state
            self.presenter.error(f"Load failed: {info['error']}")
            self.statusBar().showMessage("Ready")
            return
        if info["cxs_error"]:
            self.presenter.warn(f"Scene: {info['cxs_error']}")
        # Replace the session + re-point everything that holds a session reference.
        self.session = info["state"]
        self.router = ToolRouter(self.bridge, self.session)
        self.workbench.attach_session(self.session)
        self._reset_view_for_session()
        self._redisplay_designs(list((getattr(self.session, "design_sessions", {}) or {}).keys()))
        self._current_session_name = info["name"]
        self.presenter.success(f"✓ Loaded session '{info['name']}'")
        self.statusBar().showMessage("Ready")

    def _on_clear_session(self) -> None:
        if QtWidgets.QMessageBox.question(
                self, "Clear session",
                "Start a fresh session? Workbench designs + analysis are cleared. "
                "Loaded ChimeraX models are left open.") != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.session = SessionState()
        self.router = ToolRouter(self.bridge, self.session)
        self.workbench.attach_session(self.session)
        self._reset_view_for_session()
        self._current_session_name = None
        try:
            Path("session.json").unlink()
        except OSError:
            pass
        self.presenter.dim("Session cleared — fresh start. Loaded ChimeraX models are untouched.")

    def _on_save_as_session(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save As", "New session name:")
        if not ok or not (name or "").strip():
            return
        import session_io
        info = session_io.save_as_session(self.bridge, self.session, name,
                                          src_name=self._current_session_name)
        if info["error"]:
            self.presenter.error(f"Save As failed: {info['error']}")
            return
        # Switch the live session to the fork — subsequent Save/Export land in it; the original is
        # frozen as a snapshot.
        self._current_session_name = info["name"]
        frm = f" (forked from '{Path(info['copied_from']).name}')" if info.get("copied_from") else ""
        self.presenter.success(f"✓ Saved As '{info['name']}'{frm} → {info['dir']}; now editing the fork.")

    def _on_export_results(self) -> None:
        """Export the session's results into {name}/exports/ (results.xlsx + csv/). Requires a named
        session — if unnamed, prompt to Save first (reuse the Save dialog). Fail-loud: skip result
        types with no data; nothing-has-data → no files written."""
        import session_io
        import session_export
        if not self._current_session_name:
            if QtWidgets.QMessageBox.question(
                    self, "Save first",
                    "Exports live inside a named session's folder. Save this session first?"
                    ) != QtWidgets.QMessageBox.StandardButton.Yes:
                return
            self._on_save_session()                    # prompts for a name + sets _current_session_name
            if not self._current_session_name:         # user cancelled the save
                return
        exports_dir = session_io.session_paths(self._current_session_name)["exports"]
        rep = session_export.export_session(getattr(self.session, "design_sessions", {}) or {},
                                            exports_dir,
                                            getattr(self.session, "proteinmpnn_results", {}) or {})
        if not rep["any"]:
            self.presenter.dim("Nothing to export yet — no fold / deviation / stability / "
                               "solubility / template-assist / alignment results in this session.")
            return
        nfig = len((rep.get("figures") or {}).get("written") or [])
        msg = f"✓ Exported {len(rep['written'])} table(s) → {exports_dir} "
        msg += f"(results.xlsx + csv/{' + sequences.fasta' if 'Sequences' in rep['written'] else ''}"
        msg += f"{f' + {nfig} figure(s)' if nfig else ''}). Written: {', '.join(rep['written'])}."
        if rep["skipped"]:
            msg += f"  Skipped (no data): {', '.join(rep['skipped'])}."
        self.presenter.success(msg)

    # ── Reconnect / refresh the ChimeraX view (relaunch + re-open + re-link) ──────────
    def _on_reconnect_clicked(self) -> None:
        # Always (re)launch ChimeraX — even on an EMPTY session. The button is the user's
        # documented recovery for a closed window ("Re-open the ChimeraX window (if closed)…"),
        # and the normal command path deliberately never auto-relaunches (the anti-window-pile
        # guard). An early-return on an empty session left the user with NO way to bring
        # ChimeraX back after closing it in a fresh session. `_do_reconnect` handles zero
        # structures fine (ensure_visible_gui then a no-op re-open loop).
        self.reconnect_action.setEnabled(False)
        self.statusBar().showMessage("Reconnecting ChimeraX…")
        if not self.session.structures:
            self.presenter.info("Reconnecting ChimeraX (relaunching the window if closed; "
                                "no structures in this session to re-open)…")
        else:
            self.presenter.info("Reconnecting ChimeraX (relaunching if closed; re-opening structures)…")
        w = _ReconnectWorker(self)
        w.signals.done.connect(self._on_reconnect_done)
        self._pool.start(w)

    def _do_reconnect(self) -> dict:
        """OFF-THREAD: ensure a VISIBLE ChimeraX is up (relaunch if it was closed), then re-open
        every session structure that is missing from the (possibly fresh) ChimeraX, capturing each
        one's NEW model id. Returns {remap old->new, reopened, errors, outcome}; no session mutation
        here (the UI thread applies it). A structure already present is mapped to itself."""
        outcome = self.bridge.ensure_visible_gui(timeout=60)

        def cur_ids():
            res = self.bridge.run_command("info models")
            val = (res.get("value") or "") if isinstance(res, dict) else ""
            return set(re.findall(r"#(\d+)", val))

        remap, reopened, errors, fallbacks = {}, [], [], []
        open_now = cur_ids()
        for old_id, info in list(self.session.structures.items()):
            name = str(info.get("name", "?"))
            if old_id in open_now:
                remap[old_id] = old_id                 # ChimeraX wasn't fully closed — reuse as-is
                continue
            target = self._resolve_reopen_target(info)
            if not target:
                errors.append(f"#{old_id} {name}: no source to re-open from "
                              "(temp fold gone, no saved-session copy, no PDB id)")
                continue
            used_fallback = bool(info.get("path")) and target != info.get("path")
            before = cur_ids()
            res = self.bridge.run_command(f"open {target}")
            if isinstance(res, dict) and res.get("error"):
                errors.append(f"{name}: {str(res['error'])[:60]}")
                continue
            after = cur_ids()
            new = sorted(after - before, key=int)
            if not new:
                errors.append(f"{name}: opened but no new model id detected")
                continue
            remap[old_id] = new[-1]
            reopened.append((name, old_id, new[-1]))
            if used_fallback:
                fallbacks.append(name)
            open_now = after
        return {"remap": remap, "reopened": reopened, "errors": errors,
                "fallbacks": fallbacks, "outcome": outcome}

    def _resolve_reopen_target(self, info: dict):
        """Best on-disk/id source to re-open a structure from, in order: its own `path` if the file
        still exists; else a DURABLE copy in a named session's `folds/` matched by basename (the
        FALLBACK for a de-novo fold whose volatile temp CIF was cleaned); else the 4-char PDB id
        (ChimeraX fetches it); else None (un-reopenable)."""
        path = info.get("path")
        if path and Path(path).is_file():
            return path
        if path:
            import session_io
            dur = session_io.find_fold_copy(Path(path).name)
            if dur:
                return dur
        name = str(info.get("name", "")).strip()
        if re.match(r"^[A-Za-z0-9]{4}$", name):
            return name
        return None

    def _denovo_fold_ids(self) -> set:
        """Model ids that are de-novo FOLD models (referenced by a `source=='sequence'` design) —
        these are re-displayed via `rehydrate_denovo`, NOT as plain structures."""
        ids: set = set()
        for dd in (getattr(self.session, "design_sessions", {}) or {}).values():
            if (dd or {}).get("source") != "sequence":
                continue
            for cd in (dd.get("chains") or {}).values():
                for slot in ("template_fold", "guided_fold"):
                    mid = (cd.get(slot) or {}).get("model_id")
                    if mid:
                        ids.add(str(mid))
                for mm, _ch in (cd.get("members") or []):
                    ids.add(str(mm))
        return ids

    @QtCore.Slot(dict)
    def _on_reconnect_done(self, payload: dict) -> None:
        self.reconnect_action.setEnabled(True)
        changed = {o: n for o, n in (payload.get("remap") or {}).items() if str(o) != str(n)}
        if changed:
            remap_session_model_ids(self.session, changed)   # old ids → fresh reopened ids
            self.workbench.attach_session(self.session)
        # Rebuild the view: re-link workbench designs (de-novo rehydrate / crystal show_model) +
        # any plain loaded structures that aren't a de-novo fold model.
        self._reset_view_for_session()
        design_keys = set(getattr(self.session, "design_sessions", {}) or {})
        self._redisplay_designs(list(design_keys))
        fold_ids = self._denovo_fold_ids()
        for mid in list(self.session.structures.keys()):
            if mid in design_keys or mid in fold_ids:
                continue
            try:
                self.show_model(mid)
            except Exception as exc:
                self.presenter.warn(f"re-show #{mid} failed: {exc}")
        try:
            self.bridge._maybe_apply_lean_layout()
        except Exception:
            pass
        if payload.get("errors"):
            self.presenter.warn("Reconnect issues: " + "; ".join(payload["errors"][:4]))
        n = len(payload.get("reopened") or [])
        nfb = len(payload.get("fallbacks") or [])
        fb = f" ({nfb} from a saved session's folds/)" if nfb else ""
        self.presenter.success(
            f"✓ ChimeraX reconnected — re-opened {n} structure(s){fb}; sequence + workbench re-linked."
            if n else "✓ ChimeraX reconnected — view re-linked.")
        self.statusBar().showMessage("Ready")

    def _run_preflight(self) -> None:
        """Worker-thread preflight: bring up ChimeraX + Ollama, render the ✓ checklist to
        the pane, offer session restore. Error-first — any dependency that won't come up
        is reported and the GUI stays usable."""
        p = self.presenter
        p.info("StructureBot — starting up… (translation is LOCAL-ONLY via Ollama)")
        # No API-key check — there is no Claude/Anthropic path.
        self._preflight_chimerax()
        self._preflight_ollama()
        self._preflight_accelerators()
        self._preflight_wsl()
        self._preflight_restore()
        p.dim("Startup complete.")

    def _preflight_chimerax(self) -> None:
        p = self.presenter
        cx = self.bridge.chimerax_path
        if not cx or not Path(cx).is_file():
            p.warn("⚠ ChimeraX not found (set CHIMERAX_PATH). Open ChimeraX manually and run "
                   "'remotecontrol rest start port 60001'.")
            return
        p.dim(f"ChimeraX: {cx}")
        # ensure_visible_gui() rejects a leftover *windowless* ChimeraX (REST-reachable
        # but with no GUI window — a zombie from a prior session): models would open into
        # an invisible viewer and "nothing appears". It relaunches a fresh visible window.
        if not self.bridge.is_running():
            p.info("ChimeraX REST not found — launching ChimeraX… (may take 20–40 s)")
        try:
            outcome = self.bridge.ensure_visible_gui(timeout=60)
            if outcome == "connected":
                p.success(f"✓ Connected to ChimeraX on port {self.bridge.port}")
            elif outcome == "relaunched":
                self._started_chimerax = True
                p.warn("⚠ A windowless ChimeraX was holding the port — relaunched a "
                       f"fresh visible window (REST on port {self.bridge.port}).")
            else:  # "started"
                self._started_chimerax = True
                p.success(f"✓ ChimeraX started — REST on port {self.bridge.port}")
        except Exception as exc:
            p.error(f"✗ Failed to start ChimeraX: {exc}")
            p.dim("Manual fix: open ChimeraX → 'remotecontrol rest start port 60001'.")
            return
        ping = self.bridge.ping()
        if ping.get("ok"):
            ver = (ping["result"].get("value") or "")[:40].strip()
            p.success(f"✓ Ping OK ({ping['latency_ms']} ms) — {ver}")
        else:
            p.warn(f"⚠ Ping failed: {ping['result'].get('error')}")
        # Baseline of models ALREADY open in this (possibly shared) ChimeraX — i.e. opened by
        # ANOTHER StructureBot window. They are hidden once THIS window opens its own model, so
        # the current working model isn't overlaid (models created later by this window — opens,
        # folds, assemblies — get ids NOT in this set, so they're never hidden).
        try:
            self._foreign_mids = set(self.bridge.visible_model_ids())
        except Exception:
            self._foreign_mids = set()
        # Structure-only window: hide Log/Models/CLI/Toolbar NOW (at startup), not just
        # on first open — so the clean structure view is up before any model loads.
        # Once-per-session guarded; the first-open call remains a no-op fallback.
        self.bridge._maybe_apply_lean_layout()

    # Blackwell-safe Ollama floor: builds below this lack the sm_120 GPU fix and run on
    # CPU silently (the 0.24.0 bug). Warn if the installed binary is older.
    _OLLAMA_MIN_VERSION = (0, 30, 0)

    def _ollama_cli_version(self):
        """(version_tuple, raw_str) from `ollama --version`, or (None, '') if unknown."""
        try:
            out = subprocess.run(["ollama", "--version"], capture_output=True, text=True,
                                  timeout=8,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
            m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
            if m:
                return (tuple(int(g) for g in m.groups()), m.group(0))
        except Exception:
            pass
        return (None, "")

    def _ollama_gpu_status(self, base: str):
        """Determine GPU vs CPU placement the REAL way (mirrors esm_bridge's
        real-kernel check, not a bare ping): warm-load the model, then read /api/ps
        `size_vram`. Returns ('gpu'|'cpu'|None, detail). None = couldn't determine."""
        try:
            requests.post(f"{base}/api/generate", json={
                "model": config.OLLAMA_MODEL, "prompt": "ok", "stream": False,
                "think": False, "options": {"num_predict": 1},
            }, timeout=90)
            r = requests.get(f"{base}/api/ps", timeout=5)
            stem = config.OLLAMA_MODEL.split(":")[0]
            for m in (r.json().get("models") or []):
                if m.get("name") == config.OLLAMA_MODEL or m.get("name", "").startswith(stem):
                    vram, total = m.get("size_vram", 0) or 0, m.get("size", 0) or 0
                    if vram > 0:
                        pct = int(100 * vram / total) if total else 100
                        return ("gpu", f"{pct}% in VRAM")
                    return ("cpu", "0% in VRAM")
            return (None, "model not resident")
        except Exception as exc:
            return (None, f"{type(exc).__name__}")

    def _preflight_ollama(self) -> None:
        p = self.presenter
        base = config.OLLAMA_BASE_URL
        where = base.split("://", 1)[-1]      # host:port for the status lines

        # Version floor (B2): a downgrade below 0.30.0 re-introduces the Blackwell/CPU bug.
        ver_tuple, ver_str = self._ollama_cli_version()
        if ver_tuple is not None and ver_tuple < self._OLLAMA_MIN_VERSION:
            p.warn(f"⚠ Ollama {ver_str} is below the Blackwell-safe floor "
                   f"{'.'.join(map(str, self._OLLAMA_MIN_VERSION))} — it may run on CPU. "
                   "Update from https://ollama.com/download.")
        elif ver_str:
            p.dim(f"  Ollama {ver_str}")

        def tags():
            try:
                r = requests.get(f"{base}/api/tags", timeout=3)
                return r.json() if r.status_code == 200 else None
            except Exception:
                return None

        t = tags()
        if t is None:
            p.info("Ollama not running — starting `ollama serve` (windowless, logged)…")
            # Bind the spawned serve to the configured host:port (so a non-default
            # OLLAMA_BASE_URL is honoured), via OLLAMA_HOST.
            host = base.split("://", 1)[-1]
            svc = ManagedService("ollama", ["ollama", "serve"],
                                 config.LOG_DIR / "ollama_serve.log",
                                 env={"OLLAMA_HOST": host})
            try:
                svc.start()
                self._services.append(svc)
            except FileNotFoundError:
                p.error("✗ REQUIRED: Ollama is not installed — translation is LOCAL-ONLY "
                        "and has NO fallback. Install from https://ollama.com/download.")
                return
            except Exception as exc:
                p.error(f"✗ REQUIRED: couldn't start Ollama: {exc} — translation cannot run "
                        "without it (no fallback).")
                return
            deadline = time.time() + 30
            while time.time() < deadline:
                t = tags()
                if t is not None:
                    break
                time.sleep(0.5)
            if t is None:
                p.error("✗ REQUIRED: Ollama did not become ready within 30 s — translation "
                        "cannot run without it (no fallback). Check `ollama serve`.")
                return
            p.success(f"✓ Ollama serve is up ({where})")
        else:
            p.success(f"✓ Connected to Ollama ({where})")
        names = [m.get("name", "") for m in (t.get("models") or [])]
        model = config.OLLAMA_MODEL
        stem = model.split(":")[0]
        if any(n == model or n.startswith(stem) for n in names):
            p.dim(f"  model {model} present")
        else:
            p.error(f"✗ REQUIRED: Ollama model {model} is missing — translation cannot run "
                    f"without it (no fallback). Run `ollama pull {model}`.")
            return

        # GPU-vs-CPU placement (B1) — the original bug class: "up" ≠ "on the GPU".
        # Determined the REAL way (warm-load + /api/ps size_vram), not a bare ping.
        status, detail = self._ollama_gpu_status(base)
        if status == "gpu":
            p.success(f"✓ Ollama on GPU ({detail})")
        elif status == "cpu":
            p.warn("⚠ Ollama is up but running on CPU — GPU expected; translations will be "
                   "slow. Check the GPU/driver and the Ollama version.")
        else:
            p.dim(f"  (Ollama GPU/CPU placement undetermined: {detail})")

    def _preflight_accelerators(self) -> None:
        """Surface the ML tools' GPU/CPU placement up front (B3), reusing the EXISTING
        real-kernel checks — so a silent CPU fallback is visible, not buried in a log."""
        p = self.presenter
        try:
            import esm_bridge
            if esm_bridge._check_venv312_cuda():
                p.success("✓ ESM / ESMFold / ThermoMPNN: GPU (venv312, cu128 sm_120)")
            else:
                p.warn("⚠ ESM / ESMFold / ThermoMPNN: CPU fallback (venv312 CUDA probe "
                       "failed — GPU expected; inference will be slow)")
        except Exception:
            pass
        try:
            from rasp_bridge import RaSPBridge
            if RaSPBridge().is_available():
                p.dim("  RaSP: CPU (by design — the 2022 stack can't drive Blackwell; ~11 s/chain)")
        except Exception:
            pass

    def _preflight_wsl(self) -> None:
        p = self.presenter
        try:
            from wsl_bridge import WSLBridge
            wsl = WSLBridge()
            if wsl.is_available():
                if wsl.check_pyrosetta():
                    p.success("✓ Rosetta: local (PyRosetta via WSL2) — publication quality")
                else:
                    p.warn("⚠ WSL2 available — PyRosetta not installed (run pyrosetta_installer).")
                self.session.wsl_available = True
            else:
                p.dim("✓ Rosetta: DynaMut2 (screening quality) — WSL2 not installed")
                self.session.wsl_available = False
        except Exception:
            pass  # WSL2 status is informational only

    def _preflight_restore(self) -> None:
        from session_state import SessionState
        state, err = SessionState.try_load("session.json")
        if err or state is None:
            return
        if not (state.structures or state.scan_results
                or getattr(state, "double_mutant_results", None) or state.command_history
                or getattr(state, "design_sessions", None)):
            return
        summary = (f"{len(state.structures)} structure(s), "
                   f"{len(state.scan_results)} scan result(s), "
                   f"{len(getattr(state, 'design_sessions', {}) or {})} workbench design(s), "
                   f"{len(state.command_history)} prior command(s).")
        choice = self._blocking_restore(summary)
        if choice == "restore":
            self.session = state
            self.router = ToolRouter(self.bridge, self.session)
            # Re-point the workbench at the RESTORED session — it was constructed with the old
            # (empty) one, so without this its rehydrate reads the wrong object and edits never
            # persist into the restored file.
            self.workbench.attach_session(state)
            # Re-DISPLAY the workbench designs on the UI thread once preflight finishes (this
            # runs on the preflight worker). ChimeraX is left running across an app restart, so
            # the crystal + fold models are still open → show_model → workbench.load_model
            # rehydrates the persisted variants/results against the live models (no re-fold).
            self._restore_mids = list((getattr(state, "design_sessions", {}) or {}).keys())
            self.presenter.success(f"✓ Restored session: {summary}")
        elif choice == "clear":
            try:
                Path("session.json").unlink()
            except Exception:
                pass
            self.presenter.dim("Previous session cleared.")
        else:
            self.presenter.dim("Starting fresh (previous session kept).")

    def _blocking_restore(self, summary: str) -> str:
        """Worker-thread blocking restore prompt (reuses the Stage-2 worker-block seam)."""
        q: "queue.Queue" = queue.Queue(maxsize=1)
        self._sig.ask.emit("restore", f"A previous session was found: {summary}", q)
        return q.get()

    # ── teardown: stop exactly what the app started ───────────────────────────────
    def closeEvent(self, event) -> None:
        # Persist the session (parity with the console's save-on-exit) so the next launch's
        # restore prompt has something to restore.
        try:
            self.session.save("session.json")
        except Exception:
            pass
        # Persist the user's grid↔console divider position for the next launch.
        try:
            self._settings.setValue("ui/splitState", self.split.saveState())
        except Exception:
            pass
        # Ollama (a bundled daemon WE started) is stopped. ChimeraX is left running like
        # the console REPL — it is the user's structure viewer; killing it on window close
        # would discard their loaded models.
        for svc in self._services:
            try:
                svc.stop()
            except Exception:
                pass
        super().closeEvent(event)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="StructureBot — unified GUI front-end")
    ap.add_argument("--port", type=int, default=0, help="ChimeraX REST port")
    ap.add_argument("--no-auto-proceed", action="store_true",
                    help="always require explicit Execute (no countdown)")
    a = ap.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    win = StructureBotWindow(
        port=a.port,
        auto_proceed=not a.no_auto_proceed,
        auto_proceed_delay=config.AUTO_PROCEED_DELAY,
    )
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
