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
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets
from rich.markup import escape

import config
from chimerax_bridge import ChimeraXBridge
from colabfold_bridge import ColabFoldBridge
from translator import CommandTranslator, probe_chimerax_verbs
from session_state import SessionState
from tool_router import ToolRouter
from request_engine import RequestEngine
from qt_presenter import QtPresenter, PresenterSignals, render_html, CANCEL
from seq_editor.controller import SequenceEditorController
from seq_editor.view import _ChainGrid


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
            try:
                probe_chimerax_verbs(self._engine.bridge.run_command)
            except Exception:
                pass  # verb-guard probe is best-effort
            self._engine.handle_request(self._text, self._presenter)
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
        self._pending_focus: List[str] = []   # model ids to focus after the request

        self._build_ui()
        self._connect()

    # ── UI ──────────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        self.setWindowTitle("StructureBot")
        self.resize(1000, 720)

        self.tabs = QtWidgets.QTabWidget()
        self.output = QtWidgets.QTextEdit(readOnly=True)
        self.output.setStyleSheet("QTextEdit{background:#1e1e1e;color:#dddddd;}")
        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Ask StructureBot…  (e.g. \"open 1hsg and show it as a cartoon\")")

        bottom = QtWidgets.QWidget()
        bl = QtWidgets.QVBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.addWidget(self.output)
        bl.addWidget(self.input)

        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        split.addWidget(self.tabs)
        split.addWidget(bottom)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        self.setCentralWidget(split)

        tb = self.addToolBar("main")
        self.cancel_action = tb.addAction("Cancel", self._on_cancel)
        self.cancel_action.setEnabled(False)
        self.statusBar().showMessage("Ready")

    def _connect(self) -> None:
        self._sig.append_html.connect(self._on_append)
        self._sig.set_busy.connect(self._on_busy)
        self._sig.ask.connect(self._on_ask)
        self.input.returnPressed.connect(self._on_submit)

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
        self.input.clear()
        self.output.append(f"<pre style='margin:0;color:#7fd1ff'><b>&gt; {escape(text)}</b></pre>")
        self._start_request(text)

    def _start_request(self, text: str) -> None:
        self._in_flight = True
        self._pending_focus = []
        self.presenter.cancelled = False
        self.input.setEnabled(False)
        self.cancel_action.setEnabled(True)
        w = _RequestWorker(self.engine, text, self.presenter)
        w.signals.done.connect(self._on_request_done)
        w.signals.failed.connect(self._on_request_failed)
        self._worker = w
        self._pool.start(w)

    @QtCore.Slot()
    def _on_request_done(self) -> None:
        focus = list(self._pending_focus)
        self._finish_request()
        for mid in focus:                       # after an open, focus the new model's tab
            self.show_model(mid)

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
        if any(k[0] == mid for k in self._grids):
            self._focus_model(mid)
            return
        w = _LoadModelWorker(self.controller, mid)
        w.signals.done.connect(self._on_model_loaded)
        w.signals.failed.connect(lambda e: self.presenter.warn(f"show #{mid} failed: {e}"))
        self._pool.start(w)

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
