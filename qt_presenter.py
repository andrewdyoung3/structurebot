"""
qt_presenter.py
---------------
QtPresenter — the GUI implementation of the Presenter contract. The SAME RequestEngine
that the console drives is driven here from a worker thread; QtPresenter is what makes
that work unchanged (the engine stays straight-line):

  • OUTPUT methods render the Rich renderable to an HTML fragment ON THE WORKER THREAD
    (pure, no widget access) and emit a queued Qt signal → a UI-thread slot appends it to
    the pane. Widgets are NEVER touched from the worker.
  • BLOCKING INPUT methods (ask_clarification / confirm / ask_edit / ask_yes_no) emit a
    signal to the UI thread to show the prompt, then BLOCK the worker on a thread-safe
    queue.Queue. The UI thread stays fully responsive; when the user answers, the UI slot
    puts the value in the queue → the worker unblocks → returns it to the engine.

Error-first: a render failure degrades to plain text; it never crashes the worker or the
window.
"""
from __future__ import annotations

import io
import queue
from typing import List, Optional

from PySide6 import QtCore
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme

from presenter import Presenter

# Same semantic theme as the console front-end (parity of meaning → colour).
THEME = Theme({
    "cmd":     "bold cyan",
    "ok":      "bold green",
    "warn":    "bold yellow",
    "err":     "bold red",
    "dim":     "dim white",
    "info":    "blue",
    "heading": "bold cyan underline",
    "hi":      "bold white",
})

# A cancel sentinel the UI puts in the reply queue; the presenter maps it per-method.
CANCEL = object()


def render_html(renderable, width: int = 100) -> str:
    """Render a Rich markup string / Table / Panel to a self-contained HTML fragment
    (inline styles) suitable for QTextEdit.append. Pure — safe on the worker thread."""
    try:
        c = Console(record=True, theme=THEME, width=width, file=io.StringIO())
        c.print(renderable)
        return c.export_html(
            inline_styles=True,
            code_format="<pre style='margin:0;white-space:pre-wrap;font-family:Consolas,monospace'>{code}</pre>",
        )
    except Exception:
        # error-first: never let a render failure escape into the worker/engine
        return f"<pre style='margin:0'>{escape(str(renderable))}</pre>"


class PresenterSignals(QtCore.QObject):
    """The worker→UI bridge. Created on the UI thread; emits cross-thread are queued."""
    append_html = QtCore.Signal(str)            # output → pane
    set_busy    = QtCore.Signal(bool, str)      # (busy, label) → status bar
    ask         = QtCore.Signal(str, object, object)   # (kind, payload, reply_queue:Queue)


class QtPresenter(Presenter):
    """Drives the pane/status/prompts of the unified GUI from the worker thread."""

    def __init__(self, signals: PresenterSignals):
        self._sig = signals
        self.cancelled = False     # set by the window's Cancel; blocking asks short-circuit

    # ── helpers ─────────────────────────────────────────────────────────────────
    def _emit(self, renderable) -> None:
        self._sig.append_html.emit(render_html(renderable))

    # ── text ──────────────────────────────────────────────────────────────────
    def info(self, text: str) -> None:
        self._emit(f"[info]{escape(text)}[/info]")

    def warn(self, text: str) -> None:
        self._emit(f"[warn]{escape(text)}[/warn]")

    def error(self, text: str) -> None:
        self._emit(f"[err]{escape(text)}[/err]")

    def success(self, text: str) -> None:
        self._emit(f"[ok]{escape(text)}[/ok]")

    def dim(self, text: str) -> None:
        self._emit(f"[dim]{escape(text)}[/dim]")

    def blank(self) -> None:
        self._sig.append_html.emit("")

    def markup(self, text: str) -> None:
        # raw Rich markup (router sequence/selection fast-paths) → HTML in the pane
        self._sig.append_html.emit(render_html(text))

    def active_site_ok(self, msg: str) -> None:
        self._emit(f"[ok]✓[/ok] {escape(msg)}")

    # ── structured (same Rich renderables as ConsolePresenter) ────────────────────
    def show_commands(self, commands: List[str], explanations: List[str], confidence: str) -> None:
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "white")
        title = (f"[bold]Proposed Commands[/bold]  "
                 f"[{conf_color}]confidence: {confidence}[/{conf_color}]")
        table = Table(title=title, border_style="blue", show_lines=True)
        table.add_column("#",          style="dim",  width=3, no_wrap=True)
        table.add_column("Command",    style="cmd",  min_width=36)
        table.add_column("What it does", style="white")
        for i, (cmd, exp) in enumerate(zip(commands, explanations), 1):
            table.add_row(str(i), escape(cmd), escape(exp or "—"))
        self._emit(table)

    def show_tool_pipeline(self, result: dict) -> None:
        steps = result.get("tool_steps_info", [])
        if not steps:
            return
        table = Table(title="[bold]Tool Pipeline[/bold]", border_style="magenta", show_lines=True)
        table.add_column("#",    style="dim",   width=3,  no_wrap=True)
        table.add_column("Tool", style="bold",  width=14, no_wrap=True)
        table.add_column("Action", style="white")
        for i, step in enumerate(steps, 1):
            icon = step.get("icon", "⚙️")
            tool = step.get("tool", "?")
            desc = step.get("description", "")
            table.add_row(str(i), f"{icon} {tool}", escape(desc))
        self._emit(table)

    def show_interface_summary(self, result: dict) -> None:
        for step in result.get("tool_step_results", []):
            if step.get("tool") == "assembly_analyser" and step.get("success"):
                data = step.get("data", {})
                header  = data.get("header", "")
                summary = data.get("interface_summary", "")
                if header:
                    self._emit(f"[bold]🔗 {escape(header)}[/bold]")
                if summary:
                    self._emit(f"[info]{escape(summary)}[/info]")
                for w in data.get("warnings", []):
                    self._emit(f"[warn]⚠ {escape(w)}[/warn]")
                excluded = data.get("excluded_count", 0)
                if excluded:
                    self._emit(f"[dim]  → {excluded} residue(s) will be excluded from mutation scan[/dim]")

    def tool_summary(self, icon: str, ok: bool, summary: str = "", tool: str = "", error: str = "") -> None:
        if ok:
            self._emit(f"  [ok]✓[/ok] {icon} {escape(summary)}")
        else:
            self._emit(f"  [err]✗[/err] {icon} {tool}: {escape(error)}")

    def analysis_panel(self, summary: str) -> None:
        self._emit(Panel(summary, title="[bold green]Analysis Summary[/bold green]",
                         border_style="green", padding=(1, 2)))

    def command_result(self, cmd: str, ok: bool, value=None, error=None, warning=None) -> None:
        if not ok:
            self._emit(f"  [err]✗[/err] [cmd]{escape(cmd)}[/cmd]")
            self._emit(f"      [err]{escape(str(error)[:120])}[/err]")
        else:
            val = (value or "").replace("\n", " ").strip()[:60]
            suffix = f" [dim]→ {escape(val)}[/dim]" if val else ""
            self._emit(f"  [ok]✓[/ok] [cmd]{escape(cmd)}[/cmd]{suffix}")
            if warning:
                self._emit(f"      [warn]⚠ {escape(str(warning))}[/warn]")

    def blocked(self, cmd: str, error: str) -> None:
        self._emit(f"  [err]✗ Blocked (emission guard):[/err] [cmd]{escape(cmd)}[/cmd]")
        self._emit(f"      [err]{escape(error[:140])}[/err]")

    def completed(self, n: int) -> None:
        self._emit(f"  [dim]Completed {n} command(s).[/dim]")

    def translation_declined(self, exc: Exception) -> None:
        self._emit(f"[warn]⚠ The model returned no usable translation: {escape(str(exc))}[/warn]")
        self._emit("[dim]An automatic retry was already attempted. This is usually transient — "
                   "try the same request again, or rephrase it slightly.[/dim]")

    def translation_error(self, exc: Exception) -> None:
        self._emit(f"[warn]⚠ Couldn't translate that request: {escape(str(exc))}[/warn]")
        self._emit("[dim]Is the local Ollama model reachable? Try again, or rephrase it.[/dim]")

    # ── status / long-running ─────────────────────────────────────────────────
    def status(self, label: str):
        sig = self._sig

        class _CM:
            def __enter__(_s):
                sig.set_busy.emit(True, label)
                return _s

            def __exit__(_s, *a):
                sig.set_busy.emit(False, "")
                return False
        return _CM()

    def running_tools(self, label: str, eta_s: float = 0.0, needs_timer: bool = False):
        # In the GUI the busy indicator covers the long run; no text ticker needed.
        return self.status(label)

    def tool_status(self, msg: str) -> None:
        self._emit(f"  [info]{escape(msg)}[/info]")

    # ── blocking input (worker thread blocks on a queue; UI thread stays live) ────
    def _ask(self, kind: str, payload, on_cancel):
        if self.cancelled:
            return on_cancel
        q: "queue.Queue" = queue.Queue(maxsize=1)
        self._sig.ask.emit(kind, payload, q)
        ans = q.get()                      # BLOCKS the worker thread, not the UI thread
        if ans is CANCEL:
            return on_cancel
        return ans

    def ask_clarification(self, question: str) -> str:
        return self._ask("clarification", question, on_cancel="")

    def confirm(self, confidence: str) -> Optional[str]:
        return self._ask("confirm", confidence, on_cancel=None)

    def ask_edit(self, original: List[str]) -> List[str]:
        return self._ask("edit", list(original), on_cancel=[])

    def ask_yes_no(self, question: str, default: str = "y") -> bool:
        return bool(self._ask("yesno", question, on_cancel=False))
