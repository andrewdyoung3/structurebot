"""
presenter.py
------------
The Presenter contract + the ConsolePresenter parity oracle.

The RequestEngine (request_engine.py) is UI-agnostic: it orchestrates a request
straight-line and performs ALL input/output through a Presenter. The interactive
methods (ask_clarification / confirm / ask_edit / ask_yes_no) are BLOCKING from the
engine's point of view — the engine calls them and gets an answer back; HOW the answer
arrives (a console Prompt.ask, or a GUI widget that blocks a worker thread) is the
presenter's business.

ConsolePresenter reproduces today's Rich console behaviour EXACTLY — it is the parity
oracle: the console REPL on the extracted engine must be indistinguishable from the
pre-extraction REPL. Every method below is a verbatim move of the corresponding
console.print / Prompt.ask site from main.py.
"""
from __future__ import annotations

import sys
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional

from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from translator import is_usage_cap_error


# ── Windows-only keyboard polling (for the auto-proceed countdown) ──────────────
if sys.platform == "win32":
    import msvcrt

    def _kbhit() -> bool:
        return msvcrt.kbhit()
else:
    import select

    def _kbhit() -> bool:
        return bool(select.select([sys.stdin], [], [], 0)[0])


# ── Elapsed-time ticker for long computational phases ───────────────────────────

class _ElapsedTicker:
    """
    Background thread that prints an elapsed-time message every *interval* seconds.
    Designed for long-running computational tool phases.

    Usage::
        with _ElapsedTicker("Running mutation_scan", interval=30):
            results = scanner.scan(...)
    """

    def __init__(self, prefix: str, interval: int = 30, eta_s: float = 0.0):
        self._prefix   = prefix
        self._interval = interval
        self._eta_s    = max(0.0, float(eta_s))   # 0 = unknown; omit ETA from the line
        self._start    = time.time()
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "_ElapsedTicker":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            elapsed = int(time.time() - self._start)
            mins, secs = divmod(elapsed, 60)
            # When an approximate ETA is known, show "elapsed / ~ETA" so a long
            # fold reads as progress rather than a hang.
            if self._eta_s > 0:
                em, es = divmod(int(self._eta_s), 60)
                tail = f" / ~{em:02d}:{es:02d} est"
            else:
                tail = ""
            msg = f"  {self._prefix}... ({mins:02d}:{secs:02d} elapsed{tail})"
            try:
                print(msg, flush=True)
            except Exception:
                pass

    def __enter__(self) -> "_ElapsedTicker":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()


# ── The Presenter contract ───────────────────────────────────────────────────────

class Presenter(ABC):
    """The I/O surface the RequestEngine speaks. A console or a GUI implements it."""

    # text
    @abstractmethod
    def info(self, text: str) -> None: ...
    @abstractmethod
    def warn(self, text: str) -> None: ...
    @abstractmethod
    def error(self, text: str) -> None: ...
    @abstractmethod
    def success(self, text: str) -> None: ...
    @abstractmethod
    def dim(self, text: str) -> None: ...
    @abstractmethod
    def blank(self) -> None: ...
    @abstractmethod
    def markup(self, text: str) -> None: ...          # render a raw Rich-markup string
    @abstractmethod
    def active_site_ok(self, msg: str) -> None: ...    # the dispatch active-site ✓ line

    # structured render
    @abstractmethod
    def show_commands(self, commands: List[str], explanations: List[str], confidence: str) -> None: ...
    @abstractmethod
    def show_tool_pipeline(self, result: dict) -> None: ...
    @abstractmethod
    def show_interface_summary(self, result: dict) -> None: ...
    @abstractmethod
    def tool_summary(self, icon: str, ok: bool, summary: str = "", tool: str = "", error: str = "") -> None: ...
    @abstractmethod
    def analysis_panel(self, summary: str) -> None: ...
    @abstractmethod
    def command_result(self, cmd: str, ok: bool, value=None, error=None, warning=None) -> None: ...
    @abstractmethod
    def blocked(self, cmd: str, error: str) -> None: ...
    @abstractmethod
    def completed(self, n: int) -> None: ...
    @abstractmethod
    def translation_declined(self, exc: Exception) -> None: ...
    @abstractmethod
    def translation_error(self, exc: Exception) -> None: ...

    # status / long-running
    @abstractmethod
    def status(self, label: str): ...
    @abstractmethod
    def running_tools(self, label: str, eta_s: float = 0.0, needs_timer: bool = False): ...
    @abstractmethod
    def tool_status(self, msg: str) -> None: ...

    # interactive (BLOCKING from the engine's view)
    @abstractmethod
    def ask_clarification(self, question: str) -> str: ...
    @abstractmethod
    def confirm(self, confidence: str) -> Optional[str]: ...   # "proceed" | "edit" | None
    @abstractmethod
    def ask_edit(self, original: List[str]) -> List[str]: ...
    @abstractmethod
    def ask_yes_no(self, question: str, default: str = "y") -> bool: ...


# ── The console parity oracle ──────────────────────────────────────────────────

class ConsolePresenter(Presenter):
    """Reproduces today's Rich console behaviour exactly. `console` is injected (the
    module-level Rich Console from main, or a test mock) so tests that patch
    `main.console` see the calls."""

    def __init__(self, console, auto_proceed: bool = True, auto_proceed_delay: int = 2):
        self._console = console
        self.auto_proceed = auto_proceed
        self.auto_proceed_delay = auto_proceed_delay

    # ── text ──────────────────────────────────────────────────────────────────
    def info(self, text: str) -> None:
        self._console.print(f"[info]{escape(text)}[/info]")

    def warn(self, text: str) -> None:
        self._console.print(f"[warn]{escape(text)}[/warn]")

    def error(self, text: str) -> None:
        self._console.print(f"[err]{escape(text)}[/err]")

    def success(self, text: str) -> None:
        self._console.print(f"[ok]{escape(text)}[/ok]")

    def dim(self, text: str) -> None:
        self._console.print(f"[dim]{escape(text)}[/dim]")

    def blank(self) -> None:
        self._console.print()

    def markup(self, text: str) -> None:
        # raw Rich markup built by the router (sequence/selection fast-paths) — printed
        # verbatim, exactly as the console _dispatch_input does.
        self._console.print(text)

    def active_site_ok(self, msg: str) -> None:
        self._console.print(f"[ok]✓[/ok] {escape(msg)}")

    # ── structured render ───────────────────────────────────────────────────────
    def show_commands(self, commands: List[str], explanations: List[str], confidence: str) -> None:
        conf_color = {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "white")
        title = (
            f"[bold]Proposed Commands[/bold]  "
            f"[{conf_color}]confidence: {confidence}[/{conf_color}]"
        )
        table = Table(title=title, border_style="blue", show_lines=True)
        table.add_column("#",          style="dim",  width=3, no_wrap=True)
        table.add_column("Command",    style="cmd",  min_width=36)
        table.add_column("What it does", style="white")
        for i, (cmd, exp) in enumerate(zip(commands, explanations), 1):
            table.add_row(str(i), escape(cmd), escape(exp or "—"))
        self._console.print(table)

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
        self._console.print(table)
        self._console.print()

    def show_interface_summary(self, result: dict) -> None:
        step_results = result.get("tool_step_results", [])
        for step in step_results:
            if step.get("tool") == "assembly_analyser" and step.get("success"):
                data = step.get("data", {})
                summary = data.get("interface_summary", "")
                header  = data.get("header", "")
                warnings = data.get("warnings", [])

                if header:
                    self._console.print(f"\n  [bold]🔗 {escape(header)}[/bold]")
                if summary:
                    self._console.print(f"  [info]{escape(summary)}[/info]")
                for w in warnings:
                    self._console.print(f"  [warn]⚠ {escape(w)}[/warn]")
                excluded = data.get("excluded_count", 0)
                if excluded:
                    self._console.print(
                        f"  [dim]  → {excluded} residue(s) will be excluded from mutation scan[/dim]"
                    )

    def tool_summary(self, icon: str, ok: bool, summary: str = "", tool: str = "", error: str = "") -> None:
        if ok:
            self._console.print(f"  [ok]✓[/ok] {icon} {escape(summary)}")
        else:
            self._console.print(f"  [err]✗[/err] {icon} {tool}: {escape(error)}")

    def analysis_panel(self, summary: str) -> None:
        self._console.print()
        self._console.print(Panel(
            summary,
            title="[bold green]Analysis Summary[/bold green]",
            border_style="green",
            padding=(1, 2),
        ))

    def command_result(self, cmd: str, ok: bool, value=None, error=None, warning=None) -> None:
        if not ok:
            self._console.print(f"  [err]✗[/err] [cmd]{escape(cmd)}[/cmd]")
            self._console.print(f"      [err]{escape(str(error)[:120])}[/err]")
        else:
            val = (value or "").replace("\n", " ").strip()[:60]
            suffix = f" [dim]→ {escape(val)}[/dim]" if val else ""
            self._console.print(f"  [ok]✓[/ok] [cmd]{escape(cmd)}[/cmd]{suffix}")
            if warning:
                self._console.print(f"      [warn]⚠ {escape(str(warning))}[/warn]")

    def blocked(self, cmd: str, error: str) -> None:
        self._console.print(
            f"  [err]✗ Blocked (emission guard):[/err] [cmd]{escape(cmd)}[/cmd]"
        )
        self._console.print(f"      [err]{escape(error[:140])}[/err]")

    def completed(self, n: int) -> None:
        self._console.print(f"\n  [dim]Completed {n} command(s).[/dim]")

    def translation_declined(self, exc: Exception) -> None:
        self._console.print(
            f"[warn]⚠ The model returned no usable translation: {escape(str(exc))}[/warn]"
        )
        self._console.print(
            "[dim]An automatic retry was already attempted. This is usually "
            "transient — try the same request again, or rephrase it slightly. "
            "(The request is fine for a structural-biology tool.)[/dim]"
        )

    def translation_error(self, exc: Exception) -> None:
        sys.stderr.write(f"[main] translation error: {type(exc).__name__}: {exc}\n")
        if is_usage_cap_error(exc):
            self._console.print(
                f"[warn]⚠ Claude API usage limit reached: {escape(str(exc))}[/warn]"
            )
            self._console.print(
                "[dim]Set TRANSLATOR_BACKEND=ollama to use the local model, or wait "
                "for the limit to reset (see the date above).[/dim]"
            )
        else:
            self._console.print(
                f"[warn]⚠ Couldn't translate that request: {escape(str(exc))}[/warn]"
            )
            self._console.print(
                "[dim]Returning to the prompt — try again, or rephrase it.[/dim]"
            )

    # ── status / long-running ─────────────────────────────────────────────────
    def status(self, label: str):
        return self._console.status(f"[cyan]{label}[/cyan]")

    @contextmanager
    def running_tools(self, label: str, eta_s: float = 0.0, needs_timer: bool = False):
        ticker = _ElapsedTicker(label, interval=30, eta_s=eta_s).start() if needs_timer else None
        try:
            with self._console.status("[cyan]Running computational tools…[/cyan]"):
                yield
        finally:
            if ticker is not None:
                ticker.stop()

    def tool_status(self, msg: str) -> None:
        self._console.print(f"  [info]{escape(msg)}[/info]")

    # ── interactive (blocking) ──────────────────────────────────────────────────
    def ask_clarification(self, question: str) -> str:
        self._console.print(f"\n[warn]❓ {escape(question)}[/warn]")
        return Prompt.ask("[hi]Answer[/hi]").strip()

    def confirm(self, confidence: str) -> Optional[str]:
        """Returns "proceed", "edit", or None (cancel). High/medium confidence →
        auto-proceed countdown if auto_proceed is on; low → always prompt. ESC during
        countdown → cancel immediately."""
        if self.auto_proceed and confidence in ("high", "medium"):
            result = self._countdown(self.auto_proceed_delay)
            if result == "escaped":
                self._console.print("[dim]Cancelled.[/dim]")
                return None
            return "proceed" if result else self._manual_confirm()
        return self._manual_confirm()

    def _manual_confirm(self) -> Optional[str]:
        choice = Prompt.ask(
            "\n[hi]Execute?[/hi] [dim][[y]es / [n]o / [e]dit][/dim]",
            default="y",
        ).strip().lower()
        if choice in ("n", "no"):
            self._console.print("[dim]Cancelled.[/dim]")
            return None
        if choice in ("e", "edit"):
            return "edit"
        return "proceed"

    def _countdown(self, seconds: int):
        """Countdown before auto-executing. Returns True to proceed, False on any other
        key (→ manual prompt), or "escaped" if ESC was pressed (→ cancel)."""
        for remaining in range(seconds, 0, -1):
            self._console.print(
                f"\r  [dim]Auto-executing in {remaining}s… "
                f"(press any key to pause, ESC to cancel)[/dim]",
                end="",
            )
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if _kbhit():
                    if sys.platform == "win32":
                        ch = msvcrt.getwch()
                    else:
                        ch = sys.stdin.read(1)
                    self._console.print()
                    if ch == "\x1b":
                        return "escaped"
                    return False
                time.sleep(0.05)
        self._console.print()
        return True

    def ask_edit(self, original: List[str]) -> List[str]:
        self._console.print("[warn]Edit mode — enter each command, blank line to finish:[/warn]")
        for cmd in original:
            self._console.print(f"  [cmd]{escape(cmd)}[/cmd]")
        self._console.print()
        edited: List[str] = []
        while True:
            line = Prompt.ask("[dim]>[/dim]", default="").strip()
            if not line:
                break
            edited.append(line)
        return edited

    def ask_yes_no(self, question: str, default: str = "y") -> bool:
        choice = Prompt.ask(
            f"[hi]{question}[/hi] [dim][[y]es / [n]o][/dim]",
            default=default,
        ).strip().lower()
        return choice in ("y", "yes", "")
