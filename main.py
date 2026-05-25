"""
main.py
-------
StructureBot — Natural Language Interface for UCSF ChimeraX.

Usage:
    python main.py
    python main.py --resume              # restore last session.json
    python main.py --chimerax PATH       # override ChimeraX path
    python main.py --port 60001          # custom REST port
    python main.py --no-auto-proceed     # always require explicit confirmation
"""

# ── Load .env.local FIRST (before any module that reads env vars) ─────────────
import config
config.load_env_file()

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ── Windows-only keyboard polling ─────────────────────────────────────────────
if sys.platform == "win32":
    import msvcrt
    def _kbhit() -> bool:
        return msvcrt.kbhit()
    def _getch() -> None:
        msvcrt.getch()
else:
    import select
    def _kbhit() -> bool:
        return bool(select.select([sys.stdin], [], [], 0)[0])
    def _getch() -> None:
        sys.stdin.read(1)

# ── Rich UI ───────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markup import escape
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.rule import Rule
    from rich.table import Table
    from rich.theme import Theme
except ImportError:
    print("ERROR: 'rich' not installed.  Run: pip install rich")
    sys.exit(1)

from chimerax_bridge import ChimeraXBridge
from translator import CommandTranslator
from session_state import SessionState
from tool_router import ToolRouter

# ── Console ───────────────────────────────────────────────────────────────────

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
console = Console(theme=THEME)

# ── Constants ─────────────────────────────────────────────────────────────────

SESSION_FILE  = "session.json"
BANNER = """\
[bold cyan]StructureBot[/bold cyan] [dim]— Natural Language Interface for UCSF ChimeraX[/dim]

  [dim]Example prompts:[/dim]
    [cyan]Open 1HSG and show it as a cartoon[/cyan]
    [cyan]Color each chain a different color[/cyan]
    [cyan]Show the ligand as spheres and color it by element[/cyan]
    [cyan]Find all residues within 4 Å of the ligand[/cyan]
    [cyan]Save a publication-quality image to my desktop as figure1.png[/cyan]
    [cyan]Run CamSol solubility analysis on the loaded structure[/cyan]
    [cyan]Color the structure by evolutionary conservation (ESM-2)[/cyan]
    [cyan]Calculate ddG for mutation V82A in chain A[/cyan]
    [cyan]Suggest mutations to improve solubility of chain A[/cyan]

  [dim]Type [bold]help[/bold] for more examples or [bold]quit[/bold] to exit.[/dim]
"""

HELP_TEXT = """
[heading]StructureBot — Example Prompts[/heading]

[bold]Loading & display[/bold]
  open 1HSG and show it as a ribbon diagram
  load my_protein.pdb as ball-and-stick
  open 1AKE and 4AKE then align them

[bold]Coloring[/bold]
  color each chain a different color
  color by secondary structure — helices blue, strands gold, loops gray
  color by B-factor using a heat-map palette
  show the electrostatic surface

[bold]Ligand analysis[/bold]
  show the ligand as spheres and color it by element
  find all residues within 4 angstroms of the ligand
  show the hydrogen bonds between the ligand and protein
  color the binding pocket by hydrophobicity

[bold]Structural comparison[/bold]
  align 1AKE onto 4AKE and report the RMSD
  show both structures side by side colored differently

[bold]Export[/bold]
  save an image to my desktop called figure1.png at 3000×3000
  save a publication-quality image with a white background

[bold]Computational analysis[/bold]
  run CamSol solubility analysis on chain A
  color the structure by aggregation-prone regions
  analyze evolutionary conservation with ESM-2
  color by conservation — conserved residues blue, variable red
  open 1HSG and show aggregation-prone patches

[bold]Protein engineering[/bold]
  calculate ddG for mutation V82A in chain A
  is mutation L75K stabilising?
  suggest mutations to improve solubility of chain A
  what mutations would reduce aggregation?
  run the full engineering pipeline on the loaded structure

[bold]Special commands[/bold]
  [cmd]history[/cmd]           show last 15 commands
  [cmd]state[/cmd]             dump current session state
  [cmd]jobs[/cmd]              show status of pending Robetta ddG jobs
  [cmd]undo[/cmd]              undo the last ChimeraX action
  [cmd]clear[/cmd]             close all models, reset session
  [cmd]save session NAME[/cmd] save ChimeraX session + state to sessions/NAME
  [cmd]load session NAME[/cmd] restore a saved session
  [cmd]reset[/cmd]             clear conversation context
  [cmd]help[/cmd]              show this help
  [cmd]quit[/cmd] / [cmd]exit[/cmd]          save & exit
"""


# ════════════════════════════════════════════════════════════════════════════════
# Application
# ════════════════════════════════════════════════════════════════════════════════

class StructureBot:
    def __init__(
        self,
        chimerax_path:    Optional[str] = None,
        port:             int = config.REST_PORT,
        resume:           bool = False,
        auto_proceed:     bool = True,
        auto_proceed_delay: int = config.AUTO_PROCEED_DELAY,
    ):
        # Bridge uses 127.0.0.1 (loopback) to match the confirmed REST endpoint
        self.bridge  = ChimeraXBridge(
            chimerax_path = chimerax_path or config.CHIMERAX_PATH,
            port          = port,
        )
        # Patch base_url to 127.0.0.1 if the bridge still uses localhost
        self.bridge.base_url = f"http://{config.REST_HOST}:{port}"
        self.bridge.run_url  = f"{self.bridge.base_url}/run"

        self.translator = CommandTranslator()
        self.session    = SessionState.load(SESSION_FILE) if resume else SessionState()
        self.router     = ToolRouter(self.bridge, self.session)

        self.auto_proceed       = auto_proceed
        self.auto_proceed_delay = auto_proceed_delay

        # Log file path
        log_name = datetime.now().strftime("session_%Y%m%d_%H%M%S.jsonl")
        self.log_file = config.LOG_DIR / log_name

    # ── Startup ───────────────────────────────────────────────────────────────

    def startup(self) -> None:
        console.print(Panel(BANNER, border_style="cyan", padding=(0, 2)))

        # 1. Check ANTHROPIC_API_KEY
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print(
                "[err]✗ ANTHROPIC_API_KEY is not set.[/err]\n"
                "  Add it to .env.local:  ANTHROPIC_API_KEY=sk-ant-...\n"
                "  Or set it in your shell before running StructureBot."
            )
            sys.exit(1)

        # 2. Verify ChimeraX path
        cx_path = Path(self.bridge.chimerax_path or "")
        if not cx_path.is_file():
            console.print(f"[err]✗ ChimeraX not found: {cx_path}[/err]")
            custom = Prompt.ask("[hi]Enter full path to ChimeraX.exe[/hi]").strip()
            if not custom or not Path(custom).is_file():
                console.print("[err]Invalid path — exiting.[/err]")
                sys.exit(1)
            self.bridge.chimerax_path = custom
        else:
            console.print(f"[info]ChimeraX:[/info] [dim]{cx_path}[/dim]")

        # 3. Start or connect
        with console.status("[cyan]Connecting to ChimeraX REST server…[/cyan]"):
            already = self.bridge.is_running()

        if already:
            console.print(
                f"[ok]✓[/ok] Connected to ChimeraX at "
                f"http://{config.REST_HOST}:{self.bridge.port}/"
            )
        else:
            console.print("[warn]ChimeraX REST server not found — launching ChimeraX…[/warn]")
            console.print("[dim]  (may take 20–40 s for ChimeraX to initialise)[/dim]")
            try:
                with console.status("[cyan]Starting ChimeraX…[/cyan]"):
                    self.bridge.start(timeout=60)
                console.print(
                    f"[ok]✓[/ok] ChimeraX started — REST server on port {self.bridge.port}."
                )
            except Exception as exc:
                console.print(f"[err]✗ Failed to start ChimeraX: {exc}[/err]")
                console.print(
                    "\n[dim]Manual fix: open ChimeraX and run in its command bar:\n"
                    "  remotecontrol rest start port 60001\n"
                    "Then re-run StructureBot.[/dim]"
                )
                sys.exit(1)

        # 4. Ping
        ping = self.bridge.ping()
        if ping["ok"]:
            ver = (ping["result"].get("value") or "")[:40].strip()
            console.print(f"[ok]✓[/ok] Ping OK ({ping['latency_ms']} ms) — {ver}")
        else:
            console.print(f"[warn]Ping failed: {ping['result'].get('error')}[/warn]")

        # 5. Resume info
        if self.session.command_history:
            console.print(
                f"[dim]Resumed session: {len(self.session.structures)} structure(s), "
                f"{len(self.session.command_history)} prior commands.[/dim]"
            )
        console.print()

    # ── REPL ──────────────────────────────────────────────────────────────────

    def run(self) -> None:
        self.startup()
        while True:
            try:
                prompt_label = self._build_prompt_label()
                user_input   = Prompt.ask(prompt_label).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Type [bold]quit[/bold] to exit cleanly.[/dim]")
                continue

            if not user_input:
                continue

            lower = user_input.lower().strip()

            # ── Special (bypass-LLM) commands ─────────────────────────────────
            if lower in ("quit", "exit", "q"):
                self._cmd_quit()
            elif lower == "history":
                self._cmd_history()
            elif lower == "state":
                self._cmd_state()
            elif lower == "undo":
                self._cmd_undo()
            elif lower == "clear":
                self._cmd_clear()
            elif lower == "reset":
                self._cmd_reset()
            elif lower == "help":
                self._cmd_help()
            elif lower == "jobs":
                self._cmd_jobs()
            elif re.match(r"^save session\b", lower):
                name = user_input.split(maxsplit=2)[2] if len(user_input.split()) > 2 else "default"
                self._cmd_save_session(name)
            elif re.match(r"^load session\b", lower):
                name = user_input.split(maxsplit=2)[2] if len(user_input.split()) > 2 else "default"
                self._cmd_load_session(name)
            else:
                self._handle_request(user_input)

    # ── Natural language pipeline ─────────────────────────────────────────────

    def _handle_request(self, user_input: str, is_retry: bool = False) -> None:
        # 1. Translate
        with console.status("[cyan]Translating…[/cyan]"):
            result = self.translator.translate(user_input, self.session)

        # 2. Route (augment with tool pipeline info; no execution yet)
        result = self.router.route(result)

        # 3. Clarification loop (max 2 rounds)
        for _ in range(2):
            q = result.get("clarification_needed")
            if not q:
                break
            console.print(f"\n[warn]❓ {escape(q)}[/warn]")
            answer = Prompt.ask("[hi]Answer[/hi]").strip()
            if not answer:
                console.print("[dim]No answer — cancelling.[/dim]")
                return
            self.translator.add_clarification(answer)
            with console.status("[cyan]Retranslating…[/cyan]"):
                result = self.translator.translate(answer, self.session)
            result = self.router.route(result)

        if result.get("clarification_needed"):
            console.print("[warn]Still ambiguous — please rephrase.[/warn]")
            return

        commands:     List[str] = result.get("commands", [])
        explanations: List[str] = result.get("explanations", [])
        warnings:     List[str] = result.get("warnings", [])
        confidence:   str       = result.get("confidence", "medium")
        has_extra     = result.get("has_extra_tools", False)
        tools_needed: List[str] = result.get("tools_needed", ["chimerax"])

        # Require at least commands OR extra tools
        if not commands and not has_extra:
            console.print("[warn]No commands generated.[/warn]")
            return

        # 4. Show warnings
        for w in warnings:
            console.print(f"[warn]⚠ {escape(w)}[/warn]")

        # 5. Preview
        console.print()
        if has_extra:
            self._show_tool_pipeline(result)
        if commands:
            self._show_preview(commands, explanations, confidence)
        elif has_extra:
            # No initial ChimeraX commands — tool output will generate viz
            console.print(
                "[dim]  (visualization commands will be generated after "
                "the tool completes)[/dim]"
            )

        # 6. Confirm / auto-proceed / edit
        should_execute = self._confirm_execution(confidence)
        if should_execute is None:
            return  # cancelled
        if should_execute == "edit" and commands:
            commands = self._edit_commands(commands)
            if not commands:
                return

        console.print()

        # 7. Execute initial ChimeraX commands (if any)
        all_commands = list(commands)
        success      = True
        failed_cmd:  Optional[str] = None
        error_msg:   Optional[str] = None

        if commands:
            success, failed_cmd, error_msg = self._execute_commands(commands)

        # 8. Execute extra tools (CamSol, ESM, etc.) if initial phase succeeded
        if success and has_extra:
            def _status(msg: str) -> None:
                console.print(f"  [info]{msg}[/info]")

            with console.status("[cyan]Running computational tools…[/cyan]"):
                result = self.router.execute(result, status_callback=_status)

            # Show tool summaries
            summaries = result.get("tool_summaries", {})
            for tool, summary in summaries.items():
                icon = ToolRouter._TOOL_ICONS.get(tool, "⚙️")
                if result.get("pipeline_success"):
                    console.print(f"  [ok]✓[/ok] {icon} {escape(summary)}")
                else:
                    err = result.get("pipeline_error", "unknown error")
                    console.print(f"  [err]✗[/err] {icon} {tool}: {escape(err)}")

            if not result.get("pipeline_success"):
                err = result.get("pipeline_error", "")
                console.print(f"\n[err]Tool pipeline failed: {escape(err[:120])}[/err]")
                # Keep going — viz commands might still be partially available

            # 9. Execute visualization commands generated by the tools
            viz_cmds = result.get("all_viz_commands", [])
            viz_exps = result.get("all_viz_explanations", [])
            if viz_cmds:
                console.print()
                console.print("[dim]  Applying visualization…[/dim]")
                self._show_preview(viz_cmds, viz_exps, "high")
                viz_ok, viz_failed, viz_err = self._execute_commands(viz_cmds)
                if not viz_ok:
                    console.print(f"[warn]  Visualization command failed: {escape(viz_err or '')}[/warn]")
                all_commands.extend(viz_cmds)

        # 10. Auto-fix on first failure (once only, ChimeraX commands only)
        if not success and not is_retry and failed_cmd and error_msg:
            console.print("\n[warn]Asking Claude for a corrected command…[/warn]")
            fix = self.translator.translate_error_fix(failed_cmd, error_msg, self.session)
            fix_cmds = fix.get("commands", [])
            fix_exps = fix.get("explanations", [])
            if fix_cmds:
                console.print("\n[warn]Suggested correction:[/warn]")
                self._show_preview(fix_cmds, fix_exps, fix.get("confidence", "medium"))
                choice = Prompt.ask(
                    "[hi]Apply fix?[/hi] [dim][[y]es / [n]o][/dim]",
                    default="y",
                ).strip().lower()
                if choice in ("y", "yes", ""):
                    fix_success, _, _ = self._execute_commands(fix_cmds)
                    if fix_success:
                        all_commands.extend(fix_cmds)

        # 11. Update state
        self.session.add_to_history(user_input, all_commands, success=success, error=error_msg)
        self._maybe_update_structure_state(all_commands)
        self.translator.trim_history()
        self._log_exchange(user_input, all_commands, success, error_msg)

    def _show_tool_pipeline(self, result: dict) -> None:
        """Display the tool pipeline before the ChimeraX command preview."""
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
        console.print(table)
        console.print()

    def _show_preview(
        self,
        commands:     List[str],
        explanations: List[str],
        confidence:   str,
    ) -> None:
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
        console.print(table)

    def _confirm_execution(self, confidence: str) -> Optional[str]:
        """
        Returns "proceed", "edit", or None (cancel).
        High/medium confidence → auto-proceed countdown if auto_proceed is on.
        Low confidence → always prompt.
        """
        if self.auto_proceed and confidence in ("high", "medium"):
            proceeded = self._countdown(self.auto_proceed_delay)
            return "proceed" if proceeded else self._manual_confirm()
        return self._manual_confirm()

    def _manual_confirm(self) -> Optional[str]:
        choice = Prompt.ask(
            "\n[hi]Execute?[/hi] [dim][[y]es / [n]o / [e]dit][/dim]",
            default="y",
        ).strip().lower()
        if choice in ("n", "no"):
            console.print("[dim]Cancelled.[/dim]")
            return None
        if choice in ("e", "edit"):
            return "edit"
        return "proceed"

    def _countdown(self, seconds: int) -> bool:
        """
        Countdown before auto-executing.
        Returns True to proceed, False if the user pressed any key.
        """
        for remaining in range(seconds, 0, -1):
            console.print(
                f"\r  [dim]Auto-executing in {remaining}s… "
                f"(press any key to pause)[/dim]",
                end="",
            )
            deadline = time.time() + 1.0
            while time.time() < deadline:
                if _kbhit():
                    _getch()
                    console.print()
                    return False
                time.sleep(0.05)
        console.print()
        return True

    def _edit_commands(self, original: List[str]) -> List[str]:
        console.print("[warn]Edit mode — enter each command, blank line to finish:[/warn]")
        for cmd in original:
            console.print(f"  [cmd]{escape(cmd)}[/cmd]")
        console.print()
        edited: List[str] = []
        while True:
            line = Prompt.ask("[dim]>[/dim]", default="").strip()
            if not line:
                break
            edited.append(line)
        return edited

    def _execute_commands(
        self, commands: List[str]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute via bridge, print results. Returns (all_ok, failed_cmd, error_msg)."""
        with console.status("[cyan]Executing…[/cyan]"):
            results = self.bridge.run_commands(commands)

        first_err_cmd: Optional[str] = None
        first_err_msg: Optional[str] = None

        for r in results:
            cmd = r["command"]
            res = r["result"]
            err = res.get("error")
            val = (res.get("value") or "").replace("\n", " ").strip()[:60]

            if err:
                console.print(f"  [err]✗[/err] [cmd]{escape(cmd)}[/cmd]")
                console.print(f"      [err]{escape(str(err)[:120])}[/err]")
                if first_err_cmd is None:
                    first_err_cmd = cmd
                    first_err_msg = str(err)
            else:
                suffix = f" [dim]→ {escape(val)}[/dim]" if val else ""
                console.print(f"  [ok]✓[/ok] [cmd]{escape(cmd)}[/cmd]{suffix}")
                # Surface blank-image warnings from the bridge's post-save check
                if res.get("warning"):
                    console.print(f"      [warn]⚠ {escape(str(res['warning']))}[/warn]")

        all_ok = first_err_cmd is None
        if all_ok:
            console.print(f"\n  [dim]Completed {len(results)} command(s).[/dim]")
        return all_ok, first_err_cmd, first_err_msg

    # ── Session state update from executed commands ────────────────────────────

    def _maybe_update_structure_state(self, commands: List[str]) -> None:
        """
        Heuristically sync session state with commands that were just run.
        Note: RCSB metadata fetch in add_structure() is the accurate path for PDB IDs.
        """
        for cmd in commands:
            s = cmd.strip().lower()

            # open <name> [from ...]
            if s.startswith("open ") and "session" not in s:
                parts = cmd.split()
                if len(parts) >= 2:
                    name = parts[1].strip("'\"")
                    mid  = self.session.next_model_id()
                    path = None
                    if name.lower().endswith((".pdb", ".cif", ".mol2")):
                        path = str(Path(self.session.working_dir) / name)
                    # add_structure fetches RCSB metadata if name is a 4-char PDB ID
                    self.session.add_structure(mid, name, path=path)

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

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_exchange(
        self,
        user_input: str,
        commands:   List[str],
        success:    bool,
        error:      Optional[str],
    ) -> None:
        entry = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "user_input": user_input,
            "commands":   commands,
            "success":    success,
            "error":      error,
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ── Special commands ──────────────────────────────────────────────────────

    def _build_prompt_label(self) -> str:
        if not self.session.structures:
            tag = "[dim][no structures][/dim]"
        else:
            parts = [f"{info['name']} #{mid}" for mid, info in self.session.structures.items()]
            joined = ", ".join(parts[:3])
            if len(parts) > 3:
                joined += f" +{len(parts)-3}"
            tag = f"[dim][{joined}][/dim]"
        return f"{tag} [bold cyan]You[/bold cyan]"

    def _cmd_history(self, n: int = 15) -> None:
        hist = self.session.get_recent_history(n)
        if not hist:
            console.print("[dim]No history yet.[/dim]")
            return
        console.print(Rule("[bold]Command History[/bold]"))
        for i, entry in enumerate(hist, 1):
            mark = "[ok]✓[/ok]" if entry.get("success") else "[err]✗[/err]"
            ts   = entry["timestamp"]
            nl   = entry["nl_input"][:70]
            console.print(f"  {mark} [dim]{ts}[/dim]  [bold]{i}.[/bold] {escape(nl)}")
            for cmd in entry.get("commands", []):
                console.print(f"       [cmd]{escape(cmd)}[/cmd]")

    def _cmd_state(self) -> None:
        console.print(Rule("[bold]Session State[/bold]"))
        console.print(self.session.get_context_summary())
        # Show tool availability
        console.print()
        console.print(Rule("[bold]Tool Status[/bold]"))
        available = self.router.available_tools()
        for tool, status in available.items():
            icon  = ToolRouter._TOOL_ICONS.get(tool, "⚙️")
            color = "ok" if status == "active" else "warn" if "stub" in status else "dim"
            console.print(f"  {icon} [bold]{tool:<12}[/bold] [{color}]{escape(status)}[/{color}]")

    def _cmd_undo(self) -> None:
        # Send undo to ChimeraX
        if not self.bridge.is_running():
            console.print("[err]ChimeraX not reachable.[/err]")
            return
        result = self.bridge.run_command("undo")
        if result.get("error"):
            console.print(f"[warn]ChimeraX undo: {escape(result['error'][:80])}[/warn]")
        else:
            console.print("[ok]✓[/ok] Undo sent to ChimeraX.")
        # Also remove from session history
        removed = self.session.undo_last()
        if removed:
            console.print(f"[dim]Removed from history: {removed['nl_input'][:60]}[/dim]")

    def _cmd_clear(self) -> None:
        confirm = Prompt.ask(
            "[warn]Close all models and reset session? (y/n)[/warn]",
            default="n",
        ).strip().lower()
        if confirm not in ("y", "yes"):
            console.print("[dim]Cancelled.[/dim]")
            return
        self.bridge.run_command("close all")
        self.session.clear_all_structures()
        self.session.named_selections.clear()
        self.session.applied_styles.clear()
        console.print("[ok]✓[/ok] All models closed, session reset.")

    def _cmd_reset(self) -> None:
        confirm = Prompt.ask(
            "[warn]Clear conversation context? Loaded structures are kept. (y/n)[/warn]",
            default="n",
        ).strip().lower()
        if confirm in ("y", "yes"):
            self.translator.reset_conversation()
            console.print("[ok]✓[/ok] Conversation context cleared.")

    def _cmd_save_session(self, name: str) -> None:
        name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        cxs_path  = config.SESSION_DIR / f"{name}.cxs"
        json_path = config.SESSION_DIR / f"{name}.json"

        # Save ChimeraX session
        cx_fwd = cxs_path.as_posix()
        result = self.bridge.run_command(f'save "{cx_fwd}"')
        if result.get("error"):
            console.print(f"[err]ChimeraX save failed: {escape(result['error'][:80])}[/err]")
        else:
            console.print(f"[ok]✓[/ok] ChimeraX session → {cxs_path}")

        # Save Python session state
        self.session.save(str(json_path))
        console.print(f"[ok]✓[/ok] Session state   → {json_path}")

    def _cmd_load_session(self, name: str) -> None:
        name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
        cxs_path  = config.SESSION_DIR / f"{name}.cxs"
        json_path = config.SESSION_DIR / f"{name}.json"

        if not cxs_path.is_file():
            console.print(f"[err]Session not found: {cxs_path}[/err]")
            return

        cx_fwd = cxs_path.as_posix()
        result = self.bridge.run_command(f'open "{cx_fwd}"')
        if result.get("error"):
            console.print(f"[err]ChimeraX open failed: {escape(result['error'][:80])}[/err]")
        else:
            console.print(f"[ok]✓[/ok] Loaded ChimeraX session from {cxs_path}")

        if json_path.is_file():
            self.session = SessionState.load(str(json_path))
            console.print(f"[ok]✓[/ok] Restored session state from {json_path}")

    def _cmd_jobs(self) -> None:
        """Show all Robetta / PyRosetta jobs tracked in this session."""
        jobs = self.session.list_rosetta_jobs()
        if not jobs:
            console.print("[dim]No Rosetta jobs in this session.[/dim]")
            return

        console.print(Rule("[bold]⚗️  Rosetta Jobs[/bold]"))
        table = Table(border_style="magenta", show_lines=True)
        table.add_column("Job ID",    style="bold cyan", no_wrap=True)
        table.add_column("Backend",   style="dim",       width=10)
        table.add_column("Status",    width=12)
        table.add_column("Mutations", width=10)
        table.add_column("Submitted", style="dim")

        for jid, job in jobs.items():
            status  = job.get("status", "?")
            backend = job.get("backend", "?")
            nmut    = len(job.get("mutations", []))
            ts      = job.get("submitted_at", "?")

            if status == "completed":
                status_str = f"[ok]{escape(status)}[/ok]"
            elif status in ("failed", "error"):
                status_str = f"[err]{escape(status)}[/err]"
            else:
                status_str = f"[warn]{escape(status)}[/warn]"

            table.add_row(
                escape(jid),
                escape(backend),
                status_str,
                str(nmut),
                escape(ts),
            )

        console.print(table)

        # Show results for completed jobs
        for jid, job in jobs.items():
            if job.get("status") == "completed" and job.get("results"):
                console.print(f"\n  [bold]Results for job #{escape(jid)}:[/bold]")
                scores = job["results"]
                for mut_key, ddg in sorted(scores.items(), key=lambda x: x[1]):
                    colour = "cyan" if ddg < 0 else "yellow" if ddg < 1.0 else "red"
                    console.print(
                        f"    [{colour}]{escape(mut_key)}[/{colour}]  "
                        f"ΔΔG = {ddg:+.3f} kcal/mol"
                    )

    def _cmd_help(self) -> None:
        console.print(HELP_TEXT)

    def _cmd_quit(self) -> None:
        self.session.save(SESSION_FILE)
        console.print(f"[ok]✓[/ok] Session saved to [dim]{SESSION_FILE}[/dim]")
        console.print("[dim]Goodbye.[/dim]")
        sys.exit(0)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="structurebot",
        description="StructureBot — Natural Language Interface for UCSF ChimeraX",
    )
    parser.add_argument("--resume",           action="store_true",
                        help="Restore last session.json")
    parser.add_argument("--chimerax",         metavar="PATH",
                        help="Override ChimeraX.exe path")
    parser.add_argument("--port",             type=int, default=config.REST_PORT,
                        help=f"REST port (default {config.REST_PORT})")
    parser.add_argument("--no-auto-proceed",  dest="auto_proceed", action="store_false",
                        help="Always require explicit confirmation (disable countdown)")
    parser.set_defaults(auto_proceed=True)
    args = parser.parse_args()

    bot = StructureBot(
        chimerax_path     = args.chimerax,
        port              = args.port,
        resume            = args.resume,
        auto_proceed      = args.auto_proceed,
        auto_proceed_delay= config.AUTO_PROCEED_DELAY,
    )
    bot.run()


if __name__ == "__main__":
    main()
