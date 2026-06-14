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
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ── UTF-8 stdout on Windows (default is cp1252 in Python 3.x) ────────────────
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

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
from presenter import ConsolePresenter
from request_engine import RequestEngine

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
    [cyan]Suggest mutations avoiding chain interfaces[/cyan]
    [cyan]Show me the interface between chains A and B[/cyan]
    [cyan]Suggest disulfide bonds to stabilise the dimer[/cyan]

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
  analyse solubility of chain A as a monomer
  suggest mutations to improve solubility avoiding chain interfaces
  what mutations would reduce aggregation?
  run the full engineering pipeline on the loaded structure

[bold]Assembly analysis[/bold]
  show me the interface between chains A and B
  analyse as multimer — avoid interface residues
  what is the biological assembly of this structure?
  find all inter-chain contacts within 5 angstroms

[bold]Disulfide engineering[/bold]
  suggest disulfide bonds to stabilise the dimer
  find disulfide candidates between chains A and B
  engineer a disulfide to cross-link the interface
  improve dimer stability with disulfide bonds

[bold]Special commands[/bold]
  [cmd]history[/cmd]           show last 15 commands
  [cmd]state[/cmd]             dump current session state
  [cmd]jobs[/cmd]              show status of pending Robetta ddG jobs
  [cmd]stats[/cmd]             show usage statistics across all sessions
  [cmd]undo[/cmd]              undo the last ChimeraX action
  [cmd]clear[/cmd]             close all models, reset session
  [cmd]clear session[/cmd]     delete session.json and start fresh (keeps ChimeraX models)
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

        # Restore-on-startup bookkeeping
        self._resume_flag = resume   # --resume already loaded session.json in load()
        self._interactive = True     # set False by run_script() to skip prompts

        # Log file path
        log_name = datetime.now().strftime("session_%Y%m%d_%H%M%S.jsonl")
        self.log_file = config.LOG_DIR / log_name

        # Request orchestration: the UI-agnostic engine + the console presenter.
        # The console REPL drives RequestEngine through ConsolePresenter (the parity
        # oracle); a future GUI hosts the same engine through a Qt presenter.
        self.presenter = None
        self.engine = None
        self._ensure_engine()

    # ── Startup ───────────────────────────────────────────────────────────────

    def startup(self) -> None:
        console.print(Panel(BANNER, border_style="cyan", padding=(0, 2)))

        # Translation is local-only (Ollama) — no API key to check.

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

        # 3. Start or connect. ensure_visible_gui() rejects a leftover *windowless*
        # ChimeraX (REST-reachable but with no GUI window — a zombie from a prior
        # session): models would open into an invisible viewer. It relaunches a
        # fresh visible window in that case.
        with console.status("[cyan]Connecting to ChimeraX REST server…[/cyan]"):
            already = self.bridge.is_running()
        if not already:
            console.print("[warn]ChimeraX REST server not found — launching ChimeraX…[/warn]")
            console.print("[dim]  (may take 20–40 s for ChimeraX to initialise)[/dim]")
        try:
            with console.status("[cyan]Starting ChimeraX…[/cyan]"):
                outcome = self.bridge.ensure_visible_gui(timeout=60)
            if outcome == "connected":
                console.print(
                    f"[ok]✓[/ok] Connected to ChimeraX at "
                    f"http://{config.REST_HOST}:{self.bridge.port}/"
                )
            elif outcome == "relaunched":
                console.print(
                    "[warn]⚠ A windowless ChimeraX was holding the port — relaunched a "
                    f"fresh visible window (REST on port {self.bridge.port}).[/warn]"
                )
            else:  # "started"
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

        # 5. WSL2 status
        try:
            from wsl_bridge import WSLBridge
            wsl = WSLBridge()
            if wsl.is_available():
                has_py = wsl.check_pyrosetta()
                if has_py:
                    console.print(
                        "[ok]✓[/ok] Rosetta: local (PyRosetta via WSL2) — publication quality"
                    )
                else:
                    console.print(
                        "[warn]⚠[/warn] WSL2 available — PyRosetta not installed "
                        "(run pyrosetta_installer in WSL2 to enable local Rosetta)"
                    )
                self.session.wsl_available = True
            else:
                console.print(
                    "[dim]✓ Rosetta: DynaMut2 (screening quality) — "
                    "WSL2 not installed (run `wsl --install -d Ubuntu-22.04` "
                    "as Administrator to enable local Rosetta)[/dim]"
                )
                self.session.wsl_available = False
        except Exception:
            pass  # WSL2 status is informational only

        # 6. Session auto-restore (mutation scans etc. survive restarts)
        self._maybe_restore_session()
        console.print()

    # ── Session restore ─────────────────────────────────────────────────────────

    def _maybe_restore_session(self) -> None:
        """
        On startup, offer to restore a previous session.json so expensive
        computed state (mutation scans, double-mutant results, interfaces,
        active-site residues, …) survives restarts.

        --resume auto-accepts (the file was already loaded in __init__).
        Otherwise the user is prompted: restore / start fresh / delete.
        Corrupt or incompatible files are reported and skipped (start fresh).
        """
        # --resume already loaded the file in __init__ → treat as accepted.
        if self._resume_flag:
            if self.session.structures or self.session.scan_results:
                console.print(
                    f"[dim]Resumed session: {len(self.session.structures)} structure(s), "
                    f"{len(self.session.scan_results)} scan result(s), "
                    f"{len(self.session.command_history)} prior commands.[/dim]"
                )
                self._reconnect_or_offer_reopen()
            return

        state, err = SessionState.try_load(SESSION_FILE)
        if err:
            console.print(
                f"[warn]⚠ Previous session ({SESSION_FILE}) could not be read "
                f"({escape(err)}); starting fresh.[/warn]"
            )
            return
        if state is None:
            return  # no session file at all

        # Anything worth restoring?
        if not (state.structures or state.scan_results
                or state.double_mutant_results or state.command_history):
            return

        # Non-interactive (script) runs never block on a prompt.
        if not self._interactive:
            return

        console.print(Panel(
            state.restore_summary(),
            title="[bold cyan]Previous session found[/bold cyan]",
            border_style="cyan",
            padding=(0, 2),
        ))
        choice = Prompt.ask(
            "[hi]Restore it?[/hi] "
            "[dim][[y]es / [n]o (keep file) / [c]lear (delete it)][/dim]",
            default="y",
        ).strip().lower()

        if choice in ("c", "clear"):
            self._wipe_session_file()
            console.print("[ok]✓[/ok] Previous session deleted — starting fresh.")
            return
        if choice in ("n", "no"):
            console.print(
                "[dim]Starting fresh. The previous session.json is kept and will "
                "be overwritten when you quit.[/dim]"
            )
            return

        # Accept → swap in the restored session and rebind the router to it.
        self.session = state
        self.router  = ToolRouter(self.bridge, self.session)
        console.print(
            f"[ok]✓[/ok] Session restored: {len(self.session.structures)} structure(s), "
            f"{len(self.session.scan_results)} scan result(s)."
        )
        self._reconnect_or_offer_reopen()

    def _chimerax_model_ids(self) -> set:
        """
        Return the set of model-id strings currently open in ChimeraX
        (e.g. {"1", "2"}).  Empty set if ChimeraX is unreachable.
        """
        try:
            if not self.bridge.is_running():
                return set()
            res = self.bridge.run_command("info models")
            val = (res.get("value") or "") if isinstance(res, dict) else ""
            return set(re.findall(r"#(\d+)", val))
        except Exception:
            return set()

    def _reconnect_or_offer_reopen(self) -> None:
        """
        For each restored structure, check whether it is still loaded in
        ChimeraX. Present ones are reused as-is; missing ones can be re-opened
        (a fast fetch for PDB IDs / local files, not a re-computation).
        """
        if not self.session.structures:
            return
        open_ids = self._chimerax_model_ids()
        for mid, info in list(self.session.structures.items()):
            name = str(info.get("name", "?"))
            if mid in open_ids:
                console.print(
                    f"  [dim]✓ #{mid} {escape(name)} still loaded in ChimeraX.[/dim]"
                )
                continue

            console.print(
                f"  [warn]⚠ #{mid} {escape(name)} is not loaded in ChimeraX "
                "(session state kept).[/warn]"
            )
            # Re-open only makes sense for a 4-char PDB ID or a known file path.
            reopenable = (
                bool(re.match(r"^[A-Za-z0-9]{4}$", name.strip()))
                or bool(info.get("path"))
            )
            if not reopenable or not self._interactive:
                continue

            choice = Prompt.ask(
                f"[hi]Re-open {escape(name)} in ChimeraX?[/hi] [dim][[y]es / [n]o][/dim]",
                default="y",
            ).strip().lower()
            if choice in ("n", "no"):
                continue

            target = info.get("path") or name
            res = self.bridge.run_command(f"open {target}")
            if isinstance(res, dict) and res.get("error"):
                console.print(
                    f"  [err]✗ open failed: {escape(str(res['error'])[:80])}[/err]"
                )
            else:
                console.print(f"  [ok]✓[/ok] Re-opened {escape(name)}.")

    def _wipe_session_file(self) -> None:
        """Delete the persisted session.json (silent if absent)."""
        try:
            p = Path(SESSION_FILE)
            if p.is_file():
                p.unlink()
        except OSError:
            pass

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
            elif lower in ("clear session", "new session", "clearsession", "newsession"):
                self._cmd_clear_session()
            elif lower == "clear":
                self._cmd_clear()
            elif lower == "reset":
                self._cmd_reset()
            elif lower == "help":
                self._cmd_help()
            elif lower == "jobs":
                self._cmd_jobs()
            elif lower == "stats":
                self._cmd_stats()
            elif re.match(r"^save session\b", lower):
                name = user_input.split(maxsplit=2)[2] if len(user_input.split()) > 2 else "default"
                self._cmd_save_session(name)
            elif re.match(r"^load session\b", lower):
                name = user_input.split(maxsplit=2)[2] if len(user_input.split()) > 2 else "default"
                self._cmd_load_session(name)
            else:
                self._dispatch_input(user_input)

    def _dispatch_input(self, user_input: str) -> None:
        """Process one user input string (REPL or script): semicolon chaining + the
        bypass-LLM fast-paths (active-site / sequence-display / live-selection) + the full
        pipeline — all via the shared engine.dispatch, the SINGLE path both front-ends use
        (the GUI worker calls the same method). Rendering goes through the presenter, so
        the console output is unchanged (ConsolePresenter reproduces it verbatim)."""
        self._ensure_engine()
        self.engine.dispatch(user_input, self.presenter)

    # ── Script runner ─────────────────────────────────────────────────────────

    def run_script(self, script_path: str) -> None:
        """
        Execute commands from a plain-text script file sequentially.

        File format:
          - One command per line
          - Lines starting with ``#`` are comments (skipped)
          - Blank lines are skipped

        Connects to ChimeraX via ``startup()`` before running, saves the session
        when done, and returns (does NOT call sys.exit so tests can use it cleanly).
        """
        path = Path(script_path)
        if not path.is_file():
            console.print(f"[err]Error: script file not found: {path}[/err]")
            import sys as _sys
            _sys.exit(1)

        # Batch mode: never block on an interactive restore/reopen prompt.
        self._interactive = False
        self.startup()

        lines = path.read_text(encoding="utf-8").splitlines()
        cmds  = [ln.strip() for ln in lines
                 if ln.strip() and not ln.strip().startswith("#")]

        console.print(
            f"[info]Running script: {path.name} "
            f"({len(cmds)} command(s))[/info]"
        )
        for cmd in cmds:
            console.print(f"\n[dim]── {escape(cmd)} ──[/dim]")
            self._handle_request(cmd)

        self.session.save(SESSION_FILE)
        console.print(f"\n[ok]✓[/ok] Script complete — session saved.")

    # ── Natural language pipeline ─────────────────────────────────────────────

    def _ensure_engine(self) -> None:
        """Build the ConsolePresenter + RequestEngine on first use. Idempotent so a
        StructureBot built via object.__new__ (tests) still gets a working engine on
        the first request."""
        if getattr(self, "engine", None) is None:
            self.presenter = ConsolePresenter(
                console,
                auto_proceed=getattr(self, "auto_proceed", True),
                auto_proceed_delay=getattr(self, "auto_proceed_delay", 2),
            )
            self.engine = RequestEngine(self)

    def _handle_request(self, user_input: str, is_retry: bool = False) -> None:
        # Thin wrapper → the shared engine (the verb-guard probe now runs inside
        # engine.handle_request, the single translate entry).
        self._ensure_engine()
        self.engine.handle_request(user_input, self.presenter, is_retry=is_retry)

    def _execute_commands(
        self,
        commands: List[str],
        origin: str = "translation",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute via the request engine (presenter-rendered). Returns
        (all_ok, failed_cmd, error_msg). Thin wrapper so existing callers and tests
        (bot._execute_commands(...)) are unchanged."""
        self._ensure_engine()
        return self.engine.execute_commands(commands, self.presenter, origin=origin)

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
                    # Display assembly type for PDB IDs
                    self._display_assembly_type_on_open(name, mid)

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
        """
        After a structure is opened, query RCSB for assembly type, display it,
        and surface an AU-vs-biological mismatch note when the file contains only
        the asymmetric unit (e.g. 2 chains) but the biological assembly is larger
        (e.g. homotetramer A4 = 4 chains).
        """
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
                console.print(f"  [dim]✓ {name.upper()} → {escape(display)}[/dim]")

                # AU-vs-biological mismatch detector (Component 2)
                n_bio = asm_info.get("n_subunits")
                if n_bio and int(n_bio) > 1:
                    try:
                        loaded_chains = self.bridge._model_chains(model_id)
                        n_loaded = len(loaded_chains)
                        if n_loaded > 0 and int(n_bio) > n_loaded:
                            asm_type = asm_info.get("assembly_type") or "biological assembly"
                            stoich   = asm_info.get("stoichiometry") or ""
                            stoich_s = f" ({stoich})" if stoich else ""
                            console.print(
                                f"  [dim yellow]⚠ Loaded the asymmetric unit "
                                f"({n_loaded} chain{'s' if n_loaded != 1 else ''}); "
                                f"biological assembly is a {asm_type}{stoich_s} — "
                                f"generate it with "
                                f"\"work as {asm_type.split()[-1]}\" / "
                                f"\"generate biological assembly\"?[/dim yellow]"
                            )
                    except Exception:
                        pass  # mismatch note is non-critical
        except Exception:
            pass  # assembly info is non-critical; never interrupt the flow

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log_exchange(
        self,
        user_input: str,
        commands:   List[str],
        success:    bool,
        error:      Optional[str],
        tool_steps: Optional[List[dict]] = None,
    ) -> None:
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

    def _cmd_clear_session(self) -> None:
        """
        Delete the persisted session.json and reset in-memory state to a fresh
        session, so the user is not stuck with stale restored state. Loaded
        ChimeraX models are left untouched (use 'clear' to close those).
        """
        self._wipe_session_file()
        self.session = SessionState()
        self.router  = ToolRouter(self.bridge, self.session)
        console.print(
            f"[ok]✓[/ok] Session cleared — fresh start "
            f"([dim]{SESSION_FILE}[/dim] removed). "
            "Loaded ChimeraX models are untouched."
        )

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

    def _cmd_stats(self) -> None:
        """Show usage statistics from session log files."""
        try:
            from log_analyser import display_stats
            display_stats(console)
        except Exception as exc:
            console.print(f"[err]Stats unavailable: {escape(str(exc))}[/err]")

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
    parser.add_argument("--script",           type=str, default=None,
                        metavar="FILE",
                        help="Path to a text file of commands to run sequentially "
                             "(one per line; blank lines and # comments skipped)")
    parser.add_argument("--console",          action="store_true",
                        help="Run the console REPL (the parity oracle / debug fallback) "
                             "instead of the default GUI")
    parser.set_defaults(auto_proceed=True)
    args = parser.parse_args()

    # The GUI is the default front-end; --console (and --script) use the console REPL.
    # Both drive the SAME RequestEngine; the console stays the parity oracle + fallback.
    if not (args.console or args.script):
        from PySide6 import QtWidgets
        from gui_app import StructureBotWindow
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        win = StructureBotWindow(
            port               = args.port,
            auto_proceed       = args.auto_proceed,
            auto_proceed_delay = config.AUTO_PROCEED_DELAY,
        )
        win.show()
        sys.exit(app.exec())

    bot = StructureBot(
        chimerax_path     = args.chimerax,
        port              = args.port,
        resume            = args.resume,
        auto_proceed      = args.auto_proceed,
        auto_proceed_delay= config.AUTO_PROCEED_DELAY,
    )

    if args.script:
        bot.run_script(args.script)
    else:
        bot.run()


if __name__ == "__main__":
    main()
