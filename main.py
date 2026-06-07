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
from translator import CommandTranslator, RefusalError, is_usage_cap_error
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


# ── Elapsed-time progress ticker ──────────────────────────────────────────────

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
        """
        Process one user input string (from REPL or script).

        Handles in order:
          1. Semicolon chaining — "cmd1; cmd2" runs both sequentially.
          2. Active-site management commands (set/clear/show).
          3. Sequence display fast-path (show designed sequences, etc.).
          4. Full LLM-backed _handle_request pipeline.
        """
        # ── Semicolon chaining ─────────────────────────────────────────────────
        if ";" in user_input:
            parts = [p.strip() for p in user_input.split(";") if p.strip()]
            for part in parts:
                self._handle_request(part)
            return

        # ── Active-site management ─────────────────────────────────────────────
        msg = self.router.handle_active_site_command(user_input)
        if msg:
            console.print(f"[ok]✓[/ok] {escape(msg)}")
            return

        # ── Sequence display fast-path ─────────────────────────────────────────
        seq_msg = self.router.handle_sequence_display_command(user_input)
        if seq_msg:
            console.print(seq_msg)
            return

        # ── Live-selection fast-path (act on the current ChimeraX selection) ───
        sel_msg = self.router.handle_selection_command(user_input)
        if sel_msg:
            console.print(sel_msg)
            return

        self._handle_request(user_input)

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

    def _report_translation_decline(self, exc: Exception) -> None:
        """
        Transparently report a declined/empty translation. Shows the REAL reason
        (the actual stop_reason carried in *exc*), NOT a generic "safety filter"
        framing — the translator already retried once automatically, and routine
        structural-biology requests are not blocked by StructureBot's own logic.
        """
        console.print(
            f"[warn]⚠ The model returned no usable translation: {escape(str(exc))}[/warn]"
        )
        console.print(
            "[dim]An automatic retry was already attempted. This is usually "
            "transient — try the same request again, or rephrase it slightly. "
            "(The request is fine for a structural-biology tool.)[/dim]"
        )

    def _report_translation_error(self, exc: Exception) -> None:
        """Surface an UNEXPECTED translation failure as a clean one-line message and
        return to the prompt — the REPL must never crash on a backend error. The full
        error goes to stderr for diagnosis (§5 error-first). A Claude usage/spend-cap
        is special-cased with actionable guidance (the cap message carries the reset
        date)."""
        sys.stderr.write(f"[main] translation error: {type(exc).__name__}: {exc}\n")
        if is_usage_cap_error(exc):
            console.print(
                f"[warn]⚠ Claude API usage limit reached: {escape(str(exc))}[/warn]"
            )
            console.print(
                "[dim]Set TRANSLATOR_BACKEND=ollama to use the local model, or wait "
                "for the limit to reset (see the date above).[/dim]"
            )
        else:
            console.print(
                f"[warn]⚠ Couldn't translate that request: {escape(str(exc))}[/warn]"
            )
            console.print(
                "[dim]Returning to the prompt — try again, or rephrase it.[/dim]"
            )

    def _handle_request(self, user_input: str, is_retry: bool = False) -> None:
        # 1. Translate
        try:
            with console.status("[cyan]Translating…[/cyan]"):
                result = self.translator.translate(user_input, self.session)
        except RefusalError as exc:
            self._report_translation_decline(exc)
            return
        except ValueError as exc:
            # Legacy/other path: only treat as a decline if the message clearly
            # indicates an empty/declined translation; otherwise re-raise.
            if any(k in str(exc).lower() for k in ("refusal", "safety", "stop_reason")):
                self._report_translation_decline(exc)
                return
            raise
        except Exception as exc:
            # Backstop for the NON-refusal escape path: any other unexpected
            # translation failure (e.g. a Claude usage-cap BadRequestError that the
            # one-way fallback couldn't reroute when fallback is off) must surface
            # cleanly and return to the prompt — the REPL never crashes on it.
            self._report_translation_error(exc)
            return

        # 2. Route (augment with tool pipeline info; no execution yet)
        result = self.router.route(result, user_input=user_input)

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

            # ── Fast-path: bypass retranslation for known tool intents ─────────
            # If the original user_input already contains glycan (or other
            # recognised) keywords, re-routing through translate() would send a
            # bare short answer ("chain A") to the model with no prior context,
            # causing a stop_reason='refusal' crash.  Detect the intent here and
            # dispatch directly instead.
            if self.router._detect_glycan_intent(user_input):
                result = self.router.route(
                    {
                        "commands":             [],
                        "explanations":         [],
                        "warnings":             [],
                        "clarification_needed": None,
                        "confidence":           "high",
                        "tools_needed":         ["glycan"],
                        "tool_inputs":          {},
                    },
                    user_input=user_input,
                )
                break

            self.translator.add_clarification(answer)
            try:
                with console.status("[cyan]Retranslating…[/cyan]"):
                    result = self.translator.translate(answer, self.session)
            except ValueError as exc:
                err_str = str(exc)
                if "refusal" in err_str.lower() or "stop_reason" in err_str.lower():
                    console.print(
                        "[warn]Sorry, I couldn't process that answer. "
                        "Try rephrasing your original request directly, "
                        "e.g. 'suggest glycosylation sites on chain A'[/warn]"
                    )
                else:
                    console.print(
                        f"[warn]Translation error: {escape(err_str[:120])}[/warn]"
                    )
                return
            except Exception as exc:
                console.print(
                    "[warn]Sorry, I couldn't process that answer. "
                    "Try rephrasing your original request directly.[/warn]"
                )
                return
            result = self.router.route(result, user_input=user_input)

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

        # 6. Take pre-scan snapshot (before confirmation so state is clean)
        _pre_scan_snapshot = self.session.snapshot()

        # 7. Confirm / auto-proceed / edit
        should_execute = self._confirm_execution(confidence)
        if should_execute is None:
            return  # cancelled
        if should_execute == "edit" and commands:
            commands = self._edit_commands(commands)
            if not commands:
                return

        console.print()

        # Initialize execution state (used after try/except block)
        all_commands: list = list(commands)
        success      = True
        failed_cmd:  Optional[str] = None
        error_msg:   Optional[str] = None

        try:
            # 8. Execute initial ChimeraX commands (if any)

            if commands:
                success, failed_cmd, error_msg = self._execute_commands(commands)  # noqa: F841

            # 9. Execute extra tools (CamSol, ESM, etc.) if initial phase succeeded
            if success and has_extra:
                def _status(msg: str) -> None:
                    console.print(f"  [info]{msg}[/info]")

                # For long-running pipelines, show elapsed time every 30s
                _long_tools = {"mutation_scan", "disulfide", "rosetta", "colabfold",
                               "validate_design"}
                _needs_timer = bool(set(tools_needed) & _long_tools)
                _ticker_label = (
                    "Running " + "/".join(
                        t for t in tools_needed if t in _long_tools
                    )
                )

                # ColabFold: surface a rough ETA beside the elapsed counter when
                # the sequence is known up front (approximate; see ColabFoldBridge).
                _eta_s = 0.0
                if "colabfold" in tools_needed:
                    try:
                        _cf_in = (result.get("tool_inputs") or {}).get("colabfold", {})
                        _seq   = _cf_in.get("sequence") or ""
                        _cop   = int(_cf_in.get("copies", 1) or 1)
                        if _seq:
                            from colabfold_bridge import ColabFoldBridge
                            _eta_s = ColabFoldBridge().estimate_runtime_s(
                                len(_seq) * _cop, 5, 3
                            )
                    except Exception:
                        _eta_s = 0.0

                if _needs_timer:
                    _ticker = _ElapsedTicker(_ticker_label, interval=30, eta_s=_eta_s)
                    _ticker.start()
                else:
                    _ticker = None

                try:
                    with console.status("[cyan]Running computational tools…[/cyan]"):
                        result = self.router.execute(result, status_callback=_status)
                finally:
                    if _ticker is not None:
                        _ticker.stop()

                # Show assembly interface summary (before other summaries)
                self._show_interface_summary(result)

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

                # 10. Execute visualization commands generated by the tools
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

                # 11. Show actionable summary panel for tools that produced one
                for step in result.get("tool_step_results", []):
                    if (step.get("success") and step.get("summary")
                            and "\n" in step.get("summary", "")):
                        console.print()
                        console.print(Panel(
                            step["summary"],
                            title="[bold green]Analysis Summary[/bold green]",
                            border_style="green",
                            padding=(1, 2),
                        ))

        except KeyboardInterrupt:
            console.print("\n[warn]Warning: Scan cancelled by user.[/warn]")
            console.print("[dim]Restoring session state to pre-scan snapshot...[/dim]")
            self.session.restore(_pre_scan_snapshot)
            console.print("[dim]Session state restored.[/dim]")
            return

        # 10. Auto-fix on first failure (once only, ChimeraX commands only)
        if not success and not is_retry and failed_cmd and error_msg:
            console.print("\n[warn]Asking for a corrected command…[/warn]")
            # Bug 6a: the actual error text is fed into translate_error_fix so the
            # model cannot re-propose the same command blind (already handled by
            # translate_error_fix, which builds the prompt from failed_command +
            # error_message verbatim — no silent re-prompt).
            fix = self.translator.translate_error_fix(failed_cmd, error_msg, self.session)
            fix_cmds = fix.get("commands", [])
            fix_exps = fix.get("explanations", [])

            # Bug 6b: no-progress detection — halt cleanly instead of looping.
            # No progress = the correction is empty (guards blocked it or model
            # refused) OR the model re-proposed the identical failing command.
            _same_cmd = bool(
                fix_cmds
                and fix_cmds[0].strip().lower() == failed_cmd.strip().lower()
            )
            if not fix_cmds or _same_cmd:
                console.print(
                    f"[warn]Couldn't auto-correct — "
                    f"error: {escape(error_msg[:200])}[/warn]"
                )
                console.print(
                    "[dim]Correction re-proposed the same command or was blocked "
                    "by a validation guard.  Try rephrasing your request.[/dim]"
                )
            else:
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

        # Build enhanced tool-step log entries from tool pipeline results
        _tool_steps: List[dict] = []
        for step in result.get("tool_step_results", []):
            if step.get("skipped"):
                continue
            tool  = step.get("tool", "")
            data  = step.get("data", {})
            entry: dict = {
                "tool":       tool,
                "elapsed_ms": step.get("elapsed_ms", 0),
                "success":    step.get("success", False),
            }
            # Tool-specific enrichment
            if tool == "mutation_scan":
                cands = data.get("candidates", [])
                entry["n_candidates"]  = len(cands)
                entry["top_candidate"] = cands[0].get("mutation_key", "") if cands else ""
                entry["top_ddg"]       = cands[0].get("ddg", None)        if cands else None
                entry["backend"]       = cands[0].get("backend", "")      if cands else ""
            elif tool == "disulfide":
                entry["n_candidates"] = data.get("count", 0)
            elif tool in ("camsol", "esm", "proteinmpnn", "rfdiffusion"):
                pass  # no extra enrichment needed
            _tool_steps.append(entry)

        self._log_exchange(user_input, all_commands, success, error_msg,
                           tool_steps=_tool_steps if _tool_steps else None)

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
        ESC during countdown → cancel immediately (return None).
        """
        if self.auto_proceed and confidence in ("high", "medium"):
            result = self._countdown(self.auto_proceed_delay)
            if result == "escaped":
                console.print("[dim]Cancelled.[/dim]")
                return None
            return "proceed" if result else self._manual_confirm()
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

    def _countdown(self, seconds: int):
        """
        Countdown before auto-executing.
        Returns True to proceed, False if the user pressed any other key,
        or "escaped" if the user pressed ESC.

        ESC ('\x1b') cancels immediately without showing a y/n prompt.
        Any other key pauses the countdown and returns False so _manual_confirm
        is shown.
        """
        for remaining in range(seconds, 0, -1):
            console.print(
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
                    console.print()
                    if ch == "\x1b":
                        return "escaped"
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
        After a structure is opened, query RCSB for assembly type and display it.
        Only runs for 4-letter PDB IDs; silently skips on any error.
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
        except Exception:
            pass  # assembly info is non-critical; never interrupt the flow

    def _show_interface_summary(self, result: dict) -> None:
        """Show interface summary from assembly_analyser step results."""
        step_results = result.get("tool_step_results", [])
        for step in step_results:
            if step.get("tool") == "assembly_analyser" and step.get("success"):
                data = step.get("data", {})
                summary = data.get("interface_summary", "")
                header  = data.get("header", "")
                warnings = data.get("warnings", [])

                if header:
                    console.print(f"\n  [bold]🔗 {escape(header)}[/bold]")
                if summary:
                    console.print(f"  [info]{escape(summary)}[/info]")
                for w in warnings:
                    console.print(f"  [warn]⚠ {escape(w)}[/warn]")
                excluded = data.get("excluded_count", 0)
                if excluded:
                    console.print(
                        f"  [dim]  → {excluded} residue(s) will be excluded from mutation scan[/dim]"
                    )

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
    parser.set_defaults(auto_proceed=True)
    args = parser.parse_args()

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
