"""
tests/test_integration.py
--------------------------
End-to-end integration test: full stack against 1HSG (HIV protease + MK1).

What it tests
-------------
Phase A — Translation only (no ChimeraX needed):
  Verify that each of the 5 prompts produces reasonable ChimeraX commands
  without the model hallucinating incorrect syntax.

Phase B — Execution (ChimeraX must be running with REST server):
  Execute each set of commands and verify no errors are returned.

Prerequisites
-------------
  1. venv activated:  .\\venv\\Scripts\\Activate.ps1
  2. Internet access (to fetch 1HSG from RCSB and RCSB metadata)
  3. For Phase B: ChimeraX running with:
       remotecontrol rest start port 60001

Run:
    python tests/test_integration.py               # translation only
    python tests/test_integration.py --execute     # translation + execution
    python tests/test_integration.py --start       # start ChimeraX then execute
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from the project root even when run from tests/
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)  # ensure relative paths (chimerax_commands.md etc.) resolve

# ── Load env ──────────────────────────────────────────────────────────────────
import config
config.load_env_file()

# ── UTF-8 on Windows ──────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from chimerax_bridge import ChimeraXBridge
from translator import CommandTranslator
from session_state import SessionState

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg: str)   -> None: print(f"  {GREEN}✓{RESET}  {msg}")
def warn(msg: str) -> None: print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg: str) -> None: print(f"  {RED}✗{RESET}  {msg}")
def info(msg: str) -> None: print(f"  {CYAN}·{RESET}  {msg}")
def section(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * 60)


# ════════════════════════════════════════════════════════════════════════════════
# Test definitions
# Each entry: (id, nl_prompt, assertion_fn, description)
# assertion_fn(commands) → (passed: bool, detail: str)
# ════════════════════════════════════════════════════════════════════════════════

def _any_contains(commands: List[str], *substrings: str) -> bool:
    return any(all(s.lower() in c.lower() for s in substrings) for c in commands)

def assert_open_ribbon(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_open = _any_contains(commands, "open", "1HSG") or _any_contains(commands, "open", "1hsg")
    has_cartoon = _any_contains(commands, "cartoon")
    if not has_open:
        return False, f"Expected 'open 1HSG' — got: {commands}"
    if not has_cartoon:
        return False, f"Expected 'cartoon' — got: {commands}"
    return True, f"open + cartoon ✓  ({len(commands)} command(s))"

def assert_color_bychain(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    if _any_contains(commands, "bychain"):
        return True, "'color bychain' found ✓"
    return False, f"Expected 'bychain' in commands — got: {commands}"

def assert_ligand_sphere(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_sphere = _any_contains(commands, "sphere")
    has_mk1    = _any_contains(commands, "MK1") or _any_contains(commands, "mk1")
    has_byelement = _any_contains(commands, "byelement")
    issues = []
    if not has_sphere:
        issues.append("'sphere' style not found")
    if not has_mk1:
        issues.append("'MK1' residue name not found (ligand not resolved from session state)")
    if not has_byelement:
        issues.append("'byelement' coloring not found")
    if issues:
        return False, "; ".join(issues) + f" — commands: {commands}"
    return True, "sphere + MK1 + byelement ✓"

def assert_zone_selection(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_zone = _any_contains(commands, "zone")
    has_show = _any_contains(commands, "show") or _any_contains(commands, "atoms")
    if not has_zone:
        return False, f"Expected 'zone' selection — got: {commands}"
    if not has_show:
        return False, f"Expected 'show' after zone selection — got: {commands}"
    return True, "zone selection + show ✓"

def assert_save_image(commands: List[str]) -> Tuple[bool, str]:
    if not commands:
        return False, "No commands generated"
    has_save = _any_contains(commands, "save")
    has_png  = _any_contains(commands, ".png")
    has_path = any("desktop" in c.lower() or "Users" in c for c in commands)
    if not has_save:
        return False, f"Expected 'save' command — got: {commands}"
    if not has_png:
        return False, f"Expected '.png' in save command — got: {commands}"
    if not has_path:
        warn(f"Desktop path not detected in save command (may still be correct): {commands}")
    return True, "save + .png ✓"


TEST_SUITE = [
    (
        1,
        "open 1HSG and show it as a ribbon diagram",
        assert_open_ribbon,
        "Open 1HSG from RCSB and display as cartoon",
    ),
    (
        2,
        "color each chain a different color",
        assert_color_bychain,
        "Per-chain coloring → must use 'color bychain'",
    ),
    (
        3,
        "show the ligand as spheres and color it by element",
        assert_ligand_sphere,
        "Ligand sphere + byelement → must resolve 'MK1' from session state",
    ),
    (
        4,
        "find all residues within 4 angstroms of the ligand",
        assert_zone_selection,
        "Zone selection within 4 Å of ligand",
    ),
    (
        5,
        "save an image called test_output.png to my desktop",
        assert_save_image,
        "Save image to Desktop with Windows path",
    ),
]


# ════════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════════

class IntegrationRunner:
    def __init__(self, execute: bool = False, start_cx: bool = False):
        self.execute   = execute
        self.start_cx  = start_cx
        self.translator = CommandTranslator()
        self.session    = SessionState()
        self.bridge: Optional[ChimeraXBridge] = None

        self.passed  = 0
        self.failed  = 0
        self.skipped = 0
        self.results: List[Dict[str, Any]] = []

    def setup_bridge(self) -> bool:
        """Connect to / start ChimeraX.  Returns True if ready."""
        self.bridge = ChimeraXBridge(port=config.REST_PORT)
        self.bridge.base_url = f"http://{config.REST_HOST}:{config.REST_PORT}"
        self.bridge.run_url  = f"{self.bridge.base_url}/run"

        if self.bridge.is_running():
            info(f"ChimeraX REST server reachable at {self.bridge.run_url}")
            return True

        if self.start_cx:
            info("Launching ChimeraX (this may take 30 s)…")
            try:
                self.bridge.start(timeout=60)
                ok("ChimeraX started.")
                return True
            except Exception as exc:
                fail(f"Could not start ChimeraX: {exc}")
                return False

        warn("ChimeraX not running — execution tests will be skipped.")
        warn("Run with --start to launch ChimeraX, or enable REST manually:")
        warn("  remotecontrol rest start port 60001")
        return False

    def run(self) -> None:
        print(f"\n{BOLD}{CYAN}StructureBot — Integration Test Suite (1HSG){RESET}")
        print("=" * 60)

        if not os.environ.get("ANTHROPIC_API_KEY"):
            fail("ANTHROPIC_API_KEY is not set — cannot run translation tests.")
            sys.exit(1)

        cx_ready = False
        if self.execute or self.start_cx:
            section("Setup — ChimeraX connection")
            cx_ready = self.setup_bridge()

        # Run each test
        for test_id, prompt, assert_fn, description in TEST_SUITE:
            self._run_one(test_id, prompt, assert_fn, description, cx_ready)

        # Cleanup
        if cx_ready and self.bridge:
            section("Cleanup")
            res = self.bridge.run_command("close all")
            if not res.get("error"):
                ok("Closed all models — session is clean.")

        self._print_summary()

    def _run_one(
        self,
        test_id:     int,
        prompt:      str,
        assert_fn,
        description: str,
        cx_ready:    bool,
    ) -> None:
        section(f"Test {test_id} — {description}")
        info(f"Prompt: {prompt!r}")

        # ── Phase A: Translation ───────────────────────────────────────────────
        try:
            result = self.translator.translate(prompt, self.session)
        except Exception as exc:
            fail(f"Translation error: {exc}")
            self.failed += 1
            return

        commands     = result.get("commands", [])
        explanations = result.get("explanations", [])
        confidence   = result.get("confidence", "?")
        clarify      = result.get("clarification_needed")
        warnings_    = result.get("warnings", [])

        info(f"Confidence: {confidence}")

        if clarify:
            warn(f"Clarification requested: {clarify}")
            warn("This means the LLM did not have enough context to translate.")
            # For tests 3-5, a clarification about ligand name is a soft failure
            if test_id in (3, 4):
                warn("Hint: RCSB metadata fetch may have failed — check internet access.")
            fail("FAIL (translation): clarification_needed is set")
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "reason": f"clarification: {clarify}"})
            return

        if not commands:
            fail("FAIL (translation): no commands generated")
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "reason": "no commands"})
            return

        for w in warnings_:
            warn(f"Model warning: {w}")

        # Print commands
        for i, (cmd, exp) in enumerate(zip(commands, explanations), 1):
            info(f"  {i}. {cmd!r}  →  {exp}")

        # Assert
        passed, detail = assert_fn(commands)
        if passed:
            ok(f"PASS (translation): {detail}")
        else:
            fail(f"FAIL (translation): {detail}")
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "reason": detail})
            return

        # Update session state (simulates what main.py does after execution)
        self._simulate_state_update(test_id, commands, prompt)

        # ── Phase B: Execution ─────────────────────────────────────────────────
        if not cx_ready:
            warn("Skipping execution (ChimeraX not running).")
            self.passed  += 1
            self.skipped += 1
            self.results.append({"id": test_id, "passed": True, "executed": False})
            return

        # Wait briefly between tests so ChimeraX can finish rendering
        if test_id > 1:
            time.sleep(1.5)

        exec_results = self.bridge.run_commands(commands)
        exec_ok = True
        for r in exec_results:
            cmd = r["command"]
            err = r["result"].get("error")
            val = (r["result"].get("value") or "").strip()[:50]
            if err:
                fail(f"FAIL (execution): {cmd!r} → {err}")
                exec_ok = False
            else:
                suf = f" → {val!r}" if val else ""
                ok(f"Executed: {cmd!r}{suf}")

        if exec_ok:
            ok(f"PASS (translation + execution): all {len(commands)} command(s) succeeded")
            self.passed += 1
            self.results.append({"id": test_id, "passed": True, "executed": True})
        else:
            self.failed += 1
            self.results.append({"id": test_id, "passed": False, "executed": True,
                                  "reason": "execution error"})

    def _simulate_state_update(self, test_id: int, commands: List[str], prompt: str) -> None:
        """Mimic what main.py's _maybe_update_structure_state does."""
        import re
        for cmd in commands:
            s = cmd.strip().lower()
            if s.startswith("open ") and "session" not in s:
                parts = cmd.split()
                if len(parts) >= 2:
                    name = parts[1].strip("'\"")
                    mid  = self.session.next_model_id()
                    # This triggers RCSB metadata fetch for 4-char PDB IDs
                    self.session.add_structure(mid, name)
            elif s.startswith("close"):
                if "all" in s:
                    self.session.clear_all_structures()
                else:
                    m = re.search(r"#(\d+)", cmd)
                    if m:
                        self.session.remove_structure(m.group(1))

        self.session.add_to_history(prompt, commands, success=True)

    def _print_summary(self) -> None:
        section("Summary")
        total = len(TEST_SUITE)
        exec_count = sum(1 for r in self.results if r.get("executed"))
        print(
            f"  {GREEN}{self.passed} passed{RESET}  "
            f"{RED}{self.failed} failed{RESET}  "
            f"({self.skipped} skipped execution)  "
            f"of {total} tests"
        )
        if exec_count:
            print(f"  {exec_count} test(s) were fully executed against ChimeraX.")
        print()

        if self.failed == 0:
            print(f"{GREEN}All tests passed!{RESET}")
            if self.skipped:
                print(f"  Re-run with --execute to also test ChimeraX execution.")
            sys.exit(0)
        else:
            print(f"{RED}Some tests failed. See details above.{RESET}")
            for r in self.results:
                if not r["passed"]:
                    print(f"  Test {r['id']}: {r.get('reason', 'unknown')}")
            sys.exit(1)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="StructureBot integration test — 1HSG full workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Execute commands against ChimeraX (must be running)",
    )
    parser.add_argument(
        "--start", action="store_true",
        help="Launch ChimeraX automatically before executing",
    )
    args = parser.parse_args()

    runner = IntegrationRunner(
        execute  = args.execute or args.start,
        start_cx = args.start,
    )
    runner.run()


if __name__ == "__main__":
    main()
