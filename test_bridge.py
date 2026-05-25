"""
test_bridge.py
--------------
Verifies that StructureBot can find, start, and communicate with the
UCSF ChimeraX REST server — run this BEFORE launching main.py to confirm
your environment is set up correctly.

Usage:
    python test_bridge.py            # non-destructive checks only
    python test_bridge.py --start    # also attempt to launch ChimeraX
    python test_bridge.py --full     # run all tests including a real open/close cycle
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows so Unicode check-marks and box chars render
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Setup: load .env.local if present ────────────────────────────────────────
_env_file = Path(__file__).parent / ".env.local"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

from chimerax_bridge import ChimeraXBridge, find_chimerax

# ── Colours (no dependency on rich) ──────────────────────────────────────────
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
    print("─" * 50)


# ══════════════════════════════════════════════════════════════════════════════
# Individual test functions
# ══════════════════════════════════════════════════════════════════════════════

def test_find_chimerax() -> bool:
    """Check that the ChimeraX executable can be located."""
    section("1 · Finding ChimeraX")

    env_path = os.environ.get("CHIMERAX_PATH")
    if env_path:
        info(f"CHIMERAX_PATH env override: {env_path}")

    path = find_chimerax()
    if path is None:
        fail("ChimeraX executable not found in any standard location.")
        print()
        print("  Fix options:")
        print("    a) Set CHIMERAX_PATH=C:\\full\\path\\to\\ChimeraX.exe")
        print("    b) Add the install location to _WINDOWS_SEARCH_PATTERNS in chimerax_bridge.py")
        return False

    if not Path(path).is_file():
        fail(f"Path resolved but file does not exist: {path}")
        return False

    ok(f"Found: {path}")
    version_dir = Path(path).parent.parent.name
    info(f"Version directory: {version_dir}")
    return True


def test_connection_check(bridge: ChimeraXBridge) -> bool:
    """Non-destructively probe whether ChimeraX is already running."""
    section("2 · Checking for existing ChimeraX REST server")
    running = bridge.is_running()
    if running:
        ok(f"REST server already reachable at http://localhost:{bridge.port}/")
    else:
        warn(f"No REST server found at http://localhost:{bridge.port}/")
        info("(This is normal if ChimeraX is not open yet — use --start to launch it.)")
    return running


def test_start_chimerax(bridge: ChimeraXBridge) -> bool:
    """Attempt to start ChimeraX with the REST server."""
    section("3 · Starting ChimeraX")
    if bridge.is_running():
        ok("Already running — skipping start.")
        return True

    if bridge.chimerax_path is None:
        fail("Cannot start: ChimeraX path unknown. Fix test 1 first.")
        return False

    info(f"Launching: {bridge.chimerax_path}")
    info("This may take 15–30 seconds for ChimeraX to initialise …")
    try:
        bridge.start(timeout=60)
        ok("ChimeraX started and REST server is up.")
        return True
    except TimeoutError as exc:
        fail(f"Timed out waiting for REST server: {exc}")
        return False
    except Exception as exc:
        fail(f"Failed to start ChimeraX: {exc}")
        return False


def test_ping(bridge: ChimeraXBridge) -> bool:
    """Send a lightweight echo command and measure round-trip latency."""
    section("4 · Ping (echo command)")
    if not bridge.is_running():
        warn("Skipping — ChimeraX not running.")
        return False

    result = bridge.ping()
    if result["ok"]:
        ok(f"Ping successful — latency {result['latency_ms']} ms")
    else:
        fail(f"Ping failed: {result['result'].get('error')}")
    return result["ok"]


def test_basic_commands(bridge: ChimeraXBridge) -> bool:
    """Run a handful of safe, read-only commands to verify the REST bridge.

    Note: commands that output to ChimeraX's GUI log (echo, help, log) return
    an empty string over REST — only commands that return data directly work.
    """
    section("5 · Basic command round-trip")
    if not bridge.is_running():
        warn("Skipping — ChimeraX not running.")
        return False

    tests = [
        # (command, expected_substring_in_value, label)
        ("version",                "ChimeraX", "version string"),
        ("info models",            "",          "info models (empty session)"),
        ("set bgColor white",      "",          "set bgColor (no-output command)"),
    ]

    all_ok = True
    for cmd, expected, label in tests:
        result = bridge.run_command(cmd)
        if result.get("error"):
            fail(f"{label}: error — {result['error'][:80]}")
            all_ok = False
        else:
            val = (result.get("value") or "")
            val_preview = val[:60].replace("\n", " ").strip()
            if expected and expected not in val:
                fail(f"{label}: expected {expected!r} in response, got {val_preview!r}")
                all_ok = False
            else:
                ok(f"{label}: {val_preview!r}")

    return all_ok


def test_open_close(bridge: ChimeraXBridge) -> bool:
    """
    Full round-trip: open a PDB from RCSB, check it loaded, then close it.
    WARNING: this fetches from the internet and modifies the ChimeraX session.
    """
    section("6 · Open / close cycle  (internet + session modification)")
    if not bridge.is_running():
        warn("Skipping — ChimeraX not running.")
        return False

    pdb_id = "1AON"  # GroEL — well-known test structure
    info(f"Opening {pdb_id} from RCSB PDB …")
    result = bridge.run_command(f"open {pdb_id}")
    if result.get("error"):
        fail(f"open {pdb_id}: {result['error']}")
        return False
    ok(f"Opened {pdb_id}")

    # Verify it's loaded
    time.sleep(1)
    result2 = bridge.run_command("info models")
    if result2.get("error"):
        warn(f"Could not query models: {result2['error']}")
    else:
        preview = (result2.get("value") or "")[:120].replace("\n", " ").strip()
        info(f"Models: {preview!r}")

    # Close it
    result3 = bridge.run_command("close #1")
    if result3.get("error"):
        warn(f"close #1: {result3['error']}")
    else:
        ok(f"Closed model #1 — session is clean")

    return True


def test_error_handling(bridge: ChimeraXBridge) -> bool:
    """Confirm that malformed commands surface as errors, not silent successes.

    ChimeraX returns errors as 200 OK with a plain-text body starting with
    'Unknown command: ...'. The bridge must detect this pattern and set
    result['error'] so callers don't treat failures as successes.
    """
    section("7 · Error handling (intentional bad command)")
    if not bridge.is_running():
        warn("Skipping — ChimeraX not running.")
        return False

    result = bridge.run_command("this_command_does_not_exist_xyz")
    if result.get("error"):
        ok(f"Bad command correctly surfaced as error: {result['error'][:80]!r}")
        return True
    else:
        fail(
            f"Bad command was NOT detected as an error.\n"
            f"  value={result.get('value','')[:80]!r}\n"
            f"  This means command failures will be silently ignored in main.py."
        )
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="StructureBot — ChimeraX connectivity test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Attempt to launch ChimeraX if it is not already running",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all tests including open/close (requires internet access)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=60001,
        help="ChimeraX REST server port (default: 60001)",
    )
    parser.add_argument(
        "--chimerax",
        metavar="PATH",
        help="Override ChimeraX executable path",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{CYAN}StructureBot — ChimeraX Bridge Test{RESET}")
    print(f"{'─' * 50}")
    print(f"  Port  : {args.port}")
    print(f"  Python: {sys.version.split()[0]}")

    bridge = ChimeraXBridge(chimerax_path=args.chimerax, port=args.port)

    passed  = 0
    skipped = 0
    failed  = 0

    def run(fn, *fn_args) -> None:
        nonlocal passed, skipped, failed
        try:
            result = fn(*fn_args)
            if result:
                passed  += 1
            else:
                skipped += 1
        except Exception as exc:
            fail(f"Unexpected exception: {exc}")
            failed += 1

    # Always run
    run(test_find_chimerax)
    run(test_connection_check, bridge)

    # Run only if --start or --full
    if args.start or args.full:
        run(test_start_chimerax, bridge)

    # Run remaining tests only if REST server is reachable
    if bridge.is_running():
        run(test_ping, bridge)
        run(test_basic_commands, bridge)
        run(test_error_handling, bridge)
        if args.full:
            run(test_open_close, bridge)
    else:
        warn(
            "\nChimeraX is not running — connectivity tests skipped.\n"
            "  • Open ChimeraX manually and run:  remotecontrol rest start port 60001\n"
            "  • Or re-run this script with:      python test_bridge.py --start"
        )

    # Summary
    section("Summary")
    total = passed + skipped + failed
    print(f"  {GREEN}{passed} passed{RESET}  "
          f"{YELLOW}{skipped} skipped{RESET}  "
          f"{RED}{failed} failed{RESET}  "
          f"(of {total} tests)")

    if failed:
        print(f"\n{RED}Some tests failed — see messages above.{RESET}")
        sys.exit(1)
    elif not bridge.is_running():
        print(f"\n{YELLOW}Basic checks passed, but ChimeraX is not running.{RESET}")
        print("Re-run with --start to test the full connection.")
        sys.exit(0)
    else:
        print(f"\n{GREEN}All tests passed — StructureBot is ready to use!{RESET}")
        sys.exit(0)


if __name__ == "__main__":
    main()
