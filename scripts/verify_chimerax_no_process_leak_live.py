"""
Live-verify — ChimeraX PROCESS-LEAK fix (the 3 compounding flaws). The gate: a mid-session REST drop
must produce ONE honest error + a clean Reconnect — NOT a pile of N windows. We count real ChimeraX
processes throughout and assert the count stays BOUNDED.

Scenario (real ChimeraX, no GPU):
  0. Clean slate — kill any ChimeraX, confirm 0.
  1. start() launches ONE ChimeraX (leak-safe) → reachable, process count == 1.
  2. Induce a mid-session REST DROP that leaves the process alive (`remotecontrol rest stop`) — the
     "zombie holding the port" case that used to make every later command spawn a colliding instance.
  3. Issue SEVERAL commands on the dropped connection → each FAILS LOUD (ConnectionError mentioning
     Reconnect) and spawns NOTHING. THE GATE: process count stays 1, not 1+N. (Old behavior: +1 window
     per command.)
  4. Recover via the existing path (`ensure_visible_gui`, what the "Reconnect ChimeraX" button calls)
     → it REPLACES the zombie (kill + one fresh) rather than colliding → reachable again, count == 1.
  5. Clean up — kill all, confirm 0.

Run: venv/Scripts/python.exe scripts/verify_chimerax_no_process_leak_live.py   (launches ChimeraX; no GPU)
"""
import sys, time, subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config
config.load_env_file()
from chimerax_bridge import ChimeraXBridge

bridge = ChimeraXBridge(port=60001)

_checks = []
def check(name, ok, detail=""):
    _checks.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _count() -> int:
    """Real ChimeraX process count (Windows tasklist)."""
    try:
        out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq ChimeraX.exe", "/NH"],
                             capture_output=True, text=True).stdout or ""
        return sum(1 for ln in out.splitlines() if "chimerax.exe" in ln.lower())
    except Exception:
        return -1


def _kill_all():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "ChimeraX.exe"], capture_output=True)
    except Exception:
        pass
    time.sleep(1.0)


def main():
    # 0) clean slate ───────────────────────────────────────────────────────────────────────
    print("0) Clean slate…", flush=True)
    _kill_all()
    check("no ChimeraX running at start", _count() == 0, f"count={_count()}")

    # 1) leak-safe start → exactly ONE ChimeraX ─────────────────────────────────────────────
    print("1) start() launches ONE ChimeraX…", flush=True)
    try:
        bridge.start(timeout=120)
    except Exception as exc:
        print(f"  start() failed: {type(exc).__name__}: {exc}")
        check("leaked nothing even though start failed", _count() == 0, f"count={_count()}")
        return 1
    time.sleep(1.0)
    c1 = _count()
    check("exactly one ChimeraX after start()", c1 == 1, f"count={c1}")
    check("REST reachable after start()", bridge.is_running())

    # 2) induce a mid-session REST drop, process stays alive (the zombie case) ───────────────
    print("2) Induce REST drop (remotecontrol rest stop)…", flush=True)
    try:
        bridge.run_command("remotecontrol rest stop")
    except Exception:
        pass
    time.sleep(1.5)
    dropped = not bridge.is_running()
    check("REST is down but the process is still alive (zombie holding the port)",
          dropped and _count() == 1, f"is_running={bridge.is_running()}, count={_count()}")

    # 3) THE GATE — N commands on the dropped connection: fail loud, spawn NOTHING ───────────
    print("3) 6 commands on the dropped connection → fail-loud, NO new windows…", flush=True)
    errors, spawned = 0, False
    for i in range(6):
        try:
            bridge.run_command(f"color #1 red  # probe {i}")
        except ConnectionError as exc:
            errors += 1
            if i == 0:
                check("error is honest + routes to Reconnect",
                      "ChimeraX is still open" in str(exc) and "Reconnect" in str(exc))
        except Exception:
            pass
    c3 = _count()
    check("all 6 dropped commands failed loud (raised)", errors == 6, f"{errors}/6 raised")
    check("THE GATE: process count stayed BOUNDED (no per-command spawn)", c3 == 1,
          f"count={c3} (was 1; old behavior would be ~7)")

    # 4) recovery via the Reconnect path → REPLACE the zombie, not collide ───────────────────
    print("4) Reconnect (ensure_visible_gui) replaces the zombie…", flush=True)
    try:
        outcome = bridge.ensure_visible_gui(timeout=120)
    except Exception as exc:
        check("reconnect recovered", False, f"{type(exc).__name__}: {exc}")
        _kill_all(); return 1
    time.sleep(1.0)
    c4 = _count()
    check("Reconnect recovered REST", bridge.is_running(), f"outcome={outcome}")
    check("still exactly one ChimeraX after recovery (replace, not pile)", c4 == 1, f"count={c4}")

    # 5) clean up ────────────────────────────────────────────────────────────────────────────
    print("5) Clean up…", flush=True)
    _kill_all()
    check("no ChimeraX left running", _count() == 0, f"count={_count()}")

    ok = all(_checks) and bool(_checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'} — {sum(_checks)}/{len(_checks)} checks", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
