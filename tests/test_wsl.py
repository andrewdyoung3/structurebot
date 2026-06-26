"""
tests/test_wsl.py
-----------------
Tests for WSLBridge (wsl_bridge.py).

These tests run regardless of whether WSL2 is actually installed.
When WSL2 is not installed, most tests check graceful degradation.
When WSL2 is installed (CI or configured machine), behaviour tests
additionally verify that actual command execution works.

Usage
-----
  cd structurebot
  python -m pytest tests/test_wsl.py -v
  # or
  python tests/test_wsl.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from wsl_bridge import WSLBridge, _check_wsl_availability

# ── Helpers ────────────────────────────────────────────────────────────────────

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

_results = {"pass": 0, "fail": 0, "skip": 0}


def _ok(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {PASS} {name}{suffix}")
    _results["pass"] += 1


def _fail(name: str, reason: str) -> None:
    print(f"  {FAIL} {name}: {reason}")
    _results["fail"] += 1


def _skip(name: str, reason: str) -> None:
    print(f"  {SKIP} {name}: {reason}")
    _results["skip"] += 1


def _assert(cond: bool, name: str, msg: str = "") -> bool:
    if cond:
        _ok(name)
        return True
    else:
        _fail(name, msg or "assertion failed")
        return False


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_import() -> None:
    """WSLBridge can be imported and instantiated without errors."""
    print("\n=== A. Import and instantiation ===")
    try:
        bridge = WSLBridge()
        _ok("WSLBridge import and instantiation")
    except Exception as exc:
        _fail("WSLBridge import and instantiation", str(exc))


def test_is_available_returns_bool() -> None:
    """WSLBridge.is_available() returns a bool."""
    print("\n=== B. is_available() ===")
    bridge = WSLBridge()
    result = bridge.is_available()
    _assert(isinstance(result, bool), "is_available() returns bool", f"got {type(result)}")


def test_is_available_stable() -> None:
    """Repeated calls return the same value (result is cached)."""
    bridge = WSLBridge()
    r1 = bridge.is_available()
    r2 = bridge.is_available()
    _assert(r1 == r2, "is_available() stable across calls", f"{r1!r} vs {r2!r}")


def test_path_translation_c_drive() -> None:
    """Windows C:\\ paths are translated to /mnt/c/..."""
    print("\n=== C. Path translation ===")
    bridge = WSLBridge()

    cases = [
        (r"C:\Users\andre\docs\file.pdb", "/mnt/c/Users/andre/docs/file.pdb"),
        (r"C:/Users/andre/docs/file.pdb", "/mnt/c/Users/andre/docs/file.pdb"),
        ("C:\\",                           "/mnt/c/"),
        (r"D:\data\protein.pdb",          "/mnt/d/data/protein.pdb"),
    ]
    all_ok = True
    for win_path, expected in cases:
        got = bridge.translate_path(win_path)
        if got != expected:
            _fail(f"translate_path({win_path!r})", f"expected {expected!r}, got {got!r}")
            all_ok = False
    if all_ok:
        _ok("translate_path — Windows drive letters")


def test_path_translation_wsl_passthrough() -> None:
    """WSL paths (/mnt/…) are passed through unchanged."""
    bridge = WSLBridge()
    cases = [
        "/mnt/c/Users/andre/file.pdb",
        "/tmp/rosetta_output.json",
        "/home/user/protein.pdb",
    ]
    all_ok = True
    for p in cases:
        got = bridge.translate_path(p)
        if got != p:
            _fail(f"translate_path passthrough ({p!r})", f"got {got!r}")
            all_ok = False
    if all_ok:
        _ok("translate_path — WSL paths passthrough")


def test_run_command_no_wsl() -> None:
    """run_command returns a well-formed error dict when WSL2 is unavailable."""
    print("\n=== D. run_command (graceful degradation) ===")
    bridge = WSLBridge()
    if bridge.is_available():
        _skip("run_command_no_wsl", "WSL2 is installed — skip no-WSL degradation test")
        return

    result = bridge.run_command("echo hello")
    _assert(isinstance(result, dict),          "run_command returns dict")
    _assert(result.get("ok") is False,         "run_command ok=False when unavailable")
    _assert(bool(result.get("error")),         "run_command has error message")
    _assert("returncode" in result,            "run_command has returncode key")
    _assert("stdout"     in result,            "run_command has stdout key")
    _assert("stderr"     in result,            "run_command has stderr key")


def test_copy_to_wsl_no_wsl() -> None:
    """copy_to_wsl returns empty string when WSL2 is unavailable."""
    bridge = WSLBridge()
    if bridge.is_available():
        _skip("copy_to_wsl_no_wsl", "WSL2 is installed — skip degradation test")
        return

    result = bridge.copy_to_wsl(r"C:\does\not\exist.pdb")
    _assert(result == "", "copy_to_wsl returns empty string when unavailable")


def test_copy_from_wsl_no_wsl() -> None:
    """copy_from_wsl returns False when WSL2 is unavailable."""
    bridge = WSLBridge()
    if bridge.is_available():
        _skip("copy_from_wsl_no_wsl", "WSL2 is installed — skip degradation test")
        return

    result = bridge.copy_from_wsl("/tmp/file.json", r"C:\temp\file.json")
    _assert(result is False, "copy_from_wsl returns False when unavailable")


def test_status_string_type() -> None:
    """status_string() returns a non-empty string."""
    print("\n=== E. status_string ===")
    bridge = WSLBridge()
    s = bridge.status_string()
    _assert(isinstance(s, str) and len(s) > 0, "status_string returns non-empty str",
            f"got {s!r}")


def test_status_string_not_installed() -> None:
    """When WSL2 is not installed, status_string mentions the install command."""
    bridge = WSLBridge()
    if bridge.is_available():
        _skip("status_string_not_installed", "WSL2 is installed")
        return
    s = bridge.status_string().lower()
    _assert(
        "wsl" in s or "ubuntu" in s or "not installed" in s,
        "status_string mentions WSL2 install when unavailable",
        f"got: {bridge.status_string()!r}",
    )


def test_run_command_wsl() -> None:
    """
    When WSL2 IS installed: verify basic command execution.
    Skipped if WSL2 is not available.
    """
    print("\n=== F. Live WSL2 execution (skipped if not installed) ===")
    bridge = WSLBridge()
    if not bridge.is_available():
        _skip("run_command echo", "WSL2 not installed")
        return

    result = bridge.run_command("echo hello-from-wsl2", timeout=15)
    _assert(result["ok"],                            "run_command ok=True for echo")
    _assert("hello-from-wsl2" in result["stdout"],   "stdout contains echo output",
            f"got: {result['stdout']!r}")


def test_copy_roundtrip_wsl() -> None:
    """
    When WSL2 IS installed: copy a file to WSL2, then copy it back.
    Skipped if WSL2 is not available.
    """
    import tempfile
    bridge = WSLBridge()
    if not bridge.is_available():
        _skip("copy_roundtrip", "WSL2 not installed")
        return

    content = "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 10.00\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pdb", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(content)
        win_src = fh.name

    try:
        wsl_path = bridge.copy_to_wsl(win_src)
        _assert(bool(wsl_path), "copy_to_wsl returns non-empty path",
                f"got {wsl_path!r}")

        if wsl_path:
            win_dest = win_src + ".roundtrip.pdb"
            ok = bridge.copy_from_wsl(wsl_path, win_dest)
            _assert(ok, "copy_from_wsl returned True")
            if ok:
                _assert(Path(win_dest).is_file(), "roundtrip file exists")
                got_content = Path(win_dest).read_text(encoding="utf-8")
                _assert(content in got_content, "roundtrip file content matches")
                try:
                    Path(win_dest).unlink()
                except OSError:
                    pass
    finally:
        try:
            os.unlink(win_src)
        except OSError:
            pass


# ── Bounded-wait (Mode-A completion-signal fix) — REAL asserts, WSL not required ──
#
# These exercise the bounded-wait path in run_command directly via the _popen_wsl seam,
# so they prove the fix with NO WSL/GPU. They use plain `assert` (pytest-enforced), unlike
# the soft _assert helper above, because the bounded-wait guarantee is load-bearing.

import subprocess as _sp
import threading as _th
import time as _tm


class _FakePopen:
    """Stand-in for the wsl.exe Popen. *behavior* is called by communicate(); set
    returncode/raise/block from there to model normal / timeout / never-draining runs."""

    def __init__(self, behavior):
        self.returncode = None
        self.killed = False
        self._behavior = behavior
        self.calls = 0

    def communicate(self, timeout=None):
        self.calls += 1
        return self._behavior(self, timeout)

    def kill(self):
        self.killed = True


def _bridge_with_fake(behavior, *, drain_timeout=0.3):
    """A WSLBridge whose wsl.exe spawn is a _FakePopen and whose WSL-tree kill is recorded
    (never spawns real wsl.exe). Returns (bridge, captured)."""
    bridge = WSLBridge()
    bridge.is_available = lambda: True                  # bypass the not-installed gate
    bridge._drain_timeout = drain_timeout
    captured = {"spawned_args": None, "killed_markers": [], "popen": None}

    def _fake_popen(wsl_args):
        captured["spawned_args"] = wsl_args
        p = _FakePopen(behavior)
        captured["popen"] = p
        return p

    def _fake_kill(marker):
        captured["killed_markers"].append(marker)

    bridge._popen_wsl = _fake_popen
    bridge._kill_wsl_tree = _fake_kill
    return bridge, captured


def test_run_command_normal_path_returns() -> None:
    """The Popen refactor preserves the success path: communicate → ok/stdout/returncode."""
    print("\n=== F. bounded-wait: normal path ===")

    def _ok_behavior(p, timeout):
        p.returncode = 0
        return ("hello\n", "")

    bridge, _ = _bridge_with_fake(_ok_behavior)
    r = bridge.run_command("echo hello", timeout=5)
    assert r["ok"] is True, r
    assert r["returncode"] == 0
    assert r["stdout"] == "hello\n"
    assert r["error"] is None
    _ok("bounded-wait normal path returns ok+stdout")


def test_run_command_bounded_drain_never_hangs() -> None:
    """THE load-bearing test: a never-draining subprocess (the real Mode-A hang) must NOT
    hang run_command — it returns an honest 'timed out' terminal error within the wall."""
    print("\n=== G. bounded-wait: never-draining subprocess ===")
    _blocked = _th.Event()                              # never set → the drain blocks forever

    def _hang_behavior(p, timeout):
        if timeout is not None:                         # the main communicate(timeout=…)
            raise _sp.TimeoutExpired(cmd="wsl", timeout=timeout)
        _blocked.wait()                                 # the post-kill drain: blocks forever
        return ("", "")

    bridge, captured = _bridge_with_fake(_hang_behavior, drain_timeout=0.3)

    box = {"r": None}

    def _call():
        box["r"] = bridge.run_command("boltz predict …", timeout=0.2)

    t = _th.Thread(target=_call, daemon=True)
    t0 = _tm.perf_counter()
    t.start()
    t.join(5.0)                                         # generous wall; the bound is ~0.5s
    elapsed = _tm.perf_counter() - t0

    assert not t.is_alive(), "run_command HUNG on a never-draining subprocess (the bug)"
    assert box["r"] is not None
    assert box["r"]["ok"] is False
    assert "timed out" in box["r"]["error"].lower()     # _run_failure_message keys on this
    assert "drain abandoned" in box["r"]["error"]       # the final drain was bounded, not awaited
    assert captured["killed_markers"], "the WSL-side tree was not reaped (only proc.kill)"
    assert captured["popen"].killed is True             # the Windows handle was killed too
    # the spawned command carried the reap marker that _kill_wsl_tree was handed
    tagged = captured["spawned_args"][-1]
    assert f"SBWSL_TAG={captured['killed_markers'][0]}" in tagged
    _ok(f"never-draining subprocess → terminal error in {elapsed:.2f}s (no hang)")


def test_run_command_timeout_reaps_then_drains() -> None:
    """On timeout the WSL-side tree is reaped BEFORE the drain (the ordering that makes the
    drain actually close), and a drain that DOES complete is reported as 'drain completed'."""
    print("\n=== H. bounded-wait: reap-then-drain ordering ===")
    order = []

    def _behavior(p, timeout):
        if timeout is not None:
            order.append("communicate_timeout")
            raise _sp.TimeoutExpired(cmd="wsl", timeout=timeout)
        order.append("drain")                           # completes promptly this time
        return ("partial out", "partial err")

    bridge, captured = _bridge_with_fake(_behavior, drain_timeout=2.0)

    def _kill(marker):
        order.append("kill_tree")
        captured["killed_markers"].append(marker)

    bridge._kill_wsl_tree = _kill
    r = bridge.run_command("boltz predict …", timeout=0.2)

    assert r["ok"] is False
    assert "drain completed" in r["error"], r["error"]
    assert r["stdout"] == "partial out"                 # a completed drain keeps the output
    # ordering: timeout → reap the WSL tree → THEN drain (so the pipe can actually close)
    assert order == ["communicate_timeout", "kill_tree", "drain"], order
    _ok("timeout reaps WSL tree before the bounded drain; completed drain keeps output")


# ── Runner ─────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("tests/test_wsl.py — WSL2 Bridge Tests")
    print("=" * 60)

    test_import()
    test_is_available_returns_bool()
    test_is_available_stable()
    test_path_translation_c_drive()
    test_path_translation_wsl_passthrough()
    test_run_command_no_wsl()
    test_copy_to_wsl_no_wsl()
    test_copy_from_wsl_no_wsl()
    test_status_string_type()
    test_status_string_not_installed()
    test_run_command_wsl()
    test_copy_roundtrip_wsl()
    test_run_command_normal_path_returns()
    test_run_command_bounded_drain_never_hangs()
    test_run_command_timeout_reaps_then_drains()

    print()
    print("=" * 60)
    print(
        f"Results: {_results['pass']} passed, "
        f"{_results['fail']} failed, "
        f"{_results['skip']} skipped"
    )
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
