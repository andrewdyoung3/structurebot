"""
tests/test_chimerax_bridge.py
-----------------------------
Tests for ChimeraXBridge automatic reconnection on a dropped REST connection.

All HTTP / process interactions are mocked (the single-attempt worker
_run_command_once and the _try_reconnect helper are patched), so no real
ChimeraX or network access occurs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from chimerax_bridge import ChimeraXBridge


def _bridge() -> ChimeraXBridge:
    # Passing an explicit path avoids find_chimerax() filesystem scanning.
    return ChimeraXBridge(chimerax_path="X", port=60001)


def test_run_command_reconnects_and_retries_on_connection_error():
    """
    First attempt raises ConnectionError (server dropped); reconnect succeeds;
    retry succeeds → run_command returns the successful result WITHOUT surfacing
    the error.
    """
    bridge = _bridge()
    ok = {"value": "done", "error": None}

    once = MagicMock(side_effect=[ConnectionError("server dropped"), ok])
    reconnect = MagicMock(return_value=True)

    with patch.object(bridge, "_run_command_once", once), \
         patch.object(bridge, "_try_reconnect", reconnect):
        result = bridge.run_command("cartoon #1")

    assert result == ok
    assert result.get("error") is None          # error not surfaced
    assert once.call_count == 2                  # initial attempt + one retry
    reconnect.assert_called_once()               # reconnected exactly once


def test_run_command_raises_clear_error_when_reconnect_fails():
    """If reconnection fails, a ConnectionError with an actionable message is raised."""
    bridge = _bridge()
    once = MagicMock(side_effect=ConnectionError("server dropped"))

    with patch.object(bridge, "_run_command_once", once), \
         patch.object(bridge, "_try_reconnect", MagicMock(return_value=False)):
        with pytest.raises(ConnectionError) as exc_info:
            bridge.run_command("cartoon #1")

    assert "ChimeraX is still open" in str(exc_info.value)
    assert once.call_count == 1                  # not retried after reconnect failed


def test_run_command_success_path_does_not_reconnect():
    """A normal successful command must not attempt any reconnection."""
    bridge = _bridge()
    ok = {"value": "", "error": None}
    once = MagicMock(return_value=ok)
    reconnect = MagicMock(return_value=True)

    with patch.object(bridge, "_run_command_once", once), \
         patch.object(bridge, "_try_reconnect", reconnect):
        result = bridge.run_command("color #1 red")

    assert result == ok
    once.assert_called_once()
    reconnect.assert_not_called()


def test_ensure_visible_gui_reuses_running_visible_chimerax():
    """REST reachable + a visible ChimeraX window → reuse it ('connected'), no relaunch."""
    bridge = _bridge()
    with patch.object(bridge, "is_running", MagicMock(return_value=True)), \
         patch.object(bridge, "_visible_chimerax_window_exists",
                      MagicMock(return_value=True)), \
         patch.object(bridge, "start", MagicMock()) as start, \
         patch.object(bridge, "_kill_all_chimerax", MagicMock()) as kill:
        assert bridge.ensure_visible_gui() == "connected"
    start.assert_not_called()
    kill.assert_not_called()


def test_ensure_visible_gui_starts_when_nothing_running():
    """No REST server → launch fresh ('started')."""
    bridge = _bridge()
    with patch.object(bridge, "is_running", MagicMock(return_value=False)), \
         patch.object(bridge, "start", MagicMock(return_value=True)) as start, \
         patch.object(bridge, "_kill_all_chimerax", MagicMock()) as kill:
        assert bridge.ensure_visible_gui() == "started"
    start.assert_called_once()
    kill.assert_not_called()


def test_ensure_visible_gui_replaces_windowless_zombie():
    """REST reachable but NO visible window → kill the zombie and relaunch a fresh
    visible instance ('relaunched'). is_running() returns False after the kill so the
    drop-wait loop exits immediately."""
    bridge = _bridge()
    running = MagicMock(side_effect=[True, False])   # before kill: up; after kill: gone
    with patch.object(bridge, "is_running", running), \
         patch.object(bridge, "_visible_chimerax_window_exists",
                      MagicMock(return_value=False)), \
         patch.object(bridge, "_kill_all_chimerax", MagicMock()) as kill, \
         patch.object(bridge, "start", MagicMock(return_value=True)) as start:
        assert bridge.ensure_visible_gui() == "relaunched"
    kill.assert_called_once()
    start.assert_called_once()


def test_run_commands_recovers_mid_list_via_reconnect():
    """
    run_commands benefits from the same reconnect: a mid-list connection drop
    recovers and the batch completes without an error result.
    """
    bridge = _bridge()
    ok = {"value": "", "error": None}
    # cmd1 ok; cmd2 drops then succeeds after reconnect.
    once = MagicMock(side_effect=[ok, ConnectionError("drop"), ok])

    with patch.object(bridge, "_run_command_once", once), \
         patch.object(bridge, "_try_reconnect", MagicMock(return_value=True)):
        results = bridge.run_commands(["cmd1", "cmd2"])

    assert len(results) == 2
    assert all(r["result"].get("error") is None for r in results)
    assert once.call_count == 3                  # cmd1 + (cmd2 fail + cmd2 retry)


# ── PROCESS-LEAK FIX (3 compounding flaws) ────────────────────────────────────────────────

def test_start_kills_spawned_process_on_timeout_no_leak():
    """FLAW 1 — the teardown backstop: a start() whose spawned ChimeraX never binds REST must
    KILL that process before raising, not leave it running (the per-failure window that piled up)."""
    bridge = _bridge()
    proc = MagicMock(); proc.poll.return_value = None          # spawned, still alive
    with patch.object(bridge, "is_running", MagicMock(return_value=False)), \
         patch.object(bridge, "_chimerax_process_exists", MagicMock(return_value=False)), \
         patch("chimerax_bridge.subprocess.Popen", MagicMock(return_value=proc)), \
         patch("chimerax_bridge.Path") as _path, \
         patch("chimerax_bridge.time.sleep", MagicMock()), \
         patch("chimerax_bridge.time.time", MagicMock(side_effect=[0, 0, 100])):  # 0 then past the deadline
        _path.return_value.is_file.return_value = True
        with pytest.raises(TimeoutError):
            bridge.start(timeout=1)
    proc.terminate.assert_called_once()                        # the leaked process was killed…
    assert bridge._process is None                             # …and the handle forgotten


def test_run_command_drop_does_not_spawn_and_fails_loud():
    """FLAW 2 — a genuine REST drop surfaces an honest error pointing at Reconnect, and NEVER
    auto-spawns a ChimeraX (the cascade). _try_reconnect re-checks WITHOUT auto_start."""
    bridge = _bridge()
    once = MagicMock(side_effect=ConnectionError("server dropped"))
    with patch.object(bridge, "_run_command_once", once), \
         patch.object(bridge, "is_running", MagicMock(return_value=False)), \
         patch.object(bridge, "start", MagicMock()) as start:
        with pytest.raises(ConnectionError) as exc:
            bridge.run_command("color #1 red")
    start.assert_not_called()                                  # NO new ChimeraX spawned on a drop
    msg = str(exc.value)
    assert "ChimeraX is still open" in msg and "Reconnect" in msg   # honest + actionable recovery


def test_try_reconnect_never_auto_starts():
    """FLAW 2 (unit) — _try_reconnect must call ensure_connected with auto_start=False (re-check
    only), so a dropped command can never relaunch ChimeraX from the command path."""
    bridge = _bridge()
    with patch.object(bridge, "ensure_connected", MagicMock(return_value=False)) as ec:
        assert bridge._try_reconnect() is False
    ec.assert_called_once_with(auto_start=False)


def test_start_replaces_zombie_process_holding_port():
    """FLAW 3 — REST down on :port BUT a ChimeraX process exists (a zombie squatting the port) →
    start() REPLACES it (kill-all) before spawning, instead of spawning a 2nd that can't bind."""
    bridge = _bridge()
    proc = MagicMock(); proc.poll.return_value = None
    # is_running: False (port dead) at entry; True after the fresh spawn binds.
    running = MagicMock(side_effect=[False, True])
    # process exists at entry (zombie), gone after kill.
    exists = MagicMock(side_effect=[True, False])
    with patch.object(bridge, "is_running", running), \
         patch.object(bridge, "_chimerax_process_exists", exists), \
         patch.object(bridge, "_kill_all_chimerax", MagicMock()) as kill, \
         patch("chimerax_bridge.subprocess.Popen", MagicMock(return_value=proc)), \
         patch("chimerax_bridge.Path") as _path, \
         patch("chimerax_bridge.time.sleep", MagicMock()):
        _path.return_value.is_file.return_value = True
        assert bridge.start(timeout=5) is True
    kill.assert_called_once()                                  # the zombie was replaced, not collided with


def test_start_reuses_reachable_rest_no_spawn():
    """A REST server already reachable on the port → reuse it, never spawn or kill."""
    bridge = _bridge()
    with patch.object(bridge, "is_running", MagicMock(return_value=True)), \
         patch("chimerax_bridge.subprocess.Popen", MagicMock()) as popen, \
         patch("chimerax_bridge.Path") as _path, \
         patch.object(bridge, "_kill_all_chimerax", MagicMock()) as kill:
        _path.return_value.is_file.return_value = True
        assert bridge.start() is True
    popen.assert_not_called()
    kill.assert_not_called()


def test_start_resets_lean_layout_on_fresh_spawn():
    """A freshly spawned ChimeraX must re-apply the lean layout: the once-per-session
    guard is reset on spawn (else the relaunched window keeps Log/Models/toolbars until
    the next open — the live-verify Test 2 issue)."""
    bridge = _bridge()
    bridge._lean_layout_applied = True                        # carried over from a prior instance
    proc = MagicMock(); proc.poll.return_value = None
    running = MagicMock(side_effect=[False, True])            # port dead at entry, up after spawn
    with patch.object(bridge, "is_running", running), \
         patch.object(bridge, "_chimerax_process_exists", MagicMock(return_value=False)), \
         patch("chimerax_bridge.subprocess.Popen", MagicMock(return_value=proc)), \
         patch("chimerax_bridge.Path") as _path, \
         patch("chimerax_bridge.time.sleep", MagicMock()):
        _path.return_value.is_file.return_value = True
        assert bridge.start(timeout=5) is True
    assert bridge._lean_layout_applied is False               # reset so the new window gets it


def test_start_reuse_does_not_reset_lean_layout():
    """Reusing a reachable REST server must NOT reset the guard (no fresh window → no re-apply)."""
    bridge = _bridge()
    bridge._lean_layout_applied = True
    with patch.object(bridge, "is_running", MagicMock(return_value=True)), \
         patch("chimerax_bridge.subprocess.Popen", MagicMock()), \
         patch("chimerax_bridge.Path") as _path:
        _path.return_value.is_file.return_value = True
        assert bridge.start() is True
    assert bridge._lean_layout_applied is True                # untouched on reuse
