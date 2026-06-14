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
