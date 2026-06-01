"""
tests/test_main.py
------------------
Unit tests for StructureBot REPL helpers:
  - Semicolon command chaining (_dispatch_input)
  - Script-file runner (run_script / --script flag)

All ChimeraX, translator, and session interactions are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import StructureBot


# ── Module-wide console patch ─────────────────────────────────────────────────
# Rich's console.print() calls that emit ✓, ✗ or similar Unicode characters
# fail on Windows when stdout uses cp1252 (the default in the test runner).
# Suppress all console output for every test in this file.

@pytest.fixture(autouse=True)
def _suppress_console(monkeypatch):
    """Replace main.console with a silent MagicMock for all tests here."""
    monkeypatch.setattr("main.console", MagicMock())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_bot() -> StructureBot:
    """
    Create a StructureBot whose heavyweight deps (bridge, translator, ChimeraX)
    are all replaced with MagicMock, bypassing __init__ entirely.
    """
    bot = object.__new__(StructureBot)
    bot.bridge              = MagicMock()
    bot.translator          = MagicMock()
    bot.session             = MagicMock()
    bot.router              = MagicMock()
    bot.auto_proceed        = False
    bot.auto_proceed_delay  = 3
    bot.log_file            = Path("test_session.jsonl")

    # Default: no active-site / sequence-display short-circuits
    bot.router.handle_active_site_command.return_value    = None
    bot.router.handle_sequence_display_command.return_value = None
    bot.router.handle_selection_command.return_value      = None

    return bot


# ════════════════════════════════════════════════════════════════════════════════
# Semicolon chaining
# ════════════════════════════════════════════════════════════════════════════════

class TestSemicolonChaining:
    def test_semicolon_split_runs_both_commands(self):
        """
        "cmd1; cmd2" must call _handle_request twice — once per part.
        """
        bot = _make_mock_bot()
        bot._handle_request = MagicMock()

        bot._dispatch_input("cmd1; cmd2")

        assert bot._handle_request.call_count == 2
        bot._handle_request.assert_any_call("cmd1")
        bot._handle_request.assert_any_call("cmd2")

    def test_semicolon_split_skips_empty_parts(self):
        """
        "cmd1;  ; cmd2" must call _handle_request exactly twice —
        the blank middle segment must not be passed to it.
        """
        bot = _make_mock_bot()
        bot._handle_request = MagicMock()

        bot._dispatch_input("cmd1;  ; cmd2")

        assert bot._handle_request.call_count == 2
        called_args = [c.args[0] for c in bot._handle_request.call_args_list]
        assert "" not in called_args
        assert "cmd1" in called_args
        assert "cmd2" in called_args

    def test_semicolon_split_preserves_order(self):
        """
        Commands must be run in the order they appear left-to-right.
        """
        bot = _make_mock_bot()
        bot._handle_request = MagicMock()

        bot._dispatch_input("alpha; beta; gamma")

        called_args = [c.args[0] for c in bot._handle_request.call_args_list]
        assert called_args == ["alpha", "beta", "gamma"]

    def test_no_semicolon_calls_handle_request_once(self):
        """
        Plain input (no semicolon) must result in exactly one _handle_request call.
        """
        bot = _make_mock_bot()
        bot._handle_request = MagicMock()

        bot._dispatch_input("suggest proline mutations to stabilise chain A")

        bot._handle_request.assert_called_once_with(
            "suggest proline mutations to stabilise chain A"
        )

    def test_active_site_command_short_circuits(self):
        """
        An active-site command must be handled by handle_active_site_command
        and NOT reach _handle_request.
        """
        bot = _make_mock_bot()
        bot.router.handle_active_site_command.return_value = "Active-site residues set: [25]."
        bot._handle_request = MagicMock()

        bot._dispatch_input("set active site residues 25")

        bot._handle_request.assert_not_called()


# ════════════════════════════════════════════════════════════════════════════════
# Script runner
# ════════════════════════════════════════════════════════════════════════════════

class TestScriptRunner:
    def test_script_flag_runs_commands_in_order(self, tmp_path):
        """
        A 3-line script file must call _handle_request 3 times in order.
        """
        script = tmp_path / "commands.txt"
        script.write_text("open 1HSG\ndesign chain A\nshow designed sequences\n",
                          encoding="utf-8")

        bot = _make_mock_bot()
        bot.startup          = MagicMock()
        bot._handle_request  = MagicMock()

        bot.run_script(str(script))

        assert bot._handle_request.call_count == 3
        called_args = [c.args[0] for c in bot._handle_request.call_args_list]
        assert called_args == [
            "open 1HSG",
            "design chain A",
            "show designed sequences",
        ]

    def test_script_flag_skips_comments_and_blanks(self, tmp_path):
        """
        Lines starting with '#' and blank lines must be silently skipped.
        """
        script = tmp_path / "commands.txt"
        script.write_text(
            "# This is a comment\n"
            "open 1HSG\n"
            "\n"
            "design chain A\n",
            encoding="utf-8",
        )

        bot = _make_mock_bot()
        bot.startup          = MagicMock()
        bot._handle_request  = MagicMock()

        bot.run_script(str(script))

        assert bot._handle_request.call_count == 2
        called_args = [c.args[0] for c in bot._handle_request.call_args_list]
        assert called_args == ["open 1HSG", "design chain A"]

    def test_script_file_not_found(self):
        """
        A non-existent script path must print an error and exit with a non-zero code.
        """
        bot = _make_mock_bot()
        bot.startup = MagicMock()

        with pytest.raises(SystemExit) as exc_info:
            bot.run_script("/nonexistent/path/that/does/not/exist.txt")

        assert exc_info.value.code != 0

    def test_script_calls_startup(self, tmp_path):
        """
        run_script() must call startup() before processing any commands.
        """
        script = tmp_path / "cmd.txt"
        script.write_text("open 1HSG\n", encoding="utf-8")

        bot = _make_mock_bot()
        bot.startup         = MagicMock()
        bot._handle_request = MagicMock()

        bot.run_script(str(script))

        bot.startup.assert_called_once()

    def test_script_saves_session(self, tmp_path):
        """
        run_script() must save the session after all commands have run.
        """
        script = tmp_path / "cmd.txt"
        script.write_text("open 1HSG\n", encoding="utf-8")

        bot = _make_mock_bot()
        bot.startup         = MagicMock()
        bot._handle_request = MagicMock()

        bot.run_script(str(script))

        bot.session.save.assert_called_once()
