"""
Async-dispatch wiring tests for the sequence editor's off-thread select
(seq_editor.view._SelectWorker). Verifies the worker calls the controller with the
coalesced selection and returns the result via signal — including error-first (a
failed select surfaces in the result dict, never raised). Qt is exercised offscreen;
no window, no live ChimeraX.
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402
from seq_editor.view import _SelectWorker  # noqa: E402


class _StubController:
    def __init__(self, result):
        self.calls = []
        self.result = result

    def select_in_3d(self, model, chain, resnums):
        self.calls.append((model, chain, tuple(resnums)))
        return self.result


@pytest.fixture(scope="module")
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_select_worker_dispatches_coalesced_set_and_emits(_app):
    ctl = _StubController({"value": "", "error": None})
    got = {}
    w = _SelectWorker(ctl, "1", "A", [50, 51, 53])
    w.signals.done.connect(lambda d: got.update(d))
    w.run()                                            # same-thread direct connection → sync
    assert ctl.calls == [("1", "A", (50, 51, 53))]     # the coalesced set, one dispatch
    assert got.get("error") is None


def test_select_worker_surfaces_error_first(_app):
    ctl = _StubController({"value": None, "error": "Connection lost"})
    got = {}
    w = _SelectWorker(ctl, "1", "A", [50])
    w.signals.done.connect(lambda d: got.update(d))
    w.run()
    assert got.get("error") == "Connection lost"       # surfaced via signal, not swallowed/raised
