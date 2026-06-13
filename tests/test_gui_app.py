"""
Stage-2 tests: QtPresenter (output-as-signal + blocking-ask round-trip via queue) and
the GUI host hooks (open -> focus the model tab; structure-state sync). Qt offscreen; no
live ChimeraX, no real window construction (which would build the translator/router).
"""
import os
import queue
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402
from qt_presenter import QtPresenter, PresenterSignals, CANCEL  # noqa: E402
from request_engine import RequestEngine  # noqa: E402
import gui_app  # noqa: E402
from seq_editor.controller import ResidueCell, ChainSeq  # noqa: E402


@pytest.fixture(scope="module")
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# ── QtPresenter: output renders to an HTML signal, never a widget ─────────────────

def test_output_emits_html_signal(_app):
    sig = PresenterSignals()
    got = []
    sig.append_html.connect(lambda h: got.append(h))
    pres = QtPresenter(sig)
    pres.success("done!")
    pres.warn("careful")
    pres.show_commands(["color #1 red"], ["recolor"], "high")
    assert any("done!" in h for h in got)
    assert any("careful" in h for h in got)
    assert any("Proposed Commands" in h and "color #1 red" in h for h in got)
    # HTML, not plain (colour spans / pre)
    assert any("<pre" in h or "<span" in h for h in got)


# ── QtPresenter: blocking ask round-trips through the queue (worker-block analog) ──

def test_ask_clarification_roundtrip(_app):
    sig = PresenterSignals()
    # UI-thread slot answers by putting into the reply queue (direct conn = synchronous)
    sig.ask.connect(lambda kind, payload, q: q.put("chain A"))
    pres = QtPresenter(sig)
    assert pres.ask_clarification("which chain?") == "chain A"


def test_confirm_maps_values(_app):
    for put, expect in [("proceed", "proceed"), ("edit", "edit"), (None, None), (CANCEL, None)]:
        sig = PresenterSignals()
        sig.ask.connect(lambda kind, payload, q, _p=put: q.put(_p))
        pres = QtPresenter(sig)
        assert pres.confirm("high") == expect


def test_ask_yes_no_and_edit(_app):
    sig = PresenterSignals()
    sig.ask.connect(lambda kind, payload, q: q.put(True if kind == "yesno" else ["x"]))
    pres = QtPresenter(sig)
    assert pres.ask_yes_no("Apply fix?") is True
    assert pres.ask_edit(["orig"]) == ["x"]


def test_cancel_shortcircuits_blocking_asks(_app):
    sig = PresenterSignals()
    sig.ask.connect(lambda *a: (_ for _ in ()).throw(AssertionError("must not emit when cancelled")))
    pres = QtPresenter(sig)
    pres.cancelled = True
    assert pres.ask_clarification("q") == ""        # on_cancel
    assert pres.confirm("high") is None
    assert pres.ask_yes_no("q") is False


# ── the engine drives QtPresenter end-to-end (same engine, GUI presenter) ─────────

def test_engine_drives_qt_presenter(_app):
    sig = PresenterSignals()
    out = []
    sig.append_html.connect(lambda h: out.append(h))
    sig.ask.connect(lambda kind, payload, q: q.put("proceed"))   # auto-confirm
    pres = QtPresenter(sig)

    host = MagicMock()
    r = {"commands": ["color #1 red"], "explanations": ["recolor"], "warnings": [],
         "confidence": "high", "tools_needed": ["chimerax"], "has_extra_tools": False,
         "clarification_needed": None}
    host.translator.translate.return_value = r
    host.router.route.return_value = r
    host.bridge.run_commands.return_value = [
        {"command": "color #1 red", "result": {"value": "", "error": None}}]
    eng = RequestEngine(host)
    eng.handle_request("color it red", pres)

    import re as _re
    text = _re.sub(r"<[^>]+>", "", " ".join(out))      # visible text (Rich spans stripped)
    assert "Proposed Commands" in text and "color #1 red" in text
    assert "Completed 1 command(s)." in text
    host.bridge.run_commands.assert_called_with(["color #1 red"])


# ── host hooks: open -> session add + record for focus; show_model -> tab ─────────

import types  # noqa: E402

W = gui_app.StructureBotWindow


def _fake(**attrs):
    """A non-QWidget stand-in carrying the attrs a method needs, with the real GUI
    methods bound to it — avoids constructing the QMainWindow (translator/router)."""
    obj = types.SimpleNamespace(**attrs)
    for name in ("_display_assembly_type_on_open", "_maybe_update_structure_state",
                 "_add_chain_tab", "_focus_model", "_on_model_loaded", "show_model"):
        setattr(obj, name, types.MethodType(getattr(W, name), obj))
    return obj


def test_maybe_update_structure_state_records_open(_app):
    w = _fake(session=MagicMock(), bridge=MagicMock(), presenter=MagicMock(), _pending_focus=[])
    w.session.next_model_id.return_value = "2"
    # foo.pdb (not a 4-char PDB id) skips the network assembly fetch
    w._maybe_update_structure_state(["open foo.pdb", "cartoon #2"])
    w.session.add_structure.assert_called_once()
    assert w._pending_focus == ["2"]               # recorded for post-request focus
    w.session.record_style.assert_called_with("cartoon #2")


def test_show_model_adds_and_focuses_tab(_app):
    w = _fake(_grids={}, tabs=QtWidgets.QTabWidget(), presenter=MagicMock())
    cells = [ResidueCell("2", "A", 1, "M", 1), ResidueCell("2", "A", 2, "G", 2)]
    chain = ChainSeq("2", "A", cells)
    w._on_model_loaded("2", [chain])
    assert ("2", "A") in w._grids
    assert w.tabs.count() == 1
    assert w.tabs.currentWidget().chain.key == ("2", "A")
