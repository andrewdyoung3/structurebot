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


# ── Stage 3: managed service, Ollama preflight, ground-truth tab focus ────────────

def test_managed_service_windowless_logged(_app, monkeypatch, tmp_path):
    cap = {}

    class FakeProc:
        def poll(self): return None

    def fake_popen(args, **kw):
        cap["args"], cap["kw"] = args, kw
        return FakeProc()

    monkeypatch.setattr(gui_app.subprocess, "Popen", fake_popen)
    svc = gui_app.ManagedService("ollama", ["ollama", "serve"], tmp_path / "o.log")
    svc.start()
    assert cap["args"] == ["ollama", "serve"]
    assert cap["kw"].get("stdout") is not None                 # logged, not inherited
    if sys.platform == "win32":
        assert cap["kw"].get("creationflags") == gui_app.subprocess.CREATE_NO_WINDOW
    assert (tmp_path / "o.log").exists()


def test_managed_service_stop_terminates(_app):
    svc = gui_app.ManagedService("x", ["x"], Path("nul"))

    class FakeProc:
        def __init__(self): self.killed = False
        def poll(self): return None
        def terminate(self): self.killed = True
        def wait(self, timeout=None): return 0
        def kill(self): pass

    fp = FakeProc()
    svc.proc = fp
    svc.stop()
    assert fp.killed and svc.proc is None


def _preflight_fake(**attrs):
    obj = types.SimpleNamespace(presenter=MagicMock(), _services=[],
                                _OLLAMA_MIN_VERSION=W._OLLAMA_MIN_VERSION, **attrs)
    obj._preflight_ollama = types.MethodType(W._preflight_ollama, obj)
    # stub the new I/O helpers so the wholesale _preflight_ollama test stays OFFLINE
    # (the version subprocess + the GPU warm-load are tested directly elsewhere).
    obj._ollama_cli_version = lambda: (None, "")
    obj._ollama_gpu_status = lambda base: (None, "undetermined")
    return obj


def test_preflight_ollama_already_up(_app, monkeypatch):
    class Resp:
        status_code = 200
        def json(self): return {"models": [{"name": "qwen3:8b"}]}
    monkeypatch.setattr(gui_app.requests, "get", lambda *a, **k: Resp())
    w = _preflight_fake()
    w._preflight_ollama()
    msgs = " ".join(str(c) for c in w.presenter.success.call_args_list)
    assert "Ollama" in msgs                                   # connected
    assert not w._services                                    # nothing spawned (already up)


def test_preflight_ollama_model_missing_is_blocking(_app, monkeypatch):
    class Resp:
        status_code = 200
        def json(self): return {"models": [{"name": "llama3:8b"}]}   # qwen3 absent
    monkeypatch.setattr(gui_app.requests, "get", lambda *a, **k: Resp())
    w = _preflight_fake()
    w._preflight_ollama()
    # Translation is local-only with NO fallback → a missing model is a BLOCKING error.
    errs = " ".join(str(c) for c in w.presenter.error.call_args_list)
    assert "REQUIRED" in errs and "missing" in errs and "pull" in errs


def test_bridge_on_structure_opened_fires(_app, monkeypatch):
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge.__new__(ChimeraXBridge)
    b._lean_layout_applied = True
    got = []
    b.on_structure_opened = lambda mid: got.append(mid)
    monkeypatch.setattr(b, "run_command", lambda c, timeout=30: {"value": "opened #3", "error": None})
    monkeypatch.setattr(b, "_maybe_apply_presentation_on_open", lambda: None)
    monkeypatch.setattr(b, "_maybe_apply_lean_layout", lambda: None)
    monkeypatch.setattr(b, "_maybe_show_sequence_on_open", lambda mid: None)
    b.run_commands(["open foo.pdb"])
    assert got == ["3"]                                       # the REAL opened id, from the bridge


def test_opened_mid_focus_uses_ground_truth(_app):
    shown = []
    w = types.SimpleNamespace(_opened_mids=["3"], _pending_focus=["1"])
    w._finish_request = lambda: None
    w.show_model = lambda mid: shown.append(mid)
    W._on_request_done(w)
    assert shown == ["3"]                                     # real id wins over next_model_id guess


def test_opened_mid_focus_falls_back_to_guess(_app):
    shown = []
    w = types.SimpleNamespace(_opened_mids=[], _pending_focus=["1"])
    w._finish_request = lambda: None
    w.show_model = lambda mid: shown.append(mid)
    W._on_request_done(w)
    assert shown == ["1"]                                     # fallback when bridge saw no open


# ── Stage 4: engine.dispatch (semicolon + fast-paths) shared with the GUI ─────────

def test_dispatch_semicolon_splits_into_requests(_app):
    eng = RequestEngine(MagicMock())
    calls = []
    eng.handle_request = lambda text, pres: calls.append(text)
    eng.dispatch("open 1hsg; cartoon #1 ;  ; color #1 red", MagicMock())
    assert calls == ["open 1hsg", "cartoon #1", "color #1 red"]   # blanks skipped, order kept


def test_dispatch_active_site_fast_path_short_circuits(_app):
    host = MagicMock()
    host.router.handle_active_site_command.return_value = "Active-site residues set: [25]."
    eng = RequestEngine(host)
    eng.handle_request = lambda *a: (_ for _ in ()).throw(AssertionError("must not reach LLM"))
    pres = MagicMock()
    eng.dispatch("set active site residues 25", pres)
    pres.active_site_ok.assert_called_once_with("Active-site residues set: [25].")


def test_dispatch_sequence_and_selection_fast_paths(_app):
    host = MagicMock()
    host.router.handle_active_site_command.return_value = None
    host.router.handle_sequence_display_command.return_value = "[bold]Sequences[/bold]"
    eng = RequestEngine(host)
    called = []
    eng.handle_request = lambda *a: called.append(1)
    pres = MagicMock()
    eng.dispatch("show designed sequences", pres)
    pres.markup.assert_called_once_with("[bold]Sequences[/bold]")
    assert not called


def test_dispatch_falls_through_to_handle_request(_app):
    host = MagicMock()
    host.router.handle_active_site_command.return_value = None
    host.router.handle_sequence_display_command.return_value = None
    host.router.handle_selection_command.return_value = None
    eng = RequestEngine(host)
    got = []
    eng.handle_request = lambda text, pres: got.append(text)
    eng.dispatch("suggest stabilising mutations", MagicMock())
    assert got == ["suggest stabilising mutations"]


def test_qt_presenter_markup_and_active_site(_app):
    sig = PresenterSignals()
    out = []
    sig.append_html.connect(lambda h: out.append(h))
    pres = QtPresenter(sig)
    pres.markup("[bold]raw markup[/bold]")
    pres.active_site_ok("set [25]")
    import re as _re
    text = _re.sub(r"<[^>]+>", "", " ".join(out))
    assert "raw markup" in text and "set [25]" in text


# ── Audit remediation: preflight GPU/CPU + version guardrails, intent_registry URL ──

def test_ollama_gpu_status_detects_gpu(_app, monkeypatch):
    monkeypatch.setattr(gui_app.requests, "post", lambda *a, **k: MagicMock())
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": "qwen3:8b",
                                          "size_vram": 5_000_000_000, "size": 6_000_000_000}]}
    monkeypatch.setattr(gui_app.requests, "get", lambda *a, **k: resp)
    status, detail = gui_app.StructureBotWindow._ollama_gpu_status(None, "http://x")
    assert status == "gpu" and "VRAM" in detail


def test_ollama_gpu_status_detects_cpu(_app, monkeypatch):
    monkeypatch.setattr(gui_app.requests, "post", lambda *a, **k: MagicMock())
    resp = MagicMock()
    resp.json.return_value = {"models": [{"name": "qwen3:8b", "size_vram": 0, "size": 6_000_000_000}]}
    monkeypatch.setattr(gui_app.requests, "get", lambda *a, **k: resp)
    status, _ = gui_app.StructureBotWindow._ollama_gpu_status(None, "http://x")
    assert status == "cpu"                          # the original-bug class, now detected


def test_ollama_version_below_blackwell_floor(_app, monkeypatch):
    monkeypatch.setattr(gui_app.subprocess, "run",
                        lambda *a, **k: MagicMock(stdout="ollama version is 0.24.0"))
    ver, s = gui_app.StructureBotWindow._ollama_cli_version(None)
    assert ver == (0, 24, 0) and s == "0.24.0"
    assert ver < gui_app.StructureBotWindow._OLLAMA_MIN_VERSION   # would warn


def test_ollama_version_at_floor_ok(_app, monkeypatch):
    monkeypatch.setattr(gui_app.subprocess, "run",
                        lambda *a, **k: MagicMock(stdout="ollama version is 0.30.8"))
    ver, _ = gui_app.StructureBotWindow._ollama_cli_version(None)
    assert ver >= gui_app.StructureBotWindow._OLLAMA_MIN_VERSION


def test_intent_classifier_uses_config_base_url(_app, monkeypatch):
    """make_llm_classify_fn falls back to config.OLLAMA_BASE_URL (not a bare hardcode)."""
    import config, intent_registry
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(config, "OLLAMA_BASE_URL", "http://sentinel:9999")
    cap = {}
    def fake_post(url, json=None, timeout=None):
        cap["url"] = url
        r = MagicMock(); r.json.return_value = {"response": "none"}
        return r
    monkeypatch.setattr(intent_registry.requests if hasattr(intent_registry, "requests") else __import__("requests"),
                        "post", fake_post)
    fn = intent_registry.make_llm_classify_fn(backend_name="ollama")
    fn("do something", ["view.cartoon_only"])
    assert cap.get("url", "").startswith("http://sentinel:9999"), cap
