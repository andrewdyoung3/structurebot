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


# ── Stage 3b: handle_tool_request enters the SAME spine (no translate, real route/execute) ──

def test_handle_tool_request_enters_the_spine(_app):
    sig = PresenterSignals()
    out = []
    sig.append_html.connect(lambda h: out.append(h))
    sig.ask.connect(lambda kind, payload, q: q.put("proceed"))   # the confirm-gate → proceed
    pres = QtPresenter(sig)

    host = MagicMock()
    routed = {"commands": [], "explanations": [], "warnings": ["Deep tier — approximate runtime ~2 h"],
              "confidence": "low", "clarification_needed": None,
              "tools_needed": ["mutation_scan"], "has_extra_tools": True,
              "tool_inputs": {"mutation_scan": {"model_id": "1", "chain": "A",
                                                "scan_positions": [10, 25], "run_rosetta": True}}}
    host.router.route.return_value = routed
    executed = dict(routed)
    executed.update({"tool_step_results": [], "all_viz_commands": [], "all_viz_explanations": [],
                     "tool_summaries": {}, "pipeline_success": True, "pipeline_error": None})
    host.router.execute.return_value = executed

    eng = RequestEngine(host)
    eng.handle_tool_request(
        "mutation_scan",
        {"model_id": "1", "chain": "A", "scan_positions": [10, 25], "run_rosetta": True},
        "[Workbench] mutation scan on chain A — 2 selected position(s), deep tier",
        pres, confidence="low")

    # It built a routed-shaped dict and entered route() → execute() (the real spine) —
    # never a parallel invocation, and translate() (the LLM) was never touched.
    built = host.router.route.call_args.args[0]
    assert built["tools_needed"] == ["mutation_scan"]
    assert built["tool_inputs"]["mutation_scan"]["run_rosetta"] is True
    assert built["confidence"] == "low"
    host.router.execute.assert_called_once()
    host.translator.translate.assert_not_called()
    # the deep-tier estimate reached the pane before the gate
    import re as _re
    text = _re.sub(r"<[^>]+>", "", " ".join(out))
    assert "approximate runtime" in text.lower()


def test_handle_tool_request_on_result_seam(_app):
    # S4a: the executed result is handed to on_result (the variant ResultSlots capture).
    sig = PresenterSignals()
    sig.ask.connect(lambda kind, payload, q: q.put("proceed"))
    pres = QtPresenter(sig)

    host = MagicMock()
    routed = {"commands": [], "explanations": [], "warnings": [], "confidence": "high",
              "clarification_needed": None, "tools_needed": ["mutation_scan"],
              "has_extra_tools": True, "tool_inputs": {"mutation_scan": {"model_id": "1"}}}
    host.router.route.return_value = routed
    executed = dict(routed)
    executed.update({"tool_step_results": [{"tool": "mutation_scan",
                     "data": {"candidates": [{"resnum": 1, "to_aa": "W", "ddg": 2.0}]}}],
                     "all_viz_commands": [], "all_viz_explanations": [], "tool_summaries": {},
                     "pipeline_success": True, "pipeline_error": None})
    host.router.execute.return_value = executed

    captured = []
    RequestEngine(host).handle_tool_request(
        "mutation_scan", {"model_id": "1", "score_mutations": {1: "W"}},
        "[Workbench] stability", pres, on_result=captured.append)

    assert len(captured) == 1
    cands = captured[0]["tool_step_results"][0]["data"]["candidates"]
    assert cands[0]["resnum"] == 1 and cands[0]["ddg"] == 2.0


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


def test_preflight_restore_captures_design_mids(_app, monkeypatch):
    # restore offers + restores a design-only session, and records the model ids to re-display
    # on the UI thread (so the workbench rehydrates against the still-open ChimeraX models).
    from session_state import SessionState
    st = SessionState()
    st.add_design_session("1", {"model_id": "1", "chains": {}, "next_id": 1})
    monkeypatch.setattr(SessionState, "try_load",
                        staticmethod(lambda path: (st, None)))
    obj = types.SimpleNamespace(presenter=MagicMock(), bridge=MagicMock(),
                                session=SessionState(), router=None, workbench=MagicMock())
    obj._blocking_restore = lambda summary: "restore"
    obj._preflight_restore = types.MethodType(W._preflight_restore, obj)
    obj._preflight_restore()
    assert obj.session is st                                   # the design-only session restored
    assert obj._restore_mids == ["1"]                         # its model id queued for re-display
    obj.workbench.attach_session.assert_called_once_with(st)   # panel re-pointed at the new session


def test_bridge_on_structure_opened_fires(_app, monkeypatch):
    from chimerax_bridge import ChimeraXBridge
    b = ChimeraXBridge.__new__(ChimeraXBridge)
    b._lean_layout_applied = True
    got = []
    b.on_structure_opened = lambda mid: got.append(mid)
    monkeypatch.setattr(b, "run_command", lambda c, timeout=30: {"value": "opened #3", "error": None})
    monkeypatch.setattr(b, "_maybe_apply_presentation_on_open", lambda: None)
    monkeypatch.setattr(b, "_maybe_apply_lean_layout", lambda: None)
    b.run_commands(["open foo.pdb"])
    assert got == ["3"]                                       # the REAL opened id, from the bridge


def test_opened_mid_focus_uses_ground_truth(_app):
    shown = []
    w = types.SimpleNamespace(_opened_mids=["3"], _pending_focus=["1"])
    w._finish_request = lambda: None
    w.show_model = lambda mid: shown.append(mid)
    w._isolate_foreign_models = lambda: None      # opened a model → isolate fires (stubbed here)
    w._ingest_new_assembly = lambda: None
    W._on_request_done(w)
    assert shown == ["3"]                                     # real id wins over next_model_id guess


def test_opened_mid_focus_falls_back_to_guess(_app):
    shown = []
    w = types.SimpleNamespace(_opened_mids=[], _pending_focus=["1"])
    w._finish_request = lambda: None
    w.show_model = lambda mid: shown.append(mid)
    w._ingest_new_assembly = lambda: None
    W._on_request_done(w)
    assert shown == ["1"]                                     # fallback when bridge saw no open


# ── PART A: a bio_assembly build → ingest the flat assembly as the active design ──

def _ingest_fake(session):
    """A stand-in carrying the assembly-ingest methods bound to a real SessionState."""
    w = types.SimpleNamespace(session=session, _assembly_snapshot={})
    w.show_model = None     # set by the caller
    for name in ("_assembly_model_ids", "_ingest_new_assembly"):
        setattr(w, name, types.MethodType(getattr(W, name), w))
    return w


def test_ingest_new_assembly_makes_it_active(_app):
    from session_state import SessionState
    s = SessionState()
    shown = []
    w = _ingest_fake(s)
    w.show_model = lambda mid: shown.append(mid)
    w._assembly_snapshot = w._assembly_model_ids()           # snapshot BEFORE the build (empty)
    # a bio_assembly build lands a flat assembly model #3 for AU #1
    s.set_generated_assembly("1", {"assembly_model_id": "3", "assembly_id": 1,
                                   "assembly_type": "Homotrimer"})
    w._ingest_new_assembly()
    assert shown == ["3"]                                    # the assembly becomes the active design


def test_ingest_noop_when_nothing_built(_app):
    from session_state import SessionState
    s = SessionState()
    shown = []
    w = _ingest_fake(s)
    w.show_model = lambda mid: shown.append(mid)
    w._assembly_snapshot = w._assembly_model_ids()
    w._ingest_new_assembly()                                 # a plain (non-assembly) request
    assert shown == []


def test_ingest_does_not_refire_for_preexisting_assembly(_app):
    from session_state import SessionState
    s = SessionState()
    s.set_generated_assembly("1", {"assembly_model_id": "3", "assembly_id": 1})
    shown = []
    w = _ingest_fake(s)
    w.show_model = lambda mid: shown.append(mid)
    w._assembly_snapshot = w._assembly_model_ids()           # assembly already present (a PRIOR request)
    w._ingest_new_assembly()
    assert shown == []                                       # unchanged → not re-ingested


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


# ── Session menu: named save / load / clear (wiring over session_io) ───────────────
from unittest.mock import MagicMock as _MM  # noqa: E402
from session_state import SessionState as _SS  # noqa: E402


def _fakew(**attrs):
    """Bind the session-menu methods to a non-QWidget stand-in (avoids the full QMainWindow)."""
    obj = types.SimpleNamespace(**attrs)
    for name in ("_on_load_session", "_on_clear_session", "_on_save_session",
                 "_redisplay_designs", "_reset_view_for_session", "show_model"):
        setattr(obj, name, types.MethodType(getattr(W, name), obj))
    return obj


def test_load_session_failloud_does_not_swap(_app, monkeypatch):
    import session_io
    orig = _SS()
    monkeypatch.setattr(session_io, "list_saved_sessions", lambda: ["bad"])
    monkeypatch.setattr(session_io, "load_named_session",
                        lambda b, n: {"name": "bad", "state": None, "error": "corrupt JSON",
                                      "cxs_ok": False, "cxs_error": None})
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *a, **k: ("bad", True))
    w = _fakew(bridge=_MM(), session=orig, router=_MM(), workbench=_MM(), presenter=_MM(),
               tabs=QtWidgets.QTabWidget(), _grids={}, statusBar=lambda: _MM())
    w._on_load_session()
    assert w.session is orig                       # FAIL-LOUD: never swapped to a fresh/loaded state
    w.presenter.error.assert_called_once()
    w.workbench.attach_session.assert_not_called()


def test_load_session_replaces_and_redisplays_denovo(_app, monkeypatch):
    import session_io
    new = _SS()
    new.add_design_session("denovo-1", {"model_id": "denovo-1", "source": "sequence",
                                        "chains": {}, "next_id": 1})
    monkeypatch.setattr(session_io, "list_saved_sessions", lambda: ["expt"])
    monkeypatch.setattr(session_io, "load_named_session",
                        lambda b, n: {"name": "expt", "state": new, "error": None,
                                      "cxs_ok": True, "cxs_error": None})
    monkeypatch.setattr(QtWidgets.QInputDialog, "getItem", lambda *a, **k: ("expt", True))
    monkeypatch.setattr(gui_app, "ToolRouter", lambda *a, **k: _MM())
    wb = _MM()
    w = _fakew(bridge=_MM(), session=_SS(), router=_MM(), workbench=wb, presenter=_MM(),
               tabs=QtWidgets.QTabWidget(), _grids={}, statusBar=lambda: _MM())
    w._on_load_session()
    assert w.session is new                         # REPLACE
    wb.attach_session.assert_called_with(new)
    wb.rehydrate_denovo.assert_called_once()        # de-novo design re-displayed via the contract


def test_clear_session_resets_state_and_view(_app, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(gui_app, "ToolRouter", lambda *a, **k: _MM())
    orig = _SS(); wb = _MM()
    w = _fakew(bridge=_MM(), session=orig, router=_MM(), workbench=wb, presenter=_MM(),
               tabs=QtWidgets.QTabWidget(), _grids={"x": 1})
    w._on_clear_session()
    assert w.session is not orig and isinstance(w.session, _SS)   # fresh state
    wb.attach_session.assert_called_once()
    wb.reset.assert_called_once()
    assert w._grids == {}                            # view reset


def test_save_session_reports_success(_app, monkeypatch):
    import session_io
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("my-expt", True))
    monkeypatch.setattr(session_io, "save_named_session",
                        lambda b, s, n: {"name": "my-expt", "dir": "/x/my-expt",
                                         "cxs_ok": True, "cxs_error": None, "json_error": None})
    w = _fakew(bridge=_MM(), session=_SS(), presenter=_MM())
    w._on_save_session()
    w.presenter.success.assert_called_once()


def test_reset_view_keeps_workbench_tab(_app):
    """REGRESSION: the Variant Workbench is a tab INSIDE self.tabs; _reset_view_for_session must
    drop the chain-grid tabs but KEEP the workbench tab (a blanket clear() destroyed the panel +
    its toolbar, leaving no way to start a new session after Clear/Load)."""
    tabs = QtWidgets.QTabWidget()
    wb = QtWidgets.QWidget()
    reset_calls = {"n": 0}
    wb.reset = lambda: reset_calls.__setitem__("n", reset_calls["n"] + 1)
    tabs.addTab(wb, "Variant Workbench")
    grid = QtWidgets.QWidget()
    tabs.addTab(grid, "#1/A")
    w = _fakew(tabs=tabs, workbench=wb, _grids={("1", "A"): grid})
    w._reset_view_for_session()
    widgets = [tabs.widget(i) for i in range(tabs.count())]
    assert wb in widgets and grid not in widgets   # workbench (toolbar) kept; grid dropped
    assert w._grids == {} and reset_calls["n"] == 1


def test_reset_view_keeps_and_clears_disulfides_tab(_app):
    """REGRESSION (layer-spanning): the persistent Disulfides tab is a SIBLING of the workbench in
    self.tabs — `_reset_view_for_session` must KEEP it (it was being swept out with the chain grids
    and never re-added, so any Clear/Load/reconnect dropped it permanently) AND CLEAR it (a Load
    brings a new session; the prior session's pairs reference a construct no longer loaded → stale).
    Spans the gui_app reset layer + the REAL panel tab — the seam neither the panel persistence test
    nor `test_reset_view_keeps_workbench_tab` (no disulfides_tab in its harness) crossed."""
    from variant_workbench import DisulfidesResultsTab
    tabs = QtWidgets.QTabWidget()
    wb = QtWidgets.QWidget(); wb.reset = lambda: None
    dtab = DisulfidesResultsTab(on_highlight=lambda *a, **k: None, on_declare=lambda *a, **k: None)
    wb.disulfides_tab = dtab
    grid = QtWidgets.QWidget()
    tabs.addTab(wb, "Variant Workbench")
    tabs.addTab(dtab, "Disulfides")                # the sibling, exactly as gui_app.__init__ adds it
    tabs.addTab(grid, "#1/A")
    dtab.populate_scan(MagicMock(), {"pairs": [
        {"chain_a": "A", "resnum_a": 5, "chain_b": "A", "resnum_b": 9, "score": 0.9}],
        "best_partner": {}, "caveat": "x"})
    assert dtab._sec["D"]["table"].rowCount() == 1            # populated BEFORE the reset
    w = _fakew(tabs=tabs, workbench=wb, _grids={("1", "A"): grid})
    w._reset_view_for_session()
    widgets = [tabs.widget(i) for i in range(tabs.count())]
    assert dtab in widgets and grid not in widgets           # the tab SURVIVES; only the grid drops
    assert dtab._sec["D"]["table"].rowCount() == 0           # …but CLEARED (no stale pairs)
    assert not dtab._sec["D"]["placeholder"].isHidden()      # back to the dormant placeholder


def test_reset_view_keeps_and_clears_proline_tab(_app):
    """The Proline tab is a SIBLING peer of the Disulfides tab — same layer-spanning contract:
    `_reset_view_for_session` must KEEP it (not sweep it out with the chain grids) AND CLEAR it
    (a Load brings a new session; the prior candidates reference a construct no longer loaded)."""
    from variant_workbench import DisulfidesResultsTab, ProlineResultsTab
    tabs = QtWidgets.QTabWidget()
    wb = QtWidgets.QWidget(); wb.reset = lambda: None
    dtab = DisulfidesResultsTab(on_highlight=lambda *a, **k: None, on_declare=lambda *a, **k: None)
    ptab = ProlineResultsTab(on_highlight=lambda *a, **k: None)
    wb.disulfides_tab = dtab
    wb.proline_tab = ptab
    grid = QtWidgets.QWidget()
    tabs.addTab(wb, "Variant Workbench")
    tabs.addTab(dtab, "Disulfides")
    tabs.addTab(ptab, "Proline")
    tabs.addTab(grid, "#1/A")
    ptab.populate(MagicMock(), {"candidates": [
        {"chain": "A", "position": 5, "from_aa": "W", "phi": -63.0, "psi": -35.0,
         "score": 0.99, "hbond_donates": False}], "existing": [], "caveat": "x"})
    assert ptab._tbl.rowCount() == 1                          # populated BEFORE the reset
    w = _fakew(tabs=tabs, workbench=wb, _grids={("1", "A"): grid})
    w._reset_view_for_session()
    widgets = [tabs.widget(i) for i in range(tabs.count())]
    assert ptab in widgets and grid not in widgets           # SURVIVES; only the grid drops
    assert ptab._tbl.rowCount() == 0                         # …but CLEARED (no stale candidates)
    assert not ptab._placeholder.isHidden()                 # back to the dormant placeholder


def test_reset_view_clears_the_design_basket(_app):
    """The Design basket (a QDockWidget, owned by the panel) must be EMPTIED on session reset —
    its staged picks reference the prior session's design (keep-and-clear, layer-spanning through
    gui_app's `_reset_view_for_session`)."""
    from variant_workbench import DesignBasketPanel
    tabs = QtWidgets.QTabWidget()
    wb = QtWidgets.QWidget()
    basket = DesignBasketPanel()
    wb.design_basket = basket
    wb.reset = lambda: None                                  # isolate the gui_app reset path
    grid = QtWidgets.QWidget()
    tabs.addTab(wb, "Variant Workbench")
    tabs.addTab(grid, "#1/A")
    basket.add_entry({"cls": "Proline", "score": 0.9,
                      "subs": [{"chain": "A", "position": 3, "from_aa": "V", "to_aa": "P"}],
                      "metrics_text": "x"})
    assert basket.entries                                    # staged BEFORE the reset
    w = _fakew(tabs=tabs, workbench=wb, _grids={("1", "A"): grid})
    w._reset_view_for_session()
    assert basket.entries == []                              # …emptied (keep-and-clear)


# ── Reconnect ChimeraX: model-id remap + reopen/relink flow ───────────────────────
def test_remap_session_model_ids_crystal_and_denovo(_app):
    """remap rewrites structures keys, crystal design keys, AND internal fold refs of a de-novo
    design (whose synthetic key is untouched but its members/template_fold ids move)."""
    from gui_app import remap_session_model_ids
    s = _SS()
    s.structures = {"1": {"name": "2hhb"}, "3": {"name": "ubq"}}
    s.add_design_session("1", {"model_id": "1", "source": "structure",
                               "chains": {"k": {"rep_model": "1", "members": [["1", "A"]],
                                                "template_fold": {}, "guided_fold": {}}}})
    s.add_design_session("denovo-x", {"model_id": "denovo-x", "source": "sequence",
        "chains": {"c": {"rep_model": "3", "members": [["3", "A"], ["3", "B"]],
                         "template_fold": {"model_id": "3"}, "guided_fold": {"model_id": "3"},
                         "wt_refs": {"boltz:assembly": {"model_id": "3"}}}}})
    remap_session_model_ids(s, {"1": "5", "3": "7"})
    assert set(s.structures) == {"5", "7"}                       # structures rekeyed
    assert "5" in s.design_sessions and "1" not in s.design_sessions  # crystal key moved
    assert s.design_sessions["5"]["chains"]["k"]["members"] == [["5", "A"]]
    dn = s.design_sessions["denovo-x"]["chains"]["c"]             # de-novo key UNCHANGED
    assert dn["rep_model"] == "7" and dn["members"] == [["7", "A"], ["7", "B"]]
    assert dn["template_fold"]["model_id"] == "7" and dn["guided_fold"]["model_id"] == "7"
    assert dn["wt_refs"]["boltz:assembly"]["model_id"] == "7"


def test_remap_noop_when_ids_unchanged(_app):
    from gui_app import remap_session_model_ids
    s = _SS(); s.structures = {"1": {"name": "x"}}
    remap_session_model_ids(s, {"1": "1"})                        # self-map → no change
    assert set(s.structures) == {"1"}


def test_remap_rewrites_structural_align_reference(_app):
    """A reconnect (fresh ids) must move the alignment reference/query ids too, or the
    alignment-visibility toggle would target a stale id."""
    from gui_app import remap_session_model_ids
    s = _SS()
    s.structures = {"9": {"name": "1mbn", "metadata": {"aligned_reference": True}}}
    s.add_design_session("denovo-y", {"model_id": "denovo-y", "source": "sequence",
        "chains": {"c": {"rep_model": "7", "members": [["7", "A"]], "template_fold": {"model_id": "7"},
                         "structural_align": {"reference_model_id": "9", "query_model_id": "7"}}}})
    remap_session_model_ids(s, {"9": "11", "7": "8"})
    sa = s.design_sessions["denovo-y"]["chains"]["c"]["structural_align"]
    assert sa["reference_model_id"] == "11" and sa["query_model_id"] == "8"
    assert set(s.structures) == {"11"}                            # the reference structure rekeyed


def test_reconnect_clicked_noop_without_structures(_app):
    w = _fakew(session=_SS(), presenter=_MM(),
               reconnect_action=_MM(), statusBar=lambda: _MM(), _pool=_MM())
    # bind the reconnect handler too
    w._on_reconnect_clicked = types.MethodType(W._on_reconnect_clicked, w)
    w._on_reconnect_clicked()
    w.presenter.dim.assert_called_once()                         # "nothing to re-open"
    w._pool.start.assert_not_called()


def test_reconnect_done_applies_remap_and_relinks(_app, monkeypatch):
    from session_state import SessionState
    s = SessionState()
    s.structures = {"1": {"name": "2hhb"}}
    s.add_design_session("1", {"model_id": "1", "source": "structure",
                               "chains": {"k": {"rep_model": "1", "members": [["1", "A"]]}}})
    shown = []
    w = _fakew(session=s, workbench=_MM(), presenter=_MM(), reconnect_action=_MM(),
               statusBar=lambda: _MM(), tabs=QtWidgets.QTabWidget(), _grids={})
    for name in ("_do_reconnect", "_on_reconnect_done", "_denovo_fold_ids",
                 "_redisplay_designs", "_reset_view_for_session"):
        setattr(w, name, types.MethodType(getattr(W, name), w))
    w.show_model = lambda mid: shown.append(mid)
    monkeypatch.setattr(gui_app, "ToolRouter", lambda *a, **k: _MM())
    w.bridge = _MM(); w.bridge._maybe_apply_lean_layout = lambda: None
    w._on_reconnect_done({"remap": {"1": "4"}, "reopened": [("2hhb", "1", "4")],
                          "errors": [], "outcome": "started"})
    assert set(s.structures) == {"4"}                            # remapped
    assert "4" in s.design_sessions                              # crystal design re-keyed
    assert shown == ["4"]                                        # re-shown at the NEW id
    w.presenter.success.assert_called_once()


def test_resolve_reopen_target_prefers_path_then_folds_then_pdbid(_app, monkeypatch, tmp_path):
    import session_io
    w = _fakew(session=_SS())
    w._resolve_reopen_target = types.MethodType(W._resolve_reopen_target, w)
    # 1) existing path wins
    real = tmp_path / "live.cif"; real.write_text("x")
    assert w._resolve_reopen_target({"path": str(real), "name": "x"}) == str(real)
    # 2) path gone -> durable folds/ copy (the FALLBACK)
    monkeypatch.setattr(session_io, "find_fold_copy",
                        lambda b: "/sessions/expt/folds/boltz_pred_z.cif" if b == "boltz_pred_z.cif" else None)
    got = w._resolve_reopen_target({"path": str(tmp_path / "boltz_pred_z.cif"), "name": "denovo"})
    assert got == "/sessions/expt/folds/boltz_pred_z.cif"
    # 3) no path, 4-char name -> PDB id fetch
    assert w._resolve_reopen_target({"name": "2hhb"}) == "2hhb"
    # 4) nothing re-openable -> None
    monkeypatch.setattr(session_io, "find_fold_copy", lambda b: None)
    assert w._resolve_reopen_target({"path": str(tmp_path / "gone.cif"), "name": "denovo-x"}) is None


# ── Session ▸ Export results… + Save As… ──────────────────────────────────────────
def _fakew_session(**attrs):
    obj = types.SimpleNamespace(**attrs)
    for name in ("_on_export_results", "_on_save_as_session", "_on_save_session"):
        setattr(obj, name, types.MethodType(getattr(W, name), obj))
    return obj


def test_export_requires_named_save_first(_app, monkeypatch):
    import session_export
    called = {"export": 0}
    monkeypatch.setattr(session_export, "export_session",
                        lambda ds, d: called.__setitem__("export", called["export"] + 1) or
                        {"any": True, "written": ["Solubility"], "skipped": [], "files": []})
    # user declines the "save first" prompt → no export
    monkeypatch.setattr(QtWidgets.QMessageBox, "question",
                        lambda *a, **k: QtWidgets.QMessageBox.StandardButton.No)
    w = _fakew_session(session=_SS(), presenter=_MM(), _current_session_name=None)
    w._on_export_results()
    assert called["export"] == 0
    w.presenter.success.assert_not_called()


def test_export_writes_when_named(_app, monkeypatch, tmp_path):
    import session_export, session_io
    monkeypatch.setattr(session_io, "session_paths", lambda n: {"exports": tmp_path / "exports"})
    monkeypatch.setattr(session_export, "export_session",
                        lambda ds, d, ppr=None: {"any": True, "written": ["Deviation", "Solubility"],
                                       "skipped": ["Stability ddG"], "files": []})
    w = _fakew_session(session=_SS(), presenter=_MM(), _current_session_name="expt")
    w._on_export_results()
    w.presenter.success.assert_called_once()


def test_export_nothing_to_export(_app, monkeypatch, tmp_path):
    import session_export, session_io
    monkeypatch.setattr(session_io, "session_paths", lambda n: {"exports": tmp_path})
    monkeypatch.setattr(session_export, "export_session",
                        lambda ds, d, ppr=None: {"any": False, "written": [], "skipped": [], "files": []})
    w = _fakew_session(session=_SS(), presenter=_MM(), _current_session_name="expt")
    w._on_export_results()
    w.presenter.dim.assert_called_once()
    w.presenter.success.assert_not_called()


def test_save_as_switches_current_and_reports(_app, monkeypatch):
    import session_io
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("fork", True))
    monkeypatch.setattr(session_io, "save_as_session",
                        lambda b, s, n, src_name=None: {"error": None, "name": "fork",
                                                        "dir": "/x/fork", "copied_from": "/x/orig"})
    w = _fakew_session(bridge=_MM(), session=_SS(), presenter=_MM(), _current_session_name="orig")
    w._on_save_as_session()
    assert w._current_session_name == "fork"           # live session SWITCHED to the fork
    w.presenter.success.assert_called_once()


def test_save_as_error_keeps_current(_app, monkeypatch):
    import session_io
    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", lambda *a, **k: ("dup", True))
    monkeypatch.setattr(session_io, "save_as_session",
                        lambda b, s, n, src_name=None: {"error": "already exists", "name": "dup"})
    w = _fakew_session(bridge=_MM(), session=_SS(), presenter=_MM(), _current_session_name="orig")
    w._on_save_as_session()
    assert w._current_session_name == "orig"           # unchanged on error
    w.presenter.error.assert_called_once()


# ── multi-window isolation: hide another window's models in a shared ChimeraX ──────
class _IsoBridge:
    def __init__(self, visible):
        self._visible = visible
        self.commands = []
    def visible_model_ids(self):
        return list(self._visible)
    def run_command(self, cmd, timeout=None):
        self.commands.append(cmd)
        return {"value": "", "error": None}


def _iso_win(**attrs):
    obj = types.SimpleNamespace(**attrs)
    obj._isolate_foreign_models = types.MethodType(W._isolate_foreign_models, obj)
    return obj


def test_isolate_hides_foreign_models_still_visible(_app):
    br = _IsoBridge(visible=["1", "2"])                 # #1 = another window's model, #2 = mine
    w = _iso_win(bridge=br, _foreign_mids={"1"})
    w._isolate_foreign_models()
    assert br.commands == ["hide #1 models"]            # the foreign model is hidden, mine untouched


def test_isolate_noop_when_foreign_already_hidden(_app):
    br = _IsoBridge(visible=["2", "3"])                 # the foreign #1 is no longer visible
    w = _iso_win(bridge=br, _foreign_mids={"1"})
    w._isolate_foreign_models()
    assert br.commands == []


def test_isolate_noop_with_no_foreign_baseline(_app):
    br = _IsoBridge(visible=["1", "2"])
    w = _iso_win(bridge=br, _foreign_mids=set())        # fresh ChimeraX at startup → nothing foreign
    w._isolate_foreign_models()
    assert br.commands == []


def test_isolate_hides_multiple_foreign_sorted(_app):
    br = _IsoBridge(visible=["1", "2", "10"])
    w = _iso_win(bridge=br, _foreign_mids={"10", "1"})  # mine = #2
    w._isolate_foreign_models()
    assert br.commands == ["hide #1,10 models"]         # numeric sort, not lexical
