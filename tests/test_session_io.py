"""
tests/test_session_io.py
------------------------
The shared named-session core (session_io) used by BOTH front-ends. Covers the directory
layout (SESSION_DIR/{name}/ with session.json + scene.cxs + folds/ + exports/), sanitise,
list, save (all parts incl. durable fold copies), and the FAIL-LOUD load contract (missing /
corrupt session.json never yields a silent fresh state).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import session_io
from session_state import SessionState


class _FakeBridge:
    """Records `save`/`open` commands; reports success unless told to fail."""
    def __init__(self, fail=False):
        self.fail = fail
        self.commands = []

    def run_command(self, cmd, timeout=None):
        self.commands.append(cmd)
        return {"error": "boom" if self.fail else None, "value": ""}


@pytest.fixture
def session_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSION_DIR", tmp_path)
    return tmp_path


def test_sanitize_and_paths(session_dir):
    assert session_io.sanitize_session_name("my run!/2") == "my_run__2"
    assert session_io.sanitize_session_name("   ") == "default"
    p = session_io.session_paths("a b")
    assert p["dir"].name == "a_b"
    assert p["json"].name == "session.json" and p["cxs"].name == "scene.cxs"
    assert p["folds"].name == "folds" and p["exports"].name == "exports"


def test_save_writes_dir_layout_and_records_scene(session_dir):
    s = SessionState()
    s.add_design_session("denovo-1", {"model_id": "denovo-1", "chains": {}, "next_id": 1})
    br = _FakeBridge()
    info = session_io.save_named_session(br, s, "expt-A")
    d = session_dir / "expt-A"
    assert info["cxs_ok"] and info["json_error"] is None
    assert (d / "session.json").is_file()
    assert (d / "folds").is_dir() and (d / "exports").is_dir()
    assert any('save "' in c and c.endswith('scene.cxs"') for c in br.commands)


def test_save_state_still_written_when_scene_fails(session_dir):
    s = SessionState()
    info = session_io.save_named_session(_FakeBridge(fail=True), s, "x")
    assert not info["cxs_ok"] and info["cxs_error"] == "boom"
    assert (session_dir / "x" / "session.json").is_file()   # state always written


def test_save_copies_fold_cifs_and_rewrites_paths_durably(session_dir, tmp_path):
    """The durability win: a fold CIF in a (volatile) temp location is COPIED into folds/ and the
    saved session.json points at the durable copy, NOT the original temp path."""
    volatile = tmp_path / "boltz_pred_xyz.cif"
    volatile.write_text("data_x\n# fake fold\n")
    s = SessionState()
    s.add_design_session("denovo-1", {
        "model_id": "denovo-1", "next_id": 1, "source": "sequence",
        "chains": {"hivca|denovo-1/A": {
            "group_key": "hivca", "rep_model": "denovo-1", "rep_chain": "A",
            "members": [["denovo-1", "A"]], "template_cells": [], "variants": [],
            "guided_fold": {"model_id": "denovo-1", "cif_path": str(volatile),
                            "template_label": "8UB2"},
        }},
    })
    session_io.save_named_session(_FakeBridge(), s, "dn")

    # The LIVE session still points at the original temp path (not mutated).
    live_cif = s.get_design_session("denovo-1")["chains"]["hivca|denovo-1/A"]["guided_fold"]["cif_path"]
    assert live_cif == str(volatile)

    # The SAVED json points at the durable copy under folds/, and the file exists.
    saved = SessionState.load(str(session_dir / "dn" / "session.json"))
    saved_cif = saved.get_design_session("denovo-1")["chains"]["hivca|denovo-1/A"]["guided_fold"]["cif_path"]
    assert Path(saved_cif).parent == (session_dir / "dn" / "folds")
    assert Path(saved_cif).is_file()
    # Survives deletion of the original temp file (the whole point).
    volatile.unlink()
    assert Path(saved_cif).is_file()


def test_folds_rebuilt_each_save_no_stale(session_dir, tmp_path):
    cif = tmp_path / "boltz_pred_a.cif"; cif.write_text("x")
    s = SessionState()
    s.add_design_session("d", {"model_id": "d", "next_id": 1, "chains": {"k": {
        "group_key": "g", "rep_model": "d", "rep_chain": "A", "members": [["d", "A"]],
        "template_cells": [], "variants": [],
        "template_fold": {"cif_path": str(cif)}}}})
    session_io.save_named_session(_FakeBridge(), s, "r")
    folds = session_dir / "r" / "folds"
    assert len(list(folds.glob("*.cif"))) == 1
    session_io.save_named_session(_FakeBridge(), s, "r")     # re-save → folds rebuilt, not doubled
    assert len(list(folds.glob("*.cif"))) == 1


def test_list_saved_sessions_keys_off_dir_with_json(session_dir):
    (session_dir / "alpha").mkdir(); (session_dir / "alpha" / "session.json").write_text("{}")
    (session_dir / "beta").mkdir();  (session_dir / "beta" / "session.json").write_text("{}")
    (session_dir / "empty").mkdir()                          # no session.json → not listed
    assert session_io.list_saved_sessions() == ["alpha", "beta"]


def test_load_missing_is_failloud(session_dir):
    info = session_io.load_named_session(_FakeBridge(), "nope")
    assert info["state"] is None and "not found" in info["error"]


def test_load_corrupt_json_is_failloud_never_fresh(session_dir):
    (session_dir / "bad").mkdir()
    (session_dir / "bad" / "session.json").write_text("{ this is not json")
    info = session_io.load_named_session(_FakeBridge(), "bad")
    assert info["state"] is None and info["error"]          # FAIL-LOUD, no silent fresh state


def test_load_roundtrip_reopens_scene(session_dir):
    s = SessionState()
    s.add_design_session("denovo-1", {"model_id": "denovo-1", "chains": {}, "next_id": 1})
    session_io.save_named_session(_FakeBridge(), s, "good")
    (session_dir / "good" / "scene.cxs").write_text("scene")  # pretend ChimeraX wrote a scene
    br2 = _FakeBridge()
    info = session_io.load_named_session(br2, "good")
    assert info["error"] is None and info["state"] is not None and info["cxs_ok"]
    assert any(c.startswith('open "') and c.endswith('scene.cxs"') for c in br2.commands)
    assert info["state"].get_design_session("denovo-1") is not None


def test_load_state_only_when_no_cxs(session_dir):
    s = SessionState()
    session_io.save_named_session(_FakeBridge(fail=True), s, "stateonly")  # scene save failed → no .cxs
    info = session_io.load_named_session(_FakeBridge(), "stateonly")
    assert info["state"] is not None and info["error"] is None
    assert not info["cxs_ok"] and "no scene.cxs" in info["cxs_error"]


def test_find_fold_copy_by_basename_and_suffix(session_dir):
    (session_dir / "expt" / "folds").mkdir(parents=True)
    (session_dir / "expt" / "folds" / "boltz_pred_abc.cif").write_text("x")
    (session_dir / "other" / "folds").mkdir(parents=True)
    (session_dir / "other" / "folds" / "boltz_pred_def__1.cif").write_text("y")
    # exact basename match (across sessions)
    got = session_io.find_fold_copy("boltz_pred_abc.cif")
    assert got and Path(got).name == "boltz_pred_abc.cif" and Path(got).parent.parent.name == "expt"
    # collision-suffixed variant matched by stem
    got2 = session_io.find_fold_copy("boltz_pred_def.cif")
    assert got2 and Path(got2).name == "boltz_pred_def__1.cif"
    # absent → None
    assert session_io.find_fold_copy("nope.cif") is None


def _session_with_fold(tmp_path):
    cif = tmp_path / "boltz_pred_q.cif"; cif.write_text("data_x\n")
    s = SessionState()
    s.add_design_session("denovo-1", {"model_id": "denovo-1", "source": "sequence",
        "chains": {"c": {"group_key": "g", "rep_model": "denovo-1", "rep_chain": "A",
                         "members": [["denovo-1", "A"]], "template_cells": [], "variants": [],
                         "template_fold": {"cif_path": str(cif)}}}})
    return s, cif


def test_save_as_forks_self_contained(session_dir, tmp_path):
    s, cif = _session_with_fold(tmp_path)
    session_io.save_named_session(_FakeBridge(), s, "orig")
    (session_dir / "orig" / "exports").mkdir(exist_ok=True)
    (session_dir / "orig" / "exports" / "results.xlsx").write_text("xlsx")  # a prior export

    info = session_io.save_as_session(_FakeBridge(), s, "fork", src_name="orig")
    assert info["error"] is None
    fork = session_dir / "fork"
    # inherited durable artifacts travel
    assert (fork / "exports" / "results.xlsx").is_file()
    assert list((fork / "folds").glob("*.cif"))                       # fold copy present
    # the fork's session.json fold path points INTO the fork (self-contained), not orig/ or temp
    forked = SessionState.load(str(fork / "session.json"))
    p = forked.get_design_session("denovo-1")["chains"]["c"]["template_fold"]["cif_path"]
    assert Path(p).parent == (fork / "folds") and Path(p).is_file()
    # original is untouched; live session unmutated
    assert (session_dir / "orig" / "session.json").is_file()
    assert s.get_design_session("denovo-1")["chains"]["c"]["template_fold"]["cif_path"] == str(cif)


def test_save_as_fork_survives_temp_fold_deletion(session_dir, tmp_path):
    s, cif = _session_with_fold(tmp_path)
    session_io.save_named_session(_FakeBridge(), s, "orig")
    cif.unlink()                                                      # the volatile temp fold is GONE
    info = session_io.save_as_session(_FakeBridge(), s, "fork2", src_name="orig")
    assert info["error"] is None
    forked = SessionState.load(str(session_dir / "fork2" / "session.json"))
    p = forked.get_design_session("denovo-1")["chains"]["c"]["template_fold"]["cif_path"]
    assert Path(p).parent == (session_dir / "fork2" / "folds") and Path(p).is_file()  # from inherited copy


def test_save_as_refuses_existing_name(session_dir, tmp_path):
    s, _ = _session_with_fold(tmp_path)
    session_io.save_named_session(_FakeBridge(), s, "taken")
    info = session_io.save_as_session(_FakeBridge(), s, "taken", src_name=None)
    assert info["error"] and "already exists" in info["error"]
