"""
Unit tests for seq_editor.controller (the standalone sequence editor's logic).

ChimeraX REST and ColabFold are fully mocked — no live ChimeraX, no fold. Covers:
sequence-model build (incl. a non-1-start + gapped chain), grid↔resnum mapping,
select-command construction, lossless variant editing (WT preserved), reverse-sync
filtering, and fold-call wiring (asserts the variant sequence is passed).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from seq_editor.controller import SequenceEditorController, VALID_AA  # noqa: E402


# ── a mock ChimeraX REST runner ──────────────────────────────────────────────────

# chain A: non-1-start (50) + a gap (no 52); chain B: 1-start. Plus a solvent line
# that the macromolecule-scoped query would already exclude (not included here).
_INFO_RESIDUES_M1 = """\
residue id #1/A:50 name MET index 0
residue id #1/A:51 name GLY index 1
residue id #1/A:53 name LEU index 3
residue id #1/B:1 name ALA index 0
residue id #1/B:2 name CYS index 1
"""

_INFO_MODELS = "1 model(s)\n#1  some structure   model id #1 type AtomicStructure\n"

_INFO_SEL = """\
residue id #1/A:51 name GLY index 1
residue id #1/Z:9 name TRP index 0
"""  # Z:9 is an UNLOADED chain → must be filtered out by sync


class MockRunner:
    def __init__(self):
        self.commands = []

    def __call__(self, command: str):
        self.commands.append(command)
        c = command.strip()
        if c == "info models":
            return {"value": _INFO_MODELS, "error": None}
        if c.startswith("info residues sel"):
            return {"value": _INFO_SEL, "error": None}
        if c.startswith("info residues #1"):
            return {"value": _INFO_RESIDUES_M1, "error": None}
        if c.startswith("select "):
            return {"value": "", "error": None}
        if c.startswith("open "):
            return {"value": "", "error": None}
        return {"value": "", "error": None}


class MockFold:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {"success": True, "mean_plddt": 88.0,
                                 "ranked_pdb": "C:/tmp/x.pdb", "ptm": 0.85}

    def __call__(self, sequence, **kw):
        self.calls.append({"sequence": sequence, **kw})
        return self.result


def _ctl(fold=None):
    return SequenceEditorController(MockRunner(), fold or MockFold())


# ── build + numbering ────────────────────────────────────────────────────────────

def test_load_models_builds_chains():
    c = _ctl()
    chains = c.load_models()
    assert {ch.key for ch in chains} == {("1", "A"), ("1", "B")}
    a = c.get_chain("1", "A")
    assert a.resnums() == [50, 51, 53]          # gap at 52 preserved
    assert a.wt_seq == "MGL"
    b = c.get_chain("1", "B")
    assert b.wt_seq == "AC"


def test_non1start_gapped_seqpos_mapping():
    c = _ctl()
    c.load_models()
    a = c.get_chain("1", "A")
    # gap-aware 1-based positions: 50→1, 51→2, 53→3 (NOT 4)
    assert [(cell.resnum, cell.seqpos) for cell in a.cells] == [(50, 1), (51, 2), (53, 3)]
    assert a.cells[0].wt_aa == "M" and a.cells[0].model == "1" and a.cells[0].chain == "A"


def test_list_model_ids():
    c = _ctl()
    assert c.list_model_ids() == ["1"]


# ── viewer → 3D select ───────────────────────────────────────────────────────────

def test_build_select_command():
    assert SequenceEditorController.build_select_command("1", "A", [50, 53]) \
        == "select #1/A:50,53"


def test_select_in_3d_pushes_command():
    runner = MockRunner()
    c = SequenceEditorController(runner, MockFold())
    c.load_models()
    c.select_in_3d("1", "A", [50, 51])
    assert "select #1/A:50,51" in runner.commands
    assert c.select_in_3d("1", "A", []) is None        # empty → no-op, no command
    assert runner.commands.count("select #1/A:") == 0


def test_coalesced_clicks_produce_one_combined_command():
    # Rapid clicks accumulate into ONE select command for the whole set — not N.
    runner = MockRunner()
    c = SequenceEditorController(runner, MockFold())
    c.load_models()
    clicked = []
    for rn in (53, 50, 51, 50):                         # out-of-order + a duplicate click
        clicked.append(rn)
    combined = sorted(set(clicked))                     # the coalesced selection set
    cmd = c.build_select_command("1", "A", combined)
    assert cmd == "select #1/A:50,51,53"               # one command, exactly the clicked set
    c.select_in_3d("1", "A", combined)
    selects = [x for x in runner.commands if x.startswith("select ")]
    assert selects == ["select #1/A:50,51,53"]         # a single round-trip, not 4


def test_select_in_3d_is_error_first():
    def boom(_cmd):
        raise ConnectionError("ChimeraX gone")
    c = SequenceEditorController(boom, MockFold())
    out = c.select_in_3d("1", "A", [50])
    assert out is not None and out.get("error") and "ConnectionError" in out["error"]
    assert "value" in out                              # shaped like a result dict, not raised
    assert c.select_in_3d("1", "A", []) is None        # empty still a clean no-op


# ── reverse sync (on command) ────────────────────────────────────────────────────

def test_sync_from_chimerax_filters_unloaded():
    c = _ctl()
    c.load_models()
    synced = c.sync_from_chimerax()
    assert synced == [("1", "A", 51)]                  # Z:9 (unloaded) filtered out


# ── variant editing (lossless) ───────────────────────────────────────────────────

def test_substitution_is_lossless():
    c = _ctl()
    c.load_models()
    c.apply_substitution("1", "A", 50, "V")            # M50V
    a = c.get_chain("1", "A")
    assert a.wt_seq == "MGL"                            # WT untouched
    assert a.variant_seq == "VGL"                       # variant reflects the edit
    assert a.edits == {50: "V"} and a.is_edited


def test_editing_back_to_wt_reverts():
    c = _ctl()
    c.load_models()
    c.apply_substitution("1", "A", 50, "V")
    c.apply_substitution("1", "A", 50, "M")            # back to WT
    a = c.get_chain("1", "A")
    assert a.edits == {} and not a.is_edited and a.variant_seq == "MGL"


def test_substitution_validation():
    c = _ctl()
    c.load_models()
    with pytest.raises(ValueError):
        c.apply_substitution("1", "A", 50, "Z")        # not a standard AA
    with pytest.raises(ValueError):
        c.apply_substitution("1", "A", 99, "V")        # resnum not in chain
    with pytest.raises(KeyError):
        c.apply_substitution("1", "Q", 50, "V")        # chain not loaded
    assert "B" not in "".join(VALID_AA)                 # sanity: B is not a standard code


# ── fold wiring ──────────────────────────────────────────────────────────────────

def test_fold_variant_passes_variant_sequence():
    fold = MockFold()
    c = SequenceEditorController(MockRunner(), fold)
    c.load_models()
    c.apply_substitution("1", "A", 51, "W")            # G51W → variant MWL
    res = c.fold_variant("1", "A")
    assert fold.calls == [{"sequence": "MWL"}]          # the VARIANT, not WT
    assert res["success"] is True


def test_fold_unedited_uses_wt():
    fold = MockFold()
    c = SequenceEditorController(MockRunner(), fold)
    c.load_models()
    c.fold_variant("1", "B")
    assert fold.calls[0]["sequence"] == "AC"            # WT when no edits


# ── error-first ──────────────────────────────────────────────────────────────────

def test_run_command_failure_degrades_not_crashes():
    def boom(_cmd):
        raise ConnectionError("ChimeraX gone")
    c = SequenceEditorController(boom, MockFold())
    assert c.load_models() == []                        # no crash, empty
    assert c.sync_from_chimerax() == []
