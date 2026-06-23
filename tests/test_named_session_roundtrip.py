"""
tests/test_named_session_roundtrip.py
-------------------------------------
ROUND-TRIP GUARD for named save/load (the load-bearing piece). The probe found the
serialization already round-trips every recent-arc field; this PINS it so a future arc
field can't silently regress. A de-novo construct carrying guided_fold + template_assist +
structural_align + wt_refs + a foldseek-derived guided fold must survive
  DesignSession.to_dict -> SessionState.save -> try_load -> DesignSession.from_dict
field-identical, and the int-key (resno) JSON coercion is asserted EXPLICITLY (json has no
int keys, so resno-keyed maps come back string-keyed — pin the shape).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from session_state import SessionState
from variant_model import DesignSession, build_design_session_from_sequence


def _rich_denovo_design():
    """A de-novo design with a variant and EVERY recent-arc field populated, including
    int-resno-keyed maps (template_assist.d_flex, wt_refs.floor) and a foldseek-derived
    guided fold (the chosen template label is the foldseek hit)."""
    d = build_design_session_from_sequence("hivca", [("ACDEFGHIKLMNPQRSTVWY", 3)])
    cd = next(iter(d.chains.values()))
    vid = d.new_variant_id()
    cd.add_variant(vid)
    cd.edit_variant(vid, 2, "W")                       # a real substitution on the variant row

    cd.template_fold = {"model_id": "denovo-x", "engine": "boltz", "target": "monomer",
                        "plddt": 91.2, "chains": ["A", "B", "C"],
                        "cif_path": "/tmp/boltz_pred_t.cif"}
    cd.guided_fold = {"model_id": "denovo-x", "engine": "boltz", "target": "assembly",
                      "templated": True, "template_label": "8UB2",   # foldseek-DISCOVERED hit
                      "force": False, "threshold": None,
                      "cif_path": "/tmp/boltz_pred_g.cif", "adoption": 0.93}
    cd.template_assist = {"template_label": "8UB2", "guided_mean_plddt": 95.1,
                          "unguided_mean_plddt": 92.0, "d_plddt": 3.1, "n_stabilized": 4,
                          "tm_adopt": 0.90, "force": False, "threshold": None,
                          "d_flex": {10: 0.42, 25: 0.11, 130: 0.30}}     # INT resno keys
    cd.structural_align = {"reference": "1AXC", "ref_label": "1AXC", "tm_ref": 0.90,
                           "tm_query": 0.88, "rmsd": 1.10, "n_aligned": 240,
                           "matrix": [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0], "norm": "ref"}
    cd.wt_refs = {"boltz:assembly": {"engine": "boltz", "target": "assembly", "seed": 0,
                                     "model_id": "denovo-x", "path": "/tmp/wt.cif",
                                     "floor": {10: 0.20, 25: 0.15}}}      # INT resno keys
    return d, cd, vid


def _norm(x):
    """The json-normalized form (tuples->lists, int keys->str keys) — what survives disk."""
    return json.loads(json.dumps(x))


def test_named_session_arc_fields_roundtrip_field_identical(tmp_path):
    d, cd, vid = _rich_denovo_design()
    blob = d.to_dict()

    s = SessionState()
    s.add_design_session(d.model_id, blob)
    f = tmp_path / "named.json"
    s.save(str(f))

    state, err = SessionState.try_load(str(f))
    assert err is None and state is not None
    restored_blob = state.get_design_session(d.model_id)

    # (1) The persisted blob round-trips byte-identical vs the json-normalized original.
    assert restored_blob == _norm(blob)

    # (2) from_dict rebuilds EVERY arc field, equal to the json-normalized source (non-empty).
    rd = DesignSession.from_dict(restored_blob)
    rcd = next(iter(rd.chains.values()))
    assert rcd.template_fold    == _norm(cd.template_fold)    and rcd.template_fold
    assert rcd.guided_fold      == _norm(cd.guided_fold)      and rcd.guided_fold
    assert rcd.template_assist  == _norm(cd.template_assist)  and rcd.template_assist
    assert rcd.structural_align == _norm(cd.structural_align) and rcd.structural_align
    assert rcd.wt_refs          == _norm(cd.wt_refs)          and rcd.wt_refs

    # (3) The variant row + its substitution survive; the foldseek-derived template label survives.
    assert [v.id for v in rcd.variants] == [vid]
    assert rcd.variants[0].mutations[0].to_aa == "W"
    assert rcd.guided_fold["template_label"] == "8UB2"

    # (4) de-novo identity survives (source + synthetic model id + members).
    assert rd.source == "sequence"
    assert rcd.members == cd.members


def test_int_resno_keys_coerce_to_str_through_json(tmp_path):
    """EXPLICIT guard: int resno keys in d_flex / floor come back STRING-keyed after a JSON
    save/load round-trip (json has no int keys). Pin the behavior so a future arc field keyed
    by an int resno can't silently change shape on restore without this test failing."""
    d, _cd, _vid = _rich_denovo_design()
    s = SessionState()
    s.add_design_session(d.model_id, d.to_dict())
    f = tmp_path / "n.json"
    s.save(str(f))

    rd = DesignSession.from_dict(SessionState.load(str(f)).get_design_session(d.model_id))
    rcd = next(iter(rd.chains.values()))

    assert set(rcd.template_assist["d_flex"]) == {"10", "25", "130"}      # int -> str
    assert set(rcd.wt_refs["boltz:assembly"]["floor"]) == {"10", "25"}
    assert rcd.template_assist["d_flex"]["10"] == 0.42                    # values preserved
    assert rcd.wt_refs["boltz:assembly"]["floor"]["25"] == 0.15
