"""
tests/test_template_assist.py
-----------------------------
Template-guided fold (first build) — router side. All mocked (no real WSL/Boltz/US-align):
  A. _resolve_boltz_templates — path passthrough, pdb_id → download, fail-closed on a bad ref.
  B. _run_template_assist     — reuses _fold_wt_reference for BOTH floors (no new floor math),
     computes ΔpLDDT + per-residue Δflexibility (unguided − guided), honest readout. NO re-fold
     of either seed-0 (the two folds are reused off disk by _fold_wt_reference's REUSE path).
  C. _run_boltz template threading — inputs["templates"] resolves + forwards to the bridge; a
     bad template fails the fold (never silently folds unguided).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from tool_router import ToolRouter


def _router() -> ToolRouter:
    return ToolRouter(bridge=MagicMock(), session=MagicMock())


# ── A. _resolve_boltz_templates ──────────────────────────────────────────────────────
class TestResolveTemplates:
    def test_none_passthrough(self):
        r = _router()
        out, err = r._resolve_boltz_templates(None)
        assert out is None and err is None

    def test_local_path_used_as_is(self, tmp_path):
        f = tmp_path / "t.cif"; f.write_text("data_\n")
        r = _router()
        out, err = r._resolve_boltz_templates([{"cif": str(f), "chain_id": "A"}])
        assert err is None
        assert out[0]["cif"] == str(f) and out[0]["chain_id"] == "A"

    def test_pdb_id_resolves_to_mmCIF_preferred(self, tmp_path, monkeypatch):
        # a pdb_id resolves to the official mmCIF (gemmi-safe), NOT the pdb — Boltz's parse_pdb
        # raises a swallowed KeyError on some ligand/entity-bearing PDBs.
        c = tmp_path / "1MBN.cif"; c.write_text("data_\n")
        r = _router()
        monkeypatch.setattr(r, "_download_cif_by_id", lambda pid: str(c))
        out, err = r._resolve_boltz_templates(
            [{"pdb_id": "1MBN", "chain_id": "A", "force": True, "threshold": 10.0}])
        assert err is None
        assert out[0]["cif"] == str(c) and "pdb" not in out[0]   # mmCIF, not pdb
        assert "pdb_id" not in out[0]                            # stripped once resolved
        assert out[0]["force"] is True and out[0]["threshold"] == 10.0   # steering preserved

    def test_pdb_id_falls_back_to_pdb_when_no_cif(self, tmp_path, monkeypatch):
        f = tmp_path / "1MBN.pdb"; f.write_text("ATOM\n")
        r = _router()
        monkeypatch.setattr(r, "_download_cif_by_id", lambda pid: None)   # cif unavailable
        monkeypatch.setattr(r, "_download_pdb_by_id", lambda pid: str(f))
        out, err = r._resolve_boltz_templates([{"pdb_id": "1MBN", "chain_id": "A"}])
        assert err is None and out[0]["pdb"] == str(f) and "cif" not in out[0]

    def test_failed_download_is_error_fail_closed(self, monkeypatch):
        r = _router()
        monkeypatch.setattr(r, "_download_cif_by_id", lambda pid: None)
        monkeypatch.setattr(r, "_download_pdb_by_id", lambda pid: None)
        out, err = r._resolve_boltz_templates([{"pdb_id": "ZZZZ"}])
        assert out is None and "Could not obtain template" in err

    def test_missing_local_file_is_error(self):
        r = _router()
        out, err = r._resolve_boltz_templates([{"cif": "/nope/missing.cif"}])
        assert out is None and "not a readable local file" in err


# ── B. _run_template_assist (reuse the floor; no new floor math) ──────────────────────
class TestRunTemplateAssist:
    def _inputs(self):
        return {
            "engine": "boltz", "target": "monomer", "multichain": False, "variant_chain": "A",
            "wt_chains": [{"id": "A", "sequence": "MKVLW"}],
            "unguided_ref": {"model_id": "7", "path": "/tmp/u.cif", "seed": 0},
            "guided_ref":   {"model_id": "9", "path": "/tmp/g.cif", "seed": 0},
            "templates": [{"pdb_id": "1MBN", "chain_id": "A"}],
            "guided_mean_plddt": 84.0, "unguided_mean_plddt": 62.0,
            "guided_plddt":   {"1": 85.0, "2": 88.0},
            "unguided_plddt": {"1": 60.0, "2": 70.0},
            "template_label": "1MBN", "force": False, "threshold": None,
        }

    def test_delta_flex_and_plddt_from_two_floors(self, monkeypatch):
        r = _router()
        # _fold_wt_reference is called TWICE: first unguided (no templates), then guided.
        calls = []
        def fake_ref(inp):
            calls.append(dict(inp))
            if inp.get("fold_templates") is None:      # unguided floor (wigglier)
                return {"floor_ddm": {"1": 2.0, "2": 1.5, "3": 0.8}, "n_floor_seeds": 4}
            return {"floor_ddm": {"1": 0.7, "2": 1.6, "3": 0.5}, "n_floor_seeds": 4}   # guided
        monkeypatch.setattr(r, "_fold_wt_reference", fake_ref)
        monkeypatch.setattr(r, "_resolve_boltz_templates", lambda t: ([], None))  # hermetic (no DL)
        monkeypatch.setattr(r, "_usalign_tm2", lambda a, b: None)
        res = r._run_template_assist(self._inputs())
        assert res.success
        d = res.data
        # Δflex = unguided − guided: res 1 stabilized (+1.3), res 2 loosened (−0.1), res 3 (+0.3)
        assert d["d_flex"]["1"] == 1.3 and d["d_flex"]["2"] == -0.1 and d["d_flex"]["3"] == 0.3
        assert d["n_stabilized"] == 2 and d["n_loosened"] == 1
        assert d["d_plddt"] == 22.0                    # 84 − 62
        assert d["d_plddt_by_res"]["1"] == 25.0 and d["d_plddt_by_res"]["2"] == 18.0
        # the GUIDED floor call carried the templates; the unguided one did NOT.
        assert calls[0]["fold_templates"] is None
        assert calls[1]["fold_templates"] == [{"pdb_id": "1MBN", "chain_id": "A"}]

    def test_honest_readout_surfaces_both_and_delta(self, monkeypatch):
        r = _router()
        monkeypatch.setattr(r, "_fold_wt_reference",
                            lambda inp: {"floor_ddm": {"1": 1.0}, "n_floor_seeds": 4})
        monkeypatch.setattr(r, "_resolve_boltz_templates", lambda t: ([], None))
        monkeypatch.setattr(r, "_usalign_tm2", lambda a, b: None)
        res = r._run_template_assist(self._inputs())
        s = res.summary.lower()
        assert "62.0" in s and "84.0" in s             # unguided AND guided shown
        assert "not a confirmation of correctness" in s          # never "rescue confirmed" (truth-dependent)
        assert "template-biased" in s

    def test_negligible_effect_honest_null(self, monkeypatch):
        r = _router()
        monkeypatch.setattr(r, "_fold_wt_reference",
                            lambda inp: {"floor_ddm": {"1": 1.0, "2": 1.0}, "n_floor_seeds": 4})
        monkeypatch.setattr(r, "_resolve_boltz_templates", lambda t: ([], None))
        monkeypatch.setattr(r, "_usalign_tm2", lambda a, b: None)
        inp = self._inputs()
        inp["guided_mean_plddt"] = 62.3                 # ~no pLDDT change; floors identical
        res = r._run_template_assist(inp)
        # honest null = tiny delta + zero stabilized, with NO correctness/rescue claim (descriptive)
        assert res.data["d_plddt"] == 0.3 and res.data["n_stabilized"] == 0
        assert "not a confirmation of correctness" in res.summary.lower()

    def test_high_adoption_caveat_fires_when_template_was_NOT_already_close(self, monkeypatch):
        # The possible-copying caveat fires on HIGH adoption (≥0.8) ONLY when the template was NOT
        # already close to the unguided fold — low pre-hoc proxy structTM(template, unguided) < 0.5.
        # That is the suspicious case: guidance pulled the fold onto a template it did not resemble.
        r = _router()
        monkeypatch.setattr(r, "_fold_wt_reference",
                            lambda inp: {"floor_ddm": {"1": 1.0}, "n_floor_seeds": 4})
        monkeypatch.setattr(r, "_resolve_boltz_templates",
                            lambda t: ([{"cif": "/x/1mbn.cif", "chain_id": "A"}], None))
        # adoption = structTM(guided "/tmp/g.cif", template); prehoc = structTM(template, unguided).
        monkeypatch.setattr(r, "_usalign_tm2",
                            lambda a, b: 0.91 if a == "/tmp/g.cif" else 0.30)   # high adopt, low prehoc
        res = r._run_template_assist(self._inputs())
        d = res.data
        assert d["max_adoption"] == 0.91 and d["high_adoption_caveat"] is True
        assert d["high_adoption_caveat_reason"] == "distant"   # MEASURED distant → refined wording
        assert d["per_template"][0]["adoption"] == 0.91
        assert d["per_template"][0]["prehoc_structTM_to_unguided"] == 0.30
        assert ("high adoption" in res.summary.lower()
                and "did not already resemble" in res.summary.lower()
                and "imposing the template" in res.summary.lower())

    def test_high_adoption_caveat_SUPPRESSED_when_template_was_already_close(self, monkeypatch):
        # NATURAL success: the template was already same-fold-close to the unguided fold (prehoc
        # ≥ 0.5), so high adoption is convergence-within-a-fold, NOT copying — the caveat must NOT
        # fire. This is the false-fire the §9 polish removes (previously fired on adoption alone).
        r = _router()
        monkeypatch.setattr(r, "_fold_wt_reference",
                            lambda inp: {"floor_ddm": {"1": 1.0}, "n_floor_seeds": 4})
        monkeypatch.setattr(r, "_resolve_boltz_templates",
                            lambda t: ([{"cif": "/x/1mbn.cif", "chain_id": "A"}], None))
        monkeypatch.setattr(r, "_usalign_tm2",
                            lambda a, b: 0.91 if a == "/tmp/g.cif" else 0.72)   # high adopt, HIGH prehoc
        res = r._run_template_assist(self._inputs())
        d = res.data
        assert d["max_adoption"] == 0.91                       # still reported (headline)
        assert d["per_template"][0]["prehoc_structTM_to_unguided"] == 0.72
        assert d["high_adoption_caveat"] is False              # NOT flagged — template was already close
        assert d["high_adoption_caveat_reason"] is None
        assert "imposing the template" not in res.summary.lower()

    def test_high_adoption_caveat_fires_GENERIC_when_prehoc_unavailable(self, monkeypatch):
        # Conservative default: if the pre-hoc proxy cannot be computed (US-align returns None for
        # the template→unguided direction), we cannot establish the template was already close, so
        # high adoption STILL fires — never silently suppress on a missing proxy. BUT it fires with
        # the GENERIC wording, NOT the refined "did not already resemble" string: the caveat must not
        # assert the "distant" condition it never measured.
        r = _router()
        monkeypatch.setattr(r, "_fold_wt_reference",
                            lambda inp: {"floor_ddm": {"1": 1.0}, "n_floor_seeds": 4})
        monkeypatch.setattr(r, "_resolve_boltz_templates",
                            lambda t: ([{"cif": "/x/1mbn.cif", "chain_id": "A"}], None))
        monkeypatch.setattr(r, "_usalign_tm2",
                            lambda a, b: 0.91 if a == "/tmp/g.cif" else None)   # high adopt, prehoc n/a
        res = r._run_template_assist(self._inputs())
        d = res.data
        assert d["max_adoption"] == 0.91
        assert d["per_template"][0]["prehoc_structTM_to_unguided"] is None
        assert d["high_adoption_caveat"] is True
        assert d["high_adoption_caveat_reason"] == "unmeasured"   # fired conservatively, NOT measured
        # generic wording fires; the refined "distant" claim must be ABSENT (never over-asserted)
        assert "high adoption" in res.summary.lower()
        assert "did not already resemble" not in res.summary.lower()
        assert "imposing the template" not in res.summary.lower()

    def test_distant_takes_precedence_over_unmeasured_across_templates(self, monkeypatch):
        # Multi-template set: one template is MEASURABLY distant-yet-adopted (prehoc 0.30), another
        # is adopted with an unavailable proxy (None). The measured "distant" claim wins → refined
        # wording (a real distant template is enough to warrant the strong claim).
        r = _router()
        monkeypatch.setattr(r, "_fold_wt_reference",
                            lambda inp: {"floor_ddm": {"1": 1.0}, "n_floor_seeds": 4})
        monkeypatch.setattr(r, "_resolve_boltz_templates",
                            lambda t: ([{"cif": "/x/t1.cif", "chain_id": "A"},
                                        {"cif": "/x/t2.cif", "chain_id": "A"}], None))
        def tm(a, b):
            if a == "/tmp/g.cif":   return 0.91     # adoption — both templates highly adopted
            if a == "/x/t1.cif":    return 0.30     # t1 prehoc MEASURED distant
            return None                             # t2 prehoc unavailable
        monkeypatch.setattr(r, "_usalign_tm2", tm)
        inp = self._inputs()
        inp["templates"] = [{"pdb_id": "T1", "chain_id": "A"}, {"pdb_id": "T2", "chain_id": "A"}]
        res = r._run_template_assist(inp)
        d = res.data
        assert d["high_adoption_caveat"] is True
        assert d["high_adoption_caveat_reason"] == "distant"        # measured-distant wins
        assert "did not already resemble" in res.summary.lower()

    def test_requires_both_folds(self):
        r = _router()
        inp = self._inputs(); inp["guided_ref"] = {}
        res = r._run_template_assist(inp)
        assert not res.success and "guided fold" in res.error.lower()

    def test_esmfold_rejected(self):
        r = _router()
        inp = self._inputs(); inp["engine"] = "esmfold"
        res = r._run_template_assist(inp)
        assert not res.success and "boltz-only" in res.error.lower()


# ── C. _run_boltz forwards/resolves templates; fail-closed on a bad ref ───────────────
class TestRunBoltzTemplateThreading:
    def test_bad_template_fails_fold_not_silent(self, monkeypatch):
        r = _router()
        monkeypatch.setattr(r, "_download_pdb_by_id", lambda pid: None)   # download fails
        res = r._run_boltz({"model_id": "1", "sequence": "MKVLW", "chain": "A",
                            "templates": [{"pdb_id": "ZZZZ"}]})
        assert not res.success and "Could not obtain template" in res.error

    def test_resolved_templates_passed_to_bridge(self, tmp_path, monkeypatch):
        f = tmp_path / "t.cif"; f.write_text("data_\n")
        r = _router()
        bridge = MagicMock()
        bridge.predict.return_value = {"success": False, "error": "stop-after-build"}
        monkeypatch.setattr(r, "_get_boltz_bridge", lambda: bridge)
        r._run_boltz({"model_id": "1", "sequence": "MKVLW", "chain": "A",
                      "templates": [{"cif": str(f), "chain_id": "A"}]})
        _, kwargs = bridge.predict.call_args
        assert kwargs["templates"] == [{"cif": str(f), "chain_id": "A"}]   # resolved list forwarded
