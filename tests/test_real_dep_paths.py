"""
tests/test_real_dep_paths.py
----------------------------
NON-MOCKED real external-dependency smokes for the LIGHT, CI-runnable bridges —
the cavity `TestRealSASAPath` pattern generalized.

WHY: cavity detection's SASA path was dead for 17 days (a wrong `Bio.PDB.SASA`
import) yet 20 unit tests stayed green because every one MOCKED `_sasa_for_chains`.
A test that mocks the dependency proves the wiring, not that the real path
imports/runs. These tests hit the REAL freesasa / Bio.PDB.SASA path against a
committed hermetic fixture (crambin, 1CRN — no network) and assert a sane,
non-empty result, so a future silent death of any of these paths fails CI loudly.

Each test is written to FAIL if its real path dies (empty / all-zero), and to
check the module's _OK import flag where one exists. Heavy bridges (ThermoMPNN /
RaSP / Rosetta / ESM / ColabFold / RFdiffusion) are out of scope here — they get
capability-checks + gated opt-in smokes in Unit B.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_CRAMBIN = Path(__file__).parent / "fixtures" / "1crn.pdb"


@pytest.fixture(autouse=True)
def _require_fixture():
    assert _CRAMBIN.is_file(), f"missing hermetic fixture {_CRAMBIN}"


class TestStructuralUtilsRealSASA:
    """structural_utils.compute_sasa — freesasa → Bio.PDB.SASA fallback."""

    def test_compute_sasa_nonempty_nonzero(self):
        import structural_utils as su
        sasa = su.compute_sasa(str(_CRAMBIN), "A")
        assert len(sasa) > 0, "real SASA path returned EMPTY map (dead import?)"
        assert sum(1 for v in sasa.values() if v > 0) >= 1, \
            "all-zero SASA — the real path is dead, not computing"


class TestGlycanRealSASA:
    """glycan_bridge._get_sasa — freesasa → Bio.PDB.SASA, normalised RSA."""

    def test_get_sasa_nonempty_nonzero(self):
        import glycan_bridge as gb
        sasa = gb._get_sasa(str(_CRAMBIN), "A", list(range(1, 21)))
        assert len(sasa) > 0, "real SASA path returned EMPTY (dead import?)"
        assert any(v > 0 for v in sasa.values()), \
            "all-zero RSA — exposed residues exist in crambin; the path is dead"


class TestSaltBridgeRealPath:
    """salt_bridge — the _OK import flags (the direct cavity-analog) + real run."""

    def test_import_flags_alive(self):
        import salt_bridge_bridge as sb
        assert sb._BIOPYTHON_OK is True, "BioPython import dead — salt-bridge SASA path"
        assert sb._FREESASA_OK is True, "freesasa import dead — salt-bridge SASA path"

    def test_find_existing_salt_bridges_real(self):
        import salt_bridge_bridge as sb
        bridges = sb.SaltBridgeBridge().find_existing_salt_bridges(str(_CRAMBIN))
        assert isinstance(bridges, list)
        # crambin has a real Arg-Glu salt bridge (A17–A23); the SASA path also
        # annotates each with `buried`
        assert len(bridges) >= 1, "real geometry/SASA path found NO salt bridge"
        assert "buried" in bridges[0], "SASA-derived buried annotation missing"


class TestProlineRealSASA:
    """proline_bridge._detect_functional_residues_sasa — Bio.PDB.SASA buried scan."""

    def test_functional_residue_sasa_nonempty(self):
        import proline_bridge as pb
        found = pb.ProlineBridge()._detect_functional_residues_sasa(str(_CRAMBIN), "A")
        assert isinstance(found, set)
        # crambin has buried Cys/polar residues (e.g. {3,4,26,32}); an EMPTY set
        # here means the real ShrakeRupley path silently died
        assert len(found) >= 1, "real Bio.PDB.SASA path returned EMPTY — dead path"
