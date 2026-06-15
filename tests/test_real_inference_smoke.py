"""
tests/test_real_inference_smoke.py
----------------------------------
B3 — GATED opt-in real-inference smokes for the heavy external-dep bridges. The
periodic deep check that complements the capability flag (Unit B): the flag
confirms the import chain resolves; this confirms one real inference actually
produces a sane result.

SKIP BY DEFAULT — runs only when STRUCTUREBOT_RUN_LIVE_DEPS=1 AND the live envs
(venv312 GPU / WSL rasp_env / RFdiffusion env / ESM model) are present. Hermetic
input: the committed crambin fixture (1CRN, 46 aa) — no network for the structure
(ESMFold/ESM may fetch their model on first use).

Each runs ONE tiny inference and asserts a non-empty, sane result. These are the
tests behind the §7 "guarded (capability + gated smoke)" status.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_RUN = os.environ.get("STRUCTUREBOT_RUN_LIVE_DEPS") == "1"
_gate = pytest.mark.skipif(
    not _RUN,
    reason="gated real-inference smoke — set STRUCTUREBOT_RUN_LIVE_DEPS=1 "
           "(needs the venv312 / WSL / model envs)")

# Every smoke here does REAL GPU/WSL inference → serialize GPU access across processes
# (conftest `_gpu_serialize`) so two real folds never contend for VRAM when run.
pytestmark = pytest.mark.gpu

_CRAMBIN = str(Path(__file__).parent / "fixtures" / "1crn.pdb")
_CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"   # 1CRN chain A


@_gate
def test_thermompnn_real_inference():
    from thermompnn_bridge import ThermoMPNNBridge
    b = ThermoMPNNBridge()
    assert b.is_available(), "ThermoMPNN capability flag False — env not ready"
    # crambin chain A: T1, R10, R17 (WT verified against the structure by the bridge)
    ddg, src = b.score_mutations(
        _CRAMBIN, "A",
        [{"position": 1, "from_aa": "T", "to_aa": "A"},
         {"position": 10, "from_aa": "R", "to_aa": "A"}])
    assert ddg, "ThermoMPNN produced NO scores on crambin (real-path regression)"


@_gate
def test_proteinmpnn_real_inference():
    from proteinmpnn_bridge import ProteinMPNNBridge
    b = ProteinMPNNBridge()
    assert b.is_available(), "ProteinMPNN capability flag False — env not ready"
    res = b.analyze({
        "model_id": "1", "pdb_path": _CRAMBIN, "chain_id": "A",
        "num_sequences": 1, "fixed_positions": [], "design_positions": None,
        "interface_design": False, "partner_chain": None,
        "omit_aas": "", "bias_aas": [],
    })
    assert res.success and res.data.get("sequences"), "ProteinMPNN produced no designs"


@_gate
def test_rasp_real_inference():
    from rasp_bridge import RaSPBridge
    b = RaSPBridge()
    assert b.is_available(), "RaSP capability flag False — env not ready"
    ddg, src = b.score_mutations(_CRAMBIN, "A",
                                 [{"position": 10, "from_aa": "R", "to_aa": "A"}])
    assert ddg, "RaSP produced NO scores on crambin (real-path regression)"


@_gate
def test_esmfold_real_inference():
    from esmfold_bridge import ESMFoldBridge
    pred = ESMFoldBridge().predict(_CRAMBIN_SEQ)
    assert isinstance(pred, dict) and pred.get("mean_plddt"), \
        "ESMFold returned no pLDDT (real-path regression)"


@_gate
def test_esm_real_inference():
    from esm_bridge import EsmBridge
    res = EsmBridge().analyze(_CRAMBIN_SEQ)
    assert res.success and res.data, "ESM-2 produced no conservation scores"


@_gate
def test_rfdiffusion_real_inference():
    # MUST go through the bridge analyze() DISPATCH (not the raw CLI) — the 06-12
    # smoke verified only the engine; the dispatch (script-path resolution) was the
    # broken, never-exercised path. symmetric is the no-PDB analyze() mode; T≥15 is
    # an RFdiffusion model constraint. Asserts a real backbone via the production path.
    from rfdiffusion_bridge import RFdiffusionBridge
    b = RFdiffusionBridge()
    assert b.is_available(), "RFdiffusion capability flag False — env not ready"
    res = b.analyze({"mode": "symmetric", "symmetry": "c2", "contigs": "60-60",
                     "num_designs": 1, "num_steps": 15})
    assert res.success and res.data.get("pdb_paths"), \
        f"RFdiffusion analyze() produced no backbone: {(res.error or '')[:160]}"
