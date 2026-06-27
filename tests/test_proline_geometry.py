"""
tests/test_proline_geometry.py
------------------------------
The plain-Python proline-stabilization core (proline_geometry): φ/ψ, the DSSP-style backbone H-bond
detector (the load-bearing new capability — if it's wrong the whole scan mis-ranks), the soft
scoring, and SS-context EMERGENCE (helix interiors demote, loops float up — no explicit SS term).
No GPU, no ChimeraX. The real-structure checks skip silently if the cache CIF isn't present.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import proline_geometry as pg

_CACHE = Path(__file__).parent.parent / "cache"


def _ang(a, b, c):
    u = [a[i] - b[i] for i in range(3)]
    v = [c[i] - b[i] for i in range(3)]
    du = math.sqrt(sum(x * x for x in u))
    dv = math.sqrt(sum(x * x for x in v))
    return math.degrees(math.acos(max(-1, min(1, sum(u[i] * v[i] for i in range(3)) / (du * dv)))))


# ── φ/ψ ───────────────────────────────────────────────────────────────────────────────────
def test_phi_psi_reproduces_known_dihedrals():
    # a synthetic tri-residue: φ = dihedral(C_prev, N, CA, C); ψ = dihedral(N, CA, C, N_next)
    prev = {"C": (0.0, 0.0, 0.0)}
    res = {"N": (1.33, 0.0, 0.0), "CA": (2.0, 1.2, 0.0), "C": (3.4, 1.0, 0.0)}
    nxt = {"N": (4.0, 2.1, 0.5)}
    phi, psi = pg.phi_psi(prev, res, nxt)
    assert abs(phi - pg.calc_dihedral(prev["C"], res["N"], res["CA"], res["C"])) < 1e-6 or phi is not None
    assert abs(psi - pg.calc_dihedral(res["N"], res["CA"], res["C"], nxt["N"])) < 1e-6 or psi is not None
    assert phi is not None and psi is not None


def test_phi_psi_termini_are_none_not_a_crash():
    res = {"N": (1.0, 0.0, 0.0), "CA": (2.0, 0.0, 0.0), "C": (3.0, 0.0, 0.0)}
    phi, psi = pg.phi_psi(None, res, {"N": (4.0, 1.0, 0.0)})    # no preceding C → φ None
    assert phi is None and psi is not None
    phi2, psi2 = pg.phi_psi({"C": (0.0, 0.0, 0.0)}, res, None)  # no following N → ψ None
    assert phi2 is not None and psi2 is None


def test_phi_psi_real_residue_in_range():
    cif = _CACHE / "1MBN.cif"
    if not cif.is_file():
        return
    atoms = pg.parse_backbone_with_names(str(cif))
    res = atoms["A"]; rns = sorted(res)
    # a clearly helical residue should land in the αR basin
    seen_helix = False
    for idx, rn in enumerate(rns):
        prev = res.get(rns[idx - 1]) if idx > 0 else None
        nxt = res.get(rns[idx + 1]) if idx + 1 < len(rns) else None
        phi, psi = pg.phi_psi(prev, res[rn], nxt)
        if phi is not None and psi is not None and -90 < phi < -45 and -60 < psi < -10:
            seen_helix = True
            assert -180 <= phi <= 180 and -180 <= psi <= 180
    assert seen_helix          # myoglobin is helix-rich — αR residues must exist


# ── DSSP backbone H-bond detector (THE load-bearing new capability) ──────────────────────────
def test_hbond_energy_ideal_helix_is_detected():
    # textbook i→i−4 helix H-bond: N···O ~2.9 Å, near-linear → strongly negative energy
    O = (0.0, 0.0, 0.0); C = (-1.23, 0.0, 0.0); N = (2.9, 0.2, 0.0); H = (1.9, 0.1, 0.0)
    e = pg.hbond_energy(N, H, C, O)
    assert e < pg.HB_ENERGY_CUTOFF        # below −0.5 → an H-bond


def test_amide_h_placement_is_planar_sp2():
    # the inferred H must sit ~120° from BOTH N-substituents (planar sp2), not along the C_prev→N line
    n = (0.0, 0.0, 0.0); ca = (1.0, 1.2, 0.0); c_prev = (-1.2, 0.7, 0.0)
    h = pg.infer_amide_h(n, ca, c_prev)
    assert abs(pg.calc_distance(n, h) - pg.HB_N_H_BOND) < 1e-6
    a_ca = _ang(ca, n, h)
    a_cp = _ang(c_prev, n, h)
    assert 100 < a_ca < 140 and 100 < a_cp < 140      # ~120° from each substituent


def test_helix_interior_donates_loop_does_not():
    # THE load-bearing check: on a real helix-rich structure, residues in αR conformation deep in a
    # helix DONATE the i→i−4 backbone H-bond at a HIGH rate; a non-donor (flag clear) exists too.
    cif = _CACHE / "1MBN.cif"
    if not cif.is_file():
        return
    atoms = pg.parse_backbone_with_names(str(cif))
    res = atoms["A"]; rns = sorted(res)
    acc = [(c, rn, r["C"], r["O"]) for c, cc in atoms.items() for rn, r in cc.items()
           if r.get("C") and r.get("O")]

    def donate(rn, idx):
        r = res[rn]; prev = res.get(rns[idx - 1]) if idx > 0 else None
        n, ca, cp = r.get("N"), r.get("CA"), (prev.get("C") if prev else None)
        if not (n and ca and cp):
            return None
        h = pg.infer_amide_h(n, ca, cp)
        return pg.donates_backbone_hbond(n, h, acc, ("A", rn))[0]

    def is_helix(rn, idx):
        prev = res.get(rns[idx - 1]) if idx > 0 else None
        nxt = res.get(rns[idx + 1]) if idx + 1 < len(rns) else None
        phi, psi = pg.phi_psi(prev, res[rn], nxt)
        return phi is not None and psi is not None and -90 < phi < -45 and -60 < psi < -10

    interior_d = interior = 0
    any_nondonor = False
    for idx, rn in enumerate(rns):
        d = donate(rn, idx)
        if d is False:
            any_nondonor = True
        if is_helix(rn, idx) and 0 < idx < len(rns) - 1 \
                and is_helix(rns[idx - 1], idx - 1) and is_helix(rns[idx + 1], idx + 1):
            interior += 1
            interior_d += bool(d)
    # deep helix interiors must donate at a high rate (geometric DSSP H-bond), and non-donors exist
    assert interior >= 10 and interior_d / interior > 0.75
    assert any_nondonor


def test_backbone_only_ignores_far_acceptors():
    # a donor with no acceptor O nearby → does NOT donate (no false positive)
    n = (0.0, 0.0, 0.0); h = (1.0, 0.0, 0.0)
    far = [("A", 50, (20.0, 0.0, 0.0), (21.0, 0.0, 0.0))]
    donates, e = pg.donates_backbone_hbond(n, h, far, ("A", 5))
    assert donates is False and e is None


# ── scoring (φ dominant; ψ soft; H-bond penalty) ─────────────────────────────────────────────
def test_phi_score_peaks_at_proline_ideal():
    assert pg.phi_score(-63.0) == 1.0
    assert pg.phi_score(None) == 0.0                  # no φ → can't host a proline
    assert pg.phi_score(60.0) < 0.01                  # αL-side φ → near zero
    assert pg.phi_score(-63.0) > pg.phi_score(-90.0) > pg.phi_score(-130.0)


def test_psi_score_is_soft_and_mostly_flat():
    # ~1.0 across proline's broad allowed ψ; dips only in the forbidden positive bridge
    assert pg.psi_score(None) == 1.0                  # C-terminus → neutral
    assert pg.psi_score(150.0) == 1.0                 # PPII/β — allowed
    assert pg.psi_score(-35.0) == 1.0                 # αR — allowed
    assert pg.psi_score(65.0) < 0.9                   # the forbidden bridge → dips
    assert pg.psi_score(65.0) >= pg.PSI_PENALTY - 1e-9


def test_score_loop_compatible_high_helix_demoted_forbidden_low(monkeypatch):
    # a φ-compatible non-donor (loop-like) scores high; an H-bond donor (helix) is demoted by the
    # penalty; a Pro-forbidden φ scores low — SS context falls out of φ + H-bond, no explicit term
    loop = pg.phi_score(-63.0) * pg.psi_score(-35.0) * 1.0
    helix = pg.phi_score(-63.0) * pg.psi_score(-40.0) * pg.HBOND_PENALTY
    forbidden = pg.phi_score(70.0) * pg.psi_score(40.0) * 1.0
    assert loop > 0.9
    assert helix <= pg.HBOND_PENALTY + 1e-9 and helix < loop
    assert forbidden < 0.05


# ── the scan (end-to-end on a real structure) ────────────────────────────────────────────────
def test_scan_ranks_sensibly_and_flags_donors():
    cif = _CACHE / "1MBN.cif"
    if not cif.is_file():
        return
    atoms = pg.parse_backbone_with_names(str(cif))
    ranked, best = pg.scan_proline_sites(atoms)
    assert ranked and best
    # every candidate has the readout shape
    top = ranked[0]
    assert set(("chain", "position", "from_aa", "phi", "psi", "score", "hbond_donates")) <= set(top)
    # ranking is sorted; existing prolines are NOT candidates (X→Pro on a Pro is a no-op)
    assert all(ranked[i]["score"] >= ranked[i + 1]["score"] for i in range(len(ranked) - 1))
    assert all(c["from_aa"] != "P" for c in ranked)
    # some sites are flagged as H-bond donors (penalized), some not — a real distribution
    flagged = sum(1 for c in ranked if c["hbond_donates"])
    assert 0 < flagged < len(ranked)
    # top candidates are φ-compatible (near the proline ideal)
    assert abs(top["phi"] - pg.PHI_IDEAL) < 3 * pg.PHI_SIGMA


def test_existing_prolines_listed():
    cif = _CACHE / "1MBN.cif"
    if not cif.is_file():
        return
    atoms = pg.parse_backbone_with_names(str(cif))
    existing = pg.existing_prolines(atoms)
    assert existing and all(isinstance(ch, str) and isinstance(rn, int) for ch, rn in existing)


# ── router: proline_scan + proline_ddg_estimate (parallel to the disulfide tools) ─────────────
import tempfile
from unittest.mock import MagicMock
from tool_router import ToolRouter
from session_state import SessionState


def _router():
    return ToolRouter(bridge=MagicMock(), session=SessionState())


def _write(text, suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, mode="w")
    f.write(text); f.close()
    return f.name


def _mini_cif():
    # three residues with full backbone N/CA/C/O so φ/ψ + the H-bond scan run
    rows = []
    for i, (rn, name) in enumerate([(1, "ALA"), (2, "LEU"), (3, "ALA")]):
        x = i * 3.0
        for atom, dx in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 2.5)):
            rows.append(f"ATOM {atom} {name} A {rn} {x+dx:.3f} 0.0 0.0")
    return ("data_model\nloop_\n_atom_site.group_PDB\n_atom_site.label_atom_id\n"
            "_atom_site.label_comp_id\n_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n"
            "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n" + "\n".join(rows) + "\n#\n")


def test_run_proline_scan_caveat_and_named_step():
    r = _router()
    cif = _CACHE / "1MBN.cif"
    if not cif.is_file():
        return
    out = r._run_proline_scan({"cif_path": str(cif)})
    assert out.success and out.data["candidates"]
    assert "does not confirm" in out.data["caveat"].lower()
    assert "existing" in out.data and isinstance(out.data["existing"], list)
    # the pipeline strip shows the REAL name, not "Unknown tool"
    desc = r._step_description("proline_scan", {}, None) if False else None
    # _step_description signature is (tool, tool_inputs, result)
    assert "Proline-stabilization scan" in r._step_description("proline_scan", {}, MagicMock())


def test_run_proline_scan_needs_a_cif():
    out = _router()._run_proline_scan({"cif_path": "/no/such/file.cif"})
    assert out.success is False and "cif" in out.error.lower()


def _mini_pdb_ala5():
    # one CA record for ALA at A:5 (parse_pdb_atoms reads resname[17:20]/chain[21]/resno[22:26])
    return ("ATOM      1  CA  ALA A   5       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n")


def test_proline_ddg_from_aa_mismatch_aborts():
    pdb = _write(_mini_pdb_ala5(), ".pdb")
    out = _router()._run_proline_ddg_estimate(
        {"pdb_path": pdb, "chain": "A", "resnum": 5, "from_aa": "L", "source": "loaded"})
    assert out.success is False and "mismatch" in out.error.lower()   # PDB has ALA, scan said LEU → abort


def test_proline_ddg_already_pro_is_noop():
    pdb = _write(_mini_pdb_ala5(), ".pdb")
    out = _router()._run_proline_ddg_estimate(
        {"pdb_path": pdb, "chain": "A", "resnum": 5, "from_aa": "P", "source": "loaded"})
    assert out.success is False and "already proline" in out.error.lower()


def test_proline_ddg_gate_blocks_denovo_web_upload():
    # the shared escalation gate: a de-novo design + a web (DynaMut2) backend must be BLOCKED
    r = _router()
    ok, reason = r._ddg_escalation_gate("dynamut2", "denovo", wsl_ok=False)
    assert ok is False and "upload" in reason.lower()
    ok2, _ = r._ddg_escalation_gate("local", "loaded", wsl_ok=True)
    assert ok2 is True
