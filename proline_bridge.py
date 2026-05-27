"""
proline_bridge.py
-----------------
Local stabilising proline substitution scanner for StructureBot.

Rationale
---------
Proline is the only natural amino acid with a pyrrolidine ring that
constrains the backbone φ dihedral to ~-60°.  When a residue already
has φ ≈ -60° in the folded state, replacing it with Pro costs little
enthalpic penalty but gains substantial stabilisation through reduction
of unfolded-state conformational entropy (ΔΔG ≈ +1 to +4 kcal/mol).

Key exclusions
--------------
- Existing Pro, Gly (no φ)
- Helix positions (Pro would break the helix H-bond pattern)
- i+1 after existing Pro (would create Pro-Pro)
- Within 3 residues of a chain terminus
- Residues in cis-peptide conformation (ψ_{i-1} ∈ [-60, 60])
- Interface residues (soft penalty, not hard exclusion)
- Low ESM tolerance (soft penalty)

Scoring
-------
    composite = phi_score × loop_bonus × esm_factor × iface_factor × hbond_factor

    phi_score  = max(0, 1 - |φ - (-60)| / 60)      # peaks at φ=-60, zero at ±60°
    loop_bonus = 1.3 if secondary structure is loop, else 1.0
    esm_factor     = 1.0 if esm_tolerance >= 0.5 else 0.6
    iface_factor   = 1.0 if not near interface, else 0.4
    hbond_factor   = 0.7 if ψ_{i-1} ∈ (-60, 60),   else 1.0

Confidence bands
----------------
    high     : composite > 0.6
    moderate : 0.3 ≤ composite ≤ 0.6
    low      : composite < 0.3

ChimeraX colors
---------------
    high     : magenta (#cc00cc)
    moderate : orange  (#cc6600)
    low      : yellow  (#cccc00)

Dependencies
------------
- BioPython (biopython ≥ 1.87) — PPBuilder for backbone angles
- DSSP optional (mkdssp/dssp binary); fallback via φ/ψ classification
- tool_router.ToolStepResult — return type
"""

from __future__ import annotations

import math
import re
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from tool_router import ToolStepResult


# ── Catalytic residue types for SASA-based active-site heuristic ─────────────
# His, Asp, Glu, Ser, Cys: common charge-relay / nucleophile residues.
_CATALYTIC_ONE_LETTER: frozenset = frozenset("HDESC")

# ── One-letter ↔ three-letter mappings ───────────────────────────────────────

_ONE_TO_THREE: Dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

_THREE_TO_ONE: Dict[str, str] = {v: k for k, v in _ONE_TO_THREE.items()}


# ── Secondary structure classification from φ/ψ ───────────────────────────────

def _classify_ss_from_angles(phi_deg: float, psi_deg: float) -> str:
    """
    Classify secondary structure from backbone dihedrals.

    Returns 'H' (helix), 'E' (sheet), or 'L' (loop/coil).
    """
    if (-90.0 < phi_deg < -45.0) and (-60.0 < psi_deg < -10.0):
        return "H"
    if (-160.0 < phi_deg < -90.0) and (90.0 < psi_deg < 175.0):
        return "E"
    return "L"


# ════════════════════════════════════════════════════════════════════════════════
# ProlineBridge
# ════════════════════════════════════════════════════════════════════════════════

class ProlineBridge:
    """
    Scan a protein chain for stabilising proline substitution candidates.

    The class is stateless; all inputs are passed per-call.
    """

    # ── 1. Backbone angle extraction ─────────────────────────────────────────

    def extract_backbone_angles(
        self,
        pdb_path: str,
        chain:    str,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract per-residue backbone geometry from a PDB file.

        Uses BioPython PPBuilder for φ/ψ angles (in degrees after conversion
        from radians).  Attempts DSSP for secondary structure; falls back to
        Ramachandran classification when DSSP is unavailable.

        Parameters
        ----------
        pdb_path : path to local PDB file (as_posix() used internally)
        chain    : chain identifier, e.g. "A"

        Returns
        -------
        dict[int, dict] keyed by 1-based residue sequence number, each entry:
          {
            "phi":     float | None,   # degrees, None for terminus
            "psi":     float | None,
            "ss":      str,            # "H", "E", or "L"
            "resname": str,            # 3-letter code, e.g. "LEU"
            "aa":      str,            # 1-letter code, e.g. "L"
          }
        """
        from Bio.PDB import PDBParser, PPBuilder
        from Bio.PDB.Polypeptide import is_aa

        pdb_path_str = Path(pdb_path).as_posix()
        parser       = PDBParser(QUIET=True)

        try:
            structure = parser.get_structure("prot", pdb_path_str)
        except Exception as exc:
            raise ValueError(f"Could not parse PDB file '{pdb_path}': {exc}") from exc

        model = structure[0]

        # Collect raw φ/ψ from PPBuilder
        builder    = PPBuilder()
        raw_angles: Dict[int, Dict[str, Any]] = {}   # seqnum → {phi, psi, resname}

        for chain_obj in model:
            if chain_obj.id != chain:
                continue
            for pp in builder.build_peptides(chain_obj):
                phi_psi = pp.get_phi_psi_list()
                for residue, (phi_rad, psi_rad) in zip(pp, phi_psi):
                    seq_num  = residue.get_id()[1]
                    resname  = residue.get_resname().strip()
                    phi_deg  = math.degrees(phi_rad) if phi_rad is not None else None
                    psi_deg  = math.degrees(psi_rad) if psi_rad is not None else None
                    raw_angles[seq_num] = {
                        "phi":     phi_deg,
                        "psi":     psi_deg,
                        "resname": resname,
                    }

        if not raw_angles:
            return {}

        # Try DSSP for secondary structure
        ss_map: Dict[int, str] = {}
        try:
            from Bio.PDB.DSSP import DSSP
            # DSSP needs the PDB path as a string (POSIX ok on all platforms for Bio)
            dssp = DSSP(model, pdb_path_str)
            for key in dssp:
                res_id  = key[1][1]   # sequence number
                ss_code = dssp[key][2]
                # DSSP codes: H E B G I T S C → map to H/E/L
                if ss_code in ("H", "G", "I"):
                    ss_map[res_id] = "H"
                elif ss_code in ("E", "B"):
                    ss_map[res_id] = "E"
                else:
                    ss_map[res_id] = "L"
        except Exception:
            # DSSP not available (common on Windows) — use φ/ψ fallback
            for seq_num, ang in raw_angles.items():
                phi = ang.get("phi")
                psi = ang.get("psi")
                if phi is not None and psi is not None:
                    ss_map[seq_num] = _classify_ss_from_angles(phi, psi)
                else:
                    ss_map[seq_num] = "L"

        # Merge into output dict
        result: Dict[int, Dict[str, Any]] = {}
        for seq_num, ang in raw_angles.items():
            resname = ang["resname"]
            result[seq_num] = {
                "phi":     ang["phi"],
                "psi":     ang["psi"],
                "ss":      ss_map.get(seq_num, "L"),
                "resname": resname,
                "aa":      _THREE_TO_ONE.get(resname, "X"),
            }

        return result

    # ── 2. Candidate scanning ────────────────────────────────────────────────

    def scan_proline_candidates(
        self,
        backbone:            Dict[int, Dict[str, Any]],
        sequence:            str,
        interface_residues:  Optional[Set[int]] = None,
        esm_scores:          Optional[Dict[int, float]] = None,
        functional_residues: Optional[Set[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Score and filter proline substitution candidates.

        Parameters
        ----------
        backbone             : output of extract_backbone_angles()
        sequence             : 1-letter amino-acid string (1-indexed via offset)
        interface_residues   : set of 1-based residue numbers near binding site
        esm_scores           : {position: esm_tolerance (0–1)}; higher = more tolerant
        functional_residues  : set of 1-based residue numbers that are known or
                               inferred active-site / functional residues.
                               Positions within 2 residues are hard-excluded.

        Returns
        -------
        List of candidate dicts, sorted by composite_score descending:
          {
            "position":        int,
            "from_aa":         str,   # 1-letter
            "to_aa":           "P",
            "phi":             float,
            "psi":             float,
            "ss":              str,
            "phi_score":       float,
            "loop_bonus":      float,
            "esm_factor":      float,
            "iface_factor":    float,
            "hbond_factor":    float,
            "composite_score": float,
            "confidence":      str,   # "high" / "moderate" / "low"
            "near_interface":  bool,
          }
        """
        interface_residues  = interface_residues  or set()
        esm_scores          = esm_scores          or {}
        functional_residues = functional_residues or set()

        positions   = sorted(backbone.keys())
        n_residues  = len(positions)
        pos_set     = set(positions)
        min_pos     = min(positions) if positions else 1
        max_pos     = max(positions) if positions else 1

        candidates: List[Dict[str, Any]] = []

        for i, pos in enumerate(positions):
            info    = backbone[pos]
            aa      = info.get("aa", "X")
            phi     = info.get("phi")
            psi     = info.get("psi")
            ss      = info.get("ss", "L")

            # ── Hard exclusions ───────────────────────────────────────────────
            # 1. Already Pro
            if aa == "P":
                continue
            # 2. Gly (φ is near -60 but no Cβ; Pro would not be meaningful)
            if aa == "G":
                continue
            # 3. Helix positions: Pro would disrupt i-4 H-bond
            if ss == "H":
                continue
            # 4. Pro-Pro: previous residue is already Pro
            if i > 0 and backbone.get(positions[i - 1], {}).get("aa") == "P":
                continue
            # 5. Too close to terminus (within 3 residues)
            if pos - min_pos < 3 or max_pos - pos < 3:
                continue
            # 6. No φ angle available (terminus residue from PPBuilder)
            if phi is None:
                continue
            # 7. Active-site / functional-site proximity (hard exclusion)
            #    Substituting Pro within 2 residues of a catalytic residue risks
            #    disrupting substrate binding geometry or key H-bond networks.
            if functional_residues and any(
                abs(pos - fr) <= 2 for fr in functional_residues
            ):
                continue

            # ── φ score ───────────────────────────────────────────────────────
            phi_score = max(0.0, 1.0 - abs(phi - (-60.0)) / 60.0)

            # ── Loop bonus ───────────────────────────────────────────────────
            loop_bonus = 1.3 if ss == "L" else 1.0

            # ── ESM tolerance factor ─────────────────────────────────────────
            esm_tol    = esm_scores.get(pos, 0.5)
            esm_factor = 1.0 if esm_tol >= 0.5 else 0.6

            # ── Interface factor ─────────────────────────────────────────────
            near_iface = any(
                abs(pos - ip) <= 5
                for ip in interface_residues
            )
            iface_factor = 0.4 if near_iface else 1.0

            # ── H-bond factor (cis-peptide penalty) ─────────────────────────
            # If the previous residue's ψ ∈ (-60, 60) the N-H of the current
            # residue is in a cis-like arrangement — Pro substitution would
            # break a backbone H-bond.
            hbond_factor = 1.0
            if i > 0:
                prev_psi = backbone.get(positions[i - 1], {}).get("psi")
                if prev_psi is not None and -60.0 < prev_psi < 60.0:
                    hbond_factor = 0.7

            # ── Composite score ───────────────────────────────────────────────
            composite = phi_score * loop_bonus * esm_factor * iface_factor * hbond_factor

            # ── Confidence ───────────────────────────────────────────────────
            if composite > 0.6:
                confidence = "high"
            elif composite >= 0.3:
                confidence = "moderate"
            else:
                confidence = "low"

            candidates.append({
                "position":        pos,
                "from_aa":         aa,
                "to_aa":           "P",
                "phi":             round(phi, 2),
                "psi":             round(psi, 2) if psi is not None else None,
                "ss":              ss,
                "phi_score":       round(phi_score, 4),
                "loop_bonus":      loop_bonus,
                "esm_factor":      esm_factor,
                "iface_factor":    iface_factor,
                "hbond_factor":    hbond_factor,
                "composite_score": round(composite, 4),
                "confidence":      confidence,
                "near_interface":  near_iface,
            })

        candidates.sort(key=lambda c: -c["composite_score"])
        return candidates

    # ── 3. DynaMut2 post-filter ──────────────────────────────────────────────

    def validate_with_dynamut2(
        self,
        candidates:       List[Dict[str, Any]],
        pdb_path:         str,
        chain:            str,
        top_n:            int = 5,
        dynamut2_bridge:  Any = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank the top-N candidates using DynaMut2 ddG predictions.

        Parameters
        ----------
        candidates       : list from scan_proline_candidates() (already sorted)
        pdb_path         : path to local PDB file
        chain            : chain identifier
        top_n            : number of candidates to score with DynaMut2
        dynamut2_bridge  : a RosettaBridge / DynaMut2Bridge instance with
                           .analyze(pdb_path, mutations, progress_callback) API.
                           If None, candidates are returned unchanged.
        progress_callback: optional callable for status messages

        Returns
        -------
        List of candidate dicts, each augmented with "ddg" key (kcal/mol),
        sorted by composite_score (ties broken by ddg ascending).
        """
        if dynamut2_bridge is None or not candidates:
            return candidates

        _prog = progress_callback or (lambda _: None)
        subset = candidates[:top_n]

        # Build mutation list for DynaMut2
        mutations = [
            {
                "chain":    chain,
                "position": c["position"],
                "from_aa":  c["from_aa"],
                "to_aa":    "P",
            }
            for c in subset
        ]

        try:
            _prog(f"DynaMut2: scoring {len(mutations)} proline candidate(s)…")
            result = dynamut2_bridge.analyze(
                pdb_path          = Path(pdb_path).as_posix(),
                mutations         = mutations,
                progress_callback = lambda msg: _prog(f"  {msg}"),
            )
            ddg_scores = result.data.get("ddg_scores", {})

            # Parse ddg_scores keys like "L37P"
            ddg_by_pos: Dict[int, float] = {}
            for key, ddg in ddg_scores.items():
                m = re.match(r"[A-Z](\d+)[A-Z]", str(key))
                if m:
                    ddg_by_pos[int(m.group(1))] = float(ddg)

        except Exception as exc:
            _prog(f"DynaMut2 validation failed: {exc}")
            return candidates   # fallback: return original list

        # Annotate subset with ddG
        for cand in subset:
            pos = cand["position"]
            cand["ddg"] = ddg_by_pos.get(pos, 0.0)

        # The rest of candidates (not submitted) get ddg=None
        for cand in candidates[top_n:]:
            cand.setdefault("ddg", None)

        # Re-sort: composite_score descending, then ddg ascending (lower = more stable)
        scored    = sorted(subset,             key=lambda c: (-c["composite_score"], c.get("ddg") or 0.0))
        unscored  = candidates[top_n:]
        return scored + unscored

    # ── 4. Orchestrator ──────────────────────────────────────────────────────

    def full_proline_scan(
        self,
        pdb_path:            str,
        chain:               str = "A",
        sequence:            Optional[str] = None,
        interface_residues:  Optional[Set[int]] = None,
        esm_scores:          Optional[Dict[int, float]] = None,
        top_n:               int = 5,
        dynamut2_bridge:     Any = None,
        progress_callback:   Optional[Callable[[str], None]] = None,
        functional_residues: Optional[Set[int]] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end proline substitution scan.

        Workflow:
          1. Auto-detect catalytic residues via SASA if functional_residues is None
          2. extract_backbone_angles(pdb_path, chain)
          3. scan_proline_candidates(backbone, sequence, ...)
          4. validate_with_dynamut2(...) if dynamut2_bridge is provided
          5. Return result dict

        Returns
        -------
        {
          "candidates":                 [...],   # sorted list of candidate dicts
          "count":                      int,
          "top":                        dict | None,
          "chain":                      str,
          "pdb_path":                   str,
          "n_residues_scanned":         int,
          "exclusion_counts":           dict,
          "functional_residues":        list[int],
          "inferred_functional_residues": list[int],  # auto-detected, may be []
        }
        """
        _prog = progress_callback or (lambda _: None)

        # ── Step 0: auto-detect functional residues if not provided ───────────
        inferred_functional: Set[int] = set()
        if functional_residues is None:
            try:
                inferred_functional = self._detect_functional_residues_sasa(pdb_path, chain)
            except Exception:
                inferred_functional = set()   # detection failure is non-fatal
            if inferred_functional:
                _prog(
                    f"Auto-detected {len(inferred_functional)} likely catalytic "
                    f"residue(s) from SASA (His/Asp/Glu/Ser/Cys buried <20 Å²): "
                    f"{sorted(inferred_functional)}"
                )
            functional_residues = inferred_functional  # may still be empty set

        _prog(f"Extracting backbone angles from chain {chain}…")
        backbone = self.extract_backbone_angles(pdb_path, chain)

        if not backbone:
            return {
                "candidates":                   [],
                "count":                        0,
                "top":                          None,
                "chain":                        chain,
                "pdb_path":                     Path(pdb_path).as_posix(),
                "n_residues_scanned":           0,
                "exclusion_counts":             {},
                "functional_residues":          sorted(functional_residues),
                "inferred_functional_residues": sorted(inferred_functional),
            }

        # Infer sequence from backbone if not provided
        if sequence is None:
            positions = sorted(backbone.keys())
            sequence  = "".join(backbone[p]["aa"] for p in positions)

        _prog(f"Scanning {len(backbone)} residues for proline sites…")
        candidates = self.scan_proline_candidates(
            backbone             = backbone,
            sequence             = sequence,
            interface_residues   = interface_residues,
            esm_scores           = esm_scores,
            functional_residues  = functional_residues or None,
        )

        if candidates and dynamut2_bridge is not None:
            _prog(f"Validating top {min(top_n, len(candidates))} with DynaMut2…")
            candidates = self.validate_with_dynamut2(
                candidates        = candidates,
                pdb_path          = pdb_path,
                chain             = chain,
                top_n             = top_n,
                dynamut2_bridge   = dynamut2_bridge,
                progress_callback = _prog,
            )

        exclusion_counts = self._count_exclusions(
            backbone, functional_residues=functional_residues or None
        )

        return {
            "candidates":                   candidates,
            "count":                        len(candidates),
            "top":                          candidates[0] if candidates else None,
            "chain":                        chain,
            "pdb_path":                     Path(pdb_path).as_posix(),
            "n_residues_scanned":           len(backbone),
            "exclusion_counts":             exclusion_counts,
            "functional_residues":          sorted(functional_residues),
            "inferred_functional_residues": sorted(inferred_functional),
        }

    # ── 4b. Exclusion counter ────────────────────────────────────────────────

    def _count_exclusions(
        self,
        backbone:            Dict[int, Dict[str, Any]],
        functional_residues: Optional[Set[int]] = None,
    ) -> Dict[str, int]:
        """
        Count how many backbone residues were excluded from candidacy and why.

        Uses the same priority order as scan_proline_candidates() so each
        residue is attributed to exactly one exclusion reason.

        Returns
        -------
        dict with integer counts for each exclusion category:
          existing_pro, glycine, helix, post_pro, terminal, no_phi, functional_site
        """
        positions = sorted(backbone.keys())
        if not positions:
            return {
                "existing_pro": 0, "glycine": 0, "helix": 0,
                "post_pro": 0, "terminal": 0, "no_phi": 0,
                "functional_site": 0,
            }

        min_pos = positions[0]
        max_pos = positions[-1]
        fr_set  = functional_residues or set()
        counts: Dict[str, int] = {
            "existing_pro":    0,
            "glycine":         0,
            "helix":           0,
            "post_pro":        0,
            "terminal":        0,
            "no_phi":          0,
            "functional_site": 0,
        }

        for i, pos in enumerate(positions):
            info = backbone[pos]
            aa   = info.get("aa", "X")
            phi  = info.get("phi")
            ss   = info.get("ss", "L")

            # Hard exclusions in priority order (mirrors scan_proline_candidates)
            if aa == "P":
                counts["existing_pro"] += 1
            elif aa == "G":
                counts["glycine"] += 1
            elif ss == "H":
                counts["helix"] += 1
            elif i > 0 and backbone.get(positions[i - 1], {}).get("aa") == "P":
                counts["post_pro"] += 1
            elif pos - min_pos < 3 or max_pos - pos < 3:
                counts["terminal"] += 1
            elif phi is None:
                counts["no_phi"] += 1
            elif fr_set and any(abs(pos - fr) <= 2 for fr in fr_set):
                counts["functional_site"] += 1
            # else: this residue passed all hard exclusions → it's a candidate

        return counts

    # ── 4c. SASA-based catalytic residue detection ───────────────────────────

    def _detect_functional_residues_sasa(
        self,
        pdb_path: str,
        chain:    str,
    ) -> Set[int]:
        """
        Heuristically identify buried catalytic residues via SASA.

        Residues of types His (H), Asp (D), Glu (E), Ser (S), Cys (C) —
        common charge-relay / nucleophile residues — with residue-level
        SASA < 20 Å² are considered likely active-site residues.

        This is a best-effort heuristic:
          * False positives: buried polar residues that are not catalytic.
          * False negatives: exposed active-site residues (e.g. Lys in some
            enzymes) are not detected.

        Returns an empty set on any failure (BioPython version too old,
        SASA computation error, PDB parse failure, etc.).
        """
        try:
            from Bio.PDB import PDBParser
            from Bio.PDB.SASA import ShrakeRupley  # requires BioPython ≥ 1.79

            parser    = PDBParser(QUIET=True)
            structure = parser.get_structure("prot", Path(pdb_path).as_posix())
            model     = structure[0]

            sr = ShrakeRupley()
            sr.compute(model, level="R")   # residue-level SASA in Å²

            inferred: Set[int] = set()
            for chain_obj in model:
                if chain_obj.id != chain:
                    continue
                for residue in chain_obj:
                    resname = residue.get_resname().strip()
                    aa      = _THREE_TO_ONE.get(resname, "X")
                    if aa not in _CATALYTIC_ONE_LETTER:
                        continue
                    sasa = getattr(residue, "sasa", None)
                    if sasa is not None and sasa < 20.0:
                        inferred.add(residue.get_id()[1])
            return inferred
        except Exception:
            return set()

    # ── 5. ChimeraX visualization ─────────────────────────────────────────────

    def generate_chimerax_commands(
        self,
        candidates: List[Dict[str, Any]],
        model_id:   str = "1",
        chain:      str = "A",
    ) -> Tuple[List[str], List[str]]:
        """
        Generate ChimeraX sphere + label commands for proline candidates.

        Colors:
          high     → magenta (#cc00cc)
          moderate → orange  (#cc6600)
          low      → yellow  (#cccc00)

        Returns
        -------
        (commands, explanations) — both list[str], same length.
        """
        if not candidates:
            return [], []

        color_map = {
            "high":     "#cc00cc",
            "moderate": "#cc6600",
            "low":      "#cccc00",
        }

        cmds: List[str] = []
        exps: List[str] = []

        for cand in candidates:
            pos        = cand["position"]
            from_aa    = cand["from_aa"]
            confidence = cand.get("confidence", "moderate")
            score      = cand.get("composite_score", 0.0)
            ddg        = cand.get("ddg")
            hex_color  = color_map.get(confidence, "#cc6600")
            phi        = cand.get("phi")

            spec = f"#{model_id}/{chain}:{pos}"

            # ── Sphere style for the Cα ───────────────────────────────────────
            cmds.append(f"style {spec} sphere")
            exps.append("")   # no separate display explanation for style

            # ── Color by confidence band ──────────────────────────────────────
            cmds.append(f"color {spec} {hex_color}")
            exps.append("")   # no separate display explanation for color

            # ── Label: "L36P phi=-62" (or "L36P phi=-62 -1.5" with ddG) ──────
            phi_str       = f" phi={phi:+.0f}" if phi is not None else ""
            ddg_str_label = f" {ddg:+.1f}" if ddg is not None else ""
            label_text    = f"{from_aa}{pos}P{phi_str}{ddg_str_label}"
            cmds.append(
                f'label {spec} text "{label_text}" '
                f'color {hex_color} height 0.8'
            )
            ddg_str = f", ddG={ddg:+.2f}" if ddg is not None else ""
            phi_exp = f", phi={phi:+.1f}" if phi is not None else ""
            exps.append(
                f"{from_aa}{pos}P — {confidence} confidence "
                f"(score={score:.3f}{phi_exp}{ddg_str})"
            )

        # Invariant: len(cmds) == len(exps) == 3 × len(candidates)
        return cmds, exps

    # ── Summary generator ─────────────────────────────────────────────────────

    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """
        Format a multi-line result summary for display in a Rich Panel.

        Includes:
          - total candidates found
          - exclusion breakdown (Pro, Gly, helix, post-Pro, terminal)
          - top candidates table: position, mutation, φ angle, confidence
        """
        count     = result.get("count", 0)
        chain     = result.get("chain", "A")
        n_scanned = result.get("n_residues_scanned", 0)
        excl      = result.get("exclusion_counts", {})
        cands     = result.get("candidates", [])

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────
        if count == 0:
            lines.append(f"Proline scan chain {chain}: no candidates found.")
            if n_scanned:
                lines.append(f"  Scanned {n_scanned} residues.")
            if excl:
                lines.append(self._format_exclusions(excl))
            return "\n".join(lines)

        lines.append(f"Proline scan chain {chain}: {count} candidate(s) of {n_scanned} residues scanned.")

        # ── Exclusion breakdown ───────────────────────────────────────────
        if excl:
            lines.append(self._format_exclusions(excl))

        # ── Top candidates table ──────────────────────────────────────────
        top_n  = min(5, len(cands))
        lines.append(f"\n  Top {top_n} candidates:")
        lines.append(f"    {'Mutation':<9} {'φ (°)':<9} {'Score':<8} {'Conf':<10} {'ddG'}")
        lines.append("    " + "-" * 44)
        for c in cands[:top_n]:
            mutation = f"{c['from_aa']}{c['position']}P"
            phi_str  = f"{c['phi']:+.1f}" if c.get("phi") is not None else "  n/a"
            score    = f"{c['composite_score']:.3f}"
            conf     = c.get("confidence", "?")
            ddg      = c.get("ddg")
            ddg_str  = f"{ddg:+.2f}" if ddg is not None else "—"
            lines.append(
                f"    {mutation:<9} {phi_str:<9} {score:<8} {conf:<10} {ddg_str}"
            )

        # ── Confidence breakdown ──────────────────────────────────────────
        high     = sum(1 for c in cands if c.get("confidence") == "high")
        moderate = sum(1 for c in cands if c.get("confidence") == "moderate")
        low      = sum(1 for c in cands if c.get("confidence") == "low")
        lines.append(
            f"\n  Confidence: {high} high (>0.6), "
            f"{moderate} moderate (0.3–0.6), "
            f"{low} low (<0.3)"
        )

        return "\n".join(lines)

    @staticmethod
    def _format_exclusions(excl: Dict[str, int]) -> str:
        """Format the exclusion count dict into a compact human-readable line."""
        parts = []
        if excl.get("existing_pro"):
            parts.append(f"{excl['existing_pro']} existing Pro")
        if excl.get("glycine"):
            parts.append(f"{excl['glycine']} Gly")
        if excl.get("helix"):
            parts.append(f"{excl['helix']} helix")
        if excl.get("post_pro"):
            parts.append(f"{excl['post_pro']} post-Pro")
        if excl.get("terminal"):
            parts.append(f"{excl['terminal']} terminal")
        if excl.get("no_phi"):
            parts.append(f"{excl['no_phi']} no-φ")
        if excl.get("functional_site"):
            parts.append(f"{excl['functional_site']} active-site-proximal")
        if not parts:
            return "  Excluded: none"
        return "  Excluded: " + ", ".join(parts)
