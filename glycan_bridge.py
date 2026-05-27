"""
glycan_bridge.py
----------------
Full N-linked glycosylation site analysis for StructureBot.

Detects canonical NXS/T sequons (N-X-S/T, X ≠ P), scores them by:
  - Surface exposure  (SASA via freesasa or BioPython ShrakeRupley)
  - Secondary structure (DSSP or Ramachandran φ/ψ fallback)
  - Interface proximity
  - ESM-2 positional tolerance
  - Composite score = sasa × loop_factor × interface_factor × esm_factor

Also suggests engineered sequon insertions on exposed loop residues.

Public API
----------
scan_sequons(sequence, chain)
score_sequon_sites(sequon_list, pdb_path, interface_residues, esm_scores)
suggest_engineered_sequons(scored_sites, sequence, chain, pdb_path, esm_scores, top_n)
full_glycan_scan(pdb_path, chain, sequence, interface_residues, esm_scores, min_score, top_n)
analyze(pdb_path, chain, sequence, model_id, session, ...)
generate_chimerax_commands(candidates, model_id, chain)
"""

from __future__ import annotations

import math
import re
import traceback as _traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Glycosylation sequon pattern ──────────────────────────────────────────────
# N-X-S/T where X ≠ P  (canonical NXS/T sequon)
_SEQUON_RE = re.compile(r"N[^P][ST]")

# ── ChimeraX color constants ──────────────────────────────────────────────────
_COLOR_NATIVE   = "#00cc00"   # green  — native sequon, high exposure
_COLOR_ENG_HIGH = "#00cccc"   # cyan   — engineered, high confidence
_COLOR_ENG_MOD  = "#cccc00"   # yellow — moderate / low confidence

# ── Ramachandran angle boundaries for SS classification ───────────────────────
_HELIX_PHI       = (-80.0, -40.0)
_HELIX_PSI       = (-60.0, -20.0)
_SHEET_PHI       = (-180.0, -90.0)
_SHEET_PSI_ABS   = (100.0, 180.0)


# ── Private helpers ───────────────────────────────────────────────────────────

def _get_sasa(pdb_path: str, chain: str, residue_numbers: List[int]) -> Dict[int, float]:
    """
    Return per-residue relative SASA (0–1) for *residue_numbers*.

    Tries freesasa first; falls back to BioPython ShrakeRupley.
    Returns {} on complete failure — caller substitutes 0.5.
    """
    path = Path(pdb_path)
    if not path.exists():
        return {}

    # ── freesasa ──────────────────────────────────────────────────────────────
    try:
        import freesasa  # type: ignore
        structure = freesasa.Structure(str(path))
        result    = freesasa.calc(structure)
        area_map: Dict[int, float] = {}
        for i in range(structure.nAtoms()):
            raw = structure.residueNumber(i)
            try:
                res_n = int(str(raw).strip())
            except (ValueError, AttributeError):
                continue
            if res_n not in residue_numbers:
                continue
            area_map[res_n] = area_map.get(res_n, 0.0) + result.atomArea(i)
        # Normalise: ~300 Å² = fully exposed residue
        if area_map:
            return {r: min(1.0, a / 300.0) for r, a in area_map.items()}
    except Exception:
        pass

    # ── BioPython ShrakeRupley fallback ───────────────────────────────────────
    try:
        from Bio.PDB import PDBParser  # type: ignore
        from Bio.PDB.SASA import ShrakeRupley  # type: ignore
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", str(path))
        sr = ShrakeRupley()
        sr.compute(struct, level="R")
        sasa_map: Dict[int, float] = {}
        for model in struct:
            for ch in model:
                if ch.id != chain:
                    continue
                for res in ch:
                    rn = res.id[1]
                    if rn in residue_numbers:
                        sasa_map[rn] = min(1.0, res.sasa / 300.0)
        return sasa_map
    except Exception:
        return {}


def _get_backbone_angles(pdb_path: str, chain: str) -> Dict[int, Tuple[float, float]]:
    """
    Return {residue_number: (phi_deg, psi_deg)} via BioPython PPBuilder.
    Returns {} if BioPython is not installed or the PDB is unreadable.
    """
    try:
        from Bio.PDB import PDBParser, PPBuilder  # type: ignore
        parser  = PDBParser(QUIET=True)
        struct  = parser.get_structure("s", str(pdb_path))
        builder = PPBuilder()
        angles: Dict[int, Tuple[float, float]] = {}
        for pp in builder.build_peptides(struct[0][chain]):
            for res, (phi, psi) in zip(pp, pp.get_phi_psi_list()):
                if phi is not None and psi is not None:
                    angles[res.id[1]] = (math.degrees(phi), math.degrees(psi))
        return angles
    except Exception:
        return {}


def _classify_ss_from_angles(phi: float, psi: float) -> str:
    """
    Classify secondary structure from Ramachandran angles.
    Returns 'H' (helix), 'E' (sheet), or 'L' (loop/coil).
    """
    if (_HELIX_PHI[0] <= phi <= _HELIX_PHI[1] and
            _HELIX_PSI[0] <= psi <= _HELIX_PSI[1]):
        return "H"
    if (_SHEET_PHI[0] <= phi <= _SHEET_PHI[1] and
            _SHEET_PSI_ABS[0] <= abs(psi) <= _SHEET_PSI_ABS[1]):
        return "E"
    return "L"


def _get_secondary_structure(pdb_path: str, chain: str) -> Dict[int, str]:
    """
    Return {residue_number: ss_code} where ss_code ∈ {'H', 'E', 'L'}.

    Tries DSSP first; falls back to Ramachandran classification.
    """
    # ── DSSP ──────────────────────────────────────────────────────────────────
    try:
        from Bio.PDB import PDBParser, DSSP  # type: ignore
        parser = PDBParser(QUIET=True)
        struct = parser.get_structure("s", str(pdb_path))
        dssp   = DSSP(struct[0], str(pdb_path))
        ss_map: Dict[int, str] = {}
        for key in dssp.keys():
            if key[0] != chain:
                continue
            res_n = key[1][1]
            code  = dssp[key][2]
            if code in ("H", "G", "I"):
                ss_map[res_n] = "H"
            elif code in ("E", "B"):
                ss_map[res_n] = "E"
            else:
                ss_map[res_n] = "L"
        if ss_map:
            return ss_map
    except Exception:
        pass

    # ── Ramachandran fallback ─────────────────────────────────────────────────
    angles = _get_backbone_angles(pdb_path, chain)
    return {
        rn: _classify_ss_from_angles(phi, psi)
        for rn, (phi, psi) in angles.items()
    }


def _compute_composite_score(
    sasa:             float,
    ss:               str,
    interface_factor: float,
    esm_factor:       float,
) -> float:
    """
    Composite glycan-site score.

      loop_factor = 1.0 (loop) | 0.5 (helix) | 0.3 (sheet)
      score = sasa × loop_factor × interface_factor × esm_factor
    """
    loop_factor = {"L": 1.0, "H": 0.5, "E": 0.3}.get(ss, 1.0)
    return round(sasa * loop_factor * interface_factor * esm_factor, 4)


def _classify_confidence(score: float) -> str:
    if score >= 0.4:
        return "high"
    if score >= 0.2:
        return "moderate"
    return "low"


# ── GlycanBridge class ────────────────────────────────────────────────────────

class GlycanBridge:
    """
    Detect and score N-linked glycosylation sites in protein sequences.

    All public methods are stateless; construct once and reuse.
    """

    # ── scan_sequons ───────────────────────────────────────────────────────────

    def scan_sequons(
        self,
        sequence: str,
        chain:    str = "A",
    ) -> List[Dict[str, Any]]:
        """
        Find all NXS/T sequons in *sequence* (1-indexed position of N).

        Returns a list of dicts: {position, sequon, chain}
        """
        sites: List[Dict[str, Any]] = []
        for m in _SEQUON_RE.finditer(sequence):
            sites.append({
                "position": m.start() + 1,
                "sequon":   m.group(),
                "chain":    chain,
            })
        return sites

    # ── score_sequon_sites ─────────────────────────────────────────────────────

    def score_sequon_sites(
        self,
        sequon_list:        List[Dict[str, Any]],
        pdb_path:           Optional[str] = None,
        interface_residues: Optional[List[int]] = None,
        esm_scores:         Optional[Dict[int, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich *sequon_list* with structural and tolerance scores.

        Adds per-site keys:
          sasa, secondary_structure, interface_proximity,
          esm_tolerance, composite_score, confidence
        """
        if not sequon_list:
            return []

        interface_residues = interface_residues or []
        esm_scores         = esm_scores         or {}
        chain = sequon_list[0].get("chain", "A")
        positions = [s["position"] for s in sequon_list]

        # Structural data — only when a valid PDB is available
        sasa_map: Dict[int, float] = {}
        ss_map:   Dict[int, str]   = {}
        if pdb_path and Path(pdb_path).exists():
            sasa_map = _get_sasa(pdb_path, chain, positions)
            ss_map   = _get_secondary_structure(pdb_path, chain)

        scored: List[Dict[str, Any]] = []
        for site in sequon_list:
            pos  = site["position"]
            sasa = sasa_map.get(pos, 0.5)
            ss   = ss_map.get(pos, "L")

            near_interface   = any(abs(pos - ir) <= 5 for ir in interface_residues)
            interface_factor = 0.5 if near_interface else 1.0
            esm_tol          = esm_scores.get(pos, 1.0)

            composite  = _compute_composite_score(sasa, ss, interface_factor, esm_tol)
            confidence = _classify_confidence(composite)

            scored.append({
                **site,
                "sasa":                round(sasa, 4),
                "secondary_structure": ss,
                "interface_proximity": near_interface,
                "esm_tolerance":       round(esm_tol, 4),
                "composite_score":     composite,
                "confidence":          confidence,
            })

        scored.sort(key=lambda s: -s["composite_score"])
        return scored

    # ── suggest_engineered_sequons ─────────────────────────────────────────────

    def suggest_engineered_sequons(
        self,
        scored_sites:       List[Dict[str, Any]],
        sequence:           str,
        chain:              str = "A",
        pdb_path:           Optional[str] = None,
        esm_scores:         Optional[Dict[int, float]] = None,
        top_n:              int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Suggest single-AA mutations that would create a new NXS/T sequon.

        Strategy: for each position i that is NOT already a native sequon N,
        check if mutating aa[i] → N would form N-aa[i+1]-aa[i+2] where
        aa[i+1] ≠ P and aa[i+2] ∈ {S, T}.  One-mutation candidates only.

        Returns the top_n candidates sorted by composite_score.
        """
        esm_scores        = esm_scores or {}
        existing_positions = {s["position"] for s in scored_sites}

        candidates: List[Dict[str, Any]] = []
        for i in range(len(sequence) - 2):
            pos        = i + 1            # 1-indexed
            aa0, aa1, aa2 = sequence[i], sequence[i + 1], sequence[i + 2]

            # Skip proline at X (would violate sequon rule)
            if aa1 == "P":
                continue
            # Skip existing native sequons
            if pos in existing_positions:
                continue
            # Skip if already N
            if aa0 == "N":
                continue
            # Require S or T at position i+2 (one-mutation only)
            if aa2 not in "ST":
                continue

            mutation   = f"{aa0}{pos}N"
            new_sequon = f"N{aa1}{aa2}"
            candidates.append({
                "position":   pos,
                "sequon":     new_sequon,
                "mutation":   mutation,
                "chain":      chain,
                "engineered": True,
            })

        if not candidates:
            return []

        # Score using same pipeline
        if pdb_path and Path(pdb_path).exists():
            candidates = self.score_sequon_sites(
                candidates, pdb_path, [], esm_scores
            )
        else:
            # No PDB — assign defaults; apply ESM factor if available
            for c in candidates:
                esm_tol   = esm_scores.get(c["position"], 1.0)
                composite = _compute_composite_score(0.5, "L", 1.0, esm_tol)
                c.update({
                    "sasa":                0.5,
                    "secondary_structure": "L",
                    "interface_proximity": False,
                    "esm_tolerance":       round(esm_tol, 4),
                    "composite_score":     composite,
                    "confidence":          _classify_confidence(composite),
                })

        candidates.sort(key=lambda c: -c.get("composite_score", 0.0))
        return candidates[:top_n]

    # ── full_glycan_scan ───────────────────────────────────────────────────────

    def full_glycan_scan(
        self,
        pdb_path:           Optional[str] = None,
        chain:              str = "A",
        sequence:           str = "",
        interface_residues: Optional[List[int]] = None,
        esm_scores:         Optional[Dict[int, float]] = None,
        min_score:          float = 0.05,
        top_n:              int = 3,
    ) -> Dict[str, Any]:
        """
        Full N-glycosylation pipeline.

        Returns
        -------
        dict:
          success, chain, pdb_path,
          native_sequons        — all NXS/T sequons with composite_score ≥ min_score
          engineered_candidates — top_n single-mutation engineering proposals
          all_ranked            — all native sequons (including below min_score)
          top_n                 — number of engineered candidates returned
          error                 — None on success
        """
        if not sequence:
            return {
                "success":               False,
                "error":                 "No sequence provided",
                "chain":                 chain,
                "pdb_path":              str(pdb_path) if pdb_path else None,
                "native_sequons":        [],
                "engineered_candidates": [],
                "all_ranked":            [],
                "top_n":                 0,
            }

        try:
            raw_sequons    = self.scan_sequons(sequence, chain)
            scored_sequons = self.score_sequon_sites(
                raw_sequons, pdb_path, interface_residues, esm_scores
            )
            native_sequons = [
                s for s in scored_sequons if s["composite_score"] >= min_score
            ]
            eng_candidates = self.suggest_engineered_sequons(
                scored_sequons, sequence, chain, pdb_path, esm_scores, top_n
            )

            return {
                "success":               True,
                "chain":                 chain,
                "pdb_path":              str(pdb_path) if pdb_path else None,
                "native_sequons":        native_sequons,
                "engineered_candidates": eng_candidates,
                "all_ranked":            scored_sequons,
                "top_n":                 len(eng_candidates),
                "error":                 None,
            }

        except Exception as exc:
            return {
                "success":               False,
                "error":                 f"{type(exc).__name__}: {exc}",
                "trace":                 _traceback.format_exc(),
                "chain":                 chain,
                "pdb_path":              str(pdb_path) if pdb_path else None,
                "native_sequons":        [],
                "engineered_candidates": [],
                "all_ranked":            [],
                "top_n":                 0,
            }

    # ── analyze ────────────────────────────────────────────────────────────────

    def analyze(
        self,
        pdb_path:           Optional[str] = None,
        chain:              str = "A",
        sequence:           str = "",
        model_id:           str = "1",
        session:            Any = None,
        interface_residues: Optional[List[int]] = None,
        esm_scores:         Optional[Dict[int, float]] = None,
        min_score:          float = 0.05,
        top_n:              int = 3,
        **kwargs:           Any,
    ) -> Dict[str, Any]:
        """
        Full analysis entry point.  Returns a plain dict for tool_router.

        Adds keys to full_glycan_scan output:
          summary, viz_commands, viz_explanations
        """
        result = self.full_glycan_scan(
            pdb_path           = pdb_path,
            chain              = chain,
            sequence           = sequence,
            interface_residues = interface_residues,
            esm_scores         = esm_scores,
            min_score          = min_score,
            top_n              = top_n,
        )

        if not result.get("success"):
            return {
                **result,
                "summary":               f"Glycan scan failed: {result.get('error', '?')}",
                "chimerax_commands":     [],
                "chimerax_explanations": [],
            }

        native     = result["native_sequons"]
        eng        = result["engineered_candidates"]
        all_ranked = result["all_ranked"]   # all sequons, no min_score filter

        # ── Visualization candidate list ───────────────────────────────────────
        # Use all_ranked (unfiltered by min_score) so proteins with no
        # N-glycosylation sequons still show engineered candidates in ChimeraX.
        # Engineered candidates are added where they don't overlap native sites.
        ranked_positions = {s["position"] for s in all_ranked}
        all_viz = all_ranked + [c for c in eng if c["position"] not in ranked_positions]

        cx_cmds, cx_exps = self.generate_chimerax_commands(
            all_viz, model_id=model_id, chain=chain
        )

        # ── Multi-line summary (triggers Rich Panel in main.py) ───────────────
        header = f"Glycan scan — chain {chain}"
        lines: List[str] = [header, ""]
        if native:
            lines.append(f"Native sequons ({len(native)} found):")
            for s in native[:5]:
                ss    = s.get("secondary_structure", "L")
                lines.append(
                    f"  N{s['position']} {s['sequon']}  {s['confidence']}"
                    f"  score={s['composite_score']:.3f}  SS={ss}"
                )
            if len(native) > 5:
                lines.append(f"  … and {len(native) - 5} more")
        elif all_ranked:
            lines.append(
                f"No native sequons above score threshold "
                f"({len(all_ranked)} detected total, all below min_score)."
            )
        else:
            lines.append("No native NXS/T sequons detected.")

        if eng:
            lines.append("")
            lines.append(f"Engineered candidates ({len(eng)} proposed):")
            for e in eng:
                mut = e.get("mutation", "?")
                seq = e.get("sequon", "")
                lines.append(
                    f"  {mut} → {seq}  {e['confidence']}"
                    f"  score={e['composite_score']:.3f}"
                )

        summary = "\n".join(lines)

        # Persist to session state when a session object is provided
        if session is not None:
            try:
                session.set_glycan_results(model_id, chain, result)
            except Exception:
                pass

        return {
            **result,
            "summary":               summary,
            "chimerax_commands":     cx_cmds,
            "chimerax_explanations": cx_exps,
        }

    # ── generate_chimerax_commands ─────────────────────────────────────────────

    def generate_chimerax_commands(
        self,
        candidates: List[Dict[str, Any]],
        model_id:   str = "1",
        chain:      str = "A",
    ) -> Tuple[List[str], List[str]]:
        """
        Generate 3 ChimeraX commands per candidate + 1 explanation per candidate.

        Commands per site:
          1. color target ar
          2. show atoms
          3. label with site ID (and mutation if engineered)

        Color logic:
          native + high     → #00cc00 (green)
          engineered + high → #00cccc (cyan)
          moderate / low    → #cccc00 (yellow)
        """
        if not candidates:
            return [], []

        cmds: List[str] = []
        exps: List[str] = []

        for cand in candidates:
            pos        = cand["position"]
            confidence = cand.get("confidence", "moderate")
            engineered = bool(cand.get("engineered", False))
            score      = cand.get("composite_score", cand.get("score", 0.0))

            if confidence == "high" and not engineered:
                hex_color = _COLOR_NATIVE
            elif confidence == "high" and engineered:
                hex_color = _COLOR_ENG_HIGH
            else:
                hex_color = _COLOR_ENG_MOD

            spec = f"#{model_id}/{chain}:{pos}"
            cmds.append(f"color {spec} {hex_color} target ar")
            cmds.append(f"show {spec} atoms")

            if engineered:
                mut        = cand.get("mutation", "?")
                label_text = f"N{pos}({mut})"
            else:
                label_text = f"N{pos}({confidence[:3]})"

            cmds.append(
                f'label {spec} text "{label_text}" '
                f'color {hex_color} height 1.0'
            )

            kind = "Engineered" if engineered else "Native"
            exps.append(
                f"{kind} glycosylation site N{pos} — {confidence} confidence "
                f"(score={score:.3f})"
            )

        return cmds, exps
