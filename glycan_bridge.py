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

# Shared structural utilities — imported lazily so the module loads even if the
# file has not yet been created (graceful degradation).
try:
    import structural_utils as _su
except ImportError:          # pragma: no cover
    _su = None  # type: ignore[assignment]

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

    Delegates to structural_utils.compute_sasa() (which tries freesasa then
    BioPython ShrakeRupley) and normalises raw Å² values to the 0–1 range
    using a 300 Å² reference for a fully exposed residue.

    Returns {} on complete failure — caller substitutes 0.5.
    """
    path = Path(pdb_path)
    if not path.exists():
        return {}

    raw: Dict[int, float] = {}
    if _su is not None:
        try:
            raw = _su.compute_sasa(pdb_path, chain)
        except Exception:
            pass

    if not raw:
        # Legacy fallback: freesasa / BioPython ShrakeRupley directly
        try:
            import freesasa  # type: ignore
            structure = freesasa.Structure(str(path))
            result    = freesasa.calc(structure)
            area_map: Dict[int, float] = {}
            for i in range(structure.nAtoms()):
                _raw = structure.residueNumber(i)
                try:
                    res_n = int(str(_raw).strip())
                except (ValueError, AttributeError):
                    continue
                if res_n not in residue_numbers:
                    continue
                area_map[res_n] = area_map.get(res_n, 0.0) + result.atomArea(i)
            if area_map:
                raw = area_map
        except Exception:
            pass

    if not raw:
        try:
            from Bio.PDB import PDBParser  # type: ignore
            from Bio.PDB.SASA import ShrakeRupley  # type: ignore
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure("s", str(path))
            sr = ShrakeRupley()
            sr.compute(struct, level="R")
            for model in struct:
                for ch in model:
                    if ch.id != chain:
                        continue
                    for res in ch:
                        rn = res.id[1]
                        if rn in residue_numbers:
                            raw[rn] = getattr(res, "sasa", 0.0) * 300.0  # un-normalised for consistency
        except Exception:
            return {}

    if not raw:
        return {}

    # Normalise: ~300 Å² = fully exposed residue
    return {r: min(1.0, raw[r] / 300.0) for r in residue_numbers if r in raw}


def _get_backbone_angles(pdb_path: str, chain: str) -> Dict[int, Tuple[float, float]]:
    """
    Return {residue_number: (phi_deg, psi_deg)} via structural_utils.

    Thin wrapper — delegates to structural_utils.extract_backbone_angles()
    and reformats the result into the legacy (phi, psi) tuple format.
    Returns {} on any failure.
    """
    if _su is not None:
        try:
            backbone = _su.extract_backbone_angles(pdb_path, chain)
            return {
                resno: (data["phi"], data["psi"])
                for resno, data in backbone.items()
                if data.get("phi") is not None and data.get("psi") is not None
            }
        except Exception:
            pass

    # Legacy fallback (structural_utils unavailable)
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
        projection_scores:  Optional[Dict[int, Dict[str, Any]]] = None,
        backbone:           Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Enrich *sequon_list* with structural and tolerance scores.

        Adds per-site keys:
          sasa, secondary_structure, interface_proximity,
          esm_tolerance, composite_score, confidence,
          projection_score, projection_category,
          sequon_geometry, sequon_geometry_factor

        Parameters
        ----------
        sequon_list        : from scan_sequons() or suggest_engineered_sequons()
        pdb_path           : local PDB path for SASA / secondary-structure lookups
        interface_residues : 1-based residue numbers near binding interface
        esm_scores         : {position: esm_tolerance (0–1)}
        projection_scores  : from structural_utils.compute_projection_score()
                             {resno: {"projection_score": float, "gly_proxy": bool}}
        backbone           : from structural_utils.extract_backbone_angles()
                             {resno: {"phi": float, "psi": float, ...}}
                             When provided, sequon_geo_factor replaces loop_factor
                             in the composite score formula.

        Projection categories
        ---------------------
        "outward"  : score ≥ 0.6
        "flat"     : 0.2 ≤ score < 0.6
        "inward"   : score < 0.2
        "unknown"  : projection_scores not supplied, position absent, or Gly proxy

        Sequon geometry factors
        -----------------------
        "beta_turn" : 1.4  (glycans enriched at β-turns)
        "loop"      : 1.2
        "extended"  : 1.0
        "helix"     : 0.5  (sterically problematic)
        "unknown"   : 1.0

        Composite score formula
        -----------------------
        When *backbone* is supplied:
          composite = sasa × proj_factor × geom_factor × iface_factor × esm_factor
        Otherwise (backward-compatible):
          composite = sasa × proj_factor × loop_factor × iface_factor × esm_factor

        Where proj_factor = projection_score (if outward/flat/inward) else 1.0.
        """
        if not sequon_list:
            return []

        interface_residues = interface_residues or []
        esm_scores         = esm_scores         or {}
        chain     = sequon_list[0].get("chain", "A")
        positions = [s["position"] for s in sequon_list]

        # Structural data — only when a valid PDB is available
        sasa_map: Dict[int, float] = {}
        ss_map:   Dict[int, str]   = {}
        if pdb_path and Path(pdb_path).exists():
            sasa_map = _get_sasa(pdb_path, chain, positions)
            ss_map   = _get_secondary_structure(pdb_path, chain)

        # Sequon geometry factor lookup
        _GEO_FACTORS: Dict[str, float] = {
            "beta_turn": 1.4,
            "loop":      1.2,
            "extended":  1.0,
            "helix":     0.5,
            "unknown":   1.0,
        }

        scored: List[Dict[str, Any]] = []
        for site in sequon_list:
            pos  = site["position"]
            sasa = sasa_map.get(pos, 0.5)
            ss   = ss_map.get(pos, "L")

            near_interface   = any(abs(pos - ir) <= 5 for ir in interface_residues)
            interface_factor = 0.5 if near_interface else 1.0
            esm_tol          = esm_scores.get(pos, 1.0)

            # ── Projection score ──────────────────────────────────────────────
            proj_entry = (projection_scores or {}).get(pos)
            if proj_entry is not None and not proj_entry.get("gly_proxy", False):
                proj_score = proj_entry["projection_score"]
                if proj_score >= 0.6:
                    proj_cat = "outward"
                elif proj_score >= 0.2:
                    proj_cat = "flat"
                else:
                    proj_cat = "inward"
                proj_factor: float = proj_score
            else:
                proj_score = None
                proj_cat   = "unknown"
                proj_factor = 1.0

            # ── Sequon geometry ───────────────────────────────────────────────
            if backbone is not None and _su is not None:
                try:
                    geom = _su.classify_sequon_geometry(backbone, pos)
                except Exception:
                    geom = "unknown"
            elif backbone is not None:
                # structural_utils unavailable — simple φ/ψ classification fallback
                geom = "unknown"
            else:
                geom = "unknown"
            geom_factor: float = _GEO_FACTORS.get(geom, 1.0)

            # ── Composite score ───────────────────────────────────────────────
            if backbone is not None:
                composite = round(
                    sasa * proj_factor * geom_factor * interface_factor * esm_tol, 3
                )
            else:
                loop_factor = {"L": 1.0, "H": 0.5, "E": 0.3}.get(ss, 1.0)
                composite   = round(
                    sasa * proj_factor * loop_factor * interface_factor * esm_tol, 3
                )

            confidence = _classify_confidence(composite)

            scored.append({
                **site,
                "sasa":                   round(sasa, 4),
                "secondary_structure":    ss,
                "interface_proximity":    near_interface,
                "esm_tolerance":          round(esm_tol, 4),
                "composite_score":        composite,
                "confidence":             confidence,
                # ── new fields ──────────────────────────────────────────────
                "projection_score":        proj_score,
                "projection_category":     proj_cat,
                "sequon_geometry":         geom,
                "sequon_geometry_factor":  geom_factor,
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
        projection_scores:  Optional[Dict[int, Dict[str, Any]]] = None,
        backbone:           Optional[Dict[int, Dict[str, Any]]] = None,
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
                candidates, pdb_path, [], esm_scores,
                projection_scores=projection_scores,
                backbone=backbone,
            )
        else:
            # No PDB — assign defaults; apply ESM factor if available
            for c in candidates:
                esm_tol   = esm_scores.get(c["position"], 1.0)
                composite = _compute_composite_score(0.5, "L", 1.0, esm_tol)
                c.update({
                    "sasa":                   0.5,
                    "secondary_structure":     "L",
                    "interface_proximity":     False,
                    "esm_tolerance":           round(esm_tol, 4),
                    "composite_score":         composite,
                    "confidence":              _classify_confidence(composite),
                    "projection_score":        None,
                    "projection_category":     "unknown",
                    "sequon_geometry":         "unknown",
                    "sequon_geometry_factor":  1.0,
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
            # ── Compute projection and backbone geometry (graceful fallback) ──
            projection_scores: Optional[Dict[int, Dict[str, Any]]] = None
            backbone:          Optional[Dict[int, Dict[str, Any]]] = None
            if pdb_path and Path(pdb_path).exists() and _su is not None:
                try:
                    _proj = _su.compute_projection_score(pdb_path, chain)
                    if _proj:
                        projection_scores = _proj
                except Exception:
                    pass
                try:
                    _bb = _su.extract_backbone_angles(pdb_path, chain)
                    if _bb:
                        backbone = _bb
                except Exception:
                    pass

            raw_sequons    = self.scan_sequons(sequence, chain)
            scored_sequons = self.score_sequon_sites(
                raw_sequons, pdb_path, interface_residues, esm_scores,
                projection_scores=projection_scores,
                backbone=backbone,
            )
            native_sequons = [
                s for s in scored_sequons if s["composite_score"] >= min_score
            ]
            eng_candidates = self.suggest_engineered_sequons(
                scored_sequons, sequence, chain, pdb_path, esm_scores, top_n,
                projection_scores=projection_scores,
                backbone=backbone,
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
            # Column header
            HDR = (f"  {'Mutation':<10}{'Sequon':<8}{'Projection':<12}"
                   f"{'Geometry':<12}{'Score':>6}  {'Conf'}")
            SEP = "  " + "-" * (len(HDR) - 2)
            lines.append(HDR)
            lines.append(SEP)
            for e in eng:
                mut      = e.get("mutation", "?")
                seq      = e.get("sequon", "")
                proj     = e.get("projection_category", "unknown")
                geom     = e.get("sequon_geometry", "unknown")
                score    = e.get("composite_score", 0.0)
                conf     = e.get("confidence", "?")
                lines.append(
                    f"  {mut:<10}{seq:<8}{proj:<12}{geom:<12}{score:>6.3f}  {conf}"
                )
            lines.append("")
            lines.append(
                "  Legend: projection = side chain direction vs solvent"
            )
            lines.append(
                "          geometry   = local backbone at sequon positions i/i+1/i+2"
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

    # ── suggest_glycosylation_positions ───────────────────────────────────────

    def suggest_glycosylation_positions(
        self,
        pdb_path:           Optional[str] = None,
        chain:              str = "A",
        sequence:           str = "",
        interface_residues: Optional[Any] = None,    # set[int] | list[int] | None
        esm_scores:         Optional[Dict[int, float]] = None,
        min_projection:     float = 0.5,
        min_sasa:           float = 50.0,
        exclude_residues:   str = "GP",
        top_n:              int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Projection-aware scan of ALL residues for glycosylation suitability.

        Unlike scan_sequons() (which only looks at existing NXS/T motifs),
        this evaluates every surface-exposed, outward-projecting residue as a
        potential glycosylation engineering target.

        Composite formula
        -----------------
        projection_factor = proj_score  (if proj_score ≥ min_projection else skip)
        sasa_factor       = min(sasa_A2 / 150.0, 1.0)  (0.5 default when no data)
        ss_factor         = 1.3 (loop) | 0.7 (helix/sheet) | 1.0 (unknown)
        esm_factor        = esm_scores[pos] if available else 1.0
        interface_factor  = 0.3 if interface_proximal else 1.0
        composite         = round(proj_factor × sasa_factor × ss_factor
                                  × esm_factor × interface_factor, 3)

        Returns
        -------
        List of dicts sorted by composite_score descending, length ≤ top_n.
        Each dict: position, chain, wildtype_residue, projection_score,
                   projection_category, sasa, sasa_category,
                   secondary_structure, esm_tolerance, composite_score,
                   confidence, interface_proximal, engineering_notes.
        """
        if not sequence:
            return []

        esm_scores         = esm_scores or {}
        iface_set: set     = set(interface_residues or [])

        # ── Gather structural data (graceful fallback) ─────────────────────────
        has_pdb = bool(pdb_path and Path(pdb_path).exists() and _su is not None)

        projection_data: Dict[int, Dict[str, Any]] = {}
        sasa_data:       Dict[int, float]           = {}
        ss_map:          Dict[int, str]             = {}

        if has_pdb:
            try:
                projection_data = _su.compute_projection_score(pdb_path, chain) or {}
            except Exception:
                pass
            try:
                sasa_data = _su.compute_sasa(pdb_path, chain) or {}
            except Exception:
                pass
            try:
                ss_map = _get_secondary_structure(pdb_path, chain)
            except Exception:
                pass

        candidates: List[Dict[str, Any]] = []

        for i, aa in enumerate(sequence):
            pos = i + 1  # 1-indexed

            # Skip excluded amino-acid types (default: G and P)
            if aa in exclude_residues:
                continue

            # ── Projection score ──────────────────────────────────────────────
            proj_entry = projection_data.get(pos)
            if proj_entry is not None and not proj_entry.get("gly_proxy", False):
                proj_score: Optional[float] = proj_entry["projection_score"]
                # Filter residues whose side chains point inward
                if proj_score < min_projection:
                    continue
                if proj_score >= 0.6:
                    proj_cat = "outward"
                elif proj_score >= 0.2:
                    proj_cat = "flat"
                else:
                    proj_cat = "inward"
                proj_factor: float = proj_score
            else:
                # No projection data — neutral; never filter on missing data
                proj_score  = None
                proj_cat    = "unknown"
                proj_factor = 1.0

            # ── SASA (raw Å²) ─────────────────────────────────────────────────
            raw_sasa = sasa_data.get(pos)
            if raw_sasa is not None:
                if raw_sasa < min_sasa:
                    continue   # too buried — skip
                sasa        = raw_sasa
                sasa_factor = min(raw_sasa / 150.0, 1.0)
                if raw_sasa >= 100.0:
                    sasa_cat = "exposed"
                elif raw_sasa >= 50.0:
                    sasa_cat = "partial"
                else:
                    sasa_cat = "buried"
            else:
                # No data — conservative neutral; never filter
                sasa        = None
                sasa_factor = 0.5
                sasa_cat    = "unknown"

            # ── Secondary structure ───────────────────────────────────────────
            ss_code = ss_map.get(pos, "L")
            ss_str  = {"L": "loop", "H": "helix", "E": "sheet"}.get(ss_code, "unknown")
            ss_factor = {"L": 1.3, "H": 0.7, "E": 0.7}.get(ss_code, 1.0)

            # ── ESM tolerance ─────────────────────────────────────────────────
            esm_tol = esm_scores.get(pos, 1.0)

            # ── Interface proximity ───────────────────────────────────────────
            interface_proximal = pos in iface_set
            interface_factor   = 0.3 if interface_proximal else 1.0

            # ── Composite score ───────────────────────────────────────────────
            composite  = round(
                proj_factor * sasa_factor * ss_factor * esm_tol * interface_factor,
                3,
            )
            confidence = _classify_confidence(composite)

            notes = self._generate_engineering_notes(
                pos, sequence, proj_score, proj_cat,
                sasa, sasa_cat, ss_code, ss_str, ss_map,
            )

            candidates.append({
                "position":           pos,
                "chain":              chain,
                "wildtype_residue":   aa,
                "projection_score":   proj_score,
                "projection_category": proj_cat,
                "sasa":               sasa,
                "sasa_category":      sasa_cat,
                "secondary_structure": ss_code,
                "esm_tolerance":      round(esm_tol, 4),
                "composite_score":    composite,
                "confidence":         confidence,
                "interface_proximal": interface_proximal,
                "engineering_notes":  notes,
            })

        candidates.sort(key=lambda c: -c["composite_score"])
        return candidates[:top_n]

    # ── validate_sequon_engineering ───────────────────────────────────────────

    def validate_sequon_engineering(
        self,
        position:          int,
        wildtype_sequence: str,
        mutations:         List[Any],     # list of "A5N" strings or dicts
        pdb_path:          str = "",
        chain:             str = "A",
        esmfold_bridge:    Any = None,
        rosetta_bridge:    Any = None,
        top_esm_designs:   int = 3,
    ) -> Dict[str, Any]:
        """
        Validate glycosylation sequon engineering at *position*.

        Runs ESMFold on the wildtype and each mutant (up to top_esm_designs),
        compares pLDDT, and calls Rosetta ddG for stability impact.

        Pass threshold: pLDDT drop ≤ 5 AND ddG ≤ 1.0 kcal/mol.

        Returns
        -------
        dict with:
          position, chain, mutations_tested, top_esm_designs,
          wildtype_plddt, best_mutation, n_passing, summary,
          results — list of per-mutation result dicts, each containing:
            mutation, mean_plddt, plddt_vs_wt, ddg, total_ddg,
            ddg_category, pass_threshold, notes.
        """
        mutation_dicts = self._parse_mutations_to_dicts(mutations, chain)

        # ── ESMFold wildtype baseline ─────────────────────────────────────────
        wildtype_plddt: Optional[float] = None
        if esmfold_bridge is not None:
            try:
                wt_result = esmfold_bridge.predict(
                    wildtype_sequence, label=f"wt_pos{position}"
                )
                if wt_result.get("success"):
                    _wt_plddt = wt_result.get("mean_plddt", 0.0)
                    if _wt_plddt is not None and _wt_plddt < 2.0:
                        _wt_plddt *= 100.0   # 0-1 scale → 0-100
                    wildtype_plddt = _wt_plddt
            except Exception:
                pass

        # ── Rosetta ddG for all mutations ─────────────────────────────────────
        ddg_scores: Dict[str, float] = {}
        if rosetta_bridge is not None and mutation_dicts:
            try:
                pdb_posix = Path(pdb_path).as_posix() if pdb_path else ""
                ros_result = rosetta_bridge.analyze(pdb_posix, mutation_dicts)
                if ros_result.success:
                    ddg_scores = ros_result.data.get("ddg_scores", {})
            except Exception:
                pass

        # ── Per-mutant ESMFold + collate ──────────────────────────────────────
        results: List[Dict[str, Any]] = []

        for mut_dict in mutation_dicts[:top_esm_designs]:
            from_aa   = mut_dict.get("from_aa", "X")
            mut_pos   = int(mut_dict.get("position", position))
            to_aa     = mut_dict.get("to_aa", "X")
            mut_label = f"{from_aa}{mut_pos}{to_aa}"

            mut_plddt: Optional[float] = None
            if esmfold_bridge is not None:
                try:
                    mut_seq    = self._apply_mutations_to_sequence(
                        wildtype_sequence, [mut_dict]
                    )
                    mut_result = esmfold_bridge.predict(
                        mut_seq, label=f"mut_{mut_label}"
                    )
                    if mut_result.get("success"):
                        _mp = mut_result.get("mean_plddt", 0.0)
                        if _mp is not None and _mp < 2.0:
                            _mp *= 100.0
                        mut_plddt = _mp
                except Exception:
                    pass

            plddt_vs_wt: Optional[float] = None
            if mut_plddt is not None and wildtype_plddt is not None:
                plddt_vs_wt = round(mut_plddt - wildtype_plddt, 2)

            ddg = ddg_scores.get(mut_label)

            # ddG classification
            if ddg is None:
                ddg_category = "unknown"
            elif ddg < -0.5:
                ddg_category = "stabilizing"
            elif ddg > 0.5:
                ddg_category = "destabilizing"
            else:
                ddg_category = "neutral"

            # Pass threshold
            _plddt_ok = (plddt_vs_wt is None) or (plddt_vs_wt >= -5.0)
            _ddg_ok   = (ddg is None) or (ddg <= 1.0)
            pass_threshold = _plddt_ok and _ddg_ok

            notes = self._generate_validation_notes(
                mut_plddt, plddt_vs_wt, ddg, ddg_category, pass_threshold,
                [mut_label],
            )

            results.append({
                "mutation":       mut_label,
                "mean_plddt":     mut_plddt,
                "plddt_vs_wt":    plddt_vs_wt,
                "ddg":            ddg,
                "total_ddg":      ddg,
                "ddg_category":   ddg_category,
                "pass_threshold": pass_threshold,
                "notes":          notes,
            })

        # ── Summarise ─────────────────────────────────────────────────────────
        passing      = [r for r in results if r["pass_threshold"]]
        best_mutation: Optional[str] = None
        if passing:
            best = max(passing, key=lambda r: (r["mean_plddt"] or 0.0))
            best_mutation = best["mutation"]

        n_pass = len(passing)
        summary = (
            f"Validation at pos {position}: {len(results)} mutation(s) tested, "
            f"{n_pass} pass fold + stability threshold"
        )
        if best_mutation:
            summary += f". Best: {best_mutation}"

        return {
            "position":        position,
            "chain":           chain,
            "mutations_tested": [r["mutation"] for r in results],
            "top_esm_designs": top_esm_designs,
            "results":         results,
            "wildtype_plddt":  wildtype_plddt,
            "best_mutation":   best_mutation,
            "n_passing":       n_pass,
            "summary":         summary,
        }

    # ── generate_positions_summary ─────────────────────────────────────────────

    def generate_positions_summary(
        self,
        candidates: List[Dict[str, Any]],
        chain:      str = "A",
        top_n:      int = 20,
        validation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build a multi-line text summary for suggest_glycosylation_positions()
        output (triggers a Rich Panel in main.py when returned as summary).

        Includes an optional validation section if *validation* is supplied.
        """
        header = f"Projection-aware glycosylation candidates — chain {chain}"
        lines: List[str] = [header, ""]

        if not candidates:
            lines.append("No suitable glycosylation positions found.")
            return "\n".join(lines)

        shown = candidates[:top_n]
        lines.append(f"Top {len(shown)} candidate(s) (of {len(candidates)} evaluated):")

        HDR = (
            f"  {'Pos':<6}{'Res':<5}{'Projection':<12}"
            f"{'SASA':<10}{'SS':<6}{'Score':>7}  {'Conf'}"
        )
        SEP = "  " + "-" * (len(HDR) - 2)
        lines.append(HDR)
        lines.append(SEP)

        for c in shown:
            pos      = c.get("position", "?")
            res      = c.get("wildtype_residue", "?")
            proj_cat = c.get("projection_category", "unknown")
            sasa_cat = c.get("sasa_category", "unknown")
            ss       = c.get("secondary_structure", "?")
            score    = c.get("composite_score", 0.0)
            conf     = c.get("confidence", "?")
            lines.append(
                f"  {str(pos):<6}{res:<5}{proj_cat:<12}{sasa_cat:<10}"
                f"{ss:<6}{score:>7.3f}  {conf}"
            )

        if validation:
            lines.append("")
            lines.append(f"Fold/stability validation: {validation.get('summary', '')}")

        return "\n".join(lines)

    # ── generate_positions_chimerax_commands ──────────────────────────────────

    def generate_positions_chimerax_commands(
        self,
        candidates: List[Dict[str, Any]],
        model_id:   str = "1",
        chain:      str = "A",
        top_n:      int = 20,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate ChimeraX commands for projection-aware glycosylation candidates.

        3 commands per candidate (color, show atoms, label), plus 2 extra for
        outward-projecting sites (Cβ sphere + resize).  1 explanation per site.

        Color logic:
          outward + high  → _COLOR_NATIVE    (#00cc00, green)
          outward/flat    → _COLOR_ENG_HIGH  (#00cccc, cyan)
          otherwise       → _COLOR_ENG_MOD   (#cccc00, yellow)
        """
        if not candidates:
            return [], []

        cmds: List[str] = []
        exps: List[str] = []

        for cand in candidates[:top_n]:
            pos      = cand["position"]
            proj_cat = cand.get("projection_category", "unknown")
            score    = cand.get("composite_score", 0.0)
            conf     = cand.get("confidence", "moderate")

            if proj_cat == "outward" and conf == "high":
                hex_color = _COLOR_NATIVE
            elif proj_cat in ("outward", "flat"):
                hex_color = _COLOR_ENG_HIGH
            else:
                hex_color = _COLOR_ENG_MOD

            spec = f"#{model_id}/{chain}:{pos}"
            cmds.append(f"color {spec} {hex_color} target ar")
            cmds.append(f"show {spec} atoms")

            label_text = f"pos{pos}({proj_cat[:3]})"
            cmds.append(
                f'label {spec} text "{label_text}" '
                f'color {hex_color} height 1.0'
            )

            # Cβ sphere for outward-projecting sites
            if proj_cat == "outward":
                spec_cb = f"#{model_id}/{chain}:{pos}@CB"
                cmds.append(f"style {spec_cb} sphere")
                cmds.append(f"size {spec_cb} atomRadius 1.5")

            exps.append(
                f"Glycosylation candidate at {chain}{pos} — "
                f"proj={proj_cat}, score={score:.3f}, {conf} confidence"
            )

        return cmds, exps

    # ── Private helpers ────────────────────────────────────────────────────────

    def _apply_mutations_to_sequence(
        self, sequence: str, mutations: List[Dict[str, Any]]
    ) -> str:
        """Apply a list of mutation dicts to *sequence*.  Returns mutated string."""
        seq_list = list(sequence)
        for mut in mutations:
            pos   = int(mut.get("position", 0))
            to_aa = str(mut.get("to_aa", "X"))
            if 1 <= pos <= len(seq_list):
                seq_list[pos - 1] = to_aa
        return "".join(seq_list)

    def _parse_mutations_to_dicts(
        self, mutations: List[Any], chain: str
    ) -> List[Dict[str, Any]]:
        """
        Normalise *mutations* to a list of dicts {chain, position, from_aa, to_aa}.

        Accepted input forms:
          - dict with keys (chain, position, from_aa, to_aa)
          - string "A5N"  (from_aa + 1-indexed position + to_aa)
        """
        result: List[Dict[str, Any]] = []
        for mut in mutations:
            if isinstance(mut, dict):
                result.append(mut)
            elif isinstance(mut, str):
                m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", mut.strip())
                if m:
                    result.append({
                        "chain":    chain,
                        "position": int(m.group(2)),
                        "from_aa":  m.group(1).upper(),
                        "to_aa":    m.group(3).upper(),
                    })
        return result

    def _generate_engineering_notes(
        self,
        pos:       int,
        sequence:  str,
        proj_score: Optional[float],
        proj_cat:  str,
        sasa:      Optional[float],
        sasa_cat:  str,
        ss_code:   str,
        ss_str:    str,
        ss_map:    Dict[int, str],
    ) -> str:
        """
        Build a human-readable engineering-suitability note for one candidate.
        """
        parts: List[str] = []

        # Projection note
        if proj_cat == "outward":
            parts.append("side chain projects outward — optimal for glycan display")
        elif proj_cat == "flat":
            parts.append("side chain projects flat — moderate glycan accessibility")
        elif proj_cat == "inward":
            parts.append("side chain projects inward — restricted glycan accessibility")

        # SASA note
        if sasa_cat == "exposed":
            parts.append("well solvent-exposed")
        elif sasa_cat == "partial":
            parts.append("partially exposed")
        elif sasa_cat == "buried":
            parts.append("partially buried")

        # Secondary-structure note
        if ss_str == "loop":
            parts.append("loop region (preferred for N-glycosylation)")
        elif ss_str == "helix":
            parts.append("helix backbone (less ideal)")
        elif ss_str == "sheet":
            parts.append("sheet backbone (less ideal)")

        if not parts:
            parts.append("no structural data — sequence-based estimate only")

        return "; ".join(parts)

    def _generate_validation_notes(
        self,
        mean_plddt:     Optional[float],
        plddt_vs_wt:    Optional[float],
        total_ddg:      Optional[float],
        ddg_category:   str,
        pass_threshold: bool,
        mutations:      List[str],
    ) -> str:
        """Build a human-readable note for one validation result."""
        parts: List[str] = []

        if mean_plddt is not None:
            conf = (
                "high"   if mean_plddt > 70 else
                "medium" if mean_plddt > 50 else
                "low"
            )
            parts.append(f"ESMFold pLDDT={mean_plddt:.1f} ({conf} confidence)")

        if plddt_vs_wt is not None:
            if plddt_vs_wt >= -2.0:
                parts.append(
                    f"pLDDT delta={plddt_vs_wt:+.1f} (minimal fold disruption)"
                )
            elif plddt_vs_wt >= -5.0:
                parts.append(
                    f"pLDDT delta={plddt_vs_wt:+.1f} (moderate fold disruption)"
                )
            else:
                parts.append(
                    f"pLDDT delta={plddt_vs_wt:+.1f} (significant fold disruption)"
                )

        if total_ddg is not None:
            parts.append(
                f"Rosetta ddG={total_ddg:+.2f} kcal/mol ({ddg_category})"
            )

        if pass_threshold:
            parts.append("[PASS] passes fold + stability threshold")
        else:
            parts.append("[FAIL] fails fold + stability threshold")

        return "; ".join(parts)

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
            proj_cat   = cand.get("projection_category", "unknown")

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

            # ── Cβ sphere — only for outward-projecting sites ─────────────────
            # Shows the Cβ prominently so the user can see projection direction.
            if proj_cat == "outward":
                spec_cb = f"#{model_id}/{chain}:{pos}@CB"
                cmds.append(f"style {spec_cb} sphere")
                cmds.append(f"size {spec_cb} atomRadius 1.5")

            kind = "Engineered" if engineered else "Native"
            proj_note = f", proj={proj_cat}" if proj_cat != "unknown" else ""
            exps.append(
                f"{kind} glycosylation site N{pos} — {confidence} confidence "
                f"(score={score:.3f}{proj_note})"
            )

        return cmds, exps
