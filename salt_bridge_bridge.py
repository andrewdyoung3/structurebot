"""
salt_bridge_bridge.py
---------------------
Salt bridge analysis for StructureBot.

Salt bridges are electrostatic interactions between oppositely charged
residues (Asp/Glu with Arg/Lys/His) within ~4 Angstroms.  Engineering
new salt bridges is a proven protein stabilisation strategy contributing
~1-3 kcal/mol per bridge depending on geometry and burial.

Dependencies: BioPython (structure parsing), FreeSASA (SASA calculation)
Both are present in the main venv.  All methods degrade gracefully if
FreeSASA is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from Bio import PDB as _PDB
    _BIOPYTHON_OK = True
except ImportError:
    _BIOPYTHON_OK = False

try:
    import freesasa as _freesasa
    _FREESASA_OK = True
except ImportError:
    _FREESASA_OK = False


# ── Residue charge classification ─────────────────────────────────────────────

# Positive residues and their charged atoms
_POS_ATOMS: Dict[str, List[str]] = {
    "ARG": ["NH1", "NH2"],
    "LYS": ["NZ"],
    "HIS": ["ND1", "NE2"],
}

# Negative residues and their charged atoms
_NEG_ATOMS: Dict[str, List[str]] = {
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
}

# One-letter codes for positive/negative residues
_POS_1L = {"R": "Arg", "K": "Lys", "H": "His"}
_NEG_1L = {"D": "Asp", "E": "Glu"}

# Residues to skip when suggesting mutations
_SKIP_MUT = {"GLY", "PRO", "CYS", "G", "P", "C"}

# Salt bridge distance cutoff (Angstroms, charged atom-to-atom)
_SALT_BRIDGE_CUTOFF = 4.0

# Suggestion search window (Angstroms, Cbeta-to-Cbeta)
_SUGGEST_MIN = 5.0
_SUGGEST_MAX = 8.0

# SASA threshold for "surface" classification (Angstroms^2)
_BURIED_THRESHOLD = 20.0
_SURFACE_THRESHOLD = 30.0


def _three_to_one(resname: str) -> str:
    """Convert 3-letter residue code to 1-letter, or '?' if unknown."""
    _MAP = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return _MAP.get(resname.upper(), "?")


def _dist(v1, v2) -> float:
    """Euclidean distance between two BioPython Vector or atom objects."""
    try:
        d = v1 - v2
        return math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
    except Exception:
        return float("inf")


def _atom_dist(a1, a2) -> float:
    """Distance between two BioPython Atom objects."""
    try:
        return float(a1 - a2)
    except Exception:
        return float("inf")


def _residue_label(res, chain_id: str) -> str:
    """Format residue as 'A32' (chain + residue number)."""
    return f"{chain_id}{res.id[1]}"


def _cbeta_coord(res):
    """Return CB coordinate (CA for Gly)."""
    if "CB" in res:
        return res["CB"].get_vector()
    if "CA" in res:
        return res["CA"].get_vector()
    return None


class SaltBridgeBridge:
    """
    Detect existing salt bridges and suggest new ones for protein stabilisation.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_existing_salt_bridges(
        self,
        pdb_path: str,
        chain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify existing salt bridges in a PDB structure.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file.
        chain : str, optional
            If given, only consider residues in this chain (inter-chain pairs
            are still reported if both residues are in chain, or if chain=None).

        Returns
        -------
        List of dicts, one per detected salt bridge.
        """
        if not _BIOPYTHON_OK:
            return []

        structure = self._load_structure(pdb_path)
        if structure is None:
            return []

        sasa_map = self._compute_sasa(pdb_path)  # resno -> sasa or {}

        bridges: List[Dict[str, Any]] = []
        residues = self._get_residues(structure)

        # Separate into positively and negatively charged residue lists
        pos_residues = [
            (r, c) for r, c in residues if r.get_resname().upper() in _POS_ATOMS
        ]
        neg_residues = [
            (r, c) for r, c in residues if r.get_resname().upper() in _NEG_ATOMS
        ]

        seen: set = set()
        for pos_res, pos_chain in pos_residues:
            for neg_res, neg_chain in neg_residues:
                # Apply chain filter: include if either res is in the requested chain
                if chain and pos_chain != chain and neg_chain != chain:
                    continue

                key = tuple(sorted([
                    (pos_chain, pos_res.id[1]),
                    (neg_chain, neg_res.id[1]),
                ]))
                if key in seen:
                    continue

                # Find closest charged atom pair distance
                min_dist = float("inf")
                pos_name = pos_res.get_resname().upper()
                neg_name = neg_res.get_resname().upper()

                for atom_name in _POS_ATOMS.get(pos_name, []):
                    if atom_name not in pos_res:
                        continue
                    for aneg in _NEG_ATOMS.get(neg_name, []):
                        if aneg not in neg_res:
                            continue
                        d = _atom_dist(pos_res[atom_name], neg_res[aneg])
                        if d < min_dist:
                            min_dist = d

                if min_dist > _SALT_BRIDGE_CUTOFF:
                    continue

                seen.add(key)

                # Burial: average SASA of both residues
                sasa1 = sasa_map.get((pos_chain, pos_res.id[1]), 50.0)
                sasa2 = sasa_map.get((neg_chain, neg_res.id[1]), 50.0)
                buried = ((sasa1 + sasa2) / 2.0) < _BURIED_THRESHOLD

                pos_1l = _three_to_one(pos_name)
                neg_1l = _three_to_one(neg_name)
                type_str = (
                    f"{_POS_1L.get(pos_1l, pos_name.capitalize())}-"
                    f"{_NEG_1L.get(neg_1l, neg_name.capitalize())}"
                )

                bridges.append({
                    "res1":       _residue_label(pos_res, pos_chain),
                    "res2":       _residue_label(neg_res, neg_chain),
                    "chain1":     pos_chain,
                    "chain2":     neg_chain,
                    "distance":   round(min_dist, 2),
                    "type":       type_str,
                    "interchain": pos_chain != neg_chain,
                    "buried":     buried,
                })

        bridges.sort(key=lambda b: b["distance"])
        return bridges

    def suggest_new_salt_bridges(
        self,
        pdb_path:            str,
        chain:               str,
        sequence:            str,
        interface_residues:  Optional[List[int]] = None,
        esm_scores:          Optional[Dict[int, float]] = None,
        top_n:               int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Suggest positions where introducing a charged residue would form a
        new salt bridge with an existing oppositely-charged partner.
        """
        if not _BIOPYTHON_OK:
            return []

        structure = self._load_structure(pdb_path)
        if structure is None:
            return []

        sasa_map = self._compute_sasa(pdb_path)
        iface_set = set(interface_residues or [])

        residues_in_chain = [
            r for r, c in self._get_residues(structure) if c == chain
        ]

        # Existing salt bridge positions (to avoid placing near them)
        existing_bridges = self.find_existing_salt_bridges(pdb_path, chain=chain)
        existing_positions: set = set()
        for b in existing_bridges:
            try:
                existing_positions.add(int(b["res1"][1:]))
                existing_positions.add(int(b["res2"][1:]))
            except (ValueError, IndexError):
                pass

        # Index charged residues by position for proximity lookup
        charged_res: List[Tuple] = []  # (resno, resname, chain_id, CB coord)
        for res, c in self._get_residues(structure):
            rn = res.get_resname().upper()
            if rn in _POS_ATOMS or rn in _NEG_ATOMS:
                cb = _cbeta_coord(res)
                if cb is not None:
                    charged_res.append((res.id[1], rn, c, cb))

        candidates: List[Dict[str, Any]] = []

        for res in residues_in_chain:
            resno  = res.id[1]
            resname = res.get_resname().upper()

            # Skip if already charged
            if resname in _POS_ATOMS or resname in _NEG_ATOMS:
                continue
            # Skip Gly, Pro, Cys
            if resname in _SKIP_MUT:
                continue
            # Skip interface residues
            if resno in iface_set:
                continue
            # Skip if within 3 residues of an existing salt bridge
            if any(abs(resno - ep) <= 3 for ep in existing_positions):
                continue

            sasa = sasa_map.get((chain, resno), 50.0)
            # Must be surface-exposed
            if sasa < _SURFACE_THRESHOLD:
                continue

            cb_self = _cbeta_coord(res)
            if cb_self is None:
                continue

            # Find best charged partner in the suggestion window
            best_partner = None
            best_dist    = float("inf")
            for p_resno, p_resname, p_chain, p_cb in charged_res:
                if p_chain == chain and abs(p_resno - resno) <= 3:
                    continue  # too close in sequence
                try:
                    diff = cb_self - p_cb
                    d    = math.sqrt(diff[0]**2 + diff[1]**2 + diff[2]**2)
                except Exception:
                    continue
                if _SUGGEST_MIN <= d <= _SUGGEST_MAX and d < best_dist:
                    best_dist    = d
                    best_partner = (p_resno, p_resname, p_chain)

            if best_partner is None:
                continue

            p_resno, p_resname, p_chain = best_partner

            # Determine what charge to introduce (opposite of partner)
            if p_resname in _POS_ATOMS:
                # Partner is positive -> introduce negative (Asp/Glu -> prefer Glu)
                mut_aa    = "E"
                mut_name  = "Glu"
                partner_type = _POS_1L.get(_three_to_one(p_resname), p_resname.capitalize())
                charge_pair  = f"Glu-{partner_type}"
            else:
                # Partner is negative -> introduce positive (Arg/Lys -> prefer Lys)
                mut_aa    = "K"
                mut_name  = "Lys"
                partner_type = _NEG_1L.get(_three_to_one(p_resname), p_resname.capitalize())
                charge_pair  = f"Lys-{partner_type}"

            wt_1l = _three_to_one(resname)

            # Scoring
            distance_score    = max(0.0, 1.0 - abs(best_dist - 5.0) / 3.0)
            sasa_score        = min(sasa / 150.0, 1.0)
            esm_tol           = (esm_scores or {}).get(resno, None)
            esm_factor        = esm_tol if esm_tol is not None else 1.0
            interface_proximal = any(abs(resno - ir) <= 5 for ir in iface_set)
            iface_factor      = 0.3 if interface_proximal else 1.0
            composite_score   = round(
                distance_score * sasa_score * esm_factor * iface_factor, 3
            )

            if composite_score <= 0:
                continue

            if composite_score >= 0.6:
                confidence = "high"
            elif composite_score >= 0.3:
                confidence = "moderate"
            else:
                confidence = "low"

            candidates.append({
                "position":          resno,
                "chain":             chain,
                "wildtype_residue":  wt_1l,
                "suggested_mutation": f"{wt_1l}{resno}{mut_aa}",
                "partner_residue":   f"{p_chain}{p_resno}",
                "partner_distance":  round(best_dist, 2),
                "charge_pair":       charge_pair,
                "sasa":              round(sasa, 1),
                "esm_tolerance":     round(esm_factor, 3) if esm_tol is not None else None,
                "composite_score":   composite_score,
                "confidence":        confidence,
            })

        candidates.sort(key=lambda x: -x["composite_score"])
        return candidates[:top_n]

    def full_salt_bridge_scan(
        self,
        pdb_path:           str,
        chain:              str,
        sequence:           str,
        interface_residues: Optional[List[int]] = None,
        esm_scores:         Optional[Dict[int, float]] = None,
        top_n:              int = 10,
    ) -> Dict[str, Any]:
        """Run the complete salt bridge analysis pipeline."""
        try:
            existing   = self.find_existing_salt_bridges(pdb_path, chain=chain)
            candidates = self.suggest_new_salt_bridges(
                pdb_path, chain, sequence,
                interface_residues=interface_residues,
                esm_scores=esm_scores,
                top_n=top_n,
            )
            return {
                "success":               True,
                "chain":                 chain,
                "existing_salt_bridges": existing,
                "candidates":            candidates,
                "total_existing":        len(existing),
                "total_candidates":      len(candidates),
                "error":                 None,
            }
        except Exception as exc:
            return {
                "success":               False,
                "chain":                 chain,
                "existing_salt_bridges": [],
                "candidates":            [],
                "total_existing":        0,
                "total_candidates":      0,
                "error":                 str(exc),
            }

    def generate_chimerax_commands(
        self,
        result:   Dict[str, Any],
        model_id: str = "1",
    ) -> Tuple[List[str], List[str]]:
        """Generate ChimeraX visualization commands for salt bridge results."""
        cmds: List[str] = []
        exps: List[str] = []

        chain = result.get("chain", "A")

        # Existing salt bridges -> orange (#ff8800), spheres, labeled
        for bridge in result.get("existing_salt_bridges", []):
            for res_label, chain_id in [
                (bridge["res1"], bridge["chain1"]),
                (bridge["res2"], bridge["chain2"]),
            ]:
                try:
                    resno = int(res_label[1:])
                except (ValueError, IndexError):
                    continue
                spec = f"#{model_id}/{chain_id}:{resno}"
                cmds.append(f"show {spec} atoms")
                exps.append(f"Show {res_label} atoms (existing salt bridge)")
                cmds.append(f"color {spec} #ff8800")
                exps.append(f"Color {res_label} orange (existing salt bridge partner)")

            # Label the pair
            try:
                r1no = int(bridge["res1"][1:])
                spec1 = f"#{model_id}/{bridge['chain1']}:{r1no}"
                dist_str = f"{bridge['distance']:.1f}A"
                label = f"{bridge['res1']}-{bridge['res2']} ({dist_str})"
                cmds.append(f'label {spec1} text "{label}" height 0.5')
                exps.append(f"Label existing salt bridge {bridge['res1']}-{bridge['res2']}")
            except Exception:
                pass

        # Engineering candidates -> lime (high) or yellow (moderate)
        for cand in result.get("candidates", []):
            resno = cand.get("position")
            conf  = cand.get("confidence", "low")
            if resno is None:
                continue
            color = "#88ff00" if conf == "high" else "#cccc00"
            spec  = f"#{model_id}/{chain}:{resno}"
            cmds.append(f"show {spec} atoms")
            exps.append(f"Show {cand['suggested_mutation']} candidate atoms")
            cmds.append(f"color {spec} {color}")
            exps.append(
                f"Color {cand['suggested_mutation']} "
                f"({'lime' if conf == 'high' else 'yellow'}) "
                f"-- composite score {cand['composite_score']:.3f}"
            )

        return cmds, exps

    def generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate a multi-line Rich Panel summary of the salt bridge analysis."""
        chain     = result.get("chain", "A")
        existing  = result.get("existing_salt_bridges", [])
        candidates = result.get("candidates", [])

        lines = [
            f"Salt bridge analysis -- chain {chain}",
            "",
        ]

        lines.append(f"Existing salt bridges: {len(existing)}")
        for b in existing[:8]:
            loc = "buried" if b.get("buried") else "surface"
            lines.append(
                f"  {b['res1']}-{b['res2']}  "
                f"{b['distance']:.1f}A  {b['type']}  {loc}"
            )
        if len(existing) > 8:
            lines.append(f"  ... and {len(existing) - 8} more")
        lines.append("")

        lines.append(f"Top engineering candidates: {len(candidates)}")
        if candidates:
            lines.append(
                f"  {'Mutation':<10} {'Partner':<8} {'Dist':>6}  {'Score':>6}  {'Conf'}"
            )
            lines.append("  " + "-" * 42)
            for c in candidates[:8]:
                lines.append(
                    f"  {c['suggested_mutation']:<10} "
                    f"{c['partner_residue']:<8} "
                    f"{c['partner_distance']:>5.1f}A  "
                    f"{c['composite_score']:>6.3f}  "
                    f"{c['confidence']}"
                )

        if result.get("error"):
            lines.append(f"\nError: {result['error']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_structure(self, pdb_path: str):
        """Load and return BioPython structure, or None on failure."""
        if not _BIOPYTHON_OK:
            return None
        try:
            parser = _PDB.PDBParser(QUIET=True)
            return parser.get_structure("struct", pdb_path)
        except Exception:
            return None

    def _get_residues(self, structure) -> List[Tuple]:
        """Return list of (residue, chain_id) for all protein residues."""
        residues = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] == " ":  # standard residue (not HETATM/water)
                        residues.append((res, chain.id))
        return residues

    def _compute_sasa(self, pdb_path: str) -> Dict[Tuple, float]:
        """
        Compute per-residue SASA using FreeSASA.
        Returns {(chain_id, resno): sasa} or empty dict on failure.
        """
        if not _FREESASA_OK:
            return {}
        try:
            struct = _freesasa.Structure(Path(pdb_path).as_posix())
            result = _freesasa.calc(struct)
            sasa_map: Dict[Tuple, float] = {}
            for i in range(struct.nAtoms()):
                chain  = struct.chainLabel(i)
                resno  = struct.residueNumber(i).strip()
                try:
                    resno_int = int(resno)
                except ValueError:
                    continue
                key = (chain, resno_int)
                sasa_map[key] = sasa_map.get(key, 0.0) + result.atomArea(i)
            return sasa_map
        except Exception:
            return {}
