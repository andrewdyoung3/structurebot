"""
cavity_bridge.py
----------------
Internal cavity detection and filling suggestion for StructureBot.

Cavities (internal voids) represent opportunities for stabilisation --
filling them with larger hydrophobic residues reduces the entropic cost
of the unfolded state.  Detection is pure geometry using FreeSASA.

Method: dual-probe SASA (standard 1.4 A vs large 5.0 A probe).
Residues buried by the standard probe but exposed to the large probe
are cavity-lining.  Groups are assembled by spatial clustering (8 A cutoff).

Note: cavity volumes are APPROXIMATE (n_residues * 15 A^3 proxy).
For precise volumes use HOLLOW or fpocket.

Dependencies: BioPython, FreeSASA.  Both must be present in the main venv.
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


# ── Side-chain volume table (Angstroms^3, approximate) ───────────────────────

_SC_VOLUME: Dict[str, float] = {
    "G": 0.0,
    "A": 67.0,
    "S": 73.0,
    "T": 93.0,
    "V": 105.0,
    "L": 124.0,
    "I": 124.0,
    "M": 124.0,
    "P": 90.0,
    "F": 135.0,
    "Y": 141.0,
    "W": 163.0,
    "C": 86.0,
    "D": 91.0,
    "E": 109.0,
    "N": 96.0,
    "Q": 114.0,
    "K": 135.0,
    "R": 148.0,
    "H": 118.0,
}

# Allowed volume-increasing mutations {wt_1l: [target_1l, ...]}
_VOLUME_MUTATIONS: Dict[str, List[str]] = {
    "G": ["A"],
    "A": ["V", "I", "L"],
    "S": ["T", "V"],
    "T": ["V", "I"],
    "V": ["I", "L"],
}

# Volume per cavity lining residue (A^3), rough proxy
_VOL_PER_RESIDUE = 15.0


def _three_to_one(resname: str) -> str:
    _MAP = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return _MAP.get(resname.upper(), "?")


class CavityBridge:
    """
    Detect internal cavities and suggest hydrophobic filling mutations.
    """

    # Probe radii for dual-probe SASA method
    _PROBE_STD   = 1.4   # standard water probe
    _PROBE_LARGE = 5.0   # large probe to identify cavity-accessible residues

    # SASA thresholds
    _BURIED_STD   = 5.0   # A^2 with standard probe -> "buried"
    _EXPOSED_LARGE = 0.1  # A^2 with large probe -> "cavity-lining"

    # Clustering distance for grouping cavity-lining residues
    _CLUSTER_DIST = 8.0

    # Minimum cavity volume to report (A^3)
    _MIN_VOLUME = 10.0

    # ESM tolerance cutoff
    _ESM_MIN = 0.4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_cavities(
        self,
        pdb_path:     str,
        probe_radius: float = 1.4,
        min_volume:   float = 10.0,
    ) -> List[Dict[str, Any]]:
        """
        Detect internal cavities using dual-probe SASA.

        Returns a list of cavity dicts, each with keys:
          cavity_id, lining_residues, centroid, estimated_volume_A3,
          volume_approximate, chain.
        """
        if not _BIOPYTHON_OK or not _FREESASA_OK:
            return []

        structure = self._load_structure(pdb_path)
        if structure is None:
            return []

        # Compute SASA with standard and large probes
        sasa_std   = self._compute_sasa_probe(pdb_path, probe_radius)
        sasa_large = self._compute_sasa_probe(pdb_path, self._PROBE_LARGE)

        if not sasa_std or not sasa_large:
            return []

        # Identify cavity-lining residues: buried by std probe, exposed by large probe
        cavity_residues: List[Tuple] = []  # (chain, resno, centroid_coord)
        for res, chain_id in self._get_residues(structure):
            resno = res.id[1]
            key   = (chain_id, resno)
            s_std   = sasa_std.get(key, 0.0)
            s_large = sasa_large.get(key, 0.0)
            if s_std < self._BURIED_STD and s_large > self._EXPOSED_LARGE:
                coord = self._residue_centroid(res)
                if coord is not None:
                    cavity_residues.append((chain_id, resno, coord))

        if not cavity_residues:
            return []

        # Cluster by spatial proximity
        clusters = self._cluster_residues(cavity_residues)

        cavities: List[Dict[str, Any]] = []
        for cid, members in enumerate(clusters, 1):
            volume = len(members) * _VOL_PER_RESIDUE
            if volume < min_volume:
                continue

            coords   = [m[2] for m in members]
            centroid = tuple(sum(c[i] for c in coords) / len(coords) for i in range(3))
            labels   = [f"{m[0]}{m[1]}" for m in members]

            # Determine dominant chain
            chain_counts: Dict[str, int] = {}
            for m in members:
                chain_counts[m[0]] = chain_counts.get(m[0], 0) + 1
            dominant_chain = max(chain_counts, key=lambda k: chain_counts[k])

            cavities.append({
                "cavity_id":            cid,
                "lining_residues":      labels,
                "centroid":             centroid,
                "estimated_volume_A3":  round(volume, 1),
                "volume_approximate":   True,
                "chain":                dominant_chain,
            })

        return cavities

    def suggest_cavity_filling(
        self,
        pdb_path:           str,
        chain:              str,
        sequence:           str,
        cavities:           List[Dict[str, Any]],
        esm_scores:         Optional[Dict[int, float]] = None,
        interface_residues: Optional[List[int]] = None,
        top_n:              int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Suggest mutations to fill detected cavities with larger hydrophobic
        residues.
        """
        if not cavities:
            return []
        if not _BIOPYTHON_OK:
            return []

        iface_set = set(interface_residues or [])
        candidates: List[Dict[str, Any]] = []

        for cavity in cavities:
            cid = cavity["cavity_id"]
            for label in cavity.get("lining_residues", []):
                # Parse chain + resno from label like "A32"
                if not label:
                    continue
                try:
                    c_chain = label[0]
                    resno   = int(label[1:])
                except (ValueError, IndexError):
                    continue

                if c_chain != chain:
                    continue
                if resno in iface_set:
                    continue

                # Get wildtype residue from sequence (1-indexed)
                seq_idx = resno - 1
                if seq_idx < 0 or seq_idx >= len(sequence):
                    continue
                wt_1l = sequence[seq_idx]

                if wt_1l not in _VOLUME_MUTATIONS:
                    continue  # no volume-increasing mutation available

                wt_vol = _SC_VOLUME.get(wt_1l, 0.0)
                esm_tol = (esm_scores or {}).get(resno, None)

                if esm_tol is not None and esm_tol < self._ESM_MIN:
                    continue  # poorly tolerated

                esm_factor = esm_tol if esm_tol is not None else 1.0

                for tgt_1l in _VOLUME_MUTATIONS[wt_1l]:
                    tgt_vol    = _SC_VOLUME.get(tgt_1l, 0.0)
                    volume_gain = tgt_vol - wt_vol
                    if volume_gain <= 0:
                        continue

                    volume_gain_score = volume_gain / 50.0
                    composite_score   = round(volume_gain_score * esm_factor, 3)

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
                        "suggested_mutation": f"{wt_1l}{resno}{tgt_1l}",
                        "cavity_id":         cid,
                        "volume_gain_A3":    round(volume_gain, 1),
                        "esm_tolerance":     round(esm_tol, 3) if esm_tol is not None else None,
                        "composite_score":   composite_score,
                        "confidence":        confidence,
                    })

        # Deduplicate by position+mutation, keep best score
        seen: Dict[str, Dict] = {}
        for c in candidates:
            k = c["suggested_mutation"]
            if k not in seen or c["composite_score"] > seen[k]["composite_score"]:
                seen[k] = c

        result = sorted(seen.values(), key=lambda x: -x["composite_score"])
        return result[:top_n]

    def full_cavity_scan(
        self,
        pdb_path:           str,
        chain:              str,
        sequence:           str,
        interface_residues: Optional[List[int]] = None,
        esm_scores:         Optional[Dict[int, float]] = None,
        top_n:              int = 10,
    ) -> Dict[str, Any]:
        """Run the complete cavity analysis pipeline."""
        try:
            cavities   = self.find_cavities(pdb_path)
            candidates = self.suggest_cavity_filling(
                pdb_path, chain, sequence, cavities,
                esm_scores=esm_scores,
                interface_residues=interface_residues,
                top_n=top_n,
            )
            return {
                "success":          True,
                "chain":            chain,
                "cavities":         cavities,
                "candidates":       candidates,
                "total_cavities":   len(cavities),
                "total_candidates": len(candidates),
                "error":            None,
            }
        except Exception as exc:
            return {
                "success":          False,
                "chain":            chain,
                "cavities":         [],
                "candidates":       [],
                "total_cavities":   0,
                "total_candidates": 0,
                "error":            str(exc),
            }

    def generate_chimerax_commands(
        self,
        result:   Dict[str, Any],
        model_id: str = "1",
    ) -> Tuple[List[str], List[str]]:
        """Generate ChimeraX visualization commands for cavity results."""
        cmds: List[str] = []
        exps: List[str] = []
        chain = result.get("chain", "A")

        # Cavity-lining residues -> teal (#008080), spheres
        for cavity in result.get("cavities", []):
            cid = cavity["cavity_id"]
            for label in cavity.get("lining_residues", []):
                try:
                    c_chain = label[0]
                    resno   = int(label[1:])
                except (ValueError, IndexError):
                    continue
                spec = f"#{model_id}/{c_chain}:{resno}"
                cmds.append(f"show {spec} atoms")
                exps.append(f"Show cavity {cid} lining residue {label}")
                cmds.append(f"color {spec} #008080")
                exps.append(f"Color cavity {cid} lining residue {label} teal")

        # Filling candidates -> gold (high) or wheat (moderate)
        for cand in result.get("candidates", []):
            resno = cand.get("position")
            conf  = cand.get("confidence", "low")
            if resno is None:
                continue
            color = "#ffd700" if conf == "high" else "#f5deb3"
            spec  = f"#{model_id}/{chain}:{resno}"
            mut   = cand.get("suggested_mutation", "")
            vol   = cand.get("volume_gain_A3", 0)
            cmds.append(f"show {spec} atoms")
            exps.append(f"Show filling candidate {mut}")
            cmds.append(f"color {spec} {color}")
            exps.append(
                f"Color {mut} {'gold' if conf == 'high' else 'wheat'} "
                f"(+{vol:.0f} A^3 volume gain)"
            )
            label_text = f"{mut} (+{vol:.0f}A^3)"
            cmds.append(f'label {spec} text "{label_text}" height 0.5')
            exps.append(f"Label {mut} with volume gain")

        return cmds, exps

    def generate_summary(self, result: Dict[str, Any]) -> str:
        """Generate a multi-line Rich Panel summary of the cavity analysis."""
        chain      = result.get("chain", "A")
        cavities   = result.get("cavities", [])
        candidates = result.get("candidates", [])

        lines = [
            f"Cavity analysis -- chain {chain}",
            "",
            f"Cavities detected: {len(cavities)}",
        ]

        for cav in cavities:
            n    = len(cav.get("lining_residues", []))
            vol  = cav.get("estimated_volume_A3", 0)
            lines.append(
                f"  Cavity {cav['cavity_id']}: {n} lining residues, "
                f"~{vol:.0f} A^3 (approximate)"
            )
        lines.append("")

        lines.append(f"Top filling candidates: {len(candidates)}")
        if candidates:
            lines.append(
                f"  {'Mutation':<10} {'Cavity':>6} {'Vol gain':>9} {'Score':>6}  Conf"
            )
            lines.append("  " + "-" * 42)
            for c in candidates[:8]:
                lines.append(
                    f"  {c['suggested_mutation']:<10} "
                    f"{c['cavity_id']:>6} "
                    f"+{c['volume_gain_A3']:>7.0f}A^3 "
                    f"{c['composite_score']:>6.3f}  "
                    f"{c['confidence']}"
                )
        lines.append("")
        lines.append(
            "Note: cavity volumes are approximate (geometric proxy)."
        )
        lines.append(
            "For precise volumes use HOLLOW or fpocket."
        )

        if result.get("error"):
            lines.append(f"\nError: {result['error']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_structure(self, pdb_path: str):
        if not _BIOPYTHON_OK:
            return None
        try:
            parser = _PDB.PDBParser(QUIET=True)
            return parser.get_structure("struct", pdb_path)
        except Exception:
            return None

    def _get_residues(self, structure) -> List[Tuple]:
        residues = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.id[0] == " ":
                        residues.append((res, chain.id))
        return residues

    def _residue_centroid(self, res) -> Optional[Tuple[float, float, float]]:
        """Compute the geometric centroid of all atoms in a residue."""
        coords = []
        for atom in res:
            coords.append(atom.get_vector())
        if not coords:
            return None
        x = sum(v[0] for v in coords) / len(coords)
        y = sum(v[1] for v in coords) / len(coords)
        z = sum(v[2] for v in coords) / len(coords)
        return (x, y, z)

    def _compute_sasa_probe(self, pdb_path: str, probe_radius: float) -> Dict[Tuple, float]:
        """Compute per-residue SASA with a given probe radius."""
        if not _FREESASA_OK:
            return {}
        try:
            params = _freesasa.Parameters({
                "probe-radius": probe_radius,
                "algorithm":    _freesasa.LeeRichards,
            })
            struct = _freesasa.Structure(Path(pdb_path).as_posix())
            result = _freesasa.calc(struct, params)
            sasa_map: Dict[Tuple, float] = {}
            for i in range(struct.nAtoms()):
                chain  = struct.chainLabel(i)
                resno_str = struct.residueNumber(i).strip()
                try:
                    resno = int(resno_str)
                except ValueError:
                    continue
                key = (chain, resno)
                sasa_map[key] = sasa_map.get(key, 0.0) + result.atomArea(i)
            return sasa_map
        except Exception:
            return {}

    def _cluster_residues(
        self,
        residues: List[Tuple],  # (chain, resno, coord)
    ) -> List[List[Tuple]]:
        """Simple distance-based clustering (single-linkage, 8 A cutoff)."""
        if not residues:
            return []

        clusters: List[List[Tuple]] = []
        assigned = [False] * len(residues)

        for i, res in enumerate(residues):
            if assigned[i]:
                continue
            cluster = [res]
            assigned[i] = True
            for j, other in enumerate(residues):
                if assigned[j]:
                    continue
                # Check distance to any member already in cluster
                for member in cluster:
                    c1, c2 = member[2], other[2]
                    try:
                        d = math.sqrt(sum((c1[k] - c2[k])**2 for k in range(3)))
                    except Exception:
                        d = float("inf")
                    if d <= self._CLUSTER_DIST:
                        cluster.append(other)
                        assigned[j] = True
                        break
            clusters.append(cluster)

        return clusters
