"""
cavity_bridge.py
----------------
Internal cavity detection and filling suggestion for StructureBot.

Cavities (internal voids) represent opportunities for stabilisation --
filling them with larger hydrophobic residues reduces the entropic cost
of the unfolded state.

Method: assembly-aware SASA + BFS geometric clustering.
  1. Compute per-residue SASA on the selected chains (BioPython ShrakeRupley).
  2. Residues with SASA < burial_sasa_threshold are "buried".
  3. BFS connected-components on buried Cα atoms (cluster_radius cutoff).
  4. Size-filtered clusters → cavity dicts.

Supports full assembly / dimer analysis via the `chains` parameter.
Pass chains=None to analyse all chains; interface cavities are flagged
automatically when a cluster contains residues from more than one chain.

Note: cavity volumes are APPROXIMATE (n_residues * 15 A^3 proxy).
For precise volumes use HOLLOW or fpocket.

Dependencies: BioPython (main venv).
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from Bio import PDB as _PDB
    _BIOPYTHON_OK = True
except ImportError:
    _BIOPYTHON_OK = False

try:
    from Bio.PDB import PDBIO as _PDBIO
    try:
        # BioPython >= 1.79 ships ShrakeRupley in Bio.PDB.SASA. The old
        # `Bio.PDB.ShrakeRupley` path never existed on shipped versions, so the
        # prior import silently disabled ALL SASA (cavity detection + the
        # solubility exposed-selector) — caught by the design-goal live-verify.
        from Bio.PDB.SASA import ShrakeRupley as _ShrakeRupley
    except ImportError:
        from Bio.PDB.ShrakeRupley import ShrakeRupley as _ShrakeRupley  # legacy fallback
    _BIOPYTHON_SASA_OK = True
except ImportError:
    _PDBIO = None          # type: ignore[assignment]
    _ShrakeRupley = None   # type: ignore[assignment]
    _BIOPYTHON_SASA_OK = False


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
    Supports single-chain and multi-chain (assembly) analysis.
    """

    # ESM tolerance cutoff
    _ESM_MIN = 0.4

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_cavities(
        self,
        pdb_path: str,
        chains: Optional[List[str]] = None,
        burial_sasa_threshold: float = 20.0,
        cluster_radius: float = 6.0,
        min_cluster_size: int = 4,
        max_cluster_size: int = 30,
        min_volume_proxy: float = 10.0,
        debug: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Detect internal cavities using assembly-aware SASA + BFS clustering.

        Parameters
        ----------
        pdb_path : str
            Path to the PDB file.
        chains : list[str] | str | None
            Chains to include.  None = all chains (assembly mode).
            A string is treated as a single-element list.
        burial_sasa_threshold : float
            Residues with SASA (Å²) below this value are considered buried.
        cluster_radius : float
            Buried Cα atoms within this distance (Å) are placed in the same cluster.
        min_cluster_size : int
            Clusters smaller than this are discarded.
        max_cluster_size : int
            Clusters larger than this are discarded (avoids flagging entire hydrophobic core).
        min_volume_proxy : float
            Minimum estimated volume (Å³) to report.
        debug : bool
            Print diagnostic information to stdout when True.

        Returns
        -------
        list[dict]
            Each dict has keys:
              cavity_id, lining_residues, centroid, estimated_volume_A3,
              volume_approximate, chain, chains_involved, is_interface_cavity,
              n_residues.
        """
        if not _BIOPYTHON_OK:
            return []

        structure = self._load_structure(pdb_path)
        if structure is None:
            return []

        # ── Normalise chains ───────────────────────────────────────────────────
        all_chain_ids: List[str] = [
            ch.id for model in structure for ch in model
        ]
        if chains is None:
            selected_chains = list(all_chain_ids)
        elif isinstance(chains, str):
            selected_chains = [chains]
        else:
            selected_chains = list(chains)
        # Keep only chains that actually exist in the structure
        selected_chains = [c for c in selected_chains if c in all_chain_ids]

        if not selected_chains:
            if debug:
                print(f"[cavity debug] No valid chains found in {pdb_path}")
            return []

        if debug:
            print(f"[cavity debug] Chains selected: {selected_chains}")

        # ── Compute SASA on filtered structure ─────────────────────────────────
        # Always delegate — _sasa_for_chains() handles its own availability check.
        sasa_map: Dict[Tuple, float] = self._sasa_for_chains(
            structure, pdb_path, selected_chains
        )

        if debug:
            print(f"[cavity debug] SASA computed for {len(sasa_map)} residues")

        # ── Identify buried residues ───────────────────────────────────────────
        buried_residues: List[Tuple] = []  # (chain_id, resno, ca_coord)
        for res, chain_id in self._get_residues(structure):
            if chain_id not in selected_chains:
                continue
            resno = res.id[1]
            key = (chain_id, resno)
            sasa_val = sasa_map.get(key, burial_sasa_threshold + 1.0)
            if sasa_val >= burial_sasa_threshold:
                continue
            # Prefer Cα; fall back to residue centroid
            ca_coord: Optional[Tuple[float, float, float]] = None
            try:
                ca_coord = tuple(res["CA"].get_vector())  # type: ignore[arg-type]
            except (KeyError, Exception):
                ca_coord = self._residue_centroid(res)
            if ca_coord is not None:
                buried_residues.append((chain_id, resno, ca_coord))

        if debug:
            print(
                f"[cavity debug] Buried residues "
                f"(SASA < {burial_sasa_threshold}): {len(buried_residues)}"
            )

        if not buried_residues:
            return []

        # ── BFS connected-components clustering ───────────────────────────────
        clusters = self._bfs_cluster_residues(buried_residues, cluster_radius)

        if debug:
            print(f"[cavity debug] Clusters before size filter: {len(clusters)}")

        # ── Build cavity dicts ─────────────────────────────────────────────────
        cavities: List[Dict[str, Any]] = []
        cid = 0
        for cluster in clusters:
            n = len(cluster)
            if n < min_cluster_size or n > max_cluster_size:
                continue
            volume = n * _VOL_PER_RESIDUE
            if volume < min_volume_proxy:
                continue

            cid += 1
            coords = [m[2] for m in cluster]
            centroid = tuple(
                sum(c[i] for c in coords) / len(coords) for i in range(3)
            )
            labels = [f"{m[0]}{m[1]}" for m in cluster]

            chains_in_cluster = sorted(set(m[0] for m in cluster))
            is_interface = len(chains_in_cluster) > 1

            # Dominant chain (most residues)
            chain_counts: Dict[str, int] = {}
            for m in cluster:
                chain_counts[m[0]] = chain_counts.get(m[0], 0) + 1
            dominant_chain = max(chain_counts, key=lambda k: chain_counts[k])

            cavities.append({
                "cavity_id":           cid,
                "lining_residues":     labels,
                "centroid":            centroid,
                "estimated_volume_A3": round(volume, 1),
                "volume_approximate":  True,
                "chain":               dominant_chain,
                "chains_involved":     chains_in_cluster,
                "is_interface_cavity": is_interface,
                "n_residues":          n,
            })

        if debug:
            print(f"[cavity debug] Cavities after filter: {len(cavities)}")
            for cav in cavities:
                print(
                    f"  Cavity {cav['cavity_id']}: {cav['n_residues']} residues, "
                    f"chains={cav['chains_involved']}, "
                    f"interface={cav['is_interface_cavity']}"
                )

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
        chains:             Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Suggest mutations to fill detected cavities with larger hydrophobic
        residues.

        Parameters
        ----------
        chains : list[str] | None
            Chains that were analysed.  Candidates are restricted to ``chain``
            (the primary chain for mutation suggestions).
        """
        if not cavities:
            return []
        if not _BIOPYTHON_OK:
            return []

        iface_set = set(interface_residues or [])
        candidates: List[Dict[str, Any]] = []

        for cavity in cavities:
            cid = cavity["cavity_id"]
            cavity_type = "interface" if cavity.get("is_interface_cavity") else "intrachain"

            for label in cavity.get("lining_residues", []):
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

                seq_idx = resno - 1
                if seq_idx < 0 or seq_idx >= len(sequence):
                    continue
                wt_1l = sequence[seq_idx]

                if wt_1l not in _VOLUME_MUTATIONS:
                    continue

                wt_vol   = _SC_VOLUME.get(wt_1l, 0.0)
                esm_tol  = (esm_scores or {}).get(resno, None)

                if esm_tol is not None and esm_tol < self._ESM_MIN:
                    continue

                esm_factor = esm_tol if esm_tol is not None else 1.0

                for tgt_1l in _VOLUME_MUTATIONS[wt_1l]:
                    tgt_vol     = _SC_VOLUME.get(tgt_1l, 0.0)
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
                        "position":           resno,
                        "chain":              chain,
                        "wildtype_residue":   wt_1l,
                        "suggested_mutation": f"{wt_1l}{resno}{tgt_1l}",
                        "cavity_id":          cid,
                        "cavity_type":        cavity_type,
                        "volume_gain_A3":     round(volume_gain, 1),
                        "esm_tolerance":      round(esm_tol, 3) if esm_tol is not None else None,
                        "composite_score":    composite_score,
                        "confidence":         confidence,
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
        chains:             Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run the complete cavity analysis pipeline.

        Parameters
        ----------
        chains : list[str] | None
            Chains to include in cavity detection.  None = all chains
            (assembly / dimer mode).  The ``chain`` parameter still selects
            the primary chain for mutation suggestions.
        """
        try:
            cavities = self.find_cavities(pdb_path, chains=chains)

            # Determine which chains were actually analysed
            if chains is None:
                structure = self._load_structure(pdb_path)
                if structure is not None:
                    chains_used: List[str] = sorted(
                        set(ch.id for model in structure for ch in model)
                    )
                else:
                    chains_used = [chain]
            elif isinstance(chains, str):
                chains_used = [chains]
            else:
                chains_used = list(chains)

            candidates = self.suggest_cavity_filling(
                pdb_path           = pdb_path,
                chain              = chain,
                sequence           = sequence,
                cavities           = cavities,
                esm_scores         = esm_scores,
                interface_residues = interface_residues,
                top_n              = top_n,
                chains             = chains,
            )

            assembly_mode = chains is None or (
                isinstance(chains, list) and len(chains) > 1
            )
            interface_count  = sum(1 for c in cavities if c.get("is_interface_cavity"))
            intrachain_count = len(cavities) - interface_count

            return {
                "success":            True,
                "chain":              chain,
                "cavities":           cavities,
                "candidates":         candidates,
                "total_cavities":     len(cavities),
                "total_candidates":   len(candidates),
                "error":              None,
                "assembly_mode":      assembly_mode,
                "chains_analysed":    chains_used,
                "interface_cavities": interface_count,
                "intrachain_cavities": intrachain_count,
            }

        except Exception as exc:
            return {
                "success":            False,
                "chain":              chain,
                "cavities":           [],
                "candidates":         [],
                "total_cavities":     0,
                "total_candidates":   0,
                "error":              str(exc),
                "assembly_mode":      False,
                "chains_analysed":    [chain],
                "interface_cavities": 0,
                "intrachain_cavities": 0,
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
        chain           = result.get("chain", "A")
        cavities        = result.get("cavities", [])
        candidates      = result.get("candidates", [])
        assembly_mode   = result.get("assembly_mode", False)
        chains_analysed = result.get("chains_analysed", [chain])

        chains_str = ", ".join(chains_analysed) if chains_analysed else chain
        if assembly_mode:
            header = f"Cavity analysis -- chains {chains_str} (assembly mode)"
        else:
            header = f"Cavity analysis -- chain {chain}"

        lines = [
            header,
            "",
            f"Cavities detected: {len(cavities)}",
        ]

        for cav in cavities:
            n        = len(cav.get("lining_residues", []))
            vol      = cav.get("estimated_volume_A3", 0)
            is_iface = cav.get("is_interface_cavity", False)
            cav_chain = cav.get("chain", chain)
            label    = "[INTERFACE]" if is_iface else f"[INTRACHAIN-{cav_chain}]"
            lines.append(
                f"  Cavity {cav['cavity_id']} {label}: {n} lining residues, "
                f"~{vol:.0f} A^3 (approximate)"
            )
        lines.append("")

        lines.append(f"Top filling candidates: {len(candidates)}")
        if candidates:
            lines.append(
                f"  {'Mutation':<10} {'Cavity':>6} {'Vol gain':>9} "
                f"{'Score':>6} {'Type':>12}  Conf"
            )
            lines.append("  " + "-" * 54)
            for c in candidates[:8]:
                ctype = c.get("cavity_type", "intrachain")
                lines.append(
                    f"  {c['suggested_mutation']:<10} "
                    f"{c['cavity_id']:>6} "
                    f"+{c['volume_gain_A3']:>7.0f}A^3 "
                    f"{c['composite_score']:>6.3f} "
                    f"{ctype:>12}  "
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

    def solvent_exposed_residues(
        self,
        pdb_path:       str,
        chain_id:       str,
        sasa_threshold: float = 40.0,
    ) -> List[int]:
        """
        Author residue numbers on *chain_id* that are SOLVENT-EXPOSED — per-residue
        absolute SASA ≥ *sasa_threshold* (Å²). The inverse of the burial test in
        ``find_cavities``; reuses the same ``_sasa_for_chains`` ShrakeRupley path.

        SASA is computed on the SELECTED CHAIN ONLY, so for a monomer redesign the
        exposure reflects the isolated chain (not buried by partners). Returns []
        on any failure (no BioPython, parse error) — callers must treat [] as
        "could not determine" and refuse, NEVER fall back to the whole chain.

        NOTE: absolute Å² is residue-size biased (a large exposed residue and a
        small partially-buried one can read similar). The threshold is the lever; a
        future refinement is relative SASA (RSA). See DESIGN_EXPOSED_SASA_THRESHOLD.
        """
        structure = self._load_structure(pdb_path)
        if structure is None:
            return []
        sasa_map = self._sasa_for_chains(structure, pdb_path, [chain_id])
        return sorted(
            resno for (ch, resno), area in sasa_map.items()
            if ch == chain_id and area >= sasa_threshold
        )

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

    def _compute_sasa_biopython(self, structure) -> Dict[Tuple, float]:
        """Compute per-residue SASA (Å²) using BioPython ShrakeRupley."""
        if not _BIOPYTHON_SASA_OK or _ShrakeRupley is None:
            return {}
        try:
            sr = _ShrakeRupley()
            sr.compute(structure, level="R")
            sasa_map: Dict[Tuple, float] = {}
            for model in structure:
                for chain in model:
                    for res in chain:
                        if res.id[0] != " ":
                            continue
                        sasa_val = getattr(res, "sasa", 0.0)
                        sasa_map[(chain.id, res.id[1])] = float(sasa_val)
            return sasa_map
        except Exception:
            return {}

    def _sasa_for_chains(
        self,
        structure,
        pdb_path: str,
        chain_ids: List[str],
    ) -> Dict[Tuple, float]:
        """
        Write a temp PDB containing only the selected chains, compute
        assembly-aware SASA, and return {(chain_id, resno): sasa_A2}.
        """
        if not _BIOPYTHON_SASA_OK or _PDBIO is None:
            return {}

        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".pdb", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                tmp_path = tmp.name

            # Write filtered structure
            selected_ids = set(chain_ids)

            class _ChainSelect(_PDB.Select):  # type: ignore[misc]
                def accept_chain(self, chain):  # type: ignore[override]
                    return chain.id in selected_ids

                def accept_residue(self, res):  # type: ignore[override]
                    return res.id[0] == " "

            io_obj = _PDBIO()
            io_obj.set_structure(structure)
            io_obj.save(tmp_path, _ChainSelect())

            # Load filtered structure and compute SASA
            parser = _PDB.PDBParser(QUIET=True)
            filtered = parser.get_structure("tmp", tmp_path)
            return self._compute_sasa_biopython(filtered)

        except Exception:
            return {}
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _bfs_cluster_residues(
        self,
        residues: List[Tuple],   # (chain_id, resno, coord)
        radius: float,
    ) -> List[List[Tuple]]:
        """BFS connected-components clustering at the given Cα–Cα radius."""
        n = len(residues)
        if n == 0:
            return []

        r2 = radius * radius

        # Build adjacency list
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            c1 = residues[i][2]
            for j in range(i + 1, n):
                c2 = residues[j][2]
                d2 = sum((c1[k] - c2[k]) ** 2 for k in range(3))
                if d2 <= r2:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        clusters: List[List[Tuple]] = []

        for start in range(n):
            if visited[start]:
                continue
            cluster: List[Tuple] = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                cluster.append(residues[node])
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        queue.append(nb)
            clusters.append(cluster)

        return clusters
