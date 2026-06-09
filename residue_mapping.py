"""
residue_mapping.py
------------------
Shared residue-identity spine for the fast-tier ddG voters (ThermoMPNN, RaSP, …).

This is the SINGLE source of the author-resnum mapping primitives, so every voter
and the scanner see the identical residue set + order — the invariant behind the
residue-identity fix (author resnum, never the sequence index).

  - candidate_key            chain-aware identity key f"{chain}:{wt}{resnum}{mut}"
  - ordered_chain_residues   the chain's coordinate residues in AUTHOR order
  - align_predictions_to_resnums   the WT-ANCHORED ALIGNMENT (exact across gaps AND
        insertion codes; hard-error on divergence, never a mis-attribution)

Extracted verbatim (behaviour-preserving) from the merged ThermoMPNN mapping so
RaSP reuses the exact same alignment rather than re-implementing it (re-implementing
position mapping was the bug the residue-identity fix removed).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# 3-letter → 1-letter for the PDB wildtype cross-check (standard AAs only).
_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def candidate_key(chain: str, resnum: int, wt: str, mut: str) -> str:
    """Chain-aware candidate key — never collides across chains in a multimer."""
    return f"{chain}:{wt}{resnum}{mut}"


def ordered_chain_residues(pdb_path, chain) -> List[Tuple[int, str, str]]:
    """The chain's coordinate residues in AUTHOR order → [(resnum, icode, aa)].

    SHARED by the scanner (seqindex→resnum spine) and every fast-tier voter
    (WT-anchored alignment) so BOTH see the identical residue set + order — the
    invariant that makes the alignment's resnum keys match the candidate keys.
    Standard residues only; insertion-coded residues ARE included (author order:
    base then insertion) so a predictor's extra insertion rows line up 1:1.
    Empty on failure → callers use the legacy seqindex==resnum path.
    """
    if not pdb_path:
        return []
    try:
        from Bio.PDB import PDBParser
        structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
        model = next(iter(structure))
        ch = chain or next(iter(model)).id
        if ch not in [c.id for c in model]:
            return []
        out: List[Tuple[int, str, str]] = []
        for r in model[ch]:
            het, resseq, icode = r.id
            if het.strip():                    # skip HETATM / water / ligand
                continue
            one = _THREE_TO_ONE.get(r.resname.strip().upper())
            if one:
                out.append((int(resseq), (icode or "").strip(), one))
        return out
    except Exception:
        return []


def align_predictions_to_resnums(
    ordered: List[Tuple[int, str, str]],
    pos_wt: Dict[int, str],
    log=lambda *_: None,
    tool: str = "predictor",
) -> Optional[Dict[int, int]]:
    """WT-ANCHORED ALIGNMENT — map each predictor position to its AUTHOR resnum.

    A predictor's present positions, in order, ARE the chain's residues in author
    order — so the k-th unique present position maps to the k-th ordered residue,
    REQUIRING the wildtype AA to match at each step.  Exact across gaps AND
    insertion codes; independent of any numbering offset.

    Returns {position: author_resnum}, or **None** on any length/AA divergence —
    a HARD ERROR (caller drops the whole batch to not_computed), never a
    probabilistic pass, never a mis-attribution.

    ``ordered``  : ordered_chain_residues() output [(resnum, icode, aa)].
    ``pos_wt``   : {predictor_position: wildtype_aa} (one entry per present residue).
    """
    upos = sorted(pos_wt)
    if len(upos) != len(ordered):
        log(f"  {tool}: alignment ABORTED — {len(upos)} predicted positions "
            f"vs {len(ordered)} structure residues (set divergence). not_computed.")
        return None
    pos_to_resnum: Dict[int, int] = {}
    for k, p in enumerate(upos):
        rn, _ic, aa = ordered[k]
        if pos_wt[p] != aa:
            log(f"  {tool}: alignment ABORTED — wildtype mismatch at residue "
                f"#{k} (predicted {pos_wt[p]} vs structure {aa}@{rn}). not_computed "
                "(never mis-attribute).")
            return None
        pos_to_resnum[p] = rn
    return pos_to_resnum
