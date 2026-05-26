"""
assembly_analyser.py
--------------------
Biological assembly analysis for StructureBot.

Provides two analysis modes:
  MONOMER  — analyses each chain independently, no interface context
  MULTIMER — detects chain-chain interfaces, passes protected residues
             to mutation_scanner to prevent interface mutations

RCSB assembly metadata
-----------------------
Queries https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/1
to determine biological assembly type (monomer, homodimer, heterodimer,
homotetramer, etc.) and reports it in the analysis header.

Interface detection
-------------------
Uses ChimeraX zone-selection to find inter-chain contacts:
  select #{model}/{chainA}@CA & (#{model}/{chainB} :< 5.0); info selection

The ChimeraX `contacts` command only returns a summary count ("N distances")
via the REST API — it does not return parseable residue data.  Zone-selection
with CA atoms is used instead: it returns one hit per protein residue, cleanly
excluding waters and ligands, and the output format is stable across versions.

For each chain pair the command is run twice (A-near-B and B-near-A) and the
residue numbers are merged.  Interface residues are stored in session_state.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


# ── Assembly type descriptions ────────────────────────────────────────────────

_STOICH_LABELS: Dict[str, str] = {
    "A1":    "monomer",
    "A2":    "homodimer",
    "A3":    "homotrimer",
    "A4":    "homotetramer",
    "A5":    "homopentamer",
    "A6":    "homohexamer",
    "A8":    "homooctamer",
    "A12":   "homododecamer",
    "AB":    "heterodimer",
    "A2B2":  "heterotetrameric (α2β2)",
    "A2B":   "heterotrimer (α2β)",
    "AB2":   "heterotrimer (αβ2)",
    "A2B2C2": "heterohexameric (α2β2γ2)",
}

# Number-of-subunit fallback
_OLIGOMER_NAMES = [
    None, "monomer", "homodimer", "homotrimer", "homotetramer",
    "homopentamer", "homohexamer", "homoheptamer", "homooctamer",
]


def _stoichiometry_label(stoich: str, n_polymer_entities: int) -> str:
    """
    Convert an RCSB stoichiometry string (e.g. 'A2', 'AB') to a readable label.
    Falls back to oligomer count name if not in the lookup table.
    """
    if stoich in _STOICH_LABELS:
        return _STOICH_LABELS[stoich]

    # All same letter → homo-oligomer
    letters = re.findall(r"[A-Z]", stoich)
    digits  = re.findall(r"\d+", stoich) or ["1"] * len(letters)
    if len(set(letters)) == 1:
        n = sum(int(d) for d in digits)
        if 1 <= n < len(_OLIGOMER_NAMES):
            return _OLIGOMER_NAMES[n]
        return f"homo-{n}mer"

    # Hetero
    if len(set(letters)) > 1:
        total = sum(int(d) for d in digits) if digits else len(letters)
        if total == 2:
            return "heterodimer"
        return f"hetero-{total}mer"

    return stoich


# ══════════════════════════════════════════════════════════════════════════════
# RCSB assembly query
# ══════════════════════════════════════════════════════════════════════════════

def fetch_assembly_info(pdb_id: str) -> Dict[str, Any]:
    """
    Query the RCSB REST API for biological assembly 1 of a PDB entry.

    Returns a dict with:
        pdb_id          : uppercase PDB ID
        assembly_type   : "monomer" | "homodimer" | "heterodimer" | …
        stoichiometry   : raw RCSB stoichiometry string (e.g. "A2")
        n_subunits      : total polymer subunit count
        chains          : list of chain IDs in the assembly
        oligomeric_state: RCSB oligomeric state string
        is_obligate     : True if RCSB marks assembly as obligate
        error           : error description if query failed (or None)

    Returns an empty / error dict silently on any network or parse failure.
    """
    result: Dict[str, Any] = {
        "pdb_id":          pdb_id.upper(),
        "assembly_type":   None,
        "stoichiometry":   None,
        "n_subunits":      None,
        "chains":          [],
        "oligomeric_state": None,
        "is_obligate":     False,
        "error":           None,
    }

    if not _REQUESTS_OK:
        result["error"] = "requests library not available"
        return result

    try:
        url  = f"https://data.rcsb.org/rest/v1/core/assembly/{pdb_id.upper()}/1"
        resp = _requests.get(url, timeout=10)
        if resp.status_code == 404:
            result["error"] = f"Assembly info not found for {pdb_id.upper()}"
            return result
        if resp.status_code != 200:
            result["error"] = f"RCSB HTTP {resp.status_code}"
            return result
        data = resp.json()
    except Exception as exc:
        result["error"] = f"Network error: {exc}"
        return result

    # ── Parse assembly info ───────────────────────────────────────────────────
    rcsb_info = data.get("rcsb_assembly_info", {}) or {}
    pdbx_struct = data.get("pdbx_struct_assembly", {}) or {}

    stoich = rcsb_info.get("polymer_entity_instance_count_protein") or \
             rcsb_info.get("selected_polymer_entity_types")

    # Preferred: pdbx_struct_assembly.oligomeric_details
    oligo_detail  = pdbx_struct.get("oligomeric_details", "") or ""
    oligo_count   = pdbx_struct.get("oligomeric_count")
    assembly_name = pdbx_struct.get("details", "")

    # n_subunits from RCSB
    n_polymer = rcsb_info.get("polymer_entity_instance_count")
    n_protein = rcsb_info.get("polymer_entity_instance_count_protein")
    n_sub = n_protein or n_polymer or oligo_count

    # Stoichiometry string
    stoich_str = rcsb_info.get("stoichiometry") or ""

    # Determine assembly type label
    if stoich_str:
        asm_type = _stoichiometry_label(stoich_str, n_sub or 1)
    elif oligo_detail:
        asm_type = oligo_detail.lower()
    elif n_sub == 1:
        asm_type = "monomer"
    elif n_sub:
        asm_type = _OLIGOMER_NAMES[n_sub] if n_sub < len(_OLIGOMER_NAMES) else f"homo-{n_sub}mer"
    else:
        asm_type = "unknown"

    # Chain IDs in the assembly
    chains: List[str] = []
    for gen in (data.get("pdbx_struct_assembly_gen") or []):
        asym_ids = gen.get("asym_id_list") or []
        chains.extend(asym_ids)

    # Is this an obligate complex? (RCSB flags some as "author_and_software_defined_assembly")
    assembly_class = pdbx_struct.get("rcsb_candidate_assembly") or ""
    is_obligate    = "homo" in asm_type or n_sub and n_sub > 1

    result.update({
        "assembly_type":   asm_type,
        "stoichiometry":   stoich_str,
        "n_subunits":      n_sub,
        "chains":          sorted(set(chains)),
        "oligomeric_state": oligo_detail or str(n_sub) if n_sub else None,
        "is_obligate":     bool(is_obligate),
    })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Interface detection
# ══════════════════════════════════════════════════════════════════════════════

def parse_contacts_output(contacts_text: str) -> Dict[Tuple[str, str], List[int]]:
    """
    Parse the text output of a ChimeraX `contacts` command.

    ChimeraX contacts output includes lines like:
        /A 10 ARG <-> /B 25 GLU  dist 3.8
    or (atom-level):
        #1/A:10@CA <-> #1/B:25@CA  dist 4.1

    Returns {(chain_A, chain_B): [list of residue IDs in chain A at interface]}
    Chain pairs are stored in sorted order: ("A","B") not ("B","A").

    Note: since ChimeraX REST API may return contacts differently than the GUI,
    we also support the condensed table format.
    """
    # Regex patterns for common ChimeraX contacts output formats
    # Pattern 1: #model/chain:resno@atom <-> #model/chain:resno@atom
    pat_atom = re.compile(
        r"#\d+/([A-Za-z])[\s:]+(\d+)[^<]*<->\s*#\d+/([A-Za-z])[\s:]+(\d+)"
    )
    # Pattern 2: /chain resno RESNAME <-> /chain resno RESNAME
    pat_simple = re.compile(
        r"/([A-Za-z])\s+(\d+)\s+\w+\s*<->\s*/([A-Za-z])\s+(\d+)"
    )
    # Pattern 3: chain:resno ... chain:resno (compact)
    pat_compact = re.compile(
        r"([A-Za-z]):(\d+)[^<]*<->\s*([A-Za-z]):(\d+)"
    )

    interfaces: Dict[Tuple[str, str], Set[int]] = {}

    def _add(c1: str, r1: int, c2: str, r2: int) -> None:
        if c1 == c2:
            return  # intra-chain contact — ignore
        pair = tuple(sorted([c1, c2]))
        if pair not in interfaces:
            interfaces[pair] = set()
        # Add both residues to the interface set for their respective chains
        interfaces[pair].add(r1)
        interfaces[pair].add(r2)

    for line in contacts_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") and "<->" not in line:
            continue

        for pat in (pat_atom, pat_simple, pat_compact):
            m = pat.search(line)
            if m:
                c1, r1, c2, r2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
                _add(c1, r1, c2, r2)
                break

    # Convert sets to sorted lists
    return {pair: sorted(resnos) for pair, resnos in interfaces.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class AssemblyAnalyser:
    """
    Biological assembly analysis with MONOMER / MULTIMER mode selection.

    Usage::

        analyser = AssemblyAnalyser(bridge=chimerax_bridge, session=session_state)

        # Monomer mode (default)
        result = analyser.analyse(model_id="1", pdb_id="1HSG", mode="monomer")

        # Multimer mode — detects interfaces, returns protected residues
        result = analyser.analyse(model_id="1", pdb_id="1HSG", mode="multimer",
                                  chain_id="A")
    """

    def __init__(
        self,
        bridge:  Any,   # ChimeraXBridge
        session: Any,   # SessionState
    ):
        self.bridge  = bridge
        self.session = session

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyse(
        self,
        model_id: str,
        pdb_id:   Optional[str] = None,
        mode:     str = "monomer",   # "monomer" | "multimer"
        chain_id: Optional[str] = None,
        contact_distance: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Run assembly analysis.

        Parameters
        ----------
        model_id         : ChimeraX model number (e.g. "1")
        pdb_id           : 4-letter PDB ID for RCSB lookup (optional)
        mode             : "monomer" or "multimer"
        chain_id         : chain to analyse (used for interface lookup in multimer mode)
        contact_distance : Å cutoff for interface contacts (default 5.0)

        Returns
        -------
        {
            "mode":              "monomer" | "multimer",
            "assembly_info":     {type, stoichiometry, ...} from RCSB
            "interfaces":        {(chainA, chainB): [resno, ...]} (multimer only)
            "protected_residues": [resno, ...] for chain_id (multimer only)
            "excluded_count":    int — number of positions excluded
            "header":            display string for console
            "warnings":          [str, ...]
        }
        """
        mode = mode.lower().strip()
        if mode not in ("monomer", "multimer"):
            mode = "monomer"

        result: Dict[str, Any] = {
            "mode":               mode,
            "model_id":           model_id,
            "chain_id":           chain_id,
            "assembly_info":      {},
            "interfaces":         {},
            "protected_residues": [],
            "excluded_count":     0,
            "header":             "",
            "warnings":           [],
        }

        # ── Fetch assembly metadata from RCSB ─────────────────────────────────
        asm_info: Dict[str, Any] = {}
        if pdb_id and re.match(r"^[A-Za-z0-9]{4}$", pdb_id):
            asm_info = fetch_assembly_info(pdb_id)
            if asm_info.get("error"):
                result["warnings"].append(
                    f"Assembly metadata unavailable: {asm_info['error']}"
                )
        result["assembly_info"] = asm_info

        # ── Store in session ──────────────────────────────────────────────────
        if pdb_id and asm_info:
            self.session.set_assembly_info(pdb_id.upper(), asm_info)
        self.session.set_analysis_mode(model_id, mode)

        # ── Build header string ───────────────────────────────────────────────
        result["header"] = self._build_header(pdb_id, asm_info, mode)

        # ── Warn if monomer analysis of obligate multimer ─────────────────────
        if mode == "monomer" and asm_info.get("is_obligate") and asm_info.get("n_subunits", 1) > 1:
            asm_type = asm_info.get("assembly_type", "multimer")
            result["warnings"].append(
                f"⚠ {pdb_id or 'This structure'} is an obligate {asm_type}. "
                "Monomer analysis ignores inter-chain contacts — "
                "consider 'analyse as multimer' for a complete picture."
            )

        # ── Multimer: detect interfaces ───────────────────────────────────────
        if mode == "multimer":
            interfaces = self.detect_interfaces(model_id, contact_distance)
            result["interfaces"] = interfaces

            # Store in session
            self.session.set_interface_residues(model_id, interfaces)

            # Determine protected residues for the requested chain
            protected: List[int] = []
            if chain_id:
                for (c1, c2), resnos in interfaces.items():
                    if chain_id.upper() in (c1.upper(), c2.upper()):
                        protected.extend(resnos)
                protected = sorted(set(protected))

            result["protected_residues"] = protected
            result["excluded_count"]     = len(protected)

            n_ifaces = len(interfaces)
            if n_ifaces > 0:
                pairs_str = ", ".join(
                    f"{c1}/{c2} ({len(r)} residues)"
                    for (c1, c2), r in interfaces.items()
                )
                result["interface_summary"] = (
                    f"🔗 {n_ifaces} interface(s) detected: {pairs_str}"
                )
            else:
                result["interface_summary"] = "No inter-chain contacts detected."
                result["warnings"].append(
                    "No interface contacts found. "
                    "Ensure multiple chains are loaded and distance cutoff is appropriate."
                )

        return result

    def detect_interfaces(
        self,
        model_id:         str,
        contact_distance: float = 5.0,
    ) -> Dict[Tuple[str, str], List[int]]:
        """
        Find inter-chain contacts using ChimeraX zone-selection on CA atoms.

        Why not `contacts`? The ChimeraX REST API returns only a summary count
        ("N distances") for the contacts command — not parseable residue data.

        Strategy
        --------
        For each ordered pair of chains (c1, c2):
          select #{model}/{c1}@CA & (#{model}/{c2} :< dist); info selection
        and again with c1/c2 swapped.  Parse ``atom id /X:RESNO@CA`` lines to
        extract residue numbers.  @CA ensures only protein residues are hit
        (waters and ligands have no alpha-carbon).

        Returns {(chain1, chain2): sorted([resno, ...])} for each chain pair
        that has at least one contact.  Keys are in sorted order: ("A","B").
        Returns {} if ChimeraX is not running or no chains are found.
        """
        if not self.bridge.is_running():
            return {}

        chains = self._get_model_chains(model_id)
        if len(chains) < 2:
            return {}

        pat: re.Pattern = re.compile(r"atom id /([A-Za-z\d]+):(\d+)@CA")
        interfaces: Dict[Tuple[str, str], set] = {}

        for i, c1 in enumerate(chains):
            for c2 in chains[i + 1:]:
                pair: Tuple[str, str] = tuple(sorted([c1, c2]))  # type: ignore[assignment]
                resnos: set = set()

                # Collect CA atoms from each chain that are near the other
                for sel_chain, near_chain in ((c1, c2), (c2, c1)):
                    cmd = (
                        f"select #{model_id}/{sel_chain}@CA"
                        f" & (#{model_id}/{near_chain} :< {contact_distance})"
                        f"; info selection"
                    )
                    res = self.bridge.run_command(cmd)
                    if res.get("error"):
                        continue
                    text = res.get("value") or ""
                    for line in text.splitlines():
                        m = pat.search(line)
                        if m:
                            resnos.add(int(m.group(2)))

                if resnos:
                    interfaces[pair] = resnos

        # Clear selection so we don't leave residues highlighted
        self.bridge.run_command("select clear")

        return {pair: sorted(resnos) for pair, resnos in interfaces.items()}

    def _get_model_chains(self, model_id: str) -> List[str]:
        """
        Return the chain IDs present in a loaded model.

        Priority:
          1. session.structures[model_id]["metadata"]["chains"] — set during
             add_structure() via parse_pdb_header() or fetch_rcsb_metadata()
          2. Empty list → caller skips interface detection

        Only single-letter alphabetic chain IDs are returned (excludes any
        multi-character auth_asym IDs that can appear in mmCIF files).
        """
        info = self.session.get_structure(model_id)
        if info:
            meta   = info.get("metadata", {})
            chains = meta.get("chains") or []
            single = [c for c in chains if len(c) == 1 and c.isalpha()]
            if single:
                return sorted(set(single))
        return []

    # ── Visualization commands ─────────────────────────────────────────────────

    def generate_interface_viz_commands(
        self,
        model_id:   str,
        interfaces: Dict[Tuple[str, str], List[int]],
        chain_pair: Optional[Tuple[str, str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate ChimeraX commands to visualize interface residues.

        If chain_pair is given, only visualize that interface.
        Otherwise visualize all detected interfaces.

        Returns (commands, explanations).
        """
        if not interfaces:
            return [], []

        # Colour scheme: each interface gets a distinct colour
        colours = ["orange", "magenta", "lime", "gold", "pink"]

        cmds = [
            f"cartoon #{model_id}",
            f"color #{model_id} white",
        ]
        exps = [
            "Switch to cartoon representation",
            "Reset all residues to white before colouring interface",
        ]

        pairs_to_show = (
            {chain_pair: interfaces[chain_pair]}
            if chain_pair and chain_pair in interfaces
            else interfaces
        )

        for i, ((c1, c2), resnos) in enumerate(pairs_to_show.items()):
            colour = colours[i % len(colours)]
            if not resnos:
                continue
            res_spec = ",".join(str(r) for r in resnos)
            spec = f"#{model_id}:{res_spec}"

            cmds.append(f"color {spec} {colour}")
            exps.append(
                f"Interface {c1}/{c2}: {len(resnos)} residues coloured {colour}"
            )
            cmds.append(f"show {spec} atoms")
            exps.append(f"Show interface residues ({c1}/{c2}) as atoms")
            cmds.append(f"style {spec} ball")
            exps.append(f"Ball style for interface residues ({c1}/{c2})")

        cmds.append(f"view #{model_id}")
        exps.append("Fit structure in view")

        return cmds, exps

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_header(
        self,
        pdb_id:   Optional[str],
        asm_info: Dict[str, Any],
        mode:     str,
    ) -> str:
        """Build the display header string for the analysis."""
        label = pdb_id or "Structure"
        asm_type = asm_info.get("assembly_type")
        chains   = asm_info.get("chains", [])

        if asm_type and asm_type != "unknown":
            chain_str = "+".join(chains) if chains else ""
            if chain_str:
                asm_desc = f"{asm_type} (chains {chain_str})"
            else:
                asm_desc = asm_type
        else:
            asm_desc = "assembly info unavailable"

        mode_str = "monomer analysis" if mode == "monomer" else "multimer analysis"
        return f"{label}: {asm_desc} — {mode_str}"

    def get_assembly_display(
        self,
        pdb_id:   Optional[str],
        asm_info: Dict[str, Any],
    ) -> str:
        """
        Return a one-line display string for structure load notification.

        E.g.: "homodimer (chains A+B), ligand MK1, 1.9Å resolution"
        """
        parts: List[str] = []

        asm_type = asm_info.get("assembly_type")
        chains   = asm_info.get("chains", [])
        if asm_type and asm_type != "unknown":
            chain_str = "+".join(chains) if chains else ""
            if chain_str:
                parts.append(f"{asm_type} ({chain_str})")
            else:
                parts.append(asm_type)

        # Ligands from session structure metadata
        if pdb_id and self.session:
            info = None
            for mid, sinfo in self.session.structures.items():
                if sinfo.get("name", "").upper() == (pdb_id or "").upper():
                    info = sinfo
                    break
            if info:
                meta = info.get("metadata", {})
                ligands = meta.get("ligand_codes") or meta.get("ligands") or []
                if ligands:
                    parts.append(f"ligand {', '.join(str(l) for l in ligands[:3])}")
                if meta.get("resolution"):
                    parts.append(f"{meta['resolution']}Å resolution")

        return ", ".join(parts) if parts else "assembly info unavailable"

    def __repr__(self) -> str:
        return (
            f"<AssemblyAnalyser bridge={self.bridge!r} "
            f"session={self.session!r}>"
        )
