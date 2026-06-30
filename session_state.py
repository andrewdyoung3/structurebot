"""
session_state.py
----------------
Tracks everything StructureBot knows about the current ChimeraX session:
loaded structures, named selections, style history, command log.

Also provides:
  - parse_pdb_header()    — lightweight local .pdb file scanner
  - fetch_rcsb_metadata() — live RCSB PDB REST API lookup (for PDB IDs)
"""

import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Local PDB header parser ────────────────────────────────────────────────────

# Standard amino acids and common solvent/ion codes to exclude from ligand list
_AMINO_ACIDS = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
    # modified / non-standard but common
    "MSE","SEC","PYL","CSE","HYP",
}
_SOLVENT_CODES = {"HOH","WAT","H2O","DOD","TIP","H2O","GOL","EDO","PEG","SO4",
                  "PO4","ACT","ACE","NI","MG","CA","NA","K","ZN","FE","MN",
                  "CO","CU","CD","HG","CL","BR","IOD","F"}


def parse_pdb_header(pdb_path: str) -> Dict[str, Any]:
    """
    Read the REMARK/HEADER section of a local PDB file and return metadata.
    Stops at the first ATOM or HETATM record to stay fast on large files.
    """
    meta: Dict[str, Any] = {}
    chains: set = set()
    ligands: Dict[str, str] = {}   # code → full name
    het_codes: set = set()

    try:
        with open(pdb_path, "r", errors="replace") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                rec  = line[:6].strip()

                if rec in ("ATOM", "HETATM", "END"):
                    break

                if rec == "HEADER":
                    meta["pdb_id"] = line[62:66].strip() or None
                    meta["date"]   = line[50:59].strip() or None

                elif rec == "TITLE":
                    meta["title"] = (meta.get("title", "") + " " + line[10:].strip()).strip()

                elif rec == "EXPDTA":
                    meta["method"] = line[10:79].strip()

                elif rec == "SOURCE":
                    src = line[10:79].strip()
                    m = re.search(r"ORGANISM_SCIENTIFIC:\s*([^;]+)", src, re.I)
                    if m:
                        meta["organism"] = m.group(1).strip()

                elif rec == "REMARK":
                    if line[7:10].strip() == "2":
                        m = re.search(r"RESOLUTION\.\s+([\d.]+)\s+ANGSTROMS", line, re.I)
                        if m:
                            meta["resolution"] = m.group(1)

                elif rec == "SEQRES":
                    ch = line[11:12].strip()
                    if ch:
                        chains.add(ch)

                elif rec == "HET":
                    code = line[7:10].strip()
                    if code and code not in _SOLVENT_CODES and code not in _AMINO_ACIDS:
                        het_codes.add(code)

                elif rec == "HETNAM":
                    code = line[11:14].strip()
                    name = line[15:70].strip()
                    if code and name and code not in _SOLVENT_CODES and code not in _AMINO_ACIDS:
                        ligands[code] = name

    except OSError:
        pass

    if chains:
        meta["chains"] = sorted(chains)

    if ligands:
        meta["ligands"] = [f"{k} ({v})" for k, v in ligands.items()]
        meta["ligand_codes"] = sorted(ligands.keys())
    elif het_codes:
        meta["ligands"] = sorted(het_codes)
        meta["ligand_codes"] = sorted(het_codes)

    return meta


# ── RCSB REST API metadata ─────────────────────────────────────────────────────

def fetch_rcsb_metadata(pdb_id: str) -> Dict[str, Any]:
    """
    Fetch entry metadata from the RCSB PDB REST API for a 4-letter PDB ID.
    Returns an empty dict silently on any network or parse failure.

    Endpoint: https://data.rcsb.org/rest/v1/core/entry/{ID}
    """
    try:
        import requests as req
        url  = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        resp = req.get(url, timeout=8)
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception:
        return {}

    meta: Dict[str, Any] = {"pdb_id": pdb_id.upper()}

    # Title
    struct = data.get("struct", {})
    if struct.get("title"):
        meta["title"] = struct["title"]

    # Experimental method
    exptl = data.get("exptl") or [{}]
    if exptl[0].get("method"):
        meta["method"] = exptl[0]["method"]

    # Resolution
    refine = data.get("refine") or [{}]
    if refine[0].get("ls_d_res_high"):
        meta["resolution"] = str(refine[0]["ls_d_res_high"])

    # Source organism
    entity_src = data.get("entity_src_gen") or data.get("entity_src_nat") or [{}]
    for src in entity_src:
        org = src.get("pdbx_gene_src_scientific_name") or src.get("pdbx_organism_scientific")
        if org:
            meta["organism"] = org
            break

    # Non-polymer ligands (excludes water, ions already filtered by RCSB)
    entry_info = data.get("rcsb_entry_info") or {}
    ligands = entry_info.get("nonpolymer_bound_components") or []
    if ligands:
        meta["ligands"]      = ligands         # list of 3-letter codes, e.g. ["MK1"]
        meta["ligand_codes"] = ligands

    # Polymer chains — the entry endpoint does NOT include polymer_entity_instances.
    # Query /rest/v1/core/polymer_entity/{pdb_id}/{entity_id} for each polymer entity
    # and collect auth_asym_ids (author chain IDs, e.g. "A", "B").
    ids_data = data.get("rcsb_entry_container_identifiers") or {}
    polymer_entity_ids = ids_data.get("polymer_entity_ids") or []
    chains: set = set()
    for entity_id in polymer_entity_ids:
        try:
            ent_url  = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
            ent_resp = req.get(ent_url, timeout=8)
            if ent_resp.status_code == 200:
                ent_data  = ent_resp.json()
                container = ent_data.get("rcsb_polymer_entity_container_identifiers") or {}
                # auth_asym_ids is the list of author chain IDs for this entity's instances
                auth_ids  = container.get("auth_asym_ids") or container.get("asym_ids") or []
                for ch in auth_ids:
                    if ch and len(ch) == 1 and ch.isalpha():
                        chains.add(ch)
        except Exception:
            pass
    if chains:
        meta["chains"] = sorted(chains)

    return meta


# ── Session state ──────────────────────────────────────────────────────────────

class SessionState:
    """
    Maintains the running state of a ChimeraX session:
    - loaded structures (model number → name, path, parsed metadata)
    - named selections
    - visual style log
    - full command history
    """

    def __init__(self, working_dir: Optional[str] = None):
        self.session_start   = datetime.now().isoformat(timespec="seconds")
        self.working_dir     = working_dir or os.getcwd()
        self.structures:     Dict[str, Dict[str, Any]] = {}  # model_id → info
        # Variant-Design Workbench: model_id → DesignSession.to_dict() (template T
        # per unique chain + ordered variants). Persisted; backward-compatible
        # (an old session.json without this key restores to {}).
        self.design_sessions: Dict[str, Dict[str, Any]] = {}
        self.named_selections: Dict[str, str] = {}
        self.applied_styles: List[str] = []
        self.command_history: List[Dict[str, Any]] = []
        # tool_results[tool_name][model_id] = result_data_dict
        self.tool_results:   Dict[str, Dict[str, Any]] = {}
        # Stability analysis cache (keyed by job_id string).
        # Populated by RosettaBridge after each successful analysis.
        # DynaMut2 (default backend) is synchronous — status is always
        # "completed" immediately.  Preserved across save/load for the
        # `jobs` display command.
        # job_id -> {status, mutations, pdb_path, backend, submitted_at, results?}
        self.rosetta_jobs:   Dict[str, Dict[str, Any]] = {}
        # Full mutation-scan results per model: model_id → scan result list
        self.scan_results:   Dict[str, Dict[str, Any]] = {}
        # Assembly analysis: pdb_id → assembly metadata from RCSB
        self.assembly_info:  Dict[str, Dict[str, Any]] = {}
        # Analysis mode per model: model_id → "monomer" | "multimer"
        self.analysis_mode:  Dict[str, str] = {}
        # Interface residues per model: model_id → {(c1,c2): [resno,...]}
        # Keys are stored as "C1:C2" strings (tuples not JSON-serialisable)
        self.interface_residues: Dict[str, Dict[str, List[int]]] = {}
        # ProteinMPNN redesign results per model: model_id → result list
        self.proteinmpnn_results: Dict[str, Dict[str, Any]] = {}
        # Disulfide candidates: model_id → {chain_pair_key: [candidate_dicts]}
        # e.g. {"1": {"A:B": [...]}}
        self.disulfide_candidates: Dict[str, Dict[str, Any]] = {}
        # Proline substitution scan results: model_id → {chain_key: result_dict}
        # e.g. {"1": {"A": {"candidates": [...], "count": 3, ...}}}
        self.proline_results: Dict[str, Dict[str, Any]] = {}
        # N-glycosylation site scan results: model_id → {chain_key: result_dict}
        # e.g. {"1": {"A": {"native_sequons": [...], "engineered_candidates": [...]}}}
        self.glycan_results: Dict[str, Dict[str, Any]] = {}
        # ProteinMPNN + ESMFold combined validation results: model_id → {result, timestamp}
        # validated_designs[*]["pdb_str"] is present in memory but stripped on save().
        self.mpnn_esmfold_results: Dict[str, Dict[str, Any]] = {}
        # Rosetta relax cache: pdb_hash → relaxed_pdb_path
        # Tracks which PDB files have been FastRelax'd in WSL2.
        self.rosetta_relax_cache: Dict[str, str] = {}
        # WSL2 availability flag (set at startup by main.py)
        self.wsl_available: bool = False
        # Known / declared active-site / functional residue numbers (1-based).
        # Set by the user with "set active site residues 25 26 27".
        # Passed to ProlineBridge.full_proline_scan() as functional_residues.
        self.functional_residues: set = set()
        # Salt bridge analysis results: model_id → result dict
        self.salt_bridge_results: Dict[str, Any] = {}
        # Cavity detection results: model_id → result dict
        self.cavity_results:      Dict[str, Any] = {}
        # Double mutant pair scoring results: model_id → result data dict
        # Populated by ToolRouter._run_double_mutant() after a successful run.
        self.double_mutant_results: Dict[str, Any] = {}
        # NetNGlyc OST recognition results: model_id → annotated candidate list
        # Populated by _run_netnglyc() / _run_glycan_positions() in tool_router.
        self.netnglyc_results:    Dict[str, Any] = {}
        # ColabFold structure-prediction results: model_id → trimmed result dict
        # Populated by ToolRouter._run_colabfold() after a successful fold.
        self.colabfold_results:   Dict[str, Any] = {}
        # Validate-design meta-tool results: model_id → evidence-rich report dict
        # Populated by ToolRouter._run_validate_design() (fold + RMSD + energy).
        self.validate_design_results: Dict[str, Any] = {}
        # Conformer-comparison results: "{model_id_a}v{model_id_b}" → shift report dict
        # Populated by ToolRouter._run_conformer_comparison() (anchor-restricted Kabsch).
        self.conformer_comparison_results: Dict[str, Any] = {}
        # Biological-assembly generation tracking: au_model_id → {assembly_model_id, assembly_id,
        # assembly_type, n_subunits, pdb_id}.  Populated by _run_bio_assembly() so downstream
        # tools (interface detection, sequence viewers) can address the full assembly.
        self.generated_assemblies: Dict[str, Dict[str, Any]] = {}
        # Interface stabilization results: model_id → {interfaces, is_assembly, submodels, …}
        # Populated by _run_interface_stabilization() after successful analysis.
        self.interface_stabilization_results: Dict[str, Any] = {}

    # ── Structure tracking ────────────────────────────────────────────────────

    def add_structure(
        self,
        model_id:  str,
        name:      str,
        path:      Optional[str] = None,
        pdb_path:  Optional[str] = None,
        metadata:  Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a newly loaded structure.

        Metadata priority:
          1. Explicitly passed *metadata* dict
          2. If *pdb_path* is given, parse its header
          3. If *name* looks like a PDB ID (4 letters/digits), try RCSB
        """
        meta: Dict[str, Any] = {}

        if metadata:
            meta = dict(metadata)
        elif pdb_path and Path(pdb_path).is_file():
            meta = parse_pdb_header(pdb_path)
        elif path and path.lower().endswith(".pdb") and Path(path).is_file():
            meta = parse_pdb_header(path)
        elif re.match(r"^[A-Za-z0-9]{4}$", name.strip()):
            meta = fetch_rcsb_metadata(name)

        self.structures[str(model_id)] = {
            "name":      name.upper() if re.match(r"^[A-Za-z0-9]{4}$", name) else name,
            "path":      path,
            "metadata":  meta,
            "loaded_at": datetime.now().isoformat(timespec="seconds"),
        }

    def remove_structure(self, model_id: str) -> None:
        self.structures.pop(str(model_id), None)

    def close_models(self, model_ids) -> None:
        """Drop one or more closed models from tracking: remove them from `structures`
        AND prune any `generated_assemblies` record that references a closed id (as the
        source AU or the assembly model). Without the assembly prune a stale record would
        keep resolving a name to an id ChimeraX may later REUSE for an unrelated model —
        a mis-close. Ids are matched at top level ('2.1' → '2')."""
        closed = {str(m).split(".")[0] for m in model_ids if str(m).strip()}
        if not closed:
            return
        for mid in list(self.structures):
            if str(mid).split(".")[0] in closed:
                self.structures.pop(mid, None)
        for au_mid, rec in list(self.generated_assemblies.items()):
            rec = rec or {}
            au = str(au_mid).split(".")[0]
            amid = str(rec.get("assembly_model_id") or "").split(".")[0]
            if au in closed or (amid and amid in closed):
                self.generated_assemblies.pop(au_mid, None)

    def clear_all_structures(self) -> None:
        self.structures.clear()
        self.generated_assemblies.clear()

    def get_structure(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.structures.get(str(model_id))

    def next_model_id(self) -> str:
        """Predict the model ID ChimeraX will assign to the next opened structure."""
        if not self.structures:
            return "1"
        return str(max(int(k) for k in self.structures) + 1)

    # ── Tool result tracking ──────────────────────────────────────────────────

    def add_tool_result(
        self,
        tool:        str,
        model_id:    str,
        data:        Dict[str, Any],
    ) -> None:
        """
        Store the result of a computational tool run.

        Parameters
        ----------
        tool     : tool name, e.g. "camsol", "esm"
        model_id : ChimeraX model ID the tool was run against
        data     : the tool's output data dict
        """
        if tool not in self.tool_results:
            self.tool_results[tool] = {}
        self.tool_results[tool][str(model_id)] = {
            "data":      data,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def get_tool_result(
        self,
        tool:     str,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached tool result data.

        Returns the data dict, or None if no result is stored.
        """
        entry = self.tool_results.get(tool, {}).get(str(model_id))
        if entry:
            return entry.get("data")
        return None

    def clear_tool_results(self, tool: Optional[str] = None) -> None:
        """Clear cached tool results. Pass *tool* to clear just one tool."""
        if tool:
            self.tool_results.pop(tool, None)
        else:
            self.tool_results.clear()

    # ── Rosetta job tracking ──────────────────────────────────────────────────

    def add_rosetta_job(
        self,
        job_id:   str,
        job_data: Dict[str, Any],
    ) -> None:
        """
        Store a stability analysis result.

        job_data keys:
          mutations    : list of {chain, position, from_aa, to_aa}
          pdb_path     : local path of the PDB file analysed
          backend      : "dynamut2" | "empirical" | "dynamut2+empirical" | "pyrosetta"
          submitted_at : ISO timestamp string
          status       : "completed" for DynaMut2/empirical; other for async backends
          results      : {mutation_key: ddg_float} (present when status="completed")
        """
        self.rosetta_jobs[str(job_id)] = dict(job_data)

    def get_rosetta_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored data for a job, or None if not found."""
        return self.rosetta_jobs.get(str(job_id))

    def update_rosetta_job(
        self,
        job_id:  str,
        updates: Dict[str, Any],
    ) -> None:
        """Merge *updates* into an existing job record.  Silent no-op if not found."""
        entry = self.rosetta_jobs.get(str(job_id))
        if entry is not None:
            entry.update(updates)

    def list_rosetta_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of all tracked job records."""
        return dict(self.rosetta_jobs)

    def clear_rosetta_job(self, job_id: str) -> None:
        """Remove a single job record."""
        self.rosetta_jobs.pop(str(job_id), None)

    # ── Scan result tracking ──────────────────────────────────────────────────

    def add_scan_result(
        self,
        model_id: str,
        data:     Dict[str, Any],
    ) -> None:
        """Store the output of a MutationScanner.scan() call."""
        self.scan_results[str(model_id)] = {
            "data":      data,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def get_scan_result(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return the most recent scan result for a model, or None."""
        entry = self.scan_results.get(str(model_id))
        if entry:
            return entry.get("data")
        return None

    # ── Assembly analysis tracking ────────────────────────────────────────────

    def set_assembly_info(self, pdb_id: str, info: Dict[str, Any]) -> None:
        """Store RCSB assembly metadata for a PDB ID."""
        self.assembly_info[pdb_id.upper()] = dict(info)

    def get_assembly_info(self, pdb_id: str) -> Optional[Dict[str, Any]]:
        """Return cached assembly info for a PDB ID, or None."""
        return self.assembly_info.get(pdb_id.upper())

    def set_generated_assembly(self, au_model_id: str, info: Dict[str, Any]) -> None:
        """Record a biological assembly generated via `sym`.
        *au_model_id* is the asymmetric-unit model; *info* should carry
        assembly_model_id, assembly_id, assembly_type, n_subunits, pdb_id."""
        self.generated_assemblies[str(au_model_id)] = dict(info)

    def get_generated_assembly(self, au_model_id: str) -> Optional[Dict[str, Any]]:
        """Return the generated-assembly record for *au_model_id*, or None."""
        return self.generated_assemblies.get(str(au_model_id))

    def set_interface_stabilization_result(
        self, model_id: str, result: Dict[str, Any]
    ) -> None:
        """Store interface stabilization result for a model."""
        self.interface_stabilization_results[str(model_id)] = dict(result)

    def get_interface_stabilization_result(
        self, model_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return cached interface stabilization result for *model_id*, or None."""
        return self.interface_stabilization_results.get(str(model_id))

    def set_analysis_mode(self, model_id: str, mode: str) -> None:
        """Set the analysis mode ('monomer' or 'multimer') for a model."""
        self.analysis_mode[str(model_id)] = mode

    def get_analysis_mode(self, model_id: str) -> str:
        """Return the current analysis mode for a model (default 'monomer')."""
        return self.analysis_mode.get(str(model_id), "monomer")

    def set_interface_residues(
        self,
        model_id:   str,
        interfaces: Dict,   # {(c1,c2): [resno,...]} — keys may be tuples
    ) -> None:
        """
        Store interface residues for a model.

        Tuple keys (c1, c2) are serialised as "c1:c2" strings for JSON compat.
        """
        serialised: Dict[str, List[int]] = {}
        for key, resnos in interfaces.items():
            if isinstance(key, tuple):
                skey = f"{key[0]}:{key[1]}"
            else:
                skey = str(key)
            serialised[skey] = sorted(resnos)
        self.interface_residues[str(model_id)] = serialised

    def get_interface_residues(
        self,
        model_id: str,
    ) -> Dict:
        """
        Return interface residues as {(c1,c2): [resno,...]} for a model.
        Returns {} if nothing stored.
        """
        raw = self.interface_residues.get(str(model_id), {})
        result = {}
        for skey, resnos in raw.items():
            parts = skey.split(":")
            if len(parts) == 2:
                result[(parts[0], parts[1])] = resnos
            else:
                result[skey] = resnos
        return result

    def get_protected_residues_for_chain(
        self,
        model_id: str,
        chain_id: str,
    ) -> List[int]:
        """
        Return all interface residue numbers for a specific chain in a model.
        Used by MutationScanner to protect interface residues in multimer mode.
        """
        interfaces = self.get_interface_residues(model_id)
        protected: set = set()
        for (c1, c2), resnos in interfaces.items():
            if chain_id.upper() in (c1.upper(), c2.upper()):
                protected.update(resnos)
        return sorted(protected)

    def add_proteinmpnn_result(
        self,
        model_id: str,
        data:     Dict[str, Any],
    ) -> None:
        """Store ProteinMPNN sequence redesign results for a model."""
        self.proteinmpnn_results[str(model_id)] = {
            "data":      data,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    def get_proteinmpnn_result(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return ProteinMPNN results for a model, or None."""
        entry = self.proteinmpnn_results.get(str(model_id))
        return entry.get("data") if entry else None

    # ── Variant-Design Workbench ───────────────────────────────────────────────
    def add_design_session(self, model_id: str, design_dict: Dict[str, Any]) -> None:
        """Store/replace the Workbench DesignSession (as a dict) for a model."""
        self.design_sessions[str(model_id)] = design_dict

    def get_design_session(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return the Workbench DesignSession dict for a model, or None."""
        return self.design_sessions.get(str(model_id))

    # ── Disulfide candidate tracking ──────────────────────────────────────────

    def set_disulfide_candidates(
        self,
        model_id:   str,
        chain_a:    str,
        chain_b:    str,
        candidates: List[Dict[str, Any]],
    ) -> None:
        """Store disulfide prediction results for a (model, chain pair)."""
        mid = str(model_id)
        if mid not in self.disulfide_candidates:
            self.disulfide_candidates[mid] = {}
        key = f"{chain_a}:{chain_b}"
        self.disulfide_candidates[mid][key] = {
            "candidates": candidates,
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
        }

    def get_disulfide_candidates(
        self,
        model_id: str,
        chain_a:  str,
        chain_b:  str,
    ) -> Optional[List[Dict[str, Any]]]:
        """Return stored disulfide candidates for a (model, chain pair), or None."""
        entry = (
            self.disulfide_candidates
            .get(str(model_id), {})
            .get(f"{chain_a}:{chain_b}")
        )
        return entry["candidates"] if entry else None

    # ── Proline scan result tracking ──────────────────────────────────────────

    def set_proline_results(
        self,
        model_id: str,
        chain:    str,
        result:   Dict[str, Any],
    ) -> None:
        """Store proline scan results for a (model, chain) pair."""
        from datetime import datetime as _dt
        mid = str(model_id)
        if mid not in self.proline_results:
            self.proline_results[mid] = {}
        self.proline_results[mid][chain] = {
            "result":    result,
            "timestamp": _dt.now().isoformat(timespec="seconds"),
        }

    def get_proline_results(
        self,
        model_id: str,
        chain:    str,
    ) -> Optional[Dict[str, Any]]:
        """Return stored proline scan results for a (model, chain) pair, or None."""
        entry = self.proline_results.get(str(model_id), {}).get(chain)
        return entry["result"] if entry else None

    # ── Glycan site scan result tracking ─────────────────────────────────────

    def set_glycan_results(
        self,
        model_id: str,
        chain:    str,
        result:   Dict[str, Any],
    ) -> None:
        """Store N-glycosylation scan results for a (model, chain) pair."""
        from datetime import datetime as _dt
        mid = str(model_id)
        if mid not in self.glycan_results:
            self.glycan_results[mid] = {}
        self.glycan_results[mid][chain] = {
            "result":    result,
            "timestamp": _dt.now().isoformat(timespec="seconds"),
        }

    def get_glycan_results(
        self,
        model_id: str,
        chain:    str,
    ) -> Optional[Dict[str, Any]]:
        """Return stored glycan scan results for a (model, chain) pair, or None."""
        entry = self.glycan_results.get(str(model_id), {}).get(chain)
        return entry["result"] if entry else None

    # ── MPNN + ESMFold validation result tracking ─────────────────────────────

    def set_mpnn_esmfold_results(
        self,
        model_id: str,
        result:   Dict[str, Any],
    ) -> None:
        """
        Store ProteinMPNN + ESMFold validation results for a model.

        The full result (including ``pdb_str`` in each validated design) is
        kept in memory.  ``pdb_str`` is stripped when the session is saved
        to disk (regeneratable by re-running ESMFold).
        """
        from datetime import datetime as _dt
        self.mpnn_esmfold_results[str(model_id)] = {
            "result":    result,
            "timestamp": _dt.now().isoformat(timespec="seconds"),
        }

    def get_mpnn_esmfold_results(
        self,
        model_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Return the MPNN + ESMFold validation result for a model, or None.
        """
        entry = self.mpnn_esmfold_results.get(str(model_id))
        return entry["result"] if entry else None

    # ── Functional / active-site residue tracking ────────────────────────────

    def set_functional_residues(self, residues: set) -> None:
        """
        Store the user-declared active-site / functional residue numbers.

        Parameters
        ----------
        residues : set of 1-based residue sequence numbers, e.g. {25, 26, 27}.
                   Pass an empty set to clear.
        """
        self.functional_residues = set(int(r) for r in residues)

    def get_functional_residues(self) -> set:
        """Return the current functional residue set (copy)."""
        return set(self.functional_residues)

    # ── NetNGlyc OST recognition tracking ────────────────────────────────────

    def set_netnglyc_results(self, model_id: str, result: Any) -> None:
        """Store NetNGlyc OST recognition results for a model."""
        from datetime import datetime as _dt
        self.netnglyc_results[str(model_id)] = {
            "result":    result,
            "timestamp": _dt.now().isoformat(timespec="seconds"),
        }

    def get_netnglyc_results(self, model_id: str) -> Optional[Any]:
        """Return stored NetNGlyc results for a model, or None."""
        entry = self.netnglyc_results.get(str(model_id))
        return entry["result"] if entry else None

    # ── Salt bridge analysis tracking ─────────────────────────────────────────

    def set_salt_bridge_results(self, model_id: str, result: Dict[str, Any]) -> None:
        """Store salt bridge analysis results for a model."""
        self.salt_bridge_results[str(model_id)] = result

    def get_salt_bridge_results(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return stored salt bridge analysis results for a model, or None."""
        return self.salt_bridge_results.get(str(model_id))

    # ── Cavity detection tracking ─────────────────────────────────────────────

    def set_cavity_results(self, model_id: str, result: Dict[str, Any]) -> None:
        """Store cavity detection results for a model."""
        self.cavity_results[str(model_id)] = result

    def get_cavity_results(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return stored cavity detection results for a model, or None."""
        return self.cavity_results.get(str(model_id))

    # ── Double mutant pair scoring result tracking ────────────────────────────

    def set_double_mutant_results(self, model_id: str, result: Dict[str, Any]) -> None:
        """Store double mutant pair scoring results for a model."""
        self.double_mutant_results[str(model_id)] = result

    def get_double_mutant_results(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return stored double mutant results for a model, or None."""
        return self.double_mutant_results.get(str(model_id))

    # ── ColabFold structure-prediction result tracking ────────────────────────

    def set_colabfold_results(self, model_id: str, result: Dict[str, Any]) -> None:
        """Store ColabFold fold results (trimmed) for a model."""
        self.colabfold_results[str(model_id)] = result

    def get_colabfold_results(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return stored ColabFold results for a model, or None."""
        return self.colabfold_results.get(str(model_id))

    # ── Validate-design result tracking ───────────────────────────────────────

    def set_validate_design_results(self, model_id: str, result: Dict[str, Any]) -> None:
        """Store validate-design report for a model."""
        self.validate_design_results[str(model_id)] = result

    def get_validate_design_results(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Return stored validate-design report for a model, or None."""
        return self.validate_design_results.get(str(model_id))

    # ── Conformer-comparison result tracking ──────────────────────────────────

    def set_conformer_comparison_results(self, key: str, result: Dict[str, Any]) -> None:
        """Store conformer-comparison shift report keyed by '{model_id_a}v{model_id_b}'."""
        self.conformer_comparison_results[str(key)] = result

    def get_conformer_comparison_results(self, key: str) -> Optional[Dict[str, Any]]:
        """Return stored conformer-comparison report for *key*, or None."""
        return self.conformer_comparison_results.get(str(key))

    # ── Rosetta relax cache tracking ──────────────────────────────────────────

    def set_relax_cache(self, pdb_hash: str, relaxed_path: str) -> None:
        """Record that a PDB (identified by MD5 hash) has been FastRelax'd."""
        self.rosetta_relax_cache[pdb_hash] = relaxed_path

    def get_relax_cache(self, pdb_hash: str) -> Optional[str]:
        """Return the path to a cached relaxed PDB, or None."""
        return self.rosetta_relax_cache.get(pdb_hash)

    # ── Selection tracking ────────────────────────────────────────────────────

    def save_selection(self, label: str, spec: str) -> None:
        self.named_selections[label] = spec

    def get_selection(self, label: str) -> Optional[str]:
        return self.named_selections.get(label)

    # ── Style tracking ────────────────────────────────────────────────────────

    def record_style(self, command: str) -> None:
        self.applied_styles.append(command)
        if len(self.applied_styles) > 60:
            self.applied_styles = self.applied_styles[-60:]

    # ── Command history ───────────────────────────────────────────────────────

    def add_to_history(
        self,
        nl_input:  str,
        commands:  List[str],
        success:   bool = True,
        error:     Optional[str] = None,
    ) -> None:
        self.command_history.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "nl_input":  nl_input,
            "commands":  commands,
            "success":   success,
            "error":     error,
        })

    def undo_last(self) -> Optional[Dict[str, Any]]:
        """Remove and return the last history entry."""
        if self.command_history:
            return self.command_history.pop()
        return None

    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        return self.command_history[-n:]

    # ── Context summary (injected into every LLM call) ────────────────────────

    def get_context_summary(self) -> str:
        lines: List[str] = []
        lines.append(f"Session started : {self.session_start}")
        lines.append(f"Working dir     : {self.working_dir}")

        if self.structures:
            lines.append(f"\nLoaded structures ({len(self.structures)}):")
            for mid, info in self.structures.items():
                meta = info.get("metadata", {})
                lines.append(f"  #{mid}  {info['name']}")
                if meta.get("title"):
                    lines.append(f"        Title      : {meta['title'][:80]}")
                if meta.get("pdb_id"):
                    lines.append(f"        PDB ID     : {meta['pdb_id']}")
                if meta.get("organism"):
                    lines.append(f"        Organism   : {meta['organism']}")
                if meta.get("method"):
                    lines.append(f"        Method     : {meta['method']}")
                if meta.get("resolution"):
                    lines.append(f"        Resolution : {meta['resolution']} Å")
                if meta.get("chains"):
                    lines.append(f"        Chains     : {', '.join(meta['chains'])}")
                # IMPORTANT: list exact ligand residue codes for the translator
                if meta.get("ligand_codes"):
                    lines.append(f"        Ligands    : {', '.join(meta['ligand_codes'])}")
                    lines.append(f"        [Use :{'/'.join(meta['ligand_codes'])} for ligand selection]")
                elif meta.get("ligands"):
                    lines.append(f"        Ligands    : {', '.join(str(l) for l in meta['ligands'])}")
        else:
            lines.append("\nNo structures currently loaded.")

        if self.functional_residues:
            lines.append(
                f"\nActive-site residues (user-declared): "
                f"{sorted(self.functional_residues)}"
            )

        if self.named_selections:
            lines.append("\nNamed selections:")
            for lbl, spec in self.named_selections.items():
                lines.append(f"  {lbl!r:20s} → {spec}")

        if self.tool_results:
            lines.append("\nCached tool results:")
            for tool, by_model in self.tool_results.items():
                for mid, entry in by_model.items():
                    ts  = entry.get("timestamp", "?")
                    dat = entry.get("data", {})
                    # Show a brief summary depending on tool
                    if tool == "camsol":
                        n = len(dat.get("scores", {}))
                        h = len(dat.get("aggregation_hot_spots", []))
                        lines.append(f"  camsol  #{mid}  [{ts}]  {n} residues, {h} hot-spots")
                    elif tool == "esm":
                        n = len(dat.get("conservation", {}))
                        m = dat.get("mean_conservation", 0)
                        lines.append(f"  esm     #{mid}  [{ts}]  {n} residues, mean cons {m:.2f}")
                    elif tool == "rosetta":
                        n = len(dat.get("ddg_scores", {}))
                        lines.append(f"  rosetta #{mid}  [{ts}]  {n} mutations scored")
                    else:
                        lines.append(f"  {tool:<8} #{mid}  [{ts}]")

        if self.scan_results:
            lines.append("\nMutation scan results:")
            for mid, entry in self.scan_results.items():
                ts  = entry.get("timestamp", "?")
                dat = entry.get("data", {})
                n   = len(dat) if isinstance(dat, list) else "?"
                mode = self.get_analysis_mode(mid)
                lines.append(f"  scan    #{mid}  [{ts}]  {n} ranked candidates  [{mode} mode]")

        if self.interface_residues:
            lines.append("\nDetected interfaces:")
            for mid, ifaces in self.interface_residues.items():
                mode = self.get_analysis_mode(mid)
                for pair_key, resnos in ifaces.items():
                    lines.append(
                        f"  #{mid}  {pair_key}  {len(resnos)} residues  [{mode} mode]"
                    )

        if self.rosetta_jobs:
            completed = {
                jid: job for jid, job in self.rosetta_jobs.items()
                if job.get("status") == "completed"
            }
            pending = {
                jid: job for jid, job in self.rosetta_jobs.items()
                if job.get("status") not in ("completed", "failed", "error")
            }
            if completed:
                lines.append(f"\nStability analyses ({len(completed)} completed):")
                for jid, job in list(completed.items())[-3:]:  # show last 3
                    backend = job.get("backend", "?")
                    ts      = job.get("submitted_at", "?")
                    nmut    = len(job.get("mutations", []))
                    nres    = len(job.get("results", {}))
                    lines.append(
                        f"  [{backend}]  {nmut} mutation(s) scored  "
                        f"{nres} results  {ts}"
                    )
            if pending:
                lines.append(f"\nPending stability jobs ({len(pending)}):")
                for jid, job in pending.items():
                    backend = job.get("backend", "?")
                    status  = job.get("status", "?")
                    ts      = job.get("submitted_at", "?")
                    nmut    = len(job.get("mutations", []))
                    lines.append(
                        f"  job #{jid}  [{backend}]  {status}  "
                        f"{nmut} mutation(s)  submitted {ts}"
                    )

        recent = self.get_recent_history(5)
        if recent:
            lines.append(f"\nRecent commands (last {len(recent)}):")
            for entry in recent:
                mark = "✓" if entry.get("success") else "✗"
                lines.append(
                    f"  {mark} [{entry['timestamp']}] {entry['nl_input'][:70]}"
                )

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = "session.json") -> None:
        # Strip pdb_str from mpnn_esmfold_results before writing to disk.
        # pdb_str can be several MB per design and is easily regeneratable.
        mpnn_esmfold_to_save: Dict[str, Dict[str, Any]] = {}
        for mid, entry in self.mpnn_esmfold_results.items():
            entry_copy  = copy.deepcopy(entry)
            result_data = entry_copy.get("result", {})
            for design in result_data.get("validated_designs", []):
                design.pop("pdb_str", None)
            mpnn_esmfold_to_save[mid] = entry_copy

        data = {
            "session_start":        self.session_start,
            "working_dir":          self.working_dir,
            "structures":           self.structures,
            "design_sessions":      self.design_sessions,
            "named_selections":     self.named_selections,
            "applied_styles":       self.applied_styles,
            "command_history":      self.command_history,
            "tool_results":         self.tool_results,
            "rosetta_jobs":         self.rosetta_jobs,
            "scan_results":         self.scan_results,
            "assembly_info":        self.assembly_info,
            "analysis_mode":        self.analysis_mode,
            "interface_residues":   self.interface_residues,
            "proteinmpnn_results":  self.proteinmpnn_results,
            "disulfide_candidates": self.disulfide_candidates,
            "proline_results":      self.proline_results,
            "glycan_results":       self.glycan_results,
            "mpnn_esmfold_results": mpnn_esmfold_to_save,
            "rosetta_relax_cache":  self.rosetta_relax_cache,
            # sets are not JSON-serialisable; store as sorted list
            "functional_residues":  sorted(self.functional_residues),
            "salt_bridge_results":    self.salt_bridge_results,
            "cavity_results":         self.cavity_results,
            "double_mutant_results":  self.double_mutant_results,
            "netnglyc_results":       self.netnglyc_results,
            "colabfold_results":      self.colabfold_results,
            "validate_design_results":      self.validate_design_results,
            "conformer_comparison_results": self.conformer_comparison_results,
            "generated_assemblies":              self.generated_assemblies,
            "interface_stabilization_results":   self.interface_stabilization_results,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Build a SessionState from an already-parsed session dict."""
        state = cls()
        state.session_start       = data.get("session_start", state.session_start)
        state.working_dir         = data.get("working_dir",   state.working_dir)
        state.structures          = data.get("structures",    {})
        state.design_sessions     = data.get("design_sessions", {})   # backward-compat default
        state.named_selections    = data.get("named_selections", {})
        state.applied_styles      = data.get("applied_styles", [])
        state.command_history     = data.get("command_history", [])
        state.tool_results        = data.get("tool_results",  {})
        state.rosetta_jobs        = data.get("rosetta_jobs",  {})
        state.scan_results        = data.get("scan_results",  {})
        state.assembly_info       = data.get("assembly_info", {})
        state.analysis_mode       = data.get("analysis_mode", {})
        state.interface_residues   = data.get("interface_residues", {})
        state.proteinmpnn_results  = data.get("proteinmpnn_results", {})
        state.disulfide_candidates = data.get("disulfide_candidates", {})
        state.proline_results      = data.get("proline_results", {})
        state.glycan_results       = data.get("glycan_results", {})
        state.mpnn_esmfold_results = data.get("mpnn_esmfold_results", {})
        state.rosetta_relax_cache  = data.get("rosetta_relax_cache", {})
        state.functional_residues  = set(data.get("functional_residues", []))
        state.salt_bridge_results    = data.get("salt_bridge_results", {})
        state.cavity_results         = data.get("cavity_results", {})
        state.double_mutant_results  = data.get("double_mutant_results", {})
        state.netnglyc_results       = data.get("netnglyc_results", {})
        state.colabfold_results      = data.get("colabfold_results", {})
        state.validate_design_results      = data.get("validate_design_results", {})
        state.conformer_comparison_results = data.get("conformer_comparison_results", {})
        state.generated_assemblies                = data.get("generated_assemblies", {})
        state.interface_stabilization_results    = data.get("interface_stabilization_results", {})
        return state

    @classmethod
    def try_load(cls, path: str = "session.json"):
        """
        Attempt to load a session file without ever raising.

        Returns a (state, error) tuple:
          (None, None)      — the file does not exist (nothing to restore)
          (None, "message") — the file exists but is corrupt / unreadable /
                              incompatible (caller should warn and start fresh)
          (state, None)     — loaded successfully
        """
        p = Path(path)
        if not p.is_file():
            return None, None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, ValueError) as exc:
            return None, f"corrupt JSON ({str(exc)[:80]})"
        except OSError as exc:
            return None, f"unreadable ({str(exc)[:80]})"
        if not isinstance(data, dict):
            return None, "session file is not a JSON object"
        try:
            return cls._from_dict(data), None
        except Exception as exc:  # malformed/incompatible field shapes
            return None, f"incompatible session data ({str(exc)[:80]})"

    @classmethod
    def load(cls, path: str = "session.json") -> "SessionState":
        """
        Load a session, never raising: returns a fresh state if the file is
        missing, corrupt, or incompatible.  Use try_load() when you need to
        distinguish those cases (e.g. to warn the user).
        """
        state, _err = cls.try_load(path)
        return state if state is not None else cls()

    def restore_summary(self) -> str:
        """
        Short human-readable summary of restorable content, for the startup
        restore prompt.  Lists structures, scan / double-mutant results,
        detected interfaces, and prior command count.
        """
        lines: List[str] = [f"Saved        : {self.session_start}"]

        if self.structures:
            structs = ", ".join(
                f"#{mid} {info.get('name', '?')}"
                for mid, info in self.structures.items()
            )
            lines.append(f"Structures   : {structs}")
        else:
            lines.append("Structures   : none")

        if self.scan_results:
            parts = []
            for mid, entry in self.scan_results.items():
                dat = entry.get("data", {})
                n = len(dat) if isinstance(dat, list) else "?"
                parts.append(f"#{mid} ({n} candidates)")
            lines.append(f"Scan results : {', '.join(parts)}")

        if self.double_mutant_results:
            keys = ", ".join(f"#{m}" for m in self.double_mutant_results)
            lines.append(f"Double-mutant: {keys}")

        if self.interface_residues:
            keys = ", ".join(f"#{m}" for m in self.interface_residues)
            lines.append(f"Interfaces   : {keys}")

        if self.functional_residues:
            lines.append(f"Active site  : {sorted(self.functional_residues)}")

        if self.command_history:
            lines.append(f"Prior commands: {len(self.command_history)}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<SessionState "
            f"structures={list(self.structures.keys())} "
            f"history={len(self.command_history)} entries>"
        )

    # ── Snapshot / restore ────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a deep-copy snapshot of all mutable state fields.
        Use restore(snap) to revert to this snapshot.
        """
        return copy.deepcopy({
            "structures":            self.structures,
            "design_sessions":       self.design_sessions,
            "named_selections":      self.named_selections,
            "applied_styles":        self.applied_styles,
            "command_history":       self.command_history,
            "tool_results":          self.tool_results,
            "rosetta_jobs":          self.rosetta_jobs,
            "scan_results":          self.scan_results,
            "assembly_info":         self.assembly_info,
            "analysis_mode":         self.analysis_mode,
            "interface_residues":    self.interface_residues,
            "proteinmpnn_results":   self.proteinmpnn_results,
            "disulfide_candidates":  self.disulfide_candidates,
            "proline_results":       self.proline_results,
            "glycan_results":        self.glycan_results,
            "mpnn_esmfold_results":  self.mpnn_esmfold_results,
            "rosetta_relax_cache":   self.rosetta_relax_cache,
            "functional_residues":   set(self.functional_residues),
            "salt_bridge_results":    self.salt_bridge_results,
            "cavity_results":         self.cavity_results,
            "double_mutant_results":  self.double_mutant_results,
            "netnglyc_results":       self.netnglyc_results,
            "colabfold_results":      self.colabfold_results,
            "validate_design_results":      self.validate_design_results,
            "conformer_comparison_results": self.conformer_comparison_results,
            "generated_assemblies":              self.generated_assemblies,
            "interface_stabilization_results":   self.interface_stabilization_results,
        })

    def restore(self, snap: Dict[str, Any]) -> None:
        """
        Restore all mutable state fields from a snapshot produced by snapshot().
        """
        self.structures            = copy.deepcopy(snap.get("structures",            {}))
        self.design_sessions       = copy.deepcopy(snap.get("design_sessions",       {}))
        self.named_selections      = copy.deepcopy(snap.get("named_selections",      {}))
        self.applied_styles        = copy.deepcopy(snap.get("applied_styles",        []))
        self.command_history       = copy.deepcopy(snap.get("command_history",       []))
        self.tool_results          = copy.deepcopy(snap.get("tool_results",          {}))
        self.rosetta_jobs          = copy.deepcopy(snap.get("rosetta_jobs",          {}))
        self.scan_results          = copy.deepcopy(snap.get("scan_results",          {}))
        self.assembly_info         = copy.deepcopy(snap.get("assembly_info",         {}))
        self.analysis_mode         = copy.deepcopy(snap.get("analysis_mode",         {}))
        self.interface_residues    = copy.deepcopy(snap.get("interface_residues",    {}))
        self.proteinmpnn_results   = copy.deepcopy(snap.get("proteinmpnn_results",   {}))
        self.disulfide_candidates  = copy.deepcopy(snap.get("disulfide_candidates",  {}))
        self.proline_results       = copy.deepcopy(snap.get("proline_results",       {}))
        self.glycan_results        = copy.deepcopy(snap.get("glycan_results",        {}))
        self.mpnn_esmfold_results  = copy.deepcopy(snap.get("mpnn_esmfold_results",  {}))
        self.rosetta_relax_cache   = copy.deepcopy(snap.get("rosetta_relax_cache",   {}))
        self.functional_residues   = set(snap.get("functional_residues",   set()))
        self.salt_bridge_results    = snap.get("salt_bridge_results", {})
        self.cavity_results         = snap.get("cavity_results", {})
        self.double_mutant_results  = copy.deepcopy(snap.get("double_mutant_results", {}))
        self.netnglyc_results       = copy.deepcopy(snap.get("netnglyc_results", {}))
        self.colabfold_results      = copy.deepcopy(snap.get("colabfold_results", {}))
        self.validate_design_results      = copy.deepcopy(snap.get("validate_design_results", {}))
        self.conformer_comparison_results = copy.deepcopy(snap.get("conformer_comparison_results", {}))
        self.generated_assemblies              = copy.deepcopy(snap.get("generated_assemblies", {}))
        self.interface_stabilization_results  = copy.deepcopy(snap.get("interface_stabilization_results", {}))
