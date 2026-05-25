"""
session_state.py
----------------
Tracks everything StructureBot knows about the current ChimeraX session:
loaded structures, named selections, style history, command log.

Also provides:
  - parse_pdb_header()    — lightweight local .pdb file scanner
  - fetch_rcsb_metadata() — live RCSB PDB REST API lookup (for PDB IDs)
"""

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

    # Polymer chains
    polymer = data.get("polymer_entities") or []
    chains: set = set()
    for ent in polymer:
        for inst in (ent.get("polymer_entity_instances") or []):
            ch = (inst.get("rcsb_polymer_entity_instance_container_identifiers") or {}).get("auth_asym_id")
            if ch:
                chains.add(ch)
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

    def clear_all_structures(self) -> None:
        self.structures.clear()

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
                lines.append(f"  scan    #{mid}  [{ts}]  {n} ranked candidates")

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
        data = {
            "session_start":     self.session_start,
            "working_dir":       self.working_dir,
            "structures":        self.structures,
            "named_selections":  self.named_selections,
            "applied_styles":    self.applied_styles,
            "command_history":   self.command_history,
            "tool_results":      self.tool_results,
            "rosetta_jobs":      self.rosetta_jobs,
            "scan_results":      self.scan_results,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str = "session.json") -> "SessionState":
        state = cls()
        if not Path(path).is_file():
            return state
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        state.session_start    = data.get("session_start", state.session_start)
        state.working_dir      = data.get("working_dir",   state.working_dir)
        state.structures       = data.get("structures",    {})
        state.named_selections = data.get("named_selections", {})
        state.applied_styles   = data.get("applied_styles", [])
        state.command_history  = data.get("command_history", [])
        state.tool_results     = data.get("tool_results",  {})
        state.rosetta_jobs     = data.get("rosetta_jobs",  {})
        state.scan_results     = data.get("scan_results",  {})
        return state

    def __repr__(self) -> str:
        return (
            f"<SessionState "
            f"structures={list(self.structures.keys())} "
            f"history={len(self.command_history)} entries>"
        )
