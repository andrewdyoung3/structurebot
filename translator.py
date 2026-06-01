"""
translator.py
-------------
Translates natural language requests into ChimeraX commands using the
Anthropic API.  Maintains rolling conversation history so follow-up requests
("make it more transparent", "now do the same for chain B") work naturally.

Prompt caching strategy
-----------------------
Block 1 (STATIC, CACHED): role + rules + full command reference.
  Marked cache_control=ephemeral.  After the first call the cache hits on every
  subsequent call in the session, cutting input-token cost dramatically.
Block 2 (DYNAMIC, UNCACHED): current session state — changes every turn.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Ensure venv site-packages takes priority over any global install ──────────
#
# On Windows, pip install --user drops packages into
#   %APPDATA%\Python\PythonXYZ\site-packages
# which may appear on sys.path *before* the venv's site-packages, causing the
# wrong (user-installed, possibly outdated) copy of anthropic to be loaded.
#
# We locate the venv relative to this file and, if any AppData path precedes
# the venv on sys.path, move the venv to position 0.  We also evict any
# already-cached anthropic.* modules so the corrected path takes effect.

def _ensure_venv_priority() -> None:
    _project_root   = Path(__file__).resolve().parent
    _venv_site_pkgs = _project_root / "venv" / "Lib" / "site-packages"
    _appdata_marker = str(Path.home() / "AppData" / "Roaming" / "Python")

    if not _venv_site_pkgs.is_dir():
        return

    _venv_idx = next(
        (i for i, p in enumerate(sys.path)
         if Path(p).resolve() == _venv_site_pkgs),
        None,
    )
    _appdata_idxs = [
        i for i, p in enumerate(sys.path)
        if _appdata_marker.lower() in p.lower()
    ]

    _needs_fix = (
        _venv_idx is None
        or (_appdata_idxs and min(_appdata_idxs) < _venv_idx)
    )

    if _needs_fix:
        _venv_str = str(_venv_site_pkgs)
        if _venv_idx is not None:
            sys.path.pop(_venv_idx)
        sys.path.insert(0, _venv_str)

        # Evict cached anthropic modules so re-import resolves against venv
        for _mod in [m for m in list(sys.modules)
                     if m == "anthropic" or m.startswith("anthropic.")]:
            del sys.modules[_mod]

_ensure_venv_priority()

import anthropic

import config
from session_state import SessionState

# ── Model ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = config.ANTHROPIC_MODEL


class RefusalError(ValueError):
    """
    Raised when the Anthropic API declines to process the request
    (stop_reason='refusal' or an empty-content response from a safety filter).

    Callers should catch RefusalError separately from generic ValueError so they
    can show a user-friendly message instead of propagating a traceback.
    """

# ── Static system block (cached) ───────────────────────────────────────────────

_STATIC_SYSTEM = """\
You are an expert UCSF ChimeraX command translator integrated into StructureBot.
Your sole job: convert a researcher's natural language request into one or more
precise, executable ChimeraX commands.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT  (strict JSON, no markdown, no prose)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always respond with exactly this JSON object and nothing else:

{{
  "commands":            ["cmd1", "cmd2", ...],
  "explanations":        ["what cmd1 does", "what cmd2 does", ...],
  "warnings":            ["anything the user should know before running"],
  "clarification_needed": null,
  "confidence":          "high",
  "tools_needed":        ["chimerax"],
  "tool_inputs":         {{}}
}}

confidence values:
  "high"   — unambiguous request, well-understood commands, likely to succeed
  "medium" — minor assumptions made; commands should work but review is advised
  "low"    — request is complex or unclear; user should carefully review

tools_needed values (list — may contain one or more):
  "chimerax"          — visualization only (ALWAYS include by default)
  "camsol"            — per-residue solubility scoring
  "esm"               — evolutionary conservation via ESM-2
  "proteinmpnn"       — fixed-backbone sequence redesign via ProteinMPNN
  "rfdiffusion"       — de novo backbone generation (binder design, motif scaffolding)
  "rosetta"           — single-mutation or batch ddG calculation
  "mutation_scan"     — full CamSol + ESM + Rosetta engineering pipeline
  "assembly_analyser" — biological assembly detection, interface mapping
  "disulfide"         — interchain disulfide bond candidate prediction
  "esmfold"           — ESMFold mutant foldability prediction via ESM Atlas API

tool_inputs: dict of tool-specific parameters, e.g.:
  {{"camsol": {{"model_id": "1", "chain": "A"}}}}
  {{"esm":    {{"model_id": "1", "chain": "A"}}}}
  When not using extra tools, set tool_inputs to {{}}.

If the request cannot be safely translated without more information:
{{
  "commands":            [],
  "explanations":        [],
  "warnings":            [],
  "clarification_needed": "A single concise question for the user",
  "confidence":          "low",
  "tools_needed":        ["chimerax"],
  "tool_inputs":         {{}}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSLATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  Only emit commands that appear in the reference below.
2.  Use model specifiers (#1, #2, …) that match the loaded structures in
    session state.  If nothing is loaded and the request needs a model, ask.
3.  Break multi-step workflows into individual commands in the correct order.
4.  Always append "view" after any command that changes geometry or visibility.
5.  Use PDB ID for open (e.g. open 1HSG), not local paths, unless the user
    explicitly says "my file" or gives a filename.
6.  Prefer `matchmaker` over `align` when structures may differ in sequence.
7.  LIGAND RESIDUE NAMES: always use the exact 3-letter code from session state.
    If session state shows "Ligands: MK1", use ":MK1", never ":LIG" or "ligand".
8.  WINDOWS PATHS: save commands must use forward slashes:
      save "C:/Users/andre/Desktop/file.png"
    Construct the full Desktop path as "C:/Users/USERNAME/Desktop/filename.ext"
    using the username from the session working directory if available.
9.  COLOR by* SYNTAX — selector ALWAYS before the keyword, NEVER after:
      color bychain           ← OK (all models)
      color #1 bychain        ← OK (specific model)
      color #1 byelement      ← OK
      color :MK1 byelement    ← OK
      color bychain #1        ← WRONG — triggers "Expected a collection" error
      color byelement #1      ← WRONG — same error
    Applies to every by* keyword: bychain, byelement, bypolymer, byhetero, bymodel.
10. "show as ribbon/cartoon" → `cartoon #N`
11. Publication-quality requests must include in order:
      preset publication
      graphics silhouettes true width 2
      set bgColor white
      lighting soft
12. BACKGROUND: use `set bgColor white` or `set bgColor black`.
    NEVER use `background color white` — that command does not exist.
13. LIGHTING: valid forms are `lighting soft`, `lighting gentle`, `lighting full`,
    `lighting simple`, `lighting flat`, `lighting preset soft`, etc.
    NEVER use `lighting preset publication` — that preset does not exist.
14. Electrostatics → `coulombic`; hydrophobicity → `mlp`.
15. Never emit Python, shell, or OS commands — only ChimeraX commands.
16. ZONE / "within N Å" SELECTIONS — use ChimeraX zone OPERATORS, never the
    Chimera-1 `zone` command. `zone #1/B 4.5` is OLD Chimera-1 syntax and is
    INVALID in ChimeraX 1.11 (it yields an empty selection). Use the zone
    operators `:<` (whole-RESIDUE zone) and `@<` (individual-ATOM zone), combined
    with `&` (intersection) and `~` (negation):
      a) Residues of chain A within 4.5 Å of chain B (an interface):
           select #1/B :<4.5 & #1/A
           info residues sel
      b) Residues within 4 Å of a ligand (e.g. MK1), excluding the ligand itself:
           select :MK1 :<4 & ~:MK1
           info residues sel
      c) ATOMS within 3 Å of that ligand (atom-level zone):
           select :MK1 @<3 & ~:MK1
    Pattern: `<reference> :<<dist> & <target>` selects whole residues of the
    target within <dist> of the reference (`@<` for atoms). Follow a residue zone
    with `info residues sel` so the matched residues are reported. NEVER emit
    `zone ...`.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chimerax         : visualization, selection, measurement, image export  [ACTIVE]
camsol           : per-residue solubility / aggregation-prone scoring  [ACTIVE]
esm              : evolutionary conservation via ESM-2 language model  [ACTIVE]
proteinmpnn      : fixed-backbone sequence redesign                    [ACTIVE — ProteinMPNN/venv312]
rfdiffusion      : de novo backbone diffusion (binder/scaffold/symmetric)[STUB — set RFDIFFUSION_DIR]
rosetta          : stability prediction, ddG calculation               [ACTIVE — DynaMut2 or local]
mutation_scan    : full CamSol + ESM + Rosetta engineering pipeline   [ACTIVE]
assembly_analyser: biological assembly detection, interface mapping    [ACTIVE]
disulfide        : interchain disulfide bond candidate prediction      [ACTIVE]
esmfold          : mutant foldability via ESM Atlas API (free)        [ACTIVE]
rosetta_local    : publication-quality ddG via PyRosetta/WSL2         [ACTIVE if WSL2 configured]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOL ROUTING RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Set tools_needed and tool_inputs when the user asks for computational analysis.
The "chimerax" tool is ONLY needed when you also have ChimeraX setup commands.

SOLUBILITY / AGGREGATION requests:
  "solubility analysis", "aggregation-prone regions", "CamSol", "color by solubility"
  → tools_needed: ["camsol"]          (no initial ChimeraX commands needed)
  → tools_needed: ["chimerax","camsol"] if you also need to open/setup the structure
  → tool_inputs: {{"camsol": {{"model_id": "1", "chain": "A"}}}}
  → commands: [] (no extra commands; CamSol bridge generates the viz automatically)

CONSERVATION / EVOLUTIONARY requests:
  "conservation", "evolutionary conservation", "important residues by evolution",
  "ESM", "mutation tolerance", "color by conservation"
  → tools_needed: ["esm"]
  → tools_needed: ["chimerax","esm"] if setup commands are needed
  → tool_inputs: {{"esm": {{"model_id": "1"}}}}
  → commands: [] or setup commands only

SEQUENCE DESIGN requests:
  "ProteinMPNN", "design sequences", "sequence redesign", "design alternative sequences"
  → tools_needed: ["proteinmpnn"]
  → tool_inputs: {{"proteinmpnn": {{"model_id": "1", "chain": "A"}}}}

DE NOVO BACKBONE DESIGN requests:
  "design a binder", "RFdiffusion", "binder design", "scaffold a motif",
  "design symmetric oligomer", "partial diffusion", "diversify backbone"
  → tools_needed: ["rfdiffusion"]
  → tool_inputs: {{"rfdiffusion": {{
       "mode":     "binder",          # or "motif_scaffold" | "symmetric" | "partial_diffusion"
       "model_id": "1",
       "chain_id": "A",
       "hotspot_residues": [82, 83, 119, 120],  # for binder mode
       "num_designs": 4
     }}}}
  → commands: [] or setup commands only

STABILITY / DDG requests (single mutation or small list):
  "calculate ddG", "how stable", "how destabilising", "mutation V82A",
  "what is the effect of L10K", "is this mutation stabilising"
  → tools_needed: ["rosetta"]
  → tool_inputs: {{"rosetta": {{
       "model_id": "1",
       "chain": "A",
       "mutations": [{{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}}]
     }}}}
  → commands: [] or setup commands only

ENGINEERING / FULL PIPELINE requests (find and rank mutations):
  "suggest mutations", "improve solubility", "engineering candidates",
  "what mutations would help", "stabilise this protein",
  "design mutations to reduce aggregation", "protein engineering"
  → tools_needed: ["mutation_scan"]
  → tool_inputs: {{"mutation_scan": {{
       "model_id": "1",
       "chain": "A",
       "focus": "solubility",
       "analysis_mode": "monomer"   // default
     }}}}
  → commands: [] (visualization generated by the scan pipeline automatically)

ASSEMBLY / INTERFACE requests:
  "analyse as monomer", "monomer analysis", "analyse chain independently"
  → set analysis_mode = "monomer" in mutation_scan tool_inputs (or assembly_analyser)
  → tools_needed: ["mutation_scan"] with analysis_mode: "monomer"

  "analyse as multimer", "analyse as complex", "avoiding interfaces",
  "suggest mutations avoiding chain interfaces", "interface-aware"
  → tools_needed: ["assembly_analyser", "mutation_scan"]
  → tool_inputs: {{
       "assembly_analyser": {{"model_id": "1", "mode": "multimer", "chain_id": "A"}},
       "mutation_scan": {{"model_id": "1", "chain": "A", "focus": "solubility",
                         "analysis_mode": "multimer"}}
     }}

  "find interface residues", "show chain contacts", "what residues are at the interface",
  "show interface between chain A and chain B"
  → tools_needed: ["assembly_analyser", "chimerax"]
  → tool_inputs: {{
       "assembly_analyser": {{"model_id": "1", "mode": "multimer"}},
       "chimerax": {{}}
     }}

DISULFIDE BOND requests:
  "suggest disulfide bonds", "find disulfide candidates", "stabilise the interface",
  "engineer disulfide", "cross-link chains", "predict disulfide",
  "disulfide bridge candidates", "disulfide positions"
  → tools_needed: ["disulfide"]
  → tool_inputs: {{
       "disulfide": {{
         "model_id": "1",
         "chain_a": "A",
         "chain_b": "B"
       }}
     }}

  "improve dimer stability" or "stabilise the complex" (when a multimer is loaded)
  → tools_needed: ["disulfide", "mutation_scan"]
  → tool_inputs: {{
       "disulfide": {{"model_id": "1", "chain_a": "A", "chain_b": "B"}},
       "mutation_scan": {{"model_id": "1", "chain": "A", "focus": "solubility",
                         "analysis_mode": "multimer"}}
     }}

  Chain specification: if the user names chains, use those:
    "suggest disulfides between chain A and chain B"
    → chain_a: "A", chain_b: "B"

FOLDABILITY / STRUCTURE VALIDATION requests:
  "will this mutation fold", "check foldability", "validate design",
  "foldability prediction", "does the mutant fold", "pLDDT", "ESMFold"
  → tools_needed: ["esmfold"]
  → tool_inputs: {{
       "esmfold": {{
         "model_id": "1",
         "sequence": "",         # leave blank — router fetches from session
         "mutation_positions": []
       }}
     }}

PURE VISUALIZATION (default — no extra tools):
  All other requests → tools_needed: ["chimerax"], tool_inputs: {{}}

CHAIN EXTRACTION: If the user specifies a chain (e.g. "analyze chain A"),
put it in tool_inputs: {{"camsol": {{"model_id": "1", "chain": "A"}}}}

EXAMPLE — "Run solubility analysis on the loaded structure":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     [],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["camsol"],
  "tool_inputs":  {{"camsol": {{"model_id": "1"}}}}
}}

EXAMPLE — "Open 1HSG then show me which residues are aggregation-prone":
{{
  "commands":     ["open 1HSG", "cartoon #1", "color bychain", "view"],
  "explanations": ["Fetch 1HSG from RCSB", "Show as cartoon", "Color by chain", "Fit in view"],
  "warnings":     [],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["chimerax", "camsol"],
  "tool_inputs":  {{"camsol": {{"model_id": "1"}}}}
}}

EXAMPLE — "Color by evolutionary conservation":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     ["ESM-2 model (~30 MB) will be downloaded on first use"],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["esm"],
  "tool_inputs":  {{"esm": {{"model_id": "1"}}}}
}}

EXAMPLE — "Calculate ddG for mutation V82A in chain A":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     ["Rosetta requires the PDB file to be available locally or downloadable from RCSB"],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["rosetta"],
  "tool_inputs":  {{
    "rosetta": {{
      "model_id": "1",
      "chain": "A",
      "mutations": [{{"chain": "A", "position": 82, "from_aa": "V", "to_aa": "A"}}]
    }}
  }}
}}

EXAMPLE — "Suggest mutations to improve solubility of chain A":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     ["Full pipeline (CamSol + ESM + Rosetta) may take several minutes"],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["mutation_scan"],
  "tool_inputs":  {{
    "mutation_scan": {{
      "model_id": "1",
      "chain": "A",
      "focus": "solubility"
    }}
  }}
}}

EXAMPLE — "Check whether the L75K mutation would stabilise this protein":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     [],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["rosetta"],
  "tool_inputs":  {{
    "rosetta": {{
      "model_id": "1",
      "mutations": [{{"chain": "A", "position": 75, "from_aa": "L", "to_aa": "K"}}]
    }}
  }}
}}

EXAMPLE — "Analyse solubility of chain A as a monomer":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     [],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["mutation_scan"],
  "tool_inputs":  {{
    "mutation_scan": {{
      "model_id": "1",
      "chain": "A",
      "focus": "solubility",
      "analysis_mode": "monomer"
    }}
  }}
}}

EXAMPLE — "Suggest mutations to improve solubility avoiding chain interfaces":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     ["Multimer analysis will detect interface contacts and exclude those residues"],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["assembly_analyser", "mutation_scan"],
  "tool_inputs":  {{
    "assembly_analyser": {{
      "model_id": "1",
      "mode": "multimer",
      "chain_id": "A"
    }},
    "mutation_scan": {{
      "model_id": "1",
      "chain": "A",
      "focus": "solubility",
      "analysis_mode": "multimer"
    }}
  }}
}}

EXAMPLE — "Show me the interface between chains A and B":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     [],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["assembly_analyser"],
  "tool_inputs":  {{
    "assembly_analyser": {{
      "model_id": "1",
      "mode": "multimer",
      "chain_id": "A",
      "visualize": true
    }}
  }}
}}

EXAMPLE — "Suggest disulfide bonds to stabilise the dimer interface":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     ["Disulfide prediction scores geometry, ESM tolerance, and DynaMut2 stability for each candidate pair"],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["disulfide"],
  "tool_inputs":  {{
    "disulfide": {{
      "model_id": "1",
      "chain_a": "A",
      "chain_b": "B"
    }}
  }}
}}

EXAMPLE — "Find disulfide candidates between chain A and chain B":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     [],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["disulfide"],
  "tool_inputs":  {{
    "disulfide": {{
      "model_id": "1",
      "chain_a": "A",
      "chain_b": "B"
    }}
  }}
}}

EXAMPLE — "Improve the stability of the 1HSG dimer":
{{
  "commands":     [],
  "explanations": [],
  "warnings":     ["Multimer analysis will run disulfide prediction and mutation scan"],
  "clarification_needed": null,
  "confidence":   "high",
  "tools_needed": ["disulfide", "assembly_analyser", "mutation_scan"],
  "tool_inputs":  {{
    "disulfide": {{"model_id": "1", "chain_a": "A", "chain_b": "B"}},
    "assembly_analyser": {{"model_id": "1", "mode": "multimer", "chain_id": "A"}},
    "mutation_scan": {{
      "model_id": "1",
      "chain": "A",
      "focus": "solubility",
      "analysis_mode": "multimer"
    }}
  }}
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHIMERAX COMMAND REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{command_reference}
"""

# ── Helper ─────────────────────────────────────────────────────────────────────

def _load_command_reference() -> str:
    ref = Path(__file__).parent / "chimerax_commands.md"
    if ref.is_file():
        return ref.read_text(encoding="utf-8")
    return "(chimerax_commands.md not found — add it to the project root)"


# ── Translator ─────────────────────────────────────────────────────────────────

class CommandTranslator:
    """
    Converts natural language into ChimeraX commands using the Anthropic API.

    Conversation history is maintained across turns so follow-up requests
    ("now do the same for chain B") work without re-stating context.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model:   str = DEFAULT_MODEL,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set.\n"
                "  Add it to .env.local or set it in your shell."
            )
        self.client  = anthropic.Anthropic(api_key=key)
        self.model   = model
        self._ref    = _load_command_reference()
        self._history: List[Dict[str, str]] = []

        # Pre-format the static block once; it never changes during a session.
        self._static_block: str = _STATIC_SYSTEM.format(command_reference=self._ref)

    # ── Public ─────────────────────────────────────────────────────────────────

    def translate(self, user_input: str, session: SessionState) -> Dict[str, Any]:
        """
        Translate *user_input* into ChimeraX commands.

        Returns::

            {
                "commands":            ["cmd1", ...],
                "explanations":        ["...", ...],
                "warnings":            ["...", ...],
                "clarification_needed": None | "question",
                "confidence":          "high" | "medium" | "low",
            }
        """
        system_blocks = [
            # Block 1: large static content — cached after first call
            {
                "type":          "text",
                "text":          self._static_block,
                "cache_control": {"type": "ephemeral"},
            },
            # Block 2: dynamic session state — not cached (changes every turn)
            {
                "type": "text",
                "text": (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "CURRENT SESSION STATE\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{session.get_context_summary()}"
                ),
            },
        ]

        # Short-circuit for requests that bypass the API entirely
        pre = self._pre_screen(user_input)
        if pre is not None:
            self._history.append({"role": "user",      "content": user_input})
            self._history.append({"role": "assistant",  "content": "{}"}  )  # placeholder
            return pre

        self._history.append({"role": "user", "content": user_input})
        raw = self._call_api(system_blocks)
        self._history.append({"role": "assistant", "content": raw})

        result = self._parse_response(raw)

        # Retry once if JSON parsing failed
        if result.get("_parse_failed"):
            retry_msg = (
                "Your previous response was not valid JSON. "
                "Respond with ONLY a JSON object matching the schema, no other text."
            )
            self._history.append({"role": "user", "content": retry_msg})
            raw2 = self._call_api(system_blocks)
            self._history.append({"role": "assistant", "content": raw2})
            result = self._parse_response(raw2)
            result.pop("_parse_failed", None)

        return result

    def translate_error_fix(
        self,
        failed_command: str,
        error_message:  str,
        session:        SessionState,
    ) -> Dict[str, Any]:
        """
        After a command fails, ask the model for a corrected version.
        Returns the same dict schema as translate().
        """
        prompt = (
            f"The ChimeraX command just executed and failed:\n\n"
            f"  Command : {failed_command}\n"
            f"  Error   : {error_message}\n\n"
            "Please suggest corrected ChimeraX command(s) that achieve the same "
            "goal.  Return the same JSON format."
        )
        return self.translate(prompt, session)

    def add_clarification(self, answer: str) -> None:
        """Append a user clarification to history before the next translate()."""
        self._history.append({"role": "user", "content": f"Clarification: {answer}"})

    def reset_conversation(self) -> None:
        """Discard conversation history (e.g. when switching to an unrelated task)."""
        self._history.clear()

    def trim_history(self, max_pairs: int | None = None) -> None:
        """
        Keep only the most recent *max_pairs* user/assistant pairs.
        Defaults to config.MAX_CONVERSATION_HISTORY.
        """
        limit = max_pairs or config.MAX_CONVERSATION_HISTORY
        if len(self._history) > limit * 2:
            self._history = self._history[-(limit * 2):]

    # ── Internals ──────────────────────────────────────────────────────────────

    # Keywords that unambiguously signal a de novo backbone design request.
    # Checked case-insensitively before the API call so we never hit a content
    # filter on "binder" / "protein design" phrasing.
    _RFD_KEYWORDS: tuple = (
        "rfdiffusion", "rf diffusion",
        "design a binder", "binder design", "protein binder",
        "de novo backbone", "de-novo backbone",
        "scaffold a motif", "motif scaffold",
        "design symmetric oligomer", "partial diffusion",
        "backbone generation", "backbone design",
    )

    def _pre_screen(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Intercept requests that are known to route to unconfigured tools —
        return a direct routing result without calling the API.

        Currently handles: RFdiffusion (de novo backbone design).
        Avoids empty/refused API responses when safety filters trigger on
        "design a binder" or similar biology phrasing.

        Returns a result dict (same shape as translate()), or None to proceed
        normally through the API.
        """
        lower = user_input.lower()
        if any(kw in lower for kw in self._RFD_KEYWORDS):
            return {
                "commands":     [],
                "explanations": [],
                "warnings":     [],
                "confidence":   "high",
                "tools_needed": ["rfdiffusion"],
                "tool_inputs":  {"rfdiffusion": {"mode": "binder"}},
            }
        return None

    def _call_api(self, system_blocks: list) -> str:
        response = self.client.messages.create(
            model      = self.model,
            max_tokens = 2048,
            system     = system_blocks,
            messages   = self._history,
        )
        if not response.content:
            stop = getattr(response, "stop_reason", "unknown")
            raise RefusalError(
                f"API returned empty response (stop_reason={stop!r}). "
                "The prompt may have triggered a safety filter or the "
                "request was malformed."
            )
        return response.content[0].text.strip()

    @staticmethod
    def _parse_response(raw: str) -> Dict[str, Any]:
        """
        Robustly parse the model's JSON.
        Handles: clean JSON, ```json fenced, stray prose around braces.
        Sets _parse_failed=True in the returned dict on unrecoverable failure.
        """
        # Strip markdown fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        text = fenced.group(1) if fenced else raw

        # Strip any prose outside the outermost braces
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            return {
                "commands":             [],
                "explanations":         [],
                "warnings":             [],
                "clarification_needed": None,
                "confidence":           "low",
                "_parse_failed":        True,
            }

        # ── Backwards compat: old schema had needs_clarification + clarifying_question
        if "needs_clarification" in result and "clarification_needed" not in result:
            q = result.pop("clarifying_question", None)
            if result.pop("needs_clarification", False):
                result["clarification_needed"] = q
            else:
                result["clarification_needed"] = None

        # Normalise all keys
        result.setdefault("commands",            [])
        result.setdefault("explanations",        [])
        result.setdefault("warnings",            [])
        result.setdefault("clarification_needed", None)
        result.setdefault("confidence",          "medium")
        result.setdefault("tools_needed",        ["chimerax"])
        result.setdefault("tool_inputs",         {})

        # Coerce confidence to one of three values
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"

        # Ensure tools_needed is always a non-empty list
        if not isinstance(result["tools_needed"], list) or not result["tools_needed"]:
            result["tools_needed"] = ["chimerax"]

        # Ensure tool_inputs is a dict
        if not isinstance(result["tool_inputs"], dict):
            result["tool_inputs"] = {}

        # Pad short explanations list
        while len(result["explanations"]) < len(result["commands"]):
            result["explanations"].append("")

        return result

    def __repr__(self) -> str:
        return (
            f"<CommandTranslator model={self.model!r} "
            f"history_turns={len(self._history) // 2}>"
        )
