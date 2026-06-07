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


# ── Chimera-1 zone-syntax guard ─────────────────────────────────────────────────
# `zone <spec> <dist>` is Chimera-1 syntax, invalid in ChimeraX 1.11 (which uses
# the zone OPERATORS `:<` / `@<`).  This deterministic guard rewrites any such
# command the model still emits, so invalid syntax can never reach ChimeraX:
#   select zone #1/B 4.5 & #1/A      →  select #1/B :<4.5 & #1/A
#   select #1/A & (zone #1/B 4.5)    →  select #1/A & (#1/B :<4.5)
# The `\S+` spec group stops at whitespace, so a trailing ')' stays put.
_ZONE_CMD_RE = re.compile(r"\bzone\s+(\S+)\s+([0-9]*\.?[0-9]+)")
# `volume zone <spec> <dist>` IS a real ChimeraX command (density-map masking) —
# never touch it.
_VOLUME_ZONE_RE = re.compile(r"\bvolume\s+zone\b")


def _sanitize_zone_syntax(
    commands: list, explanations: list
) -> tuple:
    """
    Rewrite Chimera-1 ``zone <spec> <dist>`` (bare or parenthesised) to the
    ChimeraX ``<spec> :<<dist>`` residue-zone operator.  Returns
    (new_commands, new_explanations, notes).

    A command whose ``zone`` cannot be safely rewritten is DROPPED (with a note)
    rather than sent on as invalid syntax.  ``volume zone`` is left untouched.
    """
    new_cmds: list = []
    new_exps: list = []
    notes:    list = []
    for i, cmd in enumerate(commands):
        exp = explanations[i] if i < len(explanations) else ""
        if _VOLUME_ZONE_RE.search(cmd):           # real command — leave as-is
            new_cmds.append(cmd)
            new_exps.append(exp)
            continue
        rewritten = _ZONE_CMD_RE.sub(r"\1 :<\2", cmd)
        if rewritten != cmd:
            notes.append(
                f"Rewrote Chimera-1 zone syntax to the ChimeraX operator: "
                f"{cmd!r} -> {rewritten!r}"
            )
            print(f"  [zone-guard] {cmd!r} -> {rewritten!r}", flush=True)
            cmd = rewritten
        if re.search(r"\bzone\b", cmd):           # still unsafe → drop it
            notes.append(
                f"Dropped an invalid Chimera-1 `zone` command "
                f"(no safe ChimeraX rewrite): {cmd!r}"
            )
            print(f"  [zone-guard] dropped unrewritable {cmd!r}", flush=True)
            continue
        new_cmds.append(cmd)
        new_exps.append(exp)
    return new_cmds, new_exps, notes


# ── Chain-reference macromolecule-scoping guard ─────────────────────────────────
# PRINCIPLE: a direct reference to "chain A–Z" must affect ONLY the macromolecule of
# that chain — never a ligand/solvent/ion that happens to share the chain id; and a
# reference to "ligand" affects ONLY the ligand. They are DISJOINT, every time,
# across ALL operations (color, hide, show, select, zone refs, …).
#
# Why: in 1HSG the MK1 ligand is assigned chain B (B:902), so a bare `/B` selector
# also hits the ligand — literally correct for `/B`, but NOT what "chain B" means
# ("colour chain B red" coloured chain B AND the ligand). This deterministic guard
# scopes every BARE chain selector to the macromolecule by appending
# `& ~ligand & ~solvent & ~ions` (verified live on 1HSG: `/B` = 885 atoms →
# `/B & ~ligand & ~solvent & ~ions` = 757 = protein only; the exclusion form keeps
# ANY polymer, so it also covers nucleic-acid chains). The `ligand` keyword is a
# separate built-in classification and is left untouched (so disjointness holds both
# ways). Backend-agnostic: both backends inherit it via CommandTranslator.translate.
#
# NOT touched: a sub-chain ref (`/B:902`, `/B@CA` — an explicit residue/atom spec),
# an already-scoped ref (`/B & protein` / `/B & ~ligand …`), a negation (`~/B`), and
# a `/` inside a file path (the boundary lookbehind excludes `cache/foo`, `C:/…`).
# The zone OPERATOR forms `/B :<4.5` / `/B:<4.5` ARE scoped (the reference chain is a
# macromolecule too) — runs AFTER _sanitize_zone_syntax so zone syntax is normalised.
_MACRO_SCOPE = "~ligand & ~solvent & ~ions"
_CHAIN_REF_RE = re.compile(
    r"(?:(?<=[\s(|,&])|^)"                  # spec boundary (NOT '~' → leave negations)
    r"(?P<tok>(?:#\d+)?/[A-Za-z0-9]+)"      # optional model + a bare chain id
)


def _scope_chain_refs_to_macromolecule(commands: list) -> tuple:
    """Scope every BARE chain selector in *commands* to the macromolecule
    (`& ~ligand & ~solvent & ~ions`). Returns (new_commands, notes). 1:1 with the
    input (no command is dropped); a command with no bare chain ref is unchanged."""
    new_cmds: list = []
    notes: list = []
    for cmd in commands:
        def _repl(m: "re.Match") -> str:
            tok = m.group("tok")
            end = m.end()
            after = cmd[end:end + 2]
            # explicit residue/atom sub-ref (`:902`, `@CA`) — but NOT the zone
            # operators `:<`/`:>`/`@<`/`@>` — is intentional: leave it.
            if after[:1] in (":", "@") and after[1:2] not in ("<", ">"):
                return m.group(0)
            # the model already scoped it (protein / macromolecule): leave it.
            rest = cmd[end:].lstrip()
            if rest.startswith(("& ~ligand", "&~ligand", "& protein", "&protein")):
                return m.group(0)
            return f"({tok} & {_MACRO_SCOPE})"
        rewritten = _CHAIN_REF_RE.sub(_repl, cmd)
        if rewritten != cmd:
            notes.append(
                f"Scoped chain reference(s) to the macromolecule "
                f"(excluded ligand/solvent/ions): {cmd!r} -> {rewritten!r}"
            )
            print(f"  [chain-scope] {cmd!r} -> {rewritten!r}", flush=True)
        new_cmds.append(rewritten)
    return new_cmds, notes


# ── Open-target validation guard (Bug 4a) ─────────────────────────────────────
# `open <target>` must only reach ChimeraX when the target resolves — an
# existing local file, a valid 4-char PDB-ID, or a .cxs session extension.
# Hallucinated targets ("sequence", "design1.pdb" when the file doesn't exist)
# are DROPPED pre-execution with a clean message instead of silently failing
# inside ChimeraX ("No such file/path").
_OPEN_TARGET_RE = re.compile(
    r'^\s*open\s+(?:"([^"]+)"|\'([^\']+)\'|(\S+))',
    re.IGNORECASE,
)
# PDB IDs are exactly 4 alphanumeric characters (digits or letters, any case).
# Examples: 1HSG, 4UNL, 2LZM, A0A1.  The first character may be a digit.
_PDB_ID_RE = re.compile(r'^[A-Za-z0-9]{4}$')


def _is_valid_open_target(target: str) -> bool:
    """True if *target* is resolvable: 4-char PDB-ID, existing file, or .cxs session."""
    t = (target or "").strip().strip('"\'')
    if not t:
        return False
    if _PDB_ID_RE.match(t):          # valid PDB-ID (4 alphanumeric chars)
        return True
    if Path(t).is_file():            # existing local file
        return True
    if t.lower().endswith(".cxs"):   # session file (may be in sessions/ dir)
        return True
    return False


def _validate_open_targets(commands: list, explanations: list) -> tuple:
    """
    Guard (Bug 4a): drop any ``open <target>`` whose target is not resolvable.
    Returns (new_commands, new_explanations, blocked_messages).

    Non-resolvable targets produce a user-visible warning and the command is
    removed so it never reaches ChimeraX.  The canonical bad cases:
    ``open sequence``, ``open design1.pdb`` (file does not exist).
    """
    new_cmds: list = []
    new_exps: list = []
    blocked:  list = []
    for i, cmd in enumerate(commands):
        exp = explanations[i] if i < len(explanations) else ""
        m = _OPEN_TARGET_RE.match(cmd)
        if m:
            raw = m.group(1) or m.group(2) or m.group(3) or ""
            # Strip any "from <source>" qualifier (e.g. "1HSG from alphafold" → "1HSG")
            target = raw.split()[0] if raw else ""
            if target and not _is_valid_open_target(target):
                blocked.append(
                    f"Blocked 'open {raw}': '{target}' is not a valid PDB-ID or "
                    "existing file path.  Use a 4-letter PDB code (e.g. open 1HSG) "
                    "or provide an existing file path."
                )
                print(f"  [open-guard] blocked {cmd!r}: '{target}' not resolvable", flush=True)
                continue
        new_cmds.append(cmd)
        new_exps.append(exp)
    return new_cmds, new_exps, blocked


# ── Invalid-verb rejection guard (Bug 4b) ─────────────────────────────────────
# ChimeraX has a finite set of registered command verbs.  A command whose
# leading word is not in this set is either hallucinated (find/search) or a
# Chimera-1 legacy verb — neither will succeed.  The guard rejects such commands
# BEFORE they reach ChimeraX so the user sees a clean actionable message instead
# of the raw "Unknown command: find" REST response.
#
# Two-tier strategy:
#   Tier 2 (allowlist/registry): if a live registry is available, block ANY verb
#   absent from it.  This GENERALISES — assembly_analyser, bio_assembly, and any
#   other non-ChimeraX verb are caught automatically without enumerating them.
#   probe_chimerax_verbs() is called in main._handle_request (before translate())
#   whenever the bridge is connected, so the registry is populated in production.
#
#   Tier 1 (denylist fallback): when the registry is unavailable, block only
#   _HALLUCINATED_VERB_DENYLIST.  Unknown-but-potentially-valid verbs are passed
#   through to avoid false positives on unlisted-but-real ChimeraX commands.

# Hallucinated verbs blocked unconditionally (Tier 1) — common English words the LLM
# generates that can never be valid ChimeraX verbs.  This list is the FALLBACK for when
# the live registry is unavailable (bridge disconnected or probe not yet called).
# Unknown non-ChimeraX verbs (assembly_analyser, bio_assembly, tool names, …) are the
# registry's job (Tier 2) and must NOT be enumerated here — the registry generalises.
_HALLUCINATED_VERB_DENYLIST: frozenset = frozenset({
    "find", "search", "locate", "lookup", "retrieve",
    "fetch_sequence", "query", "get_sequence", "list_structures",
    "display_structure",
})

# Module-level registry cache populated by probe_chimerax_verbs().
# None = not yet probed.
_chimerax_verb_registry: Optional[frozenset] = None


def probe_chimerax_verbs(run_command_fn) -> Optional[frozenset]:
    """
    Probe the live ChimeraX command registry via a REST runscript and cache the
    result in ``_chimerax_verb_registry``.  Idempotent — returns the cached set
    on subsequent calls without re-issuing a REST request.  Returns ``None`` if
    the probe fails; the guard then falls back to the denylist.

    ChimeraX 1.11.1 registry surface (verified live):
      ``chimerax.core.commands.cli._command_info.commands.subcommands``
      → OrderedDict of top-level verb → _WordInfo; 193 entries.
      ``list_commands`` does NOT exist in 1.11.1 (import fails).

    *run_command_fn* must accept a ChimeraX command string and return a dict
    with at least a ``"value"`` key (same signature as
    ``ChimeraXBridge.run_command()``).
    """
    global _chimerax_verb_registry
    if _chimerax_verb_registry is not None:
        return _chimerax_verb_registry
    import tempfile
    import os as _os
    # ChimeraX 1.11.1 stores registered top-level commands in
    # `cli._command_info.commands.subcommands` (an OrderedDict of verb→_WordInfo).
    # Verified live: 193 entries incl. open/color/matchmaker/view; find/search absent.
    script = (
        "try:\n"
        "    from chimerax.core.commands import cli\n"
        "    names = sorted(cli._command_info.commands.subcommands.keys())\n"
        "    print(','.join(names))\n"
        "except Exception as _probe_err:\n"
        "    print('ERROR:' + str(_probe_err))\n"
    )
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as _f:
            _f.write(script)
            _tmp = _f.name
        try:
            result = run_command_fn(f'runscript "{_tmp}"')
        finally:
            try:
                _os.unlink(_tmp)
            except Exception:
                pass
        val = (result.get("value") or "").strip()
        if val and not val.startswith("ERROR:") and "," in val:
            verbs = frozenset(v.strip().lower() for v in val.split(",") if v.strip())
            if len(verbs) > 10:      # sanity: real ChimeraX has hundreds of commands
                _chimerax_verb_registry = verbs
                return verbs
    except Exception:
        pass
    return None


def _validate_command_verbs(
    commands:    list,
    explanations: list,
    known_verbs: Optional[frozenset] = None,
) -> tuple:
    """
    Guard (Bug 4b): reject commands whose leading verb is not a registered
    ChimeraX command.  Returns (new_commands, new_explanations, blocked_messages).

    *known_verbs* overrides the module-level registry (useful for testing).
    If neither is set, only the hardcoded denylist is applied so valid
    commands that happen not to be listed are never accidentally dropped.
    """
    active = known_verbs if known_verbs is not None else _chimerax_verb_registry
    new_cmds: list = []
    new_exps: list = []
    blocked:  list = []

    for i, cmd in enumerate(commands):
        exp = explanations[i] if i < len(explanations) else ""
        stripped = cmd.strip()
        if not stripped or stripped.startswith("#"):
            new_cmds.append(cmd)
            new_exps.append(exp)
            continue

        verb = stripped.split()[0].lower()

        # Tier 1: always block the observed-hallucination denylist
        if verb in _HALLUCINATED_VERB_DENYLIST:
            blocked.append(
                f"Rejected '{verb}' in {cmd!r}: "
                f"'{verb}' is not a ChimeraX command (hallucinated verb).  "
                "No action taken."
            )
            print(f"  [verb-guard] rejected hallucinated verb {verb!r}: {cmd!r}", flush=True)
            continue

        # Tier 2: if a live registry is available, block verbs absent from it
        if active is not None and verb not in active:
            blocked.append(
                f"Rejected '{verb}' in {cmd!r}: "
                f"'{verb}' is not in the ChimeraX command registry.  "
                "No action taken."
            )
            print(f"  [verb-guard] rejected unknown verb {verb!r}: {cmd!r}", flush=True)
            continue

        new_cmds.append(cmd)
        new_exps.append(exp)

    return new_cmds, new_exps, blocked


# ── Static system block (cached) ───────────────────────────────────────────────

_STATIC_SYSTEM = """\
You are an expert UCSF ChimeraX command translator integrated into StructureBot.
Your sole job: convert a researcher's natural language request into one or more
precise, executable ChimeraX commands.

SCOPE: StructureBot is a structural-biology tool operating on deposited protein
structures for legitimate research. Routine protein-engineering and design
operations — redesigning interface/dimer/surface residues, making positions
hydrophilic/hydrophobic/charged/polar, excluding or introducing cysteines,
stabilising or destabilising mutations, scanning for solubility, mapping
epitopes/binding sites — are STANDARD computational structural biology. Translate
them to ChimeraX commands (and tool routing) normally; do NOT decline or return
an empty response for such requests. They are benign visualisation/analysis work
on coordinate files, not anything hazardous.
Conversely, a request that is genuinely OUTSIDE this scope — not about a deposited
protein structure at all (file-system/shell/network/OS actions, or other non-
structural tasks) — must be DECLINED via the refusal shape below (refused:true), NOT
forced into invented ChimeraX commands. Decline the out-of-scope, execute the
in-scope; never invent an action you cannot actually perform.

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

  proteinmpnn — use EXACTLY these keys (do not invent synonyms):
    {{"proteinmpnn": {{
        "model_id": "1", "chain": "A",
        "design_scope": "selected" | "interface" | "chain",
        "design_positions": [20, 21, 22, ..., 30],
        "partner_chain": "B",
        "exclude_amino_acids": ["C"],
        "bias_amino_acids": ["D","E","N","Q","H","K","R","S","T"]
    }}}}
    • design_scope "selected" = redesign only the residues the user has SELECTED
      in ChimeraX; "interface" = redesign only the chain/partner_chain interface;
      "chain" = the whole chain. Use "selected" whenever the request says "the
      selected residues"; use "interface" for interface/dimer-interface requests.
    • design_positions = EXPLICIT residue numbers when the user names a range or
      list (e.g. "residues 20-30 of chain A" → [20,21,...,30], "residues 5, 9, 14"
      → [5,9,14]). Always emit the EXPANDED integer list (never the string
      "20-30"). Set design_scope "selected" alongside it. A named explicit range is
      NOT a whole-chain redesign.
    • exclude_amino_acids = HARD exclusions ("no cysteines" → ["C"], "no prolines"
      → ["P"]).
    • bias_amino_acids = SOFT preference ("hydrophilic" / "more soluble" / "reduce
      aggregation" → the polar/charged set D E N Q H K R S T). Omit keys that don't
      apply.

If the request cannot be safely translated without more information, ASK — emit a
pure question with NO tool and NO command (do not guess a chain or default to
acting). But do NOT over-clarify: a clear, answerable request must still execute.
{{
  "commands":            [],
  "explanations":        [],
  "warnings":            [],
  "clarification_needed": "A single concise question for the user",
  "confidence":          "low",
  "tools_needed":        [],
  "tool_inputs":         {{}}
}}

If the request is OUTSIDE StructureBot's scope or is UNSAFE — anything that is not
visualisation/analysis of a protein structure (e.g. file-system, shell, network, or
operating-system actions) — DECLINE. Do NOT invent ChimeraX commands; emit a clean
non-action shape with the structured refusal flag and a one-line reason:
{{
  "commands":            [],
  "explanations":        [],
  "warnings":            ["One sentence: why this is outside StructureBot's scope."],
  "clarification_needed": null,
  "confidence":          "low",
  "tools_needed":        [],
  "tool_inputs":         {{}},
  "refused":             true
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
7.  THE LIGAND: for "the ligand" / "the bound ligand" generically, use the built-in
    `ligand` keyword selector (e.g. `color ligand white`, `style ligand stick`) — it
    selects the bound ligand(s) WITHOUT needing the residue name, and is disjoint from
    the protein chains. For a SPECIFICALLY named ligand you may instead use its exact
    3-letter code from session state (e.g. ":MK1"). NEVER invent a name: do NOT emit
    `:LIG` or `/LIG` — there is no chain or residue called "LIG".
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
    An EXPLICITLY named colour (white / red / blue / …) is applied LITERALLY —
    `color <spec> <name>` (e.g. `color ligand white`, `color #1/B red`). Do NOT
    substitute `byelement`/`byhetero` unless the user explicitly asks to colour BY
    element / heteroatom.
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
16. ZONE / "within N Å" / proximity / interface SELECTIONS — use ChimeraX zone
    OPERATORS, never the Chimera-1 `zone` command. `zone <spec> <dist>` (e.g.
    `select zone #1/B 4.5 & #1/A`, or `select #1/A & (zone #1/B 4.5)`) is OLD
    Chimera-1 syntax and is INVALID in ChimeraX 1.11 — it raises "Expected a
    keyword" / "Expected an objects specifier" and silently selects nothing.
    NEVER emit `zone` / `select zone …`. The zone OPERATORS attach a distance
    to a spec:
      `:<dist` = whole RESIDUES within dist   `:>dist` = residues beyond dist
      `@<dist` = individual ATOMS within dist  `@>dist` = atoms beyond dist
    A proximity/interface selection is ONE command — do NOT decompose it.
    Worked examples:
      • chain-A residues within 4.5 Å of chain B (an interface):
          select #1/B :<4.5 & #1/A
          info residues sel
      • residues within 5 Å of a ligand LIG in chain A:
          select /A:LIG :<5
          info residues sel
      • atoms within 4 Å of the current selection:
          select sel @<4
    Pattern: `<reference> :<<dist> & <target>`. To list/report the selected
    residues use `info residues sel` — NOT `info sel`, which lists atoms.
17. HIDE / SHOW a chain or selection — target the DISPLAYED REPRESENTATION, not
    just atoms. A bare `hide #1/B` only hides atoms/bonds, so a cartoon/ribbon
    stays fully visible (no visible effect). Name the representation:
      • hide chain B (shown as cartoon):   hide #1/B cartoon
      • hide everything for chain B:        hide #1/B target ac   (atoms+cartoon)
      • show chain B's cartoon again:       show #1/B cartoon
    Use the same rule for `show`. When unsure which representation is on, prefer
    `target ac` so both atoms and cartoon are affected.
18. SWITCH a chain/selection to a STICK / SPHERE / BALL representation — a chain
    shown as cartoon has its ATOMS HIDDEN, so a bare `style <spec> stick` styles
    nothing visible (no effect). Emit, IN ORDER:
      show <spec> atoms      ← reveal the atoms (`style` alone NEVER reveals them)
      style <spec> stick     ← stick | sphere | ball+stick, as requested
      hide <spec> cartoon    ← when "stick view" / "as sticks" REPLACES the ribbon
                               (the usual intent for a whole chain)
    Worked example — "show chain A as sticks":
      show #1/A atoms
      style #1/A stick
      hide #1/A cartoon
      view

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


# ── Pluggable LLM backend ───────────────────────────────────────────────────────
# The prompt assembly, JSON schema and parsing live ONCE on CommandTranslator
# (the single shared place). A backend orchestrates one translation using those
# shared primitives — so a future backend (e.g. a local model) reuses the exact
# same prompt/schema and only swaps the model call.  The deterministic guards
# (e.g. _sanitize_zone_syntax) are applied by CommandTranslator AFTER the backend
# returns, so they are backend-agnostic; a backend returns the parsed-but-
# unguarded normalized dict.

class TranslatorBackend:
    """
    Interface for an NL→ChimeraX translation backend.

    ``translate(translator, user_input, session)`` returns the SAME normalized
    translation dict that ``CommandTranslator.translate`` returns
    (``commands`` / ``explanations`` / ``warnings`` / ``clarification_needed`` /
    ``confidence`` / ``tools_needed`` / ``tool_inputs``), built from the given
    translator's shared prompt + parsing helpers.  Everything downstream is
    unchanged regardless of which backend produced it.
    """
    name: str = "base"

    def translate(self, translator: "CommandTranslator",
                  user_input: str, session) -> Dict[str, Any]:
        raise NotImplementedError


class ClaudeBackend(TranslatorBackend):
    """
    Default backend — the Anthropic Claude API.  Wraps the current logic
    verbatim, using the translator's shared system prompt
    (``_build_system_blocks``) and primitives (``_pre_screen`` / ``_call_api`` /
    ``_parse_response`` / ``_history`` / ``client``).
    """
    name = "claude"

    def translate(self, translator: "CommandTranslator",
                  user_input: str, session) -> Dict[str, Any]:
        system_blocks = translator._build_system_blocks(session)

        # Short-circuit for requests that bypass the API entirely.
        pre = translator._pre_screen(user_input)
        if pre is not None:
            translator._history.append({"role": "user",      "content": user_input})
            translator._history.append({"role": "assistant",  "content": "{}"})
            return pre

        translator._history.append({"role": "user", "content": user_input})
        raw = translator._call_api(system_blocks)
        translator._history.append({"role": "assistant", "content": raw})

        result = translator._parse_response(raw)

        # Retry once if JSON parsing failed.
        if result.get("_parse_failed"):
            retry_msg = (
                "Your previous response was not valid JSON. "
                "Respond with ONLY a JSON object matching the schema, no other text."
            )
            translator._history.append({"role": "user", "content": retry_msg})
            raw2 = translator._call_api(system_blocks)
            translator._history.append({"role": "assistant", "content": raw2})
            result = translator._parse_response(raw2)
            result.pop("_parse_failed", None)

        return result


# JSON schema for Ollama's structured output — EXACTLY the normalized 7-key
# translation object (the shared schema; NOT a fork of the prompt text).
# `tools_needed` items are ENUM-constrained to the real router registry
# (config.TRANSLATOR_TOOL_NAMES) so constrained decoding cannot emit a
# misspelled / hallucinated / wrong-cased tool — it must pick from the valid set.
# (Claude does not use this schema, so its path is unaffected.)
TRANSLATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "commands":             {"type": "array", "items": {"type": "string"}},
        "explanations":         {"type": "array", "items": {"type": "string"}},
        "warnings":             {"type": "array", "items": {"type": "string"}},
        "clarification_needed": {"type": ["string", "null"]},
        "confidence":           {"type": "string", "enum": ["high", "medium", "low"]},
        "tools_needed":         {"type": "array",
                                 "items": {"type": "string",
                                           "enum": list(config.TRANSLATOR_TOOL_NAMES)}},
        "tool_inputs":          {"type": "object"},
        # Structured decline signal for out-of-scope / unsafe requests (a clean
        # non-action shape: tools_needed:[], commands:[], refused:true). Optional —
        # absent/false on every normal response.
        "refused":              {"type": "boolean"},
    },
    "required": ["commands", "explanations", "warnings", "clarification_needed",
                 "confidence", "tools_needed", "tool_inputs"],
}

# Module flag: an Ollama model MAY be resident in VRAM. Set after ANY Ollama
# request (forced backend OR a Claude→Ollama fallback both set it), so
# ensure_translator_unloaded() unloads correctly even under TRANSLATOR_BACKEND
# ="claude" when a fallback loaded the local model.
_OLLAMA_MAY_BE_LOADED = False


class OllamaBackend(TranslatorBackend):
    """
    Local-LLM backend via Ollama (benchmark + fallback role; Claude stays the
    default). Same interface and SAME normalized output as ClaudeBackend: reuses
    the translator's shared prompt (`_build_system_blocks`) and normalization
    (`_parse_response`) — no prompt fork.

    Output mechanism (locked): **schema-constrained JSON** via Ollama's native
    `/api/chat` `format` (NOT the model's tool-call template); `temperature=0`
    for determinism; `keep_alive` for explicit VRAM unload control. The
    deterministic guards (`_sanitize_zone_syntax`) run later in
    `CommandTranslator.translate`, identically for both backends.
    """
    name = "ollama"

    # Last `/api/chat` response metadata (prompt_eval_count / done_reason / …),
    # stashed by _chat for the benchmark runner's TRUNCATION instrumentation. Purely
    # additive — never read on the production path, never changes behaviour.
    _LAST_META: Dict[str, Any] = {}

    @classmethod
    def last_meta(cls) -> Dict[str, Any]:
        return dict(cls._LAST_META)

    def translate(self, translator: "CommandTranslator",
                  user_input: str, session) -> Dict[str, Any]:
        # Same pre-screen short-circuit as Claude (backend-agnostic routing).
        pre = translator._pre_screen(user_input)
        if pre is not None:
            translator._history.append({"role": "user",      "content": user_input})
            translator._history.append({"role": "assistant",  "content": "{}"})
            return pre

        system_text = self._system_text(translator, session)
        translator._history.append({"role": "user", "content": user_input})
        content = self._chat(system_text, translator._history)
        translator._history.append({"role": "assistant", "content": content})
        return translator._parse_response(content)

    @staticmethod
    def _system_text(translator: "CommandTranslator", session) -> str:
        # Flatten the shared system blocks into one system message (Ollama has no
        # cache_control blocks). SAME prompt text — no fork.
        return "\n\n".join(
            b["text"] for b in translator._build_system_blocks(session))

    @staticmethod
    def _few_shot() -> list:
        """Targeted few-shot demos (LOCAL backend only) for the categories the
        local model fails — sourced from the disjoint EXAMPLE_POOL. Best-effort:
        no demos if the corpus module is unavailable."""
        try:
            from translator_corpus import few_shot_messages
            return few_shot_messages()
        except Exception:
            return []

    @staticmethod
    def _chat(system_text: str, history: list) -> str:
        global _OLLAMA_MAY_BE_LOADED
        import requests
        # Few-shot demos sit between the system prompt and the real turn; they are
        # NOT persisted into translator._history (so they don't accumulate).
        messages = ([{"role": "system", "content": system_text}]
                    + OllamaBackend._few_shot() + list(history))
        payload = {
            "model":      config.OLLAMA_MODEL,
            "messages":   messages,
            "format":     TRANSLATION_JSON_SCHEMA,   # constrained JSON, not tool-calls
            "stream":     False,
            "think":      False,   # direct schema JSON, no chain-of-thought (faster, cleaner)
            # EXPLICIT sampling — never inherit modelfile/global defaults silently
            # (a silent throttle is indistinguishable from a weak model). Greedy +
            # no repeat penalty (the latter distorts the repeated braces/quotes/keys
            # of constrained JSON), no top-p/top-k clipping, fixed seed for
            # reproducibility, and a generous output cap so the 7-key JSON never
            # truncates mid-generation.
            "options": {
                "temperature":    0,
                "top_p":          1.0,
                "top_k":          0,
                "repeat_penalty": 1.0,
                "seed":           0,
                "num_predict":    int(config.OLLAMA_NUM_PREDICT),
                "num_ctx":        int(config.OLLAMA_NUM_CTX),
            },
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
        }
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json=payload, timeout=int(config.OLLAMA_TIMEOUT),
        )
        _OLLAMA_MAY_BE_LOADED = True   # a request was issued — model may be resident
        resp.raise_for_status()
        data = resp.json() or {}
        # Stash response metadata for the benchmark runner's truncation honesty guard
        # (a silently truncated prompt scored as a model failure was the original
        # num_ctx bug). Additive — the returned content is unchanged.
        OllamaBackend._LAST_META = {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count":        data.get("eval_count"),
            "done_reason":       data.get("done_reason"),
            "num_ctx":           int(config.OLLAMA_NUM_CTX),
            "num_predict":       int(config.OLLAMA_NUM_PREDICT),
        }
        return (data.get("message", {}) or {}).get("content", "") or ""


def ensure_translator_unloaded() -> None:
    """
    Free the Ollama translator model from VRAM (native unload via keep_alive=0).

    INVARIANT — call this BEFORE any GPU-heavy bridge run (ColabFold /
    ProteinMPNN / RFdiffusion-later) so the local LLM never contends with a fold
    for VRAM (a mid-run OOM is the failure mode this prevents). Cheap +
    idempotent; a NO-OP when no Ollama model has been loaded this session (e.g.
    Claude-only — nothing local is resident).
    """
    global _OLLAMA_MAY_BE_LOADED
    if not _OLLAMA_MAY_BE_LOADED:
        return
    try:
        import requests
        requests.post(
            f"{config.OLLAMA_BASE_URL}/api/chat",
            json={"model": config.OLLAMA_MODEL, "messages": [], "keep_alive": 0},
            timeout=10,
        )
    except Exception:
        pass   # best-effort; the keep_alive timer is the backstop
    finally:
        _OLLAMA_MAY_BE_LOADED = False


# Claude API failures that justify a fallback (real transport/auth/quota issues,
# NOT a successful-but-imperfect or refused response). APITimeoutError is a
# subclass of APIConnectionError; AuthenticationError/RateLimitError of APIStatusError.
_CLAUDE_FALLBACK_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.AuthenticationError,
    anthropic.RateLimitError,
)


def is_usage_cap_error(exc: BaseException) -> bool:
    """True for a Claude API USAGE/SPEND-CAP rejection: a ``BadRequestError``
    (HTTP 400, ``invalid_request_error``) whose message reports a usage limit
    ("You have reached your specified API usage limits …").

    NARROW on purpose. A usage-cap is a 400, not a 429 (`RateLimitError`) or 401
    (`AuthenticationError`), so it matched none of `_CLAUDE_FALLBACK_ERRORS`. But a
    genuinely malformed request is ALSO a 400 `BadRequestError` — it must NOT match
    here (it has to surface, never silently reroute to Ollama). The discriminator is
    the "usage limit" phrase in the message, narrowed by the `invalid_request_error`
    error type when the SDK exposes the parsed body.
    """
    if not isinstance(exc, anthropic.BadRequestError):
        return False
    msg = (getattr(exc, "message", None) or str(exc) or "").lower()
    if "usage limit" not in msg:
        return False
    etype = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            etype = str(err.get("type", "")).lower()
    # accept the cap when the type confirms it OR the SDK didn't expose a parsed body
    return etype in ("", "invalid_request_error")


_BACKENDS = {"claude": ClaudeBackend, "ollama": OllamaBackend}


def make_backend(name: str) -> TranslatorBackend:
    """
    Return the translation backend for *name* (``config.TRANSLATOR_BACKEND``).
    Only ``"claude"`` exists today; an unknown name falls back to it.
    """
    cls = _BACKENDS.get((name or "claude").strip().lower())
    if cls is None:
        sys.stderr.write(
            f"[translator] unknown TRANSLATOR_BACKEND {name!r}; using 'claude'.\n")
        cls = ClaudeBackend
    return cls()


# ── Translator ─────────────────────────────────────────────────────────────────

class CommandTranslator:
    """
    Converts natural language into ChimeraX commands via a pluggable LLM backend
    (default: the Anthropic Claude API — see TRANSLATOR_BACKEND / ClaudeBackend).

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

        # Pluggable translation backend (default: Claude). The backend reuses the
        # shared prompt/parsing on this translator; the Claude primitives
        # (client / _call_api / _pre_screen / _static_block) stay here unchanged.
        self._backend: TranslatorBackend = make_backend(
            getattr(config, "TRANSLATOR_BACKEND", "claude")
        )

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

        Four deterministic backend-agnostic guards run on every result:
          1. _sanitize_zone_syntax   — rewrite Chimera-1 zone → :< operator
          2. _scope_chain_refs_to_macromolecule — exclude ligand/solvent/ions
          3. _validate_open_targets  — block unresolvable open targets (Bug 4a)
          4. _validate_command_verbs — reject unregistered leading verbs (Bug 4b)
        """
        result   = self._translate_via_backend(user_input, session)
        cmds     = result.get("commands")     or []
        exps     = result.get("explanations") or []
        warnings = list(result.get("warnings") or [])

        # Guard 1: rewrite Chimera-1 `zone` syntax → ChimeraX zone operator
        cmds, exps, zone_notes = _sanitize_zone_syntax(cmds, exps)
        if zone_notes:
            warnings.extend(zone_notes)

        # Guard 2: scope bare chain refs to the macromolecule (silent correctness fix)
        cmds, _scope_notes = _scope_chain_refs_to_macromolecule(cmds)

        # Guard 3 (Bug 4a): block unresolvable `open <target>` before execution
        cmds, exps, open_blocked = _validate_open_targets(cmds, exps)
        if open_blocked:
            warnings.extend(open_blocked)

        # Guard 4 (Bug 4b): reject commands with an unregistered leading verb
        cmds, exps, verb_blocked = _validate_command_verbs(cmds, exps)
        if verb_blocked:
            warnings.extend(verb_blocked)

        result["commands"]     = cmds
        result["explanations"] = exps
        result["warnings"]     = warnings
        return result

    def _translate_via_backend(self, user_input: str, session) -> Dict[str, Any]:
        """
        Run the active backend, with the LOCKED one-directional fallback:

        - active backend == "claude": on a REAL Claude API failure
          (connection/timeout/auth/rate-limit) AND config.TRANSLATOR_FALLBACK,
          fall back to the local Ollama backend. Any other error (e.g. a
          RefusalError from an empty/declined-but-successful response) propagates
          unchanged — never a fallback trigger.
        - a Claude USAGE/SPEND-CAP rejection (a 400 BadRequestError, NOT a 429/401)
          is also treated as a fallback trigger when active == claude; a genuinely
          malformed 400 must SURFACE, never reroute (see is_usage_cap_error).
        - active backend == "ollama" (forced/benchmark): NEVER falls back to
          Claude; its error surfaces (benchmark honesty).
        """
        try:
            return self._backend.translate(self, user_input, session)
        except _CLAUDE_FALLBACK_ERRORS as exc:
            if not self._may_fall_back():
                raise
            return self._fall_back_to_ollama(user_input, session, type(exc).__name__)
        except anthropic.BadRequestError as exc:
            # A usage/spend-cap rejection is a 400 (invalid_request_error), so it
            # never matched _CLAUDE_FALLBACK_ERRORS. Fall back ONLY for the cap; any
            # other 400 (a real malformed request) re-raises and surfaces.
            if not (is_usage_cap_error(exc) and self._may_fall_back()):
                raise
            return self._fall_back_to_ollama(user_input, session, "usage-limit cap")

    def _may_fall_back(self) -> bool:
        """The one-directional fallback is allowed only when the ACTIVE backend is
        claude and TRANSLATOR_FALLBACK is on — a forced 'ollama' NEVER falls back."""
        return (self._backend.name == "claude"
                and getattr(config, "TRANSLATOR_FALLBACK", True))

    def _fall_back_to_ollama(self, user_input: str, session, reason: str) -> Dict[str, Any]:
        """Drop the dangling user turn ClaudeBackend appended before failing (so the
        fallback doesn't double-append it) and re-run on the local Ollama backend."""
        if self._history and self._history[-1].get("role") == "user":
            self._history.pop()
        sys.stderr.write(
            f"[translator] Claude API failure ({reason}); "
            "falling back to the local Ollama backend.\n")
        return make_backend("ollama").translate(self, user_input, session)

    def _build_system_blocks(self, session: SessionState) -> list:
        """
        Assemble the system prompt blocks (the single shared prompt every backend
        reuses): a cached static block (role + rules + command reference) and an
        uncached dynamic block with the current session state.
        """
        return [
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
                "commands":             [],
                "explanations":         [],
                "warnings":             [],
                "clarification_needed": None,   # fully-normalized 7-key result
                "confidence":           "high",
                "tools_needed":         ["rfdiffusion"],
                "tool_inputs":          {"rfdiffusion": {"mode": "binder"}},
            }
        return None

    def _call_api(self, system_blocks: list) -> str:
        # One automatic retry. An empty/declined response on a routine
        # structural-biology request is usually transient/over-eager — re-issuing
        # the IDENTICAL request typically succeeds — so we try twice before
        # surfacing anything to the user. (Genuine content concerns refuse again.)
        stop = "unknown"
        for _attempt in range(2):
            response = self.client.messages.create(
                model      = self.model,
                max_tokens = 2048,
                system     = system_blocks,
                messages   = self._history,
            )
            if response.content:
                return response.content[0].text.strip()
            stop = getattr(response, "stop_reason", "unknown")
        # Both attempts came back empty — surface the REAL reason (the actual
        # stop_reason), not a generic "safety filter" assumption.
        raise RefusalError(
            f"the model returned no content (stop_reason={stop!r}) after an "
            "automatic retry. Routine structural-biology requests can hit a "
            "transient/over-eager decline — try again or rephrase slightly."
        )

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
        result.setdefault("tools_needed",        [])
        result.setdefault("tool_inputs",         {})
        result["refused"] = bool(result.get("refused", False))

        # Coerce confidence to one of three values
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"

        # tools_needed must be a list. Default to ["chimerax"] ONLY for an ACTION
        # response — a pure clarification (clarification_needed set) or an explicit
        # refusal (refused:true) legitimately carries NO tool (a clean non-action
        # shape), so do NOT inject boilerplate ["chimerax"] there. (That boilerplate
        # previously poisoned eval scoring: it counted as "acting" and tripped the
        # refuse cases' forbidden ["chimerax"].)
        if not isinstance(result["tools_needed"], list):
            result["tools_needed"] = []
        _is_nonaction = bool(result.get("clarification_needed")) or result["refused"]
        if not result["tools_needed"] and not _is_nonaction:
            result["tools_needed"] = ["chimerax"]

        # Ensure tool_inputs is a dict
        if not isinstance(result["tool_inputs"], dict):
            result["tool_inputs"] = {}

        # Defensive: the string-list fields MUST contain only strings. A weaker
        # local model can emit a stray non-string element (e.g. a nested object)
        # even under constrained decoding; downstream + the guards assume strings.
        # (Claude output is already well-formed, so this is a no-op there.)
        for _k in ("commands", "explanations", "warnings"):
            v = result.get(_k)
            result[_k] = [x for x in v if isinstance(x, str)] if isinstance(v, list) else []

        # Pad short explanations list
        while len(result["explanations"]) < len(result["commands"]):
            result["explanations"].append("")

        return result

    def __repr__(self) -> str:
        return (
            f"<CommandTranslator model={self.model!r} "
            f"history_turns={len(self._history) // 2}>"
        )
