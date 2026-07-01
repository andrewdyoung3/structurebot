"""
tool_router.py
--------------
Routes translator output through the appropriate computational tools.

Dispatcher for the StructureBot tool pipeline:
  chimerax     -> ChimeraXBridge  (visualization — always handled by main.py)
  camsol       -> CamsolBridge    (per-residue solubility scoring)
  esm          -> EsmBridge       (evolutionary conservation via ESM-2)
  proteinmpnn  -> ProteinMPNNBridge (fixed-backbone sequence redesign)
  rfdiffusion  -> RFdiffusionBridge (de novo backbone diffusion — stub)

The translator emits a 'tools_needed' list such as:
  ["chimerax"]                      - visualization only (default)
  ["camsol"]                        - solubility + auto-visualization
  ["esm"]                           - conservation + auto-visualization
  ["camsol", "esm"]                 - both analyses, then visualize
  ["chimerax", "camsol"]            - open/setup, then solubility

Workflow
--------
1.  route(translator_result)
      Augments the translator dict with tool routing metadata.
      Safe to call before user confirmation; no tool runs.

2.  execute(routed_result, status_callback=...)
      Runs the non-chimerax tools in order.
      Returns the same dict augmented with step results and viz_commands.
      main.py then runs those viz_commands through ChimeraXBridge.

Each bridge's analyze() method returns a ToolStepResult with:
  - data          : tool-specific output (dict)
  - viz_commands  : ChimeraX commands to visualize results
  - viz_explanations : human-readable explanations for each viz command
  - summary       : one-line result description
  - error         : error string or None
"""

from __future__ import annotations

import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from chimerax_bridge import ChimeraXBridge
    from session_state import SessionState


# ── Tool step result ──────────────────────────────────────────────────────────

class ToolStepResult:
    """Encapsulates the result of one non-chimerax tool step."""

    def __init__(
        self,
        tool:              str,
        success:           bool,
        data:              Optional[Dict[str, Any]] = None,
        viz_commands:      Optional[List[str]] = None,
        viz_explanations:  Optional[List[str]] = None,
        summary:           str = "",
        error:             Optional[str] = None,
        elapsed_ms:        float = 0.0,
    ):
        self.tool             = tool
        self.success          = success
        self.data             = data or {}
        self.viz_commands     = viz_commands or []
        self.viz_explanations = viz_explanations or []
        self.summary          = summary
        self.error            = error
        self.elapsed_ms       = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool":             self.tool,
            "success":          self.success,
            "data":             self.data,
            "viz_commands":     self.viz_commands,
            "viz_explanations": self.viz_explanations,
            "summary":          self.summary,
            "error":            self.error,
            "elapsed_ms":       round(self.elapsed_ms, 1),
        }


# ── Router ─────────────────────────────────────────────────────────────────────

# Hydrophilic residues to bias toward when the request asks for a hydrophilic
# (more soluble / more polar) redesign — used when the translator emits
# `bias_toward: "hydrophilic"` instead of an explicit `bias_amino_acids` list.
_HYDROPHILIC_AAS = "DENQHKRST"

# GPU-heavy bridges. Before dispatching any of these, the local-LLM translator is
# freed from VRAM (translator.ensure_translator_unloaded) so it never contends
# with a fold/design run for GPU memory. No-op under the Claude backend.
_GPU_BRIDGE_TOOLS = frozenset({
    "proteinmpnn", "mpnn_esmfold", "rfdiffusion", "colabfold",
    "validate_design", "esmfold", "esm", "boltz",
    "variant_deviation",   # S4c: may fold the WT reference set (engine-driven) when absent
    "disulfide_discovery", # Mode A: folds N unconstrained seeds (Mode B geometry-readout is cheap)
})

# ColabFold remote-MSA consent — the SINGLE boundary-crossing message both entry points
# (Workbench engine-picker dialog AND the NL colabfold/alphafold intent) surface BEFORE any
# network call. ColabFold is the one fold engine that leaves LOCAL-ONLY (its MSA is remote),
# so crossing that boundary is always a conscious, surfaced choice (default-off, never silent).
_COLABFOLD_REMOTE_CONSENT_MSG = (
    "Fold with ColabFold — uses the remote MSA server (leaves LOCAL-ONLY). "
    "Comparative reference vs the local Boltz fold; the sequence is sent to the "
    "ColabFold MSA server."
)

# Representation classifier — created once so _claude_capped latch persists across calls
_repr_classify_fn = None

# Color classifier — created once so the _claude_capped latch persists across calls
_color_classify_fn = None

# Design-goal classifier — created once (LLM tier for the design-intent op-class)
_design_classify_fn = None


# ── biological-assembly chain-id normalization (pure helpers; unit-testable) ───────────
# Matches submodel addressing `#N.M/<chain>` in BOTH `info chains` (`chain id #2.1/A …`) and
# `info residues` (`residue id #2.1/A:12 …`) output — the char class stops at ':' so a residue
# spec's chain id is captured cleanly. `info residues` is the canonical source because it ALSO
# reports NON-POLYMER chains (glycans/ligands); `info chains` lists polymer chains only, and that
# omission was the glyco-assembly "1 chain" bug: a NAG-bearing copy's glycan chains (B, C, …) were
# invisible to the planner, so the copy-rename targets collided with them (`changechains` rejected)
# and `combine retainIds` then refused the duplicate chain-A copies.
_SUBMODEL_CHAIN_RE = re.compile(r"#(\d+\.\d+)/([^:/\s]+)")
_INT_MODEL_RE = re.compile(r"model id #(\d+)\b")


def _chain_id_candidates():
    """Unique chain-id supply for assembly copies: A–Z, a–z, 0–9, then AA, AB, … (mmCIF auth
    chain ids may be multi-char), so even a large assembly never runs out."""
    import string
    singles = string.ascii_uppercase + string.ascii_lowercase + string.digits
    for c in singles:
        yield c
    for a in singles:
        for b in singles:
            yield a + b


def _parse_submodel_chains(text: str, group_model_id: str):
    """`info residues #N` text → ordered [(submodel_id, [chain_ids]), …] for SUBMODELS of the group
    (`#N.M/chain` lines only). [] when the text has no submodel addressing (already a flat model →
    nothing to normalize). Preserves first-seen order of submodels and of chains within each.

    Sourced from `info residues` (not `info chains`) so NON-POLYMER chains — glycans/ligands like
    NAG, which sit in their own chain ids B, C, … — are enumerated too; otherwise the copy-rename
    plan collides with those hidden chains and normalization fails (the glyco-assembly bug). The
    regex also accepts `info chains` lines, so this stays usable either way."""
    prefix = f"{group_model_id}."
    order: List[str] = []
    by_sub: Dict[str, List[str]] = {}
    for sub, ch in _SUBMODEL_CHAIN_RE.findall(text or ""):
        if not sub.startswith(prefix):
            continue
        if sub not in by_sub:
            by_sub[sub] = []
            order.append(sub)
        if ch not in by_sub[sub]:
            by_sub[sub].append(ch)
    return [(s, by_sub[s]) for s in order]


def plan_assembly_chain_renames(submodel_chains):
    """Plan unique chain ids across assembly copies. *submodel_chains* = ordered
    [(submodel_id, [chain_ids])]. Returns ``(renames, final_chains)`` where *renames* =
    [(submodel_id, old_chain, new_chain)] ONLY for chains that COLLIDE with an already-claimed id
    (a chain whose id is still free is LEFT ALONE — a hetero ASU's first copy keeps A,B; only the
    duplicating copies are relabelled), and *final_chains* is the resulting full chain-id list.
    Pure / unit-testable."""
    used: set = set()
    renames: List[Tuple[str, str, str]] = []
    final: List[str] = []
    for sub, chains in submodel_chains:
        for ch in chains:
            if ch not in used:
                used.add(ch)
                final.append(ch)
                continue
            new = next(c for c in _chain_id_candidates() if c not in used)
            used.add(new)
            final.append(new)
            renames.append((sub, ch, new))
    return renames, final


class ToolRouter:
    """
    Routes translator output through the correct computational tools.

    Usage::

        router = ToolRouter(bridge, session)

        # After translation — augments result, no execution yet:
        routed = router.route(translator_result)

        # After user confirms — runs computational tools:
        routed = router.execute(routed, status_callback=console.status)
        # routed["all_viz_commands"] and routed["all_viz_explanations"]
        # are now populated; main.py runs them through the bridge.
    """

    _TOOL_ICONS: Dict[str, str] = {
        "chimerax":          "🎨",
        "camsol":            "💧",
        "esm":               "🧬",
        "esmfold":           "🔮",
        "boltz":             "🧬🔮",
        "proteinmpnn":       "🔬",
        "mpnn_esmfold":      "🔬🔮",
        "rfdiffusion":       "🌀",
        "rosetta":           "⚗️",
        "mutation_scan":     "🔬⚗️",
        "assembly_analyser": "🔗",
        "bio_assembly":             "🏗️",
        "interface_stabilization": "🔗🛡️",
        "disulfide":               "🔗⚗️",
        "disulfide_discovery":     "🔗📊",
        "disulfide_geometry":      "🔗📐",
        "disulfide_scan":          "🔗🔍",
        "disulfide_interface_scan": "🔗🤝",
        "disulfide_ddg_estimate":  "🔗⚡",
        "proline":           "🧪",
        "proline_scan":      "🧪🔍",
        "proline_ddg_estimate": "🧪⚡",
        "cavity_scan":       "🕳️🔍",
        "cavity_ddg_estimate": "🕳️⚡",
        "saltbridge_scan":      "🧂🔍",
        "saltbridge_ddg_estimate": "🧂⚡",
        "glycan":            "🍬",
        "glycan_positions":  "🍬🔮",
        "netnglyc":          "🔬🍬",
        "salt_bridge":       "⚡",
        "cavity":            "🕳",
        "double_mutant":     "⚗️🔗",
        "validate_ddg":          "⚗️✅",
        "colabfold":             "🧬🔮",
        "validate_design":       "🧬✅",
        "conformer_comparison":  "🔄",
        "variant_deviation":     "📐",
        "representation":        "🖼️",
        "color":                 "🎨",
        "transparency":          "👻",
        "design_goal":           "🧪",
    }

    # Conformer-comparison intent keywords.  Specific enough to avoid false
    # positives with existing tools; compound check covers "compare.*conformer".
    _CONFORMER_COMPARISON_KEYWORDS: tuple = (
        "compare conformer",
        "conformer comparison",
        "conformational change",
        "conformational comparison",
        "per-residue shift",
        "per-residue displacement",
        "residue shift",
        "anchored overlay",
        "overlay anchored on",
        "anchor on residues",
        "anchor the overlay",
        "anchoring the conserved",
        "hinge motion",
        "domain motion",
        "rigid core and show",
        "morph analysis",
        "open vs closed",
        "open and closed",
        "two conformers",
        "two conformations",
        "conformation comparison",
    )

    # Validate-design meta-tool: the HEAVY ColabFold+Rosetta orchestrator. It
    # fires ONLY on explicit high-accuracy phrasing — BARE "validate design" /
    # "check this design" etc. deliberately fall through to the light
    # mpnn_esmfold / ESMFold fast screen (its prior behaviour). The "... with
    # colabfold/alphafold" case is handled separately in _detect (compound).
    # Checked BEFORE the bare match in route(), so a QUALIFIED phrase like
    # "high-accuracy validate design" hits the meta-tool while bare
    # "validate design" does not.
    _VALIDATE_DESIGN_KEYWORDS: tuple = (
        "full validation",
        "full design validation",
        "high-accuracy validation", "high accuracy validation",
        "high-confidence validation", "high confidence validation",
        "high-accuracy validate", "high accuracy validate",
        "high-confidence validate", "high confidence validate",
        "thoroughly validate", "thorough validation",
    )

    # Explicit ColabFold (AF2) folding keywords — the high-accuracy structure-
    # prediction path. Kept tight so it neither swallows nor is swallowed by the
    # esmfold / mpnn_esmfold pipeline (see route()).
    _COLABFOLD_KEYWORDS: tuple = (
        "colabfold", "alphafold", "alpha fold", "af2",
    )

    # Oligomer words → homo-oligomer copy count (used for both intent + parsing).
    _OLIGOMER_COPIES: Dict[str, int] = {
        "monomer": 1, "monomeric": 1,
        "dimer": 2, "homodimer": 2, "dimeric": 2,
        "trimer": 3, "homotrimer": 3, "trimeric": 3,
        "tetramer": 4, "homotetramer": 4, "tetrameric": 4,
        "pentamer": 5, "pentameric": 5,
        "hexamer": 6, "hexameric": 6,
    }

    # Keywords that signal a biological-assembly generation request.
    # "open as X" / "work as X" / "generate biological assembly" etc.
    # Kept tight: the key distinguisher is the "biological assembly" / oligomer
    # + generation verb pairing, NOT bare oligomer mentions (which appear in
    # mutation-scan requests like "redesign the interface of the dimer").
    _BIO_ASSEMBLY_KEYWORDS: tuple = (
        "generate biological assembly",
        "generate the biological assembly",
        "generate assembly",   # covers "generate assembly 2" with explicit id
        "build biological assembly",
        "build the biological assembly",
        "build biological unit",
        "build the biological unit",
        "apply crystal symmetry",
        "apply symmetry",
        "generate the full assembly",
        "make the full assembly",
        "make the full tetramer",
        "make the full dimer",
        "make the full trimer",
        "make the full hexamer",
        "open as tetramer",
        "open as a tetramer",
        "open as dimer",
        "open as a dimer",
        "open as trimer",
        "open as a trimer",
        "open as hexamer",
        "open as a hexamer",
        "open as oligomer",
        "work as tetramer",
        "work as a tetramer",
        "work as dimer",
        "work as a dimer",
        "work as trimer",
        "work as a trimer",
        "work as hexamer",
        "work as the full",
        "view as tetramer",
        "view as a tetramer",
        "show full assembly",
        "show the full assembly",
        "show biological assembly",
        "show the biological assembly",
        "full biological assembly",
        "complete biological assembly",
        "load biological assembly",
        "load the biological assembly",
        "expand to full assembly",
    )

    # Keywords that signal an interface-stabilization request.
    # Tight enough to avoid false positives with mutation-scan / disulfide.
    # Checked BEFORE generic disulfide or mutation-scan routing.
    _INTERFACE_STABILIZATION_KEYWORDS: tuple = (
        "stabilize the interface",
        "stabilise the interface",
        "stabilize interface",
        "stabilise interface",
        "stabilize the dimer interface",
        "stabilize the tetramer interface",
        "stabilize the trimer interface",
        "stabilize the oligomer interface",
        "lock the interface",
        "lock the dimer",
        "lock the tetramer",
        "lock with disulfide",
        "lock with a disulfide",
        "crosslink the interface",
        "crosslink the dimer",
        "crosslink the tetramer",
        "engineer interface disulfide",
        "interface disulfide",
        "inter-subunit disulfide",
        "inter-chain disulfide",
        "interchain disulfide",
        "interface stabilization",
        "interface stabilisation",
        "strengthen the interface",
        "reinforce the interface",
        "reinforce the assembly",
        "detect interfaces",
        "characterize interface",
        "characterise interface",
        "characterize the interface",
        "characterise the interface",
        "interface contacts",
        "interface residues",
        "map the interface",
        "identify the interface",
        "buried interface area",
        "buried surface area",
    )

    # Keywords that signal a ProteinMPNN + ESMFold validation request.
    # Checked case-insensitively; any match replaces 'proteinmpnn' (or 'esmfold'
    # when session has MPNN results) in the pipeline.
    _MPNN_ESMFOLD_KEYWORDS: tuple = (
        "validate design",
        "validate sequence",
        "fold design",
        "fold sequence",
        "esmfold mpnn",
        "esmfold proteinmpnn",
        "mpnn esmfold",
        "proteinmpnn esmfold",
        "fold redesign",
        "structural integrity",
        "assess fold",
        "check fold",
        "validate mpnn",
        "esmfold top",
        "fold top",
        "sequences to assess",
        "designed sequences",
        "redesigned sequences",
        # Phrases that arise when the user refers to "top N sequences" after redesign
        "top 2-3 sequences",
        "top 2–3 sequences",   # en-dash variant (U+2013)
        "top 3 sequences",
        "top sequences",
        "assess structural integrity",
        "structural fidelity",
    )

    # Contextual keywords: when present alongside a session with MPNN results,
    # an 'esmfold' dispatch is redirected to _run_mpnn_esmfold().
    _MPNN_CONTEXT_KEYWORDS: tuple = (
        "sequence",
        "design",
        "top",
        "integrity",
        "redesign",
    )

    # Keywords that trigger the "show designed sequences" fast-path handler
    # (bypasses LLM when session already has ProteinMPNN results).
    _SEQUENCE_DISPLAY_KEYWORDS: tuple = (
        "show designed sequences",
        "show design sequences",
        "show sequences",
        "list designed sequences",
        "list sequences",
        "print designed sequences",
        "print sequences",
        "display designed sequences",
        "display sequences",
        "output sequences",
        "output the sequences",
        "what are the designed sequences",
        "what are the sequences",
        "show mpnn sequences",
        "what sequences did",
        "show me the sequences",
        "sequences for the redesigns",
        "redesign sequences",
    )

    # Verbs used in fuzzy sequence-display matching (see handle_sequence_display_command).
    _SEQUENCE_DISPLAY_VERBS: tuple = (
        "show", "output", "print", "list", "display", "what",
    )

    # Phrasings that mean "RUN/RE-RUN a redesign" — a display request must never
    # match these (re-running is stochastic and overwrites the design).
    _MPNN_RUN_TRIGGERS: tuple = (
        "with proteinmpnn", "with protein mpnn", "with mpnn", "with ligandmpnn",
        "using proteinmpnn", "using mpnn", "run proteinmpnn", "run mpnn",
        "redesign chain", "redesign the chain", "redesign again", "rerun", "re-run",
        "another design", "new design", "design again", "redesign the dimer",
        "redesign the interface", "redesign selected", "make it hydrophilic",
        "make it hydrophobic",
    )
    # Primary visualization verbs: a request that OPENS with one of these is a
    # structural viz op — NOT an MPNN sequence-display request — regardless of
    # whether "redesigned"/"redesign" appears later.  This prevents "remove chain B
    # … show the overlay … redesigned chain A" from dumping the MPNN design list.
    # (Acceptance case R1 / Bug 5 fix.)
    _PRIMARY_VIZ_VERBS: tuple = (
        "remove", "hide", "close", "overlay", "color", "colour",
        "cartoon", "style", "align", "matchmaker",
    )
    _MPNN_DISPLAY_VERBS: tuple = (
        "show", "output", "print", "display", "list", "what", "see", "view",
        "give", "tell", "get",
    )
    # Phrasings that ask for the WT-vs-redesign ALIGNMENT / change set.
    _MPNN_ALIGNMENT_KEYWORDS: tuple = (
        "alignment", "wt vs", "wildtype vs", "vs wildtype", "vs wt",
        "compare to wildtype", "compared to wildtype", "compare to wt",
        "what changed", "what positions changed", "which positions changed",
        "show the changes", "the changes", "show the redesign", "redesign view",
        "what mutations", "mutations made", "side by side", "side-by-side",
    )

    _FASTA_EXPORT_KEYWORDS: tuple = (
        "export sequences",
        "save sequences",
        "save fasta",
        "export fasta",
        "write sequences",
        "download sequences",
    )

    _SALT_BRIDGE_KEYWORDS: tuple = (
        "salt bridge",
        "salt-bridge",
        "electrostatic interaction",   # narrowed from "electrostatic" — avoids matching
        "ionic interaction",           # "show electrostatic surface" (visualization)
        "charge pair",
        "engineer salt",
        "new salt bridge",
    )

    # Cavity keyword list — does NOT include bare "void" (matches "avoiding").
    # Bare "void" / "voids" is handled with word-boundary regex in
    # _detect_cavity_intent().  Only explicit cavity-specific compound forms here.
    # Removed: "hydrophobic core" (too broad — matches mutation scan requests)
    _CAVITY_KEYWORDS: tuple = (
        "cavit",           # cavity, cavities, cavity-filling
        "internal void",   # compound — "internal voids" also matches via substring
        "buried void",     # compound — "buried voids" also matches via substring
        "fill void",
        "cavity fill",
        "fill cavity",
        "packing defect",
        "buried pocket",   # pocket + burial context
        "internal pocket",
        "fill pocket",
        "hollow",          # hollow protein core
        "fpocket",         # explicit cavity-detection tool name
        "tunnel",          # protein tunnel / channel
    )

    # Assembly / dimer keywords trigger chains=None (all chains) in find_cavities.
    # Removed: "binding pocket" — too broad; a binding-pocket analysis request is
    # not the same as asking for dimer cavity detection.
    _CAVITY_ASSEMBLY_KEYWORDS: tuple = (
        "full dimer",
        "full assembly",
        "both chains",
        "all chains",
        "dimer cavity",
        "oligomer",
        "assembly cavity",
        "interface cavity",
        "interface pocket",
        "active site cavity",
    )

    # Keywords that signal an N-glycosylation site scan request.
    # Checked case-insensitively; any match rewrites tools_needed to ["glycan"]
    # and clears any translator clarification_needed flag.
    _GLYCAN_KEYWORDS: tuple = (
        "glycan",
        "glycosylat",        # glycosylation, glycosylated, glycosylate, …
        "n-linked",
        "nxs",               # N-X-S sequon notation
        "nxt",               # N-X-T sequon notation
        "sequon",
        "glycoengineering",
        "glycosylation site",
        "n-glycan",
        "sugar",
    )

    # Keywords that signal a projection-aware glycosylation position scan
    # (distinct from the classic NXS/T sequon scan).  Checked BEFORE the
    # general glycan block so these phrases are never swallowed by _GLYCAN_KEYWORDS.
    _GLYCAN_POSITIONS_KEYWORDS: tuple = (
        "glycosylation positions",
        "glycan candidates",
        "glycan sites",
        "domain masking",
        "immunosilence",
        "glycan engineering candidates",
        "surface glycosylation",
        "projection-aware glycosylation",
    )

    # Keywords that trigger a standalone NetNGlyc OST recognition prediction.
    # These fire when the user explicitly requests OST/NetNGlyc scoring
    # without going through glycan_positions (which auto-calls NetNGlyc on
    # top-5 candidates already).
    _NETNGLYC_KEYWORDS: tuple = (
        "netnglyc",
        "ost recognition",
        "ost score",
        "oligosaccharyltransferase",
        "ost prediction",
        "glycosylation efficiency",
        "sequon recognition",
        "glycan ost",
        "ost glycosylation",
    )

    # Keywords that signal a mutation scan / engineering scan request.
    # Used as a fallback when the translator returns an unexpected tool
    # (e.g. cavity) for a clear solubility / mutation engineering request.
    # Matches both "mutation" (singular) and "mutations" (plural) via "mutati".
    _MUTATION_SCAN_KEYWORDS: tuple = (
        "suggest mutation",       # singular
        "suggest mutations",      # plural
        "improve solubility",     # solubility improvement without explicit mutation
        "reduce aggregation",     # aggregation engineering
        "mutation scan",          # explicit scan name
        "solubility scan",        # explicit scan name
        "mutation to improve",    # singular + context
        "mutations to improve",   # plural + context
        "engineering candidate",  # engineering language
        "stabilising mutation",   # stability mutation (British)
        "stabilizing mutation",   # stability mutation (American)
        "stability mutation",
    )

    # Keywords that trigger double mutant pair scoring.
    # Checked BEFORE mutation_scan dispatch so these phrases are never
    # sent through the single-point scan pipeline.
    _DOUBLE_MUTANT_KEYWORDS: tuple = (
        "double mutant",
        "combine mutations",
        "combined mutations",
        "synergistic",
        "epistasis",
        "pair mutations",
        "two mutations",
        "mutation combination",
        "epitope preserv",
        "preserve epitope",
        "scaffold stabili",
    )

    # Keywords that signal the high-accuracy ddG VALIDATION tier (multi-trajectory
    # median ddG on a small explicit set of mutations / top scan candidates).
    # Distinct from the fast single-trajectory mutation_scan path. Checked
    # case-insensitively; checked BEFORE mutation_scan / double_mutant overrides.
    _VALIDATE_DDG_KEYWORDS: tuple = (
        "validate ddg",
        "validate the ddg",
        "high accuracy ddg",
        "high-accuracy ddg",
        "high confidence stability",
        "high-confidence stability",
        "confirm ddg",
        "confirm the ddg",
        "confirm stability",
        "validate stability",
        "high accuracy stability",
        "multi-trajectory",
        "multitrajectory",
    )

    # Keywords that signal a proline substitution scan request.
    # Checked case-insensitively; any match overrides generic mutation_scan routing.
    _PROLINE_KEYWORDS: tuple = (
        "proline",           # catches "proline mutations", "proline scan", etc.
        "pro substitut",     # "pro substitution(s)"
        "pro mutation",      # "pro mutation(s)"
        "backbone stabili",  # "backbone stabilisation / stabilise"
        "entropic stabili",  # "entropic stabilisation"
        "phi angle",         # explicit backbone geometry reference
        "φ angle",      # φ angle (Unicode phi)
        "rigidif",           # "rigidify", "rigidification"
        "proline scan",      # explicit tool name
        "proline candidate", # explicit candidate language
    )

    def __init__(
        self,
        bridge:  "ChimeraXBridge",
        session: "SessionState",
    ):
        self.bridge  = bridge
        self.session = session

        # Bridges are instantiated lazily on first use
        self._camsol_bridge:           Optional[Any] = None
        self._esm_bridge:              Optional[Any] = None
        self._esmfold_bridge:          Optional[Any] = None
        self._boltz_bridge:            Optional[Any] = None
        self._proteinmpnn_bridge:      Optional[Any] = None
        self._mpnn_esmfold_pipeline:   Optional[Any] = None
        self._rfdiffusion_bridge:      Optional[Any] = None
        self._rosetta_bridge:          Optional[Any] = None
        self._mutation_scanner:        Optional[Any] = None
        self._assembly_analyser:       Optional[Any] = None
        self._disulfide_bridge:        Optional[Any] = None
        self._proline_bridge:          Optional[Any] = None
        self._glycan_bridge:           Optional[Any] = None
        self._netnglyc_bridge:         Optional[Any] = None
        self._salt_bridge_bridge:      Optional[Any] = None
        self._cavity_bridge:           Optional[Any] = None
        self._double_mutant_bridge:    Optional[Any] = None
        self._colabfold_bridge:        Optional[Any] = None

        # Per-target representation snapshots keyed by ChimeraX spec string (e.g. "#1")
        # Each value is a list of restore commands captured before the last render.
        self._repr_snapshots:          Dict[str, List[str]] = {}
        # Per-target transparency level (0=opaque … 100=invisible), keyed by the resolved
        # atomspec, so a RELATIVE request ("increase transparency by 50%") adjusts the
        # tracked level and emits an absolute `transparency` (ChimeraX has no relative form).
        self._transparency_levels:     Dict[str, int] = {}

    # ── Phase 1: Route (no execution) ─────────────────────────────────────────

    # ── Cavity intent helpers ─────────────────────────────────────────────────

    @classmethod
    def _detect_cavity_intent(cls, text: str) -> bool:
        """
        Return True if *text* signals a cavity detection / filling request.

        "void" / "voids" are matched with a word-boundary regex (\\bvoids?\\b) so
        that "avoiding" does NOT trigger cavity routing.  All other cavity keywords
        are matched as substrings (they are long enough to be unambiguous).
        """
        lower = text.lower()
        if re.search(r'\bvoids?\b', lower):
            return True
        return any(kw in lower for kw in cls._CAVITY_KEYWORDS)

    # ── Mutation scan intent helpers ──────────────────────────────────────────

    @classmethod
    def _detect_mutation_scan_intent(cls, text: str) -> bool:
        """
        Return True if *text* signals a mutation scan / engineering request.

        Matches 'mutation' (singular) and 'mutations' (plural) via shared
        substring, plus standalone solubility / aggregation engineering phrases.
        Used as a routing fallback when the translator returns an unexpected tool.
        """
        lower = text.lower()
        # Substring check covers mutation / mutations / mutational
        if "mutati" in lower and any(
            kw in lower for kw in ("solubility", "aggregation", "stabili", "engineer",
                                   "improve", "suggest", "scan", "candidate")
        ):
            return True
        return any(kw in lower for kw in cls._MUTATION_SCAN_KEYWORDS)

    # ── Mutation-scan tiering: scope + opt-in Rosetta (Priority 1) ─────────────
    # Deep-tier mutation grid: each scoped position expands to candidates_per_pos
    # substitutions (mutation_scanner defaults), so the Rosetta call count is
    # positions × subs, NOT positions.  Used by the runtime estimate.
    _DEFAULT_CANDS_PER_POS  = 3      # mutation_scanner.scan() default
    _DEFAULT_MAX_CANDIDATES = 20     # whole-chain scan cap

    # Explicit deep-tier trigger.  Word-boundary so bare "rose" does NOT match
    # ("rosette", "the value rose", "rosé") — same class as the viewer
    # "drop"-in-"hydrophobic" fix.
    _TIER_ROSETTA_RE = re.compile(r"\b(?:rosetta|rosie)\b", re.IGNORECASE)
    # Thoroughness phrases do NOT auto-run deep — they raise a tier-choice surface.
    _THOROUGHNESS_RE = re.compile(
        r"\b(?:exhaustive|deep[\s-]?dive|comprehensive|"
        r"gold[\s-]?standard|gold[\s-]?quality)\b",
        re.IGNORECASE,
    )
    # Live-selection scope phrases.
    _SELECTION_SCOPE_RE = re.compile(
        r"\b(?:selected|selection|highlighted)\b", re.IGNORECASE
    )
    # Deep-tier SHORTLIST opt-in (validate top-K instead of the full grid).
    _SHORTLIST_RE = re.compile(
        r"\b(?:shortlist|short-list|top[- ]?\d+|top\s+candidates|"
        r"just\s+the\s+top|best\s+candidates)\b", re.IGNORECASE
    )
    # ASYMMETRIC ddG opt-in (single cached-WT reference; faster, noisier basis).
    _ASYMMETRIC_RE = re.compile(r"\basymmetric\b", re.IGNORECASE)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 90:
            return f"~{int(round(seconds))} s"
        if seconds < 3600:
            return f"~{seconds / 60:.0f} min"
        return f"~{seconds / 3600:.1f} h"

    def _resolve_deep_workers(self, n_residues: Optional[int] = None) -> int:
        """Resolved deep-tier pool size — capped to hardware AND the POSE-SIZE-
        scaled per-worker footprint (so a large complex shrinks the pool, matching
        what the real run will do; prevents the estimate from dividing by more
        workers than will actually fit)."""
        import config as _cfg
        try:
            from rosetta_bridge import resolve_rosetta_workers, worker_footprint_mb
            return resolve_rosetta_workers(
                configured     = getattr(_cfg, "ROSETTA_MAX_WORKERS", 8),
                physical_cores = getattr(_cfg, "ROSETTA_PHYSICAL_CORES", 8),
                mem_budget_mb  = getattr(_cfg, "ROSETTA_WSL_MEM_BUDGET_MB", 12000),
                per_worker_mb  = worker_footprint_mb(n_residues),
            )
        except Exception:
            return 1

    def _pose_residue_count(self, model_id: str) -> Optional[int]:
        """Total residues of the FULL pose Rosetta will load (ALL chains — e.g. the
        whole 2HHB tetramer, 574, not just the scanned chain).  Uses the cached/
        downloaded PDB (the same file the deep run uses).  None if unavailable."""
        try:
            from rosetta_bridge import count_pdb_residues
            p = self._ensure_pdb_file(model_id)
            return count_pdb_residues(p) if p else None
        except Exception:
            return None

    def _estimate_rosetta_runtime(
        self,
        n_mutations: Optional[int],
        n_residues:  Optional[int],
        workers:     int = 1,
    ) -> Optional[str]:
        """
        Honest deep-tier estimate, or None if the mutation count is unknown.

        est ≈ n_mutations × per_mutation_sec(n_residues) / workers
        — counts the ACTUAL dispatched mutations (positions × candidates_per_pos),
        scales per-mutation cost SUPER-linearly with the FULL pose size, and
        divides by the (footprint-capped) worker count.  Biased high; "approximate".
        """
        secs = self._estimate_rosetta_secs(n_mutations, n_residues, workers)
        return self._format_duration(secs) if secs is not None else None

    def _estimate_rosetta_secs(
        self,
        n_mutations: Optional[int],
        n_residues:  Optional[int],
        workers:     int = 1,
    ) -> Optional[float]:
        """Raw seconds for the deep-tier estimate (None if mutation count unknown)."""
        if not n_mutations:
            return None
        from rosetta_bridge import per_mutation_sec, parallel_efficiency
        # Large-pose guard (BUILD 3): above the measured anchor, single-channel
        # DDR5 makes parallel scaling sub-linear, so shrink the effective worker
        # count → estimate stays biased HIGH (never undershoots a big complex).
        eff_workers = max(1.0, max(1, workers) * parallel_efficiency(n_residues))
        return n_mutations * per_mutation_sec(n_residues) / eff_workers

    def _parse_scan_scope(
        self,
        user_input: str,
        model_id:   str,
        chain:      str,
    ) -> Tuple[Optional[List[int]], bool]:
        """
        Parse an assignable scan scope from *user_input*.

        Returns (positions, scope_requested):
          - (None, False) — no scope named → whole chain (current behaviour).
          - (list, True)  — explicit residues/range OR the live ChimeraX selection
                            (may be [] when a scope was requested but resolved to
                            nothing → caller errors, no full-chain fallback).
        """
        text = user_input or ""

        # (1) Live-selection scope ("the selected residues", "current selection").
        if self._SELECTION_SCOPE_RE.search(text):
            try:
                sel = self._read_selected_residues(model_id, chain)
            except Exception:
                sel = []
            return sorted(set(sel)), True

        # (2) Explicit range: "residues 30-45", "positions 30 to 45", or bare "30-45".
        rng = re.search(
            r"(?:residues?|positions?|res)\s+(\d+)\s*(?:-|–|—|to|through)\s*(\d+)",
            text, re.IGNORECASE,
        ) or re.search(r"\b(\d+)\s*(?:-|–|—)\s*(\d+)\b", text)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            if a > b:
                a, b = b, a
            return list(range(a, b + 1)), True

        # (3) Explicit list: "residues 30, 32, 40" / "positions 12 and 15".
        lst = re.search(
            r"(?:residues?|positions?)\s+((?:\d+\s*(?:,|and|&)\s*)+\d+)",
            text, re.IGNORECASE,
        )
        if lst:
            nums = [int(x) for x in re.findall(r"\d+", lst.group(1))]
            return sorted(set(nums)), True

        # (4) Single residue: "residue 30".
        one = re.search(r"\b(?:residue|position)\s+(\d+)\b", text, re.IGNORECASE)
        if one:
            return [int(one.group(1))], True

        return None, False

    def _apply_mutation_scan_tiering(
        self,
        result:     Dict[str, Any],
        user_input: str,
    ) -> None:
        """
        Augment a routed `mutation_scan` request with (a) assignable scope, (b) the
        triage→validate tier (default = fast CamSol+ESM; opt-in Rosetta), and (c)
        the pre-launch surfaces (deep-tier estimate warning / tier-choice prompt).

        Mutates *result* in place; no-op when the request is not a mutation scan.
        """
        if not user_input:
            return
        if "mutation_scan" not in (result.get("tools_needed") or []):
            return

        ti  = result.setdefault("tool_inputs", {})
        inp = ti.get("mutation_scan")
        if not isinstance(inp, dict):
            inp = {}
            ti["mutation_scan"] = inp

        model_id = str(inp.get("model_id") or self._primary_model_id())
        chain    = inp.get("chain") or "A"

        # (a) Scope
        positions, scope_requested = self._parse_scan_scope(user_input, model_id, chain)
        if scope_requested:
            inp["scan_positions"] = positions   # may be [] → _run_mutation_scan errors

        # (b) Tier + options
        # `deep` and the scan scope may arrive PRE-SET in `inp` (the panel tool-launch
        # path builds the result dict deterministically — no text to parse), OR be parsed
        # from user_input (the NL/typed path). Honor the pre-set first so the estimate +
        # confirm-gate below fire identically for both; the text path is unchanged.
        import config as _cfg
        deep      = bool(inp.get("run_rosetta")) or bool(self._TIER_ROSETTA_RE.search(user_input))
        thorough  = bool(self._THOROUGHNESS_RE.search(user_input))
        shortlist = bool(self._SHORTLIST_RE.search(user_input))
        asym      = bool(self._ASYMMETRIC_RE.search(user_input))
        inp["run_rosetta"] = deep
        inp["ddg_basis"]   = "asymmetric" if asym else getattr(_cfg, "ROSETTA_DDG_BASIS", "symmetric")
        _K = int(getattr(_cfg, "ROSETTA_SHORTLIST_K", 15))
        # FULL coverage is the default (max data, the user's bias); shortlist is an
        # explicit opt-in (never silent).  rosetta_shortlist_k=None → full grid.
        inp["rosetta_shortlist_k"] = _K if shortlist else None

        # Estimate inputs (only needed when a deep estimate will be surfaced):
        #   n_pos  — positions to scan (scope, else chain length)
        #   n_mut  — ACTUAL full-grid Rosetta calls = positions × candidates_per_pos
        #   n_res  — FULL pose residues (all chains; drives per-mutation cost)
        if deep or thorough:
            _preset_scope = inp.get("scan_positions")
            if scope_requested and positions:
                n_pos: Optional[int] = len(positions)
            elif _preset_scope:                 # panel-supplied scope (no text to parse)
                n_pos = len(_preset_scope)
            else:
                _seq  = self._fetch_sequence(model_id, chain) or ""
                n_pos = len(_seq) or None
            cands = int(inp.get("candidates_per_pos") or self._DEFAULT_CANDS_PER_POS)
            _scoped = scope_requested or bool(_preset_scope)
            if n_pos:
                n_mut = n_pos * cands
                if not _scoped:                # whole-chain scan is capped
                    n_mut = min(n_mut, self._DEFAULT_MAX_CANDIDATES)
            else:
                n_mut = None
            n_res    = self._pose_residue_count(model_id)
            workers  = self._resolve_deep_workers(n_res)
            n_short  = min(_K, n_mut) if n_mut else None
            full_s   = self._estimate_rosetta_secs(n_mut, n_res, workers)
            full_est = self._format_duration(full_s) if full_s is not None else None
            short_est = self._estimate_rosetta_runtime(n_short, n_res, workers)
            offer_sec = int(getattr(_cfg, "ROSETTA_FULL_GRID_OFFER_SEC", 300))
            _basis_note = "" if not asym else " [asymmetric ddG basis — faster, noisier]"

        # (c) Surfaces
        if deep:
            if shortlist and short_est:
                result.setdefault("warnings", []).append(
                    f"Deep tier — SHORTLIST top {n_short} of {n_mut} candidates by "
                    f"fast score ({workers} worker(s)): approximate runtime "
                    f"{short_est} on a {n_res or '?'}-residue pose{_basis_note}. "
                    "The rest are retained as 'not computed'."
                )
            elif full_est:
                _msg = (
                    f"Deep tier — FULL coverage ({workers} worker(s)): approximate "
                    f"runtime {full_est} for {n_mut} mutation(s) across {n_pos} "
                    f"position(s) on a {n_res or '?'}-residue pose{_basis_note} — "
                    "shown before launch (no mid-run cancel)."
                )
                # Offer the shortlist alternative when the full grid is expensive
                # (data-vs-speed becomes an explicit, estimated choice — never silent).
                if full_s and full_s > offer_sec and short_est and n_mut and n_mut > _K:
                    _msg += (
                        f"  Faster option: say 'shortlist' to validate just the top "
                        f"{_K} by fast score (≈ {short_est}); the rest stay 'not computed'."
                    )
                result.setdefault("warnings", []).append(_msg)
        elif thorough:
            # Thoroughness phrasing does NOT auto-run Rosetta — raise a tier choice.
            _fe = full_est or "several minutes"
            _extra = ""
            if full_s and full_s > offer_sec and short_est and n_mut and n_mut > _K:
                _extra = f", or Deep-shortlist (top {_K}) ≈ {short_est}"
            result["clarification_needed"] = (
                f"Base tier (CamSol + ESM) ≈ 2 s, or Deep tier (+Rosetta ddG, "
                f"{workers} worker(s)) full {_fe} for {n_mut} mutation(s) across "
                f"{n_pos if n_pos else 'the chain'} position(s){_extra} — which? "
                "(base / deep / shortlist)"
            )
            result["_tier_choice"] = True

    # ── Double mutant intent helpers ──────────────────────────────────────────

    @classmethod
    def _detect_double_mutant_intent(cls, text: str) -> bool:
        """Return True if *text* signals a double mutant pair scoring request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._DOUBLE_MUTANT_KEYWORDS)

    # ── Proline intent helpers ────────────────────────────────────────────────

    @classmethod
    def _detect_validate_ddg_intent(cls, text: str) -> bool:
        """Return True if *text* requests the high-accuracy ddG validation tier."""
        lower = (text or "").lower()
        return any(kw in lower for kw in cls._VALIDATE_DDG_KEYWORDS)

    @classmethod
    def _detect_proline_intent(cls, text: str) -> bool:
        """Return True if *text* signals a proline substitution scan request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._PROLINE_KEYWORDS)

    def _rewrite_as_proline(
        self,
        tools_needed: List[str],
        tool_inputs:  Dict[str, Any],
    ) -> tuple:
        """
        Replace 'mutation_scan' with 'proline' in the tool pipeline.

        Builds proline tool_inputs from the mutation_scan inputs so model_id
        and chain are preserved.  Other tools (assembly_analyser, disulfide,
        chimerax) are kept unchanged.

        Returns (new_tools_needed, new_tool_inputs).
        """
        new_tools  = []
        new_inputs = dict(tool_inputs)

        for tool in tools_needed:
            if tool == "mutation_scan":
                new_tools.append("proline")
                ms = tool_inputs.get("mutation_scan", {})
                # Always target the primary (crystal) structure — not a
                # secondary ESMFold prediction that the translator may have
                # guessed because it was the active model in ChimeraX.
                new_inputs["proline"] = {
                    "model_id": self._primary_model_id(),
                    "chain":    ms.get("chain", "A"),
                    "pdb_path": ms.get("pdb_path"),
                    "top_n":    5,
                }
                new_inputs.pop("mutation_scan", None)
            else:
                new_tools.append(tool)

        # If mutation_scan wasn't in the list, still ensure proline is added
        if "proline" not in new_tools:
            new_tools.append("proline")
            new_inputs["proline"] = {
                "model_id": self._primary_model_id(),
                "chain":    "A",
                "top_n":    5,
            }

        return new_tools, new_inputs

    # ── Glycan intent helpers ─────────────────────────────────────────────────

    @classmethod
    def _detect_glycan_intent(cls, text: str) -> bool:
        """Return True if *text* signals an N-glycosylation site scan request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._GLYCAN_KEYWORDS)

    def _rewrite_as_glycan(
        self,
        tools_needed: List[str],
        tool_inputs:  Dict[str, Any],
    ) -> tuple:
        """
        Replace the current tool pipeline with a single 'glycan' step.

        Inherits model_id and chain from any existing tool_inputs so that
        context set by the translator (chain letter, model number) is preserved.
        Initial ChimeraX commands in ``result["commands"]`` are unaffected.

        Returns (new_tools_needed, new_tool_inputs).
        """
        new_inputs = dict(tool_inputs)

        # Inherit model_id / chain from whatever the translator already parsed
        model_id: Optional[str] = None
        chain:    str           = "A"
        for inp in tool_inputs.values():
            if isinstance(inp, dict):
                if inp.get("model_id") and not model_id:
                    model_id = str(inp["model_id"])
                if inp.get("chain"):
                    chain = inp["chain"]

        model_id = model_id or self._first_model_id()

        new_inputs["glycan"] = {
            "model_id": model_id,
            "chain":    chain,
            "top_n":    3,
        }
        # Drop other analysis tool inputs; keep chimerax passthrough untouched
        for drop in ("mutation_scan", "proline", "camsol", "esm"):
            new_inputs.pop(drop, None)

        return ["glycan"], new_inputs

    @classmethod
    def _detect_glycan_positions_intent(cls, text: str) -> bool:
        """
        Return True if *text* signals a projection-aware glycosylation
        position scan (distinct from the classic NXS/T sequon scan).
        """
        lower = text.lower()
        return any(kw in lower for kw in cls._GLYCAN_POSITIONS_KEYWORDS)

    @classmethod
    def _detect_netnglyc_intent(cls, text: str) -> bool:
        """Return True if *text* signals a standalone NetNGlyc OST prediction request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._NETNGLYC_KEYWORDS)

    # ── MPNN+ESMFold intent helpers ───────────────────────────────────────────

    @classmethod
    def _detect_mpnn_esmfold_intent(cls, text: str) -> bool:
        """Return True if *text* signals a ProteinMPNN + ESMFold validation request."""
        lower = text.lower()
        return any(kw in lower for kw in cls._MPNN_ESMFOLD_KEYWORDS)

    def _rewrite_as_mpnn_esmfold(
        self,
        tools_needed: List[str],
        tool_inputs:  Dict[str, Any],
    ) -> tuple:
        """
        Replace ``'proteinmpnn'`` with ``'mpnn_esmfold'`` in the tool pipeline.

        Builds mpnn_esmfold tool_inputs from the proteinmpnn inputs so that
        model_id and chain are preserved.  Other tools are kept unchanged.

        Returns (new_tools_needed, new_tool_inputs).
        """
        new_tools  = []
        new_inputs = dict(tool_inputs)

        for tool in tools_needed:
            if tool == "proteinmpnn":
                new_tools.append("mpnn_esmfold")
                pi = tool_inputs.get("proteinmpnn", {})
                new_inputs["mpnn_esmfold"] = {
                    "model_id": pi.get("model_id") or self._first_model_id(),
                    "chain_id": pi.get("chain_id") or pi.get("chain", "A"),
                }
                new_inputs.pop("proteinmpnn", None)
            elif tool == "esmfold":
                # Redirect standalone ESMFold → mpnn_esmfold when session has
                # MPNN results (route() gate already checked this condition).
                new_tools.append("mpnn_esmfold")
                ei = tool_inputs.get("esmfold", {})
                new_inputs["mpnn_esmfold"] = {
                    "model_id": ei.get("model_id") or self._first_model_id(),
                    "chain_id": ei.get("chain_id") or ei.get("chain", "A"),
                }
                new_inputs.pop("esmfold", None)
            else:
                new_tools.append(tool)

        # If neither proteinmpnn nor esmfold was in the list, still add mpnn_esmfold
        if "mpnn_esmfold" not in new_tools:
            new_tools.append("mpnn_esmfold")
            new_inputs["mpnn_esmfold"] = {
                "model_id": self._first_model_id(),
                "chain_id": "A",
            }

        return new_tools, new_inputs

    # ── Phase 1: Route (no execution) ─────────────────────────────────────────

    def route(
        self,
        translator_result: Dict[str, Any],
        user_input:        str = "",
    ) -> Dict[str, Any]:
        """
        Augment a translator result with tool routing metadata.

        Adds the following keys (safe to call before user confirmation):
          "tools_needed"    — list of tools from translator (default ["chimerax"])
          "tool_steps_info" — list of {tool, icon, description} for each step
          "has_extra_tools" — True if any non-chimerax tools are present
          "_user_input"     — original user text, used by execute() for proline guard

        If *user_input* contains proline intent keywords and 'mutation_scan' is
        in tools_needed, the pipeline is rewritten to use 'proline' instead.
        """
        tools_needed = translator_result.get("tools_needed") or ["chimerax"]
        tool_inputs  = translator_result.get("tool_inputs") or {}

        # ── Intent-override PRECEDENCE lock ────────────────────────────────────
        # The override chain below is ORDERED (documented §2):
        #   validate_design → validate_ddg → colabfold → proline → mpnn_esmfold →
        #   glycan_positions → netnglyc → glycan → salt_bridge → cavity →
        #   double_mutant → mutation_scan (fallback, last).
        # Each FULL-REPLACE override used to guard only on its OWN tool's absence,
        # so a LOWER-precedence override could clobber a HIGHER one when a single
        # prompt legitimately triggered both (e.g. "validate the ddG with
        # AlphaFold" → colabfold stealing validate_ddg). `_claimed` makes the
        # FIRST (highest-precedence) override that fires the deterministic winner:
        # once routing is claimed, no later full-replace override may overwrite it.
        # (The two REWRITE overrides — proline, mpnn_esmfold — also set it, since
        # they sit high in the order; they are naturally inert once a higher
        # full-replace has claimed, because their target tool is no longer present.)
        # A PANEL tool request (engine.handle_tool_request) already chose the tool
        # deterministically → pre-claim so the NL intent overrides can't re-route it on the
        # synthetic label (e.g. "[Workbench] fold V1 … boltz monomer" must run Boltz, not be
        # stolen by the ColabFold intent's "fold"+"monomer" match). The additive mutation-scan
        # tiering runs after this block regardless, so panel Scan launches are unaffected.
        _claimed     = bool(translator_result.get("_explicit_tool"))
        _ba_intent   = False  # bio-assembly generation override (set below)
        _repr_intent = False  # viewer representation override (set below)
        _repr_key    = None
        _color_intent = False  # color op-class override (set below)
        _color_key    = None
        _transparency_intent = False  # transparency op-class override (set below)
        _design_intent = False  # design-intent op-class override (set below)

        # ── Viewer representation intent override (FIRST — highest priority) ────
        # Fires before ALL analysis-tool overrides because representation is
        # orthogonal: "show cartoon" cannot mean "run a mutation scan".
        # detect_category_phrase() uses the broad viewer vocabulary; alias
        # resolution populates intent_key up front; LLM tier runs in execute().
        # Translator-emitted commands are cleared (rendered deterministically).
        # EXCEPTION — bio-assembly phrasing wins over representation: "show/present/display as
        # <oligomer>" / "… biological assembly" means BUILD the assembly, not set a cartoon/surface
        # rep. Without this guard the broad "show" vocabulary claims it first → the rep render fails
        # (the 5HRZ "show as trimeric assembly" → blocked cartoon #1 bug).
        if user_input and not self._detect_bio_assembly_intent(user_input):
            from intent_registry import VIEWER_REGISTRY as _vreg
            if _vreg.detect_category_phrase(user_input):
                # Strip the target so an intervening target ("show the ligand as
                # sticks") doesn't break the contiguous alias match → deterministic
                # intent_key, no flaky LLM tier.
                _repr_key    = _vreg.resolve_alias(
                    self._strip_target_for_alias(user_input))  # None → LLM in execute
                tools_needed = ["representation"]
                tool_inputs  = {
                    "representation": {
                        "_user_input": user_input,
                        "intent_key":  _repr_key,
                        # Preserved for the resolver's DEFER path: a finer target
                        # (residue range / zone / pocket) runs these scoped commands
                        # instead of a regenerated whole-model render.
                        "_translator_commands": list(
                            translator_result.get("commands") or []),
                    }
                }
                _repr_intent = True
                _claimed     = True

        # ── Color op-class intent override ─────────────────────────────────────
        # Orthogonal to analysis tools ("color chain A red" cannot mean a scan).
        # Checked after representation (a phrase can't be both) and guarded by
        # _claimed so it never clobbers a higher-precedence override.  Alias
        # resolution populates intent_key up front; LLM/solid resolution runs in
        # execute().  Translator-emitted commands are cleared (rendered here).
        if user_input and not _claimed:
            from intent_registry import COLOR_REGISTRY as _creg
            if _creg.detect_category_phrase(user_input, "color"):
                _color_key   = _creg.resolve_alias(
                    self._strip_target_for_alias(user_input))  # None → LLM/solid in execute
                tools_needed = ["color"]
                tool_inputs  = {
                    "color": {
                        "_user_input": user_input,
                        "intent_key":  _color_key,
                        # Preserved for the resolver's DEFER path (see representation).
                        "_translator_commands": list(
                            translator_result.get("commands") or []),
                    }
                }
                _color_intent = True
                _claimed      = True

        # ── Transparency op-class override ─────────────────────────────────────
        # Orthogonal to color/representation ("make it 50% transparent" is neither a
        # colour scheme nor a display style). Checked after them (a phrase can't be
        # both) and guarded by _claimed. The level/relative parsing + the all-visible
        # model scope run in _run_transparency; translator commands are cleared.
        if user_input and not _claimed and self._detect_transparency_phrase(user_input):
            tools_needed = ["transparency"]
            tool_inputs  = {"transparency": {"_user_input": user_input}}
            _transparency_intent = True
            _claimed             = True

        # ── Design-intent op-class override (goal → tool-invocation profile) ────
        # SINGLE SOURCE OF TRUTH for a goal-directed redesign. Fires only on the
        # conservative design floor (redesign verb + goal objective) so bare
        # "redesign chain A" never enters and "suggest mutations to improve
        # solubility" (no redesign verb) stays mutation_scan. Claims so the later
        # proteinmpnn/mutation_scan routing cannot also fire (supersede, not
        # double-route). Alias resolved up front; the LLM tier + profile lookup +
        # the over-attraction MISS-handback run in _run_design_goal.
        if user_input and not _claimed:
            from intent_registry import DESIGN_GOAL_REGISTRY as _dreg
            if _dreg.detect_category_phrase(user_input, "design"):
                # Strip an intervening target ("redesign CHAIN A for solubility") so
                # the contiguous alias lands deterministically (else the LLM tier).
                _design_goal_key = _dreg.resolve_alias(
                    self._strip_target_for_alias(user_input))   # None → LLM in execute
                _chain = "A"
                for _inp in list(tool_inputs.values()):
                    if isinstance(_inp, dict) and _inp.get("chain"):
                        _chain = _inp["chain"]; break
                tools_needed = ["design_goal"]
                tool_inputs  = {
                    "design_goal": {
                        "_user_input": user_input,
                        "intent_key":  _design_goal_key,
                        "model_id":    self._primary_model_id(),
                        "chain":       _chain,
                    }
                }
                _design_intent = True
                _claimed       = True

        # ── Validate-design meta-tool override ─────────────────────────────────
        # Checked FIRST: "validate design" appears in _MPNN_ESMFOLD_KEYWORDS too,
        # so this must win and replace tools_needed before that override runs.
        # Distinct from bare `colabfold` (no fold/oligomer/template trigger) and
        # from `validate_ddg` (its keywords are about ddG/stability, not design).
        _vd_intent = bool(
            user_input
            and self._detect_validate_design_intent(user_input)
            and "validate_design" not in tools_needed
        )
        if _vd_intent:
            _vd_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _vd_chain = _inp["chain"]
                    break
            vd_inputs: Dict[str, Any] = {
                "model_id":    self._primary_model_id(),
                "chain":       _vd_chain,
                "_user_input": user_input,
            }
            vd_inputs.update(self._parse_colabfold_options(user_input))  # seq/copies/template/quick
            vd_inputs.update(self._parse_validate_design_options(user_input))
            tools_needed = ["validate_design"]
            tool_inputs  = {"validate_design": vd_inputs}
            _claimed = True

        # ── High-accuracy ddG validation tier override ─────────────────────────
        # Checked FIRST so "validate ddg" / "high-accuracy stability" route to the
        # multi-trajectory validation tier, NOT the fast single-trajectory scan or
        # double-mutant pipeline. Operates on an explicit mutation list (parsed in
        # _run_validate_ddg) or the top candidates from existing scan_results.
        _vddg_intent = bool(user_input and self._detect_validate_ddg_intent(user_input))
        if _vddg_intent and "validate_ddg" not in tools_needed and not _claimed:
            _v_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _v_chain = _inp["chain"]
                    break
            tools_needed = ["validate_ddg"]
            tool_inputs  = {
                "validate_ddg": {
                    "model_id":    self._primary_model_id(),
                    "chain":       _v_chain,
                    "_user_input": user_input,
                }
            }
            _claimed = True

        # ── ColabFold intent override ──────────────────────────────────────────
        # Explicit high-accuracy folding path (AF2 via the WSL2 ColabFold env).
        # Fires only on explicit keywords (colabfold/alphafold, "fold … as a
        # <oligomer>", or "use PDB XXXX as template to fold"), so it neither
        # swallows nor is swallowed by the esmfold / mpnn_esmfold pipeline.
        _cf_intent = bool(
            user_input
            and self._detect_colabfold_intent(user_input)
            and "colabfold" not in tools_needed
            and "validate_design" not in tools_needed  # meta-tool already claimed it
            and not _claimed
        )
        if _cf_intent:
            _cf_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _cf_chain = _inp["chain"]
                    break
            cf_inputs: Dict[str, Any] = {
                "model_id":    self._primary_model_id(),
                "chain":       _cf_chain,
                "_user_input": user_input,
            }
            cf_inputs.update(self._parse_colabfold_options(user_input))
            tools_needed = ["colabfold"]
            tool_inputs  = {"colabfold": cf_inputs}
            _claimed = True

        # ── Biological-assembly generation intent override ────────────────────
        # Fires when user asks to "work as tetramer" / "generate biological
        # assembly" / "build the biological unit" / "apply crystal symmetry" etc.
        # Emits `sym #N assembly M copies true` on the EXISTING loaded model.
        # Does NOT re-open the structure (that was the duplicate-#2 bug).
        _ba_intent = bool(
            user_input
            and self._detect_bio_assembly_intent(user_input)
            and "bio_assembly" not in tools_needed
            and not _claimed
        )
        if _ba_intent:
            # Target the VISIBLE model (what the user is focused on), not the always-#1
            # heuristic — else "show as assembly" with A+B open re-assembles the first-opened
            # A instead of the visible B. Falls back to _primary_model_id when no visible info.
            _ba_model_id   = self._visible_focus_model_id() or self._primary_model_id()
            _ba_assembly_id = self._parse_bio_assembly_id(user_input)
            tools_needed = ["bio_assembly"]
            tool_inputs  = {
                "bio_assembly": {
                    "model_id":    _ba_model_id,
                    "assembly_id": _ba_assembly_id,
                    "_user_input": user_input,
                }
            }
            _claimed = True

        # ── Interface-stabilization intent override ───────────────────────────
        # Fires on "stabilize the interface", "lock with disulfide", "interface
        # contacts", etc.  Uses the assembly model when one has been generated;
        # falls back to the primary AU model otherwise.
        _is_intent = bool(
            user_input
            and self._detect_interface_stabilization_intent(user_input)
            and "interface_stabilization" not in tools_needed
            and not _claimed
        )
        if _is_intent:
            _is_model_id = self._primary_assembly_model_id()
            tools_needed = ["interface_stabilization"]
            tool_inputs  = {
                "interface_stabilization": {
                    "model_id":    _is_model_id,
                    "_user_input": user_input,
                }
            }
            _claimed = True

        # ── Conformer-comparison intent override ──────────────────────────────
        # Fires on explicit phrasing only ("compare conformers", "anchored overlay",
        # "per-residue shift", "morph analysis", etc.) — never on generic phrases.
        _cc_intent = bool(
            user_input
            and self._detect_conformer_comparison_intent(user_input)
            and "conformer_comparison" not in tools_needed
            and not _claimed
        )
        if _cc_intent:
            cc_inputs = self._parse_conformer_comparison_options(user_input, translator_result)
            tools_needed = ["conformer_comparison"]
            tool_inputs  = {"conformer_comparison": cc_inputs}
            _claimed = True

        # ── Proline intent override ────────────────────────────────────────────
        # Check BEFORE building step_info so the icon/description are correct.
        if user_input and self._detect_proline_intent(user_input) and not _claimed:
            if "mutation_scan" in tools_needed:
                tools_needed, tool_inputs = self._rewrite_as_proline(
                    tools_needed, tool_inputs
                )
                _claimed = True

        # ── MPNN+ESMFold intent override ───────────────────────────────────────
        # Always rewrite if 'proteinmpnn' is in the pipeline.
        # Only rewrite 'esmfold' → 'mpnn_esmfold' when the session already holds
        # ProteinMPNN results (prevents "check fold mutation I64E" from hijacking
        # a plain ESMFold foldability request on a structure with no designs).
        if user_input and self._detect_mpnn_esmfold_intent(user_input) and not _claimed:
            _session_has_mpnn = (
                self.session.get_proteinmpnn_result(self._first_model_id()) is not None
            )
            if "proteinmpnn" in tools_needed or (
                _session_has_mpnn and "esmfold" in tools_needed
            ):
                tools_needed, tool_inputs = self._rewrite_as_mpnn_esmfold(
                    tools_needed, tool_inputs
                )
                _claimed = True

        # ── Glycan positions intent override ──────────────────────────────────
        # Must fire BEFORE the general glycan check so that phrases like
        # "glycan candidates" / "domain masking" are not swallowed by the
        # broader glycan keyword set.
        _glycan_positions_intent = bool(
            user_input and self._detect_glycan_positions_intent(user_input)
        )
        if _glycan_positions_intent and "glycan_positions" not in tools_needed and not _claimed:
            _gp_model_id = self._first_model_id()
            _gp_chain    = "A"
            for _inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _gp_chain = _inp["chain"]
                    break
            tools_needed = ["glycan_positions"]
            tool_inputs  = {
                "glycan_positions": {
                    "model_id":    _gp_model_id,
                    "chain":       _gp_chain,
                    "top_n":       20,
                    "_user_input": user_input,
                }
            }
            _claimed = True

        # ── NetNGlyc intent override ───────────────────────────────────────────
        # Fires for explicit OST recognition requests (e.g. "run NetNGlyc on
        # my sequence" / "what is the OST score for position 42?").
        _netnglyc_intent = bool(
            user_input and self._detect_netnglyc_intent(user_input)
        )
        if _netnglyc_intent and "netnglyc" not in tools_needed and not _claimed:
            _ng_model_id = self._first_model_id()
            _ng_chain    = "A"
            for _inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _ng_chain = _inp["chain"]
                    break
            tools_needed = ["netnglyc"]
            tool_inputs  = {
                "netnglyc": {
                    "model_id":    _ng_model_id,
                    "chain":       _ng_chain,
                    "_user_input": user_input,
                }
            }
            _claimed = True

        # ── Glycan intent override ─────────────────────────────────────────
        # Fires when glycan keywords are present and the translator did NOT
        # already emit "glycan" in tools_needed (wrong routing or unclear query).
        # ALSO clears any clarification_needed flag from the translator — this
        # prevents the clarification retry loop in main.py from asking the user a
        # question whose answer would be re-sent to translate(), which crashes with
        # stop_reason='refusal' when the short answer ("chain A") has no prior
        # context the model can work with.
        _glycan_intent = bool(user_input and self._detect_glycan_intent(user_input))
        if (_glycan_intent and "glycan" not in tools_needed
                and not _glycan_positions_intent and not _netnglyc_intent and not _claimed):
            tools_needed, tool_inputs = self._rewrite_as_glycan(
                tools_needed, tool_inputs
            )
            _claimed = True

        # ── Salt bridge intent override ────────────────────────────────────────
        _sb_intent = bool(user_input and any(kw in user_input.lower() for kw in self._SALT_BRIDGE_KEYWORDS))
        if _sb_intent and "salt_bridge" not in tools_needed and not _claimed:
            # Rewrite to salt_bridge tool
            tools_needed = ["salt_bridge"]
            tool_inputs = {"salt_bridge": {"model_id": self._primary_model_id(), "chain": "A"}}
            # Extract chain from any existing tool_input
            for inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(inp, dict) and inp.get("chain"):
                    tool_inputs["salt_bridge"]["chain"] = inp["chain"]
                    break
            _claimed = True

        # ── Cavity intent override ──────────────────────────────────────────────
        _cav_intent = bool(user_input and self._detect_cavity_intent(user_input))
        if _cav_intent and "cavity" not in tools_needed and not _claimed:
            tools_needed = ["cavity"]
            tool_inputs = {"cavity": {"model_id": self._primary_model_id(), "chain": "A"}}
            for inp in list(translator_result.get("tool_inputs", {}).values()):
                if isinstance(inp, dict) and inp.get("chain"):
                    tool_inputs["cavity"]["chain"] = inp["chain"]
                    break
            _claimed = True

        # ── Cavity assembly mode detection ──────────────────────────────────────
        # When assembly keywords are present, find_cavities() uses chains=None
        # (all chains), enabling interface cavity detection for dimers/oligomers.
        if "cavity" in tool_inputs:
            _cav_assembly = bool(
                user_input and any(
                    kw in user_input.lower()
                    for kw in self._CAVITY_ASSEMBLY_KEYWORDS
                )
            )
            tool_inputs["cavity"]["assembly_mode"] = _cav_assembly

        # ── Double mutant intent override ─────────────────────────────────────
        # Fires when double mutant keywords are detected AND the tool is not
        # already set.  Runs AFTER other overrides so proline/glycan/etc. take
        # priority when their more-specific keywords also appear.
        _dm_intent = bool(
            user_input
            and self._detect_double_mutant_intent(user_input)
            and "double_mutant" not in tools_needed
            and not _claimed
        )
        if _dm_intent:
            _dm_chain = "A"
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict) and _inp.get("chain"):
                    _dm_chain = _inp["chain"]
                    break
            tools_needed = ["double_mutant"]
            tool_inputs  = {
                "double_mutant": {
                    "model_id":    self._primary_model_id(),
                    "chain":       _dm_chain,
                    "_user_input": user_input,
                }
            }
            _claimed = True

        # ── Mutation scan intent fallback ──────────────────────────────────────
        # Fires when user clearly wants a mutation scan but the translator returned
        # a different tool (e.g. cavity, chimerax).  Handles singular "mutation"
        # as well as "improve solubility" without the mutation keyword.
        # Runs LAST: all more-specific overrides (proline, glycan, double_mutant)
        # have already fired and take priority.
        # Blocks on ACTUAL user keywords (not translator output) so that cavity
        # and salt-bridge are only respected when the user's words match those tools.
        _lower_ui_ms = user_input.lower() if user_input else ""
        _ms_intent = bool(
            user_input
            and not _claimed                       # a higher-precedence override already won
            and self._detect_mutation_scan_intent(user_input)
            and "mutation_scan" not in tools_needed
            # DISTRACTOR non-capture: only UPGRADE a generic/cavity mis-route to a
            # mutation scan. NEVER clobber a specialized tool the translator
            # deliberately chose — e.g. "CamSol *solubility scan*" legitimately
            # routes camsol, and a *proteinmpnn* redesign that mentions "improve
            # solubility"/"reduce aggregation" must stay proteinmpnn (the
            # collisions the new eval corpus forbids). `cavity` stays an allowed
            # upgrade target because it is itself a frequent mis-route for a
            # "suggest mutation to improve solubility" request.
            and all(t in ("chimerax", "cavity") for t in tools_needed)
            and not self._detect_proline_intent(user_input)
            and not self._detect_glycan_intent(user_input)
            and not self._detect_double_mutant_intent(user_input)
            # Only skip if the user's words genuinely match cavity / salt-bridge tools.
            and not self._detect_cavity_intent(user_input)
            and not any(kw in _lower_ui_ms for kw in self._SALT_BRIDGE_KEYWORDS)
        )
        if _ms_intent:
            _ms_chain    = "A"
            _ms_model_id = self._primary_model_id()
            for _inp in list(tool_inputs.values()):
                if isinstance(_inp, dict):
                    if _inp.get("chain"):
                        _ms_chain = _inp["chain"]
                    if _inp.get("model_id"):
                        _ms_model_id = str(_inp["model_id"])
            tools_needed = ["mutation_scan"]
            tool_inputs  = {
                "mutation_scan": {
                    "model_id": _ms_model_id,
                    "chain":    _ms_chain,
                    "focus":    "solubility",
                }
            }

        result = dict(translator_result)
        result["tools_needed"] = tools_needed
        result["tool_inputs"]  = tool_inputs
        result["_user_input"]  = user_input   # passed to execute() for guard

        # Suppress translator clarification when we've resolved the intent ourselves
        if _glycan_intent or _glycan_positions_intent or _netnglyc_intent:
            result["clarification_needed"] = None

        # Representation: clear translator-emitted commands — rendered deterministically.
        if _repr_intent:
            result["commands"]     = []
            result["explanations"] = []

        # Color: clear translator-emitted commands — rendered deterministically
        # (scoped) in _run_color so chain colors never bleed onto ligand/solvent.
        if _color_intent:
            result["commands"]     = []
            result["explanations"] = []

        # Transparency: clear translator-emitted commands — rendered deterministically
        # (absolute level from the tracked state) in _run_transparency.
        if _transparency_intent:
            result["commands"]     = []
            result["explanations"] = []

        # Design-intent: the op-class owns the whole redesign; drop any translator
        # commands so nothing runs alongside the profile-driven ProteinMPNN.
        if _design_intent:
            result["commands"]     = []
            result["explanations"] = []

        # Bio-assembly: clear any translator-emitted commands (e.g. a re-open) —
        # the sym command is issued from within _run_bio_assembly, not from the
        # commands list; a spurious re-open would duplicate the AU model.
        if _ba_intent:
            result["commands"]     = []
            result["explanations"] = []

        # Mutation-scan tiering (Priority 1): assignable scope + triage→validate
        # (default fast CamSol+ESM, opt-in Rosetta) + pre-launch estimate / tier
        # choice.  Runs for any routed mutation_scan (translator- or fallback-routed).
        self._apply_mutation_scan_tiering(result, user_input)

        # ── Bug 4c: suppress translator commands that duplicate a fold/design tool ──
        # When colabfold / validate_design / esmfold / mpnn_esmfold is dispatched,
        # the tool opens, folds, and visualises the result itself.  Any `open` or
        # `matchmaker` commands the translator emitted in parallel are spurious —
        # they desync model IDs (the #2-vs-#3 bug from the live fold-overlay session).
        _FOLD_TOOLS = frozenset({"colabfold", "validate_design", "esmfold", "mpnn_esmfold", "boltz"})
        if any(t in _FOLD_TOOLS for t in tools_needed):
            _raw_cmds = result.get("commands") or []
            _raw_exps = result.get("explanations") or []
            _suppress_re = re.compile(r"^\s*(open|matchmaker)\s", re.IGNORECASE)
            _kept = [
                (_raw_cmds[i], _raw_exps[i] if i < len(_raw_exps) else "")
                for i in range(len(_raw_cmds))
                if not _suppress_re.match(_raw_cmds[i])
            ]
            _suppressed = [c for c in _raw_cmds if _suppress_re.match(c)]
            if _suppressed:
                result["commands"]     = [p[0] for p in _kept]
                result["explanations"] = [p[1] for p in _kept]
                print(
                    f"  [fold-guard] suppressed {len(_suppressed)} open/matchmaker "
                    f"command(s) alongside tool dispatch: {_suppressed}",
                    flush=True,
                )

        step_info: List[Dict[str, Any]] = []
        for tool in tools_needed:
            icon = self._TOOL_ICONS.get(tool, "⚙️")
            step_info.append({
                "tool":        tool,
                "icon":        icon,
                "description": self._step_description(tool, tool_inputs, result),
            })

        result["tool_steps_info"] = step_info
        result["has_extra_tools"] = any(t != "chimerax" for t in tools_needed)
        return result

    def _step_description(
        self,
        tool:        str,
        tool_inputs: Dict[str, Any],
        result:      Dict[str, Any],
    ) -> str:
        if tool == "chimerax":
            n = len(result.get("commands", []))
            return f"Execute {n} ChimeraX command(s)"
        if tool == "representation":
            inp = tool_inputs.get("representation", {})
            key = inp.get("intent_key")
            if key:
                from intent_registry import VIEWER_REGISTRY as _vreg
                defn = _vreg.get_defn(key)
                desc = defn.description if defn else key
                return f"Viewer representation: {desc} (deterministic render)"
            return "Viewer representation (resolving intent…)"
        if tool == "color":
            inp = tool_inputs.get("color", {})
            key = inp.get("intent_key")
            if key:
                from intent_registry import COLOR_REGISTRY as _creg
                defn = _creg.get_defn(key)
                desc = defn.description if defn else key
                return f"Color: {desc} (deterministic render)"
            return "Color (resolving intent…)"
        if tool == "transparency":
            return "Transparency (deterministic render)"
        if tool == "design_goal":
            inp = tool_inputs.get("design_goal", {})
            key = inp.get("intent_key")
            if key:
                from intent_registry import DESIGN_PROFILES
                prof = DESIGN_PROFILES.get(key)
                return (f"Design goal: {prof.description}" if prof
                        else f"Design goal: {key}")
            return "Design goal (resolving goal…)"
        if tool == "bio_assembly":
            inp   = tool_inputs.get("bio_assembly", {})
            mid   = inp.get("model_id") or self._primary_model_id()
            asmid = inp.get("assembly_id", 1)
            return f"Generate biological assembly {asmid} from AU model #{mid} (sym)"
        if tool == "interface_stabilization":
            inp = tool_inputs.get("interface_stabilization", {})
            mid = inp.get("model_id") or self._primary_assembly_model_id()
            return (
                f"Interface stabilization — detect contacts, buried area, "
                f"inter-chain disulfide scan on #{mid}"
            )
        if tool == "camsol":
            inp   = tool_inputs.get("camsol", {})
            chain = inp.get("chain", "all chains")
            mid   = inp.get("model_id") or self._first_model_id()
            return f"CamSol solubility analysis — #{mid} chain {chain}"
        if tool == "esm":
            inp = tool_inputs.get("esm", {})
            mid = inp.get("model_id") or self._first_model_id()
            return f"ESM-2 evolutionary conservation — #{mid}"
        if tool == "esmfold":
            inp = tool_inputs.get("esmfold", {})
            mid = inp.get("model_id") or self._first_model_id()
            return f"ESMFold foldability prediction — #{mid} (ESM Atlas API)"
        if tool == "boltz":
            inp    = tool_inputs.get("boltz", {})
            nch    = len(inp.get("chains") or []) or 1
            shape  = "monomer" if nch <= 1 else f"{nch}-chain assembly"
            return f"Boltz-2 fold — {shape} (LOCAL-ONLY, seed-pinned)"
        if tool == "colabfold":
            inp     = tool_inputs.get("colabfold", {})
            copies  = inp.get("copies", 1)
            shape   = "monomer" if copies <= 1 else f"{copies}-copy homo-oligomer"
            tmpl    = " (templated)" if inp.get("template") else ""
            return (f"ColabFold AF2 structure prediction — {shape}{tmpl} "
                    f"(REMOTE MSA — LEAVES LOCAL-ONLY, sequence sent to the ColabFold MSA server)")
        if tool == "align_folds":
            return ("Compare two folds — US-align TM/RMSD + per-residue deviation "
                    "(e.g. local Boltz vs MSA-informed ColabFold; the asymmetry is stated)")
        if tool == "validate_design":
            return ("Validate design — ColabFold confidence + matchmaker RMSD + "
                    "Rosetta folding-energy sanity (evidence-rich report)")
        if tool == "conformer_comparison":
            inp = tool_inputs.get("conformer_comparison", {})
            a   = inp.get("model_id_a", "?")
            b   = inp.get("model_id_b", "?")
            anc = inp.get("anchor", "auto")
            return (
                f"Conformer comparison #{a}↔#{b} — anchor-restricted Kabsch "
                f"(anchor: {anc}) + per-residue Cα shift map (blue=rigid, red=mobile)"
            )
        if tool == "proteinmpnn":
            return "ProteinMPNN fixed-backbone sequence redesign"
        if tool == "mpnn_esmfold":
            inp   = tool_inputs.get("mpnn_esmfold", {})
            mid   = inp.get("model_id") or self._first_model_id()
            top_n = inp.get("top_n", 3)
            return (
                f"MPNN + ESMFold validation — #{mid} "
                f"top {top_n} design(s) (pLDDT confidence)"
            )
        if tool == "rfdiffusion":
            inp  = tool_inputs.get("rfdiffusion", {})
            mode = inp.get("mode", "binder")
            return f"RFdiffusion de novo backbone generation (mode: {mode})"
        if tool == "rosetta":
            inp  = tool_inputs.get("rosetta", {})
            mid  = inp.get("model_id") or self._first_model_id()
            muts = inp.get("mutations", [])
            return (
                f"Rosetta ddG — #{mid}, "
                f"{len(muts)} mutation(s)"
                if muts else f"Rosetta ddG — #{mid}"
            )
        if tool == "mutation_scan":
            inp   = tool_inputs.get("mutation_scan", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            focus = inp.get("focus", "solubility")
            mode  = inp.get("analysis_mode", "monomer")
            return f"CamSol + ESM + Rosetta scan — #{mid} chain {chain} [{focus}] [{mode} mode]"
        if tool == "assembly_analyser":
            inp  = tool_inputs.get("assembly_analyser", {})
            mid  = inp.get("model_id") or self._first_model_id()
            mode = inp.get("mode", "multimer")
            ch   = inp.get("chain_id", "")
            return (
                f"Assembly analysis — #{mid} [{mode} mode]"
                + (f", chain {ch}" if ch else "")
            )
        if tool == "disulfide":
            inp  = tool_inputs.get("disulfide", {})
            mid  = inp.get("model_id") or self._first_model_id()
            ca   = inp.get("chain_a", "A")
            cb   = inp.get("chain_b", "B")
            return (
                f"Disulfide candidate prediction — #{mid} chains {ca}/{cb} "
                "(geometry + ESM + DynaMut2)"
            )
        if tool == "disulfide_discovery":
            n = (tool_inputs.get("disulfide_discovery", {}) or {}).get("n_seeds")
            return ("Disulfide discovery — assess existing Cys pairs by multi-fold bonding "
                    "frequency" + (f" ({n} seeds)" if n else "") + " (unconstrained)")
        if tool == "disulfide_geometry":
            return "Disulfide geometry — measure existing Cys-pair geometry (this fold)"
        if tool == "disulfide_interface_scan":
            return "Disulfide interface scan — find NOVEL inter-chain sites at the interface (this fold)"
        if tool == "disulfide_scan":
            return ("Disulfide engineering scan — find NOVEL installable sites (backbone, "
                    "geometric compatibility only)")
        if tool == "disulfide_ddg_estimate":
            return ("Disulfide ΔΔG estimate (legacy) — energetic read for ONE flagged interface "
                    "pair (uncalibrated; ranking/sign only)")
        if tool == "proline_scan":
            return ("Proline-stabilization scan — rank X→Pro sites by backbone φ/ψ proline-"
                    "compatibility + a backbone-H-bond-donor penalty (this structure)")
        if tool == "proline_ddg_estimate":
            return ("Proline ΔΔG estimate (legacy) — energetic read for ONE flagged X→Pro site "
                    "(uncalibrated; ranking/sign only)")
        if tool == "cavity_scan":
            return ("Cavity-filling scan — detect internal voids + rank small→larger hydrophobic "
                    "fills that pack them clash-free (this structure)")
        if tool == "cavity_ddg_estimate":
            return ("Cavity-fill ΔΔG estimate (legacy) — energetic read for ONE flagged fill "
                    "(uncalibrated; ranking/sign only)")
        if tool == "saltbridge_scan":
            return ("Salt-bridge scan — assess existing Asp/Glu↔Arg/Lys pairs + rank NOVEL "
                    "complementary charge-pair sites (geometry × burial; this structure)")
        if tool == "saltbridge_ddg_estimate":
            return ("Salt-bridge ΔΔG estimate (legacy) — energetic read for ONE flagged "
                    "charge-pair (two positions; uncalibrated; ranking/sign only)")
        if tool == "proline":
            inp   = tool_inputs.get("proline", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"Proline substitution scan — #{mid} chain {chain} "
                "(backbone φ/ψ + ESM tolerance)"
            )
        if tool == "glycan":
            inp   = tool_inputs.get("glycan", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"N-glycosylation site scan — #{mid} chain {chain} "
                "(NXS/T sequons, SASA + SS + ESM)"
            )
        if tool == "glycan_positions":
            inp   = tool_inputs.get("glycan_positions", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"Projection-aware glycosylation position scan — #{mid} chain {chain} "
                "(all residues, SASA + projection + ESM)"
            )
        if tool == "netnglyc":
            inp   = tool_inputs.get("netnglyc", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return (
                f"NetNGlyc OST recognition prediction — #{mid} chain {chain} "
                "(DTU Health Tech API)"
            )
        if tool == "salt_bridge":
            inp   = tool_inputs.get("salt_bridge", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return f"Salt bridge analysis -- #{mid} chain {chain}"
        if tool == "cavity":
            inp   = tool_inputs.get("cavity", {})
            mid   = inp.get("model_id") or self._first_model_id()
            chain = inp.get("chain", "A")
            return f"Cavity detection and filling -- #{mid} chain {chain}"
        if tool == "double_mutant":
            inp  = tool_inputs.get("double_mutant", {})
            mid  = inp.get("model_id") or self._first_model_id()
            ui   = inp.get("_user_input", "")
            mode = "epitope" if any(kw in ui.lower() for kw in ("epitope", "binding", "preserve")) else "stability"
            return (
                f"Double mutant pair scoring — #{mid} [{mode} mode] "
                "(DynaMut2 prediction_mm + epistasis)"
            )
        if tool == "validate_ddg":
            inp = tool_inputs.get("validate_ddg", {})
            mid = inp.get("model_id") or self._first_model_id()
            import config as _cfg
            n   = getattr(_cfg, "ROSETTA_VALIDATION_TRAJECTORIES", 5)
            cyc = getattr(_cfg, "ROSETTA_VALIDATION_CYCLES", 8)
            return (
                f"High-accuracy ddG validation — #{mid} "
                f"({n} trajectories x {cyc} cycles, median + confidence)"
            )
        if tool == "variant_deviation":
            inp    = tool_inputs.get("variant_deviation", {})
            engine = inp.get("engine", "esmfold")
            target = inp.get("target", "monomer")
            return (f"Variant-vs-WT Cα deviation ({engine}:{target}) — folds the WT "
                    "reference if absent, then per-residue floor-gated deviation")
        if tool == "structural_align":
            inp = tool_inputs.get("structural_align", {})
            ref = inp.get("ref_label") or inp.get("reference_pdb_id") or "reference"
            return (f"Structural alignment vs {ref} — US-align sequence-independent "
                    "superposition (TM-score + RMSD), LOCAL-ONLY")
        return f"Unknown tool: {tool}"

    # ── Phase 2: Execute (non-chimerax tools) ─────────────────────────────────

    def execute(
        self,
        routed_result:   Dict[str, Any],
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run all non-chimerax tools in the pipeline.

        Adds to the returned dict:
          "tool_step_results"    — list of ToolStepResult.to_dict() per non-cx step
          "all_viz_commands"     — ChimeraX commands from all tools combined
          "all_viz_explanations" — corresponding human-readable explanations
          "tool_summaries"       — {tool_name: summary_string}
          "pipeline_success"     — True if no tool step failed
          "pipeline_error"       — first error message, or None
        """
        tools_needed = routed_result.get("tools_needed") or ["chimerax"]
        # Expose the progress callback to long multi-step tools (e.g. Mode-A discovery's
        # N-seed fold loop) so a long-but-healthy run advances VISIBLY (seed k/N), instead
        # of one "Running…" line then hours of silence — the done-vs-stuck legibility half.
        self._status_callback = status_callback
        # Case-insensitive tool_inputs lookup (mirrors the case-insensitive
        # dispatch) so "CamSol"-keyed inputs still reach the camsol bridge.
        tool_inputs  = {str(k).strip().lower(): v
                        for k, v in (routed_result.get("tool_inputs") or {}).items()}
        # user_input stored by route() for the proline guard
        user_input   = routed_result.get("_user_input", "")

        step_results:   List[Dict[str, Any]] = []
        all_viz_cmds:   List[str] = []
        all_viz_exps:   List[str] = []
        tool_summaries: Dict[str, str] = {}
        pipeline_error: Optional[str] = None

        for _raw_tool in tools_needed:
            tool = str(_raw_tool).strip().lower()   # case-insensitive routing
            if tool == "chimerax":
                # ChimeraX execution is handled by main.py; skip here
                step_results.append({
                    "tool":    "chimerax",
                    "skipped": True,
                    "note":    "executed by main.py",
                })
                continue

            icon = self._TOOL_ICONS.get(tool, "⚙️")
            if status_callback:
                status_callback(f"{icon} Running {tool}…")

            t0   = time.perf_counter()
            step = self._dispatch_tool(tool, tool_inputs.get(tool) or {}, user_input=user_input)
            step.elapsed_ms = (time.perf_counter() - t0) * 1000

            step_results.append(step.to_dict())
            tool_summaries[tool] = step.summary

            if step.success:
                # Cache results in session state for later use / display
                mid = (tool_inputs.get(tool) or {}).get("model_id") or self._first_model_id()
                self.session.add_tool_result(tool, mid, step.data)

                if step.viz_commands:
                    all_viz_cmds.extend(step.viz_commands)
                    all_viz_exps.extend(step.viz_explanations)
            else:
                pipeline_error = pipeline_error or step.error

        result = dict(routed_result)
        result["tool_step_results"]    = step_results
        result["all_viz_commands"]     = all_viz_cmds
        result["all_viz_explanations"] = all_viz_exps
        result["tool_summaries"]       = tool_summaries
        result["pipeline_success"]     = pipeline_error is None
        result["pipeline_error"]       = pipeline_error
        return result

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def _dispatch_tool(
        self,
        tool:       str,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Route to the correct bridge; return ToolStepResult.

        Secondary guards:
          - Proline: if tool is 'mutation_scan' but user_input contains proline
            keywords, redirect to _run_proline() (backbone φ-angle scanner).
          - Glycan: if tool is 'mutation_scan' but user_input contains glycan
            keywords, redirect to _run_glycan() (N-glycosylation site scan).
        These guards fire if route() was not called with user_input.
        """
        # Case-INSENSITIVE tool resolution: the registry literals below are
        # lowercase, so a local model emitting "CamSol"/"ESM" still routes. The
        # benchmark scorer matches case-insensitively too (kept in lockstep).
        tool = (tool or "").strip().lower()

        # VRAM invariant: before any GPU-heavy bridge, free the local LLM
        # translator from VRAM (no-op under Claude / when nothing is loaded).
        # The explicit guard — not the idle timer — is the contract that prevents
        # a mid-run OOM from the translator contending with a fold.
        if tool in _GPU_BRIDGE_TOOLS:
            try:
                from translator import ensure_translator_unloaded
                ensure_translator_unloaded()
            except Exception:
                pass

        try:
            if tool == "camsol":
                return self._run_camsol(inputs)
            if tool == "esm":
                return self._run_esm(inputs)
            if tool == "esmfold":
                # ── MPNN+ESMFold guard (secondary safety net) ────────────────
                # If the user's request contains MPNN/ESMFold validation keywords
                # *or* general context keywords (design, sequence, top, …) and
                # the session already holds ProteinMPNN results, redirect to the
                # combined pipeline instead of the bare ESMFold tool.
                if user_input:
                    _lower_ui         = user_input.lower()
                    _has_mpnn_kw      = self._detect_mpnn_esmfold_intent(user_input)
                    _has_ctx_kw       = any(kw in _lower_ui for kw in self._MPNN_CONTEXT_KEYWORDS)
                    _session_has_mpnn = (
                        self.session.get_proteinmpnn_result(self._first_model_id()) is not None
                    )
                    if (_has_mpnn_kw or _has_ctx_kw) and _session_has_mpnn:
                        return self._run_mpnn_esmfold({
                            "model_id": inputs.get("model_id") or self._first_model_id(),
                            "chain_id": inputs.get("chain_id") or inputs.get("chain", "A"),
                        })
                return self._run_esmfold(inputs)
            if tool == "boltz":
                return self._run_boltz(inputs)
            if tool == "proteinmpnn":
                return self._run_proteinmpnn(inputs)
            if tool == "mpnn_esmfold":
                return self._run_mpnn_esmfold(inputs)
            if tool == "rfdiffusion":
                return self._run_rfdiffusion(inputs)
            if tool == "rosetta":
                return self._run_rosetta(inputs)
            if tool == "mutation_scan":
                # ── Double mutant guard (highest-priority secondary net) ──────
                # Fires when user_input contains "double" or "combine" regardless
                # of the full keyword list — covers "double mutations", "combine
                # these mutations", etc. that route() may not have intercepted.
                _lower_ui = user_input.lower() if user_input else ""
                if _lower_ui and ("double" in _lower_ui or "combine" in _lower_ui):
                    dm_inputs = {
                        "model_id":    self._primary_model_id(),
                        "chain":       inputs.get("chain", "A"),
                        "_user_input": user_input,
                    }
                    return self._run_double_mutant(dm_inputs)
                # ── Proline guard (secondary safety net) ─────────────────────
                # If the user asked about proline and route() didn't catch it
                # (e.g. user_input was not passed to route()), redirect here.
                if user_input and self._detect_proline_intent(user_input):
                    proline_inputs = {
                        "model_id": self._primary_model_id(),
                        "chain":    inputs.get("chain", "A"),
                        "pdb_path": inputs.get("pdb_path"),
                        "top_n":    5,
                    }
                    return self._run_proline(proline_inputs)
                # ── Glycan guard (secondary safety net) ──────────────────────
                # If the user asked about glycans and route() didn't catch it,
                # redirect here.
                if user_input and self._detect_glycan_intent(user_input):
                    glycan_inputs = {
                        "model_id": inputs.get("model_id") or self._first_model_id(),
                        "chain":    inputs.get("chain", "A"),
                        "top_n":    3,
                    }
                    return self._run_glycan(glycan_inputs)
                return self._run_mutation_scan(inputs, user_input=user_input)
            if tool == "double_mutant":
                return self._run_double_mutant(inputs, user_input=user_input)
            if tool == "validate_ddg":
                return self._run_validate_ddg(inputs, user_input=user_input)
            if tool == "colabfold":
                return self._run_colabfold(inputs, user_input=user_input)
            if tool == "validate_design":
                return self._run_validate_design(inputs, user_input=user_input)
            if tool == "conformer_comparison":
                return self._run_conformer_comparison(inputs, user_input=user_input)
            if tool == "variant_deviation":
                return self._run_variant_deviation(inputs)
            if tool == "structural_align":
                return self._run_structural_align(inputs)
            if tool == "align_folds":
                return self._run_align_folds(inputs)
            if tool == "template_assist":
                return self._run_template_assist(inputs)
            if tool == "representation":
                return self._run_representation(inputs, user_input=user_input)
            if tool == "color":
                return self._run_color(inputs, user_input=user_input)
            if tool == "transparency":
                return self._run_transparency(inputs, user_input=user_input)
            if tool == "design_goal":
                return self._run_design_goal(inputs, user_input=user_input)
            if tool == "bio_assembly":
                return self._run_bio_assembly(inputs, user_input=user_input)
            if tool == "interface_stabilization":
                return self._run_interface_stabilization(inputs, user_input=user_input)
            if tool == "assembly_analyser":
                return self._run_assembly_analyser(inputs)
            if tool == "disulfide":
                return self._run_disulfide(inputs)
            if tool == "disulfide_discovery":
                return self._run_disulfide_discovery(inputs)
            if tool == "disulfide_geometry":
                return self._run_disulfide_geometry(inputs)
            if tool == "disulfide_scan":
                return self._run_disulfide_scan(inputs)
            if tool == "disulfide_interface_scan":
                return self._run_disulfide_interface_scan(inputs)
            if tool == "disulfide_ddg_estimate":
                return self._run_disulfide_ddg_estimate(inputs)
            if tool == "proline_scan":
                return self._run_proline_scan(inputs)
            if tool == "proline_ddg_estimate":
                return self._run_proline_ddg_estimate(inputs)
            if tool == "cavity_scan":
                return self._run_cavity_scan(inputs)
            if tool == "cavity_ddg_estimate":
                return self._run_cavity_ddg_estimate(inputs)
            if tool == "saltbridge_scan":
                return self._run_saltbridge_scan(inputs)
            if tool == "saltbridge_ddg_estimate":
                return self._run_saltbridge_ddg_estimate(inputs)
            if tool == "proline":
                return self._run_proline(inputs)
            if tool == "glycan":
                return self._run_glycan(inputs)
            if tool == "glycan_positions":
                return self._run_glycan_positions(inputs)
            if tool == "netnglyc":
                return self._run_netnglyc(inputs)
            if tool == "salt_bridge":
                return self._run_salt_bridge(inputs)
            if tool == "cavity":
                return self._run_cavity(inputs)
            return ToolStepResult(
                tool=tool, success=False,
                error=(
                    f"Unknown tool '{tool}'. "
                    "Available: chimerax, camsol, esm, esmfold, proteinmpnn, "
                    "mpnn_esmfold, rfdiffusion, rosetta, mutation_scan, "
                    "assembly_analyser, bio_assembly, disulfide, proline, glycan, "
                    "glycan_positions, netnglyc, salt_bridge, cavity, "
                    "double_mutant, validate_ddg, colabfold, validate_design."
                ),
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool=tool, success=False,
                error=f"{tool} raised an unexpected error: {exc}",
            )

    def _run_disulfide(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run interchain disulfide bond candidate prediction."""
        bridge   = self._get_disulfide_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain_a  = inputs.get("chain_a", "A")
        chain_b  = inputs.get("chain_b", "B")

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="disulfide", success=False,
                error=(
                    "Disulfide prediction requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        # Pull any interface residues the assembly analyser already found
        binding_site: Optional[List[int]] = None
        interface_data = self.session.get_interface_residues(model_id)
        if interface_data:
            # Collect binding-site residues for chain_a
            binding_site = self.session.get_protected_residues_for_chain(
                model_id, chain_a
            ) or None

        import time as _time
        t0 = _time.perf_counter()

        result = bridge.analyze(
            pdb_path              = pdb_path,
            chain_a               = chain_a,
            chain_b               = chain_b,
            session               = self.session,
            model_id              = model_id,
            binding_site_residues = binding_site,
        )
        result.elapsed_ms = (_time.perf_counter() - t0) * 1000

        if result.success and result.data.get("candidates"):
            self.session.set_disulfide_candidates(
                model_id, chain_a, chain_b, result.data["candidates"]
            )

        return result

    def _run_proline(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run proline substitution scan on a chain."""
        import time as _time
        bridge   = self._get_proline_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 5))

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="proline", success=False,
                error=(
                    "Proline scan requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        # Pull ESM tolerance scores from session if already computed
        esm_scores: Optional[Dict[int, float]] = None
        esm_data = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_data, dict):
            raw = esm_data.get("per_residue_tolerance") or esm_data.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        # Pull interface residues from session if already computed
        iface_set: Optional[set] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                iface_set = set(protected)

        # Optional DynaMut2 validation
        dynamut2_bridge: Optional[Any] = None
        use_dynamut2 = inputs.get("use_dynamut2", False)
        if use_dynamut2:
            try:
                dynamut2_bridge = self._get_rosetta_bridge()
            except Exception:
                pass   # proceed without DynaMut2 if unavailable

        # Pull user-declared active-site residues from session
        # (None → trigger SASA auto-detection in proline_bridge)
        session_fr = self.session.get_functional_residues()
        functional_residues: Optional[set] = session_fr if session_fr else None

        t0 = _time.perf_counter()
        try:
            result = bridge.full_proline_scan(
                pdb_path             = pdb_path,
                chain                = chain,
                interface_residues   = iface_set,
                esm_scores           = esm_scores,
                top_n                = top_n,
                dynamut2_bridge      = dynamut2_bridge,
                functional_residues  = functional_residues,
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool="proline", success=False,
                error=f"Proline scan failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        candidates = result.get("candidates", [])
        if result.get("count", 0) == 0:
            return ToolStepResult(
                tool       = "proline",
                success    = True,
                data       = result,
                summary    = f"Proline scan chain {chain}: no candidates found.",
                elapsed_ms = elapsed_ms,
            )

        # Store in session
        self.session.set_proline_results(model_id, chain, result)

        # Visualization
        viz_cmds, viz_exps = bridge.generate_chimerax_commands(
            candidates[:top_n], model_id=model_id, chain=chain
        )

        summary = bridge._generate_summary(result)

        return ToolStepResult(
            tool             = "proline",
            success          = True,
            data             = result,
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_glycan(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run N-glycosylation site prediction on a chain."""
        import time as _time
        bridge   = self._get_glycan_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 3))
        min_score= float(inputs.get("min_score", 0.05))

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)
        if not sequence:
            return ToolStepResult(
                tool="glycan", success=False,
                error=(
                    "Glycan scan requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)

        # Pull ESM tolerance scores from session if available
        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        # Pull interface residues from session if available
        interface_residues: Optional[List[int]] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = protected

        t0 = _time.perf_counter()
        try:
            result = bridge.analyze(
                pdb_path           = pdb_path,
                chain              = chain,
                sequence           = sequence,
                model_id           = model_id,
                session            = self.session,
                interface_residues = interface_residues,
                esm_scores         = esm_scores,
                min_score          = min_score,
                top_n              = top_n,
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool="glycan", success=False,
                error=f"Glycan scan failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="glycan", success=False,
                error=result.get("error", "Glycan scan failed"),
                elapsed_ms=elapsed_ms,
            )

        cx_cmds = result.get("chimerax_commands", [])
        cx_exps = result.get("chimerax_explanations", [])

        if not cx_cmds:
            # Defensive: if analyze() returned nothing, log clearly so it's visible
            n_native = len(result.get("native_sequons", []))
            n_eng    = len(result.get("engineered_candidates", []))
            print(
                f"  [glycan] WARNING: no ChimeraX commands generated "
                f"(native={n_native}, engineered={n_eng})"
            )

        return ToolStepResult(
            tool             = "glycan",
            success          = True,
            data             = {k: v for k, v in result.items()
                                if k not in ("chimerax_commands", "chimerax_explanations", "summary")},
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = result.get("summary", "Glycan scan complete."),
            elapsed_ms       = elapsed_ms,
        )

    def _run_glycan_positions(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """
        Run projection-aware glycosylation position scan (all residues).

        Unlike _run_glycan() which detects existing NXS/T sequons, this method
        scans every surface-exposed, outward-projecting residue and ranks them
        as engineering targets for de-novo glycan attachment.
        """
        import time as _time
        bridge   = self._get_glycan_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 20))

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)
        if not sequence:
            return ToolStepResult(
                tool="glycan_positions", success=False,
                error=(
                    "Glycan position scan requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)

        # Pull ESM tolerance scores from session if available
        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        # Pull interface residues from session if available
        interface_residues: Optional[set] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = set(protected)

        t0 = _time.perf_counter()
        try:
            candidates = bridge.suggest_glycosylation_positions(
                pdb_path          = pdb_path,
                chain             = chain,
                sequence          = sequence,
                interface_residues= interface_residues,
                esm_scores        = esm_scores,
                top_n             = top_n,
            )
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            return ToolStepResult(
                tool="glycan_positions", success=False,
                error=f"Glycan position scan failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        # ── Auto-call NetNGlyc on top N candidates (when enabled) ───────────────
        netnglyc_annotated: bool = False
        try:
            import config as _cfg
            _netnglyc_enabled = getattr(_cfg, "NETNGLYC_ENABLED", True)
            _netnglyc_top_n   = getattr(_cfg, "NETNGLYC_TOP_N", 5)
        except ImportError:
            _netnglyc_enabled = True
            _netnglyc_top_n   = 5

        if _netnglyc_enabled and candidates:
            try:
                ng_bridge = self._get_netnglyc_bridge()
                top_cands = candidates[:_netnglyc_top_n]
                # Build an engineered sequence: apply the first candidate's
                # mutation to the sequence so NxS/T sequons appear at each pos.
                # We use the raw sequence as proxy (NetNGlyc scans all N positions).
                annotated = ng_bridge.integrate_with_glycan_candidates(
                    candidates          = top_cands,
                    engineered_sequence = sequence,
                )
                # Merge ost_score, ost_category, combined_confidence back into candidates
                annotated_map = {c.get("position"): c for c in annotated}
                for cand in candidates:
                    pos = cand.get("position")
                    if pos in annotated_map:
                        ann = annotated_map[pos]
                        cand["ost_score"]          = ann.get("ost_score")
                        cand["ost_category"]       = ann.get("ost_category")
                        cand["ost_prediction"]     = ann.get("ost_prediction")
                        cand["ost_confidence"]     = ann.get("ost_confidence")
                        cand["combined_confidence"] = ann.get("combined_confidence")
                # Store in session
                try:
                    self.session.set_netnglyc_results(model_id, annotated)
                except Exception:
                    pass
                netnglyc_annotated = True
            except Exception:
                pass   # NetNGlyc failure is non-fatal — proceed without OST scores

        cx_cmds, cx_exps = bridge.generate_positions_chimerax_commands(
            candidates, model_id=model_id, chain=chain, top_n=top_n
        )
        summary = bridge.generate_positions_summary(
            candidates, chain=chain, top_n=top_n
        )
        if netnglyc_annotated:
            summary += "\n  [OST scores from NetNGlyc 1.0 added to top candidates]"

        return ToolStepResult(
            tool             = "glycan_positions",
            success          = True,
            data             = {
                "candidates":         candidates,
                "count":              len(candidates),
                "netnglyc_annotated": netnglyc_annotated,
            },
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_netnglyc(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """
        Run NetNGlyc 1.0 OST recognition prediction for a sequence.

        Fetches the sequence from session, submits to the NetNGlyc REST API,
        and returns per-sequon OST scores annotated with confidence categories.
        Stores annotated results in session.netnglyc_results.
        """
        import time as _time
        try:
            import config as _cfg
            if not getattr(_cfg, "NETNGLYC_ENABLED", True):
                return ToolStepResult(
                    tool="netnglyc", success=False,
                    error="NetNGlyc is disabled (NETNGLYC_ENABLED=false in config).",
                )
        except ImportError:
            pass

        bridge   = self._get_netnglyc_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain", "A")
        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        if not sequence:
            return ToolStepResult(
                tool="netnglyc", success=False,
                error=(
                    "NetNGlyc requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        # Use the position hint from user input if present (score_engineered_sequon mode)
        sequon_position: Optional[int] = inputs.get("sequon_position")
        engineered_seq:  Optional[str] = inputs.get("engineered_sequence")
        wildtype_seq:    Optional[str] = inputs.get("wildtype_sequence") or sequence

        t0 = _time.perf_counter()
        if sequon_position and engineered_seq:
            result = bridge.score_engineered_sequon(
                sequon_position      = sequon_position,
                engineered_sequence  = engineered_seq,
                wildtype_sequence    = wildtype_seq,
            )
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if not result.get("success"):
                return ToolStepResult(
                    tool="netnglyc", success=False,
                    error=result.get("error", "NetNGlyc prediction failed"),
                    elapsed_ms=elapsed_ms,
                )
            summary = (
                f"NetNGlyc — position {sequon_position}: "
                f"OST score={result['ost_score']:.3f} ({result['ost_category']}). "
                f"{result.get('notes', '')}"
            )
            data = result
        else:
            result = bridge.predict_glycosylation(
                sequence = sequence,
                name     = f"model{model_id}_chain{chain}",
            )
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if not result.get("success"):
                return ToolStepResult(
                    tool="netnglyc", success=False,
                    error=result.get("error", "NetNGlyc prediction failed"),
                    elapsed_ms=elapsed_ms,
                )
            n_found = result["n_sequons_found"]
            n_high  = result["n_sequons_high_score"]
            summary = (
                f"NetNGlyc — {n_found} sequon(s) found, "
                f"{n_high} with high OST score (>0.7)."
            )
            data = result

        # Store in session
        try:
            self.session.set_netnglyc_results(model_id, data)
        except Exception:
            pass

        return ToolStepResult(
            tool       = "netnglyc",
            success    = True,
            data       = data,
            summary    = summary,
            elapsed_ms = elapsed_ms,
        )

    @staticmethod
    def _parse_glycan_validation_request(user_input: str) -> Optional[Dict[str, Any]]:
        """
        Parse a glycan validation sub-request from user text.

        Returns a dict with position if found, {"validate": True} if "validate"
        is present but no position, or None if the input is not a validation request.
        """
        if not user_input:
            return None
        lower = user_input.lower()
        if "validate" not in lower and "validation" not in lower:
            return None
        m = re.search(r"\b(?:position|pos|at)\s+(\d+)\b", lower)
        if m:
            return {"position": int(m.group(1))}
        return {"validate": True}

    def _run_salt_bridge(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run salt bridge analysis."""
        import time as _time
        bridge   = self._get_salt_bridge_bridge()
        model_id = inputs.get("model_id") or self._primary_model_id()
        chain    = inputs.get("chain", "A")
        top_n    = int(inputs.get("top_n", 10))

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="salt_bridge", success=False,
                error=(
                    "Salt bridge analysis requires a local PDB file.\n"
                    "  Load a structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download from RCSB."
                ),
            )

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        # Pull ESM scores and interface residues from session
        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        interface_residues: Optional[List[int]] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = protected

        t0 = _time.perf_counter()
        try:
            result = bridge.full_salt_bridge_scan(
                pdb_path           = pdb_path,
                chain              = chain,
                sequence           = sequence or "",
                interface_residues = interface_residues,
                esm_scores         = esm_scores,
                top_n              = top_n,
            )
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            return ToolStepResult(
                tool="salt_bridge", success=False,
                error=f"Salt bridge analysis failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="salt_bridge", success=False,
                error=result.get("error", "Salt bridge analysis failed"),
                elapsed_ms=elapsed_ms,
            )

        self.session.set_salt_bridge_results(model_id, result)

        cx_cmds, cx_exps = bridge.generate_chimerax_commands(result, model_id=model_id)
        summary = bridge.generate_summary(result)

        return ToolStepResult(
            tool             = "salt_bridge",
            success          = True,
            data             = result,
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_cavity(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run cavity detection and filling suggestion."""
        import time as _time
        bridge        = self._get_cavity_bridge()
        model_id      = inputs.get("model_id") or self._primary_model_id()
        chain         = inputs.get("chain", "A")
        top_n         = int(inputs.get("top_n", 10))
        assembly_mode = bool(inputs.get("assembly_mode", False))
        # chains=None → all chains (assembly mode); chains=[chain] → single chain
        chains_for_cavity = None if assembly_mode else [chain]

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="cavity", success=False,
                error=(
                    "Cavity detection requires a local PDB file.\n"
                    "  Load a structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download from RCSB."
                ),
            )

        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        esm_scores: Optional[Dict[int, float]] = None
        esm_entry = self.session.tool_results.get("esm", {}).get(model_id)
        if isinstance(esm_entry, dict):
            raw = esm_entry.get("per_residue_tolerance") or esm_entry.get("scores")
            if isinstance(raw, dict):
                esm_scores = {int(k): float(v) for k, v in raw.items()}

        interface_residues: Optional[List[int]] = None
        iface_data = self.session.get_interface_residues(model_id)
        if iface_data:
            protected = self.session.get_protected_residues_for_chain(model_id, chain)
            if protected:
                interface_residues = protected

        t0 = _time.perf_counter()
        try:
            result = bridge.full_cavity_scan(
                pdb_path           = pdb_path,
                chain              = chain,
                sequence           = sequence or "",
                interface_residues = interface_residues,
                esm_scores         = esm_scores,
                top_n              = top_n,
                chains             = chains_for_cavity,
            )
        except Exception as exc:
            import traceback as _tb
            _tb.print_exc()
            return ToolStepResult(
                tool="cavity", success=False,
                error=f"Cavity analysis failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="cavity", success=False,
                error=result.get("error", "Cavity analysis failed"),
                elapsed_ms=elapsed_ms,
            )

        self.session.set_cavity_results(model_id, result)

        cx_cmds, cx_exps = bridge.generate_chimerax_commands(result, model_id=model_id)
        summary = bridge.generate_summary(result)

        return ToolStepResult(
            tool             = "cavity",
            success          = True,
            data             = result,
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_camsol(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge   = self._get_camsol_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain")
        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        if not sequence:
            return ToolStepResult(
                tool="camsol", success=False,
                error=(
                    "No amino-acid sequence available. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )
        return bridge.analyze(
            sequence,
            model_id=model_id,
            chain=chain,
            session=self.session,
        )

    def _run_esm(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge   = self._get_esm_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain    = inputs.get("chain")
        sequence = inputs.get("sequence") or self._fetch_sequence(model_id, chain)

        if not sequence:
            return ToolStepResult(
                tool="esm", success=False,
                error=(
                    "No amino-acid sequence available. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )
        return bridge.analyze(sequence, model_id=model_id, session=self.session)

    def _run_esmfold(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run ESMFold foldability prediction via ESM Atlas API."""
        import time as _time
        bridge   = self._get_esmfold_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()

        # Resolve sequence
        sequence = inputs.get("sequence") or self._fetch_sequence(
            model_id, inputs.get("chain")
        )
        if not sequence:
            return ToolStepResult(
                tool="esmfold", success=False,
                error=(
                    "ESMFold requires an amino-acid sequence. "
                    "Load a structure first, or pass a sequence explicitly."
                ),
            )

        # ── Stage 4b: workbench fold — open the predicted model, pLDDT-colour it, and
        # matchmaker onto the reference, all LOCAL-ONLY (never the remote Atlas). This is
        # the engine-agnostic fold seam the Variant Workbench launches through; Boltz
        # later joins it via the same _fold_viz_commands path. ─────────────────────────
        if inputs.get("open_model"):
            local_only = bool(inputs.get("local_only", True))
            t0 = _time.perf_counter()
            res = bridge.predict(sequence, label=f"#{model_id}",
                                 allow_remote=not local_only)
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if not res.get("success"):
                return ToolStepResult(
                    tool="esmfold", success=False,
                    error=f"ESMFold fold failed: {res.get('error')}",
                    elapsed_ms=elapsed_ms,
                )
            # LOCAL-ONLY gate (defensive; allow_remote already blocked the network):
            # a non-local source on a local_only fold is a breach, not a silent remote fold.
            if local_only and res.get("source") != "local_venv312":
                return ToolStepResult(
                    tool="esmfold", success=False,
                    error=(f"LOCAL-ONLY breach: ESMFold returned source "
                           f"'{res.get('source')}', not local_venv312. Refusing the fold."),
                    elapsed_ms=elapsed_ms,
                )
            # Persist the predicted PDB so the viz can open it (and survive session work).
            import tempfile as _tf
            _tmp = _tf.NamedTemporaryFile(mode="w", suffix=".pdb",
                                          prefix=f"esmfold_{model_id}_", delete=False)
            _tmp.write(res.get("pdb_str", ""))
            _tmp.close()
            pdb_path = _tmp.name
            ref_id = None if inputs.get("no_reference") else (inputs.get("compare_to") or model_id)
            # Open LIVE + read the REAL model id back (V3 fix — no next_model_id() guess).
            new_id, cmds, exps = self._open_and_viz_fold_live(
                Path(pdb_path).as_posix(), {**inputs, "compare_to": ref_id})
            # Register the predicted model so next_model_id() advances — otherwise a
            # SECOND workbench fold reuses the same id (the open path doesn't otherwise
            # update session.structures). Keeps per-variant predicted models distinct.
            try:
                self.session.add_structure(
                    new_id, f"{inputs.get('engine', 'esmfold')}_pred_{model_id}",
                    path=pdb_path, metadata={"predicted": True,
                                             "engine": inputs.get("engine", "esmfold")})
            except Exception:
                pass
            mean_plddt = res.get("mean_plddt", 0.0)
            conf = ("very high" if mean_plddt > 90 else "high" if mean_plddt > 70
                    else "low" if mean_plddt > 50 else "very low")
            return ToolStepResult(
                tool="esmfold", success=True,
                data={
                    "engine":             "esmfold",
                    "target":             "monomer",   # ESMFold is monomer-only
                    "new_model_id":       new_id,
                    "reference_model_id": str(ref_id),
                    "mean_plddt":         mean_plddt,
                    "plddt":              res.get("plddt", {}),
                    "length":             res.get("length"),
                    "source":             res.get("source"),
                    "pdb_path":           pdb_path,
                    "commands":           cmds,         # executed live (transparency)
                },
                viz_commands=[],          # already executed live against the REAL id
                viz_explanations=[],
                summary=(f"ESMFold (local): model #{new_id} — mean pLDDT "
                         f"{mean_plddt:.1f} ({conf} confidence), superposed on #{ref_id}."),
                elapsed_ms=elapsed_ms,
            )

        mutation_positions: List[int] = inputs.get("mutation_positions") or []

        t0 = _time.perf_counter()
        if mutation_positions:
            # Compare wildtype vs mutant sequences at specified positions
            mut_sequence = inputs.get("mut_sequence", "")
            if not mut_sequence:
                # Single-sequence foldability prediction (no mutant provided)
                result = bridge.predict(sequence, label=f"#{model_id}")
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                if not result["success"]:
                    return ToolStepResult(
                        tool="esmfold", success=False,
                        error=f"ESMFold prediction failed: {result.get('error')}",
                        elapsed_ms=elapsed_ms,
                    )
                mean_plddt = result["mean_plddt"]
                summary = (
                    f"ESMFold: model #{model_id} — mean pLDDT {mean_plddt:.1f} "
                    f"({'high' if mean_plddt > 70 else 'low'} confidence)."
                )
                return ToolStepResult(
                    tool="esmfold", success=True,
                    data={
                        "mean_plddt":  mean_plddt,
                        "plddt":       result["plddt"],
                        "length":      result["length"],
                        "source":      result.get("source"),
                    },
                    summary=summary,
                    elapsed_ms=elapsed_ms,
                )
            else:
                cmp = bridge.compare_to_wildtype(sequence, mut_sequence, mutation_positions)
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                if not cmp["success"]:
                    return ToolStepResult(
                        tool="esmfold", success=False,
                        error=f"ESMFold comparison failed: {cmp.get('error')}",
                        elapsed_ms=elapsed_ms,
                    )
                risk = cmp["foldability_risk"]
                drop = cmp["plddt_drop"]
                summary = (
                    f"ESMFold: foldability risk = {risk} "
                    f"(mean pLDDT drop {drop:.1f} at mutation positions)."
                )
                if cmp.get("warning"):
                    summary += f"\n  {cmp['warning']}"
                return ToolStepResult(
                    tool="esmfold", success=True,
                    data=cmp,
                    summary=summary,
                    elapsed_ms=elapsed_ms,
                )
        else:
            # Plain foldability check — no mutation comparison
            result = bridge.predict(sequence, label=f"#{model_id}")
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if not result["success"]:
                return ToolStepResult(
                    tool="esmfold", success=False,
                    error=f"ESMFold prediction failed: {result.get('error')}",
                    elapsed_ms=elapsed_ms,
                )
            mean_plddt = result["mean_plddt"]
            conf       = "high" if mean_plddt > 70 else "medium" if mean_plddt > 50 else "low"
            summary    = (
                f"ESMFold: model #{model_id} — mean pLDDT {mean_plddt:.1f} "
                f"({conf} confidence, {result['length']} residues)."
            )
            return ToolStepResult(
                tool="esmfold", success=True,
                data={
                    "mean_plddt": mean_plddt,
                    "plddt":      result["plddt"],
                    "length":     result["length"],
                    "source":     result.get("source"),
                },
                summary=summary,
                elapsed_ms=elapsed_ms,
            )

    def _run_boltz(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """LOCAL-ONLY Boltz-2 fold (monomer or assembly) on the engine-agnostic seam: open
        the predicted (multi-chain) model, pLDDT-colour it, matchmaker onto the WT reference,
        surface ipTM. Seed-pinned; the bridge's fail-closed guard refuses any remote-MSA
        breach. Mirrors _run_esmfold's open_model branch — feeds the SAME _fold_viz_commands
        + fold_summary so Boltz is a second engine, not a parallel path."""
        import time as _time
        model_id   = inputs.get("model_id") or self._first_model_id()
        local_only = bool(inputs.get("local_only", True))

        # Chains to fold: an explicit assembly `chains` list, else a single chain from the
        # variant's exact `sequence` (or the loaded chain).
        chains = inputs.get("chains")
        if not chains:
            seq = inputs.get("sequence") or self._fetch_sequence(model_id, inputs.get("chain"))
            if not seq:
                return ToolStepResult(
                    tool="boltz", success=False,
                    error="Boltz needs a sequence (or explicit chains) to fold.")
            chains = [{"id": inputs.get("chain", "A"), "sequence": seq}]

        bridge = self._get_boltz_bridge()
        # TEMPLATE-GUIDED (optional): resolve each template entry to a LOCAL on-disk structure
        # path here (the router owns `_download_pdb_by_id`), then thread the list through to the
        # bridge, which translate_path's the cif/pdb into WSL. A template is not an MSA → the
        # fail-closed LOCAL-ONLY guard is unaffected. Absent → a plain de-novo fold. Fail-closed:
        # an unresolvable template fails the fold (never silently folds unguided).
        templates, terr = self._resolve_boltz_templates(inputs.get("templates"))
        if terr:
            return ToolStepResult(tool="boltz", success=False, error=terr)
        # DECLARED DISULFIDE BOND (Mode C, optional): `disulfide_constraints` is the emit-ready
        # Boltz `bond` list (the CALLER already mapped author-resnum→1-based chain index via
        # disulfide_geometry — the tested conversion); `disulfide_bonds` is the author-resnum pairs
        # kept for provenance + the readout. A constraint BIASES toward the bond, it does NOT enforce
        # geometry — the geometry is MEASURED on the result, never assumed (honesty layer).
        constraints  = inputs.get("disulfide_constraints")
        declared_ss  = inputs.get("disulfide_bonds")          # [(resnum_a, resnum_b), …] author resnums
        t0  = _time.perf_counter()
        res = bridge.predict(chains, seed=inputs.get("seed"), allow_remote=not local_only,
                             templates=templates, constraints=constraints)
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        if not res.get("success"):
            return ToolStepResult(
                tool="boltz", success=False,
                error=f"Boltz fold failed: {res.get('error')}", elapsed_ms=elapsed_ms)
        # LOCAL-ONLY gate (defensive; the bridge's fail-closed guard already blocked the network).
        if local_only and res.get("source") != "local_boltz_env":
            return ToolStepResult(
                tool="boltz", success=False,
                error=(f"LOCAL-ONLY breach: Boltz returned source '{res.get('source')}', "
                       f"not local_boltz_env. Refusing the fold."),
                elapsed_ms=elapsed_ms)

        cif_path = res.get("cif_path")
        ref_id   = None if inputs.get("no_reference") else (inputs.get("compare_to") or model_id)
        # Open LIVE + read the REAL model id back (V3 fix — no next_model_id() guess).
        new_id, cmds, exps = self._open_and_viz_fold_live(
            Path(cif_path).as_posix(), {**inputs, "compare_to": ref_id})
        try:
            self.session.add_structure(
                new_id, f"boltz_pred_{model_id}", path=cif_path,
                metadata={"predicted": True, "engine": "boltz"})
        except Exception:
            pass

        mean_plddt = res.get("mean_plddt", 0.0)
        iptm = res.get("iptm")
        conf = ("very high" if mean_plddt > 90 else "high" if mean_plddt > 70
                else "low" if mean_plddt > 50 else "very low")
        nch   = len(chains)
        shape = "monomer" if nch <= 1 else f"{nch}-chain assembly"
        templated = bool(inputs.get("templates"))
        # IMMEDIATE ADOPTION readout (use-time, no ground truth): how much the guided fold
        # resembles each template — structTM(guided fold, template). Gives instant "did it
        # reflect the template" feedback at fold time (the assist's full ΔpLDDT/Δflex needs an
        # unguided baseline; adoption does not). HIGH adoption ⇒ the fold FOLLOWS the template
        # (possible copying — not proof of correctness). Off-thread (this runs on the worker).
        adoption = None
        per_template_adopt: List[Dict[str, Any]] = []
        if templated and cif_path:
            for t in (templates or []):
                tpath = t.get("cif") or t.get("pdb")
                per_template_adopt.append({"template": tpath, "adoption": self._usalign_tm2(cif_path, tpath)})
            _av = [d["adoption"] for d in per_template_adopt if d["adoption"] is not None]
            adoption = max(_av) if _av else None
        return ToolStepResult(
            tool="boltz", success=True,
            data={
                "engine":             "boltz",
                "target":             ("monomer" if nch <= 1 else "assembly"),
                "templated":          templated,
                "adoption":           adoption,            # max structTM(guided, template) — use-time
                "per_template_adoption": per_template_adopt,
                "new_model_id":       new_id,
                "reference_model_id": str(ref_id),
                "mean_plddt":         mean_plddt,
                "iptm":               iptm,
                "chains_ptm":         res.get("chains_ptm"),
                "plddt":              res.get("plddt", {}),
                "plddt_by_chain":     res.get("plddt_by_chain"),   # {chain: {idx: pLDDT}} (hetero re-point)
                "chain_ids":          res.get("chain_ids"),        # observed CIF order (read-back guard)
                "length":             res.get("length"),
                "source":             res.get("source"),
                "seed":               res.get("seed"),
                "cif_path":           cif_path,
                "disulfide_bonds":    declared_ss,    # PROVENANCE: the declared bond(s) (author resnums)
                "constrained":        bool(constraints),  # this fold was BIASED by a declared bond
                "commands":           cmds,         # executed live (transparency)
            },
            viz_commands=[],          # already executed live against the REAL id
            viz_explanations=[],
            summary=(f"Boltz-2 ({shape}{', template-guided' if templated else ''}"
                     f"{', SS-constrained' if constraints else ''}, local): "
                     f"model #{new_id} — mean pLDDT {mean_plddt:.1f} ({conf})"
                     + (f", ipTM {iptm:.3f}" if isinstance(iptm, (int, float)) else "")
                     + (f", adopted template at {adoption:.0%}" if adoption is not None else "")
                     + f", seed={res.get('seed')}."),
            elapsed_ms=elapsed_ms,
        )

    # ── Disulfide suite — Mode A discovery (multi-seed frequency) + Mode B geometry readout ──
    def _fold_n_seeds(self, chains: List[Dict[str, str]], n: int, *,
                      templates: Optional[List[Dict[str, Any]]] = None,
                      constraints: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Fold *chains* across N seeds — the SHARED multi-seed primitive (the SAME seed convention
        as the deviation floor: `BOLTZ_SEED` base + a contiguous range), so there is ONE seeding
        scheme, not a forked second path. Returns the cif paths of the folds that SUCCEEDED.
        Mode A calls this UNCONSTRAINED (`constraints=None` — the discovery question is what the
        model thinks UNPROMPTED). Cheap callers (Mode B) must NEVER reach here."""
        import config as _cfg, time as _time
        base = int(getattr(_cfg, "BOLTZ_SEED", 0))
        bridge = self._get_boltz_bridge()
        cb = getattr(self, "_status_callback", None)
        total = max(1, int(n))
        paths: List[str] = []
        t0 = _time.perf_counter()
        for i, s in enumerate(range(base, base + total), start=1):
            if cb:                                 # per-seed heartbeat: a long run advances VISIBLY
                cb(f"🔗📊 Folding seed {i}/{total} (unconstrained)…")
            r = bridge.predict(chains, seed=s, allow_remote=False,
                               templates=templates, constraints=constraints)
            if r.get("success") and r.get("cif_path"):
                paths.append(r["cif_path"])
            if cb:                                 # …and reports each completion (done-vs-stuck)
                cb(f"🔗📊 Seed {i}/{total} folded — {len(paths)} ok so far "
                   f"({_time.perf_counter() - t0:.0f}s elapsed).")
        return paths

    def _resolve_disulfide_chains(self, inputs: Dict[str, Any]):
        chains = inputs.get("chains")
        if not chains:
            seq = inputs.get("sequence")
            if not seq:
                return None
            chains = [{"id": inputs.get("chain", "A"), "sequence": seq}]
        return chains

    def _run_disulfide_discovery(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """MODE A — DISCOVERY (observe, frequency-based). Fold the construct UNCONSTRAINED across N
        seeds and report, per cysteine pair, HOW OFTEN it sits in disulfide-bonding geometry. This
        frequency IS the model's learned pairing prior expressed EMPIRICALLY (measured, not asserted
        — always reported with N). Distinct from Mode C (which IMPOSES a bond): A observes what the
        model thinks unprompted. Reuses `_fold_n_seeds` (the shared seed machinery) + the shared
        geometry core (`pair_geometry.bonding_compatible` = SG–SG within the bonding window)."""
        import config as _cfg, itertools, time as _time
        from disulfide_geometry import parse_cys_atoms, pair_geometry
        chains = self._resolve_disulfide_chains(inputs)
        if not chains:
            return ToolStepResult(tool="disulfide_discovery", success=False,
                                  error="Disulfide discovery needs the construct's chains/sequence.")
        n = int(inputs.get("n_seeds") or getattr(_cfg, "DEVIATION_FLOOR_N", 4))
        t0 = _time.perf_counter()
        paths = self._fold_n_seeds(chains, n)             # UNCONSTRAINED — the discovery invariant
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        if not paths:
            return ToolStepResult(tool="disulfide_discovery", success=False,
                                  error="No unconstrained folds succeeded — cannot tally pairing frequency.")
        n_done = len(paths)
        tally: Dict[tuple, Dict[str, Any]] = {}
        for p in paths:
            cys = parse_cys_atoms(p)
            for ch, residues in cys.items():
                for ra, rb in itertools.combinations(sorted(residues), 2):
                    pg = pair_geometry(residues[ra], residues[rb])
                    t = tally.setdefault((ch, ra, rb), {"compat": 0, "sg": []})
                    if pg["bonding_compatible"]:
                        t["compat"] += 1
                    if pg["sg_sg"] is not None:
                        t["sg"].append(pg["sg_sg"])
        pairs = []
        for (ch, ra, rb), t in tally.items():
            sgs = sorted(t["sg"])
            pairs.append({
                "chain_a": ch, "resnum_a": ra, "chain_b": ch, "resnum_b": rb,  # intrachain: both = ch
                "n_compatible": t["compat"], "n_folds": n_done,
                "frequency": round(t["compat"] / n_done, 3),
                "median_sg_sg": (round(sgs[len(sgs) // 2], 3) if sgs else None),
            })
        pairs.sort(key=lambda d: (-d["frequency"], -d["n_compatible"]))
        top = pairs[0] if pairs else None
        if top is None:
            summary = (f"Disulfide discovery: no cysteine pairs in the construct "
                       f"(across {n_done} unconstrained folds).")
        else:
            head = "; ".join(
                f"Cys{d['resnum_a']}–Cys{d['resnum_b']}: bonding-compatible in "
                f"{d['n_compatible']}/{d['n_folds']} folds" for d in pairs[:3])
            summary = (f"Disulfide discovery ({n_done} unconstrained folds — the model's empirical "
                       f"pairing prior, measured): {head}.")
        return ToolStepResult(
            tool="disulfide_discovery", success=True,
            data={"pairs": pairs, "n_folds": n_done, "mode": "discovery"},
            summary=summary, elapsed_ms=elapsed_ms)

    def _run_disulfide_geometry(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """MODE B — GEOMETRY READOUT (observe, any pair). Report the MEASURED geometry (Cα–Cα,
        Cβ–Cβ, SG–SG, χSS vs canonical windows) for cysteine pairs in ONE existing fold — a
        FACTUAL measurement of the produced structure, NOT a declaration that a bond exists. CHEAP:
        it READS coordinates from a fold already on disk; it NEVER folds (must not trigger Mode A's
        multi-seed run). Reads `cif_path` (an existing fold); optional `pairs` = [(ra,rb)] else
        every cysteine pair."""
        import itertools
        import os as _os
        from disulfide_geometry import parse_cys_atoms, pair_geometry
        cif = inputs.get("cif_path")
        if not cif or not _os.path.isfile(cif):
            return ToolStepResult(tool="disulfide_geometry", success=False,
                                  error="Geometry readout needs an existing fold CIF on disk.")
        cys = parse_cys_atoms(cif)
        want = inputs.get("pairs")
        out = []
        for ch, residues in cys.items():
            combos = (want if want else list(itertools.combinations(sorted(residues), 2)))
            for ra, rb in combos:
                if ra in residues and rb in residues:
                    out.append({"chain_a": ch, "resnum_a": ra, "chain_b": ch, "resnum_b": rb,  # intrachain
                                **pair_geometry(residues[ra], residues[rb])})
        out.sort(key=lambda d: (d["sg_sg"] if d["sg_sg"] is not None else 1e9))
        if not out:
            summary = "Disulfide geometry: no cysteine pairs to measure in this fold."
        else:
            b = out[0]
            chi = f"{b['chi_ss']:.0f}°" if b["chi_ss"] is not None else "n/a"
            verdict = ("disulfide-compatible geometry" if b["bonding_compatible"]
                       else "geometrically incompatible")
            summary = (f"Disulfide geometry (measured, this fold): Cys{b['resnum_a']}–Cys{b['resnum_b']} "
                       f"— {verdict} (Cα–Cα {b['ca_ca']} Å, SG–SG {b['sg_sg']} Å, χSS {chi}).")
        return ToolStepResult(tool="disulfide_geometry", success=True,
                              data={"pairs": out, "mode": "geometry"}, summary=summary)

    # The load-bearing caveat for the engineering scan — rides with every Mode-D readout (the
    # heatmap is the most over-read-prone surface; the caveat must never separate from it).
    _DISULFIDE_SCAN_CAVEAT = (
        "Geometrically viable disulfide-engineering sites in THIS predicted fold — a starting "
        "point. Ranked by ROTAMER Sγ-REACHABILITY (idealized Cys sidechains placed over a χ1 sweep: "
        "can the two Sγ reach ~2.05 Å at a good χSS), with a rigid-backbone clash flag. The geometry "
        "PERMITS a disulfide; it does NOT confirm the X→C mutations are tolerated, that packing "
        "accommodates the Cys after repacking (the clash check is on the FIXED backbone — repacking "
        "may relieve a flagged clash, or introduce a new one), that the protein still folds, or that "
        "the bond will form. Validate by introducing the Cys pair and re-folding (Mode C)."
    )

    def _run_disulfide_scan(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """MODE D — ENGINEERING SCAN (find NOVEL installable sites). An all-pairs, residue-AGNOSTIC
        BACKBONE scan of ONE existing fold: where COULD a disulfide be installed if both residues
        were mutated to Cys (Cα–Cα + Cβ–Cβ + Cα-Cβ-Cβ-Cα orientation, soft-graded; NO χSS — no Sγ
        pre-mutation). CHEAP — reads coordinates, NEVER folds; a cheap Cα prefilter keeps the O(N²)
        scan fast on assemblies (output-lossless). Distinct from A/B (existing cysteines). Returns a
        ranked candidate list (source of truth) + a per-residue best-partner map (the heatmap index)
        + the load-bearing geometric-only CAVEAT. Reads `cif_path`."""
        import os as _os
        from disulfide_geometry import (parse_backbone_atoms, parse_heavy_atoms,
                                        scan_engineerable_sites, ClashGrid)
        cif = inputs.get("cif_path")
        if not cif or not _os.path.isfile(cif):
            return ToolStepResult(tool="disulfide_scan", success=False,
                                  error="Engineering scan needs an existing fold CIF on disk.")
        atoms = parse_backbone_atoms(cif)
        # Tier (b): a heavy-atom clash grid (built once) so each surfaced site's best-reach Sγ is
        # tested for vdW overlap — flags + softly demotes a sulfur-reachable-but-clashing site.
        grid = ClashGrid(parse_heavy_atoms(cif))
        ranked, best = scan_engineerable_sites(atoms, clash_grid=grid)
        # best_partner keyed by (chain, resnum); flatten to {chain: {resnum: score}} for the heatmap.
        best_by_chain: Dict[str, Dict[int, float]] = {}
        for (ch, rn), sc in best.items():
            best_by_chain.setdefault(ch, {})[rn] = round(sc, 4)
        if not ranked:
            summary = ("Engineering scan: no geometrically viable disulfide-engineering sites in "
                       "this fold. " + self._DISULFIDE_SCAN_CAVEAT)
        else:
            head = "; ".join(f"{d['resnum_a']}–{d['resnum_b']} (score {d['score']:.2f})"
                             for d in ranked[:3])
            summary = (f"Engineering scan — {len(ranked)} candidate site(s); top: {head}. "
                       + self._DISULFIDE_SCAN_CAVEAT)
        return ToolStepResult(
            tool="disulfide_scan", success=True,
            data={"pairs": ranked, "best_partner": best_by_chain, "mode": "engineering_scan",
                  "caveat": self._DISULFIDE_SCAN_CAVEAT},
            summary=summary)

    def _run_disulfide_interface_scan(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """INTERFACE SCAN — find NOVEL INTER-CHAIN (inter-subunit) installable disulfide sites: the
        CROSS-chain analogue of Mode D. Reads a MULTI-chain fold's backbone and scores every residue
        pair across DIFFERENT chains by the SAME `backbone_pair_score` (the shared primitive — no new
        geometry loop), bounded to the INTERFACE by the Cα prefilter (a pair close enough across
        chains to bond IS interface-proximal). CHEAP — reads coordinates, NEVER folds. Surfaces a
        ranked cross-chain pair list (chain_a≠chain_b) + the load-bearing geometric-only CAVEAT.
        Found pairs feed the (proven) cross-chain Mode-C declare. Reads `cif_path`; needs ≥2 chains."""
        import os as _os
        from disulfide_geometry import (parse_backbone_atoms, parse_heavy_atoms,
                                        scan_interface_sites, ClashGrid)
        cif = inputs.get("cif_path")
        if not cif or not _os.path.isfile(cif):
            return ToolStepResult(tool="disulfide_interface_scan", success=False,
                                  error="Interface scan needs an existing fold CIF on disk.")
        atoms = parse_backbone_atoms(cif)
        if len(atoms) < 2:
            return ToolStepResult(tool="disulfide_interface_scan", success=False,
                                  error="Interface scan needs a MULTI-CHAIN fold (≥2 chains).")
        # Tier (b): heavy-atom clash grid (built once) for the surfaced cross-chain candidates.
        grid = ClashGrid(parse_heavy_atoms(cif))
        ranked, best = scan_interface_sites(atoms, clash_grid=grid)
        best_by_chain: Dict[str, Dict[int, float]] = {}
        for (ch, rn), sc in best.items():
            best_by_chain.setdefault(ch, {})[rn] = round(sc, 4)
        if not ranked:
            summary = ("Interface scan: no geometrically viable INTER-CHAIN disulfide sites at the "
                       "interface of this fold. " + self._DISULFIDE_SCAN_CAVEAT)
        else:
            head = "; ".join(f"{d['chain_a']}:{d['resnum_a']}–{d['chain_b']}:{d['resnum_b']} "
                             f"(score {d['score']:.2f})" for d in ranked[:3])
            summary = (f"Interface scan — {len(ranked)} inter-chain candidate site(s); top: {head}. "
                       + self._DISULFIDE_SCAN_CAVEAT)
        return ToolStepResult(
            tool="disulfide_interface_scan", success=True,
            data={"pairs": ranked, "best_partner": best_by_chain, "mode": "interface_scan",
                  "caveat": self._DISULFIDE_SCAN_CAVEAT},
            summary=summary)

    # The ΔΔG-escalation caveat — rides with EVERY escalated readout. ΔΔG is a SECOND soft signal on
    # the geometric suggestion, NOT confirmation the bond forms; our ddG path is uncalibrated (§7:
    # ranking/sign only, ~±2.7 kcal/mol). The de-novo two-layer note is appended display-side.
    _DISULFIDE_DDG_CAVEAT = (
        "ΔΔG estimate (legacy, uncalibrated — ranking/sign only, ~±2.7 kcal/mol). A second soft "
        "signal on the geometric suggestion — not confirmation the bond will form. Validate by "
        "declaring the bond, re-folding, and measuring the as-produced geometry."
    )

    @staticmethod
    def _ddg_escalation_gate(backend: str, source: str, wsl_ok: bool) -> Tuple[bool, str]:
        """PURE escalation gate (testable, no IO). Decide whether a ΔΔG escalation may proceed given
        the resolved stability *backend*, the structure *source* ('denovo'|'loaded'), and whether the
        local PyRosetta/WSL path is available. Two boundaries:
          • LOCAL-ONLY analog — DynaMut2 is a WEB service (biosig.lab.uq.edu.au). Escalating a DE-NOVO
            design over it would UPLOAD possibly-unpublished coordinates to a third party → BLOCK
            (default to local PyRosetta). A loaded PDB (already public) is allowed over the web.
          • local backend with no WSL/PyRosetta → BLOCK (else `_score_stability` silently returns 0.0,
            a fabricated 'neutral' for scoring that never ran)."""
        be = (backend or "").strip().lower()
        if be == "dynamut2" and source == "denovo":
            return False, ("ΔΔG escalation is blocked: the active stability backend is DynaMut2 (a web "
                           "service, biosig.lab.uq.edu.au) and this is a DE-NOVO design — running it "
                           "would upload your unpublished coordinates off-machine. Set "
                           "ROSETTA_BACKEND=local (PyRosetta/WSL2) to estimate ΔΔG locally.")
        if be == "local" and not wsl_ok:
            return False, ("ΔΔG escalation needs the local PyRosetta path (ROSETTA_BACKEND=local), but "
                           "WSL2/PyRosetta is not available. Configure it, or this stays a "
                           "geometry-only suggestion.")
        return True, ""

    def _run_disulfide_ddg_estimate(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """ΔΔG-ESCALATION — an energetic read for ONE geometrically-flagged interface pair (the §9
        analysis-side convergence: the fold-based suite acquires an energetic estimate WITHOUT merging
        into the legacy candidate-finding tool). Reuses the legacy ΔΔG engine via
        `DisulfideBridge._score_stability` (the NARROW primitive); NEVER `disulfide_bridge.analyze`
        (its 4.5 Å Cβ / ESM / ddG filters could silently DROP the very pair the user clicked).

        Inputs: ``pdb_path`` (a PDB on disk — PyRosetta `cleanATOM` is PDB-only, so the caller saves
        the LIVE model to PDB, not the mmCIF), ``chain_a/resnum_a/from_aa_a`` + the b-side, and
        ``source`` ('denovo'|'loaded'). Correctness gates, fail-CLOSED:
          1. from_aa VERIFICATION — the claimed WT residue (recovered design-side from template_cells)
             must match the residue AT THAT (chain, resnum) in the EXACT PDB being scored. A mismatch
             (off-by-one / wrong chain / stale-after-edit) would silently score a DIFFERENT mutation
             and return a plausible-but-wrong ΔΔG → abort.
          2. backend gate (`_ddg_escalation_gate`) — no de-novo web upload; local needs WSL."""
        from disulfide_bridge import _three_to_one, parse_pdb_atoms
        pdb = inputs.get("pdb_path")
        ca, ra, aa_a = inputs.get("chain_a"), inputs.get("resnum_a"), inputs.get("from_aa_a")
        cb, rb, aa_b = inputs.get("chain_b"), inputs.get("resnum_b"), inputs.get("from_aa_b")
        source = (inputs.get("source") or "loaded").strip().lower()
        if not pdb or not Path(pdb).is_file():
            return ToolStepResult(tool="disulfide_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs a structure (PDB) on disk.")
        if None in (ca, ra, aa_a, cb, rb, aa_b):
            return ToolStepResult(tool="disulfide_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs both positions with their WT residue (chain, resnum, from_aa).")
        ca, cb, ra, rb = str(ca), str(cb), int(ra), int(rb)
        aa_a, aa_b = str(aa_a).upper(), str(aa_b).upper()

        # 1) VERIFY each from_aa against the EXACT structure being scored (silent-wrong-residue guard)
        atoms = parse_pdb_atoms(pdb)
        for ch, rn, claimed in ((ca, ra, aa_a), (cb, rb, aa_b)):
            res = (atoms.get(ch) or {}).get(rn)
            if res is None:
                return ToolStepResult(tool="disulfide_ddg_estimate", success=False,
                                      error=f"ΔΔG estimate: residue {ch}:{rn} not found in the scored structure.")
            struct_aa = _three_to_one(res.get("resname", "UNK"))
            if struct_aa != claimed:
                return ToolStepResult(tool="disulfide_ddg_estimate", success=False,
                                      error=(f"ΔΔG estimate ABORTED — residue mismatch at {ch}:{rn}: the design says "
                                             f"{claimed} but the structure has {struct_aa}. Scoring would target the "
                                             f"wrong mutation (off-by-one / wrong chain / stale edit). Re-scan first."))

        # 2) backend gate (fail-closed BEFORE any compute/upload)
        from rosetta_bridge import _select_backend
        backend = _select_backend()
        wsl_ok = False
        if backend == "local":
            try:
                from wsl_bridge import WSLBridge
                _w = WSLBridge()
                wsl_ok = bool(_w.is_available() and _w.check_pyrosetta())
            except Exception:
                wsl_ok = False
        ok, reason = self._ddg_escalation_gate(backend, source, wsl_ok)
        if not ok:
            return ToolStepResult(tool="disulfide_ddg_estimate", success=False, error=reason)

        # 3) score EXACTLY this pair via the narrow primitive (per-chain X→C ddG; never analyze())
        bridge = self._get_disulfide_bridge()
        cand = [{"chain_a_residue": ra, "chain_b_residue": rb,
                 "chain_a_aa": aa_a, "chain_b_aa": aa_b}]
        scored = bridge._score_stability(cand, pdb, ca, cb)
        c0 = scored[0] if scored else {}
        ddg_a, ddg_b = c0.get("ddg_a"), c0.get("ddg_b")
        ddg_mean = None if (ddg_a is None or ddg_b is None) else round((ddg_a + ddg_b) / 2.0, 3)
        summary = (f"ΔΔG (legacy, {backend}) for {ca}:{ra}{aa_a}→C / {cb}:{rb}{aa_b}→C — "
                   f"{aa_a}{ra}C {ddg_a:+.2f}, {aa_b}{rb}C {ddg_b:+.2f} kcal/mol "
                   f"(mean {ddg_mean:+.2f}). {self._DISULFIDE_DDG_CAVEAT}")
        return ToolStepResult(
            tool="disulfide_ddg_estimate", success=True,
            data={"chain_a": ca, "resnum_a": ra, "from_aa_a": aa_a,
                  "chain_b": cb, "resnum_b": rb, "from_aa_b": aa_b,
                  "ddg_a": ddg_a, "ddg_b": ddg_b, "ddg_mean": ddg_mean,
                  "backend": backend, "source": source, "caveat": self._DISULFIDE_DDG_CAVEAT},
            summary=summary)

    # ── Proline-stabilization scan (panel-primary; the legacy NL `proline`/ProlineBridge stays parallel) ──
    # The load-bearing caveat — rides with EVERY proline-scan readout (measured-not-promised).
    _PROLINE_SCAN_CAVEAT = (
        "Geometrically favourable proline-substitution sites in THIS predicted/loaded structure — a "
        "starting point. Ranked by backbone φ/ψ proline-compatibility, with a backbone-H-bond-donor "
        "flag (proline has no amide H, so it cannot DONATE the N–H···O bond loops tolerate but helices/"
        "sheets rely on — flagged + soft-penalized, NOT excluded; H-bond detection on a predicted/rigid "
        "backbone is uncertain). Geometry PERMITS a stabilizing proline; it does NOT confirm the X→Pro "
        "mutation is tolerated, that the protein still folds, or that it stabilizes. Validate by "
        "substituting → re-folding (no constraint) → comparing; the X→Pro ΔΔG is a second soft signal."
    )

    def _run_proline_scan(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """PROLINE-STABILIZATION SCAN — per-residue X→Pro candidates ranked by backbone φ/ψ proline-
        compatibility × a backbone-H-bond-donor penalty (`proline_geometry`). CHEAP — reads the CIF,
        NEVER folds; works on a de-novo fold OR a LOADED crystal/model (multimers native). The correct
        FRESH version (real geometric DSSP-style H-bond detection + a SOFT penalty), DISTINCT from the
        legacy NL `proline`/`ProlineBridge` (BioPython/PDB + ESM/DynaMut2 console path — kept parallel,
        §9). Returns a ranked candidate list (source of truth) + a per-residue best-score heatmap map +
        the existing prolines + the load-bearing CAVEAT. Reads `cif_path`."""
        import os as _os
        from proline_geometry import parse_backbone_with_names, scan_proline_sites, existing_prolines
        cif = inputs.get("cif_path")
        if not cif or not _os.path.isfile(cif):
            return ToolStepResult(tool="proline_scan", success=False,
                                  error="Proline scan needs an existing structure CIF on disk.")
        atoms = parse_backbone_with_names(cif)
        ranked, best = scan_proline_sites(atoms)
        existing = [list(p) for p in existing_prolines(atoms)]
        best_by_chain: Dict[str, Dict[int, float]] = {}
        for (ch, rn), sc in best.items():
            best_by_chain.setdefault(ch, {})[rn] = round(sc, 4)
        if not ranked:
            summary = ("Proline scan: no proline-favourable sites in this structure. "
                       + self._PROLINE_SCAN_CAVEAT)
        else:
            head = "; ".join(
                f"{d['from_aa']}{d['position']}→P (score {d['score']:.2f}"
                + (", H-bond donor" if d['hbond_donates'] else "") + ")" for d in ranked[:3])
            summary = (f"Proline scan — {len(ranked)} candidate site(s); top: {head}. "
                       + self._PROLINE_SCAN_CAVEAT)
        return ToolStepResult(
            tool="proline_scan", success=True,
            data={"candidates": ranked, "best_partner": best_by_chain, "existing": existing,
                  "caveat": self._PROLINE_SCAN_CAVEAT, "mode": "proline_scan"},
            summary=summary)

    _PROLINE_DDG_CAVEAT = (
        "X→Pro ΔΔG (legacy, uncalibrated — ranking/sign only, ~±2.7 kcal/mol). The stabilization "
        "estimate itself, but a SOFT signal — not confirmation the mutation stabilizes or that the "
        "protein still folds. Validate by substituting → re-folding → comparing."
    )

    def _run_proline_ddg_estimate(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """ΔΔG-ESCALATION for ONE proline candidate — the `disulfide_ddg_estimate` pattern with
        ``to_aa='P'``. A DIRECT single-mutation `RosettaBridge.analyze` (NOT the legacy
        `full_proline_scan` pipeline — no SASA/ESM/interface machinery). from_aa is VERIFIED against
        the residue AT (chain, resnum) in the EXACT PDB scored (silent-wrong-mutation guard); de-novo
        + web backend BLOCKED (no off-machine upload). X→Pro ΔΔG IS the stabilization estimate.
        Inputs: ``pdb_path``, ``chain``/``resnum``/``from_aa``, ``source`` ('denovo'|'loaded')."""
        from disulfide_bridge import _three_to_one, parse_pdb_atoms
        pdb = inputs.get("pdb_path")
        ch, rn, aa = inputs.get("chain"), inputs.get("resnum"), inputs.get("from_aa")
        source = (inputs.get("source") or "loaded").strip().lower()
        if not pdb or not Path(pdb).is_file():
            return ToolStepResult(tool="proline_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs a structure (PDB) on disk.")
        if None in (ch, rn, aa):
            return ToolStepResult(tool="proline_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs the position with its WT residue (chain, resnum, from_aa).")
        ch, rn, aa = str(ch), int(rn), str(aa).upper()
        if aa == "P":
            return ToolStepResult(tool="proline_ddg_estimate", success=False,
                                  error="That position is already proline — X→Pro is a no-op.")
        # 1) VERIFY from_aa against the EXACT scored structure (silent-wrong-residue guard)
        atoms = parse_pdb_atoms(pdb)
        res = (atoms.get(ch) or {}).get(rn)
        if res is None:
            return ToolStepResult(tool="proline_ddg_estimate", success=False,
                                  error=f"ΔΔG estimate: residue {ch}:{rn} not found in the scored structure.")
        struct_aa = _three_to_one(res.get("resname", "UNK"))
        if struct_aa != aa:
            return ToolStepResult(tool="proline_ddg_estimate", success=False,
                                  error=(f"ΔΔG estimate ABORTED — residue mismatch at {ch}:{rn}: the scan says {aa} "
                                         f"but the structure has {struct_aa}. Scoring would target the wrong mutation "
                                         f"(off-by-one / wrong chain / stale edit). Re-scan first."))
        # 2) backend gate (fail-closed BEFORE any compute/upload) — reuses the disulfide gate
        from rosetta_bridge import _select_backend, RosettaBridge
        backend = _select_backend()
        wsl_ok = False
        if backend == "local":
            try:
                from wsl_bridge import WSLBridge
                _w = WSLBridge()
                wsl_ok = bool(_w.is_available() and _w.check_pyrosetta())
            except Exception:
                wsl_ok = False
        ok, reason = self._ddg_escalation_gate(backend, source, wsl_ok)
        if not ok:
            return ToolStepResult(tool="proline_ddg_estimate", success=False, error=reason)
        # 3) score EXACTLY this X→Pro mutation (single mutation; NOT the legacy full pipeline)
        bridge = RosettaBridge()
        r = bridge.analyze(pdb_path=pdb,
                           mutations=[{"chain": ch, "position": rn, "from_aa": aa, "to_aa": "P"}])
        ddg = None
        if getattr(r, "success", False):
            for key, v in (r.data.get("ddg_scores", {}) or {}).items():
                m = re.match(r"[A-Z](\d+)[A-Z]", str(key))
                if m and int(m.group(1)) == rn:
                    ddg = float(v); break
        if ddg is None:
            return ToolStepResult(tool="proline_ddg_estimate", success=False,
                                  error=f"ΔΔG estimate produced no score for {aa}{rn}P ({backend}).")
        summary = (f"X→Pro ΔΔG (legacy, {backend}) for {ch}:{aa}{rn}P — {ddg:+.2f} kcal/mol. "
                   + self._PROLINE_DDG_CAVEAT)
        return ToolStepResult(
            tool="proline_ddg_estimate", success=True,
            data={"chain": ch, "resnum": rn, "from_aa": aa, "ddg": ddg,
                  "backend": backend, "source": source, "caveat": self._PROLINE_DDG_CAVEAT},
            summary=summary)

    # ── Cavity-filling scan (panel-primary; the legacy NL `cavity`/CavityBridge stays parallel) ──
    # The load-bearing caveat — rides with EVERY cavity-scan readout. CONTEXT-DEPENDENT and honest to
    # BOTH the cautionary literature (generic thermostability) and the success literature (the RSV
    # prefusion-F vaccines): cavity-filling's value depends on the cavity's STRUCTURAL ROLE, which a
    # geometric scan cannot judge — the tool surfaces viable fills, the designer supplies the insight.
    _CAVITY_SCAN_CAVEAT = (
        "A buried cavity exists in THIS predicted/loaded structure; filling it PERMITS better packing. "
        "Cavity-filling gives modest, variable gains for generic thermostability (Matthews, T4 lysozyme; "
        "Machicado, apoflavodoxin — measured 0.0–0.6 kcal/mol, sub-~20 Å³ voids destabilize regardless), "
        "but is a PROVEN, POWERFUL technique for CONFORMATIONAL stabilization — locking a target state — "
        "central to the RSV prefusion-F vaccines (McLellan/Graham, DS-Cav1). Its value depends heavily on "
        "the cavity's structural role, which this geometric scan CANNOT judge: the tool surfaces "
        "geometrically-viable fills; you supply the structural insight about whether a cavity is "
        "conformationally important. Does NOT confirm the substitution is tolerated, that the protein "
        "still folds, that it stabilizes, or that the cavity isn't functional (a ligand/water/catalytic "
        "site). Validate by substituting → re-folding (no constraint) → comparing; the ΔΔG is a second "
        "soft signal."
    )

    def _run_cavity_scan(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """CAVITY-FILLING SCAN — detect internal voids (grid + probe-sphere + exterior solvent flood)
        then rank small→larger hydrophobic FILLS that reach into a void clash-free, ROTAMER-AWARE
        (`cavity_geometry`). CHEAP — reads the CIF, NEVER folds; works on a de-novo fold OR a LOADED
        crystal/model (multimers native — interface cavities fall out). The correct FRESH version (true
        geometric void detection + rotamer-placement + clash), DISTINCT from the legacy `cavity_bridge`
        (SASA-burial proxy + static volume-table lookup — kept parallel, §9). Returns the ranked
        candidate list (source of truth) + a per-residue best-score heatmap map + the per-void summary +
        the load-bearing CAVEAT. Reads `cif_path`."""
        import os as _os
        from cavity_geometry import parse_residue_atoms, parse_heavy_atoms, scan_cavity_sites
        cif = inputs.get("cif_path")
        if not cif or not _os.path.isfile(cif):
            return ToolStepResult(tool="cavity_scan", success=False,
                                  error="Cavity scan needs an existing structure CIF on disk.")
        heavy = parse_heavy_atoms(cif)
        residues = parse_residue_atoms(cif)
        candidates, best, cavities = scan_cavity_sites(heavy, residues)
        # best is already {chain: {resnum: score}} — the heatmap map shape the panel expects
        best_by_chain: Dict[str, Dict[int, float]] = {
            ch: {int(rn): round(float(sc), 4) for rn, sc in rmap.items()} for ch, rmap in best.items()}
        if not cavities:
            summary = ("Cavity scan: no internal cavities ≥ the volume floor in this structure "
                       "(well-packed at this probe). " + self._CAVITY_SCAN_CAVEAT)
        elif not candidates:
            summary = (f"Cavity scan — {len(cavities)} internal cavity(ies), but no clash-free "
                       f"small→larger fill reaches a void. " + self._CAVITY_SCAN_CAVEAT)
        else:
            head = "; ".join(
                f"{c['from_aa']}{c['position']}{c['to_aa']} (cav {c['cavity_id']}, {c['void_volume']:.0f} Å³"
                + (", ⚠ clash" if c["clash"] else "") + ")" for c in candidates[:3])
            summary = (f"Cavity scan — {len(cavities)} internal cavity(ies), {len(candidates)} viable "
                       f"fill(s); top: {head}. " + self._CAVITY_SCAN_CAVEAT)
        return ToolStepResult(
            tool="cavity_scan", success=True,
            data={"candidates": candidates, "best_partner": best_by_chain, "cavities": cavities,
                  "caveat": self._CAVITY_SCAN_CAVEAT, "mode": "cavity_scan"},
            summary=summary)

    _CAVITY_DDG_CAVEAT = (
        "Cavity-fill ΔΔG (legacy, uncalibrated — ranking/sign only, ~±2.7 kcal/mol). A SOFT signal on "
        "the geometric fill — not confirmation it stabilizes, that the protein still folds, or that the "
        "cavity isn't conformationally/functionally important. Validate by substituting → re-folding → "
        "comparing."
    )

    def _run_cavity_ddg_estimate(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """ΔΔG-ESCALATION for ONE cavity-fill — the `proline_ddg_estimate` pattern, but the target
        is VARIABLE (``to_aa`` = the filling residue, not a fixed 'P'/'C'). A DIRECT single-mutation
        `RosettaBridge.analyze`. from_aa is VERIFIED against the residue AT (chain, resnum) in the EXACT
        PDB scored (silent-wrong-mutation guard); de-novo + web backend BLOCKED (no off-machine upload).
        Inputs: ``pdb_path``, ``chain``/``resnum``/``from_aa``/``to_aa``, ``source`` ('denovo'|'loaded')."""
        from disulfide_bridge import _three_to_one, parse_pdb_atoms
        pdb = inputs.get("pdb_path")
        ch, rn = inputs.get("chain"), inputs.get("resnum")
        aa, to_aa = inputs.get("from_aa"), inputs.get("to_aa")
        source = (inputs.get("source") or "loaded").strip().lower()
        if not pdb or not Path(pdb).is_file():
            return ToolStepResult(tool="cavity_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs a structure (PDB) on disk.")
        if None in (ch, rn, aa, to_aa):
            return ToolStepResult(tool="cavity_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs the position with its WT + target residue "
                                        "(chain, resnum, from_aa, to_aa).")
        ch, rn, aa, to_aa = str(ch), int(rn), str(aa).upper(), str(to_aa).upper()
        if aa == to_aa:
            return ToolStepResult(tool="cavity_ddg_estimate", success=False,
                                  error="The fill target equals the WT residue — no mutation to score.")
        # 1) VERIFY from_aa against the EXACT scored structure (silent-wrong-residue guard)
        atoms = parse_pdb_atoms(pdb)
        res = (atoms.get(ch) or {}).get(rn)
        if res is None:
            return ToolStepResult(tool="cavity_ddg_estimate", success=False,
                                  error=f"ΔΔG estimate: residue {ch}:{rn} not found in the scored structure.")
        struct_aa = _three_to_one(res.get("resname", "UNK"))
        if struct_aa != aa:
            return ToolStepResult(tool="cavity_ddg_estimate", success=False,
                                  error=(f"ΔΔG estimate ABORTED — residue mismatch at {ch}:{rn}: the scan says {aa} "
                                         f"but the structure has {struct_aa}. Scoring would target the wrong mutation "
                                         f"(off-by-one / wrong chain / stale edit). Re-scan first."))
        # 2) backend gate (fail-closed BEFORE any compute/upload) — reuses the shared escalation gate
        from rosetta_bridge import _select_backend, RosettaBridge
        backend = _select_backend()
        wsl_ok = False
        if backend == "local":
            try:
                from wsl_bridge import WSLBridge
                _w = WSLBridge()
                wsl_ok = bool(_w.is_available() and _w.check_pyrosetta())
            except Exception:
                wsl_ok = False
        ok, reason = self._ddg_escalation_gate(backend, source, wsl_ok)
        if not ok:
            return ToolStepResult(tool="cavity_ddg_estimate", success=False, error=reason)
        # 3) score EXACTLY this fill mutation (single mutation)
        bridge = RosettaBridge()
        r = bridge.analyze(pdb_path=pdb,
                           mutations=[{"chain": ch, "position": rn, "from_aa": aa, "to_aa": to_aa}])
        ddg = None
        if getattr(r, "success", False):
            for key, v in (r.data.get("ddg_scores", {}) or {}).items():
                m = re.match(r"[A-Z](\d+)[A-Z]", str(key))
                if m and int(m.group(1)) == rn:
                    ddg = float(v); break
        if ddg is None:
            return ToolStepResult(tool="cavity_ddg_estimate", success=False,
                                  error=f"ΔΔG estimate produced no score for {aa}{rn}{to_aa} ({backend}).")
        summary = (f"Cavity-fill ΔΔG (legacy, {backend}) for {ch}:{aa}{rn}{to_aa} — {ddg:+.2f} kcal/mol. "
                   + self._CAVITY_DDG_CAVEAT)
        return ToolStepResult(
            tool="cavity_ddg_estimate", success=True,
            data={"chain": ch, "resnum": rn, "from_aa": aa, "to_aa": to_aa, "ddg": ddg,
                  "backend": backend, "source": source, "caveat": self._CAVITY_DDG_CAVEAT},
            summary=summary)

    # ── Salt-bridge scan (panel-primary; the legacy NL `salt_bridge`/SaltBridgeBridge stays parallel) ──
    # The load-bearing caveat — rides with EVERY salt-bridge readout. CONTEXT-DEPENDENT and honest to the
    # subtle literature: a salt bridge is favourable when good geometry COINCIDES with partial burial and
    # a complementary environment, but carries a DESOLVATION penalty — surface bridges are often only
    # marginally stabilizing or neutral, and a continuum analysis (Hendsch & Tidor 1994) found a majority
    # DEstabilizing once desolvation is counted (Kumar & Nussinov 1999: most stabilizing but geometry +
    # burial-dependent). The geometric scan CANNOT resolve the electrostatic/desolvation balance.
    _SALTBRIDGE_SCAN_CAVEAT = (
        "Salt bridges are CONTEXT-DEPENDENT stabilizers. Favourable when good geometry (closest "
        "carboxyl-O↔basic-N ≤4 Å, Barlow & Thornton 1983) coincides with partial burial and a "
        "complementary environment; but burying charges costs a DESOLVATION penalty, so surface salt "
        "bridges are often only marginally stabilizing or even neutral, and a continuum-electrostatics "
        "analysis (Hendsch & Tidor 1994) found a majority DEstabilizing once desolvation is counted "
        "(Kumar & Nussinov 1999: most stabilizing, but geometry- and burial-dependent). This scan ranks "
        "by geometry × a burial factor and surfaces geometric candidates — it CANNOT resolve the full "
        "electrostatic/desolvation balance: you judge whether the context is favourable; the re-fold "
        "validates. Novel reach uses a χ1 charged-group reach model (an approximation, not a full rotamer "
        "library). Does NOT confirm the substitution is tolerated, that the protein still folds, or that "
        "it stabilizes. Validate by substituting both positions → re-folding (NO constraint — charged "
        "residues fold natively) → comparing; the ΔΔG is a second soft signal."
    )
    SB_NOVEL_TOP_N = 100        # cap the displayed novel shortlist (the heatmap best_partner keeps ALL);
                                # salt bridges are geometrically easy to introduce, so bound the table

    def _run_saltbridge_scan(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """SALT-BRIDGE SCAN — two halves on ONE existing structure (de-novo fold OR a LOADED crystal/
        model, multimers native): (1) ASSESS existing Asp/Glu↔Arg/Lys pairs (closest carboxyl-O↔basic-N
        + H-bond flag + burial; 4–5 Å near-misses flagged optimizable — pure measurement); (2) suggest
        NOVEL complementary charge-pair sites (place acid+base over a χ1 reach-ring, ROTAMER-AWARE clash,
        intra + inter-chain). CHEAP — reads the CIF, NEVER folds. Geometry × a FreeSASA burial factor.
        The correct FRESH version (real geometry + reachability), DISTINCT from the legacy NL
        `salt_bridge`/SaltBridgeBridge (proximity + SASA proxy — kept parallel, §9). Returns the existing
        + novel ranked lists (source of truth) + a per-residue best-score heatmap map + the load-bearing
        context-dependent CAVEAT. Reads `cif_path`."""
        import os as _os
        from disulfide_geometry import ClashGrid, parse_heavy_atoms
        import saltbridge_geometry as sbg
        cif = inputs.get("cif_path")
        if not cif or not _os.path.isfile(cif):
            return ToolStepResult(tool="saltbridge_scan", success=False,
                                  error="Salt-bridge scan needs an existing structure CIF on disk.")
        residues = sbg.parse_residues(cif)
        sasa = sbg.compute_sasa_map(cif)         # FreeSASA (declared dep); {} → NEUTRAL burial (geometry-only)
        existing, _ex_best = sbg.scan_existing_pairs(residues, sasa_map=sasa)
        grid = ClashGrid(parse_heavy_atoms(cif))
        novel, best = sbg.scan_novel_sites(residues, clash_grid=grid, sasa_map=sasa)
        if len(residues) >= 2:                   # multimer → also the cross-chain interface scan
            inovel, ibest = sbg.scan_novel_interface(residues, clash_grid=grid, sasa_map=sasa)
            novel = sorted(novel + inovel, key=lambda d: -d["score"])
            for k, v in ibest.items():
                best[k] = max(best.get(k, 0.0), v)
        n_novel_total = len(novel)
        novel = novel[:self.SB_NOVEL_TOP_N]      # cap the TABLE; best_partner (heatmap) keeps all
        best_by_chain: Dict[str, Dict[int, float]] = {}
        for (ch, rn), sc in best.items():
            best_by_chain.setdefault(ch, {})[rn] = round(sc, 4)
        burial_note = "" if sasa else " (burial unavailable — geometry-only score; the burial factor needs FreeSASA)"
        cap_note = (f" (showing top {self.SB_NOVEL_TOP_N} of {n_novel_total})"
                    if n_novel_total > self.SB_NOVEL_TOP_N else "")
        parts = [f"Salt-bridge scan — {len(existing)} existing pair(s), {n_novel_total} novel candidate(s){cap_note}"]
        if existing:
            e = existing[0]
            parts.append(f"; top existing {e['type']} {e['chain_a']}:{e['resnum_a']}↔{e['chain_b']}:{e['resnum_b']} "
                         f"{e['on_dist']}Å" + (" (optimizable)" if e.get("optimizable") else ""))
        if novel:
            c = novel[0]
            parts.append(f"; top novel {c['from_aa_a']}{c['resnum_a']}{c['to_aa_a']}+{c['from_aa_b']}{c['resnum_b']}{c['to_aa_b']} "
                         f"(score {c['score']:.2f})")
        summary = "".join(parts) + burial_note + ". " + self._SALTBRIDGE_SCAN_CAVEAT
        return ToolStepResult(
            tool="saltbridge_scan", success=True,
            data={"existing": existing, "novel": novel, "best_partner": best_by_chain,
                  "n_novel_total": n_novel_total, "burial_available": bool(sasa),
                  "caveat": self._SALTBRIDGE_SCAN_CAVEAT, "mode": "saltbridge_scan"},
            summary=summary)

    _SALTBRIDGE_DDG_CAVEAT = (
        "Salt-bridge ΔΔG (legacy, uncalibrated — ranking/sign only, ~±2.7 kcal/mol). The two positions "
        "are scored as INDEPENDENT single mutations (not the cooperative pair energy), and ΔΔG does NOT "
        "model the desolvation/electrostatic balance — a SOFT signal on the geometric suggestion. "
        "Validate by substituting both → re-folding (no constraint) → comparing."
    )

    def _run_saltbridge_ddg_estimate(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """ΔΔG-ESCALATION for ONE novel salt-bridge candidate — the `disulfide_ddg_estimate` pattern for
        a 2-residue change, but the targets are VARIABLE (to_aa_a/to_aa_b = the chosen acid/base, not a
        fixed 'C'). A DIRECT two-mutation `RosettaBridge.analyze`. Each from_aa is VERIFIED against the
        residue AT (chain, resnum) in the EXACT PDB scored (silent-wrong-mutation guard, BOTH positions);
        de-novo + web backend BLOCKED (no off-machine upload). Inputs: ``pdb_path``, ``chain_a/resnum_a/
        from_aa_a/to_aa_a`` + the b-side, ``source`` ('denovo'|'loaded')."""
        from disulfide_bridge import _three_to_one, parse_pdb_atoms
        pdb = inputs.get("pdb_path")
        ca, ra, aa_a, ta = (inputs.get("chain_a"), inputs.get("resnum_a"),
                            inputs.get("from_aa_a"), inputs.get("to_aa_a"))
        cb, rb, aa_b, tb = (inputs.get("chain_b"), inputs.get("resnum_b"),
                            inputs.get("from_aa_b"), inputs.get("to_aa_b"))
        source = (inputs.get("source") or "loaded").strip().lower()
        if not pdb or not Path(pdb).is_file():
            return ToolStepResult(tool="saltbridge_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs a structure (PDB) on disk.")
        if None in (ca, ra, aa_a, ta, cb, rb, aa_b, tb):
            return ToolStepResult(tool="saltbridge_ddg_estimate", success=False,
                                  error="ΔΔG estimate needs both positions with their WT + target residue "
                                        "(chain, resnum, from_aa, to_aa).")
        ca, cb, ra, rb = str(ca), str(cb), int(ra), int(rb)
        aa_a, aa_b, ta, tb = aa_a.upper(), aa_b.upper(), ta.upper(), tb.upper()
        # 1) VERIFY each from_aa against the EXACT scored structure (silent-wrong-residue guard, BOTH)
        atoms = parse_pdb_atoms(pdb)
        for ch, rn, claimed in ((ca, ra, aa_a), (cb, rb, aa_b)):
            res = (atoms.get(ch) or {}).get(rn)
            if res is None:
                return ToolStepResult(tool="saltbridge_ddg_estimate", success=False,
                                      error=f"ΔΔG estimate: residue {ch}:{rn} not found in the scored structure.")
            struct_aa = _three_to_one(res.get("resname", "UNK"))
            if struct_aa != claimed:
                return ToolStepResult(tool="saltbridge_ddg_estimate", success=False,
                                      error=(f"ΔΔG estimate ABORTED — residue mismatch at {ch}:{rn}: the scan says "
                                             f"{claimed} but the structure has {struct_aa}. Scoring would target the "
                                             f"wrong mutation (off-by-one / wrong chain / stale edit). Re-scan first."))
        # 2) backend gate (fail-closed BEFORE any compute/upload) — reuses the shared escalation gate
        from rosetta_bridge import _select_backend, RosettaBridge
        backend = _select_backend()
        wsl_ok = False
        if backend == "local":
            try:
                from wsl_bridge import WSLBridge
                _w = WSLBridge()
                wsl_ok = bool(_w.is_available() and _w.check_pyrosetta())
            except Exception:
                wsl_ok = False
        ok, reason = self._ddg_escalation_gate(backend, source, wsl_ok)
        if not ok:
            return ToolStepResult(tool="saltbridge_ddg_estimate", success=False, error=reason)
        # 3) score the two mutations (independent single-mutation ΔΔG each — the legacy engine's unit)
        bridge = RosettaBridge()
        r = bridge.analyze(pdb_path=pdb,
                           mutations=[{"chain": ca, "position": ra, "from_aa": aa_a, "to_aa": ta},
                                      {"chain": cb, "position": rb, "from_aa": aa_b, "to_aa": tb}])
        ddg_a = ddg_b = None
        if getattr(r, "success", False):
            for key, v in (r.data.get("ddg_scores", {}) or {}).items():
                m = re.match(r"[A-Z](\d+)([A-Z])", str(key))      # from-aa, POSITION, TO-aa
                if not m:
                    continue
                pos, to = int(m.group(1)), m.group(2)
                # disambiguate by (position, target): a salt bridge always pairs an ACID with a BASE,
                # so ta != tb — this stays correct even for a cross-chain pair that shares a resnum.
                if pos == ra and to == ta and ddg_a is None:
                    ddg_a = float(v)
                elif pos == rb and to == tb and ddg_b is None:
                    ddg_b = float(v)
        if ddg_a is None or ddg_b is None:
            return ToolStepResult(tool="saltbridge_ddg_estimate", success=False,
                                  error=f"ΔΔG estimate produced no score for {aa_a}{ra}{ta} / {aa_b}{rb}{tb} ({backend}).")
        ddg_mean = round((ddg_a + ddg_b) / 2.0, 3)
        summary = (f"Salt-bridge ΔΔG (legacy, {backend}) for {ca}:{ra}{aa_a}→{ta} / {cb}:{rb}{aa_b}→{tb} — "
                   f"{aa_a}{ra}{ta} {ddg_a:+.2f}, {aa_b}{rb}{tb} {ddg_b:+.2f} kcal/mol (mean {ddg_mean:+.2f}). "
                   + self._SALTBRIDGE_DDG_CAVEAT)
        return ToolStepResult(
            tool="saltbridge_ddg_estimate", success=True,
            data={"chain_a": ca, "resnum_a": ra, "from_aa_a": aa_a, "to_aa_a": ta,
                  "chain_b": cb, "resnum_b": rb, "from_aa_b": aa_b, "to_aa_b": tb,
                  "ddg_a": ddg_a, "ddg_b": ddg_b, "ddg_mean": ddg_mean,
                  "backend": backend, "source": source, "caveat": self._SALTBRIDGE_DDG_CAVEAT},
            summary=summary)

    def _resolve_boltz_templates(
        self, templates: Optional[List[Dict[str, Any]]]
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        """Resolve every template entry to a LOCAL on-disk structure path. An entry may carry an
        explicit ``cif``/``pdb`` path (used as-is) OR a ``pdb_id`` (downloaded via RCSB). Returns
        (resolved_list, None) or (None, error). The per-template steering fields
        (chain_id/template_id/force/threshold) pass through verbatim; ``pdb_id`` is stripped once
        resolved so only path + steering fields reach the bridge.

        TEMPLATE FORMAT: a ``pdb_id`` is resolved to the official RCSB **mmCIF** (not PDB) — Boltz's
        parse_pdb (gemmi) raises a swallowed KeyError on the entity/subchain mapping for some
        PDB-format files (ligand/entity-bearing, e.g. 1G6P/1SRO) and produces no model; the mmCIF
        carries proper entity records and parses cleanly. Falls back to PDB only if the CIF is
        unavailable.

        Fail-closed: an entry that resolves to no readable file returns an error (the caller fails
        the fold) rather than silently folding unguided — the §0 silent-wrong guard."""
        if not templates:
            return None, None
        out: List[Dict[str, Any]] = []
        for t in templates:
            t2 = {k: v for k, v in t.items() if k != "pdb_id"}
            path = t2.get("cif") or t2.get("pdb")
            if not path and t.get("pdb_id"):
                cif = self._download_cif_by_id(str(t["pdb_id"]))   # mmCIF preferred (gemmi-safe)
                if cif:
                    t2.pop("pdb", None)
                    t2["cif"] = cif
                    path = cif
                else:
                    path = self._download_pdb_by_id(str(t["pdb_id"]))   # fallback
                    if not path:
                        return None, (f"Could not obtain template '{t['pdb_id']}' for the guided "
                                      f"fold. Provide a valid 4-char PDB id or a local .cif/.pdb path.")
                    t2["pdb"] = path
            if not path or not Path(str(path)).is_file():
                return None, (f"Template structure '{path or t.get('pdb_id')}' is not a readable "
                              f"local file — refusing the guided fold (would silently fold unguided).")
            out.append(t2)
        return out, None

    def _read_selected_residues(self, model_id: str, chain_id: str) -> List[int]:
        """
        Read the LIVE ChimeraX selection and return the selected residue numbers
        on *chain_id* of *model_id*.

        Uses ``info residues sel & #<model>/<chain>`` whose output lines look like
        ``residue id /A:8 name GLN index 7`` — we extract the PDB residue numbers.
        Returns [] on any failure (caller decides the fallback).
        """
        try:
            cmd = f"info residues sel & #{model_id}/{chain_id}"
            res = self.bridge.run_command(cmd)
            text = (res.get("value") or "") if isinstance(res, dict) else ""
            nums = [
                int(m.group(1))
                for m in re.finditer(rf"/{re.escape(chain_id)}:(-?\d+)\b", text)
            ]
            return sorted(set(nums))
        except Exception:
            return []

    def _run_design_goal(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Design-intent op-class handler (the SINGLE source of truth for goal→profile).

        Resolution: alias (route) → LLM classifier → MISS. On MISS — including a
        non-solubility goal the classifier must NOT force-fit (the over-attraction
        guard) — hands back to the plain `_run_proteinmpnn` path; never errors,
        never widens. On a resolved goal: realises the profile (solvent-exposed
        designable set, soluble bias, Cys omitted), runs ProteinMPNN with FULLY
        RESOLVED inputs (`_resolved_profile` makes the whole-chain path unreachable),
        then RANKS by CamSol (cross-sequence scalar) + an ESMFold fold-check —
        never MPNN log-likelihood.
        """
        from intent_registry import (
            DESIGN_GOAL_REGISTRY, DESIGN_PROFILES, make_llm_classify_fn,
            _DESIGN_TASK_BLOCK,
        )
        from camsol_bridge import camsol_solubility_score
        import config as _cfg

        user_input = user_input or inputs.get("_user_input", "")
        intent_key = inputs.get("intent_key")        # alias-resolved in route(), or None
        model_id   = str(inputs.get("model_id") or self._first_model_id())
        chain_id   = inputs.get("chain") or inputs.get("chain_id") or "A"
        resolution = "alias"

        # Tier (b): LLM classifier when the alias missed. The over-attraction guard
        # is HERE — a different goal ("thermostable") must come back None, not be
        # rubber-stamped to the only offered label.
        if intent_key is None:
            global _design_classify_fn
            if _design_classify_fn is None:
                _design_classify_fn = make_llm_classify_fn(
                    registry=DESIGN_GOAL_REGISTRY, task_block=_DESIGN_TASK_BLOCK)
            classify = _design_classify_fn
            intent_key, resolution = DESIGN_GOAL_REGISTRY.resolve(
                self._strip_target_for_alias(user_input),
                llm_classify_fn=lambda t, ls: classify(t, ls))

        # Tier (c): MISS → hand back to the plain redesign path (never widen/error).
        profile = DESIGN_PROFILES.get(intent_key) if intent_key else None
        if profile is None:
            return self._run_proteinmpnn({
                "model_id": model_id, "chain_id": chain_id,
                "_user_input": user_input,
            })

        pdb_path = self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="design_goal", success=False,
                error=("Design-goal redesign needs a local PDB file and none could "
                       "be resolved for the loaded model."))

        # ── Designable set per the profile ─────────────────────────────────────
        design_positions: Optional[List[int]] = None
        if profile.designable == "solvent_exposed":
            cav = self._get_cavity_bridge()
            exposed = cav.solvent_exposed_residues(
                pdb_path, chain_id,
                sasa_threshold=_cfg.DESIGN_EXPOSED_SASA_THRESHOLD)
            if not exposed:
                return ToolStepResult(
                    tool="design_goal", success=False,
                    error=(f"Could not determine solvent-exposed positions for "
                           f"chain {chain_id} (SASA unavailable). Refusing rather "
                           f"than redesigning the whole chain."))
            design_positions = exposed

        # ── Fully-resolved ProteinMPNN inputs (profile → params) ───────────────
        mpnn_inputs: Dict[str, Any] = {
            "model_id":          model_id,
            "chain_id":          chain_id,
            "pdb_path":          pdb_path,
            "design_positions":  design_positions,
            "_resolved_profile": intent_key,
        }
        if profile.bias == "soluble":
            mpnn_inputs["bias_toward"] = "soluble"   # → _HYDROPHILIC_AAS bias
        if profile.omit:
            mpnn_inputs["exclude_amino_acids"] = list(profile.omit)

        result = self._run_proteinmpnn(mpnn_inputs)
        if not result.success:
            return result

        # ── Ranking: CamSol scalar (comparable) + ESMFold fold-check ───────────
        data    = result.data or {}
        designs = data.get("sequences") or []
        wt_seq  = data.get("wildtype_sequence") or ""
        wt_camsol = camsol_solubility_score(wt_seq) if wt_seq else None

        for d in designs:
            seq = d.get("sequence", "")
            d["camsol"] = camsol_solubility_score(seq) if seq else None
            d["camsol_gain"] = (
                None if (d["camsol"] is None or wt_camsol is None)
                else round(d["camsol"] - wt_camsol, 3))
        designs.sort(
            key=lambda d: d["camsol"] if d.get("camsol") is not None else float("-inf"),
            reverse=True)

        # ESMFold fold-guard on the top-K only (GPU/venv312 cost); graceful if down.
        folded = 0
        for d in designs[: int(_cfg.DESIGN_FOLD_CHECK_TOP_K)]:
            try:
                pred = self._get_esmfold_bridge().predict(d.get("sequence", ""))
                mp = pred.get("mean_plddt") if isinstance(pred, dict) else None
            except Exception:
                mp = None
            d["mean_plddt"] = mp
            if mp is not None:
                folded += 1
                if mp < float(_cfg.DESIGN_FOLD_PLDDT_FLOOR):
                    d["fold_flag"] = (
                        f"low pLDDT {mp:.0f} (<{_cfg.DESIGN_FOLD_PLDDT_FLOOR:.0f}) "
                        f"— possible misfolder")
        fold_note = "" if folded else " (ESMFold unavailable — ranked by CamSol only)"

        data["sequences"]     = designs
        data["ranking"]       = "camsol+esmfold"
        data["wt_camsol"]     = wt_camsol
        data["design_profile"] = intent_key

        # ── Transparency: state the profile applied ────────────────────────────
        n_exp    = len(design_positions or [])
        chain_len = len(wt_seq) if wt_seq else None
        n_fixed  = (chain_len - n_exp) if chain_len is not None else None
        top      = designs[0] if designs else None
        lines = [
            f"[Design profile: solubility — {resolution}-resolved]",
            (f"Designed {n_exp} solvent-exposed position(s)"
             + (f", held {n_fixed} buried/core fixed" if n_fixed is not None else "")
             + f" on chain {chain_id}; soluble bias on, Cys omitted."),
            (f"Ranked by CamSol{' + ESMFold' if folded else ''}{fold_note}."
             + (f" WT CamSol baseline = {wt_camsol:+.2f}." if wt_camsol is not None else "")),
        ]
        if top is not None and top.get("camsol") is not None:
            lines.append(
                f"Top design CamSol = {top['camsol']:+.2f}"
                + (f" (gain {top['camsol_gain']:+.2f} vs WT)"
                   if top.get("camsol_gain") is not None else "")
                + (f", mean pLDDT {top['mean_plddt']:.0f}"
                   if top.get("mean_plddt") is not None else ""))

        return ToolStepResult(
            tool             = "design_goal",
            success          = True,
            data             = data,
            viz_commands     = result.viz_commands,
            viz_explanations = result.viz_explanations,
            summary          = "\n".join(lines) + "\n\n" + (result.summary or ""),
        )

    def _run_proteinmpnn(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """
        Run ProteinMPNN fixed-backbone sequence redesign.

        Consumes the STRUCTURED constraint fields the translator emits
        (``exclude_amino_acids``, ``bias_amino_acids``, ``design_mode``,
        ``use_selection``) — no natural-language re-parsing — and resolves the
        designable set from the LIVE ChimeraX selection when the request is
        selection-scoped, falling back to a direct interface computation, and
        NEVER silently to a whole-chain redesign.
        """
        bridge   = self._get_proteinmpnn_bridge()
        model_id = inputs.get("model_id") or self._first_model_id()
        chain_id = inputs.get("chain_id") or inputs.get("chain", "A")

        # ── Resolve PDB path ──────────────────────────────────────────────────
        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool    = "proteinmpnn",
                success = False,
                error   = (
                    "ProteinMPNN requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        # ── Structured constraint fields (straight from the translator) ────────
        # The model is not perfectly consistent about field names, so accept the
        # observed synonyms rather than re-parsing the NL text.
        exclude_aas = (inputs.get("exclude_amino_acids")
                       or inputs.get("omit_amino_acids") or [])
        omit_aas = "".join(sorted({
            ch.upper() for aa in exclude_aas for ch in str(aa) if ch.isalpha()
        }))

        bias_aas = [str(a).upper() for a in (inputs.get("bias_amino_acids") or [])
                    if str(a).strip()]
        # A "more soluble" / "hydrophilic" / "reduce aggregation" objective maps to a
        # SOFT positive bias toward the polar/charged set. The model signals it either
        # structurally (bias_amino_acids, handled above) or in free text via
        # bias_toward/design_scope — accept the solubility synonyms here so the
        # objective is actually HONOURED (not silently dropped). Only "hydrophil"
        # was matched before, so bias_toward:"soluble" — the corpus's own wording —
        # fell through.
        _bias_hint = (str(inputs.get("bias_toward") or "") + " "
                      + str(inputs.get("design_scope") or "")).lower()
        if not bias_aas and any(tok in _bias_hint for tok in
                                ("hydrophil", "solub", "soluble", "polar", "aggregation")):
            bias_aas = list(_HYDROPHILIC_AAS)   # D E N Q H K R S T

        # design_scope is the canonical key (see translator prompt); design_mode
        # and the boolean flags below are tolerated synonyms.
        design_mode = (str(inputs.get("design_scope") or "")
                       + " " + str(inputs.get("design_mode") or "")).lower()
        partner_chain = (inputs.get("partner_chain")
                         or inputs.get("interface_partner_chain"))
        design_positions = inputs.get("design_positions")
        # A SCOPED (selection / interface-only) redesign — detected robustly,
        # since the model is inconsistent about exact field names.  Any boolean
        # "design only the selection/interface" flag, an interface/selection
        # design_mode, an explicit position list, or a named partner chain all
        # mean "do NOT redesign the whole chain".
        # `_resolved_profile` (set by _run_design_goal) means a design-intent
        # op-class already resolved the FULL scope+params — treat as scoped so the
        # under-scoped whole-chain+bias path is STRUCTURALLY unreachable for a
        # resolved goal, keyed on PROFILE PRESENCE not design_positions content
        # (defense in depth; the goal handler already refuses on an empty exposed
        # set, so an empty set never reaches here).
        resolved_profile = bool(inputs.get("_resolved_profile"))
        scoped = bool(
            resolved_profile
            or design_positions
            or partner_chain
            or inputs.get("use_selection")
            or inputs.get("design_only_interface")
            or inputs.get("redesign_selected")
            or inputs.get("selected_only")
            or "interface" in design_mode
            or "select" in design_mode
        )

        # ── Designable set: explicit > live selection > interface > (error) ────
        interface_design = bool(inputs.get("interface_design"))
        if scoped and not design_positions:
            # The user's live ChimeraX selection is the ground truth for "the
            # selected residues"; if none was captured, compute the interface
            # directly from coordinates (deterministic, never whole-chain).
            design_positions = self._read_selected_residues(model_id, chain_id)
            if not design_positions:
                interface_design = True

        # ── Legacy "protect cached interface residues" path ────────────────────
        # Only for an UNRESTRICTED design (no selection / interface / explicit set);
        # restricted paths fix the complement themselves.
        fixed_positions: List[int] = inputs.get("fixed_positions") or []
        if (not fixed_positions and not scoped
                and not design_positions and not interface_design):
            interface_data = self.session.get_interface_residues(model_id)
            if interface_data:
                fixed_positions = (
                    self.session.get_protected_residues_for_chain(model_id, chain_id)
                    or []
                )

        full_inputs: Dict[str, Any] = {
            "model_id":         model_id,
            "pdb_path":         pdb_path,
            "chain_id":         chain_id,
            "fixed_positions":  fixed_positions,
            "num_sequences":    int(inputs.get("num_sequences", 8)),
            "temperature":      float(inputs.get("temperature", 0.1)),
            # Design-scope + AA constraints (structured fields, consumed directly)
            "design_positions": design_positions,
            "interface_design": interface_design,
            "partner_chain":    partner_chain,
            "omit_aas":         omit_aas,
            "bias_aas":         bias_aas,
        }
        return bridge.analyze(full_inputs, session=self.session)

    def _run_mpnn_esmfold(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run ProteinMPNN + ESMFold combined redesign and validation pipeline."""
        import time as _time
        import config as _cfg

        pipeline  = self._get_mpnn_esmfold_pipeline()
        model_id  = inputs.get("model_id") or self._first_model_id()
        chain_id  = inputs.get("chain_id") or inputs.get("chain", "A")
        top_n     = int(inputs.get("top_n", _cfg.MPNN_ESMFOLD_TOP_N))
        include_wt = bool(inputs.get("include_wildtype", _cfg.MPNN_ESMFOLD_INCLUDE_WT))
        plddt_threshold = float(inputs.get("plddt_threshold", 70.0))

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)

        t0 = _time.perf_counter()
        try:
            result = pipeline.run(
                model_id         = model_id,
                pdb_path         = pdb_path,
                chain_id         = chain_id,
                session          = self.session,
                top_n            = top_n,
                plddt_threshold  = plddt_threshold,
                include_wildtype = include_wt,
            )
        except Exception as exc:
            traceback.print_exc()
            return ToolStepResult(
                tool    = "mpnn_esmfold",
                success = False,
                error   = f"MPNN+ESMFold pipeline failed: {exc}",
            )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool       = "mpnn_esmfold",
                success    = False,
                error      = result.get("error", "Pipeline failed"),
                elapsed_ms = elapsed_ms,
            )

        # Store in session (pdb_str is stripped automatically on save)
        self.session.set_mpnn_esmfold_results(model_id, result)

        cx_cmds = result.get("chimerax_commands", [])
        cx_exps = result.get("chimerax_explanations", [])

        return ToolStepResult(
            tool             = "mpnn_esmfold",
            success          = True,
            data             = {
                k: v for k, v in result.items()
                if k not in ("chimerax_commands", "chimerax_explanations", "summary")
            },
            viz_commands     = cx_cmds,
            viz_explanations = cx_exps,
            summary          = result.get("summary", "MPNN+ESMFold validation complete."),
            elapsed_ms       = elapsed_ms,
        )

    def _run_rfdiffusion(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge = self._get_rfdiffusion_bridge()
        return bridge.analyze(inputs, session=self.session)

    def _run_rosetta(self, inputs: Dict[str, Any]) -> ToolStepResult:
        bridge    = self._get_rosetta_bridge()
        model_id  = inputs.get("model_id") or self._first_model_id()
        mutations = inputs.get("mutations", [])
        chain     = inputs.get("chain")

        if not mutations:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "No mutations supplied to rosetta tool.\n"
                    "  Provide tool_inputs: {\"rosetta\": {\"mutations\": "
                    "[{\"chain\": \"A\", \"position\": 82, \"from_aa\": \"V\", "
                    "\"to_aa\": \"A\"}]}}"
                ),
            )

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="rosetta", success=False,
                error=(
                    "Rosetta requires a local PDB file.\n"
                    "  Load the structure from a local .pdb file, or ensure\n"
                    "  internet access so StructureBot can download it from RCSB."
                ),
            )

        return bridge.analyze(
            pdb_path  = pdb_path,
            mutations = mutations,
            session   = self.session,
            model_id  = model_id,
            chain     = chain,
        )

    def _run_validate_ddg(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        High-accuracy ddG validation tier: multi-trajectory MEDIAN ddG + spread
        + confidence on a SMALL explicit set of mutations (named in the request)
        or the top candidates from existing scan_results. NOT a full re-scan.
        """
        import re as _re
        user_input = user_input or inputs.get("_user_input", "")
        model_id   = inputs.get("model_id") or self._primary_model_id()
        chain      = inputs.get("chain", "A")

        pdb_path = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="validate_ddg", success=False,
                error=(
                    "No structure loaded or PDB file unavailable. "
                    "Run 'open 1HSG' (or your structure) first."
                ),
            )

        # 1) Explicit mutations named in the request, e.g. "I72R", "V82A".
        mutations: List[Dict[str, Any]] = []
        seen: set = set()
        for m in _re.finditer(
            r"\b([ACDEFGHIKLMNPQRSTVWY])(\d{1,4})([ACDEFGHIKLMNPQRSTVWY])\b",
            user_input or "",
        ):
            frm, pos, to = m.group(1), int(m.group(2)), m.group(3)
            key = f"{frm}{pos}{to}"
            if key in seen:
                continue
            seen.add(key)
            mutations.append({"chain": chain, "position": pos,
                              "from_aa": frm, "to_aa": to})

        source_note = ""
        if mutations:
            source_note = f"{len(mutations)} mutation(s) named in request"
        else:
            # 2) Fall back to the top candidates from a prior scan.
            scan_data = self.session.get_scan_result(model_id)
            if not scan_data:
                return ToolStepResult(
                    tool="validate_ddg", success=False,
                    error=(
                        "No mutations named and no prior scan found. "
                        "Name the mutations (e.g. 'validate ddg for I72R and V82A'), "
                        "or run a mutation scan first then ask to validate the top hits."
                    ),
                )
            top = sorted(
                scan_data, key=lambda c: c.get("combined_score", 0.0), reverse=True
            )[:3]
            for c in top:
                mutations.append({
                    "chain":    c.get("chain", chain),
                    "position": c.get("position"),
                    "from_aa":  c.get("from_aa"),
                    "to_aa":    c.get("to_aa"),
                })
            source_note = f"top {len(mutations)} candidate(s) from prior scan"

        if not mutations:
            return ToolStepResult(
                tool="validate_ddg", success=False,
                error="Could not determine any mutations to validate.",
            )

        def _status(msg: str) -> None:
            print(msg, flush=True)

        bridge = self._get_rosetta_bridge()
        result = bridge.validate_ddg(
            pdb_path          = pdb_path,
            mutations         = mutations,
            session           = self.session,
            model_id          = model_id,
            chain             = chain,
            progress_callback = _status,
        )

        if not result.success:
            return ToolStepResult(
                tool="validate_ddg", success=False,
                error=result.error or "Validation-tier ddG failed.",
            )

        data    = result.data or {}
        scores  = data.get("ddg_scores", {})
        spreads = data.get("ddg_spread", {})
        confs   = data.get("ddg_confidence", {})
        srcs    = data.get("ddg_source", {})
        n_traj  = data.get("n_trajectories", "?")

        # Build a confidence table (multi-line → shown in a Rich Panel).
        lines: List[str] = [
            f"=== High-Accuracy ddG Validation ({n_traj} trajectories, "
            f"median + MAD spread) ===",
            f"Mutations: {source_note}",
            "",
            f"  {'Mutation':<9} {'ddG(med)':>9} {'spread':>7} {'confidence':<12} {'source'}",
            "  " + "-" * 56,
        ]
        for key in scores:
            ddg = scores.get(key)
            sp  = spreads.get(key)
            cf  = confs.get(key, "?")
            sr  = srcs.get(key, "?")
            ddg_s = f"{ddg:+.3f}" if ddg is not None else "  N/A"
            sp_s  = f"{sp:.2f}" if sp is not None else "  —"
            lines.append(f"  {key:<9} {ddg_s:>9} {sp_s:>7} {cf:<12} {sr}")
        lines.append("")
        for w in data.get("warnings", []) or []:
            lines.append(f"[!] {w}")
        summary = "\n".join(lines)

        return ToolStepResult(
            tool             = "validate_ddg",
            success          = True,
            data             = data,
            viz_commands     = result.viz_commands,
            viz_explanations = result.viz_explanations,
            summary          = summary,
        )

    def _cx_value(self, command: str) -> str:
        """run_command → its text 'value' (str), '' on anything unexpected. Never raises."""
        try:
            r = self.bridge.run_command(command)
        except Exception:
            return ""
        if isinstance(r, dict):
            v = r.get("value")
            return v if isinstance(v, str) else ""
        return r if isinstance(r, str) else ""

    def _normalize_assembly_to_flat_model(self, group_model_id: str, asm_name: str):
        """FIX A: `sym … copies true` makes submodel-per-copy with DUPLICATE chain ids
        (#N.1/A, #N.2/A, …), which (1) makes native `color bychain` paint every copy the same and
        (2) is INVISIBLE to StructureBot's ingestion (its parser only matches integer `#N/chain`
        addressing and keys by chain letter alone — so it'd see zero/one chain). Give every copy
        unique chain ids via `changechains`, then `combine … retainIds true` into ONE integer-id
        AtomicStructure that ingestion + native commands handle correctly.

        Returns ``(flat_model_id, final_chains, note)``: on success ``(id, [chains], None)``; when
        there's nothing to normalize (already flat) ``(None, None, None)``; on FAILURE
        ``(None, None, "<reason>")`` — the caller falls back to the submodel group AND surfaces the
        reason (never a SILENT fallback: a swallowed failure would look identical to the pre-Fix-A
        symptom, so the reason must reach the summary to be diagnosable). Best-effort: never raises
        (a failed normalization must not break assembly generation). The submodel group is hidden so
        only the flat assembly shows; the AU is handled by the caller.
        """
        try:
            submodel_chains = _parse_submodel_chains(
                self._cx_value(f"info residues #{group_model_id}"), group_model_id)
            if not submodel_chains:
                return None, None, None                 # not submodel-addressed → already flat (no-op)
            renames, final_chains = plan_assembly_chain_renames(submodel_chains)
            for sub, old, new in renames:               # only collisions are renamed (uniques kept)
                self.bridge.run_command(f"changechains #{sub}/{old} {new}")
            before = {int(i) for i in _INT_MODEL_RE.findall(self._cx_value("info models"))}
            self.bridge.run_command(f'combine #{group_model_id} name "{asm_name}" retainIds true')
            after = {int(i) for i in _INT_MODEL_RE.findall(self._cx_value("info models"))}
            new_ids = sorted(after - before)
            if not new_ids:
                return None, None, "combine produced no new model id"
            flat_id = str(new_ids[-1])
            self.bridge.run_command(f"hide #{group_model_id} models")   # show only the flat assembly
            return flat_id, final_chains, None
        except Exception as exc:
            return None, None, f"{type(exc).__name__}: {exc}"

    def _run_bio_assembly(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """Thin orchestrator: generate the biological assembly via ChimeraX `sym`.

        Emits ``sym #N assembly M copies true`` on the EXISTING loaded model —
        does NOT re-open (that was the duplicate-#2 bug).  Tracks the generated
        assembly model and the original AU model in session_state so downstream
        tools (interface detection, sequence viewers) can address the assembly.

        Error-first: if no model is loaded, or the assembly ID is invalid, a
        clean error is returned that lists available assemblies (from ``sym #N``).
        """
        import time as _time
        t0 = _time.perf_counter()
        user_input = user_input or inputs.get("_user_input", "")

        model_id    = str(inputs.get("model_id")
                          or self._visible_focus_model_id()
                          or self._primary_model_id())
        assembly_id = int(inputs.get("assembly_id") or 1)

        # Guard: must have a loaded model
        if not self.session.structures:
            return ToolStepResult(
                tool="bio_assembly", success=False,
                error=(
                    "No structure is loaded.  Open a structure first, then "
                    "ask to generate the biological assembly."
                ),
            )

        if self.bridge is None:
            return ToolStepResult(
                tool="bio_assembly", success=False,
                error="ChimeraX bridge unavailable.",
            )

        # Check what assemblies are available (sym #N with no further args) — RAW value (no
        # sentinel string, so the no-assembly detector below isn't fooled by the word "assemblies").
        def _list_assemblies_raw() -> str:
            try:
                r = self.bridge.run_command(f"sym #{model_id}")
                return (r.get("value") or "").strip()
            except Exception:
                return ""

        def _fail(reason: str) -> ToolStepResult:
            """Graceful failure: when the structure has NO deposited assembly, say so plainly (the
            AU is likely already the biological unit) rather than dumping a raw sym error."""
            listing = _list_assemblies_raw()
            has_any = bool(re.search(r"assembl|cop(y|ies)\s+of", listing, re.I))
            if not has_any:
                msg = (f"{reason}\n#{model_id} has no deposited biological assembly — the "
                       f"asymmetric unit is likely already the biological unit, so there is "
                       f"nothing to build.")
            else:
                msg = f"{reason}\nAvailable assemblies for #{model_id}: {listing or '(none listed)'}"
            return ToolStepResult(tool="bio_assembly", success=False, error=msg)

        # Generate the assembly
        sym_cmd = f"sym #{model_id} assembly {assembly_id} copies true"
        try:
            result = self.bridge.run_command(sym_cmd)
        except Exception as exc:
            return _fail(f"ChimeraX error running `{sym_cmd}`: {exc}")

        err = result.get("error") if isinstance(result, dict) else None
        val = (result.get("value") or "") if isinstance(result, dict) else str(result)

        if err:
            return _fail(f"sym assembly generation failed: {err}")

        # Parse the assembly model ID from the ChimeraX response.
        # Response format: "Made N copies for <name> assembly M"
        # The new group model is next after the current highest model number.
        assembly_model_id: Optional[str] = None
        try:
            models_r = self.bridge.run_command("info models")
            models_val = (models_r.get("value") or "") if isinstance(models_r, dict) else ""
            # Find model named "<name> assembly <id>"
            struct_info = self.session.get_structure(model_id)
            struct_name = (struct_info.get("name") or "").lower() if struct_info else ""
            for line in models_val.splitlines():
                if f"assembly {assembly_id}" in line.lower():
                    m = re.search(r"model id #(\d+)\b", line)
                    if m:
                        assembly_model_id = m.group(1)
                        break
            # Fallback: highest numeric model id in session
            if assembly_model_id is None:
                ids = re.findall(r"model id #(\d+)\b", models_val)
                if ids:
                    assembly_model_id = str(max(int(i) for i in ids))
        except Exception:
            pass

        # FIX A: normalize the submodel-per-copy assembly (duplicate chain ids) into ONE flat
        # integer-id model with UNIQUE chain ids — so native `color bychain` distinguishes copies
        # AND StructureBot's ingestion (which only parses integer `#N/chain` addressing) enumerates
        # every copy as its own chain. Best-effort: falls back to the submodel group on any failure.
        group_model_id = assembly_model_id
        flat_model_id, final_chains, normalize_note = (None, None, None)
        if assembly_model_id:
            struct_info0 = self.session.get_structure(model_id)
            base_name = ((struct_info0.get("name") if struct_info0 else None) or "assembly")
            flat_model_id, final_chains, normalize_note = self._normalize_assembly_to_flat_model(
                assembly_model_id, f"{base_name} assembly {assembly_id}")
        # downstream (ingestion, interface scan) addresses the FLAT model when normalization ran
        if flat_model_id:
            assembly_model_id = flat_model_id

        # Fetch assembly info for the pdb_id (may already be cached)
        pdb_id = None
        n_subunits = None
        asm_type = None
        try:
            struct_info = self.session.get_structure(model_id)
            name = (struct_info.get("name") or "") if struct_info else ""
            if re.match(r"^[A-Za-z0-9]{4}$", name):
                pdb_id = name.upper()
                cached = self.session.get_assembly_info(pdb_id)
                asm_info = cached
                if not asm_info:
                    from assembly_analyser import fetch_assembly_info
                    asm_info = fetch_assembly_info(pdb_id)
                    if asm_info and not asm_info.get("error"):
                        self.session.set_assembly_info(pdb_id, asm_info)
                if asm_info and not asm_info.get("error"):
                    n_subunits = asm_info.get("n_subunits")
                    asm_type   = asm_info.get("assembly_type")
        except Exception:
            pass

        # Track in session_state. `assembly_model_id` is the model downstream tools address — the
        # FLAT normalized model when Fix A ran, else the raw submodel group. `group_model_id` keeps
        # the sym submodel group for reference; `normalized` records whether unique chain ids were
        # assigned (so a homo-oligomer's copies ingest as distinct member chains).
        gen_record: Dict[str, Any] = {
            "au_model_id":       model_id,
            "assembly_model_id": assembly_model_id,
            "group_model_id":    group_model_id,
            "normalized":        bool(flat_model_id),
            "assembly_chains":   list(final_chains) if final_chains else None,
            "normalize_error":   normalize_note,        # non-None only when normalization was ATTEMPTED and FAILED
            "assembly_id":       assembly_id,
            "assembly_type":     asm_type,
            "n_subunits":        n_subunits,
            "pdb_id":            pdb_id,
        }
        self.session.set_generated_assembly(model_id, gen_record)

        # Build summary
        asm_label = asm_type or f"assembly {assembly_id}"
        if final_chains:
            copies_note = f"{len(final_chains)} unique chains ({', '.join(final_chains[:8])}" \
                          + ("…" if len(final_chains) > 8 else "") + ")"
        else:
            copies_note = f"{n_subunits} chains" if n_subunits else "multiple chains"
        summary = (
            f"Generated {asm_label} from AU #{model_id} "
            + (f"→ assembly model #{assembly_model_id}" if assembly_model_id else "")
            + (" (flat, unique chain ids)" if flat_model_id else "")
            + f" ({copies_note}); AU #{model_id} kept and hidden"
        )
        # NEVER a silent fallback: if normalization was attempted but FAILED, the copies keep
        # duplicate chain ids (the pre-Fix-A symptom). Surface the reason so it's diagnosable rather
        # than looking identical to an un-fixed build.
        if normalize_note:
            summary += (f"  ⚠ chain-id normalization failed ({normalize_note}) — copies keep "
                        f"DUPLICATE ids on submodel group #{group_model_id}; color-by-chain and "
                        f"per-copy ops will NOT distinguish them.")

        # VALIDATE a user-ASSERTED oligomer against the deposited assembly (the oligomer word is
        # optional; when present it's an assertion to check, not a build directive). We built the
        # DEPOSITED assembly — if the user named a different stoichiometry, warn (never silently the
        # wrong thing, never force the user's word over the file).
        requested = self._parse_requested_oligomer_count(user_input)
        actual = len(final_chains) if final_chains else n_subunits
        if requested is not None and actual is not None and requested != actual:
            gen_record["oligomer_mismatch"] = {"requested": requested, "actual": actual}
            summary += (f"  ⚠ you asked for a {requested}-mer but {pdb_id or 'this structure'}'s "
                        f"deposited assembly {assembly_id} is a {actual}-mer — built the DEPOSITED "
                        f"assembly. To build a different one, name its assembly id (e.g. 'assembly 2').")

        elapsed_ms = (_time.perf_counter() - t0) * 1000
        return ToolStepResult(
            tool         = "bio_assembly",
            success      = True,
            data         = gen_record,
            viz_commands = ["view"],
            viz_explanations = ["Fit the full assembly in view"],
            summary      = summary,
            elapsed_ms   = elapsed_ms,
        )

    def _run_interface_stabilization(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Orchestrate inter-subunit interface detection + disulfide scan.

        Automatically uses the assembly model (from generated_assemblies) if
        one exists; falls back to the primary AU model.  Error-first: explicit
        guard on PDB availability before any heavy computation.
        """
        import time as _time
        t0 = _time.perf_counter()
        user_input = user_input or inputs.get("_user_input", "")

        if self.bridge is None:
            return ToolStepResult(
                tool="interface_stabilization", success=False,
                error="ChimeraX bridge unavailable.",
            )
        if not self.session.structures:
            return ToolStepResult(
                tool="interface_stabilization", success=False,
                error=(
                    "No structure is loaded.  Open a structure first, then ask "
                    "to stabilize the interface."
                ),
            )

        model_id = str(inputs.get("model_id") or self._primary_assembly_model_id())

        # Resolve PDB path from the AU model (assembly model has no local PDB)
        au_model_id = model_id
        for mid, rec in self.session.generated_assemblies.items():
            asm_mid = str(rec.get("assembly_model_id") or "")
            if asm_mid == model_id:
                au_model_id = str(mid)
                break

        pdb_path = self._ensure_pdb_file(au_model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="interface_stabilization", success=False,
                error=(
                    "Interface stabilization requires a local PDB file for the "
                    "disulfide geometry scan.\n"
                    "  Load the structure from a local .pdb file, or ensure "
                    "internet access so StructureBot can download it from RCSB."
                ),
            )

        # Resolve PDB ID for informational output
        pdb_id: Optional[str] = None
        struct_info = self.session.get_structure(au_model_id)
        if struct_info:
            name = struct_info.get("name", "")
            if re.match(r"^[A-Za-z0-9]{4}$", name):
                pdb_id = name.upper()

        from interface_stabilization import InterfaceStabilization
        stab = InterfaceStabilization(bridge=self.bridge, session=self.session)

        result = stab.analyze(
            model_id  = model_id,
            pdb_path  = pdb_path,
            pdb_id    = pdb_id,
        )
        if result.elapsed_ms is None:
            result.elapsed_ms = (_time.perf_counter() - t0) * 1000
        return result

    def _run_assembly_analyser(self, inputs: Dict[str, Any]) -> ToolStepResult:
        """Run biological assembly analysis (MONOMER or MULTIMER mode)."""
        import time as _time
        t0 = _time.perf_counter()

        model_id = inputs.get("model_id") or self._first_model_id()

        # Guard: when multiple models are loaded (e.g. original structure #1 +
        # ESMFold prediction #2), the translator may target #2 because it is
        # the most recently opened / currently active model.  Interface analysis
        # must always run on the original crystal structure (#1).
        _n_models = len(self.session.structures)
        if _n_models > 1 and model_id != "1" and "1" in self.session.structures:
            print(
                f"  [WARN] Multiple models loaded -- interface analysis targeting #1 "
                f"(original structure), not #{model_id}",
                flush=True,
            )
            model_id = "1"

        mode     = inputs.get("mode", "multimer")
        chain_id = inputs.get("chain_id") or inputs.get("chain")
        visualize= inputs.get("visualize", False)
        dist     = float(inputs.get("contact_distance", 5.0))

        # Determine PDB ID for RCSB assembly query
        pdb_id: Optional[str] = None
        info = self.session.get_structure(model_id)
        if info:
            name = info.get("name", "")
            if re.match(r"^[A-Za-z0-9]{4}$", name):
                pdb_id = name.upper()

        analyser = self._get_assembly_analyser()
        result   = analyser.analyse(
            model_id          = model_id,
            pdb_id            = pdb_id,
            mode              = mode,
            chain_id          = chain_id,
            contact_distance  = dist,
        )

        elapsed_ms = (_time.perf_counter() - t0) * 1000

        # Build summary string
        asm_info   = result.get("assembly_info", {})
        asm_type   = asm_info.get("assembly_type", "unknown")
        n_ifaces   = len(result.get("interfaces", {}))
        n_excluded = result.get("excluded_count", 0)
        mode_str   = result.get("mode", mode)
        header     = result.get("header", "")

        summary_parts = [f"Assembly: {asm_type} [{mode_str} mode]"]
        if mode_str == "multimer":
            summary_parts.append(f"{n_ifaces} interface(s) detected")
            if n_excluded:
                ch_label = chain_id or "chain"
                summary_parts.append(
                    f"{n_excluded} positions excluded from scan (chain {ch_label} interface)"
                )
        summary = " — ".join(summary_parts)

        # Warnings
        warnings_out = result.get("warnings", [])

        # Visualization commands (if requested and interfaces detected)
        viz_cmds: List[str] = []
        viz_exps: List[str] = []
        if visualize and mode_str == "multimer" and result.get("interfaces"):
            viz_cmds, viz_exps = analyser.generate_interface_viz_commands(
                model_id   = model_id,
                interfaces = result["interfaces"],
            )

        return ToolStepResult(
            tool             = "assembly_analyser",
            success          = True,
            data             = {
                "mode":               mode_str,
                "assembly_type":      asm_type,
                "assembly_info":      asm_info,
                "interfaces":         {
                    f"{k[0]}:{k[1]}": v
                    for k, v in result.get("interfaces", {}).items()
                },
                "protected_residues": result.get("protected_residues", []),
                "excluded_count":     n_excluded,
                "header":             header,
                "interface_summary":  result.get("interface_summary", ""),
                "warnings":           warnings_out,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = summary,
            elapsed_ms       = elapsed_ms,
        )

    def _run_mutation_scan(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        model_id       = inputs.get("model_id") or self._first_model_id()
        chain          = inputs.get("chain", "A")
        focus          = inputs.get("focus", "solubility")
        analysis_mode  = inputs.get("analysis_mode", "monomer")
        sequence       = inputs.get("sequence") or self._fetch_sequence(model_id, chain)
        pdb_path       = inputs.get("pdb_path") or self._ensure_pdb_file(model_id)
        # Tiering (set by route()._apply_mutation_scan_tiering): assignable scope +
        # opt-in Rosetta.  Defaults: whole chain, FAST tier (no Rosetta).
        scan_positions = inputs.get("scan_positions")        # None | list (maybe [])
        run_rosetta    = bool(inputs.get("run_rosetta", False))
        shortlist_k    = inputs.get("rosetta_shortlist_k")   # None = full coverage
        ddg_basis      = inputs.get("ddg_basis") or "symmetric"
        # Workbench "Test stability": score a variant's EXACT mutations {resnum: to_aa}
        # (not candidate generation). Implies the scope, so an empty score set ≠ a
        # whole-chain fallback — it's caught by the scope guard below.
        score_mutations = inputs.get("score_mutations")      # None | {resnum: to_aa}
        if score_mutations:
            scan_positions = [int(r) for r in score_mutations]

        # A scope that was requested but resolved to nothing → ERROR, no full-chain
        # fallback (mirrors the ProteinMPNN restricted-design convention).
        if scan_positions is not None and len(scan_positions) == 0:
            return ToolStepResult(
                tool="mutation_scan", success=False,
                error=(
                    "The requested scan scope resolved to no residues — nothing is "
                    "selected in ChimeraX, or the named range is out of bounds. "
                    "Select residues (or name a valid range), then retry."
                ),
            )

        if not sequence:
            return ToolStepResult(
                tool="mutation_scan", success=False,
                error=(
                    "No amino-acid sequence available for mutation scan.\n"
                    "  Load a structure first, or pass a sequence explicitly."
                ),
            )
        # Only the DEEP (Rosetta) tier hard-requires a local PDB. The fast tier is
        # sequence-driven (CamSol + ESM) with structure-based voters (ThermoMPNN, RaSP)
        # that degrade to silence when no structure is present — so a de-novo construct
        # with no downloadable PDB still gets a fast stability result.
        if run_rosetta and not pdb_path:
            return ToolStepResult(
                tool="mutation_scan", success=False,
                error=(
                    "The deep (Rosetta) stability tier requires a local PDB file.\n"
                    "  StructureBot will attempt to download from RCSB if the\n"
                    "  structure has a 4-letter PDB ID and internet is available.\n"
                    "  For a de-novo construct, re-run the FAST tier (no Rosetta)."
                ),
            )

        # Build filters from inputs
        filters: Dict[str, Any] = {}
        for key in ("camsol_threshold", "esm_threshold", "max_candidates",
                    "candidates_per_pos", "binding_site_residues",
                    "w_ddg", "w_sol", "w_tol"):
            if key in inputs:
                filters[key] = inputs[key]
        filters["focus"] = focus

        # In multimer mode, pull interface residues from session state
        protected_residues: List[int] = []
        if analysis_mode == "multimer":
            protected_residues = self.session.get_protected_residues_for_chain(
                model_id, chain
            )
            if not protected_residues:
                # Interface data not yet computed — it should have been run
                # via assembly_analyser first, but we proceed without it
                pass

        from mutation_scanner import MutationScanner

        def _progress(msg: str) -> None:
            pass  # progress shown by main.py via status_callback

        scanner = MutationScanner(
            session           = self.session,
            model_id          = model_id,
            progress_callback = _progress,
        )

        import time as _time
        t0      = _time.perf_counter()
        results = scanner.scan(
            pdb_path           = pdb_path,
            chain_id           = chain,
            sequence           = sequence,
            filters            = filters,
            protected_residues = protected_residues,
            analysis_mode      = analysis_mode,
            include_positions  = scan_positions,
            run_rosetta        = run_rosetta,
            rosetta_shortlist_k = shortlist_k,
            ddg_basis          = ddg_basis,
            score_mutations    = score_mutations,
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        # ── Voter visibility (B2): an EXPECTED voter that silently produced nothing
        # is the insidious case ("thought I had 4 axes, got 3"). LOUD for
        # capability-passed-then-empty (the formerly-silent drop, with the reason);
        # a QUIET one-liner for capability-absent (axis count stays visible);
        # SILENT for a deliberately-disabled voter (no note recorded). Normal runs
        # add nothing.
        _vn        = getattr(scanner, "voter_notes", [])
        _loud      = [n for n in _vn if n.get("state") == "empty"]
        _quiet     = [n for n in _vn if n.get("state") == "unavailable"]
        _vlines: List[str] = []
        for _n in _loud:
            _r = _n.get("reason") or "produced no output"
            _vlines.append(
                f"⚠  {_n['voter']} was available but produced no scores — dropped "
                f"from the ensemble for this run (reason: {_r}). The ranking below "
                f"uses the remaining voters.")
        if _quiet:
            _names = ", ".join(_n["voter"] for _n in _quiet)
            _vlines.append(
                f"· {_names} not available this run — the ensemble used the "
                f"remaining axes.")
        _vheader = ("\n".join(_vlines) + "\n\n") if _vlines else ""

        if not results:
            excluded_note = ""
            if protected_residues:
                excluded_note = (
                    f" ({len(protected_residues)} positions excluded "
                    f"due to interface contacts)"
                )
            return ToolStepResult(
                tool      = "mutation_scan",
                success   = True,
                data      = {"candidates": [], "count": 0,
                             "excluded_count": len(protected_residues),
                             "voter_notes": _vn},
                summary   = (_vheader + f"Mutation scan complete — no candidates met "
                             f"the criteria.{excluded_note}"),
                elapsed_ms = elapsed_ms,
            )

        # Generate visualization for top 5
        viz_cmds, viz_exps = scanner.generate_chimerax_commands(results, top_n=5)

        top = results[0]
        excluded_note = ""
        if protected_residues:
            excluded_note = (
                f" [{len(protected_residues)} interface position(s) excluded]"
            )

        _top_ddg = top.get("ddg")
        _ddg_txt = (
            f"ddG={_top_ddg:+.3f} kcal/mol [{top.get('ddg_source', '?')}]"
            if _top_ddg is not None
            else "ddG=not computed (opt in with 'rosetta')"
        )
        _tier_txt = "deep" if run_rosetta else "fast (CamSol+ESM)"
        one_liner = (
            f"Mutation scan [{analysis_mode} mode, {_tier_txt} tier]: "
            f"{len(results)} candidate(s) found.{excluded_note} "
            f"Top: {top['from_aa']}{top['position']}{top['to_aa']} "
            f"(score={top['combined_score']:+.2f}, "
            f"{_ddg_txt}, "
            f"solubility delta={top['solubility_delta']:+.2f})"
        )

        detailed_summary = _vheader + scanner._generate_summary(results)

        return ToolStepResult(
            tool             = "mutation_scan",
            success          = True,
            data             = {
                "candidates":    results,
                "count":         len(results),
                "top":           top,
                "excluded_count": len(protected_residues),
                "analysis_mode": analysis_mode,
                "voter_notes":   _vn,
            },
            viz_commands     = viz_cmds,
            viz_explanations = viz_exps,
            summary          = detailed_summary,
            elapsed_ms       = elapsed_ms,
        )

    # ── Double mutant tool ────────────────────────────────────────────────────

    def _run_double_mutant(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Score double mutant pairs from existing scan results.

        Reads scan_results from session state, builds the mutations list,
        routes pairs via DynaMut2 prediction_mm (or PyRosetta for close pairs),
        and stores results in session_state.double_mutant_results.
        """
        import time as _time

        user_input = user_input or inputs.get("_user_input", "")
        model_id   = inputs.get("model_id") or self._primary_model_id()
        lower      = user_input.lower()

        # Step 1 — detect mode
        _epitope_kw = ("epitope", "binding", "interface", "preserve", "target")
        mode = "epitope" if any(kw in lower for kw in _epitope_kw) else "stability"
        print(f"[DoubleMutant] Mode: {mode}", flush=True)

        # Step 2 — prerequisites: PDB file
        pdb_path = self._ensure_pdb_file(model_id)
        if not pdb_path:
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=(
                    "No structure loaded or PDB file unavailable. "
                    "Run 'open 1HSG' (or your structure) first."
                ),
            )

        # Step 2 — prerequisites: scan results
        scan_data = self.session.get_scan_result(model_id)
        if not scan_data:
            # Diagnostic: show which session fields have data to aid debugging
            populated = [
                k for k in vars(self.session)
                if k not in ("session_start", "working_dir")
                and getattr(self.session, k, None)
            ]
            print(
                f"[DoubleMutant] Session state fields with data: {populated}",
                flush=True,
            )
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=(
                    "No single-point scan results found. Run a mutation scan first, e.g.\n"
                    "'suggest mutations to improve solubility of chain A',\n"
                    "then ask for double mutant combinations."
                ),
            )

        # Map scan result dicts → DoubleMutantBridge schema
        # mutation_scanner uses 'solubility_delta'; bridge expects 'camsol_delta'
        mutations: List[Dict[str, Any]] = []
        for m in scan_data:
            mutations.append({
                "chain":             m.get("chain", "A"),
                "position":          m.get("position"),
                "from_aa":           m.get("from_aa"),
                "to_aa":             m.get("to_aa"),
                "ddg":               m.get("ddg") or 0.0,   # fast-tier scan → None; treat as 0.0
                "camsol_delta":      m.get("solubility_delta") or m.get("camsol_delta", 0.0),
                "esm_tolerance":     m.get("esm_tolerance", 1.0),
                "interface_proximal": m.get("interface_proximal", False),
            })

        if len(mutations) < 2:
            return ToolStepResult(
                tool="double_mutant", success=False,
                error=(
                    "Need at least 2 candidate mutations to generate pairs. "
                    "Run a wider mutation scan first."
                ),
            )

        # Step 3 — detect run_pyrosetta flag
        _pr_kw = ("pyrosetta", "rosetta", "accurate", "high accuracy", "validate")
        run_pyrosetta = any(kw in lower for kw in _pr_kw)
        if run_pyrosetta:
            print(
                "⚠ PyRosetta validation requested for close pairs — this adds ~30 min "
                "per close pair. Close pairs only.",
                flush=True,
            )

        # Step 4 — gather session context
        iface_dict = self.session.get_interface_residues(model_id)
        iface_set: Optional[set] = None
        if iface_dict:
            iface_set = set()
            for resnos in iface_dict.values():
                iface_set.update(resnos)

        func_set = self.session.get_functional_residues()
        func_residues: Optional[set] = func_set if func_set else None

        import config as _cfg_dm
        bridge_inputs: Dict[str, Any] = {
            "pdb_path":            pdb_path,
            "mutations":           mutations,
            "mode":                mode,
            "interface_residues":  iface_set,
            "functional_residues": func_residues,
            "top_n":               _cfg_dm.DOUBLE_MUTANT_TOP_N,
            "run_pyrosetta":       run_pyrosetta,
        }

        bridge = self._get_double_mutant_bridge()
        t0 = _time.perf_counter()
        result = bridge.analyze(bridge_inputs, self.session)
        result.elapsed_ms = (_time.perf_counter() - t0) * 1000

        # Step 5 — handle result
        if not result.success:
            return result

        self.session.set_double_mutant_results(model_id, result.data)

        # Generate ChimeraX visualization
        viz_cmds, viz_exps = self._build_double_mutant_viz(
            result.data.get("top_pairs", []), model_id
        )
        result.viz_commands     = viz_cmds
        result.viz_explanations = viz_exps

        return result

    def _build_double_mutant_viz(
        self,
        top_pairs: List[Dict[str, Any]],
        model_id:  str,
    ) -> tuple:
        """Generate ChimeraX commands to visualize top double mutant pairs."""
        if not top_pairs:
            return [], []

        _CONF_COLORS = {
            "high":     "#6495ed",  # cornflower blue
            "moderate": "#ffd700",  # gold
            "low":      "#c0c0c0",  # light grey
        }

        cmds: List[str] = []
        exps: List[str] = []

        chain = top_pairs[0]["mutation_a"].get("chain", "A")
        cmds.append(f"cartoon #{model_id}")
        exps.append("Switch to cartoon for double mutant visualization")
        cmds.append(f"color #{model_id}/{chain} white")
        exps.append(f"Reset chain {chain} to white before pair coloring")

        for pair in top_pairs[:5]:
            m_a      = pair["mutation_a"]
            m_b      = pair["mutation_b"]
            chain_a  = m_a.get("chain", "A")
            chain_b  = m_b.get("chain", "A")
            pos_a    = m_a["position"]
            pos_b    = m_b["position"]
            conf     = pair.get("confidence", "low")
            color    = _CONF_COLORS.get(conf, "#c0c0c0")
            pair_key = pair.get("pair_key", f"pos{pos_a}+pos{pos_b}")
            epistasis = pair.get("epistasis")
            ep_str   = f" (e={epistasis:+.1f})" if epistasis is not None else ""
            label    = f"{pair_key}{ep_str}"

            spec_a = f"#{model_id}/{chain_a}:{pos_a}"
            spec_b = f"#{model_id}/{chain_b}:{pos_b}"

            cmds.append(f"color {spec_a} {color}")
            exps.append(f"{pair_key}: color residue {pos_a} by confidence ({conf})")
            cmds.append(f"color {spec_b} {color}")
            exps.append(f"{pair_key}: color residue {pos_b} by confidence ({conf})")

            cmds.append(f"show {spec_a} atoms")
            exps.append(f"{pair_key}: show residue {pos_a} as atoms")
            cmds.append(f"style {spec_a} sphere")
            exps.append(f"{pair_key}: sphere style for residue {pos_a}")
            cmds.append(f"show {spec_b} atoms")
            exps.append(f"{pair_key}: show residue {pos_b} as atoms")
            cmds.append(f"style {spec_b} sphere")
            exps.append(f"{pair_key}: sphere style for residue {pos_b}")

            cmds.append(f"distance {spec_a}@CA {spec_b}@CA")
            exps.append(f"Draw Ca-Ca distance line for {pair_key}")

            cmds.append(f'label {spec_a} text "{label}" size 12 color white')
            exps.append(f"Label {pair_key} at residue {pos_a} with pair key and epistasis")

        cmds.append(f"view #{model_id}")
        exps.append("Fit structure in view to show all labeled pairs")

        return cmds, exps

    # ── ColabFold intent detection + option parsing ────────────────────────────

    def _detect_colabfold_intent(self, text: str) -> bool:
        """
        True only for an EXPLICIT high-accuracy folding request:
          * 'colabfold' / 'alphafold' / 'af2' anywhere, OR
          * 'fold … as a <oligomer>' (oligomer word present with a fold verb), OR
          * 'use PDB XXXX as template to fold …' (template + fold).
        Intentionally tight so generic 'fold sequence' / 'fold design' stays with
        the ESMFold / MPNN+ESMFold pipeline.
        """
        if not text:
            return False
        low = text.lower()
        if any(kw in low for kw in self._COLABFOLD_KEYWORDS):
            return True
        _has_fold = "fold" in low or "predict" in low or "structure" in low
        if _has_fold and any(w in low for w in self._OLIGOMER_COPIES):
            return True
        if "template" in low and "fold" in low:
            return True
        return False

    def _parse_colabfold_options(self, text: str) -> Dict[str, Any]:
        """
        Parse copies (oligomer word), an optional template PDB id, an optional
        explicit sequence, and a 'quick' preset flag from the user text.
        """
        opts: Dict[str, Any] = {}
        if not text:
            return opts
        low = text.lower()

        # copies ← first oligomer word found
        for word, n in self._OLIGOMER_COPIES.items():
            if re.search(rf"\b{word}\b", low):
                opts["copies"] = n
                break

        # template ← "use PDB XXXX as template" / "template XXXX" (4-char id)
        m = re.search(
            r"(?:pdb\s+)?\b([0-9][a-z0-9]{3})\b\s+as\s+(?:a\s+)?template", low
        ) or re.search(r"template[:\s]+(?:pdb\s+)?\b([0-9][a-z0-9]{3})\b", low)
        if m:
            opts["template"] = m.group(1).upper()

        # quick preset
        if "quick" in low or "fast fold" in low or "rough fold" in low:
            opts["quick"] = True

        # explicit sequence ← a long run of standard one-letter codes (>=20 aa)
        m = re.search(r"\b([ACDEFGHIKLMNPQRSTVWY]{20,})\b", text.upper())
        if m:
            opts["sequence"] = m.group(1)

        return opts

    # ── ColabFold dispatch ──────────────────────────────────────────────────────

    def _run_colabfold(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> "ToolStepResult":
        """
        Run AF2 structure prediction via the WSL2 ColabFold env and build viz.

        Resolves the sequence (explicit input/parsed sequence, else the loaded
        structure's chain — MPNN-result auto-pull is DEFERRED), folds via
        ColabFoldBridge, stores a trimmed result in session_state.colabfold_results,
        and builds ChimeraX confidence-map viz (open ranked PDB → native AlphaFold
        pLDDT palette → Sequence Viewer → optional matchmaker RMSD vs compare_to).
        Opens the PAE/pLDDT/coverage PNGs in the OS image viewer (best-effort).
        """
        import time as _time

        user_input   = user_input or inputs.get("_user_input", "")
        model_id     = inputs.get("model_id") or self._first_model_id()
        copies       = int(inputs.get("copies", 1) or 1)
        template     = inputs.get("template")
        quick        = bool(inputs.get("quick", False))
        allow_remote = bool(inputs.get("allow_remote"))
        chains       = inputs.get("chains")           # hetero Workbench construct: [{id,sequence}, …]

        # ── REMOTE-MSA CONSENT GATE (load-bearing) — BOTH entry points funnel here ──
        # No network call without surfaced consent. The Workbench engine-picker dialog and
        # the NL colabfold/alphafold banner each set allow_remote=True ONLY AFTER the user
        # accepts leaving LOCAL-ONLY; until then we return the consent prompt and NEVER touch
        # the bridge (which is itself fail-closed on the same flag — defense in depth).
        if not allow_remote:
            return ToolStepResult(
                tool="colabfold", success=False,
                error=_COLABFOLD_REMOTE_CONSENT_MSG,
                data={"remote_consent_required": True,
                      "consent_message":  _COLABFOLD_REMOTE_CONSENT_MSG,
                      "colabfold_inputs": dict(inputs)},   # re-dispatch payload once accepted
            )

        # ── Resolve sequence (homo/monomer path; skipped when `chains` is given) ────
        # Priority: explicit/pasted sequence → MPNN top design (auto-pull, when the
        # request refers to "the redesign"/"the top design"; RETRIEVED, never re-runs
        # MPNN) → the loaded structure's chain.
        sequence = inputs.get("sequence")
        mpnn_src = None
        if not chains:
            if not sequence and self._refers_to_mpnn_design(user_input):
                sequence, mpnn_src = self._mpnn_top_sequence(model_id)
            if not sequence:
                sequence = self._fetch_sequence(model_id, inputs.get("chain"))
            if not sequence:
                return ToolStepResult(
                    tool="colabfold", success=False,
                    error=(
                        "ColabFold needs an amino-acid sequence. Provide one explicitly "
                        "(e.g. 'fold MKT... as a dimer with colabfold'), redesign a chain "
                        "with ProteinMPNN first then 'fold the top design', or load a "
                        "structure so its chain sequence can be used."
                    ),
                )
            if mpnn_src:
                print(f"  ColabFold: folding the top ProteinMPNN design ({mpnn_src}, no re-run).",
                      flush=True)

        # ── Resolve optional template (PDB id → local file) ────────────────────────
        template_path: Optional[str] = None
        if template:
            if Path(str(template)).is_file():
                template_path = str(template)
            else:
                template_path = self._download_pdb_by_id(str(template))
                if not template_path:
                    return ToolStepResult(
                        tool="colabfold", success=False,
                        error=(
                            f"Could not obtain template '{template}'. Provide a 4-char "
                            "PDB id (downloadable from RCSB) or a local .pdb path."
                        ),
                    )

        bridge = self._get_colabfold_bridge()

        # num_models/num_recycle default to config inside the bridge when None;
        # quick=True overrides both to 1/1 there.
        t0 = _time.perf_counter()
        result = bridge.predict(
            sequence=sequence, copies=copies, chains=chains, template=template_path,
            num_models=None, num_recycle=None, quick=quick,
            label=f"model{model_id}", allow_remote=True,   # consent already established above
        )
        elapsed_ms = (_time.perf_counter() - t0) * 1000

        if not result.get("success"):
            return ToolStepResult(
                tool="colabfold", success=False,
                error=result.get("error", "ColabFold prediction failed"),
                data={"oom_risk": result.get("oom_risk", False)},
                elapsed_ms=elapsed_ms,
            )

        # ── Store a trimmed result in session (omit the heavy PAE matrix) ──────────
        trimmed = {
            k: result[k] for k in (
                "ranked_pdb", "mean_plddt", "ptm", "iptm", "length", "copies",
                "total_residues", "num_models", "num_recycle", "png_paths",
                "cached", "source",
            ) if k in result
        }
        self.session.set_colabfold_results(model_id, trimmed)

        # ── Open the confidence-map PNGs in the OS viewer (best-effort) ─────────────
        self._open_pngs(result.get("png_paths", {}))

        # ── Build ChimeraX viz (open ranked PDB → pLDDT palette → seq → matchmaker) ─
        viz_cmds, viz_exps, new_id = self._build_colabfold_viz(result, inputs, model_id)

        # ── Make the result a fold_summary-shaped step datum (engine-agnostic seam) ──
        # So the Workbench construct-fold path (`apply_construct_fold_result`/`apply_fold_result`)
        # consumes ColabFold exactly like Boltz/ESMFold — carrying the `remote_msa` provenance
        # all the way to the panel badge + export rows. `new_model_id` is the LIVE opened id.
        n_chains = int(result.get("n_chains", result.get("copies", 1)) or 1)
        ref_id   = None if inputs.get("no_reference") else inputs.get("compare_to")
        result["new_model_id"]       = new_id
        result["target"]             = "monomer" if n_chains <= 1 else "assembly"
        result["reference_model_id"] = str(ref_id) if ref_id is not None else None
        try:
            self.session.add_structure(
                new_id, f"colabfold_pred_{model_id}", path=result.get("ranked_pdb"),
                metadata={"predicted": True, "engine": "colabfold", "remote_msa": True})
        except Exception:
            pass

        mean_plddt = result["mean_plddt"]
        conf = "very high" if mean_plddt > 90 else "high" if mean_plddt > 70 else \
               "low" if mean_plddt > 50 else "very low"
        ptm = result.get("ptm")
        iptm = result.get("iptm")
        _shape = "monomer" if n_chains <= 1 else f"{n_chains}-mer"
        summary = (
            f"ColabFold ({_shape}, REMOTE MSA — left local-only): "
            f"mean pLDDT {mean_plddt:.1f} ({conf} confidence)"
            + (f", pTM {ptm:.2f}" if isinstance(ptm, (int, float)) else "")
            + (f", ipTM {iptm:.2f}" if isinstance(iptm, (int, float)) else "")
            + (" [cached]" if result.get("cached") else "")
            + f".\n  Ranked model: {result.get('ranked_pdb', '?')}"
        )
        return ToolStepResult(
            tool="colabfold", success=True,
            data=result,
            viz_commands=viz_cmds,
            viz_explanations=viz_exps,
            summary=summary,
            elapsed_ms=elapsed_ms,
        )

    def _build_colabfold_viz(
        self,
        result:   Dict[str, Any],
        inputs:   Dict[str, Any],
        model_id: str,
    ) -> tuple:
        """
        ChimeraX commands: open the ranked PDB as a NEW model, colour it by the
        native AlphaFold pLDDT palette (B-factor holds pLDDT), open the Sequence
        Viewer, and — when a compare_to structure resolves — superpose with
        matchmaker. The new model id is taken from session.next_model_id(), which
        is exactly what main.py assigns to the open command it later state-tracks.
        Returns (cmds, exps, new_id) — new_id threads into the result so the Workbench
        fold seam re-points to the LIVE opened model (engine-agnostic, like Boltz)."""
        ranked = result.get("ranked_pdb", "")
        if not ranked:
            return [], [], None
        pdb_posix = Path(ranked).as_posix()
        cmds, exps, new_id = self._fold_viz_commands(pdb_posix, inputs)
        return cmds, exps, new_id

    def _fold_viz_commands(
        self,
        pdb_posix: str,
        inputs:    Dict[str, Any],
    ) -> tuple:
        """ENGINE-AGNOSTIC fold viz: open a predicted-structure PDB as a NEW model,
        colour by the native AlphaFold pLDDT palette (B-factor holds pLDDT for both
        ColabFold AND ESMFold), open the Sequence Viewer, and — when a compare_to
        reference resolves — superpose with matchmaker. Returns (cmds, exps, new_id).
        The new id is session.next_model_id(), exactly what the spine assigns to the
        open command it state-tracks. Shared by _build_colabfold_viz and the workbench
        ESMFold fold so a later engine (Boltz) reuses one viz, not a parallel path."""
        new_id = str(self.session.next_model_id())

        cmds: List[str] = []
        exps: List[str] = []

        cmds.append(f'open "{pdb_posix}"')
        exps.append(f"Open the predicted model as #{new_id}")
        # Native AlphaFold pLDDT colouring (canonical blue→orange palette over the
        # B-factor column, where the engine stores per-residue pLDDT).
        cmds.append(f"color byattribute bfactor #{new_id} palette alphafold target acs")
        exps.append("Colour by pLDDT using the native AlphaFold palette (blue=confident)")
        cmds.append(f"cartoon #{new_id}")
        exps.append("Cartoon representation for the predicted model")
        # ChimeraX is structure-only — no Sequence Viewer (sequence lives in the
        # StructureBot window). Removed 2026-06-16.

        # ── Optional structural comparison (matchmaker RMSD) ────────────────────────
        ref_spec = self._resolve_colabfold_compare_to(inputs, exclude_model=new_id)
        if ref_spec:
            cmds.append(f"matchmaker #{new_id} to {ref_spec}")
            exps.append(
                f"Superpose the predicted model onto {ref_spec} (matchmaker RMSD "
                "reported in ChimeraX log)"
            )
        cmds.append(f"view #{new_id}")
        exps.append("Fit the predicted model in view")
        return cmds, exps, new_id

    def _open_and_viz_fold_live(self, pdb_posix: str, inputs: Dict[str, Any]) -> tuple:
        """Open a predicted-structure file LIVE and read the REAL assigned model id back from
        the `open` response — NOT a `next_model_id()` GUESS. The guess desyncs from ChimeraX's
        actual id (e.g. after the S4c deviation WT-reference folds consume ids), so a guessed
        id mis-targets colour/matchmaker/view AND poisons the stored `model_id` → the fold
        appears to do nothing and the active-row HIDE switch shows the wrong model (the 'V3'
        failure). Here we open, parse the real id, then colour-by-pLDDT + matchmaker + view
        against THAT id, all via self.bridge (so the spine's `on_structure_opened` still
        captures the open for tab focus). Returns (real_id, cmds_run, explanations). The
        caller returns viz_commands=[] — these already executed, and re-running the `open`
        would duplicate the model."""
        open_cmd = f'open "{pdb_posix}"'
        r = self.bridge.run_command(open_cmd)
        guess = str(self.session.next_model_id())
        real_id = (self._parse_model_spec(r, guess) or f"#{guess}").lstrip("#")
        cmds = [open_cmd]
        exps = [f"Open the predicted model as #{real_id} (id read back from the open response)"]
        # target atoms+cartoons+surfaces so the pLDDT colour survives a later
        # representation change (e.g. an NL "show as spheres" reveals coloured atoms,
        # not default-coloured ones).
        viz = [f"color byattribute bfactor #{real_id} palette alphafold target acs",
               f"cartoon #{real_id}"]
        vexp = ["Colour by pLDDT (AlphaFold palette)", "Cartoon representation"]
        ref_spec = self._resolve_colabfold_compare_to(inputs, exclude_model=real_id)
        if ref_spec:
            viz.append(f"matchmaker #{real_id} to {ref_spec}")
            vexp.append(f"Superpose onto {ref_spec} (matchmaker RMSD in the log)")
        viz.append(f"view #{real_id}")
        vexp.append("Fit the predicted model in view")
        for c in viz:
            self.bridge.run_command(c)
        return real_id, cmds + viz, exps + vexp

    def _resolve_colabfold_compare_to(
        self,
        inputs:        Dict[str, Any],
        exclude_model: str,
    ) -> Optional[str]:
        """
        Resolve the compare_to reference for matchmaker.

        Default: the currently-loaded primary model (if any), as a ``#id`` spec,
        with optional chain. Overridable via inputs['compare_to'] = a ``#id``/
        ``#id/chain`` spec or a model id. Returns None when nothing to compare to
        (e.g. the predicted model is the only structure).

        ``inputs['no_reference']`` (DE-NOVO constructs) FORCES None — the fold has no
        reference structure, so matchmaker is skipped explicitly and NEVER falls back to
        whatever primary happens to be loaded (which would silently superpose the de-novo
        fold onto an unrelated model).
        """
        if inputs.get("no_reference"):
            return None
        compare_to = inputs.get("compare_to")
        if compare_to:
            spec = str(compare_to).strip()
            return spec if spec.startswith("#") else f"#{spec}"
        # Default = the existing primary model, if one is loaded and it isn't the
        # model we just predicted.
        try:
            primary = self._primary_model_id()
        except Exception:
            primary = None
        if primary and str(primary) != str(exclude_model) and self.session.get_structure(primary):
            chain = inputs.get("chain")
            return f"#{primary}/{chain}" if chain else f"#{primary}"
        return None

    def _download_pdb_by_id(self, pdb_id: str) -> Optional[str]:
        """Download a 4-char PDB id from RCSB into cache/ and return the local path."""
        return self._download_rcsb(pdb_id, "pdb")

    def _download_cif_by_id(self, pdb_id: str) -> Optional[str]:
        """Download a 4-char PDB id from RCSB as mmCIF. PREFERRED for Boltz TEMPLATES: Boltz's
        parse_pdb (gemmi) raises a KeyError on the entity/subchain mapping for some PDB-format
        files (those with ligands/entities the PDB→gemmi path maps to a subchain with no entity,
        e.g. AXP in 1G6P/1SRO) and SWALLOWS it (exits 0, no model). The official mmCIF carries
        proper entity records and parses cleanly."""
        return self._download_rcsb(pdb_id, "cif")

    @staticmethod
    def _download_rcsb(pdb_id: str, fmt: str) -> Optional[str]:
        if not re.match(r"^[A-Za-z0-9]{4}$", pdb_id):
            return None
        cache_dir = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local = cache_dir / f"{pdb_id.upper()}.{fmt}"
        if local.is_file():
            return str(local)
        try:
            import requests
            resp = requests.get(
                f"https://files.rcsb.org/download/{pdb_id.upper()}.{fmt}", timeout=30
            )
            resp.raise_for_status()
            local.write_bytes(resp.content)
            return str(local)
        except Exception:
            return None

    @staticmethod
    def _open_pngs(png_paths: Dict[str, str]) -> None:
        """Open the confidence-map PNGs in the OS default image viewer (best-effort)."""
        import os as _os
        import sys as _sys
        import subprocess as _sub
        for _kind, _path in (png_paths or {}).items():
            if not _path or not Path(_path).is_file():
                continue
            try:
                if _sys.platform == "win32":
                    _os.startfile(_path)             # type: ignore[attr-defined]
                elif _sys.platform == "darwin":
                    _sub.Popen(["open", _path])
                else:
                    _sub.Popen(["xdg-open", _path])
            except Exception:
                pass

    # ══════════════════════════════════════════════════════════════════════════
    # Conformer-comparison (thin orchestrator — NOT a new bridge)
    # ══════════════════════════════════════════════════════════════════════════
    #
    # Reuses:  validate-design's _per_residue_ca_deviation Kabsch pattern
    #          (generalised to anchor-restricted fit via _anchor_kabsch)
    #          + _build_deviation_color_cmds grouped-color idiom
    #          + proteinmpnn_bridge.chain_resnum_to_seqpos for residue mapping
    #
    # STEP 0 probe findings (ChimeraX 1.11.1):
    #   • align #N/chain:range@CA to #M/chain:range@CA  CONFIRMED WORKS — moves
    #     model N; output "RMSD between K atom pairs is X.XXX angstroms"; requires
    #     equal atom counts (same residue numbers in both specs).
    #   • Cα extraction via runscript: r.find_atom('CA') returns None for residues
    #     without CA — requires if-guard.  JSON output via print(json.dumps(coords))
    #     works cleanly.  4AKE/1AKE: 214 residues each, all 1–214, no gaps.
    #   • The numpy Kabsch path is AUTHORITATIVE for numbers; `align` is for the
    #     visible overlay only.

    @staticmethod
    def _ca_coords_live(
        bridge: "ChimeraXBridge",
        model_id: str,
        chain: str,
    ) -> Dict[int, "Any"]:
        """
        ``{resno: np.array([x, y, z])}`` for Cα atoms of *chain* in the live
        ChimeraX model *model_id* (e.g. ``"4"``).  Reads from the CURRENT model
        state (post any prior operations).  Returns ``{}`` on error.

        Implementation note: writes Cα coordinates to a temp JSON file (the
        same write-then-read pattern as the ESMFold/PyRosetta workers), because
        large ``print()`` output from a ChimeraX runscript can be truncated by
        the REST API's response buffer.
        """
        import json
        import numpy as _np
        import tempfile
        import os as _os

        try:
            # Out-file receives the JSON — ChimeraX reads/writes, Python reads
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, encoding="utf-8", mode="w"
            ) as jf:
                out_path = jf.name

            # Use forward-slash path inside the script (safe on Windows)
            out_posix = out_path.replace("\\", "/")

            script = (
                "import json as _json\n"
                "from chimerax.atomic import AtomicStructure\n"
                f"_mid = {model_id!r}\n"
                f"_out = {out_posix!r}\n"
                "_models = {m.id_string: m for m in session.models "
                "if isinstance(m, AtomicStructure)}\n"
                "_m = _models.get(_mid)\n"
                "_c = {}\n"
                "if _m:\n"
                f"    for _ch in _m.chains:\n"
                f"        if _ch.chain_id != {chain!r}: continue\n"
                "        for _r in _ch.residues:\n"
                "            _ca = _r.find_atom('CA')\n"
                "            if _ca is not None:\n"
                "                _c[str(_r.number)] = "
                "[round(float(_x), 4) for _x in _ca.coord]\n"
                "with open(_out, 'w') as _fh:\n"
                "    _json.dump(_c, _fh)\n"
                "print('OK:' + str(len(_c)))\n"
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as sf:
                sf.write(script)
                script_path = sf.name

            try:
                result = bridge.run_command(f'runscript "{script_path}"')
            finally:
                try:
                    _os.unlink(script_path)
                except Exception:
                    pass

            val = (result.get("value") or "").strip()
            # Validate that the script ran and wrote the file
            if not val.startswith("OK:") or not _os.path.isfile(out_path):
                return {}
            with open(out_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            return {int(k): _np.array(v, dtype=float) for k, v in raw.items()}
        except Exception:
            return {}
        finally:
            try:
                _os.unlink(out_path)
            except Exception:
                pass

    @staticmethod
    def _ca_coords_live_multichain(
        bridge: "ChimeraXBridge",
        model_id: str,
    ) -> "Dict[Tuple[str,int], Any]":
        """
        ``{(chain_id, resno): np.array([x, y, z])}`` for ALL protein chains in
        *model_id*.  Used for multimer quaternary comparison so the Kabsch fit
        operates on the JOINT multi-chain Cα set rather than a single chain.
        Returns ``{}`` on error.
        """
        import json
        import numpy as _np
        import tempfile
        import os as _os

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".json", delete=False, encoding="utf-8", mode="w"
            ) as jf:
                out_path = jf.name

            out_posix = out_path.replace("\\", "/")

            script = (
                "import json as _json\n"
                "from chimerax.atomic import AtomicStructure\n"
                f"_mid = {model_id!r}\n"
                f"_out = {out_posix!r}\n"
                "_models = {m.id_string: m for m in session.models "
                "if isinstance(m, AtomicStructure)}\n"
                "_m = _models.get(_mid)\n"
                "_c = {}\n"
                "if _m:\n"
                "    for _ch in _m.chains:\n"
                "        _cid = _ch.chain_id\n"
                "        for _r in _ch.residues:\n"
                "            _ca = _r.find_atom('CA')\n"
                "            if _ca is not None:\n"
                "                _c[_cid + ':' + str(_r.number)] = "
                "[round(float(_x), 4) for _x in _ca.coord]\n"
                "with open(_out, 'w') as _fh:\n"
                "    _json.dump(_c, _fh)\n"
                "print('OK:' + str(len(_c)))\n"
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as sf:
                sf.write(script)
                script_path = sf.name

            try:
                result = bridge.run_command(f'runscript "{script_path}"')
            finally:
                try:
                    _os.unlink(script_path)
                except Exception:
                    pass

            val = (result.get("value") or "").strip()
            if not val.startswith("OK:") or not _os.path.isfile(out_path):
                return {}
            with open(out_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Keys are "CHAIN:RESNO" strings — convert to (chain, resno) tuples
            out = {}
            for k, v in raw.items():
                try:
                    ch, rn = k.split(":", 1)
                    out[(ch, int(rn))] = _np.array(v, dtype=float)
                except (ValueError, TypeError):
                    pass
            return out
        except Exception:
            return {}
        finally:
            try:
                _os.unlink(out_path)
            except Exception:
                pass

    @staticmethod
    def _anchor_kabsch(
        coords_a:       Dict[int, "Any"],
        coords_b:       Dict[int, "Any"],
        anchor_resnums: List[int],
    ) -> "Tuple[Optional[Dict[int,float]], Optional[float], Optional[float]]":
        """
        Anchor-restricted Kabsch superposition (generalised from
        ``_per_residue_ca_deviation``).

        1. Fits rotation R and translation t to align the *anchor* subset of B
           onto A (minimises anchor RMSD).
        2. Applies R, t to ALL common residues of B.
        3. Returns ``(per_resno_shift_Å, anchor_residual_rmsd, all_pairs_rmsd)``.

        ``anchor_residual_rmsd ≈ 0`` confirms the anchor is internally rigid
        and the shift map is trustworthy.  Returns ``(None, None, None)`` when
        the anchor or common-residue sets are too small (< 3).
        """
        import numpy as _np

        anc_set  = set(anchor_resnums)
        a_set    = set(coords_a)
        b_set    = set(coords_b)
        anc_com  = sorted(anc_set & a_set & b_set)
        all_com  = sorted(a_set & b_set)

        if len(anc_com) < 3 or len(all_com) < 3:
            return None, None, None

        P_anc = _np.array([coords_b[r] for r in anc_com])  # B anchor
        Q_anc = _np.array([coords_a[r] for r in anc_com])  # A anchor

        p_c = P_anc.mean(0)
        q_c = Q_anc.mean(0)
        Pc  = P_anc - p_c
        Qc  = Q_anc - q_c
        H   = Pc.T @ Qc
        U, _S, Vt = _np.linalg.svd(H)
        d   = _np.sign(_np.linalg.det(Vt.T @ U.T))
        R   = Vt.T @ _np.diag([1.0, 1.0, d]) @ U.T
        t   = q_c - R @ p_c

        per_shift: Dict[int, float] = {}
        all_diffs = []
        for rn in all_com:
            b_tr = R @ coords_b[rn] + t
            diff = coords_a[rn] - b_tr
            per_shift[rn] = round(float(_np.sqrt((diff ** 2).sum())), 3)
            all_diffs.append(diff)

        anc_diffs  = _np.array([coords_a[r] - (R @ coords_b[r] + t) for r in anc_com])
        anchor_rms = round(float(_np.sqrt((anc_diffs ** 2).sum(1).mean())), 3)
        all_arr    = _np.array(all_diffs)
        all_rms    = round(float(_np.sqrt((all_arr ** 2).sum(1).mean())), 3)

        return per_shift, anchor_rms, all_rms

    @classmethod
    def _auto_anchor_resnums(
        cls,
        coords_a: Dict["Any", "Any"],
        coords_b: Dict["Any", "Any"],
        common:   "List[Any]",
    ) -> "Tuple[Optional[List[Any]], str]":
        """Iterative-prune the rigid common-residue anchor: repeatedly drop the
        top-displaced residues (above the 40th-percentile shift) and re-fit until
        convergence, leaving the rigid core the Kabsch superposes on.  Converges on
        ONE rigid domain even for multimers whose chains are individually conserved
        but whose quaternary arrangement changes (e.g. haemoglobin T↔R).

        Key-type agnostic — *common* may be resnos (single-chain) or (chain, resno)
        tuples (multichain), exactly as ``_anchor_kabsch`` consumes.  Returns
        ``(anchor_keys, source_str)`` or ``(None, reason)`` when the fit fails.

        SHARED by conformer comparison and the S4c variant-vs-WT deviation so both
        localize divergence against the same rigid-core definition (one source)."""
        import numpy as _np
        anchor_resnums = list(common)
        n_iters = 0
        min_anchor = max(10, int(len(common) * 0.05))
        for _iter in range(12):
            shifts_g, _, _ = cls._anchor_kabsch(coords_a, coords_b, anchor_resnums)
            if shifts_g is None:
                return None, "Auto-anchor Kabsch fit failed (too few common residues)."
            cutoff = float(_np.percentile(sorted(shifts_g.values()), 40))
            new_anchor = [r for r in anchor_resnums if shifts_g[r] <= cutoff]
            n_iters += 1
            if len(new_anchor) < min_anchor or set(new_anchor) == set(anchor_resnums):
                break
            anchor_resnums = new_anchor
        return anchor_resnums, (
            f"auto-iterative ({n_iters} prune steps → {len(anchor_resnums)} residues)"
        )

    @staticmethod
    def _parse_anchor_spec(anchor_str: str, common_resnums: "set") -> List[int]:
        """
        Parse ``"1-29,124-214"`` or ``"1,2,3"`` into a sorted list of residue
        numbers that are also in *common_resnums*.  Returns ``[]`` on failure.
        """
        result: set = set()
        for part in anchor_str.split(","):
            part = part.strip()
            if "-" in part:
                try:
                    lo_s, hi_s = part.split("-", 1)
                    result.update(range(int(lo_s.strip()), int(hi_s.strip()) + 1))
                except ValueError:
                    pass
            else:
                try:
                    result.add(int(part))
                except ValueError:
                    pass
        return sorted(result & common_resnums)

    @staticmethod
    def _resnums_to_chimerax_range(resnums: List[int]) -> str:
        """
        Compress a sorted list of residue numbers to ChimeraX range notation.
        ``[1,2,3,5,6,10]`` → ``"1-3,5-6,10"``.
        """
        if not resnums:
            return ""
        runs: List[tuple] = []
        start = prev = resnums[0]
        for r in resnums[1:]:
            if r == prev + 1:
                prev = r
            else:
                runs.append((start, prev))
                start = prev = r
        runs.append((start, prev))
        return ",".join(str(s) if s == e else f"{s}-{e}" for s, e in runs)

    @classmethod
    def _conformer_shift_color_cmds(
        cls,
        per_shift:  Dict[int, float],
        model_spec: str,
        chain:      str,
    ) -> "Tuple[List[str], List[str]]":
        """
        Colour *model_spec* by per-residue Cα displacement using adaptive
        percentile-based buckets (blue=rigid → red=mobile).

        Uses the same grouped-run idiom as ``_build_deviation_color_cmds``
        but scales to the actual shift magnitude via percentiles, so the
        colour gradient is informative for any conformational change magnitude
        (small ≈1 Å or large ≈15+ Å like adenylate kinase).
        """
        if not per_shift:
            return [], []
        import numpy as _np

        vals = sorted(per_shift.values())
        p25, p50, p70, p85 = (
            float(_np.percentile(vals, 25)),
            float(_np.percentile(vals, 50)),
            float(_np.percentile(vals, 70)),
            float(_np.percentile(vals, 85)),
        )
        thresholds = [
            (p25, "blue"),
            (p50, "cornflower blue"),
            (p70, "white"),
            (p85, "orange"),
            (float("inf"), "red"),
        ]

        def _bucket(v: float) -> str:
            for hi, col in thresholds:
                if v <= hi:
                    return col
            return "red"

        base = f"{model_spec}/{chain}" if chain else model_spec
        cmds = [f"color {base} white"]
        exps = ["Reset to white before per-residue shift colouring"]

        runs: List[tuple] = []
        for rn in sorted(per_shift):
            col = _bucket(per_shift[rn])
            if runs and runs[-1][0] == col:
                runs[-1][1].append(rn)
            else:
                runs.append((col, [rn]))

        for col, resnos in runs:
            if col == "white":
                continue
            if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
                spec = f":{resnos[0]}-{resnos[-1]}"
            else:
                spec = ":" + ",".join(str(r) for r in resnos)
            cmds.append(f"color {model_spec}{spec} {col}")
            exps.append(f"Colour {spec} {col} (shift percentile bucket)")

        return cmds, exps

    @classmethod
    def _conformer_shift_color_cmds_mc(
        cls,
        per_shift:  "Dict[Tuple[str,int], float]",
        model_spec: str,
    ) -> "Tuple[List[str], List[str]]":
        """
        Multi-chain version of ``_conformer_shift_color_cmds``.
        *per_shift* is keyed by ``(chain_id, resno)`` tuples (multichain mode).
        Percentile buckets are computed globally across ALL chains so the colour
        scale is uniform and directly comparable between chains.
        """
        if not per_shift:
            return [], []
        import numpy as _np

        vals = sorted(per_shift.values())
        p25, p50, p70, p85 = (
            float(_np.percentile(vals, 25)),
            float(_np.percentile(vals, 50)),
            float(_np.percentile(vals, 70)),
            float(_np.percentile(vals, 85)),
        )
        thresholds = [
            (p25, "blue"),
            (p50, "cornflower blue"),
            (p70, "white"),
            (p85, "orange"),
            (float("inf"), "red"),
        ]

        def _bucket(v: float) -> str:
            for hi, col in thresholds:
                if v <= hi:
                    return col
            return "red"

        cmds = [f"color {model_spec} white"]
        exps = ["Reset to white before per-residue shift colouring"]

        # Group by chain → runs of consecutive residues with same colour
        chain_map: Dict[str, List[int]] = {}
        for (ch, rn) in sorted(per_shift):
            chain_map.setdefault(ch, []).append(rn)

        for chain in sorted(chain_map):
            resnos = sorted(chain_map[chain])
            runs: List[tuple] = []
            for rn in resnos:
                col = _bucket(per_shift[(chain, rn)])
                if runs and runs[-1][0] == col:
                    runs[-1][1].append(rn)
                else:
                    runs.append((col, [rn]))

            for col, res_list in runs:
                if col == "white":
                    continue
                if len(res_list) > 1 and res_list == list(range(res_list[0], res_list[-1] + 1)):
                    spec = f"/{chain}:{res_list[0]}-{res_list[-1]}"
                else:
                    spec = f"/{chain}:" + ",".join(str(r) for r in res_list)
                cmds.append(f"color {model_spec}{spec} {col}")
                exps.append(f"Colour chain {chain}{spec} {col} (shift percentile bucket)")

        return cmds, exps

    @classmethod
    def _mc_anchor_to_align_specs(
        cls,
        anchor_keys: "List[Any]",
        model_id_a:  str,
        model_id_b:  str,
    ) -> "Tuple[str, str]":
        """
        Build ChimeraX ``align`` atom specs for a multi-chain anchor.

        *anchor_keys* is a list of ``(chain_id, resno)`` tuples.

        - If all chains share the same residue numbers: ``#mid/A,B,C:range@CA``
        - Otherwise (asymmetric anchor): ``#mid/A,B,C@CA`` using all Cα of
          the anchor chains (approximate but always valid ChimeraX syntax).

        Returns ``(spec_for_b, spec_for_a)``.
        """
        chain_resnums: Dict[str, List[int]] = {}
        for (ch, rn) in anchor_keys:
            chain_resnums.setdefault(ch, []).append(rn)

        chains = sorted(chain_resnums.keys())
        chain_str = ",".join(chains)

        # Check whether all chains share the same sorted residue list
        sorted_sets = [tuple(sorted(chain_resnums[c])) for c in chains]
        if len(set(sorted_sets)) == 1:
            # Same range for all chains → compact /CHAINS:range spec
            range_str = cls._resnums_to_chimerax_range(list(sorted_sets[0]))
            spec_b = f"#{model_id_b}/{chain_str}:{range_str}@CA"
            spec_a = f"#{model_id_a}/{chain_str}:{range_str}@CA"
        else:
            # Asymmetric ranges → align on all Cα of anchor chains (approximate)
            spec_b = f"#{model_id_b}/{chain_str}@CA"
            spec_a = f"#{model_id_a}/{chain_str}@CA"

        return spec_b, spec_a

    def _run_conformer_comparison(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> "ToolStepResult":
        """
        Thin orchestrator: anchor-restricted conformer comparison.

        (1) Reads live Cα coordinates for both conformers via runscript.
            Multichain mode (chain="ALL") reads all chains as (chain,resno) keys.
        (2) Determines anchor residues via iterative prune (repeatedly removes
            the top-displaced residues until convergence).  Reliably converges on
            ONE rigid domain even for multimers (quaternary motion detection).
        (3) Anchor-restricted Kabsch: fits R+t on anchor, applies to all
            matched residues → per-residue Cα shift map.
        (4) Issues ``align #B/<anchor>@CA to #A/<anchor>@CA`` for visual overlay.
            Model A rendered fully opaque (reference frame); model B at
            CONFORMER_B_TRANSPARENCY% (default 50%) so both are visible.
        (5) Colours model B by per-residue shift (adaptive blue→red).
        (6) Writes CSV artifact, persists to session.

        Caveats baked into the output (project honesty ethos):
          • Geometric only — Cα displacement, NOT energetics.
          • Anchor quality = anchor residual RMSD (should be ≈ 0).
          • For two-domain motion, run twice anchoring on each domain.
        """
        import csv
        import time as _time
        from datetime import datetime
        from pathlib import Path

        t0 = _time.perf_counter()
        user_input = user_input or inputs.get("_user_input", "")

        model_id_a = str(inputs.get("model_id_a") or self._first_model_id())
        model_id_b = str(inputs.get("model_id_b") or self._second_model_id())
        chain_a    = str(inputs.get("chain_a") or "A").upper()
        chain_b    = str(inputs.get("chain_b") or "A").upper()
        anchor_str = str(inputs.get("anchor") or "auto").strip()
        # Multichain mode: chain_a=="ALL" → read all chains; keys become (chain,resno) tuples.
        multichain = (chain_a == "ALL")

        if model_id_a == model_id_b:
            return ToolStepResult(
                tool="conformer_comparison", success=False,
                error="model_id_a and model_id_b must be different models.",
            )
        if self.bridge is None:
            return ToolStepResult(
                tool="conformer_comparison", success=False,
                error="ChimeraX bridge unavailable — cannot read Cα coordinates.",
            )

        # ── 1. Read live Cα coordinates ───────────────────────────────────────
        if multichain:
            coords_a = self._ca_coords_live_multichain(self.bridge, model_id_a)
            coords_b = self._ca_coords_live_multichain(self.bridge, model_id_b)
            chain_display = "all chains"
        else:
            coords_a = self._ca_coords_live(self.bridge, model_id_a, chain_a)
            coords_b = self._ca_coords_live(self.bridge, model_id_b, chain_b)
            chain_display = f"{chain_a}/{chain_b}"

        if not coords_a:
            return ToolStepResult(
                tool="conformer_comparison", success=False,
                error=(f"Could not read Cα coordinates for model #{model_id_a} "
                       + (f"(all chains)" if multichain else f"chain {chain_a}.")
                       + "  Check the model is open."),
            )
        if not coords_b:
            return ToolStepResult(
                tool="conformer_comparison", success=False,
                error=(f"Could not read Cα coordinates for model #{model_id_b} "
                       + (f"(all chains)" if multichain else f"chain {chain_b}.")
                       + "  Check the model is open."),
            )

        common = sorted(set(coords_a) & set(coords_b))
        if len(common) < 10:
            return ToolStepResult(
                tool="conformer_comparison", success=False,
                error=(f"Too few common Cα residues ({len(common)}) between "
                       f"#{model_id_a} and #{model_id_b} ({chain_display}).  "
                       "Check that both conformers are the same protein and chain IDs match."),
            )

        # ── 2. Determine anchor residues ──────────────────────────────────────
        import numpy as _np
        common_set = set(common)

        if anchor_str.lower() == "auto":
            anchor_resnums, anchor_source = self._auto_anchor_resnums(
                coords_a, coords_b, common
            )
            if anchor_resnums is None:
                return ToolStepResult(
                    tool="conformer_comparison", success=False,
                    error=anchor_source,
                )
        else:
            anchor_resnums = self._parse_anchor_spec(anchor_str, common_set)
            if len(anchor_resnums) < 3:
                return ToolStepResult(
                    tool="conformer_comparison", success=False,
                    error=(f"Anchor spec {anchor_str!r} resolved to only "
                           f"{len(anchor_resnums)} common Cα residues (need ≥ 3).  "
                           "Use a wider range or 'auto'."),
                )
            anchor_source = f"user-specified ({anchor_str}; {len(anchor_resnums)} residues)"

        # ── 3. Anchor-restricted Kabsch ───────────────────────────────────────
        per_shift, anchor_rmsd, all_rmsd = self._anchor_kabsch(
            coords_a, coords_b, anchor_resnums
        )
        if per_shift is None:
            return ToolStepResult(
                tool="conformer_comparison", success=False,
                error="Anchor-restricted Kabsch fit failed.",
            )

        anchor_quality = (
            "GOOD" if anchor_rmsd is not None and anchor_rmsd < 0.5
            else "FAIR" if anchor_rmsd is not None and anchor_rmsd < 2.0
            else "POOR"
        )

        # ── 4. Visual overlay: ChimeraX align on anchor ───────────────────────
        viz_cmds: List[str] = []
        viz_exps: List[str] = []

        if multichain:
            align_spec_b, align_spec_a = self._mc_anchor_to_align_specs(
                anchor_resnums, model_id_a, model_id_b
            )
        else:
            anc_range    = self._resnums_to_chimerax_range(anchor_resnums)
            align_spec_b = f"#{model_id_b}/{chain_b}:{anc_range}@CA"
            align_spec_a = f"#{model_id_a}/{chain_a}:{anc_range}@CA"
        align_cmd = f"align {align_spec_b} to {align_spec_a}"

        align_res = self.bridge.run_command(align_cmd)
        align_val = (align_res.get("value") or "").strip()
        align_err = (align_res.get("error") or "").strip()

        # Fallback: if range notation fails (rare — unequal counts edge case),
        # try an explicit comma-separated residue list (single-chain only)
        if align_err and "Unequal" in align_err and not multichain:
            anc_list = ",".join(str(r) for r in anchor_resnums)
            align_cmd = (f"align #{model_id_b}/{chain_b}:{anc_list}@CA "
                         f"to #{model_id_a}/{chain_a}:{anc_list}@CA")
            align_res = self.bridge.run_command(align_cmd)
            align_val = (align_res.get("value") or "").strip()
            align_err = (align_res.get("error") or "").strip()

        viz_cmds.append(align_cmd)
        viz_exps.append(
            f"Anchor-restricted overlay: #{model_id_b} onto #{model_id_a} "
            f"(anchor: {anchor_source})"
        )

        # Parse ChimeraX-reported RMSD from the align command (independent check)
        align_rmsd: Optional[float] = None
        m_a = re.search(r"RMSD between\s+\d+\s+atom pairs is\s+([\d.]+)", align_val)
        if m_a:
            align_rmsd = round(float(m_a.group(1)), 3)

        # DEFAULT PRESENTATION: model A opaque (fixed reference frame);
        # model B semi-transparent so the superposition reads clearly.
        # Verified: "target c" required — bare "transparency #N 50" silently
        # has no effect on cartoons in ChimeraX 1.11.1 (§5 silent-success gap).
        import config as _cfg
        _transp = getattr(_cfg, "CONFORMER_B_TRANSPARENCY", 50)
        viz_cmds.append(f"transparency #{model_id_a} 0 target c")
        viz_exps.append(f"#{model_id_a} fully opaque (reference — fixed frame)")
        viz_cmds.append(f"transparency #{model_id_b} {_transp} target c")
        viz_exps.append(
            f"#{model_id_b} {_transp}% transparent (mobile — coloured by shift)"
        )

        viz_cmds.append("view")
        viz_exps.append("Fit aligned models in view")

        # ── 5. Colour model B by per-residue shift ────────────────────────────
        if multichain:
            color_cmds, color_exps = self._conformer_shift_color_cmds_mc(
                per_shift, f"#{model_id_b}"
            )
        else:
            color_cmds, color_exps = self._conformer_shift_color_cmds(
                per_shift, f"#{model_id_b}", chain_b
            )
        viz_cmds.extend(color_cmds)
        viz_exps.extend(color_exps)

        # ── 6. Compute summary statistics ─────────────────────────────────────
        anchor_set   = set(anchor_resnums)
        non_anchor   = {r: v for r, v in per_shift.items() if r not in anchor_set}
        all_vals     = list(per_shift.values())
        non_anc_vals = list(non_anchor.values()) if non_anchor else all_vals
        max_shift    = max(all_vals) if all_vals else 0.0
        mean_nonanc  = (round(sum(non_anc_vals) / len(non_anc_vals), 3)
                        if non_anc_vals else 0.0)
        top_k = sorted(per_shift.items(), key=lambda kv: kv[1], reverse=True)[:10]
        if multichain:
            top_shifted = [{"chain": rn[0], "resno": rn[1], "shift_A": sh}
                           for rn, sh in top_k]
        else:
            top_shifted = [{"chain": chain_b, "resno": rn, "shift_A": sh}
                           for rn, sh in top_k]

        # ── 7. Write CSV artifact ─────────────────────────────────────────────
        cache_dir = Path("cache")
        cache_dir.mkdir(exist_ok=True)
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = cache_dir / f"conformer_cmp_{model_id_a}v{model_id_b}_{ts}.csv"
        csv_written: Optional[str] = None
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["chain", "resno", "shift_A", "region"])
                if multichain:
                    for key_mc in sorted(per_shift):
                        ch_mc, rn_mc = key_mc
                        w.writerow([ch_mc, rn_mc, per_shift[key_mc],
                                    "anchor" if key_mc in anchor_set else "mobile"])
                else:
                    for rn in sorted(per_shift):
                        w.writerow([chain_b, rn, per_shift[rn],
                                    "anchor" if rn in anchor_set else "mobile"])
            csv_written = str(csv_path)
        except Exception:
            pass

        # ── 8. Build report ───────────────────────────────────────────────────
        lines = [
            f"Conformer comparison: #{model_id_a} (reference) ↔ #{model_id_b} (mobile), "
            f"{chain_display}",
            f"Anchor: {anchor_source}",
            f"  Anchor residual RMSD : {anchor_rmsd:.3f} Å  (quality: {anchor_quality})",
            f"  All-residue RMSD     : {all_rmsd:.3f} Å  (after anchor fit)",
            f"  Max shift            : {max_shift:.1f} Å",
            f"  Mean (non-anchor)    : {mean_nonanc:.1f} Å",
        ]
        if align_rmsd is not None:
            lines.append(f"  ChimeraX align check : {align_rmsd:.3f} Å (anchor subset)")
        lines += ["", "Top displaced residues (chain, resno, shift Å):"]
        for e in top_shifted:
            lines.append(f"  {e['chain']} {e['resno']:>4d}  {e['shift_A']:.1f} Å")
        if anchor_quality == "POOR":
            lines += [
                "",
                f"⚠ Anchor residual RMSD {anchor_rmsd:.2f} Å > 2 Å — the anchor may contain "
                "mobile residues.  Consider specifying a tighter anchor range.",
            ]
        lines += [
            "",
            "CAVEATS:",
            "  • Geometric only — Cα displacement, NOT energetics.",
            "  • Anchor quality (residual RMSD ≈ 0) is the validity check.",
            "  • For two-domain / quaternary motion: run twice anchoring on each domain.",
            "",
            f"Model #{model_id_a} opaque (reference); #{model_id_b} semi-transparent, "
            "coloured blue (rigid) → red (mobile).",
        ]
        if csv_written:
            lines.append(f"CSV: {csv_written}")
        summary_text = "\n".join(lines)

        # ── 9. Persist to session ─────────────────────────────────────────────
        elapsed_ms = round((_time.perf_counter() - t0) * 1000)
        # Anchor head: convert tuple keys to [chain,resno] pairs for JSON compat
        if multichain:
            anchor_head = [[k[0], k[1]] for k in anchor_resnums[:30]]
        else:
            anchor_head = anchor_resnums[:30]
        result_data: Dict[str, Any] = {
            "model_id_a":          model_id_a,
            "model_id_b":          model_id_b,
            "chain_a":             chain_a,
            "chain_b":             chain_b,
            "multichain":          multichain,
            "anchor":              anchor_source,
            "anchor_resnums_head": anchor_head,
            "anchor_rmsd":         anchor_rmsd,
            "anchor_quality":      anchor_quality,
            "all_rmsd":            all_rmsd,
            "align_rmsd":          align_rmsd,
            "max_shift":           round(max_shift, 3),
            "mean_non_anchor":     mean_nonanc,
            "top_shifted":         top_shifted,
            "per_shift":           per_shift,    # full map persisted
            "csv_path":            csv_written,
            "elapsed_ms":          elapsed_ms,
        }
        sess_key = f"{model_id_a}v{model_id_b}"
        self.session.set_conformer_comparison_results(sess_key, result_data)

        top_label = top_shifted[0] if top_shifted else {}
        summary_line = (
            f"#{model_id_a}↔#{model_id_b} {chain_display}: "
            f"anchor residual {anchor_rmsd:.2f} Å ({anchor_quality}), "
            f"max shift {max_shift:.1f} Å "
            f"(chain {top_label.get('chain','?')} res {top_label.get('resno','?')})"
        )
        return ToolStepResult(
            tool="conformer_comparison",
            success=True,
            data=result_data,
            viz_commands=viz_cmds,
            viz_explanations=viz_exps,
            summary=summary_line + "\n" + summary_text,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # S4c: variant-vs-WT per-residue Cα deviation + per-residue noise floor
    # ══════════════════════════════════════════════════════════════════════════
    # GLOBAL minimum effective floor (Å). Guards a small-N per-residue floor from
    # under-gating toward ~0, and gives a deterministic engine (ESMFold, floor 0) a
    # non-zero floor so sub-resolution coordinate noise is never painted as signal.
    _DEVIATION_FLOOR_MIN_A = 0.25

    # ── Cα-lDDT (superposition-FREE local-distance-difference) parameters ─────────
    # lDDT localizes WHERE the variant's local structure changed without any rigid-body
    # superposition — so a domain that merely re-orients (a rigid-body / lever-arm effect that
    # made the old Kabsch deviation read tens of Å, and inflated the WT cross-seed floor to
    # 10-15 Å) keeps its internal distances and stays high. Standard Mariani-2013 settings.
    _LDDT_INCLUSION_RADIUS_A = 15.0
    _LDDT_THRESHOLDS_A = (0.5, 1.0, 2.0, 4.0)
    # A residue with lDDT ≥ this is treated as conserved regardless of the cross-seed floor
    # (caps the gate so near-identical local structure is never painted as signal; the analog
    # of _DEVIATION_FLOOR_MIN_A, in lDDT space where HIGHER = more conserved).
    _LDDT_NEUTRAL_CAP = 0.9
    # ── per-residue dRMSD (all-pairs distance-RMSD, the PAINTED signal) ────────────
    # dRMSD captures BOTH local change AND rigid-body DISPLACEMENT relative to the rest (an
    # intact element that swung away keeps its internal distances but its distances to the
    # stationary part change → nonzero), while ignoring a WHOLE-body rigid move (every distance
    # preserved → 0). Superposition-free, so no anchor/lever-arm artifact. Global-min floor (Å)
    # so identical-up-to-noise is never painted.
    _DDM_FLOOR_MIN_A = 0.5

    @staticmethod
    def _dev_key(k: "Any") -> str:
        """JSON-safe per-residue key: ``(chain, resno)`` → ``"chain:resno"``; else ``str``."""
        if isinstance(k, tuple):
            return f"{k[0]}:{k[1]}"
        return str(k)

    @classmethod
    def _per_residue_lddt(cls, coords_a: Dict["Any", "Any"], coords_b: Dict["Any", "Any"],
                          common: "List[Any]") -> Dict["Any", float]:
        """Per-residue Cα-lDDT of B vs the reference A over *common* residues — the
        SUPERPOSITION-FREE local-distance-difference test (Mariani et al. 2013). For each
        residue, the fraction of its reference neighbour distances (those < inclusion radius in
        A) that B preserves within each of the four tolerances, averaged over the tolerances →
        lDDT ∈ [0,1] (1 = locally identical). No alignment is performed, so rigid-body / domain
        motion does NOT lower it — only genuine local geometry change (e.g. an insertion pushing
        neighbours apart) does. Returns ``{key: lddt}``; key-type agnostic like _anchor_kabsch.
        Pure / numpy-vectorized (one n×n matrix op)."""
        import numpy as _np
        keys = list(common)
        n = len(keys)
        if n < 2:
            return {}
        A = _np.array([coords_a[k] for k in keys], dtype=float)   # (n,3) reference
        B = _np.array([coords_b[k] for k in keys], dtype=float)   # (n,3) variant
        DA = _np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)   # (n,n) ref distances
        DB = _np.linalg.norm(B[:, None, :] - B[None, :, :], axis=-1)   # (n,n) variant distances
        include = DA < cls._LDDT_INCLUSION_RADIUS_A
        _np.fill_diagonal(include, False)                        # exclude self-pairs
        diff = _np.abs(DA - DB)
        preserved = _np.zeros_like(DA)
        for t in cls._LDDT_THRESHOLDS_A:
            preserved += (diff < t)
        preserved /= len(cls._LDDT_THRESHOLDS_A)                 # mean over tolerances → [0,1]
        out: Dict["Any", float] = {}
        for i, k in enumerate(keys):
            mask = include[i]
            cnt = int(mask.sum())
            out[k] = round(float(preserved[i][mask].mean()), 4) if cnt else 1.0
        return out

    @classmethod
    def _per_residue_ddm(cls, coords_a: Dict["Any", "Any"], coords_b: Dict["Any", "Any"],
                         common: "List[Any]") -> Dict["Any", float]:
        """Per-residue distance-RMSD (dRMSD, Å) of B vs reference A over *common* residues — the
        SUPERPOSITION-FREE all-pairs distance-difference. For residue i it is the RMS over every
        other residue j of ``|d_B(i,j) − d_A(i,j)|``. Unlike lDDT (local, 15 Å) this rises for a
        rigidly DISPLACED-but-intact element (its distances to the stationary part change) AND
        for local change — but stays ~0 for a whole-body rigid move (all distances preserved), so
        there is no anchor/lever-arm artifact. Returns ``{key: dRMSD}``. Pure / numpy-vectorized
        (one n×n op). Key-type agnostic like _anchor_kabsch."""
        import numpy as _np
        keys = list(common)
        n = len(keys)
        if n < 2:
            return {}
        A = _np.array([coords_a[k] for k in keys], dtype=float)
        B = _np.array([coords_b[k] for k in keys], dtype=float)
        DA = _np.linalg.norm(A[:, None, :] - A[None, :, :], axis=-1)
        DB = _np.linalg.norm(B[:, None, :] - B[None, :, :], axis=-1)
        sq = (DA - DB) ** 2
        _np.fill_diagonal(sq, 0.0)
        out: Dict["Any", float] = {}
        for i, k in enumerate(keys):
            out[k] = round(float(_np.sqrt(sq[i].sum() / (n - 1))), 3)   # RMS over the n−1 others
        return out

    def _read_fold_ca(self, model_id: str, multichain: bool, chain: str) -> Dict["Any", "Any"]:
        """Live Cα of a predicted fold model — ``{(chain,resno):xyz}`` (multichain) else
        ``{resno:xyz}``. Reuses the conformer-comparison live readers (one source)."""
        if multichain:
            return self._ca_coords_live_multichain(self.bridge, str(model_id))
        return self._ca_coords_live(self.bridge, str(model_id), chain)

    def _fold_wt_reference(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Establish the seed-pinned WT reference fold (+ per-residue noise floor) for a
        (engine, target) combo. Folds the template T sequence(s) through the SAME engine
        the variant used (fold-vs-fold cancellation only holds same-engine/target):

          • reference  = the pinned-seed fold, opened + pLDDT-coloured + matchmadered onto
            the crystal WT via the shared `_fold_viz_commands` (executed inline, the
            conformer-style live pattern — the spine path only BUILDS those commands);
          • floor (Boltz only) = fold WT at the N-1 extra seeds, Kabsch-align each onto the
            reference via the SHARED auto-anchor, and take the per-residue CROSS-SEED MAX
            displacement (conservative at small N); effective floor = max(that, the global
            minimum). ESMFold is deterministic → no extra seeds, floor = the global min.

        Returns the wt_ref dict ``{engine,target,seed,model_id,path,floor(str-keyed)}`` (the
        workbench caches it on ``cd.wt_refs[combo]``), or None on a fold failure."""
        engine     = inputs["engine"]
        target     = inputs.get("target", "monomer")
        multichain = bool(inputs.get("multichain"))
        chain      = inputs.get("variant_chain", "A")
        wt_chains  = inputs.get("wt_chains") or []
        compare_to = inputs.get("compare_to")
        model_id   = inputs.get("model_id")
        seeds      = list(inputs.get("seeds") or [])
        # TEMPLATE-GUIDED floor (assist): when given, the cross-seed floor folds carry the SAME
        # template steering as the seed-0 reference, so the "guided flexibility floor" measures
        # the steered ensemble's wiggle. None for the unguided floor (the normal path).
        fold_templates, _terr = self._resolve_boltz_templates(inputs.get("fold_templates"))
        if engine == "boltz" and not seeds:
            import config as _cfg
            base = int(getattr(_cfg, "BOLTZ_SEED", 0))
            n    = int(getattr(_cfg, "DEVIATION_FLOOR_N", 4))   # N=4: 1 reference + 3 floor
            seeds = list(range(base, base + max(1, n)))

        # ── reference: REUSE an existing fold (de-novo T-fold) OR fold T fresh ────
        reuse = inputs.get("wt_ref") or {}
        if reuse.get("model_id"):
            # REUSE PATH (de-novo): the construct's T-fold IS the reference (folded at the pinned
            # seed already). Skip the reference fold entirely — read its Cα as the seed-0 baseline;
            # only the floor seeds below are folded. Reopen from path if it was closed mid-session.
            ref_mid  = str(reuse["model_id"])
            ref_path = reuse.get("path")
            ref_seed = reuse.get("seed")
            ref_ca   = self._read_fold_ca(ref_mid, multichain, chain)
            if not ref_ca and ref_path:
                ro = self.bridge.run_command(f'open "{Path(ref_path).as_posix()}"')
                spec = self._parse_model_spec(ro, None)
                if spec:
                    ref_mid = spec.lstrip("#")
                    ref_ca = self._read_fold_ca(ref_mid, multichain, chain)
            if not ref_ca:
                return None
        else:
            # FRESH-FOLD PATH (crystal design): fold the template T at the pinned seed.
            if engine == "esmfold":
                seq = (wt_chains[0]["sequence"] if wt_chains else inputs.get("wt_sequence"))
                if not seq:
                    return None
                rres = self._get_esmfold_bridge().predict(seq, label="WTref", allow_remote=False)
                ref_path = None
                if rres.get("success"):
                    import tempfile as _tf
                    _t = _tf.NamedTemporaryFile(mode="w", suffix=".pdb",
                                                prefix="wtref_esmfold_", delete=False)
                    _t.write(rres.get("pdb_str", "")); _t.close()
                    ref_path = _t.name
                ref_seed = None
            else:
                if not wt_chains:
                    return None
                rres = self._get_boltz_bridge().predict(
                    wt_chains, seed=(seeds[0] if seeds else None), allow_remote=False,
                    templates=fold_templates)
                ref_path = rres.get("cif_path")
                ref_seed = rres.get("seed")
            if not rres.get("success") or not ref_path:
                return None

            # Open LIVE + read the REAL id back (V3 fix — a guessed id would mis-target the viz
            # AND the CA read below, silently corrupting the deviation against this reference).
            ref_mid, _cmds, _exps = self._open_and_viz_fold_live(
                Path(ref_path).as_posix(), {**inputs, "compare_to": compare_to})
            try:                                  # consume the id so it isn't reused
                self.session.add_structure(
                    ref_mid, f"wtref_{engine}_{model_id}", path=ref_path,
                    metadata={"predicted": True, "engine": engine, "wt_reference": True})
            except Exception:
                pass

            ref_ca = self._read_fold_ca(ref_mid, multichain, chain)
            if not ref_ca:
                return None

        # ── per-residue noise floors: cross-seed WT variation (Boltz only) ────────
        # Both SUPERPOSITION-FREE, from the extra-seed WT folds — how much the WT's OWN structure
        # varies seed↔seed, the noise the variant must beat to count as a real change:
        #   • ddm_floor_raw  = cross-seed MAX per-residue dRMSD (Å).
        #   • lddt_floor_raw = cross-seed MIN per-residue lDDT (worst local self-consistency).
        lddt_floor_raw: Dict["Any", float] = {}
        ddm_floor_raw: Dict["Any", float] = {}
        if engine == "boltz" and len(seeds) > 1:
            bridge = self._get_boltz_bridge()
            for s in seeds[1:]:
                fr = bridge.predict(wt_chains, seed=s, allow_remote=False,
                                    templates=fold_templates)
                fpath = fr.get("cif_path") if fr.get("success") else None
                if not fpath:
                    continue
                ro = self.bridge.run_command(f'open "{Path(fpath).as_posix()}"')
                spec = self._parse_model_spec(ro, None)
                if not spec:
                    continue
                fmid = spec.lstrip("#")
                fca = self._read_fold_ca(fmid, multichain, chain)
                self.bridge.run_command(f"close #{fmid}")   # floor folds are not kept
                if not fca:
                    continue
                common = sorted(set(ref_ca) & set(fca))
                for k, lv in self._per_residue_lddt(ref_ca, fca, common).items():
                    if k not in lddt_floor_raw or lv < lddt_floor_raw[k]:
                        lddt_floor_raw[k] = lv           # worst (lowest) self-consistency
                for k, dv in self._per_residue_ddm(ref_ca, fca, common).items():
                    if dv > ddm_floor_raw.get(k, 0.0):
                        ddm_floor_raw[k] = dv            # cross-seed MAX dRMSD

        # lDDT floor = min(cross-seed-min lDDT, neutral cap) — local-integrity noise.
        floor_lddt: Dict[str, float] = {
            self._dev_key(k): round(min(lddt_floor_raw.get(k, 1.0), self._LDDT_NEUTRAL_CAP), 4)
            for k in ref_ca
        }
        # dRMSD floor (Å) = max(cross-seed max, global min) — the PAINTED signal's gate. Built
        # superposition-free (no anchor), so it does NOT inflate the way a rigid-body-fit floor
        # does on re-orienting WT regions; the variant must move MORE than the WT does seed↔seed.
        floor_ddm: Dict[str, float] = {
            self._dev_key(k): round(max(ddm_floor_raw.get(k, 0.0), self._DDM_FLOOR_MIN_A), 3)
            for k in ref_ca
        }
        return {
            "engine": engine, "target": target, "seed": ref_seed,
            "model_id": str(ref_mid), "path": ref_path,
            "floor_lddt": floor_lddt, "floor_ddm": floor_ddm,
            "n_floor_seeds": (len(seeds) if engine == "boltz" else 1),
        }

    def _run_variant_deviation(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """Per-residue variant-vs-WT deviation of a FOLDED variant vs the seed-pinned WT REFERENCE
        FOLD (fold-vs-fold, real atoms — not the S2 sequence preview, not the crystal).

        Ensures the (engine, target) WT reference exists (folding it + its cross-seed floors once
        per design when absent — the expensive path; reused via the cached `wt_ref` otherwise),
        then computes two SUPERPOSITION-FREE per-residue signals over the indel-paired residues:
        `ddm` (all-pairs distance-RMSD, the painted magnitude) and `lddt` (local-distance-difference,
        the secondary local-integrity signal), each gated by its cross-seed noise floor. Per-chain
        for an assembly (multichain CA). Data-only: the floor-gated 3-tier rendering is the panel's
        deviation colour mode (one source: `color_modes.combined_disruption_color`), feeding the
        SAME seam as ddG/pLDDT — not a parallel render path."""
        variant_mid = inputs.get("variant_model_id")
        multichain  = bool(inputs.get("multichain"))
        chain       = inputs.get("variant_chain", "A")
        engine      = inputs.get("engine", "esmfold")
        target      = inputs.get("target", "monomer")
        if not variant_mid:
            return ToolStepResult(tool="variant_deviation", success=False,
                                  error="variant_deviation needs the folded variant's model id.")

        # ── 1. Ensure the WT reference fold (+ floor) for this combo ──────────────
        wt_ref = inputs.get("wt_ref")
        ref_ca: Dict["Any", "Any"] = {}
        if wt_ref and wt_ref.get("model_id"):
            ref_ca = self._read_fold_ca(wt_ref["model_id"], multichain, chain)
            if not ref_ca and wt_ref.get("path"):     # cached/reused but not open → reopen
                ro = self.bridge.run_command(f'open "{Path(wt_ref["path"]).as_posix()}"')
                spec = self._parse_model_spec(ro, None)
                if spec:
                    wt_ref = {**wt_ref, "model_id": spec.lstrip("#")}
                    ref_ca = self._read_fold_ca(wt_ref["model_id"], multichain, chain)
            # REUSED reference (a de-novo construct's T-fold) WITHOUT a floor yet → establish the
            # cross-seed floor ONLY (fold the N-1 extra seeds against this model as seed-0; NO fresh
            # fold of T). The result carries the floor so the workbench caches the full wt_ref.
            if ref_ca and not wt_ref.get("floor_ddm"):
                established = self._fold_wt_reference({**inputs, "wt_ref": wt_ref})
                if established:
                    wt_ref = established
                    ref_ca = self._read_fold_ca(wt_ref["model_id"], multichain, chain) or ref_ca
        if not ref_ca:
            wt_ref = self._fold_wt_reference(inputs)
            if not wt_ref:
                return ToolStepResult(
                    tool="variant_deviation", success=False,
                    error=(f"Could not establish the WT reference fold for {engine}:{target} "
                           "(fold failed or no Cα read). Deviation not computed."))
            ref_ca = self._read_fold_ca(wt_ref["model_id"], multichain, chain)

        # ── 2. Variant fold Cα + matched-residue Kabsch against the reference ─────
        var_ca = self._read_fold_ca(variant_mid, multichain, chain)
        if not var_ca:
            return ToolStepResult(
                tool="variant_deviation", success=False,
                error=(f"Could not read Cα for the variant fold #{variant_mid}. "
                       "Check the predicted model is open."))
        # INDEL-AWARE column pairing (additive): re-key the variant fold onto the REFERENCE fold
        # numbering via the panel-built {variant_fold_resnum: reference_fold_resnum} map, so a
        # deletion's downstream residues pair to the correct template position (not the one-off
        # mis-pair resnum==resnum would give), and an INSERTED residue (a variant residue at a
        # template-gap column — build_fold_column_map omits it by design, no WT counterpart) is
        # DROPPED (excluded, rendered neutral). Identity map (substitution-only) → no-op; an
        # ABSENT map → resnum==resnum. The pairing is what every per-residue value below is keyed
        # on, so it's the load-bearing correctness guarantee for indels.
        fold_map = inputs.get("fold_column_map")
        applied_map: Optional[Dict[int, int]] = None
        if fold_map and not multichain:
            applied_map = {int(k): int(v) for k, v in fold_map.items()}
            var_ca = {applied_map[j]: xyz for j, xyz in var_ca.items() if j in applied_map}
        common = sorted(set(ref_ca) & set(var_ca))
        if len(common) < 3:
            return ToolStepResult(
                tool="variant_deviation", success=False,
                error=(f"Only {len(common)} common Cα between the variant fold and the WT "
                       "reference (need ≥3) — chain ids/numbering mismatch?"))

        # ── PAINTED signal: superposition-free per-residue dRMSD (variant vs WT ref) ──
        # Distances only (no rigid-body fit), so it rises for a rigidly DISPLACED-but-intact
        # element (its distances to the stationary part change — what lDDT misses) AND for local
        # change, but stays ~0 for a whole-body move. Keyed by REFERENCE resnum over `common`;
        # inserted residues are absent (no WT counterpart) → excluded.
        ddm = self._per_residue_ddm(ref_ca, var_ca, common)
        ddm_str = {self._dev_key(k): v for k, v in ddm.items()}
        floor_ddm = (wt_ref or {}).get("floor_ddm") or {}
        dmin = self._DDM_FLOOR_MIN_A
        disrupted = [k for k, v in ddm_str.items() if v > floor_ddm.get(k, dmin)]
        max_ddm = max(ddm.values()) if ddm else 0.0
        # SECONDARY (reported + used in the 3-tier gate, not the magnitude): Cα-lDDT — local-fold
        # integrity, distinguishing a MELTED region (low lDDT) from one that merely MOVED intact
        # (high lDDT, high dRMSD).
        lddt = self._per_residue_lddt(ref_ca, var_ca, common)
        lddt_str = {self._dev_key(k): v for k, v in lddt.items()}
        floor_lddt = (wt_ref or {}).get("floor_lddt") or {}
        min_lddt = min(lddt.values()) if lddt else 1.0
        mean_lddt = round(sum(lddt.values()) / len(lddt), 4) if lddt else 1.0

        data = {
            "engine":               engine,
            "target":               target,
            "multichain":           multichain,
            "variant_chain":        chain,            # predicted monomer chain (3D target)
            "variant_model_id":     str(variant_mid),
            "reference_model_id":   str(wt_ref.get("model_id")),
            "ddm":                  ddm_str,          # PAINTED magnitude: per-residue dRMSD (Å)
            "floor_ddm":            floor_ddm,        # cross-seed dRMSD noise floor (gate)
            "lddt":                 lddt_str,         # per-residue Cα-lDDT (1=conserved)
            "floor_lddt":           floor_lddt,       # cross-seed lDDT floor (gate)
            # {variant_fold_resnum: reference_fold_resnum} actually applied (monomer indel) — the
            # per-residue maps are keyed by REFERENCE resnum; the 3D push inverts this to paint
            # the VARIANT model in its OWN numbering (inserted residues, absent here, stay
            # neutral). None → identity (substitution-only / no map).
            "fold_column_map":      ({str(j): r for j, r in applied_map.items()}
                                     if applied_map else None),
            "max_ddm":              round(float(max_ddm), 3),
            "min_lddt":             round(float(min_lddt), 4),
            "mean_lddt":            mean_lddt,
            "n_residues":           len(ddm),
            "n_disrupted":          len(disrupted),   # residues above the dRMSD floor
            "floor_kind":           ("deterministic" if engine == "esmfold" else "measured"),
            "wt_ref":               wt_ref,           # so the workbench caches it per combo
        }
        return ToolStepResult(
            tool="variant_deviation", success=True, data=data,
            summary=(f"Deviation vs WT ({engine}:{target}) — dRMSD: "
                     f"{len(disrupted)}/{len(ddm)} residues disrupted (above the cross-seed "
                     f"floor); max {max_ddm:.2f} Å · local integrity (lDDT) min "
                     f"{min_lddt:.3f}, mean {mean_lddt:.3f}."))

    # ── Stage 3: US-align sequence-INDEPENDENT structural alignment ────────────────────
    @staticmethod
    def _parse_usalign_output(stdout: str) -> Optional[Dict[str, Any]]:
        """Parse US-align `-outfmt 2 -m -` stdout: the tab DATA line + the 3×4 rotation
        matrix. The tab columns are
            PDBchain1  PDBchain2  TM1  TM2  RMSD  ID1  ID2  IDali  L1  L2  Lali
        (TM1 normalized by structure-1 = the query/construct fold; TM2 by structure-2 = the
        reference). The matrix rows are `m  t[m]  u[m][0]  u[m][1]  u[m][2]`; we return it
        ROW-MAJOR as [u00,u01,u02,t0, u10,u11,u12,t1, u20,u21,u22,t2] — exactly ChimeraX's
        `view matrix models #N,<12>` order (live-verified to reproduce US-align's superposition,
        no transpose/sign flip). Returns None if neither scores nor the full matrix parse."""
        if not stdout:
            return None
        lines = stdout.splitlines()
        data: Optional[List[str]] = None
        for i, ln in enumerate(lines):
            if ln.startswith("#PDBchain1") and i + 1 < len(lines):
                parts = lines[i + 1].split("\t")
                if len(parts) >= 11:
                    data = parts
                break
        if data is None:
            return None
        try:
            out: Dict[str, Any] = {
                "tm1": float(data[2]), "tm2": float(data[3]), "rmsd": float(data[4]),
                "id1": float(data[5]), "id2": float(data[6]), "idali": float(data[7]),
                "l1": int(data[8]), "l2": int(data[9]), "lali": int(data[10]),
            }
        except (ValueError, IndexError):
            return None
        rows: Dict[int, tuple] = {}
        for ln in lines:
            s = ln.split()
            if len(s) == 5 and s[0] in ("0", "1", "2"):
                try:
                    m = int(s[0]); t = float(s[1])
                    u0, u1, u2 = float(s[2]), float(s[3]), float(s[4])
                except ValueError:
                    continue
                rows[m] = (u0, u1, u2, t)                  # row-major: r0,r1,r2,tx
        out["matrix"] = ([v for m in (0, 1, 2) for v in rows[m]]
                         if all(m in rows for m in (0, 1, 2)) else None)
        return out

    @staticmethod
    def _view_matrix_command(model_id: str, matrix12: List[float]) -> str:
        """ChimeraX command to place model *model_id* at the US-align transform (option B):
        `view matrix models #N,m00,…,m23` (row-major 3×4). Pure/testable — the one new
        ChimeraX seam, its convention live-confirmed against US-align's own superposition."""
        nums = ",".join(f"{float(v):.10g}" for v in matrix12)
        mid = str(model_id).lstrip("#")
        return f"view matrix models #{mid},{nums}"

    def _run_align_folds(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """Compare TWO existing folds of the same construct (ANY engines) — the comparative-fold
        readout. REUSE ONLY, no new alignment code: US-align (LOCAL-ONLY WSL binary) for the
        whole-structure TM/RMSD, and the superposition-free per-residue deviation machinery
        (`_per_residue_ddm`/`_per_residue_lddt` over the open models' Cα) for agreement.

        Framed HONESTLY: the headline case is the LOCAL single-sequence Boltz fold vs the
        MSA-informed ColabFold fold — NOT a fair model-vs-model test (one has an MSA, one does
        not), so it largely measures the MSA's value as an accuracy yardstick for the local fold.
        The asymmetry is stated up front in the summary; provenance (which fold was remote) is
        carried through so the readout never implies same-footing."""
        import os as _os
        import shlex as _shlex
        import config as _cfg
        a = inputs.get("fold_a") or {}
        b = inputs.get("fold_b") or {}
        path_a, path_b = a.get("path"), b.get("path")
        if not path_a or not _os.path.isfile(path_a) or not path_b or not _os.path.isfile(path_b):
            return ToolStepResult(tool="align_folds", success=False,
                                  error="Align folds needs BOTH fold files on disk (fold each first).")
        label_a = a.get("label") or a.get("engine") or "fold A"
        label_b = b.get("label") or b.get("engine") or "fold B"
        # ── US-align A vs B (LOCAL-ONLY WSL binary; reads CIF/PDB directly) ────────
        from wsl_bridge import WSLBridge
        wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
        if not wsl.is_available():
            return ToolStepResult(tool="align_folds", success=False,
                                  error="WSL2 unavailable — US-align runs in WSL.")
        qa = wsl.translate_path(_os.path.abspath(path_a))
        rb = wsl.translate_path(_os.path.abspath(path_b))
        exe = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")
        cmd = f"{_shlex.quote(exe)} {_shlex.quote(qa)} {_shlex.quote(rb)} -outfmt 2 -m -"
        res = wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 120))
        if not res.get("ok"):
            return ToolStepResult(tool="align_folds", success=False,
                                  error=f"US-align failed: {res.get('error') or res.get('stderr','')[:200]}")
        parsed = self._parse_usalign_output(res.get("stdout", ""))
        if not parsed:
            return ToolStepResult(tool="align_folds", success=False,
                                  error="Could not parse US-align output (no scores).")
        # ── Per-residue agreement over the OPEN models' Cα (superposition-free) ────
        multichain = bool(inputs.get("multichain"))
        chain      = inputs.get("chain", "A")
        mean_ddm = mean_lddt = None
        n_common = 0
        ca_a = self._read_fold_ca(a.get("model_id"), multichain, chain) if a.get("model_id") else {}
        ca_b = self._read_fold_ca(b.get("model_id"), multichain, chain) if b.get("model_id") else {}
        common = sorted(set(ca_a) & set(ca_b))
        n_common = len(common)
        if n_common >= 3:
            ddm  = self._per_residue_ddm(ca_a, ca_b, common)
            lddt = self._per_residue_lddt(ca_a, ca_b, common)
            mean_ddm  = round(sum(ddm.values()) / len(ddm), 3) if ddm else None
            mean_lddt = round(sum(lddt.values()) / len(lddt), 4) if lddt else None
        remote_a, remote_b = bool(a.get("remote_msa")), bool(b.get("remote_msa"))
        # Asymmetry framing: name the local-single-seq vs MSA-informed pairing when present.
        if remote_a ^ remote_b:
            local_lbl  = label_b if remote_a else label_a
            remote_lbl = label_a if remote_a else label_b
            framing = (f"local single-sequence {local_lbl} vs MSA-informed {remote_lbl} — "
                       f"NOT a fair model-vs-model test (one has an MSA, one does not); this "
                       f"largely measures the MSA's value as an accuracy yardstick for the local fold")
        else:
            framing = (f"{label_a} vs {label_b} — both same-provenance "
                       f"({'remote MSA' if remote_a else 'local single-sequence'})")
        data = {
            "fold_a": {k: a.get(k) for k in ("label", "engine", "model_id", "remote_msa")},
            "fold_b": {k: b.get(k) for k in ("label", "engine", "model_id", "remote_msa")},
            "tm": round(parsed.get("tm2", 0.0), 4), "tm_a": round(parsed.get("tm1", 0.0), 4),
            "rmsd": round(parsed.get("rmsd", 0.0), 3), "n_aligned": parsed.get("lali"),
            "mean_ddm_A": mean_ddm, "mean_lddt": mean_lddt, "n_common": n_common,
            "framing": framing,
        }
        agree = ("HIGH" if data["tm"] >= 0.8 else "moderate" if data["tm"] >= 0.5 else "LOW")
        summary = (
            f"Align folds — {framing}.\n"
            f"  {label_a} ↔ {label_b}: TM {data['tm']:.3f} ({agree} structural agreement), "
            f"RMSD {data['rmsd']:.2f} Å over {data['n_aligned']} residues"
            + (f"; mean per-residue dRMSD {mean_ddm:.2f} Å, Cα-lDDT {mean_lddt:.3f} "
               f"(over {n_common} common Cα)" if mean_ddm is not None else
               " (per-residue agreement skipped — a model was not open for Cα).")
        )
        return ToolStepResult(tool="align_folds", success=True, data=data, summary=summary)

    def _run_structural_align(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """Sequence-INDEPENDENT structural alignment of a de-novo construct's FOLD onto a chosen
        reference PDB via US-align (LOCAL-ONLY WSL C++ binary) — the case ChimeraX matchmaker
        can't reach (matchmaker is sequence-guided and fails closed at zero homology).

        Captures BOTH TM-scores (default-surfaced: reference-normalized = TM2), RMSD, # aligned,
        and the 3×4 transform; then OVERLAYS by opening the reference and placing the construct
        fold model onto it with `view matrix` (option B — preserves the fold's pLDDT colour, no
        extra model). Data-captured (matchmaker's RMSD is fired-and-ignored; this one is kept)."""
        import os as _os
        import shlex as _shlex
        import time as _time
        import config as _cfg
        t0 = _time.perf_counter()
        query_path     = inputs.get("query_path")
        query_model_id = inputs.get("query_model_id")
        ref_path       = inputs.get("reference_path")
        ref_id         = inputs.get("reference_pdb_id")
        ref_model_id   = inputs.get("reference_model_id")     # a loaded model (panel saved its file)
        ref_label      = inputs.get("ref_label") or ref_id or "reference"
        if not query_path or not _os.path.isfile(query_path):
            return ToolStepResult(tool="structural_align", success=False,
                                  error="structural_align needs the construct fold file on disk.")
        if not ref_path and ref_id:
            ref_path = self._download_pdb_by_id(ref_id)
        if not ref_path or not _os.path.isfile(ref_path):
            return ToolStepResult(tool="structural_align", success=False,
                                  error=f"Could not resolve the reference structure ({ref_label}).")
        # ── US-align in WSL (LOCAL-ONLY; reads CIF/PDB directly) ──────────────────
        from wsl_bridge import WSLBridge
        wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
        if not wsl.is_available():
            return ToolStepResult(tool="structural_align", success=False,
                                  error="WSL2 unavailable — US-align runs in WSL.")
        q = wsl.translate_path(_os.path.abspath(query_path))
        r = wsl.translate_path(_os.path.abspath(ref_path))
        exe = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")
        cmd = f"{_shlex.quote(exe)} {_shlex.quote(q)} {_shlex.quote(r)} -outfmt 2 -m -"
        res = wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 120))
        if not res.get("ok"):
            return ToolStepResult(tool="structural_align", success=False,
                                  error=f"US-align failed: {res.get('error') or res.get('stderr','')[:200]}")
        parsed = self._parse_usalign_output(res.get("stdout", ""))
        if not parsed or parsed.get("matrix") is None:
            return ToolStepResult(tool="structural_align", success=False,
                                  error="Could not parse US-align output (no scores/transform).")
        # ── LIVE overlay (option B): open the reference, place the fold onto it ────
        overlay_cmds: List[str] = []
        if query_model_id:
            if not ref_model_id:                       # PDB-id reference → open it live
                ro = self.bridge.run_command(f'open "{Path(ref_path).as_posix()}"')
                spec = self._parse_model_spec(ro, None)
                ref_model_id = spec.lstrip("#") if spec else None
                if ref_model_id:
                    # TRACK the just-opened reference in session.structures so it is (a) under the
                    # workbench's alignment-visibility authority (toggleable like a fold) and (b)
                    # reconnect-aware: name = the 4-char PDB id → `_resolve_reopen_target` re-fetches
                    # it by id on reload. (The loaded-model reference is ALREADY its own session
                    # structure and re-opens from that entry — no separate tracking needed.)
                    try:
                        self.session.add_structure(
                            ref_model_id, (ref_id or ref_label),
                            path=ref_path,
                            metadata={"aligned_reference": True, "ref_pdb_id": (ref_id or None)})
                    except Exception:
                        pass
            vm = self._view_matrix_command(query_model_id, parsed["matrix"])
            self.bridge.run_command(vm); overlay_cmds.append(vm)
            if ref_model_id:                           # neutral grey ref under the pLDDT fold
                for c in (f"cartoon #{ref_model_id}",
                          f"color #{ref_model_id} gray target c"):
                    self.bridge.run_command(c); overlay_cmds.append(c)
            self.bridge.run_command("view"); overlay_cmds.append("view")
        shared = parsed["tm2"] >= 0.5
        data = {
            "reference":          ref_id or ref_label,
            "ref_label":          ref_label,
            "reference_path":     ref_path,
            "reference_model_id": str(ref_model_id) if ref_model_id else None,
            "query_model_id":     str(query_model_id) if query_model_id else None,
            "tm_ref":             round(parsed["tm2"], 4),   # reference-normalized (US-align default)
            "tm_query":           round(parsed["tm1"], 4),   # query (construct fold) normalized
            "rmsd":               round(parsed["rmsd"], 3),
            "n_aligned":          parsed["lali"],
            "seq_id_ali":         round(parsed["idali"], 4),
            "query_len":          parsed["l1"],
            "ref_len":            parsed["l2"],
            "matrix":             parsed["matrix"],          # 12 row-major (ChimeraX view-matrix order)
            "norm_default":       "reference",
            "shared_fold":        bool(shared),
            "overlay_commands":   overlay_cmds,
        }
        tier = "shared fold" if shared else "NOT structurally similar"
        summary = (f"Structural alignment vs {ref_label}: TM-score {parsed['tm2']:.3f} (ref-norm) "
                   f"/ {parsed['tm1']:.3f} (query-norm), RMSD {parsed['rmsd']:.2f} Å over "
                   f"{parsed['lali']} residues — {tier} (TM>0.5 = shared fold).")
        return ToolStepResult(tool="structural_align", success=True, data=data,
                              viz_commands=overlay_cmds, summary=summary,
                              elapsed_ms=(_time.perf_counter() - t0) * 1000)

    def _run_template_assist(self, inputs: Dict[str, Any]) -> "ToolStepResult":
        """TEMPLATE-ASSIST readout — did the structural template actually help the fold? Compares
        the GUIDED fold against the construct's existing UNGUIDED T-fold (both already on disk —
        NEITHER is re-folded; we reuse them as the seed-0 of their own cross-seed ensembles):

          • ΔpLDDT          = guided.mean_plddt − unguided.mean_plddt (confidence shift).
          • Δflexibility[k] = unguided_floor[k] − guided_floor[k] (per-residue cross-seed dRMSD;
            POSITIVE = the template made residue k MORE rigid seed↔seed = stabilized).

        Reuses `_fold_wt_reference` (its de-novo REUSE path) for BOTH floors — the only new wiring
        is threading the template list into the GUIDED floor's seed folds (`fold_templates`), so
        the guided ensemble wiggles WITH the template. No new floor math (`_per_residue_ddm`
        inside `_fold_wt_reference`). HONEST: guided pLDDT↑ is template bias, never proof of
        native — the readout surfaces guided AND unguided AND the delta. Cost: the guided +
        unguided floors fold ~2×(N−1) extra Boltz seeds.

        Inputs: engine/target/multichain/variant_chain, wt_chains, unguided_ref/guided_ref
        ({model_id,path,seed}), guided_mean_plddt/unguided_mean_plddt, templates, optional
        guided_plddt/unguided_plddt (author-resnum-keyed → per-residue ΔpLDDT), template_label,
        force, threshold, seeds."""
        import time as _time
        t0 = _time.perf_counter()
        engine = inputs.get("engine", "boltz")
        if engine != "boltz":
            return ToolStepResult(tool="template_assist", success=False,
                error="Template-assist is Boltz-only (the cross-seed flexibility floor needs "
                      "Boltz's seed sampling; ESMFold is deterministic).")
        unguided_ref = inputs.get("unguided_ref") or {}
        guided_ref   = inputs.get("guided_ref") or {}
        if not unguided_ref.get("model_id") or not guided_ref.get("model_id"):
            return ToolStepResult(tool="template_assist", success=False,
                error="Template-assist needs BOTH the unguided baseline fold and the guided fold "
                      "(fold the construct unguided AND guided first).")
        base = {
            "engine":        "boltz",
            "target":        inputs.get("target", "monomer"),
            "multichain":    bool(inputs.get("multichain")),
            "variant_chain": inputs.get("variant_chain", "A"),
            "wt_chains":     inputs.get("wt_chains") or [],
            "model_id":      inputs.get("model_id"),
            "compare_to":    inputs.get("compare_to"),
            "seeds":         list(inputs.get("seeds") or []),
        }
        # UNGUIDED floor (no templates) — REUSE the on-disk T-fold as seed-0.
        ung = self._fold_wt_reference({**base, "wt_ref": unguided_ref, "fold_templates": None})
        # GUIDED floor (same templates as the guided fold) — REUSE the on-disk guided fold.
        gud = self._fold_wt_reference({**base, "wt_ref": guided_ref,
                                       "fold_templates": inputs.get("templates")})
        if ung is None or gud is None:
            return ToolStepResult(tool="template_assist", success=False,
                error="Template-assist could not establish a flexibility floor for one of the "
                      "folds (a floor-seed fold failed). See the Boltz log.")
        unguided_floor = ung.get("floor_ddm") or {}
        guided_floor   = gud.get("floor_ddm") or {}
        common = sorted(set(unguided_floor) & set(guided_floor), key=self._dev_sort_key)
        d_flex = {k: round(unguided_floor[k] - guided_floor[k], 3) for k in common}
        n_stab = sum(1 for v in d_flex.values() if v > 0)        # template made it MORE rigid
        n_loose = sum(1 for v in d_flex.values() if v < 0)       # template made it LESS rigid
        mean_dflex = round(sum(d_flex.values()) / len(d_flex), 3) if d_flex else 0.0
        g_plddt = inputs.get("guided_mean_plddt")
        u_plddt = inputs.get("unguided_mean_plddt")
        d_plddt = (round(float(g_plddt) - float(u_plddt), 2)
                   if isinstance(g_plddt, (int, float)) and isinstance(u_plddt, (int, float)) else None)
        # Optional per-residue ΔpLDDT (author-resnum-keyed; both maps must be present).
        gp, up = inputs.get("guided_plddt") or {}, inputs.get("unguided_plddt") or {}
        d_plddt_by_res: Dict[str, float] = {}
        for rn in sorted(set(gp) & set(up), key=lambda x: int(x)):
            try:
                d_plddt_by_res[str(rn)] = round(float(gp[rn]) - float(up[rn]), 2)
            except (TypeError, ValueError):
                pass
        # ── ADOPTION + pre-hoc PROXY (per-template, USE-TIME-knowable — no ground truth) ──────
        # ADOPTION = structTM(guided fold, template): how much the fold FOLLOWS each template.
        #   HIGH adoption is a COPYING caveat — without an experimental structure we cannot tell
        #   independent convergence from the fold tracing the template. (The eval-only "unlocking
        #   test" TM_G≫template needs structTM-to-TRUTH, which does NOT exist at use time, so it is
        #   NOT computed here — the honesty layer ships only truth-free signals.)
        # PRE-HOC PROXY = structTM(template, UNGUIDED fold): a WEAK prior on whether the template
        #   is in-family. CIRCULAR — it is similarity to a fold we don't trust, and it diverges
        #   from structTM-to-truth exactly when the unguided fold is bad (the case guidance is for).
        guided_path   = guided_ref.get("path")
        unguided_path = unguided_ref.get("path")
        resolved, _terr = self._resolve_boltz_templates(inputs.get("templates"))
        per_template: List[Dict[str, Any]] = []
        for ti_t, src in zip(resolved or [], inputs.get("templates") or []):
            tpath = ti_t.get("cif") or ti_t.get("pdb")
            per_template.append({
                "label":     src.get("label") or src.get("pdb_id") or src.get("cif") or src.get("pdb"),
                "adoption":  self._usalign_tm2(guided_path, tpath),       # guided FOLLOWS template
                "prehoc_structTM_to_unguided": self._usalign_tm2(tpath, unguided_path),  # weak prior
            })
        adoptions = [p["adoption"] for p in per_template if p["adoption"] is not None]
        max_adoption = max(adoptions) if adoptions else None
        # The possible-COPYING caveat has THREE states, each with its own wording (see below):
        #  • "distant"    — a template is strongly adopted (≥0.8) AND the pre-hoc proxy
        #                   structTM(template, unguided) is LOW (< 0.5 ≈ a different fold): the
        #                   template was NOT already close, so high adoption is genuinely suspicious
        #                   (the REFINED "…did not already resemble…" wording — a measured claim).
        #  • "unmeasured" — strongly adopted but the proxy is None (US-align unavailable): we cannot
        #                   establish the template was close, so it STILL fires (conservative; never
        #                   suppress on a missing proxy) BUT with the GENERIC wording — it must not
        #                   assert the "distant" condition it never measured.
        #  • suppress     — strongly adopted but prehoc ≥ 0.5 (template already same-fold-close): the
        #                   NATURAL-success case (convergence within a fold the unguided model already
        #                   found), not copying → NOT flagged. (Adoption ALONE false-fired here; §9.)
        ADOPT_HI, PREHOC_ALREADY_CLOSE = 0.8, 0.5
        def _high(p: Dict[str, Any]) -> bool:
            a = p.get("adoption")
            return a is not None and a >= ADOPT_HI
        distant = any(_high(p) and (p.get("prehoc_structTM_to_unguided") is not None
                                    and p["prehoc_structTM_to_unguided"] < PREHOC_ALREADY_CLOSE)
                      for p in per_template)
        unmeasured = any(_high(p) and p.get("prehoc_structTM_to_unguided") is None
                         for p in per_template)
        high_adopt = distant or unmeasured
        # "distant" takes precedence: if ANY template is measurably distant-yet-adopted, the strong
        # measured claim is warranted even if another template's proxy was unavailable.
        caveat_reason = "distant" if distant else ("unmeasured" if unmeasured else None)
        data = {
            "template_label":       inputs.get("template_label"),
            "n_templates":          len(inputs.get("templates") or []),
            "force":                bool(inputs.get("force")),
            "threshold":            inputs.get("threshold"),
            "guided_mean_plddt":    g_plddt,
            "unguided_mean_plddt":  u_plddt,
            "d_plddt":              d_plddt,
            "d_plddt_by_res":       d_plddt_by_res,
            "d_flex":               d_flex,                       # per-residue (unguided − guided) Å
            "mean_d_flex":          mean_dflex,
            "n_stabilized":         n_stab,
            "n_loosened":           n_loose,
            "n_residues":           len(d_flex),
            "per_template":         per_template,                 # adoption + pre-hoc proxy each
            "max_adoption":         max_adoption,
            "high_adoption_caveat": high_adopt,
            "high_adoption_caveat_reason": caveat_reason,   # "distant" | "unmeasured" | None (wording)
            "guided_model_id":      str(guided_ref.get("model_id")),
            "unguided_model_id":    str(unguided_ref.get("model_id")),
            "n_floor_seeds":        ung.get("n_floor_seeds"),
        }
        # ── HONEST readout — only USE-TIME-knowable signals; NEVER "rescue confirmed" ──────────
        plddt_txt = (f"confidence pLDDT {u_plddt:.1f}→{g_plddt:.1f} (Δ{d_plddt:+.1f})"
                     if d_plddt is not None else "pLDDT n/a")
        flex_txt = (f"cross-seed variation {n_stab}/{len(d_flex)} residues tightened "
                    f"(mean Δ {mean_dflex:+.2f} Å)" if d_flex else "no residue overlap")
        adopt_txt = (f"fold adopted the template(s) at {max_adoption:.0%}" if max_adoption is not None
                     else "adoption n/a")
        nlab = inputs.get("template_label") or f"{len(inputs.get('templates') or [])} template(s)"
        if caveat_reason == "distant":          # measured: template was NOT already close
            caveat = ("  ⚠ HIGH adoption of a template the unguided fold did NOT already resemble — "
                      "guidance may be IMPOSING the template fold rather than the construct "
                      "independently converging; without an experimental structure this cannot be "
                      "ruled out (copying vs unlocking is truth-dependent).")
        elif caveat_reason == "unmeasured":     # fired conservatively; proxy unavailable — generic
            caveat = ("  ⚠ HIGH adoption — the fold may be FOLLOWING the template rather than "
                      "independently converging; without an experimental structure this cannot be "
                      "ruled out (copying vs unlocking is truth-dependent).")
        else:
            caveat = ""
        summary = (f"Template assist ({nlab}): {plddt_txt}; {flex_txt}; {adopt_txt}. "
                   f"These are the USE-TIME-knowable effects — NOT a confirmation of correctness "
                   f"(that needs an experimental structure). Guided confidence is template-biased; "
                   f"shown against the unguided baseline.{caveat}")
        return ToolStepResult(tool="template_assist", success=True, data=data, summary=summary,
                              elapsed_ms=(_time.perf_counter() - t0) * 1000)

    def _usalign_tm2(self, query_path: Optional[str], ref_path: Optional[str]) -> Optional[float]:
        """US-align *query* onto *ref* (LOCAL-ONLY WSL binary) → the reference-normalized TM
        (tm2), or None. Reused by the honesty layer for adoption (guided-vs-template) and the
        pre-hoc proxy (template-vs-unguided). Same engine as `_run_structural_align`."""
        import os as _os, shlex as _shlex, config as _cfg
        if not (query_path and ref_path and _os.path.isfile(query_path) and _os.path.isfile(ref_path)):
            return None
        try:
            from wsl_bridge import WSLBridge
            wsl = WSLBridge(distribution=getattr(_cfg, "USALIGN_WSL_DISTRO", "Ubuntu-24.04"))
            if not wsl.is_available():
                return None
            q = wsl.translate_path(_os.path.abspath(query_path))
            r = wsl.translate_path(_os.path.abspath(ref_path))
            exe = getattr(_cfg, "USALIGN_EXE", "/home/andre/USalign/USalign")
            cmd = f"{_shlex.quote(exe)} {_shlex.quote(q)} {_shlex.quote(r)} -outfmt 2 -m -"
            res = wsl.run_command(cmd, timeout=getattr(_cfg, "USALIGN_TIMEOUT", 120))
            if not res.get("ok"):
                return None
            parsed = self._parse_usalign_output(res.get("stdout", ""))
            return round(parsed["tm2"], 4) if parsed and parsed.get("tm2") is not None else None
        except Exception:
            return None

    @staticmethod
    def _dev_sort_key(k: "Any"):
        """Sort a deviation key — ``"123"`` (monomer resno) or ``"A:123"`` (chain:resno)."""
        s = str(k)
        if ":" in s:
            ch, rn = s.split(":", 1)
            try:
                return (ch, int(rn))
            except ValueError:
                return (ch, 0)
        try:
            return ("", int(s))
        except ValueError:
            return ("", 0)

    # ══════════════════════════════════════════════════════════════════════════
    # Validate-design meta-tool (thin orchestrator — NOT a new bridge)
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_validate_design_intent(self, text: str) -> bool:
        """
        True ONLY for explicit high-accuracy validation phrasing:
          * an explicit qualifier (full / high-accuracy / high-confidence /
            thorough validation, or '<qualifier> validate'), OR
          * a 'validate ... with colabfold/alphafold' request.
        Bare 'validate design' / 'check this design' return False (they revert to
        the light mpnn_esmfold / ESMFold fast screen).
        """
        if not text:
            return False
        low = text.lower()
        if any(kw in low for kw in self._VALIDATE_DESIGN_KEYWORDS):
            return True
        # "validate ... with colabfold/alphafold" → explicit high-accuracy path.
        if "validat" in low and ("colabfold" in low or "alphafold" in low):
            return True
        return False

    # ── Interface-stabilization intent helpers ────────────────────────────────

    @classmethod
    def _detect_interface_stabilization_intent(cls, text: str) -> bool:
        """True if *text* requests interface detection / stabilization."""
        if not text:
            return False
        low = text.lower()
        return any(kw in low for kw in cls._INTERFACE_STABILIZATION_KEYWORDS)

    def _primary_assembly_model_id(self) -> str:
        """
        Return the assembly model ID if one has been generated (e.g. '2'),
        otherwise fall back to the primary AU model ID (e.g. '1').

        Iterates generated_assemblies in reverse insertion order so the most
        recently generated assembly is preferred.
        """
        # Prefer the most recently generated assembly model
        for rec in reversed(list(self.session.generated_assemblies.values())):
            asm_mid = rec.get("assembly_model_id")
            if asm_mid:
                return str(asm_mid)
        return self._primary_model_id()

    # ── Conformer-comparison intent helpers ───────────────────────────────────

    # "… as [a/an/the] <oligomer>" — "show as trimer", "present model as a homotetramer".
    _AS_OLIGOMER_RE = re.compile(
        r"\bas\s+(?:a\s+|an\s+|the\s+)?(?:homo|hetero)?"
        r"(?:mono|di|tri|tetra|penta|hexa|hepta|octa|nona|deca)mer")
    # "<oligomer> assembly/unit/complex" — "trimeric assembly", "build the trimeric assembly".
    _OLIGOMER_ASSEMBLY_RE = re.compile(
        r"(?:mono|di|tri|tetra|penta|hexa|hepta|octa|nona|deca)mer(?:ic)?\s+"
        r"(?:assembly|unit|complex)")
    # BUILD/PRESENT verbs — gate the "biological assembly" / "<oligomer> assembly" cues so an
    # ANALYSIS verb ("detect/analyse/find the biological assembly") routes to the analyser, not the
    # builder. The `as <oligomer>` construction is inherently a build directive → not gated.
    _ASSEMBLY_PRESENT_VERB_RE = re.compile(
        r"\b(assemble|reassemble|present|display|show|view|render|build|generate|"
        r"make|create|form|expand|load|open|work)\b")
    # word → copy count, for validating a user-asserted oligomer against the deposited metadata.
    _OLIGOMER_COUNTS = {
        "monomer": 1, "dimer": 2, "trimer": 3, "tetramer": 4, "pentamer": 5,
        "hexamer": 6, "heptamer": 7, "octamer": 8, "nonamer": 9, "decamer": 10,
    }

    @classmethod
    def _detect_bio_assembly_intent(cls, text: str) -> bool:
        """True if *text* requests building the biological assembly. Beyond the explicit keyword
        list, fires on the PHRASING FAMILY (per the product spec): an explicit "biological
        assembly/unit", OR "… as <oligomer>" (show/present/display/assemble/work … as a trimer),
        OR "<oligomer> assembly/unit/complex" (trimeric assembly). PRECISE — requires an `as
        <oligomer>` connector or an `<oligomer> assembly` compound, so an incidental oligomer
        mention ("show the trimer interface") is NOT swept in. The oligomer word is OPTIONAL: a bare
        "display the biological assembly" suffices (the deposited assembly drives the oligomer)."""
        if not text:
            return False
        low = text.lower()
        if any(kw in low for kw in cls._BIO_ASSEMBLY_KEYWORDS):
            return True
        if cls._AS_OLIGOMER_RE.search(low):             # "… as <oligomer>" — inherently a build directive
            return True
        # "biological assembly/unit" or "<oligomer> assembly" — only with a BUILD/PRESENT verb, so
        # "detect/analyse the biological assembly" routes to the analyser, not the builder.
        cue = ("biological assembly" in low or "biological unit" in low
               or bool(cls._OLIGOMER_ASSEMBLY_RE.search(low)))
        if cue and cls._ASSEMBLY_PRESENT_VERB_RE.search(low):
            return True
        return False

    @classmethod
    def _parse_requested_oligomer_count(cls, text: str) -> Optional[int]:
        """The copy count a user ASSERTED (dimer→2, trimer→3, …, or 'N-mer'/'Nmer'), for validating
        against the deposited assembly metadata. None when no oligomer word is supplied (the common
        case — the oligomer then comes purely from the file). An ASSERTION to validate, NOT a build
        directive: a conflict warns, it never forces the wrong oligomer."""
        if not text:
            return None
        low = text.lower()
        for word, n in cls._OLIGOMER_COUNTS.items():
            if word in low:
                return n
        m = re.search(r"\b(\d+)\s*-?\s*mer\b", low)        # "3-mer" / "12mer"
        if m:
            return int(m.group(1))
        return None

    @staticmethod
    def _parse_bio_assembly_id(text: str) -> int:
        """Extract an explicit assembly ID from user text (e.g. 'assembly 2') → 2.
        Returns 1 (the default biological assembly) if none is found."""
        m = re.search(r"assembly\s+(\d+)", text.lower())
        if m:
            return int(m.group(1))
        return 1

    @classmethod
    def _detect_conformer_comparison_intent(cls, text: str) -> bool:
        """
        True if *text* requests a conformational-change / conformer-comparison
        analysis (anchor-restricted Kabsch + per-residue Cα shift map).
        Fires only on explicit phrasing — NOT on generic "compare" requests.
        """
        if not text:
            return False
        low = text.lower()
        if any(kw in low for kw in cls._CONFORMER_COMPARISON_KEYWORDS):
            return True
        # Compound: "compar" + ("conformer" | "conformation" | "state")
        if "compar" in low and any(w in low for w in ("conformer", "conformation", " state")):
            return True
        # Compound: "overlay" + ("anchor" | "conserved" | "rigid core")
        # Catches: "open and overlay A and B, anchoring the conserved domains/core"
        #          "overlay anchored on the conserved region"
        if "overlay" in low and any(w in low for w in ("anchor", "conserved", "rigid core")):
            return True
        # Compound: "align" + "rigid core"
        # Catches: "align A and B on the rigid core and show the shift"
        if "align" in low and "rigid core" in low:
            return True
        return False

    def _parse_conformer_comparison_options(
        self,
        user_input: str,
        translator_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Parse conformer-comparison inputs from *user_input*.

        Extracts model IDs (from ``#N`` patterns or session order),
        chain letters, and anchor residue spec (``"1-29,124-214"`` or ``"auto"``).
        """
        opts: Dict[str, Any] = {"_user_input": user_input}

        # Model IDs: first two ``#N`` refs, else first two session structures
        model_refs = re.findall(r"#(\d+)", user_input or "")
        if len(model_refs) >= 2:
            opts["model_id_a"] = model_refs[0]
            opts["model_id_b"] = model_refs[1]
        else:
            opts["model_id_a"] = self._first_model_id()
            opts["model_id_b"] = self._second_model_id()

        # Chain IDs: "all chains" | "chain X" | "chain pair X/Y"
        if re.search(r"\ball\s+chains?\b", user_input or "", re.I):
            opts["chain_a"] = "ALL"
            opts["chain_b"] = "ALL"
        else:
            ch_pair = re.search(r"chain\s+([A-Za-z])/([A-Za-z])", user_input or "", re.I)
            ch_single = re.search(r"\bchain\s+([A-Za-z])\b", user_input or "", re.I)
            if ch_pair:
                opts["chain_a"] = ch_pair.group(1).upper()
                opts["chain_b"] = ch_pair.group(2).upper()
            elif ch_single:
                opts["chain_a"] = ch_single.group(1).upper()
                opts["chain_b"] = ch_single.group(1).upper()

        # Anchor spec: "anchor on <range>", "anchor residues <range>",
        #              "core domain <range>", "anchored on <range>"
        anc_m = re.search(
            r"(?:anchor(?:ed)?(?:\s+on|\s+residues?)?|core\s+domain)\s+([\d\-,\s]+)",
            user_input or "", re.I,
        )
        if anc_m:
            raw = anc_m.group(1).strip().rstrip(",")
            # Keep only digits, hyphens, commas
            clean = re.sub(r"[^\d\-,]", "", raw)
            opts["anchor"] = clean if clean else "auto"
        else:
            opts["anchor"] = "auto"

        return opts

    def _parse_validate_design_options(self, text: str) -> Dict[str, Any]:
        """
        Parse the RMSD reference (fold preservation) and the energy reference
        (relative stability) — they are SEPARATE. Cross-topology is fine for the
        RMSD reference (matchmaker on the selected chain); the energy reference is
        explicit and only ever scored relative when topologies match.
        """
        opts: Dict[str, Any] = {}
        if not text:
            return opts
        low = text.lower()
        # RMSD reference: "vs / against / compare(d) to / relative to PDB XXXX [chain Y]"
        m = re.search(
            r"(?:vs\.?|versus|against|compared?\s+to|relative\s+to|compare\s+with|"
            r"superpose\s+(?:on|onto))\s+(?:pdb\s+)?([0-9][a-z0-9]{3})"
            r"(?:\s+chain\s+([a-z0-9]))?",
            low,
        )
        if m:
            opts["rmsd_ref"] = {
                "pdb": m.group(1).upper(),
                "chain": (m.group(2).upper() if m.group(2) else None),
            }
        # Explicit energy reference → implies a relative-stability request.
        me = re.search(r"energy\s+reference\s+(?:pdb\s+)?([0-9][a-z0-9]{3})", low)
        if me:
            opts["energy_ref"] = me.group(1).upper()
            opts["requested_relative"] = True
        # Relative-stability intent without an explicit ref id (decision will
        # still DECLINE unless a same-topology energy ref is actually resolved).
        if (("relative" in low and ("energy" in low or "stability" in low))
                or "energy delta" in low or "stability delta" in low
                or "compare energy" in low or "compare the energy" in low):
            opts["requested_relative"] = True
        return opts

    @staticmethod
    def _design_energy_decision(
        design_topology: Optional[tuple],
        ref_topology:    Optional[tuple],
        requested_relative: bool,
        ref_name: str = "the energy reference",
    ) -> Dict[str, Any]:
        """
        PURE honesty logic for the folding-energy axis (no I/O — unit-testable).

        Topology = ``(n_chains, tuple(sorted(per_chain_lengths)))``.

        Returns ``{"mode": ..., "reason": ...}`` where mode is:
          * "sanity"   — no relative number; report fold-plausibility only.
          * "relative" — topologies MATCH and a relative score was requested.
          * "declined" — relative was requested but topologies DON'T match;
                         NO relative number is emitted and a reason is given.
        """
        if not requested_relative or ref_topology is None:
            return {
                "mode": "sanity",
                "reason": (
                    "No explicit same-topology energy reference — the relaxed ref2015 "
                    "energy is reported as a fold-PLAUSIBILITY sanity signal (total REU + "
                    "per-residue density + clash check), NOT a stability-vs-reference claim."
                ),
            }
        if design_topology is not None and design_topology == ref_topology:
            return {
                "mode": "relative",
                "reason": (
                    f"Topologies match (same chain count and per-chain lengths) — relative "
                    f"ΔREU vs {ref_name} reported. Ranking-reliable; absolute magnitude "
                    f"approximate (~±2.7 kcal/mol, see PROJECT_CONTEXT §7 calibration verdict)."
                ),
            }
        d_n, d_lens = design_topology if design_topology else (0, ())
        r_n, r_lens = ref_topology
        return {
            "mode": "declined",
            "reason": (
                f"Relative stability vs {ref_name} isn't meaningful — {ref_name} is a "
                f"{r_n}-mer (chain lengths {list(r_lens)}), the design is a {d_n}-mer "
                f"(chain lengths {list(d_lens)}). ref2015 totals across different topologies "
                f"are not comparable; the per-residue energy density is given for the design "
                f"as a sanity signal instead."
            ),
        }

    @staticmethod
    def _topology_from_fold(fold: Dict[str, Any]) -> Optional[tuple]:
        """(n_chains, sorted per-chain lengths) for a ColabFold result (homo-oligomer)."""
        copies = int(fold.get("copies", 1) or 1)
        length = fold.get("length")
        if not length:
            plddt = fold.get("plddt") or {}
            length = len(plddt) // copies if plddt and copies else (len(plddt) or 0)
        if not length:
            return None
        return (copies, tuple([int(length)] * copies))

    @staticmethod
    def _topology_from_pdb(pdb_path: Optional[str]) -> Optional[tuple]:
        """(n_chains, sorted per-chain CA counts) parsed from a PDB file, or None."""
        if not pdb_path or not Path(pdb_path).is_file():
            return None
        per_chain: Dict[str, int] = {}
        seen: set = set()
        try:
            with open(pdb_path, errors="replace") as fh:
                for line in fh:
                    if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
                        ch  = line[21]
                        key = (ch, line[22:27])
                        if key in seen:
                            continue
                        seen.add(key)
                        per_chain[ch] = per_chain.get(ch, 0) + 1
        except Exception:
            return None
        if not per_chain:
            return None
        lengths = sorted(per_chain.values())
        return (len(lengths), tuple(lengths))

    def _acquire_design_fold(self, inputs: Dict[str, Any], user_input: str = "") -> Dict[str, Any]:
        """
        Get the design's ColabFold fold, REUSING an existing result instead of
        re-folding when possible (guardrail). Priority:
          1. an explicit ``colabfold_result`` dict (chaining / tests);
          2. an in-session fold for this model (no re-fold) — enriched from the
             on-disk full result.json when present;
          3. the MPNN top design (auto-pull, when the request refers to "the design"
             and no fold exists yet) → folded via the bridge (whose hash-cache reuses
             a prior fold); RETRIEVED, never re-runs MPNN;
          4. fold the given sequence via the bridge (hash-cache reused).
        """
        if isinstance(inputs.get("colabfold_result"), dict):
            r = dict(inputs["colabfold_result"])
            r.setdefault("success", True)
            r.setdefault("fold_source", "provided")
            return r

        model_id = inputs.get("model_id") or self._first_model_id()
        sequence = inputs.get("sequence")
        user_input = user_input or inputs.get("_user_input", "")

        if not sequence:
            sess = self.session.get_colabfold_results(model_id) if self.session else None
            if sess and sess.get("ranked_pdb"):
                r = dict(sess)
                r["success"] = True
                r["fold_source"] = "reused (session)"
                self._enrich_fold_from_disk(r)
                return r
            # No in-session fold → pull the MPNN top design (validate the redesign
            # WITHOUT a prior explicit fold), letting the bridge hash-cache reuse it.
            if self._refers_to_mpnn_design(user_input):
                sequence, _src = self._mpnn_top_sequence(model_id)
            if not sequence:
                return {
                    "success": False,
                    "error": (
                        "No sequence provided and no in-session ColabFold result for this "
                        "model. Fold a sequence first (e.g. 'fold <seq> with colabfold'), "
                        "redesign a chain with ProteinMPNN then 'validate the top design', "
                        "or give a sequence to validate."
                    ),
                }

        bridge = self._get_colabfold_bridge()
        r = dict(bridge.predict(
            sequence=sequence, copies=int(inputs.get("copies", 1) or 1),
            template=inputs.get("template"), quick=bool(inputs.get("quick", False)),
            label=f"design{model_id}",
        ))
        r["fold_source"] = "reused (cache)" if r.get("cached") else "folded"
        return r

    @staticmethod
    def _enrich_fold_from_disk(r: Dict[str, Any]) -> None:
        """Backfill pae/per-residue pLDDT/pTM from the on-disk full result.json
        beside the ranked PDB (the session copy is trimmed). Best-effort."""
        try:
            rp = r.get("ranked_pdb")
            if not rp:
                return
            meta = Path(rp).parent / "result.json"
            if not meta.is_file():
                return
            import json as _j
            full = _j.loads(meta.read_text(encoding="utf-8"))
            for k in ("pae", "plddt", "ptm", "iptm", "mean_plddt"):
                if r.get(k) in (None, {}, []) and full.get(k) is not None:
                    r[k] = full[k]
        except Exception:
            pass

    @staticmethod
    def _parse_model_spec(resp: dict, fallback_id: Optional[str]) -> Optional[str]:
        """Parse '#N' from a ChimeraX open response; fall back to '#fallback_id'."""
        val = (resp or {}).get("value")
        if isinstance(val, str):
            m = re.search(r"#(\d+)", val)
            if m:
                return f"#{m.group(1)}"
        return f"#{fallback_id}" if fallback_id else None

    # ── Fold-preservation helpers (pure — unit-testable) ────────────────────────

    @staticmethod
    def _parse_matchmaker_rmsds(val: str) -> Optional[Dict[str, Any]]:
        """
        Parse BOTH RMSDs + pair counts from a ChimeraX matchmaker response:
          "RMSD between N pruned atom pairs is X angstroms; (across all M pairs: Y)"
        Returns {pruned_n, pruned_rmsd, total_n, all_pairs_rmsd}. If the
        "(across all M pairs: Y)" clause is absent (no pruning occurred), falls
        back to all-pairs == pruned (total_n = pruned_n, all_pairs_rmsd = pruned).
        Returns None if no RMSD could be parsed at all.
        """
        if not val:
            return None
        mp = re.search(r"RMSD between\s+(\d+)\s+(?:pruned\s+)?atom pairs is\s+([\d.]+)", val)
        if not mp:
            mf = re.search(r"RMSD[^0-9]*([\d.]+)\s*angstrom", val, re.IGNORECASE)
            if not mf:
                return None
            x = round(float(mf.group(1)), 3)
            return {"pruned_n": None, "pruned_rmsd": x, "total_n": None, "all_pairs_rmsd": x}
        pruned_n   = int(mp.group(1))
        pruned_rmsd = round(float(mp.group(2)), 3)
        ma = re.search(r"across all\s+(\d+)\s+pairs?:?\s*([\d.]+)", val)
        if ma:
            total_n = int(ma.group(1))
            all_pairs = round(float(ma.group(2)), 3)
        else:
            total_n = pruned_n
            all_pairs = pruned_rmsd
        return {"pruned_n": pruned_n, "pruned_rmsd": pruned_rmsd,
                "total_n": total_n, "all_pairs_rmsd": all_pairs}

    @staticmethod
    def _concentration_descriptor(parsed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Free variance-concentration read from the matchmaker numbers. matchmaker's
        pruning already isolates the divergent subsection, so a big all-pairs−pruned
        gap from FEW pruned residues = "a lot of variation from a small subsection".
        Descriptive only — NO threshold pass/fail.
        """
        allp   = parsed.get("all_pairs_rmsd")
        pruned = parsed.get("pruned_rmsd")
        n      = parsed.get("pruned_n")     # core (kept) pairs
        m      = parsed.get("total_n")      # all pairs
        out: Dict[str, Any] = {
            "gap": None, "pruned_count": n, "total_count": m,
            "pruned_fraction": None, "descriptor": None,
        }
        if allp is None or pruned is None:
            out["descriptor"] = "single RMSD only (no pruned/all-pairs split available)."
            return out
        gap = round(allp - pruned, 3)
        out["gap"] = gap
        frac = None
        if n is not None and m and m > 0:
            frac = round((m - n) / m, 3)        # fraction of residues pruned out
            out["pruned_fraction"] = frac
        # Descriptive read.
        if gap < 0.3:
            out["descriptor"] = (
                f"divergence is low/uniform — all-pairs {allp:.2f} Å vs core {pruned:.2f} Å "
                f"(gap {gap:.2f} Å); the fit is even across the chain."
            )
        elif frac is not None and frac <= 0.25:
            pct = round(frac * 100)
            out["descriptor"] = (
                f"divergence is CONCENTRATED — all-pairs {allp:.2f} Å but core {pruned:.2f} Å; "
                f"~{pct}% of residues ({m - n}/{m}) drive most of the gap ({gap:.2f} Å), "
                f"i.e. a localized over-represented subsection."
            )
        else:
            out["descriptor"] = (
                f"broadly divergent — all-pairs {allp:.2f} Å vs core {pruned:.2f} Å "
                f"(gap {gap:.2f} Å) with many residues pruned"
                + (f" ({m - n}/{m})" if (n is not None and m) else "")
                + "; not a single localized pocket."
            )
        return out

    @staticmethod
    def _ca_coords(pdb_path: str, chain: Optional[str]) -> Dict[int, "Any"]:
        """{resno: (x,y,z)} for Cα atoms of *chain* (all chains if None). First
        occurrence per (chain,resno) wins (skips altlocs/duplicates)."""
        import numpy as _np
        out: Dict[int, Any] = {}
        seen: set = set()
        try:
            with open(pdb_path, errors="replace") as fh:
                for line in fh:
                    if not line.startswith(("ATOM", "HETATM")):
                        continue
                    if line[12:16].strip() != "CA":
                        continue
                    ch = line[21]
                    if chain and ch != chain:
                        continue
                    try:
                        resno = int(line[22:26])
                        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    except ValueError:
                        continue
                    key = (ch, resno)
                    if key in seen:
                        continue
                    seen.add(key)
                    out[resno] = _np.array(xyz, dtype=float)
        except Exception:
            return {}
        return out

    @classmethod
    def _per_residue_ca_deviation(
        cls,
        design_pdb:   str,
        design_chain: Optional[str],
        ref_pdb:      str,
        ref_chain:    Optional[str],
    ) -> Tuple[Optional[Dict[int, float]], Optional[float]]:
        """
        Per-residue Cα deviation (Å) of the design vs reference, over an
        independent Kabsch superposition of matched Cα (matched by residue
        number). Returns ({design_resno: deviation}, all_pairs_rmsd) or
        (None, None). The matchmaker superposition isn't exposed over REST, so
        this is an independent Cα fit — adequate for localizing the divergence.
        """
        import numpy as _np
        d = cls._ca_coords(design_pdb, design_chain)
        r = cls._ca_coords(ref_pdb, ref_chain)
        common = sorted(set(d) & set(r))
        if len(common) < 3:
            return None, None
        P = _np.array([d[i] for i in common])   # design
        Q = _np.array([r[i] for i in common])   # reference
        Pc, Qc = P - P.mean(0), Q - Q.mean(0)
        # Kabsch: rotation fitting P onto Q.
        H = Pc.T @ Qc
        U, _S, Vt = _np.linalg.svd(H)
        dsign = _np.sign(_np.linalg.det(Vt.T @ U.T))
        D = _np.diag([1.0, 1.0, dsign])
        R = Vt.T @ D @ U.T
        P_fit = Pc @ R.T
        diff = P_fit - Qc
        per_res = _np.sqrt((diff ** 2).sum(1))
        dev = {resno: round(float(per_res[i]), 3) for i, resno in enumerate(common)}
        all_pairs = round(float(_np.sqrt((per_res ** 2).mean())), 3)
        return dev, all_pairs

    @staticmethod
    def _topk_deviant(dev: Dict[int, float], chain: Optional[str], k: int = 10) -> List[Dict[str, Any]]:
        """Top-K most-deviant residues (chain, resno, deviation Å), descending."""
        if not dev:
            return []
        top = sorted(dev.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [{"chain": chain or "?", "resno": rn, "deviation": dv} for rn, dv in top]

    # Deviation → colour buckets (blue=low … red=high). Documented scale; >=3.5 Å
    # caps at red so a few large outliers don't wash out the gradient.
    _DEVIATION_BUCKETS = (
        (0.5, "blue"),
        (1.0, "cornflower blue"),
        (2.0, "white"),
        (3.5, "orange"),
        (float("inf"), "red"),
    )

    @classmethod
    def _build_deviation_color_cmds(
        cls,
        dev:         Dict[int, float],
        design_spec: str,
        chain:       Optional[str],
    ) -> Tuple[List[str], List[str]]:
        """
        Colour the design model by per-residue Cα deviation, reusing the CamSol
        grouped-run idiom (consecutive same-colour residues → one `color` command).
        blue=low … red=high (>=3.5 Å). design_spec is the live '#N' model spec.
        """
        if not dev:
            return [], []
        def _bucket(v: float) -> str:
            for hi, col in cls._DEVIATION_BUCKETS:
                if v < hi:
                    return col
            return "red"
        chain_spec = f"/{chain}" if chain else ""
        base = design_spec if "/" in design_spec else f"{design_spec}{chain_spec}"
        cmds = [f"color {base} white"]
        exps = ["Reset to white before per-residue deviation colouring"]
        runs: List[Tuple[str, List[int]]] = []
        for resno in sorted(dev):
            col = _bucket(dev[resno])
            if runs and runs[-1][0] == col:
                runs[-1][1].append(resno)
            else:
                runs.append((col, [resno]))
        for col, resnos in runs:
            if col == "white":
                continue
            if len(resnos) > 1 and resnos == list(range(resnos[0], resnos[-1] + 1)):
                spec = f":{resnos[0]}-{resnos[-1]}"
            else:
                spec = ":" + ",".join(str(r) for r in resnos)
            cmds.append(f"color {design_spec}{spec} {col}")
            exps.append(f"Colour {spec} {col} (Cα deviation bucket)")
        return cmds, exps

    def _matchmaker_rmsd_live(
        self,
        fold:   Dict[str, Any],
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Open the predicted model, apply ColabFold-style pLDDT viz, and superpose
        it onto the RMSD reference with matchmaker; parse and return Cα RMSD.

        RMSD reference: default = the loaded session primary model; overridable to
        a named PDB with chain selection (cross-topology is fine here). Degrades
        gracefully (rmsd=None + note) when ChimeraX or a reference is unavailable.
        Runs live during execute (the RMSD value needs a live run); the issued
        commands are returned for transparency.
        """
        out: Dict[str, Any] = {"rmsd_ca": None, "reference": None, "commands": [], "note": None}
        ranked = fold.get("ranked_pdb")
        if self.bridge is None:
            out["note"] = "ChimeraX bridge unavailable — RMSD skipped."
            return out
        if not ranked or not Path(str(ranked)).is_file():
            out["note"] = "No ranked PDB available — RMSD skipped."
            return out

        cmds: List[str] = []
        try:
            # Open + colour the predicted model (ColabFold viz, reused).
            design_guess = str(self.session.next_model_id()) if self.session else None
            open_cmd = f'open "{Path(ranked).as_posix()}"'
            r = self.bridge.run_command(open_cmd)
            cmds.append(open_cmd)
            design_spec = self._parse_model_spec(r, design_guess)
            if design_spec:
                # Structure-only: pLDDT colour the model, but NO Sequence Viewer
                # (sequence lives in the StructureBot window). Removed 2026-06-16.
                c = f"color byattribute bfactor {design_spec} palette alphafold"
                self.bridge.run_command(c)
                cmds.append(c)

            # Resolve the reference spec AND a LOCAL reference PDB path (the
            # latter for the offline per-residue Cα Kabsch).
            rmsd_ref      = inputs.get("rmsd_ref")
            ref_spec      = None
            ref_pdb_path  = None
            ref_chain     = None
            design_chain  = inputs.get("design_chain", "A")  # ColabFold monomer → chain A
            if rmsd_ref and rmsd_ref.get("pdb"):
                ref_pdb_path = self._download_pdb_by_id(rmsd_ref["pdb"])
                if not ref_pdb_path:
                    out["note"] = f"Could not fetch RMSD reference {rmsd_ref['pdb']}; RMSD skipped."
                    out["commands"] = cmds
                    return out
                ropen = f'open "{Path(ref_pdb_path).as_posix()}"'
                rr = self.bridge.run_command(ropen)
                cmds.append(ropen)
                ref_model = self._parse_model_spec(rr, None)
                ref_chain = rmsd_ref.get("chain")
                ref_spec = (f"{ref_model}/{ref_chain}" if (ref_model and ref_chain) else ref_model)
                out["reference"] = (
                    f"{rmsd_ref['pdb']}" + (f" chain {ref_chain}" if ref_chain else "")
                )
            else:
                # Default: the loaded primary model (assumed already open).
                primary = self._primary_model_id()
                if primary and design_spec and str(primary) != design_spec.lstrip("#") \
                        and self.session and self.session.get_structure(primary):
                    ref_chain = inputs.get("chain")
                    ref_spec = f"#{primary}/{ref_chain}" if ref_chain else f"#{primary}"
                    out["reference"] = f"loaded model #{primary}" + (f" chain {ref_chain}" if ref_chain else "")
                    ref_pdb_path = self._ensure_pdb_file(primary)
                else:
                    out["note"] = (
                        "No RMSD reference (no loaded structure and none specified) — "
                        "fold preservation not assessed."
                    )

            if design_spec and ref_spec:
                mm = f"matchmaker {design_spec} to {ref_spec}"
                mr = self.bridge.run_command(mm)
                cmds.append(mm)
                val = (mr or {}).get("value") or ""
                parsed = self._parse_matchmaker_rmsds(val)
                if parsed:
                    # HEADLINE = all-pairs (honest "did the fold drift overall").
                    out["rmsd_ca"]        = parsed["all_pairs_rmsd"]
                    out["all_pairs_rmsd"] = parsed["all_pairs_rmsd"]
                    out["pruned_rmsd"]    = parsed["pruned_rmsd"]
                    out["pruned_n"]       = parsed["pruned_n"]
                    out["total_n"]        = parsed["total_n"]
                    conc = self._concentration_descriptor(parsed)
                    out["gap"]             = conc["gap"]
                    out["pruned_fraction"] = conc["pruned_fraction"]
                    out["concentration"]   = conc["descriptor"]
                else:
                    out["note"] = "matchmaker ran but RMSD could not be parsed from the response."

                # ── Localize: per-residue Cα deviation + colour the design model ──
                if ref_pdb_path and Path(str(ref_pdb_path)).is_file():
                    dev, dev_allpairs = self._per_residue_ca_deviation(
                        ranked, design_chain, ref_pdb_path, ref_chain
                    )
                    if dev:
                        out["per_residue_deviation"] = dev
                        out["per_residue_all_pairs_rmsd"] = dev_allpairs
                        out["top_deviant_residues"] = self._topk_deviant(dev, design_chain, k=10)
                        ccmds, _cexp = self._build_deviation_color_cmds(
                            dev, design_spec, design_chain
                        )
                        for c in ccmds:
                            self.bridge.run_command(c)
                            cmds.append(c)
                        out["colored_by_deviation"] = bool(ccmds)
                    else:
                        out["per_residue_note"] = (
                            "per-residue deviation unavailable (could not match >=3 Cα "
                            "between design and reference chains)."
                        )
                else:
                    out["per_residue_note"] = (
                        "per-residue deviation skipped (no local reference PDB)."
                    )

                self.bridge.run_command(f"view {design_spec}")
                cmds.append(f"view {design_spec}")
        except Exception as exc:
            out["note"] = f"RMSD step error: {exc}"
        out["commands"] = cmds
        return out

    def _run_validate_design(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> "ToolStepResult":
        """
        Thin orchestrator: fold confidence (ColabFold, reused if present) +
        fold preservation (matchmaker RMSD) + folding energy (Rosetta relax +
        ref2015, HONEST: sanity by default, relative only on same-topology, decline
        cross-topology). EVIDENCE-RICH report; no binary verdict. Stored in
        session_state.validate_design_results.
        """
        import time as _time
        user_input = user_input or inputs.get("_user_input", "")
        model_id   = inputs.get("model_id") or self._first_model_id()
        t0 = _time.perf_counter()

        # ── 1. Fold confidence (reuse-or-fold) ────────────────────────────────────
        fold = self._acquire_design_fold(inputs, user_input=user_input)
        if not fold.get("success"):
            return ToolStepResult(
                tool="validate_design", success=False,
                error=fold.get("error", "could not obtain a fold for the design"),
                data={"oom_risk": fold.get("oom_risk", False)},
            )

        plddt_map  = fold.get("plddt") or {}
        mean_plddt = fold.get("mean_plddt") or (
            round(sum(plddt_map.values()) / len(plddt_map), 2) if plddt_map else None
        )
        fold_flags: List[str] = []
        if isinstance(mean_plddt, (int, float)) and mean_plddt < 70:
            fold_flags.append(f"low mean pLDDT ({mean_plddt:.1f} < 70) — low-confidence fold")
        fold_conf = {
            "label":            "AF2 fold confidence (ColabFold)",
            "fold_source":      fold.get("fold_source"),
            "mean_plddt":       mean_plddt,
            "per_residue_plddt": plddt_map or None,
            "pae_available":    fold.get("pae") is not None,
            "ptm":              fold.get("ptm"),
            "iptm":             fold.get("iptm"),
            "ranked_pdb":       fold.get("ranked_pdb"),
            "flags":            fold_flags,
        }

        # ── 2. Fold preservation (matchmaker all-pairs RMSD + concentration + where) ─
        rmsd = self._matchmaker_rmsd_live(fold, inputs)
        rmsd_flags: List[str] = []
        if isinstance(rmsd.get("rmsd_ca"), (int, float)) and rmsd["rmsd_ca"] > 4.0:
            rmsd_flags.append(
                f"high all-pairs Cα RMSD ({rmsd['rmsd_ca']:.2f} Å > 4 Å) — fold may not be preserved"
            )
        _gap = rmsd.get("gap")
        _frac = rmsd.get("pruned_fraction")
        if isinstance(_gap, (int, float)) and _gap >= 0.3 and isinstance(_frac, (int, float)) and _frac <= 0.25:
            rmsd_flags.append(
                f"divergence concentrated in ~{round(_frac*100)}% of residues "
                f"(all-pairs−core gap {_gap:.2f} Å)"
            )
        fold_pres = {
            "label":               "Fold preservation — matchmaker Cα RMSD vs reference (headline = all-pairs)",
            "rmsd_ca":             rmsd.get("rmsd_ca"),            # all-pairs (headline)
            "all_pairs_rmsd":      rmsd.get("all_pairs_rmsd"),
            "pruned_rmsd":         rmsd.get("pruned_rmsd"),        # core (pruned) — flatters agreement
            "pruned_n":            rmsd.get("pruned_n"),
            "total_n":             rmsd.get("total_n"),
            "pruned_fraction":     rmsd.get("pruned_fraction"),
            "gap":                 rmsd.get("gap"),
            "concentration":       rmsd.get("concentration"),
            "top_deviant_residues": rmsd.get("top_deviant_residues"),
            "per_residue_deviation": rmsd.get("per_residue_deviation"),
            "colored_by_deviation": rmsd.get("colored_by_deviation", False),
            "reference":           rmsd.get("reference"),
            "note":                rmsd.get("note") or rmsd.get("per_residue_note"),
            "flags":               rmsd_flags,
        }

        # ── 3. Folding energy (HONEST) ────────────────────────────────────────────
        rosetta = self._get_rosetta_bridge()
        energy_flags: List[str] = []
        energy: Dict[str, Any] = {
            "label": "Folding energy (Rosetta FastRelax + ref2015)",
            "mode":  "sanity",
        }
        ranked_pdb = fold.get("ranked_pdb")
        score = rosetta.relax_and_score(ranked_pdb) if ranked_pdb else {"success": False, "error": "no ranked PDB"}

        # Decide sanity / relative / declined (pure logic).
        requested_relative = bool(inputs.get("requested_relative", False))
        energy_ref = inputs.get("energy_ref")
        ref_topo = None
        ref_name = str(energy_ref) if energy_ref else "the energy reference"
        energy_ref_pdb = None
        if requested_relative and energy_ref:
            energy_ref_pdb = (energy_ref if Path(str(energy_ref)).is_file()
                              else self._download_pdb_by_id(str(energy_ref)))
            ref_topo = self._topology_from_pdb(energy_ref_pdb)
        decision = self._design_energy_decision(
            self._topology_from_fold(fold), ref_topo, requested_relative, ref_name
        )
        energy["mode"]   = decision["mode"]
        energy["reason"] = decision["reason"]

        if not score.get("success"):
            energy["available"] = False
            energy["error"]     = score.get("error")
            energy_flags.append("Rosetta relax/score unavailable — folding-energy axis not assessed")
        else:
            energy.update({
                "available":           True,
                "total_reu":           score.get("total_reu"),
                "per_residue_density": score.get("per_residue_density"),
                "fa_rep":              score.get("fa_rep"),
                "clash_ok":            score.get("clash_ok"),
                "converged":           score.get("converged"),
                "relaxed_pdb":         score.get("relaxed_pdb"),
            })
            if score.get("clash_ok") is False:
                energy_flags.append("post-relax repulsive energy is high — possible clashes / strained packing")
            # RELATIVE number ONLY on the relative path (never on declined/sanity).
            if decision["mode"] == "relative" and energy_ref_pdb:
                ref_score = rosetta.relax_and_score(energy_ref_pdb)
                if ref_score.get("success"):
                    delta = round(score["total_reu"] - ref_score["total_reu"], 3)
                    energy["relative_delta_reu"]  = delta
                    energy["reference_total_reu"] = ref_score["total_reu"]
                    energy["energy_reference"]    = ref_name
                    energy["relative_note"] = (
                        "ΔREU = design − reference (negative = design lower-energy). "
                        "Ranking-reliable; magnitude approximate (~±2.7, §7)."
                    )
                    if delta > 5.0:
                        energy_flags.append(
                            f"design is much higher-energy than {ref_name} (ΔREU {delta:+.1f})"
                        )
                else:
                    # Could not score the reference → downgrade to sanity, stay honest.
                    energy["mode"] = "sanity"
                    energy["reason"] = (
                        f"Relative comparison requested and topologies matched, but {ref_name} "
                        f"could not be scored ({ref_score.get('error')}); reporting the design's "
                        "sanity signal only."
                    )
        energy["flags"] = energy_flags

        # ── Assemble evidence-rich report (no binary verdict) ─────────────────────
        report = {
            "success":          True,
            "design": {
                "sequence_length": fold.get("length"),
                "copies":          fold.get("copies", 1),
                "fold_source":     fold.get("fold_source"),
            },
            "fold_confidence":  fold_conf,
            "fold_preservation": fold_pres,
            "folding_energy":   energy,
            "artifacts": {
                "ranked_pdb":  fold.get("ranked_pdb"),
                "relaxed_pdb": energy.get("relaxed_pdb"),
                "png_paths":   fold.get("png_paths", {}),
            },
            "viz_applied": bool(rmsd.get("commands")),
            "commands":    rmsd.get("commands", []),
            "elapsed_s":   round(_time.perf_counter() - t0, 1),
        }
        if self.session:
            self.session.set_validate_design_results(model_id, report)

        # Auto-open the confidence-map PNGs (best-effort).
        self._open_pngs(fold.get("png_paths", {}))

        # ── Evidence-rich multi-line summary (honesty labels, NO pass/fail) ───────
        lines: List[str] = ["Validate-design report (evidence — not a verdict):"]
        _mp = f"{mean_plddt:.1f}" if isinstance(mean_plddt, (int, float)) else "n/a"
        _ptm = fold.get("ptm"); _iptm = fold.get("iptm")
        lines.append(
            f"  • Fold confidence [{fold.get('fold_source')}]: mean pLDDT {_mp}"
            + (f", pTM {_ptm:.2f}" if isinstance(_ptm, (int, float)) else "")
            + (f", ipTM {_iptm:.2f}" if isinstance(_iptm, (int, float)) else "")
            + (f", PAE matrix available" if fold.get("pae") is not None else "")
        )
        if fold_pres["rmsd_ca"] is not None:
            _pru = fold_pres.get("pruned_rmsd")
            _pru_s = f" / core(pruned) {_pru:.2f} Å" if isinstance(_pru, (int, float)) else ""
            lines.append(
                f"  • Fold preservation: all-pairs Cα RMSD {fold_pres['rmsd_ca']:.2f} Å"
                f"{_pru_s} vs {fold_pres['reference']}"
            )
            if fold_pres.get("concentration"):
                lines.append(f"      ↳ {fold_pres['concentration']}")
            _topk = fold_pres.get("top_deviant_residues") or []
            if _topk:
                _shown = ", ".join(f"{d['chain']}{d['resno']} {d['deviation']:.1f}Å" for d in _topk[:5])
                lines.append(f"      ↳ most-deviant residues: {_shown}"
                             + (" …" if len(_topk) > 5 else "")
                             + ("  [design coloured by per-residue deviation in ChimeraX]"
                                if fold_pres.get("colored_by_deviation") else ""))
        else:
            lines.append(f"  • Fold preservation: not assessed — {fold_pres['note']}")
        if energy.get("available"):
            if energy["mode"] == "relative" and "relative_delta_reu" in energy:
                lines.append(
                    f"  • Folding energy [RELATIVE]: ΔREU {energy['relative_delta_reu']:+.1f} vs "
                    f"{energy.get('energy_reference')} (total {energy['total_reu']:.1f} REU; "
                    f"ranking-reliable, magnitude approximate). {energy['reason']}"
                )
            elif energy["mode"] == "declined":
                lines.append(
                    f"  • Folding energy [SANITY — relative DECLINED]: total {energy['total_reu']:.1f} REU, "
                    f"density {energy['per_residue_density']:.2f} REU/res, "
                    f"clash_ok={energy['clash_ok']}. {energy['reason']}"
                )
            else:
                lines.append(
                    f"  • Folding energy [SANITY]: total {energy['total_reu']:.1f} REU, "
                    f"density {energy['per_residue_density']:.2f} REU/res, "
                    f"clash_ok={energy['clash_ok']}. {energy['reason']}"
                )
        else:
            lines.append(f"  • Folding energy: unavailable — {energy.get('error')}")
        _allflags = fold_flags + rmsd_flags + energy_flags
        if _allflags:
            lines.append("  • Flags: " + "; ".join(_allflags))
        summary = "\n".join(lines)

        return ToolStepResult(
            tool="validate_design", success=True,
            data=report,
            viz_commands=[],            # viz already applied live (with the RMSD run)
            viz_explanations=[],
            summary=summary,
            elapsed_ms=(_time.perf_counter() - t0) * 1000,
        )

    # ── Bridge accessors (lazy init) ───────────────────────────────────────────

    def _get_assembly_analyser(self):
        if self._assembly_analyser is None:
            from assembly_analyser import AssemblyAnalyser
            self._assembly_analyser = AssemblyAnalyser(
                bridge  = self.bridge,
                session = self.session,
            )
        return self._assembly_analyser

    def _get_disulfide_bridge(self):
        if self._disulfide_bridge is None:
            from disulfide_bridge import DisulfideBridge
            self._disulfide_bridge = DisulfideBridge(chimerax_bridge=self.bridge)
        return self._disulfide_bridge

    def _get_proline_bridge(self):
        if self._proline_bridge is None:
            from proline_bridge import ProlineBridge
            self._proline_bridge = ProlineBridge()
        return self._proline_bridge

    def _get_glycan_bridge(self):
        if self._glycan_bridge is None:
            from glycan_bridge import GlycanBridge
            self._glycan_bridge = GlycanBridge()
        return self._glycan_bridge

    def _get_netnglyc_bridge(self):
        if self._netnglyc_bridge is None:
            from netnglyc_bridge import NetNGlycBridge
            self._netnglyc_bridge = NetNGlycBridge()
        return self._netnglyc_bridge

    def _get_salt_bridge_bridge(self):
        if self._salt_bridge_bridge is None:
            from salt_bridge_bridge import SaltBridgeBridge
            self._salt_bridge_bridge = SaltBridgeBridge()
        return self._salt_bridge_bridge

    def _get_cavity_bridge(self):
        if self._cavity_bridge is None:
            from cavity_bridge import CavityBridge
            self._cavity_bridge = CavityBridge()
        return self._cavity_bridge

    def _get_double_mutant_bridge(self):
        if self._double_mutant_bridge is None:
            from double_mutant_bridge import DoubleMutantBridge
            self._double_mutant_bridge = DoubleMutantBridge()
        return self._double_mutant_bridge

    def _get_colabfold_bridge(self):
        if self._colabfold_bridge is None:
            from colabfold_bridge import ColabFoldBridge
            self._colabfold_bridge = ColabFoldBridge()
        return self._colabfold_bridge

    def _get_camsol_bridge(self):
        if self._camsol_bridge is None:
            from camsol_bridge import CamsolBridge
            self._camsol_bridge = CamsolBridge()
        return self._camsol_bridge

    def _get_esm_bridge(self):
        if self._esm_bridge is None:
            from esm_bridge import EsmBridge
            self._esm_bridge = EsmBridge()
        return self._esm_bridge

    def _get_esmfold_bridge(self):
        if self._esmfold_bridge is None:
            from esmfold_bridge import ESMFoldBridge
            self._esmfold_bridge = ESMFoldBridge()
        return self._esmfold_bridge

    def _get_boltz_bridge(self):
        if self._boltz_bridge is None:
            from boltz_bridge import BoltzBridge
            self._boltz_bridge = BoltzBridge()
        return self._boltz_bridge

    def _get_proteinmpnn_bridge(self):
        if self._proteinmpnn_bridge is None:
            from proteinmpnn_bridge import ProteinMPNNBridge
            self._proteinmpnn_bridge = ProteinMPNNBridge()
        return self._proteinmpnn_bridge

    def _get_mpnn_esmfold_pipeline(self):
        if self._mpnn_esmfold_pipeline is None:
            from mpnn_esmfold_pipeline import MPNNESMFoldPipeline
            self._mpnn_esmfold_pipeline = MPNNESMFoldPipeline()
        return self._mpnn_esmfold_pipeline

    def _get_rfdiffusion_bridge(self):
        if self._rfdiffusion_bridge is None:
            from rfdiffusion_bridge import RFdiffusionBridge
            self._rfdiffusion_bridge = RFdiffusionBridge()
        return self._rfdiffusion_bridge

    def _get_rosetta_bridge(self):
        if self._rosetta_bridge is None:
            from rosetta_bridge import RosettaBridge
            self._rosetta_bridge = RosettaBridge()
        return self._rosetta_bridge

    # ── PDB file retrieval ────────────────────────────────────────────────────

    def _ensure_pdb_file(self, model_id: str) -> Optional[str]:
        """
        Get or download the PDB file for a loaded structure.

        Priority:
          1. session.structures[model_id]["path"] (if file exists locally)
          2. Download from RCSB if the structure name is a 4-char PDB ID
             → cached to cache/<ID>.pdb

        Returns the local path string, or None on failure.
        """
        info = self.session.get_structure(model_id)
        if not info:
            return None

        # Check for a cached local path
        path = info.get("path")
        if path and Path(path).is_file():
            return path

        # Try downloading from RCSB for known PDB IDs
        name = info.get("name", "")
        if not re.match(r"^[A-Za-z0-9]{4}$", name):
            return None

        cache_dir  = Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / f"{name.upper()}.pdb"

        if local_path.is_file():
            info["path"] = str(local_path)
            return str(local_path)

        try:
            import requests
            url  = f"https://files.rcsb.org/download/{name.upper()}.pdb"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
            info["path"] = str(local_path)
            return str(local_path)
        except Exception:
            return None

    # ── Sequence retrieval ─────────────────────────────────────────────────────

    def _fetch_sequence(
        self,
        model_id: str,
        chain:    Optional[str] = None,
    ) -> Optional[str]:
        """
        Attempt to get an amino-acid sequence for a loaded structure.

        Priority:
          1. Session state cache (sequences stored after first fetch)
          2. RCSB FASTA API (if the structure name is a 4-char PDB ID)
          3. ChimeraX 'sequence chain' command (may not return text via REST)
        """
        info = self.session.get_structure(model_id)
        if info:
            meta = info.get("metadata", {})
            # Check cached sequences dict
            if chain and isinstance(meta.get("sequences"), dict):
                seq = meta["sequences"].get(chain)
                if seq:
                    return seq
            elif meta.get("sequence"):
                return meta["sequence"]

            # Try RCSB for 4-letter PDB IDs
            name = info.get("name", "")
            if re.match(r"^[A-Za-z0-9]{4}$", name):
                seq = self._fetch_rcsb_fasta(name, chain)
                if seq:
                    # Cache for next time
                    if "sequences" not in meta:
                        meta["sequences"] = {}
                    key = chain or "A"
                    meta["sequences"][key] = seq
                    return seq

        # Last resort: ask ChimeraX
        return self._fetch_sequence_from_chimerax(model_id, chain)

    def _fetch_rcsb_fasta(
        self,
        pdb_id: str,
        chain:  Optional[str] = None,
    ) -> Optional[str]:
        """Fetch the amino-acid FASTA sequence from RCSB for a PDB ID + chain."""
        try:
            import requests
            url  = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}/display"
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                return None
            # Parse multi-entry FASTA; find the matching chain
            sequences = self._parse_fasta(resp.text)
            if not sequences:
                return None
            if chain:
                # RCSB FASTA headers contain the chain letter, e.g.:
                # >1HSG_1|Chain A|...
                for header, seq in sequences.items():
                    if f"|Chain {chain.upper()}|" in header or f"Chain {chain.upper()}" in header:
                        return seq
            # Fall back to first sequence
            return next(iter(sequences.values()))
        except Exception:
            return None

    @staticmethod
    def _parse_fasta(text: str) -> Dict[str, str]:
        """Parse a FASTA string into {header: sequence} dict."""
        sequences: Dict[str, str] = {}
        current_header = ""
        current_seq:    List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_seq:
                    sequences[current_header] = "".join(current_seq)
                current_header = line[1:]
                current_seq    = []
            else:
                current_seq.append(line)
        if current_header and current_seq:
            sequences[current_header] = "".join(current_seq)
        return sequences

    def _fetch_sequence_from_chimerax(
        self,
        model_id: str,
        chain:    Optional[str] = None,
    ) -> Optional[str]:
        """
        Best-effort last-resort: read a model/chain's sequence from ChimeraX via a
        RUNSCRIPT over the residues' one-letter codes — NOT the `sequence chain`
        command, which would pop the Sequence Viewer (StructureBot keeps ChimeraX
        structure-only; sequence viewing lives in the StructureBot window). Returns
        the standard-AA sequence, or None.
        """
        if not self.bridge.is_running():
            return None
        import tempfile as _tf
        import os as _os
        try:
            script = (
                "from chimerax.atomic import AtomicStructure\n"
                f"_mid = {str(model_id)!r}\n"
                f"_chain = {(chain or '')!r}\n"
                "_models = {m.id_string: m for m in session.models "
                "if isinstance(m, AtomicStructure)}\n"
                "_m = _models.get(_mid)\n"
                "_seq = []\n"
                "if _m:\n"
                "    for _ch in _m.chains:\n"
                "        if _chain and _ch.chain_id != _chain:\n"
                "            continue\n"
                "        for _r in _ch.residues:\n"
                "            if _r is None:\n"
                "                continue\n"
                "            _olc = getattr(_r, 'one_letter_code', None)\n"
                "            if _olc:\n"
                "                _seq.append(_olc)\n"
                "print('SEQ:' + ''.join(_seq))\n"
            )
            with _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                        encoding="utf-8") as sf:
                sf.write(script)
                script_path = sf.name
            try:
                result = self.bridge.run_command(f'runscript "{script_path}"')
            finally:
                try:
                    _os.unlink(script_path)
                except Exception:
                    pass
        except Exception:
            return None
        value = (result or {}).get("value") if isinstance(result, dict) else None
        if (result or {}).get("error") or not isinstance(value, str):
            return None
        raw = ""
        for line in value.splitlines():
            if line.strip().startswith("SEQ:"):
                raw = line.split("SEQ:", 1)[1].strip()
                break
        seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "", raw.upper())
        return seq if len(seq) >= 5 else None

    # ── Active-site command handler ────────────────────────────────────────────

    # Patterns for active-site management commands (matched case-insensitively).
    _ACTIVE_SITE_SET_RE   = re.compile(
        r"^\s*set\s+active[\s\-]?site\s+residues?\s+([\d\s,]+)\s*$", re.I
    )
    _ACTIVE_SITE_CLEAR_RE = re.compile(
        r"^\s*clear\s+active[\s\-]?site\s+residues?\s*$", re.I
    )
    _ACTIVE_SITE_SHOW_RE  = re.compile(
        r"^\s*show\s+active[\s\-]?site\s+residues?\s*$", re.I
    )

    def handle_active_site_command(self, user_input: str) -> Optional[str]:
        """
        Handle active-site residue management commands without LLM translation.

        Supported patterns:
          "set active site residues 25 26 27"  → stores {25, 26, 27} in session
          "clear active site residues"          → clears the stored set
          "show active site residues"           → prints current set

        Returns a human-readable result string if the command was handled,
        or None if the input is not an active-site command (caller should
        fall through to LLM translation).
        """
        text = user_input.strip()

        m = self._ACTIVE_SITE_SET_RE.match(text)
        if m:
            nums_str = m.group(1)
            nums = {
                int(x)
                for x in re.split(r"[\s,]+", nums_str.strip())
                if x.strip()
            }
            self.session.set_functional_residues(nums)
            return (
                f"Active-site residues set: {sorted(nums)}.\n"
                f"All subsequent proline scans will exclude positions "
                f"within 2 residues of these sites."
            )

        if self._ACTIVE_SITE_CLEAR_RE.match(text):
            self.session.set_functional_residues(set())
            return "Active-site residues cleared — proline scans will use SASA auto-detection."

        if self._ACTIVE_SITE_SHOW_RE.match(text):
            current = self.session.get_functional_residues()
            if current:
                return f"Active-site residues: {sorted(current)}."
            return "No active-site residues declared (SASA auto-detection will be used)."

        return None   # not an active-site command

    # ── Sequence display command handler ──────────────────────────────────────

    def _export_sequences_fasta(self) -> str:
        """Export ProteinMPNN designed sequences to a FASTA file on the desktop."""
        import config as _cfg
        model_id  = self._primary_model_id()
        mpnn_data = self.session.get_proteinmpnn_result(model_id)
        if not mpnn_data:
            return (
                "No ProteinMPNN sequences in session -- run "
                "'redesign chain A with ProteinMPNN' first."
            )
        sequences = mpnn_data.get("sequences", [])
        wildtype  = mpnn_data.get("wildtype_sequence", "")

        try:
            desktop = _cfg.desktop_path()
        except Exception:
            desktop = Path.home() / "Desktop"

        out_path = Path(desktop) / "proteinmpnn_designs.fasta"
        lines: list = []

        if wildtype:
            lines.append(f">wildtype length={len(wildtype)}")
            lines.append(wildtype)

        for i, entry in enumerate(sequences, 1):
            seq       = entry.get("sequence", "")
            score     = entry.get("score", 0.0)
            recovery  = entry.get("recovery", 0.0)
            mutations = entry.get("mutations", [])
            n_muts    = len(mutations)
            lines.append(
                f">design_{i} score={score:.3f} "
                f"recovery={recovery:.3f} mutations={n_muts}"
            )
            lines.append(seq)

        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return f"Sequences saved to {out_path}"

    # ── Live-selection commands (the ChimeraX selection as a first-class input) ─

    _SELECTION_REPORT_PHRASES = (
        "what's selected", "whats selected", "what is selected",
        "what residues are selected", "what's the selection", "whats the selection",
        "show the selection", "show selection", "describe the selection",
        "describe selection", "report the selection", "current selection",
        "list the selection", "the current selection",
    )
    _SELECTION_SCAN_PHRASES = (
        "scan the selection", "scan selection", "scan the selected",
        "analyze the selection", "analyse the selection",
        "analyze the selected", "analyse the selected",
        "solubility of the selection", "camsol the selection",
        "score the selection", "score the selected",
    )
    _SELECTION_REDESIGN_PHRASES = (
        "redesign the selection", "redesign the selected", "redesign selection",
        "design the selection", "design the selected residues",
        "mpnn the selection", "mpnn the selected",
    )

    def _detect_selection_intent(self, user_input: str) -> Optional[str]:
        """Return 'report' | 'scan' | 'redesign' for a selection-consuming phrasing,
        else None (the caller falls through to the LLM)."""
        low = user_input.lower().strip()
        if any(p in low for p in self._SELECTION_REPORT_PHRASES):
            return "report"
        if any(p in low for p in self._SELECTION_SCAN_PHRASES):
            return "scan"
        if any(p in low for p in self._SELECTION_REDESIGN_PHRASES):
            return "redesign"
        return None

    def handle_selection_command(self, user_input: str) -> Optional[str]:
        """
        Poll-on-command fast-path: when the user asks to act on "the selection",
        read the LIVE ChimeraX selection over REST and either report it or feed it
        into the matching tool. Returns a printable string, or None to fall through.

        Empty selection is a graceful, friendly message — never an error.
        """
        intent = self._detect_selection_intent(user_input)
        if intent is None:
            return None
        if getattr(self, "bridge", None) is None:
            return ("[warn]ChimeraX is not connected.[/warn] Open it, select residues, "
                    "then try again.")

        from selection import read_selection
        sel = read_selection(self.bridge.run_command)
        if sel.is_empty:
            return ("[warn]Nothing is selected in ChimeraX.[/warn]\n"
                    "[dim]Ctrl-click residues in the 3D view (or drag across them in the "
                    "Sequence Viewer), then run the command again.[/dim]")

        if intent == "report":
            return self._format_selection_report(sel)
        if intent == "scan":
            return self._scan_selection_camsol(sel)
        if intent == "redesign":
            return self._redesign_selection(sel)
        return None

    def _chain_resnums(self, model_id: str, chain: str) -> List[int]:
        """Ordered residue numbers of a loaded chain (via `info residues #M/CH`),
        parsed with the same selection parser. [] on failure."""
        try:
            from selection import parse_selection_text
            res = self.bridge.run_command(f"info residues #{model_id}/{chain}")
            text = res.get("value") if isinstance(res, dict) else ""
            refs = parse_selection_text(text or "", default_model=str(model_id))
            return sorted({r for _, c, r, _ in refs if c == chain})
        except Exception:
            return []

    def _format_selection_report(self, sel) -> str:
        by = sel.by_chain()
        lines = [f"[hi]Current ChimeraX selection[/hi] — [bold]{sel.count}[/bold] "
                 f"residue(s) across {len(by)} chain(s):"]
        for chain, rns in by.items():
            spec = self._compact_resspec(rns)
            lines.append(f"  chain [bold]{chain}[/bold]: {len(rns)} residue(s)  "
                         f"([dim]{spec}[/dim])")
        lines.append("[dim]→ 'scan the selection' (solubility) or 'redesign the "
                     "selection' (ProteinMPNN) act on exactly these residues.[/dim]")
        return "\n".join(lines)

    def _scan_selection_camsol(self, sel) -> str:
        """Run CamSol on the selected chain(s) and report/colour EXACTLY the
        selected residues (their per-residue solubility), nothing else."""
        from proteinmpnn_bridge import chain_resnum_to_seqpos
        from camsol_bridge import _assign_colour
        bridge = self._get_camsol_bridge()
        lines = ["[hi]CamSol solubility — selected residues only[/hi]:"]
        color_cmds: List[str] = []
        any_scored = False
        for chain, sel_resnums in sel.by_chain().items():
            model_id = (sel.models[0] if sel.models else None) or self._first_model_id()
            seq = self._fetch_sequence(model_id, chain)
            if not seq:
                lines.append(f"  chain {chain}: [warn]no sequence available[/warn]")
                continue
            ordered = self._chain_resnums(model_id, chain) or list(range(1, len(seq) + 1))
            pos1 = chain_resnum_to_seqpos(ordered)          # resnum -> 1-based position
            res = bridge.analyze(seq, model_id=model_id, chain=chain,
                                 start_resno=ordered[0], session=self.session)
            scores = res.data.get("scores", {}) if res.success else {}
            scores_list = [scores[k] for k in sorted(scores)]   # 0-indexed by position
            band: Dict[str, List[int]] = {}
            for r in sel_resnums:
                p = pos1.get(r)
                sc = scores_list[p - 1] if (p and 1 <= p <= len(scores_list)) else None
                if sc is None:
                    lines.append(f"  {chain}:{r}  (not scored)")
                    continue
                any_scored = True
                tag = "aggregation-prone" if sc < -0.5 else ("soluble" if sc > 0.5 else "neutral")
                lines.append(f"  {chain}:{r}  z={sc:+.2f}  [{tag}]")
                band.setdefault(_assign_colour(sc), []).append(r)
            for colour, rs in band.items():
                if colour == "white":
                    continue
                color_cmds.append(
                    f"color #{model_id}/{chain}:{self._compact_resspec(rs)} {colour}")
        if color_cmds:
            try:
                self.bridge.run_commands(color_cmds)
                lines.append("[dim]Selected residues coloured in 3D by solubility "
                             "(red=aggregation-prone … blue=soluble).[/dim]")
            except Exception:
                pass
        if not any_scored:
            lines.append("[dim](no residues could be scored)[/dim]")
        return "\n".join(lines)

    def _redesign_selection(self, sel) -> str:
        """Feed the live selection into ProteinMPNN as its designable set."""
        chain = sel.chains[0]
        model_id = (sel.models[0] if sel.models else None) or self._first_model_id()
        result = self._run_proteinmpnn({
            "model_id": model_id, "chain_id": chain,
            "design_positions": sel.resnums(chain),
        })
        if result.viz_commands:
            try:
                self.bridge.run_commands(result.viz_commands)
            except Exception:
                pass
        if not result.success:
            return f"[warn]{result.error or 'ProteinMPNN failed'}[/warn]"
        return (f"[ok]ProteinMPNN redesigned the {len(sel.resnums(chain))} selected "
                f"residue(s) on chain {chain}.[/ok]\n{result.summary or ''}")

    def handle_sequence_display_command(self, user_input: str) -> Optional[str]:
        """
        Handle "show designed sequences" and similar commands without LLM translation.

        Returns a human-readable string if BOTH conditions hold:
          1. *user_input* matches a display keyword OR fuzzy rule (see below), AND
          2. The session already holds ProteinMPNN results.

        Returns None if either condition is false — the caller should fall through
        to LLM translation so the model can answer naturally.

        Fuzzy rule: plural "sequences" + any display verb ("show", "output",
        "print", "list", "display", "what") also triggers the handler when the
        session has MPNN results, catching natural phrasings like
        "can you output the sequences" that aren't covered by explicit keywords.
        """
        lower = user_input.lower().strip()

        # FASTA export takes priority
        if any(kw in lower for kw in self._FASTA_EXPORT_KEYWORDS):
            return self._export_sequences_fasta()

        intent = self._detect_mpnn_display_intent(user_input)
        if intent is None:
            # Back-compat: the original plural-keyword / fuzzy rule still counts as
            # a 'sequence' display request.
            if any(kw in lower for kw in self._SEQUENCE_DISPLAY_KEYWORDS) or (
                "sequences" in lower and any(v in lower for v in self._SEQUENCE_DISPLAY_VERBS)
            ):
                intent = "sequence"

        if intent is None:
            return None   # not a display request — let the LLM handle it (incl. runs)

        model_id = self._first_model_id()
        mpnn_data, src = self._resolve_mpnn_data(model_id)
        if not mpnn_data:
            # A DISPLAY request with no design anywhere must NOT fall through to a
            # fresh MPNN run — inform instead.
            return ("No ProteinMPNN design found in this session or the cache "
                    "(cache/proteinmpnn/). Run a redesign first, e.g. "
                    "'redesign chain A with ProteinMPNN'.")

        if intent == "alignment":
            return self._show_mpnn_alignment(mpnn_data, model_id, src)
        return self._show_designed_sequences(mpnn_data=mpnn_data, model_id=model_id)

    def _detect_mpnn_display_intent(self, text: str) -> Optional[str]:
        """
        Classify a redesign-DISPLAY request (never a run): 'alignment', 'sequence',
        or None. Run/re-run phrasings ('redesign chain A with MPNN') return None so
        they fall through to the normal pipeline.

        Bug 5 fix: a request whose FIRST verb is a primary visualization op
        (remove/hide/overlay/color/cartoon/…) is NEVER MPNN display retrieval,
        even if "redesigned"/"redesign" appears later in the text.
        """
        if not text:
            return None
        low = text.lower()

        # Bug 5: if the request opens with a pure viz verb, it's structural viz —
        # not a sequence-list request — regardless of "redesigned" appearing later.
        # (e.g. "remove chain B … redesigned chain A" → hide/overlay viz, not MPNN)
        low_stripped = low.lstrip()
        for vv in self._PRIMARY_VIZ_VERBS:
            if low_stripped.startswith(vv + " ") or low_stripped == vv:
                return None

        if any(t in low for t in self._MPNN_RUN_TRIGGERS):
            return None
        if any(k in low for k in self._MPNN_ALIGNMENT_KEYWORDS):
            return "alignment"
        has_verb = any(v in low for v in self._MPNN_DISPLAY_VERBS)
        has_noun = any(n in low for n in (
            "sequence", "sequences", "redesign", "redesigned", "designed",
            "the design", "mpnn", "mutation",
        ))
        if has_verb and has_noun:
            return "sequence"
        return None

    def _resolve_mpnn_data(self, model_id: Optional[str] = None):
        """
        Return the most recent ProteinMPNN result WITHOUT re-running: prefer the
        in-session result, fall back to the latest persisted cache FASTA. Returns
        (mpnn_data | None, source_str).
        """
        model_id = model_id or self._first_model_id()
        data = self.session.get_proteinmpnn_result(model_id)
        if data:
            return data, "session"
        # Fall back to the on-disk design cache (survives an unsaved session).
        try:
            from proteinmpnn_bridge import latest_cached_fasta, read_designs_fasta
            fa = latest_cached_fasta(model_id) or latest_cached_fasta()
            if fa:
                return read_designs_fasta(fa), f"cache:{fa.name}"
        except Exception:
            pass
        return None, "none"

    # Phrases that mean "the ProteinMPNN design" (so a fold/validate request pulls
    # the redesigned sequence instead of the loaded WT chain). Distinct from
    # rfdiffusion "design a binder" — these only matter once colabfold/validate_design
    # has already been dispatched.
    _MPNN_DESIGN_REFS = (
        "redesign", "redesigned", "top design", "best design", "the design",
        "designed sequence", "designed seq", "mpnn design", "mpnn sequence",
        "mpnn result", "the redesign",
    )

    def _refers_to_mpnn_design(self, text: str) -> bool:
        low = (text or "").lower()
        return any(p in low for p in self._MPNN_DESIGN_REFS)

    def _mpnn_top_sequence(self, model_id: Optional[str] = None):
        """The TOP ProteinMPNN designed sequence for *model_id*, RETRIEVED never
        re-run (session → persisted cache FASTA, via `_resolve_mpnn_data`). Returns
        (sequence | None, source). "Top" = lowest ProteinMPNN score (best); falls
        back to the first design when scores are absent/equal."""
        data, src = self._resolve_mpnn_data(model_id)
        seqs = [s for s in ((data or {}).get("sequences") or []) if s.get("sequence")]
        if not seqs:
            return None, src
        best = min(seqs, key=lambda s: s["score"] if isinstance(s.get("score"), (int, float)) else 0.0)
        return best.get("sequence"), src

    def _show_designed_sequences(self, mpnn_data=None, model_id=None) -> str:
        """
        Build a human-readable display of ProteinMPNN designed sequences,
        retrieving (never re-running) from the session or the persistent cache.
        """
        model_id  = model_id or self._first_model_id()
        if mpnn_data is None:
            mpnn_data, _src = self._resolve_mpnn_data(model_id)
        if not mpnn_data:
            return (
                "No ProteinMPNN design found (session or cache). "
                "Run a sequence redesign first "
                "(e.g. 'redesign chain A with ProteinMPNN')."
            )

        sequences   = mpnn_data.get("sequences", [])
        wildtype    = mpnn_data.get("wildtype_sequence", "")
        backend     = mpnn_data.get("backend", "unknown")
        n_sequences = len(sequences)

        if not sequences:
            return "ProteinMPNN ran but produced no designed sequences."

        lines = [
            f"ProteinMPNN Designed Sequences — model #{model_id} ({backend})",
            f"  Wildtype length: {len(wildtype)} residues",
            f"  Designs: {n_sequences}",
            "",
        ]

        for i, seq_entry in enumerate(sequences, 1):
            seq       = seq_entry.get("sequence", "")
            score     = seq_entry.get("score", 0.0)
            recovery  = seq_entry.get("recovery", 0.0)
            mutations = list(seq_entry.get("mutations") or [])

            # Compute mutations from diff if not stored
            if not mutations and wildtype and seq:
                try:
                    from mpnn_esmfold_pipeline import _diff_sequences
                    mutations = _diff_sequences(wildtype, seq)
                except ImportError:
                    pass

            n_muts      = len(mutations)
            mut_display = ", ".join(mutations)   # full set — never hide the design

            lines.append(f"  Design {i}/{n_sequences}")
            lines.append(f"    Score:     {score:.3f}")
            lines.append(f"    Recovery:  {recovery:.1%}")
            lines.append(
                f"    Mutations: {n_muts} — {mut_display if mut_display else 'none (identical to WT)'}"
            )
            seq_display = seq
            lines.append(f"    Sequence:  {seq_display}")
            lines.append("")

        return "\n".join(lines)

    # ── WT-vs-redesign alignment (console + interactive ChimeraX) ────────────────

    def _show_mpnn_alignment(self, mpnn_data, model_id: str, src: str) -> str:
        """
        Render the numbered WT-vs-top-redesign console alignment AND (best-effort,
        if ChimeraX is up) open the interactive ChimeraX Sequence Viewer associated
        with the loaded chain. Retrieval only — never re-runs MPNN.
        """
        seqs = mpnn_data.get("sequences") or []
        wt   = mpnn_data.get("wildtype_sequence") or ""
        if not seqs or not wt:
            return "No ProteinMPNN design available to align."
        top   = seqs[0]
        chain = mpnn_data.get("chain", "A")
        console_view = self._build_mpnn_alignment_console(
            wt, top.get("sequence", ""), top, model_id, chain, src
        )
        viz_note = self._open_mpnn_sequence_viewer(
            wt, top.get("sequence", ""), model_id, chain
        )
        return console_view + ("\n" + viz_note if viz_note else "")

    @staticmethod
    def _alignment_rows(wt: str, design: str) -> List[int]:
        """1-based positions where design differs from WT (over the common length)."""
        return [i for i, (w, d) in enumerate(zip(wt, design), 1) if w != d]

    def _build_mpnn_alignment_console(
        self, wt: str, design: str, top: Dict[str, Any],
        model_id: str, chain: str, src: str, width: int = 50,
    ) -> str:
        """
        Numbered, Rich-marked WT-vs-redesign alignment in residue numbering
        (1-based, consistent with the recovery viz), in blocks of *width*, every
        changed column flagged. No truncation.
        """
        n = min(len(wt), len(design))
        changed = set(self._alignment_rows(wt, design))
        mutations = list(top.get("mutations") or [
            f"{wt[i-1]}{i}{design[i-1]}" for i in sorted(changed)
        ])
        rec = top.get("recovery")
        out: List[str] = [
            f"[bold]WT vs ProteinMPNN redesign[/bold] — model #{model_id} chain {chain}  "
            f"[dim](source: {src})[/dim]",
            f"  length {n} · [red]{len(changed)} changed[/red] · "
            + (f"recovery {rec:.1%}" if isinstance(rec, (int, float)) else "")
            + "   (numbering = 1-based residue position)",
            "",
        ]
        for start in range(0, n, width):
            end = min(start + width, n)
            ruler = "".join(
                ("|" if (p % 10 == 0) else (":" if (p % 5 == 0) else "."))
                for p in range(start + 1, end + 1)
            )
            wt_row, de_row, mark = [], [], []
            for p in range(start + 1, end + 1):
                w, d = wt[p - 1], design[p - 1]
                if p in changed:
                    wt_row.append(w)
                    de_row.append(f"[bold red]{d}[/bold red]")
                    mark.append("^")
                else:
                    wt_row.append(w)
                    de_row.append(d)
                    mark.append(" ")
            lbl = f"{start + 1:>5}"
            out.append(f"  pos {lbl} {ruler}")
            out.append(f"  WT      {''.join(wt_row)}")
            out.append(f"  design  {''.join(de_row)}")
            out.append(f"          {''.join(mark)}")
            out.append("")
        out.append(f"  [bold]Changes ({len(mutations)}):[/bold] " + ", ".join(mutations))
        return "\n".join(out)

    def _open_mpnn_sequence_viewer(
        self, wt: str, design: str, model_id: str, chain: str,
    ) -> Optional[str]:
        """
        Build a 2-sequence ungapped FASTA (WT + top redesign) and open it in
        ChimeraX's Sequence Viewer associated with the loaded chain, so selecting a
        column selects the 3D residue (and vice versa). Best-effort; returns a note.

        Association subtlety: the redesign is heavily changed (> ChimeraX's ~10%
        auto-association threshold) so the redesign row won't auto-associate, but
        the WT row (identical to chain A) does, and the ungapped 1:1 columns still
        map to the correct 3D residues through it.
        """
        if self.bridge is None:
            return ("  (ChimeraX not connected — run while ChimeraX is open to get the "
                    "interactive Sequence Viewer: changed columns highlighted; select a "
                    "column → highlights the 3D residue.)")
        try:
            from proteinmpnn_bridge import build_alignment_fasta
            import config as _cfg
            out = Path(_cfg.PROTEINMPNN_CACHE_DIR) / f"alignment_model{model_id}.fa"
            build_alignment_fasta(wt, design, out,
                                  wt_label="WT", design_label="redesign_top")
            posix = Path(out).as_posix()

            changed   = self._alignment_rows(wt, design)            # 1-based changed columns
            n         = min(len(wt), len(design))
            conserved = [i for i in range(1, n + 1) if i not in set(changed)]
            spec = f"#{model_id}/{chain}"

            cmds = [
                f'open "{posix}"',                       # opens the Sequence Viewer + auto-associates WT↔chain
                f"sequence associate {spec}",            # force-associate the chain to the alignment
                # ── Auto-decorate (one consistent story across sequence + 3D) ──
                # ChimeraX 1.11.1 has no `sequence region`/`sequence color` command,
                # so we (a) colour the 3D STRUCTURE with the MPNN convention
                # (changed=tomato, conserved=cornflower blue), and (b) SELECT the
                # changed residues — through the WT association the Sequence Viewer
                # mirrors this as a highlighted region over exactly the changed
                # columns. The auto-shown conservation header also marks them.
                f"cartoon {spec}",
                f"color {spec} white",
            ]
            if conserved:
                cmds.append(f"color {spec}:{self._compact_resspec(conserved)} cornflower blue")
            if changed:
                cmds.append(f"color {spec}:{self._compact_resspec(changed)} tomato")
                cmds.append(f"select {spec}:{self._compact_resspec(changed)}")

            ran_ok = True
            for c in cmds:
                r = self.bridge.run_command(c)
                if isinstance(r, dict) and r.get("error"):
                    ran_ok = False
            if ran_ok:
                return (f"  [green]Interactive alignment open in ChimeraX[/green] — "
                        f"{len(changed)} changed columns highlighted in the Sequence Viewer "
                        f"and coloured [red]tomato[/red] on chain {chain} in 3D (conserved = "
                        "cornflower blue). Select a column → highlights the 3D residue (and "
                        "vice versa); the WT row is associated with the structure.")
            return f"  (Alignment FASTA written to {out}; open it in ChimeraX to interact.)"
        except Exception as exc:
            return f"  (Could not open the ChimeraX Sequence Viewer: {exc})"

    @staticmethod
    def _compact_resspec(positions: List[int]) -> str:
        """Compact a sorted 1-based position list into a ChimeraX spec, e.g.
        [1,2,3,5,8,9] → '1-3,5,8-9'."""
        if not positions:
            return ""
        ps = sorted(set(positions))
        runs, start, prev = [], ps[0], ps[0]
        for p in ps[1:]:
            if p == prev + 1:
                prev = p
            else:
                runs.append((start, prev)); start = prev = p
        runs.append((start, prev))
        return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _first_model_id(self) -> str:
        """Return the first loaded structure's model ID, or '1'."""
        if self.session.structures:
            return next(iter(self.session.structures))
        return "1"

    def _second_model_id(self) -> str:
        """Return the second loaded structure's model ID, or '2'."""
        ids = list(self.session.structures)
        if len(ids) >= 2:
            return ids[1]
        return "2"

    def _primary_model_id(self) -> str:
        """
        Return the primary (crystal-structure) model ID.

        When multiple models are loaded — e.g. the original structure (#1)
        alongside an ESMFold prediction (#2) — tools that operate on the
        crystal structure (proline scan, assembly analysis, …) should always
        target #1.  This helper returns '1' whenever that model is in the
        session, falling back to the first loaded model otherwise.
        """
        structures = self.session.structures
        if not structures:
            return "1"
        if "1" in structures:
            return "1"
        return next(iter(structures))

    def _visible_focus_model_id(self) -> Optional[str]:
        """The model the user is focused on — i.e. what's VISIBLE on screen — so an
        unspecified prompt acts on it rather than on a stale heuristic (the always-#1
        bias of `_primary_model_id`). Read from the bridge's live display state:
          • exactly one visible tracked structure → that one (the clear case);
          • several visible → the most-recently-added (highest numeric id), a proxy for
            "what the user just brought up";
          • no bridge / nothing visible / probe fails → None, so the caller falls back to
            `_primary_model_id()` unchanged (tests + headless stay deterministic).
        Submodel ids collapse to top level (visible_model_ids already returns top-level)."""
        if self.bridge is None:
            return None
        try:
            visible = list(self.bridge.visible_model_ids() or [])
        except Exception:
            return None
        if not visible:
            return None
        # Prefer visible ids that are tracked structures (skip e.g. a bare surface/volume).
        tracked = [v for v in visible if v in self.session.structures]
        pool = tracked or visible
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]
        return max(pool, key=lambda s: int(s) if str(s).isdigit() else -1)

    def available_tools(self) -> Dict[str, str]:
        """Return {tool_name: status_string} for all tools."""
        status: Dict[str, str] = {"chimerax": "active"}

        for name in ("camsol_bridge", "esm_bridge"):
            try:
                __import__(name)
                status[name.replace("_bridge", "")] = "active"
            except ImportError:
                status[name.replace("_bridge", "")] = "module not found"

        try:
            from proteinmpnn_bridge import ProteinMPNNBridge
            _b = ProteinMPNNBridge()
            status["proteinmpnn"] = (
                f"{_b._backend} — {_b._dir}" if _b._available
                else "not configured (set PROTEINMPNN_DIR)"
            )
        except ImportError:
            status["proteinmpnn"] = "module not found"

        try:
            from rfdiffusion_bridge import RFdiffusionBridge
            _rfd = RFdiffusionBridge()
            status["rfdiffusion"] = (
                f"rfdiffusion — {_rfd._dir}" if _rfd._available
                else "not configured (set RFDIFFUSION_DIR)"
            )
        except ImportError:
            status["rfdiffusion"] = "module not found"

        # Rosetta: report which backend is configured
        try:
            from rosetta_bridge import RosettaBridge, _select_backend
            backend = _select_backend()
            import os
            api_key = os.environ.get("ROBETTA_API_KEY", "").strip()
            if backend == "pyrosetta":
                status["rosetta"] = "PyRosetta (local) — active"
            elif api_key:
                status["rosetta"] = "Robetta web API — active"
            else:
                status["rosetta"] = "Robetta web API — set ROBETTA_API_KEY to enable"
        except ImportError:
            status["rosetta"] = "module not found"

        # mutation_scan depends on rosetta_bridge + mutation_scanner
        try:
            __import__("mutation_scanner")
            status["mutation_scan"] = "active (depends on rosetta)"
        except ImportError:
            status["mutation_scan"] = "module not found"

        # assembly_analyser
        try:
            __import__("assembly_analyser")
            status["assembly_analyser"] = "active"
        except ImportError:
            status["assembly_analyser"] = "module not found"

        # disulfide
        try:
            __import__("disulfide_bridge")
            status["disulfide"] = "active (geometry + ESM + DynaMut2)"
        except ImportError:
            status["disulfide"] = "module not found"

        # proline
        try:
            __import__("proline_bridge")
            status["proline"] = "active (backbone φ/ψ + ESM tolerance + optional DynaMut2)"
        except ImportError:
            status["proline"] = "module not found"

        # esmfold
        try:
            __import__("esmfold_bridge")
            status["esmfold"] = "active (ESM Atlas API — free, no auth)"
        except ImportError:
            status["esmfold"] = "module not found"

        # glycan
        try:
            __import__("glycan_bridge")
            status["glycan"] = "active (NXS/T sequon detection + SASA/SS/ESM scoring)"
            status["glycan_positions"] = "active (projection-aware all-residue scan)"
        except ImportError:
            status["glycan"] = "module not found"
            status["glycan_positions"] = "module not found"

        # salt_bridge
        try:
            __import__("salt_bridge_bridge")
            status["salt_bridge"] = "active (BioPython + FreeSASA)"
        except ImportError:
            status["salt_bridge"] = "module not found"

        # cavity
        try:
            __import__("cavity_bridge")
            status["cavity"] = "active (BioPython + FreeSASA dual-probe)"
        except ImportError:
            status["cavity"] = "module not found"

        # mpnn_esmfold
        try:
            __import__("mpnn_esmfold_pipeline")
            status["mpnn_esmfold"] = "active (ProteinMPNN + ESMFold combined validation)"
        except ImportError:
            status["mpnn_esmfold"] = "module not found"

        # rosetta_local via WSL2
        try:
            from wsl_bridge import WSLBridge
            wsl = WSLBridge()
            if wsl.is_available():
                if wsl.check_pyrosetta():
                    status["rosetta_local"] = "active (PyRosetta via WSL2)"
                else:
                    status["rosetta_local"] = "WSL2 available — PyRosetta not installed"
            else:
                status["rosetta_local"] = "WSL2 not installed"
        except ImportError:
            status["rosetta_local"] = "wsl_bridge module not found"

        return status

    # ── Representation (Intent/Render separation — viewer instance) ────────────

    def _snapshot_repr(self, spec: str, bridge: Any) -> List[str]:
        """
        Probe current representation state for *spec* via runscript; return restore commands.
        Returns [] on any failure — undo simply won't be available for that step.
        """
        if bridge is None:
            return []
        model_root = spec.lstrip("#").split("/")[0].split(".")[0]
        script = (
            "from chimerax.atomic import all_atomic_structures\n"
            "for m in all_atomic_structures(session):\n"
            f"    if m.id_string.startswith('{model_root}'):\n"
            "        atoms_shown   = bool(m.atoms.displays.any())\n"
            "        cartoon_shown = bool(m.residues.ribbon_displays.any())\n"
            "        modes = list(m.atoms.draw_modes)\n"
            "        mode  = max(set(modes), key=modes.count) if modes else 1\n"
            "        print(f'{atoms_shown},{cartoon_shown},{mode}')\n"
            "        break\n"
        )
        try:
            import tempfile as _tmp, os as _os
            fd, path = _tmp.mkstemp(suffix=".py")
            try:
                with _os.fdopen(fd, "w") as f:
                    f.write(script)
                r = bridge.run_command(f"runscript {path}")
            finally:
                try:
                    _os.unlink(path)
                except OSError:
                    pass
            val = (r.get("value") or "").strip()
            if "," not in val:
                return []
            parts = val.split(",", 2)
            if len(parts) < 3:
                return []
            atoms_shown   = parts[0].strip().lower() == "true"
            cartoon_shown = parts[1].strip().lower() == "true"
            mode          = int(parts[2].strip()) if parts[2].strip().isdigit() else 1
            _style_map    = {0: "sphere", 1: "stick", 2: "ball"}
            cmds: List[str] = []
            if atoms_shown:
                cmds.append(f"show {spec} atoms")
                cmds.append(f"style {spec} {_style_map.get(mode, 'stick')}")
            else:
                cmds.append(f"hide {spec} atoms")
            if cartoon_shown:
                cmds.append(f"show {spec} cartoons")
            else:
                cmds.append(f"hide {spec} cartoons")
            return cmds
        except Exception:
            return []

    # ── Shared op-class target resolver (single source of truth) ───────────────
    # The op-class handlers (_run_color / _run_representation) rebuild a command
    # from the user's English via the intent/render registry, so both MUST agree on
    # WHAT the command applies to. This one resolver is that agreement — neither
    # handler reimplements target parsing.
    #
    # SAFETY INVARIANT (the regression class this closes): an op-class handler must
    # NEVER silently broaden scope. It renders a deterministic command ONLY for a
    # target it can fully express — the whole model, a chain, or a
    # ligand/solvent/ions keyword selector. For any FINER target (residue
    # ranges/zones, a binding/active site, an interface, a live selection) it
    # DEFERS: the caller runs the translator's already-scoped command instead of a
    # regenerated whole-model one. Widening (ligand → whole model) is thereby made
    # structurally impossible, not merely unlikely.

    # Markers of a target finer than a bare chain/keyword. Presence of ANY forces a
    # defer — even when a chain/keyword is also present, because the real target is
    # then relative/narrower ("residues 50-60 in chain A", "within 5 Å of the
    # ligand"). Deliberately EXCLUDES secondary-structure words (helix/strand/loop)
    # so "color by secondary structure" still routes to the registry scheme.
    _OPCLASS_FINER_TARGET_RE = re.compile(
        r"""\bresidues?\b | \bresid\b | \bresno\b
          | :\s*\d
          | \b\d+\s*(?:-|–|to|thru|through)\s*\d+\b
          | \bwithin\b | \bnear\b | \bnearby\b | \bzone\b
          | \bpocket\b | \bbinding\s+site\b | \bactive\s+site\b
          | \binterface\b
          | \bselection\b | \bselected\b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    def _opclass_model_scope(self, user_input: str, model_id: str) -> str:
        """The model scope for an op-class command (the §0 rule), used for BOTH whole-model
        and subregion targets: the ENGLISH is authoritative. Precedence:
          1. explicit `#N` / `model N` → that model;
          2. SEMANTIC model words — "template"/"reference"/"crystal"/"WT" → the loaded
             (primary) structure; "variant"/"fold"/"prediction" → the OTHER visible models
             (the predicted folds), i.e. visible minus primary;
          3. otherwise ALL VISIBLE models (hidden ones untouched), never a stale default.
        A subregion (chain/residue/keyword) narrows WITHIN this scope (`#1,2/A`); it never
        re-broadens the model dimension to a hidden model. Falls back to `model_id` only
        when display state is unavailable. Returns a bare atomspec model part (may be
        comma-joined)."""
        ui = user_input or ""
        mexp = (re.search(r"#(\d+(?:\.\d+)*)\b", ui)
                or re.search(r"\bmodels?\s+#?(\d+(?:\.\d+)*)\b", ui, re.I))
        if mexp:
            return mexp.group(1)                              # 1. explicit model wins
        primary = str(self._primary_model_id())
        if re.search(r"\b(template|reference|crystal|wild[\s-]?type|wt)\b", ui, re.I):
            return primary                                    # 2a. the loaded reference model
        try:
            vis = self.bridge.visible_model_ids() if self.bridge is not None else None
        except Exception:
            vis = None
        vis = [str(v) for v in vis] if isinstance(vis, list) else []
        if re.search(r"\b(variants?|folds?|predictions?|predicted)\b", ui, re.I):
            others = [v for v in vis if v != primary]         # 2b. the predicted fold(s)
            if others:
                return ",".join(others)                       # (visible minus the template)
        if vis:
            return ",".join(vis)                              # 3. ALL VISIBLE models
        return str(model_id)                                  # probe unavailable → default

    def _resolve_opclass_target(
        self, user_input: str, model_id: str,
    ) -> Dict[str, Any]:
        """Resolve the target an op-class (color/representation) command applies to.

        Returns a dict:
          {"spec": str|None, "chain": str|None, "keyword": str|None,
           "target_desc": str, "defer": bool}

        defer=False → render deterministically against ``spec`` (whole model /
                      ``#N/CHAIN`` / ``#N & <keyword>``). Chain specs are scoped to
                      exclude ligand/solvent/ions by the shared chain-scope guard at
                      the call site.
        defer=True  → ``spec`` is None; the target is finer than this resolver can
                      express. The caller MUST run the translator's command instead
                      of regenerating a whole-model one (the safety invariant).

        Precedence: a finer-target marker → DEFER (wins over chain/keyword, since
        the marker means the real target is narrower); else explicit chain ref;
        else ligand/solvent/ions keyword; else the whole model.
        """
        ui  = user_input or ""
        # `mid` is the §0 model scope for EVERY target (whole-model AND subregion): an
        # explicit #N, else ALL VISIBLE models, else the default. A subregion narrows
        # within it (`#1,2/A`); it never targets a hidden model. One resolution → one
        # bridge probe per op-class command.
        mid = self._opclass_model_scope(ui, model_id)

        # 1. Explicit residue selection (range / list / single), optionally
        #    chain-qualified → an EXPRESSIBLE `:N` / `:N-M` / `:n,m` spec. Done
        #    BEFORE the finer-target defer so a residue range renders
        #    deterministically (the translator's residue specs are unreliable —
        #    `/50-60` instead of `:50-60`) while zones/pockets still defer.
        res_chain = None
        cm = re.search(r"\bchain\s+([A-Za-z0-9])\b", ui, re.I) \
            or re.search(r"(?<![A-Za-z0-9])/([A-Za-z0-9])(?![A-Za-z0-9:])", ui)
        if cm:
            res_chain = cm.group(1).upper()
        rng    = re.search(r"\bresidues?\s+(\d+)\s*(?:-|–|to|through|thru)\s*(\d+)\b",
                           ui, re.I)
        rlist  = re.search(r"\bresidues?\s+(\d+(?:\s*,\s*\d+)+)", ui, re.I)
        rone   = re.search(r"\bresidues?\s+(\d+)\b", ui, re.I)
        res_frag = None
        if rng:
            res_frag = f":{rng.group(1)}-{rng.group(2)}"
        elif rlist:
            res_frag = ":" + re.sub(r"\s+", "", rlist.group(1))
        elif rone:
            res_frag = f":{rone.group(1)}"
        if res_frag:
            chain_part = f"/{res_chain}" if res_chain else ""
            desc = (f"residues {res_frag[1:]}"
                    + (f" in chain {res_chain}" if res_chain else ""))
            return {"spec": f"#{mid}{chain_part}{res_frag}", "chain": res_chain,
                    "keyword": None, "target_desc": desc, "defer": False}

        # 2. A finer-than-chain target marker we can't express (distance zone,
        #    binding/active site, interface, live selection) → DEFER (highest
        #    remaining precedence). A bare chain/keyword in the same phrase would
        #    WIDEN it ("within 5 Å of the ligand" is NOT the whole ligand), so never
        #    render in this case.
        if self._OPCLASS_FINER_TARGET_RE.search(ui):
            return {"spec": None, "chain": None, "keyword": None,
                    "target_desc": "a finer selection", "defer": True}

        # 3. Explicit whole-model phrasing — an intended, explicit broadening. Applies
        #    to ALL VISIBLE models (the §0 rule), not just the default one.
        if re.search(r"\b(?:each|every|all|both|per)\s+chains?\b"
                     r"|\bwhole\s+(?:model|structure|thing)\b|\beverything\b",
                     ui, re.I):
            return {"spec": f"#{mid}", "chain": None, "keyword": None,
                    "target_desc": "whole model", "defer": False}

        # 4. Explicit chain ref ("chain A" / "/A"), but NOT residue-qualified
        #    (a `/A:50` is handled as a residue selection in step 1).
        m = re.search(r"\bchain\s+([A-Za-z0-9])\b", ui, re.I)
        if not m:
            m = re.search(r"(?<![A-Za-z0-9])/([A-Za-z0-9])(?![A-Za-z0-9:])", ui)
        if m:
            chain = m.group(1).upper()
            return {"spec": f"#{mid}/{chain}", "chain": chain, "keyword": None,
                    "target_desc": f"chain {chain}", "defer": False}

        # 5. Keyword selectors (ligand / solvent|water / ions) — disjoint from the
        #    protein chains; the bare ChimeraX keyword, scoped to this model.
        keyword = None
        if re.search(r"\bligands?\b", ui, re.I):
            keyword = "ligand"
        elif re.search(r"\b(?:solvent|waters?)\b", ui, re.I):
            keyword = "solvent"
        elif re.search(r"\bions?\b", ui, re.I):
            keyword = "ions"
        if keyword:
            return {"spec": f"#{mid} & {keyword}", "chain": None,
                    "keyword": keyword, "target_desc": f"the {keyword}",
                    "defer": False}

        # 6. No target named → ALL VISIBLE models (the §0 rule; the common "color by
        #    chain" / "show only as cartoon" case — the reported regression).
        return {"spec": f"#{mid}", "chain": None, "keyword": None,
                "target_desc": "whole model", "defer": False}

    # Target phrases removed before ALIAS / scheme matching only. A target sitting
    # between the verb and the representation/scheme noun ("show THE LIGAND as
    # sticks", "show CHAIN A as cartoon") breaks the registry's contiguous-phrase
    # alias match, dropping the request to the flaky LLM tier. The target and the
    # color are resolved separately from the ORIGINAL text (_resolve_opclass_target
    # / extract_named_color), so removing the target here is safe and only helps the
    # alias land deterministically.
    _TARGET_PHRASE_RE = re.compile(
        r"\bthe\s+ligands?\b|\bligands?\b|\bthe\s+solvent\b|\bsolvent\b|\bwaters?\b"
        r"|\bions?\b|\bchain\s+[A-Za-z0-9]\b|(?<![A-Za-z0-9])/[A-Za-z0-9]\b",
        re.IGNORECASE,
    )

    def _strip_target_for_alias(self, user_input: str) -> str:
        """Remove a target phrase so the verb+noun alias core matches (see
        _TARGET_PHRASE_RE). Used ONLY for alias/scheme resolution."""
        stripped = self._TARGET_PHRASE_RE.sub(" ", user_input or "")
        return re.sub(r"\s{2,}", " ", stripped).strip()

    def _execute_deferred_opclass(
        self, tool: str, commands: List[str],
    ) -> ToolStepResult:
        """Safety-invariant defer path: the op-class resolver could not express the
        target, so run the translator's already-scoped command(s) verbatim rather
        than a regenerated whole-model command. Never widens scope. If there is no
        translator command to fall back to, REFUSE (never silently widen)."""
        if self.bridge is None:
            return ToolStepResult(tool=tool, success=False,
                                  error="ChimeraX bridge unavailable.")
        if not commands:
            return ToolStepResult(
                tool=tool, success=False,
                error=(
                    f"This {tool} request targets a finer selection (e.g. a residue "
                    "range, a distance zone, or a binding pocket) that the "
                    "deterministic resolver can't express, and no fallback command "
                    "was available. Rephrase with a chain (e.g. 'chain A') or "
                    "'the ligand', or name the residues explicitly."),
            )
        executed: List[str] = []
        for cmd in commands:
            r = self.bridge.run_command(cmd)
            if r.get("error"):
                return ToolStepResult(
                    tool=tool, success=False,
                    error=(f"Command failed (deferred to the translator's scoped "
                           f"command): {cmd!r} → {r['error']}"),
                )
            executed.append(cmd)
        return ToolStepResult(
            tool=tool, success=True,
            summary=(f"Applied {tool} to a finer target via the translator's scoped "
                     f"command(s) — op-class resolver deferred (NOT whole-model)\n"
                     f"Commands: {'; '.join(executed)}"),
            viz_commands=[], viz_explanations=[],
            data={"resolution": "deferred", "commands": executed},
        )

    def _run_representation(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Deterministic viewer representation handler.

        Resolution pipeline (intent_registry):
          (a) alias match   — instant; 100% for listed phrases
          (b) LLM classifier — constrained to return a LABEL (or "none"), never syntax
          (c) graceful miss  — lists available intents and asks user to rephrase

        Commands are executed via bridge.run_command() directly and verified
        via post-command probe.  viz_commands=[] (already run); summary carries result.
        """
        from intent_registry import VIEWER_REGISTRY, make_llm_classify_fn
        from translator import _scope_chain_refs_to_macromolecule

        user_input = user_input or inputs.get("_user_input", "")
        intent_key = inputs.get("intent_key")   # pre-resolved by route() alias match
        model_id   = str(inputs.get("model_id") or self._primary_model_id())
        resolution = "alias"

        # Target resolution via the shared op-class resolver. DEFER (run the
        # translator's already-scoped command) for any target finer than a chain /
        # ligand keyword, so "show the ligand as sticks" never re-styles the WHOLE
        # model (the op-class widening regression). Undo carries no target marker,
        # so it resolves to the whole-model spec and is unaffected.
        tgt = self._resolve_opclass_target(user_input, model_id)
        if tgt["defer"]:
            return self._execute_deferred_opclass(
                "representation", inputs.get("_translator_commands") or [])
        spec = tgt["spec"]

        # Tier (b): LLM constrained classifier — fires when alias match missed
        if intent_key is None:
            global _repr_classify_fn
            if _repr_classify_fn is None:
                _repr_classify_fn = make_llm_classify_fn()
            classify = _repr_classify_fn
            labels   = VIEWER_REGISTRY.list_intent_keys("view")
            # Resolve on the target-stripped text so the alias core lands and the
            # LLM sees the bare representation phrase, not the target.
            intent_key, resolution = VIEWER_REGISTRY.resolve(
                self._strip_target_for_alias(user_input),
                llm_classify_fn=lambda t, ls: classify(t, ls),
            )

        # Tier (c): graceful miss
        if intent_key is None:
            miss_msg = VIEWER_REGISTRY.graceful_miss_message(user_input, "view")
            return ToolStepResult(
                tool    = "representation",
                success = False,
                error   = miss_msg,
            )

        # ── Undo/revert special case ───────────────────────────────────────────
        if intent_key == "view.undo_representation":
            restore_cmds = self._repr_snapshots.get(spec)
            if not restore_cmds:
                return ToolStepResult(
                    tool    = "representation",
                    success = False,
                    error   = (
                        f"No prior representation state recorded for {spec}. "
                        "Apply a representation change first, then use undo/revert."
                    ),
                )
            if self.bridge is None:
                return ToolStepResult(
                    tool="representation", success=False,
                    error="ChimeraX bridge unavailable.",
                )
            for cmd in restore_cmds:
                r = self.bridge.run_command(cmd)
                if r.get("error"):
                    return ToolStepResult(
                        tool    = "representation",
                        success = False,
                        error   = f"Undo failed: {cmd!r} → {r['error']}",
                    )
            del self._repr_snapshots[spec]
            return ToolStepResult(
                tool    = "representation",
                success = True,
                summary = f"Reverted representation for {spec}",
                viz_commands     = [],
                viz_explanations = [],
                data = {"intent_key": "view.undo_representation", "spec": spec},
            )

        # Render layer — single source of truth for syntax
        commands = VIEWER_REGISTRY.render(intent_key, spec)
        defn     = VIEWER_REGISTRY.get_defn(intent_key)

        # Chain-scope guard: a chain rep ("show chain A as cartoon") excludes
        # ligand/solvent/ions so it never bleeds onto them — matching the color
        # path and the translator. Whole-model `#N` and keyword `#N & ligand`
        # specs carry no bare chain ref and pass through unchanged.
        commands, _scope_notes = _scope_chain_refs_to_macromolecule(commands)

        if self.bridge is None:
            return ToolStepResult(
                tool="representation", success=False,
                error="ChimeraX bridge unavailable.",
            )

        # Snapshot BEFORE executing render commands — enables one-level undo
        snapshot = self._snapshot_repr(spec, self.bridge)
        if snapshot:
            self._repr_snapshots[spec] = snapshot

        # Execute commands — bridge._ERROR_PREFIXES catches "Expected …" and others
        executed: List[str] = []
        for cmd in commands:
            r = self.bridge.run_command(cmd)
            if r.get("error"):
                return ToolStepResult(
                    tool    = "representation",
                    success = False,
                    error   = (
                        f"Command failed ({resolution}-resolved {intent_key!r}): "
                        f"{cmd!r} → {r['error']}"
                    ),
                )
            executed.append(cmd)

        # Post-command verify guard
        verify_ok = VIEWER_REGISTRY.verify(intent_key, spec, self.bridge)
        verify_note = ""
        if verify_ok is False:
            verify_note = " [⚠ verify: display state may not have changed]"

        desc    = defn.description if defn else intent_key
        cmd_str = "; ".join(executed)
        return ToolStepResult(
            tool    = "representation",
            success = True,
            summary = (
                f"Applied {desc} to {tgt['target_desc']} "
                f"({resolution}-resolved){verify_note}\n"
                f"Commands: {cmd_str}"
            ),
            viz_commands     = [],   # already executed above
            viz_explanations = [],
            data = {
                "intent_key": intent_key,
                "spec":       spec,
                "chain":      tgt["chain"],
                "resolution": resolution,
                "commands":   executed,
            },
        )

    def _run_color(
        self,
        inputs:     Dict[str, Any],
        user_input: str = "",
    ) -> ToolStepResult:
        """
        Deterministic color handler (Intent/Render op-class — mirrors
        _run_representation).

        Resolution:
          (a) scheme alias (by_chain/by_element/by_heteroatom/rainbow/by_attribute)
          (b) color.solid — a recognised named color in the phrase
          (c) LLM constrained classifier (label only, never syntax)
          (d) graceful miss

        Chain-scoped colors reuse the translator chain-scope guard
        (`& ~ligand & ~solvent & ~ions`) so coloring a chain never bleeds onto its
        ligand/solvent/ions (the payoff bug).  Whole-model targets are left
        unscoped (coloring "everything" is the user's explicit intent).
        """
        from intent_registry import (
            COLOR_REGISTRY, make_llm_classify_fn, extract_named_color,
            _COLOR_TASK_BLOCK,
        )
        from translator import _scope_chain_refs_to_macromolecule

        user_input = user_input or inputs.get("_user_input", "")
        intent_key = inputs.get("intent_key")        # pre-resolved scheme alias, or None
        model_id   = str(inputs.get("model_id") or self._primary_model_id())
        resolution = "alias"

        # Target resolution via the shared op-class resolver (the SAME one
        # _run_representation uses). DEFER (run the translator's already-scoped
        # command) for any target finer than a chain / ligand keyword, so
        # "colour the ligand white" never paints the whole model (the regression
        # this whole op-class audit closes).
        tgt    = self._resolve_opclass_target(user_input, model_id)
        if tgt["defer"]:
            return self._execute_deferred_opclass(
                "color", inputs.get("_translator_commands") or [])
        spec   = tgt["spec"]
        chain  = tgt["chain"]

        # color.solid carries a parsed color value (no alias tier).
        color_name = None

        # Tier (b)/(c): resolve when no scheme alias matched up front.
        if intent_key is None:
            color_name = extract_named_color(user_input)
            if color_name is not None:
                intent_key, resolution = "color.solid", "solid"
            else:
                global _color_classify_fn
                if _color_classify_fn is None:
                    _color_classify_fn = make_llm_classify_fn(
                        registry=COLOR_REGISTRY, task_block=_COLOR_TASK_BLOCK,
                    )
                classify = _color_classify_fn
                labels   = COLOR_REGISTRY.list_intent_keys("color")
                # Scheme resolution on the target-stripped text (target + color are
                # parsed separately from the original), so "color the ligand by
                # element" still resolves the by-element scheme.
                intent_key, resolution = COLOR_REGISTRY.resolve(
                    self._strip_target_for_alias(user_input),
                    llm_classify_fn=lambda t, ls: classify(t, ls),
                )
                # LLM picked color.solid but no named color present → ask which.
                if intent_key == "color.solid":
                    color_name = extract_named_color(user_input)
                    if color_name is None:
                        return ToolStepResult(
                            tool    = "color",
                            success = False,
                            error   = (
                                "Which color? I detected a request to apply a solid "
                                "color but no recognised color name (e.g. 'red', "
                                "'blue', 'cornflower blue')."
                            ),
                        )

        # Tier (d): graceful miss
        if intent_key is None:
            return ToolStepResult(
                tool    = "color",
                success = False,
                error   = COLOR_REGISTRY.graceful_miss_message(user_input, "color"),
            )

        # Render layer — single source of truth for syntax
        if intent_key == "color.solid":
            commands = [f"color {spec} {color_name}"]
        else:
            commands = COLOR_REGISTRY.render(intent_key, spec)

        # Chain-scope guard: rewrite a bare chain ref to exclude ligand/solvent/ions
        # so chain colors never bleed.  (Whole-model `#N` specs are not rewritten.)
        commands, _scope_notes = _scope_chain_refs_to_macromolecule(commands)

        if self.bridge is None:
            return ToolStepResult(
                tool="color", success=False,
                error="ChimeraX bridge unavailable.",
            )

        executed: List[str] = []
        for cmd in commands:
            r = self.bridge.run_command(cmd)
            if r.get("error"):
                return ToolStepResult(
                    tool    = "color",
                    success = False,
                    error   = (
                        f"Command failed ({resolution}-resolved {intent_key!r}): "
                        f"{cmd!r} → {r['error']}"
                    ),
                )
            executed.append(cmd)

        defn    = COLOR_REGISTRY.get_defn(intent_key)
        desc    = defn.description if defn else intent_key
        if intent_key == "color.solid":
            desc = f"solid color {color_name}"
        target  = tgt["target_desc"]
        cmd_str = "; ".join(executed)
        return ToolStepResult(
            tool    = "color",
            success = True,
            summary = (
                f"Applied {desc} to {target} ({resolution}-resolved)\n"
                f"Commands: {cmd_str}"
            ),
            viz_commands     = [],   # already executed above
            viz_explanations = [],
            data = {
                "intent_key": intent_key,
                "spec":       spec,
                "chain":      chain,
                "color_name": color_name,
                "resolution": resolution,
                "commands":   executed,
            },
        )

    # ── Transparency op-class ───────────────────────────────────────────────────────
    _TRANSPARENCY_RE = re.compile(
        r"\btransparen\w*\b|\bopaque\b|\bsee[\s-]?through\b", re.IGNORECASE)

    def _detect_transparency_phrase(self, user_input: str) -> bool:
        """A transparency request (transparent / transparency / opaque / see-through)."""
        return bool(self._TRANSPARENCY_RE.search(user_input or ""))

    def _parse_transparency_request(self, user_input: str) -> Tuple[str, int]:
        """Parse → (mode, amount). mode='absolute' → amount is the target % (0=opaque,
        100=invisible); mode='relative' → amount is a signed delta. ChimeraX `transparency`
        is ABSOLUTE-only, so a relative ask is applied against the tracked per-target level.
        The number regex requires %/percent or a `by/to/at` lead-in so a model id (`#2`) is
        never mistaken for a level."""
        low = (user_input or "").lower()
        # Strip model refs ('#2', 'model 2') BEFORE the number parse so a model id is never
        # read as a transparency level — the model scope is resolved separately.
        low = re.sub(r"#[\d.,]+", " ", low)
        low = re.sub(r"\bmodels?\s+#?\d+", " ", low)
        if re.search(r"\bopaque\b|\bno\s+transparency\b|\bnot\s+transparent\b"
                     r"|\bsolid\b|\bremove\s+transparency\b", low):
            return ("absolute", 0)
        num_m = (re.search(r"(\d+)\s*(?:%|percent|pct)", low)        # "50%", "50 percent"
                 or re.search(r"\b(?:by|to|at)\s+(\d+)\b", low)       # "by 50", "to 30"
                 or re.search(r"(\d+)\s*(?:%|percent)?\s*transparen", low))  # "50 transparent"
        num = int(num_m.group(1)) if num_m else None
        inc = bool(re.search(r"\b(increase|increased|more|raise|add|up)\b", low))
        dec = bool(re.search(r"\b(decrease|decreased|less|reduce|lower|down)\b", low))
        if inc or dec:
            delta = num if num is not None else 25          # bare "more transparent" → a step
            return ("relative", delta if inc else -delta)
        if num is not None:
            return ("absolute", num)
        if re.search(r"\b(fully|completely|totally|max(?:imum)?|very)\b", low):
            return ("absolute", 100)
        return ("absolute", 50)                              # bare "make it transparent"

    def _run_transparency(self, inputs: Dict[str, Any], user_input: str = "") -> ToolStepResult:
        """Set ChimeraX transparency on the §0 op-class scope (ALL VISIBLE models / explicit
        #N / a chain etc. — reusing the shared resolver). ABSOLUTE sets the level; RELATIVE
        adjusts the tracked per-target level (ChimeraX has no relative form), clamped 0–100.
        Chain refs are macromolecule-scoped like color/representation (no ligand bleed)."""
        from translator import _scope_chain_refs_to_macromolecule
        if self.bridge is None:
            return ToolStepResult(tool="transparency", success=False,
                                  error="ChimeraX bridge unavailable.")
        ui       = user_input or inputs.get("_user_input", "")
        model_id = str(inputs.get("model_id") or self._primary_model_id())
        tgt      = self._resolve_opclass_target(ui, model_id)
        if tgt["defer"]:
            return ToolStepResult(
                tool="transparency", success=False,
                error=("This transparency request targets a finer selection (e.g. a residue "
                       "range, a zone, or a binding pocket) the resolver can't express. "
                       "Rephrase with a chain (e.g. 'chain A'), 'the ligand', or the whole "
                       "model."))
        spec          = tgt["spec"]
        mode, amount  = self._parse_transparency_request(ui)
        if mode == "absolute":
            level = max(0, min(100, amount))
        else:
            level = max(0, min(100, self._transparency_levels.get(spec, 0) + amount))
        self._transparency_levels[spec] = level
        # EXPLICIT target abcs = atoms+bonds+cartoons+surfaces, so transparency applies
        # whatever the shown representation. Two ChimeraX-1.11.1 gotchas this avoids:
        # (1) a BARE `transparency #N 50` (no target) SILENTLY no-ops on cartoons (the
        #     verified §5 silent-success gap — see _run_conformer_comparison); and
        # (2) `target acs` omits BONDS, so a stick/ball-and-stick rep couldn't be set/reset.
        # abcs covers both — cartoon/surface (c/s), spheres (a), and sticks/ball-and-stick (b).
        cmd  = f"transparency {spec} {level} target abcs"
        cmd  = _scope_chain_refs_to_macromolecule([cmd])[0][0]   # scope a bare chain ref
        r = self.bridge.run_command(cmd)
        if r.get("error"):
            return ToolStepResult(tool="transparency", success=False,
                                  error=f"Command failed: {cmd!r} → {r['error']}")
        return ToolStepResult(
            tool="transparency", success=True,
            summary=(f"Set transparency to {level}% on {tgt['target_desc']} "
                     f"({mode})\nCommands: {cmd}"),
            viz_commands=[], viz_explanations=[],
            data={"resolution": mode, "level": level, "spec": spec, "commands": [cmd]},
        )

    def __repr__(self) -> str:
        extra = [t for t in (self._camsol_bridge, self._esm_bridge, self._proteinmpnn_bridge)
                 if t is not None]
        return (
            f"<ToolRouter bridge={self.bridge!r} "
            f"loaded_bridges={len(extra)}>"
        )
