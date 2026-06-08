"""
intent_registry.py
------------------
Intent/Render separation framework.

DESIGN PRINCIPLE (§0):
  "Intent/Render separation — the LLM infers and clarifies intent; it never writes
  tool syntax."  For covered operations, the translation backend returns a canonical
  INTENT LABEL from a closed set — never raw ChimeraX/tool syntax.  Syntax lives in
  the render layer (here), is probe-verified once, and cannot be reintroduced wrong
  by the model.  A classifier may pick the wrong label (recoverable, surfaced), but
  cannot emit malformed syntax.  Uncovered ops fall through to the guarded free-
  translation path.

Resolution pipeline for a user phrase:
  (a) Deterministic alias match — no LLM; instant; 100% for listed phrases.
  (b) LLM constrained classifier — given the closed intent list, it returns a
      LABEL (or "none"); NEVER syntax.  This is the only LLM role for covered ops.
  (c) Graceful miss — if unresolved, list available intents and ask.

Adding an intent: add a new IntentDef to the registry; syntax change → edit render_fn only.
Adding a synonym: add one line to the aliases tuple.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Intent definition ─────────────────────────────────────────────────────────

@dataclass
class IntentDef:
    name:        str          # canonical key, e.g. "view.cartoon_only"
    category:    str          # grouping, e.g. "view"
    aliases:     tuple        # lowercase trigger phrases for alias tier
    render_fn:   Callable     # (spec: str) -> List[str]
    description: str = ""     # human-readable label shown in graceful-miss list
    verify_fn:   Optional[Callable] = None  # (spec: str, bridge: Any) -> Optional[bool]


# ── Registry ──────────────────────────────────────────────────────────────────

class IntentRegistry:
    """
    Registry for covered intent operations.

    Each registered IntentDef holds: aliases (tier-a), a render function
    (single source of truth for syntax), and an optional verify function.

    Resolution pipeline: resolve_alias → LLM fallback (if llm_fn supplied) → miss.
    """

    def __init__(self) -> None:
        self._intents: Dict[str, IntentDef] = {}

    def register(self, defn: IntentDef) -> None:
        self._intents[defn.name] = defn

    # ── Tier (a): deterministic alias match ───────────────────────────────────

    def resolve_alias(self, text: str) -> Optional[str]:
        """Tier (a): deterministic alias match. Returns intent key or None."""
        low = text.lower().strip()
        for name, defn in self._intents.items():
            if any(alias in low for alias in defn.aliases):
                return name
        return None

    # ── Full resolution pipeline ──────────────────────────────────────────────

    def resolve(
        self,
        text: str,
        llm_classify_fn: Optional[Callable[[str, List[str]], Optional[str]]] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Full 3-tier resolution pipeline.

        Returns (intent_key, method) where method is "alias", "llm", or "miss".
        The LLM classifier is invoked with (text, labels) and must return a valid
        intent key or None — never syntax.  Any non-key response is treated as None.
        """
        key = self.resolve_alias(text)
        if key:
            return key, "alias"

        if llm_classify_fn:
            labels = list(self._intents.keys())
            try:
                key = llm_classify_fn(text, labels)
            except Exception:
                key = None
            # Enforce: accept only valid keys — never syntax, never free text
            if key and key in self._intents:
                return key, "llm"

        return None, "miss"

    # ── Render layer ──────────────────────────────────────────────────────────

    def render(self, intent_key: str, spec: str) -> List[str]:
        """Single source of truth for syntax: intent + spec → verified command list."""
        defn = self._intents.get(intent_key)
        if defn is None:
            raise KeyError(f"Unknown intent: {intent_key!r}")
        return defn.render_fn(spec)

    # ── Post-command verify guard ─────────────────────────────────────────────

    def verify(self, intent_key: str, spec: str, bridge: Any) -> Optional[bool]:
        """
        Run post-command verify check. Returns True (ok), False (state unchanged),
        or None (no verify registered / probe failed).
        """
        defn = self._intents.get(intent_key)
        if defn is None or defn.verify_fn is None:
            return None
        try:
            return defn.verify_fn(spec, bridge)
        except Exception:
            return None

    # ── Catalogue helpers ─────────────────────────────────────────────────────

    def list_intent_keys(self, category: Optional[str] = None) -> List[str]:
        if category:
            return [k for k, d in self._intents.items() if d.category == category]
        return list(self._intents.keys())

    def get_defn(self, intent_key: str) -> Optional[IntentDef]:
        return self._intents.get(intent_key)

    def graceful_miss_message(self, text: str, category: str = "view") -> str:
        intents = [d for d in self._intents.values() if d.category == category]
        lines = [
            f'Could not match "{text}" to a known {category} representation.',
            "",
            f"Available {category} representations:",
        ]
        for defn in intents:
            lines.append(f"  {defn.name:<28} — {defn.description}")
        lines.extend([
            "",
            "Please rephrase using one of these, or specify a representation directly.",
        ])
        return "\n".join(lines)

    # ── Interception helpers ──────────────────────────────────────────────────

    def detect_category_phrase(self, text: str, category: str = "view") -> bool:
        """
        Broad category check for pre-translate interception.  True if the phrase
        likely belongs to this category.

        For the "view" category two tiers run in order:
          (1a) Authoritative alias match — resolve_alias() positive → True.
          (1b) Conservative noun-floor — unambiguous representation nouns are always
               flagged; ambiguous nouns (atom/sphere/surface) require a nearby display
               or removal verb (show/hide/add/ditch/…).
        False positives route through alias/LLM resolution → graceful miss, not bad syntax.
        """
        if not text:
            return False
        if category == "view":
            low = text.lower()
            if self.resolve_alias(text) is not None:
                return True
            return _representation_noun_floor(low)
        low = text.lower()
        triggers = _CATEGORY_TRIGGERS.get(category, ())
        return any(t in low for t in triggers)

    def is_covered_phrase(self, text: str) -> bool:
        """True if text matches any registered alias (fastest interception check)."""
        return self.resolve_alias(text) is not None


# ── Noun-floor: structural representation detection (PART 1) ─────────────────
# detect_category_phrase() for "view" uses two tiers:
#   1a. resolve_alias() positive → authoritative, always True
#   1b. _representation_noun_floor() — unambiguous rep nouns are always flagged;
#       ambiguous nouns (atom/sphere/surface) require a nearby display/removal verb.
#       "surface" is verb-required so bare analysis phrases ("binding surface",
#       "surface of the dimer") reach free-translation rather than the rep gate.

_REP_UNAMBIGUOUS_NOUNS: frozenset = frozenset([
    # Representation type names — always a display concept in structural biology
    "cartoon", "ribbon",
    "spacefill", "space-fill", "space fill",
    "sticks", "stick",
    "ball-and-stick", "ball and stick", "ball & stick", "balls",
    "wireframe", "wire", "mesh", "licorice", "cpk",
])

# Ambiguous nouns — only flag when a display/removal verb is also in the phrase.
# "surface"/"surfaces" move here (verb-required tier) so bare analysis phrases like
# "binding surface", "surface of the dimer", "surface contacts between chains"
# reach free-translation rather than the representation gate.
_REP_AMBIGUOUS_NOUNS: frozenset = frozenset([
    "atom", "atoms", "sphere", "spheres", "surface", "surfaces",
])
_REP_DISPLAY_VERBS: frozenset = frozenset([
    "show", "hide", "remove", "display", "style", "reveal",
    "add", "apply", "ditch", "lose", "drop", "get rid of", "no more",
])


def _has_display_verb(low: str) -> bool:
    """True if the lowercased phrase contains a display/removal verb as a whole word."""
    for verb in _REP_DISPLAY_VERBS:
        # Multi-word verbs ("get rid of", "no more") are safe to substring-match
        # because they're long enough to be unambiguous.  Single-word verbs need
        # word-boundary protection so "drop" doesn't fire inside "hydrophobic" and
        # "lose" doesn't fire inside "closely".
        if " " in verb:
            if verb in low:
                return True
        else:
            if re.search(r"\b" + re.escape(verb) + r"\b", low):
                return True
    return False


def _representation_noun_floor(low: str) -> bool:
    """
    Conservative noun-floor for representation detection.
    True iff the lowercased phrase structurally refers to a viewer representation.

    Tiers:
      1. Unambiguous nouns (cartoon/sticks/balls/wireframe/…) — always flagged.
      2. Bare "surface"/"surfaces" as the complete input — flagged (most likely
         a rep command, no other context to disambiguate).
      3. Ambiguous nouns (atom/atoms/sphere/spheres/surface/surfaces) — flagged
         ONLY when a display/removal verb is also present in the phrase.
    """
    for noun in _REP_UNAMBIGUOUS_NOUNS:
        if noun in low:
            return True
    if low.strip() in ("surface", "surfaces"):
        return True
    if any(noun in low for noun in _REP_AMBIGUOUS_NOUNS) and _has_display_verb(low):
        return True
    return False


# ── Emission guard (PART 2) ───────────────────────────────────────────────────
# Representation-shaped commands must originate from the render layer.
# Used by main._execute_commands() to block any rep-shaped command that arrives
# via free-translation (the only other execution path).

_REP_CMD_RE = re.compile(
    r"^~?surface\s+#|"                                          # surface/~surface #spec
    r"^style\s+#[^\s]+\s+(sphere|stick|ball)\b|"               # style #spec sphere/stick/ball
    r"^(hide|show)\s+#[^\s]+"
    r"\s+(atoms?|cartoons?|surfaces?|sphere|spheres?"
    r"|stick|sticks?|ball|balls?|ribbon)\b|"                   # hide/show #spec <rep-target>
    r"^(cartoon|ribbon|spacefill|licorice|wireframe|cpk)\s+#", # bare rep cmd with spec
    re.IGNORECASE,
)


def is_representation_shaped(cmd: str) -> bool:
    """
    True if cmd looks like a viewer representation command that must only originate
    from the render layer (_run_representation), never from free-translation.
    Requires a model spec (#…) to avoid blocking 'hide solvent atoms' etc.
    """
    return bool(_REP_CMD_RE.match(cmd.strip()))


# ── Category triggers (broader than aliases) ──────────────────────────────────
# Fallback phrase list for non-"view" categories (currently "view" uses noun-floor).
# Retained for forward compatibility if other categories are added.

_CATEGORY_TRIGGERS: Dict[str, tuple] = {
    "view": (
        # Representation type words (in structural-biology context, nearly
        # unambiguous; surface/ribbon/sphere almost always mean display modes)
        "cartoon", "ribbon",
        "spacefill", "space-fill", "space fill",
        "licorice", "cpk", "wireframe",
        # Sphere/stick nouns combined with action or as standalone rep intent
        "show as sphere", "show as sticks", "show as stick",
        "sphere mode", "sphere style", "sphere representation",
        "stick mode", "stick style", "stick representation",
        "ball-and-stick", "ball and stick", "ball & stick",
        # Surface display specifically (not surface area / accessibility)
        # "show surface" omitted — too ambiguous ("show surface area" false-positive)
        "show as surface", "hide surface",
        "surface mode", "surface representation",
        # Atom show/hide as representation operation
        "show atoms", "hide atoms", "show cartoons", "hide cartoons",
        "show the atoms", "hide the atoms",
        "hide spheres", "remove spheres", "hide the spheres", "remove the spheres",
        # Plural/quantified sphere variants ("remove all spheres / atoms", "all spheres")
        "all spheres", "all atoms",
        # Natural-language paraphrases that contain the representation nouns
        "back to ribbon", "back to cartoon",
        "just the ribbon", "just the cartoon",
        "just ribbon", "just cartoon",
        "show as cartoon", "show as ribbon",
        "change to cartoon", "switch to cartoon", "switch to ribbon",
        "change to sphere", "change to ribbon",
        "cartoon mode", "ribbon mode",
        "atom display", "atom style",
        "display mode", "representation mode",
        # hide/remove cartoon
        "hide cartoon", "hide ribbon", "remove cartoon", "remove ribbon",
        "no ribbon", "no cartoon",
        # Undo/revert representation — triggers for "strip it back", "undo that", etc.
        "strip it back", "undo that", "put it back",
        "revert that", "revert the view", "restore the view",
        "go back to the previous",
    ),
}


# ── Verify helper ─────────────────────────────────────────────────────────────

def _probe_atom_count(spec: str, bridge: Any) -> Optional[int]:
    """
    Query the number of displayed atoms on the model matching *spec* via runscript.
    Returns atom count or None on failure.
    """
    model_root = spec.lstrip("#").split("/")[0].split(".")[0]
    script = (
        "from chimerax.atomic import all_atomic_structures\n"
        "for m in all_atomic_structures(session):\n"
        f"    if m.id_string.startswith('{model_root}'):\n"
        "        print(int(m.atoms.displays.sum()))\n"
        "        break\n"
    )
    try:
        fd, path = tempfile.mkstemp(suffix=".py")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(script)
            r = bridge.run_command(f"runscript {path}")
            val = (r.get("value") or "").strip()
            if val.isdigit():
                return int(val)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except Exception:
        pass
    return None


# ── LLM classifier factory ────────────────────────────────────────────────────

def make_llm_classify_fn(backend_name: Optional[str] = None) -> Callable:
    """
    Create a constrained LLM classifier for tier (b) resolution.

    Returns callable: (text: str, labels: List[str]) -> Optional[str].
    The callable returns a valid intent key, "uncovered" (representation-category
    but no specific match), or None — NEVER syntax.

    Backend: tries the configured TRANSLATOR_BACKEND first; falls back to Ollama
    on any failure (e.g. API cap).  The Ollama call uses `think: false` at the
    TOP LEVEL of the request body — required for qwen3; NOT inside options{}.
    """
    import config as _cfg
    _backend      = backend_name or getattr(_cfg, "TRANSLATOR_BACKEND", "claude")
    _ollama_model = os.environ.get("OLLAMA_MODEL", getattr(_cfg, "OLLAMA_MODEL", "qwen3:8b"))
    _ollama_url   = (
        os.environ.get("OLLAMA_HOST")
        or getattr(_cfg, "OLLAMA_HOST", None)
        or "http://localhost:11434"
    ).rstrip("/")

    def _build_prompt(text: str, all_labels: List[str]) -> str:
        lines = []
        for lbl in all_labels:
            defn = VIEWER_REGISTRY._intents.get(lbl)
            desc = f" — {defn.description}" if defn else ""
            lines.append(f"  {lbl}{desc}")
        label_list = "\n".join(lines)
        return (
            f'User request: "{text}"\n\n'
            "Which viewer representation intent best matches?\n"
            "Direction rules:\n"
            "  HIDE: 'lose', 'get rid of', 'remove', 'ditch', 'hide' "
            "→ view.hide_atoms (sticks/balls/spheres), view.hide_cartoon, or view.hide_surface.\n"
            "  SHOW: 'show', 'display', 'reveal' + a rep noun "
            "→ use the matching show intent (e.g. 'show the balls' → view.spacefill).\n"
            "IMPORTANT: there is no view.hide_sticks, view.hide_balls, "
            "view.hide_spheres, or view.hide_surface_area.\n"
            "Reply with ONLY the exact key string, nothing else.\n"
            "Use 'uncovered' if about representation but no intent matches.\n"
            "Use 'none' if NOT a representation request.\n\n"
            f"Valid intents:\n{label_list}"
        )

    def _parse_resp(resp: str, all_labels: List[str]) -> Optional[str]:
        r = resp.strip().lower()
        if r in all_labels:
            return r
        first = r.split()[0].rstrip(".,;:") if r else ""
        return first if first in all_labels else None

    def _call_ollama(text: str, labels: List[str]) -> Optional[str]:
        import requests as _req
        all_labels = list(labels) + ["uncovered", "none"]
        r = _req.post(
            f"{_ollama_url}/api/generate",
            json={
                "model":   _ollama_model,
                "prompt":  _build_prompt(text, all_labels),
                "stream":  False,
                "think":   False,          # top-level flag for qwen3 — NOT inside options
                "options": {"temperature": 0.0, "num_predict": 60},
            },
            timeout=20,
        )
        return _parse_resp(r.json().get("response", ""), all_labels)

    # Latch: set to True on first Claude API failure — subsequent calls skip Claude
    _claude_capped = [False]

    def classify(text: str, labels: List[str]) -> Optional[str]:
        all_labels = list(labels) + ["uncovered", "none"]

        if _backend == "claude" and not _claude_capped[0]:
            try:
                import anthropic
                client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
                msg = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=30,
                    messages=[{"role": "user", "content": _build_prompt(text, all_labels)}],
                )
                result = _parse_resp(msg.content[0].text or "", all_labels)
                if result is not None:
                    return result
            except Exception:
                _claude_capped[0] = True  # latch — skip Claude for this session

        # Ollama — primary when backend=ollama, fallback when Claude failed/capped
        try:
            return _call_ollama(text, labels)
        except Exception:
            return None

    return classify


# ── Viewer intent registry ────────────────────────────────────────────────────
# Probe-verified command sequences (STEP 0 — ChimeraX 1.11.1, 2D31):
#   hide #N atoms          → atoms_shown=0
#   show #N cartoons       → ribbon_shown>0
#   show #N atoms          → atoms_shown=6124 (2D31)
#   style #N sphere        → "Changed 6124 atom styles"
#   style #N stick         → "Changed 6124 atom styles"
#   style #N ball          → "Changed 6124 atom styles"
#   surface #N             → OK
#   ~surface #N            → OK (no-op when no surface)
#   hide #N surfaces       → OK
#   sub-model spec #2.1/A  → all commands accept it unchanged

VIEWER_REGISTRY = IntentRegistry()

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.cartoon_only",
    category    = "view",
    description = "cartoon/ribbon view — atoms hidden, backbone trace shown",
    aliases     = (
        "cartoon only", "cartoon mode", "cartoon-only",
        "cartoon representation", "cartoon view",
        "change to cartoon", "switch to cartoon",
        "show as cartoon", "show cartoon", "just cartoon",
        "ribbon only", "ribbon mode", "ribbon-only",
        "ribbon representation", "ribbon view",
        "switch to ribbon", "just ribbon",
        "show ribbon", "show the ribbon",
        "just the ribbon", "just the cartoon",
        "cartoons only", "ribbons only",
        "back to ribbon", "back to cartoon",
        "change to ribbon",
    ),
    render_fn   = lambda spec: [
        f"hide {spec} atoms",
        f"show {spec} cartoons",
        f"~surface {spec}",
    ],
    verify_fn   = lambda spec, bridge: (
        lambda n: None if n is None else (n == 0)
    )(_probe_atom_count(spec, bridge)),
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.spacefill",
    category    = "view",
    description = "switch to sphere/spacefill/CPK style — show atoms as spheres (aka balls/CPK)",
    aliases     = (
        "space filled",
        "sphere mode", "sphere style", "sphere representation",
        "show as spheres", "show as spacefill", "show as cpk",
        "vdw", "van der waals",
        "atom representation", "show atoms as spheres",
        "change to spheres", "change to sphere",
    ),
    render_fn   = lambda spec: [
        f"show {spec} atoms",
        f"style {spec} sphere",
    ],
    verify_fn   = lambda spec, bridge: (
        lambda n: None if n is None else (n > 0)
    )(_probe_atom_count(spec, bridge)),
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.sticks",
    category    = "view",
    description = "stick representation",
    aliases     = (
        "stick mode", "stick style",
        "show as sticks", "show as stick",
        "stick representation",
        "show sticks", "all sticks",
    ),
    render_fn   = lambda spec: [
        f"show {spec} atoms",
        f"style {spec} stick",
    ],
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.ball_and_stick",
    category    = "view",
    description = "ball-and-stick style — show atoms as small balls connected by sticks",
    aliases     = (
        "ball stick", "balls and sticks",
        "ball-and-stick representation", "ball and stick mode",
        "ball and stick style",
    ),
    render_fn   = lambda spec: [
        f"show {spec} atoms",
        f"style {spec} ball",
    ],
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.surface",
    category    = "view",
    description = "molecular surface display",
    aliases     = (
        "show as surface",
        "molecular surface", "surface representation",
        "solvent accessible surface", "solvent-accessible surface",
        "surface mode", "sas", "surface view",
        "show the surface",
    ),
    render_fn   = lambda spec: [
        f"surface {spec}",
    ],
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.hide_atoms",
    category    = "view",
    description = "HIDE/REMOVE atom display — sticks, spheres, balls, ball-and-stick (keeps cartoon/surface if shown)",
    aliases     = (
        "hide atoms", "hide the atoms", "no atoms", "atoms off",
        "hide spheres", "hide the spheres",
        "remove atoms", "remove the atoms",
        "remove spheres", "remove the spheres",
        "turn off atoms",
    ),
    render_fn   = lambda spec: [
        f"hide {spec} atoms",
    ],
    verify_fn   = lambda spec, bridge: (
        lambda n: None if n is None else (n == 0)
    )(_probe_atom_count(spec, bridge)),
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.hide_cartoon",
    category    = "view",
    description = "hide cartoon/ribbon display",
    aliases     = (
        "hide cartoon", "hide the cartoon",
        "hide ribbon", "hide the ribbon",
        "remove cartoon", "remove ribbon",
        "no ribbon", "no cartoon",
        "cartoon off", "ribbon off",
        "turn off cartoon", "turn off ribbon",
    ),
    render_fn   = lambda spec: [
        f"hide {spec} cartoons",
    ],
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.hide_surface",
    category    = "view",
    description = "remove surface display",
    aliases     = (
        "hide surface", "hide the surface",
        "remove surface", "surface off",
        "no surface", "turn off surface",
    ),
    render_fn   = lambda spec: [
        f"~surface {spec}",
    ],
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.show_atoms",
    category    = "view",
    description = "show atom display",
    aliases     = (
        "show atoms", "show the atoms", "display atoms",
        "atoms on", "reveal atoms",
        "turn on atoms", "show all atoms",
    ),
    render_fn   = lambda spec: [
        f"show {spec} atoms",
    ],
    verify_fn   = lambda spec, bridge: (
        lambda n: None if n is None else (n > 0)
    )(_probe_atom_count(spec, bridge)),
))

VIEWER_REGISTRY.register(IntentDef(
    name        = "view.undo_representation",
    category    = "view",
    description = "undo/revert last representation change for this target",
    aliases     = (
        "strip it back", "undo that", "undo the last",
        "put it back", "revert that", "revert the view",
        "restore the view", "restore view",
        "go back to the previous view",
        "never mind the change", "undo the representation",
        "cancel the representation change",
    ),
    render_fn   = lambda spec: [],  # Overridden in _run_representation; never called directly
))
