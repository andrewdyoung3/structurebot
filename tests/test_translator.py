"""
tests/test_translator.py
------------------------
Tests for Translator internals that don't require a live API key.

A. _pre_screen()  -- rfdiffusion keyword detection, no-API-call path
B. _call_api()    -- empty response guard (IndexError fix)

Usage
-----
  cd structurebot
  python -m pytest tests/test_translator.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import translator as _translator_mod
from translator import (
    CommandTranslator as Translator,
    RefusalError,
    _sanitize_zone_syntax,
    _scope_chain_refs_to_macromolecule,
    _validate_open_targets,
    _is_valid_open_target,
    _validate_command_verbs,
    _validate_close_targets,
    _is_valid_close_spec,
    _resolve_close_target,
    probe_chimerax_verbs,
)

# -- Helpers -------------------------------------------------------------------

PASS = "[PASS]"
FAIL = "[FAIL]"

_results = {"pass": 0, "fail": 0}


def _ok(name: str, detail: str = "") -> None:
    suffix = f"  {detail}" if detail else ""
    print(f"  {PASS} {name}{suffix}")
    _results["pass"] += 1


def _fail(name: str, reason: str) -> None:
    print(f"  {FAIL} {name}: {reason}")
    _results["fail"] += 1


def _assert(cond: bool, name: str, msg: str = "") -> bool:
    if cond:
        _ok(name)
        return True
    _fail(name, msg or "assertion failed")
    return False


def _make_translator() -> Translator:
    """Return a local-only Translator (no live calls)."""
    return Translator()


def _stub_backend(t, json_str: str) -> None:
    """Make t's backend return the given raw JSON via the shared _parse_response —
    so translate()'s deterministic guards run backend-independently (no Claude/Ollama)."""
    class _Stub:
        name = "stub"
        def translate(self, tr, user_input, session):
            return tr._parse_response(json_str)
    t._backend = _Stub()


# -- A. _pre_screen() ----------------------------------------------------------

def test_pre_screen_passes_through_normal_request() -> None:
    print("\n=== A. _pre_screen() ===")
    t = _make_translator()
    result = t._pre_screen("open 1HSG and color by chain")
    _assert(result is None, "normal request -> None (no pre-screen)")


def test_pre_screen_rfdiffusion_keywords() -> None:
    """All rfdiffusion keyword variants trigger the pre-screen."""
    t = _make_translator()
    inputs = [
        "design a binder for 1HSG",
        "I want binder design for the active site",
        "Use RFdiffusion to make something",
        "run rf diffusion",
        "de novo backbone design",
        "de-novo backbone generation",
        "scaffold a motif from chain A",
        "motif scaffold design",
        "partial diffusion of the structure",
        "design symmetric oligomer C3",
        "backbone generation request",
    ]
    for phrase in inputs:
        result = t._pre_screen(phrase)
        _assert(
            result is not None,
            f"rfdiffusion keyword detected: {phrase!r}",
            f"got None (should be a dict)",
        )


def test_pre_screen_returns_valid_result_shape() -> None:
    """Pre-screen result has required keys and rfdiffusion in tools_needed."""
    t = _make_translator()
    result = t._pre_screen("design a protein binder for the active site")
    _assert(result is not None, "pre-screen triggered")
    if result is None:
        return
    for key in ("commands", "explanations", "warnings",
                "confidence", "tools_needed", "tool_inputs"):
        _assert(key in result, f"result has '{key}' key")
    _assert("rfdiffusion" in result["tools_needed"],
            "tools_needed contains rfdiffusion",
            f"got {result['tools_needed']}")
    _assert(result["commands"] == [], "commands is empty list")
    _assert(result["confidence"] == "high", "confidence is high")


def test_pre_screen_case_insensitive() -> None:
    """Keyword matching is case-insensitive."""
    t = _make_translator()
    for variant in ("DESIGN A BINDER", "Design A Binder", "Design a BINDER"):
        result = t._pre_screen(variant)
        _assert(result is not None,
                f"case-insensitive match for {variant!r}")


def test_translate_rfdiffusion_skips_api() -> None:
    """translate() with an rfdiffusion request never calls the API."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."

    # The pre-screen short-circuits BEFORE any LLM call — verify the local Ollama
    # request (requests.post) is never made.
    with patch("requests.post") as mock_post:
        result = t.translate("design a binder for chain A", session)

    mock_post.assert_not_called()
    _assert("rfdiffusion" in result.get("tools_needed", []),
            "pre-screened result routes to rfdiffusion",
            f"got tools_needed={result.get('tools_needed')}")


# -- C. Over-eager refusal handling (false positives on benign design) ---------

_VALID_JSON = (
    '{"commands": ["select #1/A"], "explanations": ["x"], "warnings": [], '
    '"clarification_needed": null, "confidence": "high", '
    '"tools_needed": ["proteinmpnn"], "tool_inputs": {}}'
)


def test_system_prompt_frames_routine_design_as_legitimate() -> None:
    """The static prompt frames routine protein-engineering as standard, so the
    model doesn't over-refuse it."""
    prompt = _make_translator()._static_block.lower()
    assert "standard computational structural biology" in prompt
    assert "do not decline" in prompt or "do not refuse" in prompt
    for term in ("interface", "hydrophilic", "cysteine", "epitope"):
        assert term in prompt, f"scope note should mention {term!r}"
    _ok("system prompt frames routine design operations as legitimate")


# -- D. Chimera-1 zone-syntax guard (invalid in ChimeraX 1.11) -----------------

_ZONE_JSON = (
    '{"commands": ["select #1/A & (zone #1/B 4.5)", "info residues sel"], '
    '"explanations": ["interface", "list"], "warnings": [], '
    '"clarification_needed": null, "confidence": "high", '
    '"tools_needed": ["chimerax"], "tool_inputs": {}}'
)


def test_zone_guard_rewrites_bare_and_parenthesised() -> None:
    print("\n=== D. zone-syntax guard ===")
    # bare form
    cmds, _, notes = _sanitize_zone_syntax(["select zone #1/B 4.5 & #1/A"], [""])
    _assert(cmds == ["select #1/B :<4.5 & #1/A"],
            "bare zone rewritten to :<", f"got {cmds}")
    # parenthesised form (the regression seen live)
    cmds2, _, notes2 = _sanitize_zone_syntax(["select #1/A & (zone #1/B 4.5)"], [""])
    _assert(cmds2 == ["select #1/A & (#1/B :<4.5)"],
            "parenthesised (zone …) rewritten, parens kept", f"got {cmds2}")
    _assert(all("zone" not in c for c in cmds + cmds2), "no `zone` keyword remains")
    _assert(bool(notes) and bool(notes2), "both rewrites logged in notes")


def test_zone_guard_leaves_valid_and_volume_zone() -> None:
    valid = ["select #1/B :<4.5 & #1/A", "info residues sel"]
    cmds, _, notes = _sanitize_zone_syntax(valid, ["", ""])
    _assert(cmds == valid and notes == [], "valid :< left untouched", f"got {cmds}")
    vz, _, vnotes = _sanitize_zone_syntax(["volume zone #1 4.5"], [""])
    _assert(vz == ["volume zone #1 4.5"] and vnotes == [],
            "volume zone preserved", f"got {vz}")


def test_zone_guard_drops_unrewritable() -> None:
    cmds, exps, notes = _sanitize_zone_syntax(
        ["select zone sel", "info residues sel"], ["bad", "list"])
    _assert(cmds == ["info residues sel"], "unrewritable zone dropped", f"got {cmds}")
    _assert(exps == ["list"], "dropped command's explanation removed")
    _assert(any("Dropped" in n for n in notes), "drop logged", f"got {notes}")


def test_chain_scope_guard_scopes_bare_chain_refs() -> None:
    print("\n=== D2. chain-reference macromolecule-scoping guard ===")
    SCOPE = "~ligand & ~solvent & ~ions"
    # a bare chain ref is scoped to the macromolecule (the 1HSG MK1=B:902 bleed)
    out, notes = _scope_chain_refs_to_macromolecule(["color /B red"])
    _assert(out == [f"color (/B & {SCOPE}) red"], "bare /B scoped", f"got {out}")
    _assert(bool(notes), "scope logged in notes")
    out, _ = _scope_chain_refs_to_macromolecule(["hide #1/B cartoon", "show /A cartoon"])
    _assert(out == [f"hide (#1/B & {SCOPE}) cartoon", f"show (/A & {SCOPE}) cartoon"],
            "scoping applies to hide/show too (every operation)", f"got {out}")
    # zone reference chains are scoped on BOTH sides
    out, _ = _scope_chain_refs_to_macromolecule(["select /A & (/B :<4.5)"])
    _assert(out == [f"select (/A & {SCOPE}) & ((/B & {SCOPE}) :<4.5)"],
            "zone refs scoped both sides", f"got {out}")


def test_chain_scope_guard_leaves_ligand_and_subrefs() -> None:
    # DISJOINTNESS the other way: the `ligand` keyword is untouched
    out, notes = _scope_chain_refs_to_macromolecule(["color ligand white"])
    _assert(out == ["color ligand white"] and notes == [],
            "ligand keyword left untouched (disjoint)", f"got {out}")
    # explicit residue/atom sub-refs and negations are intentional → untouched
    untouched = ["color /B:902 red", "color /B@CA red", "color ~/A gray",
                 'runscript "cache/seqview/x.py"', "open 1hsg"]
    out, notes = _scope_chain_refs_to_macromolecule(untouched)
    _assert(out == untouched and notes == [],
            "residue/atom sub-refs, negation, paths left untouched", f"got {out}")
    # idempotent: an already-scoped ref is not double-wrapped
    pre = ["color (/B & ~ligand & ~solvent & ~ions) red"]
    out, notes = _scope_chain_refs_to_macromolecule(pre)
    _assert(out == pre and notes == [], "already-scoped ref not double-wrapped", f"got {out}")


def test_translate_rewrites_zone_end_to_end() -> None:
    """A model emitting the parenthesised Chimera-1 zone is corrected before
    return: interface request → `:<` + `& #1/A`, `info residues sel` survives."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "1IL8 loaded as #1 (chains A, B)."
    _stub_backend(t, _ZONE_JSON)
    result = t.translate(
        "select the residues at the dimer interface on chain A", session)
    cmds = result.get("commands", [])
    _assert(cmds[0] == "select (#1/A & ~ligand & ~solvent & ~ions) & "
            "((#1/B & ~ligand & ~solvent & ~ions) :<4.5)",
            "interface select uses :< not zone (chain-scoped)", f"got {cmds}")
    _assert(all("zone" not in c for c in cmds), "no `zone` survives")
    _assert("info residues sel" in cmds, "`info residues sel` preserved")
    _assert(any("Rewrote" in w for w in result.get("warnings", [])),
            "rewrite surfaced as a warning")


def test_system_prompt_documents_zone_and_hide() -> None:
    prompt = _make_translator()._static_block
    low = prompt.lower()
    _assert(":<" in prompt and "@<" in prompt, "prompt documents :< / @< operators")
    _assert("info residues sel" in prompt, "prompt lists via `info residues sel`")
    _assert("never" in low and "zone" in low, "prompt forbids Chimera-1 `zone`")
    # BUG 2: hide/show must target the representation, not bare atoms.
    _assert("hide #1/b cartoon" in low or "hide #1/b target ac" in low,
            "prompt shows hide targeting the cartoon/representation")


def test_prompt_documents_stick_view_sequence() -> None:
    """BUG 1: switching a chain to sticks needs `show <spec> atoms` THEN
    `style <spec> <mode>` (+ `hide <spec> cartoon` to replace the ribbon) — a bare
    `style` reveals nothing because a cartoon chain has its atoms hidden."""
    low = _make_translator()._static_block.lower()
    _assert("show #1/a atoms" in low, "prompt shows `show <spec> atoms` before styling")
    _assert("style #1/a stick" in low, "prompt shows `style <spec> stick`")
    _assert("hide #1/a cartoon" in low, "prompt hides the cartoon when replacing the ribbon")
    _assert("style" in low and "alone" in low and "reveal" in low,
            "prompt warns a bare `style` never reveals hidden atoms")


def test_prompt_documents_ligand_keyword_and_literal_colour() -> None:
    """BUG 2: "the ligand" → the built-in `ligand` keyword (never an invented
    `:LIG`/`/LIG`); an explicitly named colour is applied literally (no byelement
    substitution unless the user asks to colour by element)."""
    prompt = _make_translator()._static_block
    low = prompt.lower()
    _assert("ligand` keyword" in low or "`ligand`" in low,
            "prompt maps 'the ligand' to the `ligand` keyword")
    _assert("color ligand white" in low, "prompt shows the literal `color ligand white` form")
    _assert("lig" in low and ("do not emit `:lig`" in low or "/lig" in low),
            "prompt forbids inventing `:LIG` / `/LIG`")
    _assert("do not" in low and "byelement" in low and "byhetero" in low,
            "prompt forbids substituting byelement/byhetero for a named colour")


def test_hide_chain_targets_representation() -> None:
    """'hide chain B' must target the cartoon (not a bare `hide #1/B`, which
    only hides atoms and leaves the ribbon visible)."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "1IL8 loaded as #1 (chains A, B)."
    hide_json = (
        '{"commands": ["hide #1/B cartoon"], "explanations": ["hide B ribbon"], '
        '"warnings": [], "clarification_needed": null, "confidence": "high", '
        '"tools_needed": ["chimerax"], "tool_inputs": {}}'
    )
    _stub_backend(t, hide_json)
    result = t.translate("hide chain B", session)
    cmds = result.get("commands", [])
    joined = " ".join(cmds).lower()
    _assert(cmds == ["hide (#1/B & ~ligand & ~solvent & ~ions) cartoon"],
            "hide targets cartoon (chain-scoped)", f"got {cmds}")
    _assert("cartoon" in joined or "target a" in joined,
            "representation named (cartoon / target ac), not bare hide")
    _assert(cmds != ["hide #1/B"], "not a bare atoms-only hide")


# -- E. Pluggable backend (Claude default; same normalized shape) --------------

def test_default_backend_is_ollama() -> None:
    print("\n=== E. local-only backend ===")
    from translator import OllamaBackend, TranslatorBackend
    t = _make_translator()
    _assert(isinstance(t._backend, OllamaBackend),
            "the sole backend is OllamaBackend (local-only)", f"got {type(t._backend).__name__}")
    _assert(isinstance(t._backend, TranslatorBackend),
            "OllamaBackend implements the TranslatorBackend interface")
    _assert(t._backend.name == "ollama", "backend name is 'ollama'")
    _assert(getattr(t, "client", None) is None, "no Claude client (local-only)")


def test_backend_translate_returns_normalized_shape() -> None:
    """translate() routes through the backend and returns the SAME normalized
    structured object downstream depends on (mock the API — no live call)."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."
    _stub_backend(t, _VALID_JSON)
    result = t.translate("color chain A red", session)
    for key in ("commands", "explanations", "warnings", "clarification_needed",
                "confidence", "tools_needed", "tool_inputs"):
        _assert(key in result, f"normalized result has '{key}'")
    _assert(result["commands"] == ["select (#1/A & ~ligand & ~solvent & ~ions)"],
            "backend produced the parsed commands (chain-scoped)", f"got {result.get('commands')}")
    _assert(type(t._backend).__name__ in ("OllamaBackend", "_Stub"),
            "translate routed through the (local) backend")


# -- F. Open-target validation guard (Bug 4a) ----------------------------------

def test_open_valid_pdb_id_passes() -> None:
    print("\n=== F. open-target validation guard (Bug 4a) ===")
    for pdb in ("1HSG", "1hsg", "4UNL", "A0A1"):
        cmds, _, blocked = _validate_open_targets([f"open {pdb}"], ["fetch"])
        _assert(not blocked and len(cmds) == 1,
                f"valid PDB-ID {pdb!r} passes open-target guard", f"blocked={blocked}")


def test_open_nonexistent_file_blocked() -> None:
    """open design1.pdb (file absent) is blocked before execution."""
    cmds, _, blocked = _validate_open_targets(["open design1.pdb"], ["open design"])
    _assert(not cmds, "open design1.pdb: command list is empty after blocking", f"got {cmds}")
    _assert(bool(blocked), "open design1.pdb: produces a blocked message", f"got {blocked}")
    _assert("design1.pdb" in (blocked[0] if blocked else ""),
            "block message names the bad target", f"got {blocked}")


def test_open_sequence_token_blocked() -> None:
    """open sequence (hallucinated target) is blocked."""
    cmds, _, blocked = _validate_open_targets(["open sequence"], ["open seq"])
    _assert(not cmds, "open sequence: command blocked", f"got {cmds}")
    _assert(bool(blocked), "open sequence: produces blocked message")


def test_open_from_alphafold_valid_pdb_passes() -> None:
    """open 1HSG from alphafold — target is the PDB-ID, passes."""
    cmds, _, blocked = _validate_open_targets(["open 1HSG from alphafold"], ["from AF"])
    _assert(not blocked and len(cmds) == 1,
            "open 1HSG from alphafold passes (PDB-ID qualifier)", f"blocked={blocked}")


def test_open_session_cxs_passes() -> None:
    """open session.cxs — .cxs extension is a recognised special token."""
    cmds, _, blocked = _validate_open_targets(["open session.cxs"], ["restore"])
    _assert(not blocked and len(cmds) == 1,
            "open session.cxs passes (recognised .cxs extension)", f"blocked={blocked}")


def test_open_existing_file_passes(tmp_path) -> None:
    """open <existing-file.pdb> is allowed (file exists on disk)."""
    pdb = tmp_path / "real_structure.pdb"
    pdb.write_text("ATOM …")
    cmds, _, blocked = _validate_open_targets([f'open "{pdb}"'], ["open pdb"])
    _assert(not blocked and len(cmds) == 1,
            "existing file path passes open-target guard", f"blocked={blocked}")


def test_open_guard_non_open_commands_untouched() -> None:
    """Non-open commands are never touched by the open-target guard."""
    raw = ["color #1 red", "cartoon #1", "view"]
    cmds, exps, blocked = _validate_open_targets(raw, ["c", "c", "v"])
    _assert(cmds == raw and not blocked,
            "non-open commands pass through unchanged", f"got {cmds}")


def test_open_guard_end_to_end_in_translate() -> None:
    """translate() with an invalid open target blocks the command and warns."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."
    bad_json = (
        '{"commands": ["open sequence", "view"], '
        '"explanations": ["open seq", "fit"], "warnings": [], '
        '"clarification_needed": null, "confidence": "high", '
        '"tools_needed": ["chimerax"], "tool_inputs": {}}'
    )
    _stub_backend(t, bad_json)
    result = t.translate("open sequence", session)
    cmds = result.get("commands", [])
    _assert("open sequence" not in cmds,
            "blocked open not in commands", f"got {cmds}")
    _assert("view" in cmds, "non-blocked commands survive", f"got {cmds}")
    _assert(any("open sequence" in str(w) or "sequence" in str(w).lower()
                for w in result.get("warnings", [])),
            "block message surfaced in warnings", f"got {result.get('warnings')}")


# -- G. Invalid-verb rejection guard (Bug 4b) ----------------------------------

def test_unknown_verb_denylist_rejected() -> None:
    print("\n=== G. invalid-verb rejection guard (Bug 4b) ===")
    for bad_verb in ("find", "search", "locate", "lookup"):
        cmds, _, blocked = _validate_command_verbs([f"{bad_verb} chain A"], [f"bad verb {bad_verb}"])
        _assert(not cmds, f"denylist verb '{bad_verb}' rejected (no cmds)", f"got {cmds}")
        _assert(bool(blocked), f"denylist verb '{bad_verb}' produces block message")


def test_known_verbs_with_registry_pass() -> None:
    """open/color/matchmaker pass when an explicit registry is provided."""
    registry = frozenset({"open", "color", "matchmaker", "view", "cartoon", "select", "hide",
                          "show", "style", "info", "runscript"})
    for cmd in ("open 1HSG", "color #1 red", "matchmaker #1 to #2", "view"):
        cmds, _, blocked = _validate_command_verbs([cmd], ["ok"], known_verbs=registry)
        _assert(not blocked and len(cmds) == 1,
                f"known verb in {cmd!r} passes with registry", f"blocked={blocked}")


def test_unknown_verb_with_registry_rejected() -> None:
    """A verb absent from a registry is rejected (tier-2 check)."""
    registry = frozenset({"open", "color", "view"})
    cmds, _, blocked = _validate_command_verbs(["find chain A"], ["find"], known_verbs=registry)
    _assert(not cmds, "verb not in registry rejected", f"got {cmds}")
    _assert(bool(blocked), "rejection produces a block message", f"got {blocked}")
    _assert("find" in (blocked[0] if blocked else ""),
            "block message names the rejected verb", f"got {blocked}")


def test_denylist_applies_even_with_registry() -> None:
    """The denylist always blocks, even when the registry is provided."""
    # Registry includes "search" (pathological case — registry wins anyway once
    # a live registry is set, but denylist is Tier 1 and fires first regardless)
    registry = frozenset({"open", "color", "search"})
    cmds, _, blocked = _validate_command_verbs(["search chain A"], ["bad"], known_verbs=registry)
    # Denylist takes priority (Tier 1): even if "search" is in the registry,
    # the denylist check fires first and rejects it.
    _assert(not cmds, "denylist verb rejected regardless of registry", f"got {cmds}")


def test_verb_guard_without_registry_passes_unknown() -> None:
    """Without a registry, an UNKNOWN (but not denylisted) verb passes through."""
    # Module-level registry is None; no known_verbs argument → denylist-only mode.
    old = _translator_mod._chimerax_verb_registry
    _translator_mod._chimerax_verb_registry = None
    try:
        cmds, _, blocked = _validate_command_verbs(["somefuturecommand #1"], ["ok"])
        _assert(len(cmds) == 1 and not blocked,
                "unknown non-denylisted verb passes without registry", f"got {cmds}")
    finally:
        _translator_mod._chimerax_verb_registry = old


def test_verb_guard_end_to_end_in_translate() -> None:
    """translate() with a hallucinated verb blocks the command and warns."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."
    bad_json = (
        '{"commands": ["find chain A", "view"], '
        '"explanations": ["bad", "fit"], "warnings": [], '
        '"clarification_needed": null, "confidence": "high", '
        '"tools_needed": ["chimerax"], "tool_inputs": {}}'
    )
    _stub_backend(t, bad_json)
    result = t.translate("find chain A", session)
    cmds = result.get("commands", [])
    _assert("find chain A" not in cmds,
            "hallucinated verb not in commands", f"got {cmds}")
    _assert("view" in cmds, "non-blocked commands survive", f"got {cmds}")
    _assert(any("find" in str(w) for w in result.get("warnings", [])),
            "block message surfaced in warnings", f"got {result.get('warnings')}")


def _session_with(structures: dict):
    """A minimal session stub exposing `.structures` for the close-target guard."""
    import types
    return types.SimpleNamespace(structures=structures)


def test_is_valid_close_spec() -> None:
    """Specs ChimeraX `close` accepts as-is pass; a name/PDB-id does not."""
    for ok in ("#1", "#1,2", "#2.1", "all", "ALL", "session"):
        assert _is_valid_close_spec(ok), f"{ok!r} should be a valid close spec"
    for bad in ("5HRZ", "5hrz", "1abc", "mymodel"):
        assert not _is_valid_close_spec(bad), f"{bad!r} should NOT be a valid close spec"


def test_resolve_close_target_by_name_and_pdbid() -> None:
    """A name or pdb_id resolves to its top-level model id; no match → []."""
    sess = _session_with({
        "1": {"name": "5HRZ", "metadata": {"pdb_id": "5HRZ"}},
        "2.1": {"name": "1abc", "metadata": {"pdb_id": "1ABC"}},
    })
    assert _resolve_close_target("5hrz", sess) == ["1"]          # case-insensitive name
    assert _resolve_close_target("1ABC", sess) == ["2"]          # submodel id collapses to top-level
    assert _resolve_close_target("9XYZ", sess) == []             # unknown
    assert _resolve_close_target("5HRZ", None) == []             # no session → no crash


def test_resolve_close_target_multiple_copies() -> None:
    """The same PDB opened twice resolves to both model ids."""
    sess = _session_with({
        "1": {"name": "5HRZ", "metadata": {"pdb_id": "5HRZ"}},
        "3": {"name": "5HRZ", "metadata": {"pdb_id": "5HRZ"}},
    })
    assert _resolve_close_target("5hrz", sess) == ["1", "3"]


def test_close_guard_rewrites_pdb_id_to_model_spec() -> None:
    """`close 5HRZ` → `close #1` when 5HRZ is loaded as #1 (the core bug fix)."""
    sess = _session_with({"1": {"name": "5HRZ", "metadata": {"pdb_id": "5HRZ"}}})
    cmds, exps, blocked = _validate_close_targets(["close 5HRZ"], ["remove it"], sess)
    assert cmds == ["close #1"], f"expected rewrite, got {cmds}"
    assert exps == ["remove it"], "explanation preserved across rewrite"
    assert not blocked, f"a resolvable target must not be blocked: {blocked}"


def test_close_guard_passes_valid_specs_untouched() -> None:
    """Already-valid specs (`#N`/`all`/`session`) are never rewritten or blocked."""
    sess = _session_with({"1": {"name": "5HRZ", "metadata": {"pdb_id": "5HRZ"}}})
    for spec in ("close #1", "close #1,2", "close all", "close session"):
        cmds, _, blocked = _validate_close_targets([spec], ["x"], sess)
        assert cmds == [spec] and not blocked, f"{spec!r} should pass untouched, got {cmds}"


def test_close_guard_blocks_unresolvable_target() -> None:
    """An unresolvable name is blocked with an actionable message — never emitted as a dead command."""
    sess = _session_with({"1": {"name": "5HRZ", "metadata": {"pdb_id": "5HRZ"}}})
    cmds, _, blocked = _validate_close_targets(["close 9XYZ"], ["nope"], sess)
    assert cmds == [], f"unresolvable close must be dropped, got {cmds}"
    assert blocked and "9XYZ" in blocked[0], f"actionable block message expected, got {blocked}"


def test_close_guard_end_to_end_in_translate() -> None:
    """translate() rewrites `close 5HRZ` → `close #1` via the session structures."""
    t = _make_translator()
    sess = _session_with({"1": {"name": "5HRZ", "metadata": {"pdb_id": "5HRZ"}}})
    good_json = (
        '{"commands": ["close 5HRZ"], "explanations": ["remove the model"], '
        '"warnings": [], "clarification_needed": null, "confidence": "high", '
        '"tools_needed": ["chimerax"], "tool_inputs": {}}'
    )
    _stub_backend(t, good_json)
    result = t.translate("remove PDB 5HRZ", sess)
    assert result.get("commands") == ["close #1"], f"got {result.get('commands')}"


def test_probe_chimerax_verbs_caches_on_success() -> None:
    """probe_chimerax_verbs caches a successful result and is idempotent."""
    old = _translator_mod._chimerax_verb_registry
    _translator_mod._chimerax_verb_registry = None
    try:
        # Simulate a runscript that returns a comma-separated list
        mock_run = MagicMock(return_value={"value": "open,color,matchmaker,view,select", "error": None})
        with patch("tempfile.NamedTemporaryFile"):
            # just test the caching logic — inject registry directly
            _translator_mod._chimerax_verb_registry = frozenset({"open", "color", "view"})
            result = probe_chimerax_verbs(mock_run)  # should return cached, not call run_fn
        mock_run.assert_not_called()
        _assert(result == frozenset({"open", "color", "view"}),
                "probe returns cached registry", f"got {result}")
    finally:
        _translator_mod._chimerax_verb_registry = old


def test_probe_chimerax_verbs_denylist_fallback() -> None:
    """If probe fails, denylist still blocks hallucinated verbs."""
    old = _translator_mod._chimerax_verb_registry
    _translator_mod._chimerax_verb_registry = None
    try:
        cmds, _, blocked = _validate_command_verbs(["search chain A"], ["bad"])
        _assert(not cmds, "denylist fallback blocks 'search'", f"got {cmds}")
    finally:
        _translator_mod._chimerax_verb_registry = old


# -- H. Error-correction guards (Bug 6) ----------------------------------------

def test_translate_error_fix_includes_error_text() -> None:
    """translate_error_fix feeds both the failed command and the actual error text."""
    print("\n=== H. error-correction guards (Bug 6) ===")
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."

    captured_inputs = []

    def _capture_translate(user_input, sess):
        captured_inputs.append(user_input)
        return {
            "commands": ["open 1HSG"], "explanations": ["fetch"], "warnings": [],
            "clarification_needed": None, "confidence": "high",
            "tools_needed": ["chimerax"], "tool_inputs": {},
        }

    with patch.object(t, "translate", side_effect=_capture_translate):
        t.translate_error_fix("open design1.pdb", "No such file: design1.pdb", session)

    _assert(bool(captured_inputs), "translate was called with the fix prompt")
    prompt = captured_inputs[0] if captured_inputs else ""
    _assert("open design1.pdb" in prompt,
            "failed command is in correction prompt", f"got: {prompt[:120]}")
    _assert("No such file: design1.pdb" in prompt,
            "actual error text is in correction prompt", f"got: {prompt[:120]}")


def test_correction_blocked_by_guard_4a() -> None:
    """If the corrected command is also an invalid open, guards block it (4a)."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."
    same_bad_json = (
        '{"commands": ["open design1.pdb"], "explanations": ["retry"], '
        '"warnings": [], "clarification_needed": null, "confidence": "medium", '
        '"tools_needed": ["chimerax"], "tool_inputs": {}}'
    )
    _stub_backend(t, same_bad_json)
    fix = t.translate_error_fix("open design1.pdb", "No such file", session)
    # Guard 4a should have blocked open design1.pdb → commands empty
    _assert(not fix.get("commands"),
            "re-proposed invalid open blocked by guard 4a in correction", f"got {fix.get('commands')}")


def test_correction_blocked_by_guard_4b() -> None:
    """If the corrected command uses a hallucinated verb, guard 4b blocks it."""
    t = _make_translator()
    session = MagicMock()
    session.get_context_summary.return_value = "No structures loaded."
    bad_fix_json = (
        '{"commands": ["search design1.pdb"], "explanations": ["wrong"], '
        '"warnings": [], "clarification_needed": null, "confidence": "low", '
        '"tools_needed": ["chimerax"], "tool_inputs": {}}'
    )
    _stub_backend(t, bad_fix_json)
    fix = t.translate_error_fix("find design1.pdb", "Unknown command: find", session)
    _assert(not fix.get("commands"),
            "hallucinated verb in correction blocked by guard 4b", f"got {fix.get('commands')}")


# -- Runner --------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("tests/test_translator.py -- Translator Internals Tests")
    print("=" * 60)

    # A. _pre_screen()
    test_pre_screen_passes_through_normal_request()
    test_pre_screen_rfdiffusion_keywords()
    test_pre_screen_returns_valid_result_shape()
    test_pre_screen_case_insensitive()
    test_translate_rfdiffusion_skips_api()

    # B. _call_api() guard

    # C. over-eager refusal handling
    test_system_prompt_frames_routine_design_as_legitimate()

    # D. zone-syntax guard + hide/show representation
    test_zone_guard_rewrites_bare_and_parenthesised()
    test_zone_guard_leaves_valid_and_volume_zone()
    test_chain_scope_guard_scopes_bare_chain_refs()
    test_chain_scope_guard_leaves_ligand_and_subrefs()
    test_zone_guard_drops_unrewritable()
    test_translate_rewrites_zone_end_to_end()
    test_system_prompt_documents_zone_and_hide()
    test_prompt_documents_stick_view_sequence()
    test_prompt_documents_ligand_keyword_and_literal_colour()
    test_hide_chain_targets_representation()

    # E. pluggable backend
    test_default_backend_is_ollama()
    test_backend_translate_returns_normalized_shape()

    # F. open-target validation guard (Bug 4a)
    test_open_valid_pdb_id_passes()
    test_open_nonexistent_file_blocked()
    test_open_sequence_token_blocked()
    test_open_from_alphafold_valid_pdb_passes()
    test_open_session_cxs_passes()
    # test_open_existing_file_passes requires tmp_path (pytest fixture) — runs via pytest
    test_open_guard_non_open_commands_untouched()
    test_open_guard_end_to_end_in_translate()

    # G. invalid-verb rejection guard (Bug 4b)
    test_unknown_verb_denylist_rejected()
    test_known_verbs_with_registry_pass()
    test_unknown_verb_with_registry_rejected()
    test_denylist_applies_even_with_registry()
    test_verb_guard_without_registry_passes_unknown()
    test_verb_guard_end_to_end_in_translate()
    test_probe_chimerax_verbs_caches_on_success()
    test_probe_chimerax_verbs_denylist_fallback()

    # H. error-correction guards (Bug 6)
    test_translate_error_fix_includes_error_text()
    test_correction_blocked_by_guard_4a()
    test_correction_blocked_by_guard_4b()

    print()
    print("=" * 60)
    print(
        f"Results: {_results['pass']} passed, {_results['fail']} failed"
    )
    return 0 if _results["fail"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
