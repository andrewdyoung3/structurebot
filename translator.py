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
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

import config
from session_state import SessionState

# ── Model ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL: str = config.ANTHROPIC_MODEL

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
  "confidence":          "high"
}}

confidence values:
  "high"   — unambiguous request, well-understood commands, likely to succeed
  "medium" — minor assumptions made; commands should work but review is advised
  "low"    — request is complex or unclear; user should carefully review

If the request cannot be safely translated without more information:
{{
  "commands":            [],
  "explanations":        [],
  "warnings":            [],
  "clarification_needed": "A single concise question for the user",
  "confidence":          "low"
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

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AVAILABLE TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chimerax   : visualization, selection, measurement, image export  [ACTIVE]
rosetta    : stability prediction, ddG calculation               [NOT YET CONFIGURED]
esm        : evolutionary scoring, mutation tolerance            [NOT YET CONFIGURED]
proteinmpnn: sequence redesign                                   [NOT YET CONFIGURED]
camsol     : solubility scoring                                  [NOT YET CONFIGURED]

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

    def _call_api(self, system_blocks: list) -> str:
        response = self.client.messages.create(
            model      = self.model,
            max_tokens = 2048,
            system     = system_blocks,
            messages   = self._history,
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

        # Coerce confidence to one of three values
        if result["confidence"] not in ("high", "medium", "low"):
            result["confidence"] = "medium"

        # Pad short explanations list
        while len(result["explanations"]) < len(result["commands"]):
            result["explanations"].append("")

        return result

    def __repr__(self) -> str:
        return (
            f"<CommandTranslator model={self.model!r} "
            f"history_turns={len(self._history) // 2}>"
        )
