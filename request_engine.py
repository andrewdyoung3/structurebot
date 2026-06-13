"""
request_engine.py
-----------------
The UI-agnostic orchestration core, extracted verbatim from main.py's _handle_request.

RequestEngine.handle_request(text, presenter) runs a request STRAIGHT-LINE: translate →
route → clarification loop → preview → confirm → execute → tools → viz → auto-fix →
state/log. It performs NO console I/O of its own — every output and every blocking
prompt goes through the injected Presenter. The interactive presenter methods
(ask_clarification / confirm / ask_edit / ask_yes_no) block from the engine's point of
view; the console runs the engine on the main thread, the GUI (later) on a worker
thread — only the presenter differs. The logic here is unchanged: this is a relocation,
not a rewrite (§0 intent/render unchanged — the LLM still infers intent).

The engine is bound to a *host* (StructureBot today, the GUI app later) for its
collaborators (bridge/translator/router/session) and a couple of side-effect hooks
(_log_exchange, _maybe_update_structure_state) that mutate session/write the log. The
verb-guard probe (probe_chimerax_verbs) is run by the host BEFORE handle_request.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from translator import RefusalError
from tool_router import ToolRouter


class RequestEngine:
    """Orchestrates one natural-language request. I/O via the Presenter; collaborators
    + side-effect hooks via the host."""

    def __init__(self, host):
        self._host = host

    # convenience accessors (read through the host so test mocks/late binding work)
    @property
    def bridge(self):
        return self._host.bridge

    @property
    def translator(self):
        return self._host.translator

    @property
    def router(self):
        return self._host.router

    @property
    def session(self):
        return self._host.session

    # ── input dispatch (pre-LLM glue, shared by front-ends) ─────────────────────
    def dispatch(self, user_input: str, presenter) -> None:
        """Front-end input glue: semicolon chaining + the bypass-LLM fast-paths
        (active-site / sequence-display / live-selection), then the full handle_request
        pipeline. Mirrors the console REPL's _dispatch_input so a GUI submit gets the
        SAME pre-LLM behaviour. (The console keeps its own _dispatch_input for now; this
        is the shared form the closeout can collapse onto.)"""
        if ";" in user_input:
            for part in [p.strip() for p in user_input.split(";") if p.strip()]:
                self.handle_request(part, presenter)
            return
        msg = self.router.handle_active_site_command(user_input)
        if msg:
            presenter.active_site_ok(msg)
            return
        seq_msg = self.router.handle_sequence_display_command(user_input)
        if seq_msg:
            presenter.markup(seq_msg)
            return
        sel_msg = self.router.handle_selection_command(user_input)
        if sel_msg:
            presenter.markup(sel_msg)
            return
        self.handle_request(user_input, presenter)

    # ── the request pipeline ──────────────────────────────────────────────────
    def handle_request(self, user_input: str, presenter, is_retry: bool = False) -> None:
        # 1. Pre-translate interception: covered intent categories bypass translation.
        # For covered ops, the intent registry resolves and renders deterministically;
        # the translator is not invoked at all (§0 Intent/Render separation principle).
        # Uncovered requests fall through to the normal translation path below.
        from intent_registry import (
            VIEWER_REGISTRY as _vreg, COLOR_REGISTRY as _creg,
        )
        if _vreg.detect_category_phrase(user_input):
            _repr_key = _vreg.resolve_alias(user_input)  # None → LLM tier in execute()
            result = {
                "commands":             [],
                "explanations":         [],
                "warnings":             [],
                "clarification_needed": None,
                "confidence":           "high",
                "tools_needed":         ["representation"],
                "tool_inputs":          {
                    "representation": {
                        "_user_input": user_input,
                        "intent_key":  _repr_key,
                    }
                },
            }
        elif _creg.detect_category_phrase(user_input, "color"):
            _color_key = _creg.resolve_alias(user_input)  # None → LLM/solid in execute()
            result = {
                "commands":             [],
                "explanations":         [],
                "warnings":             [],
                "clarification_needed": None,
                "confidence":           "high",
                "tools_needed":         ["color"],
                "tool_inputs":          {
                    "color": {
                        "_user_input": user_input,
                        "intent_key":  _color_key,
                    }
                },
            }
        else:
            # 1b. Translate (existing path — uncovered requests only)
            try:
                with presenter.status("Translating…"):
                    result = self.translator.translate(user_input, self.session)
            except RefusalError as exc:
                presenter.translation_declined(exc)
                return
            except ValueError as exc:
                # Legacy/other path: only treat as a decline if the message clearly
                # indicates an empty/declined translation; otherwise re-raise.
                if any(k in str(exc).lower() for k in ("refusal", "safety", "stop_reason")):
                    presenter.translation_declined(exc)
                    return
                raise
            except Exception as exc:
                # Backstop for the NON-refusal escape path: any other unexpected
                # translation failure (e.g. a Claude usage-cap BadRequestError that the
                # one-way fallback couldn't reroute when fallback is off) must surface
                # cleanly and return to the prompt — the REPL never crashes on it.
                presenter.translation_error(exc)
                return

        # 2. Route (augment with tool pipeline info; no execution yet)
        result = self.router.route(result, user_input=user_input)

        # 3. Clarification loop (max 2 rounds)
        for _ in range(2):
            q = result.get("clarification_needed")
            if not q:
                break
            answer = presenter.ask_clarification(q)
            if not answer:
                presenter.dim("No answer — cancelling.")
                return

            # ── Fast-path: mutation-scan tier choice (base vs deep) ────────────
            # The tier-choice surface is a local question, not a re-translation.
            # Interpret the answer and set run_rosetta directly — never re-route.
            if result.get("_tier_choice"):
                _ans = answer.lower()
                _want_shortlist = bool(re.search(r"\b(shortlist|short-list|top)\b", _ans))
                _want_deep = _want_shortlist or bool(re.search(
                    r"\b(deep|rosetta|rosie|full|yes|2)\b", _ans
                ))
                _ti = (result.get("tool_inputs") or {}).get("mutation_scan")
                if isinstance(_ti, dict):
                    _ti["run_rosetta"] = _want_deep
                    if _want_shortlist:
                        import config as _cfg
                        _ti["rosetta_shortlist_k"] = int(getattr(_cfg, "ROSETTA_SHORTLIST_K", 15))
                result["clarification_needed"] = None
                result.pop("_tier_choice", None)
                _label = ("deep-shortlist (top-K Rosetta ddG)" if _want_shortlist
                          else "deep (full Rosetta ddG)" if _want_deep
                          else "fast (CamSol+ESM)")
                presenter.dim(f"Running the {_label} tier.")
                break

            # ── Fast-path: bypass retranslation for known tool intents ─────────
            # If the original user_input already contains glycan (or other
            # recognised) keywords, re-routing through translate() would send a
            # bare short answer ("chain A") to the model with no prior context,
            # causing a stop_reason='refusal' crash.  Detect the intent here and
            # dispatch directly instead.
            if self.router._detect_glycan_intent(user_input):
                result = self.router.route(
                    {
                        "commands":             [],
                        "explanations":         [],
                        "warnings":             [],
                        "clarification_needed": None,
                        "confidence":           "high",
                        "tools_needed":         ["glycan"],
                        "tool_inputs":          {},
                    },
                    user_input=user_input,
                )
                break

            self.translator.add_clarification(answer)
            try:
                with presenter.status("Retranslating…"):
                    result = self.translator.translate(answer, self.session)
            except ValueError as exc:
                err_str = str(exc)
                if "refusal" in err_str.lower() or "stop_reason" in err_str.lower():
                    presenter.warn(
                        "Sorry, I couldn't process that answer. "
                        "Try rephrasing your original request directly, "
                        "e.g. 'suggest glycosylation sites on chain A'"
                    )
                else:
                    presenter.warn(f"Translation error: {err_str[:120]}")
                return
            except Exception as exc:
                presenter.warn(
                    "Sorry, I couldn't process that answer. "
                    "Try rephrasing your original request directly."
                )
                return
            result = self.router.route(result, user_input=user_input)

        if result.get("clarification_needed"):
            presenter.warn("Still ambiguous — please rephrase.")
            return

        commands:     List[str] = result.get("commands", [])
        explanations: List[str] = result.get("explanations", [])
        warnings:     List[str] = result.get("warnings", [])
        confidence:   str       = result.get("confidence", "medium")
        has_extra     = result.get("has_extra_tools", False)
        tools_needed: List[str] = result.get("tools_needed", ["chimerax"])

        # Require at least commands OR extra tools
        if not commands and not has_extra:
            presenter.warn("No commands generated.")
            return

        # 4. Show warnings
        for w in warnings:
            presenter.warn(f"⚠ {w}")

        # 5. Preview
        presenter.blank()
        if has_extra:
            presenter.show_tool_pipeline(result)
        if commands:
            presenter.show_commands(commands, explanations, confidence)
        elif has_extra:
            # No initial ChimeraX commands — tool output will generate viz
            presenter.dim(
                "  (visualization commands will be generated after the tool completes)"
            )

        # 6. Take pre-scan snapshot (before confirmation so state is clean)
        _pre_scan_snapshot = self.session.snapshot()

        # 7. Confirm / auto-proceed / edit
        should_execute = presenter.confirm(confidence)
        if should_execute is None:
            return  # cancelled
        if should_execute == "edit" and commands:
            commands = presenter.ask_edit(commands)
            if not commands:
                return

        presenter.blank()

        # Initialize execution state (used after try/except block)
        all_commands: list = list(commands)
        success      = True
        failed_cmd:  Optional[str] = None
        error_msg:   Optional[str] = None

        try:
            # 8. Execute initial ChimeraX commands (if any)
            if commands:
                success, failed_cmd, error_msg = self.execute_commands(commands, presenter)  # noqa: F841

            # 9. Execute extra tools (CamSol, ESM, etc.) if initial phase succeeded
            if success and has_extra:
                # For long-running pipelines, show elapsed time every 30s
                _long_tools = {"mutation_scan", "disulfide", "rosetta", "colabfold",
                               "validate_design"}
                _needs_timer = bool(set(tools_needed) & _long_tools)
                _ticker_label = (
                    "Running " + "/".join(
                        t for t in tools_needed if t in _long_tools
                    )
                )

                # ColabFold: surface a rough ETA beside the elapsed counter when
                # the sequence is known up front (approximate; see ColabFoldBridge).
                _eta_s = 0.0
                if "colabfold" in tools_needed:
                    try:
                        _cf_in = (result.get("tool_inputs") or {}).get("colabfold", {})
                        _seq   = _cf_in.get("sequence") or ""
                        _cop   = int(_cf_in.get("copies", 1) or 1)
                        if _seq:
                            from colabfold_bridge import ColabFoldBridge
                            _eta_s = ColabFoldBridge().estimate_runtime_s(
                                len(_seq) * _cop, 5, 3
                            )
                    except Exception:
                        _eta_s = 0.0

                with presenter.running_tools(_ticker_label, _eta_s, _needs_timer):
                    result = self.router.execute(result, status_callback=presenter.tool_status)

                # Show assembly interface summary (before other summaries)
                presenter.show_interface_summary(result)

                # Show tool summaries
                summaries = result.get("tool_summaries", {})
                for tool, summary in summaries.items():
                    icon = ToolRouter._TOOL_ICONS.get(tool, "⚙️")
                    if result.get("pipeline_success"):
                        presenter.tool_summary(icon, True, summary=summary)
                    else:
                        err = result.get("pipeline_error", "unknown error")
                        presenter.tool_summary(icon, False, tool=tool, error=err)

                if not result.get("pipeline_success"):
                    err = result.get("pipeline_error", "")
                    presenter.blank()
                    presenter.error(f"Tool pipeline failed: {err[:120]}")
                    # Keep going — viz commands might still be partially available

                # 10. Execute visualization commands generated by the tools
                viz_cmds = result.get("all_viz_commands", [])
                viz_exps = result.get("all_viz_explanations", [])
                if viz_cmds:
                    presenter.blank()
                    presenter.dim("  Applying visualization…")
                    presenter.show_commands(viz_cmds, viz_exps, "high")
                    viz_ok, viz_failed, viz_err = self.execute_commands(viz_cmds, presenter, origin="tool_viz")
                    if not viz_ok:
                        presenter.warn(f"  Visualization command failed: {viz_err or ''}")
                    all_commands.extend(viz_cmds)

                # 11. Show actionable summary panel for tools that produced one
                for step in result.get("tool_step_results", []):
                    if (step.get("success") and step.get("summary")
                            and "\n" in step.get("summary", "")):
                        presenter.analysis_panel(step["summary"])

        except KeyboardInterrupt:
            presenter.blank()
            presenter.warn("Warning: Scan cancelled by user.")
            presenter.dim("Restoring session state to pre-scan snapshot...")
            self.session.restore(_pre_scan_snapshot)
            presenter.dim("Session state restored.")
            return

        # 10. Auto-fix on first failure (once only, ChimeraX commands only)
        if not success and not is_retry and failed_cmd and error_msg:
            presenter.blank()
            presenter.warn("Asking for a corrected command…")
            # Bug 6a: the actual error text is fed into translate_error_fix so the
            # model cannot re-propose the same command blind (already handled by
            # translate_error_fix, which builds the prompt from failed_command +
            # error_message verbatim — no silent re-prompt).
            fix = self.translator.translate_error_fix(failed_cmd, error_msg, self.session)
            fix_cmds = fix.get("commands", [])
            fix_exps = fix.get("explanations", [])

            # Bug 6b: no-progress detection — halt cleanly instead of looping.
            # No progress = the correction is empty (guards blocked it or model
            # refused) OR the model re-proposed the identical failing command.
            _same_cmd = bool(
                fix_cmds
                and fix_cmds[0].strip().lower() == failed_cmd.strip().lower()
            )
            # Bug 6c (3b fix): halt if the correction reuses the SAME non-existent
            # verb that was just rejected.  The verb guard in translate_error_fix
            # blocks the command and empties fix_cmds, but when the registry is
            # unavailable and the verb isn't in the denylist it may slip through.
            # Extract the leading verb of the failed command; if the correction's
            # first command starts with the same verb, it's the same hallucination.
            _failed_verb = failed_cmd.strip().split()[0].lower() if failed_cmd.strip() else ""
            _fix_verb    = fix_cmds[0].strip().split()[0].lower() if fix_cmds else ""
            _same_verb   = bool(
                _failed_verb and _fix_verb and _failed_verb == _fix_verb
                and _failed_verb not in ("open", "close", "color", "colour",
                                         "select", "hide", "show", "cartoon",
                                         "surface", "style", "align", "view",
                                         "transparency", "sym", "matchmaker")
            )
            if not fix_cmds or _same_cmd or _same_verb:
                presenter.warn(
                    f"Couldn't auto-correct — error: {error_msg[:200]}"
                )
                _reason = (
                    "Correction re-proposed the same non-existent verb "
                    f"'{_failed_verb}' — halting to prevent a loop."
                    if _same_verb else
                    "Correction re-proposed the same command or was blocked "
                    "by a validation guard.  Try rephrasing your request."
                )
                presenter.dim(_reason)
            else:
                presenter.blank()
                presenter.warn("Suggested correction:")
                presenter.show_commands(fix_cmds, fix_exps, fix.get("confidence", "medium"))
                if presenter.ask_yes_no("Apply fix?"):
                    fix_success, _, _ = self.execute_commands(fix_cmds, presenter)
                    if fix_success:
                        all_commands.extend(fix_cmds)

        # 11. Update state
        self.session.add_to_history(user_input, all_commands, success=success, error=error_msg)
        self._host._maybe_update_structure_state(all_commands)
        self.translator.trim_history()

        # Build enhanced tool-step log entries from tool pipeline results
        _tool_steps: List[dict] = []
        for step in result.get("tool_step_results", []):
            if step.get("skipped"):
                continue
            tool  = step.get("tool", "")
            data  = step.get("data", {})
            entry: dict = {
                "tool":       tool,
                "elapsed_ms": step.get("elapsed_ms", 0),
                "success":    step.get("success", False),
            }
            # Tool-specific enrichment
            if tool == "mutation_scan":
                cands = data.get("candidates", [])
                entry["n_candidates"]  = len(cands)
                entry["top_candidate"] = cands[0].get("mutation_key", "") if cands else ""
                entry["top_ddg"]       = cands[0].get("ddg", None)        if cands else None
                entry["backend"]       = cands[0].get("backend", "")      if cands else ""
            elif tool == "disulfide":
                entry["n_candidates"] = data.get("count", 0)
            elif tool in ("camsol", "esm", "proteinmpnn", "rfdiffusion"):
                pass  # no extra enrichment needed
            _tool_steps.append(entry)

        self._host._log_exchange(user_input, all_commands, success, error_msg,
                                 tool_steps=_tool_steps if _tool_steps else None)

    # ── command execution (presenter-rendered) ──────────────────────────────────
    def execute_commands(
        self,
        commands: List[str],
        presenter,
        origin: str = "translation",
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute via bridge, render results through the presenter. Returns
        (all_ok, failed_cmd, error_msg).

        origin — "translation" (default): free-translated commands from the LLM; the
                  emission guard fires here to block representation-shaped commands.
                 "tool_viz": visualization commands generated by tool steps (ColabFold,
                  double-mutant, etc.); these are trusted and bypass the guard.
        """
        # Emission guard (§0): representation-shaped commands must originate from
        # the render layer (_run_representation), never from free-translation.
        # Tool-viz commands (origin="tool_viz") are trusted and exempt.
        if origin == "translation":
            from intent_registry import is_representation_shaped
            for cmd in (c.strip() for c in commands if c.strip()):
                if is_representation_shaped(cmd):
                    err = (
                        f"Representation command {cmd!r} blocked — "
                        "use a representation phrase (cartoon/sticks/surface/…) "
                        "so it routes via the render layer."
                    )
                    presenter.blocked(cmd, err)
                    return False, cmd, err

        with presenter.status("Executing…"):
            results = self.bridge.run_commands(commands)

        first_err_cmd: Optional[str] = None
        first_err_msg: Optional[str] = None

        for r in results:
            cmd = r["command"]
            res = r["result"]
            err = res.get("error")

            if err:
                presenter.command_result(cmd, ok=False, error=err)
                if first_err_cmd is None:
                    first_err_cmd = cmd
                    first_err_msg = str(err)
            else:
                presenter.command_result(cmd, ok=True, value=res.get("value"),
                                         warning=res.get("warning"))

        all_ok = first_err_cmd is None
        if all_ok:
            presenter.completed(len(results))
        return all_ok, first_err_cmd, first_err_msg
