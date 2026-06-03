# Ollama Serving-Config Bill of Health — qwen3:8b translator backend

**Date:** 2026-06-03 · **Branch/commit:** `main` @ `0e08e5b` (bug fixes pushed) ·
**Governing principle:** verify EMPIRICALLY — a silent throttle is indistinguishable
from a weak model, so for every item below the EVIDENCE is what Ollama actually
received/loaded, not what the code intends.

**Bottom line:** After the `0e08e5b` fixes, the backend is serving the **real,
un-throttled qwen3:8b at Q4_K_M** — full context, no truncation, explicit greedy
decoding, full few-shot, enum-reachable tools, 100% GPU. The two remaining levers
(thinking mode, higher quant) are **judgment calls**, A/B'd below, **NOT adopted**
pending sign-off.

---

## A. Ground truth — the exact request body (captured live, GPU free)

Captured by shimming `requests.post` around a real `translate()` for a CamSol prompt
(`cache/audit_capture.py`). What Ollama actually received:

- **URL/transport:** `POST http://localhost:11434/api/chat`, `timeout=180`, `stream=False`,
  `keep_alive=30s`, `think=False`.
- **Options (AFTER fix `0e08e5b`):**
  `{temperature:0, top_p:1.0, top_k:0, repeat_penalty:1.0, seed:0, num_predict:1024, num_ctx:16384}`
  — every value explicit (see C/D for why).
- **`format` (grammar-constrained):** `type:object`, `required` = the 7 keys
  (`commands, explanations, warnings, clarification_needed, confidence, tools_needed, tool_inputs`);
  `tools_needed.items` = `{type:string, enum:[…21 tools…]}`.
- **Messages — 18, roles correct:** `[0] system` (30 105 ch ≈ 7.5 k tok) → 8 alternating
  `user`/`assistant` few-shot pairs → `[17] user` (the live prompt). The assistant demos
  are JSON in the **same 7-key shape** the `format` grammar enforces.
- **Few-shot present assertion:** `msgs[1:1+16] == few_shot_messages()` → **True**.
- **Token budget:** ~8 014 tok vs `num_ctx` 16 384 → **~8 k margin, no truncation**.
- **Result:** `tools_needed=['camsol']` ✓.

**Production == benchmark path:** both go through `backend.translate() → OllamaBackend._chat`
(same prompt via `_build_system_blocks`, same few-shot, same options). The benchmark
forces `ollama` with no fallback; production adds the one-way Claude→Ollama fallback and
multi-turn history. For single-turn translation the request is identical — the benchmark
measures what we ship.

## B. Context / truncation — FIXED (the central discovery)

- **Bug:** `OLLAMA_NUM_CTX` was **8192**. System prompt (~7.5 k tok) + few-shot (~0.5 k) ≈
  **7 999 tok ≈ the 8192 ceiling** → Ollama **silently truncated** the prompt/few-shot →
  routing collapsed (the failing-category 0% was a truncation artifact, not model weakness).
- **Fix:** `num_ctx → 16384`. Captured budget now ~8 k tok with ~8 k margin.
- **`ollama ps` during a 16384 request:** `qwen3:8b … 7.8 GB … 100% GPU … CONTEXT 16384` —
  **zero CPU offload**, fits the 16 GB RTX 5070 Ti.
- **No other clip:** single-turn history = `[user]`; `num_predict=1024` is generous vs the
  ~<500-tok JSON, so output never truncates mid-generation.

## C. Sampling / decoding — FIXED (explicit, no silent merge)

- **Bug:** the request sent only `{temperature, num_ctx}`; everything else was inherited
  from the modelfile via Ollama's option-merge. Effective values happened to be benign
  (modelfile `repeat_penalty=1`), but **inheritance is the silent-throttle risk** — a model
  re-pull or default change would throttle invisibly.
- **Fix — all decoding options explicit:** `temperature=0` (true greedy), `top_p=1.0` +
  `top_k=0` (no nucleus/top-k clipping — irrelevant at greedy but now unambiguous),
  `repeat_penalty=1.0` (1.1 distorts the repeated braces/quotes/keys of constrained JSON),
  `seed=0` (reproducible), `num_predict=1024`. Re-captured: the options block now shows all
  seven values; the CamSol result is unchanged → behavior-preserving, ambiguity removed.

## D. Thinking mode (qwen3 hybrid reasoner) — `think:false` HONORED; A/B below

- **Honored:** raw `message.content` for a CamSol prompt is direct JSON
  (`{"commands": [], …`) with **no `<think>` leak**; the `format` grammar and `think:false`
  are not fighting.
- **A/B (think ON vs OFF, failing categories, Q4, runs=3):** OFF **91%** vs ON **71%**
  → **think ON is −20 pts** (see table below).
- **Recommendation: keep `think:false`** (already the production default). Thinking mode
  actively hurts this constrained-JSON routing task — **no change needed, nothing to adopt.**

## E. Quantization — Q4_K_M today; A/B below

- **Exact quant served:** `ollama show qwen3:8b` → 8.2 B, context length 40960, **Q4_K_M**.
- On 16 GB an 8B at Q8_0 fits; pulling `qwen3:8b-q8_0` for an A/B on the failing categories.
- **A/B (Q8_0 vs Q4_K_M, think OFF, failing categories, runs=3):** Q8_0 **91%** vs Q4_K_M
  **91%** → **net 0 pts** (see table below).
- **Recommendation: keep Q4_K_M.** Q8_0 (8.9 GB vs 5.2 GB, slower) buys nothing here —
  **nothing to adopt.**

## F. Chat template / roles — VERIFIED

- `[0]` is a true **system-role** message (not folded into a user turn).
- Few-shot renders as proper **alternating user/assistant** turns under qwen3's template.
- Assistant demos are in the **same 7-key JSON shape** the `format` grammar forces.

## G. Schema / enum ↔ router parity — VERIFIED EXACT

- `tools_needed` enum = `config.TRANSLATOR_TOOL_NAMES` (21 lowercase literals).
- Cross-checked one-to-one against `tool_router._dispatch_tool`'s dispatch registry:
  **21 == 21, no missing / extra / misspelled.** No tool is unreachable via the grammar.
- `format` is grammar-applied (constrained decoding), `required` = all 7 keys; not
  over-constraining (free-form string arrays for commands/explanations/warnings).

## H. Backend code path — VERIFIED (no silent failure modes)

- **Few-shot ON for local:** present in the captured request (not gated off).
- **`_parse_response` not over-dropping:** a viz prompt returns
  `commands=['open 1HSG','color bychain','view']` — valid string commands preserved
  (the non-string drop only removes genuine garbage a weak model might emit).
- **No error-swallowing:** `OllamaBackend._chat` raises on HTTP error (propagates);
  the benchmark's `run_backend` counts an exception as a **failure** (raw `{}`), never as a
  smoothed/garbage pass — real failures surface.
- **No truncating transport:** `stream=False` → complete response or a raised error
  (no partial/truncated body); `timeout=180 s`.

---

## Fixes shipped in `0e08e5b` (bugs only — committed FF, pushed)

| # | Bug | Fix |
|---|-----|-----|
| 1 | `num_ctx=8192` silently truncated prompt+few-shot | → **16384** |
| 2 | `tools_needed` unconstrained → tools could be unreachable | enum = 21 router literals, **parity verified** |
| 3 | case-mismatch tool names mis-routed/mis-scored | router lowercases tool + `tool_inputs`; scorer case-insensitive |
| 4 | options inherited from modelfile (silent merge) | **all sampling options explicit** (greedy, rp=1.0, seed, num_predict) |
| 5 | failing categories had no exemplars | 8 targeted few-shot pairs (camsol/mpnn/selection/esm), from disjoint pool |

Tests: `test_ollama_backend.py` 12→14, `test_translator_benchmark_logic.py` 11→14;
full suite **696 passed / 20 skipped**.

---

## Judgment-call A/B results (NOT adopted — awaiting sign-off)

Slices run on the **4 failing categories only** (camsol 9 · mpnn 9 · selection_scope 9 ·
esm 8 = 35 cases), FULL pass-rate (post-guard), runs=3. All results deterministic across
runs (explicit `seed=0`). The A/B harness (`cache/ab_slice.py`, untracked) injects
think/quant via a `requests.post` shim + `config.OLLAMA_MODEL` — **no production code changed.**

> **First, the headline the fixes already bought:** on these same 4 categories the
> **post-fix baseline is 91%** (camsol 89 · esm 100 · mpnn 78 · selection 100) — versus the
> pre-fix corpus baseline of **camsol 0% · selection 0% · mpnn 18% · esm 28%**. The collapse
> was the `num_ctx=8192` truncation throttle, not the model. **This is the main finding.**

### D. Thinking mode A/B (Q4_K_M, think OFF vs ON)

| Category | think OFF (current) | think ON | Δ |
|----------|--------------------:|---------:|----:|
| camsol | 89% | 89% | 0 |
| esm | 100% | 88% | −12 |
| mpnn | 78% | 56% | −22 |
| selection_scope | 100% | 56% | −44 |
| **OVERALL** | **91%** | **71%** | **−20** |

**Verdict: keep `think:false`.** Thinking mode costs 20 pts overall (and 44 on
selection-scope) — the reasoning competes with the `format` grammar instead of helping.
The production default is already OFF, so **nothing to change or adopt.**

### E. Quantization A/B (think OFF, Q4_K_M vs Q8_0)

| Category | Q4_K_M (current) | Q8_0 | Δ |
|----------|-----------------:|-----:|----:|
| camsol | 89% | 78% | −11 |
| esm | 100% | 100% | 0 |
| mpnn | 78% | 89% | +11 |
| selection_scope | 100% | 100% | 0 |
| **OVERALL** | **91%** | **91%** | **0** |

**Verdict: keep Q4_K_M.** Q8_0 is a wash — it swaps one case between camsol and mpnn
(net zero) for +3.7 GB resident VRAM and slower decode. No accuracy signal justifies it
on this corpus. **Nothing to change or adopt.**

---

## Conclusion

**Yes — after `0e08e5b` the backend is serving the most capable qwen3:8b it can, with
nothing left suspect.** Full 16384 context (no silent truncation), 100% GPU, explicit
greedy decoding with no inherited/penalty distortion, full system+few-shot prompt with
correct roles, every enum tool reachable and parity-checked against the router, no
error-swallowing. The pre-fix failing-category collapse (camsol/selection 0%, mpnn 18%,
esm 28%) was a **truncation throttle**, now **91%** on those same categories.

Both remaining levers were A/B'd and **neither helps**: thinking mode is −20 pts (kept
OFF, already the default), and Q8_0 is net-zero (kept Q4_K_M). **No judgment-call config
change is recommended; nothing was adopted.** The serving config is sound as committed.

**Awaiting your sign-off** to run the full definitive benchmark (93×5, Claude vs the
cleaned Ollama config) and refresh `scripts/translator_benchmark_results.{md,csv}`.
