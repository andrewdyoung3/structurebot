# Translator backend benchmark — 2026-06-03

_Ollama model: **qwen3:8b**_  ·  eval corpus: 93 cases × 11 categories  ·  **N = 5 runs** (mean [min–max]). FULL = post-guard (`translate()`); RAW = pre-guard (`backend.translate()`). Bar = mean over N≥5.

> **Scope of this score: ROUTING + COMMAND SYNTAX only.** A case passes on correct tool routing and well-formed ChimeraX commands; **`tool_inputs` (arguments / scope) are NOT checked**, so this is *not* an end-to-end correctness number. The argument/scope-checking upgrade is a separate item.
>
> _Measured on the cleaned serving config (`0e08e5b`: num_ctx=16384, enum-constrained tools, case-insensitive routing, explicit sampling seed=0, targeted few-shot). The prior 57% (commit `71e2779`) was measured on the throttled pre-fix config — stale, not comparable, discarded._

## Overall (mean [min–max] over N runs)

| Metric | claude | ollama |
|--------|---|---|
| Pass-rate FULL (post-guard) | 99% [99–99] | 92% [92–92] |
| Pass-rate RAW (pre-guard) | 99% [98–99] | 88% [87–88] |
| Schema-valid | 99% [99–99] | 100% [100–100] |
| Tool-routing accuracy | 99% [99–99] | 93% [93–93] |
| Latency median (s) | 5.3 | 3.5 |
| Latency max (s) | 13.5 | 10.6 |
| Errors (total) | 5 | 0 |

## Per-category — FULL mean [min–max] · RAW mean [min–max]

| Category (n) | claude | ollama |
|----------|---|---|
| camsol (9) | 100% [100–100] · 100% [100–100] | 89% [89–89] · 89% [89–89] |
| disulfide (8) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| esm (8) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| hide_show (9) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| mpnn (9) | 89% [89–89] · 89% [89–89] | 78% [78–78] · 78% [78–78] |
| multi_tool (8) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| proline (8) | 100% [100–100] · 100% [100–100] | 75% [75–75] · 75% [75–75] |
| rfdiffusion (8) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| selection_scope (9) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| viz (8) | 100% [100–100] · 100% [100–100] | 88% [88–88] · 88% [88–88] |
| zone (9) | 100% [100–100] · 98% [89–100] | 89% [89–89] · 42% [33–44] |
