# Translator backend benchmark — 2026-06-02

_Ollama model: **qwen3:8b**_  ·  eval corpus: 93 cases × 11 categories  ·  **N = 5 runs** (mean [min–max]). FULL = post-guard (`translate()`); RAW = pre-guard (`backend.translate()`). Bar = mean over N≥5.

## Overall (mean [min–max] over N runs)

| Metric | claude | ollama |
|--------|---|---|
| Pass-rate FULL (post-guard) | 98% [92–99] | 57% [56–58] |
| Pass-rate RAW (pre-guard) | 97% [91–99] | 50% [48–52] |
| Schema-valid | 98% [92–99] | 100% [99–100] |
| Tool-routing accuracy | 97% [90–99] | 48% [46–49] |
| Latency median (s) | 5.4 | 8.3 |
| Latency max (s) | 25.1 | 182.1 |
| Errors (total) | 11 | 2 |

## Per-category — FULL mean [min–max] · RAW mean [min–max]

| Category (n) | claude | ollama |
|----------|---|---|
| camsol (9) | 91% [56–100] · 91% [56–100] | 0% [0–0] · 0% [0–0] |
| disulfide (8) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| esm (8) | 100% [100–100] · 100% [100–100] | 28% [25–38] · 28% [25–38] |
| hide_show (9) | 100% [100–100] · 100% [100–100] | 78% [78–78] · 78% [78–78] |
| mpnn (9) | 91% [89–100] · 91% [89–100] | 18% [11–22] · 18% [11–22] |
| multi_tool (8) | 100% [100–100] · 100% [100–100] | 68% [62–75] · 68% [62–75] |
| proline (8) | 100% [100–100] · 100% [100–100] | 80% [50–100] · 80% [50–100] |
| rfdiffusion (8) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| selection_scope (9) | 93% [67–100] · 93% [67–100] | 0% [0–0] · 0% [0–0] |
| viz (8) | 100% [100–100] · 100% [100–100] | 100% [100–100] · 100% [100–100] |
| zone (9) | 100% [100–100] · 91% [78–100] | 69% [67–78] · 0% [0–0] |
