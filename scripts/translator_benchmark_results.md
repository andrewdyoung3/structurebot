# Translator backend benchmark — 2026-06-02

_Ollama model: **qwen3:8b**_  ·  corpus: 17 cases, 11 categories. FULL = post-guard (`translate()`); RAW = pre-guard (`backend.translate()`).

## Overall

| Metric | claude | ollama |
|--------|---|---|
| Pass-rate (FULL, post-guard) | 100% (17/17) | 47% (8/17) |
| Pass-rate (RAW, pre-guard) | 100% (17/17) | 41% (7/17) |
| Schema-valid | 100% | 100% |
| Tool-routing accuracy | 100% (14/14) | 43% (6/14) |
| Latency median (s) | 5.7 | 8.0 |
| Latency max (s) | 9.3 | 11.1 |
| Errors | 0 | 0 |

## Per-category (FULL · RAW pass)

| Category | claude | ollama |
|----------|---|---|
| camsol | 2/2 · 2/2 | 0/2 · 0/2 |
| disulfide | 1/1 · 1/1 | 1/1 · 1/1 |
| esm | 1/1 · 1/1 | 1/1 · 1/1 |
| hide_show | 2/2 · 2/2 | 1/2 · 1/2 |
| mpnn | 2/2 · 2/2 | 0/2 · 0/2 |
| multi_tool | 2/2 · 2/2 | 1/2 · 1/2 |
| proline | 1/1 · 1/1 | 1/1 · 1/1 |
| rfdiffusion | 1/1 · 1/1 | 1/1 · 1/1 |
| selection_scope | 2/2 · 2/2 | 0/2 · 0/2 |
| viz | 1/1 · 1/1 | 1/1 · 1/1 |
| zone | 2/2 · 2/2 | 1/2 · 0/2 |
