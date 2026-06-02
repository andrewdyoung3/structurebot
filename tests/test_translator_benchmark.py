"""
tests/test_translator_benchmark.py
----------------------------------
OPT-IN translator benchmark RUN (Claude reference vs local Ollama, per rule
category). A MEASUREMENT tool, NOT a CI gate — the local model is expected to
show gaps.

SKIP BY DEFAULT — runs live ONLY when STRUCTUREBOT_RUN_TRANSLATOR_BENCHMARK=1
(needs a live Anthropic key + a running Ollama with OLLAMA_MODEL pulled). CI / no
opt-in → collects + skips, 0 model calls. Mirrors the rosetta/colabfold gating.

  STRUCTUREBOT_RUN_TRANSLATOR_BENCHMARK=1 pytest tests/test_translator_benchmark.py -v -s
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

_RUN = os.environ.get("STRUCTUREBOT_RUN_TRANSLATOR_BENCHMARK") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN,
    reason="opt-in benchmark — set STRUCTUREBOT_RUN_TRANSLATOR_BENCHMARK=1 "
           "(needs a live Anthropic key + a running Ollama with OLLAMA_MODEL).",
)


def test_run_translator_benchmark():
    """Run the full Claude-vs-Ollama benchmark, print + persist the report.

    This is a measurement, so it does NOT assert a local-model bar (the local
    model is expected to lag). It DOES sanity-assert that the harness ran every
    case and the reference (Claude) is healthy — a guardrail against a broken
    harness masquerading as a model result.
    """
    import config
    import translator_corpus as corpus
    import translator_benchmark as bm

    # Runs per case — default 1 for the test smoke; set STRUCTUREBOT_BENCHMARK_RUNS
    # (or use `python translator_benchmark.py --runs 5`) for the official mean±range.
    runs = int(os.environ.get("STRUCTUREBOT_BENCHMARK_RUNS", "1"))
    comp = bm.run_comparison(("claude", "ollama"), runs=runs)
    bm.print_rich(comp, model_label=config.OLLAMA_MODEL)
    paths = bm.write_artifacts(comp, model_label=config.OLLAMA_MODEL)
    print(f"\nArtifacts written: {paths['md']} · {paths['csv']}")

    n = len(corpus.EVAL_CORPUS)
    for backend in ("claude", "ollama"):
        for rows in comp[backend]["runs"]:
            assert len(rows) == n, f"{backend} did not run every eval case"

    # Harness-health guardrail (NOT a model gate): the reference backend should be
    # schema-valid throughout and clear a generous bar — if Claude looks bad, the
    # harness/corpus is broken, not the model.
    claude = comp["claude"]["summary"]
    assert claude["schema"]["mean"] == 1.0, "reference (Claude) schema-validity < 100% → harness/corpus bug"
    assert claude["full"]["mean"] >= 0.8, f"reference (Claude) pass-rate {claude['full']['mean']:.0%} too low → corpus mis-calibrated"
