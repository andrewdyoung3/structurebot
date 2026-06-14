"""
tests/test_seq_library.py
-------------------------
The reusable pure library: unique-sequence grouping (homo-oligomer collapse), the
grouping-key formula PINNED to the ChimeraX-native runscript (they can't share a
call across the process boundary, so they must not drift), and the ruler re-export.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import seq_library as sl


class _FakeChain:
    def __init__(self, model, chain, seq, resnums):
        self.model, self.chain, self.wt_seq, self._rns = model, chain, seq, resnums
    def resnums(self):
        return self._rns


class TestGroupingKey:
    def test_key_is_md5_12_plus_sorted_resnums(self):
        assert sl.sequence_group_key("ABCDE", [3, 1, 2]) == (
            hashlib.md5(b"ABCDE").hexdigest()[:12], (1, 2, 3))

    def test_native_runscript_uses_the_same_formula(self):
        # PIN: the ChimeraX-native consolidation grouping can't import this helper
        # (runscript boundary) — assert its inline key formula is still md5(...)[:12]
        # + sorted resnums, so a runscript change that diverges trips this test.
        src = (Path(__file__).parent.parent / "sequence_viewer.py").read_text()
        assert "hexdigest()[:12]" in src
        assert re.search(r"sorted\(r\.number", src)
        assert "key = (seq_hash, rns)" in src


class TestGrouping:
    def test_homo_oligomer_collapses_to_one_group(self):
        chains = [_FakeChain("1", "A", "MKV", [1, 2, 3]),
                  _FakeChain("1", "B", "MKV", [1, 2, 3])]
        groups = sl.group_chains_by_sequence(chains)
        assert len(groups) == 1
        g = groups[0]
        assert g.members == [("1", "A"), ("1", "B")] and g.rep == ("1", "A")

    def test_hetero_gives_distinct_groups(self):
        chains = [_FakeChain("1", "A", "MKV", [1, 2, 3]),
                  _FakeChain("1", "B", "WYF", [1, 2, 3])]
        assert len(sl.group_chains_by_sequence(chains)) == 2

    def test_same_seq_different_numbering_not_merged(self):
        chains = [_FakeChain("1", "A", "MKV", [1, 2, 3]),
                  _FakeChain("1", "B", "MKV", [10, 11, 12])]
        assert len(sl.group_chains_by_sequence(chains)) == 2   # resnums part of the key


class TestRulerReExport:
    def test_ruler_is_the_same_pure_fn(self):
        # the ruler is a true shared call (no runscript boundary) — re-export, one copy
        import sequence_viewer
        assert sl.build_numbering_header_content is sequence_viewer.build_numbering_header_content
        ruler = sl.build_numbering_header_content([1, 2, 3, 4, 5], interval=5)
        assert "5" in ruler and len(ruler) == 5
