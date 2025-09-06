import pytest

from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import match_stage1_with_skips


@pytest.fixture()
def haydn_root():
    canonical = [
        (1001, ["12", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
        (1002, ["10", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
    ]
    return build_trie_from_canonical(canonical, reverse=True)


def test_fuzzy_transposition_accepts_with_numeric(haydn_root):
    # Messy has adjacent transposition HADYN vs canonical HAYDN
    addr = "12 HADYN PARK ROAD"
    assert match_stage1_with_skips(addr.split(), haydn_root) == 1001


def test_fuzzy_substitution_accepts_with_numeric(haydn_root):
    # One substitution: HAYDN -> HAYDN (exact) would pass; test close variant HAYEN
    addr = "10 HAYEN PARK ROAD"
    assert match_stage1_with_skips(addr.split(), haydn_root) == 1002


def test_no_numeric_guard_blocks(haydn_root):
    # Without a numeric token, guard should block even if fuzzy could bridge
    addr = "HADYN PARK ROAD"
    assert match_stage1_with_skips(addr.split(), haydn_root) is None


def test_fuzzy_numeric_anchor_not_allowed():
    # Numeric must be exact: DL=1 on the house number should not satisfy the guard
    canonical = [
        (2001, ["12", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
    ]
    root = build_trie_from_canonical(canonical, reverse=True)
    # '12' vs '13' is DL=1 but numeric_must_be_exact=True → reject
    assert match_stage1_with_skips("13 HAYDN PARK ROAD".split(), root) is None
    # sanity: exact numeric passes
    assert match_stage1_with_skips("12 HAYDN PARK ROAD".split(), root) == 2001
