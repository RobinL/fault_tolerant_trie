import pytest

from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import match_stage1_with_skips


@pytest.fixture()
def love_lane_root():
    canonical_love_lane = [
        (1, ["5", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (2, ["9", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (3, ["8", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (4, ["7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (5, ["ANNEX", "7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (6, ["6", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (7, ["4", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    ]
    return build_trie_from_canonical(canonical_love_lane, reverse=True)


@pytest.mark.parametrize(
    "addr, expected_uprn",
    [
        # Inner noise (non-tail) requires a skip → accept
        ("4 LOVE EXTRA LANE KINGS LANGLEY", 7),
        # Redundant token inside should be 0-cost skipped by counts
        ("4 LOVE LANE KINGS HERTFORDSHIRE LANGLEY", 7),
        # Business prefix + redundant tail: both handled (blocked accept and/or skips)
        ("KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE", 7),
    ],
)
def test_with_skips_accepts(love_lane_root, addr, expected_uprn):
    tokens = addr.split()
    assert match_stage1_with_skips(tokens, love_lane_root) == expected_uprn


def test_with_skips_respects_numeric_guard(love_lane_root):
    # No numeric anchor → reject under default require_numeric=True
    tokens = "LOVE LANE KINGS LANGLEY".split()
    assert match_stage1_with_skips(tokens, love_lane_root) is None


def test_with_skips_respects_min_exact_hits(love_lane_root):
    # Force a high min_exact_hits so it fails despite otherwise valid path
    tokens = "KIMS NAILS 4 LOVE LANE KINGS LANGLEY".split()
    assert (
        match_stage1_with_skips(
            tokens, love_lane_root, min_exact_hits=6, require_numeric=True
        )
        is None
    )

