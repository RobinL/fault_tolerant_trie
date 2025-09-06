import pytest

from matcher.trie_builder import build_trie_from_canonical, count_tail_L2R
from matcher.matcher_stage1 import peel_end_tokens_with_trie


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


def test_count_tail_wrapper(love_lane_root):
    assert count_tail_L2R(love_lane_root, ["LANGLEY"]) == 7
    # Unknown tokens after LANGLEY should not exist as anchors
    assert count_tail_L2R(
        love_lane_root, ["LANGLEY", "HERTFORDSHIRE", "ENGLAND"]
    ) == 0


def test_peel_redundant_tail_removed(love_lane_root):
    messy = [
        "KIMS",
        "NAILS",
        "4",
        "LOVE",
        "LANE",
        "KINGS",
        "LANGLEY",
        "HERTFORDSHIRE",
        "ENGLAND",
    ]
    peeled = peel_end_tokens_with_trie(messy, love_lane_root, steps=4, max_k=2)
    assert peeled == [
        "KIMS",
        "NAILS",
        "4",
        "LOVE",
        "LANE",
        "KINGS",
        "LANGLEY",
    ]


def test_peel_no_redundant_tail(love_lane_root):
    tokens = ["4", "LOVE", "LANE", "KINGS", "LANGLEY"]
    assert peel_end_tokens_with_trie(tokens, love_lane_root) == tokens


def test_peel_single_extra_token_after_langley(love_lane_root):
    tokens = ["4", "LOVE", "LANE", "KINGS", "LANGLEY", "EXTRA"]
    assert peel_end_tokens_with_trie(tokens, love_lane_root) == [
        "4",
        "LOVE",
        "LANE",
        "KINGS",
        "LANGLEY",
    ]


def test_peel_respects_max_k(love_lane_root):
    # With max_k=1 we can't jump over two unknown tail tokens at once,
    # so nothing should be peeled in one step. Our implementation stops.
    tokens = ["4", "LOVE", "LANE", "KINGS", "LANGLEY", "HERTFORDSHIRE", "ENGLAND"]
    assert (
        peel_end_tokens_with_trie(tokens, love_lane_root, steps=4, max_k=1)
        == tokens
    )
