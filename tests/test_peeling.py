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


def _peel_text(s: str, root) -> str:
    tokens = [t for t in s.split(" ") if t]
    out = peel_end_tokens_with_trie(tokens, root, steps=4, max_k=2)
    return " ".join(out)

@pytest.mark.parametrize(
    "inp, expected",
    [
        (
            "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND",
            "KIMS NAILS 4 LOVE LANE KINGS LANGLEY",
        ),
        ("4 LOVE LANE KINGS LANGLEY", "4 LOVE LANE KINGS LANGLEY"),
        ("4 LOVE LANE KINGS LANGLEY EXTRA", "4 LOVE LANE KINGS LANGLEY"),
    ],
)
def test_peel_pairs(love_lane_root, inp: str, expected: str):
    assert _peel_text(inp, love_lane_root) == expected


@pytest.mark.parametrize(
    "inp, max_k, expected",
    [
        # Can't jump two tokens when max_k=1 → unchanged
        (
            "4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND",
            1,
            "4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND",
        ),
        # Single tail token can be peeled with max_k=1
        ("4 LOVE LANE KINGS LANGLEY EXTRA", 1, "4 LOVE LANE KINGS LANGLEY"),
    ],
)
def test_peel_respects_max_k(love_lane_root, inp: str, max_k: int, expected: str):
    tokens = [t for t in inp.split(" ") if t]
    out = peel_end_tokens_with_trie(tokens, love_lane_root, steps=4, max_k=max_k)
    assert " ".join(out) == expected
