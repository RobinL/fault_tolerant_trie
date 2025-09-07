import pytest
from matcher.trie_builder import count_tail_L2R
from matcher.matcher_stage1 import peel_end_tokens_with_trie


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


def _peel_text_params(s: str, root, *, steps: int, max_k: int) -> str:
    tokens = [t for t in s.split(" ") if t]
    out = peel_end_tokens_with_trie(tokens, root, steps=steps, max_k=max_k)
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
        # Tie case: counts equal for KINGS vs LANE anchors, do not peel
        ("4 LOVE LANE KINGS", "4 LOVE LANE KINGS"),
        # Unknown last token base=0, jump increases to >0 → peel
        ("4 LOVE LANE KINGS LANGLEY ZZZ", "4 LOVE LANE KINGS LANGLEY"),
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


@pytest.mark.parametrize(
    "inp, steps, max_k, expected",
    [
        # With max_k=2, a three-token tail cannot be peeled at all
        # because we can't "see" back to LANGLEY in one hop (need k>=3).
        ("4 LOVE LANE KINGS LANGLEY A B C", 1, 2, "4 LOVE LANE KINGS LANGLEY A B C"),
        ("4 LOVE LANE KINGS LANGLEY A B C", 2, 2, "4 LOVE LANE KINGS LANGLEY A B C"),
        # With max_k=2, a two-token tail can be peeled in a single step.
        ("4 LOVE LANE KINGS LANGLEY A B", 1, 2, "4 LOVE LANE KINGS LANGLEY"),
        # With max_k=3 and one step, remove all three noisy tokens at once.
        (
            "4 LOVE LANE KINGS LANGLEY A B C",
            1,
            3,
            "4 LOVE LANE KINGS LANGLEY",
        ),
        # With max_k=1, even many steps cannot remove a two-token noisy tail.
        (
            "4 LOVE LANE KINGS LANGLEY A B",
            3,
            1,
            "4 LOVE LANE KINGS LANGLEY A B",
        ),
        # With max_k=1 and a single tail token, one step is enough.
        (
            "4 LOVE LANE KINGS LANGLEY EXTRA",
            1,
            1,
            "4 LOVE LANE KINGS LANGLEY",
        ),
        # With steps=0, nothing happens regardless of max_k.
        ("4 LOVE LANE KINGS LANGLEY EXTRA", 0, 10, "4 LOVE LANE KINGS LANGLEY EXTRA"),
    ],
)
def test_peel_steps_and_max_k(love_lane_root, inp: str, steps: int, max_k: int, expected: str):
    assert _peel_text_params(inp, love_lane_root, steps=steps, max_k=max_k) == expected
