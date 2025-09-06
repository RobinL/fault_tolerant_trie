import pytest

from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import walk_exact


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
        ("4 LOVE LANE KINGS LANGLEY", 7),            # unique & blocked
        ("ANNEX 7 LOVE LANE KINGS LANGLEY", 5),      # unique deeper leaf
        ("7 LOVE LANE KINGS LANGLEY", 4),            # exact-exhausted terminal
        # Business tokens in front should still accept at unique node when next cannot descend
        ("KIMS NAILS 4 LOVE LANE KINGS LANGLEY", 7),
    ],
)
def test_walk_exact_accepts_when_unique_or_terminal(love_lane_root, addr, expected_uprn):
    tokens = addr.split()
    assert walk_exact(tokens, love_lane_root) == expected_uprn


@pytest.mark.parametrize(
    "addr",
    [
        # Extra token in the middle blocks traversal (no skips in Step 3)
        "4 LOVE EXTRA LANE KINGS LANGLEY",
        # Unknown start that doesn't lead into trie
        "UNKNOWN STREET",
        # exhausted at non-terminal → reject
        "LOVE LANE KINGS LANGLEY",
    ],
)
def test_walk_exact_rejects_when_not_unique_or_no_path(love_lane_root, addr):
    tokens = addr.split()
    assert walk_exact(tokens, love_lane_root) is None


def test_can_disable_terminal_acceptance_if_needed(love_lane_root):
    # With the knob off, '7 ...' goes back to rejection (count==2)
    tokens = "7 LOVE LANE KINGS LANGLEY".split()
    assert walk_exact(tokens, love_lane_root, accept_terminal_if_exhausted=False) is None
