import pytest
from matcher.matcher_stage1 import match_stage1_exact_only


@pytest.mark.parametrize(
    "addr, expected_uprn",
    [
        ("4 LOVE LANE KINGS LANGLEY", 7),
        ("7 LOVE LANE KINGS LANGLEY", 4),
        ("ANNEX 7 LOVE LANE KINGS LANGLEY", 5),
        # Peeling removes trailing noise
        ("4 LOVE LANE KINGS LANGLEY EXTRA", 7),
        ("KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND", 7),
    ],
)
def test_match_stage1_exact_only_accepts(love_lane_root, addr, expected_uprn):
    tokens = addr.split()
    assert match_stage1_exact_only(tokens, love_lane_root) == expected_uprn


@pytest.mark.parametrize(
    "addr",
    [
        # Non-terminal exhaustion
        "LOVE LANE KINGS LANGLEY",
        # Inner noise not at tail -> peel doesn't help, exact walk blocks
        "4 LOVE EXTRA LANE KINGS LANGLEY",
        # No path
        "UNKNOWN STREET",
    ],
)
def test_match_stage1_exact_only_rejects(love_lane_root, addr):
    tokens = addr.split()
    assert match_stage1_exact_only(tokens, love_lane_root) is None
