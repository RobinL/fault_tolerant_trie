import pytest

from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import match_stage1, Params


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


def test_stage1_api_accepts_exact_and_terminal(love_lane_root):
    r1 = match_stage1("4 LOVE LANE KINGS LANGLEY".split(), love_lane_root)
    assert r1["matched"] and r1["uprn"] == 7

    r2 = match_stage1("7 LOVE LANE KINGS LANGLEY".split(), love_lane_root)
    assert r2["matched"] and r2["uprn"] == 4


def test_stage1_api_handles_peeling_and_skip(love_lane_root):
    r = match_stage1(
        "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE".split(), love_lane_root
    )
    assert r["matched"] and r["uprn"] == 7
    # Ensure we report peeled tokens used
    assert r["peeled_tokens"][-1] == "LANGLEY"


def test_stage1_api_respects_guards(love_lane_root):
    # No numeric -> reject by default
    r = match_stage1("LOVE LANE KINGS LANGLEY".split(), love_lane_root)
    assert not r["matched"]

