import pytest

from matcher.matcher_stage1 import match_stage1, Params


def test_swap_adjacent_accepts_when_enabled(love_lane_root):
    # Messy has LANE LOVE but canonical expects LOVE after LANE (R2L expects LANE first).
    # Use house number 4 path to avoid tie with ANNEX sibling.
    addr = "4 LANE LOVE KINGS LANGLEY"
    params = Params(allow_swap_adjacent=True)
    res = match_stage1(addr.split(), love_lane_root, params)
    assert res["matched"] is True
    assert res["uprn"] == 7  # UPRN for '4 ...' in fixture
    assert res["cost"] == params.swap_cost


def test_swap_adjacent_off_preserves_behavior(love_lane_root):
    addr = "4 LANE LOVE KINGS LANGLEY"
    params = Params(allow_swap_adjacent=False)
    res = match_stage1(addr.split(), love_lane_root, params)
    # With swap disabled, this should not match under strict guards
    assert res["matched"] is False


def test_no_swap_when_exact_succeeds(love_lane_root):
    # Normal order matches exactly, swap shouldn't be used
    addr = "4 LOVE LANE KINGS LANGLEY"
    params = Params(allow_swap_adjacent=True)
    res = match_stage1(addr.split(), love_lane_root, params)
    assert res["matched"] is True
    assert res["cost"] == 0
