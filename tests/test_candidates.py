from matcher.matcher_stage1 import match_stage1, Params


def test_candidate_enumeration_on_small_subtree(love_lane_root):
    # No match: unknown house number, but small subtree under LOVE
    tokens = "700 LOVE LANE KINGS LANGLEY ENGLAND".split()
    res = match_stage1(tokens, love_lane_root)
    assert res["matched"] is False
    assert res["final_node_count"] is not None
    # With default limit=10 and count=7, we should include candidate_uprns
    cands = res.get("candidate_uprns")
    assert cands is not None
    assert set(cands) == {1, 2, 3, 4, 5, 6, 7}
    assert res.get("limit_used") == 10


def test_candidate_enumeration_respects_limit(love_lane_root):
    tokens = "700 LOVE LANE KINGS LANGLEY ENGLAND".split()
    params = Params(max_uprns_to_return=3)
    res = match_stage1(tokens, love_lane_root, params)
    assert res["matched"] is False
    # final_node_count exceeds limit, so omit candidate_uprns
    assert "candidate_uprns" not in res
