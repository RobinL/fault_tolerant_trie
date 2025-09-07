from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import match_stage1, match_stage1_with_skips, Params
from matcher.trace_utils import Trace


def _build_love_lane_root():
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


def test_trace_optional_and_noninvasive(capsys):
    root = _build_love_lane_root()
    params = Params()

    addrs = [
        ("4 LOVE LANE KINGS LANGLEY", 7),
        ("7 LOVE LANE KINGS LANGLEY", 4),
        ("500 LOVE LANE KINGS LANGLEY ENGLAND", None),
        ("KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND", 7),
    ]

    for addr, expected_uprn in addrs:
        tokens = addr.split()
        # Without trace
        r1 = match_stage1(tokens, root, params)
        # With trace
        r2 = match_stage1(tokens, root, params, trace=Trace(enabled=True))
        assert r1["uprn"] == expected_uprn
        assert r2["uprn"] == expected_uprn
        assert r1["matched"] == (expected_uprn is not None)
        assert r2["matched"] == (expected_uprn is not None)

    # Ensure no prints from matcher functions
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""


def test_trace_optional_for_match_stage1_with_skips():
    root = _build_love_lane_root()
    # With default params, this should accept 7 despite EXTRA
    tokens = "KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND".split()
    # No trace
    uprn_no_trace = match_stage1_with_skips(tokens, root)
    # With trace
    uprn_with_trace = match_stage1_with_skips(tokens, root, trace=Trace(enabled=True))
    assert uprn_no_trace == 7
    assert uprn_with_trace == 7
