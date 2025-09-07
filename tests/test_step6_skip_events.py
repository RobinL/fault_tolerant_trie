from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import match_stage1, Params
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


def test_skip_penalized_extra_between_love_and_lane():
    root = _build_love_lane_root()
    tokens = "KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND".split()
    tr = Trace(enabled=True)
    res = match_stage1(tokens, root, Params(), trace=tr)

    assert res["matched"] and res["uprn"] == 7

    # Ensure EXTRA is skipped but penalized (unknown child at this anchor)
    skips = [ev for ev in tr.events if ev.get("action") in ("SKIP_REDUNDANT", "SKIP_PENALIZED")]
    assert any(ev.get("messy") == "EXTRA" and ev.get("action") == "SKIP_PENALIZED" for ev in skips), tr.events
