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


def test_stop_no_child_when_house_number_not_found():
    root = _build_love_lane_root()
    # 500 is not a known child under LOVE
    tokens = "500 LOVE LANE KINGS LANGLEY ENGLAND".split()
    tr = Trace(enabled=True)
    r = match_stage1(tokens, root, Params(), trace=tr)
    assert not r["matched"] and r["uprn"] is None

    # We should have exact descents for LANGLEY, KINGS, LANE, LOVE
    exact = [ev for ev in tr.events if ev.get("action") == "EXACT_DESCEND"]
    canon_seq = [ev.get("canon") for ev in exact]
    assert canon_seq == ["LANGLEY", "KINGS", "LANE", "LOVE"], canon_seq

    # And a STOP_NO_CHILD on the failing token '500'
    stops = [ev for ev in tr.events if str(ev.get("action", "")).startswith("STOP_")]
    assert stops and stops[-1]["action"] == "STOP_NO_CHILD"
