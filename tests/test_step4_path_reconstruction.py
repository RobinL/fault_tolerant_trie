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


def test_path_events_exact_and_accept():
    root = _build_love_lane_root()
    tokens = "4 LOVE LANE KINGS LANGLEY".split()
    trace = Trace(enabled=True)

    res = match_stage1(tokens, root, Params(), trace=trace)
    assert res["matched"] and res["uprn"] == 7

    # Collect ordered EXACT_DESCEND events
    exact = [ev for ev in trace.events if ev.get("action") == "EXACT_DESCEND"]
    canon_seq = [ev.get("canon") for ev in exact]
    assert canon_seq == ["LANGLEY", "KINGS", "LANE", "LOVE", "4"]

    # Last event should be an ACCEPT_* with star position at the '4'
    accept_ev = [ev for ev in trace.events if str(ev.get("action", "")).startswith("ACCEPT_")]
    assert accept_ev, "missing ACCEPT event"
    last = accept_ev[-1]
    assert tokens[last["at_m_index"]] == "4"
