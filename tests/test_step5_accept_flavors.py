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


def _last_accept_action(trace):
    acc = [ev for ev in trace.events if str(ev.get("action", "")).startswith("ACCEPT_")]
    assert acc, "no ACCEPT event recorded"
    return acc[-1]["action"]


def test_accept_unique_for_unique_leaf():
    root = _build_love_lane_root()
    tokens = "4 LOVE LANE KINGS LANGLEY".split()
    tr = Trace(enabled=True)
    r = match_stage1(tokens, root, Params(), trace=tr)
    assert r["matched"] and r["uprn"] == 7
    assert _last_accept_action(tr) == "ACCEPT_UNIQUE"


def test_accept_terminal_when_exhausted_and_not_unique():
    root = _build_love_lane_root()
    tokens = "7 LOVE LANE KINGS LANGLEY".split()
    tr = Trace(enabled=True)
    r = match_stage1(tokens, root, Params(), trace=tr)
    assert r["matched"] and r["uprn"] == 4
    assert _last_accept_action(tr) == "ACCEPT_TERMINAL"
