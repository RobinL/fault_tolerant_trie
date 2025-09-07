from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import _search_with_skips
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


def test_step3_returns_best_state_on_exact_path():
    root = _build_love_lane_root()
    tokens = "4 LOVE LANE KINGS LANGLEY".split()
    trace = Trace(enabled=True)

    uprn, best_cost, runner_cost, best_state, parents = _search_with_skips(
        tokens,
        root,
        trace=trace,
    )

    # Exact match should be found; tracing should provide a best_state
    assert uprn == 7
    assert best_state is not None
    assert parents is not None and isinstance(parents, dict)
