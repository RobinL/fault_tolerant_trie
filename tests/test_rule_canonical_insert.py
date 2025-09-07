import pytest

from matcher.matcher_stage1 import match_stage1, Params
from matcher.trace_utils import Trace, build_alignment_table
from matcher.trie_builder import build_trie_from_canonical, TrieNode


def test_canonical_insert_positive_exact_lookahead(love_lane_root: TrieNode):
    # Missing 'LOVE' between '5' and 'LANE' should be fixed by canonical insert
    addr = "5 LANE KINGS LANGLEY"
    tokens = addr.split()
    tr = Trace(enabled=True)
    params = Params()  # defaults enable canonical insert
    res = match_stage1(tokens, love_lane_root, params, trace=tr)

    assert res["matched"] is True
    # Known UPRN for this path in fixture is 1 (house number 5)
    assert res["uprn"] == 1
    assert res["cost"] == params.canonical_insert_cost

    # Alignment condition should include inserted canonical metadata
    table = build_alignment_table(tokens, tr.events)
    assert any("ins=LOVE" in c for c in table.get("condition", []))


def test_canonical_insert_only_when_stuck(love_lane_root: TrieNode):
    # Full canonical path present: no insertion should occur
    addr = "ANNEX 7 LOVE LANE KINGS LANGLEY"
    tokens = addr.split()
    tr = Trace(enabled=True)
    params = Params()
    res_with = match_stage1(tokens, love_lane_root, params, trace=tr)
    # Baseline with canonical insert disabled
    res_without = match_stage1(tokens, love_lane_root, Params(allow_canonical_insert=False))
    assert res_with["uprn"] == res_without["uprn"]
    # Ensure no event carries inserted_canonical metadata
    assert all("inserted_canonical" not in ev for ev in tr.events)


def test_canonical_insert_disallow_numeric():
    # Build a tiny trie: B -> 2 -> A (R->L)
    # Canonical L2R sequence: A, 2, B
    canonical = [
        (999, ["A", "2", "B"], "XX1 1XX"),
    ]
    root = build_trie_from_canonical(canonical, reverse=True)

    # Messy omits the numeric '2': A B
    tokens = ["A", "B"]
    params = Params(canonical_insert_disallow_numeric=True)
    res = match_stage1(tokens, root, params)
    # With numeric inserts disallowed, this should not match
    assert res["matched"] is False


def test_canonical_insert_candidate_cap_blocks_rule():
    # Construct a node where three children (A1,A2,A3) each have child '7'
    # Path: ANNEX <- 7 <- A{1,2,3} <- LANE <- KINGS <- LANGLEY (R->L)
    canonical = [
        (101, ["ANNEX", "7", "A1", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (102, ["ANNEX", "7", "A2", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (103, ["ANNEX", "7", "A3", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    ]
    root = build_trie_from_canonical(canonical, reverse=True)

    # Messy omits the inserted canonical token (A1/A2/A3)
    tokens = ["ANNEX", "7", "LANE", "KINGS", "LANGLEY"]
    params = Params(canonical_insert_max_candidates=2)
    res = match_stage1(tokens, root, params)
    # Since there are 3 viable inserts and cap is 2, rule aborts -> no match
    assert res["matched"] is False


def test_canonical_insert_trace_stability(love_lane_root: TrieNode):
    # Trigger an insert and check star placement and canonical rows
    addr = "5 LANE KINGS LANGLEY"
    tokens = addr.split()
    tr = Trace(enabled=True)
    params = Params()
    res = match_stage1(tokens, love_lane_root, params, trace=tr)
    assert res["matched"] is True

    table = build_alignment_table(tokens, tr.events)
    # There should be a star somewhere (✓★)
    assert any(a == "✓★" for a in table.get("action", []))
    # Canonical row should still show consumed labels at the star column
    star_cols = [i for i, a in enumerate(table.get("action", [])) if a == "✓★"]
    if star_cols:
        j = star_cols[0]
        assert table.get("canonical", [])[j] != "—"
    # And condition shows inserted canonical
    assert any("ins=LOVE" in c for c in table.get("condition", []))


def test_canonical_insert_missing_north_chain():
    # Augment canonical with a NORTH branch requiring an inserted token at KINGS
    canonical = [
        (1, ["5", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (2, ["9", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (3, ["8", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (4, ["7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (6, ["6", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (7, ["4", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        # Missing-NORTH chain (reusing a UPRN id for simplicity)
        (700, ["10", "LOVE", "LANE", "NORTH", "KINGS", "LANGLEY"], "WD4 9HW"),
    ]
    root = build_trie_from_canonical(canonical, reverse=True)
    addr = "10 LOVE LANE KINGS LANGLEY"
    tokens = addr.split()
    tr = Trace(enabled=True)
    params = Params()  # default allows insert
    res = match_stage1(tokens, root, params, trace=tr)
    assert res["matched"] is True
    assert res["uprn"] == 700
    assert res["cost"] == params.canonical_insert_cost

    table = build_alignment_table(tokens, tr.events)
    # Ensure we note the inserted canonical
    assert any("ins=NORTH" in c for c in table.get("condition", []))
